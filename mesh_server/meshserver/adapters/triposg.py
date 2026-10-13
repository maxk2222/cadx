"""VAST-AI TripoSG 1.5B (image->mesh rectified-flow model).

Primary path: CUDA. MPS is supported experimentally (MESH_TRIPOSG_MPS=1).

The upstream repo ships no packaging metadata — the adapter resolves it from
`MESH_TRIPOSG_SRC` or `.src/TripoSG` inside the mesh_server repo and adds it to
sys.path on demand. On non-CUDA devices the CUDA-only `diso` extension is
replaced by a scikit-image marching-cubes shim injected before import.

Install:
    git clone https://github.com/VAST-AI-Research/TripoSG.git mesh_server/.src/TripoSG
    pip install diffusers transformers einops opencv-python omegaconf \
        scikit-image peft jaxtyping typeguard   # NOT diso (CUDA-only)

License: MIT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import trimesh
from PIL import Image

from meshserver.adapters.base import (
    AdapterGenResult,
    GenOpts,
    MeshAdapter,
    hard_unload,
    missing,
)
from meshserver.adapters.hunyuan3d21_mlx import _to_trimesh


def _triposg_src() -> Path | None:
    env = os.environ.get("MESH_TRIPOSG_SRC")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2] / ".src" / "TripoSG")
    for c in candidates:
        if (c / "triposg" / "pipelines" / "pipeline_triposg.py").is_file():
            return c
    return None


def _install_diso_shim() -> None:
    """`diso.DiffDMC` is a CUDA-only extension. On MPS/CPU register a
    scikit-image marching-cubes shim BEFORE triposg imports `diso`."""
    try:
        import diso  # noqa: F401

        return
    except Exception:
        pass
    if "diso" in sys.modules:
        return
    import types

    import numpy as np
    import torch
    from skimage import measure

    class DiffDMC(torch.nn.Module):  # minimal duck-type for diso.DiffDMC
        def __init__(self, dtype: Any = torch.float32) -> None:
            super().__init__()

        def forward(
            self,
            sdf: Any,
            deform: Any = None,
            return_quads: bool = False,
            normalize: bool = False,
        ) -> tuple[Any, Any]:
            arr = sdf.detach().to(torch.float32).cpu().numpy()
            # hierarchical octree leaves NaN/inf outside the evaluated region —
            # map them to "outside" before marching_cubes
            arr = np.nan_to_num(arr, nan=1e6, posinf=1e6, neginf=-1e6)
            verts, faces, _n, _v = measure.marching_cubes(arr, level=0.0)
            return (
                torch.from_numpy(verts.copy()).float(),
                torch.from_numpy(faces.copy()).long(),
            )

    mod = types.ModuleType("diso")
    mod.DiffDMC = DiffDMC  # type: ignore[attr-defined]
    sys.modules["diso"] = mod


def _import_pipeline() -> Any:
    """Import TripoSGPipeline, injecting path+diso shims as needed."""
    try:
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline

        return TripoSGPipeline
    except ImportError:
        pass
    src = _triposg_src()
    if src is None:
        return None
    _install_diso_shim()
    sys.path.insert(0, str(src))
    try:
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline

        return TripoSGPipeline
    except Exception:
        return None


def _device_ok() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if os.environ.get("MESH_TRIPOSG_MPS") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return ""


def _load_triposg_pipe(pipeline_cls: Any, dtype: Any, model_id: str = "VAST-AI/TripoSG") -> Any:
    """Load a TripoSG pipeline component-wise.

    `from_pretrained` fails on diffusers >= 0.33: model_index.json references
    `triposg.schedulers.*` / `triposg.models.*` as custom components and the
    validator demands the .py files inside the HF repo. Constructing the
    pipeline from per-subfolder loads sidesteps that path entirely. Optional
    components (tokenizer/text_encoder for the scribble variant) are only
    loaded when the pipeline __init__ declares them.
    """
    import inspect

    from diffusers import FlowMatchEulerDiscreteScheduler
    from huggingface_hub import snapshot_download
    from transformers import BitImageProcessor, Dinov2Model

    from triposg.models.autoencoders import TripoSGVAEModel
    from triposg.models.transformers import TripoSGDiTModel

    root = snapshot_download(model_id)
    kw: dict[str, Any] = {
        "vae": TripoSGVAEModel.from_pretrained(root, subfolder="vae", torch_dtype=dtype),
        "transformer": TripoSGDiTModel.from_pretrained(
            root, subfolder="transformer", torch_dtype=dtype
        ),
        "scheduler": FlowMatchEulerDiscreteScheduler.from_pretrained(
            root, subfolder="scheduler"
        ),
        "image_encoder_dinov2": Dinov2Model.from_pretrained(
            root, subfolder="image_encoder_dinov2"
        ),
        "feature_extractor_dinov2": BitImageProcessor.from_pretrained(
            root, subfolder="feature_extractor_dinov2"
        ),
    }
    params = set(inspect.signature(pipeline_cls.__init__).parameters)
    if "tokenizer" in params:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        kw["tokenizer"] = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer")
        kw["text_encoder"] = CLIPTextModelWithProjection.from_pretrained(
            root, subfolder="text_encoder", torch_dtype=dtype
        )
    return pipeline_cls(**kw)


class _RunDevice:
    """Reports the device the adapter would actually run on."""

    def __get__(self, obj: Any, owner: type | None = None) -> str:
        return _device_ok() or "cuda"


class TripoSGAdapter(MeshAdapter):
    id = "triposg"
    name = "TripoSG 1.5B"
    device = _RunDevice()
    backend = "triposg"
    supports_multiview = False
    max_images = 1
    license_note = "MIT"

    @classmethod
    def available(cls, cfg: Any) -> bool:
        return _import_pipeline() is not None and bool(_device_ok())

    def load(self) -> None:
        try:
            import torch
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        TripoSGPipeline = _import_pipeline()
        if TripoSGPipeline is None:
            missing(
                type(self),
                "triposg not importable — clone into mesh_server/.src/TripoSG "
                "or set MESH_TRIPOSG_SRC",
            )
        self._device = _device_ok() or "cpu"
        # fp16 is unstable on MPS (rectified-flow DiT + Metal) — run fp32 there;
        # CUDA keeps the fast fp16 path
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._pipe = _load_triposg_pipe(TripoSGPipeline, dtype).to(self._device)
        self._torch = torch
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
        hard_unload()
        self._loaded = False

    def generate(
        self,
        images: list[Image.Image],
        opts: GenOpts,
        progress=None,
        cancelled=None,
    ) -> AdapterGenResult:
        if not self._loaded:
            self.load()
        if progress:
            progress(0.05, "running TripoSG")
        kwargs: dict[str, Any] = {
            "image": images[0],
            "num_inference_steps": opts.steps or 30,
        }
        # resolution → octree depth (grid = 2**depth); upstream defaults are
        # dense=8 / hierarchical=9 (512). Detail loss shows up mainly in the
        # hierarchical depth — derive it from opts.resolution when set.
        if opts.resolution:
            import math

            depth = int(math.ceil(math.log2(max(opts.resolution, 64))))
            depth = min(max(depth, 7), 10)
            kwargs["hierarchical_octree_depth"] = depth
            kwargs["dense_octree_depth"] = max(depth - 1, 5)
            kwargs["flash_octree_depth"] = depth
        if self._device == "cuda":
            kwargs["generator"] = self._torch.Generator(device="cuda").manual_seed(
                opts.seed & 0xFFFFFFFF
            )
        else:
            # torch.Generator(mps/cpu) is flaky across versions — seed globally;
            # and the flash decoder's DiffDMC is CUDA-only — the hierarchical
            # decoder uses plain skimage.marching_cubes instead
            self._torch.manual_seed(opts.seed & 0xFFFFFFFF)
            kwargs["use_flash_decoder"] = False
        if opts.guidance is not None:
            kwargs["guidance_scale"] = opts.guidance
        out = self._pipe(**kwargs)
        # TripoSG returns meshes=[(vertices, faces), ...] or a samples dict
        mesh = _triposg_to_trimesh(out)
        if progress:
            progress(0.9, "mesh extracted")
        return AdapterGenResult(
            mesh=mesh,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=kwargs["num_inference_steps"],
            guidance=opts.guidance,
            resolution=opts.resolution,
        )


def _triposg_to_trimesh(out: Any) -> trimesh.Trimesh:
    import numpy as np

    samples = getattr(out, "samples", None)
    if samples is None and isinstance(out, dict):
        samples = out.get("samples") or out.get("meshes")
    if samples is None:
        samples = getattr(out, "meshes", None)
    if samples is None:
        return _to_trimesh(out)
    item = samples[0] if isinstance(samples, (list, tuple)) else samples
    if isinstance(item, (list, tuple)) and len(item) == 2:
        # TripoSG returns raw (vertices, faces) tuples from the extractor
        verts, faces = item
        try:
            import torch

            if isinstance(verts, torch.Tensor):
                verts = verts.detach().cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()
        except Exception:
            pass
        return trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces),
                               process=False)
    if isinstance(item, dict):
        verts = item.get("vertices") or item.get("v")
        faces = item.get("faces") or item.get("f")
        if verts is not None and faces is not None:
            try:
                import torch

                if isinstance(verts, torch.Tensor):
                    verts = verts.detach().cpu().numpy()
                if isinstance(faces, torch.Tensor):
                    faces = faces.detach().cpu().numpy()
            except Exception:
                pass
            return trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces),
                                   process=False)
    return _to_trimesh(item)
