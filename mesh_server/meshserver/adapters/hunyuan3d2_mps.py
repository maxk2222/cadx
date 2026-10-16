"""Hunyuan3D-2 (torch) on Apple Silicon via MPS and on AMD via ROCm —
single-view and multi-view.

Install:
    mesh_server/scripts/install_hunyuan_torch.sh   # clones Tencent-Hunyuan/Hunyuan3D-2
    pip install -e <clone>                          # provides `hy3dgen`
    huggingface-cli download tencent/Hunyuan3D-2mini
    huggingface-cli download tencent/Hunyuan3D-2mv  # for multi-view

Requires PYTORCH_ENABLE_MPS_FALLBACK=1 on MPS (set by the adapter when
missing). On ROCm the pipeline runs on the torch "cuda" device namespace
(`torch.version.hip` set) — fp16 checkpoint variants work on RDNA3+/MI300.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, ClassVar

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
from meshserver.logging import get_logger
from meshserver.torch_util import is_rocm, mps_ok, torch_backend

log = get_logger(__name__)

_LICENSE = (
    "Tencent Hunyuan Community License — restricted in EU/UK/KR; "
    "prefer TRELLIS.2/TripoSG (MIT) where required"
)

_VARIANTS = {
    "hunyuan3d-2mini": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini", "fp16"),
    "hunyuan3d-2mini-turbo": (
        "tencent/Hunyuan3D-2mini",
        "hunyuan3d-dit-v2-mini-turbo",
        "fp16",
    ),
    "hunyuan3d-2": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0", "fp16"),
    "hunyuan3d-2mv": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv", "fp16"),
    "hunyuan3d-2mv-turbo": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-turbo", "fp16"),
}


def _mps_ok() -> bool:
    return mps_ok()


def _hy3d_device() -> str:
    """Torch device string this adapter runs on: mps | cuda (ROCm) | ""."""
    if _mps_ok():
        return "mps"
    if is_rocm():
        return "cuda"  # ROCm torch uses the cuda device namespace
    return ""


class _Hy3DDevice:
    """Class-level device reporting: mps on Apple Silicon, rocm on HIP,
    cpu/empty elsewhere (the adapter isn't available there anyway)."""

    def __get__(self, obj: Any, owner: type | None = None) -> str:
        return torch_backend() or "mps"


# transformers>=5 renamed DINOv2 params — the bundled Tencent checkpoints
# still carry the tf4 layout, so remap before the pipeline's plain
# load_state_dict sees them:
#   attention.attention.{query,key,value} -> attention.{q,k,v}_proj
#   attention.output.dense                -> attention.o_proj
#   mlp.weights_in (fused SwiGLU w12)     -> mlp.gate_proj + mlp.up_proj
#   mlp.weights_out                       -> mlp.down_proj
# tf4 Dinov2SwiGLUFFN chunked weights_in as (x1=gate, x2=up): silu(x1) * x2 —
# so the first half of the fused weight is gate_proj, the second is up_proj.
_TF5_DINO_REMAP = {
    "attention.attention.query.": "attention.q_proj.",
    "attention.attention.key.": "attention.k_proj.",
    "attention.attention.value.": "attention.v_proj.",
    "attention.output.dense.": "attention.o_proj.",
    "mlp.weights_out.": "mlp.down_proj.",
}


def _remap_tf5_dino(state_dict):
    fixed = {}
    for k, v in state_dict.items():
        if "mlp.weights_in." in k:
            half = v.shape[0] // 2  # fused w12: [gate | up] — tf4 chunk order
            suffix = k.split("mlp.weights_in.", 1)[1]  # weight | bias
            fixed[f"{k.split('mlp.weights_in.', 1)[0]}mlp.gate_proj.{suffix}"] = v[:half]
            fixed[f"{k.split('mlp.weights_in.', 1)[0]}mlp.up_proj.{suffix}"] = v[half:]
            continue
        for old, new in _TF5_DINO_REMAP.items():
            if old in k:
                k = k.replace(old, new)
                break
        fixed[k] = v
    return fixed


def _patch_tf5_dino_keys() -> None:
    try:
        import transformers

        if int(transformers.__version__.split(".")[0]) < 5:
            return
        from hy3dgen.shapegen.models import conditioner
    except Exception:
        return

    def _wrap(cls) -> None:
        if getattr(cls, "_tf5_keys_patched", False):
            return
        orig = cls.load_state_dict

        def patched(self, state_dict, *args, **kwargs):
            return orig(self, _remap_tf5_dino(state_dict), *args, **kwargs)

        cls.load_state_dict = patched
        cls._tf5_keys_patched = True

    for name in ("SingleImageEncoder", "DualImageEncoder"):
        cls = getattr(conditioner, name, None)
        if cls is not None:
            _wrap(cls)


class _Hy3DTorchBase(MeshAdapter):
    device = _Hy3DDevice()
    backend = "torch"
    license_note = _LICENSE
    canonical_up = "y"  # image-space convention: raw output is Y-up
    default_steps = 30
    use_flashvdm: ClassVar[bool] = False
    _variant_key: str = ""

    @classmethod
    def available(cls, cfg: Any) -> bool:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except Exception as e:
            # dependency drift (e.g. transformers/diffusers/huggingface_hub
            # version bump breaking the pinned hy3dgen stack) silently made
            # this adapter vanish once — surface the real import error via
            # registry._probe -> ModelInfo.last_error in /v1/health
            cls.available_error = str(e)[:300]
            return False
        cls.available_error = None
        return bool(_hy3d_device())

    def _variant(self) -> tuple[str, str, str]:
        return _VARIANTS[self._variant_key]

    def load(self) -> None:
        if _mps_ok():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            # unified memory: an unbounded MPS allocator can out-allocate the
            # machine's real RAM — the Metal queue then stalls mid-copy
            # (observed: frozen mps_copy_ at 0% CPU → full macOS freeze, not a
            # clean OOM). The watermarks cap MPS at a fraction of physical
            # memory so the job errors out instead of taking the host down.
            # setdefault keeps operator overrides intact; must be set before
            # the first MPS tensor allocation (i.e. before pipe.to("mps")).
            # 0.65 x recommendedMaxWorkingSetSize (~48GB @64GB) ~ 31GB budget
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.65")
            os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.55")
        try:
            import torch
            from hy3dgen.shapegen import (
                DegenerateFaceRemover,
                FaceReducer,
                FloaterRemover,
                Hunyuan3DDiTFlowMatchingPipeline,
            )
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        _patch_tf5_dino_keys()
        repo, subfolder, variant = self._variant()
        device = _hy3d_device() or "cpu"
        self._pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            repo, subfolder=subfolder, variant=variant, device=device
        )
        # MPS/ROCm: keep float16 where supported, fall back otherwise
        with contextlib.suppress(Exception):
            self._pipe.to(device)
        # FlashVDM accelerates the volume-decode stage (the dominant cost
        # after diffusion on turbo runs — 2000+ chunks at ~1-3k chunks/s);
        # upstream example pairs it with 5 steps + num_chunks=20000. Some
        # kernels may be CUDA-only — failure leaves the stock decoder in
        # place, so it is strictly best-effort.
        self._flashvdm = False
        if self.use_flashvdm:
            try:
                self._pipe.enable_flashvdm(mc_algo="mc")
                self._flashvdm = True
            except Exception as e:
                log.warning("hunyuan.flashvdm_unavailable", error=str(e)[:200])
        self._device_name = device
        self._cleaner = [FloaterRemover(), DegenerateFaceRemover(), FaceReducer()]
        self._torch = torch
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
        hard_unload()
        self._loaded = False

    def _images_arg(self, images: list[Image.Image], opts: GenOpts):
        return images[0]

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
            progress(0.05, f"running {self.id} ({getattr(self, '_device_name', '?')})")
        gen = self._torch.Generator(device="cpu").manual_seed(opts.seed & 0xFFFFFFFF)
        # upstream demo default is 384 — on MPS the volume-decode grid at that
        # size was the memory peak that froze the host (unified memory); 256
        # is the safety default there, CUDA/ROCm keeps 384
        default_res = 256 if self._device_name == "mps" else 384
        kwargs: dict[str, Any] = {
            "image": self._images_arg(images, opts),
            "num_inference_steps": opts.steps or self.default_steps or 30,
            "octree_resolution": opts.resolution or default_res,
            "generator": gen,
            "output_type": "trimesh",
        }
        if getattr(self, "_flashvdm", False):
            kwargs["num_chunks"] = 20000
        if opts.guidance is not None:
            kwargs["guidance_scale"] = opts.guidance
        out = self._pipe(**kwargs)
        mesh = out[0] if isinstance(out, (list, tuple)) else out
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = _to_trimesh(mesh)
        try:
            for cleaner in getattr(self, "_cleaner", []):
                mesh = cleaner(mesh)
        except Exception:
            pass
        if progress:
            progress(0.9, "mesh extracted")
        return AdapterGenResult(
            mesh=mesh,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=kwargs["num_inference_steps"],
            guidance=opts.guidance,
            resolution=kwargs["octree_resolution"],
        )


class Hunyuan3D2MiniAdapter(_Hy3DTorchBase):
    id = "hunyuan3d-2mini"
    name = "Hunyuan3D-2 mini Shape (torch)"
    _variant_key = "hunyuan3d-2mini"


class Hunyuan3D2MiniTurboAdapter(_Hy3DTorchBase):
    id = "hunyuan3d-2mini-turbo"
    name = "Hunyuan3D-2 mini turbo Shape (torch)"
    _variant_key = "hunyuan3d-2mini-turbo"
    # step-distilled: upstream runs it at 5 steps + FlashVDM
    default_steps = 5
    use_flashvdm = True


class Hunyuan3D2MvAdapter(_Hy3DTorchBase):
    id = "hunyuan3d-2mv"
    name = "Hunyuan3D-2mv multi-view Shape (torch)"
    supports_multiview = True
    max_images = 4
    _variant_key = "hunyuan3d-2mv"

    def _images_arg(self, images: list[Image.Image], opts: GenOpts):
        # multiview pipelines take a {view: image} mapping. Only feed views
        # whose role was actually identified — an unlabeled "other" image
        # dropped into the wrong slot (e.g. a second front shot as "back")
        # is worse than a missing view, it corrupts the latent geometry.
        roles = (opts.image_roles or [])[: len(images)]
        wanted = ["front", "back", "left", "right"]
        mapping: dict[str, Image.Image] = {}
        for role, img in zip(roles, images, strict=False):
            if role in wanted and role not in mapping:
                mapping[role] = img
        # unlabeled images fill the front slot only — never other views
        if "front" not in mapping:
            mapping["front"] = next(
                (img for role, img in zip(roles, images, strict=False) if role in ("other", "")),
                images[0],
            )
        return mapping


class Hunyuan3D2MvTurboAdapter(Hunyuan3D2MvAdapter):
    id = "hunyuan3d-2mv-turbo"
    name = "Hunyuan3D-2mv turbo multi-view Shape (torch)"
    _variant_key = "hunyuan3d-2mv-turbo"
    # step-distilled: upstream runs it at 5 steps + FlashVDM
    default_steps = 5
    use_flashvdm = True
