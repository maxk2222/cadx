"""Step1X-3D geometry (stepfun-ai/Step1X-3D) — two-stage VAE+DiT, CUDA only.

The upstream repo ships no packaging metadata — the adapter resolves the
clone from `MESH_STEP1X_SRC` or `.src/Step1X-3D` inside the mesh_server repo
and adds it to sys.path on demand.

Two upstream quirks are shimmed at import time:

* ``step1x3d_geometry/__init__.py`` eagerly imports the *training* stack
  (``data`` -> mosaicml-streaming, ``systems`` -> deepspeed) that the
  geometry inference pipeline never touches. The adapter binds a minimal
  namespace package carrying only the ``register``/``find`` helpers the
  inference modules actually use.
* ``pipeline.py`` imports ``FluxIPAdapterMixin`` etc. from
  ``diffusers.loaders`` — added in diffusers 0.32 while the shared stack
  pins 0.31 for the Hunyuan/huggingface_hub combo. The mixin classes are
  pure capability mixins; a missing name is shimmed with an empty class,
  exactly like the diso shim in triposg.py.

Note for the watertight workstream: the clone's
``data/watertight_and_sampling.py`` carries the reference watertight-mesh
implementation (depth_test + winding_number parity check, requires
nvdiffrast/libigl/PyMCubes — see its imports). WS-B can lift the method
from there without depending on the CUDA-only renderer bits.

Install (on a CUDA box):
    mesh_server/scripts/install_step1x.sh   # clones stepfun-ai/Step1X-3D
    # weights (stepfun-ai/Step1X-3D, subfolder Step1X-3D-Geometry-1300m)
    # download on first load

License: Apache-2.0. NVIDIA CUDA only — upstream pins a cu124 torch stack
and the geometry stack is untested on ROCm; on HIP or MPS this adapter
stays unavailable.
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
from meshserver.torch_util import cuda_ok, is_rocm

_WEIGHTS_REPO = "stepfun-ai/Step1X-3D"
_GEOMETRY_SUBFOLDER = "Step1X-3D-Geometry-1300m"

# loader mixins pipeline.py pulls from diffusers.loaders — capability
# mixins only, safe to stub when the pinned diffusers lacks them
_DIFFUSERS_MIXINS = (
    "FluxIPAdapterMixin",
    "FluxLoraLoaderMixin",
    "FromSingleFileMixin",
    "TextualInversionLoaderMixin",
)


def _step1x_src() -> Path | None:
    env = os.environ.get("MESH_STEP1X_SRC")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2] / ".src" / "Step1X-3D")
    for c in candidates:
        if (c / "step1x3d_geometry" / "models" / "pipelines" / "pipeline.py").is_file():
            return c
    return None


def _install_diffusers_shim() -> None:
    try:
        import diffusers.loaders as loaders
    except Exception:
        return
    for name in _DIFFUSERS_MIXINS:
        if not hasattr(loaders, name):
            setattr(loaders, name, type(name, (), {}))


def _install_pkg_stub(src: Path) -> None:
    """Bind a minimal ``step1x3d_geometry`` package for the clone.

    The real ``__init__.py`` unconditionally imports the training stack
    (mosaicml-streaming, deepspeed); inference only needs the lightweight
    ``register``/``find`` registry replicated here. No-op when a real
    (pip-installed) package is already resident.
    """
    import importlib
    import logging
    import types

    if "step1x3d_geometry" in sys.modules:
        return
    pkg = types.ModuleType("step1x3d_geometry")
    pkg.__path__ = [str(src / "step1x3d_geometry")]
    modules: dict[str, Any] = {}

    def register(name: str):
        def decorator(cls: type) -> type:
            if name in modules:
                raise ValueError(f"Module {name} already exists!")
            modules[name] = cls
            return cls

        return decorator

    def find(name: str) -> type:
        if name in modules:
            return modules[name]
        module_string = ".".join(name.split(".")[:-1])
        cls_name = name.split(".")[-1]
        return getattr(importlib.import_module(module_string), cls_name)

    pkg.register = register  # type: ignore[attr-defined]
    pkg.find = find  # type: ignore[attr-defined]
    pkg.__modules__ = modules  # type: ignore[attr-defined]
    pkg.logger = logging.getLogger("pytorch_lightning")  # type: ignore[attr-defined]
    pkg.debug = pkg.info = pkg.warn = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["step1x3d_geometry"] = pkg


def _import_pipeline() -> Any:
    try:
        from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline

        return Step1X3DGeometryPipeline
    except Exception:
        pass
    src = _step1x_src()
    if src is None:
        return None
    sys.path.insert(0, str(src))
    _install_diffusers_shim()
    _install_pkg_stub(src)
    try:
        from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline

        return Step1X3DGeometryPipeline
    except Exception:
        return None


def _device_ok() -> bool:
    # torch.cuda.is_available() is also True on ROCm — require a real NVIDIA
    # build (no HIP), matching upstream's cu124 pin
    return cuda_ok() and not is_rocm()


class Step1X3DAdapter(MeshAdapter):
    id = "step1x-3d"
    name = "Step1X-3D Geometry 1.3B (CUDA)"
    device = "cuda"
    backend = "step1x-3d"
    supports_multiview = False
    max_images = 1
    license_note = "Apache-2.0"
    default_steps = 50  # upstream inference.py default
    est_mem_mb = 12000

    @classmethod
    def available(cls, cfg: Any) -> bool:
        pipe = _import_pipeline()
        if pipe is None:
            cls.available_error = (
                "step1x3d_geometry not importable — clone into "
                "mesh_server/.src/Step1X-3D or set MESH_STEP1X_SRC, and "
                "install the upstream requirements (install_step1x.sh)"
            )
            return False
        cls.available_error = None
        return _device_ok()

    def load(self) -> None:
        try:
            import torch
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        pipe_cls = _import_pipeline()
        if pipe_cls is None:
            missing(
                type(self),
                "step1x3d_geometry not importable — clone into "
                "mesh_server/.src/Step1X-3D or set MESH_STEP1X_SRC",
            )
        self._pipe = pipe_cls.from_pretrained(
            _WEIGHTS_REPO, subfolder=_GEOMETRY_SUBFOLDER
        ).to("cuda")
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
            progress(0.05, "running Step1X-3D geometry")
        gen = self._torch.Generator(device="cuda").manual_seed(opts.seed & 0xFFFFFFFF)
        kwargs: dict[str, Any] = {
            # the pipeline accepts a PIL image directly (also path/tensor)
            "image": images[0],
            "num_inference_steps": opts.steps or self.default_steps or 50,
            # upstream demo uses 7.5 / 384 — the pipeline defaults are 30
            # steps and a 200k face cap, both fine to keep
            "guidance_scale": opts.guidance if opts.guidance is not None else 7.5,
            "octree_resolution": opts.resolution or 384,
            "generator": gen,
            "output_type": "trimesh",
        }
        out = self._pipe(**kwargs)
        # Step1X3DGeometryPipelineOutput(image=..., mesh=[trimesh, ...])
        meshes = getattr(out, "mesh", None) or (out[0] if isinstance(out, tuple) else None)
        mesh = meshes[0] if isinstance(meshes, (list, tuple)) else meshes
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = _to_trimesh(mesh)
        if progress:
            progress(0.9, "mesh extracted")
        return AdapterGenResult(
            mesh=mesh,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=kwargs["num_inference_steps"],
            guidance=kwargs["guidance_scale"],
            resolution=kwargs["octree_resolution"],
        )
