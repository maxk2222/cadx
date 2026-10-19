"""Hunyuan3D-Omni (torch) — controllable image-to-3D on MPS and CUDA.

Separate upstream clone from Hunyuan3D-2: the Omni repo ships its own
`hy3dshape` package (SiT flow-matching pipeline, 3.3B) that adds a unified
control encoder on top of the Hunyuan3D-2.1 architecture. Conditioning is
selected via `GenOpts.control_type`:

    bbox   GenOpts.control_input = [x, y, z] dims in mm (or
           [xmin, ymin, zmin, xmax, ymax, zmax]) — the model consumes the
           relative proportions, normalised so the largest dim is 1.
           This is the primary use-case: scale/shape guidance from the
           agent's measured bounding box.
    point  control_input = path to a mesh/point cloud, or (N, 3) points in
           the model's normalised [-1, 1] space.
    voxel  same input as point; surface samples are voxelised to a 16^3
           grid by the encoder.
    pose   control_input = path to a bone-points .txt (np.loadtxt format)
           or an (N, C) array — skeletal pose control for characters.

The Omni control encoder *requires* one control signal — when the request
carries none, the adapter falls back to a neutral bbox of [1, 1, 1] so the
model degenerates to plain image-to-3D rather than erroring.

Install:
    mesh_server/scripts/install_omni.sh   # clones Tencent-Hunyuan/Hunyuan3D-Omni
    # weights are pulled from tencent/Hunyuan3D-Omni on first load

Requires PYTORCH_ENABLE_MPS_FALLBACK=1 on MPS (set by the adapter when
missing), same as the other Hunyuan torch adapters.

License: Tencent Hunyuan 3D Omni Community License — territory-restricted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from meshserver.torch_util import cuda_ok, is_rocm, mps_ok

log = get_logger(__name__)

_LICENSE = (
    "Tencent Hunyuan 3D Omni Community License — restricted in EU/UK/KR; "
    "prefer TRELLIS.2/TripoSG (MIT) where required"
)

_CONTROL_TYPES = ("bbox", "voxel", "point", "pose")


def _omni_src() -> Path | None:
    """Locate the Hunyuan3D-Omni clone (env override first)."""
    env = os.environ.get("MESH_HUNYUAN_OMNI_SRC")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2] / ".src" / "Hunyuan3D-Omni")
    for c in candidates:
        if (c / "hy3dshape" / "pipelines" / "pipeline_generation_sit_omni.py").is_file():
            return c
    return None


def _install_pkg(src: Path) -> None:
    """Bind ``sys.modules['hy3dshape']`` to the Omni clone's package.

    Upstream ships ``hy3dshape/`` WITHOUT an ``__init__.py`` — a namespace
    package always loses against the regular ``hy3dshape`` package of the
    MLX fork (.src/Hunyuan3D-2.1-mlx), no matter the sys.path order. So the
    module is created explicitly with ``__path__`` pointed at the clone;
    submodule imports (``hy3dshape.pipelines`` etc.) then resolve straight
    from it. A resident foreign ``hy3dshape`` is evicted first — safe
    because the registry keeps at most one heavy model loaded at a time.
    """
    import types

    pkg_dir = str(src / "hy3dshape")
    mod = sys.modules.get("hy3dshape")
    if mod is not None:
        paths = [str(p) for p in (getattr(mod, "__path__", None) or [])]
        if pkg_dir in paths:
            return  # already bound to the Omni clone
        log.warning("hunyuan_omni.evict_foreign_hy3dshape", path=paths)
        for name in [
            k for k in sys.modules if k == "hy3dshape" or k.startswith("hy3dshape.")
        ]:
            sys.modules.pop(name, None)
    pkg = types.ModuleType("hy3dshape")
    pkg.__path__ = [pkg_dir]
    sys.modules["hy3dshape"] = pkg


def _import_omni_pipeline() -> Any:
    """Import Hunyuan3DOmniSiTFlowMatchingPipeline from the clone."""
    try:
        from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline

        return Hunyuan3DOmniSiTFlowMatchingPipeline
    except Exception:
        pass
    src = _omni_src()
    if src is None:
        raise ImportError(
            "Hunyuan3D-Omni clone not found — run "
            "mesh_server/scripts/install_omni.sh or set MESH_HUNYUAN_OMNI_SRC"
        )
    _install_pkg(src)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import importlib

    return importlib.import_module(
        "hy3dshape.pipelines"
    ).Hunyuan3DOmniSiTFlowMatchingPipeline


def _omni_device() -> str:
    """Torch device string this adapter runs on: mps | cuda (NVIDIA/ROCm) | ""."""
    if mps_ok():
        return "mps"
    if cuda_ok():
        return "cuda"  # ROCm torch also uses the cuda device namespace
    return ""


class _OmniDevice:
    """Class-level device reporting for /v1/models."""

    def __get__(self, obj: Any, owner: type | None = None) -> str:
        if mps_ok():
            return "mps"
        if cuda_ok():
            return "rocm" if is_rocm() else "cuda"
        return "cuda"


def _normalise_mesh(mesh: trimesh.Trimesh, scale: float = 0.9999) -> trimesh.Trimesh:
    """Center + scale into [-scale, scale] — mirrors upstream inference.py."""
    bounds = mesh.bounds
    centre = (bounds[1] + bounds[0]) / 2
    extent = float((bounds[1] - bounds[0]).max())
    mesh.apply_translation(-centre)
    mesh.apply_scale(1 / extent * 2 * scale)
    return mesh


class Hunyuan3DOmniAdapter(MeshAdapter):
    id = "hunyuan3d-omni"
    name = "Hunyuan3D-Omni controllable Shape (torch)"
    device = _OmniDevice()
    backend = "torch"
    supports_multiview = False
    max_images = 1
    license_note = _LICENSE
    canonical_up = "y"  # same image-space convention as the other Hunyuan adapters
    default_steps = 50  # upstream inference.py default
    est_mem_mb = 14000

    @classmethod
    def available(cls, cfg: Any) -> bool:
        try:
            _import_omni_pipeline()
        except Exception as e:
            # surface the real import error via registry._probe ->
            # ModelInfo.last_error in /v1/health (hunyuan3d2_mps pattern)
            cls.available_error = str(e)[:300]
            return False
        cls.available_error = None
        return bool(_omni_device())

    def load(self) -> None:
        if mps_ok():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            # unified memory: cap the MPS allocator below physical RAM so a
            # runaway job errors out instead of freezing the host (same
            # watermark policy as hunyuan3d2_mps; must precede first alloc)
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.65")
            os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.55")
        try:
            import torch

            pipeline_cls = _import_omni_pipeline()
            from hy3dshape.postprocessors import (
                DegenerateFaceRemover,
                FloaterRemover,
            )
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        device = _omni_device() or "cpu"
        self._pipe = pipeline_cls.from_pretrained(
            "tencent/Hunyuan3D-Omni",
            device=device,
            dtype=torch.float16,
        )
        self._device_name = device
        self._torch = torch
        self._cleaner = [FloaterRemover(), DegenerateFaceRemover()]
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
        hard_unload()
        self._loaded = False

    def _control_kwargs(self, opts: GenOpts) -> dict[str, Any]:
        """Map GenOpts.control_type/control_input to pipeline kwargs.

        Control tensors land on the pipeline device/dtype; shapes follow
        upstream inference.py: bbox (1,1,3) of relative dims 0..1, point /
        voxel / pose (1,N,3+) of normalised coordinates.
        """
        torch = self._torch
        device, dtype = self._pipe.device, self._pipe.dtype
        control_type = opts.control_type or "bbox"
        if control_type not in _CONTROL_TYPES:
            raise ValueError(
                f"unknown control_type {control_type!r} — expected one of {_CONTROL_TYPES}"
            )
        ci = opts.control_input
        if ci is None and control_type != "bbox":
            raise ValueError(f"control_type={control_type!r} needs control_input")

        if control_type == "bbox":
            dims = self._bbox_dims(ci)
            bbox = torch.FloatTensor(dims).reshape(1, 1, 3).to(device).to(dtype)
            return {"bbox": bbox}
        if control_type == "pose":
            # bone points: path to a np.loadtxt-able file or an (N, C) array
            pts = np.loadtxt(ci) if isinstance(ci, (str, Path)) else np.asarray(ci)
            pose = torch.as_tensor(pts, dtype=torch.float32).unsqueeze(0).to(device).to(dtype)
            return {"pose": pose}
        # point | voxel — file path to a mesh/point cloud, or an (N, 3)
        # array already in the model's normalised [-1, 1] space
        if isinstance(ci, (str, Path)):
            mesh = trimesh.load(str(ci))
            if control_type == "voxel":
                # upstream applies a -90 deg X rotation before normalising
                rot = trimesh.transformations.rotation_matrix(
                    angle=np.radians(-90), direction=[1, 0, 0]
                )
                mesh.apply_transform(rot)
            mesh = _normalise_mesh(mesh, scale=0.98 if control_type == "point" else 0.9999)
            surf = mesh.sample(81920) if control_type == "voxel" else np.asarray(mesh.vertices)
        else:
            surf = np.asarray(ci, dtype=np.float32)
        pts = torch.as_tensor(surf, dtype=torch.float32).unsqueeze(0).to(device).to(dtype)
        return {control_type: pts}

    @staticmethod
    def _bbox_dims(control_input: Any) -> list[float]:
        """Relative bbox proportions [x, y, z] in 0..1 (largest dim = 1).

        Accepts [dx, dy, dz] (mm — units cancel out) or
        [xmin, ymin, zmin, xmax, ymax, zmax]. No input -> neutral cube.
        """
        if control_input is None:
            return [1.0, 1.0, 1.0]
        vals = [float(v) for v in np.asarray(control_input, dtype=np.float64).ravel()]
        if len(vals) == 6:
            vals = [abs(vals[i + 3] - vals[i]) for i in range(3)]
        elif len(vals) != 3:
            raise ValueError(
                f"bbox control_input needs 3 dims or 6 bounds, got {len(vals)} values"
            )
        m = max(vals)
        if m <= 0:
            return [1.0, 1.0, 1.0]
        return [v / m for v in vals]

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
        # 512 is the upstream demo default — on MPS the volume decode at that
        # size is the memory peak that can freeze the host (unified memory);
        # 256 is the safety default there (same policy as hunyuan3d2_mps)
        default_res = 256 if self._device_name == "mps" else 384
        kwargs: dict[str, Any] = {
            # pipeline.prepare_image iterates the argument — pass a list so a
            # single PIL image works (a bare PIL image is not iterable)
            "image": [images[0]],
            "num_inference_steps": opts.steps or self.default_steps or 50,
            "octree_resolution": opts.resolution or default_res,
            "guidance_scale": opts.guidance if opts.guidance is not None else 4.5,
            "mc_level": 0,
            "generator": gen,
        }
        kwargs.update(self._control_kwargs(opts))
        out = self._pipe(**kwargs)
        # upstream: {'shapes': [[trimesh, ...]], 'sampled_point': ..., 'image': ...}
        shapes = out.get("shapes") if isinstance(out, dict) else getattr(out, "shapes", None)
        mesh = shapes[0][0] if shapes and isinstance(shapes[0], (list, tuple)) else shapes[0]
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
            guidance=kwargs["guidance_scale"],
            resolution=kwargs["octree_resolution"],
            extra={"control_type": opts.control_type or "bbox"},
        )
