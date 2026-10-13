"""Hunyuan3D-2.1 shape stage via the community MLX port (Apple Silicon).

Install:
    mesh_server/scripts/install_hunyuan_mlx.sh   # clones the
    dgrauet/Hunyuan3D-2.1-mlx fork into mesh_server/.src/ and installs
    mlx + mlx-arsenal. Converted weights (dgrauet/hunyuan3d-2.1-mlx) are
    downloaded from HF on first load; override the location with
    HUNYUAN3D_MLX_WEIGHTS_DIR or the clone location with MESH_HUNYUAN_MLX_SRC.

License: Tencent Hunyuan3D-2.1 Community License — note the territory
restrictions (not licensed for EU/UK/South-Korea use); shown to clients.
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

WEIGHTS_REPO = "dgrauet/hunyuan3d-2.1-mlx"


def _src_root() -> Path:
    """Directory containing the clone of dgrauet/Hunyuan3D-2.1-mlx."""
    env = os.environ.get("MESH_HUNYUAN_MLX_SRC")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / ".src" / "Hunyuan3D-2.1-mlx"


def _pkg_root() -> Path:
    # the importable `hy3dshape` package lives in <clone>/hy3dshape/hy3dshape
    return _src_root() / "hy3dshape"


def _patch_terminal_sigma() -> None:
    """Fix the appended terminal sigma for ascending sigma schedules.

    pipeline_mlx feeds ascending sigmas ``linspace(0, 1, steps)`` — matching the
    upstream torch pipeline — into mlx_arsenal's FlowMatchEulerDiscreteScheduler,
    which appends a terminal ``0.0`` intended for *descending* schedules.
    Upstream (hy3dshape.schedulers) appends ``1.0`` instead, making the final
    step a no-op. With the 0.0 terminal the last of N steps jumps the latent
    a full sigma back toward noise, producing a degenerate SDF volume.

    We do not patch the vendor clone; fix the scheduler class at import time
    so the terminal sigma equals the last real one (zero-length final step).
    """
    from mlx_arsenal.diffusion import FlowMatchEulerDiscreteScheduler

    if getattr(FlowMatchEulerDiscreteScheduler, "_terminal_sigma_fixed", False):
        return
    orig = FlowMatchEulerDiscreteScheduler.set_timesteps

    def patched(self: Any, num_inference_steps: int, sigmas: Any = None) -> None:
        orig(self, num_inference_steps, sigmas=sigmas)
        s = self.sigmas
        if s is not None and len(s) == num_inference_steps + 1 and s[-2] > s[-3]:
            # ascending schedule -> upstream appends the last sigma again
            import mlx.core as mx

            self.sigmas = mx.concatenate([s[:-1], s[-2:-1]])

    patched._terminal_sigma_fixed = True  # type: ignore[attr-defined]
    FlowMatchEulerDiscreteScheduler.set_timesteps = patched  # type: ignore[method-assign]
    FlowMatchEulerDiscreteScheduler._terminal_sigma_fixed = True  # type: ignore[attr-defined]


def _import_pipeline() -> Any:
    root = _pkg_root()
    if not (root / "hy3dshape" / "pipeline_mlx.py").exists():
        raise ImportError(
            f"MLX fork not found at {root} — run "
            "mesh_server/scripts/install_hunyuan_mlx.sh"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _patch_terminal_sigma()
    from hy3dshape.pipeline_mlx import ShapePipeline

    return ShapePipeline


class Hunyuan3D21MlxAdapter(MeshAdapter):
    id = "hunyuan3d-2.1-mlx"
    name = "Hunyuan3D-2.1 Shape (MLX)"
    device = "mlx"
    backend = "mlx"
    supports_multiview = False
    max_images = 1
    license_note = (
        "Tencent Hunyuan Community License — restricted in EU/UK/KR; "
        "prefer TRELLIS.2/TripoSG (MIT) where required"
    )

    @classmethod
    def available(cls, cfg: Any) -> bool:
        try:
            import mlx.core  # noqa: F401

            _import_pipeline()
        except Exception:
            return False
        return True

    def load(self) -> None:
        try:
            ShapePipeline = _import_pipeline()
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        self._pipe = ShapePipeline.from_pretrained()  # auto-downloads WEIGHTS_REPO
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
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
            progress(0.05, "running Hunyuan3D-2.1 (MLX)")

        kwargs: dict[str, Any] = {
            "image": images[0],
            "num_inference_steps": opts.steps or 50,
            # upstream demo default is 384 — 256 leaves ~0.5mm MC cells on
            # palm-sized parts (visible stair steps, lost sub-mm features)
            "octree_resolution": opts.resolution or 384,
            "seed": opts.seed,
        }
        if opts.guidance is not None:
            kwargs["guidance_scale"] = opts.guidance
        if progress:
            progress(0.2, "diffusion steps")
        # pipeline __call__ frees the DiT (`del self.dit`) to release memory;
        # keep our reference so the loaded pipeline stays reusable across jobs.
        dit = self._pipe.dit
        mesh = self._pipe(**kwargs)
        if self._pipe.dit is None:
            self._pipe.dit = dit
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
            guidance=opts.guidance,
            resolution=kwargs["octree_resolution"],
        )


def _to_trimesh(obj: Any) -> trimesh.Trimesh:
    """Best-effort conversion of generator outputs into a Trimesh."""
    if isinstance(obj, trimesh.Trimesh):
        return obj
    verts = getattr(obj, "vertices", None)
    faces = getattr(obj, "faces", None)
    if verts is not None and faces is not None:
        import numpy as np

        return trimesh.Trimesh(
            vertices=np.asarray(verts), faces=np.asarray(faces), process=False
        )
    raise ValueError(f"cannot convert {type(obj).__name__} to trimesh")
