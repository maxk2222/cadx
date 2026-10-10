"""Hunyuan3D-2.1 shape stage via the community MLX port (Apple Silicon).

Install:
    mesh_server/scripts/install_hunyuan_mlx.sh   # clones + pip-installs the
    dgrauet/Hunyuan3D-2.1-mlx fork into this venv
    huggingface-cli download tencent/Hunyuan3D-2.1 --include "hunyuan3d-dit-v2-1/*"

License: Tencent Hunyuan3D-2.1 Community License — note the territory
restrictions (not licensed for EU/UK/South-Korea use); shown to clients.
"""

from __future__ import annotations

from typing import Any

import trimesh
from PIL import Image

from meshserver.adapters.base import AdapterGenResult, GenOpts, MeshAdapter, missing

MODEL_REPO = "tencent/Hunyuan3D-2.1"
MODEL_SUBFOLDER = "hunyuan3d-dit-v2-1"


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
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> None:
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        # the MLX port mirrors the pytorch signature
        self._pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_REPO, subfolder=MODEL_SUBFOLDER
        )
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
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

        import mlx.core as mx  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {
            "image": images[0],
            "num_inference_steps": opts.steps or 30,
            "octree_resolution": opts.resolution or 256,
            "num_chunks": 20000,
            "generator": mx.random.key(opts.seed & 0xFFFFFFFF),
            "output_type": "trimesh",
        }
        if opts.guidance is not None:
            kwargs["guidance_scale"] = opts.guidance
        if progress:
            progress(0.2, "diffusion steps")
        out = self._pipe(**kwargs)
        mesh = out[0] if isinstance(out, (list, tuple)) else out
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

        v = np.asarray(verts)
        f = np.asarray(faces)
        try:
            import torch

            if isinstance(verts, torch.Tensor):
                v = verts.detach().cpu().numpy()
            if isinstance(faces, torch.Tensor):
                f = faces.detach().cpu().numpy()
        except Exception:
            pass
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
    raise ValueError(f"cannot convert {type(obj).__name__} to trimesh")
