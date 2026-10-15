"""Microsoft TRELLIS.2-4B (CUDA) — geometry only (shape stage, no PBR bake).

Install (on a CUDA box, >=24 GB VRAM recommended):
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    . TRELLIS.2/setup.sh --new-env --basic --flash-attn --cumesh --o-voxel --flexgemm
    huggingface-cli download microsoft/TRELLIS.2-4B

License: MIT. NVIDIA CUDA only — flash-attn/CuMesh/spconv/o-voxel have no
viable ROCm build; on HIP torch (`torch.version.hip` set) this adapter stays
unavailable and the registry falls back to TripoSG.
"""

from __future__ import annotations

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
from meshserver.torch_util import is_rocm


def _cuda_ok() -> bool:
    try:
        import torch

        # torch.cuda.is_available() is also True on ROCm — require a real
        # NVIDIA build (no HIP) since all TRELLIS.2 extensions are CUDA-only
        return bool(torch.cuda.is_available()) and not is_rocm()
    except Exception:
        return False


class Trellis2Adapter(MeshAdapter):
    id = "trellis2"
    name = "TRELLIS.2-4B (CUDA, shape only)"
    device = "cuda"
    backend = "trellis2"
    supports_multiview = False
    max_images = 1
    license_note = "MIT"

    @classmethod
    def available(cls, cfg: Any) -> bool:
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401
        except Exception:
            return False
        return _cuda_ok()

    def load(self) -> None:
        try:
            import torch
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        self._pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        self._pipe.cuda()
        self._torch = torch
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
        try:
            self._torch.cuda.empty_cache()
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
            progress(0.05, "running TRELLIS.2 (CUDA)")
        with self._torch.no_grad():
            # pipeline.run returns [mesh] — the SparseMesh exposes
            # vertices/faces (plus PBR attrs we deliberately drop: geometry only)
            res = self._pipe.run(images[0])
        out = res[0] if isinstance(res, (list, tuple)) else res
        mesh = _to_trimesh(out)
        if progress:
            progress(0.9, "mesh extracted")
        return AdapterGenResult(
            mesh=mesh,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            resolution=opts.resolution,
        )
