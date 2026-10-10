"""VAST-AI TripoSG 1.5B (image->mesh rectified-flow model).

Primary path: CUDA. MPS is supported experimentally (MESH_TRIPOSG_MPS=1).

Install:
    git clone https://github.com/VAST-AI-Research/TripoSG.git
    pip install -r TripoSG/requirements.txt -r TripoSG/requirements.txt  # CUDA wheels
    pip install -e TripoSG

License: MIT.
"""

from __future__ import annotations

import os
from typing import Any

import trimesh
from PIL import Image

from meshserver.adapters.base import AdapterGenResult, GenOpts, MeshAdapter, missing
from meshserver.adapters.hunyuan3d21_mlx import _to_trimesh


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


class TripoSGAdapter(MeshAdapter):
    id = "triposg"
    name = "TripoSG 1.5B"
    device = "cuda"
    backend = "triposg"
    supports_multiview = False
    max_images = 1
    license_note = "MIT"

    @classmethod
    def available(cls, cfg: Any) -> bool:
        try:
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: F401
        except Exception:
            try:
                from triposg.pipeline_triposg import TripoSGPipeline  # noqa: F401
            except Exception:
                return False
        return bool(_device_ok())

    def load(self) -> None:
        try:
            import torch
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        try:
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        except Exception:
            try:
                from triposg.pipeline_triposg import TripoSGPipeline
            except Exception as e:  # pragma: no cover
                missing(type(self), str(e))
        self._device = _device_ok() or "cpu"
        self._pipe = TripoSGPipeline.from_pretrained(
            "VAST-AI/TripoSG", torch_dtype=torch.float16
        ).to(self._device)
        self._torch = torch
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
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
        gen = self._torch.Generator(device=self._device).manual_seed(
            opts.seed & 0xFFFFFFFF
        )
        kwargs: dict[str, Any] = {
            "image": images[0],
            "generator": gen,
            "num_inference_steps": opts.steps or 30,
        }
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
