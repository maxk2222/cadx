"""Hunyuan3D-2 (torch) on Apple Silicon via MPS — single-view and multi-view.

Install:
    mesh_server/scripts/install_hunyuan_torch.sh   # clones Tencent-Hunyuan/Hunyuan3D-2
    pip install -e <clone>                          # provides `hy3dgen`
    huggingface-cli download tencent/Hunyuan3D-2mini
    huggingface-cli download tencent/Hunyuan3D-2mv  # for multi-view

Requires PYTORCH_ENABLE_MPS_FALLBACK=1 (set by the adapter when missing).
"""

from __future__ import annotations

import os
from typing import Any

import trimesh
from PIL import Image

from meshserver.adapters.base import AdapterGenResult, GenOpts, MeshAdapter, missing
from meshserver.adapters.hunyuan3d21_mlx import _to_trimesh

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
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


class _Hy3DTorchBase(MeshAdapter):
    device = "mps"
    backend = "torch-mps"
    license_note = _LICENSE
    _variant_key: str = ""

    @classmethod
    def available(cls, cfg: Any) -> bool:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except Exception:
            return False
        return _mps_ok()

    def _variant(self) -> tuple[str, str, str]:
        return _VARIANTS[self._variant_key]

    def load(self) -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import torch
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        repo, subfolder, variant = self._variant()
        self._pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            repo, subfolder=subfolder, variant=variant
        )
        # MPS: keep float16 where supported, fall back otherwise
        try:
            self._pipe.to("mps")
        except Exception:
            pass
        self._cleaner = [FloaterRemover(), DegenerateFaceRemover(), FaceReducer()]
        self._torch = torch
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
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
            progress(0.05, f"running {self.id} (MPS)")
        gen = self._torch.Generator(device="cpu").manual_seed(opts.seed & 0xFFFFFFFF)
        kwargs: dict[str, Any] = {
            "image": self._images_arg(images, opts),
            "num_inference_steps": opts.steps or 30,
            "octree_resolution": opts.resolution or 256,
            "generator": gen,
            "output_type": "trimesh",
        }
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
    name = "Hunyuan3D-2 mini Shape (torch/MPS)"
    _variant_key = "hunyuan3d-2mini"


class Hunyuan3D2MiniTurboAdapter(_Hy3DTorchBase):
    id = "hunyuan3d-2mini-turbo"
    name = "Hunyuan3D-2 mini turbo Shape (torch/MPS)"
    _variant_key = "hunyuan3d-2mini-turbo"


class Hunyuan3D2MvAdapter(_Hy3DTorchBase):
    id = "hunyuan3d-2mv"
    name = "Hunyuan3D-2mv multi-view Shape (torch/MPS)"
    supports_multiview = True
    max_images = 4
    _variant_key = "hunyuan3d-2mv"

    def _images_arg(self, images: list[Image.Image], opts: GenOpts):
        # multiview pipelines take a {view: image} mapping
        roles = (opts.image_roles or [])[: len(images)]
        wanted = ["front", "back", "left", "right"]
        mapping: dict[str, Image.Image] = {}
        for role, img in zip(roles, images, strict=False):
            if role in wanted and role not in mapping:
                mapping[role] = img
        for img, want in zip(images, wanted, strict=False):
            mapping.setdefault(want, img)
        return mapping or {"front": images[0]}


class Hunyuan3D2MvTurboAdapter(Hunyuan3D2MvAdapter):
    id = "hunyuan3d-2mv-turbo"
    name = "Hunyuan3D-2mv turbo multi-view Shape (torch/MPS)"
    _variant_key = "hunyuan3d-2mv-turbo"
