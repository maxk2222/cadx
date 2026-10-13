"""VAST-AI TripoSG-scribble — scribble/sketch + text conditioned image->mesh.

Unlike plain TripoSG, the scribble variant takes a line/silhouette sketch and
a text prompt, which makes it a strong fit for CADX flat parts: the measured
plan-view outline is rendered as a filled silhouette and `GenOpts.subject`
carries the part description. The CADX worker sends the silhouette as the
first image with role 'scribble'.

Install (same package as TripoSG):
    git clone https://github.com/VAST-AI-Research/TripoSG.git && pip install -e TripoSG
    huggingface-cli download VAST-AI/TripoSG-scribble

License: MIT. Experimental — MPS only via MESH_TRIPOSG_MPS=1.
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
from meshserver.adapters.triposg import _device_ok, _RunDevice, _triposg_to_trimesh

_MODEL_ID = "VAST-AI/TripoSG-scribble"


def _scribble_pipeline_cls() -> Any:
    """Locate the scribble pipeline across triposg package layouts."""
    for mod, attr in (
        ("triposg.pipelines.pipeline_triposg_scribble", "TripoSGScribblePipeline"),
        ("triposg.pipeline_triposg_scribble", "TripoSGScribblePipeline"),
        ("triposg.pipelines.pipeline_triposg", "TripoSGPipeline"),
        ("triposg.pipeline_triposg", "TripoSGPipeline"),
    ):
        try:
            m = __import__(mod, fromlist=[attr])
            return getattr(m, attr)
        except Exception:
            continue
    return None


class TripoSGScribbleAdapter(MeshAdapter):
    id = "triposg-scribble"
    name = "TripoSG-scribble (sketch+text)"
    device = _RunDevice()
    backend = "triposg"
    supports_multiview = False
    max_images = 1
    license_note = "MIT (experimental)"

    @classmethod
    def available(cls, cfg: Any) -> bool:
        return _scribble_pipeline_cls() is not None and bool(_device_ok())

    def load(self) -> None:
        try:
            import torch
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        cls = _scribble_pipeline_cls()
        if cls is None:  # pragma: no cover
            missing(type(self), "triposg scribble pipeline not importable")
        self._device = _device_ok() or "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._pipe = cls.from_pretrained(_MODEL_ID, torch_dtype=dtype).to(self._device)
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
            progress(0.05, "running TripoSG-scribble")
        kwargs: dict[str, Any] = {
            "image": images[0],
            "num_inference_steps": opts.steps or 30,
        }
        if opts.subject:
            kwargs["prompt"] = opts.subject  # scribble variant is text-guided
        if self._device == "cuda":
            kwargs["generator"] = self._torch.Generator(device="cuda").manual_seed(
                opts.seed & 0xFFFFFFFF
            )
        else:
            self._torch.manual_seed(opts.seed & 0xFFFFFFFF)
        if opts.guidance is not None:
            kwargs["guidance_scale"] = opts.guidance
        out = self._pipe(**kwargs)
        mesh = _triposg_to_trimesh(out)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("TripoSG-scribble returned no mesh")
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
