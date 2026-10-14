"""FLUX.1-schnell text-to-image via diffusers (CUDA).

Install:
    pip install 'meshserver[image-torch]'
Weights are pulled from HF on first load (black-forest-labs/FLUX.1-schnell,
Apache-2.0, ungated — no HF token required).

Memory ladder (config cuda_offload): full GPU residency is fastest but needs
~24 GB free VRAM; 'model' moves each pipeline stage between RAM and VRAM
(~12-14 GB peak); 'sequential' streams per-module (~6-8 GB, slowest).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class FluxSchnellAdapter(ImageAdapter):
    id = "flux-schnell"
    name = "FLUX.1-schnell (diffusers, CUDA)"
    device = "cuda"
    backend = "diffusers"
    license_note = "Apache-2.0 (FLUX.1-schnell)"
    default_size = 768
    max_count = 4
    max_side = 1024
    supports_edit: ClassVar[bool] = True  # img2img via FluxImg2ImgPipeline.from_pipe

    @classmethod
    def available(cls, cfg: Any) -> bool:  # noqa: ARG003
        try:
            from diffusers import FluxPipeline  # noqa: F401
        except Exception:
            return False
        return _cuda_ok()

    def load(self) -> None:
        try:
            import torch
            from diffusers import FluxPipeline
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        self._torch = torch
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir=str(self.cfg.model_dir),
        )
        offload = getattr(self.cfg, "cuda_offload", "auto")
        if offload == "auto":
            try:
                free_gb = torch.cuda.mem_get_info()[0] / 1e9
                offload = "model" if free_gb < self.cfg.cuda_offload_vram_gb else "none"
            except Exception:
                offload = "model"
        if offload == "model":
            pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to("cuda")
        self._pipe = pipe
        self._loaded = True

    def _edit_pipe(self) -> Any:
        """img2img pipeline sharing the txt2img weights (no extra VRAM)."""
        if getattr(self, "_pipe_edit", None) is None:
            from diffusers import FluxImg2ImgPipeline

            self._pipe_edit = FluxImg2ImgPipeline.from_pipe(self._pipe)
        return self._pipe_edit

    def unload(self) -> None:
        self._pipe = None
        self._pipe_edit = None
        self._loaded = False
        hard_unload()

    def generate(
        self,
        opts: ImageGenOpts,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ImageGenResult:
        if not self._loaded:
            self.load()
        torch = self._torch
        images = []
        for i in range(max(1, opts.count)):
            if cancelled is not None and cancelled():
                break
            seed = opts.seed + i * 1000
            gen = torch.Generator(device="cuda").manual_seed(seed)
            if progress:
                progress(0.1 + 0.8 * i / max(1, opts.count), f"FLUX.1-schnell image {i + 1}")
            if opts.image_path is not None:
                out = self._edit_pipe()(
                    prompt=opts.prompt,
                    negative_prompt=opts.negative or None,
                    image=Image.open(opts.image_path).convert("RGB"),
                    strength=min(max(opts.image_strength or 0.6, 0.05), 0.99),
                    num_inference_steps=opts.steps or 8,
                    guidance_scale=opts.guidance if opts.guidance is not None else 0.0,
                    generator=gen,
                )
            else:
                out = self._pipe(
                    prompt=opts.prompt,
                    negative_prompt=opts.negative or None,
                    num_inference_steps=opts.steps or 4,
                    guidance_scale=opts.guidance if opts.guidance is not None else 0.0,
                    width=opts.width,
                    height=opts.height,
                    generator=gen,
                )
            images.append(out.images[0])
        return ImageGenResult(
            images=images,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=opts.steps or 4,
            guidance=opts.guidance,
        )
