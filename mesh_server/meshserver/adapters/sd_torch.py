"""SDXL / SD1.5 text-to-image fallbacks via diffusers (CUDA | MPS | CPU).

Used when FLUX does not fit or is not installed. Both models are smaller and
work with diffusers' generic AutoPipelineForText2Image.

    sdxl  -> stabilityai/stable-diffusion-xl-base-1.0   (OpenRAIL++-M)
    sd15  -> runwayml/stable-diffusion-v1-5             (CreativeML OpenRAIL-M)

Install: pip install 'meshserver[image-torch]'
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


def _device_ok(device: str, cfg: Any) -> bool:
    try:
        import torch

        if device == "cuda":
            return bool(torch.cuda.is_available())
        if device == "mps":
            return bool(torch.backends.mps.is_available())
        return device == "cpu" and getattr(cfg, "allow_cpu", False)
    except Exception:
        return False


class _SdBase(ImageAdapter):
    repo: str = "?"
    device_candidates: tuple[str, ...] = ()

    @classmethod
    def available(cls, cfg: Any) -> bool:
        try:
            from diffusers import AutoPipelineForText2Image  # noqa: F401
        except Exception:
            return False
        return any(_device_ok(d, cfg) for d in cls.device_candidates)

    def _device(self) -> str:
        for d in self.device_candidates:
            if _device_ok(d, self.cfg):
                return d
        return self.device_candidates[0]

    def load(self) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        self._torch = torch
        dev = self._device()
        dtype = torch.float32 if dev == "cpu" else torch.float16
        kw: dict[str, Any] = {
            "torch_dtype": dtype,
            "cache_dir": str(self.cfg.model_dir),
        }
        if dev != "cpu":
            kw["variant"] = "fp16"
        pipe = AutoPipelineForText2Image.from_pretrained(self.repo, **kw)
        if dev == "mps":
            pipe.enable_model_cpu_offload()  # MPS: keep peak under unified RAM
        else:
            pipe = pipe.to(dev)
        self._pipe = pipe
        self._gen_device = dev
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
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
            if progress:
                progress(0.1 + 0.8 * i / max(1, opts.count), f"{self.id} image {i + 1}")
            gen = torch.Generator(device=self._gen_device).manual_seed(seed)
            out = self._pipe(
                prompt=opts.prompt,
                negative_prompt=opts.negative or None,
                num_inference_steps=opts.steps or 25,
                guidance_scale=opts.guidance if opts.guidance is not None else 7.5,
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
            steps=opts.steps or 25,
            guidance=opts.guidance if opts.guidance is not None else 7.5,
        )


class SdxlAdapter(_SdBase):
    id = "sdxl"
    name = "Stable Diffusion XL (diffusers)"
    device = "cuda"  # advertised device; mps/cpu handled via device_candidates
    backend = "diffusers"
    repo = "stabilityai/stable-diffusion-xl-base-1.0"
    license_note = "OpenRAIL++-M (Stability AI)"
    device_candidates = ("cuda", "mps")
    default_size = 1024
    max_count = 4
    max_side = 1024


class Sd15Adapter(_SdBase):
    id = "sd15"
    name = "Stable Diffusion 1.5 (diffusers)"
    device = "cuda"
    backend = "diffusers"
    repo = "runwayml/stable-diffusion-v1-5"
    license_note = "CreativeML OpenRAIL-M (Runway/CompVis)"
    device_candidates = ("cuda", "mps", "cpu")
    default_size = 512
    max_count = 4
    max_side = 768
