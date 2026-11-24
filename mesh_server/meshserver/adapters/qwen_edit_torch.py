"""Qwen-Image-Edit-2509 via diffusers (CUDA, ROCm) — instruction-based image
editing, the torch counterpart of qwen_edit_mlx.

Install:
    pip install 'meshserver[image-torch]'   (diffusers>=0.36)
Weights are pulled from HF on first load (Qwen/Qwen-Image-Edit-2509,
Apache-2.0, ungated — no HF token required).

Unlike img2img adapters (sdxl/flux-schnell), this is a true instruction-edit
model: it conditions on the init image and edits only what the prompt asks.
~20B params — always run with CPU offload on <=32 GB cards; the shared
cuda_offload ladder applies (auto -> 'model' offload below the VRAM cutoff).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.flux_diffusers import _cuda_ok
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult

_HF_REPO = "Qwen/Qwen-Image-Edit-2509"


class QwenImageEditTorchAdapter(ImageAdapter):
    id = "qwen-image-edit-torch"
    name = "Qwen-Image-Edit-2509 (diffusers)"
    device = "cuda"  # torch cuda namespace (covers ROCm builds too)
    backend = "diffusers"
    license_note = "Apache-2.0 (Qwen-Image-Edit-2509)"
    default_size = 1024
    max_count = 2
    max_side = 1024
    supports_edit: ClassVar[bool] = True
    requires_image: ClassVar[bool] = True  # edit-only: no plain txt2img
    est_mem_mb = 16000
    # set by load() from the resolved pipeline class; default matches the
    # Plus signature so direct _pipe injection in tests works
    _image_list: bool = True

    @classmethod
    def available(cls, cfg: Any) -> bool:
        try:
            from diffusers import QwenImageEditPlusPipeline  # noqa: F401
        except Exception:
            try:
                from diffusers import QwenImageEditPipeline  # noqa: F401
            except Exception:
                return False
        return _cuda_ok()

    def load(self) -> None:
        try:
            import torch
            from diffusers import QwenImageEditPipeline

            try:
                from diffusers import QwenImageEditPlusPipeline as pipe_cls
            except ImportError:
                pipe_cls = QwenImageEditPipeline
        except Exception as e:  # pragma: no cover
            missing(type(self), str(e))
        self._torch = torch
        # the Plus pipeline (2509) takes image lists; the base edit pipeline
        # takes a single PIL image — remember which signature to use
        self._image_list = pipe_cls.__name__.endswith("PlusPipeline")
        pipe = pipe_cls.from_pretrained(
            _HF_REPO,
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
        # ~20B bf16 — model offload is the sensible default; 'none' only fits
        # ~40 GB+ cards or when the LLM is unloaded
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        self._pipe = pipe
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
        if opts.image_path is None:
            raise RuntimeError("qwen-image-edit-torch requires an init image")
        if not self._loaded:
            self.load()
        torch = self._torch
        init_image = Image.open(opts.image_path).convert("RGB")
        # instruction-edit guidance lives in true_cfg_scale; guidance_scale
        # stays at the pipeline's distilled default (1.0)
        true_cfg = opts.guidance if opts.guidance is not None else 4.0
        steps = opts.steps or 40
        images = []
        for i in range(max(1, opts.count)):
            if cancelled is not None and cancelled():
                break
            gen = torch.Generator(device="cuda").manual_seed(opts.seed + i * 1000)
            if progress:
                progress(0.1 + 0.8 * i / max(1, opts.count), f"Qwen-Image-Edit {i + 1}")
            out = self._pipe(
                image=[init_image] if self._image_list else init_image,
                prompt=opts.prompt,
                negative_prompt=opts.negative or " ",
                num_inference_steps=steps,
                true_cfg_scale=true_cfg,
                height=min(opts.height, self.max_side),
                width=min(opts.width, self.max_side),
                generator=gen,
            )
            images.append(out.images[0])
        return ImageGenResult(
            images=images,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=steps,
            guidance=true_cfg,
        )
