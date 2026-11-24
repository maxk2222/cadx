"""Qwen-Image-Edit-2509 via diffusers (CUDA, ROCm) — instruction-based image
editing, the torch counterpart of qwen_edit_mlx.

Install:
    pip install 'meshserver[image-torch]'   (diffusers>=0.36)
Weights are pulled from HF on first load (Qwen/Qwen-Image-Edit-2509,
Apache-2.0, ungated — no HF token required).

Unlike img2img adapters (sdxl/flux-schnell), this is a true instruction-edit
model: it conditions on the init image and edits only what the prompt asks.
~20B params (~40 GB bf16 transformer) — model-level offload can't fit the
transformer alone on <=40 GB cards, so 'auto' resolves to 'sequential'
there (see image_base.resolve_offload).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.flux_diffusers import _cuda_ok
from meshserver.adapters.image_base import (
    ImageAdapter,
    ImageGenOpts,
    ImageGenResult,
    bnb_quant_config,
    offload_peak_mb,
    quant_tiers,
    resolve_offload,
)

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
    # worker RSS: offloaded weights stay resident in host RAM (~45 GB bf16)
    est_mem_mb = 40000
    # VRAM peaks per offload tier — the ~40 GB transformer makes 'model'
    # offload pointless on <=40 GB cards (its peak IS the whole model);
    # bnb8 halves it (~20 GB transformer, ~4 GB Qwen2.5-VL) so 'none' fits
    _OFFLOAD_MB: ClassVar[dict[str, int]] = {
        "none": 47000,
        "model": 42000,
        "sequential": 14000,
    }
    _OFFLOAD_MB_Q: ClassVar[dict[str, int]] = {
        "none": 26000,
        "model": 22000,
        "sequential": 10000,
    }

    @classmethod
    def vram_need_mb(cls, cfg: Any) -> int:
        from meshserver.memory import total_vram_mb

        return offload_peak_mb(
            cfg, total_vram_mb(), quant_tiers(cfg, cls._OFFLOAD_MB, cls._OFFLOAD_MB_Q)
        )
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
        quant = bnb_quant_config(self.cfg)
        kw: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": str(self.cfg.model_dir),
        }
        if quant is not None:
            # component-level: the ~20B transformer + the Qwen2.5-VL text
            # encoder via bnb — VAE stays bf16
            from diffusers import QwenImageTransformer2DModel
            from transformers import Qwen2_5_VLForConditionalGeneration

            transformer = QwenImageTransformer2DModel.from_pretrained(
                _HF_REPO, subfolder="transformer", quantization_config=quant, **kw
            )
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                _HF_REPO, subfolder="text_encoder", quantization_config=quant, **kw
            )
            pipe = pipe_cls.from_pretrained(
                _HF_REPO, transformer=transformer, text_encoder=text_encoder, **kw
            )
        else:
            pipe = pipe_cls.from_pretrained(_HF_REPO, **kw)
        offload = resolve_offload(
            self.cfg,
            torch.cuda.get_device_properties(0).total_memory / 1e6,
            quant_tiers(self.cfg, self._OFFLOAD_MB, self._OFFLOAD_MB_Q),
        )
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
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
