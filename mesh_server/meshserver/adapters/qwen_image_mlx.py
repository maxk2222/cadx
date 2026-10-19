"""Qwen-Image text-to-image via mflux (MLX, Apple Silicon) — the default
txt2img adapter on MLX hosts.

Unlike FLUX.1-schnell's HF repo the weights (Qwen/Qwen-Image) are Apache-2.0
and ungated, so 'auto' can serve txt2img without an HF token — and the
~20B Qwen backbone follows prompts far more faithfully than SDXL. Also
accepts an init image (mflux generate_image takes image_path/
image_strength), so img2img edit jobs can land here when the dedicated
qwen-image-edit adapter is unavailable.

Install: same mflux extra as FLUX.1-schnell
    pip install 'meshserver[image-mlx]'   or   mesh_server/scripts/install_image_mlx.sh
First load downloads ~10-14 GB (4-bit quantized) to ~/.cache/mflux.

load() applies _patch_mflux_tf4_qwen2() from qwen_edit_mlx first — Qwen-Image
shares the Qwen2 tokenizer whose mflux workaround only works under
transformers>=5.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


class QwenImageMlxAdapter(ImageAdapter):
    id = "qwen-image"
    name = "Qwen-Image (mflux, MLX)"
    device = "mlx"
    backend = "mflux"
    license_note = "Apache-2.0 (Qwen-Image)"
    default_size = 1024
    max_count = 2
    max_side = 1328
    supports_edit: ClassVar[bool] = (
        True  # mflux generate_image takes image_path/strength
    )
    est_mem_mb = 16000  # same ~20B backbone as qwen-image-edit
    default_steps = 20  # not a distilled model — needs a real denoise schedule

    @classmethod
    def available(cls, cfg: Any) -> bool:  # noqa: ARG003
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            from mflux.models.qwen.variants.txt2img.qwen_image import (  # noqa: F401
                QwenImage,
            )
        except Exception:
            return False
        return True

    def load(self) -> None:
        from meshserver.adapters.qwen_edit_mlx import _patch_mflux_tf4_qwen2

        _patch_mflux_tf4_qwen2()
        quant = int(getattr(self.cfg, "flux_mlx_quant", 4) or 4)
        try:
            from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        except ImportError as e:  # pragma: no cover
            missing(type(self), f"mflux qwen-image import failed: {e}")
        try:
            # model_config defaults to ModelConfig.qwen_image() in the ctor
            self._pipe = QwenImage(quantize=quant)
        except TypeError:
            self._pipe = QwenImage(quantize=None)
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
        steps = opts.steps or 20
        images = []
        for i in range(max(1, opts.count)):
            if cancelled is not None and cancelled():
                break
            if progress:
                progress(0.1 + 0.8 * i / max(1, opts.count), f"Qwen-Image {i + 1}")
            out = self._pipe.generate_image(
                seed=(opts.seed + i * 1000) & 0xFFFFFFFF,
                prompt=opts.prompt,
                num_inference_steps=steps,
                height=min(opts.height, self.max_side),
                width=min(opts.width, self.max_side),
                guidance=opts.guidance if opts.guidance is not None else 4.0,
                negative_prompt=opts.negative or None,
                **(
                    {
                        "image_path": str(opts.image_path),
                        "image_strength": min(
                            max(opts.image_strength or 0.6, 0.05), 1.0
                        ),
                    }
                    if opts.image_path is not None
                    else {}
                ),
            )
            img = out.image if hasattr(out, "image") else out
            if not isinstance(img, Image.Image):
                img = Image.open(getattr(out, "path", out))
            images.append(img)
        return ImageGenResult(
            images=images,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=steps,
            guidance=opts.guidance if opts.guidance is not None else 4.0,
        )
