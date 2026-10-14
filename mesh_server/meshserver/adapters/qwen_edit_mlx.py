"""Qwen-Image-Edit-2509 via mflux (MLX, Apple Silicon) — instruction-based
image editing: init image + text instruction ("make it black anodized
aluminum") -> edited image.

Unlike FLUX.1-Kontext the weights are Apache-2.0 and ungated on HF, so this
is the default edit adapter on MLX hosts (the registry's edit ladder prefers
it). Edit-only (`requires_image=True`): it never serves plain txt2img jobs.

Install: same mflux extra as FLUX.1-schnell
    pip install 'meshserver[image-mlx]'   or   mesh_server/scripts/install_image_mlx.sh
First load downloads ~10-14 GB (4-bit quantized) to ~/.cache/mflux.
"""

from __future__ import annotations

import platform
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


class QwenImageEditAdapter(ImageAdapter):
    id = "qwen-image-edit"
    name = "Qwen-Image-Edit 2509 (mflux, MLX)"
    device = "mlx"
    backend = "mflux"
    license_note = "Apache-2.0 (Qwen-Image-Edit-2509)"
    default_size = 1024
    max_count = 2
    max_side = 1024
    supports_edit: ClassVar[bool] = True
    requires_image: ClassVar[bool] = True

    @classmethod
    def available(cls, cfg: Any) -> bool:  # noqa: ARG003
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            from mflux.models.qwen.variants.edit.qwen_image_edit import (  # noqa: F401
                QwenImageEdit,
            )
        except Exception:
            return False
        return True

    def load(self) -> None:
        quant = int(getattr(self.cfg, "flux_mlx_quant", 4) or 4)
        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
        except ImportError as e:  # pragma: no cover
            missing(type(self), f"mflux qwen-edit import failed: {e}")
        errors: list[str] = []
        for mc_name in ("qwen_image_edit_2511", "qwen_image_edit"):
            mc = getattr(ModelConfig, mc_name, None)
            if mc is None:
                continue
            try:
                self._pipe = QwenImageEdit(quantize=quant, model_config=mc())
                self._loaded = True
                return
            except TypeError as e:
                errors.append(f"QwenImageEdit({mc_name}): {e}")
        try:
            self._pipe = QwenImageEdit(quantize=quant)
        except TypeError as e:  # pragma: no cover
            missing(type(self), "no working QwenImageEdit ctor: " + "; ".join(errors + [str(e)]))
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False
        hard_unload()

    def generate(
        self,
        opts: ImageGenOpts,
        progress=None,
        cancelled=None,
    ) -> ImageGenResult:
        if opts.image_path is None:
            raise RuntimeError("qwen-image-edit requires an init image")
        if not self._loaded:
            self.load()
        images = []
        for i in range(max(1, opts.count)):
            if cancelled is not None and cancelled():
                break
            if progress:
                progress(0.1 + 0.8 * i / max(1, opts.count), f"Qwen-Image-Edit {i + 1}")
            steps = opts.steps or 28
            out = self._pipe.generate_image(
                seed=(opts.seed + i * 1000) & 0xFFFFFFFF,
                prompt=opts.prompt,
                image_paths=[str(opts.image_path)],
                num_inference_steps=steps,
                height=min(opts.height, self.max_side),
                width=min(opts.width, self.max_side),
                guidance=opts.guidance if opts.guidance is not None else 4.0,
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
            steps=opts.steps or 28,
            guidance=opts.guidance if opts.guidance is not None else 4.0,
        )
