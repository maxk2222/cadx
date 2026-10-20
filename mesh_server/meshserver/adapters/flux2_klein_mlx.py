"""FLUX.2-klein-4B edit via mflux (MLX, Apple Silicon) — the fast
instruction-edit adapter.

FLUX.2-klein-4B (Black Forest Labs, Jan 2026, Apache-2.0, ungated) is a
4B rectified-flow MMDiT distilled to 4 steps — versus ~50s/step on the
~20B Qwen-Image-Edit backbone that was the previous edit default. On
Apple Silicon a 4-step 768px edit takes tens of seconds instead of ~8
minutes, which is what makes the synth-side-view loop affordable. Edit
support is native (image_paths reference conditioning, multi-image).

Install: same mflux extra as qwen-image-edit / z-image-turbo
    pip install 'meshserver[image-mlx]'
First load downloads ~9 GB (4-bit quantized weights + encoders).
"""

from __future__ import annotations

import platform
from typing import ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


class Flux2KleinEditAdapter(ImageAdapter):
    id = "flux2-klein-4b-edit"
    name = "FLUX.2-klein-4B Edit (mflux, MLX)"
    device = "mlx"
    backend = "mflux"
    license_note = "Apache-2.0 (FLUX.2-klein-4B)"
    default_size = 768
    max_count = 4
    max_side = 1024
    supports_edit: ClassVar[bool] = True
    requires_image: ClassVar[bool] = True  # edit-only, like qwen-image-edit
    est_mem_mb = 9000  # 4B transformer at 4-bit + Qwen3 text encoder + VAE
    default_steps = 4  # distilled — the model card's official NFE count

    @classmethod
    def available(cls, cfg) -> bool:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            from mflux.models.flux2.variants import Flux2KleinEdit  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> None:
        quant = int(getattr(self.cfg, "flux_mlx_quant", 4) or 4)
        try:
            from mflux.models.common.config import ModelConfig
            from mflux.models.flux2.variants import Flux2KleinEdit
        except ImportError as e:  # pragma: no cover
            missing(type(self), f"mflux flux2-klein import failed: {e}")
        try:
            self._pipe = Flux2KleinEdit(
                quantize=quant, model_config=ModelConfig.flux2_klein_4b()
            )
        except TypeError:
            self._pipe = Flux2KleinEdit(quantize=quant)
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
            raise RuntimeError("flux2-klein-4b-edit requires an init image")
        if not self._loaded:
            self.load()
        steps = opts.steps or self.default_steps
        images = []
        for i in range(max(1, opts.count)):
            if cancelled is not None and cancelled():
                break
            if progress:
                progress(
                    0.1 + 0.8 * i / max(1, opts.count), f"FLUX.2-klein-edit {i + 1}"
                )
            out = self._pipe.generate_image(
                seed=(opts.seed + i * 1000) & 0xFFFFFFFF,
                prompt=opts.prompt,
                num_inference_steps=steps,
                height=min(opts.height, self.max_side),
                width=min(opts.width, self.max_side),
                image_paths=[str(opts.image_path)],
                **({"image_strength": opts.image_strength} if opts.image_strength else {}),
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
            guidance=None,
        )
