"""Z-Image-Turbo text-to-image via mflux (MLX, Apple Silicon) — the fast
default txt2img adapter on MLX hosts.

Tongyi-MAI/Z-Image-Turbo is a 6B distilled model (Apache-2.0, ungated)
needing only ~8 NFEs — versus ~20 steps on the ~20B Qwen-Image backbone
that was the previous default. On Apple Silicon that turns a ~13 minute
1024px generation into roughly a minute, at benchmark parity for
photorealism and prompt adherence — the difference that makes the
hero-image verify/retry loop affordable.

Install: same mflux extra as qwen-image
    pip install 'meshserver[image-mlx]'   or   mesh_server/scripts/install_image_mlx.sh
First load downloads ~4-8 GB (4-bit quantized) to ~/.cache/mflux.

Unlike FLUX/Qwen this is a distilled single-stream DiT: no CFG guidance
(supports_guidance=False), so `guidance` is always passed as None.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


class ZImageTurboAdapter(ImageAdapter):
    id = "z-image-turbo"
    name = "Z-Image-Turbo (mflux, MLX)"
    device = "mlx"
    backend = "mflux"
    license_note = "Apache-2.0 (Tongyi-MAI/Z-Image-Turbo)"
    default_size = 1024
    max_count = 4
    max_side = 1328
    supports_edit: ClassVar[bool] = (
        True  # mflux generate_image takes image_path/strength
    )
    est_mem_mb = 8000  # ~6B backbone at 4-bit + activations
    default_steps = 8  # distilled — the model card's official NFE count

    @classmethod
    def available(cls, cfg: Any) -> bool:  # noqa: ARG003
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            from mflux.models.z_image.variants.z_image import ZImage  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> None:
        quant = int(getattr(self.cfg, "flux_mlx_quant", 4) or 4)
        try:
            from mflux.models.z_image.variants.z_image import ZImage
        except ImportError as e:  # pragma: no cover
            missing(type(self), f"mflux z-image import failed: {e}")
        try:
            # model_config defaults to ModelConfig.z_image_turbo() in the ctor
            self._pipe = ZImage(quantize=quant)
        except TypeError:
            self._pipe = ZImage(quantize=None)
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
        steps = opts.steps or self.default_steps
        images = []
        for i in range(max(1, opts.count)):
            if cancelled is not None and cancelled():
                break
            if progress:
                progress(0.1 + 0.8 * i / max(1, opts.count), f"Z-Image-Turbo {i + 1}")
            out = self._pipe.generate_image(
                seed=(opts.seed + i * 1000) & 0xFFFFFFFF,
                prompt=opts.prompt,
                num_inference_steps=steps,
                height=min(opts.height, self.max_side),
                width=min(opts.width, self.max_side),
                # distilled model — no CFG; mflux expects None here
                guidance=None,
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
            guidance=None,
        )
