"""Deterministic text-to-image test adapter — always available, no GPU needed.

Produces a seed-derived gradient PNG with the prompt rendered as text so
integration tests can exercise the whole /v1/image-jobs API surface.
"""

from __future__ import annotations

import hashlib
import textwrap
from collections.abc import Callable

from PIL import Image, ImageDraw

from meshserver.adapters.base import ProgressFn
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


class FakeImageAdapter(ImageAdapter):
    id = "fake-image"
    name = "FakeImage (test fixture)"
    device = "fake"
    backend = "fake"
    license_note = "MIT (test fixture)"
    default_size = 512
    max_count = 4
    max_side = 2048
    supports_edit = True  # echoes the init image — covers the edit API in tests

    @classmethod
    def available(cls, cfg) -> bool:  # noqa: ARG003
        return True

    def __init__(self, cfg=None) -> None:  # noqa: ARG002
        self._loaded = True

    def generate(
        self,
        opts: ImageGenOpts,
        progress: ProgressFn | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ImageGenResult:
        del cancelled
        if progress:
            progress(0.5, "fake image generation")
        if opts.image_path is not None:
            try:
                return ImageGenResult(
                    images=[Image.open(opts.image_path).convert("RGB")],
                    model=self.id,
                    backend=self.backend,
                    seed=opts.seed,
                    steps=opts.steps or 0,
                    guidance=opts.guidance,
                )
            except Exception as e:
                raise ValueError(f"fake edit: unreadable init image: {e}") from e
        imgs = [
            self._fake_png(opts.seed + i * 1000, opts.prompt, opts.width, opts.height)
            for i in range(max(1, opts.count))
        ]
        return ImageGenResult(
            images=imgs,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=opts.steps or 0,
            guidance=opts.guidance,
        )

    @staticmethod
    def _fake_png(seed: int, prompt: str, w: int, h: int) -> Image.Image:
        hval = int(hashlib.sha256(f"{seed}-{prompt}".encode()).hexdigest()[:8], 16)
        c0 = ((hval >> 16) & 0xFF, (hval >> 8) & 0xFF, hval & 0xFF)
        c1 = (255 - c0[0], 255 - c0[1], 255 - c0[2])
        img = Image.new("RGB", (w, h))
        px = img.load()
        for y in range(h):
            f = y / max(1, h - 1)
            row = tuple(int(c0[c] * (1 - f) + c1[c] * f) for c in range(3))
            for x in range(w):
                px[x, y] = row
        d = ImageDraw.Draw(img)
        for i, line in enumerate(textwrap.wrap(prompt or "fake image", 40)[:4]):
            d.text((8, 8 + i * 12), line, fill=(255, 255, 255))
        return img
