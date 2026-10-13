"""FLUX.1-Kontext-dev via mflux (MLX, Apple Silicon) — instruction-based
image editing. Consumes an init image + a text instruction ("add a deeper
knurled grip texture") and returns the edited image; used by the backend's
`mesh_regen_edit` tool to steer mesh regeneration.

Edit-only adapter (`requires_image=True`): it never serves plain txt2img
jobs. Weights are large even at 4 bit, so availability is gated on
`MESH_FLUX_KONTEXT=1` — an unexpected multi-GB download must never happen
from a bare `available()` probe.
"""

from __future__ import annotations

import os
import platform
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


class FluxKontextMlxAdapter(ImageAdapter):
    id = "flux-kontext-mlx"
    name = "FLUX.1-Kontext-dev (mflux, MLX)"
    device = "mlx"
    backend = "mflux"
    license_note = "FLUX.1 [dev] Non-Commercial License"
    default_size = 768
    max_count = 1  # kontext edits one image per call
    max_side = 1024
    supports_edit: ClassVar[bool] = True
    requires_image: ClassVar[bool] = True

    @classmethod
    def available(cls, cfg: Any) -> bool:  # noqa: ARG003
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        if not os.environ.get("MESH_FLUX_KONTEXT"):
            return False
        try:
            from mflux.models.flux.variants.kontext.flux_kontext import (  # noqa: F401
                Flux1Kontext,
            )
        except Exception:
            return False
        return True

    def load(self) -> None:
        quant = int(getattr(self.cfg, "flux_kontext_quant", 4) or 4)
        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
        except ImportError as e:  # pragma: no cover
            missing(type(self), f"mflux kontext import failed: {e}")
        errors: list[str] = []
        for mc_name in ("dev_kontext", "kontext_dev"):
            mc = getattr(ModelConfig, mc_name, None)
            if mc is None:
                continue
            try:
                self._pipe = Flux1Kontext(quantize=quant, model_config=mc())
                self._loaded = True
                return
            except TypeError as e:
                errors.append(f"Flux1Kontext({mc_name}): {e}")
        try:
            self._pipe = Flux1Kontext(quantize=quant)
        except TypeError as e:  # pragma: no cover
            missing(type(self), "no working Flux1Kontext ctor: " + "; ".join(errors + [str(e)]))
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
            raise RuntimeError("flux-kontext requires an init image")
        if not self._loaded:
            self.load()
        if progress:
            progress(0.1, "running FLUX Kontext edit")
        steps = opts.steps or 20
        out = self._pipe.generate_image(
            seed=opts.seed & 0xFFFFFFFF,
            prompt=opts.prompt,
            num_inference_steps=steps,
            height=min(opts.height, self.max_side),
            width=min(opts.width, self.max_side),
            guidance=opts.guidance if opts.guidance is not None else 2.5,
            image_path=str(opts.image_path),
            image_strength=opts.image_strength,
        )
        img = out.image if hasattr(out, "image") else out
        if not isinstance(img, Image.Image):
            img = Image.open(getattr(out, "path", out))
        if progress:
            progress(0.95, "edit done")
        return ImageGenResult(
            images=[img],
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=steps,
            guidance=opts.guidance,
        )
