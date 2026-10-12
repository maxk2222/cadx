"""FLUX.1-schnell text-to-image via mflux on Apple Silicon (MLX).

Install:
    pip install 'meshserver[image-mlx]'
or  mesh_server/scripts/install_image_mlx.sh

Weights download from HF on first load (black-forest-labs/FLUX.1-schnell,
Apache-2.0, ungated). Unified memory means the "VRAM" budget is system RAM —
the quantized weight file is the lever (MESH_FLUX_MLX_QUANT, default 4 bit
≈ 10-12 GB resident); schnell needs only 2-4 inference steps.

The mflux Python entry point moved between releases (Flux1/ModelConfig lived
under mflux.flux.flux / mflux.config in older versions, mflux.models.* in
newer ones) — load() and generate() probe the known shapes defensively so a
version bump degrades to 'unavailable' or a clear error instead of a crash.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from typing import Any

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult


class FluxSchnellMlxAdapter(ImageAdapter):
    id = "flux-schnell-mlx"
    name = "FLUX.1-schnell (mflux, MLX)"
    device = "mlx"
    backend = "mflux"
    license_note = "Apache-2.0 (FLUX.1-schnell)"
    default_size = 768
    max_count = 4
    max_side = 1024

    @classmethod
    def available(cls, cfg: Any) -> bool:  # noqa: ARG003
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            import mflux  # noqa: F401
        except Exception:
            return False
        return True

    # -- loading ---------------------------------------------------------- #

    def _model_config(self) -> Any:
        for modname in ("mflux.config.model_config", "mflux.config"):
            try:
                mod = __import__(modname, fromlist=["ModelConfig"])
                mc_cls = getattr(mod, "ModelConfig", None)
                if mc_cls is not None and hasattr(mc_cls, "schnell"):
                    return mc_cls.schnell()
            except Exception:
                continue
        return None

    def load(self) -> None:
        quant = getattr(self.cfg, "flux_mlx_quant", 4)
        errors: list[str] = []
        pipe = self._build_pipe(quant, errors)
        if pipe is None:  # pragma: no cover
            missing(type(self), "no working mflux entry point: " + "; ".join(errors[:3]))
        self._pipe = pipe
        limit = getattr(self.cfg, "mlx_mem_limit_gb", 0)
        if limit:
            try:
                import mlx.core as mx

                mx.set_memory_limit(int(limit * 1e9))
            except Exception:
                pass
        self._loaded = True

    def _build_pipe(self, quant: int, errors: list[str]) -> Any | None:
        mc = self._model_config()
        candidates: list[tuple[Any, list[dict[str, Any]]]] = []
        try:
            from mflux.flux.flux import Flux1

            kwargs = [{"model_config": mc, "quantize": quant}, {"model_config": mc}, {}]
            candidates.append((Flux1, kwargs))
        except ImportError as e:
            errors.append(f"mflux.flux.flux: {e}")
        try:
            from mflux.models.flux import Flux1 as Flux1New

            if hasattr(Flux1New, "from_pretrained"):
                for kw in ({"quantize": quant}, {}):
                    try:
                        return Flux1New.from_pretrained("schnell", **kw)
                    except TypeError as e:
                        errors.append(f"Flux1.from_pretrained: {e}")
            candidates.append((Flux1New, [{"quantize": quant}, {}]))
        except ImportError as e:
            errors.append(f"mflux.models.flux: {e}")
        for cls, kw_list in candidates:
            for kw in kw_list:
                if kw.get("model_config") is None and "model_config" in kw:
                    continue
                try:
                    return cls(**kw)
                except TypeError as e:
                    errors.append(f"{cls.__name__}: {e}")
        return None

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False
        hard_unload()

    # -- generation -------------------------------------------------------- #

    def _steps_config(self, steps: int, opts: ImageGenOpts) -> Any | None:
        for modname in ("mflux.config.config", "mflux.config"):
            try:
                mod = __import__(modname, fromlist=["Config"])
                cfg_cls = getattr(mod, "Config", None)
                if cfg_cls is None:
                    continue
                for kw in (
                    {"num_inference_steps": steps, "width": opts.width, "height": opts.height},
                    {"num_inference_steps": steps},
                    {},
                ):
                    try:
                        return cfg_cls(**kw)
                    except TypeError:
                        continue
            except Exception:
                continue
        return None

    def _call_generate(self, opts: ImageGenOpts, seed: int) -> Image.Image:
        pipe = self._pipe
        steps = opts.steps or 4
        kwargs: dict[str, Any] = {"prompt": opts.prompt, "seed": seed}
        attempts = [
            lambda: pipe.generate_image(
                num_inference_steps=steps, width=opts.width, height=opts.height, **kwargs
            ),
            lambda: pipe.generate_image(num_inference_steps=steps, **kwargs),
            lambda: pipe.generate_image(config=self._steps_config(steps, opts), **kwargs),
        ]
        for attempt in attempts:
            try:
                out = attempt()
            except (TypeError, ValueError):
                continue
            return out.image if hasattr(out, "image") else out
        out = pipe.generate_image(**kwargs)  # last resort — surfaces the real error
        return out.image if hasattr(out, "image") else out

    def generate(
        self,
        opts: ImageGenOpts,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ImageGenResult:
        if not self._loaded:
            self.load()
        images = []
        for i in range(max(1, opts.count)):
            if cancelled is not None and cancelled():
                break
            seed = opts.seed + i * 1000
            if progress:
                progress(0.1 + 0.8 * i / max(1, opts.count), f"FLUX.1-schnell image {i + 1}")
            images.append(self._call_generate(opts, seed))
        return ImageGenResult(
            images=images,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=opts.steps or 4,
            guidance=opts.guidance,
        )
