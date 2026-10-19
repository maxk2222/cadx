"""Qwen-Image-Edit-2509 via mflux (MLX, Apple Silicon) — instruction-based
image editing: init image + text instruction ("make it black anodized
aluminum") -> edited image.

Unlike FLUX.1-Kontext the weights are Apache-2.0 and ungated on HF, so this
is the default edit adapter on MLX hosts (the registry's edit ladder prefers
it). Edit-only (`requires_image=True`): it never serves plain txt2img jobs.

Install: same mflux extra as FLUX.1-schnell
    pip install 'meshserver[image-mlx]'   or   mesh_server/scripts/install_image_mlx.sh
First load downloads ~10-14 GB (4-bit quantized) to ~/.cache/mflux.

When ServerConfig.qwen_edit_lightning is on (default; env
MESH_QWEN_EDIT_LIGHTNING) the LightX2V "Lightning" distillation LoRA is
baked into the transformer at load time, collapsing the edit schedule
from ~28 steps to 8 — roughly 4x faster per synth view. The LoRA file is
resolved with hf_hub_download (single safetensors, not a repo snapshot —
the lightx2v repo bundles ~20 variants) and passed as a local lora path.
Falls back to the un-distilled 28-step model when the download fails.
"""

from __future__ import annotations

import logging
import platform
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import hard_unload, missing
from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult

logger = logging.getLogger(__name__)

# bf16 variant — matches the MLX compute dtype; the fp32 twin doubles the
# download for no quality gain at inference.
_LIGHTNING_REPO = "lightx2v/Qwen-Image-Lightning"
_LIGHTNING_FILE = (
    "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
)
LIGHTNING_STEPS = 8


def _resolve_lightning_lora() -> str | None:
    """Download just the 8-step edit LoRA file; None when unreachable."""
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=_LIGHTNING_REPO, filename=_LIGHTNING_FILE)
    except Exception as e:  # noqa: BLE001 — any failure means "no LoRA"
        logger.warning("qwen_edit_lightning_lora_unavailable", error=str(e)[:200])
        return None


def _patch_mflux_tf4_qwen2() -> None:
    """mflux ships a Qwen2Tokenizer workaround written for the transformers-5
    constructor signature (``cls(vocab=…, merges=…)``). Under transformers<5
    that call raises TypeError before reaching the working ``from_pretrained``
    path, so every Qwen tokenizer load fails. Restore the standard loader when
    running on transformers 4.x — the workaround is a no-op there anyway
    (tf4 reads vocab/merges from the files correctly)."""
    try:
        import transformers

        if int(transformers.__version__.split(".")[0]) >= 5:
            return
        from mflux.models.common.tokenizer import tokenizer_loader as _tl
    except Exception:
        return

    def _tf4_load(tokenizer_path, cls):  # noqa: ANN001, ANN202
        return cls.from_pretrained(
            pretrained_model_name_or_path=str(tokenizer_path),
            local_files_only=True,
        )

    _tl.TokenizerLoader._load_qwen2_tokenizer_workaround = staticmethod(_tf4_load)


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
    est_mem_mb = 16000
    # set by load() once the Lightning LoRA actually made it into the pipe;
    # class default keeps direct _pipe injection in tests working
    _lightning: bool = False

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
        _patch_mflux_tf4_qwen2()
        self._lightning = False
        lora_kw: dict[str, Any] = {}
        if getattr(self.cfg, "qwen_edit_lightning", False):
            lora_path = _resolve_lightning_lora()
            if lora_path:
                lora_kw = {"lora_paths": [lora_path], "lora_scales": [1.0]}
                self._lightning = True
        # ModelConfig moved between mflux releases; it is optional anyway —
        # the bare ctor falls back to the default edit config on every
        # version that ships QwenImageEdit at all.
        try:
            from mflux.models.common.config.model_config import (
                ModelConfig,
            )  # mflux >= 0.14
        except ImportError:
            try:
                from mflux.config.model_config import ModelConfig  # mflux <= 0.12.x
            except ImportError:
                ModelConfig = None  # type: ignore[assignment,misc]
        try:
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
        except ImportError as e:  # pragma: no cover
            missing(type(self), f"mflux qwen-edit import failed: {e}")
        errors: list[str] = []
        for mc_name in ("qwen_image_edit_2511", "qwen_image_edit"):
            mc = (
                getattr(ModelConfig, mc_name, None) if ModelConfig is not None else None
            )
            if mc is None:
                continue
            try:
                self._pipe = QwenImageEdit(quantize=quant, model_config=mc(), **lora_kw)
                self._loaded = True
                return
            except TypeError as e:
                errors.append(f"QwenImageEdit({mc_name}): {e}")
        try:
            self._pipe = QwenImageEdit(quantize=quant, **lora_kw)
        except TypeError as e:  # pragma: no cover
            missing(
                type(self),
                "no working QwenImageEdit ctor: " + "; ".join(errors + [str(e)]),
            )
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
            # Lightning is distilled for exactly its NFE count at cfg 1.0 —
            # more steps or real guidance degrade instead of improving, so
            # pin both when the LoRA is baked in.
            steps = LIGHTNING_STEPS if self._lightning else (opts.steps or 28)
            guidance = (
                1.0
                if self._lightning
                else (opts.guidance if opts.guidance is not None else 4.0)
            )
            out = self._pipe.generate_image(
                seed=(opts.seed + i * 1000) & 0xFFFFFFFF,
                prompt=opts.prompt,
                image_paths=[str(opts.image_path)],
                num_inference_steps=steps,
                height=min(opts.height, self.max_side),
                width=min(opts.width, self.max_side),
                guidance=guidance,
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
            guidance=guidance,
        )
