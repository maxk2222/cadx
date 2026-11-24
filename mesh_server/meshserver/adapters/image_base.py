"""Adapter interface for text-to-image backends (second adapter family).

Shares the registry's single-loaded-model invariant with the mesh adapters:
instances hold a heavyweight pipeline and must cooperate on GPU memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from meshserver.adapters.base import ProgressFn


@dataclass
class ImageGenOpts:
    prompt: str = ""
    negative: str = ""
    seed: int = 0
    count: int = 1  # 1..4 (clamped to adapter.max_count by the job runner)
    width: int = 768
    height: int = 768
    steps: int | None = None
    guidance: float | None = None
    work_dir: Path | None = None
    # image-edit (img2img / instruction-edit) inputs — only honoured by
    # adapters with supports_edit=True (flux-kontext-mlx, sd_* img2img)
    image_path: Path | None = None
    image_strength: float | None = None  # img2img noise level 0..1


@dataclass
class ImageGenResult:
    images: list[Image.Image] = field(default_factory=list)  # PIL images
    model: str = ""
    backend: str = ""
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# -- shared diffusers CPU-offload ladder ---------------------------------- #
# Each torch adapter declares per-tier VRAM peaks (MB) in _OFFLOAD_MB:
# "none" = full GPU residency, "model" = whole-module offload (peak =
# largest single module), "sequential" = per-layer streaming.


def resolve_offload(cfg: Any, total_vram: float | None, tiers: dict[str, int]) -> str:
    """Pick the offload tier for cfg.cuda_offload.

    Explicit modes pass through. "auto" picks the FASTEST tier whose
    declared peak fits the card's *total* VRAM plus the
    cuda_offload_vram_gb margin — total, not transient free: the queue's
    VRAM admission (vram_need_mb) evicts other residents until the declared
    peak is really free at load time, so probing free VRAM here would
    double-count evictions and drift between admission and load.
    """
    mode = str(getattr(cfg, "cuda_offload", "auto") or "auto")
    if mode in tiers:
        return mode
    if total_vram is None:
        return "model" if "model" in tiers else "sequential"
    margin_mb = float(getattr(cfg, "cuda_offload_vram_gb", 3) or 0) * 1000
    for tier in ("none", "model", "sequential"):
        peak = tiers.get(tier)
        if peak is not None and peak + margin_mb <= total_vram:
            return tier
    return "sequential"


def offload_peak_mb(cfg: Any, total_vram: float | None, tiers: dict[str, int]) -> int:
    """VRAM peak (MB) of the tier resolve_offload selects — the number the
    queue's VRAM admission must hold free before the load starts."""
    return tiers[resolve_offload(cfg, total_vram, tiers)]


class ImageAdapter(ABC):
    """One text-to-image backend. Instances hold the loaded pipeline."""

    id: str = "?"
    name: str = "?"
    device: str = "?"  # cuda | mps | mlx | cpu | fake
    backend: str = "?"  # short backend tag reported to clients
    kind: ClassVar[str] = "image"
    license_note: str = ""
    default_size: int = 768
    max_count: int = 4
    max_side: int = 1024  # request clamp: width/height <= max_side
    supports_edit: ClassVar[bool] = False  # accepts an init image to edit
    requires_image: ClassVar[bool] = False  # edit-only adapters (no txt2img)
    # declared resident footprint in MB — the memory ledger prefers the
    # measured peak RSS once a generation subprocess has reported one
    est_mem_mb: int | None = None

    @classmethod
    def available(cls, cfg: Any) -> bool:
        """Are dependencies + weights usable in this environment?"""
        return False

    @classmethod
    def create(cls, cfg: Any) -> "ImageAdapter":
        return cls(cfg)  # type: ignore[call-arg]

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load weights / build the pipeline."""
        self._loaded = True

    def unload(self) -> None:
        """Free GPU memory (best-effort)."""
        self._loaded = False

    @abstractmethod
    def generate(
        self,
        opts: ImageGenOpts,
        progress: ProgressFn | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ImageGenResult:
        """Blocking generation call — run inside a worker thread."""
        raise NotImplementedError
