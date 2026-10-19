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
