"""Adapter interface for image-to-3D backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh
from PIL import Image

ProgressFn = Callable[[float, str], None]  # (fraction 0..1, message)


@dataclass
class GenOpts:
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None
    subject: str = ""
    image_roles: list[str] = field(default_factory=list)
    work_dir: Path | None = None


@dataclass
class AdapterGenResult:
    mesh: trimesh.Trimesh
    model: str = ""
    backend: str = ""
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class MeshAdapter(ABC):
    """One model backend. Instances hold the loaded pipeline (GPU memory)."""

    id: str = "?"
    name: str = "?"
    device: str = "?"  # cuda | mps | mlx | cpu | fake
    backend: str = "?"  # short backend tag reported to clients
    supports_multiview: bool = False
    max_images: int = 1
    license_note: str = ""

    @classmethod
    def available(cls, cfg: Any) -> bool:
        """Are dependencies + weights usable in this environment?"""
        return False

    @classmethod
    def create(cls, cfg: Any) -> "MeshAdapter":
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
        images: list[Image.Image],
        opts: GenOpts,
        progress: ProgressFn | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AdapterGenResult:
        """Blocking generation call — run inside a worker thread."""
        raise NotImplementedError


def missing(cls: type[MeshAdapter], what: str) -> None:
    raise RuntimeError(
        f"adapter '{cls.id}' is not available: {what}. "
        "Install the matching optional dependencies (see mesh_server/README.md)."
    )
