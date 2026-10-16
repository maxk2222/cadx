"""Adapter interfaces: image-to-3D (MeshAdapter) and text-to-image (ImageAdapter)."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

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
    kind: ClassVar[str] = "mesh"  # registry task family — image adapters use "image"
    supports_multiview: bool = False
    max_images: int = 1
    license_note: str = ""
    # adapter-native inference steps when the request leaves steps=None —
    # distilled ("turbo") models are built for ~5 steps; running them at 30
    # just burns wall-clock without improving geometry
    default_steps: int | None = None
    # canonical up-axis of the raw output frame ("y" for image-space
    # generators like Hunyuan3D) — clients use it as an orientation prior
    canonical_up: str | None = None

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


def missing(cls: type, what: str) -> None:
    raise RuntimeError(
        f"adapter '{cls.id}' is not available: {what}. "
        "Install the matching optional dependencies (see mesh_server/README.md)."
    )


def hard_unload() -> None:
    """gc + device cache clear — call after every adapter.unload() and on OOM.

    Plain `del`/None assignment only drops Python references; torch/MLX keep
    their allocator caches, so the freed memory is invisible to the next model
    load without an explicit cache purge.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
