"""Adapter registry: device detection, model auto-selection, lazy load/unload."""

from __future__ import annotations

import platform
import threading
import time

from meshserver.adapters.base import MeshAdapter
from meshserver.adapters.fake import FakeAdapter
from meshserver.config import ServerConfig
from meshserver.schemas import ModelInfo


def detect_device(cfg: ServerConfig) -> str:
    """auto -> cuda | mlx | mps | cpu."""
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401

            return "mlx"
        except Exception:
            pass
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


class Registry:
    """Knows every adapter class, instantiates available ones lazily."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.device = detect_device(cfg)
        self._lock = threading.Lock()
        self._instances: dict[str, MeshAdapter] = {}
        self._last_used: dict[str, float] = {}
        self._classes: list[type[MeshAdapter]] = self._adapter_classes()

    @staticmethod
    def _adapter_classes() -> list[type[MeshAdapter]]:
        classes: list[type[MeshAdapter]] = [FakeAdapter]
        from meshserver.adapters.hunyuan3d2_mps import (
            Hunyuan3D2MiniAdapter,
            Hunyuan3D2MiniTurboAdapter,
            Hunyuan3D2MvAdapter,
            Hunyuan3D2MvTurboAdapter,
        )
        from meshserver.adapters.hunyuan3d21_mlx import Hunyuan3D21MlxAdapter
        from meshserver.adapters.trellis2_cuda import Trellis2Adapter
        from meshserver.adapters.triposg import TripoSGAdapter

        classes += [
            Hunyuan3D21MlxAdapter,
            Hunyuan3D2MiniAdapter,
            Hunyuan3D2MiniTurboAdapter,
            Hunyuan3D2MvAdapter,
            Hunyuan3D2MvTurboAdapter,
            Trellis2Adapter,
            TripoSGAdapter,
        ]
        return classes

    def _probe(self, cls: type[MeshAdapter]) -> bool:
        try:
            return cls.available(self.cfg)
        except Exception:
            return False

    def models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for cls in self._classes:
            if cls.id == "fake" and not self.cfg.fake:
                # the fake adapter is only listed when explicitly enabled
                avail = False
            else:
                avail = self._probe(cls)
            inst = self._instances.get(cls.id)
            out.append(
                ModelInfo(
                    id=cls.id,
                    name=cls.name,
                    device=cls.device,
                    backend=cls.backend,
                    supports_multiview=cls.supports_multiview,
                    max_images=cls.max_images,
                    license_note=cls.license_note,
                    loaded=bool(inst and inst.loaded),
                    available=avail,
                )
            )
        return out

    def available(self) -> list[type[MeshAdapter]]:
        return [cls for cls in self._classes if self._probe(cls)]

    def select(self, requested: str, n_images: int = 1) -> type[MeshAdapter] | None:
        """Pick an adapter class for a request. 'auto' chooses by device."""
        avail = {cls.id: cls for cls in self.available()}
        if requested and requested != "auto":
            cls = avail.get(requested)
            return cls  # explicit but unavailable -> None (error to client)
        # auto
        device = self.device
        pref: list[str]
        if device == "cuda":
            pref = ["trellis2", "triposg"]
        elif device == "mlx":
            pref = ["hunyuan3d-2.1-mlx", "hunyuan3d-2mv", "hunyuan3d-2mini", "triposg"]
        elif device == "mps":
            pref = (
                ["hunyuan3d-2mv", "hunyuan3d-2mv-turbo"]
                if n_images > 1
                else ["hunyuan3d-2mini", "hunyuan3d-2mini-turbo", "triposg"]
            )
        elif self.cfg.fake:
            pref = ["fake"]
        else:
            pref = []
        for pid in pref:
            cls = avail.get(pid)
            if cls and (cls.supports_multiview or n_images <= 1):
                return cls
        return avail.get("fake") if self.cfg.fake else None

    def get(self, cls: type[MeshAdapter]) -> MeshAdapter:
        """Instantiate + load the adapter; only one model stays loaded."""
        with self._lock:
            inst = self._instances.get(cls.id)
            if inst is None:
                inst = cls.create(self.cfg)
                self._instances[cls.id] = inst
            if not inst.loaded:
                # free other models first — one heavy model at a time
                for other_id, other in self._instances.items():
                    if other_id != cls.id and other.loaded:
                        try:
                            other.unload()
                        except Exception:
                            pass
                inst.load()
            self._last_used[cls.id] = time.monotonic()
            return inst

    def maybe_unload_idle(self) -> None:
        if self.cfg.unload_idle_s <= 0:
            return
        now = time.monotonic()
        with self._lock:
            for aid, inst in self._instances.items():
                if (
                    inst.loaded
                    and now - self._last_used.get(aid, 0.0) > self.cfg.unload_idle_s
                ):
                    try:
                        inst.unload()
                    except Exception:
                        pass
