"""Adapter registry: device detection, model auto-selection, lazy load/unload."""

from __future__ import annotations

import platform
import threading
import time

from meshserver.adapters.base import MeshAdapter, hard_unload
from meshserver.adapters.fake import FakeAdapter
from meshserver.adapters.fake_image import FakeImageAdapter
from meshserver.adapters.image_base import ImageAdapter
from meshserver.config import ServerConfig
from meshserver.schemas import ModelInfo
from meshserver.torch_util import is_rocm


def detect_device(cfg: ServerConfig) -> str:
    """auto -> cuda | rocm | mlx | mps | cpu."""
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch

        if torch.cuda.is_available():
            # ROCm builds expose the cuda device namespace — tell them apart
            # via torch.version.hip so adapter selection can skip CUDA-only
            # extensions (diso flash decoder, TRELLIS.2)
            return "rocm" if is_rocm() else "cuda"
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


def _is_fake(cls: type) -> bool:
    return str(getattr(cls, "id", "")).startswith("fake")


class Registry:
    """Knows every adapter class, instantiates available ones lazily.

    One heavyweight model stays loaded at a time across BOTH adapter families
    (mesh + image): loading a different model unloads all others first — the
    same VRAM/unified-memory budget is shared.
    """

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.device = detect_device(cfg)
        self._lock = threading.Lock()
        self._instances: dict[str, object] = {}
        self._last_used: dict[str, float] = {}
        self._failed: dict[str, str] = {}  # adapter id -> last load/OOM error
        self._classes: list[type] = self._adapter_classes() + self._image_adapter_classes()

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
        from meshserver.adapters.triposg_scribble import TripoSGScribbleAdapter

        classes += [
            Hunyuan3D21MlxAdapter,
            Hunyuan3D2MiniAdapter,
            Hunyuan3D2MiniTurboAdapter,
            Hunyuan3D2MvAdapter,
            Hunyuan3D2MvTurboAdapter,
            Trellis2Adapter,
            TripoSGAdapter,
            TripoSGScribbleAdapter,
        ]
        return classes

    @staticmethod
    def _image_adapter_classes() -> list[type[ImageAdapter]]:
        classes: list[type[ImageAdapter]] = [FakeImageAdapter]
        from meshserver.adapters.flux_diffusers import FluxSchnellAdapter
        from meshserver.adapters.flux_kontext_mlx import FluxKontextMlxAdapter
        from meshserver.adapters.flux_mlx import FluxSchnellMlxAdapter
        from meshserver.adapters.qwen_edit_mlx import QwenImageEditAdapter
        from meshserver.adapters.sd_torch import Sd15Adapter, SdxlAdapter

        classes += [
            FluxKontextMlxAdapter,
            QwenImageEditAdapter,
            FluxSchnellMlxAdapter,
            FluxSchnellAdapter,
            SdxlAdapter,
            Sd15Adapter,
        ]
        return classes

    def _probe(self, cls: type) -> bool:
        try:
            return cls.available(self.cfg)
        except Exception:
            return False

    def models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for cls in self._classes:
            if _is_fake(cls) and not self.cfg.fake:
                # the fake adapters are only listed when explicitly enabled
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
                    kind=getattr(cls, "kind", "mesh"),
                    supports_multiview=getattr(cls, "supports_multiview", False),
                    max_images=getattr(cls, "max_images", 0),
                    license_note=cls.license_note,
                    loaded=bool(inst and inst.loaded),
                    available=avail,
                    last_error=self._failed.get(cls.id),
                    default_size=getattr(cls, "default_size", None),
                    supports_edit=getattr(cls, "supports_edit", False),
                    requires_image=getattr(cls, "requires_image", False),
                    max_count=getattr(cls, "max_count", None),
                )
            )
        return out

    def available(self) -> list[type]:
        return [cls for cls in self._classes if self._probe(cls)]

    def select(self, requested: str, n_images: int = 1) -> type[MeshAdapter] | None:
        """Pick a mesh adapter class for a request. 'auto' chooses by device
        and skips models that failed to load/OOM'd earlier in this process."""
        all_mesh = {
            cls.id: cls for cls in self.available() if getattr(cls, "kind", "mesh") == "mesh"
        }
        if requested and requested != "auto":
            if requested in self._failed:
                # explicit request for an adapter that already stalled/OOM'd
                # in this process — retrying it just burns a full stall
                # window; surface the recorded failure instead
                return None
            cls = all_mesh.get(requested)
            return cls  # explicit but unavailable -> None (error to client)
        avail = {k: v for k, v in all_mesh.items() if k not in self._failed}
        # auto
        device = self.device
        pref: list[str]
        if device == "cuda":
            pref = ["trellis2", "triposg"]
        elif device == "rocm":
            # no TRELLIS.2 on AMD (CUDA-only extensions); the hunyuan torch
            # variants are the only multiview-capable adapters here
            pref = (
                ["hunyuan3d-2mv", "hunyuan3d-2mv-turbo", "triposg"]
                if n_images > 1
                else ["triposg", "hunyuan3d-2mini", "hunyuan3d-2mini-turbo"]
            )
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

    def select_image(
        self, requested: str, need_edit: bool = False
    ) -> type[ImageAdapter] | None:
        """Pick an image adapter class. Same rules as select(): 'auto' follows
        the device preference ladder and skips badlisted adapters; explicit
        requests surface their own error. Edit jobs (init image) only match
        adapters with supports_edit."""
        all_img = {
            cls.id: cls
            for cls in self.available()
            if getattr(cls, "kind", "mesh") == "image"
        }
        # edit-only adapters can't serve plain txt2img; non-edit adapters
        # can't serve edit jobs — filter by job type
        if need_edit:
            all_img = {
                k: v for k, v in all_img.items() if getattr(v, "supports_edit", False)
            }
        else:
            all_img = {
                k: v for k, v in all_img.items() if not getattr(v, "requires_image", False)
            }
        if requested and requested != "auto":
            cls = all_img.get(requested)
            if need_edit and cls is not None and not getattr(cls, "supports_edit", False):
                return None
            return cls
        avail = {k: v for k, v in all_img.items() if k not in self._failed}
        device = self.device
        if self.cfg.fake:
            # fixture runs must never touch a real model, even when installed
            pref = ["fake-image"]
        elif need_edit:
            # qwen-image-edit first: Apache-2.0 + ungated weights; kontext is
            # the better FLUX edit model but non-commercial and MESH_FLUX_KONTEXT
            # opt-in, sdxl/sd15 are plain img2img fallbacks. flux-schnell (the
            # diffusers build) supports img2img too and is the only FLUX edit
            # path on ROCm — the *-mlx edit adapters don't exist on Linux.
            if device in ("cuda", "mlx"):
                pref = [
                    "qwen-image-edit",
                    "flux-kontext-mlx",
                    "flux-schnell-mlx",
                    "sdxl",
                    "sd15",
                ]
            elif device == "rocm":
                pref = ["flux-schnell", "sdxl", "sd15"]
            else:
                pref = ["sdxl", "sd15"] if device == "mps" else ["sd15"]
        elif device in ("cuda", "rocm"):
            pref = ["flux-schnell", "sdxl", "sd15"]
        elif device == "mlx":
            pref = ["flux-schnell-mlx", "sdxl", "sd15"]
        elif device == "mps":
            pref = ["sdxl", "sd15"]
        else:
            pref = ["sd15"] if self.cfg.allow_cpu else []
        for pid in pref:
            if cls := avail.get(pid):
                return cls
        # no ladder hit — any remaining adapter is better than failing
        return next(iter(avail.values()), None)

    def get(self, cls: type) -> object:
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
                try:
                    inst.load()
                except Exception as e:
                    # remember the failure so 'auto' skips this adapter next
                    # time; purge whatever half-loaded state remains
                    self._failed[cls.id] = str(e)[:300]
                    try:
                        inst.unload()
                    except Exception:
                        pass
                    hard_unload()
                    raise
            self._last_used[cls.id] = time.monotonic()
            return inst

    def mark_failed(self, adapter_id: str, error: str) -> None:
        """Record an inference-time failure (e.g. OOM inside generate())."""
        self._failed[adapter_id] = error[:300]

    def unload_all(self) -> None:
        """Unload every resident model + purge device caches (OOM recovery)."""
        with self._lock:
            for inst in self._instances.values():
                if inst.loaded:
                    try:
                        inst.unload()
                    except Exception:
                        pass
        hard_unload()

    def maybe_unload_idle(self) -> None:
        if self.cfg.unload_idle_s <= 0:
            return
        now = time.monotonic()
        with self._lock:
            for aid, inst in self._instances.items():
                if inst.loaded and now - self._last_used.get(aid, 0.0) > self.cfg.unload_idle_s:
                    try:
                        inst.unload()
                    except Exception:
                        pass
