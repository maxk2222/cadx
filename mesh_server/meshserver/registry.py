"""Adapter registry: device detection, model auto-selection, lazy load/unload."""

from __future__ import annotations

import contextlib
import platform
import threading
import time

from meshserver.adapters.base import MeshAdapter, hard_unload
from meshserver.adapters.fake import FakeAdapter
from meshserver.adapters.fake_image import FakeImageAdapter
from meshserver.adapters.image_base import ImageAdapter
from meshserver.config import ServerConfig
from meshserver.logging import get_logger
from meshserver.memory import MemoryLedger
from meshserver.schemas import ModelInfo
from meshserver.torch_util import is_rocm

log = get_logger(__name__)


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
        self._probe_errors: dict[str, str] = {}  # adapter id -> availability-check error
        self._classes: list[type] = self._adapter_classes() + self._image_adapter_classes()
        # llm adapter family — kept separate so select()/select_image() never
        # see kind="llm" classes; refresh_llm_classes() re-reads the catalog +
        # custom_models.json (app.py calls it after register/unregister)
        self._llm_classes: list[type] = []
        self.refresh_llm_classes()
        # optional DownloadManager handle — app.py sets it post-construction;
        # models() reads it for downloading/size_gb
        self.downloads: object | None = None
        # pinned models are never evicted for memory pressure (llm pin)
        self._pinned: set[str] = set()
        # learned footprints + resident bookkeeping for admission control
        self.ledger = MemoryLedger(cfg)

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
        from meshserver.adapters.hunyuan_omni import Hunyuan3DOmniAdapter
        from meshserver.adapters.step1x_3d import Step1X3DAdapter
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
            Hunyuan3DOmniAdapter,
            Step1X3DAdapter,
        ]
        return classes

    @staticmethod
    def _image_adapter_classes() -> list[type[ImageAdapter]]:
        classes: list[type[ImageAdapter]] = [FakeImageAdapter]
        from meshserver.adapters.flux2_klein_mlx import Flux2KleinEditAdapter
        from meshserver.adapters.flux_diffusers import FluxSchnellAdapter
        from meshserver.adapters.flux_kontext_mlx import FluxKontextMlxAdapter
        from meshserver.adapters.flux_mlx import FluxSchnellMlxAdapter
        from meshserver.adapters.qwen_edit_mlx import QwenImageEditAdapter
        from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter
        from meshserver.adapters.sd_torch import Sd15Adapter, SdxlAdapter
        from meshserver.adapters.z_image_mlx import ZImageTurboAdapter

        classes += [
            Flux2KleinEditAdapter,
            FluxKontextMlxAdapter,
            QwenImageEditAdapter,
            QwenImageMlxAdapter,
            ZImageTurboAdapter,
            FluxSchnellMlxAdapter,
            FluxSchnellAdapter,
            SdxlAdapter,
            Sd15Adapter,
        ]
        return classes

    def _probe(self, cls: type) -> bool:
        try:
            ok = cls.available(self.cfg)
        except Exception as e:
            self._probe_errors[cls.id] = str(e)[:300]
            return False
        if not ok:
            err = getattr(cls, "available_error", None)
            if err:
                self._probe_errors[cls.id] = err
        return ok

    def refresh_llm_classes(self) -> None:
        """(Re)load the llm adapter catalog — builtin + custom_models.json.

        Called once at construction and again after register/unregister so
        custom models serve without a restart."""
        with contextlib.suppress(Exception):
            from meshserver.llm_models import llm_adapter_classes

            self._llm_classes = llm_adapter_classes(self.cfg)

    def find_class(self, model_id: str) -> type | None:
        """Adapter class lookup across all kinds (download/management routes)."""
        for cls in self._classes + self._llm_classes:
            if cls.id == model_id:
                return cls
        return None

    def _weights_size_gb(self, cls: type) -> float | None:
        """On-disk weights size for an llm adapter — HF repo cache for
        repo-based models, the gguf dir files for llama.cpp."""
        from meshserver.adapters.llm_base import dir_size_bytes, hf_repo_cache_dir

        with contextlib.suppress(Exception):
            files = getattr(cls, "gguf_files", None) or []
            if files:
                base = self.cfg.model_dir / "gguf"
                total = sum((base / f).stat().st_size for f in files if (base / f).exists())
                return round(total / 1e9, 2) if total else None
            repo = getattr(cls, "hf_repo", "")
            if repo:
                total = dir_size_bytes(hf_repo_cache_dir(repo))
                return round(total / 1e9, 2) if total else None
        return None

    def models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for cls in self._classes + self._llm_classes:
            kind = getattr(cls, "kind", "mesh")
            # the fake adapters are only listed when explicitly enabled
            avail = self._probe(cls) if (not _is_fake(cls) or self.cfg.fake) else False
            inst = self._instances.get(cls.id)
            est = self.ledger.est_mb(cls.id, getattr(cls, "est_mem_mb", None))
            downloaded: bool = avail
            size_gb: float | None = None
            if kind == "llm":
                with contextlib.suppress(Exception):
                    downloaded = bool(cls.downloaded(self.cfg))
                if downloaded:
                    size_gb = self._weights_size_gb(cls)
            out.append(
                ModelInfo(
                    id=cls.id,
                    name=cls.name,
                    device=cls.device,
                    backend=cls.backend,
                    kind=kind,
                    supports_multiview=getattr(cls, "supports_multiview", False),
                    max_images=getattr(cls, "max_images", 0),
                    license_note=cls.license_note,
                    loaded=bool(inst and inst.loaded),
                    available=avail,
                    last_error=self._failed.get(cls.id) or self._probe_errors.get(cls.id),
                    default_size=getattr(cls, "default_size", None),
                    supports_edit=getattr(cls, "supports_edit", False),
                    requires_image=getattr(cls, "requires_image", False),
                    max_count=getattr(cls, "max_count", None),
                    default_steps=getattr(cls, "default_steps", None),
                    canonical_up=getattr(cls, "canonical_up", None),
                    # mesh/image weights live in the HF/pipeline cache —
                    # availability implies presence; llm adapters report the
                    # real downloaded state + on-disk size
                    downloaded=downloaded,
                    downloading=bool(
                        self.downloads and self.downloads.is_downloading(cls.id)
                    ),
                    size_gb=size_gb,
                    est_mem_mb=round(est) if est else None,
                    supports_vision=getattr(cls, "supports_vision", False),
                    supports_tools=getattr(cls, "supports_tools", False),
                    # effective context the server will run at — cfg.llm_ctx
                    # overrides the adapter's catalog value (llama.cpp
                    # server_argv resolves the same way)
                    context_len=(
                        (
                            int(getattr(self.cfg, "llm_ctx", 0) or 0)
                            or getattr(cls, "context_len", None)
                        )
                        if kind == "llm"
                        else None
                    ),
                    pinned=cls.id in self._pinned,
                    custom=getattr(cls, "custom", False),
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
            return all_mesh.get(requested)  # explicit but unavailable -> None (error to client)
        avail = {k: v for k, v in all_mesh.items() if k not in self._failed}
        # auto
        device = self.device
        pref: list[str]
        if device == "cuda":
            pref = ["trellis2", "triposg", "step1x-3d", "hunyuan3d-omni"]
        elif device == "rocm":
            # no TRELLIS.2/Step1X-3D on AMD (CUDA-only); the hunyuan torch
            # variants are the only multiview-capable adapters here. Omni sits
            # last — its required control signal is only a neutral-bbox
            # fallback in 'auto' mode.
            pref = (
                ["hunyuan3d-2mv", "hunyuan3d-2mv-turbo", "triposg"]
                if n_images > 1
                else ["triposg", "hunyuan3d-2mini", "hunyuan3d-2mini-turbo", "hunyuan3d-omni"]
            )
        elif device == "mlx":
            # triposg joins when MESH_TRIPOSG_MPS=1 + a working MPS torch —
            # its available() checks the env flag itself
            pref = [
                "hunyuan3d-2.1-mlx",
                "hunyuan3d-2mv",
                "hunyuan3d-2mini",
                "triposg",
                "hunyuan3d-omni",
            ]
        elif device == "mps":
            pref = (
                ["hunyuan3d-2mv", "hunyuan3d-2mv-turbo"]
                if n_images > 1
                else [
                    "hunyuan3d-2mini",
                    "hunyuan3d-2mini-turbo",
                    "triposg",  # MESH_TRIPOSG_MPS=1 opt-in — see triposg.py
                    "hunyuan3d-omni",
                ]
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
                    # flux2-klein-4b-edit first: Apache-2.0 + ungated, distilled
                    # to 4 steps — ~10x faster than qwen-image-edit on MLX for
                    # the conditioning-view edits these jobs run; qwen stays
                    # as the bigger/steadier fallback
                    "flux2-klein-4b-edit",
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
            # z-image-turbo first: ungated Apache-2.0, 6B distilled at 8
            # steps — ~10x faster than qwen-image at benchmark parity;
            # qwen stays as the quality fallback, gated flux last
            pref = ["z-image-turbo", "qwen-image", "flux-schnell-mlx", "sdxl", "sd15"]
        elif device == "mps":
            pref = ["sdxl", "sd15"]
        else:
            pref = ["sd15"] if self.cfg.allow_cpu else []
        for pid in pref:
            if cls := avail.get(pid):
                return cls
        # no ladder hit — any remaining adapter is better than failing
        return next(iter(avail.values()), None)

    def select_llm(self, requested: str) -> type | None:
        """Pick an llm adapter class. 'auto' returns the first available and
        downloaded class; explicit ids return the class (or None). The family
        list may be empty until llm adapters are registered.

        Unknown explicit ids (e.g. an external provider's model name leaking
        through because llm_sidecar_model is unset) fall through to auto —
        the sidecar is the model authority in local mode, and a stale name
        should degrade to "any local model" instead of a hard 404."""
        all_llm = {
            cls.id: cls
            for cls in self._classes + self._llm_classes
            if getattr(cls, "kind", "mesh") == "llm"
        }
        if requested and requested != "auto":
            if cls := all_llm.get(requested):
                return cls
            log.warning("registry.unknown_llm_model", requested=requested)
        for cls in all_llm.values():
            if cls.id in self._failed:
                continue
            with contextlib.suppress(Exception):
                if not cls.downloaded(self.cfg):
                    continue
            if self._probe(cls):
                return cls
        return None

    def get(self, cls: type) -> object:
        """Instantiate + load the adapter, evicting LRU residents when the
        host can't keep mem_headroom_mb free with the newcomer aboard.

        Multiple models may stay resident as long as they fit — the old
        single-model invariant is replaced by memory admission."""
        with self._lock:
            inst = self._instances.get(cls.id)
            if inst is None:
                inst = cls.create(self.cfg)
                self._instances[cls.id] = inst
            if not inst.loaded:
                # llm servers pin when configured — a generation job must not
                # evict a pinned chat backend mid-conversation
                if getattr(cls, "kind", "") == "llm" and self.cfg.llm_pin:
                    self._pinned.add(cls.id)
                need = self.ledger.est_mb(cls.id, getattr(cls, "est_mem_mb", None))
                # best-effort eviction: if nothing is evictable the load runs
                # anyway and the OOM handler sorts out the wreckage
                while not self.ledger.fits(need):
                    if self._evict_lru_locked(exclude={cls.id}) is None:
                        break
                try:
                    inst.load()
                except Exception as e:
                    # remember the failure so 'auto' skips this adapter next
                    # time; purge whatever half-loaded state remains
                    self._failed[cls.id] = str(e)[:300]
                    with contextlib.suppress(Exception):
                        inst.unload()
                    hard_unload()
                    raise
                self.ledger.reserve(
                    cls.id, need, pinned=cls.id in self._pinned, kind="model"
                )
            self._last_used[cls.id] = time.monotonic()
            self.ledger.touch(cls.id)
            return inst

    def mark_failed(self, adapter_id: str, error: str) -> None:
        """Record an inference-time failure (e.g. OOM inside generate())."""
        self._failed[adapter_id] = error[:300]

    def pin(self, adapter_id: str, on: bool) -> None:
        """Pin/unpin a resident model — pinned ids are never evicted for
        memory pressure (the llm family uses this via cfg.llm_pin)."""
        with self._lock:
            if on:
                self._pinned.add(adapter_id)
            else:
                self._pinned.discard(adapter_id)
            self.ledger.set_pinned(adapter_id, on)

    def _unload_instance(self, adapter_id: str, inst: object) -> None:
        """Unload one adapter instance and drop its ledger reservation —
        every unload path funnels here so the ledger stays consistent."""
        with contextlib.suppress(Exception):
            inst.unload()
        self.ledger.release(adapter_id)

    def _evict_lru_locked(self, exclude: set[str]) -> str | None:
        """Unload the least-recently-used non-pinned loaded instance not in
        `exclude`; returns its adapter id or None when nothing is evictable.
        Caller must hold _lock."""
        candidates = [
            (self._last_used.get(aid, 0.0), aid)
            for aid, inst in self._instances.items()
            if inst.loaded and aid not in exclude and aid not in self._pinned
        ]
        if not candidates:
            return None
        _, aid = min(candidates)
        self._unload_instance(aid, self._instances[aid])
        # purge allocator caches — without this the "freed" memory stays in
        # torch/MLX pools and the newcomer still can't fit
        hard_unload()
        return aid

    def evict_lru(self, exclude: set[str] | None = None) -> str | None:
        """Queue-admission hook: evict one LRU resident to make room for a
        new model (idle warm workers are evicted first by the caller)."""
        with self._lock:
            return self._evict_lru_locked(set(exclude or ()))

    def unload_all(self) -> None:
        """Unload every resident model + purge device caches (OOM recovery)."""
        with self._lock:
            for aid, inst in self._instances.items():
                if inst.loaded:
                    self._unload_instance(aid, inst)
        hard_unload()

    def maybe_unload_idle(self) -> None:
        if self.cfg.unload_idle_s <= 0:
            return
        now = time.monotonic()
        unloaded = False
        with self._lock:
            for aid, inst in self._instances.items():
                if (
                    inst.loaded
                    and aid not in self._pinned
                    and now - self._last_used.get(aid, 0.0) > self.cfg.unload_idle_s
                ):
                    self._unload_instance(aid, inst)
                    unloaded = True
        if unloaded:
            hard_unload()
