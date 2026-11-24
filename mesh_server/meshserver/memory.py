"""Host-memory bookkeeping for model residency + queue admission.

The ledger tracks two things:

- learned per-model footprints — peak RSS measured in generation
  subprocesses, persisted to ``model_dir/model_stats.json`` so restarts
  start with realistic estimates instead of the static ``est_mem_mb``
  declarations on the adapter classes;
- the resident set — loaded in-process adapters (kind="model") and warm
  generation workers (kind="warm", entry id "warm:<adapter>") — so queue
  admission and /v1/queue can see who is holding memory.

``fits()`` is the admission check: a new residency is only allowed when the
host keeps ``cfg.mem_headroom_mb`` free after accounting for the newcomer.
When psutil can't report free memory the check passes — the RSS watchdog on
generation children stays the last line of defence either way.

The ledger accounts host RAM only — device VRAM is probed live via
``free_vram_mb`` in the admission loop (jobs.admit_resolved), since a
resident model can pin the card while RAM still fits.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from meshserver.config import ServerConfig
from meshserver.logging import get_logger

log = get_logger(__name__)


def _vram_info() -> tuple[float, float] | None:
    """(free, total) VRAM in MB on CUDA/ROCm hosts; None elsewhere or when
    no probe is available. Process-agnostic (driver-level counters) — sees
    llama.cpp servers and generation subprocesses alike.

    pynvml is preferred: probing via torch.cuda inits a CUDA context in the
    caller, which would permanently pin a few hundred MB of the very
    resource being measured.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.free) / 1e6, float(info.total) / 1e6
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return float(free) / 1e6, float(total) / 1e6
    except Exception:
        return None


def free_vram_mb() -> float | None:
    """Free VRAM in MB (see _vram_info); None on non-CUDA hosts."""
    info = _vram_info()
    return info[0] if info else None


def total_vram_mb() -> float | None:
    """Total VRAM in MB (see _vram_info); None on non-CUDA hosts."""
    info = _vram_info()
    return info[1] if info else None


class MemoryLedger:
    """Memory bookkeeping: learned per-model footprints + admission control."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._stats_path: Path = cfg.model_dir / "model_stats.json"
        self._stats: dict[str, float] = {}  # adapter id -> measured peak RSS (MB)
        self._residents: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        # mb freed in the last few seconds that psutil can't see yet — an
        # admission loop that evicts then immediately re-checks fits() would
        # over-evict without this (the OS needs a beat to report the release)
        self._pending_free: list[tuple[float, float]] = []
        self._load_stats()

    # -- measured footprints --------------------------------------------- #

    def _load_stats(self) -> None:
        try:
            raw = json.loads(self._stats_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            if self._stats_path.exists():
                # a torn write or disk hiccup must not take down the server —
                # estimates just fall back to the declared class attrs
                log.warning("memory.stats_corrupt", path=str(self._stats_path), error=str(e))
            return
        if isinstance(raw, dict):
            self._stats = {
                str(k): float(v)
                for k, v in raw.get("peak_rss_mb", {}).items()
                if isinstance(v, (int, float))
            }

    def _save_stats(self) -> None:
        tmp = self._stats_path.with_suffix(".tmp")
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"peak_rss_mb": self._stats}, indent=1))
            tmp.replace(self._stats_path)
        except OSError as e:
            log.warning("memory.stats_write_failed", path=str(self._stats_path), error=str(e))

    def record_peak(self, adapter_id: str, rss_mb: float) -> None:
        """Fold a measured child peak into the stats — the max wins, so one
        lucky low-RSS run never shrinks the estimate below the real peak."""
        if not adapter_id or adapter_id == "auto" or rss_mb <= 0:
            return
        with self._lock:
            prev = self._stats.get(adapter_id, 0.0)
            if rss_mb <= prev:
                return
            self._stats[adapter_id] = round(rss_mb, 1)
            self._save_stats()

    def est_mb(self, adapter_id: str, declared: int | None) -> float:
        """Estimated residency cost: the measured peak (+10% safety) wins
        over the adapter's declared est_mem_mb once known."""
        measured = self._stats.get(adapter_id, 0.0)
        return max(float(declared or 0), 1.1 * measured)

    # -- admission --------------------------------------------------------- #

    def free_mb(self, exclude_spec: set[str] | None = None) -> float | None:
        """Host RAM available right now (MB); None when psutil can't tell.

        Subtracts speculative reservations (loads in flight whose RSS
        psutil can't see yet) and re-adds recently-released memory
        (note_release) the OS hasn't reported back yet — together they
        bound both over-admission (two concurrent loads passing fits()
        on the same stale snapshot) and over-eviction in admit loops.
        ``exclude_spec`` ids skip a caller's own speculative reservation —
        a loader judging OTHER holds must not count its declared need twice."""
        try:
            import psutil

            free = psutil.virtual_memory().available / 1e6
        except Exception:
            return None
        with self._lock:
            cutoff = time.monotonic() - 4.0
            self._pending_free = [(t, m) for t, m in self._pending_free if t > cutoff]
            pending = sum(m for _, m in self._pending_free)
            spec = sum(
                e["mb"]
                for e in self._residents.values()
                if e.get("spec") and (not exclude_spec or e["id"] not in exclude_spec)
            )
        return free - spec + pending

    def note_release(self, mb: float) -> None:
        """Mark `mb` as just-freed-but-not-yet-visible to psutil (4s window)."""
        if mb <= 0:
            return
        with self._lock:
            self._pending_free.append((time.monotonic(), mb))

    def fits(self, need_mb: float, *, exclude_spec: set[str] | None = None) -> bool:
        """Admission check: keep mem_headroom_mb free after the newcomer.
        Unknown free memory -> admit (the RSS watchdog bounds the damage).
        ``exclude_spec`` flows into free_mb — the caller's own speculative
        reservation isn't counted against it."""
        free = self.free_mb(exclude_spec)
        if free is None:
            return True
        return free - need_mb >= self.cfg.effective("mem_headroom_mb")

    # -- resident set ------------------------------------------------------- #

    def reserve(
        self,
        entry_id: str,
        mb: float,
        *,
        pinned: bool,
        kind: str,
        spec: bool = False,
        vram_mb: float = 0.0,
    ) -> None:
        """Track a residency reservation. ``spec`` marks memory that isn't
        real yet — a model between fits() and load() — so a concurrent
        admission can't pass on the same stale free-mem snapshot (the
        TOCTOU the old global-lock regime hid by serializing all loads).
        ``vram_mb`` is the resident's declared device-memory peak — the
        committer for VRAM admission (a loading model's allocation isn't
        visible to a live free-VRAM probe yet)."""
        with self._lock:
            self._residents[entry_id] = {
                "id": entry_id,
                "mb": round(mb, 1),
                "vram_mb": round(float(vram_mb), 1),
                "kind": kind,
                "pinned": pinned,
                "spec": spec,
                "last_used": time.monotonic(),
            }

    def mark_resident(self, entry_id: str) -> None:
        """A speculative reservation just became real RSS (load finished) —
        psutil sees it now, so it leaves the free_mb() subtraction."""
        with self._lock:
            entry = self._residents.get(entry_id)
            if entry is not None:
                entry["spec"] = False

    def is_speculative(self, entry_id: str) -> bool:
        with self._lock:
            entry = self._residents.get(entry_id)
            return bool(entry and entry.get("spec"))

    def release(self, entry_id: str) -> float:
        """Drop a reservation; returns its mb so callers can note_release()."""
        with self._lock:
            entry = self._residents.pop(entry_id, None)
        return float(entry["mb"]) if entry is not None else 0.0

    def has(self, entry_id: str) -> bool:
        with self._lock:
            return entry_id in self._residents

    def set_pinned(self, entry_id: str, pinned: bool) -> None:
        with self._lock:
            entry = self._residents.get(entry_id)
            if entry is not None:
                entry["pinned"] = pinned

    def touch(self, entry_id: str) -> None:
        with self._lock:
            entry = self._residents.get(entry_id)
            if entry is not None:
                entry["last_used"] = time.monotonic()

    def committed_vram_mb(self, exclude: set[str] | None = None) -> float:
        """Sum of residents' declared VRAM peaks (MB) — including
        speculative entries whose device allocation is still ramping up."""
        with self._lock:
            return sum(
                float(e.get("vram_mb") or 0.0)
                for e in self._residents.values()
                if not exclude or e["id"] not in exclude
            )

    def residents(self) -> list[dict[str, Any]]:
        """[{id, mb, vram_mb, kind, pinned, idle_s}] — /v1/queue view."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "id": e["id"],
                    "mb": e["mb"],
                    "vram_mb": e.get("vram_mb", 0.0),
                    "kind": e["kind"],
                    "pinned": e["pinned"],
                    "idle_s": round(now - e["last_used"], 1),
                }
                for e in self._residents.values()
            ]

    def evictable(self, exclude: set[str]) -> list[str]:
        """Resident entry ids eligible for eviction — non-pinned, least
        recently used first."""
        with self._lock:
            entries = [
                e
                for e in self._residents.values()
                if not e["pinned"] and e["id"] not in exclude
            ]
        entries.sort(key=lambda e: e["last_used"])
        return [e["id"] for e in entries]
