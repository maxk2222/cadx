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
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from meshserver.config import ServerConfig
from meshserver.logging import get_logger

log = get_logger(__name__)


class MemoryLedger:
    """Memory bookkeeping: learned per-model footprints + admission control."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._stats_path: Path = cfg.model_dir / "model_stats.json"
        self._stats: dict[str, float] = {}  # adapter id -> measured peak RSS (MB)
        self._residents: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
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

    def free_mb(self) -> float | None:
        """Host RAM available right now (MB); None when psutil can't tell."""
        try:
            import psutil

            return psutil.virtual_memory().available / 1e6
        except Exception:
            return None

    def fits(self, need_mb: float) -> bool:
        """Admission check: keep mem_headroom_mb free after the newcomer.
        Unknown free memory -> admit (the RSS watchdog bounds the damage)."""
        free = self.free_mb()
        if free is None:
            return True
        return free - need_mb >= self.cfg.mem_headroom_mb

    # -- resident set ------------------------------------------------------- #

    def reserve(self, entry_id: str, mb: float, *, pinned: bool, kind: str) -> None:
        with self._lock:
            self._residents[entry_id] = {
                "id": entry_id,
                "mb": round(mb, 1),
                "kind": kind,
                "pinned": pinned,
                "last_used": time.monotonic(),
            }

    def release(self, entry_id: str) -> None:
        with self._lock:
            self._residents.pop(entry_id, None)

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

    def residents(self) -> list[dict[str, Any]]:
        """[{id, mb, kind, pinned, idle_s}] — the /v1/queue resident view."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "id": e["id"],
                    "mb": e["mb"],
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
