"""DownloadManager: background weight downloads for all adapter families.

One daemon thread per model calls the adapter's classmethod
``download_weights(cfg, progress_cb, cancel_flag)`` — no model gets loaded
while weights stream in. HF transfers resume partial files, so cancel() is
just an event flip and the next start() picks up where it left off.

Adapter contract (see adapters/llm_base.py): any class exposing
``downloaded(cfg)``, ``download_weights(cfg, progress_cb, cancel_flag)``
and a weights source (``hf_repo`` / ``gguf_files``) is downloadable; the
existing mesh/image adapters aren't and report ``downloadable() == False``.
"""

from __future__ import annotations

import contextlib
import shutil
import threading
from pathlib import Path
from typing import Any

from meshserver.adapters.llm_base import DownloadCancelled, hf_repo_cache_dir
from meshserver.logging import get_logger
from meshserver.schemas import DownloadStatus

log = get_logger(__name__)


def downloadable(cls: type) -> bool:
    """Does this adapter class support managed weight downloads?"""
    return callable(getattr(cls, "download_weights", None)) and bool(
        getattr(cls, "hf_repo", "") or getattr(cls, "gguf_files", [])
    )


class DownloadManager:
    """Tracks one background download job per model id."""

    def __init__(self, cfg: Any, is_loaded: Any = None) -> None:
        self.cfg = cfg
        # optional probe from the registry: model id -> resident in memory —
        # delete_weights refuses to yank files out from under a live server
        self._is_loaded = is_loaded if callable(is_loaded) else (lambda _mid: False)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, model_id: str, cls: type) -> DownloadStatus:
        """Kick off cls.download_weights in a daemon thread. Concurrent
        starts for the same model are a no-op; already-downloaded short-
        circuits to 'done'."""
        with self._lock:
            job = self._jobs.get(model_id)
            if job is not None and job["status"] == "downloading":
                return self._to_status(job)
            try:
                if cls.downloaded(self.cfg):
                    job = {
                        "status": "done",
                        "bytes_done": 0,
                        "bytes_total": None,
                        "error": None,
                        "_cancel": threading.Event(),
                        "_thread": None,
                    }
                    self._jobs[model_id] = job
                    return self._to_status(job)
            except Exception:  # probe failure → still try the download
                pass
            cancel = threading.Event()
            job = {
                "status": "downloading",
                "bytes_done": 0,
                "bytes_total": None,
                "error": None,
                "_cancel": cancel,
                "_thread": None,
            }
            self._jobs[model_id] = job
            t = threading.Thread(
                target=self._run,
                args=(model_id, cls, job, cancel),
                daemon=True,
                name=f"dl-{model_id}",
            )
            job["_thread"] = t
            t.start()
            log.info("download.start", model=model_id, repo=getattr(cls, "hf_repo", ""))
            return self._to_status(job)

    def _run(
        self,
        model_id: str,
        cls: type,
        job: dict[str, Any],
        cancel: threading.Event,
    ) -> None:
        def progress(done: int, total: int | None) -> None:
            with self._lock:
                job["bytes_done"] = done
                job["bytes_total"] = total

        try:
            nbytes = cls.download_weights(self.cfg, progress, cancel)
            with self._lock:
                job["status"] = "done"
                job["bytes_done"] = max(int(job["bytes_done"]), int(nbytes or 0))
            log.info("download.done", model=model_id, bytes=nbytes)
        except DownloadCancelled:
            # HF resumes partial transfers → back to idle, restartable
            with self._lock:
                job["status"] = "idle"
            log.info("download.cancelled", model=model_id)
        except Exception as e:
            with self._lock:
                job["status"] = "error"
                job["error"] = str(e)[:500]
            log.warning("download.error", model=model_id, error=str(e))

    # -- queries ------------------------------------------------------------ #

    def _to_status(self, job: dict[str, Any]) -> DownloadStatus:
        return DownloadStatus(
            status=str(job["status"]),
            bytes_done=int(job["bytes_done"]),
            bytes_total=job["bytes_total"],
            error=job["error"],
        )

    def status(self, model_id: str) -> DownloadStatus:
        with self._lock:
            job = self._jobs.get(model_id)
            return self._to_status(job) if job is not None else DownloadStatus(status="idle")

    def is_downloading(self, model_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(model_id)
            return job is not None and job["status"] == "downloading"

    def cancel(self, model_id: str) -> bool:
        """Flag a running download to stop; the thread lands back at 'idle'
        once the flag propagates (HF resumes on the next start)."""
        with self._lock:
            job = self._jobs.get(model_id)
            if job is None or job["status"] != "downloading":
                return False
            job["_cancel"].set()
            return True

    # -- deletion ----------------------------------------------------------- #

    def delete_weights(self, model_id: str, cls: type) -> bool:
        """Remove cached weights from disk. Refuses while a download is in
        flight or the model is loaded (registry probe)."""
        if self.is_downloading(model_id):
            return False
        if self._is_loaded(model_id):
            return False
        removed = False
        repo = getattr(cls, "hf_repo", "")
        gguf_files = list(getattr(cls, "gguf_files", []) or [])
        mmproj = getattr(cls, "gguf_mmproj", None)
        if gguf_files:
            base = Path(self.cfg.model_dir) / "gguf"
            for rel in gguf_files + ([mmproj] if mmproj else []):
                p = base / str(rel)
                if p.is_file():
                    with contextlib.suppress(OSError):
                        p.unlink()
                        removed = True
            # prune now-empty parent dirs up to (not incl.) the gguf dir
            for rel in gguf_files + ([mmproj] if mmproj else []):
                parent = (base / str(rel)).parent
                while parent != base and parent.is_dir():
                    try:
                        parent.rmdir()  # fails unless empty — fine
                    except OSError:
                        break
                    parent = parent.parent
        elif repo:
            cache = hf_repo_cache_dir(repo)
            if cache.is_dir():
                with contextlib.suppress(OSError):
                    shutil.rmtree(cache)
                    removed = True
        with self._lock:
            job = self._jobs.get(model_id)
            if job is not None and job["status"] != "downloading":
                job["status"] = "idle"
                job["bytes_done"] = 0
        log.info("download.deleted", model=model_id, removed=removed)
        return True
