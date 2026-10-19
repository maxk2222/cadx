"""Adapter interface for local OpenAI-compatible LLM servers (third family).

An LlmAdapter wraps a *managed server subprocess* — mlx_lm.server or
llama.cpp's llama-server — that speaks the OpenAI /v1/chat/completions
protocol on a loopback port. The registry treats it like the other families
(one heavy model resident at a time) and llm_proxy.py forwards client
requests to the subprocess verbatim.

Weight management is separate from the runtime: ``downloaded()`` only probes
the local cache and ``download_weights()`` is a classmethod, so a weights
fetch never needs the model — or the GPU — loaded.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from meshserver.logging import get_logger

log = get_logger(__name__)

# download progress callback: (bytes_done, bytes_total_or_None_when_unknown)
DownloadProgressFn = Callable[[int, int | None], None]

# grace period between SIGTERM and SIGKILL on the server process group
_TERM_GRACE_S = 5.0


class DownloadCancelled(RuntimeError):
    """Raised inside download_weights when the manager's cancel event fires.

    HF downloads are resumable, so a cancelled job simply drops back to
    'idle' — the next start() picks up where it left off.
    """


def free_port() -> int:
    """OS-assigned free TCP port on loopback (bind(0))."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def hf_repo_cache_dir(repo: str) -> Path:
    """HF cache snapshot dir for a repo id: <HF_HUB_CACHE>/models--org--name."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / ("models--" + repo.strip("/").replace("/", "--"))


def dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                with contextlib.suppress(OSError):
                    total += p.stat().st_size
    except OSError:
        pass
    return total


def progress_tqdm(
    progress_cb: DownloadProgressFn | None,
    cancel_flag: threading.Event | None,
    what: str = "",
) -> type | None:
    """tqdm subclass that bridges HF download progress into our callback and
    aborts the transfer when the cancel event fires (raising inside update()
    propagates through huggingface_hub's chunked copy loop).

    Returns None when no callback is needed or tqdm can't be imported —
    huggingface_hub then uses its own default bar.
    """
    if progress_cb is None and cancel_flag is None:
        return None
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        return None

    class _CbTqdm(_tqdm):
        def update(self, n: int | float = 1):  # noqa: D102
            if cancel_flag is not None and cancel_flag.is_set():
                raise DownloadCancelled(f"download cancelled: {what}")
            out = super().update(n)
            if progress_cb is not None:
                progress_cb(int(self.n), int(self.total) if self.total else None)
            return out

        def close(self):  # noqa: D102
            if progress_cb is not None:
                with contextlib.suppress(Exception):
                    progress_cb(int(self.n), int(self.total) if self.total else None)
            super().close()

    return _CbTqdm


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """huggingface_hub moves fast — only pass tqdm_class when the installed
    version actually accepts it."""
    try:
        import inspect

        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class LlmAdapter(ABC):
    """One local LLM backend. Instances hold a managed server subprocess."""

    id: str = "?"
    name: str = "?"
    device: str = "?"  # mlx | cpu | cuda | ...
    backend: str = "?"  # mlx-lm | llama.cpp
    kind: ClassVar[str] = "llm"
    license_note: str = ""
    hf_repo: str = ""  # HF repo id the weights come from
    gguf_files: list[str] = []  # llama.cpp: weight files inside hf_repo
    gguf_mmproj: str | None = None  # llama.cpp: vision projector file in hf_repo
    supports_vision: bool = False
    supports_tools: bool = True
    est_mem_mb: int | None = None  # estimated resident footprint
    context_len: int = 16384
    custom: bool = False  # user-registered via /v1/models/register

    # -- class-level probes ------------------------------------------------ #

    @classmethod
    def available(cls, cfg: Any) -> bool:
        """Can a server for this backend be spawned in this environment?"""
        return False

    @classmethod
    def downloaded(cls, cfg: Any) -> bool:
        """Weights present in the local cache. Must never trigger a download."""
        return False

    @classmethod
    def download_weights(
        cls,
        cfg: Any,
        progress_cb: DownloadProgressFn | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> int:
        """Fetch weights into the local cache; returns bytes on disk.

        Classmethod on purpose: the download manager calls it without loading
        the model (no GPU held while gigabytes stream in).
        """
        raise NotImplementedError

    @classmethod
    def create(cls, cfg: Any) -> "LlmAdapter":
        return cls(cfg)  # type: ignore[call-arg]

    # -- instance lifecycle ------------------------------------------------ #

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._loaded = False
        self._proc: subprocess.Popen | None = None
        self._log_fp: Any = None
        self._served_id: str = ""
        self.port: int | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded and self._proc is not None and self._proc.poll() is None

    def server_argv(self, port: int) -> list[str]:
        """Command line that starts the OpenAI-compatible server on `port`."""
        raise NotImplementedError

    def ready_paths(self) -> tuple[str, ...]:
        """Endpoints polled until one responds (any status < 500 = up)."""
        return ("/v1/models", "/health")

    def load(self) -> None:
        """Spawn the server subprocess and wait for readiness (bounded by
        cfg.llm_load_timeout_s). Failure kills the child before raising."""
        if self.loaded:
            return
        port = free_port()
        argv = self.server_argv(port)
        log_path = Path(self.cfg.job_dir) / f"llm-{self.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("llm.load.start", model=self.id, port=port, argv=" ".join(argv))
        # the handle must outlive load() — closed in unload(), no ctx manager
        self._log_fp = log_path.open("ab")
        try:
            self._proc = subprocess.Popen(
                argv,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group → killpg on unload
            )
        except Exception:
            with contextlib.suppress(Exception):
                self._log_fp.close()
            self._log_fp = None
            raise
        self.port = port
        try:
            self._wait_ready(log_path)
        except Exception:
            self.unload()
            raise
        self._loaded = True
        log.info("llm.load.ready", model=self.id, port=port, served=self._served_id)

    def _wait_ready(self, log_path: Path) -> None:
        timeout = float(getattr(self.cfg, "llm_load_timeout_s", 300) or 300)
        deadline = time.monotonic() + timeout
        assert self._proc is not None and self.port is not None
        while time.monotonic() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"llm '{self.id}' server exited early (rc={rc}); see {log_path}"
                )
            if self._probe_ready():
                return
            time.sleep(0.5)
        raise TimeoutError(f"llm '{self.id}' not ready within {timeout:.0f}s; see {log_path}")

    def _probe_ready(self) -> bool:
        """One poll round over ready_paths(). Any HTTP response (even 4xx)
        proves the server is up; connection refused means keep waiting.
        A parseable /v1/models body also captures the served model id."""
        assert self.port is not None
        for path in self.ready_paths():
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
                with urllib.request.urlopen(req, timeout=2) as r:
                    if path == "/v1/models" and r.status == 200:
                        with contextlib.suppress(Exception):
                            items = json.loads(r.read().decode()).get("data") or []
                            ids = [
                                str(m["id"])
                                for m in items
                                if isinstance(m, dict) and m.get("id")
                            ]
                            if ids:
                                # mlx_vlm lists every cached HF repo, not just
                                # the served one — prefer the id matching this
                                # adapter's repo, else fall back to the first
                                # (llama.cpp derives its id from the .gguf path)
                                self._served_id = (
                                    self.hf_repo if self.hf_repo in ids else ids[0]
                                )
                    if r.status < 500:
                        return True
            except urllib.error.HTTPError as e:
                if e.code < 500:  # endpoint missing/method odd — still alive
                    return True
            except Exception:
                continue
        return False

    def unload(self) -> None:
        """Terminate the server process group: SIGTERM, SIGKILL after grace."""
        proc = self._proc
        self._loaded = False
        self._proc = None
        self.port = None
        self._served_id = ""
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)  # start_new_session → pgid == pid
            except (ProcessLookupError, PermissionError, OSError):
                with contextlib.suppress(Exception):
                    proc.terminate()
            try:
                proc.wait(timeout=_TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                log.warning("llm.unload.kill", model=self.id, pid=proc.pid)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    with contextlib.suppress(Exception):
                        proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=_TERM_GRACE_S)
        if self._log_fp is not None:
            with contextlib.suppress(Exception):
                self._log_fp.close()
            self._log_fp = None
        log.info("llm.unload", model=self.id)

    def base_url(self) -> str:
        """OpenAI API root of the managed server (…/v1)."""
        if self.port is None:
            raise RuntimeError(f"llm '{self.id}' is not loaded")
        return f"http://127.0.0.1:{self.port}/v1"

    def served_model_id(self) -> str:
        """Model name the inner server expects in request bodies.

        Prefer the id the live server reported via GET /v1/models (llama.cpp
        in particular derives it from the .gguf path); fall back to the HF
        repo / adapter id for servers that don't list one.
        """
        return self._served_id or self.hf_repo or self.id
