"""LLM adapter for Apple Silicon: `mlx_lm.server` (MLX, OpenAI-compatible).

Install: ``pip install mlx-lm`` (the mlx_lm package).

Weights are plain HF repos under mlx-community/*; ``load()`` spawns
``python -m mlx_lm.server --model <hf_repo> --port <p>`` as a detached
process group and polls the OpenAI endpoints for readiness. Downloads go
through ``snapshot_download`` — progress is bridged into the manager's
callback via a tqdm shim when the installed huggingface_hub supports
``tqdm_class``, otherwise a cache-growth watcher thread approximates it.
"""

from __future__ import annotations

import platform
import sys
import threading
import time
from typing import Any

from meshserver.adapters.llm_base import (
    DownloadProgressFn,
    LlmAdapter,
    _accepts_kwarg,
    dir_size_bytes,
    hf_repo_cache_dir,
    progress_tqdm,
)
from meshserver.logging import get_logger

log = get_logger(__name__)


class _CacheGrowthWatcher(threading.Thread):
    """Fallback progress source: polls the HF cache dir while a snapshot
    download runs (older huggingface_hub has no tqdm_class hook)."""

    def __init__(self, repo: str, progress_cb: DownloadProgressFn) -> None:
        super().__init__(daemon=True, name=f"llm-dl-watch-{repo}")
        self._dir = hf_repo_cache_dir(repo)
        self._cb = progress_cb
        self._stop = threading.Event()
        self._baseline = dir_size_bytes(self._dir)

    def run(self) -> None:
        while not self._stop.wait(2.0):
            self._cb(dir_size_bytes(self._dir) - self._baseline, None)

    def stop(self) -> None:
        self._stop.set()


class MlxLmAdapter(LlmAdapter):
    """mlx_lm.server subprocess serving one HF repo."""

    device = "mlx"
    backend = "mlx-lm"

    @classmethod
    def available(cls, cfg: Any) -> bool:  # noqa: ARG003
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            cls.available_error = "mlx-lm requires Apple Silicon (darwin-arm64)"
            return False
        try:
            import mlx_lm  # noqa: F401

            return True
        except Exception as e:
            cls.available_error = (
                f"mlx_lm backend not installed ({e}) — optional extra: "
                "run mesh_server/scripts/install_llm.sh or "
                "'uv pip install -e .[llm]' inside mesh_server/"
            )
            return False

    @classmethod
    def downloaded(cls, cfg: Any) -> bool:  # noqa: ARG003
        """Repo present in the HF cache — probe config.json only, no network."""
        if not cls.hf_repo:
            return False
        try:
            from huggingface_hub import try_to_load_from_cache

            p = try_to_load_from_cache(cls.hf_repo, "config.json")
            return isinstance(p, str)
        except Exception:
            return False

    @classmethod
    def download_weights(
        cls,
        cfg: Any,  # noqa: ARG003
        progress_cb: DownloadProgressFn | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> int:
        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {}
        watcher: _CacheGrowthWatcher | None = None
        if _accepts_kwarg(snapshot_download, "tqdm_class"):
            tcls = progress_tqdm(progress_cb, cancel_flag, cls.hf_repo)
            if tcls is not None:
                kwargs["tqdm_class"] = tcls
        elif progress_cb is not None:
            watcher = _CacheGrowthWatcher(cls.hf_repo, progress_cb)
            watcher.start()
        before = dir_size_bytes(hf_repo_cache_dir(cls.hf_repo))
        try:
            snapshot_download(cls.hf_repo, **kwargs)
        finally:
            if watcher is not None:
                watcher.stop()
        return max(0, dir_size_bytes(hf_repo_cache_dir(cls.hf_repo)) - before)

    def server_argv(self, port: int) -> list[str]:
        # minimal flags — mlx_lm.server applies the repo's chat template and
        # picks sensible sampling defaults; context is bound by the model
        return [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            self.hf_repo,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]


def make_mlx_adapter(
    repo: str,
    *,
    id: str,
    name: str = "",
    vision: bool = False,
    tools: bool = True,
    ctx: int = 16384,
    est_mb: int | None = None,
    custom: bool = False,
) -> type[LlmAdapter]:
    """Factory: an adapter class serving `repo` via mlx_lm.server.

    Lets llm_models.py turn catalog rows / persisted custom entries into
    adapter classes without a file per model.
    """
    return type(
        f"MlxLm_{id.replace('-', '_')}",
        (MlxLmAdapter,),
        {
            "id": id,
            "name": name or repo,
            "hf_repo": repo,
            "supports_vision": bool(vision),
            "supports_tools": bool(tools),
            "context_len": int(ctx),
            "est_mem_mb": est_mb,
            "custom": bool(custom),
        },
    )
