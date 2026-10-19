"""LLM adapter for Apple Silicon: `mlx_lm.server` / `mlx_vlm.server` (MLX,
OpenAI-compatible).

Install: ``pip install mlx-lm`` (text models) and ``pip install mlx-vlm``
(multimodal models). Both packages require ``transformers>=5``, which
conflicts with the torch extras in the shared mesh venv — they therefore
live in a dedicated ``mesh_server/.venv-llm`` created by
``mesh_server/scripts/install_llm.sh``. The adapter resolves its
interpreter as ``cfg.llm_python`` (env ``MESH_LLM_PYTHON``) →
``.venv-llm/bin/python`` → ``sys.executable`` (legacy shared-venv
fallback for older architectures).

Weights are plain HF repos under mlx-community/*; ``load()`` spawns
``python -m <module>.server --model <hf_repo> --port <p>`` as a detached
process group and polls the OpenAI endpoints for readiness. Downloads go
through ``snapshot_download`` — progress is bridged into the manager's
callback via a tqdm shim when the installed huggingface_hub supports
``tqdm_class``, otherwise a cache-growth watcher thread approximates it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
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


def _llm_venv_python() -> Path:
    """Dedicated LLM venv next to the shared mesh venv: <repo>/mesh_server/.venv-llm."""
    return Path(__file__).resolve().parents[2] / ".venv-llm" / "bin" / "python"


def _resolve_python(cfg: Any) -> str:
    """Interpreter that should host the mlx server for this adapter.

    Order: explicit ``cfg.llm_python`` (env MESH_LLM_PYTHON) → dedicated
    ``.venv-llm`` when it exists → ``sys.executable`` (the shared mesh venv,
    legacy fallback holding the tf4-compatible ``mlx-lm<0.30``).
    """
    override = str(getattr(cfg, "llm_python", "") or "")
    if override:
        return override
    venv_py = _llm_venv_python()
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


_MODULE_IMPORT_CACHE: dict[tuple[str, str], bool] = {}


def _module_importable(python: str, module: str) -> bool:
    """`import <module>` probe in `python` — subprocess because the module may
    only exist in the dedicated .venv-llm, not in our own environment."""
    key = (python, module)
    cached = _MODULE_IMPORT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        rc = subprocess.run(
            [python, "-c", f"import {module}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).returncode
        ok = rc == 0
    except Exception:
        ok = False
    _MODULE_IMPORT_CACHE[key] = ok
    return ok


class MlxLmAdapter(LlmAdapter):
    """mlx_lm.server / mlx_vlm.server subprocess serving one HF repo."""

    device = "mlx"
    backend = "mlx-lm"
    # server module spawned as `python -m <required_module>.server`;
    # vision adapters use mlx_vlm — mlx_lm strips image parts for the
    # qwen3_5 architecture family (its server has no Qwen vision processor)
    required_module = "mlx_lm"

    @classmethod
    def available(cls, cfg: Any) -> bool:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            cls.available_error = "mlx-lm requires Apple Silicon (darwin-arm64)"
            return False
        if _module_importable(_resolve_python(cfg), cls.required_module):
            return True
        cls.available_error = (
            f"{cls.required_module} backend not installed — run "
            "mesh_server/scripts/install_llm.sh (creates the dedicated "
            "mesh_server/.venv-llm with mlx-lm + mlx-vlm)"
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
        # minimal flags — the server applies the repo's chat template and
        # picks sensible sampling defaults; context is bound by the model.
        # mlx_vlm.server is CLI-compatible (--model/--host/--port).
        return [
            _resolve_python(self.cfg),
            "-m",
            f"{self.required_module}.server",
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
    """Factory: an adapter class serving `repo` via mlx_lm/mlx_vlm server.

    Vision entries route to ``mlx_vlm.server`` — mlx_lm's server has no
    Qwen-family vision processor and would silently drop image parts.
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
            "required_module": "mlx_vlm" if vision else "mlx_lm",
        },
    )
