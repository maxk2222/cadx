"""LLM adapter for llama.cpp: `llama-server` binary or `python -m
llama_cpp.server` (llama-cpp-python) — both OpenAI-compatible.

Binary resolution order:
    1. cfg.llama_server_bin (explicit override)
    2. llama-server / llama-server.exe on PATH
    3. python -m llama_cpp.server when llama_cpp is importable

Weights are GGUF files inside a HF repo, downloaded via hf_hub_download into
``cfg.model_dir/"gguf"`` (the filename's repo-relative path is preserved, so
``local_dir/<filename>`` is the on-disk path). A ``gguf_mmproj`` entry adds a
vision projector via the server's --mmproj flag.
"""

from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path
from typing import Any

from meshserver.adapters.llm_base import (
    DownloadCancelled,
    DownloadProgressFn,
    LlmAdapter,
    _accepts_kwarg,
    progress_tqdm,
)
from meshserver.logging import get_logger

log = get_logger(__name__)


class LlamaCppAdapter(LlmAdapter):
    """llama.cpp server subprocess serving GGUF weights."""

    device = "cpu"  # llama.cpp runs everywhere; GPU builds share the same CLI
    backend = "llama.cpp"

    # -- server resolution -------------------------------------------------- #

    @classmethod
    def _resolve_server(cls, cfg: Any) -> tuple[list[str], str] | None:
        """(argv prefix, flavor) — flavor 'bin' is llama-server CLI, 'py' is
        llama_cpp.server (different flag names)."""
        if getattr(cfg, "llama_server_bin", ""):
            return ([str(cfg.llama_server_bin)], "bin")
        for name in ("llama-server", "llama-server.exe"):
            if p := shutil.which(name):
                return ([p], "bin")
        try:
            import llama_cpp  # noqa: F401

            return ([sys.executable, "-m", "llama_cpp.server"], "py")
        except Exception:
            return None

    @classmethod
    def available(cls, cfg: Any) -> bool:
        if not cls.gguf_files:
            cls.available_error = "no gguf_files declared"
            return False
        if cls._resolve_server(cfg) is None:
            cls.available_error = (
                "llama.cpp backend not installed — install the llm extra "
                "('uv pip install -e .[llm]' / mesh_server/scripts/install_llm.sh), "
                "a system llama-server ('brew install llama.cpp'), or set LLAMA_SERVER_BIN"
            )
            return False
        return True

    # -- weights ------------------------------------------------------------ #

    @classmethod
    def _gguf_dir(cls, cfg: Any) -> Path:
        return Path(cfg.model_dir) / "gguf"

    @classmethod
    def _weight_paths(cls, cfg: Any) -> list[Path]:
        return [cls._gguf_dir(cfg) / f for f in cls.gguf_files]

    @classmethod
    def downloaded(cls, cfg: Any) -> bool:
        files = list(cls.gguf_files) + ([cls.gguf_mmproj] if cls.gguf_mmproj else [])
        return bool(files) and all((cls._gguf_dir(cfg) / f).is_file() for f in files)

    @classmethod
    def download_weights(
        cls,
        cfg: Any,
        progress_cb: DownloadProgressFn | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> int:
        from huggingface_hub import hf_hub_download

        dest = cls._gguf_dir(cfg)
        dest.mkdir(parents=True, exist_ok=True)
        tcls = progress_tqdm(progress_cb, cancel_flag, cls.hf_repo)
        kwargs: dict[str, Any] = {}
        if tcls is not None and _accepts_kwarg(hf_hub_download, "tqdm_class"):
            kwargs["tqdm_class"] = tcls
        total = 0
        for fname in list(cls.gguf_files) + ([cls.gguf_mmproj] if cls.gguf_mmproj else []):
            if cancel_flag is not None and cancel_flag.is_set():
                raise DownloadCancelled(f"download cancelled: {cls.hf_repo}")
            # resumes partial transfers automatically (HF .incomplete files)
            p = hf_hub_download(cls.hf_repo, fname, local_dir=dest, **kwargs)
            try:
                total += Path(p).stat().st_size
            except OSError:
                pass
        return total

    # -- server ------------------------------------------------------------- #

    def _model_path(self) -> Path:
        if not self.gguf_files:
            raise RuntimeError(f"llm '{self.id}' declares no gguf_files")
        # multi-file GGUFs (split parts) are loaded via their first part
        return self._gguf_dir(self.cfg) / self.gguf_files[0]

    def _mmproj_path(self) -> Path | None:
        if not self.gguf_mmproj:
            return None
        p = self._gguf_dir(self.cfg) / self.gguf_mmproj
        return p if p.is_file() else None

    def server_argv(self, port: int) -> list[str]:
        resolved = self._resolve_server(self.cfg)
        if resolved is None:
            raise RuntimeError(
                f"llm '{self.id}': no llama-server binary on PATH and llama_cpp "
                "not importable — install llama-cpp-python or set LLAMA_SERVER_BIN"
            )
        prefix, flavor = resolved
        model = self._model_path()
        ctx = int(getattr(self.cfg, "llm_ctx", 0) or self.context_len or 16384)
        if flavor == "bin":
            argv = prefix + [
                "-m",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--ctx-size",
                str(ctx),
                "--jinja",  # honor the repo's chat template (tool calls too)
            ]
            if mmproj := self._mmproj_path():
                argv += ["--mmproj", str(mmproj)]
            return argv
        # llama_cpp.server — different flag names, no --mmproj support worth
        # relying on across versions; vision customs should use the binary
        return prefix + [
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--n_ctx",
            str(ctx),
        ]


def make_gguf_adapter(
    repo: str,
    files: list[str],
    mmproj: str | None = None,
    *,
    id: str,
    name: str = "",
    vision: bool = False,
    tools: bool = True,
    ctx: int = 16384,
    est_mb: int | None = None,
    custom: bool = False,
) -> type[LlmAdapter]:
    """Factory: an adapter class serving `files` (GGUF inside `repo`) via
    llama.cpp."""
    return type(
        f"LlamaCpp_{id.replace('-', '_')}",
        (LlamaCppAdapter,),
        {
            "id": id,
            "name": name or repo,
            "hf_repo": repo,
            "gguf_files": list(files),
            "gguf_mmproj": mmproj,
            "supports_vision": bool(vision or mmproj),
            "supports_tools": bool(tools),
            "context_len": int(ctx),
            "est_mem_mb": est_mb,
            "custom": bool(custom),
        },
    )
