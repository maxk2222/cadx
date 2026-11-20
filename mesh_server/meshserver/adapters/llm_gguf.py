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

import re
import shutil
import subprocess
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

# `llama-server --help` flag sets, cached per resolved argv prefix — an
# unknown flag crashes the spawn, so newer knobs (unified KV pool, cache
# types, flash-attn, prompt cache) are only emitted when the binary
# advertises them. None = probe failed (treat as "only classic flags").
_FLAG_RE = re.compile(r"-{1,2}[a-zA-Z][a-zA-Z0-9_-]*")
_HELP_CACHE: dict[str, set[str] | None] = {}


def _server_flags(prefix: list[str]) -> set[str] | None:
    """Flag set from `<prefix> --help`; None when the probe itself failed."""
    key = " ".join(prefix)
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    flags: set[str] | None = None
    try:
        r = subprocess.run(
            prefix + ["--help"],
            capture_output=True,
            text=True,
            # python -m llama_cpp.server imports torch — cold probes can
            # exceed the usual instant --help; cached afterwards anyway
            timeout=20,
            check=False,
        )
        flags = set(_FLAG_RE.findall(r.stdout + r.stderr))
    except (OSError, subprocess.SubprocessError, ValueError):
        flags = None
    _HELP_CACHE[key] = flags
    if flags is None:
        log.warning("llm.gguf.flag_probe_failed", argv=key)
    return flags


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
        effective = getattr(self.cfg, "effective", None)

        def get(name: str, d: Any = None) -> Any:
            return effective(name) if effective else getattr(self.cfg, name, d)

        ctx = int(get("llm_ctx", 0) or self.context_len or 16384)
        parallel = int(get("llm_parallel", 1) or 1)
        ngl = int(get("llm_gpu_layers", -1))
        ctype = str(get("llm_kv_cache_type", "") or "").strip()
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
                "--n-gpu-layers",
                str(ngl),
            ]
            if mmproj := self._mmproj_path():
                argv += ["--mmproj", str(mmproj)]
            flags = _server_flags(prefix) or set()
            # unified KV pool (-kvu): -c becomes ONE pool shared by all
            # parallel slots (vLLM-style) instead of ctx-per-slot division.
            # Explicit --parallel without -kvu disables it on new builds —
            # emit both deliberately.
            if get("llm_kv_unified", True):
                if "--kv-unified" in flags:
                    argv += ["--kv-unified"]
                    per_slot = int(get("llm_kv_per_slot", 0) or 0)
                    if per_slot and "--kv-unified-per-slot" in flags:
                        argv += ["--kv-unified-per-slot", str(per_slot)]
                else:
                    log.warning(
                        "llm.gguf.kvu_unsupported",
                        model=self.id,
                        hint="update llama-server (>= ~b70xx) for the shared KV pool",
                    )
            if ctype:
                if "--cache-type-k" in flags:
                    argv += ["--cache-type-k", ctype, "--cache-type-v", ctype]
                else:
                    log.warning("llm.gguf.cache_type_unsupported", model=self.id)
            if get("llm_flash_attn", True):
                if "--flash-attn" in flags:
                    argv += ["--flash-attn"]
                elif "-fa" in flags:
                    argv += ["-fa"]
            cache_mb = int(get("llm_prompt_cache_mb", 0) or 0)
            if cache_mb and "--cache-ram" in flags:
                if mmproj:
                    # llama.cpp's host-RAM prompt cache is incompatible
                    # with mtmd — never emit it on a projector build
                    log.warning(
                        "llm.gguf.cache_ram_mtmd",
                        model=self.id,
                        hint="prompt cache disabled on multimodal (mmproj) servers",
                    )
                else:
                    argv += ["--cache-ram", str(cache_mb)]
                    if "--cache-idle-slots" in flags:
                        argv += ["--cache-idle-slots"]
            # runtime-mutable slot count — concurrent client requests get
            # real parallel decode (continuous batching) instead of the
            # default single sequence queue; with -kvu the ctx pool is
            # shared dynamically, without it each slot gets ctx/parallel
            if parallel > 1:
                argv += ["--parallel", str(parallel)]
            if kv_bpt := int(getattr(self, "kv_bytes_per_token", 0) or 0):
                eff_b = kv_bpt // (4 if ctype.startswith("q4") else 2 if ctype == "q8_0" else 1)
                log.info(
                    "llm.gguf.kv_budget",
                    model=self.id,
                    ctx=ctx,
                    parallel=parallel,
                    kv_mb_est=ctx * eff_b // 1_000_000,
                )
            return argv
        # llama_cpp.server — different flag names, no KVU/continuous batching
        # (requests serialize); vision customs should use the binary. Kept as
        # a working fallback, incl. the GPU fix: its default n_gpu_layers=0
        # would silently run the model on CPU.
        argv = prefix + [
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--n_ctx",
            str(ctx),
            "--n_gpu_layers",
            str(ngl),  # ancient flag, always safe — fixes the CPU default
        ]
        # optional flags probed too — llama-cpp-python versions vary
        py_flags = _server_flags(prefix) or set()
        if ctype and "--cache_type_k" in py_flags:
            argv += ["--cache_type_k", ctype, "--cache_type_v", ctype]
        if get("llm_flash_attn", True) and "--flash_attn" in py_flags:
            argv += ["--flash_attn"]
        if get("llm_kv_unified", True) or parallel > 1:
            log.warning(
                "llm.gguf.pyflavor_serial",
                model=self.id,
                hint="install a llama-server CUDA binary "
                "(mesh_server/scripts/install_llm_cuda.sh) for parallel slots / unified KV",
            )
        return argv


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
    kv_bytes_per_token: int = 0,
    audio: bool = False,
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
            "supports_audio": bool(audio),
            "supports_tools": bool(tools),
            "context_len": int(ctx),
            "est_mem_mb": est_mb,
            "custom": bool(custom),
            "kv_bytes_per_token": int(kv_bytes_per_token),
        },
    )
