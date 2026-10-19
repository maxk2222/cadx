"""LLM model catalog: built-in entries + user-registered custom repos.

Built-ins are a small curated list per backend (mlx-lm on Apple Silicon,
llama.cpp elsewhere). Custom entries persist to
``cfg.model_dir/"custom_models.json"`` — one JSON list of::

    {"repo": "org/name", "backend": "mlx-lm"|"llama.cpp"|"",
     "name": str, "vision": bool, "tools": bool, "est_mem_mb": int|null,
     "files": [gguf...], "mmproj": str|null, "ctx": int}

``backend`` empty resolves to mlx-lm on darwin-arm64, llama.cpp elsewhere;
entries missing pieces their backend needs (e.g. no gguf files for
llama.cpp) are skipped with a warning rather than breaking the catalog.
"""

from __future__ import annotations

import contextlib
import json
import platform
import re
from pathlib import Path
from typing import Any

from meshserver.adapters.llm_base import LlmAdapter
from meshserver.adapters.llm_gguf import make_gguf_adapter
from meshserver.adapters.llm_mlx import make_mlx_adapter
from meshserver.logging import get_logger
from meshserver.schemas import ModelInfo, RegisterModelRequest

log = get_logger(__name__)

# HF repo ids: "org/name" — alnum start, then alnum/./_/- on each side
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def _default_backend() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx-lm"
    return "llama.cpp"


def _slug_id(repo: str, backend: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", repo.split("/")[-1].lower()).strip("-")
    suffix = "mlx" if backend == "mlx-lm" else "gguf"
    return f"llm-{slug}-{suffix}"


# -- built-in catalog ------------------------------------------------------- #


def builtin_llm_classes() -> list[type[LlmAdapter]]:
    """Curated defaults — small enough for unified memory, ungated repos."""
    return [
        make_mlx_adapter(
            "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            id="qwen2.5-vl-7b-mlx",
            name="Qwen2.5-VL 7B Instruct 4-bit (MLX)",
            vision=True,
            ctx=32768,
            est_mb=5600,
        ),
        make_mlx_adapter(
            "mlx-community/Qwen3-8B-4bit",
            id="qwen3-8b-mlx",
            name="Qwen3 8B 4-bit (MLX)",
            vision=False,
            ctx=32768,
            est_mb=5200,
        ),
        make_gguf_adapter(
            "bartowski/Qwen2.5-7B-Instruct-GGUF",
            ["Qwen2.5-7B-Instruct-Q4_K_M.gguf"],
            id="qwen2.5-7b-gguf",
            name="Qwen2.5 7B Instruct Q4_K_M (llama.cpp)",
            ctx=32768,
            est_mb=5000,
        ),
    ]


# -- custom entries --------------------------------------------------------- #


def _custom_path(cfg: Any) -> Path:
    return Path(cfg.model_dir) / "custom_models.json"


def _load_entries(cfg: Any) -> list[dict]:
    p = _custom_path(cfg)
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        return []
    except Exception as e:
        log.warning("llm.custom.parse_failed", path=str(p), error=str(e))
        return []
    if not isinstance(data, list):
        log.warning("llm.custom.bad_shape", path=str(p))
        return []
    return [e for e in data if isinstance(e, dict)]


def _save_entries(cfg: Any, entries: list[dict]) -> None:
    p = _custom_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2) + "\n")


def _infer_gguf_files(repo: str) -> tuple[list[str], str | None]:
    """Best-effort: pick one GGUF (+ mmproj) from the repo file list.

    Prefers a Q4_K_M quant — the sane default for unified memory — else the
    first non-mmproj .gguf. Network call; callers must treat failure as
    non-fatal (the entry is persisted anyway and skipped until fixed).
    """
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    mmproj = next((f for f in files if "mmproj" in Path(f).name.lower()), None)
    plain = [f for f in files if f != mmproj]
    pick = next((f for f in plain if "q4_k_m" in Path(f).name.lower()), None)
    return ([pick or plain[0]] if plain else []), mmproj


def _entry_id(entry: dict) -> str:
    repo = str(entry.get("repo", ""))
    backend = str(entry.get("backend") or "") or _default_backend()
    return str(entry.get("id") or "") or _slug_id(repo, backend)


def _entry_to_class(entry: dict) -> type[LlmAdapter] | None:
    repo = str(entry.get("repo", "")).strip()
    if not _REPO_RE.match(repo):
        log.warning("llm.custom.bad_repo", repo=repo)
        return None
    backend = str(entry.get("backend") or "") or _default_backend()
    ctx = int(entry.get("ctx") or entry.get("context_len") or 16384)
    common: dict[str, Any] = {
        "id": _entry_id(entry),
        "name": str(entry.get("name") or repo),
        "vision": bool(entry.get("vision", False)),
        "tools": bool(entry.get("tools", True)),
        "ctx": ctx,
        "est_mb": entry.get("est_mem_mb"),
        "custom": True,
    }
    if backend == "mlx-lm":
        return make_mlx_adapter(repo, **common)
    if backend == "llama.cpp":
        files = entry.get("files") or entry.get("gguf_files") or []
        if not files:
            log.warning(
                "llm.custom.skip_no_files",
                repo=repo,
                hint="llama.cpp entries need a 'files' list of GGUF names",
            )
            return None
        return make_gguf_adapter(repo, list(files), mmproj=entry.get("mmproj"), **common)
    log.warning("llm.custom.unknown_backend", repo=repo, backend=backend)
    return None


def custom_llm_classes(cfg: Any) -> list[type[LlmAdapter]]:
    """Adapter classes for persisted custom entries (skips broken ones)."""
    out: list[type[LlmAdapter]] = []
    for entry in _load_entries(cfg):
        cls = _entry_to_class(entry)
        if cls is not None:
            out.append(cls)
    return out


def llm_adapter_classes(cfg: Any) -> list[type[LlmAdapter]]:
    """Full catalog: built-ins + custom. Custom wins on id collisions."""
    out: dict[str, type[LlmAdapter]] = {}
    for cls in builtin_llm_classes() + custom_llm_classes(cfg):
        if cls.id in out:
            log.warning("llm.catalog.duplicate_id", id=cls.id)
        out[cls.id] = cls
    return list(out.values())


def register_custom_model(cfg: Any, req: RegisterModelRequest) -> ModelInfo:
    """Validate + persist a custom model entry; returns its ModelInfo."""
    repo = req.repo.strip()
    if not _REPO_RE.match(repo):
        raise ValueError(f"invalid repo '{req.repo}' — expected 'org/name'")
    if req.kind != "llm":
        raise ValueError(f"only kind='llm' models can be registered (got '{req.kind}')")
    backend = req.backend or _default_backend()
    entry: dict[str, Any] = {
        "repo": repo,
        "backend": backend,
        "name": req.name or repo,
        "vision": req.vision,
        "tools": req.tools,
        "est_mem_mb": req.est_mem_mb,
    }
    if backend == "llama.cpp":
        # no file list on the request — try to infer one from the repo so
        # the entry is usable immediately; stays skippable when offline
        try:
            files, mmproj = _infer_gguf_files(repo)
            if files:
                entry["files"] = files
            if mmproj:
                entry["mmproj"] = mmproj
                entry["vision"] = True
        except Exception as e:
            log.warning("llm.custom.infer_files_failed", repo=repo, error=str(e))
    cls = _entry_to_class(entry)
    if cls is None:
        # nothing persisted — e.g. llama.cpp backend with no resolvable files
        raise ValueError(f"model '{repo}' is incomplete for backend '{backend}'")
    entries = _load_entries(cfg)
    entries = [e for e in entries if _entry_id(e) != _entry_id(entry)]  # idempotent re-register
    entries.append(entry)
    _save_entries(cfg, entries)
    log.info("llm.custom.registered", id=cls.id, repo=repo, backend=backend)
    return ModelInfo(
        id=cls.id,
        name=cls.name,
        device=cls.device,
        backend=cls.backend,
        kind="llm",
        supports_vision=cls.supports_vision,
        supports_tools=cls.supports_tools,
        est_mem_mb=cls.est_mem_mb,
        custom=True,
        pinned=bool(getattr(cfg, "llm_pin", False)),
        available=cls.available(cfg),
        downloaded=cls.downloaded(cfg),
    )


def unregister_custom_model(cfg: Any, model_id: str) -> bool:
    """Drop a custom entry by resolved id. Returns False when absent."""
    entries = _load_entries(cfg)
    keep = [e for e in entries if _entry_id(e) != model_id]
    if len(keep) == len(entries):
        return False
    _save_entries(cfg, keep)
    log.info("llm.custom.unregistered", id=model_id)
    # drop the persisted file entirely when nothing custom remains
    if not keep:
        with contextlib.suppress(OSError):
            _custom_path(cfg).unlink()
    return True
