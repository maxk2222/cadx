"""Disk cache for image->3D mesh generations.

Generative calls are the slowest single step (~6 min) and job restarts or
strategy retries re-issue identical requests. A content hash over the input
images + every generation parameter makes the cache collision-free; entries
live under data/mesh_cache and are capped at 50 files (oldest pruned).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from cadx.logging import get_logger

log = get_logger(__name__)

_MAX_ENTRIES = 50


def _cache_dir() -> Path:
    from cadx import storage

    return storage.data_root() / "mesh_cache"


def cache_key(
    image_paths: list[Path],
    roles: list[str],
    *,
    model: str,
    seed: int,
    steps: int | None,
    guidance: float | None,
    resolution: int | None,
    subject: str,
) -> str:
    """SHA-256 over image contents and all generation parameters."""
    h = hashlib.sha256()
    for p in sorted(image_paths, key=lambda x: x.name):
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(str(p).encode())
    meta = {
        "roles": roles,
        "model": model,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "resolution": resolution,
        "subject": subject,
    }
    h.update(repr(sorted(meta.items())).encode())
    return h.hexdigest()[:32]


def lookup(key: str) -> Path | None:
    """Return the cached GLB path for `key`, or None."""
    p = _cache_dir() / f"{key}.glb"
    return p if p.exists() else None


def lookup_meta(key: str) -> dict[str, Any]:
    """Sidecar metadata of a cached entry (e.g. the adapter that produced
    it) — empty dict for entries predating the sidecar."""
    try:
        import json

        p = _cache_dir() / f"{key}.meta.json"
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def store(key: str, mesh_path: Path, meta: dict[str, Any] | None = None) -> None:
    """Copy a generated GLB into the cache and prune old entries.

    `meta` is an optional sidecar (`<key>.meta.json`) for provenance that the
    key cannot encode — e.g. the adapter that actually produced the mesh when
    the request model was "auto".
    """
    try:
        import json

        d = _cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mesh_path, d / f"{key}.glb")
        if meta:
            (d / f"{key}.meta.json").write_text(json.dumps(meta))
        files = sorted(d.glob("*.glb"), key=lambda p: p.stat().st_mtime)
        for old in files[: max(0, len(files) - _MAX_ENTRIES)]:
            old.unlink(missing_ok=True)
            (d / f"{old.stem}.meta.json").unlink(missing_ok=True)
    except Exception as e:
        log.warning("mesh_cache.store_failed", error=str(e)[:200])
