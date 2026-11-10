"""GET/PUT /v1/config — runtime-mutable scheduling knobs.

The overlay file (``model_dir/server_config.json``) follows the same
pattern as ``custom_models.json``: a small JSON dict of
``RUNTIME_MUTABLE`` keys, validated per-key, atomically replaced on write.
``applies`` tells clients whether a change is picked up per-request
("immediate") or baked into the next managed-server argv
("next_model_load").
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException

from meshserver.config import RUNTIME_MUTABLE, ServerConfig


def _coerce(key: str, value: Any) -> Any:
    typ, _ = RUNTIME_MUTABLE[key]
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"'{key}' expects a boolean")
    if typ is int:
        if isinstance(value, bool):  # bool is an int subclass — reject explicitly
            raise ValueError(f"'{key}' expects an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError(f"'{key}' expects an integer")
    if typ is float:
        if isinstance(value, bool):
            raise ValueError(f"'{key}' expects a number")
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"'{key}' expects a number")
    return value


def config_view(cfg: ServerConfig) -> dict:
    """Effective values + provenance for GET /v1/config."""
    overlay = cfg.overlay()
    env_keys = {
        "llm_max_concurrent": "MESH_LLM_MAX_CONCURRENT",
        "llm_queue_wait_s": "MESH_LLM_QUEUE_WAIT_S",
        "llm_parallel": "MESH_LLM_PARALLEL",
        "llm_pin": "MESH_LLM_PIN",
        "openai_image_wait_s": "MESH_OPENAI_IMAGE_WAIT_S",
        "unload_idle_s": "MESH_UNLOAD_IDLE_S",
        "job_min_free_mb": "MESH_JOB_MIN_FREE_MB",
        "mem_headroom_mb": "MESH_MEM_HEADROOM_MB",
    }
    out: dict[str, Any] = {}
    for key, (_typ, applies) in RUNTIME_MUTABLE.items():
        if key in overlay:
            source = "overlay"
        elif os.environ.get(env_keys.get(key, ""), "") != "":
            source = "env"
        else:
            source = "default"
        out[key] = {
            "value": cfg.effective(key),
            "applies": applies,
            "source": source,
        }
    return {"config": out, "overlay_path": str(cfg.overlay_path)}


def apply_patch(cfg: ServerConfig, patch: dict) -> dict:
    """Validate + persist a config patch; returns the full new view.

    Unknown keys are rejected (a typo must not silently no-op); a ``null``
    value deletes the key from the overlay so the env/default applies again.
    """
    unknown = [k for k in patch if k not in RUNTIME_MUTABLE]
    if unknown:
        raise HTTPException(
            422,
            f"unknown config key(s): {', '.join(sorted(unknown))} — "
            f"mutable: {', '.join(sorted(RUNTIME_MUTABLE))}",
        )
    errors: list[str] = []
    validated: dict[str, Any] = {}
    overlay = cfg.overlay()
    for key, value in patch.items():
        if value is None:
            overlay.pop(key, None)  # explicit reset to env/default
            continue
        try:
            v = _coerce(key, value)
            if isinstance(v, (int, float)) and v < 0:
                raise ValueError(f"'{key}' must be >= 0")
            validated[key] = v
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise HTTPException(422, "; ".join(errors))
    overlay.update(validated)
    cfg.set_overlay(overlay)
    return config_view(cfg)
