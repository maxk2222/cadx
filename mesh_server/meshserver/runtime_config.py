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

# llama.cpp --cache-type-k/-v values (server README)
_KV_CACHE_TYPES = frozenset(
    {
        "f32",
        "f16",
        "bf16",
        "q8_0",
        "q4_0",
        "q4_1",
        "iq4_nl",
        "q5_0",
        "q5_1",
        "q6_k",
        "q3_k",
        "q2_k",
        "iq3_xxs",
        "mxfp4",
    }
)


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
    if typ is str:
        if not isinstance(value, str):
            raise ValueError(f"'{key}' expects a string")
        return value.strip()
    return value


def config_view(cfg: ServerConfig) -> dict:
    """Effective values + provenance for GET /v1/config."""
    overlay = cfg.overlay()
    env_keys = {
        "llm_max_concurrent": "MESH_LLM_MAX_CONCURRENT",
        "llm_queue_wait_s": "MESH_LLM_QUEUE_WAIT_S",
        "llm_queue_fair_s": "MESH_LLM_QUEUE_FAIR_S",
        "llm_queue_max": "MESH_LLM_QUEUE_MAX",
        "llm_stream_queue_wait_s": "MESH_LLM_STREAM_QUEUE_WAIT_S",
        "llm_keepalive_s": "MESH_LLM_KEEPALIVE_S",
        "llm_prio_internal": "MESH_LLM_PRIO_INTERNAL",
        "llm_prio_external": "MESH_LLM_PRIO_EXTERNAL",
        "llm_upstream_timeout_s": "MESH_LLM_UPSTREAM_TIMEOUT_S",
        "llm_stream_idle_s": "MESH_LLM_STREAM_IDLE_S",
        "llm_failed_ttl_s": "MESH_LLM_FAILED_TTL_S",
        "llm_parallel": "MESH_LLM_PARALLEL",
        "llm_pin": "MESH_LLM_PIN",
        "openai_image_wait_s": "MESH_OPENAI_IMAGE_WAIT_S",
        "unload_idle_s": "MESH_UNLOAD_IDLE_S",
        "job_min_free_mb": "MESH_JOB_MIN_FREE_MB",
        "mem_headroom_mb": "MESH_MEM_HEADROOM_MB",
        "llm_ctx": "MESH_LLM_CTX",
        "llm_kv_bits": "MESH_LLM_KV_BITS",
        "llm_prefill_step": "MESH_LLM_PREFILL_STEP",
        "llm_draft": "MESH_LLM_DRAFT",
        "llm_max_num_seqs": "MESH_LLM_MAX_NUM_SEQS",
        "llm_prompt_cache_mb": "MESH_LLM_PROMPT_CACHE_MB",
        "llm_gpu_layers": "MESH_LLM_GPU_LAYERS",
        "llm_kv_unified": "MESH_LLM_KV_UNIFIED",
        "llm_kv_per_slot": "MESH_LLM_KV_PER_SLOT",
        "llm_kv_cache_type": "MESH_LLM_KV_CACHE_TYPE",
        "llm_flash_attn": "MESH_LLM_FLASH_ATTN",
        "llm_apc": "MESH_LLM_APC",
        "llm_kv_scheme": "MESH_LLM_KV_SCHEME",
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
            if isinstance(v, (int, float)) and v < 0 and key != "llm_gpu_layers":
                raise ValueError(f"'{key}' must be >= 0")
            if key == "llm_gpu_layers" and v < -1:
                raise ValueError("'llm_gpu_layers' must be -1 (all) or >= 0")
            if key == "llm_ctx" and v and v < 4096:
                raise ValueError("'llm_ctx' must be 0 (catalog default) or >= 4096")
            if key == "llm_kv_bits" and v and not 2 <= v <= 8:
                raise ValueError("'llm_kv_bits' must be 0 (off) or in 2..8")
            if key == "llm_kv_scheme" and v and v not in ("uniform", "turboquant"):
                raise ValueError("'llm_kv_scheme' must be '', 'uniform', or 'turboquant'")
            if key == "llm_kv_cache_type" and v and v.lower() not in _KV_CACHE_TYPES:
                raise ValueError(
                    "'llm_kv_cache_type' must be one of " + ", ".join(sorted(_KV_CACHE_TYPES))
                )
            if key == "llm_kv_per_slot" and v and v < 1024:
                raise ValueError("'llm_kv_per_slot' must be 0 (off) or >= 1024")
            if key == "llm_draft" and v:
                kinds = ("dflash", "eagle3", "mtp")
                tail = v.rsplit(":", 1)[-1] if ":" in v else ""
                if v != "auto" and v not in kinds and tail and tail not in kinds:
                    raise ValueError(
                        "'llm_draft' must be '', 'auto', a kind, or 'hf/repo[:dflash|eagle3|mtp]'"
                    )
            validated[key] = v
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise HTTPException(422, "; ".join(errors))
    overlay.update(validated)
    cfg.set_overlay(overlay)
    return config_view(cfg)
