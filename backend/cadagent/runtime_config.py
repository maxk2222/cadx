"""DB-backed application runtime settings and one-time legacy environment import."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from cadagent.db.models import AppSetting


class RuntimeSettings(BaseModel):
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model_vision: str = "gpt-4o"
    llm_model_code: str = "gpt-4o"
    llm_json_mode: Literal["json_object", "json_schema", "none"] = "json_object"
    llm_temperature_code: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=8000, ge=1, le=2_000_000)
    llm_timeout_s: float = Field(default=180, gt=0)
    llm_stream: bool = True
    llm_idle_timeout_s: float = Field(default=180, gt=0)
    llm_total_timeout_s: float = Field(default=1800, gt=0)
    llm_extra_body: str = ""
    llm_tools: Literal["auto", "on", "off"] = "auto"
    agent_max_iterations: int = Field(default=3, ge=1, le=200)
    agent_max_repairs: int = Field(default=3, ge=0, le=200)
    agent_pass_score: int = Field(default=80, ge=0, le=100)
    agent_dim_tolerance_pct: float = Field(default=2, ge=0, le=100)
    agent_dim_tolerance_mm: float = Field(default=0.5, ge=0, le=1000)
    agent_clarification_mode: Literal["assume", "ask"] = "assume"
    agent_max_input_images: int = Field(default=8, ge=0, le=100)
    agent_max_tokens_per_job: int = Field(default=400_000, ge=1)
    agent_max_wall_s: float = Field(default=900, gt=0)
    agent_default_gen_mode: Literal["auto", "cad", "mesh", "hybrid", "topopt"] = "auto"
    agent_param_autofix: bool = True
    agent_patch_mode: Literal["auto", "always", "never"] = "auto"
    agent_patch_retries: int = Field(default=2, ge=0, le=20)
    agent_feature_feedback: bool = True
    agent_critique_reask: bool = True
    agent_tool_max_calls: int = Field(default=10, ge=1, le=100)
    agent_tool_max_calls_info: int = Field(default=6, ge=1, le=100)
    agent_tool_verify: bool = True
    agent_tool_verify_max: int = Field(default=2, ge=0, le=20)
    agent_mesh_tools: bool = True
    agent_mesh_max_ops: int = Field(default=24, ge=1, le=200)
    agent_mesh_preview_max: int = Field(default=4, ge=0, le=20)
    agent_mesh_refine_rounds: int = Field(default=1, ge=0, le=10)
    fem_voxel_mm: float = Field(default=1.5, gt=0, le=100)
    fem_max_elements: int = Field(default=150_000, ge=1, le=5_000_000)
    fem_timeout_s: int = Field(default=600, ge=1, le=86_400)
    topopt_timeout_s: int = Field(default=2700, ge=1, le=86_400)
    topopt_max_iterations: int = Field(default=80, ge=1, le=1000)
    topopt_default_volfrac: float = Field(default=0.4, gt=0, le=1)
    topopt_default_rmin_mm: float = Field(default=3, gt=0, le=1000)
    mesh_base_url: str = ""
    mesh_model: str = "auto"
    mesh_timeout_s: int = Field(default=900, ge=1, le=86_400)
    mesh_decimate_faces: int = Field(default=30_000, ge=1)
    mesh_max_faces_for_cad: int = Field(default=60_000, ge=1)
    exec_timeout_s: int = Field(default=180, ge=1, le=86_400)
    exec_mem_limit_mb: int = Field(default=4096, ge=256, le=1_048_576)

    @field_validator("llm_extra_body")
    @classmethod
    def validate_extra_body(cls, value: str) -> str:
        if not value:
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("must be a JSON object")
        return value

    def to_storage(self) -> dict[str, Any]:
        """Sparse dict: only non-default values are persisted. This keeps the
        legacy-env import precedence honest — fields still at their default
        were never configured and may be filled by an import."""
        value = self.model_dump(mode="python", exclude_defaults=True)
        key = self.llm_api_key.get_secret_value()
        if key:
            value["llm_api_key"] = key
        return value

    def public_values(self) -> dict[str, Any]:
        value = self.model_dump(exclude={"llm_api_key"})
        value["llm_api_key_set"] = bool(self.llm_api_key.get_secret_value())
        return value

    def executor_values(self) -> dict[str, Any]:
        """Subset shipped to the cad-executor process inside each task payload."""
        return self.model_dump(mode="json", include=_EXECUTOR_FIELDS)


_EXECUTOR_FIELDS = {
    "exec_timeout_s",
    "exec_mem_limit_mb",
    "fem_voxel_mm",
    "fem_max_elements",
    "topopt_default_volfrac",
    "topopt_default_rmin_mm",
    "topopt_max_iterations",
    "mesh_max_faces_for_cad",
    "mesh_decimate_faces",
}

_SETTING_KEYS = {
    "llm": {
        "llm_base_url": "llm_base_url",
        "llm_model_vision": "llm_model_vision",
        "llm_model_code": "llm_model_code",
        "llm_json_mode": "llm_json_mode",
    },
    "mesh": {
        "mesh_base_url": "mesh_base_url",
        "mesh_model": "mesh_model",
        "agent_default_gen_mode": "agent_default_gen_mode",
    },
}

_ENV_TO_FIELD = {
    "LLM_BASE_URL": "llm_base_url",
    "LLM_API_KEY": "llm_api_key",
    "LLM_MODEL_VISION": "llm_model_vision",
    "LLM_MODEL_CODE": "llm_model_code",
    "LLM_JSON_MODE": "llm_json_mode",
    "LLM_TEMPERATURE_CODE": "llm_temperature_code",
    "LLM_MAX_TOKENS": "llm_max_tokens",
    "LLM_TIMEOUT_S": "llm_timeout_s",
    "LLM_STREAM": "llm_stream",
    "LLM_IDLE_TIMEOUT_S": "llm_idle_timeout_s",
    "LLM_TOTAL_TIMEOUT_S": "llm_total_timeout_s",
    "LLM_EXTRA_BODY": "llm_extra_body",
    "LLM_TOOLS": "llm_tools",
    "AGENT_MAX_ITERATIONS": "agent_max_iterations",
    "AGENT_MAX_REPAIRS": "agent_max_repairs",
    "AGENT_PASS_SCORE": "agent_pass_score",
    "AGENT_DIM_TOLERANCE_PCT": "agent_dim_tolerance_pct",
    "AGENT_DIM_TOLERANCE_MM": "agent_dim_tolerance_mm",
    "AGENT_CLARIFICATION_MODE": "agent_clarification_mode",
    "AGENT_MAX_INPUT_IMAGES": "agent_max_input_images",
    "AGENT_MAX_TOKENS_PER_JOB": "agent_max_tokens_per_job",
    "AGENT_MAX_WALL_S": "agent_max_wall_s",
    "AGENT_DEFAULT_GEN_MODE": "agent_default_gen_mode",
    "AGENT_PARAM_AUTOFIX": "agent_param_autofix",
    "AGENT_PATCH_MODE": "agent_patch_mode",
    "AGENT_PATCH_RETRIES": "agent_patch_retries",
    "AGENT_FEATURE_FEEDBACK": "agent_feature_feedback",
    "AGENT_CRITIQUE_REASK": "agent_critique_reask",
    "AGENT_TOOL_MAX_CALLS": "agent_tool_max_calls",
    "AGENT_TOOL_MAX_CALLS_INFO": "agent_tool_max_calls_info",
    "AGENT_TOOL_VERIFY": "agent_tool_verify",
    "AGENT_TOOL_VERIFY_MAX": "agent_tool_verify_max",
    "AGENT_MESH_TOOLS": "agent_mesh_tools",
    "AGENT_MESH_MAX_OPS": "agent_mesh_max_ops",
    "AGENT_MESH_PREVIEW_MAX": "agent_mesh_preview_max",
    "AGENT_MESH_REFINE_ROUNDS": "agent_mesh_refine_rounds",
    "FEM_VOXEL_MM": "fem_voxel_mm",
    "FEM_MAX_ELEMENTS": "fem_max_elements",
    "FEM_TIMEOUT_S": "fem_timeout_s",
    "TOPOPT_TIMEOUT_S": "topopt_timeout_s",
    "TOPOPT_MAX_ITERATIONS": "topopt_max_iterations",
    "TOPOPT_DEFAULT_VOLFRAC": "topopt_default_volfrac",
    "TOPOPT_DEFAULT_RMIN_MM": "topopt_default_rmin_mm",
    "MESH_BASE_URL": "mesh_base_url",
    "MESH_MODEL": "mesh_model",
    "MESH_TIMEOUT_S": "mesh_timeout_s",
    "MESH_DECIMATE_FACES": "mesh_decimate_faces",
    "MESH_MAX_FACES_FOR_CAD": "mesh_max_faces_for_cad",
    "EXEC_TIMEOUT_S": "exec_timeout_s",
    "EXEC_MEM_LIMIT_MB": "exec_mem_limit_mb",
}

_INFRA_ENV_KEYS = {
    "DATABASE_URL",
    "REDIS_URL",
    "DATA_DIR",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "MESH_HOST",
    "MESH_PORT",
    "MESH_DEVICE",
    "MESH_MODEL_DIR",
    "MESH_JOB_DIR",
    "MESH_JOB_TTL_S",
    "MESH_MAX_QUEUE",
    "MESH_UNLOAD_IDLE_S",
    "MESH_FAKE",
    "PYVISTA_OFF_SCREEN",
    "LOG_LEVEL",
}


def _legacy_db_values(
    runtime_row: AppSetting | None, legacy_rows: dict[str, AppSetting | None]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group, key_map in _SETTING_KEYS.items():
        row = legacy_rows.get(group)
        if row:
            values.update(
                {field: row.value[key] for key, field in key_map.items() if key in row.value}
            )
    if runtime_row:
        raw = runtime_row.value or {}
        values.update(raw.get("settings", raw))
    return values


async def load_runtime_settings(db: AsyncSession) -> RuntimeSettings:
    runtime_row = await db.get(AppSetting, "runtime")
    legacy_rows = {key: await db.get(AppSetting, key) for key in _SETTING_KEYS}
    values = RuntimeSettings().to_storage()
    values.update(_legacy_db_values(runtime_row, legacy_rows))
    return RuntimeSettings.model_validate(values)


async def save_runtime_settings(
    db: AsyncSession,
    settings: RuntimeSettings,
    *,
    legacy_env_imported: bool | None = None,
    seeded: bool = False,
) -> None:
    row = await db.get(AppSetting, "runtime")
    old = (row.value or {}) if row else {}
    imported = legacy_env_imported
    if imported is None:
        imported = bool(old.get("legacy_env_imported", False))
    stored = {
        "schema_version": 1,
        "legacy_env_imported": imported,
        "seeded": seeded,
        "settings": settings.to_storage(),
    }
    if row is None:
        db.add(AppSetting(key="runtime", value=stored))
    else:
        row.value = stored
    for key in _SETTING_KEYS:
        legacy = await db.get(AppSetting, key)
        if legacy is not None:
            await db.delete(legacy)


def legacy_env_values(path: Path) -> tuple[dict[str, Any], list[str]]:
    raw = dotenv_values(path)
    imported: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in raw.items():
        if key in _ENV_TO_FIELD:
            if value is not None and value != "":
                imported[_ENV_TO_FIELD[key]] = value
        elif key in _INFRA_ENV_KEYS or key == "LLM_FAKE_SCRIPT":
            ignored.append(key)
        elif value is not None:
            raise ValueError(
                f"unsupported legacy setting '{key}'; preserve the file and migrate it manually"
            )
    return imported, ignored


async def runtime_from_db_or_default() -> RuntimeSettings:
    """CLI/eval entry point: DB snapshot when reachable, else code defaults."""
    try:
        from cadagent.db.session import session_scope

        async with session_scope() as db:
            return await load_runtime_settings(db)
    except Exception:
        return RuntimeSettings()


async def ensure_runtime_seeded(db: AsyncSession) -> bool:
    """Create the 'runtime' row once. mesh_base_url/mesh_model are seeded from
    MESH_BASE_URL/MESH_MODEL env vars (deployment topology set by the launcher);
    afterwards the DB row is the only source and env changes are ignored."""
    import os

    if await db.get(AppSetting, "runtime") is not None:
        return False
    settings = RuntimeSettings()
    seed_url = os.environ.get("MESH_BASE_URL", "").strip()
    if seed_url:
        settings.mesh_base_url = seed_url
    seed_model = os.environ.get("MESH_MODEL", "").strip()
    if seed_model:
        settings.mesh_model = seed_model
    await save_runtime_settings(db, settings, legacy_env_imported=False, seeded=True)
    return True


async def import_legacy_env(db: AsyncSession, path: Path) -> dict[str, Any]:
    row = await db.get(AppSetting, "runtime")
    if row and (row.value or {}).get("legacy_env_imported"):
        return {"already_imported": True, "imported": [], "ignored": []}

    current = await load_runtime_settings(db)
    legacy_rows = {key: await db.get(AppSetting, key) for key in _SETTING_KEYS}
    # an auto-seeded row holds only defaults/env seed — it must not block a
    # legacy .env import; real (user-saved) DB values always win
    seed_only = bool(
        row and (row.value or {}).get("seeded") and not (row.value or {}).get("legacy_env_imported")
    )
    existing = _legacy_db_values(None if seed_only else row, legacy_rows)
    imported, ignored = legacy_env_values(path)
    for key in existing:
        imported.pop(key, None)
    values = current.to_storage()
    values.update(imported)
    settings = RuntimeSettings.model_validate(values)
    await save_runtime_settings(db, settings, legacy_env_imported=True)
    return {
        "already_imported": False,
        "imported": sorted(imported),
        "ignored_infrastructure": sorted(ignored),
    }
