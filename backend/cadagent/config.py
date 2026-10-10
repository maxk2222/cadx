"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Global settings. Per-project overrides live in ProjectSettings (db models)."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model_vision: str = "gpt-4o"
    llm_model_code: str = "gpt-4o"
    llm_json_mode: Literal["json_object", "json_schema", "none"] = "json_object"
    llm_temperature_code: float = 0.2
    llm_max_tokens: int = 8000
    llm_timeout_s: float = 180.0
    llm_stream: bool = True
    # streaming: abort only when no data arrives for this long (slow models OK)
    llm_idle_timeout_s: float = 180.0
    # hard cap for a single streamed call regardless of progress
    llm_total_timeout_s: float = 1800.0
    # JSON object merged into every request payload (provider-specific extras,
    # e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for Qwen3 on vLLM)
    llm_extra_body: str = ""
    # Test/dev only: path to a JSON FakeLLM script (mounted file) or inline JSON.
    llm_fake_script: str = ""

    # --- Agent defaults ---
    agent_max_iterations: int = 3
    agent_max_repairs: int = 3
    agent_pass_score: int = 80
    agent_dim_tolerance_pct: float = 2.0
    agent_dim_tolerance_mm: float = 0.5
    agent_clarification_mode: Literal["assume", "ask"] = "assume"
    agent_max_input_images: int = 8
    agent_max_tokens_per_job: int = 400_000
    agent_max_wall_s: float = 900.0
    agent_default_gen_mode: Literal["auto", "cad", "mesh", "hybrid"] = "auto"

    # --- Mesh generator (external mesh-server sidecar) ---
    mesh_base_url: str = ""
    mesh_model: str = "auto"
    mesh_timeout_s: int = 900
    mesh_fake: bool = False
    mesh_decimate_faces: int = 30_000
    mesh_max_faces_for_cad: int = 60_000

    # --- Executor ---
    exec_timeout_s: int = 180
    exec_mem_limit_mb: int = 4096

    # --- Infra ---
    data_dir: str = "./data"
    database_url: str = "postgresql://cadagent:cadagent@localhost:5432/cadagent"
    redis_url: str = "redis://localhost:6379/0"

    log_level: str = "INFO"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
