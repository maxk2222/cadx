"""Bootstrap settings required before the application database is available."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Infrastructure-only defaults; application runtime settings live in the DB."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    data_dir: str = "./data"
    database_url: str = "postgresql://cadx:cadx@localhost:5432/cadx"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    # bearer token for the MCP server's HTTP transport (env CADX_MCP_TOKEN);
    # empty = unauthenticated (local/dev default)
    mcp_token: str = Field(default="", validation_alias="CADX_MCP_TOKEN")
    # outer cap on one agent job (arq job_timeout); slow local models with
    # meshgen + many improve iterations legitimately exceed 1h
    agent_job_timeout_s: int = 14400

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
