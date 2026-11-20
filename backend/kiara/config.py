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
    database_url: str = "postgresql://kiara:kiara@localhost:5432/kiara"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    # bearer token for the MCP server's HTTP transport (env KIARA_MCP_TOKEN);
    # empty = unauthenticated (local/dev default)
    mcp_token: str = Field(default="", validation_alias="KIARA_MCP_TOKEN")
    # outer cap on one agent job (arq job_timeout); slow local models with
    # meshgen + many improve iterations legitimately exceed 1h
    agent_job_timeout_s: int = 14400
    # AR-guide media upload caps + assimp conversion (KIARA web-suite parity)
    media_max_image_bytes: int = 25 * 1024 * 1024
    media_max_video_bytes: int = 500 * 1024 * 1024
    media_max_audio_bytes: int = 100 * 1024 * 1024
    media_max_cad_bytes: int = 100 * 1024 * 1024
    media_max_image_pixels: int = 50_000_000
    assimp_docker_image: str = "kiara-assimp:5.3.1"
    assimp_timeout_seconds: int = 120

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
