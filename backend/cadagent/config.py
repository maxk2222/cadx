"""Bootstrap settings required before the application database is available."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Infrastructure-only defaults; application runtime settings live in the DB."""

    model_config = SettingsConfigDict(extra="ignore")

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
