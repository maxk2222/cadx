"""Async SQLAlchemy engine + session helpers.

Accepts DATABASE_URL in either sync (`postgresql://...`) or async form.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from cadagent.config import get_settings


def _async_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    return url


def make_engine(url: str | None = None):
    return create_async_engine(
        _async_url(url or get_settings().database_url),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


engine = make_engine()
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        yield session


async def create_all() -> None:
    """Dev/test convenience: create tables without alembic."""
    import cadagent.db.models  # noqa: F401  (registers tables)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
