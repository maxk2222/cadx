"""Shared API dependencies: DB session + arq enqueue helper."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import arq
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from cadagent.config import get_settings
from cadagent.db.session import async_session

AGENT_QUEUE = "agent"

_pool: ArqRedis | None = None


async def get_pool() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await arq.create_pool(
            arq.connections.RedisSettings.from_dsn(get_settings().redis_url)
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


async def enqueue(func: str, payload: dict[str, Any]) -> str | None:
    """Enqueue on the 'agent' queue; returns arq job id or None if enqueue failed."""
    try:
        pool = await get_pool()
        job = await pool.enqueue_job(func, payload, _queue_name=AGENT_QUEUE)
        return job.job_id if job else None
    except Exception:
        from cadagent.logging import get_logger

        get_logger(__name__).warning("enqueue_failed", func=func)
        return None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as s:
        yield s
