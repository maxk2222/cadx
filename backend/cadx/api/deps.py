"""Shared API dependencies: DB session + arq enqueue helper."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import arq
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.config import get_settings
from cadx.db.session import async_session

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


async def enqueue(func: str, payload: dict[str, Any], dedup_id: str | None = None) -> str | None:
    """Enqueue on the 'agent' queue; returns the arq job id or None on failure.

    With `dedup_id`, arq enforces job uniqueness (`arq:job:<id>` + recent
    `arq:result:<id>`); a duplicate is treated as *the entry exists* — success
    — and the dedup_id is returned. Without it, arq's None return still maps
    to failure (fresh random ids never legitimately collide)."""
    try:
        pool = await get_pool()
        job = await pool.enqueue_job(func, payload, _job_id=dedup_id, _queue_name=AGENT_QUEUE)
        if job is None:
            return dedup_id
        return job.job_id
    except Exception:
        from cadx.logging import get_logger

        get_logger(__name__).warning("enqueue_failed", func=func)
        return None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as s:
        yield s


def payload_envelope(func: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """{func,args} replay envelope for `Job.payload` — the queue sweeper
    replays it verbatim after queue loss. arq serializes payloads with
    pickle, so args may contain objects JSON can't express; storing one of
    those would break the committing transaction. Returns None (→ job.payload
    stays NULL → sweeper fails honestly) when args aren't JSON-safe."""
    import json

    try:
        return json.loads(json.dumps({"func": func, "args": args}))
    except (TypeError, ValueError):
        return None
