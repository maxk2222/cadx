"""Shared API dependencies: DB session + arq enqueue helper."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import arq
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.config import get_settings
from cadx.db.session import async_session

AGENT_QUEUE = "agent"


def chat_turn_dedup_key(chat_id: Any, msg_id: Any) -> str:
    """Deterministic arq key for a chat turn — `chat:<chat>:<msg>`.

    Shared by the API (POST/run enqueue), the worker startup reaper and the
    queue sweeper: they all agree on one key so a live turn provably leaves
    `arq:job|in-progress:chat:<id>:*` — a random key would defeat both the
    sweeper's exists-check and its turn-alive scan."""
    return f"chat:{chat_id}:{msg_id}"


_pool: ArqRedis | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> ArqRedis:
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:  # concurrent first-callers must not race two pools
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

    The arq key is DETERMINISTIC: `dedup_id` wins, else `payload["job_id"]`
    (every job payload carries it). Random keys would be invisible to the
    queue sweeper — a job waiting >grace looks "lost" and gets re-enqueued
    into a duplicate run. Deterministic keys also make arq's own dedup
    (`arq:job:<id>` + recent `arq:result:<id>`) a real idempotency guard;
    a duplicate is treated as *the entry exists* — success — and the
    dedup_id is returned. Payloads with no job_id (e.g. chat turns keyed
    per message) still fall back to a random arq id when dedup_id is None.
    """
    if dedup_id is None:
        jid = payload.get("job_id")
        if isinstance(jid, str) and jid:
            dedup_id = jid
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
