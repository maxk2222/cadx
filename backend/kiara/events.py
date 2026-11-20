"""Job progress events via Redis pub/sub (channel `job:<job_id>`)."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from kiara.config import get_settings


def channel(job_id: str) -> str:
    return f"job:{job_id}"


def chat_channel(chat_id: str) -> str:
    """Separate non-terminating stream for the conversational agent."""
    return f"chat:{chat_id}"


async def _publish(chan: str, event: dict[str, Any]) -> None:
    r = aioredis.from_url(get_settings().redis_url)
    try:
        await r.publish(chan, json.dumps(event, default=str))
    finally:
        await r.aclose()


async def publish(job_id: str, event: dict[str, Any]) -> None:
    await _publish(channel(job_id), event)


async def publish_chat(chat_id: str, event: dict[str, Any]) -> None:
    await _publish(chat_channel(chat_id), event)


def _subscribe() -> tuple[aioredis.Redis, aioredis.client.PubSub]:
    """Caller owns the client and must close it — the channel subscribe is
    the caller's job (same shape for every channel)."""
    r = aioredis.from_url(get_settings().redis_url)
    return r, r.pubsub()


def subscribe(job_id: str) -> tuple[aioredis.Redis, aioredis.client.PubSub]:
    return _subscribe()


def subscribe_chat(chat_id: str) -> tuple[aioredis.Redis, aioredis.client.PubSub]:
    return _subscribe()


def subscribe_jobs() -> tuple[aioredis.Redis, aioredis.client.PubSub]:
    return _subscribe()


# --------------------------------------------------------------------------- #
# global job lifecycle channel — one event per Job row create/status change   #
# --------------------------------------------------------------------------- #

JOBS_CHANNEL = "jobs:lifecycle"

_SYNC_CLIENT: Any = None


def _sync_client() -> Any:
    """Lazy sync redis client for publishes from sync contexts (e.g. the
    SQLAlchemy flush listener — no running loop there to await on)."""
    global _SYNC_CLIENT
    if _SYNC_CLIENT is None:
        import redis

        _SYNC_CLIENT = redis.Redis.from_url(
            get_settings().redis_url, socket_timeout=1, socket_connect_timeout=1
        )
    return _SYNC_CLIENT


def _lifecycle_event(job_id: str, version_id: str, status: str, job_type: str) -> str:
    return json.dumps(
        {
            "type": "lifecycle",
            "job_id": job_id,
            "version_id": version_id,
            "status": status,
            "job_type": job_type,
        },
        default=str,
    )


def publish_lifecycle_sync(job_id: str, version_id: str, status: str, job_type: str) -> None:
    """Fire-and-forget publish of a job create/status transition. Safe to call
    from inside a session flush: ~0.3ms sync publish, every failure swallowed —
    the SSE stream's keepalive drift-check repairs a lost event anyway."""
    import contextlib

    with contextlib.suppress(Exception):
        _sync_client().publish(JOBS_CHANNEL, _lifecycle_event(job_id, version_id, status, job_type))


async def publish_lifecycle(job_id: str, version_id: str, status: str, job_type: str) -> None:
    """Async variant for call sites that already run inside a loop."""
    await _publish(JOBS_CHANNEL, _lifecycle_event(job_id, version_id, status, job_type))
