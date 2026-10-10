"""Job progress events via Redis pub/sub (channel `job:<job_id>`)."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from cadagent.config import get_settings


def channel(job_id: str) -> str:
    return f"job:{job_id}"


async def publish(job_id: str, event: dict[str, Any]) -> None:
    r = aioredis.from_url(get_settings().redis_url)
    try:
        await r.publish(channel(job_id), json.dumps(event, default=str))
    finally:
        await r.aclose()


def subscribe(job_id: str) -> tuple[aioredis.Redis, aioredis.client.PubSub]:
    """Caller owns the client and must close it."""
    r = aioredis.from_url(get_settings().redis_url)
    return r, r.pubsub()
