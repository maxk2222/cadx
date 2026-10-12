"""Server-sent events: live job progress + log stream."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from cadx import events
from cadx.api.deps import get_session
from cadx.db.models import Job, JobStatus

router = APIRouter(tags=["sse"])

_TERMINAL = {
    JobStatus.succeeded,
    JobStatus.failed,
    JobStatus.cancelled,
    JobStatus.waiting_for_input,
}


async def _stream(job_id: str, db: AsyncSession):
    """Initial job snapshot, then live events; closes on terminal state."""

    async def gen():
        j = await db.get(Job, uuid.UUID(job_id))
        if j:
            yield _fmt("status", {"status": j.status.value, "step": j.step, "log": j.log})
            if j.status in _TERMINAL:
                yield _fmt("end", {})
                return
        client, ps = events.subscribe(job_id)
        try:
            await ps.subscribe(events.channel(job_id))
            while True:
                msg = await ps.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if msg and msg.get("data"):
                    try:
                        data = json.loads(msg["data"])
                    except Exception:
                        continue
                    yield _fmt(data.get("type", "event"), data)
                    if data.get("type") in ("done", "question"):
                        break
                else:
                    yield ": keepalive\n\n"
        finally:
            await ps.unsubscribe(events.channel(job_id))
            await client.aclose()

    return gen()


def _fmt(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, db: AsyncSession = Depends(get_session)):
    gen = await _stream(job_id, db)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
