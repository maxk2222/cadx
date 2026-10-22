"""Server-sent events: live job progress + log stream."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
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

# pub/sub poll interval — doubles as the keepalive + DB re-check cadence
_KEEPALIVE_S = 15.0


def _status_payload(j: Job) -> dict:
    return {
        "status": j.status.value,
        "step": j.step,
        "iteration": j.iteration,
        "progress": j.progress,
        "started_at": j.started_at,
        "live": j.live,
        "log": j.log,
    }


async def _stream(job_id: str, db: AsyncSession):
    """Job snapshot, then live events; closes on terminal state.

    Subscribes BEFORE reading the snapshot — Redis pub/sub has no backlog,
    so a terminal event published between the snapshot read and subscribe
    would be lost and the stream would hang on keepalives forever. The
    keepalive tick also re-polls the Job row (authoritative), so a dropped
    terminal event still closes the stream.
    """

    jid = uuid.UUID(job_id)

    async def gen():
        client, ps = events.subscribe(job_id)
        try:
            await ps.subscribe(events.channel(job_id))
            j = await db.get(Job, jid)
            if j:
                yield _fmt("status", _status_payload(j))
                if j.status in _TERMINAL:
                    yield _fmt("end", {})
                    return
            while True:
                msg = await ps.get_message(ignore_subscribe_messages=True, timeout=_KEEPALIVE_S)
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
                    # the row is authoritative — a terminal event lost on the
                    # wire must not pin this stream open on keepalives
                    j = await db.get(Job, jid, populate_existing=True)
                    if j is not None and j.status in _TERMINAL:
                        yield _fmt("status", _status_payload(j))
                        yield _fmt("end", {})
                        return
        finally:
            await ps.unsubscribe(events.channel(job_id))
            await client.aclose()

    return gen()


def _fmt(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, db: AsyncSession = Depends(get_session)):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(404, "job not found") from None
    gen = await _stream(job_id, db)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# global job lifecycle stream — drives the nav badge + status lists            #
#                                                                              #
# NOTE: the route lives in `jobs.py` (registered BEFORE `/jobs/{job_id}` so    #
# 'events' isn't swallowed as a job id); the generator stays here with the     #
# other SSE machinery.                                                         #
# --------------------------------------------------------------------------- #


def _jobs_payload(items: list) -> dict:
    return {"items": [it.model_dump() for it in items]}


def _active_sig(items: list) -> frozenset:
    """(id, status) pairs — drift vs. last emit means a lifecycle event was
    lost on the wire and the keepalive tick repairs it."""
    return frozenset((str(it.id), str(it.status)) for it in items)


async def jobs_lifecycle_stream(db: AsyncSession):
    """Snapshot of currently-active jobs, then one `jobs` event per lifecycle
    transition (create/status change — published by the Job model listener on
    the `jobs:lifecycle` redis channel). The keepalive tick re-checks the DB:
    a lost message re-emits the active list instead of pinning a stale view."""
    from cadx.api.routers.jobs import job_list_items

    client, ps = events.subscribe_jobs()
    try:
        await ps.subscribe(events.JOBS_CHANNEL)
        items = await job_list_items(db, active=True, limit=100)
        sig = _active_sig(items)
        yield _fmt("jobs", _jobs_payload(items))
        while True:
            msg = await ps.get_message(ignore_subscribe_messages=True, timeout=_KEEPALIVE_S)
            if msg and msg.get("data"):
                try:
                    data = json.loads(msg["data"])
                except Exception:
                    continue
                yield _fmt(data.get("type", "event"), data)
                items = await job_list_items(db, active=True, limit=100)
                sig = _active_sig(items)
                yield _fmt("jobs", _jobs_payload(items))
            else:
                yield ": keepalive\n\n"
                items = await job_list_items(db, active=True, limit=100)
                cur = _active_sig(items)
                if cur != sig:
                    sig = cur
                    yield _fmt("jobs", _jobs_payload(items))
    finally:
        await ps.unsubscribe(events.JOBS_CHANNEL)
        await client.aclose()
