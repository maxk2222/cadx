"""SSE job stream (cadx.api.routers.sse): the subscribe-gap fix.

Redis pub/sub has no backlog — a terminal event published between the
snapshot read and the subscribe used to be lost, pinning the EventSource
open on keepalives forever. The fixed stream subscribes FIRST and re-polls
the Job row on every keepalive tick.
"""

from __future__ import annotations

import json
import uuid

import pytest

from cadx.db.session import create_all, session_scope


@pytest.fixture(scope="module", autouse=True)
async def _tables():
    await create_all()


class _FakePubSub:
    """redis.asyncio PubSub stand-in — no broker in tests."""

    def __init__(self) -> None:
        self.channels: list[str] = []
        self.unsubscribed = False
        self.queued: list[str] = []  # raw JSON payloads to deliver

    async def subscribe(self, channel: str) -> None:
        self.channels.append(channel)

    async def unsubscribe(self, channel: str) -> None:
        self.unsubscribed = True

    async def get_message(
        self,
        ignore_subscribe_messages: bool = True,
        timeout=None,  # noqa: ASYNC109 — mirrors redis PubSub
    ):
        if self.queued:
            return {"type": "message", "data": self.queued.pop(0)}
        return None  # keepalive tick, immediately


class _FakeClient:
    def __init__(self, ps: _FakePubSub) -> None:
        self._ps = ps
        self.closed = False

    def pubsub(self) -> _FakePubSub:
        return self._ps

    async def aclose(self) -> None:
        self.closed = True


def _patch_subscribe(monkeypatch: pytest.MonkeyPatch, ps: _FakePubSub) -> _FakeClient:
    client = _FakeClient(ps)
    monkeypatch.setattr("cadx.events.subscribe", lambda job_id: (client, ps))
    return client


async def _mk_running_job() -> str:
    from cadx.db.models import Job, JobStatus, JobType, Project, Version, VersionKind

    async with session_scope() as db:
        p = Project(name="sse-proj")
        db.add(p)
        await db.flush()
        v = Version(project_id=p.id, number=1, kind=VersionKind.generate)
        db.add(v)
        await db.flush()
        job = Job(version_id=v.id, type=JobType.generate, status=JobStatus.running)
        db.add(job)
        await db.commit()
        return str(job.id)


async def _set_status(jid: str, status) -> None:
    from cadx.db.models import Job

    async with session_scope() as db:
        j = await db.get(Job, uuid.UUID(jid))
        assert j is not None
        j.status = status
        await db.commit()


async def test_sse_terminal_during_gap_closes_stream(monkeypatch):
    """Regression: the terminal publish lands in the old snapshot→subscribe
    gap (simulated by flipping the row inside subscribe()). With subscribe
    FIRST the snapshot must already see the final state and close."""
    from cadx.api.routers import sse
    from cadx.db.models import JobStatus

    jid = await _mk_running_job()

    class GapPubSub(_FakePubSub):
        async def subscribe(self, channel: str) -> None:
            await super().subscribe(channel)
            # worker's terminal commit lands exactly where the lost-publish
            # window used to be — the post-subscribe snapshot must see it
            await _set_status(jid, JobStatus.succeeded)

    ps = GapPubSub()
    _patch_subscribe(monkeypatch, ps)

    async with session_scope() as db:
        gen = await sse._stream(jid, db)
        ev = await gen.__anext__()
        assert ev.startswith("event: status")
        assert json.loads(ev.split("data: ", 1)[1])["status"] == "succeeded"
        assert (await gen.__anext__()).startswith("event: end")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
    assert ps.unsubscribed


async def test_sse_keepalive_repoll_detects_terminal(monkeypatch):
    """Job goes terminal while subscribed but the pub/sub 'done' is dropped —
    the keepalive DB re-check still emits status+end and closes."""
    from cadx.api.routers import sse
    from cadx.db.models import JobStatus

    jid = await _mk_running_job()
    ps = _FakePubSub()
    _patch_subscribe(monkeypatch, ps)

    async with session_scope() as db:
        gen = await sse._stream(jid, db)
        ev = await gen.__anext__()  # snapshot
        assert '"status": "running"' in ev
        assert (await gen.__anext__()) == ": keepalive\n\n"  # repoll: still running

        await _set_status(jid, JobStatus.failed)  # no pub/sub event — lost on the wire

        # the repoll fires as soon as the generator resumes past the
        # keepalive yield — the next frame is already the terminal status
        ev = await gen.__anext__()
        assert '"status": "failed"' in ev
        assert (await gen.__anext__()).startswith("event: end")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
    assert ps.unsubscribed


async def test_sse_pubsub_done_still_closes(monkeypatch):
    """Happy path unchanged: a live 'done' event ends the stream."""
    from cadx.api.routers import sse

    jid = await _mk_running_job()
    ps = _FakePubSub()
    ps.queued.append(json.dumps({"type": "done", "status": "succeeded"}))
    _patch_subscribe(monkeypatch, ps)

    async with session_scope() as db:
        gen = await sse._stream(jid, db)
        ev = await gen.__anext__()
        assert '"status": "running"' in ev
        ev = await gen.__anext__()
        assert ev.startswith("event: done")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
    assert ps.unsubscribed


async def test_sse_bad_job_id_404():
    """Non-UUID job ids 404 cleanly instead of raising ValueError -> 500."""
    from httpx import ASGITransport, AsyncClient

    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/jobs/not-a-uuid/events")
        assert r.status_code == 404
