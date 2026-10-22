"""Global job lifecycle stream (GET /jobs/events).

Two layers under test:
  - the Job model listener in `db/models.py` — ONE chokepoint that publishes
    every create/status transition onto `jobs:lifecycle` (spawn, worker,
    cancel, MCP — current and future call sites);
  - `sse.jobs_lifecycle_stream` — snapshot of active jobs, then per-transition
    `jobs` events, with a keepalive DB re-check that repairs lost messages.
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
        self.unsubscribed = False
        self.queued: list[str] = []  # raw JSON payloads to deliver

    async def subscribe(self, channel: str) -> None:
        self.channel = channel

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


def _patch_subscribe_jobs(monkeypatch: pytest.MonkeyPatch, ps: _FakePubSub) -> _FakeClient:
    client = _FakeClient(ps)
    monkeypatch.setattr("cadx.events.subscribe_jobs", lambda: (client, ps))
    return client


async def _mk_job(status) -> str:
    from cadx.db.models import Job, JobType, Project, Version, VersionKind

    async with session_scope() as db:
        p = Project(name="lifecycle-proj")
        db.add(p)
        await db.flush()
        v = Version(project_id=p.id, number=1, kind=VersionKind.generate)
        db.add(v)
        await db.flush()
        job = Job(version_id=v.id, type=JobType.generate, status=status)
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


def _payload(ev: str) -> dict:
    assert ev.startswith("event: ")
    return json.loads(ev.split("data: ", 1)[1])


async def test_job_insert_and_status_change_publish(monkeypatch):
    """The model listener is the only publish site: inserting a row emits
    `queued`, a status assignment emits the transition, and touching any
    other column emits nothing."""
    from cadx.db.models import Job, JobStatus

    published: list[dict] = []
    monkeypatch.setattr(
        "cadx.events.publish_lifecycle_sync",
        lambda job_id, version_id, status, job_type: published.append(
            {"job_id": job_id, "status": status, "job_type": job_type}
        ),
    )

    jid = await _mk_job(JobStatus.queued)
    assert [p["status"] for p in published] == ["queued"]
    assert published[0]["job_id"] == jid

    await _set_status(jid, JobStatus.running)
    assert [p["status"] for p in published] == ["queued", "running"]

    # progress/log writes are per-job stream material, not lifecycle
    async with session_scope() as db:
        j = await db.get(Job, uuid.UUID(jid))
        assert j is not None
        j.log = "noise"
        await db.commit()
    assert [p["status"] for p in published] == ["queued", "running"]


async def test_jobs_stream_snapshot_then_lifecycle(monkeypatch):
    """Subscribe-first, snapshot second: the first frame is the active list,
    a queued lifecycle message re-emits the refreshed list, and a terminal
    transition lost on the wire is repaired by the keepalive re-check."""
    from cadx.api.routers import sse
    from cadx.db.models import JobStatus

    jid = await _mk_job(JobStatus.running)
    ps = _FakePubSub()
    client = _patch_subscribe_jobs(monkeypatch, ps)

    async with session_scope() as db:
        gen = sse.jobs_lifecycle_stream(db)
        try:
            # 1) snapshot — the running job is in the active list
            snap = _payload(await gen.__anext__())
            ids = [str(i["id"]) for i in snap["items"]]
            assert jid in ids

            # 2) a lifecycle message lands → 'lifecycle' + refreshed 'jobs'
            ps.queued.append(
                json.dumps(
                    {
                        "type": "lifecycle",
                        "job_id": jid,
                        "version_id": str(uuid.uuid4()),
                        "status": "running",
                        "job_type": "generate",
                    }
                )
            )
            life = _payload(await gen.__anext__())
            assert life["type"] == "lifecycle"
            assert life["job_id"] == jid
            _payload(await gen.__anext__())  # refreshed jobs list

            # 3) terminal transition with the publish LOST — the keepalive
            #    DB re-check detects the drift and re-emits the empty list
            await _set_status(jid, JobStatus.succeeded)
            assert (await gen.__anext__()) == ": keepalive\n\n"
            healed = _payload(await gen.__anext__())
            assert jid not in [str(i["id"]) for i in healed["items"]]
        finally:
            await gen.aclose()
    assert ps.unsubscribed
    assert client.closed


async def test_jobs_events_route_registered_before_job_id():
    """`/jobs/events` must win over `/jobs/{job_id}` — 'events' is not a
    UUID, so a reversed registration 404s the whole stream."""
    from cadx.api.routers import jobs

    paths = [r.path for r in jobs.router.routes]
    assert paths.index("/jobs/events") < paths.index("/jobs/{job_id}")
