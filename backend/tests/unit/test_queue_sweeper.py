"""Queue sweeper: deterministic arq keys + redis-loss replay.

The contract being tested: spawn enqueues every job under `arq:job:<job.id>`
(deps.enqueue derives the dedup id from payload["job_id"]), so the sweeper's
exists-check can tell "still queued" from "lost". A WRONG key scheme is the
double-execution bug this file guards against.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import cadx.api.deps as deps
from cadx import queue as q
from cadx.api.deps import enqueue as real_enqueue  # bound at import — survives conftest's stub
from cadx.db.models import (
    Chat,
    ChatMessage,
    ChatRole,
    Job,
    JobStatus,
    JobType,
    Project,
    Version,
    VersionKind,
)
from cadx.db.session import create_all, session_scope


@pytest.fixture(scope="module", autouse=True)
async def _tables():
    await create_all()


class FakeRedis:
    """Minimal in-memory stand-in for the keys the sweeper touches."""

    def __init__(self, keys: set[str] | None = None):
        self.keys = set(keys or ())
        self.deleted: list[str] = []

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self.keys)

    async def delete(self, *keys: str) -> int:
        self.deleted.extend(keys)
        n = 0
        for k in keys:
            if k in self.keys:
                self.keys.discard(k)
                n += 1
        return n

    async def scan(self, cursor: int, match: str, count: int = 200):
        prefix = match.rstrip("*")
        return 0, [k for k in self.keys if k.startswith(prefix)]


def _rec_enqueue(calls: list):
    async def _fn(func, payload, dedup_id=None):
        calls.append((func, payload, dedup_id))
        return "ok"

    return _fn


async def _mk_job(*, age_s: float = 300, payload: dict | None = ...) -> Job:
    async with session_scope() as db:
        p = Project(name=f"sweep-{uuid.uuid4().hex[:6]}")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id, part_id=uuid.uuid4(), number=1, kind=VersionKind.generate, prompt="x"
        )
        db.add(v)
        await db.flush()
        args = {"job_id": "", "version_id": str(v.id)}
        j = Job(
            version_id=v.id,
            type=JobType.generate,
            status=JobStatus.queued,
            created_at=datetime.now(UTC) - timedelta(seconds=age_s),
        )
        db.add(j)
        await db.flush()
        args["job_id"] = str(j.id)
        if payload is not ...:
            j.payload = payload
        else:
            j.payload = {"func": "run_agent_job", "args": args}
        await db.commit()
        await db.refresh(j)
        return j


async def test_sweeper_skips_live_arq_entry(monkeypatch):
    """The core regression: a job still queued in arq must NOT be re-enqueued.
    Requires the deterministic-key contract — the sweeper checks
    `arq:job:<job.id>`."""
    j = await _mk_job()
    calls: list[tuple[str, dict, str | None]] = []

    async def fake_enqueue(func, payload, dedup_id=None):
        calls.append((func, payload, dedup_id))
        return "ok"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)
    ctx = {"arq_redis": FakeRedis({f"arq:job:{j.id}"})}
    await q._sweep_once(ctx)
    assert not [c for c in calls if c[1].get("job_id") == str(j.id)]  # live → no dup


async def test_sweeper_reenqueues_lost_job(monkeypatch):
    """Redis loss → no arq entry → replay the payload envelope under the
    same dedup id (idempotent)."""
    j = await _mk_job()
    calls: list[tuple[str, dict, str | None]] = []

    async def fake_enqueue(func, payload, dedup_id=None):
        calls.append((func, payload, dedup_id))
        return "ok"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)
    ctx = {"arq_redis": FakeRedis()}
    await q._sweep_once(ctx)
    # module-scoped db: earlier tests leave swept rows behind — assert on ours
    mine = [c for c in calls if c[1].get("job_id") == str(j.id)]
    assert len(mine) == 1
    func, _payload, dedup_id = mine[0]
    assert func == "run_agent_job"
    assert dedup_id == str(j.id)  # replays under the same deterministic key

    async with session_scope() as db:
        j2 = await db.get(Job, j.id)
        assert "re-enqueued by queue sweeper" in (j2.log or "")
        assert j2.status == JobStatus.queued


async def test_sweeper_fails_payload_less_job(monkeypatch):
    """Pre-migration rows have no envelope — fail them honestly instead of
    guessing a func from the ambiguous JobType."""
    j = await _mk_job(payload=None)
    monkeypatch.setattr(deps, "enqueue", _rec_enqueue([]))
    ctx = {"arq_redis": FakeRedis()}
    await q._sweep_once(ctx)
    async with session_scope() as db:
        j2 = await db.get(Job, j.id)
        assert j2.status == JobStatus.failed
        assert "predates payload capture" in (j2.log or "")


async def test_sweeper_respects_grace_period(monkeypatch):
    """Jobs fresher than SWEEP_GRACE_S are left alone — the spawn may still
    be mid-enqueue."""
    j = await _mk_job(age_s=5)
    calls: list = []
    monkeypatch.setattr(deps, "enqueue", _rec_enqueue(calls))
    ctx = {"arq_redis": FakeRedis()}
    await q._sweep_once(ctx)
    assert str(j.id) not in [c[2] for c in calls]


async def test_enqueue_derives_dedup_id_from_payload(monkeypatch):
    """payload['job_id'] becomes the arq key when dedup_id isn't given —
    this is what makes the sweeper's exists-check able to find the entry."""
    seen: dict[str, Any] = {}

    class Pool:
        async def enqueue_job(self, func, payload, _job_id=None, _queue_name=None):
            seen["job_id"] = _job_id
            return type("J", (), {"job_id": _job_id})()

    monkeypatch.setattr(deps, "_pool", Pool())
    jid = str(uuid.uuid4())
    out = await real_enqueue("run_agent_job", {"job_id": jid, "version_id": "v"})
    assert out == jid
    assert seen["job_id"] == jid
    # explicit dedup_id still wins
    out2 = await real_enqueue("run_chat_turn", {"chat_id": "c"}, dedup_id="chat:c:m")
    assert out2 == "chat:c:m"
    assert seen["job_id"] == "chat:c:m"


async def test_sweeper_chat_pending_lost_turn(monkeypatch):
    """Pending message with no live arq key → re-enqueue under the per-msg
    key; a stale result key is deleted first so arq dedup can't suppress."""
    async with session_scope() as db:
        p = Project(name=f"chat-sweep-{uuid.uuid4().hex[:6]}")
        db.add(p)
        await db.flush()
        c = Chat(project_id=p.id, title="t")
        db.add(c)
        await db.flush()
        m = ChatMessage(
            chat_id=c.id,
            role=ChatRole.user,
            content="hi",
            pending=True,
            created_at=datetime.now(UTC) - timedelta(seconds=300),
        )
        db.add(m)
        await db.commit()
        key = deps.chat_turn_dedup_key(c.id, m.id)

    calls: list[str | None] = []

    async def fake_enqueue(func, payload, dedup_id=None):
        calls.append(dedup_id)
        return "ok"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)
    redis = FakeRedis({f"arq:result:{key}"})  # stale claim-lost result
    await q._sweep_once({"arq_redis": redis})
    assert key in calls
    assert f"arq:result:{key}" in redis.deleted  # stale result purged


async def test_sweeper_chat_pending_live_turn_skipped(monkeypatch):
    """Pending message whose turn is queued/running must not double-fire."""
    async with session_scope() as db:
        p = Project(name=f"chat-sweep-{uuid.uuid4().hex[:6]}")
        db.add(p)
        await db.flush()
        c = Chat(project_id=p.id, title="t")
        db.add(c)
        await db.flush()
        m = ChatMessage(
            chat_id=c.id,
            role=ChatRole.user,
            content="hi",
            pending=True,
            created_at=datetime.now(UTC) - timedelta(seconds=300),
        )
        db.add(m)
        await db.commit()
        key = deps.chat_turn_dedup_key(c.id, m.id)

    calls: list = []
    monkeypatch.setattr(deps, "enqueue", _rec_enqueue(calls))
    await q._sweep_once({"arq_redis": FakeRedis({f"arq:job:{key}"})})
    assert key not in [c[2] for c in calls]


async def test_sweeper_turn_flag_only_cleared_when_dead(monkeypatch):
    """turn_running survives while a live arq key exists for the chat — a
    slow single LLM call (>grace between updated_at writes) must NOT look
    orphaned, or a second turn could claim concurrently."""
    async with session_scope() as db:
        p = Project(name=f"chat-sweep-{uuid.uuid4().hex[:6]}")
        db.add(p)
        await db.flush()
        c = Chat(project_id=p.id, title="t", turn_running=True)
        c.updated_at = datetime.now(UTC) - timedelta(seconds=300)
        db.add(c)
        await db.commit()
        cid = c.id

    # live turn key present → flag survives
    await q._sweep_once({"arq_redis": FakeRedis({f"arq:in-progress:chat:{cid}:m1"})})
    async with session_scope() as db:
        assert (await db.get(Chat, cid)).turn_running is True

    # no live key → orphan → flag cleared
    await q._sweep_once({"arq_redis": FakeRedis()})
    async with session_scope() as db:
        assert (await db.get(Chat, cid)).turn_running is False
