"""System update API: proxy to the host updater daemon + active-jobs gate."""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

import cadx.api.routers.system as system
from cadx.db.models import Job, JobStatus, JobType, Project, Version, VersionKind
from cadx.db.session import async_session, create_all


@pytest.fixture(scope="module", autouse=True)
async def _db():
    await create_all()


@pytest.fixture()
async def client():
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _mk_job(status: JobStatus = JobStatus.running) -> Job:
    async with async_session() as db:
        p = Project(name=f"upd-{uuid.uuid4().hex[:6]}")
        db.add(p)
        await db.flush()
        v = Version(project_id=p.id, part_id=uuid.uuid4(), number=1, kind=VersionKind.generate, prompt="x")
        db.add(v)
        await db.flush()
        j = Job(version_id=v.id, type=JobType.generate, status=status)
        db.add(j)
        await db.commit()
        await db.refresh(j)
        return j


def _git_info() -> dict:
    return {"sha": "abc1234", "branch": "dev", "dirty": 0, "ts": "2026-10-21T00:00:00Z"}


async def test_version_online(client, monkeypatch):
    async def fake_call(method, path, *, body=None, timeout_s=15.0):
        assert path == "/v1/version"
        return _git_info()

    monkeypatch.setattr(system, "_call", fake_call)
    r = await client.get("/api/system/version")
    assert r.status_code == 200
    d = r.json()
    assert d["updater_ok"] is True
    assert d["git"]["sha"] == "abc1234"
    assert d["version"]


async def test_version_degrades_when_daemon_down(client, monkeypatch):
    async def down(method, path, *, body=None, timeout_s=15.0):
        raise HTTPException(503, system._OFFLINE)

    monkeypatch.setattr(system, "_call", down)
    r = await client.get("/api/system/version")
    assert r.status_code == 200
    d = r.json()
    assert d["updater_ok"] is False and d["git"] is None


async def test_check_proxy(client, monkeypatch):
    async def fake_call(method, path, *, body=None, timeout_s=15.0):
        return {
            "behind": 3,
            "ahead": 0,
            "commits": [{"sha": "deadbee", "subject": "x", "date": "2026-10-20"}],
            "branches": ["dev", "main"],
            "current_ref": "dev",
            "target_ref": "dev",
            "current_sha": "abc1234",
            "update_available": True,
            "dirty": [],
            "checked_at": "2026-10-21T00:00:00Z",
        }

    monkeypatch.setattr(system, "_call", fake_call)
    r = await client.get("/api/system/update/check")
    assert r.status_code == 200
    assert r.json()["behind"] == 3
    assert r.json()["update_available"] is True


async def test_check_503(client, monkeypatch):
    async def down(method, path, *, body=None, timeout_s=15.0):
        raise HTTPException(503, system._OFFLINE)

    monkeypatch.setattr(system, "_call", down)
    r = await client.get("/api/system/update/check")
    assert r.status_code == 503
    assert "updater daemon offline" in r.json()["detail"]


async def test_update_no_jobs_passes_through(client, monkeypatch):
    called = []

    async def fake_call(method, path, *, body=None, timeout_s=15.0):
        called.append((method, path, body))
        return {"started": True}

    monkeypatch.setattr(system, "_call", fake_call)
    # the test DB is shared across the suite — quiesce any active job rows so
    # the gate sees zero regardless of module order
    async with async_session() as db:
        from sqlmodel import select

        rows = (await db.execute(select(Job))).scalars().all()
        for j in rows:
            if j.status in (JobStatus.queued, JobStatus.running, JobStatus.waiting_for_input):
                j.status = JobStatus.succeeded
        await db.commit()

    r = await client.post("/api/system/update", json={})
    assert r.status_code == 200
    assert called[-1][2] == {}


async def test_update_blocked_by_active_jobs(client, monkeypatch):
    await _mk_job(JobStatus.running)
    called = []

    async def fake_call(method, path, *, body=None, timeout_s=15.0):
        called.append((method, path, body))
        return {"started": True}

    monkeypatch.setattr(system, "_call", fake_call)
    r = await client.post("/api/system/update", json={})
    assert r.status_code == 409
    assert "active job" in r.json()["detail"]
    assert called == []

    # explicit confirm overrides; no force is forwarded so the script's
    # dirty-tree refusal still applies to GUI-triggered updates
    r = await client.post("/api/system/update", json={"confirm_jobs": True, "ref": "main"})
    assert r.status_code == 200
    assert called[-1][1] == "/v1/update"
    assert called[-1][2] == {"ref": "main"}


async def test_status_proxy(client, monkeypatch):
    async def fake_call(method, path, *, body=None, timeout_s=15.0):
        return {
            "state": "running",
            "phase": "deps",
            "log_tail": ["== CADX update", "* deps"],
            "version": _git_info(),
        }

    monkeypatch.setattr(system, "_call", fake_call)
    r = await client.get("/api/system/update/status")
    assert r.status_code == 200
    assert r.json()["state"] == "running"
    assert r.json()["log_tail"][0] == "== CADX update"


async def test_rollback_requires_confirm(client, monkeypatch):
    called = []

    async def fake_call(method, path, *, body=None, timeout_s=15.0):
        called.append((method, path, body))
        return {"started": True}

    monkeypatch.setattr(system, "_call", fake_call)
    r = await client.post("/api/system/update/rollback", json={})
    assert r.status_code == 400
    r = await client.post("/api/system/update/rollback", json={"confirm": True})
    assert r.status_code == 200
    assert called[-1] == ("POST", "/v1/update/rollback", {"confirm": True})
