"""Refine plumbing: POST /versions/{id}/refine guards + auto-refine hook.

D1: `_maybe_spawn_refine` creates a refine child version + job and enqueues
when `auto_refine_features` is set — and stays silent otherwise.
D5: API contract — 404/409 paths, extra=forbid on RefineRequest, and the
OptimizeSpec fields added in TopOpt v2 parse through the request model.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from cadx.db.models import Job, JobType, Project, Version, VersionKind, VersionStatus
from cadx.db.session import async_session, create_all
from cadx.llm.schemas import OptimizeSpec, PartSpec


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _mk_parent(status: VersionStatus = VersionStatus.succeeded) -> Version:
    async with async_session() as db:
        p = Project(name="refine-test")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=status,
        )
        db.add(v)
        await db.commit()
        await db.refresh(v)
        return v


async def test_refine_endpoint_creates_child(client):
    parent = await _mk_parent()
    r = await client.post(f"/api/versions/{parent.id}/refine", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    async with async_session() as db:
        child = await db.get(Version, uuid.UUID(body["version_id"]))
        job = await db.get(Job, uuid.UUID(body["job_id"]))
        assert child is not None and child.kind == VersionKind.refine
        assert child.parent_id == parent.id
        assert job is not None and job.type == JobType.refine


async def test_refine_endpoint_guards(client):
    parent = await _mk_parent()
    # refine a refine child → 409
    r = await client.post(f"/api/versions/{parent.id}/refine", json={})
    child_id = r.json()["version_id"]
    async with async_session() as db:
        child = await db.get(Version, uuid.UUID(child_id))
        child.status = VersionStatus.succeeded
        db.add(child)
        await db.commit()
    r2 = await client.post(f"/api/versions/{child_id}/refine", json={})
    assert r2.status_code == 409
    # missing version → 404
    r3 = await client.post(f"/api/versions/{uuid.uuid4()}/refine", json={})
    assert r3.status_code == 404
    # running/pending parent → 409
    busy = await _mk_parent(status=VersionStatus.running)
    r4 = await client.post(f"/api/versions/{busy.id}/refine", json={})
    assert r4.status_code == 409


async def test_refine_request_forbids_extra(client):
    parent = await _mk_parent()
    r = await client.post(
        f"/api/versions/{parent.id}/refine",
        json={"features": [], "bogus_field": True},
    )
    assert r.status_code == 422


async def test_maybe_spawn_refine_hook(monkeypatch):
    from cadx.agent import worker_jobs as wj

    enqueued: list[dict] = []

    async def fake_enqueue(func, payload):
        enqueued.append({"func": func, "payload": payload})
        return "fake-job-id"

    import cadx.api.deps as deps

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)

    parent = await _mk_parent()
    # the guard needs something to re-cut: a declared spec feature
    spec = PartSpec(
        name="block",
        length_x={"value": 40.0},
        width_y={"value": 20.0},
        height_z={"value": 10.0},
        features=[
            {
                "name": "m4 hole",
                "type": "hole",
                "description": "M4 through-hole",
                "dims": {"diameter": 4.0},
                "position": "center",
                "count": 1,
                "source": "stated",
            }
        ],
    )
    async with async_session() as db:
        parent = await db.get(Version, parent.id)
        ok = await wj._maybe_spawn_refine(
            db,
            parent,
            spec=spec,
            optimize={"auto_refine_features": True, "keep_regions": []},
            load_cases=[],
            material="pla",
        )
        assert ok is True
    assert enqueued and enqueued[0]["func"] == "run_refine_job"
    async with async_session() as db:
        child_id = enqueued[0]["payload"]["version_id"]
        child = await db.get(Version, uuid.UUID(child_id))
        assert child.kind == VersionKind.refine
        assert child.parent_id == parent.id
        assert (child.options or {}).get("auto_refined") is True


async def test_maybe_spawn_refine_noop(monkeypatch):
    """Without the flag or on a failed parent no child is created."""
    from cadx.agent import worker_jobs as wj

    enqueued: list = []

    async def fake_enqueue(func, payload):
        enqueued.append(payload)
        return

    import cadx.api.deps as deps

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)

    parent = await _mk_parent()
    async with async_session() as db:
        parent = await db.get(Version, parent.id)
        ok = await wj._maybe_spawn_refine(
            db, parent, spec=None, optimize={}, load_cases=[], material=""
        )
        assert ok is False
    assert not enqueued

    failed = await _mk_parent(status=VersionStatus.failed)
    async with async_session() as db:
        failed = await db.get(Version, failed.id)
        ok = await wj._maybe_spawn_refine(
            db,
            failed,
            spec=None,
            optimize={"auto_refine_features": True},
            load_cases=[],
            material="",
        )
        assert ok is False
    assert not enqueued


def test_optimize_spec_new_fields_parse():
    """TopOpt v2 fields accepted by OptimizeSpec (API dict → model)."""
    spec = OptimizeSpec(
        enabled=True,
        auto_refine_features=True,
        restore_planes=False,
        self_supporting=True,
        print_direction="-y",
        filter="heaviside",
        penal=3.5,
        multires=True,
    )
    assert spec.auto_refine_features and spec.self_supporting
    assert spec.print_direction == "-y" and spec.filter == "heaviside"
    assert spec.penal == 3.5 and spec.multires and spec.restore_planes is False
