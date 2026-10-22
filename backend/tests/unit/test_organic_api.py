"""Organic restyle plumbing: POST /versions/{id}/organic.

Child version kind='organic' + Job type='organic' + enqueues
'run_organic_job' with the request's organic params. Guards: 404 missing
parent, 409 on non-terminal parent, 422 on extra fields / bad params.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from cadx.db.models import Job, JobType, Project, Version, VersionKind, VersionStatus
from cadx.db.session import async_session, create_all


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _mk_parent(status: VersionStatus = VersionStatus.succeeded) -> Version:
    async with async_session() as db:
        p = Project(name="organic-test")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=status,
        )
        db.add(v)
        await db.commit()
        await db.refresh(v)
        return v


async def test_organic_endpoint_creates_child(client):
    parent = await _mk_parent()
    r = await client.post(
        f"/api/versions/{parent.id}/organic",
        json={"style": "voronoi", "seed": 7, "volfrac": 0.4},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    async with async_session() as db:
        child = await db.get(Version, uuid.UUID(body["version_id"]))
        job = await db.get(Job, uuid.UUID(body["job_id"]))
        assert child is not None and child.kind == VersionKind.organic
        assert child.parent_id == parent.id
        assert job is not None and job.type == JobType.organic
        req = (child.options or {}).get("organic_request") or {}
        assert req.get("style") == "voronoi" and req.get("seed") == 7
        # runtime-injected field must not leak into the stored request
        assert "anchors" not in req


async def test_organic_endpoint_guards(client):
    # missing version → 404
    r = await client.post(f"/api/versions/{uuid.uuid4()}/organic", json={})
    assert r.status_code == 404
    # non-terminal parent → 409
    busy = await _mk_parent(status=VersionStatus.running)
    r2 = await client.post(f"/api/versions/{busy.id}/organic", json={})
    assert r2.status_code == 409


async def test_organic_request_forbids_extra(client):
    parent = await _mk_parent()
    r = await client.post(
        f"/api/versions/{parent.id}/organic",
        json={"style": "gyroid", "bogus_field": True},
    )
    assert r.status_code == 422


async def test_organic_request_validates_params(client):
    parent = await _mk_parent()
    r = await client.post(
        f"/api/versions/{parent.id}/organic",
        json={"style": "lattice"},
    )
    assert r.status_code == 422
    r2 = await client.post(
        f"/api/versions/{parent.id}/organic",
        json={"volfrac": 3.0},
    )
    assert r2.status_code == 422


async def test_organic_enqueue_payload(client, monkeypatch):
    """The worker job receives the organic params verbatim."""
    enqueued: list[dict] = []

    async def fake_enqueue(func, payload):
        enqueued.append({"func": func, "payload": payload})
        return "fake"

    # spawn.py resolves deps.enqueue lazily at call time — patch the source
    import cadx.api.deps as deps

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)
    parent = await _mk_parent()
    r = await client.post(
        f"/api/versions/{parent.id}/organic",
        json={
            "style": "branches",
            "volfrac": 0.3,
            "seed": 99,
            "floor_mm": 4.0,
            "keep_regions": [{"kind": "all"}],
        },
    )
    assert r.status_code == 200, r.text
    assert enqueued and enqueued[0]["func"] == "run_organic_job"
    org = enqueued[0]["payload"]["organic"]
    assert org["style"] == "branches" and org["seed"] == 99
    assert org["floor_mm"] == 4.0
    assert [r["kind"] for r in org["keep_regions"]] == ["all"]


async def test_organic_surface_open_roundtrip(client):
    """surface_open is stored verbatim and bounds-checked."""
    parent = await _mk_parent()
    r = await client.post(
        f"/api/versions/{parent.id}/organic",
        json={"surface_open": 0.5},
    )
    assert r.status_code == 200, r.text
    async with async_session() as db:
        child = await db.get(Version, uuid.UUID(r.json()["version_id"]))
        req = (child.options or {}).get("organic_request") or {}
        assert req.get("surface_open") == 0.5
    r_bad = await client.post(
        f"/api/versions/{parent.id}/organic",
        json={"surface_open": 1.5},
    )
    assert r_bad.status_code == 422


async def test_organic_resample_guard_accepts_organic_density(client):
    """topopt-resample accepts organic versions carrying a density field."""
    parent = await _mk_parent()
    async with async_session() as db:
        v = Version(
            project_id=parent.project_id,
            part_id=uuid.uuid4(),
            parent_id=parent.id,
            number=2,
            kind=VersionKind.organic,
            prompt="organic block",
            status=VersionStatus.succeeded,
            mesh={"organic": {"density_relpath": "x/y/organic/density.npz"}},
        )
        db.add(v)
        await db.commit()
        await db.refresh(v)
        vid = v.id
    r = await client.post(f"/api/versions/{vid}/topopt-resample", json={"density_threshold": 0.6})
    assert r.status_code == 200, r.text
    async with async_session() as db:
        child = await db.get(Version, uuid.UUID(r.json()["version_id"]))
        assert child is not None and child.kind == VersionKind.organic
