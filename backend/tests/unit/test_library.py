"""Reusable part library: upload/CRUD + save-version-to-library."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from cadx.db.session import create_all, session_scope


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _project(client) -> str:
    r = await client.post("/api/projects", json={"name": "lib-proj"})
    return r.json()["id"]


async def test_library_upload_crud(client):
    # upload a STEP body into the library
    r = await client.post(
        "/api/library",
        files=[("files", ("spacer.step", b"ISO-10303-21; fake step", "model/step"))],
        params={"name": "spacer", "caption": "10mm spacer"},
    )
    assert r.status_code == 200, r.text
    part = r.json()[0]
    assert part["kind"] == "step" and part["name"] == "spacer"
    pid = part["id"]

    # list + get + file download
    lib = (await client.get("/api/library")).json()
    assert any(p["id"] == pid for p in lib)
    assert (await client.get(f"/api/library/{pid}")).json()["caption"] == "10mm spacer"
    r = await client.get(f"/api/library/{pid}/file")
    assert r.status_code == 200 and b"fake step" in r.content

    # patch + delete
    r = await client.patch(f"/api/library/{pid}", json={"caption": "updated"})
    assert r.json()["caption"] == "updated"
    r = await client.delete(f"/api/library/{pid}")
    assert r.status_code == 200
    assert (await client.get(f"/api/library/{pid}")).status_code == 404


async def test_library_upload_rejects_bad_type(client):
    r = await client.post(
        "/api/library",
        files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))],
    )
    assert r.status_code == 415


async def test_save_version_as_template_and_step(client):
    pid = await _project(client)

    # fabricate a finished version with code + a STEP artifact on disk
    from cadx import storage
    from cadx.db.models import Artifact, ArtifactKind, Version, VersionKind

    async with session_scope() as db:
        v = Version(
            project_id=uuid.UUID(pid),
            number=1,
            kind=VersionKind.generate,
            code="PARAMS = {}\nresult = None\n",
            params={"length": 40.0},
        )
        db.add(v)
        await db.flush()
        step_path = storage.version_dir(pid, v.id) / "exports" / "model.step"
        step_path.parent.mkdir(parents=True, exist_ok=True)
        step_path.write_bytes(b"ISO-10303-21; fake")
        db.add(
            Artifact(
                version_id=v.id,
                kind=ArtifactKind.step,
                path=storage.relpath(step_path),
                size=step_path.stat().st_size,
            )
        )
        await db.commit()
        vid = v.id

    # template: keeps the parametric code
    r = await client.post(
        f"/api/library/from-version/{vid}",
        json={"name": "plate tpl", "caption": "", "as_template": True},
    )
    assert r.status_code == 200, r.text
    tpl = r.json()
    assert tpl["kind"] == "template" and tpl["params"]["length"] == 40.0

    # step: copies the exported file
    r = await client.post(
        f"/api/library/from-version/{vid}",
        json={"name": "plate body", "caption": "", "as_template": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "step"
    r = await client.get(body["url"])
    assert r.status_code == 200 and b"fake" in r.content


async def test_generate_accepts_library_part_ids(client):
    pid = await _project(client)
    r = await client.post(
        "/api/library",
        files=[("files", ("insert.step", b"ISO-10303-21; fake", "model/step"))],
    )
    part_id = r.json()[0]["id"]
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "holder", "library_part_ids": [part_id]},
    )
    assert r.status_code == 200, r.text
    v = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    assert v["id"]  # version created; ids persisted (checked via DB below)

    from sqlmodel import select

    from cadx.db.models import Version

    async with session_scope() as db:
        row = (
            await db.execute(select(Version).where(Version.id == uuid.UUID(v["id"])))
        ).scalar_one()
        assert row.library_part_ids == [part_id]
