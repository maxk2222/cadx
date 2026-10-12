"""API smoke tests against an isolated SQLite DB (no redis/LLM)."""

from __future__ import annotations

import io
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from cadx.db.session import create_all


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _png_bytes(size=(16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (128, 64, 200)).save(buf, format="PNG")
    return buf.getvalue()


async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


async def test_project_crud_and_generate_flow(client):
    # create project
    r = await client.post("/api/projects", json={"name": "demo", "description": "t"})
    assert r.status_code == 200
    pid = r.json()["id"]

    # list + get
    assert any(p["id"] == pid for p in (await client.get("/api/projects")).json())
    assert (await client.get(f"/api/projects/{pid}")).json()["name"] == "demo"

    # upload asset
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("ref.png", _png_bytes(), "image/png"))],
        params={"caption": "front view"},
    )
    assert r.status_code == 200, r.text
    assets = r.json()
    aid = assets[0]["id"]
    assert assets[0]["width"] == 16

    # asset file download
    r = await client.get(f"/api/assets/{aid}/file")
    assert r.status_code == 200 and len(r.content) > 50

    # generate (enqueue will fail without redis but version+job are created)
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "a 40x20x10 block", "asset_ids": [aid]},
    )
    assert r.status_code == 200, r.text
    vid, jid = r.json()["version_id"], r.json()["job_id"]

    # version detail + tree
    v = (await client.get(f"/api/versions/{vid}")).json()
    assert v["status"] == "queued" and v["kind"] == "generate"
    tree = (await client.get(f"/api/projects/{pid}/versions")).json()
    assert len(tree) == 1 and tree[0]["number"] == 1

    # job exists; cancel it
    j = (await client.get(f"/api/jobs/{jid}")).json()
    assert j["type"] == "generate"
    r = await client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 200
    assert (await client.get(f"/api/jobs/{jid}")).json()["status"] == "cancelled"

    # edit rejected while parent not succeeded
    r = await client.post(f"/api/versions/{vid}/edit", json={"prompt": "add a hole"})
    assert r.status_code == 409


async def test_edit_library_parts_inherit_replace_and_clear(client):
    from cadx.db.models import Version, VersionStatus
    from cadx.db.session import session_scope

    project = (await client.post("/api/projects", json={"name": "library-edit"})).json()
    created = await client.post(
        f"/api/projects/{project['id']}/generate", json={"prompt": "a bracket"}
    )
    version_id = created.json()["version_id"]
    inherited = [str(uuid.uuid4())]
    async with session_scope() as db:
        parent = await db.get(Version, uuid.UUID(version_id))
        assert parent is not None
        parent.status = VersionStatus.succeeded
        parent.code = "from build123d import *\\nPARAMS = {}\\nresult = Box(1, 1, 1)"
        parent.library_part_ids = inherited
        await db.commit()

    child = await client.post(f"/api/versions/{version_id}/edit", json={"prompt": "keep"})
    assert child.status_code == 200, child.text
    detail = (await client.get(f"/api/versions/{child.json()['version_id']}")).json()
    assert detail["library_part_ids"] == inherited

    cleared = await client.post(
        f"/api/versions/{version_id}/edit",
        json={"prompt": "clear", "library_part_ids": []},
    )
    assert cleared.status_code == 200, cleared.text
    detail = (await client.get(f"/api/versions/{cleared.json()['version_id']}")).json()
    assert detail["library_part_ids"] == []

    selected = [str(uuid.uuid4())]
    replaced = await client.post(
        f"/api/versions/{version_id}/edit",
        json={"prompt": "replace", "library_part_ids": selected},
    )
    assert replaced.status_code == 200, replaced.text
    detail = (await client.get(f"/api/versions/{replaced.json()['version_id']}")).json()
    assert detail["library_part_ids"] == selected


async def test_project_delete_cascades(client):
    r = await client.post("/api/projects", json={"name": "del"})
    pid = r.json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("ref.png", _png_bytes(), "image/png"))],
    )
    assert r.status_code == 200
    r = await client.post(f"/api/projects/{pid}/generate", json={"prompt": "box"})
    vid = r.json()["version_id"]

    # leaf version delete removes its job + artifacts
    r = await client.delete(f"/api/versions/{vid}")
    assert r.status_code == 200
    # project delete removes remaining children
    r = await client.delete(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert (await client.get(f"/api/projects/{pid}")).status_code == 404


async def test_answer_requires_waiting_job(client):
    r = await client.post("/api/projects", json={"name": "p2"})
    pid = r.json()["id"]
    r = await client.post(f"/api/projects/{pid}/generate", json={"prompt": "x"})
    jid = r.json()["job_id"]
    r = await client.post(f"/api/jobs/{jid}/answer", json={"answers": [{"answer": "2mm"}]})
    assert r.status_code == 409


async def test_settings_roundtrip(client):
    r = await client.get("/api/settings")
    assert r.status_code == 200
    r = await client.put("/api/settings", json={"llm_model_code": "my-model"})
    assert r.json()["llm_model_code"] == "my-model"
    r = await client.get("/api/settings")
    assert r.json()["llm_model_code"] == "my-model"


async def test_generate_mode_persisted(client):
    r = await client.post("/api/projects", json={"name": "modes"})
    pid = r.json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "organic knob", "mode": "mesh"},
    )
    assert r.status_code == 200, r.text
    vid = r.json()["version_id"]
    v = (await client.get(f"/api/versions/{vid}")).json()
    assert v["gen_mode"] == "mesh"
    # invalid mode rejected
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "x", "mode": "bogus"},
    )
    assert r.status_code == 422


async def test_mesh_endpoints_unconfigured(client):
    # no runtime row / empty mesh_base_url in the test DB → unconfigured
    r = await client.get("/api/mesh/health")
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "not configured" in (r.json().get("error") or "")
    r = await client.get("/api/mesh/models")
    assert r.status_code == 200
    assert r.json() == []


async def test_catalog(client):
    r = await client.get("/api/catalog")
    assert r.status_code == 200
    cat = r.json()
    assert "screws" in cat and "gears" in cat
    entry = cat["screws"]["CounterSunkScrew"]
    assert entry["kind"] == "normed" and "iso10642" in entry["types"]

    r = await client.get("/api/catalog/search?q=m4")
    assert r.status_code == 200 and r.json()["results"]

    r = await client.get(
        "/api/catalog/lookup",
        params={"class_name": "CounterSunkScrew", "fastener_type": "iso10642", "size": "M4-0.7"},
    )
    assert r.status_code == 200
    assert r.json()["clearance_hole_diameters"]["Normal"] == 4.5

    r = await client.get("/api/catalog/lookup", params={"class_name": "Bogus"})
    assert r.status_code == 404


async def test_optimize_endpoint(client):
    from cadx.db.models import Version, VersionStatus
    from cadx.db.session import session_scope

    r = await client.post("/api/projects", json={"name": "opt"})
    pid = r.json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/generate", json={"prompt": "bracket", "mode": "cad"}
    )
    vid = r.json()["version_id"]

    # optimize is only allowed on finished versions
    r = await client.post(f"/api/versions/{vid}/optimize", json={})
    assert r.status_code == 409

    async with session_scope() as db:
        v = await db.get(Version, uuid.UUID(vid))
        assert v is not None
        v.status = VersionStatus.succeeded
        await db.commit()

    r = await client.post(
        f"/api/versions/{vid}/optimize",
        json={
            "load_cases": [
                {
                    "name": "c",
                    "fixed": [{"kind": "face", "axis": "-x"}],
                    "loads": [{"region": {"kind": "face", "axis": "+x"}, "force_n": [0, 0, -50]}],
                }
            ],
            "optimize": {"target_volume_fraction": 0.4},
        },
    )
    assert r.status_code == 200, r.text
    vid2 = r.json()["version_id"]
    v2 = (await client.get(f"/api/versions/{vid2}")).json()
    assert v2["kind"] == "optimize"
    assert v2["gen_mode"] == "topopt"
    assert v2["parent_id"] == vid
    assert v2["options"].get("topopt") is True


async def test_404s(client):
    z = uuid.uuid4()
    assert (await client.get(f"/api/projects/{z}")).status_code == 404
    assert (await client.get(f"/api/versions/{z}")).status_code == 404
    assert (await client.get(f"/api/jobs/{z}")).status_code == 404
    assert (await client.get(f"/api/assets/{z}/file")).status_code == 404


async def test_version_preview_endpoint(client):
    """Provisional renders are served while a job runs; newest iteration wins."""
    from cadx import storage

    pid = (await client.post("/api/projects", json={"name": "preview"})).json()["id"]
    vid = (await client.post(f"/api/projects/{pid}/generate", json={"prompt": "x"})).json()[
        "version_id"
    ]

    assert (await client.get(f"/api/versions/{vid}/preview")).status_code == 404

    # iteration 1 render appears (iso preferred over other views)
    d1 = storage.iteration_dir(pid, vid, 1) / "renders"
    d1.mkdir(parents=True)
    (d1 / "front.png").write_bytes(_png_bytes(size=(16, 16)))
    (d1 / "iso.png").write_bytes(_png_bytes(size=(24, 24)))
    r = await client.get(f"/api/versions/{vid}/preview")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "no-store"
    assert Image.open(io.BytesIO(r.content)).size == (24, 24)

    # iteration 2 only has a contact sheet — newest iteration wins
    d2 = storage.iteration_dir(pid, vid, 2) / "renders"
    d2.mkdir(parents=True)
    (d2 / "contact_sheet.png").write_bytes(_png_bytes(size=(32, 32)))
    r = await client.get(f"/api/versions/{vid}/preview")
    assert Image.open(io.BytesIO(r.content)).size == (32, 32)


async def test_job_finish_endpoint(client):
    from cadx.db.models import Job, JobStatus
    from cadx.db.session import session_scope

    pid = (await client.post("/api/projects", json={"name": "finish-test"})).json()["id"]
    created = await client.post(f"/api/projects/{pid}/generate", json={"prompt": "a block"})
    jid = created.json()["job_id"]

    # queued jobs can't 'finish early' — nothing to keep yet
    r = await client.post(f"/api/jobs/{jid}/finish")
    assert r.status_code == 409

    async with session_scope() as s:
        j = await s.get(Job, uuid.UUID(jid))
        j.status = JobStatus.running
        await s.commit()

    r = await client.post(f"/api/jobs/{jid}/finish")
    assert r.status_code == 200
    j = (await client.get(f"/api/jobs/{jid}")).json()
    assert j["finish_requested"] is True

    # terminal states reject it
    async with session_scope() as s:
        j = await s.get(Job, uuid.UUID(jid))
        j.status = JobStatus.failed
        await s.commit()
    r = await client.post(f"/api/jobs/{jid}/finish")
    assert r.status_code == 409

    r = await client.post(f"/api/jobs/{uuid.uuid4()}/finish")
    assert r.status_code == 404


async def test_job_restart_generate(client):
    """Restarting a terminal generate job clones inputs into a new version."""
    pid = (await client.post("/api/projects", json={"name": "restart-gen"})).json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("ref.png", _png_bytes(), "image/png"))],
    )
    aid = r.json()[0]["id"]
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "a knob", "asset_ids": [aid], "mode": "cad"},
    )
    jid, vid = r.json()["job_id"], r.json()["version_id"]

    # running/queued jobs can't restart
    r = await client.post(f"/api/jobs/{jid}/restart")
    assert r.status_code == 409

    await client.post(f"/api/jobs/{jid}/cancel")
    r = await client.post(f"/api/jobs/{jid}/restart")
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["version_id"] != vid and new["job_id"] != jid

    nv = (await client.get(f"/api/versions/{new['version_id']}")).json()
    assert nv["prompt"] == "a knob"
    assert nv["asset_ids"] == [aid]
    assert nv["gen_mode"] == "cad"
    assert nv["status"] == "queued"
    nj = (await client.get(f"/api/jobs/{new['job_id']}")).json()
    assert nj["type"] == "generate" and nj["version_id"] == new["version_id"]

    r = await client.post(f"/api/jobs/{uuid.uuid4()}/restart")
    assert r.status_code == 404


async def test_job_restart_params(client):
    """A params job restart carries the same effective overrides."""
    from cadx.db.models import Job, JobStatus, Version
    from cadx.db.session import session_scope

    pid = (await client.post("/api/projects", json={"name": "restart-params"})).json()["id"]
    r = await client.post(f"/api/projects/{pid}/generate", json={"prompt": "a block"})
    vid, jid = r.json()["version_id"], r.json()["job_id"]

    # give the parent code + succeeded so /params is allowed
    async with session_scope() as s:
        v = await s.get(Version, uuid.UUID(vid))
        v.code = 'PARAMS = {"w": 10}\nresult = None\n'
        v.status = "succeeded"
        j = await s.get(Job, uuid.UUID(jid))
        j.status = JobStatus.succeeded
        await s.commit()

    r = await client.post(f"/api/versions/{vid}/params", json={"params": {"w": 20}})
    assert r.status_code == 200, r.text
    p_jid = r.json()["job_id"]

    await client.post(f"/api/jobs/{p_jid}/cancel")
    r = await client.post(f"/api/jobs/{p_jid}/restart")
    assert r.status_code == 200, r.text
    nv = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    assert nv["kind"] == "params" and nv["params"]["w"] == 20
    assert nv["parent_id"] == vid
