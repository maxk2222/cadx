"""API smoke tests against an isolated SQLite DB (no redis/LLM)."""

from __future__ import annotations

import io
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from cadagent.db.session import create_all


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadagent.main import app

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
    assert "screws" in cat or "nuts" in cat


async def test_404s(client):
    z = uuid.uuid4()
    assert (await client.get(f"/api/projects/{z}")).status_code == 404
    assert (await client.get(f"/api/versions/{z}")).status_code == 404
    assert (await client.get(f"/api/jobs/{z}")).status_code == 404
    assert (await client.get(f"/api/assets/{z}/file")).status_code == 404
