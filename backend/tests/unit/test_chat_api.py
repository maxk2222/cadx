"""Chat API: CRUD + message posting against isolated SQLite (no redis)."""

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


async def _mk_project(client: AsyncClient, name: str = "chat-proj") -> str:
    r = await client.post("/api/projects", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["id"]


async def test_chat_crud_and_message_flow(client):
    pid = await _mk_project(client)

    # create
    r = await client.post("/api/chats", json={"project_id": pid})
    assert r.status_code == 201, r.text
    chat = r.json()
    cid = chat["id"]
    assert chat["title"] == "Chat" and chat["turn_running"] is False

    # listed for the project
    chats = (await client.get(f"/api/projects/{pid}/chats")).json()
    assert any(c["id"] == cid for c in chats)

    # empty state
    state = (await client.get(f"/api/chats/{cid}")).json()
    assert state["chat"]["id"] == cid and state["messages"] == []

    # post a message — persisted immediately, pending=True (no worker in tests)
    r = await client.post(f"/api/chats/{cid}/messages", json={"content": "make me a bracket"})
    assert r.status_code == 201, r.text
    msg = r.json()
    assert msg["role"] == "user" and msg["pending"] is True

    # title adopted from the first message
    state = (await client.get(f"/api/chats/{cid}")).json()
    assert state["chat"]["title"] == "make me a bracket"
    assert len(state["messages"]) == 1

    # rename
    r = await client.patch(f"/api/chats/{cid}", json={"title": "bracket chat"})
    assert r.status_code == 200 and r.json()["title"] == "bracket chat"

    # run endpoint is idempotent (enqueue fails w/o redis but must not 500)
    assert (await client.post(f"/api/chats/{cid}/run")).status_code == 200

    # cancel requires a running turn
    assert (await client.post(f"/api/chats/{cid}/cancel")).status_code == 409

    # delete
    assert (await client.delete(f"/api/chats/{cid}")).status_code == 200
    assert (await client.get(f"/api/chats/{cid}")).status_code == 404


async def test_chat_message_with_assets(client):
    pid = await _mk_project(client, "chat-assets")
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("ref.png", _png_bytes(), "image/png"))],
    )
    aid = r.json()[0]["id"]

    cid = (await client.post("/api/chats", json={"project_id": pid})).json()["id"]

    # image-only message is allowed
    r = await client.post(f"/api/chats/{cid}/messages", json={"asset_ids": [aid]})
    assert r.status_code == 201, r.text
    assert r.json()["asset_ids"] == [aid]

    # foreign / missing asset ids are rejected
    r = await client.post(
        f"/api/chats/{cid}/messages",
        json={"content": "x", "asset_ids": [str(uuid.uuid4())]},
    )
    assert r.status_code == 404


async def test_chat_validation_and_404s(client):
    pid = await _mk_project(client, "chat-404")
    cid = (await client.post("/api/chats", json={"project_id": pid})).json()["id"]

    # empty message (no text, no assets)
    r = await client.post(f"/api/chats/{cid}/messages", json={"content": "   "})
    assert r.status_code == 422

    missing = str(uuid.uuid4())
    assert (await client.get(f"/api/chats/{missing}")).status_code == 404
    assert (
        await client.post(f"/api/chats/{missing}/messages", json={"content": "hi"})
    ).status_code == 404
    assert (
        await client.post("/api/chats", json={"project_id": str(uuid.uuid4())})
    ).status_code == 404


async def test_unbound_chat_and_attach(client):
    """Chats can live without a project; the attach endpoint binds an
    existing project or creates one on the fly."""
    # project-less chat
    r = await client.post("/api/chats", json={})
    assert r.status_code == 201, r.text
    chat = r.json()
    cid = chat["id"]
    assert chat["project_id"] is None

    # assets rejected while unbound
    r = await client.post(f"/api/chats/{cid}/messages", json={"asset_ids": [str(uuid.uuid4())]})
    assert r.status_code == 409

    # attach by name → creates + binds
    r = await client.post(f"/api/chats/{cid}/project", json={"name": "knob"})
    assert r.status_code == 200, r.text
    pid = r.json()["project_id"]
    assert pid is not None
    proj = (await client.get(f"/api/projects/{pid}")).json()
    assert proj["name"] == "knob"

    # rebind to an existing project
    pid2 = await _mk_project(client, "other")
    r = await client.post(f"/api/chats/{cid}/project", json={"project_id": pid2})
    assert r.status_code == 200 and r.json()["project_id"] == pid2

    # invalid project id → 404
    r = await client.post(f"/api/chats/{cid}/project", json={"project_id": str(uuid.uuid4())})
    assert r.status_code == 404

    # global list shows the project name
    chats = (await client.get("/api/chats")).json()
    row = next(c for c in chats if c["id"] == cid)
    assert row["project_name"] == "other"

    # unbound chat shows null project_name
    cid2 = (await client.post("/api/chats", json={"title": "loose"})).json()["id"]
    chats = (await client.get("/api/chats")).json()
    row2 = next(c for c in chats if c["id"] == cid2)
    assert row2["project_name"] is None


async def test_attach_creates_default_project(client):
    """Empty attach body → 'New project' created and bound."""
    cid = (await client.post("/api/chats", json={})).json()["id"]
    r = await client.post(f"/api/chats/{cid}/project", json={})
    assert r.status_code == 200
    pid = r.json()["project_id"]
    proj = (await client.get(f"/api/projects/{pid}")).json()
    assert proj["name"] == "New project"


async def test_delete_project_cascades_chats(client):
    """Deleting a project removes its bound chats (FK would fire otherwise)."""
    pid = await _mk_project(client, "del-proj")
    cid = (await client.post("/api/chats", json={"project_id": pid})).json()["id"]
    r = await client.post(f"/api/chats/{cid}/messages", json={"content": "hi"})
    assert r.status_code == 201
    assert (await client.delete(f"/api/projects/{pid}")).status_code == 200
    assert (await client.get(f"/api/chats/{cid}")).status_code == 404
