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
