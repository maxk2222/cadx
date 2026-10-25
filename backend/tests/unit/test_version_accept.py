"""Version acceptance: user-verified master version per part.

Contract: at most one accepted version per part; any status is eligible;
accepting a sibling clears the previous one; new spawns never steal the
flag; deleting the master leaves the part without one (warned, manual).
Read surfaces: REST summaries/details/parts + agent list tools.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from cadx.db.session import create_all


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _project(client, name="accept-proj") -> str:
    r = await client.post("/api/projects", json={"name": name})
    assert r.status_code == 200
    return r.json()["id"]


async def _spawn(client, pid: str, prompt: str, part_id: str) -> str:
    r = await client.post(
        f"/api/projects/{pid}/generate", json={"prompt": prompt, "part_id": part_id}
    )
    assert r.status_code == 200, r.text
    return r.json()["version_id"]


# --------------------------------------------------------------------------- #
# REST surface                                                                 #
# --------------------------------------------------------------------------- #


async def test_accept_sets_flag_and_part_master(client):
    pid = await _project(client, "accept-basic")
    part = (await client.post(f"/api/projects/{pid}/parts", json={"name": "gear"})).json()
    vid = await _spawn(client, pid, "v1", part["id"])

    # even a queued version is eligible (any status may be accepted)
    r = await client.patch(f"/api/versions/{vid}", json={"accepted": True})
    assert r.status_code == 200, r.text
    assert r.json()["accepted_at"] is not None

    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert parts[0]["accepted_version_id"] == vid

    tree = (await client.get(f"/api/projects/{pid}/versions")).json()
    assert tree[0]["accepted_at"] is not None

    detail = (await client.get(f"/api/versions/{vid}")).json()
    assert detail["accepted_at"] is not None


async def test_accept_switch_clears_sibling(client):
    """One master per part: accepting v2 unmarks v1 atomically."""
    pid = await _project(client, "accept-switch")
    part = (await client.post(f"/api/projects/{pid}/parts", json={"name": "p"})).json()
    v1 = await _spawn(client, pid, "first", part["id"])
    v2 = await _spawn(client, pid, "second", part["id"])

    await client.patch(f"/api/versions/{v1}", json={"accepted": True})
    await client.patch(f"/api/versions/{v2}", json={"accepted": True})

    tree = {v["id"]: v for v in (await client.get(f"/api/projects/{pid}/versions")).json()}
    assert tree[v1]["accepted_at"] is None
    assert tree[v2]["accepted_at"] is not None

    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert parts[0]["accepted_version_id"] == v2


async def test_unaccept_and_reaccept(client):
    pid = await _project(client, "accept-clear")
    part = (await client.post(f"/api/projects/{pid}/parts", json={"name": "p"})).json()
    vid = await _spawn(client, pid, "v1", part["id"])

    await client.patch(f"/api/versions/{vid}", json={"accepted": True})
    r = await client.patch(f"/api/versions/{vid}", json={"accepted": False})
    assert r.json()["accepted_at"] is None
    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert parts[0]["accepted_version_id"] is None

    # re-accept is allowed (idempotent flag semantics)
    r = await client.patch(f"/api/versions/{vid}", json={"accepted": True})
    assert r.json()["accepted_at"] is not None

    # combined rename + accept in one PATCH works
    r = await client.patch(f"/api/versions/{vid}", json={"name": "final", "accepted": True})
    assert r.status_code == 200 and r.json()["name"] == "final"


async def test_accept_scoped_to_own_part(client):
    """Accepting in part A never touches part B's master."""
    pid = await _project(client, "accept-scope")
    pa = (await client.post(f"/api/projects/{pid}/parts", json={"name": "a"})).json()
    pb = (await client.post(f"/api/projects/{pid}/parts", json={"name": "b"})).json()
    va = await _spawn(client, pid, "x", pa["id"])
    vb = await _spawn(client, pid, "y", pb["id"])

    await client.patch(f"/api/versions/{va}", json={"accepted": True})
    await client.patch(f"/api/versions/{vb}", json={"accepted": True})

    parts = {p["id"]: p for p in (await client.get(f"/api/projects/{pid}/parts")).json()}
    assert parts[pa["id"]]["accepted_version_id"] == va
    assert parts[pb["id"]]["accepted_version_id"] == vb


async def test_new_spawn_keeps_master_and_delete_warns(client):
    """A newer version never inherits the flag; deleting the master reports
    it so the UI can warn — the part is then masterless (manual choice)."""
    pid = await _project(client, "accept-delete")
    part = (await client.post(f"/api/projects/{pid}/parts", json={"name": "p"})).json()
    v1 = await _spawn(client, pid, "old", part["id"])
    await client.patch(f"/api/versions/{v1}", json={"accepted": True})

    v2 = await _spawn(client, pid, "new", part["id"])
    tree = {v["id"]: v for v in (await client.get(f"/api/projects/{pid}/versions")).json()}
    assert tree[v1]["accepted_at"] is not None
    assert tree[v2]["accepted_at"] is None
    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert parts[0]["accepted_version_id"] == v1

    r = await client.delete(f"/api/versions/{v1}")
    assert r.status_code == 200 and r.json()["deleted_master"] is True
    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert parts[0]["accepted_version_id"] is None

    # deleting a non-master reports False
    r = await client.delete(f"/api/versions/{v2}")
    assert r.status_code == 200 and r.json()["deleted_master"] is False


async def test_accept_unknown_version_404(client):
    r = await client.patch(f"/api/versions/{uuid.uuid4()}", json={"accepted": True})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# agent / MCP surface                                                          #
# --------------------------------------------------------------------------- #


async def _chat_session(client, pid: str):
    """ChatToolSession bound to a chat on the given project (FakeLLM)."""
    from cadx.agent.chat_tools import ChatToolSession
    from cadx.agent.trace import MemoryTrace
    from cadx.agent.types import AgentCtx
    from cadx.db.models import Chat, ProjectSettings
    from cadx.db.session import session_scope
    from cadx.llm.fake import FakeLLM
    from cadx.runtime_config import RuntimeSettings

    async with session_scope() as db:
        chat = Chat(project_id=uuid.UUID(pid), title="t")
        db.add(chat)
        await db.commit()
        cid = chat.id

    async def emit(*_a, **_k) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    ctx = AgentCtx(
        llm=FakeLLM([]),
        executor=None,
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id=f"chat:{cid}",
        project_id=None,
        version_id=cid,
        mode="chat",
        prompt="",
        settings=ProjectSettings(),
        runtime_settings=RuntimeSettings(),
    )
    return ChatToolSession(ctx, chat_id=cid, attach_images=False)


async def test_agent_accept_version_and_list_markers(client):
    pid = await _project(client, "accept-agent")
    part = (await client.post(f"/api/projects/{pid}/parts", json={"name": "arm"})).json()
    v1 = await _spawn(client, pid, "one", part["id"])
    v2 = await _spawn(client, pid, "two", part["id"])

    session = await _chat_session(client, pid)

    out = await session.execute("accept_version", {"version_id": v1})
    assert out["ok"] and out["accepted"] is True and out["number"] == 1

    # switch via the tool — sibling cleared
    out = await session.execute("accept_version", {"version_id": v2, "accepted": True})
    assert out["ok"]

    vers = await session.execute("list_versions", {"part_id": part["id"]})
    by_id = {v["id"]: v for v in vers["versions"]}
    assert by_id[v1]["accepted"] is False
    assert by_id[v2]["accepted"] is True

    parts = await session.execute("list_parts", {})
    p = next(p for p in parts["parts"] if p["id"] == part["id"])
    assert p["accepted_version_id"] == v2
    assert p["accepted_version"] == "v2"

    det = await session.execute("version_details", {"version_id": v2})
    assert det["accepted"] is True

    # clear via the tool
    out = await session.execute("accept_version", {"version_id": v2, "accepted": False})
    assert out["ok"] and out["accepted"] is False
    det = await session.execute("version_details", {"version_id": v2})
    assert det["accepted"] is False

    # invalid arg
    out = await session.execute("accept_version", {"version_id": v2, "accepted": "yes"})
    assert out["ok"] is False


def test_accept_version_tool_surface():
    """accept_version is deterministic → lands in SAFE_ACTION_TOOLS, which
    the LLM-free MCP session picks up automatically."""
    from cadx.agent.project_tools import (
        CHAT_ACTION_TOOLS,
        LLM_ONLY_TOOL_NAMES,
        SAFE_ACTION_TOOLS,
    )

    names = {t["function"]["name"] for t in CHAT_ACTION_TOOLS}
    safe = {t["function"]["name"] for t in SAFE_ACTION_TOOLS}
    assert "accept_version" in names
    assert "accept_version" in safe
    assert "accept_version" not in LLM_ONLY_TOOL_NAMES
