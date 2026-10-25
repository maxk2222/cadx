"""Parts: model, resolution rules, REST CRUD/cascade, agent + MCP surface.

Covers the whole feature contract end to end on isolated SQLite:
auto-parts on legacy root spawns, explicit part_id/part_name resolution,
child inheritance, the ?part_id= tree filter, part deletion with the
running-job guard, and the agent/MCP tool exposure boundary.
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


async def _project(client, name="parts-proj") -> str:
    r = await client.post("/api/projects", json={"name": name})
    assert r.status_code == 200
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# resolution rules                                                             #
# --------------------------------------------------------------------------- #


async def test_root_generate_auto_creates_part(client):
    """Legacy spawn without part → deterministic auto part derived from prompt."""
    pid = await _project(client, "auto-part")
    r = await client.post(f"/api/projects/{pid}/generate", json={"prompt": "a 40x20 plate"})
    assert r.status_code == 200, r.text
    vid = r.json()["version_id"]

    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert len(parts) == 1
    assert parts[0]["number"] == 1 and parts[0]["name"] == "a 40x20 plate"
    assert parts[0]["version_count"] == 1

    v = (await client.get(f"/api/versions/{vid}")).json()
    assert v["part_id"] == parts[0]["id"]

    # second root, different prompt → its own part P2 (list is newest-first)
    await client.post(f"/api/projects/{pid}/generate", json={"prompt": "a knob"})
    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert len(parts) == 2 and parts[0]["number"] == 2 and parts[1]["number"] == 1
    p = (await client.get(f"/api/projects/{pid}")).json()
    assert p["parts_count"] == 2


async def test_explicit_part_id_and_name_resolution(client):
    pid = await _project(client, "explicit-part")
    r = await client.post(f"/api/projects/{pid}/parts", json={"name": "bracket"})
    assert r.status_code == 200, r.text
    part = r.json()
    assert part["number"] == 1 and part["name"] == "bracket"

    # explicit part_id → joins that part
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "first try", "part_id": part["id"]},
    )
    assert r.status_code == 200, r.text
    v = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    assert v["part_id"] == part["id"]

    # part_name case-insensitive match → same part, no duplicate
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "second try", "part_name": "BRACKET"},
    )
    assert r.status_code == 200, r.text
    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert len(parts) == 1
    v = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    assert v["part_id"] == part["id"]

    # new part_name → created on the fly
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "x", "part_name": "housing"},
    )
    parts = (await client.get(f"/api/projects/{pid}/parts")).json()
    assert len(parts) == 2 and parts[0]["name"] == "housing"


async def test_part_id_validation(client):
    pid = await _project(client, "bad-part")
    other = await _project(client, "other-proj")
    foreign = (await client.post(f"/api/projects/{other}/parts", json={"name": "x"})).json()

    # cross-project reference → 404
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "x", "part_id": foreign["id"]},
    )
    assert r.status_code == 404

    # malformed id → 422
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "x", "part_id": "not-a-uuid"},
    )
    assert r.status_code == 422


async def test_child_inherits_parent_part(client):
    from cadx.db.models import Version, VersionStatus
    from cadx.db.session import session_scope

    pid = await _project(client, "inherit-part")
    r = await client.post(f"/api/projects/{pid}/generate", json={"prompt": "base"})
    vid = r.json()["version_id"]
    parent = (await client.get(f"/api/versions/{vid}")).json()

    async with session_scope() as db:
        v = await db.get(Version, uuid.UUID(vid))
        assert v is not None
        v.status = VersionStatus.succeeded
        v.code = "from build123d import *\nPARAMS = {}\nresult = Box(1, 1, 1)"
        await db.commit()

    r = await client.post(f"/api/versions/{vid}/edit", json={"prompt": "tweak"})
    assert r.status_code == 200, r.text
    child = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    assert child["part_id"] == parent["part_id"]
    assert child["parent_id"] == vid


async def test_versions_part_filter(client):
    pid = await _project(client, "filter-part")
    p1 = (await client.post(f"/api/projects/{pid}/parts", json={"name": "a"})).json()
    await client.post(f"/api/projects/{pid}/generate", json={"prompt": "one", "part_id": p1["id"]})
    await client.post(f"/api/projects/{pid}/generate", json={"prompt": "two"})

    tree = (await client.get(f"/api/projects/{pid}/versions")).json()
    assert len(tree) == 2
    # newest version first
    assert [v["number"] for v in tree] == [2, 1]
    scoped = (await client.get(f"/api/projects/{pid}/versions?part_id={p1['id']}")).json()
    assert len(scoped) == 1 and scoped[0]["part_id"] == p1["id"]


# --------------------------------------------------------------------------- #
# CRUD + cascade                                                               #
# --------------------------------------------------------------------------- #


async def test_part_crud_and_cascade_delete(client):
    from cadx.db.models import Job, Version
    from cadx.db.session import session_scope

    pid = await _project(client, "crud-part")
    r = await client.post(f"/api/projects/{pid}/generate", json={"prompt": "gear"})
    jid, vid = r.json()["job_id"], r.json()["version_id"]
    part = (await client.get(f"/api/projects/{pid}/parts")).json()[0]

    # running job (queued counts as non-terminal) blocks the delete → 409
    r = await client.delete(f"/api/parts/{part['id']}")
    assert r.status_code == 409

    # cancel the job, then the cascade succeeds
    await client.post(f"/api/jobs/{jid}/cancel")
    r = await client.delete(f"/api/parts/{part['id']}")
    assert r.status_code == 200 and r.json()["versions_deleted"] == 1

    async with session_scope() as db:
        assert await db.get(Version, uuid.UUID(vid)) is None
        assert await db.get(Job, uuid.UUID(jid)) is None

    # rename via PATCH
    p2 = (await client.post(f"/api/projects/{pid}/parts", json={"name": "x"})).json()
    r = await client.patch(f"/api/parts/{p2['id']}", json={"name": "shaft"})
    assert r.status_code == 200 and r.json()["name"] == "shaft"

    # unknown ids → 404
    assert (await client.patch(f"/api/parts/{uuid.uuid4()}", json={"name": "y"})).status_code == 404
    assert (await client.delete(f"/api/parts/{uuid.uuid4()}")).status_code == 404


async def test_empty_part_delete(client):
    pid = await _project(client, "empty-part")
    p = (await client.post(f"/api/projects/{pid}/parts", json={"name": "empty"})).json()
    r = await client.delete(f"/api/parts/{p['id']}")
    assert r.status_code == 200 and r.json()["versions_deleted"] == 0


async def test_part_cascade_fk_ordering():
    """Regression: version.parent_id is a self-FK — ORM `db.delete` puts
    version rows into the unit-of-work's cyclic group, flushed AFTER the
    part delete (fk_version_part violation on Postgres). The service must
    emit immediate ordered deletes instead. Runs on an in-memory engine
    with real FK enforcement — the shared test engine leaves FKs off."""
    from sqlalchemy import event, func, select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from cadx.db.models import (
        Job,
        JobStatus,
        Part,
        Project,
        SQLModel,
        Version,
        VersionKind,
        VersionStatus,
    )
    from cadx.parts import delete_part

    eng = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(eng.sync_engine, "connect")
    def _fk_on(dbapi_conn, _rec):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as c:
        await c.run_sync(SQLModel.metadata.create_all)
    sm = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        proj = Project(name="fk-repro")
        db.add(proj)
        await db.flush()
        part = Part(project_id=proj.id, number=1, name="p")
        db.add(part)
        await db.flush()
        v1 = Version(
            project_id=proj.id,
            part_id=part.id,
            number=1,
            kind=VersionKind.generate,
            status=VersionStatus.succeeded,
        )
        db.add(v1)
        await db.flush()
        v2 = Version(
            project_id=proj.id,
            part_id=part.id,
            parent_id=v1.id,
            number=2,
            kind=VersionKind.edit,
            status=VersionStatus.cancelled,
        )
        db.add(v2)
        await db.flush()
        db.add(Job(version_id=v2.id, type="generate", status=JobStatus.cancelled, payload="{}"))
        await db.commit()
        pid = part.id

    async with sm() as db:
        part = await db.get(Part, pid)
        assert await delete_part(db, part) == 2
        left = await db.execute(select(func.count()).select_from(Version))
        assert left.scalar_one() == 0
    await eng.dispose()


# --------------------------------------------------------------------------- #
# service unit checks                                                          #
# --------------------------------------------------------------------------- #


def test_derive_part_name():
    from cadx.parts import derive_part_name

    assert derive_part_name("", 3) == "Part 3"
    assert derive_part_name("   ", 1) == "Part 1"
    assert derive_part_name("a 40x20 plate with holes", 1) == "a 40x20 plate with holes"
    long_latin = "a very descriptive mounting bracket for the enclosure " * 2
    n = derive_part_name(long_latin, 2)
    assert len(n) <= 48 and not n.endswith(" ")
    cjk = "これは非常に長いプロンプトです部品の設計" * 4
    assert len(derive_part_name(cjk, 4)) <= 48


# --------------------------------------------------------------------------- #
# agent + MCP surface                                                          #
# --------------------------------------------------------------------------- #


async def test_agent_list_parts_and_scoped_list_versions(client):
    from cadx.agent.chat_tools import ChatToolSession
    from cadx.agent.trace import MemoryTrace
    from cadx.agent.types import AgentCtx
    from cadx.db.models import Chat, ProjectSettings
    from cadx.db.session import session_scope
    from cadx.llm.fake import FakeLLM
    from cadx.runtime_config import RuntimeSettings

    pid = await _project(client, "agent-part")
    await client.post(f"/api/projects/{pid}/generate", json={"prompt": "plate"})

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
    session = ChatToolSession(ctx, chat_id=cid, attach_images=False)
    out = await session.execute("list_parts", {})
    assert out["ok"] and len(out["parts"]) == 1
    part = out["parts"][0]
    assert part["number"] == 1 and part["version_count"] == 1

    vers = await session.execute("list_versions", {"part_id": part["id"]})
    assert len(vers["versions"]) == 1
    assert vers["versions"][0]["part"].startswith("P1")
    empty = await session.execute("list_versions", {"part_id": str(uuid.uuid4())})
    assert empty["versions"] == []


def test_mcp_part_surface_boundary():
    """list_parts arrives via inherited CHAT_INFO_TOOLS; submit_code/
    mesh_generate expose part args; the LLM/delete boundary holds."""
    from cadx.agent.project_tools import CHAT_INFO_TOOLS
    from cadx.mcpserver.tools import MCP_TOOL_NAMES, MCP_TOOL_SCHEMAS

    info_names = {t["function"]["name"] for t in CHAT_INFO_TOOLS}
    assert "list_parts" in info_names

    props = {
        t["function"]["name"]: t["function"]["parameters"]["properties"] for t in MCP_TOOL_SCHEMAS
    }
    assert "part_id" in props["submit_code"] and "part_name" in props["submit_code"]
    assert "part_id" in props["mesh_generate"] and "part_name" in props["mesh_generate"]

    # hard boundary: no LLM-driven or delete tools on MCP
    assert MCP_TOOL_NAMES.isdisjoint({"generate_part", "edit_version", "answer_questions"})
    assert not any(n.startswith("delete_") for n in MCP_TOOL_NAMES)


# --------------------------------------------------------------------------- #
# backfill                                                                     #
# --------------------------------------------------------------------------- #


async def test_backfill_groups_trees_and_orphans():
    """Versions under one root share a part; orphans get their own."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from cadx.db.backfill import backfill_parts

    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as c:
        await c.exec_driver_sql("CREATE TABLE project (id CHAR(32) PRIMARY KEY)")
        await c.exec_driver_sql(
            "CREATE TABLE part (id CHAR(32) PRIMARY KEY, project_id CHAR(32),"
            " number INTEGER, name VARCHAR, created_at DATETIME, updated_at DATETIME)"
        )
        await c.exec_driver_sql(
            "CREATE TABLE version (id CHAR(32) PRIMARY KEY, project_id CHAR(32),"
            " part_id CHAR(32), parent_id CHAR(32), number INTEGER, prompt VARCHAR,"
            " name VARCHAR)"
        )

        proj = uuid.uuid4()

        def h(u: uuid.UUID) -> str:
            return u.hex

        await c.exec_driver_sql(f"INSERT INTO project VALUES ('{h(proj)}')")
        v1, v2, v3, v4, v5, v6 = (uuid.uuid4() for _ in range(6))
        rows = [
            (v1, None, 1, "bracket 40x20", ""),
            (v2, v1, 2, "edit", ""),
            (v3, v2, 3, "refine", ""),
            (v4, None, 4, "housing", "Main housing"),
            (v5, v4, 5, "edit", ""),
            (v6, uuid.uuid4(), 6, "dangling", ""),  # parent points nowhere → orphan
        ]
        for vid, par, num, prompt, name in rows:
            await c.exec_driver_sql(
                "INSERT INTO version (id, project_id, part_id, parent_id, number,"
                " prompt, name) VALUES (:id, :p, NULL, :par, :n, :pr, :na)",
                {
                    "id": h(vid),
                    "p": h(proj),
                    "par": h(par) if par else None,
                    "n": num,
                    "pr": prompt,
                    "na": name,
                },
            )

        await c.run_sync(backfill_parts)

        parts = (await c.exec_driver_sql("SELECT id, number, name FROM part ORDER BY number")).all()
        assert len(parts) == 3

        # sa.Uuid may hand back UUID objects or hex strings depending on the
        # dialect path — normalise to 32-hex for comparison
        def norm(x):
            return x.hex if isinstance(x, uuid.UUID) else x

        by_number = {r[1]: (norm(r[0]), r[2]) for r in parts}
        assert by_number[1][1] == "bracket 40x20"  # derived from root prompt
        assert by_number[2][1] == "Main housing"  # root version name wins
        assert by_number[3][1] == "dangling"  # orphan → own part

        vs = (await c.exec_driver_sql("SELECT id, part_id FROM version")).all()
        part_of = {norm(r[0]): norm(r[1]) for r in vs}
        for vid in (h(v1), h(v2), h(v3)):
            assert part_of[vid] == by_number[1][0]
        for vid in (h(v4), h(v5)):
            assert part_of[vid] == by_number[2][0]
        assert part_of[h(v6)] == by_number[3][0]
    await eng.dispose()
