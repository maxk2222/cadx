"""Backend audit fixes — focused behavioral coverage.

Covers: QueueExecutor/MCP session-pool lifecycle (task 1), version_tree
thumbnail batching (task 2), search_library SQL pushdown (task 3), shared
httpx clients + cancel-aware backoff (tasks 4/8), emit DB-write throttle +
Job.log cap (task 5), guide_dir trust (task 6), sweeper zscan (task 7),
spawn builder invariants (task 12), upload caps (task 15), service
allowlist (task 16).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
import uuid
from typing import Any

import pytest
from starlette.datastructures import Headers, UploadFile

from cadx.db.session import create_all, session_scope


@pytest.fixture(scope="module", autouse=True)
async def _db() -> None:
    await create_all()


# --------------------------------------------------------------------------- #
# task 1 — QueueExecutor.aclose + bounded MCP session pool                     #
# --------------------------------------------------------------------------- #


class _FakePool:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def test_executor_aclose_only_owned_pool() -> None:
    from cadx.agent.executor_client import QueueExecutor

    # caller-supplied pool is shared — never closed by the executor
    shared = _FakePool()
    ex = QueueExecutor(shared)
    await ex.aclose()
    assert not shared.closed

    # lazily-created pool IS closed
    ex2 = QueueExecutor(None)
    owned = _FakePool()
    ex2._pool = owned
    ex2._owns_pool = True
    await ex2.aclose()
    assert owned.closed
    assert ex2._pool is None

    # aclose is idempotent / a re-created pool is the new owner
    await ex.aclose()
    await ex2.aclose()


async def test_mcp_session_pool_bounded_ttl_and_drop() -> None:
    from cadx.mcpserver import session as mcp_session

    saved_sessions, saved_touched = dict(mcp_session.SESSIONS), dict(mcp_session._SESSION_TOUCHED)
    mcp_session.SESSIONS.clear()
    mcp_session._SESSION_TOUCHED.clear()
    try:
        # beyond the cap, the oldest (LRU) entry is evicted and closed
        first = await mcp_session.get_or_create("k0")
        evicted: list[str] = []
        orig_k0_aclose = first.aclose

        async def _evict_spy() -> None:
            evicted.append("k0")
            await orig_k0_aclose()

        first.aclose = _evict_spy  # type: ignore[method-assign]
        for i in range(1, mcp_session._MAX_SESSIONS + 1):
            await mcp_session.get_or_create(f"k{i}")
        assert len(mcp_session.SESSIONS) == mcp_session._MAX_SESSIONS
        assert "k0" not in mcp_session.SESSIONS  # evicted by the loop itself
        assert evicted == ["k0"]  # evicted session was async-closed

        overflow = await mcp_session.get_or_create("overflow")
        assert len(mcp_session.SESSIONS) == mcp_session._MAX_SESSIONS
        assert "overflow" in mcp_session.SESSIONS

        # an LRU hit keeps the same pooled session (moved to the back)
        again = await mcp_session.get_or_create("overflow")
        assert again is overflow

        # TTL: an idle-expired entry is closed + dropped on the next access
        mcp_session._SESSION_TOUCHED["overflow"] = (
            time.monotonic() - mcp_session._SESSION_TTL_S - 1
        )
        await mcp_session.get_or_create("k1")
        assert "overflow" not in mcp_session.SESSIONS

        # drop() removes the session and schedules its async close
        key = "dropme"
        sess = await mcp_session.get_or_create(key)
        closed = False
        orig_aclose = sess.aclose

        async def spy() -> None:
            nonlocal closed
            closed = True
            await orig_aclose()

        sess.aclose = spy  # type: ignore[method-assign]
        mcp_session.drop(key)
        assert key not in mcp_session.SESSIONS
        for _ in range(10):
            await asyncio.sleep(0)
            if closed:
                break
        assert closed

        mcp_session.drop("nonexistent")  # no-op, no raise
    finally:
        for s in mcp_session.SESSIONS.values():
            with contextlib.suppress(Exception):
                await s.aclose()
        mcp_session.SESSIONS.clear()
        mcp_session.SESSIONS.update(saved_sessions)
        mcp_session._SESSION_TOUCHED.clear()
        mcp_session._SESSION_TOUCHED.update(saved_touched)


async def test_mcp_session_aclose_releases_executor_pool() -> None:
    from cadx.mcpserver.session import McpToolSession

    sess = McpToolSession()
    owned = _FakePool()
    sess.ctx.executor._pool = owned
    sess.ctx.executor._owns_pool = True
    await sess.aclose()
    assert owned.closed


# --------------------------------------------------------------------------- #
# task 2 — version_tree batched thumbnails                                     #
# --------------------------------------------------------------------------- #


async def test_version_tree_batches_thumbnails() -> None:
    from cadx.api.routers.versions import _summary, version_tree
    from cadx.db.models import Artifact, ArtifactKind, Project, Version, VersionKind

    pid, part = uuid.uuid4(), uuid.uuid4()
    thumb_v: Version | None = None
    async with session_scope() as db:
        db.add(Project(id=pid, name="tree-proj"))
        versions = []
        for i in (1, 2):
            v = Version(
                project_id=pid,
                part_id=part,
                number=i,
                kind=VersionKind.generate,
                prompt=f"v{i}",
            )
            db.add(v)
            versions.append(v)
        await db.flush()
        thumb_v = versions[0]
        db.add(
            Artifact(
                version_id=thumb_v.id,
                kind=ArtifactKind.thumbnail,
                name="thumb",
                path="projects/x/thumb.png",
            )
        )
        await db.commit()

    async with session_scope() as db:
        # count every execute() — the tree must batch its lookups regardless
        # of how many versions the project carries (was one query per version)
        calls: list[str] = []
        real_execute = db.execute

        async def counted(stmt: Any, *a: Any, **k: Any) -> Any:
            calls.append(str(stmt))
            return await real_execute(stmt, *a, **k)

        db.execute = counted  # type: ignore[method-assign]
        out = await version_tree(pid, None, db)

        assert len(out) == 2
        by_id = {r.id: r for r in out}
        assert by_id[thumb_v.id].thumbnail_url is not None
        others = [r for r in out if r.id != thumb_v.id]
        assert all(r.thumbnail_url is None for r in others)
        # exactly 3 statements: versions + active jobs + ONE thumbnail query
        assert len(calls) == 3
        artifact_queries = [c for c in calls if "artifact" in c.lower()]
        assert len(artifact_queries) == 1

        # _summary stays backward-compatible: explicit thumbnail=None means
        # "looked up, none" — it must NOT issue a per-version query
        class _BoomDB:
            async def execute(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("_summary issued a query despite explicit thumbnail")

        s = await _summary(thumb_v, _BoomDB(), None, None)  # type: ignore[arg-type]
        assert s.thumbnail_url is None


# --------------------------------------------------------------------------- #
# task 3 — search_library SQL pushdown                                         #
# --------------------------------------------------------------------------- #


async def _mk_part(name: str, kind: str, filename: str, caption: str = "") -> None:
    from cadx.db.models import LibraryPart

    async with session_scope() as db:
        db.add(LibraryPart(name=name, kind=kind, filename=filename, caption=caption))
        await db.commit()


async def test_search_library_sql_filtering() -> None:
    from cadx.agent.toolspecs import ToolSession

    suffix = uuid.uuid4().hex[:8]
    await _mk_part(f"spacer-{suffix}", "step", "spacer.step", "10mm")
    await _mk_part(f"bolt-{suffix}", "stl", "bolt.stl", "m8")
    # discriminating case for wildcard escaping: 'a_b' as a literal must NOT
    # match 'axb' — an unescaped ilike '_' matches any single char
    await _mk_part(f"axb-{suffix}", "step", "axb.step", "")

    selfless: Any = None
    res = await ToolSession._tool_search_library(
        selfless, {"query": f"spacer-{suffix}"}
    )
    assert res["ok"] and res["count"] == 1
    assert res["results"][0]["name"] == f"spacer-{suffix}"

    res = await ToolSession._tool_search_library(selfless, {"kind": "step", "query": suffix})
    names = {r["name"] for r in res["results"]}
    assert f"spacer-{suffix}" in names and f"bolt-{suffix}" not in names

    # literal '_' escapes — 'a_b' must not match the 'axb-' name
    res = await ToolSession._tool_search_library(selfless, {"query": f"a_b-{suffix}"})
    assert res["count"] == 0


async def test_search_library_limit() -> None:
    from cadx.agent.toolspecs import ToolSession
    from cadx.db.models import LibraryPart

    marker = uuid.uuid4().hex[:8]
    async with session_scope() as db:
        for i in range(25):
            db.add(
                LibraryPart(name=f"bulk-{marker}-{i:02d}", kind="step", filename=f"b{i}.step")
            )
        await db.commit()

    res = await ToolSession._tool_search_library(None, {"query": f"bulk-{marker}"})
    assert res["count"] == 20 and len(res["results"]) == 20


# --------------------------------------------------------------------------- #
# tasks 4/8 — shared httpx clients + cancel-aware backoff                      #
# --------------------------------------------------------------------------- #


async def test_llm_client_shared_pool_and_aclose() -> None:
    from cadx.llm.client import LLMClient

    c = LLMClient(base_url="http://localhost:9", api_key="k", model_code="m", model_vision="m")
    c1 = c._get_client()
    assert c1 is c._get_client()  # same pooled client across calls
    await c.aclose()
    assert c1.is_closed
    c2 = c._get_client()  # transparent rebuild after close
    assert c2 is not c1
    await c.aclose()


async def test_llm_backoff_cancel_aware() -> None:
    from cadx.llm.client import LLMClient

    async def aborted() -> bool:
        return True

    c = LLMClient(
        base_url="http://localhost:9",
        api_key="k",
        model_code="m",
        model_vision="m",
        should_abort=aborted,
    )
    t0 = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        await c._backoff(30.0)
    assert time.monotonic() - t0 < 5.0  # did NOT sleep the full 30s

    # no abort callback -> plain sleep, no error
    c2 = LLMClient(base_url="http://localhost:9", api_key="k", model_code="m", model_vision="m")
    await c2._backoff(0.01)
    await c.aclose()
    await c2.aclose()


async def test_mesh_and_image_clients_share_one_pool() -> None:
    from cadx.agent.image_client import HttpImageClient
    from cadx.agent.mesh_client import HttpMeshClient

    for cls, kwargs in (
        (HttpMeshClient, {"base_url": "http://localhost:9"}),
        (HttpImageClient, {"base_url": "http://localhost:9"}),
    ):
        client = cls(**kwargs)
        c1 = client._get_client()
        assert c1 is client._get_client()
        await client.aclose()
        assert c1.is_closed


# --------------------------------------------------------------------------- #
# task 5 — emit DB-write throttle + Job.log cap                                #
# --------------------------------------------------------------------------- #


async def test_emit_throttles_db_writes_and_flush() -> None:
    from cadx.agent.worker_jobs import make_emit
    from cadx.db.models import Job, JobType, Project, Version, VersionKind

    async with session_scope() as db:
        p = Project(name="emit-proj")
        db.add(p)
        await db.flush()
        v = Version(project_id=p.id, part_id=uuid.uuid4(), number=1, kind=VersionKind.generate)
        db.add(v)
        await db.flush()
        job = Job(version_id=v.id, type=JobType.generate)
        db.add(job)
        await db.commit()
        jid = job.id

    emit = make_emit(jid)
    await emit("plan", 0, "first")  # step change -> immediate write
    async with session_scope() as db:
        j = await db.get(Job, jid)
        assert j is not None and "[plan:0] first" in (j.log or "")
        assert j.step == "plan"

    # same-step events are throttled — pending lines are NOT yet in the row
    await emit("plan", 1, "second")
    await emit("plan", 2, "third")
    async with session_scope() as db:
        j = await db.get(Job, jid)
        assert "second" not in (j.log or "")

    # terminal flush lands everything still pending
    await emit.flush()  # type: ignore[attr-defined]
    async with session_scope() as db:
        j = await db.get(Job, jid)
        assert "[plan:1] second" in j.log and "[plan:2] third" in j.log
        assert j.iteration == 2

    # a step change always flushes immediately even inside the throttle window
    await emit("execute", 0, "next step")
    async with session_scope() as db:
        j = await db.get(Job, jid)
        assert "[execute:0] next step" in j.log and j.step == "execute"


async def test_emit_caps_job_log() -> None:
    from cadx.agent.worker_jobs import _cap_job_log, make_emit
    from cadx.db.models import Job, JobType, Project, Version, VersionKind

    # unit-level cap: >500 lines keeps the newest with an elision marker
    blob = "".join(f"line {i}\n" for i in range(600))
    capped = _cap_job_log(blob)
    lines = capped.splitlines()
    assert len(lines) == 501  # marker + 500 kept
    assert "elided" in lines[0]
    assert lines[-1] == "line 599"
    assert _cap_job_log("short\n") == "short\n"

    # end-to-end: a burst of unique events lands capped in Job.log
    async with session_scope() as db:
        p = Project(name="cap-proj")
        db.add(p)
        await db.flush()
        v = Version(project_id=p.id, part_id=uuid.uuid4(), number=1, kind=VersionKind.generate)
        db.add(v)
        await db.flush()
        job = Job(version_id=v.id, type=JobType.generate)
        db.add(job)
        await db.commit()
        jid = job.id

    emit = make_emit(jid)
    for i in range(600):
        await emit("s", i, f"msg {i}")
    await emit.flush()  # type: ignore[attr-defined]
    async with session_scope() as db:
        j = await db.get(Job, jid)
        n = len((j.log or "").splitlines())
        assert n <= 501
        assert "elided" in j.log.splitlines()[0]


# --------------------------------------------------------------------------- #
# task 6 — guide_dir trust                                                      #
# --------------------------------------------------------------------------- #


async def test_resolve_guide_dir_confines_to_data_root(tmp_path) -> None:
    from cadx import storage
    from cadx.agent.worker_jobs import _resolve_guide_dir

    version = type("V", (), {"project_id": uuid.uuid4(), "id": uuid.uuid4()})()
    canonical = storage.version_dir(version.project_id, version.id) / "guide"

    # empty/missing -> canonical per-version dir
    assert _resolve_guide_dir(None, version) == canonical
    assert _resolve_guide_dir("", version) == canonical

    # a dir under data_root is honoured
    inside = storage.data_root() / "projects" / "x" / "guide"
    assert _resolve_guide_dir(str(inside), version) == inside.resolve()

    # anything outside the root is rejected -> canonical fallback
    assert _resolve_guide_dir("/etc/passwd", version) == canonical
    outside = storage.data_root() / ".." / "evil"
    assert _resolve_guide_dir(str(outside), version) == canonical
    # traversal that resolves INSIDE data_root is normalised and honoured
    vdir = storage.version_dir(version.project_id, version.id)
    sneaky_inside = str(vdir / ".." / ".." / "keep")
    resolved_inside = (
        storage.data_root() / "projects" / str(version.project_id) / "keep"
    ).resolve()
    assert _resolve_guide_dir(sneaky_inside, version) == resolved_inside
    # traversal that escapes data_root entirely is rejected
    sneaky_outside = str(vdir / ".." / ".." / ".." / ".." / ".." / "escape")
    assert _resolve_guide_dir(sneaky_outside, version) == canonical


# --------------------------------------------------------------------------- #
# task 7 — sweeper zscan, no full zrange                                        #
# --------------------------------------------------------------------------- #


class _FakeRedis:
    """Enough of the arq redis surface for _chat_turn_alive — scan + zscan."""

    def __init__(self, members: dict[bytes, float] | None = None, inprog: list[str] = ()) -> None:
        self._members = members or {}
        self._inprog = inprog
        self.calls: list[str] = []

    async def scan(self, cursor: int, match: str | None = None, count: int = 0):
        self.calls.append("scan")
        prefix = (match or "").removesuffix("*")
        return 0, [k for k in self._inprog if k.startswith(prefix)]

    async def zscan(self, name: str, cursor: int, match: str | None = None, count: int = 0):
        self.calls.append("zscan")
        prefix = (match or "").removesuffix("*")
        return 0, {m: s for m, s in self._members.items() if m.decode().startswith(prefix)}


async def test_chat_turn_alive_uses_zscan_not_zrange() -> None:
    from cadx.queue import _chat_turn_alive

    fake = _FakeRedis(members={b"chat:c1:m1": 1.0, b"job:other": 2.0})
    assert await _chat_turn_alive(fake, "c1") is True
    assert fake.calls == ["scan", "zscan"]  # queued member found via zscan

    fake2 = _FakeRedis(members={b"chat:c1:m1": 1.0})
    assert await _chat_turn_alive(fake2, "c2") is False  # no members for c2
    assert "zscan" in fake2.calls

    # an in-progress marker short-circuits before the queue scan
    fake3 = _FakeRedis(inprog=["arq:in-progress:chat:c9:abc"])
    assert await _chat_turn_alive(fake3, "c9") is True
    assert fake3.calls == ["scan"]

    # no redis == unknown == alive (never double-fire)
    assert await _chat_turn_alive(None, "c1") is True


# --------------------------------------------------------------------------- #
# task 12 — spawn builder preserves job envelopes                               #
# --------------------------------------------------------------------------- #


async def test_spawn_params_via_builder() -> None:
    from cadx.db.models import JobType, Version, VersionKind
    from cadx.spawn import spawn_params

    pid, part = uuid.uuid4(), uuid.uuid4()
    async with session_scope() as db:
        parent = Version(
            project_id=pid,
            part_id=part,
            number=1,
            kind=VersionKind.generate,
            code="result = None\n",
            params={"length": 10.0},
        )
        db.add(parent)
        await db.commit()
        parent_id = parent.id

    async with session_scope() as db:
        parent = await db.get(Version, parent_id)
        v, job = await spawn_params(db, parent, {"length": 42.0})
        assert v.parent_id == parent_id and v.kind == VersionKind.params
        assert v.params == {"length": 42.0}
        assert job.type == JobType.params
        # the arq envelope the sweeper replays must carry func + full args
        assert job.payload is not None
        env = job.payload
        assert env["func"] == "run_params_job"
        assert env["args"]["params"] == {"length": 42.0}
        assert env["args"]["job_id"] == str(job.id)


async def test_spawn_optimize_via_builder() -> None:
    from cadx.db.models import JobType, Version, VersionKind, VersionStatus
    from cadx.spawn import spawn_optimize

    pid, part = uuid.uuid4(), uuid.uuid4()
    async with session_scope() as db:
        parent = Version(
            project_id=pid,
            part_id=part,
            number=1,
            kind=VersionKind.generate,
            code="x",
            prompt="bracket",
            status=VersionStatus.succeeded,
        )
        db.add(parent)
        await db.commit()
        parent_id = parent.id

    async with session_scope() as db:
        parent = await db.get(Version, parent_id)
        v, job = await spawn_optimize(
            db, parent, load_cases=[{"n": 1}], optimize={"volfrac": 0.3}, material="pla"
        )
        assert job.type == JobType.optimize
        env = job.payload
        assert env["func"] == "run_optimize_job"
        assert env["args"]["load_cases"] == [{"n": 1}]
        assert env["args"]["optimize"] == {"volfrac": 0.3}
        # request is recorded on the child version for restarts
        assert v.options["optimize_request"]["material"] == "pla"


# --------------------------------------------------------------------------- #
# task 10 — declared capability flags (no signature probing)                    #
# --------------------------------------------------------------------------- #


def test_capability_flags_declared() -> None:
    from cadx.agent import orchestrator
    from cadx.agent.steps import critique
    from cadx.cad.fem.analyze import fem_feedback_lines

    assert critique.ACCEPTS_PREV_ISSUES is True
    assert fem_feedback_lines.ACCEPTS_LIGHTWEIGHT_REQUESTED is True
    assert orchestrator._CRITIQUE_PREV_ISSUES is True
    assert orchestrator._FEM_LIGHTWEIGHT_KW is True


# --------------------------------------------------------------------------- #
# task 11 — executor_client lazy cad imports                                   #
# --------------------------------------------------------------------------- #


def test_executor_client_does_not_bind_cad_tasks() -> None:
    import cadx.agent.executor_client as ec

    # the module must not bind cad.tasks symbols at import time — API and
    # worker processes stay OCC-free; InProcessExecutor imports lazily
    assert "cadx" not in str(vars(ec).get("tasks", ""))
    assert not any(k.startswith("run_") and k.endswith("_payload") for k in vars(ec))


# --------------------------------------------------------------------------- #
# task 15 — bounded upload reads                                                #
# --------------------------------------------------------------------------- #


def _upload(data: bytes, *, filename: str = "f.step", content_length: str | None = None):
    headers = (
        Headers({"content-length": content_length})
        if content_length is not None
        else Headers({})
    )
    return UploadFile(file=io.BytesIO(data), filename=filename, headers=headers)


async def test_read_capped_bounds_body() -> None:
    from fastapi import HTTPException

    from cadx.api.routers.assets import _read_capped

    # under the cap -> full body
    assert await _read_capped(_upload(b"0123456789"), 10) == b"0123456789"

    # over the cap without a length header -> bounded read still rejects
    with pytest.raises(HTTPException) as ei:
        await _read_capped(_upload(b"01234567890"), 10)
    assert ei.value.status_code == 413

    # a lying declared length that exceeds the cap -> early 413
    with pytest.raises(HTTPException) as ei:
        await _read_capped(_upload(b"01234567890", content_length="99999999"), 10)
    assert ei.value.status_code == 413

    # a non-numeric declared length falls through to the authoritative read
    with pytest.raises(HTTPException) as ei:
        await _read_capped(_upload(b"01234567890", content_length="bogus"), 10)
    assert ei.value.status_code == 413
    # but a small body with a broken header is fine
    assert await _read_capped(_upload(b"tiny", content_length="bogus"), 10) == b"tiny"


async def test_library_read_capped_same_contract() -> None:
    from fastapi import HTTPException

    from cadx.api.routers.library import MAX_LIBRARY_BYTES, _read_capped

    assert MAX_LIBRARY_BYTES == 50 * 1024 * 1024
    with pytest.raises(HTTPException) as ei:
        await _read_capped(_upload(b"x" * 11, content_length=str(11)), 10)
    assert ei.value.status_code == 413
    assert await _read_capped(_upload(b"abc"), 10) == b"abc"


# --------------------------------------------------------------------------- #
# task 16 — service allowlist                                                   #
# --------------------------------------------------------------------------- #


async def test_stack_restart_allowlist() -> None:
    from fastapi import HTTPException

    from cadx.api.routers.system import _RESTARTABLE_SERVICES, stack_restart

    assert {"api", "worker", "cad-executor", "frontend", "mesh-server"} == _RESTARTABLE_SERVICES

    # a compose service that is NOT supervised must not reach the daemon URL
    with pytest.raises(HTTPException) as ei:
        await stack_restart("postgres")
    assert ei.value.status_code == 400

    with pytest.raises(HTTPException) as ei:
        await stack_restart("../../etc")
    assert ei.value.status_code == 400

    # a supervised service passes validation and fails at the daemon call
    # (no updater running in tests -> 503, never a 400)
    with pytest.raises(HTTPException) as ei:
        await stack_restart("api")
    assert ei.value.status_code == 503
