"""MCP server tools: catalog shape, LLM-free guarantee, handler guards and an
in-process client round-trip.

SA-2 contract under test (`cadx/mcpserver/session.py` + `server.py`):
  * `McpToolSession(ProjectToolSession)` — `ctx.llm is None`, `ctx.executor` is
    a QueueExecutor, `enqueue_fn=cadx.api.deps.enqueue`, `execute()` checks
    `MCP_HANDLERS` before inherited `_tool_*` methods.
  * `server.py` maps the session's tool schemas onto MCP tools and dict
    results/`_images` onto MCP content.

The handler-level tests use `_StubSession`, a minimal stand-in implementing
the session contract (`ctx`, `_pid`, `spawned`, `tool_log`, `_project`,
`_version`, `_spawn_and_wait`, `execute`). The server-level tests exercise
the real `McpToolSession` + module `server` over MCP memory streams.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from cadx import storage
from cadx.db.models import (
    Asset,
    AssetKind,
    Job,
    JobStatus,
    JobType,
    Project,
    ProjectSettings,
    Version,
    VersionKind,
    VersionStatus,
)
from cadx.db.session import create_all, session_scope
from cadx.mcpserver.tools import MCP_HANDLERS, MCP_TOOL_NAMES, MCP_TOOL_SCHEMAS
from cadx.runtime_config import RuntimeSettings
from cadx.spawn import SpawnError

# tools that must NEVER be exposed over MCP — every one of them drives the
# WebGUI's LLM (or destroys data); the MCP surface is strictly LLM-free
_FORBIDDEN_TOOLS = {
    "generate_part",
    "edit_version",
    "answer_questions",
    "enqueue_resume",
    "run_agent_job",
}

_TRIVIAL_CODE = (
    "from build123d import *\n"
    "PARAMS = {'length': 40.0, 'width': 20.0, 'height': 10.0}\n"
    "with BuildPart() as p:\n"
    "    Box(PARAMS['length'], PARAMS['width'], PARAMS['height'])\n"
    "result = p.part\n"
)


@pytest.fixture(scope="module", autouse=True)
async def _tables():
    await create_all()


# --------------------------------------------------------------------------- #
# stub session — implements the frozen ProjectToolSession/McpToolSession       #
# contract so the handlers can be exercised without SA-2's classes             #
# --------------------------------------------------------------------------- #


class _FakeExecutor:
    """Captures executor payloads and returns a canned result."""

    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []
        self.audited: list[dict[str, Any]] = []
        self.parametrized: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {"ok": True, "art_dir": "scratch"}

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.executed.append(payload)
        return self.result

    async def audit_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.audited.append(payload)
        return {"ok": True, "status": "robust", "params": {}}

    async def parametrize_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.parametrized.append(payload)
        return {"ok": True, "kind": "revolve", "code": "# recon", "chamfer_mm": 0.4}


class _StubCtx:
    """Minimal AgentCtx stand-in — LLM-free by construction."""

    def __init__(self, executor: Any = None) -> None:
        self.llm = None
        self.executor = executor if executor is not None else _FakeExecutor()
        self.settings = ProjectSettings()
        self.runtime_settings = RuntimeSettings()
        self.project_id: uuid.UUID | None = None
        self.version_id = uuid.uuid4()
        self.assets: list[Any] = []
        self.image = None
        self.image_available = False


class _StubSession:
    """Session contract per SA-2: `ctx`, `_pid`, `spawned`, `tool_log`,
    `_project`, `_version`, `_spawn_and_wait`, `execute`, `tools`,
    `allowed_names` — with MCP_HANDLERS-first dispatch."""

    def __init__(self, pid: uuid.UUID | None = None, executor: Any = None) -> None:
        self.ctx = _StubCtx(executor)
        self._pid = pid
        self.ctx.project_id = pid
        self.spawned: list[dict[str, Any]] = []
        self.tool_log: list[dict[str, Any]] = []
        self._enqueue_fn = None
        self.image_left = 0

    async def _project(self, db: Any) -> Project:
        if self._pid is None:
            raise SpawnError("no project bound — call use_project first", status=409)
        p = await db.get(Project, self._pid)
        if p is None:
            raise SpawnError("project not found", status=404)
        return p

    async def _version(self, db: Any, version_id: str) -> Version:
        await self._project(db)
        try:
            vid = uuid.UUID(str(version_id))
        except ValueError:
            raise SpawnError("invalid version id", status=422) from None
        v = await db.get(Version, vid)
        if v is None or v.project_id != self._pid:
            raise SpawnError("version not found in this project", status=404)
        return v

    async def _spawn_and_wait(self, name: str, spawn: Any, wait: bool = True) -> dict[str, Any]:
        try:
            async with session_scope() as db:
                out = await spawn(db)
        except SpawnError as e:
            return {"ok": False, "error": str(e)}
        job = out[1] if isinstance(out, tuple) else out
        version_id = out[0].id if isinstance(out, tuple) else getattr(job, "version_id", None)
        self.spawned.append({"tool": name, "job_id": str(job.id), "version_id": str(version_id)})
        return {
            "ok": True,
            "status": "queued",
            "job_id": str(job.id),
            "version_id": str(version_id),
        }

    async def _tool_attach_project(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            pid = uuid.UUID(str(args.get("project_id", "")))
        except ValueError:
            return {"ok": False, "error": "invalid project id"}
        async with session_scope() as db:
            p = await db.get(Project, pid)
            if p is None:
                return {"ok": False, "error": "project not found"}
            self._pid = pid
            self.ctx.project_id = pid
            return {"ok": True, "project_id": str(pid), "project_name": p.name, "bound": True}

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = MCP_HANDLERS.get(name)
        if handler is None:
            return {"ok": False, "error": f"unknown tool '{name}'"}
        try:
            return await handler(self, args)
        except SpawnError as e:
            return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------------- #
# seeding helpers                                                              #
# --------------------------------------------------------------------------- #


async def _mk_project(name: str = "mcp-proj") -> uuid.UUID:
    async with session_scope() as db:
        p = Project(name=name)
        db.add(p)
        await db.commit()
        return p.id


async def _mk_version(
    pid: uuid.UUID,
    *,
    kind: VersionKind = VersionKind.generate,
    status: VersionStatus = VersionStatus.succeeded,
    **kw: Any,
) -> Version:
    async with session_scope() as db:
        v = Version(project_id=pid, number=1, kind=kind, status=status, **kw)
        db.add(v)
        await db.commit()
        await db.refresh(v)
        return v


async def _mk_job(
    v: Version, *, status: JobStatus = JobStatus.queued, jtype: JobType = JobType.params
) -> Job:
    async with session_scope() as db:
        j = Job(version_id=v.id, type=jtype, status=status)
        db.add(j)
        await db.commit()
        await db.refresh(j)
        return j


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# catalog shape                                                                #
# --------------------------------------------------------------------------- #


def test_catalog_shape_and_symmetry():
    assert len(MCP_TOOL_SCHEMAS) >= 20
    names = set()
    for spec in MCP_TOOL_SCHEMAS:
        fn = spec["function"]
        assert spec["type"] == "function"
        assert fn["name"] and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert set(params["required"]) <= set(params["properties"])
        names.add(fn["name"])
    # every declared schema has a handler; extra handlers are shadow impls
    # for inherited session tools whose schemas come from ProjectToolSession
    assert names == MCP_TOOL_NAMES
    assert set(MCP_HANDLERS) >= MCP_TOOL_NAMES
    assert set(MCP_HANDLERS) - MCP_TOOL_NAMES == {"view_image", "generate_image"}


def test_catalog_excludes_llm_and_delete_tools():
    assert MCP_TOOL_NAMES.isdisjoint(_FORBIDDEN_TOOLS)
    assert not any(n.startswith("delete_") for n in MCP_TOOL_NAMES)
    # LLM-free code loop is the core of the MCP surface
    assert {"codegen_contract", "check_code", "execute_cad", "submit_code"} <= MCP_TOOL_NAMES


def test_handlers_are_async():
    import inspect

    for name, fn in MCP_HANDLERS.items():
        assert inspect.iscoroutinefunction(fn), f"{name} handler must be async"


# --------------------------------------------------------------------------- #
# docs + static checks                                                         #
# --------------------------------------------------------------------------- #


async def test_codegen_contract_returns_docs():
    s = _StubSession()
    res = await s.execute("codegen_contract", {})
    assert res["ok"] and res["doc"] == "contract"
    assert "PARAMS" in res["text"] and "result" in res["text"]
    for doc in ("build123d", "cadlib", "bd_warehouse", "examples", "dfm"):
        r = await s.execute("codegen_contract", {"doc": doc})
        assert r["ok"], doc
        assert r["text"].strip(), doc


async def test_codegen_contract_unknown_doc():
    res = await _StubSession().execute("codegen_contract", {"doc": "nope"})
    assert res["ok"] is False
    assert "contract" in res["error"]


async def test_check_code_valid():
    res = await _StubSession().execute("check_code", {"code": _TRIVIAL_CODE})
    assert res["ok"], res
    assert res["errors"] == []
    assert res["params"] == {"length": 40.0, "width": 20.0, "height": 10.0}
    assert res["standard_part_errors"] == []


async def test_check_code_reports_contract_errors():
    res = await _StubSession().execute("check_code", {"code": "import os\nx = 1\n"})
    assert res["ok"] is False
    assert any("result" in e or "PARAMS" in e or "import" in e for e in res["errors"])


async def test_check_code_requires_code():
    res = await _StubSession().execute("check_code", {})
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# execute / audit payloads                                                     #
# --------------------------------------------------------------------------- #


async def test_execute_cad_requires_code_and_executor():
    s = _StubSession()
    assert (await s.execute("execute_cad", {}))["ok"] is False
    s.ctx.executor = None
    assert (await s.execute("execute_cad", {"code": _TRIVIAL_CODE}))["ok"] is False


async def test_execute_cad_scratch_payload():
    """Unbound session → zero-uuid project + fresh version id; the code runs
    in the sandbox with no persistence."""
    ex = _FakeExecutor()
    s = _StubSession(executor=ex)
    res = await s.execute(
        "execute_cad", {"code": _TRIVIAL_CODE, "params": {"length": 55.0}, "light": True}
    )
    assert res == {"ok": True, "art_dir": "scratch"}
    (p,) = ex.executed
    assert p["project_id"] == "00000000-0000-0000-0000-000000000000"
    uuid.UUID(p["version_id"])  # parseable fresh id
    uuid.UUID(p["job_id"])
    assert p["code"] == _TRIVIAL_CODE
    assert p["params_override"] == {"length": 55.0}
    assert p["references"] == []
    assert p["light"] is True
    assert p["iteration"] == 1


async def test_execute_cad_bound_version_and_hybrid_dirs():
    pid = await _mk_project()
    v = await _mk_version(
        pid,
        kind=VersionKind.params,
        resolved_mode="hybrid",
        mesh={"bbox_size": [40.0, 40.0, 40.0]},
    )
    solid = storage.version_dir(pid, v.id) / "mesh" / "solid.brep"
    solid.parent.mkdir(parents=True, exist_ok=True)
    solid.write_bytes(b"brep")

    ex = _FakeExecutor()
    s = _StubSession(pid=pid, executor=ex)
    res = await s.execute("execute_cad", {"code": _TRIVIAL_CODE, "version_id": str(v.id)})
    assert res["ok"]
    (p,) = ex.executed
    assert p["project_id"] == str(pid)
    assert p["version_id"] == str(v.id)
    assert p["mesh_solid"] == str(solid)
    assert p["mesh_bbox"] == [40.0, 40.0, 40.0]


async def test_execute_cad_version_outside_project_rejected():
    pid = await _mk_project()
    other = await _mk_version(await _mk_project("other"))
    res = await _StubSession(pid=pid).execute(
        "execute_cad", {"code": _TRIVIAL_CODE, "version_id": str(other.id)}
    )
    assert res["ok"] is False


async def test_audit_params_merges_declared_and_overrides():
    ex = _FakeExecutor()
    s = _StubSession(executor=ex)
    res = await s.execute("audit_params", {"code": _TRIVIAL_CODE, "params": {"length": 42.0}})
    assert res["ok"]
    (p,) = ex.audited
    assert p["params"] == {"length": 42.0, "width": 20.0, "height": 10.0}
    assert p["code"] == _TRIVIAL_CODE


# --------------------------------------------------------------------------- #
# spawn tools (exercise the real SA-1 spawn_* functions end-to-end)            #
# --------------------------------------------------------------------------- #


async def test_submit_code_unbound_is_error():
    res = await _StubSession().execute("submit_code", {"code": _TRIVIAL_CODE})
    assert res["ok"] is False
    assert "use_project" in res["error"] or "bound" in res["error"]


async def test_submit_code_spawns_version_and_job():
    """Real SA-1 spawn_code: contract-checked, persisted as kind=code,
    enqueued via deps.enqueue (which degrades without redis)."""
    pid = await _mk_project()
    parent = await _mk_version(pid)
    s = _StubSession(pid=pid)
    res = await s.execute(
        "submit_code",
        {
            "code": _TRIVIAL_CODE,
            "params": {"length": 60.0},
            "parent_version_id": str(parent.id),
            "prompt": "mcp submit",
        },
    )
    assert res["ok"], res
    assert res["status"] == "queued"
    assert s.spawned[0]["tool"] == "submit_code"
    async with session_scope() as db:
        v = await db.get(Version, uuid.UUID(res["version_id"]))
        j = await db.get(Job, uuid.UUID(res["job_id"]))
        assert v.kind == VersionKind.code and v.parent_id == parent.id
        assert v.code == _TRIVIAL_CODE
        assert v.params == {"length": 60.0}
        assert v.prompt == "mcp submit"
        assert j.type == JobType.code


async def test_submit_code_rejects_bad_code():
    """spawn_code contract-checks before persisting — invalid code fails."""
    pid = await _mk_project()
    res = await _StubSession(pid=pid).execute("submit_code", {"code": "import os\nresult = 1\n"})
    assert res["ok"] is False


async def test_mesh_generate_spawns():
    """Real SA-1 spawn_meshgen: creates a kind=meshgen version + JobType.meshgen
    job and stores the generation inputs on version.options for restart."""
    pid = await _mk_project()
    s = _StubSession(pid=pid)
    res = await s.execute(
        "mesh_generate",
        {"asset_ids": ["a1", "a2"], "subject": "knob", "seed": 7, "post": {"scale": 1.5}},
    )
    assert res["ok"], res
    async with session_scope() as db:
        v = await db.get(Version, uuid.UUID(res["version_id"]))
        j = await db.get(Job, uuid.UUID(res["job_id"]))
        assert v.kind == VersionKind.meshgen and j.type == JobType.meshgen
        assert v.asset_ids == ["a1", "a2"]
        opts = v.options or {}
        assert opts["subject"] == "knob" and opts["seed"] == 7
        assert opts["post"] == {"scale": 1.5}


async def test_mesh_generate_bad_post_rejected():
    pid = await _mk_project()
    res = await _StubSession(pid=pid).execute(
        "mesh_generate", {"asset_ids": ["a1"], "post": {"scale": "not-a-number"}}
    )
    assert res["ok"] is False


async def test_mesh_generate_requires_assets():
    res = await _StubSession().execute("mesh_generate", {"asset_ids": []})
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# mesh ops                                                                     #
# --------------------------------------------------------------------------- #


async def test_mesh_preview_requires_ops_and_mesh():
    pid = await _mk_project()
    v = await _mk_version(pid)
    s = _StubSession(pid=pid)
    assert (await s.execute("mesh_preview", {"version_id": str(v.id)}))["ok"] is False
    res = await s.execute(
        "mesh_preview", {"version_id": str(v.id), "ops": [{"op": "scale", "args": {}}]}
    )
    assert res["ok"] is False
    assert "no mesh" in res["error"]


async def test_mesh_preview_unknown_op():
    pid = await _mk_project()
    v = await _mk_version(pid)
    res = await _StubSession(pid=pid).execute(
        "mesh_preview", {"version_id": str(v.id), "ops": [{"op": "explode", "args": {}}]}
    )
    assert res["ok"] is False
    assert "unknown mesh op" in res["error"]


async def test_mesh_preview_applies_ops(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pid = await _mk_project()
    v = await _mk_version(pid)
    out_dir = storage.version_dir(pid, v.id) / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(10, 20, 30)).export(str(out_dir / "model.stl"))

    res = await _StubSession(pid=pid).execute(
        "mesh_preview",
        {"version_id": str(v.id), "ops": [{"op": "scale", "args": {"factor": 2.0}}]},
    )
    assert res["ok"], res
    assert res["faces"] > 0
    assert res["bbox_size"] == [20.0, 40.0, 60.0]
    assert res["ops_report"][0]["op"] == "scale"


async def test_editable_features_no_brep():
    pid = await _mk_project()
    v = await _mk_version(pid)
    res = await _StubSession(pid=pid).execute("editable_features", {"version_id": str(v.id)})
    assert res["ok"]
    assert res["features"] == []
    assert "reason" in res


async def test_edit_feature_requires_diameter():
    pid = await _mk_project()
    v = await _mk_version(pid)
    s = _StubSession(pid=pid)
    res = await s.execute("edit_feature", {"version_id": str(v.id), "feature_id": 1})
    assert res["ok"] is False
    assert "diameter" in res["error"]


async def test_edit_feature_spawns_child():
    """Real spawn_edit_feature path: a succeeded version with a solid.brep
    accepts the feature edit and enqueues the executor job."""
    pid = await _mk_project()
    v = await _mk_version(pid)
    solid = storage.version_dir(pid, v.id) / "mesh" / "solid.brep"
    solid.parent.mkdir(parents=True, exist_ok=True)
    solid.write_bytes(b"brep")  # existence is enough — the executor parses it
    res = await _StubSession(pid=pid).execute(
        "edit_feature",
        {"version_id": str(v.id), "feature_id": 0, "params": {"diameter": 8.0}},
    )
    assert res["ok"], res
    assert res["status"] == "queued"
    assert res["version_id"] != str(v.id)  # child version spawned


async def test_parametrize_mesh_no_artifact():
    pid = await _mk_project()
    v = await _mk_version(pid)
    res = await _StubSession(pid=pid).execute("parametrize_mesh", {"version_id": str(v.id)})
    assert res["ok"] is False
    assert "no mesh" in res["error"]


async def test_parametrize_mesh_dispatches():
    trimesh = pytest.importorskip("trimesh")
    pid = await _mk_project()
    v = await _mk_version(pid, kind=VersionKind.generate, iterations=1)
    mesh_dir = storage.version_dir(pid, v.id) / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(10, 10, 10)).export(str(mesh_dir / "mesh_raw.glb"))

    ex = _FakeExecutor()
    res = await _StubSession(pid=pid, executor=ex).execute(
        "parametrize_mesh", {"version_id": str(v.id)}
    )
    assert res["ok"], res
    (p,) = ex.parametrized
    assert p["version_id"] == str(v.id)
    assert p["mesh_rel"] and p["mesh_rel"].endswith("mesh_raw.glb")
    assert p["subdir"] == "parametrize"


# --------------------------------------------------------------------------- #
# jobs                                                                         #
# --------------------------------------------------------------------------- #


async def test_list_jobs_project_scoped():
    pid = await _mk_project()
    v = await _mk_version(pid)
    j = await _mk_job(v, status=JobStatus.running)
    # noise in another project must not leak in
    other_v = await _mk_version(await _mk_project("other"))
    await _mk_job(other_v)

    s = _StubSession(pid=pid)
    res = await s.execute("list_jobs", {})
    assert res["ok"]
    assert [x["id"] for x in res["jobs"]] == [str(j.id)]
    assert res["jobs"][0]["version_number"] == 1
    assert res["jobs"][0]["status"] == "running"


async def test_list_jobs_unbound():
    res = await _StubSession().execute("list_jobs", {})
    assert res["ok"] is False


async def test_cancel_job():
    pid = await _mk_project()
    v = await _mk_version(pid, status=VersionStatus.running)
    j = await _mk_job(v, status=JobStatus.queued)
    res = await _StubSession(pid=pid).execute("cancel_job", {"job_id": str(j.id)})
    assert res["ok"], res
    async with session_scope() as db:
        jj = await db.get(Job, j.id)
        vv = await db.get(Version, v.id)
        assert jj.status == JobStatus.cancelled and jj.cancel_requested
        assert vv.status == VersionStatus.cancelled


async def test_cancel_job_terminal_and_cross_project():
    pid = await _mk_project()
    v = await _mk_version(pid)
    done = await _mk_job(v, status=JobStatus.failed)
    s = _StubSession(pid=pid)
    assert (await s.execute("cancel_job", {"job_id": str(done.id)}))["ok"] is False
    foreign = await _mk_job(await _mk_version(await _mk_project("other")))
    assert (await s.execute("cancel_job", {"job_id": str(foreign.id)}))["ok"] is False


async def test_finish_job():
    pid = await _mk_project()
    v = await _mk_version(pid)
    running = await _mk_job(v, status=JobStatus.running)
    queued = await _mk_job(v)
    s = _StubSession(pid=pid)
    res = await s.execute("finish_job", {"job_id": str(running.id)})
    assert res["ok"]
    async with session_scope() as db:
        assert (await db.get(Job, running.id)).finish_requested
    assert (await s.execute("finish_job", {"job_id": str(queued.id)}))["ok"] is False


async def test_restart_job_terminal_only():
    pid = await _mk_project()
    parent = await _mk_version(pid, kind=VersionKind.generate)
    v = await _mk_version(pid, kind=VersionKind.params, parent_id=parent.id)
    running = await _mk_job(v, status=JobStatus.running)
    s = _StubSession(pid=pid)
    assert (await s.execute("restart_job", {"job_id": str(running.id)}))["ok"] is False

    # real SA-1 spawn_restart on a deterministic (params) version → sibling
    done = await _mk_job(v, status=JobStatus.failed)
    res = await s.execute("restart_job", {"job_id": str(done.id)})
    assert res["ok"], res
    assert res["status"] == "queued"
    async with session_scope() as db:
        new_v = await db.get(Version, uuid.UUID(res["version_id"]))
        new_j = await db.get(Job, uuid.UUID(res["job_id"]))
        assert new_v.kind == VersionKind.params and new_v.id != v.id
        assert new_j.type == JobType.params


async def test_restart_job_refuses_llm_versions():
    """generate/edit/resume jobs re-run via run_agent_job — the LLM. The MCP
    surface must refuse those job types; deterministic jobs on LLM-kind
    versions (fem, params, …) replay without an LLM and are allowed."""
    pid = await _mk_project()
    v = await _mk_version(pid, kind=VersionKind.generate)
    done = await _mk_job(v, status=JobStatus.failed, jtype=JobType.generate)
    res = await _StubSession(pid=pid).execute("restart_job", {"job_id": str(done.id)})
    assert res["ok"] is False
    assert "deterministic" in res["error"] or "LLM" in res["error"]

    # a deterministic job on the same LLM-kind version is allowed through
    det = await _mk_job(v, status=JobStatus.failed, jtype=JobType.params)
    res2 = await _StubSession(pid=pid).execute("restart_job", {"job_id": str(det.id)})
    assert res2["ok"], res2


async def test_task_status_survives_missing_redis():
    """No redis in unit tests — the handler must degrade to ok:false, not raise."""
    res = await asyncio.wait_for(
        _StubSession().execute("task_status", {"task_id": "execute_version-abc"}),
        timeout=15,
    )
    assert res["ok"] is False
    assert (await _StubSession().execute("task_status", {}))["ok"] is False


# --------------------------------------------------------------------------- #
# assets / images                                                              #
# --------------------------------------------------------------------------- #


async def test_upload_asset_guards():
    s = _StubSession()
    assert (await s.execute("upload_asset", {}))["ok"] is False
    assert (
        await s.execute(
            "upload_asset", {"path": "/x", "data_base64": base64.b64encode(b"a").decode()}
        )
    )["ok"] is False
    res = await s.execute("upload_asset", {"data_base64": "!!!not-b64!!!"})
    assert res["ok"] is False and "base64" in res["error"]
    res = await s.execute("upload_asset", {"path": "/definitely/missing.png"})
    assert res["ok"] is False and "not a readable file" in res["error"]
    # unbound session rejects even valid payloads
    res = await s.execute("upload_asset", {"data_base64": base64.b64encode(b"x").decode()})
    assert res["ok"] is False
    assert "bound" in res["error"]


async def test_upload_asset_image_and_view(tmp_path):
    pid = await _mk_project()
    s = _StubSession(pid=pid)
    res = await s.execute(
        "upload_asset",
        {"data_base64": base64.b64encode(_png_bytes()).decode(), "name": "ref.png"},
    )
    assert res["ok"], res
    asset = res["asset"]
    assert asset["filename"].endswith(".jpg")  # normalised
    assert asset["width"] == 8 and asset["height"] == 8
    assert storage.resolve((await _asset_row(asset["id"])).path).exists()
    # ctx.assets was refreshed so downstream tools see the upload immediately
    assert any(a.asset_id == asset["id"] for a in s.ctx.assets)

    view = await s.execute("view_image", {"asset_id": asset["id"]})
    assert view["ok"] and view["_images"]
    assert await asyncio.to_thread(lambda: Path(view["_images"][0]).exists())


async def test_upload_asset_from_path(tmp_path):
    pid = await _mk_project()
    f = tmp_path / "plate.step"
    f.write_text("ISO-10303-21;\nEND-ISO-10303-21;", encoding="utf-8")
    s = _StubSession(pid=pid)
    res = await s.execute(
        "upload_asset", {"path": str(f), "kind": "reference_step", "caption": "plate"}
    )
    assert res["ok"], res
    assert res["asset"]["kind"] == "reference_step"
    listed = await s.execute("list_assets", {"kind": "reference_step"})
    assert [a["id"] for a in listed["assets"]] == [res["asset_id"]]


async def test_list_assets_kind_validation():
    pid = await _mk_project()
    s = _StubSession(pid=pid)
    assert (await s.execute("list_assets", {"kind": "bogus"}))["ok"] is False
    assert (await s.execute("list_assets", {}))["assets"] == []


async def test_view_image_guards():
    s = _StubSession()
    assert (await s.execute("view_image", {}))["ok"] is False
    pid = await _mk_project()
    s2 = _StubSession(pid=pid)
    assert (await s2.execute("view_image", {"asset_id": str(uuid.uuid4())}))["ok"] is False


async def _asset_row(asset_id: str) -> Asset:
    async with session_scope() as db:
        return await db.get(Asset, uuid.UUID(asset_id))


async def test_remove_bg_rejects_non_image():
    pid = await _mk_project()
    async with session_scope() as db:
        a = Asset(
            project_id=pid,
            kind=AssetKind.reference_step,
            filename="x.step",
            path="projects/x.step",
        )
        db.add(a)
        await db.commit()
        aid = a.id
    res = await _StubSession(pid=pid).execute("remove_bg", {"asset_id": str(aid)})
    assert res["ok"] is False
    assert "not an image" in res["error"]


async def test_generate_image_no_adapter():
    """No image sidecar + no inherited session tool → clean ok:false."""
    res = await _StubSession().execute("generate_image", {"prompt": "a knob"})
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# system                                                                       #
# --------------------------------------------------------------------------- #


async def test_settings_get_contains_no_secrets():
    res = await _StubSession().execute("settings_get", {})
    assert res["ok"]
    settings = res["settings"]
    # NEVER the key material itself — only the presence flags
    presence_flags = {"llm_api_key_set", "tripo_api_key_set", "image_api_key_set"}
    assert "llm_api_key" not in settings
    assert "llm_api_key_set" in settings
    forbidden = [
        k
        for k in settings
        if any(t in k.lower() for t in ("secret", "password", "api_key", "auth"))
        and k not in presence_flags
    ]
    assert forbidden == [], forbidden
    json.dumps(settings)  # everything returned must be JSON-serialisable


async def test_stats_counts():
    pid = await _mk_project()
    v = await _mk_version(pid)
    await _mk_job(v)
    res = await _StubSession(pid=pid).execute("stats", {})
    assert res["ok"]
    assert res["totals"]["projects"] >= 1
    assert res["totals"]["versions"] >= 1
    assert res["totals"]["jobs"] >= 1
    assert res["project"]["project_id"] == str(pid)
    assert res["project"]["versions"] >= 1


async def test_mesh_health_without_sidecar():
    """No mesh server configured in tests → structured failure, not a raise."""
    res = await _StubSession().execute("mesh_health", {})
    assert "ok" in res  # ok False (not configured) or a probe result


async def test_image_models():
    res = await _StubSession().execute("image_models", {})
    assert res["ok"]
    assert res["models"] == []


# --------------------------------------------------------------------------- #
# use_project alias                                                            #
# --------------------------------------------------------------------------- #


async def test_use_project_binds_via_attach():
    pid = await _mk_project("bind-me")
    s = _StubSession()
    res = await s.execute("use_project", {"project_id": str(pid)})
    assert res["ok"] and res["bound"]
    assert s._pid == pid
    # bound now — project tools work
    assert (await s.execute("list_jobs", {}))["ok"]
    bad = await s.execute("use_project", {"project_id": str(uuid.uuid4())})
    assert bad["ok"] is False


# --------------------------------------------------------------------------- #
# SA-2 integration — server + real session (skipped until the modules land)    #
# --------------------------------------------------------------------------- #


def _build_server() -> Any:
    """Construct the MCP server object SA-2 ships in `cadx/mcpserver/server.py`.

    The frozen contract says the module builds a `mcp.server.lowlevel.Server`
    calling `session.execute(name, args)`; the factory/attribute name is an
    integration guess — adjust here if SA-2 names it differently.
    """
    mod = pytest.importorskip("cadx.mcpserver.server")
    for name in ("build_server", "create_server", "make_server"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn()
    for name in ("server", "mcp_server", "mcp"):
        srv = getattr(mod, name, None)
        if srv is not None:
            return srv
    pytest.fail("cadx.mcpserver.server exposes no Server factory/instance")


def _make_mcp_session() -> Any:
    """Construct SA-2's `McpToolSession` (contract: ctx.llm=None, QueueExecutor)."""
    mod = pytest.importorskip("cadx.mcpserver.session")
    return mod.McpToolSession()


@asynccontextmanager
async def _connected_client(server: Any) -> AsyncIterator[Any]:
    """In-process MCP client wired to the server over memory streams."""
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    # fresh session pool per test — the pool is process-global and a bound
    # project would otherwise leak across tests
    from cadx.mcpserver.session import SESSIONS

    SESSIONS.clear()

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        task = asyncio.create_task(
            server.run(
                server_streams[0],
                server_streams[1],
                server.create_initialization_options(),
            )
        )
        try:
            async with ClientSession(client_streams[0], client_streams[1]) as client:
                await client.initialize()
                yield client
        finally:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task


def test_server_object_constructs():
    server = _build_server()
    from mcp.server.lowlevel import Server

    assert isinstance(server, Server)


async def test_session_is_llm_free(monkeypatch):
    """The MCP session must never construct or call an LLM."""
    import cadx.llm.client

    def _boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("LLMClient constructed inside the MCP session")

    monkeypatch.setattr(cadx.llm.client, "LLMClient", _boom)
    session = _make_mcp_session()
    assert session.ctx.llm is None
    assert getattr(session.ctx, "executor", None) is not None


async def test_tools_list_exposes_catalog():
    async with _connected_client(_build_server()) as client:
        tools = (await client.list_tools()).tools
        names = {t.name for t in tools}
        # tool names on the wire must be unique (no schema duplication)
        assert len(tools) == len(names)
        # new deterministic tools are exposed
        assert names >= MCP_TOOL_NAMES
        # LLM-touching chat tools and delete_* stay OFF the wire
        assert names.isdisjoint(_FORBIDDEN_TOOLS)
        assert not any(n.startswith("delete_") for n in names)


async def test_call_tool_text_and_image_content():
    """dict results -> TextContent JSON; _images -> ImageContent."""
    pid = await _mk_project("mcp-e2e")
    # seed an image asset directly (upload itself is covered above)
    png = _png_bytes()
    async with session_scope() as db:
        a = Asset(
            project_id=pid, kind=AssetKind.image, filename="ref.png", path="", mime="image/png"
        )
        path = storage.asset_path(pid, a.id, a.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        a.path = storage.relpath(path)
        db.add(a)
        await db.commit()
        aid = str(a.id)

    import mcp.types as mt

    async with _connected_client(_build_server()) as client:
        # codegen_contract over the wire — deterministic text content
        res = await client.call_tool("codegen_contract", {"doc": "contract"})
        assert not res.is_error
        text = next(c for c in res.content if isinstance(c, mt.TextContent))
        payload = json.loads(text.text)
        assert payload["ok"] and "PARAMS" in payload["text"]

        # use_project -> list_versions flow on the seeded project
        res = await client.call_tool("use_project", {"project_id": str(pid)})
        assert not res.is_error
        res = await client.call_tool("list_versions", {})
        assert not res.is_error
        payload = json.loads(next(c for c in res.content if isinstance(c, mt.TextContent)).text)
        assert payload["ok"]

        # view_image emits image content
        res = await client.call_tool("view_image", {"asset_id": aid})
        assert not res.is_error
        assert any(isinstance(c, mt.ImageContent) for c in res.content)
