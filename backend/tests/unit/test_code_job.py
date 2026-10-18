"""MCP LLM-free jobs: `code` (persist+execute external build123d source) and
`meshgen` (image->mesh via the sidecar), plus the shared `spawn_restart`
dispatch the REST router and the MCP server both use.
"""

from __future__ import annotations

import uuid

import pytest

from cadx import spawn, storage
from cadx.db.models import (
    Asset,
    AssetKind,
    Job,
    JobStatus,
    JobType,
    Project,
    Version,
    VersionKind,
    VersionStatus,
)
from cadx.db.session import async_session, create_all
from cadx.runtime_config import RuntimeSettings, save_runtime_settings

CODE = (
    'PARAMS = {"w": 20.0}\n'
    "from build123d import Box\n"
    'result = Box(PARAMS["w"], 10.0, 5.0)\n'
)
CODE_NO_PARAMS = "PARAMS = {}\nfrom build123d import Box\nresult = Box(10, 10, 5)\n"


@pytest.fixture(autouse=True)
async def _runtime_defaults():
    """Fresh tables + default runtime row per test (mirrors test_injob_refine)."""
    await create_all()
    async with async_session() as db:
        await save_runtime_settings(db, RuntimeSettings())
        await db.commit()


def _enqueue_recorder() -> tuple[list[dict], object]:
    calls: list[dict] = []

    async def fake_enqueue(func, payload):
        calls.append({"func": func, "payload": payload})
        return "fake-arq-id"

    return calls, fake_enqueue


async def _mk_project() -> Project:
    async with async_session() as db:
        p = Project(name="mcp-jobs")
        db.add(p)
        await db.commit()
        return p


# --------------------------------------------------------------------------- #
# spawn_code                                                                  #
# --------------------------------------------------------------------------- #


async def test_spawn_code_creates_code_version_and_job():
    p = await _mk_project()
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        v, j = await spawn.spawn_code(db, p, CODE, params={"w": 25.0}, enqueue_fn=enq)
    assert v.kind == VersionKind.code
    assert j.type == JobType.code
    assert v.code == CODE
    assert v.params == {"w": 25.0}
    assert v.params_schema and v.params_schema[0]["name"] == "w"
    assert v.params_schema[0]["value"] == 20.0  # declared default, not the override
    assert calls == [
        {"func": "run_code_job", "payload": {"job_id": str(j.id), "version_id": str(v.id)}}
    ]


async def test_spawn_code_rejects_empty_code():
    p = await _mk_project()
    async with async_session() as db:
        with pytest.raises(spawn.SpawnError) as ei:
            await spawn.spawn_code(db, p, "   \n ")
        assert ei.value.status == 422


async def test_spawn_code_rejects_contract_errors():
    p = await _mk_project()
    async with async_session() as db:
        with pytest.raises(spawn.SpawnError) as ei:
            await spawn.spawn_code(db, p, "import os\nresult = None\n")
        assert ei.value.status == 422
        assert "code check failed" in str(ei.value)


async def test_spawn_code_inherits_parent_inputs():
    async with async_session() as db:
        p = Project(name="mcp-code-parent")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=VersionStatus.succeeded,
            asset_ids=[str(uuid.uuid4())],
            library_part_ids=[str(uuid.uuid4())],
        )
        db.add(parent)
        await db.commit()
        _calls, enq = _enqueue_recorder()
        v, _j = await spawn.spawn_code(db, p, CODE, parent=parent, enqueue_fn=enq)
        assert v.parent_id == parent.id
        assert v.asset_ids == parent.asset_ids
        assert v.library_part_ids == parent.library_part_ids


# --------------------------------------------------------------------------- #
# spawn_meshgen                                                               #
# --------------------------------------------------------------------------- #


async def test_spawn_meshgen_requires_asset_ids():
    p = await _mk_project()
    async with async_session() as db:
        with pytest.raises(spawn.SpawnError) as ei:
            await spawn.spawn_meshgen(db, p, asset_ids=[])
        assert ei.value.status == 422


async def test_spawn_meshgen_stores_options_and_enqueues():
    p = await _mk_project()
    aid = uuid.uuid4()
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        v, j = await spawn.spawn_meshgen(
            db,
            p,
            asset_ids=[aid],
            subject="a knob",
            model="hunyuan",
            seed=7,
            post={"scale": 1.2},
            enqueue_fn=enq,
        )
    assert v.kind == VersionKind.meshgen
    assert j.type == JobType.meshgen
    assert v.gen_mode == "mesh"
    assert v.asset_ids == [str(aid)]
    assert v.options["subject"] == "a knob"
    assert v.options["model"] == "hunyuan"
    assert v.options["seed"] == 7
    assert v.options["post"] == {"scale": 1.2}
    assert calls and calls[0]["func"] == "run_meshgen_job"


# --------------------------------------------------------------------------- #
# run_code_job                                                                #
# --------------------------------------------------------------------------- #


class _FakeExec:
    """QueueExecutor stand-in — records the execute payload, no real sandbox."""

    def __init__(self, result: dict | None = None) -> None:
        self.execute_payload: dict | None = None
        self.audit_payload: dict | None = None
        self.result = result or {
            "ok": True,
            "exports": {},
            "analysis": {"volume": 1000.0},
            "params_used": {"w": 20.0},
            "renders": {},
        }

    async def execute(self, payload: dict) -> dict:
        self.execute_payload = payload
        return dict(self.result)

    async def audit_params(self, payload: dict) -> dict:
        self.audit_payload = payload
        return {"ok": True, "status": "robust", "audited": 1, "params": {}}


async def _mk_code_version(code: str = CODE_NO_PARAMS) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    async with async_session() as db:
        p = Project(name="code-job")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.code,
            prompt="run my code",
            code=code,
            params={},
            params_schema=[],
        )
        db.add(v)
        await db.flush()
        j = Job(version_id=v.id, type=JobType.code)
        db.add(j)
        await db.commit()
        return p.id, v.id, j.id


async def test_run_code_job_executes_and_succeeds(monkeypatch):
    from cadx.agent import worker_jobs as wj

    fake = _FakeExec()
    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: fake)
    pid, vid, jid = await _mk_code_version()

    res = await wj.run_code_job({}, {"job_id": str(jid), "version_id": str(vid)})
    assert res["ok"] is True
    assert fake.execute_payload is not None
    assert fake.execute_payload["code"] == CODE_NO_PARAMS
    assert fake.execute_payload["params_override"] == {}
    assert fake.execute_payload["full_exports"] is True

    async with async_session() as db:
        v = await db.get(Version, vid)
        j = await db.get(Job, jid)
        assert v.status == VersionStatus.succeeded
        assert v.resolved_mode == "cad"
        assert v.params == {"w": 20.0}  # params_used wins
        assert v.iterations == 1
        assert j.status == JobStatus.succeeded
    v_dir = storage.version_dir(pid, vid)
    assert (v_dir / "model.py").read_text() == CODE_NO_PARAMS
    assert (v_dir / "params.json").exists()


async def test_run_code_job_audits_numeric_params(monkeypatch):
    from cadx.agent import worker_jobs as wj

    fake = _FakeExec()
    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: fake)
    _pid, vid, jid = await _mk_code_version(CODE)  # declares numeric PARAMS.w

    res = await wj.run_code_job({}, {"job_id": str(jid), "version_id": str(vid)})
    assert res["ok"] is True
    assert fake.audit_payload is not None
    assert fake.audit_payload["code"] == CODE
    assert fake.audit_payload["params"]["w"] == 20.0
    async with async_session() as db:
        v = await db.get(Version, vid)
        assert any(i["source"] == "design_audit" for i in (v.issues or []))


async def test_run_code_job_exec_failure_marks_failed(monkeypatch):
    from cadx.agent import worker_jobs as wj

    fake = _FakeExec({"ok": False, "error": "sandbox blew up"})
    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: fake)
    _pid, vid, jid = await _mk_code_version()

    res = await wj.run_code_job({}, {"job_id": str(jid), "version_id": str(vid)})
    assert res["ok"] is False
    async with async_session() as db:
        v = await db.get(Version, vid)
        j = await db.get(Job, jid)
        assert v.status == VersionStatus.failed
        assert v.error == "sandbox blew up"
        assert j.status == JobStatus.failed


# --------------------------------------------------------------------------- #
# run_params_job — version's own code wins over the inherited parent code     #
# --------------------------------------------------------------------------- #


async def test_run_params_job_prefers_version_code(monkeypatch):
    from cadx.agent import worker_jobs as wj

    fake = _FakeExec()
    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: fake)

    code_b = CODE_NO_PARAMS.replace("Box(10, 10, 5)", "Box(30, 10, 5)")
    async with async_session() as db:
        p = Project(name="params-src")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=VersionStatus.succeeded,
            code=CODE,
            params={"w": 20.0},
        )
        db.add(parent)
        await db.flush()
        child = Version(
            project_id=p.id,
            parent_id=parent.id,
            number=2,
            kind=VersionKind.params,
            prompt="params change",
            code=code_b,  # child carries its own source
            params={"w": 30.0},
        )
        db.add(child)
        await db.flush()
        j = Job(version_id=child.id, type=JobType.params)
        db.add(j)
        await db.commit()
        cid, jid = child.id, j.id

    res = await wj.run_params_job(
        {}, {"job_id": str(jid), "version_id": str(cid), "params": {"w": 30.0}}
    )
    assert res["ok"] is True
    assert fake.execute_payload["code"] == code_b  # child's source, not parent's
    async with async_session() as db:
        v = await db.get(Version, cid)
        assert v.code == code_b
        assert v.status == VersionStatus.succeeded


# --------------------------------------------------------------------------- #
# spawn_restart                                                               #
# --------------------------------------------------------------------------- #


async def _mk_terminal_job(
    *, kind: VersionKind, jtype: JobType, **vkw
) -> tuple[Project, Version, Job]:
    async with async_session() as db:
        p = Project(name="restart-src")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id,
            number=1,
            kind=kind,
            prompt="src",
            status=VersionStatus.succeeded,
            **vkw,
        )
        db.add(v)
        await db.flush()
        j = Job(version_id=v.id, type=jtype, status=JobStatus.succeeded)
        db.add(j)
        await db.commit()
        return p, v, j


async def test_spawn_restart_dispatches_code():
    p, v, j = await _mk_terminal_job(kind=VersionKind.code, jtype=JobType.code, code=CODE)
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        proj = await db.get(Project, p.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.id != v.id
    assert new_v.kind == VersionKind.code
    assert new_v.code == CODE
    assert new_j.type == JobType.code
    assert calls and calls[0]["func"] == "run_code_job"
    assert proj is not None


async def test_spawn_restart_dispatches_meshgen():
    aid = str(uuid.uuid4())
    _p, v, j = await _mk_terminal_job(
        kind=VersionKind.meshgen,
        jtype=JobType.meshgen,
        gen_mode="mesh",
        asset_ids=[aid],
        options={"subject": "knob", "model": "auto", "seed": 3, "post": {"scale": 0.9}},
    )
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.id != v.id
    assert new_v.kind == VersionKind.meshgen
    assert new_v.asset_ids == [aid]
    assert new_v.options["seed"] == 3
    assert new_v.options["post"] == {"scale": 0.9}
    assert new_j.type == JobType.meshgen
    assert calls and calls[0]["func"] == "run_meshgen_job"


async def test_spawn_restart_rejects_unknown_kind():
    _p, _v, j = await _mk_terminal_job(kind=VersionKind.mesh_edit, jtype=JobType.mesh_edit)
    async with async_session() as db:
        job = await db.get(Job, j.id)
        with pytest.raises(spawn.SpawnError) as ei:
            await spawn.spawn_restart(db, job)
        assert ei.value.status == 409


async def test_spawn_restart_rejects_running_job():
    _p, _v, j = await _mk_terminal_job(kind=VersionKind.code, jtype=JobType.code, code=CODE)
    async with async_session() as db:
        job = await db.get(Job, j.id)
        job.status = JobStatus.running
        await db.commit()
        with pytest.raises(spawn.SpawnError) as ei:
            await spawn.spawn_restart(db, job)
        assert ei.value.status == 409


# --------------------------------------------------------------------------- #
# run_meshgen_job                                                             #
# --------------------------------------------------------------------------- #


class _FakeMesh:
    """MeshPort stand-in — writes a stub GLB, no sidecar."""

    def __init__(self) -> None:
        self.req = None

    async def health(self):
        from cadx.agent.mesh_client import MeshHealth

        return MeshHealth(ok=True, device="fake")

    async def generate(self, req, on_progress, should_abort=None):
        from cadx.agent.mesh_client import MeshGenResult

        self.req = req
        req.out_path.parent.mkdir(parents=True, exist_ok=True)
        req.out_path.write_bytes(b"glb")
        return MeshGenResult(
            model=req.model,
            backend="fake",
            seed=req.seed,
            duration_s=0.01,
            mesh_path=req.out_path,
            triangles=3,
        )


class _FakeMeshExec(_FakeExec):
    async def process_mesh(self, payload: dict) -> dict:
        self.execute_payload = payload
        it_dir = storage.iteration_dir(
            uuid.UUID(payload["project_id"]), uuid.UUID(payload["version_id"]), 1
        )
        run = storage.ensure(it_dir / "_run")
        (run / "model.stl").write_text("solid m\nendsolid m\n")
        return {
            "ok": True,
            "exports": {},
            "analysis": {"volume": 42.0},
            "mesh": {"post": payload["post"], "triangles": 3},
            "renders": {},
        }


async def test_run_meshgen_job_generates_processes_and_succeeds(monkeypatch):
    from cadx.agent import worker_jobs as wj

    mesh = _FakeMesh()
    fake = _FakeMeshExec()
    monkeypatch.setattr(wj, "_make_mesh", lambda _rt: mesh)
    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: fake)

    async with async_session() as db:
        p = Project(name="meshgen-job")
        db.add(p)
        await db.flush()
        a = Asset(
            project_id=p.id,
            kind=AssetKind.image,
            filename="ref.png",
            path=f"projects/{p.id}/assets/ref.png",
        )
        db.add(a)
        await db.flush()
        v = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.meshgen,
            prompt="a knob",
            gen_mode="mesh",
            asset_ids=[str(a.id)],
            options={"subject": "a knob", "model": "fake", "seed": 5, "post": {"scale": 1.1}},
        )
        db.add(v)
        await db.flush()
        j = Job(version_id=v.id, type=JobType.meshgen)
        db.add(j)
        await db.commit()
        pid, vid, jid = p.id, v.id, j.id

    res = await wj.run_meshgen_job({}, {"job_id": str(jid), "version_id": str(vid)})
    assert res["ok"] is True
    assert mesh.req is not None
    assert mesh.req.model == "fake"
    assert mesh.req.seed == 5
    assert len(mesh.req.images) == 1

    async with async_session() as db:
        v = await db.get(Version, vid)
        assert v.status == VersionStatus.succeeded
        assert v.resolved_mode == "mesh"
        assert v.params["scale"] == 1.1
        assert v.params_schema  # mesh post-params schema for the params UI
        assert (v.mesh or {}).get("generation", {}).get("model") == "fake"
        assert v.iterations == 1
    assert (storage.version_dir(pid, vid) / "mesh" / "mesh_raw.glb").exists()
