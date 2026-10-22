"""MCP LLM-free jobs: `code` (persist+execute external build123d source) and
`meshgen` (image->mesh via the sidecar), plus the shared `spawn_restart`
dispatch the REST router and the MCP server both use.
"""

from __future__ import annotations

import uuid

import pytest
from sqlmodel import select

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

CODE = 'PARAMS = {"w": 20.0}\nfrom build123d import Box\nresult = Box(PARAMS["w"], 10.0, 5.0)\n'
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
            part_id=uuid.uuid4(),
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
            part_id=uuid.uuid4(),
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
            part_id=uuid.uuid4(),
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
            part_id=uuid.uuid4(),
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
            part_id=uuid.uuid4(),
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
    # a mesh_edit version whose job never ran has no recorded ops to replay
    _p, _v, j = await _mk_terminal_job(kind=VersionKind.mesh_edit, jtype=JobType.mesh_edit)
    async with async_session() as db:
        job = await db.get(Job, j.id)
        with pytest.raises(spawn.SpawnError) as ei:
            await spawn.spawn_restart(db, job)
        assert ei.value.status == 409


async def _mk_parent_and_child(
    *,
    parent_kind: VersionKind,
    child_kind: VersionKind,
    child_jtype: JobType,
    child_options: dict | None = None,
    child_prompt: str = "child",
    child_params: dict | None = None,
    parent_mesh: dict | None = None,
    job_kwargs: dict | None = None,
) -> tuple[Project, Version, Version, Job]:
    """succeeded parent + terminal child job, wired as a spawn tree."""
    async with async_session() as db:
        p = Project(name="restart-tree")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=1,
            kind=parent_kind,
            prompt="base",
            status=VersionStatus.succeeded,
            mesh=parent_mesh,
        )
        db.add(parent)
        await db.flush()
        child = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            parent_id=parent.id,
            number=2,
            kind=child_kind,
            prompt=child_prompt,
            status=VersionStatus.succeeded,
            options=child_options or {},
            params=child_params,
        )
        db.add(child)
        await db.flush()
        j = Job(
            version_id=child.id, type=child_jtype, status=JobStatus.failed, **(job_kwargs or {})
        )
        db.add(j)
        await db.commit()
        return p, parent, child, j


async def test_spawn_restart_resample_replays_stored_request():
    # a resample child shares kind=optimize with real optimize children —
    # dispatch must be on job.type or it re-runs a full SIMP
    _p, parent, v, j = await _mk_parent_and_child(
        parent_kind=VersionKind.optimize,
        parent_mesh={"topopt": {"density_relpath": "projects/x/density.npz"}},
        child_kind=VersionKind.optimize,
        child_jtype=JobType.resample,
        child_options={
            "topopt": True,
            "optimize_request": {
                "optimize": {"density_threshold": 0.35, "min_member_mm": 2.5},
                "material": "pla",
            },
        },
    )
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.id != v.id
    assert new_v.parent_id == parent.id
    assert new_v.kind == VersionKind.optimize
    assert new_j.type == JobType.resample
    assert calls and calls[0]["func"] == "run_topopt_resample_job"
    payload = calls[0]["payload"]
    assert payload["optimize"] == {"density_threshold": 0.35, "min_member_mm": 2.5}
    assert payload["material"] == "pla"


async def test_spawn_restart_refine_replays_stored_request():
    _p, parent, v, j = await _mk_parent_and_child(
        parent_kind=VersionKind.optimize,
        child_kind=VersionKind.refine,
        child_jtype=JobType.refine,
        child_options={
            "refine_request": {
                "features": [{"type": "bore", "dia": 5.0}],
                "load_cases": [{"name": "lc"}],
                "keep_regions": [{"kind": "box"}],
                "material": "petg",
                "voxel_mm": 0.8,
                "restore_planes": True,
                "helical_threads": False,
                "auto_plane_restore": True,
            }
        },
    )
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.id != v.id
    assert new_v.parent_id == parent.id
    assert new_v.kind == VersionKind.refine
    assert new_j.type == JobType.refine
    assert calls and calls[0]["func"] == "run_refine_job"
    payload = calls[0]["payload"]
    assert payload["features"] == [{"type": "bore", "dia": 5.0}]
    assert payload["load_cases"] == [{"name": "lc"}]
    assert payload["voxel_mm"] == 0.8
    assert payload["auto_plane_restore"] is True


async def test_spawn_restart_organic_replays_stored_request():
    _p, parent, _v, j = await _mk_parent_and_child(
        parent_kind=VersionKind.generate,
        child_kind=VersionKind.organic,
        child_jtype=JobType.organic,
        child_options={"organic_request": {"style": "gyroid", "seed": 4, "volfrac": 0.3}},
    )
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.parent_id == parent.id
    assert new_v.kind == VersionKind.organic
    assert new_j.type == JobType.organic
    assert calls and calls[0]["func"] == "run_organic_job"
    assert calls[0]["payload"]["organic"] == {"style": "gyroid", "seed": 4, "volfrac": 0.3}


async def test_spawn_restart_mesh_edit_replays_recorded_ops():
    p, parent, _v, j = await _mk_parent_and_child(
        parent_kind=VersionKind.generate,
        child_kind=VersionKind.mesh_edit,
        child_jtype=JobType.mesh_edit,
        child_prompt="visual edit of v1",
        child_params={
            "ops": [{"op": "rotate", "args": {"axis": "z", "deg": 45}}],
            "scale": 1.0,
        },
    )
    # mesh_edit_source guard: a finished STL on the parent
    exp_dir = storage.export_dir(p.id, parent.id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "model.stl").write_text("solid m\nendsolid m\n")
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.parent_id == parent.id
    assert new_v.kind == VersionKind.mesh_edit
    assert new_j.type == JobType.mesh_edit
    assert calls and calls[0]["func"] == "run_mesh_edit_job"
    payload = calls[0]["payload"]
    assert payload["ops"] == [{"op": "rotate", "args": {"axis": "z", "deg": 45}}]
    assert payload["post"] == {"scale": 1.0}


async def test_spawn_restart_fem_same_version():
    async with async_session() as db:
        p = Project(name="fem-restart")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=1,
            kind=VersionKind.generate,
            prompt="bracket",
            status=VersionStatus.succeeded,
        )
        db.add(v)
        await db.flush()
        j = Job(
            version_id=v.id,
            type=JobType.fem,
            status=JobStatus.failed,
            resume_state={
                "fem_request": {
                    "load_cases": [{"name": "down", "load_n": [0, 0, -50]}],
                    "keep_regions": [],
                    "material": "pla",
                }
            },
        )
        db.add(j)
        await db.commit()
        vid = v.id
    # spawn_fem requires a mesh artifact on disk
    exp_dir = storage.export_dir(p.id, vid)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "model.stl").write_text("solid m\nendsolid m\n")

    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        res = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert isinstance(res, Job)  # same-version job — no new Version row
    assert res.version_id == vid
    assert res.type == JobType.fem
    assert calls and calls[0]["func"] == "run_fem_job"
    payload = calls[0]["payload"]
    assert payload["version_id"] == str(vid)
    assert payload["load_cases"] == [{"name": "down", "load_n": [0, 0, -50]}]
    assert payload["material"] == "pla"


async def test_spawn_restart_fem_without_stored_request_409s():
    # fem jobs spawned before requests were recorded have nothing to replay
    async with async_session() as db:
        p = Project(name="fem-restart-old")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=1,
            kind=VersionKind.generate,
            prompt="bracket",
            status=VersionStatus.succeeded,
        )
        db.add(v)
        await db.flush()
        j = Job(version_id=v.id, type=JobType.fem, status=JobStatus.failed)
        db.add(j)
        await db.commit()
        jid = j.id
    async with async_session() as db:
        job = await db.get(Job, jid)
        with pytest.raises(spawn.SpawnError) as ei:
            await spawn.spawn_restart(db, job)
        assert ei.value.status == 409


async def test_spawn_restart_reexport_same_version():
    _p, v, j = await _mk_terminal_job(
        kind=VersionKind.generate, jtype=JobType.reexport, iterations=3
    )
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        res = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert isinstance(res, Job)
    assert res.version_id == v.id
    assert res.type == JobType.reexport
    assert calls and calls[0]["func"] == "run_reexport_job"
    assert calls[0]["payload"] == {
        "job_id": str(res.id),
        "version_id": str(v.id),
        "iteration": 3,
    }


async def test_spawn_restart_edit_feature_replays_gate():
    # kind=edit + type=edit but spawned by spawn_edit_feature — must replay
    # run_edit_feature_job, not the LLM edit branch
    p, parent, v, j = await _mk_parent_and_child(
        parent_kind=VersionKind.generate,
        child_kind=VersionKind.edit,
        child_jtype=JobType.edit,
        child_prompt="feature edit: diameter=8.0 on feature 3 (of v1)",
        child_options={
            "edit_feature_request": {"feature_id": 3, "parameter": "diameter", "value": 8.0}
        },
    )
    # feature_edit_source guard: a B-Rep on the parent
    m_dir = storage.version_dir(p.id, parent.id) / "mesh"
    m_dir.mkdir(parents=True, exist_ok=True)
    (m_dir / "solid.brep").write_bytes(b"brep")
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.id != v.id
    assert new_v.parent_id == parent.id
    assert new_v.kind == VersionKind.edit
    assert new_j.type == JobType.edit
    assert calls and calls[0]["func"] == "run_edit_feature_job"
    payload = calls[0]["payload"]
    assert payload["feature_id"] == 3
    assert payload["parameter"] == "diameter"
    assert payload["value"] == 8.0


async def test_spawn_restart_llm_edit_ignores_inherited_feature_request():
    # a genuine LLM edit child of a feature-edit version inherits the
    # options key — the prompt is the discriminator
    _p, parent, _v, j = await _mk_parent_and_child(
        parent_kind=VersionKind.generate,
        child_kind=VersionKind.edit,
        child_jtype=JobType.edit,
        child_prompt="make it taller",
        child_options={
            "edit_feature_request": {"feature_id": 3, "parameter": "diameter", "value": 8.0}
        },
    )
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.parent_id == parent.id
    assert new_v.kind == VersionKind.edit
    assert new_j.type == JobType.edit
    assert calls and calls[0]["func"] == "run_agent_job"


async def test_spawn_restart_resume_preserves_type_and_state():
    resume_state = {"spec": {"name": "bracket", "load_cases": []}}
    _p, v, j = await _mk_terminal_job(
        kind=VersionKind.generate,
        jtype=JobType.resume,
        gen_mode="auto",
    )
    async with async_session() as db:
        job = await db.get(Job, j.id)
        job.resume_state = resume_state
        await db.commit()
    calls, enq = _enqueue_recorder()
    async with async_session() as db:
        job = await db.get(Job, j.id)
        new_v, new_j = await spawn.spawn_restart(db, job, enqueue_fn=enq)
    assert new_v.id != v.id
    assert new_v.kind == VersionKind.generate
    # must stay `resume` — run_agent_job only restores the stored spec for
    # JobType.resume
    assert new_j.type == JobType.resume
    assert new_j.resume_state == resume_state
    assert calls and calls[0]["func"] == "run_agent_job"


async def test_spawn_enqueue_failure_raises_503_and_fails_rows(monkeypatch):
    """deps.enqueue returning None must surface as SpawnError(503) and the
    committed Version/Job rows get marked failed — nothing may stay queued
    with no arq task behind it."""
    import cadx.api.deps as deps

    async def _none_enqueue(func, payload):
        return None

    monkeypatch.setattr(deps, "enqueue", _none_enqueue)
    p = await _mk_project()
    with pytest.raises(spawn.SpawnError) as ei:
        async with async_session() as db:
            await spawn.spawn_generate(db, p, "a box")
    assert ei.value.status == 503

    async with async_session() as db:
        vs = (await db.execute(select(Version).where(Version.project_id == p.id))).scalars().all()
        js = (await db.execute(select(Job))).scalars().all()
    assert len(vs) == 1
    job = next(j for j in js if j.version_id == vs[0].id)
    assert vs[0].status == VersionStatus.failed
    assert job.status == JobStatus.failed


async def test_spawn_enqueue_exception_marks_rows_failed():
    async def _boom(func, payload):
        raise RuntimeError("redis down")

    p = await _mk_project()
    with pytest.raises(RuntimeError):
        async with async_session() as db:
            await spawn.spawn_generate(db, p, "a box", enqueue_fn=_boom)
    async with async_session() as db:
        vs = (await db.execute(select(Version).where(Version.project_id == p.id))).scalars().all()
        js = (await db.execute(select(Job))).scalars().all()
    job = next(j for j in js if j.version_id == vs[0].id)
    assert vs[0].status == VersionStatus.failed
    assert job.status == JobStatus.failed


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
    monkeypatch.setattr(wj, "_make_mesh", lambda _rt, **kw: mesh)
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
            part_id=uuid.uuid4(),
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
