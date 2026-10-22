"""Full worker-path e2e: run_agent_job with FakeLLM + in-process CAD pipeline.

Covers DB persistence, artifact registration, status transitions — everything
except redis/arq transport itself.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlmodel import select

import cadx.agent.worker_jobs as wj
from cadx import storage
from cadx.agent.executor_client import InProcessExecutor
from cadx.db.models import (
    Artifact,
    Job,
    JobStatus,
    JobType,
    Project,
    Version,
    VersionKind,
    VersionStatus,
)
from cadx.db.session import async_session, create_all
from cadx.llm.fake import FakeLLM

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOX_CODE = (FIXTURES / "simple_box.py").read_text()

SPEC = {
    "name": "block",
    "summary": "40x20x10 block",
    "units": "mm",
    "length_x": {"value": 40.0, "source": "stated"},
    "width_y": {"value": 20.0, "source": "stated"},
    "height_z": {"value": 10.0, "source": "stated"},
    "open_questions": [],
}
PLAN = {"steps": ["box + fillet"], "params": [], "standard_parts": [], "notes": ""}
CODE = {
    "reasoning": "box",
    "params": [{"name": "length", "value": 40.0}],
    "expected_bbox": [40.0, 20.0, 10.0],
    "code": BOX_CODE,
}
CRITIQUE = {"score": 95, "passed": True, "discrepancies": [], "matches": [], "confidence": 0.9}


@pytest.fixture(scope="module")
async def _db():
    await create_all()


async def _mk_version() -> tuple[Job, Version]:
    async with async_session() as db:
        p = Project(name="e2e")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=1,
            kind=VersionKind.generate,
            prompt="a 40x20x10 block",
        )
        db.add(v)
        await db.flush()
        j = Job(version_id=v.id, type=JobType.generate)
        db.add(j)
        await db.commit()
        await db.refresh(j)
        await db.refresh(v)
        return j, v


@pytest.mark.slow
async def test_run_agent_job_end_to_end(_db, monkeypatch):
    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: InProcessExecutor())
    llm = FakeLLM([SPEC, PLAN, CODE, CRITIQUE])
    monkeypatch.setattr(wj, "_make_llm", lambda rt, **kw: llm)

    job, version = await _mk_version()
    result = await wj.run_agent_job({}, {"job_id": str(job.id), "version_id": str(version.id)})
    assert result["ok"] is True

    async with async_session() as db:
        v = await db.get(Version, version.id)
        j = await db.get(Job, job.id)
        assert v.status == VersionStatus.succeeded
        assert j.status == JobStatus.succeeded
        assert v.code == BOX_CODE
        assert v.analysis["bbox_size"][0] == pytest.approx(40.0, abs=2.0)
        assert v.score == 95
        arts = (
            (await db.execute(select(Artifact).where(Artifact.version_id == v.id))).scalars().all()
        )
        kinds = {a.kind.value for a in arts}
        assert {"step", "stl", "glb", "contact_sheet", "thumbnail"} <= kinds


async def test_guarded_cancelled_marks_job_failed(_db):
    """arq job_timeout cancels the task with CancelledError (BaseException) —
    the crash guard must still fail the job/version rows instead of leaving
    them stuck in 'running'."""
    import asyncio

    job, version = await _mk_version()
    async with async_session() as db:
        j = await db.get(Job, job.id)
        j.status = JobStatus.running
        v = await db.get(Version, version.id)
        v.status = VersionStatus.running
        await db.commit()

    async def _boom(ctx, payload):
        raise asyncio.CancelledError

    guarded = wj._guarded(_boom)
    with pytest.raises(asyncio.CancelledError):
        await guarded({}, {"job_id": str(job.id), "version_id": str(version.id)})

    async with async_session() as db:
        j = await db.get(Job, job.id)
        v = await db.get(Version, version.id)
        assert j.status == JobStatus.failed
        assert v.status == VersionStatus.failed
        assert "cancelled" in (j.log or "")


@pytest.mark.slow
async def test_run_agent_job_waiting_then_resume(_db, monkeypatch):
    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: InProcessExecutor())
    spec = dict(SPEC)
    spec["open_questions"] = [{"question": "wall?", "blocking": True, "default_assumption": "2"}]
    llm = FakeLLM([spec])
    monkeypatch.setattr(wj, "_make_llm", lambda rt, **kw: llm)
    job, version = await _mk_version()
    # force clarification_mode=ask via project settings
    async with async_session() as db:
        p = await db.get(Project, version.project_id)
        p.settings = {"clarification_mode": "ask"}
        await db.commit()

    result = await wj.run_agent_job({}, {"job_id": str(job.id), "version_id": str(version.id)})
    assert result["status"] == "waiting_for_input"

    async with async_session() as db:
        j = await db.get(Job, job.id)
        v = await db.get(Version, version.id)
        assert j.status == JobStatus.waiting_for_input
        assert v.status == VersionStatus.waiting_for_input
        assert j.questions and j.questions[0]["blocking"]
        assert j.resume_state["spec"]


@pytest.mark.slow
async def test_run_params_job_mesh_parent(_db, monkeypatch):
    """Params job on a mesh version re-runs the post-pipeline without an LLM."""
    import trimesh

    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: InProcessExecutor())

    async with async_session() as db:
        p = Project(name="mesh-params")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=1,
            kind=VersionKind.generate,
            prompt="organic knob",
            status=VersionStatus.succeeded,
            resolved_mode="mesh",
            gen_mode="mesh",
            params={"scale": 1.0},
            spec=SPEC,
            mesh={"post": {"scale": 1.0}, "generation": {"model": "fake"}},
        )
        db.add(parent)
        await db.flush()
        child = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=2,
            parent_id=parent.id,
            kind=VersionKind.params,
            prompt="scale 2x",
        )
        db.add(child)
        await db.flush()
        job = Job(version_id=child.id, type=JobType.params)
        db.add(job)
        await db.commit()
        await db.refresh(job)
        await db.refresh(child)
        await db.refresh(parent)

    # drop a fake raw mesh into the parent's mesh dir
    mesh_dir = storage.ensure(storage.version_dir(parent.project_id, parent.id) / "mesh")
    blob = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    blob.export(str(mesh_dir / "mesh_raw.glb"), file_type="glb")

    result = await wj.run_params_job(
        {},
        {
            "job_id": str(job.id),
            "version_id": str(child.id),
            "params": {"scale": 2.0},
        },
    )
    assert result["ok"] is True, result.get("error")

    async with async_session() as db:
        v = await db.get(Version, child.id)
        assert v.status == VersionStatus.succeeded
        assert v.resolved_mode == "mesh"
        assert v.params["scale"] == 2.0
        assert v.mesh["from_params_of"] == str(parent.id)
        # scaled blob: normalized ~unit sphere x stated 40 mm x 2
        assert v.analysis["bbox_size"][0] == pytest.approx(80.0, abs=5.0)
        assert v.code is None
        arts = (
            (await db.execute(select(Artifact).where(Artifact.version_id == v.id))).scalars().all()
        )
        kinds = {a.kind.value for a in arts}
        assert "mesh_raw" in kinds


@pytest.mark.slow
async def test_run_params_job_mesh_ops_replayed(_db, monkeypatch):
    """persisted ops lists replay deterministically on params reruns."""
    import trimesh

    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: InProcessExecutor())

    ops = [
        {"op": "flatten_bottom", "args": {"height_mm": 1.0}},
        {"op": "scale", "args": {"factor": 1.5}},
    ]
    async with async_session() as db:
        p = Project(name="mesh-ops-params")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=1,
            kind=VersionKind.generate,
            prompt="organic knob",
            status=VersionStatus.succeeded,
            resolved_mode="mesh",
            gen_mode="mesh",
            params={"scale": 1.0, "ops": ops},
            spec=SPEC,
            mesh={"post": {"scale": 1.0, "ops": ops}, "generation": {"model": "fake"}},
        )
        db.add(parent)
        await db.flush()
        child = Version(
            project_id=p.id,
            part_id=uuid.uuid4(),
            number=2,
            parent_id=parent.id,
            kind=VersionKind.params,
            prompt="tweak",
        )
        db.add(child)
        await db.flush()
        job = Job(version_id=child.id, type=JobType.params)
        db.add(job)
        await db.commit()
        await db.refresh(job)
        await db.refresh(child)
        await db.refresh(parent)

    mesh_dir = storage.ensure(storage.version_dir(parent.project_id, parent.id) / "mesh")
    blob = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    blob.export(str(mesh_dir / "mesh_raw.glb"), file_type="glb")

    result = await wj.run_params_job(
        {},
        {"job_id": str(job.id), "version_id": str(child.id), "params": {"scale": 1.2}},
    )
    assert result["ok"] is True, result.get("error")

    async with async_session() as db:
        v = await db.get(Version, child.id)
        assert v.status == VersionStatus.succeeded
        # parent ops inherited by merge_post_params, scale override applied
        assert [o["op"] for o in v.params["ops"]] == ["flatten_bottom", "scale"]
        assert v.params["scale"] == 1.2
        assert [r["op"] for r in v.mesh["ops_report"]] == ["flatten_bottom", "scale"]
