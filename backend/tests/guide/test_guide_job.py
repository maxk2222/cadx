"""run_guide_job lifecycle: spawn → mission outcome → waiting_for_input →
resume carries answers through resume_state (state file is canonical)."""

from __future__ import annotations

import pytest

import cadx.agent.worker_jobs as wj
from cadx.agent.orchestrator import RunOutcome
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
from cadx.spawn import enqueue_resume, spawn_guide


@pytest.fixture(scope="module")
async def _db():
    await create_all()


async def _mk_project() -> Project:
    async with async_session() as db:
        p = Project(name="guide-e2e")
        db.add(p)
        await db.commit()
        await db.refresh(p)
        return p


async def _mk_pdf_asset(project: Project) -> Asset:
    async with async_session() as db:
        a = Asset(
            project_id=project.id,
            kind=AssetKind.pdf,
            filename="manual.pdf",
            path="dummy/manual.pdf",
        )
        db.add(a)
        await db.commit()
        await db.refresh(a)
        return a


async def test_spawn_guide_rows(_db):
    """spawn_guide creates Part/Version(kind=guide)/Job(type=guide) with the
    guide_dir in resume_state and a run_guide_job payload."""
    project = await _mk_project()
    asset = await _mk_pdf_asset(project)
    async with async_session() as db:
        p = await db.get(Project, project.id)
        v, job = await spawn_guide(db, p, pdf_asset_id=asset.id)
        assert v.kind == VersionKind.guide
        assert job.type == JobType.guide
        assert job.version_id == v.id
        assert job.payload["func"] == "run_guide_job"
        assert job.resume_state["guide_dir"].endswith("/guide")
        assert str(asset.id) in v.asset_ids


async def test_run_guide_job_waiting_then_resume(_db, monkeypatch):
    """waiting_for_input parks with questions; enqueue_resume dispatches
    run_guide_job again and answers land in resume_state."""
    project = await _mk_project()
    asset = await _mk_pdf_asset(project)
    async with async_session() as db:
        p = await db.get(Project, project.id)
        v, job = await spawn_guide(db, p, pdf_asset_id=asset.id)

    # stub the mission loop: first call → parked with a question; second → done
    calls = {"n": 0}

    async def fake_run_guide(
        actx, *, state_dir, resume_state, allow_generation=True, attach_images=True, park_job=None
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            assert "guide_dir" in resume_state
            return RunOutcome(
                status="waiting_for_input",
                questions=[{"text": "welche Schraubenlänge?", "blocking": True}],
                resume_state={"guide_dir": str(state_dir)},
            )
        # resume: answers_text must arrive
        assert resume_state.get("answers_text") == "M4x12"
        assert resume_state.get("guide_dir")
        return RunOutcome(status="succeeded", score=9)

    import cadx.guide.guide_runner as gr

    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: object())
    monkeypatch.setattr(wj, "_make_llm", lambda rt, **kw: object())
    monkeypatch.setattr(gr, "run_guide", fake_run_guide)
    # run_guide is imported lazily inside run_guide_job — patch the module attr
    res = await wj.run_guide_job({}, {"job_id": str(job.id), "version_id": str(v.id)})
    assert res["status"] == "waiting_for_input"

    async with async_session() as db:
        j = await db.get(Job, job.id)
        assert j.status == JobStatus.waiting_for_input
        assert j.questions[0]["text"].startswith("welche")
        _v, resume = await enqueue_resume(db, j, "M4x12")
        assert resume.payload["func"] == "run_guide_job"
        assert resume.resume_state["answers_text"] == "M4x12"
        assert resume.resume_state["guide_dir"]

    res2 = await wj.run_guide_job({}, {"job_id": str(resume.id), "version_id": str(v.id)})
    assert res2["status"] == "succeeded"
    async with async_session() as db:
        v2 = await db.get(Version, v.id)
        assert v2.status == VersionStatus.succeeded


async def test_unity_artifact_written(_db, monkeypatch, tmp_path):
    """emit writes unity_guide.json into the guide output dir."""
    from cadx import storage
    from cadx.guide.guide_runner import _emit_artifacts
    from cadx.guide.state import GuideState

    # production guide dirs live under storage.data_root() (version_dir/guide)
    state_dir = storage.data_root() / "test-guide" / "guide"
    state_dir.mkdir(parents=True, exist_ok=True)
    state = GuideState(output_dir=str(state_dir))
    state.save(state_dir)
    arts = await _emit_artifacts(state, state_dir)
    assert (state_dir / "output" / "unity_guide.json").exists()
    assert arts["guide:unity_guide"].endswith("unity_guide.json")
    assert storage.resolve(arts["guide:unity_guide"]).exists()
