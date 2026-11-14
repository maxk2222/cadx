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


async def test_spawn_guide_wizard_options(_db):
    """Wizard options land on version.options — run_guide_job reads them
    from there (language/audience seed the durable state)."""
    project = await _mk_project()
    asset = await _mk_pdf_asset(project)
    async with async_session() as db:
        p = await db.get(Project, project.id)
        v, _job = await spawn_guide(
            db,
            p,
            pdf_asset_id=asset.id,
            allow_generation=False,
            attach_images=False,
            language="fr",
            audience="beginner",
        )
        opts = v.options or {}
        assert opts["guide_allow_generation"] is False
        assert opts["guide_attach_images"] is False
        assert opts["guide_language"] == "fr"
        assert opts["guide_audience"] == "beginner"


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
        actx,
        *,
        state_dir,
        resume_state,
        allow_generation=True,
        attach_images=True,
        language="",
        audience="",
        park_job=None,
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


async def test_load_or_init_state_seeds_wizard_options(tmp_path):
    """language/audience kwargs seed the fresh GuideState — a set language
    is locked against doc-mission autodetect."""
    from types import SimpleNamespace

    from cadx.guide.guide_runner import _load_or_init_state
    from cadx.guide.state import GuideState

    ctx = SimpleNamespace(project_id="p1", version_id="v1", prompt="note text")
    state = _load_or_init_state(
        ctx, tmp_path, {}, allow_generation=True, language=" EN ", audience="pro"
    )
    assert state.language == "en"
    assert state.language_locked is True
    assert state.audience == "pro"
    assert state.user_notes == ["note text"]

    # defaults: auto-detect stays open, no rewrite at emit
    state2 = _load_or_init_state(ctx, tmp_path / "b", {}, allow_generation=True)
    assert state2.language == "de"
    assert state2.language_locked is False
    assert state2.audience == ""

    # resume: an existing state file wins over spawn kwargs
    GuideState(language="it", language_locked=True, audience="beginner").save(tmp_path)
    state3 = _load_or_init_state(
        ctx, tmp_path, {}, allow_generation=True, language="en", audience="pro"
    )
    assert state3.language == "it"
    assert state3.audience == "beginner"


async def test_apply_rewrite_stores_variant_and_applies():
    """apply_rewrite: _original snapshot + variant key + canonical steps
    updated — the shared core used by REST and the runner."""
    from types import SimpleNamespace

    from cadx.guide.rewrite import apply_rewrite
    from cadx.guide.state import GuideState, GuideStep

    class _FakeLLM:
        async def chat(self, role, messages, **kw):
            return SimpleNamespace(
                parsed={
                    "steps": [
                        {"number": 1, "title": "T1", "text": "easy", "display_text": "short"},
                        {"number": 2, "title": "", "text": "easier", "display_text": "short2"},
                    ]
                },
                text="",
            )

    state = GuideState()
    state.steps = [
        GuideStep(number=1, title="old", text="orig1", display_text="d1"),
        GuideStep(number=2, title="keep", text="orig2", display_text="d2"),
    ]
    out = await apply_rewrite(state, audience="beginner", language="en", llm=_FakeLLM())
    assert out["key"] == "beginner:en"
    assert out["applied"] == 2
    assert state.steps[0].text == "easy"
    # empty rewrite fields keep the original (title was "" in the response)
    assert state.steps[1].title == "keep"
    assert state.variants["_original"][0]["text"] == "orig1"
    assert state.variants["beginner:en"][1]["display_text"] == "short2"

    # second rewrite keeps the FIRST _original snapshot
    out2 = await apply_rewrite(state, audience="pro", language="", llm=_FakeLLM())
    assert out2["key"] == "pro:original"
    assert state.variants["_original"][0]["text"] == "orig1"


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


async def test_guide_note_endpoint_writes_jsonl(_db):
    """POST /versions/{id}/guide/notes appends to user_notes.jsonl — the
    exact line format attach_guide_note writes from chat, and it works
    before state.json exists (early-run notes)."""
    import json as _json

    from cadx import storage
    from cadx.api.routers.guide import GuideNoteIn, add_guide_note

    project = await _mk_project()
    asset = await _mk_pdf_asset(project)
    async with async_session() as db:
        p = await db.get(Project, project.id)
        v, _job = await spawn_guide(db, p, pdf_asset_id=asset.id)
        res = await add_guide_note(v.id, GuideNoteIn(note="check bolts"), db)
        assert res["ok"] is True
    p = storage.version_dir(project.id, v.id) / "guide" / "user_notes.jsonl"
    assert _json.loads(p.read_text().strip()) == {"note": "check bolts"}


def _seed_state(project, vid, n_steps=2):
    """Write a minimal durable GuideState so the guide endpoints load."""
    from cadx import storage
    from cadx.guide.state import GuideState, GuideStep

    guide_dir = storage.version_dir(project.id, vid) / "guide"
    guide_dir.mkdir(parents=True, exist_ok=True)
    st = GuideState()
    st.steps = [
        GuideStep(number=i + 1, title=f"s{i + 1}", text=f"step {i + 1}")
        for i in range(n_steps)
    ]
    st.save(guide_dir)
    return guide_dir


async def _mk_img_asset(project: Project) -> Asset:
    async with async_session() as db:
        a = Asset(
            project_id=project.id,
            kind=AssetKind.image,
            filename="photo.png",
            path="dummy/photo.png",
        )
        db.add(a)
        await db.commit()
        await db.refresh(a)
        return a


async def test_annotation_crud_and_renumber(_db):
    """POST/PATCH/DELETE guide annotations round-trip through state.json."""
    from cadx.api.routers.guide import (
        AnnotationIn,
        AnnotationPatch,
        add_annotation,
        delete_annotation,
        patch_annotation,
    )

    project = await _mk_project()
    asset = await _mk_pdf_asset(project)
    async with async_session() as db:
        p = await db.get(Project, project.id)
        v, _job = await spawn_guide(db, p, pdf_asset_id=asset.id)
        guide_dir = _seed_state(project, v.id)
        r = await add_annotation(
            v.id,
            AnnotationIn(
                kind="arrow", step=2, part_id="p1", direction="down", label="push"
            ),
            db,
        )
        assert r["ok"] is True and r["index"] == 0
        r = await add_annotation(
            v.id,
            AnnotationIn(kind="text", step=1, position=[1.0, 2.0, 3.0], text="note"),
            db,
        )
        assert r["index"] == 1
        p2 = await patch_annotation(
            v.id, 1, AnnotationPatch(text="updated", color="#ff0000", size=2.0), db
        )
        assert p2["annotation"]["text"] == "updated"
        assert p2["annotation"]["size"] == 2.0
        d = await delete_annotation(v.id, 0, db)
        assert d["annotations"] == 1

    from cadx.guide.state import GuideState

    st = GuideState.load(guide_dir)
    assert len(st.annotations) == 1
    assert st.annotations[0]["kind"] == "text"
    assert st.annotations[0]["color"] == "#ff0000"


async def test_annotation_axis_normalization(_db):
    """a raw vec3 is normalized; named directions resolve via _DIR_VECS."""
    from cadx import storage
    from cadx.api.routers.guide import AnnotationIn, add_annotation

    project = await _mk_project()
    asset = await _mk_pdf_asset(project)
    async with async_session() as db:
        p = await db.get(Project, project.id)
        v, _job = await spawn_guide(db, p, pdf_asset_id=asset.id)
        _seed_state(project, v.id)
        r = await add_annotation(
            v.id,
            AnnotationIn(kind="arrow", step=1, approach_axis=[0, 0, -7]),
            db,
        )
        assert r["ok"] is True
        r = await add_annotation(
            v.id, AnnotationIn(kind="arrow", step=1, direction="right"), db
        )
        assert r["ok"] is True

    from cadx.guide.state import GuideState

    st = GuideState.load(storage.version_dir(project.id, v.id) / "guide")
    assert st.annotations[0]["approach_axis"] == [0.0, 0.0, -1.0]
    assert st.annotations[1]["approach_axis"] == [1.0, 0.0, 0.0]


async def test_guide_media_upload_and_from_asset(_db):
    """multipart upload lands in guide/output/media; from-asset copies the
    project image file into the same dir; collisions get a -2 suffix."""
    import io

    from fastapi import UploadFile

    from cadx import storage
    from cadx.api.routers.guide import (
        MediaFromAsset,
        guide_media_from_asset,
        upload_guide_media,
    )

    project = await _mk_project()
    pdf = await _mk_pdf_asset(project)
    img = await _mk_img_asset(project)
    src = storage.asset_path(project.id, img.id, img.filename)
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"\x89PNG\r\n\x1a\nfakeimg")

    async with async_session() as db:
        p = await db.get(Project, project.id)
        v, _job = await spawn_guide(db, p, pdf_asset_id=pdf.id)
        guide_dir = _seed_state(project, v.id)
        up = await upload_guide_media(
            v.id, UploadFile(io.BytesIO(b"\x89PNG snap"), filename="snap.png"), db
        )
        assert up["ok"] is True and up["name"] == "snap.png"
        cp = await guide_media_from_asset(v.id, MediaFromAsset(asset_id=img.id), db)
        assert cp["ok"] is True and cp["name"] == "photo.png"
        cp2 = await guide_media_from_asset(v.id, MediaFromAsset(asset_id=img.id), db)
        assert cp2["name"] == "photo-2.png"

    media_dir = guide_dir / "output" / "media"
    assert (media_dir / "snap.png").is_file()
    assert (media_dir / "photo.png").read_bytes() == b"\x89PNG\r\n\x1a\nfakeimg"
