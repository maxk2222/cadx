"""Structured live job state (Job.live): make_emit rolls emit() fields into a
persisted snapshot (iteration budget, score history, step timings, tokens,
sub-progress), publishes it on every SSE event, and exposes it via JobOut.
Plus the live-renders manifest + ?view= preview selection."""

from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

import cadx.agent.worker_jobs as wj
from cadx import storage
from cadx.agent.worker_jobs import make_emit
from cadx.db.models import Job, JobType, Project, Version, VersionKind
from cadx.db.session import async_session, create_all


@pytest.fixture(scope="module", autouse=True)
async def _db():
    await create_all()


async def _mk_job(project_name: str = "live") -> Job:
    async with async_session() as db:
        p = Project(name=project_name)
        db.add(p)
        await db.flush()
        v = Version(project_id=p.id, number=1, kind=VersionKind.generate, prompt="x")
        db.add(v)
        await db.flush()
        j = Job(version_id=v.id, type=JobType.generate)
        db.add(j)
        await db.commit()
        await db.refresh(j)
        return j


async def _live(job_id) -> dict:
    async with async_session() as db:
        return dict((await db.get(Job, job_id)).live or {})


async def test_emit_rolls_fields_into_job_live(monkeypatch):
    published: list[dict] = []

    async def fake_publish(job_id: str, event: dict) -> None:
        published.append(event)

    monkeypatch.setattr(wj.events, "publish", fake_publish)

    job = await _mk_job()
    emit = make_emit(job.id)
    await emit("plan", 0, "approach: cad — ok", max_iterations=15)
    await emit("codegen", 1, "writing code")
    await emit("critique", 1, "score 62/100", score=62)
    await emit("meshgen", 2, "running hunyuan (40%)", sub_progress=0.4, sub_label="hunyuan")

    live = await _live(job.id)
    assert live["max_iterations"] == 15
    assert live["scores"] == [{"it": 1, "score": 62}]
    assert live["step"] == "meshgen"
    assert live["sub_progress"] == 0.4
    assert live["sub_label"] == "hunyuan"
    assert [h["step"] for h in live["step_history"]] == ["plan", "codegen", "critique"]

    # real progress: (iteration - 1 + sub_progress) / max_iterations
    async with async_session() as db:
        j = await db.get(Job, job.id)
        assert j.progress == pytest.approx((2 - 1 + 0.4) / 15)

    # every step event carries the live snapshot + computed progress
    assert published[-1]["live"]["max_iterations"] == 15
    assert published[-1]["live"]["scores"] == [{"it": 1, "score": 62}]


async def test_emit_sub_progress_cleared_on_step_change():
    job = await _mk_job()
    emit = make_emit(job.id)
    await emit("meshgen", 1, "gen (50%)", sub_progress=0.5, sub_label="gen")
    await emit("meshprocess", 1, "post-processing")
    live = await _live(job.id)
    assert live["step"] == "meshprocess"
    assert "sub_progress" not in live
    assert "sub_label" not in live


async def test_emit_score_dedup_per_iteration():
    job = await _mk_job()
    emit = make_emit(job.id)
    await emit("critique", 1, "score 60", score=60)
    await emit("critique", 1, "score 60 again", score=60)
    live = await _live(job.id)
    assert live["scores"] == [{"it": 1, "score": 60}]


async def test_emit_progress_fallback_without_budget():
    job = await _mk_job()
    emit = make_emit(job.id)
    await emit("intake", 0, "analysing")
    await emit("plan", 0, "planning")
    async with async_session() as db:
        j = await db.get(Job, job.id)
        assert j.progress == pytest.approx(0.10)  # legacy +0.05/event bump
        assert j.live is not None and "max_iterations" not in j.live


async def test_emit_tokens_fn():
    job = await _mk_job()
    emit = make_emit(job.id, tokens_fn=lambda: {"prompt_tokens": 10, "completion_tokens": 5})
    await emit("intake", 0, "x")
    live = await _live(job.id)
    assert live["tokens"] == {"prompt_tokens": 10, "completion_tokens": 5}


async def test_emit_metrics_merge_and_none_filter(monkeypatch):
    published: list[dict] = []

    async def fake_publish(job_id: str, event: dict) -> None:
        published.append(event)

    monkeypatch.setattr(wj.events, "publish", fake_publish)

    job = await _mk_job()
    emit = make_emit(job.id)
    await emit("optimize", 1, "fem probe", metrics={"probe_sf": 1.8})
    await emit("optimize", 2, "probe again", metrics={"probe_sf": 1.4, "worst_sf": None})
    await emit("fem", 2, "verify done", metrics={"worst_sf": 1.4})

    live = await _live(job.id)
    # latest value wins per name; the None entry dropped instead of erasing
    assert live["metrics"] == {"probe_sf": 1.4, "worst_sf": 1.4}
    # metrics alone must not create a runs table
    assert "runs" not in live

    # raw field lands on the event top-level, merged snapshot under .live
    assert published[0]["metrics"] == {"probe_sf": 1.8}
    assert published[-1]["metrics"] == {"worst_sf": 1.4}
    assert published[-1]["live"]["metrics"] == {"probe_sf": 1.4, "worst_sf": 1.4}


async def test_emit_runs_append_and_cap_40(monkeypatch):
    published: list[dict] = []

    async def fake_publish(job_id: str, event: dict) -> None:
        published.append(event)

    monkeypatch.setattr(wj.events, "publish", fake_publish)

    job = await _mk_job()
    emit = make_emit(job.id)
    for i in range(45):
        await emit("optimize", i, f"run {i} done", run={"index": i, "type": "topopt"})

    live = await _live(job.id)
    assert len(live["runs"]) == 40
    assert live["runs"][0]["index"] == 5  # oldest 5 evicted by the cap
    assert live["runs"][-1] == {"index": 44, "type": "topopt"}
    # run rows don't touch the scalar metrics map
    assert "metrics" not in live

    # raw run field on each event + persisted snapshot under .live
    assert published[-1]["run"] == {"index": 44, "type": "topopt"}
    assert len(published[-1]["live"]["runs"]) == 40


async def test_emit_metrics_and_run_leave_existing_fields(monkeypatch):
    job = await _mk_job()
    emit = make_emit(job.id)
    await emit("plan", 0, "approach", max_iterations=10)
    await emit("critique", 1, "score 62", score=62)
    await emit(
        "optimize",
        2,
        "run 0 done",
        metrics={"probe_sf": 2.0},
        run={"index": 0, "type": "topopt"},
        sub_progress=0.5,
        sub_label="simp",
    )

    live = await _live(job.id)
    # new fields alongside the existing whitelist/score behaviour
    assert live["max_iterations"] == 10
    assert live["scores"] == [{"it": 1, "score": 62}]
    assert live["sub_progress"] == 0.5 and live["sub_label"] == "simp"
    assert live["metrics"] == {"probe_sf": 2.0}
    assert live["runs"] == [{"index": 0, "type": "topopt"}]
    assert [h["step"] for h in live["step_history"]] == ["plan", "critique"]

    async with async_session() as db:
        j = await db.get(Job, job.id)
        # progress still computed from the iteration budget, not the new fields
        assert j.progress == pytest.approx((2 - 1 + 0.5) / 10)


# --------------------------------------------------------------------------- #
# API-level: live-renders manifest + ?view= preview + global /jobs list


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
async def client():
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_live_renders_and_view_selection(client):
    async with async_session() as db:
        p = Project(name="renders")
        db.add(p)
        await db.flush()
        v = Version(project_id=p.id, number=1, kind=VersionKind.generate, prompt="x")
        db.add(v)
        await db.flush()
        vid, pid = v.id, p.id
        await db.commit()

    r = await client.get(f"/api/versions/{vid}/live-renders")
    assert r.status_code == 200
    assert r.json()["views"] == [] and r.json()["iteration"] is None

    # fake a running job's iteration-3 renders on the shared DATA_DIR
    rdir = storage.iteration_dir(pid, vid, 3) / "renders"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "iso.png").write_bytes(_png_bytes())
    (rdir / "front.png").write_bytes(_png_bytes())
    (rdir.parent / "contact_sheet.png").write_bytes(_png_bytes())

    r = await client.get(f"/api/versions/{vid}/live-renders")
    assert r.status_code == 200
    assert r.json()["iteration"] == 3
    assert sorted(r.json()["views"]) == ["contact_sheet", "front", "iso"]
    assert r.json()["final"] is False

    r = await client.get(f"/api/versions/{vid}/preview?view=front")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "no-store"
    r = await client.get(f"/api/versions/{vid}/preview?view=contact_sheet")
    assert r.status_code == 200
    r = await client.get(f"/api/versions/{vid}/preview?view=../escape")
    assert r.status_code == 400
    r = await client.get(f"/api/versions/{vid}/preview?view=missing")
    assert r.status_code == 404

    # finalized renders win over iteration dirs
    fdir = storage.version_dir(pid, vid) / "renders"
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "iso.png").write_bytes(_png_bytes())
    r = await client.get(f"/api/versions/{vid}/live-renders")
    assert r.json()["final"] is True and r.json()["iteration"] is None


async def test_jobs_list_endpoint(client):
    job = await _mk_job("jobslist")
    r = await client.get("/api/jobs")
    assert r.status_code == 200
    hit = next((x for x in r.json() if x["id"] == str(job.id)), None)
    assert hit is not None
    assert hit["project_name"] == "jobslist"
    assert hit["version_number"] == 1
    assert hit["version_status"] == "queued"
    # fresh job is queued → included in the active filter
    r = await client.get("/api/jobs", params={"active": "true"})
    assert any(x["id"] == str(job.id) for x in r.json())


async def test_version_summary_active_job(client):
    job = await _mk_job("activetree")
    async with async_session() as db:
        v = await db.get(Version, job.version_id)
        pid = v.project_id
        live = {"max_iterations": 9}
        (await db.get(Job, job.id)).live = live
        await db.commit()
    tree = (await client.get(f"/api/projects/{pid}/versions")).json()
    vs = next(x for x in tree if x["id"] == str(job.version_id))
    assert vs["active_job"]["id"] == str(job.id)
    assert vs["active_job"]["status"] == "queued"
    assert vs["active_job"]["max_iterations"] == 9
