"""Structured live job state (Job.live): make_emit rolls emit() fields into a
persisted snapshot (iteration budget, score history, step timings, tokens,
sub-progress), publishes it on every SSE event, and exposes it via JobOut.
Plus the live-renders manifest + ?view= preview selection."""

from __future__ import annotations

import io
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

import cadx.agent.worker_jobs as wj
from cadx import storage
from cadx.agent.worker_jobs import make_emit
from cadx.db.models import Job, JobStatus, JobType, Project, Version, VersionKind
from cadx.db.session import async_session, create_all


@pytest.fixture(scope="module", autouse=True)
async def _db():
    await create_all()


async def _mk_job(project_name: str = "live") -> Job:
    async with async_session() as db:
        p = Project(name=project_name)
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id, part_id=uuid.uuid4(), number=1, kind=VersionKind.generate, prompt="x"
        )
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
        v = Version(
            project_id=p.id, part_id=uuid.uuid4(), number=1, kind=VersionKind.generate, prompt="x"
        )
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


async def _mk_version(name: str) -> tuple[uuid.UUID, uuid.UUID]:
    async with async_session() as db:
        p = Project(name=name)
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id, part_id=uuid.uuid4(), number=1, kind=VersionKind.generate, prompt="x"
        )
        db.add(v)
        await db.flush()
        vid, pid = v.id, p.id
        await db.commit()
    return vid, pid


def _touch(p: Path, when: float) -> None:
    os.utime(p, (when, when))


async def test_live_renders_attempt_subdirs(client):
    """Per-attempt artifact dirs (a1, a2, parametrize, …) must be discovered —
    the executor never writes iterations/<n>/renders directly anymore."""
    vid, pid = await _mk_version("renders-attempts")

    it = storage.iteration_dir(pid, vid, 2)
    a1 = it / "a1" / "renders"
    a1.mkdir(parents=True)
    (a1 / "iso.png").write_bytes(_png_bytes())
    (it / "a1" / "contact_sheet.png").write_bytes(_png_bytes())

    r = await client.get(f"/api/versions/{vid}/live-renders")
    assert r.status_code == 200
    assert r.json()["iteration"] == 2
    assert sorted(r.json()["views"]) == ["contact_sheet", "iso"]
    assert r.json()["final"] is False

    r = await client.get(f"/api/versions/{vid}/preview?view=iso")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    # no-view fallback must reach into the attempt dir too
    r = await client.get(f"/api/versions/{vid}/preview")
    assert r.status_code == 200

    # a newer attempt outranks the earlier one — its extra view proves the pick
    a2 = it / "a2" / "renders"
    a2.mkdir(parents=True)
    (a2 / "iso.png").write_bytes(_png_bytes())
    (a2 / "top.png").write_bytes(_png_bytes())
    newer = time.time() + 60
    _touch(a2 / "iso.png", newer)
    _touch(a2 / "top.png", newer)

    r = await client.get(f"/api/versions/{vid}/live-renders")
    # union across attempts: a2's fresh views plus a1's contact_sheet
    assert sorted(r.json()["views"]) == ["contact_sheet", "iso", "top"]
    assert r.json()["iteration"] == 2
    # per-view mtimes: a2's iso/top are newer than a1's contact_sheet
    mtimes = r.json()["mtimes"]
    assert mtimes["top"] == pytest.approx(newer)
    assert mtimes["contact_sheet"] < mtimes["iso"]


async def test_live_renders_fem_only_fresh_set(client):
    """When the freshest write is a fem/regions probe (fem-only views), the
    manifest must still offer the model renders from older sets, ?view= must
    resolve them, and the no-view preview must prefer a model render."""
    vid, pid = await _mk_version("renders-femonly")

    # older model render set
    it = storage.iteration_dir(pid, vid, 5)
    rdir = it / "renders"
    rdir.mkdir(parents=True)
    (rdir / "iso.png").write_bytes(_png_bytes())
    (rdir / "front.png").write_bytes(_png_bytes())
    old = time.time() - 120
    _touch(rdir / "iso.png", old)
    _touch(rdir / "front.png", old)

    # newest write: fem region previews only — with a space in the case name
    fdir = it / "fem"
    fdir.mkdir(parents=True)
    (fdir / "regions_chain pull.png").write_bytes(_png_bytes())
    (fdir / "vonmises_chain pull.png").write_bytes(_png_bytes())

    r = await client.get(f"/api/versions/{vid}/live-renders")
    d = r.json()
    assert d["iteration"] == 5 and d["final"] is False
    assert sorted(d["views"]) == [
        "fem:regions_chain pull",
        "fem:vonmises_chain pull",
        "front",
        "iso",
    ]

    # view names with spaces must resolve (dict lookup — not a path join)
    r = await client.get(f"/api/versions/{vid}/preview?view=fem:regions_chain pull")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"

    # the no-view preview prefers the model render over the fem plot
    r = await client.get(f"/api/versions/{vid}/preview")
    assert r.status_code == 200
    # iso was in _PREVIEW_PREFER — it wins over the fem-only fresh set
    body = r.content
    assert len(body) > 0


async def test_version_detail_active_job(client):
    """GET /versions/{id} must surface the newest non-terminal job so the
    page can restore the live stream/preview after a reload."""
    vid, _pid = await _mk_version("detail-activejob")

    r = await client.get(f"/api/versions/{vid}")
    assert r.status_code == 200
    assert r.json()["active_job"] is None

    async with async_session() as db:
        job = Job(
            version_id=vid,
            type=JobType.generate,
            status=JobStatus.running,
            step="codegen",
            iteration=3,
        )
        db.add(job)
        await db.flush()
        jid = job.id
        await db.commit()

    r = await client.get(f"/api/versions/{vid}")
    aj = r.json()["active_job"]
    assert aj is not None
    assert aj["status"] == "running" and aj["step"] == "codegen" and aj["iteration"] == 3

    # terminal jobs must not resurface as active
    async with async_session() as db:
        j = await db.get(Job, jid)
        j.status = JobStatus.succeeded
        await db.commit()

    r = await client.get(f"/api/versions/{vid}")
    assert r.json()["active_job"] is None


async def test_live_renders_aux_and_rerun_freshness(client):
    """orient_sheet/fem views are listed, and a re-run on a version with
    finalized renders must surface the fresh iteration (always newest)."""
    vid, pid = await _mk_version("renders-rerun")

    # finalized root renders (old)
    rdir = storage.version_dir(pid, vid) / "renders"
    rdir.mkdir(parents=True)
    (rdir / "iso.png").write_bytes(_png_bytes())
    old = time.time() - 3600
    _touch(rdir / "iso.png", old)

    r = await client.get(f"/api/versions/{vid}/live-renders")
    assert r.json()["final"] is True

    # a new run's attempt renders are newer than the finalized copy
    base = storage.iteration_dir(pid, vid, 7) / "a3"
    (base / "renders").mkdir(parents=True)
    (base / "fem").mkdir()
    (base / "renders" / "iso.png").write_bytes(_png_bytes())
    (base / "orient_sheet.png").write_bytes(_png_bytes())
    (base / "fem" / "stress.png").write_bytes(_png_bytes())

    r = await client.get(f"/api/versions/{vid}/live-renders")
    d = r.json()
    assert d["final"] is False and d["iteration"] == 7
    assert sorted(d["views"]) == ["fem:stress", "iso", "orient_sheet"]

    r = await client.get(f"/api/versions/{vid}/preview?view=orient_sheet")
    assert r.status_code == 200
    r = await client.get(f"/api/versions/{vid}/preview?view=fem:stress")
    assert r.status_code == 200


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


async def test_terminal_status_closes_live_step(client):
    """Regression: a job reaching a terminal status must fold live.step into
    step_history — otherwise the persisted snapshot keeps a 'running' step
    forever and every step display computes now - step_started_ts."""
    vid, _pid = await _mk_version("live-close")
    t0 = datetime.now(UTC) - timedelta(seconds=25)
    async with async_session() as db:
        job = Job(
            version_id=vid,
            type=JobType.fem,
            status=JobStatus.running,
            started_at=datetime.now(UTC) - timedelta(seconds=30),
            live={
                "step": "fem",
                "step_started_ts": t0.isoformat(),
                "step_history": [{"step": "voxelize", "it": 0, "dt_s": 1.5}],
                "sub_progress": 0.4,
            },
        )
        db.add(job)
        await db.commit()
        jid = job.id
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        await db.commit()

    r = await client.get(f"/api/jobs/{jid}")
    assert r.status_code == 200
    live = r.json()["live"]
    assert live.get("step") is None
    assert live.get("step_started_ts") is None
    assert live.get("sub_progress") is None
    hist = live["step_history"]
    assert [h["step"] for h in hist] == ["voxelize", "fem"]
    assert 20 <= hist[-1]["dt_s"] <= 60

    # an already-terminal job re-checked must not append twice
    async with async_session() as db:
        j = await db.get(Job, jid)
        j.progress = 1.0
        await db.commit()
    r = await client.get(f"/api/jobs/{jid}")
    assert [h["step"] for h in r.json()["live"]["step_history"]] == ["voxelize", "fem"]
