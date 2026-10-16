"""Hardening fixes from the live E2E test: emit dedup, crash guard, disk
guard, trace isolation, adaptive budget, critique score clamp."""

from __future__ import annotations

import pytest

import cadx.agent.worker_jobs as wj
from cadx.agent.orchestrator import (
    adaptive_max_iterations,
    stated_dim_count,
)
from cadx.agent.steps.critique import normalize_score
from cadx.agent.worker_jobs import (
    DbTrace,
    make_emit,
)
from cadx.db.models import Job, JobStatus, JobType, Project, Version, VersionKind, VersionStatus
from cadx.db.session import async_session, create_all
from cadx.llm.schemas import Critique, Discrepancy, Feature, FeatureVerdict, PartSpec

SPEC = {
    "name": "block",
    "summary": "40x20x10 block",
    "units": "mm",
    "length_x": {"value": 40.0, "source": "stated"},
    "width_y": {"value": 20.0, "source": "stated"},
    "height_z": {"value": 10.0, "source": "stated"},
    "open_questions": [],
}


@pytest.fixture(scope="module", autouse=True)
async def _db():
    await create_all()


async def _mk_job() -> Job:
    async with async_session() as db:
        p = Project(name="hardening")
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


# --------------------------------------------------------------------------- #
# emit dedup


async def test_emit_dedups_identical_messages(monkeypatch):
    job = await _mk_job()
    published: list[dict] = []

    async def fake_publish(job_id: str, event: dict) -> None:
        published.append(event)

    monkeypatch.setattr(wj.events, "publish", fake_publish)
    emit = make_emit(job.id)
    for _ in range(20):  # the mesh-server poll pattern
        await emit("meshgen", 1, "preprocessing images (0%)")
    await emit("meshgen", 1, "running hunyuan3d-2mv (9%)")
    await emit("meshgen", 1, "running hunyuan3d-2mv (9%)")

    async with async_session() as db:
        j = await db.get(Job, job.id)
        lines = [ln for ln in (j.log or "").splitlines() if ln]
        assert len(lines) == 2  # deduped, not 22
        assert "preprocessing" in lines[0]
        assert "9%" in lines[1]
    assert len(published) == 2


async def test_emit_same_message_different_iteration_passes():
    job = await _mk_job()
    emit = make_emit(job.id)
    await emit("meshgen", 1, "generating")
    await emit("meshgen", 2, "generating")
    async with async_session() as db:
        j = await db.get(Job, job.id)
        assert len((j.log or "").splitlines()) == 2


# --------------------------------------------------------------------------- #
# crash guard


async def test_guarded_marks_job_and_version_failed(monkeypatch):
    monkeypatch.setattr(wj, "_disk_free_gb", lambda: 99.0)
    job = await _mk_job()

    @wj._guarded
    async def boom(ctx, payload):
        raise RuntimeError("finalize exploded")

    result = await boom({}, {"job_id": str(job.id), "version_id": str(job.version_id)})
    assert result["ok"] is False

    async with async_session() as db:
        j = await db.get(Job, job.id)
        v = await db.get(Version, j.version_id)
        assert j.status == JobStatus.failed
        assert v.status == VersionStatus.failed
        assert "finalize exploded" in (v.error or "")


async def test_disk_guard_refuses_job(monkeypatch):
    monkeypatch.setattr(wj, "_disk_free_gb", lambda: 0.5)
    job = await _mk_job()
    ran = False

    @wj._guarded
    async def work(ctx, payload):
        nonlocal ran
        ran = True
        return {"ok": True}

    result = await work({}, {"job_id": str(job.id), "version_id": str(job.version_id)})
    assert result["ok"] is False
    assert ran is False
    async with async_session() as db:
        j = await db.get(Job, job.id)
        assert j.status == JobStatus.failed
        assert "free" in (j.log or "")


# --------------------------------------------------------------------------- #
# trace isolation


async def test_trace_record_survives_db_failure(monkeypatch):
    """A broken trace insert must never propagate into the agent step."""

    class BrokenSession:
        async def __aenter__(self):
            raise RuntimeError("fk violation — job deleted")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(wj, "session_scope", lambda: BrokenSession())
    trace = DbTrace(__import__("uuid").uuid4())
    await trace.record(
        step="codegen",
        iteration=1,
        model="m",
        messages=[],
        response_text="x",
        parsed={},
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1,
    )  # must not raise
    assert trace.records[0]["step"] == "codegen"  # memory trace still written


# --------------------------------------------------------------------------- #
# adaptive budget


def test_adaptive_budget_simple_dimensioned_part():
    spec = PartSpec(**{**SPEC, "features": [{"name": "hole", "type": "hole"}]})
    assert adaptive_max_iterations(spec, 100) == 5


def test_adaptive_budget_medium_part():
    feats = [{"name": f"f{i}", "type": "hole"} for i in range(4)]
    spec = PartSpec(**{**SPEC, "features": feats})
    assert adaptive_max_iterations(spec, 100) == 15


def test_adaptive_budget_no_stated_dims():
    spec = PartSpec(
        **{
            **SPEC,
            "length_x": {"value": 40.0, "source": "inferred"},
            "width_y": {"value": 20.0, "source": "assumed"},
            "height_z": {"value": 10.0, "source": "inferred"},
        }
    )
    assert stated_dim_count(spec) == 0
    assert adaptive_max_iterations(spec, 100) == 100


def test_adaptive_budget_many_features_unrestricted():
    feats = [{"name": f"f{i}", "type": "other"} for i in range(8)]
    spec = PartSpec(**{**SPEC, "features": feats})
    assert adaptive_max_iterations(spec, 100) == 100


def test_adaptive_budget_multi_image_never_simple():
    # image-driven parts need the full loop even when intake under-reports
    # features (weak models drop them -> stated spec looks "simple")
    spec = PartSpec(**{**SPEC, "features": [{"name": "hole", "type": "hole"}]})
    assert adaptive_max_iterations(spec, 100, n_images=2) == 100
    assert adaptive_max_iterations(spec, 100, n_images=4) == 100


def test_adaptive_budget_organic_mode_never_simple():
    spec = PartSpec(**{**SPEC, "features": []})
    assert adaptive_max_iterations(spec, 100, organic=True) == 100


def test_adaptive_budget_single_image_still_capped():
    spec = PartSpec(**{**SPEC, "features": [{"name": "hole", "type": "hole"}]})
    assert adaptive_max_iterations(spec, 100, n_images=1) == 5


# --------------------------------------------------------------------------- #
# LLM JSON-schema required fields


def test_critique_schema_requires_actionable_fields():
    # guided decoding must not allow {"score":N,"passed":false} shorthand —
    # the model is forced to emit discrepancies + per-feature verdicts
    req = Critique.model_json_schema()["required"]
    for f in ("score", "passed", "discrepancies", "feature_verdicts", "matches"):
        assert f in req


def test_flat_bottom_cue_detection():
    from cadx.agent.orchestrator import _spec_wants_flat_bottom

    def spec(summary: str) -> PartSpec:
        return PartSpec(**{**SPEC, "summary": summary})

    assert _spec_wants_flat_bottom(spec("organic top, flat back (-z)"))
    assert _spec_wants_flat_bottom(spec("rueckseite flach"))
    assert _spec_wants_flat_bottom(spec("ebene Unterseite"))
    assert not _spec_wants_flat_bottom(spec("flat top face"))
    assert not _spec_wants_flat_bottom(spec("rounded back"))


def test_flat_back_ops_appends_smooth_preserve():
    from cadx.agent.orchestrator import _flat_back_ops
    from cadx.cad.mesh_ops import MeshPostParams

    ops = _flat_back_ops(MeshPostParams(), bow_mm=5.0, smooth_iters=10)
    assert [o["op"] for o in ops] == ["flatten_bottom", "smooth_preserve"]
    assert ops[0]["args"]["height_mm"] == 5.0
    assert ops[1]["args"]["iterations"] == 10
    # smoothing disabled → flatten only
    ops0 = _flat_back_ops(MeshPostParams(), bow_mm=5.0, smooth_iters=0)
    assert [o["op"] for o in ops0] == ["flatten_bottom"]


def test_partspec_schema_requires_features_and_roles():
    schema = PartSpec.model_json_schema()
    for f in ("length_x", "width_y", "height_z", "features", "image_roles"):
        assert f in schema["required"]
    assert "name" in Feature.model_json_schema()["required"]
    assert "status" in FeatureVerdict.model_json_schema()["required"]
    assert "severity" in Discrepancy.model_json_schema()["required"]


# --------------------------------------------------------------------------- #
# critique score clamp


def test_normalize_score_zero_with_no_issues():
    c = Critique(score=0, passed=False, discrepancies=[], confidence=0.0)
    assert normalize_score(c).score == 60


def test_normalize_score_ceiling_by_severity():
    c = Critique(
        score=92,
        passed=False,
        discrepancies=[
            Discrepancy(what="a", severity="major"),
            Discrepancy(what="b", severity="minor"),
        ],
        confidence=0.5,
    )
    assert normalize_score(c).score == 65  # 100 - 25 - 10


def test_normalize_score_consistent_untouched():
    c = Critique(score=40, passed=False, discrepancies=[Discrepancy(what="a", severity="major")])
    assert normalize_score(c).score == 40  # already below the 75 ceiling


# --------------------------------------------------------------------------- #
# measured-vs-claimed 'missing' verdicts


def test_verify_missing_verdict_refuted_by_measurement():
    """Firesteel E2E: the critic claimed 'tube missing' while the bore was
    measured — a refuted 'missing' must downgrade to 'wrong', and the
    discrepancy repeating the claim loses its major severity."""
    from cadx.agent.steps.critique import _verify_missing_verdicts

    spec = PartSpec(
        **{
            **SPEC,
            "features": [
                {"name": "boss bore", "type": "hole", "count": 1, "dims": {"diameter": 8}},
                {"name": "grip ribs", "type": "rib", "count": 4},
            ],
        }
    )
    c = Critique(
        score=20,
        passed=False,
        feature_verdicts=[
            FeatureVerdict(name="boss bore", status="missing", note="no bore visible"),
            FeatureVerdict(name="grip ribs", status="missing", note="ribs absent"),
        ],
        discrepancies=[
            Discrepancy(what="boss bore is missing", severity="major"),
            Discrepancy(what="grip ribs are missing", severity="major"),
        ],
        confidence=0.5,
    )
    features = {"total_holes": 1, "slots": 0}  # bore measured present
    _verify_missing_verdicts(c, spec, features)
    by_name = {v.name: v for v in c.feature_verdicts}
    assert by_name["boss bore"].status == "wrong"  # refuted by measurement
    assert by_name["grip ribs"].status == "missing"  # not a measurable kind
    bore_d = next(d for d in c.discrepancies if "boss bore" in d.what)
    rib_d = next(d for d in c.discrepancies if "grip ribs" in d.what)
    assert bore_d.severity == "minor"
    assert rib_d.severity == "major"


def test_verify_missing_verdict_untouched_when_absent():
    from cadx.agent.steps.critique import _verify_missing_verdicts

    spec = PartSpec(**{**SPEC, "features": [{"name": "mount", "type": "hole", "count": 2}]})
    c = Critique(
        score=30,
        passed=False,
        feature_verdicts=[FeatureVerdict(name="mount", status="missing", note="")],
        discrepancies=[Discrepancy(what="mount is missing", severity="major")],
        confidence=0.5,
    )
    _verify_missing_verdicts(c, spec, {"total_holes": 0})
    assert c.feature_verdicts[0].status == "missing"
    assert c.discrepancies[0].severity == "major"


def test_verify_wrong_verdict_refuted_when_dims_match():
    """D.1: 'wrong' is refuted only when spec dims match the measurement —
    Ø AND axis. A wrong-diameter or wrong-axis verdict stands."""
    from cadx.agent.steps.critique import _verify_wrong_verdicts

    spec = PartSpec(
        **{
            **SPEC,
            "features": [
                {
                    "name": "boss bore",
                    "type": "hole",
                    "count": 1,
                    "axis": "y",
                    "dims": {"diameter": 8},
                }
            ],
        }
    )
    features = {
        "total_holes": 1,
        "hole_groups": [{"diameter": 8.2, "count": 1, "axis": "+y", "elongated": False}],
    }
    c = Critique(
        score=40,
        passed=False,
        feature_verdicts=[FeatureVerdict(name="boss bore", status="wrong", note="misplaced")],
        discrepancies=[Discrepancy(what="boss bore is wrong", severity="major")],
        confidence=0.5,
    )
    _verify_wrong_verdicts(c, spec, features)
    assert c.feature_verdicts[0].status == "unsure"
    assert c.discrepancies[0].severity == "minor"

    # different diameter → verdict stands
    c2 = Critique(
        score=40,
        passed=False,
        feature_verdicts=[FeatureVerdict(name="boss bore", status="wrong", note="")],
        discrepancies=[Discrepancy(what="boss bore is wrong", severity="major")],
        confidence=0.5,
    )
    _verify_wrong_verdicts(
        c2,
        spec,
        {
            "total_holes": 1,
            "hole_groups": [{"diameter": 12.0, "count": 1, "axis": "+y", "elongated": False}],
        },
    )
    assert c2.feature_verdicts[0].status == "wrong"
    assert c2.discrepancies[0].severity == "major"

    # different axis → verdict stands
    c3 = Critique(
        score=40,
        passed=False,
        feature_verdicts=[FeatureVerdict(name="boss bore", status="wrong", note="")],
        discrepancies=[Discrepancy(what="boss bore is wrong", severity="major")],
        confidence=0.5,
    )
    _verify_wrong_verdicts(
        c3,
        spec,
        {
            "total_holes": 1,
            "hole_groups": [{"diameter": 8.2, "count": 1, "axis": "+z", "elongated": False}],
        },
    )
    assert c3.feature_verdicts[0].status == "wrong"


# --------------------------------------------------------------------------- #
# mesh fit-to-stated rescue


def _stated_spec(x=119.0, y=38.0, z=8.0):
    return PartSpec(
        **{
            **SPEC,
            "length_x": {"value": x, "source": "stated"},
            "width_y": {"value": y, "source": "stated"},
            "height_z": {"value": z, "source": "stated"},
        }
    )


def test_fit_ops_plate_too_thick_scales_z():
    """Live case: 119x73x27 mesh vs 119x38x8 spec → bounded per-axis scale.

    A slice would keep only the top crust of the fat mesh — z-scale preserves
    the shape (relief gets shallower, which matches a thin scale)."""
    from cadx.agent.orchestrator import _fit_ops_to_stated

    ops, _note = _fit_ops_to_stated(_stated_spec(), [119.0, 73.0, 27.0])
    assert [o.op for o in ops] == ["scale"]
    assert ops[0].args["y"] == round(38 / 73, 4)
    assert ops[0].args["z"] == round(8 / 27, 4)
    assert "x" not in ops[0].args


def test_fit_ops_exact_dims_no_ops():
    from cadx.agent.orchestrator import _fit_ops_to_stated

    ops, _ = _fit_ops_to_stated(_stated_spec(), [119.0, 38.0, 8.0])
    assert ops == []


def test_fit_ops_extreme_factor_rejects():
    """10x off is the wrong object, not a proportion error."""
    from cadx.agent.orchestrator import _fit_ops_to_stated

    ops, _ = _fit_ops_to_stated(_stated_spec(), [1190.0, 73.0, 27.0])
    assert ops == []


def test_fit_ops_thin_mesh_scales_up():
    from cadx.agent.orchestrator import _fit_ops_to_stated

    ops, _ = _fit_ops_to_stated(_stated_spec(), [119.0, 38.0, 5.0])
    assert [o.op for o in ops] == ["scale"]
    assert ops[0].args["z"] == 1.6


def test_fit_ops_blocky_part_scales_z():
    """Non-plate parts (height not smallest stated) get Z scaled, not sliced."""
    from cadx.agent.orchestrator import _fit_ops_to_stated

    spec = _stated_spec(x=30.0, y=30.0, z=40.0)
    ops, _ = _fit_ops_to_stated(spec, [30.0, 30.0, 90.0])
    assert [o.op for o in ops] == ["scale"]
    assert ops[0].args["z"] == round(40 / 90, 4)


def test_fit_ops_inflated_mesh_strong_z_squash():
    """Live case: 119x73.9x52.1 blob vs 119x38x8 spec — hunyuan3d-2mv inflates
    flat parts ~6x; fz=0.153 must still fit (degenerate-volume guard catches
    crusts downstream)."""
    from cadx.agent.orchestrator import _fit_ops_to_stated

    ops, _note = _fit_ops_to_stated(_stated_spec(), [119.0, 73.92, 52.12])
    assert [o.op for o in ops] == ["scale"]
    assert ops[0].args["y"] == round(38 / 73.92, 4)
    assert ops[0].args["z"] == round(8 / 52.12, 4)


def test_fit_ops_z_extreme_still_rejects():
    """fz below 0.12 (8x+ off) is the wrong object even for a flat part."""
    from cadx.agent.orchestrator import _fit_ops_to_stated

    ops, _ = _fit_ops_to_stated(_stated_spec(), [119.0, 38.0, 80.0])
    assert ops == []


async def test_trace_orphan_circuit_breaker(monkeypatch):
    """FK violation → DbTrace goes dead, stops hitting the DB, keeps memory."""

    class OrphanSession:
        calls = 0

        async def __aenter__(self):
            OrphanSession.calls += 1
            raise RuntimeError(
                'insert or update on table "agenttrace" violates foreign key '
                'constraint "agenttrace_job_id_fkey"'
            )

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(wj, "session_scope", lambda: OrphanSession())
    trace = DbTrace(__import__("uuid").uuid4())
    kw = dict(
        step="codegen",
        iteration=1,
        model="m",
        messages=[],
        response_text="x",
        parsed={},
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1,
    )
    await trace.record(**kw)
    assert trace._dead
    await trace.record(**kw)
    assert OrphanSession.calls == 1  # second record never touched the DB
    assert len(trace.records) == 2  # memory trace still grows


async def test_executor_startup_does_not_reap_agent_jobs(monkeypatch):
    """cad_startup must not reap running agent-queue jobs — every Job row is
    enqueued on the 'agent' queue, so only an agent-worker restart orphans
    them. Regression test for the shared agent_startup reaping live jobs
    whenever the cad-executor container was rebuilt mid-run."""
    import arq.connections

    from cadx.queue import AgentWorkerSettings, CadWorkerSettings, agent_startup, cad_startup

    assert AgentWorkerSettings.on_startup is agent_startup
    assert CadWorkerSettings.on_startup is cad_startup

    async def fake_pool(*a, **k):
        return None

    monkeypatch.setattr(arq.connections, "create_pool", fake_pool)

    job = await _mk_job()
    async with async_session() as db:
        (await db.get(Job, job.id)).status = JobStatus.running
        await db.commit()

    await cad_startup({})  # executor restart — must not touch the job
    async with async_session() as db:
        assert (await db.get(Job, job.id)).status == JobStatus.running

    await agent_startup({})  # agent worker restart — reaps it
    async with async_session() as db:
        assert (await db.get(Job, job.id)).status == JobStatus.failed


async def test_cad_tasks_offload_blocking_payloads(monkeypatch, tmp_path):
    """arq tasks must run their heavy sync payloads in a worker thread —
    a blocking call on the event loop starves dequeue/health/cancel for the
    whole worker. Regression: a ~50min SIMP run froze the cad queue, so
    process_mesh and execute_version calls timed out without ever starting."""
    import asyncio
    import time

    import cadx.cad.tasks as tasks

    def slow(*args: object, **kw: object) -> dict:
        time.sleep(0.15)
        return {"ok": True}

    monkeypatch.setattr(tasks, "run_execute_payload", slow)
    monkeypatch.setattr(tasks, "_run_mesh_payload_subprocess", slow)
    monkeypatch.setattr(tasks, "_artifact_dir", lambda p: tmp_path)
    monkeypatch.setattr(tasks, "run_fem_payload", slow)
    monkeypatch.setattr(tasks, "run_optimize_payload", slow)

    async def run_one(coro):
        ticked = False

        async def ticker() -> None:
            nonlocal ticked
            await asyncio.sleep(0.02)
            ticked = True

        result, _ = await asyncio.gather(coro({}, {"job_id": None}), ticker())
        assert result["ok"] is True
        assert ticked, f"{coro.__name__} blocked the event loop"

    for coro in (
        tasks.execute_version,
        tasks.process_mesh,
        tasks.fem_analyze,
        tasks.optimize,
    ):
        await run_one(coro)


def test_mesh_payload_subprocess_surfaces_child_failure(tmp_path):
    """A dying mesh-payload child must return ok=False — never take the arq
    worker down with it (regression: accumulated in-process RSS tripped the
    container cgroup, SIGKILLing worker + orphaning the job; zombie arq retries
    then killed the next worker too)."""
    import cadx.cad.tasks as tasks

    d = tasks._run_mesh_payload_subprocess({"project_id": "x", "mode": "final"}, tmp_path)
    assert d["ok"] is False
    assert d["error"]
