"""Optimize-session salvage — when the agent exhausts its budget without
done(), the best completed run is committed instead of discarding all
executor work and re-running the deterministic single-shot path."""

from __future__ import annotations

import pytest

from cadx.agent.steps.optimize import _best_run_index, _opt_overrides
from cadx.agent.toolspecs import ToolSession


def _session_with_runs() -> ToolSession:
    from cadx.llm.fake import FakeLLM
    from tests.unit.test_session_condense import _ctx

    return ToolSession(_ctx(FakeLLM([{"content": "x"}])), kind="optimize")


def test_recut_bores_defaults_on_and_can_be_disabled() -> None:
    from cadx.llm.schemas import OptimizeSpec

    assert OptimizeSpec().recut_bores is True
    assert _opt_overrides({"recut_bores": False}) == {"recut_bores": False}


async def test_topopt_run_merges_request_opt_for_unexpressible_fields(tmp_path) -> None:
    """The robot_arm_infill_guided failure: the eval stage declares
    structure_mode='infill' + infill_scope='guided' but the agent called
    topopt_run — request fields the tool cannot express must survive into
    the executor payload instead of silently degrading to plain SIMP."""
    import itertools
    import uuid
    from typing import Any

    from cadx.agent.steps.optimize import _make_topopt_fn, _SessionOps
    from cadx.llm.fake import FakeLLM
    from cadx.llm.schemas import Dim, PartSpec
    from tests.unit.test_session_condense import _ctx

    captured: dict[str, Any] = {}

    class _CaptureExec:
        async def optimize(self, payload):
            captured.update(payload)
            return {"ok": True}

    ctx = _ctx(FakeLLM([]))
    ctx.executor = _CaptureExec()
    spec = PartSpec(
        name="bracket",
        summary="",
        length_x=Dim(value=100, source="stated"),
        width_y=Dim(value=80, source="stated"),
        height_z=Dim(value=40, source="stated"),
    )
    ops = _SessionOps(
        ctx=ctx,
        base={
            "job_id": "t",
            "project_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
            "mesh_rel": "x",
            "runtime_settings": {},
            "name": "model",
        },
        spec=spec,
        spec_cases=[],
        default_material="",
        alloc=itertools.count(1),
        contact_sheet=None,
        request_opt={
            "structure_mode": "infill",
            "infill_scope": "guided",
            "infill_guide_solver": "beso",
            "target_volume_fraction": 0.5,
        },
    )
    fn = _make_topopt_fn(ops)
    res = await fn({"load_cases": []})
    assert res["ok"] is True
    opt = captured["optimize"]
    assert opt["structure_mode"] == "infill"
    assert opt["infill_scope"] == "guided"
    assert opt["infill_guide_solver"] == "beso"
    assert opt["target_volume_fraction"] == 0.5
    # tool args still override the request baseline
    res = await fn({"load_cases": [], "infill_scope": "full"})
    assert res["ok"] is True
    assert captured["optimize"]["infill_scope"] == "full"


def test_successful_in_pipeline_recut_prevents_duplicate_auto_refine() -> None:
    import uuid

    from cadx.agent.worker_jobs import _injob_refine_committed
    from cadx.db.models import Version, VersionKind

    version = Version(
        project_id=uuid.uuid4(),
        part_id=uuid.uuid4(),
        number=2,
        kind=VersionKind.optimize,
        mesh={"topopt": {"recut": {"applied": [{"name": "mount"}], "verify_misses": []}}},
    )
    assert _injob_refine_committed(version) is True
    version.mesh["topopt"]["recut"]["verify_misses"] = [{"name": "mount"}]
    assert _injob_refine_committed(version) is False


def test_best_run_prefers_sf_export() -> None:
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 1.4},
                "fem": {"cases": [{"safety_factor": 3.0}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 2.9},
                "fem": {"cases": [{"safety_factor": 1.1}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 1


def test_best_run_falls_back_to_worst_case_sf() -> None:
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {},
                "fem": {"cases": [{"safety_factor": 4.0}, {"safety_factor": 1.2}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {},
                "fem": {"cases": [{"safety_factor": 0.9}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 0  # worst-case SF 1.2 beats 0.9


def test_best_run_none_without_metrics() -> None:
    s = _session_with_runs()
    s.struct_runs.append({"index": 0, "res": {"topopt": {}}, "params": {}})
    assert _best_run_index(s) is None
    assert _best_run_index(_session_with_runs()) is None


def test_best_run_skips_disconnected_and_invalid() -> None:
    """A high-SF run whose export mesh is several bodies — or whose export
    verification was flagged degenerate — must never be salvaged: the SF
    describes only the loaded island, not a printable part."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 19.2, "n_components": 26},
                "fem": {"cases": [{"safety_factor": 19.2}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 3.1, "n_bodies_export": 2},
                "fem": {"cases": [{"safety_factor": 3.1}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 2,
            "res": {
                "topopt": {"sf_export": 5.5, "verification": {"valid": False}},
                "fem": {"cases": [{"safety_factor": 5.5}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 3,
            "res": {
                "topopt": {"sf_export": 2.0, "n_components": 1, "n_bodies_export": 1},
                "fem": {"cases": [{"safety_factor": 2.0}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 3  # only the connected, verified run counts


def test_best_run_skips_verify_spec_fail() -> None:
    """A canonical FAIL (e.g. envelope overshoot) disqualifies a run even
    when its safety factor is the best in the table."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 9.0},
                "fem": {"cases": [{"safety_factor": 9.0}]},
                "verify_spec": {"status": "FAIL", "checks": [{"name": "envelope_x"}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 1.8},
                "fem": {"cases": [{"safety_factor": 1.8}]},
                "verify_spec": {"status": "PASS"},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 1


def test_best_run_skips_broken_load_path() -> None:
    """load_path_ok=false means at least one case cannot reach its fixtures
    through the printed part — structurally disconnected even when
    n_bodies reads 1 (smoothing bridges voxel gaps). A stellar sf_export
    on such a run describes the loaded island, not the part. None means
    'certificate skipped' and must stay eligible."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 6.34, "load_path_ok": False, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 6.34}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 2.1, "load_path_ok": None, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 2.1}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 1
    s.struct_runs.append(
        {
            "index": 2,
            "res": {
                "topopt": {"sf_export": 3.0, "load_path_ok": True, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 3.0}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 2


def test_best_run_ranks_divergent_sf_by_pessimistic_bound() -> None:
    """sf_mask 0.05 / sf_export 1.37 means the load junction is one thin
    voxel wall the smoothing fattened — the part's honest bound is the
    pessimistic solve, so the divergent run must lose to a convergent one."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_mask": 0.05, "sf_export": 1.37},
                "fem": {"cases": [{"safety_factor": 0.05}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                # convergent solves — mask and export agree
                "topopt": {"sf_mask": 1.2, "sf_export": 1.3},
                "fem": {"cases": [{"safety_factor": 1.3}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 1  # 1.3 > pessimistic 0.05


def test_best_run_skips_refine_verify_misses() -> None:
    """A refine run whose 'applied' bore failed the post-cut rescan (void
    cut, wrong diameter) must not be salvaged — the unrefined parent run
    stays the honest result."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 2.0, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 2.0}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 9.0, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 9.0}]},
                "features": {
                    "refine": {
                        "applied": [{"name": "m6"}],
                        "verify_misses": [{"x": -30.0, "y": 0.0, "dia": 6.0}],
                    }
                },
            },
            "params": {"refine_of": 0},
        }
    )
    assert _best_run_index(s) == 0


def test_topopt_images_capped_at_budget(tmp_path) -> None:
    """N4: a run result carrying 10+ renders attaches at most
    _TOPOPT_IMAGE_BUDGET and says how many were omitted — the on-disk
    files are untouched, only the context payload shrinks."""
    import types

    from cadx import storage
    from cadx.agent.steps.optimize import _TOPOPT_IMAGE_BUDGET, _topopt_images

    assert _TOPOPT_IMAGE_BUDGET == 8
    img_dir = storage.ensure(storage.data_root() / "_test_imgs" / str(tmp_path.name))
    for i in range(12):
        (img_dir / f"v{i}.png").write_bytes(b"png")
    res = {"renders": {f"view{i}": storage.relpath(img_dir / f"v{i}.png") for i in range(12)}}
    ops = types.SimpleNamespace(contact_sheet=None)
    images = _topopt_images(ops, res, tmp_path)
    assert len(images) == _TOPOPT_IMAGE_BUDGET
    assert all((img_dir / f"v{i}.png").exists() for i in range(12))
    assert any("omitted" in w for w in res.get("warnings") or [])


# ---------------------------------------------------------------------------
# Optimize call envelope — caller wait covers optimizer budget + tail, and
# the shared struct budget covers at least one full call
# ---------------------------------------------------------------------------


async def test_optimize_wait_covers_optimizer_budget_plus_tail() -> None:
    """The agent-side wait must outlast the whole task envelope:
    topopt_timeout_s bounds the OPTIMIZER inside the pipeline; extraction,
    multi-realization export verification, finishing and serialize still
    run afterwards (observed tail ~400-800s). A wait of timeout+120 alone
    discarded a verified artifact ~3min before it returned."""
    from cadx.agent.executor_client import _OPT_TAIL_S, QueueExecutor
    from cadx.runtime_config import RuntimeSettings

    ex = QueueExecutor(runtime_settings=RuntimeSettings(topopt_timeout_s=4800))
    seen: dict = {}

    async def fake_run(func, payload, timeout_s):
        seen["timeout_s"] = timeout_s
        return {"ok": True}

    ex._run_task = fake_run  # type: ignore[assignment]
    assert await ex.optimize({"optimize": {}}) == {"ok": True}
    assert seen["timeout_s"] == 4800 + _OPT_TAIL_S + 120


def test_struct_budget_covers_a_full_optimize_call() -> None:
    """agent_struct_budget_s must afford ONE full optimize envelope —
    smaller defaults let a single maxed call zero the session's remaining
    structural tools even when the call committed a usable run."""
    from cadx.runtime_config import RuntimeSettings

    s = RuntimeSettings()
    assert s.agent_struct_budget_s >= s.topopt_timeout_s + 1020.0


async def test_topopt_fn_refuses_call_that_cannot_land(tmp_path) -> None:
    """The ops budget guard must estimate the FULL call envelope
    (topopt_timeout_s + tail + wait) — a smaller estimate would let the
    agent start a call that cannot return before the job deadline."""
    import itertools
    import time
    import uuid

    from cadx.agent.steps.optimize import _make_topopt_fn, _SessionOps
    from cadx.llm.fake import FakeLLM
    from cadx.llm.schemas import Dim, PartSpec
    from tests.unit.test_session_condense import _ctx

    ctx = _ctx(FakeLLM([]))
    spec = PartSpec(
        name="bracket",
        summary="",
        length_x=Dim(value=100, source="stated"),
        width_y=Dim(value=80, source="stated"),
        height_z=Dim(value=40, source="stated"),
    )
    ops = _SessionOps(
        ctx=ctx,
        base={
            "job_id": "t",
            "project_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
            "mesh_rel": "x",
            "runtime_settings": {},
            "name": "model",
        },
        spec=spec,
        spec_cases=[{"name": "c", "loads": [], "fixed": []}],
        default_material="",
        alloc=itertools.count(1),
        contact_sheet=None,
        deadline=time.monotonic() + 100.0,  # less than one full envelope
        request_opt={},
    )
    fn = _make_topopt_fn(ops)
    res = await fn({"load_cases": [{"name": "c", "loads": [], "fixed": []}]})
    assert res["ok"] is False
    assert "budget" in (res.get("error") or "")


# ---------------------------------------------------------------------------
# queue poll must tolerate transient redis/transport blips — surfacing them
# strands a running task and invites a double-submit that queues behind it
# ---------------------------------------------------------------------------


class _FakeJob:
    def __init__(self, script: list, result=None) -> None:
        self._script = list(script)
        self._result = result if result is not None else {"ok": True}
        self.status_calls = 0

    async def status(self):
        self.status_calls += 1
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    async def result(self):
        if isinstance(self._result, BaseException):
            err, self._result = self._result, {"ok": True}
            raise err
        return self._result


class _FakePool:
    def __init__(self, enqueue_exc: Exception | None = None) -> None:
        self._enqueue_exc = enqueue_exc

    async def enqueue_job(self, *a, **kw):
        if self._enqueue_exc is not None:
            raise self._enqueue_exc

    async def setex(self, *a, **kw):
        pass


async def test_run_task_tolerates_transient_status_errors(monkeypatch) -> None:
    """A redis blip mid-wait must NOT surface as call failure — the arq task
    keeps running server-side, and an agent retry would queue behind it."""
    from arq.jobs import JobStatus

    import cadx.agent.executor_client as ec
    from cadx.agent.executor_client import QueueExecutor

    blip = ConnectionError("Timeout connecting to server")
    job = _FakeJob([blip, blip, JobStatus.complete], {"ok": True, "sf": 3.0})
    monkeypatch.setattr(ec, "Job", lambda *a, **k: job)
    ex = QueueExecutor(pool=_FakePool())

    res = await ex._run_task("optimize", {}, 60.0)
    assert res["ok"] is True and res["sf"] == 3.0
    assert job.status_calls == 3


async def test_run_task_runtime_budget_starts_when_job_is_in_progress(monkeypatch) -> None:
    from arq.jobs import JobStatus

    import cadx.agent.executor_client as ec
    from cadx.agent.executor_client import QueueExecutor

    clock = [0.0]
    monkeypatch.setattr(ec.time, "monotonic", lambda: clock[0])

    async def sleep(delay: float) -> None:
        clock[0] += delay

    monkeypatch.setattr(ec.asyncio, "sleep", sleep)
    job = _FakeJob(
        [JobStatus.queued, JobStatus.queued, JobStatus.in_progress, JobStatus.complete],
        {"ok": True},
    )
    monkeypatch.setattr(ec, "Job", lambda *a, **k: job)

    result = await QueueExecutor(pool=_FakePool())._run_task("optimize", {}, 2.0)

    assert result["ok"] is True
    assert result["queued_s"] == 1.25
    assert result["queue_start_observed"] is True


async def test_run_task_queue_wait_has_independent_hard_cap(monkeypatch) -> None:
    import cadx.agent.executor_client as ec
    from cadx.agent.executor_client import QueueExecutor

    clock = [0.0]
    monkeypatch.setattr(ec.time, "monotonic", lambda: clock[0])

    async def sleep(delay: float) -> None:
        clock[0] += delay

    monkeypatch.setattr(ec.asyncio, "sleep", sleep)
    job = _FakeJob([ec.JobStatus.queued])
    monkeypatch.setattr(ec, "Job", lambda *a, **k: job)

    class _Pool(_FakePool):
        cancel_at: float | None = None

        async def setex(self, *a, **kw):
            self.cancel_at = clock[0]

    pool = _Pool()
    result = await QueueExecutor(pool=pool)._run_task("optimize", {}, 2.0)

    assert result["ok"] is False
    assert result["queue_timed_out"] is True
    assert result["queued_s"] == 3.0
    assert pool.cancel_at == 3.0


async def test_run_task_times_out_after_dequeue_runtime(monkeypatch) -> None:
    from arq.jobs import JobStatus

    import cadx.agent.executor_client as ec
    from cadx.agent.executor_client import QueueExecutor

    clock = [0.0]
    monkeypatch.setattr(ec.time, "monotonic", lambda: clock[0])

    async def sleep(delay: float) -> None:
        clock[0] += delay

    monkeypatch.setattr(ec.asyncio, "sleep", sleep)
    job = _FakeJob([JobStatus.queued, JobStatus.in_progress])
    monkeypatch.setattr(ec, "Job", lambda *a, **k: job)
    pool = _FakePool()

    result = await QueueExecutor(pool=pool)._run_task("optimize", {}, 2.0)

    assert result["queue_timed_out"] is False
    assert result["timed_out"] is True
    assert result["queued_s"] == 0.5


async def test_run_task_marks_maybe_running_after_poll_patience(monkeypatch) -> None:
    """Past the tolerated streak the result must say MAYBE-running so the
    agent never resubmits a duplicate behind its own live task."""
    import cadx.agent.executor_client as ec
    from cadx.agent.executor_client import QueueExecutor

    monkeypatch.setattr(ec, "_POLL_FAIL_PATIENCE_S", 2.0)
    job = _FakeJob([ConnectionError("down")])
    monkeypatch.setattr(ec, "Job", lambda *a, **k: job)
    ex = QueueExecutor(pool=_FakePool())

    res = await ex._run_task("optimize", {}, 600.0)
    assert res["ok"] is False and res["task_maybe_running"] is True
    assert "do NOT resubmit" in res["error"]


async def test_run_task_submit_failure_says_safe_to_retry() -> None:
    """A failed enqueue means no task exists — the agent can and should be
    told a retry is safe (the opposite of the poll-failure case)."""
    from cadx.agent.executor_client import QueueExecutor

    ex = QueueExecutor(pool=_FakePool(enqueue_exc=ConnectionError("refused")))
    res = await ex._run_task("fem_analyze", {}, 60.0)
    assert res["ok"] is False
    assert "safe to retry" in res["error"]


async def test_run_task_result_fetch_retries_once(monkeypatch) -> None:
    """A blip on job.result() must not mask a finished task — retry once."""
    from arq.jobs import JobStatus

    import cadx.agent.executor_client as ec
    from cadx.agent.executor_client import QueueExecutor

    job = _FakeJob([JobStatus.complete], result=ConnectionError("blip"))
    monkeypatch.setattr(ec, "Job", lambda *a, **k: job)
    ex = QueueExecutor(pool=_FakePool())

    res = await ex._run_task("optimize", {}, 60.0)
    assert res["ok"] is True


def test_stage_deadline_caps_agent_optimize_budget() -> None:
    from datetime import UTC, datetime, timedelta

    from cadx.agent.worker_jobs import _agent_optimize_budget_s

    now = datetime(2026, 11, 4, 12, tzinfo=UTC)
    payload = {"_stage_deadline_at": (now + timedelta(seconds=1000)).isoformat()}

    assert _agent_optimize_budget_s(payload, 10_000, now_utc=now) == 880.0
    assert _agent_optimize_budget_s({}, 10_000, now_utc=now) == 9700.0
    expired = {"_stage_deadline_at": (now + timedelta(seconds=60)).isoformat()}
    assert _agent_optimize_budget_s(expired, 10_000, now_utc=now) == 0.0
    assert _agent_optimize_budget_s({"_stage_deadline_at": "invalid"}, 10_000) == 0.0


def test_optimize_request_accepts_optional_bounded_stage_timeout() -> None:
    from pydantic import ValidationError

    from cadx.api.schemas import OptimizeRequest

    assert OptimizeRequest().stage_timeout_s is None
    assert OptimizeRequest(stage_timeout_s=7200).stage_timeout_s == 7200
    with pytest.raises(ValidationError):
        OptimizeRequest(stage_timeout_s=0)
    with pytest.raises(ValidationError):
        OptimizeRequest(stage_timeout_s=172801)
