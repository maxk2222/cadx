"""M6: evalkit run lifecycle — status vs verdict, crash-safe persist."""

from __future__ import annotations

import json
from typing import Any

import pytest

import cadx.evalkit.runner as runner
from cadx.evalkit.models import StageOutcome
from cadx.evalkit.runner import run_case
from cadx.evalkit.suite import EvalCase, EvalStage


def _stage(id_: str = "gen", gates: list[dict[str, Any]] | None = None) -> EvalStage:
    return EvalStage(
        id=id_,
        type="generate",
        prompt="a block",
        gates=gates if gates is not None else [{"type": "status", "expect": ["succeeded"]}],
    )


def _case(stages: list[EvalStage]) -> EvalCase:
    return EvalCase(name="t_case", project="eval:t_case", stages=stages)


def _http_stub(monkeypatch: pytest.MonkeyPatch, tmp_path, dispatch):
    """Run run_case against a stubbed HttpDriver; RESULTS_DIR -> tmp_path."""
    import cadx.evalkit.http as http

    dispatched: list[str] = []

    class StubDriver:
        def __init__(self, *_a, **_kw):
            pass

        async def health(self) -> bool:
            return True

        async def find_or_create_project(self, _project, _settings):
            return {"id": "p1"}

        async def close(self):
            return None

        async def dispatch(self, stage, dctx):
            dispatched.append(stage.id)
            return await dispatch(stage, dctx)

    StubDriver.dispatched = dispatched  # type: ignore[attr-defined]
    monkeypatch.setattr(http, "HttpDriver", StubDriver)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    return StubDriver


def _result_json(dctx) -> dict:
    return json.loads((dctx.out_dir / "result.json").read_text())


async def test_run_persists_running_then_finishes(monkeypatch, tmp_path):
    snap: dict = {}

    async def dispatch(stage, dctx):
        # the run must already be persisted as running when stages dispatch
        snap["before"] = _result_json(dctx)
        return StageOutcome(status="succeeded", wall_s=1.0)

    _http_stub(monkeypatch, tmp_path, dispatch)
    res = await run_case(_case([_stage()]), driver="http", api_url="http://x")

    assert snap["before"]["status"] == "running"
    assert snap["before"]["passed"] is None
    assert snap["before"]["stages"] == []
    assert res.status == "finished"
    assert res.passed is True
    assert res.finished_at and res.updated_at
    data = json.loads((tmp_path / "t_case" / res.run_id / "result.json").read_text())
    assert data["status"] == "finished" and data["schema_version"] == 3


async def test_mid_run_snapshot_is_running_not_finished(monkeypatch, tmp_path):
    mid: dict = {}

    async def dispatch(stage, dctx):
        if stage.id == "gen2":
            # during stage 2 the persisted result already shows stage 1 —
            # lifecycle still running, no finished_at yet
            mid.update(_result_json(dctx))
        return StageOutcome(status="succeeded", wall_s=1.0)

    _http_stub(monkeypatch, tmp_path, dispatch)
    res = await run_case(
        _case([_stage("gen1"), _stage("gen2")]), driver="http", api_url="http://x"
    )

    assert mid["status"] == "running"
    assert len(mid["stages"]) == 1
    assert not mid["finished_at"]
    assert mid["updated_at"]
    assert res.status == "finished" and res.passed is True


async def test_gate_fail_is_finished_with_passed_false(monkeypatch, tmp_path):
    async def dispatch(stage, dctx):
        return StageOutcome(status="failed", wall_s=1.0, error="executor died")

    _http_stub(monkeypatch, tmp_path, dispatch)
    res = await run_case(_case([_stage()]), driver="http", api_url="http://x")

    # runner completed normally -> finished; the gate verdict fails on its own axis
    assert res.status == "finished"
    assert res.passed is False
    assert res.finished_at


async def test_dispatch_exception_is_stage_failure_not_runner_failure(monkeypatch, tmp_path):
    async def dispatch(stage, dctx):
        raise RuntimeError("dispatch exploded")

    _http_stub(monkeypatch, tmp_path, dispatch)
    res = await run_case(_case([_stage()]), driver="http", api_url="http://x")

    # expected stage error -> runner still completes normally
    assert res.status == "finished"
    assert res.passed is False
    assert res.stages[0].status == "failed"


async def test_setup_exception_marks_run_failed(monkeypatch, tmp_path):
    import cadx.evalkit.http as http

    class BrokenDriver:
        def __init__(self, *_a, **_kw):
            pass

        async def health(self) -> bool:
            return True

        async def find_or_create_project(self, _project, _settings):
            raise RuntimeError("http layer exploded")

        async def close(self):
            return None

    monkeypatch.setattr(http, "HttpDriver", BrokenDriver)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    res = await run_case(_case([_stage()]), driver="http", api_url="http://x")

    assert res.status == "failed"
    assert "http layer exploded" in (res.error or "")
    assert res.finished_at


async def test_fail_fast_records_skipped_stages(monkeypatch, tmp_path):
    async def dispatch(stage, dctx):
        return StageOutcome(status="failed", wall_s=1.0)

    drv = _http_stub(monkeypatch, tmp_path, dispatch)
    res = await run_case(
        _case([_stage("s1"), _stage("s2"), _stage("s3")]),
        driver="http",
        api_url="http://x",
        fail_fast=True,
    )

    assert res.status == "finished" and res.passed is False
    assert drv.dispatched == ["s1"]
    statuses = {s.stage_id: s.status for s in res.stages}
    assert statuses == {"s1": "failed", "s2": "skipped", "s3": "skipped"}
    skipped = [s for s in res.stages if s.status == "skipped"]
    assert all(s.passed is None for s in skipped)
    assert "fail-fast" in (res.error or "")


async def test_no_fail_fast_runs_all_stages(monkeypatch, tmp_path):
    async def dispatch(stage, dctx):
        return StageOutcome(status="failed" if stage.id == "s1" else "succeeded", wall_s=1.0)

    drv = _http_stub(monkeypatch, tmp_path, dispatch)
    res = await run_case(
        _case([_stage("s1"), _stage("s2")]),
        driver="http",
        api_url="http://x",
        fail_fast=False,
    )

    assert drv.dispatched == ["s1", "s2"]
    assert res.status == "finished" and res.passed is False
    assert all(s.status != "skipped" for s in res.stages)


async def test_api_unreachable_marks_failed(monkeypatch, tmp_path):
    import cadx.evalkit.http as http

    class DeadDriver:
        def __init__(self, *_a, **_kw):
            pass

        async def health(self) -> bool:
            return False

        async def close(self):
            return None

    monkeypatch.setattr(http, "HttpDriver", DeadDriver)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    res = await run_case(_case([_stage()]), driver="http", api_url="http://x")

    assert res.status == "failed" and "unreachable" in (res.error or "")
    assert res.finished_at


async def test_atomic_persist_leaves_no_tmp_files(monkeypatch, tmp_path):
    async def dispatch(stage, dctx):
        return StageOutcome(status="succeeded", wall_s=1.0)

    _http_stub(monkeypatch, tmp_path, dispatch)
    res = await run_case(
        _case([_stage("s1"), _stage("s2")]), driver="http", api_url="http://x"
    )

    out = tmp_path / "t_case" / res.run_id
    assert not list(out.rglob("*.tmp"))
    assert (out / "result.json").is_file()
    assert (out / "summary.md").read_text().startswith("# eval t_case")
    assert "status: **finished**" in (out / "summary.md").read_text()


def test_write_json_roundtrip_atomic(tmp_path):
    runner._write_json(tmp_path / "a.json", {"x": 1})
    runner._write_json(tmp_path / "a.json", {"x": 2})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 2}
    assert not list(tmp_path.glob("*.tmp"))
