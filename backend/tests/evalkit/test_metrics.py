"""Metrics harvest + CaseResult serialization (schema_version 2)."""

from __future__ import annotations

from cadx.evalkit.metrics import harvest
from cadx.evalkit.models import CaseResult, GateResult, StageOutcome, StageResult
from cadx.evalkit.runner import _aggregate_verdict, _persist, _summary_md


def _out(**kw) -> StageOutcome:
    o = StageOutcome(status="succeeded")
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_harvest_usage_and_exec_stats():
    o = _out(
        usage={"prompt_tokens": 1200, "completion_tokens": 400, "calls": 6},
        extras={"execs": 3, "exec_s": 12.34},
        iterations=4,
        analysis={"bbox_size": [80.123, 60, 66], "watertight": True},
        features={"hole_groups": []},
        score=88,
    )
    m = harvest(o)
    assert m["usage"]["prompt_tokens"] == 1200
    assert m["execs"] == 3 and m["exec_s"] == 12.3
    assert m["iterations"] == 4
    assert m["bbox"] == [80.12, 60, 66]
    assert m["watertight"] is True
    assert m["score"] == 88


def test_harvest_reports_feature_evidence_availability():
    m = harvest(_out(features_available=False, features_error="no artifact"))
    assert m["features_available"] is False
    assert m["features_error"] == "no artifact"


def test_harvest_trace_fallbacks():
    # iterations falls back to the trace when outcome.iterations is 0
    recs = [
        {
            "step": "codegen",
            "iteration": 2,
            "latency_ms": 10,
            "prompt_tokens": 5,
            "completion_tokens": 3,
        },
        {"step": "codegen", "iteration": 5, "latency_ms": 10},
    ]
    o = _out(trace_records=recs, iterations=0)
    m = harvest(o)
    assert m["iterations"] == 5
    assert m["step_stats"]["codegen"]["calls"] == 2
    assert m["step_stats"]["codegen"]["prompt_tokens"] == 5


def test_harvest_fem_topopt_verify_spec():
    o = _out(
        fem={"worst_safety_factor": 1.9, "cases": [{"name": "c1", "safety_factor": 1.9}]},
        mesh={
            "topopt": {
                "sf_export": 1.8,
                "volfrac_achieved": 0.5,
                "verify_spec": {"status": "FAIL", "checks": [{"name": "hole_1", "status": "FAIL"}]},
            }
        },
        issues=[{"source": "verify_spec", "what": "hole broken", "fix_hint": "keep it"}],
    )
    m = harvest(o)
    assert m["fem"]["worst_safety_factor"] == 1.9
    assert m["topopt"]["sf_export"] == 1.8
    assert m["verify_spec"]["status"] == "FAIL"
    assert m["verify_spec"]["fails"] == ["hole_1"]
    assert m["issues"] == ["hole broken keep it"]


def _stage(sid="codegen", passed=True):
    return StageResult(
        stage_id=sid,
        stage_type="generate",
        status="succeeded",
        passed=passed,
        wall_s=10.0,
        version_id="v1",
        job_id="j1",
        metrics={"score": 80},
        gates=[GateResult("status", True)],
        artifacts=[],
        error=None,
    )


def test_case_result_roundtrip_schema_v2():
    res = CaseResult(
        case="l_bracket_rib",
        driver="http",
        variant="default",
        run_id="r1",
        passed=True,
        wall_s=42.0,
        project="eval:l-bracket-rib",
        project_id="p1",
        stages=[_stage(), _stage("fem")],
        finished_at="t",
        meta={"git_sha": "abc"},
        variant_overrides={"exec_mem_limit_mb": 4096},
    )
    d = res.to_dict()
    assert d["schema_version"] == 2
    res2 = CaseResult.from_dict(d)
    assert res2.case == "l_bracket_rib"
    assert len(res2.stages) == 2
    assert res2.stages[0].gates[0].name == "status"
    assert res2.variant_overrides == {"exec_mem_limit_mb": 4096}


def test_persist_writes_result_meta_summary(tmp_path):
    res = CaseResult(
        case="c",
        driver="local",
        variant="v",
        run_id="r",
        passed=False,
        wall_s=1.0,
        stages=[_stage(passed=False)],
        error="boom",
    )
    _persist(res, tmp_path)
    import json

    data = json.loads((tmp_path / "result.json").read_text())
    assert data["case"] == "c" and data["passed"] is False
    assert (tmp_path / "meta.json").is_file()
    md = (tmp_path / "summary.md").read_text()
    assert "eval c" in md and "boom" in md


def test_summary_md_lists_failed_gates():
    s = _stage()
    s.gates.append(GateResult("holes", False, detail="want 4 got 2"))
    res = CaseResult(
        case="c", driver="http", variant="v", run_id="r", passed=False, wall_s=1.0, stages=[s]
    )
    md = _summary_md(res)
    assert "holes" in md


def test_inconclusive_verdict_aggregation_and_report():
    assert _aggregate_verdict([True, None]) is None
    assert _aggregate_verdict([None, False]) is False
    assert _aggregate_verdict([True, True]) is True

    stage = _stage(passed=None)
    stage.gates = [GateResult("status", True), GateResult("holes", None, detail="no artifact")]
    res = CaseResult(
        case="c",
        driver="local",
        variant="v",
        run_id="r",
        passed=None,
        wall_s=1.0,
        stages=[stage],
    )
    md = _summary_md(res)
    assert "INCONCLUSIVE" in md
    assert "| codegen | generate | succeeded | INCONCLUSIVE |" in md
    assert "| - | holes |" in md
