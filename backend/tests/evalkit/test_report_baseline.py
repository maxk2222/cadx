"""Report aggregation + baseline save/diff — incl. scratch exclusion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadx.evalkit import baseline as bl
from cadx.evalkit.models import CaseResult, GateResult, StageResult
from cadx.evalkit.report import aggregate, load_results, render_markdown


def _res(
    case="c1",
    variant="default",
    driver="http",
    passed=True,
    score=80.0,
    run="r1",
    wall=100.0,
    ptok=1000,
    finished="2025-01-01T00:00:00",
):
    s = StageResult(
        stage_id="codegen",
        stage_type="generate",
        status="succeeded",
        passed=passed,
        wall_s=wall,
        version_id="v",
        job_id="j",
        metrics={
            "score": score,
            "iterations": 3,
            "exec_s": 5.0,
            "usage": {"prompt_tokens": ptok, "completion_tokens": 200, "calls": 4},
            "step_stats": {
                "codegen": {
                    "prompt_tokens": ptok,
                    "completion_tokens": 200,
                    "thinking_completion_tokens": 50,
                }
            },
        },
        gates=[GateResult("status", True), GateResult("bbox", passed)],
    )
    return CaseResult(
        case=case,
        driver=driver,
        variant=variant,
        run_id=run,
        passed=passed,
        wall_s=wall,
        stages=[s],
        finished_at=finished,
    )


def _write(root: Path, res: CaseResult):
    d = root / res.case / res.run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res.to_dict()), encoding="utf-8")


def test_load_results_skips_malformed(tmp_path):
    root = tmp_path / "results"
    (root / "bad").mkdir(parents=True)
    (root / "bad" / "result.json").write_text("{not json")
    _write(root, _res())
    out = load_results(root)
    assert len(out) == 1 and out[0].case == "c1"


def test_aggregate_groups_by_variant_case_driver(tmp_path):
    root = tmp_path / "r"
    _write(root, _res(run="r1"))
    _write(root, _res(run="r2", passed=False, score=40.0))
    _write(root, _res(run="r3", variant="fast", driver="local"))
    _write(root, _res(case="c2", run="r4"))
    rows = aggregate(load_results(root))
    assert {(r.variant, r.case, r.driver) for r in rows} == {
        ("default", "c1", "http"),
        ("fast", "c1", "local"),
        ("default", "c2", "http"),
    }
    r_c1 = next(r for r in rows if r.variant == "default" and r.case == "c1")
    assert r_c1.runs == 2 and r_c1.passed == 1
    assert r_c1.mean_score == pytest.approx(60.0)
    assert r_c1.prompt_tokens == 2000
    assert r_c1.thinking_completion_tokens == 100
    assert r_c1.stage_pass["codegen"] == pytest.approx(0.5)


def test_scratch_excluded_by_dir_not_results(tmp_path):
    results = tmp_path / "results"
    scratch = tmp_path / "scratch"
    _write(results, _res(run="r1"))
    _write(scratch, _res(run="s1", driver="local"))
    durable = load_results(results)
    assert {r.run_id for r in durable} == {"r1"}
    both = load_results(results, scratch)
    assert {r.run_id for r in both} == {"r1", "s1"}


def test_render_markdown_table():
    rows = aggregate([_res()])
    md = render_markdown(rows)
    assert "| variant |" in md or "variant" in md
    assert "c1" in md and "default" in md
    assert "codegen" in md  # stage pass-rate table


def test_baseline_save_picks_latest_per_cell(tmp_path):
    root = tmp_path / "res"
    out = tmp_path / "base"
    _write(root, _res(run="r1", passed=False, score=10, ptok=1, finished="2025-01-01T00:00:00"))
    _write(root, _res(run="r2", score=90, ptok=1, finished="2025-01-02T00:00:00"))
    path = bl.save("l1", root=root, out_dir=out)
    data = bl.load_baseline("l1", out_dir=out)
    assert path.name == "l1.json"
    cell = data["cells"]["c1|default"]
    assert cell["run_id"] == "r2"  # latest finished_at wins
    assert cell["score"] == pytest.approx(90.0)
    assert cell["stages"]["codegen"]["gates"]["status"] is True


def test_baseline_diff_flips_and_regressions(tmp_path):
    root = tmp_path / "res"
    out = tmp_path / "base"
    _write(root, _res(score=80, ptok=1000))
    bl.save("a", root=root, out_dir=out)
    # regression: score drop, tokens up, gate flip
    bad = _res(score=60, ptok=2000, passed=False)
    bad.stages[0].gates[1].ok = False
    _write(root, bad)
    bl.save("b", root=root, out_dir=out)
    diffs = bl.diff(bl.load_baseline("a", out_dir=out), bl.load_baseline("b", out_dir=out))
    kinds = {d.kind for d in diffs}
    assert "metric" in kinds  # score + token regression
    assert "gate_flip" in kinds
    assert "pass_flip" in kinds
    md = bl.render_diff(diffs, "a", "b")
    assert "regression" in md
    # identical -> clean
    assert bl.diff(bl.load_baseline("a", out_dir=out), bl.load_baseline("a", out_dir=out)) == []


def test_baseline_missing_cell_flagged(tmp_path):
    root = tmp_path / "res"
    out = tmp_path / "base"
    _write(root, _res(case="c1"))
    bl.save("a", root=root, out_dir=out)
    _write(root, _res(case="c2"))
    bl.save("b", root=root, out_dir=out)
    diffs = bl.diff(bl.load_baseline("a", out_dir=out), bl.load_baseline("b", out_dir=out))
    assert any(d.kind == "missing" and "c2" in d.cell for d in diffs)
