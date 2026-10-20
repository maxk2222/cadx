"""Optimize-session salvage — when the agent exhausts its budget without
done(), the best completed run is committed instead of discarding all
executor work and re-running the deterministic single-shot path."""

from __future__ import annotations

from cadx.agent.steps.optimize import _best_run_index
from cadx.agent.toolspecs import ToolSession


def _session_with_runs() -> ToolSession:
    from cadx.llm.fake import FakeLLM
    from tests.unit.test_session_condense import _ctx

    return ToolSession(_ctx(FakeLLM([{"content": "x"}])), kind="optimize")


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
