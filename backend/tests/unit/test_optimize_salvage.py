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
