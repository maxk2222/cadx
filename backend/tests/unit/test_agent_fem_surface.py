"""Agent surface M3 — _job_summary carries the fem/topopt evidence (the
waited-job result used to be a bare status with no numbers),
suggest_load_cases drafts deterministically, and _METRIC_KEYS keeps the
new structural fields through compaction."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

from cadx.agent.steps.optimize import _SessionOps, _make_suggest_fn
from cadx.agent.toolloop import _metrics_hint
from cadx.db.models import JobStatus, JobType
from cadx.spawn import _job_summary


class _E:
    def __init__(self, v: str):
        self.value = v


def _job(t: JobType, status: JobStatus = JobStatus.succeeded) -> Any:
    return SimpleNamespace(
        type=t,
        status=status,
        id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        questions=None,
    )


def _version(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "number": 3,
        "status": _E("ready"),
        "kind": _E("generate"),
        "score": None,
        "iterations": 1,
        "resolved_mode": None,
        "error": None,
        "analysis": {},
        "issues": [],
        "fem": None,
        "mesh": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


_FEM = {
    "worst_safety_factor": 2.4,
    "model": "resolved",
    "cases": [
        {
            "name": "tip",
            "safety_factor": 2.4,
            "max_vm_mpa": 21.5,
            "max_disp_mm": 0.42,
            "work_disp_mm": 0.31,
            "converged": True,
            "reactions": {
                "resultant_n": [0.0, 0.0, -100.0],
                "fixed_centroid_mm": [-30.0, 0.0, 6.0],
            },
            "applied": [
                {"kind": "surface", "centroid_mm": [30.0, 0.0, 6.0], "force_mag_n": 100.0},
            ],
            "warnings": ["w1"],
            "feature_peaks": [{"index": 1, "vm_mpa": 40.0, "pos": [1, 2, 3]}],
        }
    ],
    "warnings": ["report-level"],
}

_TOP = {
    "solver_used": "beso",
    "status": "converged",
    "converged": True,
    "volfrac_achieved": 0.45,
    "sf_verified": 1.9,
    "sf_export": 2.1,
    "sf_mask": 2.6,
    "load_path_ok": True,
    "worse_than_uniform": False,
    "uniform_compliance_mj": 30.0,
    "warnings": [],
}


def test_job_summary_includes_fem_block() -> None:
    out = _job_summary(_job(JobType.fem), _version(fem=_FEM))
    fem = out["fem"]
    assert fem["worst_safety_factor"] == 2.4
    case = fem["cases"][0]
    assert case["safety_factor"] == 2.4 and case["work_disp_mm"] == 0.31
    assert case["resultant_n"] == [0.0, 0.0, -100.0]
    assert case["fixed_centroid_mm"] == [-30.0, 0.0, 6.0]
    assert case["load_centroid_mm"] == [30.0, 0.0, 6.0]
    assert case["feature_peaks"][0]["vm_mpa"] == 40.0
    assert fem["warnings"] == ["report-level"]


def test_job_summary_includes_topopt_block() -> None:
    out = _job_summary(_job(JobType.optimize), _version(mesh={"topopt": _TOP}))
    top = out["topopt"]
    assert top["solver_used"] == "beso"
    assert top["sf_verified"] == 1.9 and top["worse_than_uniform"] is False
    assert top["uniform_compliance_mj"] == 30.0


def test_job_summary_wrong_job_type_shows_nothing() -> None:
    # a fem report on the version must not leak into an optimize summary
    # (it would be a stale verdict from an earlier run)
    out = _job_summary(_job(JobType.optimize), _version(fem=_FEM))
    assert "fem" not in out
    out = _job_summary(_job(JobType.fem), _version(mesh={"topopt": _TOP}))
    assert "topopt" not in out


def test_suggest_load_cases_drafts_from_anchors() -> None:
    ops = _SessionOps(
        ctx=SimpleNamespace(),
        base={
            "anchors": [
                {"index": 1, "kind": "hole", "pos": [10, 0, 6], "dia": 5.0},
                {"index": 2, "kind": "hole", "pos": [-10, 0, 6], "dia": 5.0},
            ]
        },
        spec=SimpleNamespace(),
        spec_cases=[],
        default_material="pla",
        alloc=iter(range(10)),
        contact_sheet=None,
    )
    res = asyncio.run(_make_suggest_fn(ops)({}))
    assert res["ok"] and res["n_anchors"] == 2
    case = res["load_cases"][0]
    assert [r["hole_index"] for r in case["fixed"]] == [1, 2]
    assert any(ld.get("normal_force_n") for ld in case["loads"])
    assert res["hints"]  # editing guidance present


def test_suggest_load_cases_returns_spec_cases_when_present() -> None:
    spec_cases = [{"name": "declared", "fixed": [{"kind": "plane", "axis": "-x"}]}]
    ops = _SessionOps(
        ctx=SimpleNamespace(),
        base={"anchors": [{"index": 1, "kind": "hole", "pos": [0, 0, 0], "dia": 3}]},
        spec=SimpleNamespace(),
        spec_cases=spec_cases,
        default_material="pla",
        alloc=iter(range(10)),
        contact_sheet=None,
    )
    res = asyncio.run(_make_suggest_fn(ops)({}))
    assert res["load_cases"] == spec_cases


def test_metrics_hint_keeps_new_structural_keys() -> None:
    res = {
        "topopt": {
            "sf_verified_first": 0.57,
            "worse_than_uniform": True,
            "uniform_compliance_mj": 5914.0,
            "other": "x",
        }
    }
    hint = _metrics_hint(res)
    assert "sf_verified_first=0.57" in hint
    assert "worse_than_uniform=True" in hint
    assert "uniform_compliance_mj=5914.0" in hint
