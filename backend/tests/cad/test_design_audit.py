"""Design-robustness audit: fake-runner verdicts + one real-sandbox smoke test."""

from __future__ import annotations

from typing import Any

import pytest

from cadx.cad.design_audit import audit_params

CODE = """from build123d import *

PARAMS = {
    "length": 40.0,
    "width": 20.0,
    "height": 10.0,
    "label": "box",
    "centered": True,
}

result = Box(PARAMS["length"], PARAMS["width"], PARAMS["height"])
"""

ONE_PARAM_CODE = """from build123d import *

PARAMS = {"length": 40.0}

result = Box(PARAMS["length"], 20.0, 10.0)
"""

MANY_PARAMS_CODE = (
    "from build123d import *\n\nPARAMS = {\n"
    + "".join(f'    "p{i}": {10.0 + i},\n' for i in range(10))
    + "}\n\nresult = Box(PARAMS['p0'], PARAMS['p1'], PARAMS['p2'])\n"
)

BASE_PARAMS = {
    "length": 40.0,
    "width": 20.0,
    "height": 10.0,
    "label": "box",
    "centered": True,
}


def ok_run(
    overrides: dict[str, object],
    *,
    volume: float = 8000.0,
    dims: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "error": None,
        "warnings": [],
        "timings": {"execute_s": 0.01},
        "analysis": {
            "is_valid": True,
            "volume": volume,
            "bbox_size": dims or [40.0, 20.0, 10.0],
            "warnings": [],
        },
    }


def test_robust_all_params():
    report = audit_params(CODE, BASE_PARAMS, runner=ok_run)
    assert report["status"] == "robust"
    assert report["audited"] == 3
    assert set(report["params"]) == {"length", "width", "height"}
    for entry in report["params"].values():
        assert entry["verdict"] == "robust"
        assert entry["plus"]["ok"] and entry["minus"]["ok"]
    assert report["baseline"]["volume"] == 8000.0


def test_non_numeric_params_skipped():
    report = audit_params(CODE, BASE_PARAMS, runner=ok_run)
    skipped = {s["name"]: s["reason"] for s in report["skipped"]}
    assert "label" in skipped and "centered" in skipped
    assert skipped["label"] == "non-numeric / categorical"


def test_fails_when_perturbation_breaks_run():
    def runner(ov: dict[str, object]) -> dict[str, Any]:
        if float(ov["length"]) > 40.0:
            return {"ok": False, "error": "boom", "analysis": None}
        return ok_run(ov)

    report = audit_params(CODE, BASE_PARAMS, runner=runner)
    assert report["status"] == "fails"
    entry = report["params"]["length"]
    assert entry["verdict"] == "fails"
    assert entry["plus"]["ok"] is False
    assert "boom" in (entry["plus"]["error"] or "")
    # other params untouched by the failure stay robust
    assert report["params"]["width"]["verdict"] == "robust"


def test_runner_exception_captured_not_raised():
    def runner(ov: dict[str, object]) -> dict[str, Any]:
        if float(ov["height"]) != 10.0:
            raise RuntimeError("runner exploded")
        return ok_run(ov)

    report = audit_params(CODE, BASE_PARAMS, runner=runner)
    assert report["params"]["height"]["verdict"] == "fails"
    assert "runner exploded" in report["params"]["height"]["reason"]
    assert report["status"] == "fails"


def test_brittle_on_volume_drift():
    def runner(ov: dict[str, object]) -> dict[str, Any]:
        # shrinking width collapses volume far beyond the perturbation
        volume = 2000.0 if float(ov["width"]) < 20.0 else 8000.0
        return ok_run(ov, volume=volume)

    report = audit_params(CODE, BASE_PARAMS, runner=runner)
    entry = report["params"]["width"]
    assert entry["verdict"] == "brittle"
    assert "volume" in entry["reason"]
    assert report["status"] == "brittle"
    # deltas are reported against the baseline run
    assert entry["minus"]["volume_delta_pct"] == pytest.approx(-75.0)


def test_brittle_on_new_warnings():
    def runner(ov: dict[str, object]) -> dict[str, Any]:
        out = ok_run(ov)
        if float(ov["width"]) < 20.0:
            out["warnings"] = ["mesh is not watertight / winding inconsistent"]
        return out

    report = audit_params(CODE, BASE_PARAMS, runner=runner)
    assert report["params"]["width"]["verdict"] == "brittle"
    assert report["params"]["length"]["verdict"] == "robust"


def test_invalid_geometry_is_fails_not_brittle():
    def runner(ov: dict[str, object]) -> dict[str, Any]:
        out = ok_run(ov)
        if float(ov["height"]) < 10.0:
            out["analysis"]["is_valid"] = False
        return out

    report = audit_params(CODE, BASE_PARAMS, runner=runner)
    assert report["params"]["height"]["verdict"] == "fails"


def test_max_params_cap_reports_skipped():
    params = {f"p{i}": 10.0 + i for i in range(10)}
    report = audit_params(MANY_PARAMS_CODE, params, max_params=3, runner=ok_run)
    assert report["audited"] == 3
    assert set(report["params"]) == {"p0", "p1", "p2"}
    capped = [s["name"] for s in report["skipped"] if s["reason"] == "max_params cap"]
    assert capped == [f"p{i}" for i in range(3, 10)]


def test_baseline_failure_aborts_audit():
    def runner(ov: dict[str, object]) -> dict[str, Any]:
        return {"ok": False, "error": "baseline exploded", "analysis": None}

    report = audit_params(CODE, BASE_PARAMS, runner=runner)
    assert report["status"] == "fails"
    assert report["audited"] == 0
    assert report["baseline"]["ok"] is False


def test_int_params_perturb_by_at_least_one():
    code = (
        "from build123d import *\n\n"
        'PARAMS = {"count": 3, "length": 40.0}\n\n'
        "result = Box(PARAMS['length'], 20.0, 10.0)\n"
    )
    report = audit_params(code, {"count": 3, "length": 40.0}, runner=ok_run)
    entry = report["params"]["count"]
    assert entry["plus"]["override"] == 4
    assert entry["minus"]["override"] == 2
    assert isinstance(entry["plus"]["override"], int)


def test_real_sandbox_smoke():
    """One param, real sandbox runner (run_pipeline light mode)."""
    report = audit_params(ONE_PARAM_CODE, {"length": 40.0}, runner=None)
    assert report["status"] == "robust", report.get("baseline", {}).get("error")
    entry = report["params"]["length"]
    assert entry["verdict"] == "robust"
    assert entry["plus"]["override"] == pytest.approx(42.0)
    assert entry["minus"]["override"] == pytest.approx(38.0)
    assert report["baseline"]["volume"] == pytest.approx(8000.0, rel=0.05)
