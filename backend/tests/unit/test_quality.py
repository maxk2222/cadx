"""det_quality: the deterministic 0..100 progress signal from verify_spec."""

from __future__ import annotations

from cadx.agent.quality import det_quality


def _checks(*statuses: str) -> dict:
    return {
        "status": "FAIL" if "FAIL" in statuses else "PASS",
        "checks": [{"name": f"c{i}", "status": s} for i, s in enumerate(statuses)],
    }


def test_no_exec_result_is_zero():
    assert det_quality(None, []) == 0
    assert det_quality({}, []) == 0
    assert det_quality({"ok": False, "error": "boom"}, []) == 0


def test_all_pass_is_100():
    res = {"ok": True, "verify_spec": _checks(*(["PASS"] * 17))}
    assert det_quality(res, []) == 100


def test_mixed_checks_ratio():
    res = {"ok": True, "verify_spec": _checks(*(["PASS"] * 13 + ["FAIL"]))}
    assert det_quality(res, []) == 93


def test_unverified_checks_are_ignored():
    res = {"ok": True, "verify_spec": _checks("PASS", "UNVERIFIED", "UNVERIFIED")}
    assert det_quality(res, []) == 100


def test_no_report_is_neutral_50():
    assert det_quality({"ok": True}, []) == 50
    assert det_quality({"ok": True, "verify_spec": {"status": "PASS", "checks": []}}, []) == 50


def test_multi_solid_caps_at_30():
    res = {
        "ok": True,
        "verify_spec": _checks(*(["PASS"] * 10)),
        "analysis": {"solid_count": 2},
    }
    assert det_quality(res, []) == 30


def test_single_valid_solid_fail_caps_at_30():
    report = _checks("PASS", "PASS", "PASS", "PASS")
    report["checks"].append({"name": "single_valid_solid", "status": "FAIL"})
    res = {"ok": True, "verify_spec": report}
    assert det_quality(res, []) == 30


def test_dim_issues_subtract_10_each():
    res = {"ok": True, "verify_spec": _checks(*(["PASS"] * 10))}
    assert det_quality(res, ["length/X off", "width/Y off"]) == 80


def test_clamped_to_zero():
    res = {"ok": True, "verify_spec": _checks("FAIL", "FAIL", "FAIL")}
    assert det_quality(res, ["d1", "d2", "d3"]) == 0
    # even a clean report cannot exceed 100
    res = {"ok": True, "verify_spec": _checks("PASS")}
    assert det_quality(res, []) == 100
