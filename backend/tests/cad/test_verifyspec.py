"""verify_spec conformance report — pure dict inputs, no geometry needed.

The report consolidates executor-side evidence (analysis / features / dfm /
step roundtrip) into a canonical PASS/FAIL/UNVERIFIED verdict per tier. A
skipped detector must read UNVERIFIED, never FAIL.
"""

import json

from cadx.cad.verifyspec import verify_spec


def _spec(**over):
    spec = {
        "name": "bracket",
        "length_x": {"value": 100.0, "source": "stated"},
        "width_y": {"value": 50.0, "source": "stated"},
        "height_z": {"value": 20.0, "source": "stated"},
        "features": [],
    }
    spec.update(over)
    return spec


def _res(**over):
    res = {
        "ok": True,
        "error": None,
        "analysis": {
            "bbox_size": [100.0, 50.0, 20.0],
            "solid_count": 1,
            "is_valid": True,
            "watertight": True,
            "volume": 100_000.0,
            "source": "brep",
        },
        "features": {
            "step_roundtrip": {"checked": True, "ok": True, "reimported_valid": True},
        },
        "dfm": {"process": "fdm", "warnings": [], "summary": ""},
    }
    res.update(over)
    return res


def _by_name(report):
    return {c["name"]: c for c in report["checks"]}


def test_clean_pass():
    report = verify_spec(_spec(), _res())
    assert report["status"] == "PASS"
    checks = _by_name(report)
    for name in ("executes", "single_valid_solid", "brep_defects", "step_roundtrip"):
        assert checks[name]["status"] == "PASS", checks[name]
    for name in ("envelope_x", "envelope_y", "envelope_z"):
        assert checks[name]["status"] == "PASS"
    assert checks["dfm"]["status"] == "PASS"
    json.dumps(report)


def test_exec_failure_is_fail():
    res = _res(ok=False, error="sandbox timeout", analysis=None, features={}, dfm=None)
    report = verify_spec(_spec(), res)
    assert report["status"] == "FAIL"
    checks = _by_name(report)
    assert checks["executes"]["status"] == "FAIL"
    assert "timeout" in checks["executes"]["evidence"]["error"]
    # nothing downstream ran — UNVERIFIED, not more FAILs
    assert checks["single_valid_solid"]["status"] == "UNVERIFIED"
    assert checks["dfm"]["status"] == "UNVERIFIED"


def test_invalid_solid_fails_structural():
    res = _res(analysis={"bbox_size": [100.0, 50.0, 20.0], "solid_count": 1, "is_valid": False})
    report = verify_spec(_spec(), res)
    assert report["status"] == "FAIL"
    assert _by_name(report)["single_valid_solid"]["status"] == "FAIL"


def test_multi_solid_fails():
    res = _res(analysis={"bbox_size": [100.0, 50.0, 20.0], "solid_count": 3, "is_valid": True})
    assert _by_name(verify_spec(_spec(), res))["single_valid_solid"]["status"] == "FAIL"


def test_dim_drift_fails_envelope():
    res = _res(analysis={"bbox_size": [100.0, 50.0, 40.0], "solid_count": 1, "is_valid": True})
    report = verify_spec(_spec(), res)
    assert report["status"] == "FAIL"
    z = _by_name(report)["envelope_z"]
    assert z["status"] == "FAIL"
    assert z["evidence"]["expected_mm"] == 20.0
    assert z["evidence"]["actual_mm"] == 40.0
    # the conforming axes still pass
    assert _by_name(report)["envelope_x"]["status"] == "PASS"


def test_inferred_dims_are_unverified_not_failed():
    spec = _spec(height_z={"value": 999.0, "source": "inferred"})
    report = verify_spec(spec, _res())
    assert _by_name(report)["envelope_z"]["status"] == "UNVERIFIED"
    assert report["status"] == "PASS"


def test_missing_feature_evidence_is_unverified():
    spec = _spec(features=[{"name": "mounting hole", "type": "hole", "count": 4}])
    res = _res(features={})
    report = verify_spec(spec, res)
    assert _by_name(report)["spec_features"]["status"] == "UNVERIFIED"
    assert report["status"] == "PASS"  # absent evidence is not a failure


def test_findings_none_is_unverified():
    spec = _spec(features=[{"name": "hole", "type": "hole"}])
    res = _res(features={"total_holes": 1, "hole_groups": [{"diameter": 5.0, "count": 1}]})
    report = verify_spec(spec, res)  # findings not supplied
    assert _by_name(report)["spec_features"]["status"] == "UNVERIFIED"
    assert report["status"] == "PASS"


def test_major_finding_fails_features():
    spec = _spec(features=[{"name": "hole", "type": "hole"}])
    res = _res(features={"total_holes": 0, "hole_groups": []})
    findings = [{"feature": "hole", "expected": "1 hole", "actual": "missing", "severity": "major"}]
    report = verify_spec(spec, res, feature_findings=findings)
    assert report["status"] == "FAIL"
    c = _by_name(report)["spec_features"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["major_findings"] == 1


def test_minor_findings_do_not_fail():
    spec = _spec(features=[{"name": "hole", "type": "hole"}])
    res = _res(features={"total_holes": 1, "hole_groups": [{"diameter": 5.1, "count": 1}]})
    findings = [{"feature": "hole", "expected": "Ø5", "actual": "Ø5.1", "severity": "minor"}]
    report = verify_spec(spec, res, feature_findings=findings)
    assert _by_name(report)["spec_features"]["status"] == "PASS"
    assert report["status"] == "PASS"


def test_defect_present_fails_with_evidence():
    res = _res(
        analysis={"bbox_size": [100.0, 50.0, 20.0], "solid_count": 1, "is_valid": False},
        features={
            "defects": [
                {"kind": "BRepCheck_InvalidImbricationOfShell", "face": 3, "at": [1.0, 2.0, 3.0]}
            ],
            "step_roundtrip": {"checked": True, "ok": True},
        },
    )
    report = verify_spec(_spec(), res)
    assert report["status"] == "FAIL"
    c = _by_name(report)["brep_defects"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["defects"][0]["face"] == 3


def test_step_roundtrip_warning_fails():
    res = _res(
        features={
            "step_roundtrip": {
                "checked": True,
                "ok": False,
                "warning": "STEP roundtrip: re-imported shape fails BRepCheck",
            }
        }
    )
    report = verify_spec(_spec(), res)
    assert report["status"] == "FAIL"
    assert _by_name(report)["step_roundtrip"]["status"] == "FAIL"


def test_step_roundtrip_skipped_is_unverified():
    res = _res(features={"step_roundtrip": {"checked": False, "skipped": "2000 faces > cap"}})
    report = verify_spec(_spec(), res)
    assert _by_name(report)["step_roundtrip"]["status"] == "UNVERIFIED"
    assert report["status"] == "PASS"


def test_skipped_detectors_never_fail():
    # light pipeline: no dfm, no feature evidence — only structural checks stand
    res = _res(features={}, dfm=None)
    report = verify_spec(_spec(min_wall_mm=0.8), res)
    assert _by_name(report)["dfm"]["status"] == "UNVERIFIED"
    assert _by_name(report)["min_wall"]["status"] == "UNVERIFIED"
    assert report["status"] == "PASS"


def test_dfm_high_warning_fails_robustness():
    res = _res(
        dfm={
            "process": "fdm",
            "warnings": [
                {"code": "thin_wall", "severity": "high", "message": "wall 0.3 mm < 0.8 mm"}
            ],
        }
    )
    report = verify_spec(_spec(min_wall_mm=0.8), res)
    assert _by_name(report)["dfm"]["status"] == "FAIL"
    assert _by_name(report)["min_wall"]["status"] == "FAIL"
    assert report["status"] == "FAIL"


def test_hole_pattern_when_recognition_ran():
    spec = _spec(features=[{"name": "bolt circle", "type": "pattern", "count": 6}])
    res = _res(
        features={
            "recognition_ran": True,
            "hole_patterns": [{"kind": "bolt_circle", "count": 6, "diameter": 42.0}],
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["hole_patterns"]["status"] == "PASS"


def test_hole_pattern_missing_fails():
    spec = _spec(features=[{"name": "bolt circle", "type": "pattern", "count": 6}])
    res = _res(features={"recognition_ran": True, "hole_patterns": []})
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["hole_patterns"]["status"] == "FAIL"
    assert report["status"] == "FAIL"


def test_pattern_unverified_when_recognition_skipped():
    spec = _spec(features=[{"name": "bolt circle", "type": "pattern", "count": 6}])
    res = _res(features={"recognition_ran": False})
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["hole_patterns"]["status"] == "UNVERIFIED"
    assert report["status"] == "PASS"


def test_no_spec_structural_only():
    report = verify_spec(None, _res())
    checks = _by_name(report)
    assert checks["envelope"]["status"] == "UNVERIFIED"
    assert "spec_features" not in checks
    assert report["status"] == "PASS"


def test_report_is_json_serializable():
    report = verify_spec(_spec(), _res())
    assert json.loads(json.dumps(report)) == report
    keys = set(report)
    assert keys == {"status", "checks", "summary"}
    for c in report["checks"]:
        assert set(c) == {"name", "tier", "status", "evidence"}
        assert c["status"] in ("PASS", "FAIL", "UNVERIFIED")


def test_mesh_source_checks_are_unverified():
    res = _res(
        analysis={
            "bbox_size": [100.0, 50.0, 20.0],
            "solid_count": 1,
            "is_valid": True,
            "watertight": True,
            "source": "mesh",
        },
        features={},
    )
    report = verify_spec(_spec(), res)
    checks = _by_name(report)
    assert checks["brep_defects"]["status"] == "UNVERIFIED"
    assert checks["step_roundtrip"]["status"] == "UNVERIFIED"
    assert checks["single_valid_solid"]["status"] == "PASS"
    assert report["status"] == "PASS"
