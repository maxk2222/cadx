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


def test_envelope_bound_shrunken_part_passes():
    # topopt semantics: stated dims are an upper bound — SIMP legitimately
    # removes material, so a smaller bbox must PASS, not "dim drift" FAIL
    res = _res(analysis={"bbox_size": [80.0, 40.0, 15.0], "solid_count": 1, "is_valid": True})
    report = verify_spec(_spec(), res, envelope_bound_only=True)
    checks = _by_name(report)
    for ax in "xyz":
        assert checks[f"envelope_{ax}"]["status"] == "PASS", checks[f"envelope_{ax}"]
        assert checks[f"envelope_{ax}"]["evidence"]["bound"] == "upper"
    assert report["status"] == "PASS"


def test_envelope_bound_overshoot_fails():
    # exceeding the stated envelope past tolerance is still a hard FAIL —
    # the bound only forgives shrinking, never growth
    res = _res(analysis={"bbox_size": [130.0, 50.0, 20.0], "solid_count": 1, "is_valid": True})
    report = verify_spec(_spec(), res, envelope_bound_only=True)
    x = _by_name(report)["envelope_x"]
    assert x["status"] == "FAIL"
    assert x["evidence"]["over_mm"] > 0
    assert report["status"] == "FAIL"


def test_envelope_bound_within_tolerance_passes():
    # a 1% overshoot sits inside the 2% tolerance band
    res = _res(analysis={"bbox_size": [101.0, 50.0, 20.0], "solid_count": 1, "is_valid": True})
    report = verify_spec(_spec(), res, envelope_bound_only=True)
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


# ---------------------------------------------------------------------------
# M4 — per-feature placement checks (the l_bracket trap class)
# ---------------------------------------------------------------------------


def _feat(**over):
    f = {
        "name": "feat",
        "type": "hole",
        "description": "",
        "dims": {},
        "position": "",
        "axis": "",
        "count": 1,
        "source": "stated",
    }
    f.update(over)
    return f


def _hole_group(**over):
    g = {
        "id": "",
        "diameter": 5.0,
        "count": 1,
        "axis": "+z",
        "depth_class": "through",
        "centers": [],
        "blocked": 0,
        "elongated": False,
    }
    g.update(over)
    return g


def test_slot_on_wrong_axis_fails():
    """l_bracket trap: spec wants the leg slot through z, the part has the
    slot bored through x — envelope stays legal, placement must FAIL."""
    spec = _spec(features=[_feat(name="leg slot", type="slot", axis="z")])
    res = _res(
        features={
            "hole_groups": [_hole_group(id="s1", diameter=12.4, axis="+x", elongated=True)],
            "slots": 1,
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_axis"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["expected_axis"] == "z"
    assert c["evidence"]["measured_axes"] == ["x"]
    assert c["evidence"]["measured_ids"] == ["s1"]
    assert report["status"] == "FAIL"


def test_round_hole_on_wrong_axis_fails():
    spec = _spec(features=[_feat(name="mount", type="hole", axis="z")])
    res = _res(features={"hole_groups": [_hole_group(id="h1", axis="+y")]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_axis"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["expected_axis"] == "z"
    assert c["evidence"]["measured_axes"] == ["y"]
    assert "h1" in c["evidence"]["measured_ids"]


def test_slot_satisfied_by_round_hole_is_kind_mismatch():
    """A round bore never satisfies a slot spec — kind mismatch FAIL even
    without a stated axis."""
    spec = _spec(features=[_feat(name="slot", type="slot")])
    res = _res(features={"hole_groups": [_hole_group(id="h1", axis="+z")]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_kind"]
    assert c["status"] == "FAIL"
    assert "kind mismatch" in c["evidence"]["reason"]
    assert c["evidence"]["measured_kinds"] == ["hole"]


def test_hole_satisfied_by_measured_slot_is_kind_mismatch():
    spec = _spec(features=[_feat(name="mount", type="hole")])
    res = _res(
        features={
            "hole_groups": [_hole_group(id="s1", axis="+z", elongated=True)],
            "slots": 1,
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["feat_0_kind"]["status"] == "FAIL"


def test_correct_placement_passes():
    spec = _spec(features=[_feat(name="mount", type="hole", axis="z", count=2)])
    res = _res(
        features={
            "hole_groups": [_hole_group(id="h1", axis="-z", count=2)],
            "total_holes": 2,
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_axis"]
    assert c["status"] == "PASS"
    assert c["evidence"]["measured_ids"] == ["h1"]
    assert report["status"] == "PASS"


def test_placement_unverified_without_measured_data():
    """No measured void evidence at all -> UNVERIFIED, never FAIL."""
    spec = _spec(features=[_feat(name="mount", type="hole", axis="z")])
    res = _res(features={"step_roundtrip": {"checked": True, "ok": True}})
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["feat_0_axis"]["status"] == "UNVERIFIED"
    assert report["status"] == "PASS"


def test_rib_proud_of_envelope_fails():
    """l_bracket trap: ribs on the ±y side faces push the measured width past
    the stated 40 mm envelope."""
    spec = _spec(
        length_x={"value": 60.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 30.0, "source": "stated"},
        features=[_feat(name="ribs", type="rib", position="outer y sides")],
    )
    res = _res(analysis={"bbox_size": [60.0, 44.0, 30.0], "solid_count": 1, "is_valid": True})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_envelope"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["envelope_mm"] == {"y": 40.0}
    assert c["evidence"]["measured_extent_mm"]["y"] == 44.0
    assert c["evidence"]["protrusions"][0]["axis"] == "y"
    assert report["status"] == "FAIL"


def test_rib_inside_envelope_passes():
    spec = _spec(features=[_feat(name="ribs", type="rib", position="outer y sides")])
    report = verify_spec(spec, _res(), feature_findings=[])
    c = _by_name(report)["feat_0_envelope"]
    assert c["status"] == "PASS"
    assert c["evidence"]["measured_extent_mm"]["y"] == 50.0


def test_protrusion_unverified_without_stated_dims():
    spec = _spec(
        length_x={"value": 60.0, "source": "inferred"},
        width_y={"value": 40.0, "source": "inferred"},
        height_z={"value": 30.0, "source": "inferred"},
        features=[_feat(name="ribs", type="rib")],
    )
    report = verify_spec(spec, _res(), feature_findings=[])
    assert _by_name(report)["feat_0_envelope"]["status"] == "UNVERIFIED"


def test_boss_wider_than_envelope_fails():
    """A measured boss Ø60 on a stated-50-wide part cannot fit inside."""
    spec = _spec(features=[_feat(name="post", type="boss")])
    res = _res(features={"bosses": [{"id": "b1", "axis": "+z", "diameter": 60.0, "height": 8.0}]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_envelope"]
    assert c["status"] == "FAIL"
    assert any(p.get("boss_id") == "b1" for p in c["evidence"]["protrusions"])
    assert c["evidence"]["measured_ids"] == ["b1"]


def test_position_offset_verified():
    spec = _spec(features=[_feat(name="mount", type="hole", position="x=12")])
    res = _res(
        features={
            "hole_groups": [_hole_group(id="h1", centers=[[12.3, 0.0]])],
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "PASS"


def test_position_offset_violated_fails():
    spec = _spec(features=[_feat(name="mount", type="hole", position="x=12")])
    res = _res(features={"hole_groups": [_hole_group(id="h1", centers=[[20.0, 0.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["violations"][0]["axis"] == "x"
    assert c["evidence"]["measured_ids"] == ["h1"]


def test_position_face_hint_checked():
    spec = _spec(features=[_feat(name="mount", type="hole", position="+y face")])
    res = _res(features={"hole_groups": [_hole_group(id="h1", centers=[[0.0, 8.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["feat_0_position"]["status"] == "PASS"
    res_bad = _res(features={"hole_groups": [_hole_group(id="h1", centers=[[0.0, -8.0]])]})
    report = verify_spec(spec, res_bad, feature_findings=[])
    assert _by_name(report)["feat_0_position"]["status"] == "FAIL"


def test_position_midplane_is_unverified_not_failed():
    spec = _spec(features=[_feat(name="mount", type="hole", position="front face")])
    res = _res(features={"hole_groups": [_hole_group(id="h1", centers=[[0.0, 0.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["feat_0_position"]["status"] == "UNVERIFIED"


def test_position_prose_is_unverified():
    spec = _spec(features=[_feat(name="mount", type="hole", position="somewhere sensible")])
    res = _res(features={"hole_groups": [_hole_group(id="h1", centers=[[0.0, 8.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "UNVERIFIED"
    assert "not machine-parseable" in c["evidence"]["reason"]


def _res_bbox90(**over):
    """90 mm long part spanning x in [-45, +45] — the end-offset fixture."""
    res = _res(
        analysis={
            "bbox_size": [90.0, 30.0, 20.0],
            "bbox_min": [-45.0, -15.0, 0.0],
            "bbox_max": [45.0, 15.0, 20.0],
            "solid_count": 1,
            "is_valid": True,
        }
    )
    res.update(over)
    return res


def test_position_end_offset_verified():
    """'75mm from -X end' resolves against the measured bbox to x=xmin+75=+30
    — not the absolute x=75 the codegen bug produced."""
    spec = _spec(
        features=[_feat(name="mount", type="hole", position="75mm from -X end, centered in Y")]
    )
    res = _res_bbox90(features={"hole_groups": [_hole_group(id="h1", centers=[[30.2, 0.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "PASS"
    assert c["evidence"]["expected"]["ends"] == [{"axis": "x", "end": "-x", "offset_mm": 75.0}]
    assert c["evidence"]["expected"]["coords"]["y"] == 0.0


def test_position_end_offset_violated_fails():
    """Bore placed at absolute x=75 instead of xmin+75=+30 -> FAIL with delta."""
    spec = _spec(features=[_feat(name="mount", type="hole", position="75mm from -X end")])
    res = _res_bbox90(features={"hole_groups": [_hole_group(id="h1", centers=[[75.0, 0.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "FAIL"
    v = c["evidence"]["violations"][0]
    assert v["axis"] == "x"
    assert v["expected_mm"] == 30.0
    assert v["measured_mm"] == 75.0
    assert v["delta_mm"] == 45.0


def test_position_end_offset_max_end():
    """'10mm from the +x edge' -> x=xmax-10=+35."""
    spec = _spec(features=[_feat(name="mount", type="hole", position="10 mm from the +x edge")])
    res = _res_bbox90(features={"hole_groups": [_hole_group(id="h1", centers=[[35.0, 0.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["feat_0_position"]["status"] == "PASS"


def test_position_end_offset_without_bbox_is_unverified():
    """End-relative hints need bbox_min/bbox_max — sizes alone can't resolve
    them, so the hint stays UNVERIFIED rather than guessing."""
    spec = _spec(features=[_feat(name="mount", type="hole", position="75mm from -X end")])
    res = _res(features={"hole_groups": [_hole_group(id="h1", centers=[[30.0, 0.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "UNVERIFIED"
    assert c["evidence"]["expected"]["ends"][0]["end"] == "-x"


def test_position_end_offset_and_centered_in_y():
    """'30 mm from -x end, centred in Y' -> x=xmin+30=-15 and y=0 both checked."""
    spec = _spec(
        features=[_feat(name="mount", type="hole", position="30 mm from -x end, centred in Y")]
    )
    res = _res_bbox90(features={"hole_groups": [_hole_group(id="h1", centers=[[-15.0, 0.4]])]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "PASS"
    assert c["evidence"]["expected"]["coords"]["y"] == 0.0
    # an off-centre y must FAIL the same check
    res_bad = _res_bbox90(features={"hole_groups": [_hole_group(id="h1", centers=[[-15.0, 9.0]])]})
    c = _by_name(verify_spec(spec, res_bad, feature_findings=[]))["feat_0_position"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["violations"][0]["axis"] == "y"


def test_position_two_end_offsets_both_checked():
    """'20mm and 75mm from -X end' -> two expected centres at xmin+20/xmin+75."""
    spec = _spec(
        features=[_feat(name="mounts", type="hole", count=2, position="20mm and 75mm from -X end")]
    )
    res = _res_bbox90(
        features={
            "hole_groups": [_hole_group(id="h1", count=2, centers=[[-24.8, 0.0], [29.9, 0.0]])]
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["feat_0_position"]["status"] == "PASS"


def test_position_multi_point_groups_checked():
    """'x=-25, y=0 and x=+45, y=0' declares TWO points — each must own one
    measured centre within ±1.5 mm. The flat coords dict only ever saw the
    last x and let a missing duplicate pass."""
    spec = _spec(
        features=[_feat(name="mounts", type="hole", count=2, position="x=-25, y=0 and x=+45, y=0")]
    )
    res = _res(
        features={
            "hole_groups": [_hole_group(id="h1", count=2, centers=[[-25.3, 0.4], [45.2, -0.3]])]
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "PASS"
    assert c["evidence"]["expected"]["points"] == [
        {"x": -25.0, "y": 0.0},
        {"x": 45.0, "y": 0.0},
    ]


def test_position_multi_point_missing_instance_fails():
    """Only the second declared point measured -> the x=-25 group has no
    centre and the check FAILs instead of passing on x=45 alone."""
    spec = _spec(
        features=[_feat(name="mounts", type="hole", count=2, position="x=-25, y=0 and x=+45, y=0")]
    )
    res = _res(features={"hole_groups": [_hole_group(id="h1", centers=[[45.0, 0.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["violations"][0]["point"]["x"] == -25.0


def test_position_end_range_stays_unverified():
    """'between 20 and 30mm from -x end' is a range — never a point check.
    The bare '-x' still degrades gracefully to a side-face hint."""
    spec = _spec(
        features=[_feat(name="mount", type="hole", position="between 20 and 30mm from -x end")]
    )
    res = _res_bbox90(features={"hole_groups": [_hole_group(id="h1", centers=[[-25.0, 0.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert "ends" not in c["evidence"]["expected"]
    assert c["status"] == "PASS"  # face hint only: centre is on the -x side


def test_placement_checks_stay_json_serializable():
    spec = _spec(
        features=[
            _feat(name="slot", type="slot", axis="z"),
            _feat(name="ribs", type="rib", position="outer y sides"),
            _feat(name="mount", type="hole", position="x=12"),
        ]
    )
    res = _res(
        features={
            "hole_groups": [
                _hole_group(id="s1", axis="+x", elongated=True),
                _hole_group(id="h1", centers=[[20.0, 0.0]]),
            ],
            "slots": 1,
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    assert json.loads(json.dumps(report)) == report


def test_position_near_end_band_fails():
    """'near the +X end' resolves to the outermost 30% of the measured
    x-span — a bore at the wrong end FAILs instead of pooling a bare '+x'
    face hint. Bbox x spans [-45, +45] -> the band starts at x=+18."""
    spec = _spec(
        features=[
            _feat(
                name="mount",
                type="hole",
                dims={"diameter": 6.8},
                position="near the +X end, centered in Y",
            )
        ]
    )
    res = _res_bbox90(
        features={"hole_groups": [_hole_group(id="h1", diameter=6.8, centers=[[-35.0, 0.0]])]}
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["expected"]["nears"] == [{"axis": "x", "end": "+x"}]
    v = c["evidence"]["violations"][0]
    assert v["near"] == "+x" and v["measured_mm"] == -35.0


def test_position_near_end_band_passes():
    spec = _spec(
        features=[
            _feat(
                name="mount",
                type="hole",
                dims={"diameter": 6.8},
                position="near the +X end, centered in Y",
            )
        ]
    )
    res = _res_bbox90(
        features={"hole_groups": [_hole_group(id="h1", diameter=6.8, centers=[[30.0, 0.1]])]}
    )
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["feat_0_position"]["status"] == "PASS"


def test_position_dia_matched_no_pool_pass():
    """Ø8 spec at x=+30 must NOT be satisfied by a Ø12 void at +30 — only
    the dia-matched Ø8 group pools into the position check (the pass-by-
    pool defect: a different hole's centre used to satisfy the hint)."""
    spec = _spec(features=[_feat(name="m8", type="hole", dims={"diameter": 8.0}, position="x=30")])
    res = _res_bbox90(
        features={
            "hole_groups": [
                _hole_group(id="big", diameter=12.0, centers=[[30.0, 0.0]]),
                _hole_group(id="small", diameter=8.0, centers=[[-35.0, 0.0]]),
            ]
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["feat_0_position"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["measured_ids"] == ["small"]


def test_position_dia_matched_pools_right_hole():
    """Same layout, but the dia-matched Ø8 hole sits at the declared x."""
    spec = _spec(features=[_feat(name="m8", type="hole", dims={"diameter": 8.0}, position="x=30")])
    res = _res_bbox90(
        features={
            "hole_groups": [
                _hole_group(id="big", diameter=12.0, centers=[[-35.0, 0.0]]),
                _hole_group(id="small", diameter=8.0, centers=[[30.2, 0.0]]),
            ]
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["feat_0_position"]["status"] == "PASS"


def test_unrequested_void_fails():
    """An extra measured hole no spec feature declares is a FAIL even when
    the declared feature itself verifies."""
    spec = _spec(
        features=[_feat(name="mount", type="hole", dims={"diameter": 8.0}, position="x=30")]
    )
    res = _res_bbox90(
        features={
            "hole_groups": [
                _hole_group(id="h1", diameter=8.0, centers=[[30.0, 0.0]]),
                _hole_group(id="h2", diameter=5.0, centers=[[-20.0, 10.0]]),
            ]
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["unrequested_voids"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["unmatched_ids"] == ["h2"]
    assert report["status"] == "FAIL"


def test_unrequested_void_excess_in_group_fails():
    """A dia-matched group carrying MORE centres than the declared count is
    the duplicate-bore case — the extra centre is unrequested."""
    spec = _spec(features=[_feat(name="mount", type="hole", dims={"diameter": 6.8}, count=1)])
    res = _res_bbox90(
        features={
            "hole_groups": [
                _hole_group(id="h1", diameter=6.8, count=2, centers=[[30.0, 0.0], [-35.0, 0.0]])
            ]
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["unrequested_voids"]
    assert c["status"] == "FAIL"
    assert c["evidence"]["unmatched_ids"] == ["h1"]


def test_unrequested_voids_pass_when_all_matched():
    """All measured voids accounted for — by dia or by declared position —
    plus a <3 mm pseudo-void that must never be counted."""
    spec = _spec(
        features=[_feat(name="mount", type="hole", dims={"diameter": 8.0}, position="x=30")]
    )
    res = _res_bbox90(
        features={
            "hole_groups": [
                _hole_group(id="h1", diameter=8.0, centers=[[30.0, 0.0]]),
                _hole_group(id="h2", diameter=8.2, centers=[[-40.0, 0.0]]),
                _hole_group(id="h3", diameter=2.0, centers=[[0.0, 0.0]]),
            ]
        }
    )
    report = verify_spec(spec, res, feature_findings=[])
    c = _by_name(report)["unrequested_voids"]
    assert c["status"] == "PASS"
    assert c["evidence"]["measured_voids"] == 2


def test_unrequested_voids_wildcard_feature_matches():
    """A spec void feature declaring neither dia nor position could be any
    measured void — in doubt the void counts as matched."""
    spec = _spec(features=[_feat(name="mount", type="hole", axis="z")])
    res = _res(features={"hole_groups": [_hole_group(id="h1", axis="+z", centers=[[9.0, 3.0]])]})
    report = verify_spec(spec, res, feature_findings=[])
    assert _by_name(report)["unrequested_voids"]["status"] == "PASS"
