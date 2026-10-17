"""Face-adjacency-graph (AAG) structural features — pockets, grooves, notches.

M11: the measured-features dict cannot see a pocket with the hole/boss scan —
these features are recognised on the convexity-labelled face adjacency graph
(occwl pattern, rule-based) built in ``features.py``.
"""

from __future__ import annotations

import pytest
from build123d import (
    Align,
    Box,
    BuildPart,
    Cylinder,
    Pos,
    Rot,
)

from cadx.cad.features import (
    MeasuredFeatures,
    MeasuredGroove,
    MeasuredNotch,
    MeasuredPocket,
    assign_feature_ids,
    extract_features,
)
from cadx.cad.verifyspec import verify_spec


def _pocket_part():
    with BuildPart() as p:
        Box(60, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, 6) * Box(30, 20, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def _channel_part():
    """Plate with an 8 mm wide through-channel cut along x on the top face."""
    with BuildPart() as p:
        Box(60, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, 4) * Box(60, 8, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def _step_part():
    with BuildPart() as p:
        Box(60, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(20, 0, 5) * Box(20, 40, 5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def _radial_hole_part():
    with BuildPart() as p:
        Cylinder(15, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(-20, 0, 10) * Rot(0, 90, 0) * Cylinder(
            3, 40, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
    return p.part


def _verify_res(features: MeasuredFeatures, **over):
    res = {
        "ok": True,
        "error": None,
        "analysis": {
            "bbox_size": [60.0, 40.0, 10.0],
            "solid_count": 1,
            "is_valid": True,
            "watertight": True,
            "volume": 20_000.0,
            "source": "brep",
        },
        "features": features.model_dump(),
        "dfm": {"process": "fdm", "warnings": [], "summary": ""},
    }
    res.update(over)
    return res


def _by_name(report):
    return {c["name"]: c for c in report["checks"]}


def test_pocket_detected_with_depth_and_axis():
    m = extract_features(_pocket_part())
    assert len(m.pockets) == 1
    pk = m.pockets[0]
    assert pk.axis == "+z"
    assert pk.depth == pytest.approx(4.0, abs=0.05)
    assert sorted(pk.size) == pytest.approx([20.0, 30.0], abs=0.05)
    assert not m.grooves and not m.notches


def test_through_channel_is_a_groove_not_a_notch():
    m = extract_features(_channel_part())
    assert len(m.grooves) == 1
    gr = m.grooves[0]
    assert gr.width == pytest.approx(8.0, abs=0.05)
    assert gr.axis in ("+x", "-x")  # channel through-direction
    assert gr.opening == "+z"
    assert gr.open_ends == 2
    assert gr.depth == pytest.approx(6.0, abs=0.05)
    assert not m.pockets and not m.notches


def test_stepped_pocket_reports_full_depth_once():
    """Two-tier recess: the deep floor's concave subgraph must climb through
    the intermediate ledge so the pocket reports depth to the true opening —
    and the shallow tier must not double-report as grooves."""
    with BuildPart() as p:
        Box(60, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, 8) * Box(40, 30, 2, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, 4) * Box(20, 15, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
    m = extract_features(p.part)
    assert len(m.pockets) == 1
    assert m.pockets[0].depth == pytest.approx(6.0, abs=0.05)
    assert not m.grooves and not m.notches


def test_step_cut_is_a_notch_not_a_groove():
    m = extract_features(_step_part())
    assert len(m.notches) == 1
    nt = m.notches[0]
    assert nt.axis == "+z"
    assert nt.depth == pytest.approx(5.0, abs=0.05)
    assert nt.width == pytest.approx(20.0, abs=0.05)
    assert not m.grooves and not m.pockets


def test_plain_box_has_no_structural_features():
    m = extract_features(Box(50, 30, 10, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    assert m.pockets == [] and m.grooves == [] and m.notches == []


def test_radial_hole_path_unaffected():
    m = extract_features(_radial_hole_part())
    assert m.total_holes >= 1
    assert any(g.id.startswith("h") for g in m.hole_groups)
    assert m.pockets == [] and m.grooves == [] and m.notches == []


def test_pocket_and_hole_coexist():
    """A through bore inside the pocket floor must stay a hole_group — the
    AAG pass only adds the pocket, it does not absorb the hole."""
    with BuildPart() as p:
        Box(50, 50, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, 5) * Box(10, 10, 5, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(-18, -18, -1) * Cylinder(2.5, 12)
    m = extract_features(p.part)
    assert len(m.pockets) == 1 and m.pockets[0].axis == "+z"
    assert m.total_holes == 1


def test_blind_channel_is_single_open_end_groove():
    """A channel cut in from the edge and closed at one end: one open end,
    still a groove — not a pocket (not fully walled) and not a notch."""
    with BuildPart() as p:
        Box(60, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(20, 0, 4) * Box(20, 6, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    m = extract_features(p.part)
    assert len(m.grooves) == 1
    assert m.grooves[0].open_ends == 1
    assert not m.pockets and not m.notches


def test_structural_ids_and_model_dump():
    m = extract_features(_pocket_part())
    assert [p.id for p in m.pockets] == ["pk1"]
    dumped = m.model_dump()
    assert dumped["pockets"][0]["id"] == "pk1"
    assert "grooves" in dumped and "notches" in dumped

    m = extract_features(_channel_part())
    assert [g.id for g in m.grooves] == ["gr1"]
    m = extract_features(_step_part())
    assert [n.id for n in m.notches] == ["nt1"]


def test_assign_feature_ids_deterministic_for_new_kinds():
    kwargs_p = [
        {"axis": "+z", "depth": 4.0, "center": [10.0, 0.0, 6.0]},
        {"axis": "+z", "depth": 2.0, "center": [-10.0, 0.0, 6.0]},
    ]
    m1 = assign_feature_ids(
        MeasuredFeatures(pockets=[MeasuredPocket(**k) for k in kwargs_p])
    )
    m2 = assign_feature_ids(
        MeasuredFeatures(pockets=[MeasuredPocket(**k) for k in reversed(kwargs_p)])
    )
    ids1 = {tuple(p.center): p.id for p in m1.pockets}
    ids2 = {tuple(p.center): p.id for p in m2.pockets}
    assert ids1 == ids2
    assert {p.id for p in m1.pockets} == {"pk1", "pk2"}
    # grooves and notches get their own prefixes
    m = assign_feature_ids(
        MeasuredFeatures(
            grooves=[MeasuredGroove(axis="+x", width=8.0, center=[0.0, 0.0, 4.0])],
            notches=[MeasuredNotch(axis="+z", width=20.0, center=[20.0, 0.0, 5.0])],
        )
    )
    assert m.grooves[0].id == "gr1"
    assert m.notches[0].id == "nt1"


def test_verify_spec_pocket_satisfied_by_measured_pocket():
    m = extract_features(_pocket_part())
    spec = {
        "name": "housing",
        "features": [
            {"name": "recess", "type": "pocket", "axis": "z", "position": "x=0 y=0"},
        ],
    }
    report = verify_spec(spec, _verify_res(m))
    checks = _by_name(report)
    assert checks["feat_0_axis"]["status"] == "PASS"
    assert "pk1" in checks["feat_0_axis"]["evidence"]["measured_ids"]


def test_verify_spec_slot_satisfied_by_groove():
    m = extract_features(_channel_part())
    spec = {
        "name": "plate",
        "features": [{"name": "slot_a", "type": "slot", "axis": "x", "count": 1}],
    }
    report = verify_spec(spec, _verify_res(m))
    checks = _by_name(report)
    assert checks["feat_0_axis"]["status"] == "PASS"
    assert "gr1" in checks["feat_0_axis"]["evidence"]["measured_ids"]


def test_verify_spec_pocket_kind_mismatch_on_hole_only():
    """A spec pocket cannot be satisfied by a measured round hole — the kind
    mismatch must FAIL, citing the pocket candidates' absence."""
    m = extract_features(_radial_hole_part())
    spec = {
        "name": "housing",
        "features": [{"name": "recess", "type": "pocket", "count": 1}],
    }
    report = verify_spec(spec, _verify_res(m))
    checks = _by_name(report)
    assert checks["feat_0_kind"]["status"] == "FAIL"
    assert "kind mismatch" in checks["feat_0_kind"]["evidence"]["reason"]


def test_verify_spec_slot_axis_fail_when_groove_runs_elsewhere():
    m = extract_features(_channel_part())
    spec = {
        "name": "plate",
        "features": [{"name": "slot_a", "type": "slot", "axis": "y", "count": 1}],
    }
    report = verify_spec(spec, _verify_res(m))
    checks = _by_name(report)
    assert checks["feat_0_axis"]["status"] == "FAIL"
