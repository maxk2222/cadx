from __future__ import annotations

from pathlib import Path

import pytest
from build123d import Align, Box, Cone, Cylinder, Pos, export_stl, import_stl

from cadx.cad.features import _mesh_through_holes, extract_features


@pytest.fixture()
def bored_stl(tmp_path: Path) -> Path:
    part = Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for x in (-12.0, 0.0, 12.0):
        part -= Pos(x, 0, 4) * Cylinder(radius=2, height=20)
    path = tmp_path / "bored.stl"
    export_stl(part, str(path))
    return path


def test_mesh_through_holes_detects_bores(bored_stl: Path) -> None:
    groups, csk, slots = _mesh_through_holes(bored_stl)
    assert slots == 0
    z_groups = [g for g in groups if g.axis == "+z"]
    assert len(z_groups) == 3
    for g in z_groups:
        assert 3.0 < g.diameter < 5.5
        assert g.depth_class == "through"
    assert csk == 0


def test_mesh_through_holes_detects_countersink(tmp_path: Path) -> None:
    part = Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part -= Pos(0, 0, 4) * Cylinder(radius=2, height=20)
    part -= Pos(0, 0, 7) * Cone(bottom_radius=2, top_radius=4.25, height=2)
    path = tmp_path / "csk.stl"
    export_stl(part, str(path))
    groups, csk, _ = _mesh_through_holes(path)
    assert len([g for g in groups if g.axis == "+z"]) == 1
    assert csk == 1


def test_mesh_through_holes_ignores_solid(tmp_path: Path) -> None:
    path = tmp_path / "solid.stl"
    export_stl(Box(40, 20, 8), str(path))
    groups, csk, _ = _mesh_through_holes(path)
    assert groups == []
    assert csk == 0


def test_mesh_through_holes_ignores_open_slot(tmp_path: Path) -> None:
    # slot open to the silhouette edge is a bay, not a tunnel — must not
    # report a hole even though rays pass through it unobstructed
    part = Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part -= Pos(0, 14, -1) * Box(10, 12, 10)
    path = tmp_path / "slot.stl"
    export_stl(part, str(path))
    groups, _, _ = _mesh_through_holes(path)
    assert [g for g in groups if g.axis == "+z"] == []


def test_mesh_through_holes_detects_enclosed_slot(tmp_path: Path) -> None:
    # 20x6 mm enclosed through-void: elongated → reported as a slot and
    # flagged on its group, not counted as a round bore
    part = Box(70, 30, 5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cutter = Box(14, 6, 10, align=(Align.CENTER,) * 3)
    cutter += Pos(-7, 0, 0) * Cylinder(3, 10) + Pos(7, 0, 0) * Cylinder(3, 10)
    part -= Pos(0, 0, 2.5) * cutter  # cutter protrudes both faces — no membrane
    path = tmp_path / "slot.stl"
    export_stl(part, str(path))
    groups, _, slots = _mesh_through_holes(path)
    assert slots >= 1
    z_groups = [g for g in groups if g.axis == "+z"]
    assert z_groups and all(g.elongated for g in z_groups)


def test_extract_features_mesh_fallback(bored_stl: Path) -> None:
    faceted = import_stl(str(bored_stl))
    no_stl = extract_features(faceted)
    assert no_stl.total_holes == 0
    with_stl = extract_features(faceted, bored_stl)
    assert with_stl.total_holes == 3
    assert with_stl.approximate is True


def test_extract_features_merge_keeps_analytic_confidence(bored_stl: Path) -> None:
    """Merge branch (analytic holes + mesh slot/tunnel scan) must NOT mark
    the result approximate — the B-Rep measurements stay authoritative and
    _fillet_check must keep FAILing on a genuinely missing declared fillet
    (regression: l_bracket shipped without its spec fillets, verify_spec
    PASSed because the merge branch poisoned `approximate`)."""
    part = Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for x in (-12.0, 0.0, 12.0):
        part -= Pos(x, 0, 4) * Cylinder(radius=2, height=20)
    out = extract_features(part, bored_stl)
    assert out.total_holes == 3
    assert out.approximate is False


def test_fillet_check_fails_on_reliable_brep(tmp_path: Path) -> None:
    """A spec fillet missing from a reliable analytic solid → FAIL, not
    UNVERIFIED (the status only survives on approximate evidence)."""
    from cadx.cad.verifyspec import _fillet_check

    checks: list[dict] = []
    _fillet_check(
        0,
        "edge_fillet",
        {"type": "fillet", "dims": {"radius": 3.0}},
        {"rounds": [], "fillets": [], "approximate": False},
        checks,
    )
    assert checks and checks[0]["status"] == "FAIL"
    checks.clear()
    _fillet_check(
        0,
        "edge_fillet",
        {"type": "fillet", "dims": {"radius": 3.0}},
        {"rounds": [], "fillets": [], "approximate": True},
        checks,
    )
    assert checks and checks[0]["status"] == "UNVERIFIED"


def test_verify_spec_flags_degraded_topopt_holes() -> None:
    """Mesh-measured features on the topopt path must reach verify_spec: a
    spec bore group that shrank/collapsed to fewer members is a MAJOR
    finding → spec_features FAIL (regression: robot_arm v9 shipped 1 of 2
    flange holes, verify_spec saw no evidence at all → UNVERIFIED → PASS)."""
    from cadx.cad.features import MeasuredFeatures, compare_features
    from cadx.cad.verifyspec import verify_spec
    from cadx.llm.schemas import Feature

    spec = {
        "features": [
            {
                "name": "flange_holes",
                "type": "hole",
                "count": 2,
                "axis": "+x",
                "dims": {"diameter": 6.5},
            }
        ]
    }
    # mesh-measured evidence: one surviving +x bore of wrong diameter —
    # exactly the degraded shape the v9 run produced
    measured = MeasuredFeatures(
        approximate=True,
        hole_groups=[
            {
                "id": "h1",
                "axis": "+x",
                "diameter": 3.87,
                "count": 1,
                "centers": [[80.0, 20.0]],
                "depth_class": "through",
            }
        ],
        total_holes=1,
    )
    feats = [Feature.model_validate(f) for f in spec["features"]]
    findings = [f.model_dump() for f in compare_features(feats, measured)]
    report = verify_spec(
        spec,
        {"ok": True, "features": measured.model_dump(), "analysis": {}},
        feature_findings=findings,
    )
    spec_check = next(c for c in report["checks"] if c["name"] == "spec_features")
    assert spec_check["status"] == "FAIL"
    assert any(f["severity"] == "major" for f in findings)


def test_boss_axis_detected_and_compared() -> None:
    """T8.4: a protruding cylinder (boss/tube) is measured with its axis, and
    a spec asking for a different axis produces a major finding."""
    from types import SimpleNamespace

    from build123d import Rot

    from cadx.cad.features import compare_features

    base = Pos(0, 0, 4) * Box(60, 20, 8)
    tube_x = Pos(0, 0, 8) * Rot(0, 90, 0) * Cylinder(radius=6, height=60)
    mf = extract_features(base + tube_x)
    assert any(b.axis in ("+x", "-x") for b in mf.bosses), mf.bosses

    # same tube standing vertical: boss axis is +z — spec wanting +x must flag
    tube_z = Pos(0, 0, 15) * Cylinder(radius=6, height=20)
    mf_z = extract_features(base + tube_z)
    feat = SimpleNamespace(
        name="firesteel tube",
        type="boss",
        description="tube",
        dims={},
        position="",
        axis="x",
        count=1,
        standard="",
    )
    findings = compare_features([feat], mf_z)
    assert any(f.severity == "major" and "z axis" in f.actual for f in findings), findings
    assert compare_features([feat], mf) == []


def test_typed_rib_with_boss_in_prose_is_not_a_tube_demand() -> None:
    """Regression (robot_arm_mount E2E): a `type:"rib"` feature whose
    description references another boss ("rib joining the bearing boss")
    must NOT be reclassified by the text regex into a boss-axis demand.
    Untyped/other features keep the rescue path."""
    from types import SimpleNamespace

    from cadx.cad.features import compare_features

    base = Pos(0, 0, 4) * Box(60, 20, 8)
    tube_z = Pos(0, 0, 15) * Cylinder(radius=6, height=20)
    mf = extract_features(base + tube_z)  # only a +z boss exists
    kw = dict(dims={}, position="", axis="x", count=1, standard="")

    rib = SimpleNamespace(
        name="gusset rib",
        type="rib",
        description="gusset joining the bearing boss to the flange",
        **kw,
    )
    assert not any("boss/tube" in f.expected for f in compare_features([rib], mf))

    untyped = SimpleNamespace(
        name="lagerstutzen",
        type="other",
        description="cylindrical boss for the shaft",
        **kw,
    )
    assert any("boss/tube" in f.expected for f in compare_features([untyped], mf))


def test_fillet_does_not_register_as_boss() -> None:
    """Edge fillets wrap ~90° — they must stay in `rounds`, not become bosses."""
    from build123d import Axis, fillet

    part = Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = fillet(part.edges().filter_by(Axis.Z), 2)
    mf = extract_features(part)
    assert mf.bosses == []
    assert mf.rounds


def test_mesh_profile_reports_spine(tmp_path: Path) -> None:
    # stepped body: a tab shifts the y-extent at one end — the profile must
    # show the drift that a bbox alone would hide
    from build123d import export_brep

    from cadx.cad.runner_entry import _mesh_profile

    part = Box(40, 10, 8) + Pos(25, 8, 0) * Box(10, 10, 8)
    path = tmp_path / "stepped.brep"
    export_brep(part, str(path))
    rows = _mesh_profile(path)["x"]
    assert len(rows) >= 2
    for r in rows:
        assert r[1] <= r[2] and r[3] <= r[4]
        assert r[3] >= -4.1 and r[4] <= 4.1  # z stays within ±4
    left = [r for r in rows if r[0] < -10]
    right = [r for r in rows if r[0] > 20]
    assert left and right
    assert all(abs(r[1] + 5) < 0.6 and abs(r[2] - 5) < 0.6 for r in left)
    assert all(r[2] > 10 for r in right)  # tab pulls ymax out to ~13
