"""Feature extraction + spec comparison tests (real geometry, fast parts)."""

from __future__ import annotations

import pytest
from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    Cone,
    CounterBoreHole,
    Cylinder,
    GeomType,
    GridLocations,
    Hole,
    Locations,
    Pos,
    chamfer,
    export_stl,
    fillet,
)

from cadx.cad.features import (
    MeasuredFeatures,
    MeasuredHoleGroup,
    MeasuredPattern,
    assign_feature_ids,
    compare_anchor_positions,
    compare_features,
    extract_features,
    mesh_features_to_measured,
)
from cadx.llm.schemas import Feature


def _plate(with_chamfer: bool = True, fillet_r: float | None = None):
    with BuildPart() as p:
        Box(100, 60, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with GridLocations(80, 40, 2, 2):
            Hole(2.25)
        if fillet_r:
            fillet(p.edges().filter_by(Axis.Z), radius=fillet_r)
        if with_chamfer:
            chamfer(p.faces().sort_by(Axis.Z)[-1].edges(), length=1)
    return p.part


def _cbore_part():
    with BuildPart() as p:
        Box(50, 50, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((0, 0, 10)):
            CounterBoreHole(radius=2.25, counter_bore_radius=4.0, counter_bore_depth=3)
    return p.part


def test_extract_holes_and_edge_features():
    m = extract_features(_plate())
    assert m.total_holes == 4
    assert m.hole_groups[0].diameter == pytest.approx(4.5, abs=0.05)
    assert m.hole_groups[0].axis in ("+z", "-z")
    assert m.hole_groups[0].depth_class == "through"
    assert m.chamfer_faces > 0


def test_extract_fillet_vs_chamfer():
    # fillet only, no chamfer: rounds detected, no chamfer planes on top edges
    with BuildPart() as p:
        Box(50, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), radius=3)
    m = extract_features(p.part)
    assert m.rounds and pytest.approx(m.rounds[0], abs=0.01) == 3.0
    assert m.total_holes == 0


def test_extract_counterbore():
    m = extract_features(_cbore_part())
    assert m.counterbores >= 1
    assert m.total_holes == 2  # stepped coaxial cylinders


def test_extract_blind_hole():
    with BuildPart() as p:
        Box(50, 50, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((0, 0, 10)):
            Hole(2.25, depth=4)
    m = extract_features(p.part)
    assert m.total_holes == 1
    assert m.hole_groups[0].depth_class == "blind"


def test_compare_matching_holes():
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=4.5, count=4, axis="+z")],
        total_holes=4,
    )
    spec = [Feature(name="mounting", type="hole", count=4, standard="M4 clearance")]
    assert compare_features(spec, m) == []


def test_compare_hole_count_mismatch():
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=4.5, count=2, axis="+z")],
        total_holes=2,
    )
    spec = [Feature(name="mounting", type="hole", count=4, standard="M4 clearance")]
    findings = compare_features(spec, m)
    assert any(f.severity == "major" and "4" in f.expected for f in findings)


def test_compare_hole_diameter_mismatch():
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=8.0, count=4, axis="+z")],
        total_holes=4,
    )
    spec = [Feature(name="mounting", type="hole", count=4, standard="M4 clearance")]
    findings = compare_features(spec, m)
    assert any("Ø" in f.actual or "Ø" in f.expected for f in findings)


def test_compare_missing_countersink():
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=4.5, count=4, axis="+z")],
        total_holes=4,
        countersinks=0,
    )
    spec = [Feature(name="csk", type="hole", count=4, standard="M4 ISO 10642 countersunk")]
    findings = compare_features(spec, m)
    assert any(f.feature == "countersinks" for f in findings)


def test_compare_missing_fillet_is_major_on_analytic_geometry():
    """A reliable B-Rep with no round surfaces must not silently satisfy an
    explicit fillet requirement."""
    sharp = MeasuredFeatures(approximate=False)
    spec = [Feature(name="inner fillet", type="fillet", dims={"radius": 3.0})]
    findings = compare_features(spec, sharp)
    assert any(
        f.feature == "inner fillet" and f.severity == "major" and "no analytic" in f.actual
        for f in findings
    )


def test_compare_missing_fillet_stays_conservative_on_approximate_geometry():
    """A faceted source cannot prove the absence of a round face."""
    findings = compare_features(
        [Feature(name="inner fillet", type="fillet", dims={"radius": 3.0})],
        MeasuredFeatures(approximate=True),
    )
    assert not any(f.feature == "inner fillet" for f in findings)


def test_compare_chamfer_vs_fillet():
    m = MeasuredFeatures(rounds=[5.0, 5.0, 5.0, 5.0], chamfer_faces=0)
    spec = [Feature(name="edge", type="chamfer", dims={"length": 1.0})]
    findings = compare_features(spec, m)
    assert any("fillet" in f.actual.lower() or "rounded" in f.actual.lower() for f in findings)


def test_compare_no_holes_measured():
    m = MeasuredFeatures()
    spec = [Feature(name="mounting", type="hole", count=4, standard="M4 clearance")]
    findings = compare_features(spec, m)
    assert any(f.severity == "major" for f in findings)


def test_compare_axis_mismatch_flagged():
    """T0.3: a spec bore along +y with measured holes only on +z is a major
    axis mismatch — the firesteel failure mode (boss bore on wrong axis)."""
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=8.0, count=1, axis="+z")],
        total_holes=1,
    )
    spec = [Feature(name="boss bore", type="hole", count=1, axis="y", dims={"diameter": 8.0})]
    findings = compare_features(spec, m)
    assert any(
        f.severity == "major" and "axis" in f.actual.lower() and "y axis" in f.expected
        for f in findings
    )


def test_compare_slot_feature_uses_slot_count():
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=12.4, count=1, axis="+z", elongated=True)],
        total_holes=1,
        slots=1,
    )
    ok = compare_features([Feature(name="slot_a", type="slot", count=1)], m)
    assert not ok
    missing = compare_features([Feature(name="slot_a", type="slot", count=2)], m)
    assert any(f.severity == "major" and "slot" in f.expected for f in missing)


def test_elongated_group_does_not_satisfy_round_hole():
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=12.4, count=1, axis="+z", elongated=True)],
        total_holes=1,
        slots=1,
    )
    findings = compare_features(
        [Feature(name="mount", type="hole", count=1, dims={"diameter": 12.0})], m
    )
    assert any(f.severity == "major" for f in findings)


def test_compare_axis_sign_insensitive():
    """A through-bore on -z satisfies a spec asking for the z axis."""
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=4.5, count=4, axis="-z")],
        total_holes=4,
    )
    spec = [Feature(name="mounting", type="hole", count=4, axis="+z")]
    assert not any("axis" in f.actual.lower() for f in compare_features(spec, m))


def test_extract_membrane_blocked_hole_is_blind():
    """The Pos(x, y, -1) * Cylinder mis-centring leaves a membrane sealing the
    top of the bore — must classify as blind, not through."""
    with BuildPart() as p:
        Box(60, 40, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, -1) * Cylinder(2.5, 9)
    m = extract_features(p.part)
    assert m.total_holes == 1
    assert m.hole_groups[0].depth_class == "blind"


def test_compare_blocked_hole_flagged():
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=5.0, count=3, axis="+z", depth_class="blind")],
        total_holes=3,
    )
    spec = [Feature(name="mounting", type="hole", count=3, standard="M5 clearance")]
    findings = compare_features(spec, m)
    assert any(f.severity == "major" and "through" in f.expected for f in findings)


def test_compare_blind_spec_hole_not_flagged():
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=5.0, count=3, axis="+z", depth_class="blind")],
        total_holes=3,
    )
    spec = [Feature(name="pocket", type="insert", count=3)]
    findings = compare_features(spec, m)
    assert not any("through" in f.expected for f in findings)


def test_compare_fine_grained_hole_types_checked():
    """Intake emits fine-grained types (through_hole etc.) — they must reach
    the same deterministic checks as the coarse 'hole' type."""
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=5.0, count=2, axis="+z", depth_class="through")],
        total_holes=2,
    )
    for t in ("through_hole", "clearance_hole", "through_hole_with_countersink"):
        spec = [Feature(name="mounting", type=t, count=3, standard="M5 clearance")]
        findings = compare_features(spec, m)
        assert findings, f"type {t} produced no findings"
        assert any(f.severity == "major" for f in findings)


def test_compare_insert_type_not_counted_as_through_hole():
    """An 'insert' spec (blind pocket) must not demand a through verdict."""
    m = MeasuredFeatures(
        hole_groups=[MeasuredHoleGroup(diameter=4.4, count=1, axis="+z", depth_class="blind")],
        total_holes=1,
    )
    spec = [Feature(name="insert pocket", type="insert", count=1)]
    assert not any("through" in f.expected for f in compare_features(spec, m))


def test_extract_coaxial_bore_segments_count_once():
    """A countersink/cone step splits one bore into several coaxial cylinder
    faces — they must merge into a single hole (regression: a 3-csk-hole
    scale read as 6 holes and got a false 'hole count' major)."""
    with BuildPart() as p:
        Box(120, 34, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # three anchors, each bore cut as two stacked same-radius cylinders —
        # exactly what a countersink transition leaves in the B-Rep
        for x in (-46.0, 2.0, 45.0):
            p.part -= Pos(x, 0.0, -0.5) * Cylinder(2.5, 6.0)
            p.part -= Pos(x, 0.0, 5.5) * Cylinder(2.5, 4.0)
    m = extract_features(p.part)
    assert m.total_holes == 3
    g = next(g for g in m.hole_groups if abs(g.diameter - 5.0) < 0.1)
    assert g.count == 3
    assert len(g.centers) == 3


def test_mesh_slot_scan_does_not_fabricate_through_axis_center(tmp_path):
    """Ray-grid centers are on the Y/Z projection plane for an X-through
    slot; the X coordinate is unobserved, not the full-part bbox midpoint."""
    from cadx.cad.cadlib.cutouts import slot_cutter
    from cadx.cad.features import _mesh_through_holes

    foot = Box(60, 40, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
    leg = Pos(-28, 0, 25) * Box(4, 40, 50)
    part = foot + leg
    cutter = slot_cutter(6.5, 15, direction="+z", normal="+x", part=part)
    part = part - Pos(-28, 0, 25) * cutter
    stl_path = tmp_path / "x_through_slot.stl"
    export_stl(part, str(stl_path))

    groups, _, _ = _mesh_through_holes(stl_path)
    slot = next(group for group in groups if group.axis == "+x" and group.elongated)
    assert slot.centers_xyz[0][0] is None
    assert slot.centers_xyz[0][1] == pytest.approx(0.0, abs=0.5)
    assert slot.centers_xyz[0][2] == pytest.approx(25.0, abs=0.5)
    assert slot.centers[0][0] == pytest.approx(0.0, abs=0.5)  # legacy field retained


def test_mesh_feature_adapter_names_projection_axes_for_x_tunnel():
    """A +x tunnel's ray coordinates are Y/Z; its legacy XY pair must not be
    mistaken for a physical X/Y center."""
    measured = mesh_features_to_measured(
        {
            "holes": [
                {
                    "kind": "tunnel",
                    "axis": "x",
                    "x": 0.0,
                    "y": 27.0,
                    "dia": 10.5,
                    "depth": 4.0,
                }
            ]
        }
    )
    group = measured.hole_groups[0]
    assert group.centers == [[0.0, 27.0]]  # compatibility payload unchanged
    assert group.centers_xyz == [[None, 0.0, 27.0]]


def test_extract_countersunk_bore_is_one_hole():
    """Bore + cone countersink at the same axis: one hole, one countersink."""
    with BuildPart() as p:
        Box(60, 40, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, -0.5) * Cylinder(2.5, 9.0)
        p.part -= Pos(0, 0, 6.0) * Cone(
            2.5, 4.25, 2.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
    m = extract_features(p.part)
    assert m.total_holes == 1
    assert m.countersinks == 1


def test_extract_rim_chamfer_and_torus_fillet():
    """Rotational part: a rim chamfer is a cone coaxial to the bore but sits
    at the outer diameter — a chamfer, not a countersink. The boss/base
    fillet is a torus face, not a cylinder."""
    with BuildPart() as p:
        Cylinder(40.0, 10.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
        Pos(0, 0, 10.0) * Cylinder(22.5, 15.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
        top_outer = [
            e for e in p.edges() if e.geom_type == GeomType.CIRCLE and abs(e.radius - 40.0) < 0.01
        ]
        chamfer(top_outer, length=1.0)
        hub_base = [
            e
            for e in p.edges()
            if e.geom_type == GeomType.CIRCLE
            and abs(e.radius - 22.5) < 0.01
            and abs(e.center().Z - 10.0) < 0.01
        ]
        fillet(hub_base, radius=2.0)
        p.part -= Pos(0, 0, -0.5) * Cylinder(15.0, 27.0)
    m = extract_features(p.part)
    assert m.countersinks == 0
    assert m.chamfer_faces >= 1
    assert any(r == pytest.approx(2.0, abs=0.05) for r in m.rounds)
    assert m.total_holes == 1


def test_assign_feature_ids_deterministic_and_stable():
    """M5: identical geometry yields identical ids regardless of input order —
    sorting is on (kind, axis, center)."""
    kwargs = [
        {"diameter": 5.0, "count": 1, "axis": "+z", "centers": [[10.0, 0.0]]},
        {"diameter": 5.0, "count": 1, "axis": "+z", "centers": [[-10.0, 0.0]]},
        {"diameter": 8.0, "count": 1, "axis": "+x", "centers": [[0.0, 4.0]], "elongated": True},
    ]
    m1 = assign_feature_ids(MeasuredFeatures(hole_groups=[MeasuredHoleGroup(**k) for k in kwargs]))
    m2 = assign_feature_ids(
        MeasuredFeatures(hole_groups=[MeasuredHoleGroup(**k) for k in reversed(kwargs)])
    )
    ids1 = {tuple(g.centers[0]) if g.centers else (): g.id for g in m1.hole_groups}
    ids2 = {tuple(g.centers[0]) if g.centers else (): g.id for g in m2.hole_groups}
    assert ids1 == ids2
    assert {g.id for g in m1.hole_groups} == {"h1", "h2", "s1"}
    # h1 sorts before h2 by centre
    assert m1.hole_groups[[g.id for g in m1.hole_groups].index("h1")].centers == [[-10.0, 0.0]]
    slot = next(g for g in m1.hole_groups if g.elongated)
    assert slot.id == "s1"


def test_extracted_features_carry_ids():
    m = extract_features(_plate())
    assert m.hole_groups and all(g.id.startswith("h") for g in m.hole_groups)
    assert [g.id for g in m.hole_groups] == [f"h{i}" for i in range(1, len(m.hole_groups) + 1)]
    # ids survive the ExecResult dict round-trip
    assert m.model_dump()["hole_groups"][0]["id"] == "h1"


def test_compare_anchor_positions_flags_wrong_axis():
    """A hole bored along Y must not satisfy a +z plan-view anchor — a
    sideways pin through the width is not a mounting hole."""
    anchors = {
        "holes": [
            {
                "index": 1,
                "x": 10.0,
                "y": 0.0,
                "dia": 5.0,
                "axis": "+z",
                "source": "image",
            }
        ]
    }
    measured = MeasuredFeatures(
        hole_groups=[
            MeasuredHoleGroup(diameter=5.0, count=1, axis="+y", centers=[(10.0, 0.0, 4.0)])
        ],
        total_holes=1,
    )
    findings = compare_anchor_positions(anchors, measured)
    assert findings, "wrong-axis hole must produce a finding"
    assert any("+y" in f.actual for f in findings)
    assert all(f.actual != "missing" for f in findings)


def test_compare_anchor_positions_z_anchor_matches_z_hole():
    anchors = {
        "holes": [
            {
                "index": 1,
                "x": 10.0,
                "y": 0.0,
                "dia": 5.0,
                "axis": "+z",
                "source": "image",
            }
        ]
    }
    measured = MeasuredFeatures(
        hole_groups=[
            MeasuredHoleGroup(diameter=5.0, count=1, axis="+z", centers=[(10.0, 0.0, 4.0)])
        ],
        total_holes=1,
    )
    assert compare_anchor_positions(anchors, measured) == []


def test_extract_bore_through_plate_only_not_blocked_by_body_below(tmp_path):
    """Bracket bug (M1): a bore through ONE plate of a multi-body part was
    classified 'blind'/'blocked' because a fused body further along the same
    axis blocked the whole-part tunnel ray. The local bore-open walk stops at
    the host boundary — the air gap means open, not a membrane."""
    part = (
        Pos(0, 0, 95) * Box(30, 30, 5, align=(Align.CENTER, Align.CENTER, Align.MIN))
        + Pos(0, 0, 40) * Box(24, 24, 40, align=(Align.CENTER, Align.CENTER, Align.MIN))
        + Pos(10, 0, 40) * Box(4, 30, 57, align=(Align.CENTER, Align.CENTER, Align.MIN))
        - Pos(0, 0, 97.5) * Cylinder(2.75, 8)
    )
    assert len(part.solids()) == 1
    stl = tmp_path / "plate.stl"
    export_stl(part, str(stl))
    m = extract_features(part, stl)
    assert m.total_holes == 1
    g = m.hole_groups[0]
    assert g.diameter == pytest.approx(5.5, abs=0.1)
    assert g.depth_class == "through"
    assert g.blocked == 0


def test_extract_blind_hole_with_mesh_is_blind(tmp_path):
    """Same as test_extract_blind_hole but with the tessellation available —
    the local walk must still find the sealed bottom."""
    with BuildPart() as p:
        Box(30, 30, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, 8.5) * Cylinder(2.5, 3)
    stl = tmp_path / "blind.stl"
    export_stl(p.part, str(stl))
    m = extract_features(p.part, stl)
    assert m.total_holes == 1
    assert m.hole_groups[0].depth_class == "blind"
    assert m.hole_groups[0].blocked == 1


def test_extract_membrane_blocked_hole_with_mesh_is_blind(tmp_path):
    """test_extract_membrane_blocked_hole_is_blind plus the mesh path: the
    membrane sealing the bore top must still classify as blind."""
    with BuildPart() as p:
        Box(60, 40, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, -1) * Cylinder(2.5, 9)
    stl = tmp_path / "membrane.stl"
    export_stl(p.part, str(stl))
    m = extract_features(p.part, stl)
    assert m.total_holes == 1
    assert m.hole_groups[0].depth_class == "blind"
    assert m.hole_groups[0].blocked == 1


def test_pattern_check_ignores_gear_teeth_pitch_circle():
    """Regression: a gear's 'involute tooth ring' feature legitimately mentions
    'pitch circle Ø40' — the pattern check must not demand 20 hole-pattern
    members from it (observed: gear scored 55 chasing a phantom pattern)."""
    m = MeasuredFeatures(
        recognition_ran=True,
        hole_patterns=[MeasuredPattern(kind="bolt_circle", count=4, diameter=28.0)],
        total_holes=5,
    )
    spec = [
        Feature(
            name="involute tooth ring",
            type="gear_teeth",
            count=20,
            description="20 involute spur teeth, module 2, pitch circle Ø40 mm",
        )
    ]
    assert compare_features(spec, m) == []


def test_pattern_check_still_fires_for_hole_bolt_circle():
    """A hole feature described as a bolt circle remains pattern-checked —
    the hole-ish gate must not swallow real pattern expectations."""
    m = MeasuredFeatures(
        recognition_ran=True,
        hole_patterns=[MeasuredPattern(kind="bolt_circle", count=4, diameter=28.0)],
        total_holes=5,
    )
    spec = [
        Feature(
            name="mounting holes",
            type="through_hole",
            count=6,
            description="6 holes on bolt circle Ø40",
        )
    ]
    findings = compare_features(spec, m)
    assert any(f.severity == "major" and "pattern" in f.expected for f in findings)


def test_small_hole_dfm_ignores_fillets_and_degenerate():
    """DFM small_hole: a real sub-1mm bore warns; a concave edge fillet and a
    degenerate zero-radius read do not (observed: gear tooth roots produced
    spurious 'hole radius 0.00 mm' warnings)."""
    from cadx.cad.analysis import analyze
    from cadx.cad.dfm.fdm import fdm_brep_checks

    with BuildPart() as p:
        Box(40, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        Hole(0.8)  # r=0.8 real through-bore -> warn
    warns = fdm_brep_checks(p.part, analyze(p.part))
    assert any(w.code == "small_hole" for w in warns)

    with BuildPart() as q:
        Box(40, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # inner pocket -> inside vertical edges; fillet them concavely —
        # a concave fillet arc is not a hole
        q.part -= Pos(0, 0, 5) * Box(20, 20, 5.01, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(q.part.edges().filter_by(Axis.Z), radius=0.5)
    warns2 = fdm_brep_checks(q.part, analyze(q.part))
    assert not any(w.code == "small_hole" for w in warns2)


def test_extract_slot_reports_through_and_elong_axes(tmp_path):
    """P4 slot semantics: a leg slot through the X-thin wall, elongated
    along Z — the group must label through_axis +x separately from
    elong_axis +z (the bare `axis` field used to be misread as the
    elongation and spawned the phantom 'slot axis wrong' critique)."""
    with BuildPart() as p:
        Box(8, 30, 40, align=(Align.CENTER, Align.CENTER, Align.MIN))
        p.part -= Pos(0, 0, 20) * Box(12, 6.5, 15)
    assert len(p.part.solids()) == 1
    stl = tmp_path / "leg_slot.stl"
    export_stl(p.part, str(stl))
    m = extract_features(p.part, stl)
    slots = [g for g in m.hole_groups if g.elongated]
    assert len(slots) == 1
    g = slots[0]
    assert g.axis == "+x"  # the tunnel/pierce direction
    assert g.through_axis == "+x"
    assert g.elong_axis == "+z"  # the span direction the slot runs along
    assert g.width_mm == pytest.approx(6.5, abs=1.0)
    assert g.length_mm == pytest.approx(15.0, abs=1.0)
    # the new fields survive the ExecResult dict round-trip
    d = next(s for s in m.model_dump()["hole_groups"] if s["elongated"])
    for key in ("through_axis", "elong_axis", "width_mm", "length_mm"):
        assert key in d
    assert d["elong_axis"] == "+z"


def test_round_hole_group_reports_through_axis_only():
    """Round bores carry through_axis (same as axis) and empty slot dims."""
    m = extract_features(_plate())
    g = m.hole_groups[0]
    assert not g.elongated
    assert g.through_axis == g.axis
    assert g.elong_axis == ""
    assert g.width_mm == 0.0 and g.length_mm == 0.0


def test_mesh_holes_bore_fit_lifts_shadowed_column(tmp_path):
    """Mesh-path regression: a Ø6.5 wall bore whose void column is partly
    shadowed by interior material reads ~Ø5 from the zero-hit raster —
    the ring-ray bore fit must lift the group back toward the true bore
    (the robot-arm flange defect: gate measured Ø5.3 on a real Ø6.5)."""
    import math

    import trimesh

    from cadx.cad.features import _mesh_through_holes

    wall = trimesh.creation.box(extents=[8, 40, 40])
    wall.apply_translation([46, 0, 0])
    rib = trimesh.creation.box(extents=[10, 5.0, 8])
    rib.apply_translation([35, 3.5, 0])  # shadows bore rays for y > 1
    bore = trimesh.creation.cylinder(radius=3.25, height=25, sections=64)
    bore.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    bore.apply_translation([42, 0, 0])
    part = wall.difference(bore, engine="manifold").union(rib, engine="manifold")
    stl = tmp_path / "shadowed_bore.stl"
    part.export(str(stl))
    groups, _, _ = _mesh_through_holes(stl)
    bore_groups = [g for g in groups if g.axis == "+x" and not g.elongated]
    assert len(bore_groups) == 1
    # true bore is Ø6.5; the shadowed void column alone reads ~Ø5
    assert bore_groups[0].diameter == pytest.approx(6.5, abs=0.6)


def test_inside_fraction_prefers_mesh_contains():
    """The mesh contains() oracle replaces OCC is_inside (~200 ms/pt on curved
    B-Reps). A stub callable proves the vectorised path is taken and controls
    the verdict."""
    from cadx.cad.features import _inside_fraction

    part = _plate(with_chamfer=False)
    seen: list[int] = []

    def contains(pts):
        seen.append(len(pts))
        return [True] * len(pts)

    frac = _inside_fraction(part, [0, 0, 4], [0, 0, 1], 2.0, contains=contains)
    assert frac == 1.0 and seen == [8]
    frac = _inside_fraction(
        part, [0, 0, 4], [0, 0, 1], 2.0, contains=lambda pts: [False] * len(pts)
    )
    assert frac == 0.0


def test_inside_fraction_mesh_real_stl(tmp_path):
    """End-to-end: a through bore classifies as hole via the real STL mesh —
    same verdict the OCC path produced, ~4000x cheaper per point."""
    from build123d import export_stl

    from cadx.cad.features import _inside_fraction, _load_watertight_mesh, _mesh_contains_fn

    part = _plate(with_chamfer=False)
    stl = tmp_path / "plate.stl"
    export_stl(part, str(stl))
    mesh = _load_watertight_mesh(stl)
    assert mesh is not None
    # hole centre: a radial sample ring around it sits inside the wall material
    frac = _inside_fraction(part, [40, 20, 4], [0, 0, 1], 2.25, contains=_mesh_contains_fn(mesh))
    assert frac >= 0.7


def test_extract_features_accepts_preloaded_mesh(tmp_path):
    """The pipeline loads the analysis STL once and shares it — the kwarg path
    must yield the same measurements as the from-path load."""
    from build123d import export_stl

    from cadx.cad.features import _load_watertight_mesh

    part = _plate(with_chamfer=False)
    stl = tmp_path / "plate.stl"
    export_stl(part, str(stl))
    m1 = extract_features(part, stl)
    m2 = extract_features(part, stl, mesh=_load_watertight_mesh(stl))
    assert m1.total_holes == m2.total_holes == 4
    assert m1.hole_groups[0].diameter == pytest.approx(m2.hole_groups[0].diameter)


def test_extract_features_cancelled_returns_early(tmp_path):
    """Cooperative cancellation: the OCC-heavy passes must yield on the
    cancelled flag — the executor cannot kill a timed-out thread."""
    part = _plate(with_chamfer=False)
    m = extract_features(part, cancelled=lambda: True)
    assert m.total_holes == 0
    assert m.hole_groups == []
