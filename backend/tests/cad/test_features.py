"""Feature extraction + spec comparison tests (real geometry, fast parts)."""

from __future__ import annotations

import pytest
from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    CounterBoreHole,
    Cylinder,
    GridLocations,
    Hole,
    Locations,
    Pos,
    chamfer,
    fillet,
)

from cadx.cad.features import (
    MeasuredFeatures,
    MeasuredHoleGroup,
    compare_features,
    extract_features,
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
