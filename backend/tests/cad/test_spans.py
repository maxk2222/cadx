"""cad/spans.py — declared from-to connections must carry material."""

import trimesh
from build123d import Box, Pos

from cadx.cad import cadlib
from cadx.cad.spans import declared_spans, plate_checks, span_checks, span_coverage


def _features(position: str) -> list[dict]:
    return [{"name": "diagonal_brace", "type": "box", "position": position}]


def test_declared_spans_finds_from_to_features():
    feats = _features(
        "from (x=-55, z=35) on the leg to (x=+15, z=95) on the arm underside, centred in Y"
    )
    spans = declared_spans(feats)
    assert len(spans) == 1
    assert spans[0]["points"] == [[-55.0, 0.0, 35.0], [15.0, 0.0, 95.0]]


def test_declared_spans_skips_voids_and_single_points():
    assert (
        declared_spans([{"name": "h", "type": "through_hole", "position": "from x=0 to x=10"}])
        == []
    )
    assert declared_spans(_features("at x=5, y=0")) == []


def test_span_coverage_connected_strut():
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    part = leg + arm + cadlib.strut((-55, 0, 35), (15, 0, 95), 10, 6)
    assert span_coverage(part, [-55, 0, 35], [15, 0, 95]) >= 0.85


def test_span_coverage_detached_strut_fails():
    """The live defect: strut built from x=-5 (leg_thickness misused as the
    x coordinate) floats between the leg and the arm — the declared path
    leg→arm is mostly air."""
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    part = leg + arm + cadlib.strut((-5, 0, 35), (15, 0, 95), 10, 6)
    assert span_coverage(part, [-55, 0, 35], [15, 0, 95]) < 0.85


def test_span_checks_end_to_end():
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    good = leg + arm + cadlib.strut((-55, 0, 35), (15, 0, 95), 10, 6)
    feats = _features(
        "from (x=-55, z=35) on the leg to (x=+15, z=95) on the arm underside, centred in Y"
    )
    res = span_checks(good, feats)
    assert len(res) == 1 and res[0]["ok"] and res[0]["coverage"] >= 0.85
    bad = leg + arm + cadlib.strut((-5, 0, 35), (15, 0, 95), 10, 6)
    res2 = span_checks(bad, feats)
    assert len(res2) == 1 and not res2[0]["ok"]


# ---------------------------------------------------------------------------
# plate_checks


def _plate_feature() -> list[dict]:
    return [
        {
            "name": "Mounting lug -x",
            "type": "pad",
            "position": "centred at x=-45, y=0; lug extends x=-60..-30, y=-10..10, z=0..6",
        }
    ]


def test_plate_checks_flags_overshoot():
    """Live hybrid failure: the mesh lug pad stays organic at ~40 mm in Y
    while the spec declares y=-10..10 — renders looked 'flat enough' so the
    critic never saw it. The band sample must flag it."""
    pad = Pos(-45, 0, 10) * Box(30, 40, 20)  # organic pad 40 wide in Y
    res = plate_checks(pad, _plate_feature())
    assert len(res) == 1
    assert not res[0]["ok"]
    assert res[0]["measured_span_mm"]["y"] > 20


def test_plate_checks_passes_declared_footprint():
    leg = Pos(-33, 0, 15) * Box(6, 14, 30)  # leg rising at pad inner edge
    pad = Pos(-45, 0, 3) * Box(30, 20, 6)
    part = pad + leg
    res = plate_checks(part, _plate_feature())
    assert len(res) == 1
    assert res[0]["ok"]


def test_plate_checks_skips_unranged_features():
    assert (
        plate_checks(Box(10, 10, 10), [{"name": "pad", "type": "pad", "position": "centred"}]) == []
    )


def test_plate_checks_synthesises_footprint_from_centre_and_dims():
    """Real handle spec: 'centered at x=-45, y=0, z=0..6' + dims
    {length,width} — no x/y ranges in text. The centre point plus dims
    must still bound the footprint check."""
    feats = [
        {
            "name": "mounting lug -x",
            "type": "pad",
            "position": "centered at x=-45, y=0, z=0..6",
            "dims": {"length": 30, "width": 20, "thickness": 6},
        }
    ]
    pad = Pos(-45, 0, 3) * Box(30, 40, 6)  # organic flange 40 in Y
    res = plate_checks(pad, feats)
    assert len(res) == 1
    assert res[0]["declared"]["x"] == (-60.0, -30.0)
    assert res[0]["declared"]["y"] == (-10.0, 10.0)
    assert not res[0]["ok"]
    # and the spec-conform pad passes the same declaration
    ok = plate_checks(Pos(-45, 0, 3) * Box(30, 20, 6), feats)
    assert ok[0]["ok"]


# ---------------------------------------------------------------------------
# plate_checks — hybrid exemption
#
# On a mesh-derived solid the runner attaches the source trimesh as
# `shape._cadx_mesh`; the band probe then reads the organic base body, whose
# whole extent may legitimately span a declared plate footprint — but only
# while it stays inside the STATED overall dims.


def _organic_envelope() -> tuple:
    """(build123d shape, source trimesh) — both cover 120x40x30 resting on
    z>=0, centred in XY: the hybrid case where the sewn solid and the raw
    mesh share the envelope."""
    shape = Pos(0, 0, 15) * Box(120, 40, 30)
    mesh = trimesh.creation.box(extents=[120, 40, 30])
    # trimesh.creation.box is origin-centred — shift up so it rests on z>=0
    mesh.apply_translation([0.0, 0.0, 15.0])
    return shape, mesh


def _organic_pad_feature() -> list[dict]:
    return [
        {
            "type": "tab",
            "name": "pad",
            "position": "centered at x=-45, y=0, z=0..6",
            "dims": {"length": 30, "width": 20, "thickness": 6},
        }
    ]


_STATED_ENVELOPE = {
    "length_x": {"value": 120, "source": "stated"},
    "width_y": {"value": 40, "source": "stated"},
}


def test_plate_checks_spec_dict_alone_does_not_exempt_cad_shape():
    """Pure-CAD behaviour unchanged: stated overall dims without a source
    mesh don't excuse the band overshoot."""
    shape, _mesh = _organic_envelope()
    res = plate_checks(shape, _organic_pad_feature(), spec_dict=_STATED_ENVELOPE)
    assert len(res) == 1
    assert not res[0]["ok"]


def test_plate_checks_mesh_solid_exempted_within_stated_envelope():
    """Band extent ~40 in Y overshoots the declared 20 mm pad width but sits
    inside the stated 120x40 envelope — the organic base body legitimately
    spans the footprint."""
    shape, mesh = _organic_envelope()
    shape._cadx_mesh = mesh
    res = plate_checks(shape, _organic_pad_feature(), spec_dict=_STATED_ENVELOPE)
    assert len(res) == 1
    assert res[0]["ok"]


def test_plate_checks_mesh_exemption_requires_stated_source():
    """An `inferred` overall dim is the agent's own guess — it cannot excuse
    the overshoot the way a stated requirement can."""
    shape, mesh = _organic_envelope()
    shape._cadx_mesh = mesh
    spec = {
        "length_x": {"value": 120, "source": "stated"},
        "width_y": {"value": 40, "source": "inferred"},
    }
    res = plate_checks(shape, _organic_pad_feature(), spec_dict=spec)
    assert len(res) == 1
    assert not res[0]["ok"]


def test_plate_checks_mesh_extent_beyond_stated_envelope_still_flags():
    """The exemption is bounded by the stated dims — a measured band wider
    than even the declared overall width is real overshoot, not organics."""
    shape, mesh = _organic_envelope()
    shape._cadx_mesh = mesh
    spec = {
        "length_x": {"value": 120, "source": "stated"},
        "width_y": {"value": 30, "source": "stated"},
    }
    res = plate_checks(shape, _organic_pad_feature(), spec_dict=spec)
    assert len(res) == 1
    assert not res[0]["ok"]
