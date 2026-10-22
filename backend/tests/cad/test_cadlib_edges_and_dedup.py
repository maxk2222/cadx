"""cadlib edge-selection helpers + mesh-fallback dedup + spec-axis checks.

Covers the sandbox-safe edge helpers added to ``cadlib.ops``
(``edges_at``/``concave_edges``/``convex_edges``/``hole_rim_edges``/
``safe_chamfer``), the ``inspect_tools.list_edges`` inventory primitive,
the ``_mesh_fallback`` in-plane dedup regression (side holes counted
twice) and the verifyspec through-axis position/phrase handling.
"""

from __future__ import annotations

from build123d import (
    Align,
    Box,
    BuildPart,
    Cylinder,
    Locations,
    Pos,
    Rot,
    export_stl,
)

from cadx.cad import inspect_tools
from cadx.cad.cadlib import (
    concave_edges,
    convex_edges,
    edges_at,
    hole_rim_edges,
    safe_chamfer,
    safe_fillet,
)
from cadx.cad.features import extract_features
from cadx.cad.verifyspec import _spec_axis, verify_spec


def _l_bracket():
    """80x40x6 plate (z in [0,6]) plus a 4 mm wall on its -x end — the
    re-entrant inner corner edge runs along y at (x=-36, z=6)."""
    plate = Box(80, 40, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    wall = Pos(-38, 0, 6) * Box(4, 40, 24, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return plate + wall


def _drilled_plate():
    """40x30x6 plate, z in [0,6], with a Ø5 through-bore on +z."""
    plate = Box(40, 30, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    plate -= Pos(0, 0, -1) * Cylinder(2.5, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return plate


def test_concave_edges_finds_l_bracket_inner_corner():
    edges = concave_edges(_l_bracket(), axis="y")
    assert edges, "inner join edge not classified concave"
    assert any(
        abs(e.center().X + 36.0) < 0.5 and abs(e.center().Z - 6.0) < 0.5 for e in edges
    )


def test_convex_edges_finds_outer_silhouette():
    edges = convex_edges(_l_bracket(), axis="z")
    assert edges  # vertical outer silhouette edges are all convex


def test_safe_fillet_and_chamfer_on_concave_edges():
    with BuildPart() as p:
        Box(80, 40, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((-38, 0, 6)):
            Box(4, 40, 24, align=(Align.CENTER, Align.CENTER, Align.MIN))
        edges = concave_edges(axis="y")
        assert edges
        assert safe_fillet(edges, radius=2.0) > 0.0
    assert len(p.part.solids()) == 1

    with BuildPart() as p2:
        Box(80, 40, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((-38, 0, 6)):
            Box(4, 40, 24, align=(Align.CENTER, Align.CENTER, Align.MIN))
        edges = concave_edges(axis="y")
        assert edges
        assert safe_chamfer(edges, length=1.5) > 0.0
    assert len(p2.part.solids()) == 1


def test_safe_chamfer_empty_selection_returns_zero():
    with BuildPart():
        Box(10, 10, 10)
        assert safe_chamfer([], length=1.0) == 0.0


def test_edges_at_filters():
    part = _l_bracket()
    assert len(edges_at(part)) == len(part.edges())
    z_edges = edges_at(part, axis="z")
    assert z_edges
    for e in z_edges:
        t = e.tangent_at(0.5)
        tn = (t.X**2 + t.Y**2 + t.Z**2) ** 0.5
        assert abs(t.Z / tn) > 0.98
    # exact-coordinate + range narrowing both reach the inner corner edge
    corner = edges_at(part, axis="y", x=-36.0, z=6.0, tol=1.0)
    assert corner
    ranged = edges_at(part, axis="y", x_range=(-40.0, -30.0), z_range=(5.0, 7.0))
    assert ranged
    assert edges_at(part, geom="circle") == []


def test_edges_at_respects_range_bounds():
    part = Box(20, 20, 20)  # x/y/z edges at +/-10
    top = edges_at(part, axis="x", z_range=(9.0, 11.0))
    assert top
    assert all(abs(e.center().Z - 10.0) < 0.5 for e in top)
    far = edges_at(part, axis="x", z_range=(30.0, 40.0))
    assert far == []


def test_hole_rim_edges():
    plate = _drilled_plate()
    rims = hole_rim_edges(plate, diameter=5.0, axis="z")
    assert len(rims) == 2  # the bore's top + bottom rims
    assert all(abs(2.0 * e.radius - 5.0) <= 0.5 for e in rims)
    top = hole_rim_edges(plate, diameter=5.0, axis="z", z=6.0)
    assert len(top) == 1
    assert abs(top[0].center().Z - 6.0) < 0.01
    # wrong axis / wrong diameter select nothing
    assert hole_rim_edges(plate, diameter=5.0, axis="x") == []
    assert hole_rim_edges(plate, diameter=9.0) == []


def test_list_edges_inventory():
    part = _l_bracket()
    out = inspect_tools.list_edges(part)
    assert "error" not in out
    assert out["total"] == len(part.edges())
    assert out["by_axis"]
    assert out["convex"]
    assert any(
        abs(e["center"][0] + 36.0) < 0.5 and abs(e["center"][2] - 6.0) < 0.5
        for e in out["concave"]
    )
    y_only = inspect_tools.list_edges(part, axis="y")
    assert y_only["total"] < out["total"]
    for e in y_only["concave"] + y_only["convex"]:
        assert e["axis"].endswith("y")
    assert "error" in inspect_tools.list_edges(part, axis="w")


def test_list_edges_circles_labelled_by_normal():
    out = inspect_tools.list_edges(_drilled_plate())
    assert len(out["circles"]) == 2
    assert {e["axis"] for e in out["circles"]} == {"+z", "-z"} or all(
        e["axis"].endswith("z") for e in out["circles"]
    )
    z = inspect_tools.list_edges(_drilled_plate(), axis="z")
    assert len(z["circles"]) == 2


def test_mesh_fallback_dedups_side_holes(tmp_path):
    """Regression: B-Rep hole groups store a raw ``[x, y]`` projection in
    ``centers`` — for +x bores that is ``[through_axis_x, y]`` — while mesh
    tunnel groups carry the bbox midpoint on the through axis there. The
    legacy literal comparison never matched, so every side hole was counted
    twice (6 instead of 4). In-plane (u, v) centres from ``centers_xyz``
    must collapse them."""
    plate = Box(80, 60, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    wall = Pos(-38, 0, 6) * Box(4, 60, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = plate + wall
    for y in (20.0, -20.0):
        part -= Pos(25, y, -1) * Cylinder(
            2.5, 10, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
    for y in (15.0, -15.0):
        part -= Pos(-38, y, 16) * Rot(0, 90, 0) * Cylinder(2.5, 10)
    stl = tmp_path / "part.stl"
    export_stl(part, str(stl))
    m = extract_features(part, stl)
    assert m.total_holes == 4
    axes = sorted(
        (g.axis or "").lstrip("+-") for g in m.hole_groups if not g.elongated
    )
    assert axes == ["x", "z"]


def _verify_res(bbox_min: list[float], bbox_max: list[float]) -> dict:
    return {
        "ok": True,
        "error": None,
        "analysis": {
            "bbox_size": [b - a for a, b in zip(bbox_min, bbox_max, strict=True)],
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "solid_count": 1,
            "is_valid": True,
            "watertight": True,
            "source": "brep",
        },
        "features": {
            "step_roundtrip": {"checked": True, "ok": True, "reimported_valid": True},
        },
        "dfm": {"process": "fdm", "warnings": [], "summary": ""},
    }


def test_verifyspec_through_axis_position_pass():
    """'through z=0..6' is a through-axis span, not an in-plane centre
    coordinate — stripping it must let the two declared hole points verify
    cleanly against measured (x, y) centres."""
    plate = Box(80, 60, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for y in (20.0, -20.0):
        plate -= Pos(25, y, -1) * Cylinder(
            2.5, 10, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
    measured = extract_features(plate).model_dump()
    spec = {
        "name": "plate",
        "length_x": {"value": 80.0, "source": "stated"},
        "width_y": {"value": 60.0, "source": "stated"},
        "height_z": {"value": 6.0, "source": "stated"},
        "features": [
            {
                "name": "base holes",
                "type": "through_hole",
                "axis": "z",
                "count": 2,
                "dims": {"diameter": 5.0},
                "position": "at (x=25, y=+20) and (x=25, y=-20), through z=0..6",
            }
        ],
    }
    res = _verify_res([-40.0, -30.0, 0.0], [40.0, 30.0, 6.0])
    report = verify_spec(spec, res, feature_findings=[], measured_features=measured)
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["feat_0_axis"]["status"] == "PASS", checks["feat_0_axis"]
    assert checks["feat_0_position"]["status"] == "PASS", checks["feat_0_position"]
    assert report["status"] == "PASS", report["summary"]


def test_verifyspec_on_axis_phrase_is_not_an_axis_decl():
    """'one hole on +Y axis' names the FACE the hole sits on — it must not
    read as a +Y through-bore declaration and fail a plate of +z holes."""
    spec = {
        "name": "plate",
        "features": [
            {
                "name": "mount holes",
                "type": "through_hole",
                "count": 6,
                "dims": {"diameter": 5.0},
                "position": "on the top face",
                "description": "5 holes on PCD 100, one hole on +Y axis",
            }
        ],
    }
    measured = {
        "hole_groups": [
            {
                "id": "h1",
                "axis": "+z",
                "through_axis": "+z",
                "diameter": 5.0,
                "count": 6,
                "centers_xyz": [[10.0, 0.0, None]],
            }
        ],
        "total_holes": 6,
    }
    res = _verify_res([-55.0, -55.0, 0.0], [55.0, 55.0, 6.0])
    report = verify_spec(spec, res, feature_findings=[], measured_features=measured)
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["feat_0_kind"]["status"] == "PASS", checks.get("feat_0_kind")
    assert checks.get("feat_0_axis", {}).get("status") != "FAIL"


def test_spec_axis_phrase_masking():
    assert _spec_axis({"description": "5 holes on PCD 100, one hole on +Y axis"}) == ""
    assert _spec_axis({"position": "on the -x axis"}) == ""
    assert _spec_axis({"description": "drill along x axis"}) == "x"
    assert _spec_axis({"axis": "+y"}) == "y"
    assert _spec_axis({"description": "bohrung in z-richtung"}) == "z"
