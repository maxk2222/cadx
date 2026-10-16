"""cadlib — native helper checks + sandbox round-trip (`import cadlib` in
generated code must work under `python -I` + the import guard)."""

import pytest
from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    Compound,
    Cylinder,
    Pos,
)

from cadx.cad import cadlib
from cadx.cad.sandbox import run_user_code

_ALIGN_MIN = (Align.CENTER, Align.CENTER, Align.MIN)


def test_boss_dims_and_bore():
    b = cadlib.boss(diameter=12, height=6, hole_d=4.3)
    bb = b.bounding_box()
    assert pytest.approx(12) == bb.size.X
    assert pytest.approx(6) == bb.size.Z
    assert pytest.approx(0) == bb.min.Z


def test_rib_rests_on_xy():
    r = cadlib.rib(length=20, height=10, thickness=3)
    bb = r.bounding_box()
    assert pytest.approx((0, 0)) == (bb.min.X, bb.min.Z)
    assert pytest.approx((20, 3, 10)) == (bb.size.X, bb.size.Y, bb.size.Z)


def test_mounting_ear_and_tip():
    e = cadlib.mounting_ear(20, 10, 3, hole_d=4.3, tip_radius=5)
    assert pytest.approx(25) == e.bounding_box().size.X
    assert len(e.solids()) == 1


def test_snap_fit_hook_height():
    s = cadlib.snap_fit_cantilever(15, 4, 1.2, hook_height=2.0)
    bb = s.bounding_box()
    assert pytest.approx(0) == bb.min.Z
    assert pytest.approx(1.2 + 2.0) == bb.max.Z


def test_knurl_ring_has_ridges():
    k = cadlib.knurl_ring(diameter=20, height=8, pitch=2.0)
    assert len(k.solids()) >= 8


def test_filleted_box_and_l_bracket():
    f = cadlib.filleted_box(20, 15, 8, 2)
    assert f.volume < 20 * 15 * 8
    lb = cadlib.l_bracket(30, 30, 15, 4)
    bb = lb.bounding_box()
    assert pytest.approx((30, 30)) == (bb.size.X, bb.size.Z)


def test_text_label():
    t = cadlib.text_label("A1", size=8, depth=0.4)
    assert pytest.approx(0.4) == t.bounding_box().size.Z


def test_dovetail_slot_cutter():
    d = cadlib.dovetail_slot(6, 10, 4, 60)
    bb = d.bounding_box()
    # runs along ±X, wider at bottom, top edge at Z=0
    assert pytest.approx((60, 10)) == (bb.size.X, bb.size.Y)
    assert pytest.approx((-4, 0)) == (bb.min.Z, bb.max.Z)


def test_hole_grid_cuts_countersinks():
    from bd_warehouse.fastener import CounterSunkScrew

    screw = CounterSunkScrew(size="M4-0.7", fastener_type="iso10642", length=12)
    with BuildPart() as p:
        Box(40, 30, 8, align=_ALIGN_MIN)
        top = p.faces().sort_by(Axis.Z)[-1]
        cadlib.hole_grid(screw, nx=2, ny=2, pitch_x=24, pitch_y=16, kind="countersink", on=top)
    assert p.part.volume < 40 * 30 * 8 - 4 * 20  # 4 countersunk bores removed


def test_hole_grid_requires_buildpart():
    from bd_warehouse.fastener import SocketHeadCapScrew

    screw = SocketHeadCapScrew(size="M4-0.7", fastener_type="iso4762", length=12)
    with pytest.raises(RuntimeError, match="BuildPart"):
        cadlib.hole_grid(screw)


def test_insert_pocket_and_bearing_seat():
    from bd_warehouse.bearing import SingleRowDeepGrooveBallBearing
    from bd_warehouse.fastener import HeatSetNut

    bearing = SingleRowDeepGrooveBallBearing(size="M8-22-7")
    insert = HeatSetNut(size="M4-0.7-Standard", fastener_type="McMaster-Carr")
    with BuildPart() as p:
        Box(60, 40, 12, align=_ALIGN_MIN)
        cadlib.bearing_seat(bearing, retain="snapring", on=Pos(-15, 0, 12))
        cadlib.insert_pocket(insert, on=Pos(15, 0, 12))
    assert p.part.volume < 60 * 40 * 12


def test_keyhole_and_t_slot():
    with BuildPart() as p:
        Box(60, 40, 12, align=_ALIGN_MIN)
        cadlib.keyhole_slot(10, 5, 15, 4, on=Pos(0, -15, 12))
        cadlib.t_slot(6, 10, 3, 6, 60, on=Pos(0, 10, 12))
    assert p.part.volume < 60 * 40 * 12


def test_keyhole_slot_actually_cuts():
    """Regression: primitives without Mode.SUBTRACT were ADDED, not cut."""
    with BuildPart() as p:
        Box(60, 40, 12, align=_ALIGN_MIN)
        cadlib.keyhole_slot(10, 5, 15, 4, on=Pos(0, -15, 12))
    assert not p.part.is_inside((0.0, -15.0, 11.0))


def test_through_hole_cuts_fully():
    """Oversized cutter: the bore clears the part even when the location sits
    mid-height — no sealing membrane."""
    with BuildPart() as p:
        Box(60, 40, 6, align=_ALIGN_MIN)
        cadlib.through_hole(5.0, on=Pos(-15, 0, 2))
    for z in (0.5, 3.0, 5.5):
        assert not p.part.is_inside((-15.0, 0.0, z))


def test_through_hole_countersink():
    from build123d import GeomType

    with BuildPart() as p:
        Box(60, 40, 6, align=_ALIGN_MIN)
        top = p.faces().sort_by(Axis.Z)[-1]
        cadlib.through_hole(5.0, csk_dia=9.0, on=top)
    assert any(f.geom_type == GeomType.CONE for f in p.part.faces())
    assert not p.part.is_inside((0.0, 0.0, 3.0))


def test_through_hole_blind_depth():
    with BuildPart() as p:
        Box(60, 40, 10, align=_ALIGN_MIN)
        top = p.faces().sort_by(Axis.Z)[-1]
        cadlib.through_hole(6.0, depth=4.0, on=top)
    assert not p.part.is_inside((0.0, 0.0, 9.0))
    assert p.part.is_inside((0.0, 0.0, 5.0))


def test_safe_fillet_empty_selection_returns_zero():
    """An empty/stale edge selection must not crash — safe_fillet retries
    down to min_radius then reports 0.0 so the part stays valid."""
    with BuildPart() as p:
        Box(30, 20, 8, align=_ALIGN_MIN)
        empty = p.edges().filter_by(lambda e: False)
        applied = cadlib.safe_fillet(empty, radius=3.0)
    assert applied == 0.0
    assert p.part is not None


def test_safe_fillet_shrinks_oversized_radius():
    """Radius that does not fit gets halved until OCC accepts it."""
    with BuildPart() as p:
        Box(30, 20, 8, align=_ALIGN_MIN)
        verts = p.edges().filter_by(Axis.Z)
        applied = cadlib.safe_fillet(verts, radius=14.0)
    assert 0.0 < applied < 14.0


def test_through_hole_requires_buildpart():
    with pytest.raises(RuntimeError, match="BuildPart"):
        cadlib.through_hole(5.0)


def test_sandbox_import_cadlib(tmp_path):
    code = (
        'PARAMS = {"plate_l": 60.0}\n'
        "import cadlib\n"
        "from build123d import *\n"
        "from bd_warehouse.fastener import CounterSunkScrew\n"
        'screw = CounterSunkScrew(size="M4-0.7", fastener_type="iso10642", length=12)\n'
        "with BuildPart() as p:\n"
        '    Box(PARAMS["plate_l"], 30, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))\n'
        "    top = p.faces().sort_by(Axis.Z)[-1]\n"
        '    cadlib.hole_grid(screw, nx=2, ny=1, pitch_x=30, kind="countersink", on=top)\n'
        "    p.part += Pos(-20, 0, 8) * cadlib.boss(10, 5, hole_d=3)\n"
        "result = p.part\n"
    )
    out = run_user_code(code, tmp_path / "work")
    assert out.ok, out.error
    assert out.params_used["plate_l"] == 60.0


def test_sandbox_cadlib_cannot_import_disallowed(tmp_path):
    """`import cadlib` works, but other modules on the same path stay blocked."""
    code = "import cadlib\nimport engine\nPARAMS = {}\nresult = None\n"
    out = run_user_code(code, tmp_path / "work")
    assert not out.ok
    assert "static" in (out.error or "")


def test_closed_polyline_tolerates_duplicates():
    pts = [(0, 0), (10, 0), (10, 0), (10, 5), (5, 8), (0, 5), (0, 5), (0, 0)]
    face = cadlib.closed_polyline(pts)
    assert face.area > 40


def test_closed_polyline_strips_repeated_closing_point():
    pts = [(0, 0), (10, 0), (10, 5), (0, 5), (0, 0), (0, 0)]
    face = cadlib.closed_polyline(pts)
    assert face.area > 40


def test_closed_polyline_rejects_degenerate():
    with pytest.raises(ValueError):
        cadlib.closed_polyline([(0, 0), (1, 0), (1, 0)])


def test_sandbox_closed_polyline(tmp_path):
    code = (
        "PARAMS = {}\n"
        "import cadlib\n"
        "from build123d import *\n"
        "import math\n"
        "pts = [(30*math.cos(t), 10*math.sin(t)) for t in [i*0.1 for i in range(63)]]\n"
        "pts += pts[:2]  # duplicate samples + repeated closing point\n"
        "face = cadlib.closed_polyline(pts)\n"
        "result = extrude(face, amount=8)\n"
    )
    out = run_user_code(code, tmp_path / "work")
    assert out.ok, out.error


def test_trim_below_flattens_underside():
    body = Pos(0, 0, -0.5) * Box(10, 10, 5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    out = cadlib.trim_below(body, z=0.0)
    assert out.bounding_box().min.Z >= -1e-6


def test_mesh_intersect_raises_on_empty():
    """Faceted-mesh booleans can return None (no exception) — the helper must
    raise a real error so the failure lands at the boolean line."""

    class _NoneBool:
        def __and__(self, other):
            return None

    with pytest.raises(ValueError, match="intersection"):
        cadlib.mesh_intersect(_NoneBool(), object())


def test_mesh_intersect_returns_solid():
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    tool = Pos(0, 0, 5) * Box(20, 20, 10)
    out = cadlib.mesh_intersect(body, tool)
    assert out.volume > 0


def test_runner_result_none_gets_boolean_hint(tmp_path):
    """`result = None` (e.g. a mesh `&` that failed) must not read as
    'no result assigned' — the error names the boolean-None cause."""
    code = (
        "from build123d import *\n"
        "PARAMS = {}\n"
        "body = Box(10, 10, 10)\n"
        "r = body & Pos(100, 0, 0) * Box(5, 5, 5)  # mesh booleans can return None\n"
        "result = None  # simulates `result = body` after body became None\n"
    )
    out = run_user_code(code, tmp_path / "work")
    assert not out.ok
    assert "None" in (out.error or "")
    assert "no `result` assigned" not in (out.error or "")
    assert "intersect" in (out.error or "").lower() or "&" in (out.error or "")


def test_runner_missing_result_keeps_old_error(tmp_path):
    code = "from build123d import *\nPARAMS = {}\nbody = Box(10, 10, 10)\n"
    out = run_user_code(code, tmp_path / "work")
    assert not out.ok
    assert "result" in (out.error or "")


def test_collapse_solids_keeps_dominant():
    """A fragmented boolean result (big part + slivers) collapses to the
    dominant solid — the slivers are debris, not features."""
    big = Pos(0, 0, 0) * Box(10, 10, 10)
    sliver = Pos(50, 0, 0) * Box(0.05, 0.05, 0.05)
    frag = Compound(children=[big, sliver])
    out = cadlib.collapse_solids(frag)
    assert out.volume == pytest.approx(1000.0, rel=0.01)


def test_collapse_solids_single_passthrough():
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    assert cadlib.collapse_solids(body).volume == pytest.approx(1000.0, rel=0.01)


def test_largest_solid_picks_biggest():
    big = Pos(0, 0, 0) * Box(10, 10, 10)
    small = Pos(50, 0, 0) * Box(2, 2, 2)
    out = cadlib.largest_solid(Compound(children=[small, big]))
    assert out.volume == pytest.approx(1000.0, rel=0.01)


def test_mesh_cut_raises_on_disjoint_tool():
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    disjoint = Pos(500, 0, 0) * Box(2, 2, 2)
    r = cadlib.mesh_cut(body, disjoint)
    # a disjoint cut returns the body unchanged — still a valid result
    assert r.volume == pytest.approx(1000.0, rel=0.01)


def test_mesh_cut_drops_slivers():
    """When the boolean sheds slivers, mesh_cut returns the dominant solid."""
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    tool = Pos(0, 0, -1) * Cylinder(radius=2, height=12)
    out = cadlib.mesh_cut(body, tool)
    assert len(out.solids()) == 1
    assert out.volume == pytest.approx(1000.0 - 3.14159 * 4 * 12, rel=0.05)


def test_bbox_dims():
    """bbox_dims removes the BoundBox API surface generated code kept
    hallucinating (xlen/xmax/size.x all invalid in build123d)."""
    box = Pos(1, 2, 3) * Box(10, 20, 30)
    assert cadlib.bbox_dims(box) == pytest.approx([10.0, 20.0, 30.0])
    cyl = Pos(0, 0, 5) * Cylinder(radius=4, height=12)
    assert cadlib.bbox_dims(cyl) == pytest.approx([8.0, 8.0, 12.0], rel=1e-3)


def test_oblong_slot_through_and_shape():
    """Stadium slot: clean through-cut, elongated void along the given axis —
    the hand-rolled slot builders kept producing membranes or missing the
    feature entirely (observed on l_bracket and firesteel)."""
    with BuildPart() as p:
        Box(60, 40, 6, align=_ALIGN_MIN)
        cadlib.oblong_slot(6.0, 20.0, on=Pos(0, 0, 6))
    # open at top and bottom along the whole slot length
    for x in (-7, 0, 7):
        for z in (0.5, 3.0, 5.5):
            assert not p.part.is_inside((float(x), 0.0, z)), (x, z)
    # but material remains outside the slot width
    assert p.part.is_inside((0.0, 10.0, 3.0))


def test_oblong_slot_along_y():
    with BuildPart() as p:
        Box(60, 40, 6, align=_ALIGN_MIN)
        cadlib.oblong_slot(6.0, 20.0, direction="+y", on=Pos(0, 0, 6))
    for y in (-7, 0, 7):
        assert not p.part.is_inside((0.0, float(y), 3.0)), y


def test_oblong_slot_vertical_through_wall():
    """Vertical slot along Z through a wall thin in X — the l_bracket leg
    slot the model kept hand-rolling wrong."""
    with BuildPart() as p:
        Box(4, 40, 50, align=_ALIGN_MIN)  # wall 4 mm thin in X
        cadlib.oblong_slot(6.5, 15.0, direction="+z", normal="+x", on=Pos(0, 0, 25))
    for z in (-4, 0, 4):
        for x in (0.5, 2.0, 3.5):
            assert not p.part.is_inside((float(x), 0.0, 25.0 + z)), (x, z)
    # material remains above/below the slot along Z
    assert p.part.is_inside((2.0, 0.0, 45.0))
    assert p.part.is_inside((2.0, 0.0, 5.0))


def test_oblong_slot_rejects_parallel_axes():
    with BuildPart():
        Box(60, 40, 6, align=_ALIGN_MIN)
        with pytest.raises(ValueError, match="perpendicular"):
            cadlib.oblong_slot(6.0, 20.0, direction="+x", normal="-x")


def test_tube_open_bore_both_ends():
    t = cadlib.tube(12.0, 8.0, 30.0)
    # bore through: centre point is empty at both end faces
    assert not t.is_inside((0.0, 0.0, 0.2))
    assert not t.is_inside((0.0, 0.0, 29.8))
    # wall material present between inner and outer radius
    assert t.is_inside((5.0, 0.0, 15.0))
