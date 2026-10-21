"""cadlib — native helper checks + sandbox round-trip (`import cadlib` in
generated code must work under `python -I` + the import guard)."""

import json
import math
from pathlib import Path

import pytest
from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    Compound,
    Cylinder,
    Pos,
    export_stl,
)

from cadx.cad import cadlib
from cadx.cad.sandbox import run_user_code

_ALIGN_MIN = (Align.CENTER, Align.CENTER, Align.MIN)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "specs"


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


def test_safe_fillet_shape_keeps_algebra_result():
    """Algebra mode must retain the new Shape; a nonzero radius alone is not
    evidence that the immutable input was changed."""
    from cadx.cad.features import extract_features

    foot = Box(60, 40, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
    leg = Pos(-28, 0, 25) * Box(4, 40, 50)
    sharp = foot + leg
    edges = [
        e
        for e in sharp.edges().filter_by(Axis.Y)
        if abs(e.center().X - (-26.0)) < 0.1 and abs(e.center().Z - 4.0) < 0.1
    ]

    rounded, applied = cadlib.safe_fillet_shape(sharp, edges, radius=3.0)
    features = extract_features(rounded)

    assert applied == pytest.approx(3.0)
    assert rounded.volume > sharp.volume
    assert features.rounds and features.rounds[0] == pytest.approx(3.0, abs=0.01)


def test_safe_fillet_outside_builder_fails_loudly():
    """The builder-only helper cannot silently return a radius while
    discarding build123d's functional Shape result."""
    shape = Box(30, 20, 8, align=_ALIGN_MIN)
    edges = shape.edges().filter_by(Axis.Z)
    with pytest.raises(RuntimeError, match="safe_fillet_shape"):
        cadlib.safe_fillet(edges, radius=2.0)


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
    """A cutter placed outside the part must fail loudly — the old silent
    no-op let misplaced bores pass every check (observed: cutters at x=75/85
    on a part spanning x ∈ [-42.6, +39.4], holes never cut)."""
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    disjoint = Pos(500, 0, 0) * Box(2, 2, 2)
    with pytest.raises(ValueError, match="does not overlap"):
        cadlib.mesh_cut(body, disjoint)


def test_mesh_cut_raises_on_noop_result():
    """OCC 'succeeding' but returning the un-cut body is still a no-op —
    the removed-volume guard must catch what the floor check cannot."""

    class _SelfCut:
        """Body whose boolean returns the un-cut body — the silent no-op
        this guard exists to catch."""

        def __init__(self):
            self._shape = Box(10, 10, 10)

        def __sub__(self, other):
            return self._shape

        def bounding_box(self):
            return self._shape.bounding_box()

        @property
        def volume(self):
            return self._shape.volume

    with pytest.raises(ValueError, match="no material"):
        cadlib.mesh_cut(_SelfCut(), Pos(0, 0, 0) * Box(2, 2, 2))


def test_mesh_cut_small_real_cut_passes():
    """The no-op guard must not eat legitimate small features — a shallow
    pocket removes far more than the delta epsilon."""
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    pocket = Pos(0, 0, 4) * Box(2, 2, 2)  # removes 2x2x1 = 4 mm^3
    out = cadlib.mesh_cut(body, pocket)
    assert out.volume == pytest.approx(1000.0 - 4.0, rel=0.01)


def test_mesh_fuse_raises_on_disjoint_tool():
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    disjoint = Pos(500, 0, 0) * Box(2, 2, 2)
    with pytest.raises(ValueError, match="does not overlap"):
        cadlib.mesh_fuse(body, disjoint)


def test_mesh_fuse_raises_on_engulfed_tool():
    """A tool fully inside the body adds nothing — collapse_solids used to
    quietly return the un-fused body."""
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    engulfed = Pos(0, 0, 0) * Box(2, 2, 2)
    with pytest.raises(ValueError, match="no material"):
        cadlib.mesh_fuse(body, engulfed)


def test_mesh_fuse_adds_material():
    body = Pos(0, 0, 0) * Box(10, 10, 10)
    boss = Pos(6, 0, 0) * Box(6, 6, 6)  # overlaps the +X face by 2 mm
    out = cadlib.mesh_fuse(body, boss)
    assert len(out.solids()) == 1
    assert out.volume == pytest.approx(1000.0 + 6 * 6 * 4, rel=0.02)


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


def test_slot_cutter_algebra_style():
    """Algebra-mode bodies subtract the returned cutter — the observed
    l_bracket model wrapped the slot in a throwaway BuildPart instead."""
    body = Box(4, 40, 50, align=_ALIGN_MIN)
    body = body - Pos(0, 0, 25) * cadlib.slot_cutter(
        6.5, 15.0, direction="+z", normal="+x", part=body
    )
    for z in (-4, 0, 4):
        for x in (0.5, 2.0, 3.5):
            assert not body.is_inside((float(x), 0.0, 25.0 + z)), (x, z)
    assert body.is_inside((2.0, 0.0, 45.0))
    assert body.is_inside((2.0, 0.0, 5.0))


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


# --------------------------------------------------------------------------- #
# strut — straight diagonal bar between two points
# --------------------------------------------------------------------------- #


def test_strut_geometry():
    """Bar from leg (x=-55, z=35) to arm underside (x=+15, z=95): the width
    axis stays parallel to world Y, ends embed 1 mm each for the fuse."""
    s = cadlib.strut((-55, 0, 35), (15, 0, 95), 10, 6)
    bb = s.bounding_box()
    assert pytest.approx(-5.0, abs=0.01) == bb.min.Y
    assert pytest.approx(5.0, abs=0.01) == bb.max.Y
    assert s.volume == pytest.approx((math.hypot(70, 60) + 2) * 10 * 6, rel=0.01)


def test_strut_fuses_into_bracket():
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    body = leg + arm + cadlib.strut((-55, 0, 35), (15, 0, 95), 10, 6)
    assert len(body.solids()) == 1


def test_strut_parallel_to_y_no_crash():
    """The bar direction == world Y degenerates the width axis — the
    fallback must still produce a solid bar."""
    s = cadlib.strut((0, -10, 5), (0, 10, 5), 4, 4)
    assert s.volume == pytest.approx((20 + 2) * 4 * 4, rel=0.01)


def test_bound_snippets_execute_bracket(tmp_path):
    """End-to-end: the bound snippets of the bracket intake fixture must
    EXECUTE against the real cadlib — strut as a bar, and host-bounded
    bores that pierce their own plate without touching the strut on the
    same axis line (the E2E cross-channel defect)."""
    from cadx.agent.featurebind import bind_features
    from cadx.cad.features import extract_features
    from cadx.llm.schemas import PartSpec

    spec = PartSpec.model_validate(
        json.loads((_FIXTURES / "bracket_strut_cad.json").read_text(encoding="utf-8"))
    )
    by_name = {b.name: b for b in bind_features(spec)}
    ns: dict = {}
    header = "from build123d import *\nfrom cadx.cad.cadlib import *\n"
    body = (
        "part = Pos(-57.5, 0.0, 50.0) * Box(5.0, 30.0, 100.0)\n"
        "part = part + Pos(0.0, 0.0, 97.5) * Box(120.0, 30.0, 5.0)\n"
    )
    exec(header + body, ns)
    exec(by_name["diagonal_brace"].snippet, ns)
    for name in ("arm_holes", "leg_holes"):
        exec(by_name[name].snippet, ns)
    part = ns["part"]
    assert len(part.solids()) == 1
    assert 33_000 <= part.volume <= 42_000
    # the bore lines continue through air/strut — the host-bound cut must
    # NOT have drilled the strut under the arm hole or beside the leg hole
    assert part.is_inside((-25.0, 0.0, 60.0))  # strut on the arm-bore line
    assert part.is_inside((-14.0, 0.0, 70.0))  # strut on the leg-bore line
    stl = tmp_path / "bracket.stl"
    export_stl(part, str(stl))
    m = extract_features(part, stl)
    by_axis: dict[str, float] = {}
    for g in m.hole_groups:
        if abs(g.diameter - 5.5) <= 0.3:
            by_axis[g.axis.lstrip("+-")] = by_axis.get(g.axis.lstrip("+-"), 0) + g.count
    assert by_axis.get("x", 0) >= 2 and by_axis.get("z", 0) >= 2, m.hole_groups


# --------------------------------------------------------------------------- #
# clip_to_box — clip a body to an axis-aligned envelope
# --------------------------------------------------------------------------- #


def test_clip_to_box_clips_oversized_x():
    """Half-extent smaller than the body cuts the overshoot — the other two
    axes must pass through untouched (x ±60 -> ±50; y ±20, z 10..30)."""
    body = Pos(0, 0, 10) * Box(120, 40, 20, align=_ALIGN_MIN)
    out = cadlib.clip_to_box(body, x=50)
    bb = out.bounding_box()
    assert pytest.approx(100.0, abs=0.5) == bb.size.X
    assert pytest.approx(40.0, abs=0.5) == bb.size.Y
    assert pytest.approx(20.0, abs=0.5) == bb.size.Z
    assert len(out.solids()) >= 1


def test_clip_to_box_inside_body_passes_through():
    """Every bound contains the body -> no cutter side engages: the body
    comes back unchanged and no empty-boolean exception is raised."""
    body = Pos(0, 0, 0) * Box(120, 40, 20, align=_ALIGN_MIN)  # z 0..20 ⊂ [0, 25]
    out = cadlib.clip_to_box(body, x=80, y=30, z_min=0, z_max=25)
    assert out.volume == pytest.approx(body.volume, rel=0.01)
    bb_out, bb_in = out.bounding_box(), body.bounding_box()
    for attr in ("min", "max", "size"):
        a, b = getattr(bb_out, attr), getattr(bb_in, attr)
        assert pytest.approx((b.X, b.Y, b.Z)) == (a.X, a.Y, a.Z)


def test_clip_to_box_z_min_lifts_underside():
    """A body sunk below the build plane gets its underside trimmed flat —
    the top of the body is untouched."""
    body = Pos(0, 0, -5) * Box(40, 40, 20, align=_ALIGN_MIN)  # z -5..15
    out = cadlib.clip_to_box(body, z_min=0)
    bb = out.bounding_box()
    assert bb.min.Z >= -0.05
    assert pytest.approx(15.0, abs=0.5) == bb.max.Z


def test_clip_to_box_combined_axes():
    """x + y + z_max bounds clip their sides together in one call."""
    body = Pos(0, 0, 0) * Box(120, 40, 20, align=_ALIGN_MIN)
    out = cadlib.clip_to_box(body, x=50, y=15, z_max=15)
    bb = out.bounding_box()
    assert pytest.approx(100.0, abs=0.5) == bb.size.X
    assert pytest.approx(30.0, abs=0.5) == bb.size.Y
    assert pytest.approx(15.0, abs=0.5) == bb.max.Z
    assert pytest.approx(0.0, abs=0.5) == bb.min.Z


def test_clip_to_box_returns_single_solid():
    """The clipped result collapses to one solid — downstream code does
    result.solids() bookkeeping on it."""
    body = Pos(0, 0, 10) * Box(120, 40, 20, align=_ALIGN_MIN)
    out = cadlib.clip_to_box(body, x=50)
    assert len(out.solids()) == 1


# --------------------------------------------------------------------------- #
# arch_bridge — arched bar lofted along a three-point arc in XZ
# --------------------------------------------------------------------------- #


def test_arch_bridge_single_solid():
    a = cadlib.arch_bridge(60, 25, 12, 8)
    assert len(a.solids()) == 1


def test_arch_bridge_top_reaches_apex():
    """The loft passes through the apex CENTRE (apex_z - thickness/2), so the
    outer surface tops out at the declared apex height."""
    a = cadlib.arch_bridge(60, 25, 12, 8)
    assert pytest.approx(25.0, abs=0.5) == a.bounding_box().max.Z


def test_arch_bridge_width_along_y():
    a = cadlib.arch_bridge(60, 25, 12, 8)
    assert pytest.approx(12.0, abs=0.5) == a.bounding_box().size.Y


def test_arch_bridge_span_plus_foot_margin():
    """Feet sit at ±span_x/2; the radial section adds at most one thickness
    of margin beyond the span on each side."""
    a = cadlib.arch_bridge(60, 25, 12, 8)
    sx = a.bounding_box().size.X
    assert 60.0 <= sx <= 60.0 + 2 * 8


def test_arch_bridge_feet_embed_below_foot_z():
    """embed sinks the feet under foot_z — a fuse into the base body needs
    material below the contact plane to bite into."""
    a = cadlib.arch_bridge(60, 25, 12, 8)  # foot_z=0.0 default
    assert a.bounding_box().min.Z <= 0.0 + 0.5
