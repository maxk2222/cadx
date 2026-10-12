"""cadlib — native helper checks + sandbox round-trip (`import cadlib` in
generated code must work under `python -I` + the import guard)."""

import pytest
from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
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
