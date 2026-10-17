"""Tests for cad.inspect_tools — the in-namespace measurement primitives
injected into the model.py exec namespace by runner_entry. Pure geometry,
no LLM."""

import math
from pathlib import Path

import pytest
from build123d import Align, Box, Cylinder, Pos, Solid

from cadx.cad import inspect_tools
from cadx.cad.sandbox import run_user_code

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _through_bore(radius: float) -> Cylinder:
    """Cutter spanning well past the plate's z range (Cylinder's default
    align is CENTER — a bare Pos(0,0,-1)*Cylinder(h=12) stops mid-plate and
    leaves a sealing membrane)."""
    return Pos(0, 0, -1) * Cylinder(
        radius=radius, height=14, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )


@pytest.fixture
def drilled_plate():
    """40x30x10 plate, z in [0, 10], with a Ø6 through-bore at the centre."""
    part = Pos(0, 0, 5) * Box(40, 30, 10)
    part -= _through_bore(3.0)
    return part


@pytest.fixture
def countersunk_plate():
    """Same plate plus a 90deg countersink (head Ø8 -> bore Ø4) at the top."""
    part = Pos(0, 0, 5) * Box(40, 30, 10)
    bore_dia, head_dia, half_angle = 4.0, 8.0, 45.0
    part -= _through_bore(bore_dia / 2)
    depth = (head_dia - bore_dia) / 2 / math.tan(math.radians(half_angle))
    # wide mouth (head dia) at the top surface z=10, tapering to bore dia at
    # z=10-depth — make_cone takes (radius_bottom, radius_top, height)
    part -= Pos(0, 0, 10 - depth) * Solid.make_cone(bore_dia / 2, head_dia / 2, depth)
    return part


def test_measure_box():
    m = inspect_tools.measure(Pos(0, 0, 5) * Box(10, 20, 10))
    assert "error" not in m
    assert m["dims"] == {"x": 10.0, "y": 20.0, "z": 10.0}
    assert m["bbox_min"] == [-5.0, -10.0, 0.0]
    assert m["volume_mm3"] == pytest.approx(2000.0, abs=0.5)
    assert m["solids"] == 1
    assert m["faces"] == 6
    assert m["face_types"] == {"PLANE": 6}
    assert m["is_valid"] is True


def test_measure_bad_input():
    m = inspect_tools.measure(object())
    assert "error" in m


def test_bbox_dims():
    d = inspect_tools.bbox_dims(Pos(1, 2, 3) * Box(7, 8, 9))
    assert d == {"x": 7.0, "y": 8.0, "z": 9.0}


def test_find_holes_through(drilled_plate):
    holes = inspect_tools.find_holes(drilled_plate)
    assert isinstance(holes, list)
    assert len(holes) == 1
    h = holes[0]
    assert h["diameter"] == pytest.approx(6.0, abs=0.05)
    assert h["axis"] in ("+z", "-z")
    assert h["through"] is True
    assert h["depth"] == pytest.approx(10.0, abs=0.1)
    assert h["center"][0] == pytest.approx(0.0, abs=0.1)
    assert h["center"][1] == pytest.approx(0.0, abs=0.1)


def test_find_holes_blind():
    part = Pos(0, 0, 5) * Box(40, 30, 10)
    # 4mm-deep pocket bored down from the top face (z=10 -> z=6)
    part -= Pos(0, 0, 6) * Cylinder(
        radius=2.0, height=4, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    holes = inspect_tools.find_holes(part)
    assert len(holes) == 1
    assert holes[0]["through"] is False
    assert holes[0]["depth"] == pytest.approx(4.0, abs=0.1)


def test_find_holes_none():
    assert inspect_tools.find_holes(Pos(0, 0, 5) * Box(10, 10, 10)) == []


def test_find_holes_stable_ids():
    """M5: hole entries carry deterministic h1… ids so issue texts can cite
    'hole h1 on +x'. Two bores sort by (axis, center)."""
    part = Pos(0, 0, 5) * Box(40, 30, 10)
    part -= Pos(10, 0, -1) * Cylinder(
        radius=2.0, height=14, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    part -= Pos(-10, 0, -1) * Cylinder(
        radius=3.0, height=14, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    holes = inspect_tools.find_holes(part)
    assert [h["id"] for h in holes] == ["h1", "h2"]
    assert holes[0]["center"][0] == pytest.approx(-10.0, abs=0.2)


def test_find_bosses():
    part = Pos(0, 0, 5) * Box(40, 30, 10)
    part += Pos(10, 0, 10) * Cylinder(
        radius=4.0, height=6, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    bosses = inspect_tools.find_bosses(part)
    assert isinstance(bosses, list)
    assert len(bosses) == 1
    b = bosses[0]
    assert b["diameter"] == pytest.approx(8.0, abs=0.05)
    assert b["axis"] in ("+z", "-z")
    assert b["height"] == pytest.approx(6.0, abs=0.1)
    assert b["center"][0] == pytest.approx(10.0, abs=0.5)


def test_find_countersinks(countersunk_plate):
    csk = inspect_tools.find_countersinks(countersunk_plate)
    assert isinstance(csk, list)
    assert len(csk) == 1
    c = csk[0]
    assert c["top_dia"] == pytest.approx(8.0, abs=0.1)
    assert c["angle_deg"] == pytest.approx(45.0, abs=2.0)
    assert c["axis"] in ("+z", "-z")
    assert c["location"][2] == pytest.approx(10.0, abs=0.2)


def test_cross_sections_stepped():
    """Two stacked boxes: 20x20 for z in [0,5], 10x10 for z in [5,10]."""
    part = Pos(0, 0, 2.5) * Box(20, 20, 5)
    part += Pos(0, 0, 7.5) * Box(10, 10, 5)
    secs = inspect_tools.cross_sections(part, axis="z", positions=[2.0, 8.0])
    assert len(secs) == 2
    assert secs[0]["area_mm2"] == pytest.approx(400.0, abs=1.0)
    assert secs[1]["area_mm2"] == pytest.approx(100.0, abs=1.0)


def test_cross_sections_hole_nets_void(drilled_plate):
    """Section through the bore must subtract the Ø6 void."""
    secs = inspect_tools.cross_sections(drilled_plate, axis="z", positions=[5.0])
    assert secs[0]["area_mm2"] == pytest.approx(1200.0 - math.pi * 9, abs=2.0)


def test_cross_sections_even_spacing():
    part = Pos(0, 0, 5) * Box(10, 10, 10)
    secs = inspect_tools.cross_sections(part, axis="x", count=4)
    assert len(secs) == 4
    assert all(s["area_mm2"] == pytest.approx(100.0, abs=1.0) for s in secs)


def test_cross_sections_bad_axis():
    out = inspect_tools.cross_sections(Pos(0, 0, 5) * Box(10, 10, 10), axis="w")
    assert "error" in out


def test_clearance_apart():
    a = Pos(0, 0, 20) * Box(4, 4, 4)
    b = Pos(0, 0, 5) * Box(40, 30, 10)
    c = inspect_tools.clearance(a, b)
    assert c["distance_mm"] == pytest.approx(8.0, abs=0.01)
    assert c["status"] == "apart"


def test_clearance_interpenetrating():
    a = Pos(0, 0, 5) * Box(4, 4, 10)
    b = Pos(0, 0, 2.5) * Box(20, 20, 5)
    c = inspect_tools.clearance(a, b)
    assert c["distance_mm"] == pytest.approx(0.0, abs=1e-6)
    assert c["status"] == "interpenetrating"
    assert c["overlap_mm3"] > 1.0


def test_face_cap_skip(monkeypatch, drilled_plate):
    monkeypatch.setattr(inspect_tools, "_FACE_CAP", 3)
    out = inspect_tools.find_holes(drilled_plate)
    assert out["skipped"] == "faceted solid"
    assert out["faces"] == len(drilled_plate.faces())
    # measure/bbox stay cheap and unguarded
    assert "error" not in inspect_tools.measure(drilled_plate)
    assert "error" not in inspect_tools.bbox_dims(drilled_plate)


def test_in_namespace_via_sandbox(tmp_path):
    """End-to-end: generated code inside the real sandbox can call the
    injected helpers — the import guard must not block them."""
    code = """\
from build123d import *
PARAMS = {}
part = Pos(0, 0, 5) * Box(40, 30, 10)
part -= Pos(0, 0, -1) * Cylinder(radius=3.0, height=14, align=(Align.CENTER, Align.CENTER, Align.MIN))
result = part
m = measure(part)
print("DIMS", m["dims"])
print("VALID", m["is_valid"])
hs = find_holes(part)
print("HOLES", len(hs), hs[0]["diameter"], hs[0]["through"])
print("CSK", find_countersinks(part))
print("SEC", cross_sections(part, positions=[5.0]))
print("BBOX", bbox_dims(part))
"""
    out = run_user_code(code, tmp_path / "work")
    assert out.ok, (out.error, out.traceback)
    assert "DIMS {'x': 40.0, 'y': 30.0, 'z': 10.0}" in out.stdout
    assert "VALID True" in out.stdout
    assert "HOLES 1 6.0 True" in out.stdout
    assert "CSK []" in out.stdout
    assert "SEC [{'position': 5.0, 'area_mm2': 1171." in out.stdout
    assert "BBOX {'x': 40.0, 'y': 30.0, 'z': 10.0}" in out.stdout


def test_lazy_import_blocked_but_helpers_work(tmp_path):
    """Generated code cannot import inspect_tools itself (guard) but the
    injected names still resolve."""
    code = """\
from build123d import *
PARAMS = {}
import inspect_tools  # noqa: F401 — must be blocked (static allow-list / guard)
result = Box(1, 1, 1)
"""
    out = run_user_code(code, tmp_path / "work")
    assert not out.ok
    assert "not allowed" in (out.error or "") + (out.traceback or "")
