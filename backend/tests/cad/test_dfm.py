from pathlib import Path

import pytest
import trimesh
from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Mode,
    Rectangle,
    extrude,
    fillet,
)

from cadx.cad.analysis import analyze
from cadx.cad.dfm.cnc import cnc_checks
from cadx.cad.dfm.common import load_mesh
from cadx.cad.dfm.fdm import fdm_checks
from cadx.cad.exporters import quick_stl

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _mesh(shape, tmp_path):
    stl = quick_stl(shape, tmp_path / "r", name="mesh")
    return load_mesh(stl)


def test_thin_wall_flagged(tmp_path):
    shape = Box(40, 40, 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    mesh = _mesh(shape, tmp_path)
    warnings = fdm_checks(shape, mesh, analyze(shape), min_wall_mm=0.8)
    assert any(w.code == "thin_wall" for w in warnings)


def test_thick_wall_not_flagged(tmp_path):
    shape = Box(40, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
    mesh = _mesh(shape, tmp_path)
    warnings = fdm_checks(shape, mesh, analyze(shape), min_wall_mm=0.8)
    assert not any(w.code == "thin_wall" for w in warnings)


def test_overhang_flagged(tmp_path):
    with BuildPart() as p:
        with BuildSketch():
            Rectangle(40, 40)
        extrude(amount=5)
        with BuildSketch(p.faces().sort_by(Axis.Z)[-1]):
            Rectangle(60, 60)
        extrude(amount=5)
    shape = p.part
    mesh = _mesh(shape, tmp_path)
    warnings = fdm_checks(shape, mesh, analyze(shape))
    assert any(w.code == "overhang" for w in warnings)


def test_overhang_ratio_directions():
    """overhang_ratio: an inverted-T's slab underside is the overhang for a
    +z build; flipped (-z) the same slab is the base and nothing overhangs.
    A plain box resting on the plate has no overhang either way."""
    from cadx.cad.dfm.fdm import overhang_ratio

    post = trimesh.creation.box(extents=[40.0, 40.0, 5.0])
    post.apply_translation([0, 0, 2.5])
    slab = trimesh.creation.box(extents=[60.0, 60.0, 5.0])
    slab.apply_translation([0, 0, 7.5])
    tee = post.union(slab, engine="manifold")
    # underside annulus 60²-40² = 2000 mm² of the ~10 400 mm² surface
    assert overhang_ratio(tee, "+z") == pytest.approx(2000 / tee.area, abs=0.02)
    # upside-down the slab rests on the plate — the overhang disappears
    assert overhang_ratio(tee, "-z") == pytest.approx(0.0, abs=0.01)
    box = trimesh.creation.box(extents=[10.0, 10.0, 5.0])
    box.apply_translation([0, 0, 2.5])
    assert overhang_ratio(box, "+z") == pytest.approx(0.0, abs=1e-6)


def test_sharp_internal_corner_cnc(tmp_path):
    with BuildPart() as p:
        Box(40, 40, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))
        top = p.faces().sort_by(Axis.Z)[-1]
        with BuildSketch(top):
            Rectangle(20, 20)
        extrude(amount=-8, mode=Mode.SUBTRACT)
    shape = p.part
    warnings = cnc_checks(shape, _mesh(shape, tmp_path), analyze(shape))
    assert any(w.code == "sharp_internal_corner" for w in warnings)


def test_filleted_corner_not_flagged_cnc(tmp_path):
    with BuildPart() as p:
        Box(40, 40, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))
        top = p.faces().sort_by(Axis.Z)[-1]
        with BuildSketch(top):
            Rectangle(20, 20)
        extrude(amount=-8, mode=Mode.SUBTRACT)
        fillet(p.edges().filter_by(Axis.Z), radius=3)
    shape = p.part
    warnings = cnc_checks(shape, _mesh(shape, tmp_path), analyze(shape), tool_radius_mm=1.5)
    assert not any(w.code == "sharp_internal_corner" for w in warnings)


def test_deep_pocket_cnc(tmp_path):
    with BuildPart() as p:
        Box(30, 30, 30, align=(Align.CENTER, Align.CENTER, Align.MIN))
        top = p.faces().sort_by(Axis.Z)[-1]
        with BuildSketch(top):
            Rectangle(5, 5)
        extrude(amount=-25, mode=Mode.SUBTRACT)
    shape = p.part
    warnings = cnc_checks(shape, _mesh(shape, tmp_path), analyze(shape))
    assert any(w.code == "deep_pocket" for w in warnings)


def test_undercut_cnc(tmp_path):
    with BuildPart() as p:
        with BuildSketch():
            Rectangle(40, 40)
        extrude(amount=5)
        with BuildSketch(p.faces().sort_by(Axis.Z)[-1]):
            Rectangle(60, 60)
        extrude(amount=5)
    shape = p.part
    warnings = cnc_checks(shape, _mesh(shape, tmp_path), analyze(shape))
    assert any(w.code == "undercut" for w in warnings)


def test_no_undercut_simple(tmp_path):
    shape = Box(40, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
    warnings = cnc_checks(shape, _mesh(shape, tmp_path), analyze(shape))
    assert not any(w.code == "undercut" for w in warnings)
