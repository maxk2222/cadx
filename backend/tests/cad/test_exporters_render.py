from pathlib import Path

import numpy as np
import pytest
import trimesh
from build123d import (
    Align,
    Box,
    BuildPart,
    Hole,
    Locations,
    Pos,
    import_brep,
    import_step,
)

from cadx.cad.exporters import export_2d, export_all_3d, quick_stl
from cadx.cad.render import render_available, render_views
from cadx.cad.sandbox import run_user_code

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def shape(tmp_path):
    out = run_user_code((FIXTURES / "plate_holes.py").read_text(), tmp_path / "w")
    assert out.ok, out.error
    return import_brep(str(out.brep_path))


def test_all_exports(shape, tmp_path):
    out_dir = tmp_path / "exports"
    results = export_all_3d(shape, out_dir)
    for fmt in ("step", "stl", "3mf", "glb"):
        assert results[fmt], fmt
        assert (out_dir / results[fmt]).stat().st_size > 0

    # STEP round-trip preserves volume
    re = import_step(str(out_dir / results["step"]))
    assert abs(re.volume - shape.volume) / shape.volume < 0.001

    # STL is watertight
    m = trimesh.load(str(out_dir / results["stl"]))
    assert m.is_watertight


def test_2d_exports_prismatic(shape, tmp_path):
    results = export_2d(shape, tmp_path / "e", "model", None)
    assert results["dxf"] and results["svg"]


def test_2d_exports_skipped_for_non_flat(shape, tmp_path):
    out = run_user_code((FIXTURES / "simple_box.py").read_text(), tmp_path / "w2")
    s = import_brep(str(out.brep_path))
    results = export_2d(s, tmp_path / "e2", "model", None)
    # filleted box is not a prismatic plate -> no 2d exports expected
    assert results["dxf"] is None or (tmp_path / "e2" / (results["dxf"] or "x")).exists()


@pytest.mark.render
def test_render_views(shape, tmp_path):
    if not render_available():
        pytest.skip("offscreen rendering unavailable")
    stl = quick_stl(shape, tmp_path / "r", name="mesh")
    renders = render_views(stl, tmp_path / "r", [100.0, 60.0, 8.0], size=256)
    assert len(renders) == 7
    for p in renders.values():
        assert p.exists() and p.stat().st_size > 1000


@pytest.mark.render
def test_render_flat_face_has_no_shading_gradient(tmp_path):
    """Regression M2: smooth_shading without split_sharp_edges blended face
    normals across sharp edges — planar faces rendered with visible gradients
    that the critic then "saw" as curved surfaces. A hole-free 64x64 patch on
    the top face measured grayscale std ~13 before the fix; with
    split_sharp_edges=True it is ~0."""
    if not render_available():
        pytest.skip("offscreen rendering unavailable")
    from PIL import Image

    with BuildPart() as p:
        Box(60, 40, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations(Pos(-18, 0, 6)):
            Hole(3.0)
    stl = quick_stl(p.part, tmp_path / "r", name="bore")
    renders = render_views(stl, tmp_path / "r", [60.0, 40.0, 6.0], size=256)
    gray = np.asarray(Image.open(renders["top"]).convert("L"))
    # 64x64 patch on the planar top face, clear of the bore and the outline
    patch = gray[64:128, 96:160]
    assert float(patch.std()) < 4.0
