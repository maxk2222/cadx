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

import cadx.cad.exporters as exporters_mod
from cadx.cad.exporters import export_2d, export_all_3d, quick_stl
from cadx.cad.render import _load_for_render, _thread_detail_camera, render_available, render_views
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
    assert set(renders) == {
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
        "iso",
        "iso_ghost",
        "front_ghost",
    }
    for p in renders.values():
        assert p.exists() and p.stat().st_size > 1000


@pytest.mark.render
def test_render_views_ghost_disabled(shape, tmp_path, monkeypatch):
    monkeypatch.setenv("CADX_RENDER_GHOST_OPACITY", "0")
    if not render_available():
        pytest.skip("offscreen rendering unavailable")
    stl = quick_stl(shape, tmp_path / "r", name="mesh")
    renders = render_views(stl, tmp_path / "r", [100.0, 60.0, 8.0], size=256)
    assert len(renders) == 7
    assert not any(k.endswith("_ghost") for k in renders)


@pytest.mark.render
def test_render_views_ghost_param(shape, tmp_path):
    """ghosts=False drops xray shots regardless of the env opacity — the
    attempt/preview path uses it to halve per-run render cost."""
    if not render_available():
        pytest.skip("offscreen rendering unavailable")
    stl = quick_stl(shape, tmp_path / "r", name="mesh")
    renders = render_views(stl, tmp_path / "r", [100.0, 60.0, 8.0], size=256, ghosts=False)
    assert len(renders) == 7
    assert not any(k.endswith("_ghost") for k in renders)


def test_render_quality_modes():
    """Attempt renders: 512px no ghosts; export re-render: full res + xray."""
    from cadx.cad.engine import _render_quality

    assert _render_quality("preview") == (512, False)
    assert _render_quality("export") == (1024, True)
    assert _render_quality("anything-else") == (512, False)


@pytest.mark.render
def test_render_ghost_view_shows_interior(shape, tmp_path):
    """The xray tile must differ from the opaque render — translucent faces
    expose interior shading the opaque iso hides."""
    if not render_available():
        pytest.skip("offscreen rendering unavailable")
    stl = quick_stl(shape, tmp_path / "r", name="mesh")
    renders = render_views(stl, tmp_path / "r", [100.0, 60.0, 8.0], size=256)
    from PIL import Image

    iso = np.asarray(Image.open(renders["iso"]).convert("L"), dtype=np.int16)
    ghost = np.asarray(Image.open(renders["iso_ghost"]).convert("L"), dtype=np.int16)
    assert float(np.abs(iso - ghost).mean()) > 0.5


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


def test_load_for_render_welds_facet_soup(tmp_path):
    """OCC STLs arrive as per-facet triangle soups (3 verts per triangle) —
    VTK's STL reader welds on read but the render loader must keep the
    weld invariant explicit: a soup PolyData breaks smooth shading and
    marks every triangle boundary as a feature edge (the jagged-thread
    regression)."""
    with BuildPart() as p:
        Box(20, 10, 5)
    stl = quick_stl(p.part, tmp_path / "r", name="b")
    welded = _load_for_render(stl)
    # a tessellated box welds to its 8 corners; any soup would leave 36
    assert welded.n_points == 8


def test_part_surface_welds_trimesh_soup():
    """FEM overlays build PolyData straight from trimesh arrays — trimesh
    keeps per-facet duplicated vertices, so the weld must happen there."""
    from cadx.cad.fem.analyze import _part_surface

    base = trimesh.creation.box()
    soup = trimesh.Trimesh(  # OCC-style: every facet carries its own vertices
        vertices=base.vertices[base.faces].reshape(-1, 3),
        faces=np.arange(3 * len(base.faces)).reshape(-1, 3),
        process=False,
    )
    assert len(soup.vertices) == 3 * len(soup.faces)  # raw soup precondition
    poly = _part_surface(soup, None)
    assert poly.n_points == 8


def test_thread_detail_camera_clips_internal_threads():
    """A side-on camera on an internal thread only sees the flat block face
    — the detail shot must clip the camera-side half to expose the bore."""
    internal = _thread_detail_camera(
        [
            {
                "axis": "+z",
                "position_mm": [0, 0, 0],
                "major_diameter": 8,
                "length": 20,
                "external": False,
            }
        ]
    )
    assert internal is not None
    assert internal[4] is not None

    external = _thread_detail_camera(
        [
            {
                "axis": "+z",
                "position_mm": [0, 0, 0],
                "major_diameter": 8,
                "length": 20,
                "external": True,
            }
        ]
    )
    assert external is not None
    assert external[4] is None


def test_quick_stl_adapts_tolerance_to_part_size(tmp_path, monkeypatch):
    """A fixed 0.05mm chord tolerance under-resolves mm-scale features on
    small parts — tolerance scales with the smallest bbox dim."""
    captured: dict = {}

    def _fake(shape, path, **kw):
        captured.update(kw)
        Path(path).write_text("solid s\nendsolid s\n")

    monkeypatch.setattr(exporters_mod, "export_stl", _fake)

    with BuildPart() as p:
        Box(5, 5, 5)
    quick_stl(p.part, tmp_path)
    assert captured["tolerance"] == pytest.approx(0.005)
    assert captured["angular_tolerance"] == pytest.approx(0.05)

    captured.clear()
    with BuildPart() as p2:
        Box(120, 80, 40)
    quick_stl(p2.part, tmp_path)
    assert captured["tolerance"] == pytest.approx(0.04)
    assert captured["angular_tolerance"] == pytest.approx(0.16)

    # explicit values still win (detail_mode="export" threads stay 0.01/0.1)
    captured.clear()
    quick_stl(p2.part, tmp_path, tolerance=0.01, angular_tolerance=0.1)
    assert captured["tolerance"] == 0.01
    assert captured["angular_tolerance"] == 0.1
