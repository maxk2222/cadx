"""Deterministic heightfield candidate for flat parts."""

from __future__ import annotations

import numpy as np
import pytest

from cadx.cad.heightfield import build_heightfield_mesh, heightfield_candidate


class _Dim:
    def __init__(self, v):
        self.value = v


class _Spec:
    def __init__(self, lx=119.0, w=34.0, h=8.0):
        self.length_x = _Dim(lx)
        self.width_y = _Dim(w)
        self.height_z = _Dim(h)


RECT = [[-60, -17], [60, -17], [60, 17], [-60, 17], [-60, -17]]


def test_build_heightfield_watertight_and_spec_shaped():
    m = build_heightfield_mesh(RECT, 8.0)
    assert m.is_watertight
    ext = m.extents
    assert ext[0] == pytest.approx(120.0, abs=2.0)
    assert ext[1] == pytest.approx(34.0, abs=2.0)
    assert ext[2] == pytest.approx(8.0, abs=2.0)
    # rests on z=0, centred in XY
    assert m.bounds[0][2] >= -0.01
    c = (m.bounds[0] + m.bounds[1]) / 2
    assert abs(c[0]) < 1.0 and abs(c[1]) < 1.0


def test_build_heightfield_rounded_outline():
    """Non-rectangular outlines are followed, not boxed."""
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    outline = [[float(30 * np.cos(a)), float(15 * np.sin(a))] for a in t]
    m = build_heightfield_mesh(outline, 6.0)
    assert m.is_watertight
    assert m.extents[0] == pytest.approx(60.0, abs=2.0)
    # ellipse area ~ pi*30*15 = 1414 — a bounding box would be ~1800
    assert m.volume / 6.0 == pytest.approx(1414, rel=0.15)


def test_heightfield_candidate_rejects_non_flat():
    spec = _Spec(lx=40.0, w=40.0, h=40.0)  # cube — not flat
    assert heightfield_candidate(spec, RECT) is None


def test_heightfield_candidate_needs_outline_and_height():
    assert heightfield_candidate(_Spec(), None) is None
    assert heightfield_candidate(_Spec(), []) is None
    spec = _Spec(h=0.0)  # no stated height
    assert heightfield_candidate(spec, RECT) is None


def test_heightfield_candidate_flat_builds():
    m = heightfield_candidate(_Spec(), RECT)
    assert m is not None
    assert m.is_watertight
    assert m.extents[2] == pytest.approx(8.0, abs=2.0)


def test_heightfield_outline_is_exact_not_rasterized():
    """The silhouette must follow the measured polygon exactly — no PIL
    raster stairs. Mid-height section polygon ≈ outline polygon."""
    from shapely.geometry import Polygon

    # organic blobby outline — rasterization would visibly stair-step it
    t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    outline = [
        [float((50 + 6 * np.sin(3 * a)) * np.cos(a)), float((20 + 4 * np.cos(5 * a)) * np.sin(a))]
        for a in t
    ]
    m = build_heightfield_mesh(outline, 8.0)
    assert m.is_watertight
    sec = m.section(plane_origin=[0, 0, 4.0], plane_normal=[0, 0, 1])
    assert sec is not None
    path2d, _ = sec.to_2D()
    mesh_poly = max(path2d.polygons_full, key=lambda p: p.area)
    ref = Polygon(outline)
    # section polygon ≈ outline (IoU-style symmetric difference)
    sym = mesh_poly.symmetric_difference(ref).area
    assert sym / ref.area < 0.01


def test_heightfield_relief_from_image(tmp_path):
    """Relief mode: luminance of the top-view image raises the top surface —
    watertight, outline stays exact, z exceeds the base height."""
    from PIL import Image

    # object = opaque 400x200 px rect (px_per_mm=5 → 80x40 mm) centered in a
    # transparent canvas; luminance bumps in the middle → top relief
    W, H = 500, 300
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    px = img.load()
    x0, y0 = (W - 400) // 2, (H - 200) // 2
    for y in range(y0, y0 + 200):
        for x in range(x0, x0 + 400):
            d = 1.0 - min(np.hypot(x - W / 2, y - H / 2) / 160.0, 1.0)
            v = int(120 + 120 * d)
            px[x, y] = (v, v, v, 255)
    img_path = tmp_path / "top.png"
    img.save(img_path)

    outline = [[-40, -20], [40, -20], [40, 20], [-40, 20], [-40, -20]]
    m = build_heightfield_mesh(outline, 8.0, relief_mm=1.5, relief_image=img_path, px_per_mm=5.0)
    assert m.is_watertight
    assert m.bounds[1][2] > 8.3  # relief raised the centre above base height
    assert len(m.faces) > 1000  # dense interior triangulation, not 12 tris
    # footprint stays the measured rect
    assert m.extents[0] == pytest.approx(80.0, abs=1.0)
    assert m.extents[1] == pytest.approx(40.0, abs=1.0)


def test_heightfield_relief_flat_fallback(tmp_path):
    """Uniform-luminance image (no texture) falls back to a flat top cap —
    still watertight, exact height."""
    from PIL import Image

    img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    px = img.load()
    for y in range(50, 250):
        for x in range(50, 250):
            px[x, y] = (160, 160, 160, 255)
    img_path = tmp_path / "flat.png"
    img.save(img_path)

    m = build_heightfield_mesh(RECT, 8.0, relief_mm=1.5, relief_image=img_path, px_per_mm=4.0)
    assert m.is_watertight
    assert m.bounds[1][2] == pytest.approx(8.0, abs=0.01)  # no relief added
