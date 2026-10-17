"""Silhouette IoU: reference-image mask vs mesh side projections (Q5)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from cadx.cad.silhouette import (
    _norm_mask,
    load_mask,
    mesh_silhouette_iou,
    reference_mask,
    save_mask,
    silhouette_iou,
)


def _alpha_png(draw_fn, size: int = 200) -> bytes:
    """RGBA test image: transparent background, drawn subject."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img), size)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _circle_mask(size: int = 128) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    yy, xx = np.ogrid[:size, :size]
    c = size / 2
    m[(yy - c) ** 2 + (xx - c) ** 2 <= (size * 0.4) ** 2] = True
    return m


def test_reference_mask_from_alpha(tmp_path):
    png = _alpha_png(
        lambda d, s: d.ellipse([s * 0.2, s * 0.2, s * 0.8, s * 0.8], fill=(9, 9, 9, 255))
    )
    p = tmp_path / "part.png"
    p.write_bytes(png)
    mask = reference_mask(p)
    assert mask is not None and mask.dtype == bool
    frac = mask.mean()
    assert 0.25 < frac < 0.6  # a disc of r≈0.3·size


def test_reference_mask_opaque_without_rembg(tmp_path, monkeypatch):
    """Opaque image without alpha returns None when no rembg model cached."""
    import cadx.cad.silhouette as sil

    monkeypatch.setattr(sil, "_rembg_cached", lambda: False)
    img = Image.new("RGB", (64, 64), (200, 30, 30))
    p = tmp_path / "photo.png"
    img.save(p)
    assert reference_mask(p) is None


def test_save_load_mask_roundtrip(tmp_path):
    m = _circle_mask()
    p = tmp_path / "mask.png"
    save_mask(m, p)
    back = load_mask(p)
    assert back is not None
    assert abs(back.mean() - m.mean()) < 0.01


def test_norm_mask_scale_invariant():
    """Same shape at different resolution → same normalised mask."""
    small = _circle_mask(64)
    big = _circle_mask(256)
    a, b = _norm_mask(small), _norm_mask(big)
    assert a is not None and b is not None
    iou = float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())
    assert iou > 0.9


def test_silhouette_iou_sphere_vs_circle_mask():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10)
    iou, view = silhouette_iou(sphere, _circle_mask())
    assert iou > 0.85
    assert view is not None


def test_silhouette_iou_rejects_wrong_shape():
    """A long rod's silhouette must not match a disc mask."""
    rod = trimesh.creation.cylinder(radius=5, height=80, sections=32)
    iou, _ = silhouette_iou(rod, _circle_mask())
    assert iou < 0.65


def test_silhouette_iou_flat_slab_matches_top_view():
    """A flat slab's side views are thin strips — a wide-rectangle mask (like
    a plan-view photo) should match the top silhouette strongly."""
    slab = trimesh.creation.box(extents=[40, 30, 4])
    flat_mask = np.zeros((128, 128), dtype=bool)
    flat_mask[20:70, 20:100] = True  # wide rectangle, like the top view
    iou, _ = silhouette_iou(slab, flat_mask, views=("+z", "-z"))
    assert iou > 0.7


def test_mesh_silhouette_iou_identical():
    a = trimesh.creation.cylinder(radius=10, height=50, sections=32)
    assert mesh_silhouette_iou(a, a.copy()) == pytest.approx(1.0, abs=0.02)


def test_mesh_silhouette_iou_detects_wreck():
    """Halving a part off (the flatten_bottom failure mode) drags the
    same-view silhouette IoU down — the shape guard's signal."""
    body = trimesh.creation.cylinder(radius=12, height=60, sections=32)
    wrecked = body.copy()
    wrecked.apply_scale([1.0, 1.0, 0.5])
    val = mesh_silhouette_iou(body, wrecked)
    # side views halve (≈0.67 each), top view unchanged → mean ≈ 0.78,
    # clearly below the 0.8 guard the shape guard applies
    assert 0.0 <= val < 0.9


def _diagonal_band_mask(deg: float = 30.0, L: int = 700, W: int = 200) -> np.ndarray:
    """Axis-aligned bar rotated in the image plane — the grip-photo failure
    mode: rembg returns a diagonal subject mask and the naive vertical-slice
    profile reads ~half the true diameter."""
    from scipy.ndimage import rotate

    band = np.ones((W, L), dtype=np.uint8)
    return rotate(band, deg, reshape=True, order=0) > 0


def test_mask_to_outline_mm_deskews_diagonal_subject():
    """A diagonally-photographed band must yield the same outline aspect as
    the upright band — the deskew normalises photo orientation."""
    from cadx.cad.silhouette import mask_to_outline_mm

    up = mask_to_outline_mm(_diagonal_band_mask(0), 130.0, 32.0)
    dg = mask_to_outline_mm(_diagonal_band_mask(30), 130.0, 32.0)
    w_up = max(c[0] for c in up) - min(c[0] for c in up)
    w_dg = max(c[0] for c in dg) - min(c[0] for c in dg)
    assert abs(w_up - w_dg) / w_up < 0.1


def test_profile_stations_deskewed_read_true_radius():
    """Regression: diagonal mask previously produced radius_mm ~half the
    true band half-height (Ø32 read as ~Ø16) — deskew restores it."""
    from cadx.cad.silhouette import mask_to_outline_mm, profile_stations_from_outline

    # band W=200 scaled to extent_v=32 -> true radius = 16 mm
    out = mask_to_outline_mm(_diagonal_band_mask(25), 130.0, 32.0)
    st = profile_stations_from_outline(out, axis="x", n_stations=16)
    radii = [s["radius_mm"] for s in st]
    assert min(radii) > 12.0  # was ~8 without deskew
    assert max(radii) <= 17.5


def test_profile_stations_square_mask_not_deskewed():
    """A disc/square mask has no dominant axis — deskew must leave it alone
    (PCA axis is meaningless there)."""
    from cadx.cad.silhouette import _deskew_mask, _mask_crop

    m = _mask_crop(_circle_mask())
    assert m is not None
    assert np.array_equal(_deskew_mask(m), m)
