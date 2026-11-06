"""Native-vs-trimesh parity for the geometry query kernels.

These gate every geometry callsite swap: `native.*` must reproduce the
trimesh results on the corpus before a call site may switch to it. Skip
cleanly when cadx_native is absent (pure-Python environments stay valid)."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
cadx_native = pytest.importorskip("cadx_native")

from cadx.cad import native  # noqa: E402
from cadx.cad.fem.voxelize import _scanline_contains  # noqa: E402


def _box(w=24.0, h=12.0, d=8.0, at=(0.0, 0.0, 0.0)):
    return trimesh.creation.box((w, d, h)).apply_translation(at)


def _sphere(r=9.0, c=(15.0, 12.0, 8.0)):
    return trimesh.creation.icosphere(subdivisions=3, radius=r).apply_translation(c)


def _bore_solid():
    outer = trimesh.creation.box((30, 30, 10)).apply_translation((15, 15, 5))
    hole = trimesh.creation.cylinder(radius=4, height=20, sections=48).apply_translation(
        (15, 15, 5)
    )
    return outer.difference(hole)


def _cracked(mesh):
    """Deterministically broken mesh — drop every 11th face."""
    m = mesh.copy()
    keep = np.ones(len(m.faces), dtype=bool)
    keep[::11] = False
    m.update_faces(keep)
    return m


def _sample_pts(mesh, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    lo, hi = mesh.bounds
    span = (hi - lo) * 1.3
    return rng.uniform(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo), (n, 3)) + 0.0 * span


@pytest.mark.parametrize("builder", [_box, _sphere, _bore_solid])
def test_contains_parity(builder):
    mesh = builder()
    idx = native.mesh_index(mesh)
    assert idx is not None
    pts = _sample_pts(mesh)
    ref = np.asarray(mesh.contains(pts), dtype=bool)
    got = np.asarray(native.contains(mesh, pts), dtype=bool)
    n_diff = int((ref != got).sum())
    # parity must be exact for regular points — report the disagreement
    # count so flaky boundary points are visible rather than hidden by atol
    assert n_diff == 0, f"{n_diff}/{len(pts)} contains disagreements"


def test_contains_parity_cracked_mesh():
    mesh = _cracked(_sphere())
    pts = _sample_pts(mesh)
    ref = np.asarray(mesh.contains(pts), dtype=bool)
    got = np.asarray(native.contains(mesh, pts), dtype=bool)
    assert np.array_equal(ref, got)


def test_ray_any_parity():
    mesh = _bore_solid()
    rng = np.random.default_rng(3)
    lo, hi = mesh.bounds
    n = 500
    origins = np.tile([lo[0] - 1.0, 0.0, 0.0], (n, 1)).astype(float)
    origins[:, 1] = rng.uniform(lo[1], hi[1], n)
    origins[:, 2] = rng.uniform(lo[2], hi[2], n)
    dirs = np.tile([1.0, 0.0, 0.0], (n, 1))
    ref = np.asarray(mesh.ray.intersects_any(origins, dirs), dtype=bool)
    got = np.asarray(native.ray_any(mesh, origins, dirs), dtype=bool)
    assert np.array_equal(ref, got)


def test_ray_locs_parity():
    mesh = _bore_solid()
    lo, _hi = mesh.bounds
    origins = np.array(
        [[lo[0] - 1.0, 15.0, 5.0], [lo[0] - 1.0, 12.0, 5.0], [0.0, 0.0, 0.0]], dtype=float
    )
    dirs = np.tile([1.0, 0.0, 0.0], (len(origins), 1))
    ref_l, ref_r, _t = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    got = native.ray_locs(mesh, origins, dirs)
    assert got is not None
    got_l, got_r, _gt = got
    # compare hit COORDINATE sets per ray — trimesh may split shared edges
    # differently; positions along the ray are the contract
    for r in np.unique(ref_r):
        a = np.sort(np.round(ref_l[ref_r == r][:, 0], 6))
        b = np.sort(np.round(got_l[got_r == r][:, 0], 6))
        assert a.shape == b.shape and np.allclose(a, b), f"ray {r}: {a} vs {b}"


def test_scanline_parity_grid():
    """Exact equality against _scanline_contains — the reference impl."""
    origin = np.array([0.0, 0.0, 0.0])
    dims = (30, 24, 18)
    pitch = 1.0
    for mesh in (_sphere(), _box(20, 10, 12, at=(6, 4, 2)), _bore_solid()):
        ref = _scanline_contains(mesh, origin, dims, pitch, offset=0.5)
        got = native.scanline_fill(mesh, origin, dims, pitch, 0.5)
        assert got is not None
        if ref is None:
            continue
        assert np.array_equal(ref, got), (
            f"scanline disagrees on {int((ref != got).sum())} cells for {type(mesh).__name__}"
        )


def test_scanline_parity_node_offset():
    mesh = _sphere()
    origin = np.array([0.0, 0.0, 0.0])
    dims = (25, 20, 14)
    ref = _scanline_contains(mesh, origin, dims, 1.0, offset=0.0)
    got = native.scanline_fill(mesh, origin, dims, 1.0, 0.0)
    assert got is not None and ref is not None
    assert np.array_equal(ref, got)


def test_mesh_index_cached_per_mesh():
    mesh = _box()
    a = native.mesh_index(mesh)
    b = native.mesh_index(mesh)
    assert a is b  # WeakKey cache — one BVH build per mesh object
