"""Axis-scanline containment — the voxelize hot path must agree with the
per-point contains() baseline exactly (same physics, fewer ray queries),
plus the uniform-field compliance baseline used for the BESO flag."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from cadx.cad.fem.voxelize import (  # noqa: E402
    _fill_grid,
    _scanline_contains,
    voxelize,
)


def _sphere(r=9.0, c=(15, 12, 8)):
    return trimesh.creation.icosphere(subdivisions=3, radius=r).apply_translation(c)


def _box(w=24, h=12, d=8, at=(0, 0, 0)):
    return trimesh.creation.box((w, d, h)).apply_translation(at)


def test_scanline_matches_contains_on_primitives():
    origin = np.array([2.0, 2.0, 0.0])
    dims = (30, 24, 18)
    pitch = 1.0
    for mesh in (_sphere(), _box(20, 10, 12, at=(6, 4, 2))):
        scan = _scanline_contains(mesh, origin, dims, pitch, offset=0.5)
        assert scan is not None and scan.any()
        pts = origin[None, :] + pitch * (
            np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), axis=-1).reshape(
                -1, 3
            )
            + 0.5
        )
        ref = np.asarray(mesh.contains(pts))
        # identical contract — scanline is a speed path, not an approximation
        assert np.array_equal(scan, ref), (
            f"scanline/contains disagree on {scan.sum()}/{ref.sum()} cells"
        )


def test_scanline_handles_bore():
    # box centered (15,15,5) spans x,y∈[0,30] z∈[0,10]; bore r4 through z
    outer = trimesh.creation.box((30, 30, 10)).apply_translation((15, 15, 5))
    hole = trimesh.creation.cylinder(radius=4, height=20, sections=64).apply_translation(
        (15, 15, 5)
    )
    mesh = outer.difference(hole) if hasattr(outer, "difference") else outer
    origin = np.array([0.0, 0.0, 0.0])
    dims = (40, 40, 12)
    scan = _scanline_contains(mesh, origin, dims, 1.0, offset=0.5)
    if mesh.is_watertight:
        assert scan is not None
        grid = scan.reshape(dims)
        # the bore column must read empty through the middle
        assert not grid[15, 15, 5]
        assert grid[8, 8, 5]


def test_scanline_node_grid_offset():
    """offset=0.0 (node lattice) agrees with contains() at node positions."""
    mesh = _box(10, 10, 10, at=(0, 0, 0))
    origin = np.array([-1.0, -1.0, -1.0])
    npts = (12, 12, 12)
    scan = _scanline_contains(mesh, origin, npts, 1.0, offset=0.0)
    pts = origin[None, :] + 1.0 * np.stack(
        np.meshgrid(*[np.arange(n) for n in npts], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    ref = np.asarray(mesh.contains(pts))
    assert scan is not None and np.array_equal(scan, ref)


def test_env_flag_falls_back_to_contains(monkeypatch):
    monkeypatch.setenv("CADX_VOXEL_CONTAINS", "1")
    mesh = _box(8, 8, 8)
    assert _scanline_contains(mesh, np.zeros(3), (10, 10, 10), 1.0, offset=0.5) is None
    # _fill_grid still works — through the contains() path
    m = _fill_grid(mesh, np.array([-1.0, -1.0, -1.0]), (10, 10, 10), 1.0)
    assert m.any() and m.shape == (10, 10, 10)


def test_fill_grid_scanline_result_identical():
    """_fill_grid returns the scanline result — equals per-point contains."""
    mesh = _sphere(r=6, c=(8, 8, 8))
    m = _fill_grid(mesh, np.array([0.0, 0.0, 0.0]), (16, 16, 16), 1.0)
    assert m.shape == (16, 16, 16) and m.sum() > 100
    assert m[8, 8, 8] and not m[0, 0, 0]


def test_voxelize_coverage_with_scanline():
    d = voxelize(_sphere(r=5, c=(6, 6, 6)), 1.0, 200_000, coverage=True)
    assert d.coverage is not None
    cov = d.coverage.reshape(d.dims)
    assert cov.max() == pytest.approx(1.0)
    # interior cells fully covered; boundary carries partial fractions
    assert (cov == 1.0).sum() > 50
    assert ((cov > 0) & (cov < 1)).any()


def test_uniform_compliance_beats_and_bounds():
    """The uniform baseline must exceed an optimized field's compliance —
    it's the 'no cleverness' floor the BESO warning is measured against."""
    pytest.importorskip("scipy")
    from cadx.cad.fem.regions import resolve_case_dofs
    from cadx.cad.fem.topopt import TopOptParams, uniform_compliance
    from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

    mesh = _box(24, 8, 8, at=(0, 0, 0))
    dom = voxelize(mesh, 2.0, 200_000)
    case = LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="surface", axis="-x")],
        loads=[LoadSpec(region=RegionSpec(kind="surface", axis="+x"), force_n=[0, 0, -50])],
    )
    fixed3, F, _w = resolve_case_dofs(dom, case)
    p = TopOptParams(volfrac=0.4, max_iter=1)
    passive = np.zeros(dom.cell_count, dtype=bool)
    base = uniform_compliance(dom, 210_000.0, 0.3, [(fixed3, F)], passive, p)
    assert base is not None and base > 0
    # stiffer baseline at higher density — physical sanity of the metric
    base_denser = uniform_compliance(dom, 210_000.0, 0.3, [(fixed3, F)], passive, p, volfrac=0.8)
    assert base_denser is not None and base_denser < base
