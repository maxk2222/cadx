"""Unit-cell homogenization (`cad/fem/homogenize.py`).

KUBC solves of 6 canonical macro-strains on synthetic single-period
cells produce C_H(rho) (unit-E effective tensor) and A(rho) (micro
stress-localization factor, >= 1 by construction). The full-solid limit
must recover _D(nu) exactly; strut cells follow the mixture rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadx.cad.fem.homogenize import (
    _unit_cell_domain,
    homogenize_cell,
    infill_homog_elems,
    lattice_table,
)
from cadx.cad.fem.infill import strut_occupancy
from cadx.cad.fem.solver import _D
from cadx.cad.fem.voxelize import VoxelDomain

NU = 0.36


def _full_cell(v: int = 8) -> VoxelDomain:
    return VoxelDomain(
        origin=np.zeros(3),
        pitch=1.0,
        dims=(v, v, v),
        solid=np.ones((v, v, v), dtype=bool),
    )


def _strut_cell(v: int, rho: float, phase: np.ndarray | None = None) -> VoxelDomain:
    """Single-period cell of pure x-struts at target density `rho`."""
    base = _full_cell(v)
    d = np.full(base.cell_count, rho)
    dirs = np.tile([1.0, 0.0, 0.0], (base.cell_count, 1))
    occ = strut_occupancy(
        base,
        d,
        dirs,
        float(v),
        margin_cells=0,
        phase_mm=phase,
        octet=False,
        normalize=True,
    )
    return VoxelDomain(
        origin=base.origin, pitch=base.pitch, dims=base.dims, solid=occ
    )


# --------------------------------------------------------------------------- #
# canonical solves                                                            #
# --------------------------------------------------------------------------- #


def test_full_solid_cell_recovers_D() -> None:
    """rho=1 cell: C_H == _D(nu) to round-off, A == 1 (no localization)."""
    C, A = homogenize_cell(_full_cell(8), NU)
    np.testing.assert_allclose(C, _D(NU), rtol=1e-6, atol=1e-9)
    assert pytest.approx(1.0, abs=1e-6) == A


def test_full_solid_via_unit_cell_domain() -> None:
    """_unit_cell_domain at rho>=~0.995 pins the whole cube solid — the
    table path sees the same full-cell limit."""
    cell = _unit_cell_domain("gyroid_graded", 8.0, 8, 1.0, octet=True)
    assert cell.solid_count == cell.cell_count
    C, _A = homogenize_cell(cell, NU)
    np.testing.assert_allclose(C, _D(NU), rtol=1e-6, atol=1e-9)


def test_x_strut_cell_mixture_rule() -> None:
    """Pure x-struts at density rho: C_H[0,0] ~ rho (mixture rule).

    The struts are phase-shifted off the cell boundary — a strut lying on
    a KUBC-pinned face reads artificially stiff (boundary-layer effect)."""
    for rho in (0.30, 0.50):
        v = 12
        cell = _strut_cell(v, rho, phase=np.array([v / 2.0, v / 2.0, 0.0]))
        assert cell.solid_count > 0
        C, _A = homogenize_cell(cell, NU)
        assert C[0, 0] == pytest.approx(rho, rel=0.15)
        # the strut direction is the stiffest response of the cell
        assert C[0, 0] == np.diag(C).max()
        assert C[0, 0] > C[1, 1]


def test_x_strut_cell_symmetric_spd() -> None:
    """C_H is symmetrized and positive (semi-)definite: every canonical
    strain direction stores non-negative energy."""
    cell = _strut_cell(12, 0.40, phase=np.array([6.0, 6.0, 0.0]))
    C, _A = homogenize_cell(cell, NU)
    np.testing.assert_allclose(C, C.T, atol=1e-12)
    eig = np.linalg.eigvalsh(C)
    assert eig.min() > -1e-9
    assert eig.max() > 0.0


def test_micro_amp_factor_at_least_one() -> None:
    """A = peak micro VM / macro VM >= 1 for every canonical load — the
    homogenized estimate can never under-report the micro stress."""
    for cell in (_full_cell(6), _strut_cell(10, 0.35)):
        _C, A = homogenize_cell(cell, NU)
        assert A >= 1.0


# --------------------------------------------------------------------------- #
# lattice_table + cache                                                       #
# --------------------------------------------------------------------------- #


def test_lattice_table_caching_deterministic(tmp_path) -> None:
    """The disk cache returns the identical table on a second call — and
    interpolation yields sane entries."""
    rhos = (0.25, 0.50)
    t1 = lattice_table(
        "struts_aligned", NU, cell_voxels=8, densities=rhos, cache_dir=tmp_path
    )
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    t2 = lattice_table(
        "struts_aligned", NU, cell_voxels=8, densities=rhos, cache_dir=tmp_path
    )
    np.testing.assert_array_equal(t1.C_grid, t2.C_grid)
    np.testing.assert_array_equal(t1.A_grid, t2.A_grid)
    np.testing.assert_array_equal(t1.rho_grid, t2.rho_grid)
    C_mid = t1.C_at(0.375)
    assert C_mid.shape == (6, 6)
    assert np.isfinite(C_mid).all()
    assert t1.A_at(0.375) >= 1.0


def test_lattice_table_monotone_in_density(tmp_path) -> None:
    """C_at is density-monotone per diagonal entry (envelope) — a denser
    lattice may not soften."""
    t = lattice_table(
        "struts_aligned",
        NU,
        cell_voxels=8,
        densities=(0.15, 0.35, 0.65),
        cache_dir=tmp_path,
    )
    diag = np.stack([np.diag(t.C_at(r)) for r in (0.15, 0.35, 0.65)])
    assert (np.diff(diag[:, :3], axis=0) >= -1e-9).all()
    assert diag[-1, :3].max() > diag[0, :3].max()  # actually grows
    a = np.array([t.A_at(r) for r in (0.15, 0.35, 0.65)])
    assert (a >= 1.0).all()
    assert (np.diff(a) <= 1e-9).all()  # localization shrinks with density


# --------------------------------------------------------------------------- #
# infill_homog_elems bucketing                                                #
# --------------------------------------------------------------------------- #


def test_infill_homog_elems_buckets(tmp_path) -> None:
    """Interior densities bucket into <= n_buckets classes on top of the
    base class 0; micro_amp carries A(rho_b) on interior cells, 1 outside."""
    t = lattice_table(
        "struts_aligned", NU, cell_voxels=8,
        densities=(0.15, 0.30, 0.50, 0.75), cache_dir=tmp_path,
    )
    dom = _full_cell(4)
    d = np.where(
        np.indices(dom.dims)[0].ravel() < 2, 0.4, 0.0
    )  # half interior at d=0.4, half void
    interior = np.indices(dom.dims)[0].ravel() < 2
    ec, cd, amp = infill_homog_elems(dom, d, interior, t, n_buckets=2)

    assert ec.shape == (dom.cell_count,) and cd.ndim == 3 and cd.shape[1:] == (6, 6)
    assert amp.shape == (dom.cell_count,)
    # interior cells with d>0 get 1+bucket of the nearest grid density
    assert (ec[interior] >= 1).all() and (ec[~interior] == 0).all()
    # all used classes are < len(class_D) and tensors are finite
    assert ec.max() < cd.shape[0]
    assert np.isfinite(cd).all()
    # micro_amp = A at the bucketed rho on interior, 1.0 elsewhere —
    # the bucket set is the n_buckets-even spread over the grid, so the
    # bucketed rho is the nearest SELECTED grid entry, not the global one
    grid = np.asarray(t.rho_grid)
    sel = np.unique(np.rint(np.linspace(0, grid.size - 1, 2)).astype(np.int64))
    rho_b = grid[sel][np.argmin(np.abs(grid[sel] - 0.4))]
    np.testing.assert_allclose(amp[interior], t.A_at(float(rho_b)))
    np.testing.assert_allclose(amp[~interior], 1.0)
    # class 0 is the isotropic base tensor when base_D is not given
    np.testing.assert_allclose(cd[0], _D(NU))


def test_infill_homog_elems_base_D_passthrough(tmp_path) -> None:
    """base_D (e.g. the aniso base tensor) is installed as class_D[0]."""
    t = lattice_table(
        "struts_aligned", NU, cell_voxels=8,
        densities=(0.3, 0.6), cache_dir=tmp_path,
    )
    dom = _full_cell(4)
    interior = np.zeros(dom.cell_count, dtype=bool)
    d = np.zeros(dom.cell_count)
    aniso_D = _D(NU, 0.5, 2)
    ec, cd, _amp = infill_homog_elems(dom, d, interior, t, base_D=aniso_D)
    np.testing.assert_allclose(cd[0], aniso_D)
    assert (ec == 0).all()  # no interior -> everything is base class
