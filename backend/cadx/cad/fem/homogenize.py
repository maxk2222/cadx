"""Unit-cell homogenization for lattice/infill materials — C_H(rho).

Numerical two-scale homogenization on the existing Hex8 machinery: a
synthetic single-period `VoxelDomain` carries the same occupancy the
runtime lattice builders produce (`gyroid_occupancy` / `strut_occupancy`
at uniform density). For each of the 6 canonical unit macro-strains the
cell is solved under KUBC (kinematic uniform boundary conditions — the
upper/stiff bound), the micro stress field is volume-averaged into one
column of the effective constitutive tensor ``C_H``, and the micro
von-Mises peak vs. the macro VM yields a stress-localization factor
``A`` used to report `sf_micro` (member-scale failure estimate).

Tables are normalized to unit Young's modulus (``Ĉ = C_H / E0``) so one
table serves any base material with the same Poisson ratio; values are
interpolated monotonically in density and cached to disk under
``storage.data_root()/"homog"``.

Documented limits: KUBC is an upper bound (slightly stiff); the struts
table is built on the isotropic family blend (orientation-resolved
rotated tensors are a documented v2); ``A`` is a max-over-canonical-loads
estimator, not a certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from cadx.cad.fem.solver import (
    _D,
    _fixed_dofs,
    assemble,
    recover_elem_stress,
    reduce_case,
    solve_reduced,
)
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.logging import get_logger

log = get_logger(__name__)

DEFAULT_RHO_GRID = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90)


def _voigt_to_tensor(e6: np.ndarray) -> np.ndarray:
    """Symmetric 3x3 strain tensor from Voigt order [xx, yy, zz, xy, yz, zx]
    — the shear entries are ENGINEERING gammas, so the tensor's
    off-diagonals are g/2 (this is what the `_B_matrix` row layout maps
    displacements to)."""
    exx, eyy, ezz, gxy, gyz, gzx = [float(v) for v in e6]
    return np.array(
        [
            [exx, gxy / 2.0, gzx / 2.0],
            [gxy / 2.0, eyy, gyz / 2.0],
            [gzx / 2.0, gyz / 2.0, ezz],
        ]
    )


def _voigt_vm(s: np.ndarray) -> float:
    """Von Mises of a Voigt-6 stress vector."""
    sx, sy, sz, txy, tyz, tzx = [float(v) for v in s]
    return float(
        math.sqrt(
            max(
                sx**2 + sy**2 + sz**2 - sx * sy - sy * sz - sz * sx
                + 3.0 * (txy**2 + tyz**2 + tzx**2),
                0.0,
            )
        )
    )


@dataclass
class LatticeTable:
    """Interpolated homogenized properties for one lattice style.

    `C_at(rho)` returns the unit-E effective constitutive matrix (6,6) —
    multiply by the base Young's modulus for absolute MPa. `A_at(rho)`
    returns the stress-localization factor: peak micro von-Mises per
    macro von-Mises, >= 1 by construction.
    """

    style: str
    nu: float
    cell_voxels: int
    octet: bool
    rho_grid: np.ndarray  # (m,) ascending densities
    C_grid: np.ndarray  # (m, 6, 6) unit-E tensors at rho_grid
    A_grid: np.ndarray  # (m,) localization factors at rho_grid

    def C_at(self, rho: float) -> np.ndarray:
        """(6,6) unit-E effective tensor at `rho`.

        Per-entry monotone envelope (a stiffer tensor grows monotonically
        in density; sampling noise on the coarse cell would otherwise
        produce non-physical dips) + linear interpolation clamped to the
        grid range."""
        x = np.asarray(self.rho_grid, dtype=float)
        env = np.maximum.accumulate(np.asarray(self.C_grid, dtype=float), axis=0)
        out = np.empty((6, 6))
        for i in range(6):
            for j in range(6):
                out[i, j] = np.interp(float(rho), x, env[:, i, j])
        return out

    def A_at(self, rho: float) -> float:
        """Stress-localization factor at `rho` (monotone non-increasing
        envelope, clamped to >= 1 — the macro estimate can never read
        stiffer/safer than the resolved micro field)."""
        env = np.minimum.accumulate(np.asarray(self.A_grid, dtype=float))
        val = np.interp(
            float(rho), np.asarray(self.rho_grid, dtype=float), env
        )
        return float(max(val, 1.0))


def _unit_cell_domain(
    style: str,
    cell_mm: float,
    cell_voxels: int,
    rho: float,
    octet: bool,
    phase: np.ndarray | None = None,
) -> VoxelDomain:
    """Synthetic single-period cell: a `cell_voxels`³ cube spanning one
    lattice period (`cell_mm`), carrying the occupancy the runtime lattice
    builders produce at UNIFORM density `rho` — gyroid via
    `gyroid_occupancy`, struts via `strut_occupancy` on isotropic frames
    (evals all-ones → every family gets w = sqrt(d/nfam): the iso blend).

    Occupied components that touch no boundary face are dropped — floating
    islands leave unconstrained nodes and make K singular under KUBC.
    """
    from cadx.cad.fem.infill import gyroid_occupancy, strut_occupancy

    v = int(cell_voxels)
    pitch = float(cell_mm) / v
    base = VoxelDomain(
        origin=np.zeros(3),
        pitch=pitch,
        dims=(v, v, v),
        solid=np.ones((v, v, v), dtype=bool),
    )
    d = np.full(base.cell_count, float(rho))
    ph = None if phase is None else np.asarray(phase, dtype=float)
    if style in ("gyroid_graded", "gyroid"):
        occ = gyroid_occupancy(base, d, cell_mm, margin_cells=0, phase_mm=ph)
    elif style in ("struts_aligned", "struts"):
        evals = np.ones((base.cell_count, 3))
        evecs = np.broadcast_to(np.eye(3), (base.cell_count, 3, 3)).copy()
        occ = strut_occupancy(
            base,
            d,
            None,
            cell_mm,
            frames=(evals, evecs),
            margin_cells=0,
            phase_mm=ph,
            octet=octet,
            normalize=True,
        )
    else:
        raise ValueError(f"unknown lattice style {style!r}")
    occ3d = np.asarray(occ, dtype=bool)
    lab, ncomp = ndimage.label(occ3d)
    if ncomp:
        edge = np.concatenate(
            [
                lab[0, :, :].ravel(),
                lab[-1, :, :].ravel(),
                lab[:, 0, :].ravel(),
                lab[:, -1, :].ravel(),
                lab[:, :, 0].ravel(),
                lab[:, :, -1].ravel(),
            ]
        )
        keep = np.unique(edge[edge > 0])
        solid = np.isin(lab, keep)
        if not solid.any():
            # nothing reaches a face — keep the largest island so the cell
            # still solves (documented degenerate, e.g. ultra-low rho)
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            solid = lab == int(sizes.argmax())
    else:
        solid = np.zeros((v, v, v), dtype=bool)
    return VoxelDomain(
        origin=base.origin, pitch=pitch, dims=(v, v, v), solid=np.asarray(solid)
    )


def homogenize_cell(cell_domain: VoxelDomain, nu: float) -> tuple[np.ndarray, float]:
    """KUBC homogenization of one prepared cell -> (C_H (6,6) unit-E, A).

    For each canonical unit macro-strain the boundary nodes are prescribed
    the affine field u = E·x, the interior solves the zero-load correction
    (kinematic uniform boundary conditions — the stiff upper bound of the
    periodic cell), and the element stress field is volume-averaged into
    column j of C_H (uniform cell volume → plain mean). ``A`` is the worst
    ratio of peak micro von-Mises to the macro column's von-Mises over the
    6 canonical loads — the member-scale stress localization factor.
    """
    E_cells = cell_domain.solid.ravel().astype(float)
    K, edof, active = assemble(cell_domain, E_cells, nu)
    coords = cell_domain.node_coords()
    lo = np.asarray(cell_domain.origin, dtype=float)
    hi = lo + cell_domain.pitch * np.asarray(cell_domain.dims, dtype=float)
    tol = 1e-9 * max(float(cell_domain.pitch), 1.0)
    on_bnd = np.zeros(cell_domain.node_count, dtype=bool)
    for a in range(3):
        on_bnd |= np.isclose(coords[:, a], lo[a], atol=tol)
        on_bnd |= np.isclose(coords[:, a], hi[a], atol=tol)
    fixed3 = np.repeat(on_bnd[:, None], 3, axis=1)
    ctx = reduce_case(cell_domain, K, edof, fixed3)
    pdofs = _fixed_dofs(fixed3, cell_domain.node_count)
    nn3 = 3 * cell_domain.node_count
    C_H = np.zeros((6, 6))
    A = 1.0
    for j in range(6):
        e6 = np.zeros(6)
        e6[j] = 1.0
        E3 = _voigt_to_tensor(e6)
        u_p = (E3 @ (coords - lo).T).T.reshape(-1)  # affine boundary field
        F = np.zeros(nn3)
        if ctx.free.size:
            F[ctx.free] = np.asarray(-(K[ctx.free][:, pdofs] @ u_p[pdofs]), dtype=float)
        sol = solve_reduced(cell_domain, ctx, edof, active, F, E_cells, nu)
        u_full = sol.u.copy()
        u_full[pdofs] = u_p[pdofs]
        vm, sig = recover_elem_stress(cell_domain, u_full, edof, active, E_cells, nu)
        if not active.size:
            continue
        # volume average over the WHOLE cell (uniform element volume →
        # void elements contribute sigma = 0): C_H = (1/V)∫sigma dV —
        # the solid-phase mean would miss the rho factor entirely
        C_H[:, j] = sig.sum(axis=0) / cell_domain.cell_count
        macro_vm = _voigt_vm(C_H[:, j])
        A = max(A, float(vm[active].max(initial=0.0)) / max(macro_vm, 1e-12))
    C_H = 0.5 * (C_H + C_H.T)
    return C_H, A


def _homog_root() -> Path:
    from cadx import storage

    return storage.data_root() / "homog"


def _load_table(
    path: Path, rhos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load a cached (rho_grid, C_grid, A_grid) triple — validated on key,
    shape and the requested grid; any problem recomputes."""
    try:
        with np.load(path) as z:
            g = np.asarray(z["rho_grid"], dtype=float)
            C = np.asarray(z["C_grid"], dtype=float)
            A = np.asarray(z["A_grid"], dtype=float)
        if (
            g.shape != rhos.shape
            or not np.allclose(g, rhos)
            or C.shape != (rhos.size, 6, 6)
            or A.shape != rhos.shape
            or not np.isfinite(C).all()
            or not np.isfinite(A).all()
        ):
            return None
        return g, C, A
    except Exception:
        return None


def _save_table(path: Path, rhos: np.ndarray, C_grid: np.ndarray, A_grid: np.ndarray) -> None:
    """Atomic npz write (tmp + rename) — a crashed write must not poison
    the cache."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, rho_grid=rhos, C_grid=C_grid, A_grid=A_grid)
        tmp.replace(path)
    except Exception as e:
        log.warning("homog.cache_write_failed", error=str(e)[:200])


def lattice_table(
    style: str,
    nu: float,
    *,
    cell_voxels: int = 16,
    densities: tuple[float, ...] | None = None,
    octet: bool = True,
    cache_dir=None,
) -> LatticeTable:
    """Build or load the homogenized table for a lattice style.

    `densities` defaults to DEFAULT_RHO_GRID. Cached per
    (style, nu, cell_voxels, octet) under
    ``storage.data_root()/"homog"`` unless `cache_dir` overrides (tests).
    C_H is scale-free in cell size, so the synthetic cell runs at pitch
    1 mm (cell_mm = cell_voxels).
    """
    rhos = np.asarray(
        tuple(densities) if densities is not None else DEFAULT_RHO_GRID, dtype=float
    )
    if rhos.ndim != 1 or not rhos.size or not (rhos > 0).all():
        raise ValueError("densities must be a non-empty grid of positive values")
    root = Path(cache_dir) if cache_dir is not None else _homog_root()
    path = root / f"{style}_{int(cell_voxels)}_{float(nu):.4f}_{int(bool(octet))}.npz"
    cached = _load_table(path, rhos)
    if cached is not None:
        g, C, A = cached
        return LatticeTable(style, float(nu), int(cell_voxels), bool(octet), g, C, A)

    cell_mm = float(cell_voxels)  # pitch = 1 mm — C_H is scale-free
    C_grid = np.zeros((rhos.size, 6, 6))
    A_grid = np.ones(rhos.size)
    for i, rho in enumerate(rhos):
        cell = _unit_cell_domain(
            style, cell_mm, int(cell_voxels), float(rho), bool(octet)
        )
        C_H, A = homogenize_cell(cell, float(nu))
        C_grid[i] = C_H
        A_grid[i] = A
        log.info(
            "homog.cell",
            style=style,
            rho=float(rho),
            A=round(float(A), 3),
            cells=int(cell.solid_count),
        )
    _save_table(path, rhos, C_grid, A_grid)
    return LatticeTable(style, float(nu), int(cell_voxels), bool(octet), rhos, C_grid, A_grid)


def infill_homog_elems(
    domain: VoxelDomain,
    d_field: np.ndarray,
    interior_mask: np.ndarray,
    table: LatticeTable,
    *,
    base_D: np.ndarray | None = None,
    n_buckets: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize the density field into constitutive classes for a
    homogenized infill solve.

    Returns ``(elem_class, class_D, micro_amp)``:

    - ``elem_class`` (ncells,) int — 0 for solid/shell/passive cells AND
      for interior cells with ``d <= 0`` (those are void in the homog
      model — the caller masks them out of the solve via
      ``domain.solid & d>0``); interior cells with ``d > 0`` get
      ``1 + bucket`` with ``bucket`` the index of the nearest
      ``table.rho_grid`` entry.
    - ``class_D`` (1+K, 6, 6) unit-E tensors — entry 0 is ``base_D`` when
      given (e.g. the aniso base ``_D(nu, k, axis)`` — it is baked in, so
      the solve runs with aniso=None), else ``_D(table.nu)``; entries
      1..K are ``Ĉ(rho_b)`` at the grid densities. At most ``n_buckets``
      DISTINCT grid entries are used (spread evenly over the grid,
      K <= len(rho_grid)); entries for unselected buckets still carry
      their grid tensor so every emitted class id resolves.
    - ``micro_amp`` (ncells,) float — ``A(rho_b)`` on interior cells, 1.0
      elsewhere; multiply the solved macro elem_vm to get the micro
      estimate.
    """
    d = np.asarray(d_field, dtype=float).ravel()
    interior = np.asarray(interior_mask, dtype=bool).ravel()
    grid = np.asarray(table.rho_grid, dtype=float)
    elem_class = np.zeros(domain.cell_count, dtype=np.int64)
    micro_amp = np.ones(domain.cell_count)

    if interior.any():
        nb = max(1, min(int(n_buckets), grid.size))
        sel = np.unique(np.rint(np.linspace(0, grid.size - 1, nb)).astype(np.int64))
        cand = grid[sel]
        dc = np.clip(d[interior], grid[0], grid[-1])
        bucket = np.argmin(np.abs(dc[:, None] - cand[None, :]), axis=1)
        rho_b = cand[bucket]
        a_map = {float(r): table.A_at(float(r)) for r in cand}
        micro_amp[interior] = np.array([a_map[float(r)] for r in rho_b])
        live_idx = np.flatnonzero(interior)
        keep = d[interior] > 0.0
        elem_class[live_idx[keep]] = 1 + sel[bucket[keep]]

    base = np.asarray(base_D, dtype=float) if base_D is not None else _D(float(table.nu))
    class_D = [base] + [table.C_at(float(r)) for r in grid]
    return elem_class, np.stack(class_D), micro_amp
