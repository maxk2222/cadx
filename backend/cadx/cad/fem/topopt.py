"""SIMP topology optimization on the voxel domain (Liu & Tovar top3d scheme).

Minimize compliance  sum_i u_i^T K u_i  subject to
    mean(x over design cells) <= volfrac
with solid-void material interpolation  E(x) = Emin + x^penal * (E0 - Emin),
an OC update with bisected Lagrange multiplier and a cone sensitivity filter.

Non-design constraints:
- cells outside the part solid are pinned to rho_min (void)
- keep/load/fix regions are pinned to 1 (passive solid — functional features
  like mounts and holes are preserved by construction)
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.spatial import cKDTree

from cadx.cad.fem.solver import _ke_unit, assemble, element_dofs
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.logging import get_logger

log = get_logger(__name__)

RHO_MIN = 1e-3
E_MIN_FRAC = 1e-9  # Emin = E0 * E_MIN_FRAC (prevents singular stiffness)

ProgressCb = Callable[[int, dict[str, Any]], None]
CancelCb = Callable[[], bool]


@dataclass
class TopOptParams:
    volfrac: float = 0.4  # material fraction of the design space to keep
    penal: float = 3.0  # SIMP penalty
    rmin_mm: float = 3.0  # sensitivity filter radius (min feature size)
    max_iter: int = 60
    tol: float = 0.01  # stop when max density change < tol
    converge_patience: int = 5  # also stop after N iters of stalled compliance
    move: float = 0.2  # OC move limit
    threshold: float = 0.5  # iso threshold for extraction
    # M3.1: density filter variant —
    #   "cone"                classic normalized sensitivity filter (H x / Hs)
    #   "heaviside"           cone filter + smoothed Heaviside projection
    #                         (beta doubles on every compliance stall up to
    #                         beta_max → crisp 0/1 designs whose extraction
    #                         is threshold-robust)
    #   "helmholtz"           Helmholtz PDE filter (Lazarov & Sigmund 2011):
    #                         x̃ = L⁻¹x with L = I - r²Δ on the solid cells,
    #                         homogeneous Neumann BC — consistent filter
    #                         length right up to the domain boundary (no
    #                         cone-region clipping → no thinned edge
    #                         members) and a sparse SPD solve instead of the
    #                         dense-distance cone matrix
    #   "helmholtz+heaviside" PDE filter + Heaviside projection (the crisp
    #                         variant without the cone boundary artefacts)
    filter: str = "cone"
    beta0: float = 1.0  # initial Heaviside sharpness
    beta_max: float = 8.0  # cap — beyond this the projection is ~binary
    # M2: Langelaar-style additive-manufacturing filter — a cell only counts
    # as printed if its 45° support cone below is printed
    self_supporting: bool = False
    print_direction: str = "+z"  # build direction; overhangs vs the bed
    # optimizer backend: "oc" (optimality criteria + bisection, volume
    # constraint only) or "mma" (Svanberg MMA via nlopt — enables real
    # constraints: minmax load cases, displacement and stress limits)
    solver: str = "oc"
    case_objective: str = "weighted"  # weighted | minmax (minmax needs MMA)
    max_displacement_mm: float | None = None  # MMA: cap max nodal displacement
    max_stress_mpa: float | None = None  # MMA: P-norm von-Mises constraint
    stress_pnorm: int = 8  # P-norm exponent (P=8 balances locality vs smoothness)
    stress_relax_q: float = 0.5  # qp-relaxation exponent (Le et al. 2010)
    # penal continuation: ramp the SIMP exponent from `penal_init` to `penal`
    # over `penal_ramp_iters` — softer start escapes grayscale local minima
    penal_init: float | None = None  # None = no continuation
    penal_ramp_iters: int = 30
    # volfrac continuation: ramp the volume target from `volfrac_init` down
    # to `volfrac` over `volfrac_ramp_iters` — starting material-rich keeps
    # the first iterations out of the sparse-layout local minima; OC only
    # (an MMA inequality bound must hold from iteration 0)
    volfrac_init: float | None = None  # None = constant target
    volfrac_ramp_iters: int = 20
    # sparse-direct / iterative backend hint: "auto" lets solver.py pick
    # (CHOLMOD > SuperLU; AMG > SSOR above the DOF cap); anything else is a
    # forced choice for benchmarking
    solver_backend: str = "auto"
    # BESO evolutionary ratio (solver="beso" only): volume-ramp step per
    # iteration — V_{k+1} = max(volfrac, V_k*(1-ER)); 0.02 is the classic
    # Huang & Xie setting
    beso_er: float = 0.02
    # beso connectivity guard: re-bond the kept set to the passive skeleton
    # every iteration — a floating load island makes F·u explode and poisons
    # the sensitivity ranking. Debug opt-out for diagnosis only.
    beso_connectivity: bool = True
    # MMA quality gates (solver="mma"): minimum FEM evaluations before the
    # nlopt xtol may end the run — a warm-started gray field otherwise
    # "converges" in ~2 evals; optional blueprint finalization at
    # mma_final_beta returns the eta-projected density (crisper field)
    mma_min_evals: int = 8
    mma_final_beta: float | None = None
    # loadpath warm-start corridor width (mm) — None keeps the field's own
    # default (2.5 x pitch); wider corridors seed thicker junctions
    loadpath_corridor_mm: float | None = None
    # orthotropic FDM stiffness: (k, axis) scales the element constitutive
    # matrix's layer-normal response (see solver._D) — the optimizer then
    # grows members that compensate for the weaker build-axis direction
    aniso: tuple[float, float] | None = None
    # robust formulation (Sigmund 2009 / Wang et al. 2011): the blueprint is
    # evaluated under three Heaviside thresholds — eroded (eta=0.5+delta),
    # intermediate (0.5) and dilated (0.5-delta) — and the optimizer works on
    # the WORST compliance (subgradient of the argmax design) while the volume
    # constraint acts on the dilated realization. Threshold-insensitive,
    # print-tolerant designs with a minimum length scale in BOTH phases.
    # Requires the Heaviside projection — filter is forced to "heaviside"
    # when robust is on. Cost: ~3x the FEM solves per iteration.
    robust: bool = False
    robust_delta_eta: float = 0.25  # eta half-width of the eroded/dilated interval
    robust_designs: Literal["edi", "e"] = "edi"  # "e" = eroded only (Fernández)
    # mirror symmetry: after every design update (OC and MMA) the blueprint
    # density is averaged with its reflection in the named domain mid-plane
    # ("xy" | "xz" | "yz") — the extracted part comes out symmetric. Only
    # the density is symmetrised, never the sensitivities; passive cells
    # are re-pinned after averaging, so asymmetric BCs still produce an
    # (acceptably) asymmetric physics response.
    symmetry: str | None = None


@dataclass
class TopOptResult:
    density: np.ndarray  # (ncells,) final densities in flat cell order
    iterations: int
    converged: bool
    history: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passive_frac: float = 0.0  # fraction of solid pinned by keep/load/fixed cells
    # how the run ended — honest outcome vocabulary for the agent surface:
    #   "converged"          loop settled on a valid optimum
    #   "infeasible_target"  settled at the passive floor above the requested
    #                        volfrac — the target is unreachable by construction
    #   "limit_cycle"        density field oscillated (move-cap streak, no
    #                        compliance progress) — not a converged design
    #   "degenerate"         solver divergence/exception mid-run
    #   "budget_exhausted"   iteration budget spent without settling
    status: str = "converged"
    blueprint: np.ndarray | None = (
        None  # (ncells,) pre-projection design field — needed to re-evaluate
        # the design under shifted Heaviside thresholds (robust formulation
        # analysis); None for early-exit paths with no optimized field
    )


_AXES = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}
_ETA = 0.5  # Heaviside projection threshold
# symmetry mid-plane -> the grid axis the mirror operation flips
_SYM_AXES = {"yz": 0, "xz": 1, "xy": 2}


def _heaviside(x: np.ndarray, beta: float, eta: float = _ETA) -> np.ndarray:
    """Smoothed Heaviside projection (Wang et al.): 1 at x>eta, ~x at beta->0."""
    den = math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta))
    return (math.tanh(beta * eta) + np.tanh(beta * (x - eta))) / den


def _heaviside_d(x: np.ndarray, beta: float, eta: float = _ETA) -> np.ndarray:
    den = math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (x - eta)) ** 2) / den


def _symmetrize_density(
    x: np.ndarray,
    dims: tuple[int, int, int],
    axis: int,
    solid: np.ndarray,
    passive_solid: np.ndarray,
    pvoid: np.ndarray,
) -> np.ndarray:
    """Mirror-average the flat blueprint density across the domain mid-plane
    perpendicular to `axis`: x <- 0.5*(x + flip(x)).

    np.flip mirrors through the voxel centre of the grid and is odd/even
    safe. The average runs over ALL cells, then passive cells are re-pinned
    to their fixed values — the symmetrisation must never unpin
    keep/load/fixed material or forbidden void regions."""
    g = x.reshape(dims)
    xs = (0.5 * (g + np.flip(g, axis=axis))).ravel()
    xs[~solid] = RHO_MIN
    xs[passive_solid] = 1.0
    xs[pvoid] = RHO_MIN
    return xs


def _mirror_index_map(dims: tuple[int, int, int], axis: int, solid: np.ndarray) -> np.ndarray:
    """flat cell index -> flat index of its mirror across the domain
    mid-plane perpendicular to `axis` (same reflection as np.flip —
    odd/even safe, mid-plane cells map to themselves); -1 where the
    mirrored cell lies outside `solid`."""
    grid = [g.ravel() for g in np.indices(dims)]
    grid[axis] = dims[axis] - 1 - grid[axis]
    mflat = np.ravel_multi_index(tuple(grid), dims)
    return np.where(np.asarray(solid, dtype=bool).ravel()[mflat], mflat, -1)


def _sym_grad(d: np.ndarray, mirror_idx: np.ndarray, mirror_free: np.ndarray) -> np.ndarray:
    """Adjoint of the mirror average S = 0.5*(I + F) that _symmetrize_density
    applies to the blueprint (with its re-pinning):
        d obj/d v_c = 0.5*(d[c] + 1[mirror(c) free] * d[mirror(c)])
    A cell whose mirror is itself a free design cell picks up half the
    mirror's sensitivity; a pinned or non-solid mirror is clamped by the
    re-pin (its raw value is constant) so it contributes nothing and the
    own term stays 0.5*d[c]. `mirror_free` marks cells with a free mirror."""
    mc = np.clip(mirror_idx, 0, None)
    return 0.5 * (d + np.where(mirror_free, d[mc], 0.0))


def _am_setup(domain: VoxelDomain, direction: str) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Bottom-up layers for the additive-manufacturing support filter.

    Each layer entry is (flat cell ids, support ids (m,5) with -1 = out of
    grid). The 45° cone: a cell is supported by the 5 cells one layer down —
    straight below + the 4 lateral neighbours. `direction` is the build
    direction ('+z' → bed at the -z side of the domain).
    """
    ax = _AXES[direction]
    au = (ax + 1) % 3
    av = (ax + 2) % 3
    dims = domain.dims
    solid = domain.solid.ravel()
    n = domain.cell_count
    # flat id grid with -1 padding so out-of-grid lookups stay cheap
    ids = np.full(tuple(d + 2 for d in dims), -1, dtype=np.int64)
    core = tuple(slice(1, d + 1) for d in dims)
    ids[core] = np.arange(n, dtype=np.int64).reshape(dims)
    # layer coordinate along the build axis; builds bottom-up along `direction`
    coord = np.indices(dims)[ax]
    if direction[0] == "-":
        coord = dims[ax] - 1 - coord
    flat = ids[core].ravel()
    layers: list[tuple[np.ndarray, np.ndarray | None]] = []
    for lv in range(dims[ax]):
        cells = np.flatnonzero((coord.ravel() == lv) & solid)
        if not cells.size:
            continue
        if lv == 0:
            layers.append((flat[cells], None))  # on the build plate — free
            continue
        cells = flat[cells]
        g = np.unravel_index(cells, dims)
        # the five support positions one layer down: centre + lateral cross.
        # `ids` is padded by 1 per side, so grid coord c indexes as c + 1;
        # the layer below is g[ax] - 1 for "+dir" and g[ax] + 1 for "-dir".
        below = g[ax] + (1 if direction[0] == "-" else -1) + 1
        offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        sup = np.empty((cells.size, len(offs)), dtype=np.int64)
        for si, (du, dv) in enumerate(offs):
            pos = [below, below, below]
            pos[au] = g[au] + du + 1
            pos[av] = g[av] + dv + 1
            sup[:, si] = ids[tuple(pos)]
        layers.append((cells, sup))
    return layers


def _am_forward(
    x: np.ndarray,
    solid: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray | None]],
) -> np.ndarray:
    """Langelaar AM filter: printed density p_e = min(x_e, mean(p_supports)).
    Air cells never support (p=0 there); the bed layer prints freely."""
    p = np.zeros_like(x)
    for cells, sup in layers:
        if sup is None:
            p[cells] = x[cells]
            continue
        vals = np.where(sup >= 0, p[np.clip(sup, 0, None)], 0.0)
        # air cells carry no material — cannot support
        vals = np.where((sup >= 0) & solid[np.clip(sup, 0, None)], vals, 0.0)
        m = vals.sum(axis=1) / sup.shape[1]
        p[cells] = np.minimum(x[cells], m)
    return p


def _am_backward(
    dp: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    solid: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray | None]],
) -> np.ndarray:
    """Chain-rule gradient dC/dx for the AM filter — top-down accumulation:
    where the min() picked the cell's own blueprint the gradient stays local;
    where it picked the support mean it is distributed to the support cells."""
    g = dp.copy()
    gx = np.zeros_like(x)
    for cells, sup in reversed(layers):
        if sup is None:
            gx[cells] += g[cells]
            continue
        vals = np.where(sup >= 0, p[np.clip(sup, 0, None)], 0.0)
        vals = np.where((sup >= 0) & solid[np.clip(sup, 0, None)], vals, 0.0)
        m = vals.sum(axis=1) / sup.shape[1]
        own = x[cells] <= m
        gx[cells] += np.where(own, g[cells], 0.0)
        share = np.where(own, 0.0, g[cells] / sup.shape[1])
        ok = sup >= 0
        share5 = np.broadcast_to(share[:, None], sup.shape)
        np.add.at(g, sup[ok], share5[ok])
    return gx


def _filter_matrix(domain: VoxelDomain, rmin_cells: float) -> sparse.csr_matrix:
    """Cone filter over solid cell centers: H[i,j] = max(0, rmin - dist)."""
    centers = domain.element_centers()[domain.solid.ravel()]
    tree = cKDTree(centers)
    pairs = tree.query_pairs(rmin_cells, output_type="ndarray")
    ns = len(centers)
    if len(pairs):
        d = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
        w = rmin_cells - d
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        vals = np.concatenate([w, w])
    else:
        rows = cols = np.empty(0, dtype=np.int64)
        vals = np.empty(0)
    rows = np.concatenate([rows, np.arange(ns)])
    cols = np.concatenate([cols, np.arange(ns)])
    vals = np.concatenate([vals, np.full(ns, rmin_cells)])
    return sparse.coo_matrix((vals, (rows, cols)), shape=(ns, ns)).tocsr()


def _helmholtz_operator(domain: VoxelDomain, r_cells: float) -> sparse.csr_matrix:
    """Helmholtz PDE-filter operator  L = I - r²·Δ  on the solid cells
    (Lazarov & Sigmund 2011, doi:10.1002/nme.3072).

    Δ is the graph Laplacian of the solid mask on the regular voxel grid in
    CELL units (the pitch is already folded into `r_cells`):
        Δx_e = Σ_{n ∈ N(e)} (x_n - x_e)     with N(e) the face neighbours
    Restricted to existing SOLID neighbours — the homogeneous Neumann
    boundary condition (∂x̃/∂n = 0, natural BC of the weak form) drops every
    term whose neighbour lies outside the part or outside the grid entirely.
    Equivalently, a ghost-cell reflection x_ghost = x_e would make
    (x_ghost - x_e) ≡ 0. Consequence: the diagonal counts the number of
    EXISTING solid neighbours (1…6), never the full 6, and L·1 = 1 is exact
    everywhere including the boundary — the filter preserves constant fields
    and keeps a consistent effective length scale right up to the domain
    edge, which is exactly where the clipped cone filter misbehaves.

    L is symmetric (-r² on both sides of every solid-solid edge) and
    strictly diagonally dominant (diag - |off-row sum| = 1) → SPD, a single
    direct factorization serves every solve.
    """
    solid_idx = np.flatnonzero(domain.solid.ravel())
    ns = solid_idx.size
    dims = domain.dims
    pos = np.full(domain.cell_count, -1, dtype=np.int64)
    pos[solid_idx] = np.arange(ns, dtype=np.int64)
    gi, gj, gk = np.unravel_index(solid_idx, dims)
    grid = (gi, gj, gk)
    r2 = float(r_cells) ** 2
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    n_nb = np.zeros(ns, dtype=np.int64)
    for ax in range(3):
        for step in (-1, 1):
            nb = [grid[0], grid[1], grid[2]]
            nb[ax] = nb[ax] + step  # rebinds a NEW array — grid stays intact
            inside = np.ones(ns, dtype=bool)
            for a in range(3):
                inside &= (nb[a] >= 0) & (nb[a] < dims[a])
            # clipped flat index is only read where `inside` — the clip keeps
            # raveling in-range instead of branching
            nflat = np.ravel_multi_index(
                tuple(np.clip(nb[a], 0, dims[a] - 1) for a in range(3)), dims
            )
            tgt = pos[nflat]  # -1 for void / non-solid neighbours
            ok = inside & (tgt >= 0)
            rows.append(np.nonzero(ok)[0])
            cols.append(tgt[ok])
            n_nb += ok
    diag = 1.0 + r2 * n_nb.astype(float)
    row_idx = np.arange(ns, dtype=np.int64)
    vals = np.concatenate([-np.full(sum(r.size for r in rows), r2), diag])
    return sparse.coo_matrix(
        (vals, (np.concatenate([*rows, row_idx]), np.concatenate([*cols, row_idx]))),
        shape=(ns, ns),
    ).tocsr()


@dataclass
class _FilterOps:
    """The linear density-filter operator on the solid cells.

    kind "cone"      — normalized cone filter:  apply x → (H x) / Hs
    kind "helmholtz" — PDE filter:              apply x → L⁻¹ x
                       (L = I - r²Δ, SPD, one factorization per simp run)

    Both operators are symmetric → the adjoint (sensitivity chain rule) is a
    second application of the same filter. `stats["solve_s"]` accumulates
    helmholtz solve time for the perf log on large domains; `stats["filter_s"]`
    accumulates all apply/adjoint/oc time — simp() snapshots it into the
    per-iteration history as `filter_s`.
    """

    kind: str  # "cone" | "helmholtz"
    solid_idx: np.ndarray
    H: sparse.csr_matrix | None = None
    Hs: np.ndarray | None = None
    solve: Callable[[np.ndarray], np.ndarray] | None = None
    stats: dict[str, float] = field(default_factory=dict)

    def apply(self, xs: np.ndarray) -> np.ndarray:
        """Filtered field x̃ of blueprint values xs (solid_idx order)."""
        t = time.monotonic()
        if self.kind == "helmholtz":
            assert self.solve is not None
            out = np.asarray(self.solve(np.asarray(xs, dtype=float)))
        else:
            assert self.H is not None and self.Hs is not None
            out = np.asarray(self.H @ xs).ravel() / self.Hs
        self.stats["filter_s"] = self.stats.get("filter_s", 0.0) + time.monotonic() - t
        return out

    def adjoint(self, d: np.ndarray) -> np.ndarray:
        """dC/dx̃ → dC/dx — both operators are symmetric (self-adjoint)."""
        t = time.monotonic()
        if self.kind == "helmholtz":
            assert self.solve is not None
            out = np.asarray(self.solve(np.asarray(d, dtype=float)))
        else:
            assert self.H is not None and self.Hs is not None
            out = np.asarray(self.H @ (d / self.Hs)).ravel()
        self.stats["filter_s"] = self.stats.get("filter_s", 0.0) + time.monotonic() - t
        return out

    def oc(self, xb: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Classic OC sensitivity heuristic (top88-style length-scale
        regularization — a heuristic, NOT the true gradient):
            cone:      H(x·d) / (x·Hs)      — Hs re-normalizes the product
            helmholtz: L⁻¹(x·d) / x         — L⁻¹ is already normalized
                                             (L·1 = 1 → L⁻¹·1 = 1), so the
                                             Hs division has no analog.
        `xb`/`d` are (nsolid,) blueprint values / raw sensitivities."""
        t = time.monotonic()
        xb = np.asarray(xb, dtype=float)
        xd = xb * np.asarray(d, dtype=float)
        if self.kind == "helmholtz":
            assert self.solve is not None
            out = np.asarray(self.solve(xd)) / np.clip(xb, 1e-3, None)
        else:
            assert self.H is not None and self.Hs is not None
            out = np.asarray(self.H @ xd).ravel() / np.clip(xb * self.Hs, 1e-3, None)
        self.stats["filter_s"] = self.stats.get("filter_s", 0.0) + time.monotonic() - t
        return out


def _build_filter_ops(
    domain: VoxelDomain,
    solid_idx: np.ndarray,
    rmin_cells: float,
    kind: str,  # "cone" | "helmholtz"
) -> _FilterOps:
    """Construct the density-filter operator — cone matrix or a factored
    Helmholtz system (the factorization happens ONCE here per simp() call).

    `rmin_cells` is the CONE-equivalent radius; Lazarov & Sigmund's PDE
    parameter r relates to the classic filter length by  rmin ≈ 2√3·r
    (matching the kernels' second moments), so the same rmin_mm spec gives
    comparable member sizes under both filters.
    """
    if kind == "cone":
        H = _filter_matrix(domain, rmin_cells)
        Hs = np.asarray(H.sum(axis=1)).ravel()
        Hs[Hs <= 0] = 1.0
        return _FilterOps(kind="cone", solid_idx=solid_idx, H=H, Hs=Hs)
    r_cells = rmin_cells / (2.0 * math.sqrt(3.0))
    L = _helmholtz_operator(domain, r_cells)
    stats: dict[str, float] = {"solve_s": 0.0}
    big = domain.cell_count > 200_000
    t_fac = time.monotonic()
    factor = None
    with contextlib.suppress(Exception):  # spsolve fallback below
        factor = spla.factorized(L.tocsc())
    log.debug(
        "topopt.helmholtz_operator",
        nsolid=int(solid_idx.size),
        r_cells=round(r_cells, 4),
        nnz=int(L.nnz),
        factorized=factor is not None,
        factor_s=round(time.monotonic() - t_fac, 3),
    )
    Lc = L.tocsc()

    def solve(b: np.ndarray) -> np.ndarray:
        t = time.monotonic()
        out = factor(b) if factor is not None else spla.spsolve(Lc, b)
        dt = time.monotonic() - t
        stats["solve_s"] += dt
        if big:
            log.debug("topopt.helmholtz_solve", nsolid=int(solid_idx.size), ms=round(dt * 1e3, 1))
        return np.asarray(out, dtype=float)

    return _FilterOps(kind="helmholtz", solid_idx=solid_idx, solve=solve, stats=stats)


class _Projector:
    """blueprint density → physical printed density chain.

    Bundles the state simp()'s former _project/_chain closures captured so
    the very same code path can be unit-tested (finite-difference gradient
    checks run against this class). `beta` is mutable — the Heaviside
    continuation writes it mid-loop via set_beta.
    """

    def __init__(
        self,
        *,
        fops: _FilterOps,
        solid: np.ndarray,  # (ncells,) bool flat solid mask
        solid_idx: np.ndarray,
        passive_solid: np.ndarray,  # (ncells,) bool — pinned rho=1
        pvoid: np.ndarray,  # (ncells,) bool — pinned rho=RHO_MIN
        am_layers: list[tuple[np.ndarray, np.ndarray | None]] | None,
        heaviside: bool,
        beta: float,
    ) -> None:
        self.fops = fops
        self.solid = solid
        self.solid_idx = solid_idx
        self.passive_solid = passive_solid
        self.pvoid = pvoid
        self.am_layers = am_layers
        self.heaviside = heaviside
        self.beta = beta

    def project(
        self, xb: np.ndarray, eta: float = _ETA
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Blueprint -> (xp, p, xt) at threshold `eta` — the filter chain
        without the FEM solve. xp is the post-Heaviside density, p the
        post-AM physical density, xt the filtered field (None without
        Heaviside). Used by the MMA volume constraint on the dilated
        realization, which needs no FEM pass."""
        xp = xb.copy()
        xt: np.ndarray | None = None
        if self.heaviside:
            xt = self.fops.apply(xb[self.solid_idx])
            xp[self.solid_idx] = _heaviside(xt, self.beta, eta)
            xp[self.passive_solid] = 1.0
            xp[self.pvoid] = RHO_MIN
        p = _am_forward(xp, self.solid, self.am_layers) if self.am_layers is not None else xp
        return xp, p, xt

    def chain(
        self,
        d_phys: np.ndarray,
        xp: np.ndarray,
        p: np.ndarray,
        xt: np.ndarray | None,
        xb: np.ndarray,
        oc_filter: bool,
        eta: float = _ETA,
    ) -> np.ndarray:
        """sensitivity wrt physical density → wrt blueprint x, back through
        the AM and Heaviside/density filters. `oc_filter` selects the
        classic OC sensitivity normalization on the unprojected path
        (heuristic length-scale regularization — NOT the true gradient;
        constraints pass oc_filter=False). `eta` must match the projection
        the (xp, p, xt) triple was computed with."""
        if self.am_layers is not None:
            d_phys = _am_backward(d_phys, xp, p, self.solid, self.am_layers)
        if self.heaviside:
            # p = heaviside(fops(x)): dx = fopsᵀ(dc · dh/dx̃), fops symmetric
            assert xt is not None
            out = np.zeros(xb.size)
            dxt = d_phys[self.solid_idx] * _heaviside_d(xt, self.beta, eta)
            out[self.solid_idx] = self.fops.adjoint(dxt)
            return out
        out = d_phys.copy()
        if oc_filter:
            out[self.solid_idx] = self.fops.oc(xb[self.solid_idx], out[self.solid_idx])
        return out


def _prepare_solve(
    Kf: sparse.csr_matrix,
    spsolve_dofs: int,
    solver_backend: str = "auto",
    *,
    domain: VoxelDomain | None = None,
    free: np.ndarray | None = None,
) -> tuple[Any, Any]:
    """Per-iteration solver setup for one reduced system.

    Returns (factor, M): `factor` is a direct-solve callable when the system
    is small enough for a direct solve (CHOLMOD when the optional
    scikit-sparse dep is present, else SuperLU in SymmetricMode), else None;
    `M` is the iterative preconditioner for CG — smoothed-aggregation AMG
    when pyamg is importable and `domain`+`free` supply the node coords the
    near-nullspace needs, else SSOR/diagonal. `solver_backend` forces a
    backend for benchmarking; "auto" picks the best available."""
    from cadx.cad.fem.solver import factor_symmetric, iterative_preconditioner

    if Kf.shape[0] <= spsolve_dofs:
        factor = factor_symmetric(Kf, backend=solver_backend)
        if factor is not None:
            return factor, None
    M = iterative_preconditioner(Kf, backend=solver_backend, domain=domain, free=free)
    if M is None:
        diag = Kf.diagonal()
        diag[diag == 0] = 1.0
        M = spla.diags(1.0 / diag)
    return None, M


# a free design cell counts as "pinned at the floor" when its blueprint
# density sits at ~rho_min — the volume-constraint bisection multiplies by
# sqrt(-dc/lam) and clips at the move limit, so a drained field lands at
# <= a few percent, never exactly 0
_FLOOR_PIN = 0.05
# O2: consecutive iterations with change saturating >= this fraction of the
# OC move limit count as a limit-cycle streak (early detection runs on a
# looser saturation than the 0.95 backstop streak)
_MOVE_SAT_FRAC = 0.9
_MOVE_SAT_STREAK = 10  # iterations
_MOVE_SAT_WINDOW = 10  # compliance-improvement window, in iterations
_FLIP_FRAC = 0.005  # share of design cells flipping membership per iteration —
# the same 0.5% the `settled` gate below uses; sustained churn over a full
# window is what "density oscillation persists" means
_BETA_DWELL = 10  # max iterations spent on one Heaviside sharpness level —
# the stall trigger (rel < tol) can starve when the objective jitters inside
# the move cap (the robust worst-case realization oscillates a few percent on
# a stable mask), which would freeze the ramp mid-way. One full window per
# level guarantees the ramp reaches beta_max while budget remains to sharpen.


def _windowed_compliance_improvement(
    history: list[dict[str, Any]], window: int
) -> float | None:
    """Relative compliance improvement, (c_old - c_now)/c_old, over the last
    `window` history records AT THE CURRENT beta. A Heaviside continuation
    step legitimately spikes the objective (the sharper projection is a
    weaker structure) then re-decays — comparing across a beta change
    misreads that transient as 'no improvement'. Restricting the window to
    the current-beta segment also skips the not-yet-full post-ramp stretch.
    Returns None when the segment isn't full or the baseline is
    non-positive — the caller treats that as 'unknown', not as stall."""
    last_beta = history[-1].get("beta") if history else None
    seg = [h for h in history if h.get("beta") == last_beta]
    if seg and seg[0] is not history[0]:
        # the first record of a new beta segment was still EVALUATED under
        # the previous projection (history stores the pre-update objective
        # with the post-update beta) — its stale objective would masquerade
        # as a regression on the next window's baseline
        seg = seg[1:]
    if len(seg) < window:
        return None
    old = float(seg[-window].get("compliance_mj") or 0.0)
    new = float(seg[-1].get("compliance_mj") or 0.0)
    if old <= 0.0:
        return None
    return (old - new) / old


@dataclass
class _CaseCtx:
    """Per-load-case reduced system for one `_eval` — the case's own free
    DOFs (fixed DOFs are per-case, never unioned across cases), its reduced
    stiffness Kf and the solver handles for the primal and adjoint solves."""

    free: np.ndarray
    Kf: sparse.csr_matrix
    factor: Any
    M: Any


def _solve_u(
    domain: VoxelDomain,
    Kf: sparse.csr_matrix,
    F: np.ndarray,
    free: np.ndarray,
    u0: np.ndarray | None,
    factor: Any = None,
    M: Any = None,
    cg_rtol: float = 1e-5,
    solver_backend: str = "auto",
) -> tuple[np.ndarray, str]:
    """Light solve: return full u vector."""
    u = np.zeros(3 * domain.node_count)
    if free.size == 0:
        return u, "none"
    if factor is not None:
        u[free] = factor(F[free])
        return u, "spsolve"
    if M is None:
        _, M = _prepare_solve(
            Kf,
            spsolve_dofs=0,
            solver_backend=solver_backend,
            domain=domain,
            free=free,
        )
    x, info = spla.cg(
        Kf,
        F[free],
        x0=u0[free] if u0 is not None else None,
        rtol=cg_rtol,
        atol=0.0,
        maxiter=4000,
        M=M,
    )
    u[free] = x
    if info > 0:
        log.warning("topopt.cg_no_converge", info=info)
    return u, "cg"


def simp(  # noqa: C901
    domain: VoxelDomain,
    elem_E0: float | np.ndarray,
    nu: float,
    cases: list[tuple[np.ndarray, np.ndarray]],  # (fixed_mask (nn,) or (nn,3), F) per case
    passive_solid: np.ndarray,  # (ncells,) bool — pinned rho=1
    params: TopOptParams,
    *,
    passive_void: np.ndarray | None = None,  # (ncells,) bool — pinned rho=rho_min
    case_weights: np.ndarray | list[float] | None = None,  # objective weight per case
    x0: np.ndarray | None = None,  # (ncells,) warm-start density field
    spsolve_dofs: int = 120_000,  # direct-solver cutoff; larger systems use SSOR-CG
    progress_cb: ProgressCb | None = None,
    cancelled: CancelCb | None = None,
    progress_file: Path | None = None,
) -> TopOptResult:
    """Run the SIMP loop. Returns the density field (flat cell order).

    `elem_E0` may be a scalar or a (ncells,) array for material zones (e.g.
    metal inserts inside a polymer part). `passive_void` pins cells to the
    void floor inside the part — forbidden material regions. `case_weights`
    scales each load case's compliance contribution (weighted multi-case).
    `x0` warm-starts the density field (e.g. a previous run's density)."""
    from cadx.cad.fem.solver import _fixed_dofs

    warnings: list[str] = []
    solid = domain.solid.ravel()
    pvoid = (
        np.zeros(domain.cell_count, dtype=bool)
        if passive_void is None
        else np.asarray(passive_void, dtype=bool).ravel() & solid
    )
    design = solid & ~passive_solid & ~pvoid
    nsolid = int(solid.sum())
    ndesign = int(design.sum())
    if ndesign == 0:
        return TopOptResult(
            density=solid.astype(float),
            iterations=0,
            converged=False,
            status="infeasible_target",  # passive floor is the whole domain
            warnings=["no free design cells — everything is kept/pinned"],
        )

    E0 = np.broadcast_to(np.asarray(elem_E0, dtype=float), domain.cell_count).copy()
    Emin = E0 * E_MIN_FRAC
    if x0 is not None and len(x0) == domain.cell_count:
        x = np.clip(np.asarray(x0, dtype=float), RHO_MIN, 1.0)
        x[~solid] = RHO_MIN
    else:
        x = np.where(solid, params.volfrac, RHO_MIN)
        # the volume budget counts passive cells pinned at 1.0 — a uniform
        # volfrac init already sits passive_frac*(1-vf) over budget. OC
        # rescales itself to v_target on iteration 1 and does not care, but
        # MMA is a local method: observed a 0.4-passive run converging on
        # the infeasible init (final volfrac 0.67 for a 0.45 target). Start
        # the free design cells at the actually-feasible mean instead.
        free_frac = (params.volfrac * nsolid - float(passive_solid.sum())) / ndesign
        x[design] = float(np.clip(free_frac, RHO_MIN, 1.0))
    x[passive_solid] = 1.0
    x[pvoid] = RHO_MIN
    x = x.astype(float)
    # mirror symmetry ("xy"/"xz"/"yz"): symmetrise the blueprint right
    # after every design update — the OC assignment below and _unpack
    # inside _run_mma — so the density filter only ever sees an already
    # mirrored-average field and the filtered/projected design stays
    # symmetric too. Sensitivities are deliberately NOT symmetrised.
    sym_axis: int | None = _SYM_AXES.get(params.symmetry) if params.symmetry else None
    if params.symmetry and sym_axis is None:
        warnings.append(f"unknown symmetry plane '{params.symmetry}' — ignored")
    elif sym_axis is not None:
        # domain sanity: mirroring density across a plane the solid itself is
        # not symmetric about fights the load paths — observed: an L-bracket
        # (bores at x=±30, wall at -Y, load at +Y front edge — X-symmetric,
        # not Y-symmetric) run with symmetry="xz" stalled into a 0.33-volfrac
        # skeleton at SF 0.04. Tell the caller the geometry doesn't support
        # the chosen plane instead of silently distorting the design.
        vox = domain.solid
        mismatch = float(np.mean(vox != np.flip(vox, axis=sym_axis)))
        if mismatch > 0.15:
            plane_names = {0: "yz (X-flip)", 1: "xz (Y-flip)", 2: "xy (Z-flip)"}
            warnings.append(
                f"symmetry '{params.symmetry}' mirror-averages density across the "
                f"{plane_names[sym_axis]} mid-plane, but {mismatch:.0%} of solid "
                "voxels are not mirrored — the part is not symmetric about this "
                "plane; the constraint distorts the load paths"
            )

    def _symmetrize(xb: np.ndarray) -> np.ndarray:
        assert sym_axis is not None
        return _symmetrize_density(xb, domain.dims, sym_axis, solid, passive_solid, pvoid)

    if sym_axis is not None:
        x = _symmetrize(x)  # warm-started fields may arrive asymmetric
    weights = (
        np.ones(len(cases))
        if case_weights is None
        else np.asarray(case_weights, dtype=float)[: len(cases)]
    )

    ke = (
        _ke_unit(nu, domain.pitch)
        if params.aniso is None
        else _ke_unit(nu, domain.pitch, float(params.aniso[0]), int(params.aniso[1]))
    )
    en = domain.element_nodes()
    edof = element_dofs(en)
    # SIMP assembles ALL cells each iteration — the sparsity pattern is
    # constant, so build the COO index arrays once and reuse them
    ij_cache = (np.repeat(edof, 24, axis=1).ravel(), np.tile(edof, (1, 24)).ravel())

    # density filter over the solid cells — "cone" (normalized distance
    # matrix H x / Hs) or "helmholtz"/"helmholtz+heaviside" (PDE filter
    # x̃ = L⁻¹x, L = I - r²Δ with Neumann BC — consistent filter length at
    # the domain boundary, sparse SPD solve instead of the cone matrix)
    rmin_cells = max(1.2, params.rmin_mm / domain.pitch)
    solid_idx = np.flatnonzero(solid)
    pde_filter = params.filter in ("helmholtz", "helmholtz+heaviside")
    fops = _build_filter_ops(domain, solid_idx, rmin_cells, "helmholtz" if pde_filter else "cone")

    # M3.1 Heaviside projection + M2 AM support filter — the "+heaviside"
    # variants run the smoothed projection on the FILTERED field; the plain
    # filters keep the unprojected sensitivity-filter interpretation
    heaviside = params.filter in ("heaviside", "helmholtz+heaviside")
    robust = bool(params.robust)
    if robust and not heaviside:
        # robust evaluates shifted projection thresholds — it needs a
        # Heaviside-bearing filter stage; keep the PDE stage if one was set
        target = "helmholtz+heaviside" if pde_filter else "heaviside"
        warnings.append(
            f"robust formulation needs the Heaviside projection — filter forced to '{target}'"
        )
        heaviside = True
    eta_d = float(np.clip(params.robust_delta_eta, 0.0, 0.45)) if robust else 0.0
    # evaluation thresholds, eroded (hardest) first — the argmax/subgradient
    # picks the worst design; the dilated eta (0.5-d) drives the volume
    # constraint. "e" evaluates the eroded design only (Fernández variant).
    etas: tuple[float, ...] = (_ETA,)
    if robust:
        etas = (
            (_ETA + eta_d,) if params.robust_designs == "e" else (_ETA + eta_d, _ETA, _ETA - eta_d)
        )
    eta_vol = _ETA - eta_d  # volume constraint always on the dilated design
    am_layers = _am_setup(domain, params.print_direction) if params.self_supporting else None
    if am_layers is not None and not am_layers:
        warnings.append("self_supporting: no solid cells found — AM filter inactive")
        am_layers = None

    proj = _Projector(
        fops=fops,
        solid=solid,
        solid_idx=solid_idx,
        passive_solid=passive_solid,
        pvoid=pvoid,
        am_layers=am_layers,
        heaviside=heaviside,
        beta=params.beta0,
    )
    _project = proj.project
    _chain = proj.chain

    def _physical(xb: np.ndarray, eta: float = _ETA) -> np.ndarray:
        """blueprint density -> physical printed density (flat cell order)."""
        return _project(xb, eta)[1]

    # free DOFs are PER CASE: K is identical across cases (E_e depends on
    # the density, not the BCs), but each case must solve on
    # `used - its own fixed DOFs`. Unioning every case's fixed DOFs into one
    # shared `free` pins case i at the OTHER cases' fixtures — an
    # over-constrained system with under-predicted compliance and wrong
    # sensitivities (multi-fixture runs, e.g. cantilever + 3-point bend).
    # Single-fixture runs reduce to the same `free` as the union, unchanged.
    used_dofs = np.unique(edof.ravel())
    case_free: list[np.ndarray] = []
    for f, _ in cases:
        fd = np.unique(_fixed_dofs(f, domain.node_count))
        case_free.append(used_dofs[~np.isin(used_dofs, fd, assume_unique=True)])
    if all(fi.size == 0 for fi in case_free):
        return TopOptResult(
            density=x,
            iterations=0,
            converged=False,
            status="degenerate",
            warnings=["no free DOFs — the whole domain is constrained"],
        )
    for ci, fi in enumerate(case_free):
        if fi.size == 0:
            warnings.append(
                f"load case {ci} has no free DOFs — fully constrained, contributes zero compliance"
            )

    v_target = params.volfrac * nsolid
    history: list[dict[str, Any]] = []
    u_prev: list[np.ndarray | None] = [None] * len(cases)
    change = 1.0
    change_box = [change]  # shared with _run_mma: cg_rtol reads the live value
    it = 0
    stall = 0
    stop: str | None = None  # how the loop ended: None | cancelled | limit_cycle | degenerate
    move_streak = 0  # consecutive iterations with change saturated at the move cap
    move_sat_streak = 0  # O2 early detection: streak at >= _MOVE_SAT_FRAC of move
    flip_streak = 0  # O2: consecutive iterations with the mask still churning
    beta_age = 0  # iterations spent on the current Heaviside sharpness level
    prev_objective: float | None = None
    prev_mask: np.ndarray | None = None  # thresholded design set, for flip count
    x_prev: np.ndarray | None = None  # blueprint before the last update — a
    # limit cycle bounces between two phases; extraction prefers the crisper
    t0 = time.monotonic()
    ncells = domain.cell_count
    it_box = [0]  # shared iteration counter (nlopt drives iterations in MMA mode)

    # per-phase wall-clock instrumentation — _eval accumulates assemble/
    # factorize/solve time, the filter ops accumulate into fops.stats; each
    # history record snapshots the deltas and resets (MMA runs share it via
    # the `phase_timings`/`phase_t` args to _run_mma)
    phase_t: dict[str, float] = {"assemble_s": 0.0, "factorize_s": 0.0, "solve_s": 0.0}
    filter_mark = [fops.stats.get("filter_s", 0.0)]

    def _phase_timings() -> dict[str, float]:
        out = {k: round(v, 3) for k, v in phase_t.items()}
        for k in phase_t:
            phase_t[k] = 0.0
        fnow = fops.stats.get("filter_s", 0.0)
        out["filter_s"] = round(fnow - filter_mark[0], 3)
        filter_mark[0] = fnow
        return out

    def _penal_now() -> float:
        """A5 penal continuation: ramp the exponent linearly from penal_init
        to penal over penal_ramp_iters iterations (None → constant)."""
        if params.penal_init is None or params.penal_init >= params.penal:
            return params.penal
        f = min(1.0, it_box[0] / max(1, params.penal_ramp_iters))
        return params.penal_init + (params.penal - params.penal_init) * f

    def _vol_now() -> float:
        """A5 volfrac continuation: ramp the volume target linearly from
        volfrac_init down to volfrac over volfrac_ramp_iters iterations.
        A material-rich start explores broad layouts before the budget
        tightens — escapes the same local minima as penal_init."""
        if params.volfrac_init is None or params.volfrac_init <= params.volfrac:
            return params.volfrac
        f = min(1.0, it_box[0] / max(1, params.volfrac_ramp_iters))
        return params.volfrac_init + (params.volfrac - params.volfrac_init) * f

    def _set_beta(b: float) -> None:
        proj.beta = b

    def _eval(xb: np.ndarray, eta: float = _ETA):
        """One FEM evaluation at blueprint density `xb` projected at `eta`.

        Returns (xp, p, xt, ce_cases, obj_cases, us, ctxs): pre-AM physical
        density, post-AM physical density, filtered density (cone or
        PDE-filtered field), per-case element strain energies, per-case
        compliances, per-case displacement vectors, and per-case solve
        contexts `_CaseCtx` (free/Kf/factor/M — cases with an identical
        fixed-DOF signature share one context)."""
        pen = _penal_now()
        xp, p, xt = _project(xb, eta)
        E_e = Emin + np.power(p, pen) * (E0 - Emin)
        t_asm = time.monotonic()
        K, _, _ = assemble(
            domain, E_e, nu, np.ones(ncells, dtype=bool), ij=ij_cache, aniso=params.aniso
        )
        phase_t["assemble_s"] += time.monotonic() - t_asm
        cg_rtol = (
            1e-3 if change_box[0] > 0.1 else (1e-4 if change_box[0] > 10 * params.tol else 1e-6)
        )
        ce_cases: list[np.ndarray] = []
        obj_cases: list[float] = []
        us: list[np.ndarray] = []
        ctxs: list[_CaseCtx] = []
        # K is identical across cases (E_e depends on the density, not the
        # BCs) — assemble once; only the per-case `free` differs. Cases with
        # an identical fixed-DOF signature (direction-cone variants share a
        # fixture) reuse one Kf extraction + factorization.
        ctx_cache: dict[bytes, _CaseCtx] = {}
        t_solve = time.monotonic()
        for ci, (_, F) in enumerate(cases):
            free_i = case_free[ci]
            ctx = ctx_cache.get(free_i.tobytes())
            if ctx is None:
                t_fac = time.monotonic()
                if free_i.size:
                    Kf_i = K[free_i][:, free_i].tocsr()
                    factor_i, M_i = _prepare_solve(
                        Kf_i,
                        spsolve_dofs,
                        params.solver_backend,
                        domain=domain,
                        free=free_i,
                    )
                else:
                    Kf_i = sparse.csr_matrix((0, 0))
                    factor_i = M_i = None
                phase_t["factorize_s"] += time.monotonic() - t_fac
                ctx = _CaseCtx(free=free_i, Kf=Kf_i, factor=factor_i, M=M_i)
                ctx_cache[free_i.tobytes()] = ctx
            ctxs.append(ctx)
            u, _solver = _solve_u(
                domain,
                ctx.Kf,
                F,
                ctx.free,
                u_prev[ci],
                factor=ctx.factor,
                M=ctx.M,
                cg_rtol=cg_rtol,
                solver_backend=params.solver_backend,
            )
            u_prev[ci] = u
            us.append(u)
            ue = u[edof]
            ce_cases.append(np.einsum("ni,ij,nj->n", ue, ke, ue))
            obj_cases.append(float(F[ctx.free] @ u[ctx.free]))
        phase_t["solve_s"] += time.monotonic() - t_solve
        return xp, p, xt, ce_cases, obj_cases, us, ctxs

    def _sens_per_case(xb: np.ndarray, ev, oc_filter: bool, eta: float = _ETA) -> list[np.ndarray]:
        """Per-case objective sensitivity wrt blueprint x for the design
        evaluated at `eta` (ev must come from _eval(xb, eta))."""
        xp, p, xt, ce_cases = ev[0], ev[1], ev[2], ev[3]
        pen = _penal_now()
        dE = -pen * np.power(np.clip(p, 1e-3, None), pen - 1) * (E0 - Emin)
        return [_chain(dE * ce_i, xp, p, xt, xb, oc_filter, eta) for ce_i in ce_cases]

    mma_x: np.ndarray | None = None
    if params.solver == "oc" and (
        params.max_displacement_mm or params.max_stress_mpa or params.case_objective == "minmax"
    ):
        warnings.append(
            "solver=oc ignores max_displacement_mm/max_stress_mpa/minmax "
            "objective — set solver=mma to enforce them"
        )
    if params.solver == "mma":
        try:
            mma_x, it, converged, change, mma_status = _run_mma(
                x,
                des=np.flatnonzero(design),
                eval_fn=_eval,
                project_fn=_project,
                chain_fn=_chain,
                sens_fn=_sens_per_case,
                etas=etas,
                eta_vol=eta_vol,
                solid_idx=solid_idx,
                weights=weights,
                params=params,
                nsolid=nsolid,
                v_target=v_target,
                nu=nu,
                domain=domain,
                edof=edof,
                ke=ke,
                E0=E0,
                Emin=Emin,
                cases=cases,
                warnings=warnings,
                history=history,
                heaviside=heaviside,
                beta0=proj.beta,
                cancelled=cancelled,
                progress_cb=progress_cb,
                progress_file=progress_file,
                t0=t0,
                penal_now=_penal_now,
                set_beta=_set_beta,
                it_box=it_box,
                passive_solid=passive_solid,
                pvoid=pvoid,
                change_box=change_box,
                symmetrize_fn=_symmetrize if sym_axis is not None else None,
                phase_t=phase_t,
                phase_timings=_phase_timings,
            )
        except ImportError:
            warnings.append("mma solver unavailable (nlopt missing) — fell back to oc")
        except Exception as e:
            # solver diverged/crashed — return the current field honestly
            # flagged instead of losing the run to an exception
            warnings.append(
                f"mma solver failed ({type(e).__name__}: {e}) — extracting the "
                "last evaluated density field"
            )
            passive_frac = float(passive_solid.sum()) / nsolid
            return TopOptResult(
                density=_physical(x),
                iterations=it_box[0],
                converged=False,
                status="degenerate",
                history=history,
                warnings=warnings,
                passive_frac=passive_frac,
                blueprint=x.copy(),
            )
    if mma_x is not None:
        warnings_out = warnings
        passive_frac = float(passive_solid.sum()) / nsolid
        density = _physical(mma_x)
        status = mma_status
        if converged:
            # settled above the volume target with every free cell drained to
            # the floor → the passive floor itself exceeds the requested
            # volfrac: the target was infeasible, not converged
            achieved = float(density[solid_idx].sum()) / nsolid
            if achieved > params.volfrac + 0.02 and float(mma_x[design].max()) <= _FLOOR_PIN:
                status = "infeasible_target"
                converged = False
        if passive_frac > params.volfrac * 1.02:
            warnings_out.append(
                f"passive (keep/load/fixed) cells cover {passive_frac:.0%} of the part — "
                f"target volfrac {params.volfrac:.0%} unreachable; floor ≈ {passive_frac:.0%}"
            )
        elif ndesign < 0.05 * nsolid:
            warnings_out.append(
                "very few free design cells (<5%) — keep regions cover most of the part"
            )
        return TopOptResult(
            density=density,
            iterations=it,
            converged=converged,
            status=status,
            history=history,
            warnings=warnings_out,
            passive_frac=passive_frac,
            blueprint=mma_x.copy(),
        )

    while it < params.max_iter and change > params.tol:
        it += 1
        it_box[0] = it
        if cancelled and cancelled():
            warnings.append(f"cancelled after iteration {it - 1}")
            stop = "cancelled"
            break

        # robust formulation: evaluate the blueprint under every eta
        # realization and let the WORST compliance drive the update
        # (subgradient of the argmax design); a single eta is the classic
        # deterministic formulation.
        try:
            evs = [_eval(x, eta=e) for e in etas]
        except Exception as e:
            # solver divergence/crash mid-iteration — the last density field
            # is still extractable, so report it flagged 'degenerate' instead
            # of propagating the exception and losing the run
            warnings.append(
                f"FEM solve failed at iteration {it} ({type(e).__name__}: {e}) — "
                "extracting the last density field"
            )
            stop = "degenerate"
            break
        obj_designs = [
            float(np.dot(weights[: len(cases)], ev[4])) if len(cases) else 0.0 for ev in evs
        ]
        wi = int(np.argmax(obj_designs))
        eta_w = etas[wi]
        xp, p, _xt, ce_cases = evs[wi][0], evs[wi][1], evs[wi][2], evs[wi][3]
        objective = obj_designs[wi]

        # sensitivity w.r.t. physical density, then chain-rule back through
        # the filters to the blueprint variable x
        dcw = np.zeros(ncells)
        for ci, ce_i in enumerate(ce_cases):
            w = float(weights[ci]) if ci < len(weights) else 1.0
            dcw += w * ce_i
        pen = _penal_now()
        dc = -pen * np.power(np.clip(p, 1e-3, None), pen - 1) * (E0 - Emin) * dcw
        dcf = _chain(dc, xp, p, _xt, x, oc_filter=True, eta=eta_w)

        # OC update with bisection on the PHYSICAL volume constraint —
        # v_now ramps down from volfrac_init when continuation is on
        v_now = _vol_now() * nsolid
        xnew = x.copy()
        l1, l2 = 0.0, 1e9
        for _ in range(60):
            lmid = 0.5 * (l1 + l2)
            cand = x * np.sqrt(np.clip(-dcf / max(lmid, 1e-12), 0.0, None))
            cand = np.clip(cand, x - params.move, x + params.move)
            cand = np.clip(cand, 0.0, 1.0)
            cand[~solid] = RHO_MIN
            cand[passive_solid] = 1.0
            cand[pvoid] = RHO_MIN
            if heaviside or am_layers is not None:
                # volume on the DILATED realization (Wang et al. 2011) —
                # eta_vol == _ETA in the non-robust case
                vol_c = _physical(cand, eta=eta_vol)[solid_idx].sum()
            else:
                vol_c = cand[solid_idx].sum()
            if vol_c > v_now:
                l1 = lmid
            else:
                l2 = lmid
            xnew = cand
            if l2 - l1 < 1e-4 * (l1 + l2 + 1e-12):
                break

        if sym_axis is not None:
            # project the update onto the symmetric subspace before the
            # next iteration's filter sees it
            xnew = _symmetrize(xnew)
        change = float(np.abs(xnew - x).max())
        change_box[0] = change
        x_prev = x
        x = xnew
        phys_now = _physical(x, eta=eta_vol)[solid_idx]
        vol_now = float(phys_now.sum()) / nsolid
        # mask churn — fraction of design cells whose 0.5-thresholded
        # membership flipped vs the previous iteration. Settles toward 0
        # under monotonic sharpening (beta ramp); a genuine limit cycle
        # keeps a persistent share of cells bouncing. Used by both the
        # stall test below and the early limit-cycle break.
        design_sel = design[solid_idx]
        if design_sel.any():
            phys_d = phys_now[design_sel]
            gray = float(np.mean((phys_d > 0.2) & (phys_d < 0.8)))
            mask_now = phys_d >= 0.5
            flips = (
                float(np.mean(mask_now != prev_mask)) if prev_mask is not None else 1.0
            )
            prev_mask = mask_now
        else:
            gray, flips = 0.0, 0.0
        rec = {
            "iteration": it,
            "compliance_mj": round(objective, 3),
            "volfrac": round(vol_now, 4),
            "change": round(change, 5),
            "flips": round(flips, 5),
            "penal": round(pen, 3),
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        if params.volfrac_init is not None and params.volfrac_init > params.volfrac:
            rec["volfrac_target"] = round(_vol_now(), 4)
        rec.update(_phase_timings())
        if robust:
            rec["obj_per_design"] = [round(o, 3) for o in obj_designs]
        if heaviside:
            rec["beta"] = round(proj.beta, 2)
        if pde_filter and ncells > 200_000:
            rec["helmholtz_solve_s"] = round(fops.stats.get("solve_s", 0.0), 2)
        history.append(rec)
        if progress_cb:
            progress_cb(it, rec)
        if progress_file:
            with contextlib.suppress(Exception):
                progress_file.write_text(
                    json.dumps({"history": history}, indent=1), encoding="utf-8"
                )
        # limit-cycle detection: change saturating the OC move cap for many
        # consecutive iterations means the density field bounces between
        # competing layouts rather than converging — surface it once so the
        # caller (agent) can retry with a symmetry plane or another filter
        move_streak = move_streak + 1 if change >= 0.95 * params.move else 0
        if move_streak == 15:
            warnings.append(
                f"density field oscillating at the move limit for {move_streak} "
                "consecutive iterations — a limit cycle, not convergence; "
                "consider symmetry=<plane> or a different filter/threshold"
            )
        # once the Heaviside continuation has hit its beta cap there is no
        # mechanism left to settle an oscillating field — it will bounce
        # between layouts until the iteration cap. Extract the current field
        # honestly instead of burning ~10-20 more solver minutes per pass.
        if move_streak >= 15 and heaviside and proj.beta >= params.beta_max:
            warnings.append(
                f"limit cycle persisted past the beta cap ({params.beta_max:g}) "
                f"at iteration {it} — no settling mechanism left; extracting "
                "the current density field. Retry with symmetry=<plane> or a "
                "different filter/threshold"
            )
            stop = "limit_cycle"
            break
        # O2 early detection — don't wait for the beta cap: when the density
        # has saturated the move limit for 10+ consecutive iterations AND
        # compliance stopped improving over that window, the field is
        # bouncing between competing layouts, not converging. Same honest
        # extraction as the beta-cap break, ~10-20 solver minutes earlier.
        move_sat_streak = (
            move_sat_streak + 1 if change >= _MOVE_SAT_FRAC * params.move else 0
        )
        # mask-churn streak — the density field must keep flipping cells to
        # count as oscillating; a Heaviside continuation can saturate the
        # move cap for a whole window while compliance sits flat on the
        # SAME mask (sharpening gray cells), which is progress, not a cycle
        flip_streak = flip_streak + 1 if flips > _FLIP_FRAC else 0
        beta_age += 1
        if move_sat_streak >= _MOVE_SAT_STREAK and flip_streak >= _MOVE_SAT_STREAK:
            # stalled = <1% improvement across a same-beta window (the
            # helper returns None while the current-beta segment is short —
            # a ramp in flight is 'unknown', never 'stalled')
            imp = _windowed_compliance_improvement(history, _MOVE_SAT_WINDOW)
            if imp is not None and imp < 0.01:
                if heaviside and proj.beta >= params.beta_max:
                    warnings.append(
                        f"limit cycle persisted past the beta cap ({params.beta_max:g}) "
                        f"at iteration {it} — density saturated the move limit for "
                        f"{move_sat_streak} consecutive iterations with <1% compliance "
                        "change; no settling mechanism left — extracting the current "
                        "density field. Retry with symmetry=<plane> or a different "
                        "filter/threshold"
                    )
                else:
                    warnings.append(
                        f"limit cycle detected at iteration {it} — density saturated "
                        f"the move limit for {move_sat_streak} consecutive iterations "
                        "with <1% compliance improvement; extracting the current "
                        "density field. Retry with symmetry=<plane> or a different "
                        "filter/threshold"
                    )
                stop = "limit_cycle"
                break

        # secondary convergence: OC 'change' can stay at the move limit while
        # the objective already settled — count a stalled objective only when
        # the field itself stopped evolving, otherwise a still-blurry density
        # is extracted while members keep churning and fragments. `change`
        # below the move cap, a near-binary field (<5% gray design cells) or
        # a stable thresholded set (<0.5% of design cells flipping — the
        # printed part is the mask, not the raw densities) all count as
        # settled. While a Heaviside continuation can still run it IS the
        # mechanism that settles the field — the projection ramp is triggered
        # by the stalled objective, so it must fire even when the density is
        # still gray; only at the beta cap does `settled` gate.
        if prev_objective is not None and prev_objective > 0:
            rel = abs(objective - prev_objective) / prev_objective
            settled = (
                change <= 0.9 * params.move or gray <= 0.05 or flips <= 0.005
            )
            continuing = heaviside and proj.beta < params.beta_max
            stall = stall + 1 if (rel < params.tol and (continuing or settled)) else 0
            if continuing and (stall >= params.converge_patience or beta_age >= _BETA_DWELL):
                # advance the ramp on an objective stall OR when a level has
                # run a full window without settling — whichever comes first
                proj.beta = min(proj.beta * 2.0, params.beta_max)
                stall = 0
                beta_age = 0
                warnings.append(f"beta continuation -> {proj.beta:g} at iteration {it}")
                continue
            if stall >= params.converge_patience:
                warnings.append(
                    f"converged: compliance stalled (<{params.tol:g} rel. change "
                    f"for {stall} iterations) at iteration {it}"
                )
                break
        prev_objective = objective

    # return the PHYSICAL (printed) density — extraction thresholds it at 0.5;
    # with Heaviside it is near-binary anyway
    warnings_out = warnings
    converged = change <= params.tol or stall >= params.converge_patience
    passive_frac = float(passive_solid.sum()) / nsolid
    # honest outcome label (B18): how the loop actually ended, not just a
    # bool — the agent distinguishes "converged" from "the request was
    # infeasible" and "the field never settled"
    status = "converged"
    if stop is not None:
        # limit_cycle | degenerate | cancelled (an externally cancelled run
        # spent its remaining budget — report it as budget_exhausted)
        converged = False
        status = "budget_exhausted" if stop == "cancelled" else stop
        if stop == "limit_cycle" and x_prev is not None:
            # extract the better phase of the bounce — a period-2 limit cycle
            # alternates between a crisp layout and its blurrier mirror, and
            # the break lands wherever the streak ends. The previous
            # blueprint is the other phase; pick the one with fewer gray
            # cells (no extra solve needed — projection is elementwise)
            def _gray(blueprint: np.ndarray) -> float:
                phys = _physical(blueprint, eta=eta_vol)[solid_idx]
                return float(np.mean((phys > 0.2) & (phys < 0.8))) if phys.size else 1.0

            if _gray(x_prev) < _gray(x):
                x = x_prev
    elif not converged:
        status = "budget_exhausted"
    elif it > 0:
        # settled — but if the field only "converged" because every free
        # design cell drained to the density floor while the pinned passive
        # region alone already exceeds the requested volume, the target was
        # infeasible by construction, not an optimum
        achieved = float(_physical(x, eta=eta_vol)[solid_idx].sum()) / nsolid
        if (
            achieved > params.volfrac + 0.02
            and float(x[design].max()) <= _FLOOR_PIN
        ):
            converged = False
            status = "infeasible_target"
            warnings_out.append(
                f"target volfrac {params.volfrac:.0%} sits below the passive floor "
                f"— achieved {achieved:.0%} with every free design cell at the "
                "density minimum; status=infeasible_target"
            )
    if passive_frac > params.volfrac * 1.02:
        warnings_out.append(
            f"passive (keep/load/fixed) cells cover {passive_frac:.0%} of the part — "
            f"target volfrac {params.volfrac:.0%} unreachable; floor ≈ {passive_frac:.0%}"
        )
    elif ndesign < 0.05 * nsolid:
        warnings_out.append(
            "very few free design cells (<5%) — keep regions cover most of the part"
        )
    return TopOptResult(
        density=_physical(x),
        iterations=it,
        converged=converged,
        status=status,
        history=history,
        warnings=warnings_out,
        passive_frac=passive_frac,
        blueprint=x.copy(),
    )


def _adjoint_solve(
    Kf: sparse.csr_matrix,
    rhs: np.ndarray,
    free: np.ndarray,
    factor: Any,
    M: Any,
    n_nodes: int,
) -> np.ndarray:
    """Solve Kᵀ λ = rhs (K symmetric → same factor/preconditioner)."""
    lam = np.zeros(3 * n_nodes)
    if free.size == 0:
        return lam
    if factor is not None:
        lam[free] = factor(rhs[free])
        return lam
    x, info = spla.cg(Kf, rhs[free], rtol=1e-5, atol=0.0, maxiter=4000, M=M)
    lam[free] = x
    if info > 0:
        log.warning("topopt.adjoint_cg_no_converge", info=info)
    return lam


def _element_vm(
    domain: VoxelDomain,
    u: np.ndarray,
    edof: np.ndarray,
    E_per_cell: np.ndarray,
    nu: float,
    aniso: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid von-Mises per cell + per-element stress vector.

    Returns (vm(ncells), sig(ncells,6), deps_due(ncells,6,24)) — sig computed
    with the UNRELAXED modulus E_per_cell so the qp-relaxation can be applied
    on top."""
    from cadx.cad.fem.solver import _B0, _D

    B0 = _B0(domain.pitch)
    Dm = _D(nu) if aniso is None else _D(nu, float(aniso[0]), int(aniso[1]))
    ue = u[edof]  # (ncells, 24)
    eps = ue @ B0.T  # (ncells, 6)
    sig = eps @ Dm.T * E_per_cell[:, None]
    sx, sy, sz, txy, tyz, tzx = sig.T
    vm = np.sqrt(
        np.clip(
            sx**2 + sy**2 + sz**2 - sx * sy - sy * sz - sz * sx + 3 * (txy**2 + tyz**2 + tzx**2),
            0.0,
            None,
        )
    )
    # dvm/due = ((Vm @ sig)ᵀ D E0 B0) / vm — Vm the von-Mises coupling matrix
    Vm = np.array(
        [
            [1.0, -0.5, -0.5, 0, 0, 0],
            [-0.5, 1.0, -0.5, 0, 0, 0],
            [-0.5, -0.5, 1.0, 0, 0, 0],
            [0, 0, 0, 3.0, 0, 0],
            [0, 0, 0, 0, 3.0, 0],
            [0, 0, 0, 0, 0, 3.0],
        ]
    )
    dvm_dsig = sig @ Vm.T  # (ncells, 6)
    safe = np.clip(vm, 1e-12, None)
    # (Vm sig)^T D — per element then ·B0 -> (ncells, 24); E0 scales sig+deriv
    dsig_due = (E_per_cell[:, None] * (dvm_dsig @ Dm.T) / safe[:, None]) @ B0
    return vm, sig, dsig_due


def _run_mma(  # noqa: C901
    x: np.ndarray,
    *,
    des: np.ndarray,
    eval_fn,
    project_fn,
    chain_fn,
    sens_fn,
    etas: tuple[float, ...],
    eta_vol: float,
    solid_idx: np.ndarray,
    weights: np.ndarray,
    params: TopOptParams,
    nsolid: int,
    v_target: float,
    nu: float,
    domain: VoxelDomain,
    edof: np.ndarray,
    ke: np.ndarray,
    E0: np.ndarray,
    Emin: np.ndarray,
    cases: list[tuple[np.ndarray, np.ndarray]],
    warnings: list[str],
    history: list[dict[str, Any]],
    heaviside: bool,
    beta0: float,
    cancelled: CancelCb | None,
    progress_cb: ProgressCb | None,
    progress_file: Path | None,
    t0: float,
    penal_now: Callable[[], float],
    set_beta: Callable[[float], None],
    it_box: list[int],
    passive_solid: np.ndarray,
    pvoid: np.ndarray,
    change_box: list[float] | None = None,
    symmetrize_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    phase_t: dict[str, float] | None = None,
    phase_timings: Callable[[], dict[str, float]] | None = None,
) -> tuple[np.ndarray, int, bool, float, str]:
    """MMA driver (Svanberg, via nlopt LD_MMA).

    Replaces the OC update: handles a real constraint set —
    volume (always), per-case minmax compliance (bound formulation with an
    extra t variable), a max-displacement cap and a per-case P-norm stress
    limit (qp-relaxation, adjoint sensitivities). Each nlopt evaluation maps
    to one SIMP "iteration" in history.

    Returns (x_blueprint, iterations, converged, last_change, status_hint) —
    status_hint is simp()'s TopOptResult.status candidate ("converged" unless
    the optimizer stopped abnormally)."""
    import nlopt

    minmax = params.case_objective == "minmax" and len(cases) > 1
    if params.case_objective == "minmax" and not minmax:
        warnings.append("case_objective=minmax needs >1 load case — ignored")
    # robust formulation: `etas` lists the realizations (eroded first);
    # the objective works on the worst one (subgradient), the volume
    # constraint on the dilated eta_vol projection and the remaining
    # constraints on the eroded (worst) design
    robust = len(etas) > 1 or float(etas[0]) != _ETA
    ndv = int(des.size)
    # smooth minmax: P-norm of the per-case compliances as the objective —
    # much better conditioned for nlopt-MMA than the bound (t-variable)
    # formulation, which can stall slightly infeasible
    minmax_p = 8.0
    nvar = ndv
    ncells = domain.cell_count
    vol_frac_target = v_target / nsolid
    solid_mask = np.zeros(ncells, dtype=bool)
    solid_mask[solid_idx] = True

    # mirror Jacobian: _unpack evaluates the SYMMETRIZED blueprint
    # xb = S(v) with S = 0.5*(I + F_mirror) plus re-pinning, so every
    # gradient wrt the raw MMA variables must pass through S's adjoint —
    # 0.5*(d[c] + d[m(c)]) for cells whose mirror m(c) is a free design
    # variable, 0.5*d[c] when the mirror is pinned or outside the solid
    # (the re-pin clamps it). Built once here; identity when off.
    sym_axis = _SYM_AXES.get(params.symmetry or "") if symmetrize_fn is not None else None
    mirror_idx: np.ndarray | None = None
    mirror_free: np.ndarray | None = None
    if sym_axis is not None:
        mirror_idx = _mirror_index_map(domain.dims, sym_axis, solid_mask)
        design_mask = np.zeros(ncells, dtype=bool)
        design_mask[des] = True
        mirror_free = (mirror_idx >= 0) & design_mask[np.clip(mirror_idx, 0, None)]

    def _grad_adj(d: np.ndarray) -> np.ndarray:
        """blueprint sensitivity -> raw-variable sensitivity through the
        mirror average (identity when symmetry is off)."""
        if mirror_idx is None or mirror_free is None:
            return d
        return _sym_grad(d, mirror_idx, mirror_free)

    opt = nlopt.opt(nlopt.LD_MMA, nvar)
    lb = np.full(nvar, RHO_MIN)
    ub = np.ones(nvar)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(max(1, params.max_iter))
    opt.set_xtol_rel(max(1e-4, params.tol * 0.2))
    opt.set_ftol_rel(1e-9)

    def _wsum(obj_cases: list[float]) -> float:
        w = weights[: len(obj_cases)]
        if len(w) < len(obj_cases):
            w = np.concatenate([w, np.ones(len(obj_cases) - len(w))])
        return float(np.dot(w, obj_cases))

    cache: dict[str, Any] = {"v": None, "evs": None, "xb": None}
    last_change = [0.0]
    if change_box is None:
        change_box = last_change
    prev_xb = [None]

    def _unpack(v: np.ndarray) -> np.ndarray:
        xb = np.full(ncells, RHO_MIN)
        xb[des] = np.clip(v[:ndv], RHO_MIN, 1.0)
        xb[passive_solid] = 1.0
        xb[pvoid] = RHO_MIN
        if symmetrize_fn is not None:
            # nlopt steps the raw v — mirror-average the blueprint so every
            # evaluated design is symmetric before the filter sees it
            xb = symmetrize_fn(xb)
        return xb

    def _ev(v: np.ndarray):
        """Evaluate the blueprint under every eta realization.

        Returns (evs, xb) — one _eval tuple per entry of `etas` (a single
        one in the non-robust case)."""
        if cache["v"] is not None and np.array_equal(cache["v"], v):
            return cache["evs"], cache["xb"]
        xb = _unpack(np.asarray(v, dtype=float))
        evs = [eval_fn(xb, eta=e) for e in etas]
        it_box[0] += 1
        if prev_xb[0] is not None:
            last_change[0] = float(np.abs(xb - prev_xb[0]).max())
            change_box[0] = last_change[0]  # eval_fn's cg_rtol reads this live
        prev_xb[0] = xb
        obj_ds = [_wsum(ev[4]) for ev in evs]
        # the enforced volume is the dilated realization's — computed via the
        # projection chain (no extra FEM solve needed)
        p_vol = project_fn(xb, eta_vol)[1]
        rec = {
            "iteration": it_box[0],
            "compliance_mj": round(max(obj_ds), 3),
            "volfrac": round(float(p_vol[solid_idx].sum()) / nsolid, 4),
            "penal": round(penal_now(), 3),
            "elapsed_s": round(time.monotonic() - t0, 1),
            "solver": "mma",
        }
        if phase_timings is not None:
            rec.update(phase_timings())
        if robust:
            rec["obj_per_design"] = [round(o, 3) for o in obj_ds]
        history.append(rec)
        if progress_cb:
            progress_cb(it_box[0], rec)
        if progress_file:
            with contextlib.suppress(Exception):
                progress_file.write_text(
                    json.dumps({"history": history}, indent=1), encoding="utf-8"
                )
        cache.update(v=np.asarray(v, dtype=float).copy(), evs=evs, xb=xb)
        if cancelled and cancelled():
            raise nlopt.ForcedStop()
        return evs, xb

    # ---- objective ------------------------------------------------------
    def _f_obj(v: np.ndarray, grad: np.ndarray) -> float:
        evs, xb = _ev(v)
        # worst case over the eta realizations (robust) — the subgradient is
        # the argmax design's sensitivity
        objs: list[float] = []
        for ev in evs:
            obj_cases = np.asarray(ev[4], dtype=float)
            if minmax:
                safe = np.clip(obj_cases, 1e-12, None)
                objs.append(float(np.power(np.sum(np.power(safe, minmax_p)), 1.0 / minmax_p)))
            else:
                objs.append(_wsum(ev[4]))
        di = int(np.argmax(objs))
        ev, eta_w = evs[di], etas[di]
        obj = objs[di]
        if grad.size:
            if minmax:
                obj_cases = np.asarray(ev[4], dtype=float)
                safe = np.clip(obj_cases, 1e-12, None)
                dcfs = sens_fn(xb, ev, oc_filter=False, eta=eta_w)
                wi = np.power(safe, minmax_p - 1.0) * obj ** (1.0 - minmax_p)
                g = np.zeros(ncells)
                for ci, d_i in enumerate(dcfs):
                    g += wi[ci] * d_i
                grad[:] = _grad_adj(g)[des]
            else:
                # true chain-rule sensitivity for the weighted objective —
                # the OC-filtered heuristic is not a gradient and misled MMA
                dcfs = sens_fn(xb, ev, oc_filter=False, eta=eta_w)
                g = np.zeros(ncells)
                for ci, d_i in enumerate(dcfs):
                    w = float(weights[ci]) if ci < len(weights) else 1.0
                    g += w * d_i
                grad[:ndv] = _grad_adj(g)[des]
        return obj

    opt.set_min_objective(_f_obj)

    # ---- volume fraction --------------------------------------------------
    def _g_vol(v: np.ndarray, grad: np.ndarray) -> float:
        _evs, xb = _ev(v)
        # volume on the DILATED realization (eta_vol = _ETA when not robust)
        xp, p, xt = project_fn(xb, eta_vol)
        if grad.size:
            dvol = np.zeros(ncells)
            dvol[solid_idx] = 1.0
            dv = chain_fn(dvol, xp, p, xt, xb, oc_filter=False, eta=eta_vol)
            grad[:ndv] = _grad_adj(dv)[des] / nsolid
        return float(p[solid_idx].sum()) / nsolid - vol_frac_target

    opt.add_inequality_constraint(_g_vol, 1e-8)

    # ---- max displacement --------------------------------------------------
    if params.max_displacement_mm:

        def _g_disp(v: np.ndarray, grad: np.ndarray) -> float:
            evs, xb = _ev(v)
            ev = evs[0]  # eroded = worst design for the constraint
            us = ev[5]
            disp = max(float(np.abs(u).max()) for u in us)
            if grad.size:
                ci = int(np.argmax([float(np.abs(u).max()) for u in us]))
                u = us[ci]
                ctx = ev[6][ci]  # the argmax case's own free DOFs/system
                dof_star = int(np.abs(u).argmax())
                rhs = np.zeros(3 * domain.node_count)
                rhs[dof_star] = np.sign(u[dof_star])
                t_adj = time.monotonic()
                lam = _adjoint_solve(ctx.Kf, rhs, ctx.free, ctx.factor, ctx.M, domain.node_count)
                if phase_t is not None:
                    phase_t["solve_s"] += time.monotonic() - t_adj
                xp, p, xt = ev[0], ev[1], ev[2]
                pen = penal_now()
                dE = pen * np.power(np.clip(p, 1e-3, None), pen - 1) * (E0 - Emin)
                per_el = -dE * np.einsum("ni,ij,nj->n", lam[edof], ke, u[edof])
                d = chain_fn(per_el, xp, p, xt, xb, oc_filter=False, eta=etas[0])
                grad[:ndv] = _grad_adj(d)[des]
            return disp - float(params.max_displacement_mm)

        opt.add_inequality_constraint(_g_disp, 1e-8)

    # ---- P-norm stress constraint (qp-relaxation, per case) ----------------
    if params.max_stress_mpa:
        P = float(params.stress_pnorm)
        q = float(params.stress_relax_q)

        def _mk_stress_con(ci: int):
            def g(v: np.ndarray, grad: np.ndarray) -> float:
                evs, xb = _ev(v)
                ev = evs[0]  # eroded = worst design for the constraint
                xp, p, xt, us, ctx = ev[0], ev[1], ev[2], ev[5], ev[6][ci]
                u = us[ci]
                vm, _sig, dvm_due = _element_vm(domain, u, edof, E0, nu, params.aniso)
                sm = np.clip(vm, 1e-9, None)
                relax = np.power(np.clip(p, 1e-6, None), q)
                srel = relax * vm  # relaxed stress
                # volume-normalized P-norm — the unnormalized sum makes the
                # same stress limit mesh-dependent (2x resolution → ~30%
                # tighter at P=8); mean keeps the limit resolution-free
                n_sol = float(solid_idx.size)
                spn = float(np.power(np.sum(np.power(srel[solid_idx], P)) / n_sol, 1.0 / P))
                if grad.size:
                    # adjoint: K^T lam = d(spn)/du  (assembled element-wise).
                    # the sum runs over solid cells only — non-solid cells
                    # contribute zero, so their sensitivity coefficient is 0
                    a = np.power(srel, P - 1.0)
                    coef = spn ** (1.0 - P) * a * relax * solid_mask / n_sol  # per cell
                    rhs_e = coef[:, None] * dvm_due  # (ncells, 24)
                    rhs = np.zeros(3 * domain.node_count)
                    np.add.at(rhs, edof.ravel(), rhs_e.ravel())
                    t_adj = time.monotonic()
                    lam = _adjoint_solve(
                        ctx.Kf, rhs, ctx.free, ctx.factor, ctx.M, domain.node_count
                    )
                    if phase_t is not None:
                        phase_t["solve_s"] += time.monotonic() - t_adj
                    pen = penal_now()
                    dE = pen * np.power(np.clip(p, 1e-3, None), pen - 1) * (E0 - Emin)
                    dgdp = (
                        -dE * np.einsum("ni,ij,nj->n", lam[edof], ke, u[edof])
                        + spn ** (1.0 - P)
                        * a
                        * q
                        * np.power(np.clip(p, 1e-6, None), q - 1.0)
                        * sm
                        * solid_mask
                        / n_sol
                    )
                    d = chain_fn(dgdp, xp, p, xt, xb, oc_filter=False, eta=etas[0])
                    grad[:ndv] = _grad_adj(d)[des] / float(params.max_stress_mpa)
                return spn / float(params.max_stress_mpa) - 1.0

            return g

        for ci in range(len(cases)):
            opt.add_inequality_constraint(_mk_stress_con(ci), 1e-8)

    # ---- drive -------------------------------------------------------------
    converged = False
    mma_status = "budget_exhausted"  # ForcedStop / maxeval without settling
    v = x[des].copy()
    v = np.asarray(v, dtype=float)
    beta_cur = beta0
    restart = 0
    while True:
        try:
            v = np.asarray(opt.optimize(v), dtype=float)
            converged = True
            mma_status = "converged"
        except nlopt.ForcedStop:
            warnings.append(f"cancelled after {it_box[0]} MMA evaluations")
            break
        except Exception as e:
            warnings.append(f"mma stopped ({type(e).__name__}: {e})")
            # nlopt's soft numeric stop (roundoff-limited) still returns a
            # usable best point; an exception raised by OUR evaluation (FEM
            # crash, numpy failure) or an nlopt failure code means the run
            # degenerated mid-optimization — don't call it converged
            soft = getattr(nlopt, "RoundoffLimited", None)
            converged = bool(soft is not None and isinstance(e, soft))
            mma_status = "converged" if converged else "degenerate"
            break
        x = _unpack(v)
        # Heaviside beta-continuation: converge at beta, then sharpen
        if heaviside and beta_cur < params.beta_max:
            beta_cur = min(beta_cur * 2.0, params.beta_max)
            set_beta(beta_cur)
            warnings.append(f"beta continuation -> {beta_cur:g} (mma restart {restart + 1})")
            restart += 1
            if restart > 4:
                break
            continue
        break
    x = _unpack(v)
    # honest feasibility check — MMA can xtol-stop slightly over the volume
    # bound; the constrained realization is the dilated one (eta_vol)
    final_p = project_fn(x, eta_vol)[1]
    final_vol = float(final_p[solid_idx].sum()) / nsolid
    if final_vol > vol_frac_target * 1.03:
        warnings.append(
            f"mma converged with volfrac {final_vol:.0%} above target "
            f"{vol_frac_target:.0%} — another constraint (stress/disp) is binding"
        )
    return x, it_box[0], converged, last_change[0], mma_status


def passive_solid_mask(
    domain: VoxelDomain,
    extra_cell_masks: list[np.ndarray],
    case_fixed: list[np.ndarray],
    case_load: list[np.ndarray],
) -> np.ndarray:
    """Cells pinned to rho=1: keep regions + fix + load regions (flat order)."""
    mask = np.zeros(domain.cell_count, dtype=bool)
    for m in extra_cell_masks:
        mask |= np.asarray(m, dtype=bool).ravel()
    # fix/load node masks -> cells touching them
    en = domain.element_nodes()
    for nm in (*case_fixed, *case_load):
        nodes = np.flatnonzero(nm)
        if nodes.size == 0:
            continue
        touch = np.isin(en, nodes).any(axis=1)
        mask |= touch & domain.solid.ravel()
    return np.asarray(mask & domain.solid.ravel())
