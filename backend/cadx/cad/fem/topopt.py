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
from typing import Any

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
    # M3.1: density filter variant — "cone" (classic sensitivity filter) or
    # "heaviside" (cone filter + smoothed Heaviside projection; beta doubles
    # on every compliance stall up to beta_max → crisp 0/1 designs whose
    # extraction is threshold-robust)
    filter: str = "cone"
    beta0: float = 1.0  # initial Heaviside sharpness
    beta_max: float = 8.0  # cap — beyond this the projection is ~binary
    # M2: Langelaar-style additive-manufacturing filter — a cell only counts
    # as printed if its 45° support cone below is printed
    self_supporting: bool = False
    print_direction: str = "+z"  # build direction; overhangs vs the bed


@dataclass
class TopOptResult:
    density: np.ndarray  # (ncells,) final densities in flat cell order
    iterations: int
    converged: bool
    history: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passive_frac: float = 0.0  # fraction of solid pinned by keep/load/fixed cells


_AXES = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}
_ETA = 0.5  # Heaviside projection threshold


def _heaviside(x: np.ndarray, beta: float) -> np.ndarray:
    """Smoothed Heaviside projection (Wang et al.): 1 at x>eta, ~x at beta->0."""
    den = math.tanh(beta * _ETA) + math.tanh(beta * (1.0 - _ETA))
    return (math.tanh(beta * _ETA) + np.tanh(beta * (x - _ETA))) / den


def _heaviside_d(x: np.ndarray, beta: float) -> np.ndarray:
    den = math.tanh(beta * _ETA) + math.tanh(beta * (1.0 - _ETA))
    return beta * (1.0 - np.tanh(beta * (x - _ETA)) ** 2) / den


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


def _prepare_solve(Kf: sparse.csr_matrix, spsolve_dofs: int) -> tuple[Any, Any]:
    """Per-iteration solver setup shared across load cases.

    Returns (factor, M): `factor` is a `spla.factorized` callable when the
    system is small enough for a direct solve, else None; `M` is the SSOR
    (fallback diagonal) preconditioner for CG."""
    from cadx.cad.fem.solver import _ssor_preconditioner

    if Kf.shape[0] <= spsolve_dofs:
        try:
            return spla.factorized(Kf.tocsc()), None
        except Exception:
            pass
    M = _ssor_preconditioner(Kf)
    if M is None:
        diag = Kf.diagonal()
        diag[diag == 0] = 1.0
        M = spla.diags(1.0 / diag)
    return None, M


def _solve_u(
    domain: VoxelDomain,
    Kf: sparse.csr_matrix,
    F: np.ndarray,
    free: np.ndarray,
    u0: np.ndarray | None,
    factor: Any = None,
    M: Any = None,
) -> tuple[np.ndarray, str]:
    """Light solve: return full u vector."""
    u = np.zeros(3 * domain.node_count)
    if factor is not None:
        u[free] = factor(F[free])
        return u, "spsolve"
    if M is None:
        _, M = _prepare_solve(Kf, spsolve_dofs=0)
    x, info = spla.cg(
        Kf,
        F[free],
        x0=u0[free] if u0 is not None else None,
        rtol=1e-5,
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
            warnings=["no free design cells — everything is kept/pinned"],
        )

    E0 = np.broadcast_to(np.asarray(elem_E0, dtype=float), domain.cell_count).copy()
    Emin = E0 * E_MIN_FRAC
    if x0 is not None and len(x0) == domain.cell_count:
        x = np.clip(np.asarray(x0, dtype=float), RHO_MIN, 1.0)
        x[~solid] = RHO_MIN
    else:
        x = np.where(solid, params.volfrac, RHO_MIN)
    x[passive_solid] = 1.0
    x[pvoid] = RHO_MIN
    x = x.astype(float)
    weights = (
        np.ones(len(cases))
        if case_weights is None
        else np.asarray(case_weights, dtype=float)[: len(cases)]
    )

    ke = _ke_unit(nu, domain.pitch)
    en = domain.element_nodes()
    edof = element_dofs(en)
    # SIMP assembles ALL cells each iteration — the sparsity pattern is
    # constant, so build the COO index arrays once and reuse them
    ij_cache = (np.repeat(edof, 24, axis=1).ravel(), np.tile(edof, (1, 24)).ravel())

    # sensitivity filter over solid cells
    rmin_cells = max(1.2, params.rmin_mm / domain.pitch)
    H = _filter_matrix(domain, rmin_cells)
    Hs = np.asarray(H.sum(axis=1)).ravel()
    Hs[Hs <= 0] = 1.0
    solid_idx = np.flatnonzero(solid)

    # M3.1 Heaviside projection + M2 AM support filter
    heaviside = params.filter == "heaviside"
    beta = params.beta0
    am_layers = _am_setup(domain, params.print_direction) if params.self_supporting else None
    if am_layers is not None and not am_layers:
        warnings.append("self_supporting: no solid cells found — AM filter inactive")
        am_layers = None

    def _physical(xb: np.ndarray) -> np.ndarray:
        """blueprint density -> physical printed density (flat cell order)."""
        xp = xb
        if heaviside:
            xp = np.full(domain.cell_count, RHO_MIN)
            xp[solid_idx] = _heaviside(np.asarray(H @ xb[solid_idx]).ravel() / Hs, beta)
            xp[passive_solid] = 1.0
            xp[pvoid] = RHO_MIN
        if am_layers is not None:
            return _am_forward(xp, solid, am_layers)
        return xp

    # free DOFs: union over cases (all cells assembled — void gets Emin)
    used_dofs = np.unique(edof.ravel())
    fixed_dofs: set[int] = set()
    for f, _ in cases:
        fixed_dofs |= set(_fixed_dofs(f, domain.node_count).tolist())
    free = np.array([d for d in used_dofs if d not in fixed_dofs], dtype=np.int64)
    if free.size == 0:
        return TopOptResult(
            density=x,
            iterations=0,
            converged=False,
            warnings=["no free DOFs — the whole domain is constrained"],
        )

    v_target = params.volfrac * nsolid
    history: list[dict[str, Any]] = []
    u_prev: list[np.ndarray | None] = [None] * len(cases)
    change = 1.0
    it = 0
    stall = 0
    prev_objective: float | None = None
    t0 = time.monotonic()

    while it < params.max_iter and change > params.tol:
        it += 1
        if cancelled and cancelled():
            warnings.append(f"cancelled after iteration {it - 1}")
            break

        # physical density: blueprint -> (cone filter -> Heaviside) -> AM filter
        xp = x.copy()
        xt: np.ndarray | None = None
        if heaviside:
            xt = np.asarray(H @ x[solid_idx]).ravel() / Hs  # cone-filtered
            xp[solid_idx] = _heaviside(xt, beta)
            xp[passive_solid] = 1.0
            xp[pvoid] = RHO_MIN
        p = _am_forward(xp, solid, am_layers) if am_layers is not None else xp

        E_e = Emin + np.power(p, params.penal) * (E0 - Emin)
        K, _, _ = assemble(domain, E_e, nu, np.ones(domain.cell_count, dtype=bool), ij=ij_cache)

        # factorization/preconditioner is identical for all cases — build once
        Kf = K[free][:, free].tocsr()
        factor, M = _prepare_solve(Kf, spsolve_dofs)

        # solve + element compliances per case (case weights scale both)
        ce = np.zeros(domain.cell_count)
        objective = 0.0
        for ci, (_, F) in enumerate(cases):
            w = float(weights[ci]) if ci < len(weights) else 1.0
            u, _solver = _solve_u(domain, Kf, F, free, u_prev[ci], factor=factor, M=M)
            u_prev[ci] = u
            ue = u[edof]
            ce_i = np.einsum("ni,ij,nj->n", ue, ke, ue)
            ce += w * ce_i
            objective += w * float(F[free] @ u[free])

        # sensitivity w.r.t. physical density, then chain-rule back through
        # the filters to the blueprint variable x
        dc = -params.penal * np.power(np.clip(p, 1e-3, None), params.penal - 1) * (E0 - Emin) * ce
        if am_layers is not None:
            dc = _am_backward(dc, xp, p, solid, am_layers)
        if heaviside:
            # p = heaviside(H x / Hs): dx = H^T (dc * dh/dx̃ / Hs), H symmetric
            dxt = dc[solid_idx] * _heaviside_d(xt, beta)
            dcf = dc.copy()
            dcf[:] = 0.0
            dcf[solid_idx] = np.asarray(H @ (dxt / Hs)).ravel()
        else:
            xdc = H @ (x[solid_idx] * dc[solid_idx])
            dcf = dc.copy()
            dcf[solid_idx] = xdc / np.clip(x[solid_idx] * Hs, 1e-3, None)

        # OC update with bisection on the PHYSICAL volume constraint
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
                vol_c = _physical(cand)[solid_idx].sum()
            else:
                vol_c = cand[solid_idx].sum()
            if vol_c > v_target:
                l1 = lmid
            else:
                l2 = lmid
            xnew = cand
            if l2 - l1 < 1e-4 * (l1 + l2 + 1e-12):
                break

        change = float(np.abs(xnew - x).max())
        x = xnew
        vol_now = float(_physical(x)[solid_idx].sum()) / nsolid
        rec = {
            "iteration": it,
            "compliance_mj": round(objective, 3),
            "volfrac": round(vol_now, 4),
            "change": round(change, 5),
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        if heaviside:
            rec["beta"] = round(beta, 2)
        history.append(rec)
        if progress_cb:
            progress_cb(it, rec)
        if progress_file:
            with contextlib.suppress(Exception):
                progress_file.write_text(
                    json.dumps({"history": history}, indent=1), encoding="utf-8"
                )

        # secondary convergence: OC 'change' can stay at the move limit while
        # the objective already settled — stop on stalled compliance too.
        # With a Heaviside filter a stall first doubles beta (continuation)
        # until the cap, THEN counts as converged.
        if prev_objective is not None and prev_objective > 0:
            rel = abs(objective - prev_objective) / prev_objective
            stall = stall + 1 if rel < params.tol else 0
            if stall >= params.converge_patience:
                if heaviside and beta < params.beta_max:
                    beta = min(beta * 2.0, params.beta_max)
                    stall = 0
                    warnings.append(f"beta continuation -> {beta:g} at iteration {it}")
                    continue
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
        history=history,
        warnings=warnings_out,
        passive_frac=passive_frac,
    )


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
