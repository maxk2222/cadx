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


@dataclass
class TopOptResult:
    density: np.ndarray  # (ncells,) final densities in flat cell order
    iterations: int
    converged: bool
    history: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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


def _solve_u(
    domain: VoxelDomain,
    K: sparse.csr_matrix,
    F: np.ndarray,
    free: np.ndarray,
    u0: np.ndarray | None,
    spsolve_dofs: int = 120_000,
) -> tuple[np.ndarray, str]:
    """Light solve: return full u vector."""
    u = np.zeros(3 * domain.node_count)
    Kf = K[free][:, free]
    if free.size <= spsolve_dofs:
        try:
            u[free] = spla.spsolve(Kf, F[free])
            return u, "spsolve"
        except Exception:
            pass
    diag = Kf.diagonal()
    diag[diag == 0] = 1.0
    M = spla.diags(1.0 / diag)
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


def simp(
    domain: VoxelDomain,
    elem_E0: float,
    nu: float,
    cases: list[tuple[np.ndarray, np.ndarray]],  # (fixed_mask, F) per case
    passive_solid: np.ndarray,  # (ncells,) bool — pinned rho=1
    params: TopOptParams,
    *,
    progress_cb: ProgressCb | None = None,
    cancelled: CancelCb | None = None,
    progress_file: Path | None = None,
) -> TopOptResult:
    """Run the SIMP loop. Returns the density field (flat cell order)."""
    warnings: list[str] = []
    solid = domain.solid.ravel()
    design = solid & ~passive_solid
    nsolid = int(solid.sum())
    ndesign = int(design.sum())
    if ndesign == 0:
        return TopOptResult(
            density=solid.astype(float),
            iterations=0,
            converged=False,
            warnings=["no free design cells — everything is kept/pinned"],
        )

    E0 = float(elem_E0)
    Emin = E0 * E_MIN_FRAC
    x = np.where(solid, params.volfrac, RHO_MIN)
    x[passive_solid] = 1.0
    x = x.astype(float)

    ke = _ke_unit(nu, domain.pitch)
    en = domain.element_nodes()
    edof = element_dofs(en)

    # sensitivity filter over solid cells
    rmin_cells = max(1.2, params.rmin_mm / domain.pitch)
    H = _filter_matrix(domain, rmin_cells)
    Hs = np.asarray(H.sum(axis=1)).ravel()
    Hs[Hs <= 0] = 1.0
    solid_idx = np.flatnonzero(solid)

    # free DOFs: union over cases (all cells assembled — void gets Emin)
    used_dofs = np.unique(edof.ravel())
    fixed_nodes = np.zeros(domain.node_count, dtype=bool)
    for f, _ in cases:
        fixed_nodes |= f
    fixed_dofs = set(
        np.concatenate([3 * np.flatnonzero(fixed_nodes) + c for c in range(3)]).tolist()
    )
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

        E_e = Emin + np.power(x, params.penal) * (E0 - Emin)
        K, _, _ = assemble(domain, E_e, nu, np.ones(domain.cell_count, dtype=bool))

        # solve + element compliances per case
        ce = np.zeros(domain.cell_count)
        objective = 0.0
        for ci, (_, F) in enumerate(cases):
            u, _solver = _solve_u(domain, K, F, free, u_prev[ci])
            u_prev[ci] = u
            ue = u[edof]
            ce_i = np.einsum("ni,ij,nj->n", ue, ke, ue)
            ce += ce_i
            objective += float(F[free] @ u[free])

        # sensitivity (summed over cases), cone-filtered on solid cells
        dc = -params.penal * np.power(np.clip(x, 1e-3, None), params.penal - 1) * (E0 - Emin) * ce
        xdc = H @ (x[solid_idx] * dc[solid_idx])
        dcf = dc.copy()
        dcf[solid_idx] = xdc / np.clip(x[solid_idx] * Hs, 1e-3, None)

        # OC update with bisection on the volume constraint (solid cells only)
        xnew = x.copy()
        l1, l2 = 0.0, 1e9
        for _ in range(60):
            lmid = 0.5 * (l1 + l2)
            cand = x * np.sqrt(np.clip(-dcf / max(lmid, 1e-12), 0.0, None))
            cand = np.clip(cand, x - params.move, x + params.move)
            cand = np.clip(cand, 0.0, 1.0)
            cand[~solid] = RHO_MIN
            cand[passive_solid] = 1.0
            if cand[solid_idx].sum() > v_target:
                l1 = lmid
            else:
                l2 = lmid
            xnew = cand
            if l2 - l1 < 1e-4 * (l1 + l2 + 1e-12):
                break

        change = float(np.abs(xnew - x).max())
        x = xnew
        vol_now = float(x[solid_idx].sum()) / nsolid
        rec = {
            "iteration": it,
            "compliance_mj": round(objective, 3),
            "volfrac": round(vol_now, 4),
            "change": round(change, 5),
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        history.append(rec)
        if progress_cb:
            progress_cb(it, rec)
        if progress_file:
            with contextlib.suppress(Exception):
                progress_file.write_text(
                    json.dumps({"history": history}, indent=1), encoding="utf-8"
                )

        # secondary convergence: OC 'change' can stay at the move limit while
        # the objective already settled — stop on stalled compliance too
        if prev_objective is not None and prev_objective > 0:
            rel = abs(objective - prev_objective) / prev_objective
            stall = stall + 1 if rel < params.tol else 0
            if stall >= params.converge_patience:
                warnings.append(
                    f"converged: compliance stalled (<{params.tol:g} rel. change "
                    f"for {stall} iterations) at iteration {it}"
                )
                break
        prev_objective = objective

    warnings_out = warnings
    converged = change <= params.tol or stall >= params.converge_patience
    if ndesign < 0.05 * nsolid:
        warnings_out.append(
            "very few free design cells (<5%) — keep regions cover most of the part"
        )
    return TopOptResult(
        density=x,
        iterations=it,
        converged=converged,
        history=history,
        warnings=warnings_out,
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
