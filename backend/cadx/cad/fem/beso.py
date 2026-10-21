"""Soft-kill BESO topology optimization on the voxel domain — a binary
sibling of the SIMP loop in topopt.py.

BESO (bidirectional evolutionary structural optimization, Huang & Xie 2007)
evolves a discrete 0/1 design instead of a continuous density: the volume
fraction is ramped down from the initial fill to the target by an
evolutionary ratio ER (``params.beso_er``, default 0.02) and each iteration
keeps the top-`V_k` design cells ranked by their filtered sensitivity
number — the element strain energy

    a_e = u_eᵀ · (E0_e · ke_unit) · u_e

evaluated at FULL stiffness (the number a removed element would score if it
re-entered — what makes the scheme bidirectional: void cells adjacent to
the load path can come back instead of staying pinned to the floor). The
numbers are smoothed through the same density filter simp() uses (cone or
Helmholtz) and stabilised by averaging with the previous iteration's
numbers — the classic BESO remedy against checkerboarding and threshold
oscillation.

SIMP-only machinery is accepted but reported as ignored (the binary design
has no projection chain): Heaviside filters, the robust formulation, the
AM support filter, penal/volfrac continuation, and the MMA-only
constraints (stress/displacement/minmax).

Same signature and result type as simp() — engine.py dispatches on
``params.solver == "beso"`` without touching topopt.py.

Non-design constraints (identical to simp()):
- cells outside the part solid are pinned to rho_min (void)
- keep/load/fix regions are pinned to 1 (passive solid)
- passive_void regions are pinned to rho_min inside the part
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from cadx.cad.fem.solver import _fixed_dofs, _ke_unit, assemble, domain_edof
from cadx.cad.fem.topopt import (
    _FLIP_FRAC,
    _FLOOR_PIN,
    _MOVE_SAT_STREAK,
    _MOVE_SAT_WINDOW,
    _SYM_AXES,
    E_MIN_FRAC,
    RHO_MIN,
    CancelCb,
    ProgressCb,
    TopOptParams,
    TopOptResult,
    _build_filter_ops,
    _CaseCtx,
    _prepare_solve,
    _solve_u,
    _windowed_compliance_improvement,
)
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.logging import get_logger

log = get_logger(__name__)

# default evolutionary ratio: V_{k+1} = max(volfrac, V_k*(1-ER)) — 2%/iter
# mirrors the classic BESO setting; override via params.beso_er
_ER_DEFAULT = 0.02
# compliance is "plateaued" below this relative improvement over the
# _MOVE_SAT_WINDOW window (same ~1% threshold simp's stall detection uses)
_PLATEAU_IMP = 0.01


def beso(  # noqa: C901
    domain: VoxelDomain,
    elem_E0: float | np.ndarray,
    nu: float,
    cases: list[tuple[np.ndarray, np.ndarray]],  # (fixed_mask (nn,) or (nn,3), F) per case
    passive_solid: np.ndarray,  # (ncells,) bool — pinned rho=1
    params: TopOptParams,
    *,
    passive_void: np.ndarray | None = None,  # (ncells,) bool — pinned rho=rho_min
    case_weights: np.ndarray | list[float] | None = None,  # objective weight per case
    x0: np.ndarray | None = None,  # (ncells,) warm-start density field (binarized at 0.5)
    spsolve_dofs: int = 120_000,  # direct-solver cutoff; larger systems use SSOR-CG
    progress_cb: ProgressCb | None = None,
    cancelled: CancelCb | None = None,
    progress_file: Path | None = None,
) -> TopOptResult:
    """Run the soft-kill BESO loop. Returns the binary density field.

    Mirrors simp()'s contract: `density` holds 1.0 on kept/passive cells and
    RHO_MIN on removed/void/non-solid cells, `history` carries the same
    per-iteration record keys and `status` uses the same honest-outcome
    vocabulary (converged / infeasible_target / limit_cycle / degenerate /
    budget_exhausted)."""
    warnings: list[str] = []
    solid = domain.solid.ravel()
    pvoid = (
        np.zeros(domain.cell_count, dtype=bool)
        if passive_void is None
        else np.asarray(passive_void, dtype=bool).ravel() & solid
    )
    passive_solid = np.asarray(passive_solid, dtype=bool).ravel() & solid
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

    # ---- params that do not map onto a discrete optimizer ------------------
    # BESO is binary by construction: the density-filtered field only feeds
    # the keep-ranking, never a projected/relaxed density — so the
    # projection, robust, AM and MMA-constraint machinery simp() applies has
    # no analogue. Accept them silently-with-warning (same convention as
    # simp's "solver=oc ignores ..." warning).
    if params.filter in ("heaviside", "helmholtz+heaviside"):
        warnings.append(
            f"beso ignores the Heaviside projection in filter='{params.filter}' — "
            "densities are binary by construction; the PDE/cone stage still applies"
        )
    if params.robust:
        warnings.append(
            "beso ignores robust=True — no projection thresholds on a binary design"
        )
    if params.self_supporting:
        warnings.append("beso ignores self_supporting — the AM support filter needs densities")
    if params.max_stress_mpa:
        warnings.append("beso ignores max_stress_mpa — stress constraints need solver=mma")
    if params.max_displacement_mm:
        warnings.append(
            "beso ignores max_displacement_mm — displacement caps need solver=mma"
        )
    if params.case_objective == "minmax":
        warnings.append("beso ignores case_objective='minmax' — cases combine by weight")
    if params.penal_init is not None:
        warnings.append("beso ignores penal_init continuation — the design is binary")
    if params.volfrac_init is not None:
        warnings.append("beso ignores volfrac_init — the ER volume ramp already covers it")

    E0 = np.broadcast_to(np.asarray(elem_E0, dtype=float), domain.cell_count).copy()
    Emin = E0 * E_MIN_FRAC
    pen = float(params.penal)
    er = float(params.beso_er or _ER_DEFAULT)
    er = float(np.clip(er, 1e-4, 0.5))  # ER=0 never ramps; >0.5 jumps chaos

    # binary design state: keep[c]=True iff cell c carries E0 this iteration
    keep = np.zeros(domain.cell_count, dtype=bool)
    if x0 is not None and len(x0) == domain.cell_count:
        keep[:] = np.asarray(x0, dtype=float) >= 0.5
    else:
        keep[:] = solid
    keep &= solid
    keep[pvoid] = False
    keep[passive_solid] = True

    # mirror symmetry: symmetrize the filtered sensitivity NUMBERS before
    # the keep-ranking — the kept set then mirrors across the mid-plane too
    # (re-pinning passive cells can still leave the field asymmetric, same
    # caveat as simp's density symmetrization)
    sym_axis: int | None = _SYM_AXES.get(params.symmetry) if params.symmetry else None
    if params.symmetry and sym_axis is None:
        warnings.append(f"unknown symmetry plane '{params.symmetry}' — ignored")
    elif sym_axis is not None:
        vox = domain.solid
        mismatch = float(np.mean(vox != np.flip(vox, axis=sym_axis)))
        if mismatch > 0.15:
            plane_names = {0: "yz (X-flip)", 1: "xz (Y-flip)", 2: "xy (Z-flip)"}
            warnings.append(
                f"symmetry '{params.symmetry}' mirror-averages sensitivity across the "
                f"{plane_names[sym_axis]} mid-plane, but {mismatch:.0%} of solid "
                "voxels are not mirrored — the part is not symmetric about this "
                "plane; the constraint distorts the load paths"
            )

    weights = (
        np.ones(len(cases))
        if case_weights is None
        else np.asarray(case_weights, dtype=float)[: len(cases)]
    )
    if len(weights) < len(cases):
        weights = np.concatenate([weights, np.ones(len(cases) - len(weights))])

    aniso_z, aniso_ax = (1.0, 2) if params.aniso is None else (float(params.aniso[0]), int(params.aniso[1]))
    ke = _ke_unit(nu, domain.pitch, aniso_z, aniso_ax)
    edof = domain_edof(domain)
    # BESO — like SIMP — assembles ALL cells every iteration (removed cells
    # keep their rho_min stiffness so they can re-enter), so the COO
    # sparsity pattern is constant: build the index arrays once
    ij_cache = (np.repeat(edof, 24, axis=1).ravel(), np.tile(edof, (1, 24)).ravel())
    ncells = domain.cell_count
    all_cells = np.ones(ncells, dtype=bool)

    # sensitivity-number filter over the solid cells — the same cone /
    # Helmholtz operator simp() filters densities with (kind from params;
    # the "+heaviside" projection half is reported as ignored above)
    rmin_cells = max(1.2, params.rmin_mm / domain.pitch)
    solid_idx = np.flatnonzero(solid)
    pde_filter = params.filter in ("helmholtz", "helmholtz+heaviside")
    fops = _build_filter_ops(domain, solid_idx, rmin_cells, "helmholtz" if pde_filter else "cone")

    # free DOFs are PER CASE — see simp() for why they must never be unioned
    used_dofs = np.unique(edof.ravel())
    case_free: list[np.ndarray] = []
    for f, _ in cases:
        fd = np.unique(_fixed_dofs(f, domain.node_count))
        case_free.append(used_dofs[~np.isin(used_dofs, fd, assume_unique=True)])

    def _density() -> np.ndarray:
        return np.where(keep, 1.0, RHO_MIN)

    if all(fi.size == 0 for fi in case_free):
        return TopOptResult(
            density=_density(),
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

    v_target = float(params.volfrac)
    des_idx = np.flatnonzero(design)
    npassive = int(passive_solid.sum())
    v_k = float(keep.sum()) / nsolid  # current kept fraction (warm starts count)
    history: list[dict[str, Any]] = []
    u_prev: list[np.ndarray | None] = [None] * len(cases)
    nums_prev: np.ndarray | None = None  # previous iteration's filtered numbers
    it = 0
    stall = 0  # iterations at target with the design settled below tol
    flip_streak = 0  # consecutive iterations with the mask still churning
    at_target_iters = 0  # iterations since the volume ramp reached volfrac
    change = 1.0  # last iteration's flip fraction (drives the CG tolerance)
    stop: str | None = None  # None | cancelled | limit_cycle | degenerate
    settled = False
    t0 = time.monotonic()

    # per-phase wall-clock instrumentation — same vocabulary as simp()
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

    while it < params.max_iter:
        it += 1
        if cancelled and cancelled():
            warnings.append(f"cancelled after iteration {it - 1}")
            stop = "cancelled"
            break

        # evolutionary volume ramp: shrink the target by ER each iteration
        # until volfrac is reached, then hold it — a sparse warm start jumps
        # straight to the target and ADDS cells by the same ranking
        v_k = min(1.0, max(v_target, v_k * (1.0 - er)))
        at_target = v_k <= v_target + 1e-9
        at_target_iters = at_target_iters + 1 if at_target else 0

        x = _density()
        try:
            E_e = Emin + np.power(x, pen) * (E0 - Emin)
            t_asm = time.monotonic()
            K, _, _ = assemble(domain, E_e, nu, all_cells, ij=ij_cache, aniso=params.aniso)
            phase_t["assemble_s"] += time.monotonic() - t_asm
            cg_rtol = 1e-3 if change > 0.1 else (1e-4 if change > 10 * params.tol else 1e-6)
            w_num = np.zeros(ncells)
            obj_cases: list[float] = []
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
                u, _solver = _solve_u(
                    domain,
                    ctx.Kf,
                    F,
                    ctx.free,
                    u_prev[ci],
                    factor=ctx.factor,
                    M=ctx.M,
                    cg_rtol=cg_rtol,
                )
                u_prev[ci] = u
                ue = u[edof]
                # BESO sensitivity number: strain energy the element carries
                # (or WOULD carry, for removed cells) at full modulus — the
                # unpenalized E0·ke is what lets a void cell score high
                # enough to re-enter; weighting by the current (penalized)
                # E_e would pin removed cells at ~0 forever
                ce_i = E0 * np.einsum("ni,ij,nj->n", ue, ke, ue)
                w_num += ce_i * (float(weights[ci]) if ci < len(weights) else 1.0)
                obj_cases.append(float(F[ctx.free] @ u[ctx.free]))
            phase_t["solve_s"] += time.monotonic() - t_solve
        except Exception as e:
            # solver divergence/crash mid-iteration — mirror simp(): report
            # the last design field flagged degenerate instead of losing it
            warnings.append(
                f"FEM solve failed at iteration {it} ({type(e).__name__}: {e}) — "
                "extracting the last density field"
            )
            stop = "degenerate"
            break

        objective = float(np.dot(weights[: len(cases)], obj_cases)) if len(cases) else 0.0

        # filter the numbers over the solid cells (cone: H w / Hs — the
        # plain neighbourhood average; helmholtz: L^-1 w), then stabilise
        # against last iteration's numbers — the Huang & Xie fix for
        # threshold oscillation between near-tied cells
        nums_s = fops.apply(w_num[solid_idx])
        if nums_prev is not None:
            nums_s = 0.5 * (nums_s + nums_prev)
        nums_prev = nums_s
        nums = np.zeros(ncells)
        nums[solid_idx] = nums_s
        if sym_axis is not None:
            g = nums.reshape(domain.dims)
            nums = (0.5 * (g + np.flip(g, axis=sym_axis))).ravel()

        # keep the top-`n_keep` design cells by (filtered, stabilised)
        # number — the exact discrete form of bisecting a threshold tau on
        # the numbers so ~V_k·nsolid cells stay solid (a bisection can't
        # split a tie at the boundary; stable argsort keeps the cut exact
        # and deterministic)
        n_keep = round(v_k * nsolid) - npassive
        n_keep = min(max(n_keep, 0), ndesign)
        if n_keep >= ndesign:
            sel = np.ones(ndesign, dtype=bool)
        elif n_keep <= 0:
            sel = np.zeros(ndesign, dtype=bool)
        else:
            sel = np.zeros(ndesign, dtype=bool)
            order = np.argsort(nums[des_idx], kind="stable")
            sel[order[ndesign - n_keep :]] = True
        keep_prev = keep
        keep = np.zeros(ncells, dtype=bool)
        keep[des_idx] = sel
        keep[passive_solid] = True
        change = float(np.mean(sel != keep_prev[des_idx]))
        vol_now = float(keep.sum()) / nsolid

        rec = {
            "iteration": it,
            "compliance_mj": round(objective, 3),
            "volfrac": round(vol_now, 4),
            "volfrac_target": round(v_k, 4),
            "change": round(change, 5),
            "flips": round(change, 5),
            "penal": round(pen, 3),
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        rec.update(_phase_timings())
        history.append(rec)
        if progress_cb:
            progress_cb(it, rec)
        if progress_file:
            with contextlib.suppress(Exception):
                progress_file.write_text(
                    json.dumps({"history": history}, indent=1), encoding="utf-8"
                )

        # termination — same vocabulary as simp()'s O2 detection:
        # * settled: the kept set stopped churning (flip fraction < tol for
        #   converge_patience iterations, only counted once the ramp has
        #   reached the target — mid-ramp the selection MUST keep moving)
        # * plateau: compliance stopped improving over a full window of
        #   at-target iterations with a stable mask -> converged
        # * plateau + sustained churn -> limit_cycle (oscillation without
        #   progress is not a converged design)
        stall = stall + 1 if (at_target and change <= params.tol) else 0
        flip_streak = flip_streak + 1 if change > _FLIP_FRAC else 0
        imp = (
            _windowed_compliance_improvement(history, _MOVE_SAT_WINDOW)
            if at_target_iters >= _MOVE_SAT_WINDOW
            else None
        )
        stalled_obj = imp is not None and imp < _PLATEAU_IMP
        if stall >= params.converge_patience:
            warnings.append(
                f"converged: design settled (flip fraction <{params.tol:g} for "
                f"{stall} iterations) at iteration {it}"
            )
            settled = True
            break
        if stalled_obj:
            if flip_streak >= _MOVE_SAT_STREAK:
                warnings.append(
                    f"limit cycle detected at iteration {it} — the kept set churned "
                    f"above {_FLIP_FRAC:.1%} of design cells for {flip_streak} "
                    "consecutive iterations with <1% compliance improvement; "
                    "extracting the current density field. Retry with "
                    "symmetry=<plane> or a different filter/threshold"
                )
                stop = "limit_cycle"
                break
            if change <= _FLIP_FRAC:
                warnings.append(
                    f"converged: compliance plateaued (<{_PLATEAU_IMP:.0%} improvement "
                    f"over {_MOVE_SAT_WINDOW} iterations) at iteration {it}"
                )
                settled = True
                break

    # ---- outcome ----------------------------------------------------------
    density = _density()
    converged = settled and stop is None
    status = "converged"
    if stop is not None:
        converged = False
        status = "budget_exhausted" if stop == "cancelled" else stop
    elif not converged:
        status = "budget_exhausted"
    else:
        # settled above the target with every design cell removed — the
        # passive floor alone exceeds the request: infeasible by
        # construction, not an optimum (same check simp() runs)
        achieved = float(density[solid_idx].sum()) / nsolid
        if achieved > params.volfrac + 0.02 and float(density[design].max(initial=0.0)) <= _FLOOR_PIN:
            converged = False
            status = "infeasible_target"
            warnings.append(
                f"target volfrac {params.volfrac:.0%} sits below the passive floor "
                f"— achieved {achieved:.0%} with every free design cell at the "
                "density minimum; status=infeasible_target"
            )
    passive_frac = float(npassive) / nsolid
    if passive_frac > params.volfrac * 1.02:
        warnings.append(
            f"passive (keep/load/fixed) cells cover {passive_frac:.0%} of the part — "
            f"target volfrac {params.volfrac:.0%} unreachable; floor ≈ {passive_frac:.0%}"
        )
    elif ndesign < 0.05 * nsolid:
        warnings.append(
            "very few free design cells (<5%) — keep regions cover most of the part"
        )
    log.debug(
        "beso.done",
        iterations=it,
        status=status,
        volfrac=round(float(density[solid_idx].sum()) / nsolid, 4),
        elapsed_s=round(time.monotonic() - t0, 2),
    )
    return TopOptResult(
        density=density,
        iterations=it,
        converged=converged,
        status=status,
        history=history,
        warnings=warnings,
        passive_frac=passive_frac,
        blueprint=density.copy(),
    )
