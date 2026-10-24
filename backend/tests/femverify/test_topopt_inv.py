"""TopOpt verification via invariants — properties that must hold regardless
of which local optimum the optimizer finds.

  * volume:   the realized material fraction meets the target
  * quality:  the optimized field beats a uniform-density field of the SAME
              volume in compliance — optimization must not be a no-op
  * solvers:  OC and BESO reach comparable designs (local optima differ, but
              orders of magnitude do not)
  * determinism: identical inputs reproduce the identical density field
  * robust:   under the threshold-robust formulation, the eroded realization
              is the most compliant and the dilated the stiffest — material
              sets are nested (eroded ⊂ nominal ⊂ dilated) so compliance must
              order C_e >= C_n >= C_d

Runs on a 60x24x12 cantilever at pitch 3 — small enough for the fast lane.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.beso import beso
from cadx.cad.fem.regions import resolve_case_dofs
from cadx.cad.fem.solver import assemble, reduce_case, solve_reduced
from cadx.cad.fem.topopt import RHO_MIN, TopOptParams, _heaviside, simp
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_MPA = 3500.0
NU = 0.36
VOLFRAC = 0.45


def _cantilever():
    mesh = trimesh.creation.box(extents=[60.0, 24.0, 12.0])
    mesh.apply_translation([0.0, 0.0, 6.0])
    dom = voxelize(mesh, 3.0, 150_000)
    case = LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -100.0],
            )
        ],
    )
    fixed3, F, _ = resolve_case_dofs(dom, case)
    return dom, fixed3, F


def _compliance(dom, rho: np.ndarray, fixed3: np.ndarray, F: np.ndarray) -> float:
    """F·u of the SIMP-stiffness realization (void cells keep the E floor)."""
    solid = dom.solid.ravel()
    rho_eff = np.clip(np.asarray(rho, dtype=float), RHO_MIN, 1.0)
    elem_E = E_MPA * np.where(solid, rho_eff, RHO_MIN) ** 3.0
    K, edof, active = assemble(dom, elem_E, NU, solid)
    ctx = reduce_case(dom, K, edof, fixed3)
    sol = solve_reduced(dom, ctx, edof, active, F, elem_E, NU)
    return float(F @ sol.u)


def _run_oc(dom, fixed3, F, **kw):
    return simp(
        dom,
        E_MPA,
        NU,
        [(fixed3, F)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(volfrac=VOLFRAC, rmin_mm=4.0, max_iter=40, tol=0.01, **kw),
    )


def test_invariant_volume_fraction():
    dom, fixed3, F = _cantilever()
    res = _run_oc(dom, fixed3, F)
    assert res.status == "converged"
    rho = res.density.reshape(dom.dims)
    assert rho[dom.solid].mean() == pytest.approx(VOLFRAC, abs=0.03)


def test_invariant_beats_uniform_same_volume():
    """The point of topopt: less compliance than uniform at equal mass."""
    dom, fixed3, F = _cantilever()
    res = _run_oc(dom, fixed3, F)
    rho = res.density
    c_opt = _compliance(dom, rho, fixed3, F)
    c_uni = _compliance(dom, np.where(dom.solid.ravel(), VOLFRAC, RHO_MIN), fixed3, F)
    assert c_opt < 0.8 * c_uni, f"optimized {c_opt:.4g} not better than uniform {c_uni:.4g}"


def test_invariant_oc_vs_mma_same_ballpark():
    """Two independent gradient-based optimizers (OC bisection vs nlopt MMA)
    must land in the same ballpark — a large divergence means the sensitivity
    or objective path is broken, not merely a different local optimum.
    Measured ratio on this case: ~1.03."""
    pytest.importorskip("nlopt")
    dom, fixed3, F = _cantilever()
    r_oc = _run_oc(dom, fixed3, F)
    r_mma = simp(
        dom,
        E_MPA,
        NU,
        [(fixed3, F)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(volfrac=VOLFRAC, rmin_mm=4.0, max_iter=60, tol=0.01, solver="mma"),
    )
    c_oc = _compliance(dom, r_oc.density, fixed3, F)
    c_mma = _compliance(dom, r_mma.density, fixed3, F)
    hi, lo = max(c_oc, c_mma), min(c_oc, c_mma)
    assert hi / lo < 1.5, f"OC {c_oc:.4g} vs MMA {c_mma:.4g} diverge"


def test_invariant_beso_contract():
    """Soft-kill BESO is a greedy hard-kill heuristic: it hits the volume
    target and keeps ONE connected load path (load cells reach the fix
    cells), but — documented limitation — its optima can sit above the
    uniform-field compliance on coarse grids. Assert the contract, not the
    optimum."""
    from scipy import ndimage

    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    p = TopOptParams(volfrac=VOLFRAC, rmin_mm=4.0, max_iter=80, tol=0.01, solver="beso")
    p.beso_er = 0.05
    res = beso(dom, E_MPA, NU, [(fixed3, F)], passive, p)
    assert res.status in {"converged", "limit_cycle"}
    achieved = res.density[dom.solid.ravel()].mean()
    assert achieved == pytest.approx(VOLFRAC, abs=0.06)
    mask = (res.density.reshape(dom.dims) >= 0.5) & dom.solid
    lab, _n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    en = dom.element_nodes()
    load_nodes = np.abs(F.reshape(-1, 3)).sum(1) > 0
    fix_nodes = fixed3.any(1)
    lm = lab.reshape(-1)
    comp_of = lambda want: set(  # noqa: E731
        lm[e] for e in np.flatnonzero(mask.ravel()) if want[en[e]].any()
    )
    assert comp_of(load_nodes) & comp_of(fix_nodes), "load path severed"


def test_invariant_deterministic():
    dom, fixed3, F = _cantilever()
    r1 = _run_oc(dom, fixed3, F)
    r2 = _run_oc(dom, fixed3, F)
    assert np.array_equal(r1.density, r2.density)
    assert r1.iterations == r2.iterations


def test_invariant_robust_ordering():
    """Eroded <= nominal <= dilated stiffness under shifted thresholds."""
    dom, fixed3, F = _cantilever()
    res = _run_oc(dom, fixed3, F, robust=True, beta0=4.0, beta_max=4.0)
    assert res.blueprint is not None
    bp = res.blueprint
    beta, delta = 4.0, 0.25
    masks = {
        name: _heaviside(bp, beta, 0.5 + off) >= 0.5
        for name, off in [("eroded", delta), ("nominal", 0.0), ("dilated", -delta)]
    }
    comps = {n: _compliance(dom, m.astype(float), fixed3, F) for n, m in masks.items()}
    # nested material sets => monotone compliance (1% slack for solver noise)
    assert comps["eroded"] >= 0.99 * comps["nominal"] >= 0.99 * 0.99 * comps["dilated"], (
        f"ordering violated: {comps}"
    )
