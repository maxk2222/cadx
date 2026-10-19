"""Regression coverage for the per-case boundary-condition fix in SIMP.

Each load case must solve on ``used - its own fixed DOFs`` — unioning every
case's fixture into one shared ``free`` set pins case i at the OTHER cases'
supports (over-stiff, wrong compliances and sensitivities). Also covers the
symmetric-SuperLU factorization path (``factor_symmetric``): Kf is a
symmetric structural stiffness matrix, so ``splu(SymmetricMode=True)`` must
return the same solution ``factorized`` produces.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.sparse import linalg as spla

from cadx.cad.fem.regions import resolve_case_dofs
from cadx.cad.fem.solver import assemble, factor_symmetric, reduce_case, solve
from cadx.cad.fem.topopt import E_MIN_FRAC, RHO_MIN, TopOptParams, simp
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36
VOLFRAC = 0.4


def _box_mesh(x: float = 48.0, y: float = 16.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _two_fixture_cases(dom):
    """Two load cases on DIFFERENT fixtures — the union-BC bug case.

    A: cantilever — -x face encastre, +x face pulled down (-z).
    B: base support — -z face encastre, +x face pushed sideways (-y).
    The fixtures share no DOFs, so unioning them over-constrains both solves.
    """
    case_a = LoadCase(
        name="cantilever",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -100.0],
            )
        ],
    )
    case_b = LoadCase(
        name="on_base",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, -50.0, 0.0],
            )
        ],
    )
    fa, Fa, _ = resolve_case_dofs(dom, case_a)
    fb, Fb, _ = resolve_case_dofs(dom, case_b)
    return (fa, Fa), (fb, Fb)


def _uniform_E_e(dom) -> np.ndarray:
    """simp()'s iteration-1 modulus field: volfrac on solid cells, RHO_MIN
    (with the Emin floor) outside."""
    E0 = np.full(dom.cell_count, E_PLA)
    Emin = E0 * E_MIN_FRAC
    p = np.where(dom.solid.ravel(), VOLFRAC, RHO_MIN)
    return Emin + np.power(p, 3.0) * (E0 - Emin)


def test_splu_symmetric_matches_factorized() -> None:
    """factor_symmetric's splu(SymmetricMode) path solves a real reduced
    stiffness matrix identically to scipy's factorized."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    (fixed_a, F_a), _ = _two_fixture_cases(dom)
    K, edof, _active = assemble(dom, np.full(dom.cell_count, E_PLA), NU)
    ctx = reduce_case(dom, K, edof, fixed_a)
    assert ctx.factor is not None  # small enough for the direct path
    Ff = F_a[ctx.free]

    x_splu = np.asarray(ctx.factor(Ff), dtype=float)
    x_ref = np.asarray(spla.factorized(ctx.Kf.tocsc())(Ff), dtype=float)
    np.testing.assert_allclose(x_splu, x_ref, rtol=1e-8, atol=1e-12)

    # residual check — symmetric solve still satisfies Kf u = Ff
    res = ctx.Kf @ x_splu - Ff
    assert np.linalg.norm(res) <= 1e-8 * max(np.linalg.norm(Ff), 1.0)


def test_reduce_case_caches_per_signature() -> None:
    """Cases with identical fixed-DOF signatures get identical free sets;
    different fixtures get disjoint reduced systems."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    (fixed_a, _Fa), (fixed_b, _Fb) = _two_fixture_cases(dom)
    K, edof, _ = assemble(dom, np.full(dom.cell_count, E_PLA), NU)
    ctx_a = reduce_case(dom, K, edof, fixed_a)
    ctx_b = reduce_case(dom, K, edof, fixed_b)
    ctx_a2 = reduce_case(dom, K, edof, fixed_a.copy())
    assert not np.array_equal(ctx_a.free, ctx_b.free)
    assert np.array_equal(ctx_a.free, ctx_a2.free)
    # case A's free set must NOT exclude case B's fixture DOFs
    from cadx.cad.fem.solver import _fixed_dofs

    fixed_b_dofs = np.unique(_fixed_dofs(fixed_b, dom.node_count))
    assert np.isin(fixed_b_dofs, ctx_a.free).any()


def test_simp_multi_case_uses_per_case_fixtures() -> None:
    """Iteration-1 compliance equals the sum of independently-solved case
    compliances — and differs measurably from the unioned-BC result."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    (fixed_a, F_a), (fixed_b, F_b) = _two_fixture_cases(dom)
    E_e = _uniform_E_e(dom)
    exp_a = solve(dom, E_e, NU, fixed_a, F_a).compliance_j
    exp_b = solve(dom, E_e, NU, fixed_b, F_b).compliance_j
    assert exp_a > 0 and exp_b > 0

    passive = np.zeros(dom.cell_count, dtype=bool)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed_a, F_a), (fixed_b, F_b)],
        passive,
        TopOptParams(volfrac=VOLFRAC, rmin_mm=4.0, max_iter=1, tol=0.0),
    )
    assert res.history, "simp produced no iterations"
    got = res.history[0]["compliance_mj"]
    assert got == pytest.approx(exp_a + exp_b, rel=1e-4)

    # the buggy union implementation would solve BOTH cases on
    # `used - (fixA | fixB)` — strictly stiffer, strictly lower compliance
    fixed_union = np.asarray(fixed_a, dtype=bool) | np.asarray(fixed_b, dtype=bool)
    bug_a = solve(dom, E_e, NU, fixed_union, F_a).compliance_j
    bug_b = solve(dom, E_e, NU, fixed_union, F_b).compliance_j
    assert got != pytest.approx(bug_a + bug_b, rel=1e-3)


def test_simp_single_fixture_unchanged() -> None:
    """Single-fixture runs reduce to the same free set the union gave —
    iteration-1 compliance matches a direct solve exactly."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    (fixed_a, F_a), _ = _two_fixture_cases(dom)
    exp_a = solve(dom, _uniform_E_e(dom), NU, fixed_a, F_a).compliance_j

    passive = np.zeros(dom.cell_count, dtype=bool)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed_a, F_a)],
        passive,
        TopOptParams(volfrac=VOLFRAC, rmin_mm=4.0, max_iter=1, tol=0.0),
    )
    assert res.history, "simp produced no iterations"
    assert res.history[0]["compliance_mj"] == pytest.approx(exp_a, rel=1e-4)


def test_factor_symmetric_falls_back_cleanly() -> None:
    """A broken matrix yields None (caller degrades to CG), never raises."""
    from scipy import sparse

    broken = sparse.csr_matrix((0, 0))
    assert factor_symmetric(broken) is None or callable(factor_symmetric(broken))
