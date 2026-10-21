"""Extended FEM solver coverage: optional direct/CG backends and the
orthotropic FDM layer model.

scikit-sparse (CHOLMOD) and pyamg are OPTIONAL dependencies — the matching
tests skip via importorskip; the splu/SSOR fallbacks they replace stay
covered by test_fem.py. The orthotropic model (solver._D aniso
congruence-scaling) is exercised end-to-end on a z-compression column.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy import sparse

from cadx.cad.fem import solver
from cadx.cad.fem.materials import resolve_material
from cadx.cad.fem.regions import resolve_case
from cadx.cad.fem.solver import (
    _D,
    CholmodFactor,
    _amg_preconditioner,
    _elastic_preconditioner,
    _fixed_dofs,
    _free_node_coords,
    _ke_unit,
    assemble,
    factor_symmetric,
    iterative_preconditioner,
    reduce_case,
    solve,
    solve_reduced,
    solver_tier,
)
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _box_mesh(x: float = 60.0, y: float = 24.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])  # contract: rests on XY, centred
    return m


def _cantilever_case(force_n: float = -100.0) -> LoadCase:
    return LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, force_n],
            )
        ],
    )


def _column_case(force_n: float = -100.0) -> LoadCase:
    """Axial crush along z: bottom face fixed, -z load on the top face."""
    return LoadCase(
        name="crush",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+z", margin_mm=1.0),
                force_n=[0.0, 0.0, force_n],
            )
        ],
    )


def _reduced_system(dom, fixed: np.ndarray) -> tuple:
    """(Kf, free) reduced system for a fully-assembled uniform-E domain."""
    K, edof, _active = assemble(dom, np.full(dom.cell_count, E_PLA), NU)
    fixed_dofs = np.unique(_fixed_dofs(fixed, dom.node_count))
    used = np.unique(edof.ravel())
    free = used[~np.isin(used, fixed_dofs, assume_unique=True)]
    return K[free][:, free].tocsr(), free


# --------------------------------------------------------------------------- #
# CHOLMOD direct-solve tier                                                    #
# --------------------------------------------------------------------------- #


def test_solver_tier_env(monkeypatch) -> None:
    monkeypatch.setenv("CADX_FEM_DIRECT_SOLVER", "splu")
    assert solver_tier() == "splu"
    monkeypatch.setenv("CADX_FEM_DIRECT_SOLVER", "auto")
    assert solver_tier() == ("cholmod" if solver._cholmod_available() else "splu")
    # explicit backend arg overrides the env (topopt's solver_backend knob)
    assert solver_tier("superlu") == "splu"
    assert solver_tier("cholmod") == (
        "cholmod" if solver._cholmod_available() else "splu"
    )


def test_backend_vocabulary_contract() -> None:
    """fem_solver_backend = auto|cholmod|superlu|amg|ssor: the direct names
    select a factorization, amg/ssor force the iterative CG path."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, _F, _ = resolve_case(dom, _cantilever_case())
    Kf, _free = _reduced_system(dom, fixed)
    assert factor_symmetric(Kf, backend="amg") is None
    assert factor_symmetric(Kf, backend="ssor") is None
    assert callable(factor_symmetric(Kf, backend="superlu"))
    assert iterative_preconditioner(Kf, backend="ssor") is not None


def test_cholmod_factor_matches_splu(monkeypatch) -> None:
    """With scikit-sparse installed, CHOLMOD and splu(SymmetricMode) solve
    the same reduced stiffness matrix to ~1e-9."""
    pytest.importorskip("sksparse.cholmod")
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, F, _ = resolve_case(dom, _cantilever_case())
    Kf, free = _reduced_system(dom, fixed)
    Ff = F[free]

    monkeypatch.setenv("CADX_FEM_DIRECT_SOLVER", "cholmod")
    f_chol = factor_symmetric(Kf)
    assert isinstance(f_chol, CholmodFactor)
    monkeypatch.setenv("CADX_FEM_DIRECT_SOLVER", "splu")
    f_splu = factor_symmetric(Kf)
    assert callable(f_splu) and not isinstance(f_splu, CholmodFactor)

    np.testing.assert_allclose(
        np.asarray(f_chol(Ff)), np.asarray(f_splu(Ff)), rtol=1e-9, atol=1e-9
    )


def test_cholmod_refactorize_density_rescale() -> None:
    """refactorize must re-solve a value-scaled K (same pattern — the SIMP
    density update shape) and refuse a mismatched pattern."""
    cholmod = pytest.importorskip("sksparse.cholmod")
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, F, _ = resolve_case(dom, _cantilever_case())
    Kf, free = _reduced_system(dom, fixed)
    Ff = F[free]

    kf = Kf.tocsc()
    if hasattr(cholmod, "CholeskyFactor"):
        # upstream API: symbolic at construction, factorize() for numerics
        raw = cholmod.CholeskyFactor(kf)
        raw.factorize(kf)
    else:
        raw = cholmod.cholesky(kf)
    fac = CholmodFactor(raw, kf)
    x0 = np.asarray(fac(Ff))
    res = Kf @ x0 - Ff
    assert np.linalg.norm(res) <= 1e-8 * max(np.linalg.norm(Ff), 1.0)

    K2 = Kf.copy()
    K2.data *= 0.63  # density rescale: identical pattern, different values
    assert fac.refactorize(K2)
    x1 = np.asarray(fac.solve(Ff))
    res = K2 @ x1 - Ff
    assert np.linalg.norm(res) <= 1e-8 * max(np.linalg.norm(Ff), 1.0)
    assert not np.allclose(x0, x1, rtol=1e-4)

    # mismatched pattern/shape is refused — the caller refactors fresh
    assert not fac.refactorize(K2[:-1, :-1])
    K3 = sparse.triu(K2, k=1).tocsr()  # drops the diagonal -> fewer nnz
    assert not fac.refactorize(K3)


def test_factor_symmetric_auto_backend_solves() -> None:
    """Whatever tier `auto` resolved to, the returned callable solves."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, F, _ = resolve_case(dom, _cantilever_case())
    Kf, free = _reduced_system(dom, fixed)
    factor = factor_symmetric(Kf)
    assert callable(factor)
    x = np.asarray(factor(F[free]))
    res = Kf @ x - F[free]
    assert np.linalg.norm(res) <= 1e-8 * max(np.linalg.norm(F[free]), 1.0)


# --------------------------------------------------------------------------- #
# PyAMG preconditioned CG                                                      #
# --------------------------------------------------------------------------- #


def test_free_node_coords_match_domain_grid() -> None:
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    rng = np.random.default_rng(0)
    nodes = np.sort(rng.choice(dom.node_count, size=200, replace=False))
    free = (3 * nodes[:, None] + np.arange(3)).ravel()
    np.testing.assert_allclose(_free_node_coords(dom, free), dom.node_coords()[nodes])


def test_amg_preconditioned_cg_matches_direct(monkeypatch) -> None:
    """spsolve_dofs=0 forces the CG path; the AMG preconditioner must be
    built and the solution must match a direct solve on the same system."""
    pytest.importorskip("pyamg")
    monkeypatch.setenv("CADX_FEM_CG_PRECONDITIONER", "amg")
    dom = voxelize(_box_mesh(), 4.0, 150_000)
    fixed, F, _ = resolve_case(dom, _cantilever_case())
    elem_E = np.full(dom.cell_count, E_PLA)
    K, edof, active = assemble(dom, elem_E, NU)

    ctx = reduce_case(dom, K, edof, fixed, spsolve_dofs=0)
    assert ctx.factor is None and ctx.M is not None
    coords = _free_node_coords(dom, ctx.free)
    assert _amg_preconditioner(ctx.Kf, coords) is not None
    # the elastic helper picked AMG for the CG context
    assert _elastic_preconditioner(ctx.Kf, dom, ctx.free) is not None

    sol_cg = solve_reduced(dom, ctx, edof, active, F, elem_E, NU)
    assert sol_cg.solver == "cg" and sol_cg.converged

    sol_ref = solve(dom, elem_E, NU, fixed, F)  # small domain -> direct path
    scale = max(np.linalg.norm(sol_ref.u), 1e-30)
    assert np.linalg.norm(sol_cg.u - sol_ref.u) / scale < 1e-3


# --------------------------------------------------------------------------- #
# orthotropic (transversely-layered) material                                  #
# --------------------------------------------------------------------------- #


def test_D_aniso_k1_is_bit_identical() -> None:
    np.testing.assert_array_equal(_D(NU), _D(NU, 1.0, 2))
    np.testing.assert_array_equal(_ke_unit(NU, 2.0), _ke_unit(NU, 2.0, 1.0, 2))


def test_D_aniso_scales_axis_terms() -> None:
    """S @ D_iso @ S with k on {zz, yz, zx}: axis-normal + axis shears scale
    by k, the in-plane block is untouched, the matrix stays SPD."""
    k = 0.4
    D = _D(NU, k, 2)
    Di = _D(NU)
    assert D[2, 2] == pytest.approx(k * Di[2, 2])
    assert D[4, 4] == pytest.approx(k * Di[4, 4])  # yz shear
    assert D[5, 5] == pytest.approx(k * Di[5, 5])  # zx shear
    assert D[0, 0] == Di[0, 0]
    assert D[1, 1] == Di[1, 1]
    assert D[3, 3] == Di[3, 3]  # xy shear does not touch z
    np.testing.assert_array_less(0.0, np.linalg.eigvalsh(D))


def test_orthotropic_column_weaker_along_axis() -> None:
    """Axial z-compression: the compliance along the axis grows exactly 1/k
    (E_z -> k·E), so k=0.5 roughly doubles the crush; weakening the in-plane
    x axis leaves the z-axial response ~unchanged."""
    dom = voxelize(_box_mesh(12.0, 12.0, 60.0), 3.0, 150_000)
    fixed, F, _ = resolve_case(dom, _column_case())
    elem_E = np.full(dom.cell_count, E_PLA)
    sol_iso = solve(dom, elem_E, NU, fixed, F)
    sol_z = solve(dom, elem_E, NU, fixed, F, aniso=(0.5, 2.0))
    sol_x = solve(dom, elem_E, NU, fixed, F, aniso=(0.5, 0.0))
    assert sol_iso.converged and sol_z.converged
    assert sol_z.max_disp_mm > 1.5 * sol_iso.max_disp_mm
    assert sol_x.max_disp_mm == pytest.approx(sol_iso.max_disp_mm, rel=0.1)


def test_material_aniso_overrides() -> None:
    base = resolve_material("pla")
    assert base.aniso_z == 1.0 and base.aniso_axis == 2
    m = resolve_material("pla", aniso_z=0.55, aniso_axis=0)
    assert m.aniso_z == 0.55 and m.aniso_axis == 0
    assert m.elastic_modulus_mpa == base.elastic_modulus_mpa
    assert m.density_g_cm3 == base.density_g_cm3
    # the isotropic default override lands back on the table entry
    assert resolve_material("pla", aniso_z=1.0) == base
