"""Per-class constitutive tensors in the FEM kernel (`cad/fem/solver.py`).

`elem_class`/`class_D` give each element its own constitutive matrix —
material zones (own Poisson ratio) and homogenized lattice cells
(effective C_H). The legacy single-D path under-reads a stiffer zone's
stress by its E ratio (~60x for a Stahl insert in PLA); the class path
must stay byte-neutral for single-material models.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadx.cad.fem.regions import material_cell_classes
from cadx.cad.fem.solver import (
    _D,
    _fixed_dofs,
    _ke_from_D,
    _ke_unit,
    assemble,
    recover_elem_stress,
    reduce_case,
    solve,
    solve_reduced,
)
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.llm.schemas import RegionSpec

NU = 0.36


def _bar(nx: int = 16, ny: int = 4, nz: int = 4, pitch: float = 1.0) -> VoxelDomain:
    return VoxelDomain(
        origin=np.zeros(3),
        pitch=pitch,
        dims=(nx, ny, nz),
        solid=np.ones((nx, ny, nz), dtype=bool),
    )


def _axial_case(dom: VoxelDomain, f_total: float) -> tuple[np.ndarray, np.ndarray]:
    """Pin the -x face (all 3 DOFs), spread `f_total` over the +x face
    nodes in +x — the series-bar setup every stress test here shares."""
    coords = dom.node_coords()
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    fixed3 = np.repeat(np.isclose(coords[:, 0], lo[0])[:, None], 3, axis=1)
    tip = np.isclose(coords[:, 0], hi[0])
    F = np.zeros(3 * dom.node_count)
    F.reshape(-1, 3)[tip, 0] = f_total / float(tip.sum())
    return fixed3, F


def _solve_bar(dom, elem_E, fixed3, F, nu=NU, **kw):
    K, edof, active = assemble(dom, elem_E, nu, **kw)
    ctx = reduce_case(dom, K, edof, fixed3)
    sol = solve_reduced(dom, ctx, edof, active, F, elem_E, nu, **kw)
    assert sol.converged
    return sol, edof, active


# --------------------------------------------------------------------------- #
# _ke_from_D / single-class parity                                            #
# --------------------------------------------------------------------------- #


def test_ke_from_D_matches_ke_unit() -> None:
    """The extracted Gauss loop must reproduce `_ke_unit` exactly —
    bit-for-bit on iso and aniso tensors."""
    for pitch in (1.0, 2.5):
        np.testing.assert_array_equal(
            _ke_from_D(_D(NU), pitch), _ke_unit(NU, pitch)
        )
        np.testing.assert_array_equal(
            _ke_from_D(_D(NU, 0.5, 2), pitch), _ke_unit(NU, pitch, 0.5, 2)
        )


def test_single_class_table_matches_legacy() -> None:
    """A one-entry class table on a uniform-E domain is the legacy solve:
    identical K, identical displacement, same VM to round-off."""
    dom = _bar(12, 4, 4)
    elem_E = np.where(dom.solid.ravel(), 3500.0, 0.0)
    fixed3, F = _axial_case(dom, 800.0)

    K0, _, _ = assemble(dom, elem_E, NU)
    ec = np.zeros(dom.cell_count, dtype=np.int64)
    cd = _D(NU)[None]
    K1, _, _ = assemble(dom, elem_E, NU, elem_class=ec, class_D=cd)
    assert (K0 - K1).nnz == 0  # same pattern AND same values

    sol0, edof, active = _solve_bar(dom, elem_E, fixed3, F)
    sol1, _, _ = _solve_bar(dom, elem_E, fixed3, F, elem_class=ec, class_D=cd)
    np.testing.assert_array_equal(sol0.u, sol1.u)
    np.testing.assert_allclose(sol0.elem_vm, sol1.elem_vm, rtol=1e-12, atol=1e-9)

    # stress recovery: legacy single-D vs class path, same field
    vm0, sig0 = recover_elem_stress(dom, sol0.u, edof, active, elem_E, NU)
    vm1, sig1 = recover_elem_stress(
        dom, sol0.u, edof, active, elem_E, NU, elem_class=ec, class_D=cd
    )
    np.testing.assert_allclose(vm0, vm1, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(sig0, sig1, rtol=1e-12, atol=1e-9)


def test_assemble_validates_class_inputs() -> None:
    """Class indices must cover every cell in [0, ncls) — a stray -1 or an
    out-of-range id is a hard error, not a silent wrap into the table."""
    dom = _bar(4, 2, 2)
    elem_E = np.ones(dom.cell_count)
    with pytest.raises(ValueError, match="elem_class"):
        assemble(dom, elem_E, NU, class_D=_D(NU)[None])  # missing elem_class
    bad = np.zeros(dom.cell_count, dtype=np.int64)
    bad[0] = -1
    with pytest.raises(ValueError, match="elem_class"):
        assemble(dom, elem_E, NU, elem_class=bad, class_D=_D(NU)[None])
    bad[0] = 5
    with pytest.raises(ValueError, match="elem_class"):
        assemble(dom, elem_E, NU, elem_class=bad, class_D=_D(NU)[None])


# --------------------------------------------------------------------------- #
# two-material series bar                                                     #
# --------------------------------------------------------------------------- #


def test_two_material_series_bar_recovers_stress() -> None:
    """Soft|stiff bar under axial load: sigma_xx = F/A in BOTH zones.

    The per-class path applies each element's own E_e — a single shared
    constitutive matrix scaled by one reference E mis-predicts the stiff
    zone by exactly the E ratio (the 60x Stahl-insert regression)."""
    dom = _bar(16, 4, 4)
    E_soft, E_stiff = 2_000.0, 200_000.0
    c3 = np.indices(dom.dims)[0].ravel()
    zone1 = c3 >= 8  # +x half is the stiff zone
    elem_E = np.where(zone1, E_stiff, E_soft)
    elem_class = zone1.astype(np.int64)
    class_D = np.stack([_D(NU), _D(NU)])  # same nu — isolates the E factor

    f = 1_600.0
    fixed3, F = _axial_case(dom, f)
    sol, edof, active = _solve_bar(
        dom, elem_E, fixed3, F, elem_class=elem_class, class_D=class_D
    )
    _vm, sig = recover_elem_stress(
        dom, sol.u, edof, active, elem_E, NU,
        elem_class=elem_class, class_D=class_D,
    )
    sxx = sig[:, 0]
    expected = f / (4.0 * 4.0)  # 100 MPa
    # interior core of each zone — away from BCs and the interface singularity
    cc = np.indices(dom.dims).reshape(3, -1).T[active]  # (nact,3) x,y,z cell ids
    zc = cc[:, 0]
    core = (cc[:, 1] >= 1) & (cc[:, 1] <= 2) & (cc[:, 2] >= 1) & (cc[:, 2] <= 2)
    m0 = core & (zc >= 2) & (zc <= 5)
    m1 = core & (zc >= 10) & (zc <= 13)
    assert m0.sum() >= 8 and m1.sum() >= 8
    assert sxx[m0].mean() == pytest.approx(expected, rel=0.05)
    assert sxx[m1].mean() == pytest.approx(expected, rel=0.05)

    # series-spring check: tip deflection matches the analytical two-zone
    # compliance delta = L/A * (1/E_soft + 1/E_stiff) * F/2... per half-length
    area = 4.0 * 4.0
    delta_exact = (f / area) * (8.0 / E_soft + 8.0 / E_stiff)
    assert sol.max_disp_mm == pytest.approx(delta_exact, rel=0.10)

    # regression guard: the same displacement through the legacy single-D
    # recovery (scaled by the BASE modulus) under-reads the stiff zone by
    # ~E_stiff/E_soft — the bug this table exists to fix
    _vm_l, sig_l = recover_elem_stress(dom, sol.u, edof, active, elem_E, NU, E_ref=E_soft)
    assert sig_l[m1, 0].mean() == pytest.approx(expected * E_soft / E_stiff, rel=0.1)
    assert sxx[m1].mean() / sig_l[m1, 0].mean() == pytest.approx(
        E_stiff / E_soft, rel=0.1
    )


# --------------------------------------------------------------------------- #
# Poisson-ratio stress regression (hydrostatic KUBC probe)                    #
# --------------------------------------------------------------------------- #


_EPS0 = 0.01


def _hydrostatic_solve(dom, elem_E, nu_solve=NU, **kw):
    """Solve the KUBC hydrostatic probe: every boundary node prescribed
    u = eps0 * x (unit-E cells -> analytic sigma_ii = eps0/(1-2*nu))."""
    K, edof, active = assemble(dom, elem_E, nu_solve, **kw)
    coords = dom.node_coords()
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    on_bnd = np.zeros(dom.node_count, dtype=bool)
    for a in range(3):
        on_bnd |= np.isclose(coords[:, a], lo[a]) | np.isclose(coords[:, a], hi[a])
    fixed3 = np.repeat(on_bnd[:, None], 3, axis=1)
    ctx = reduce_case(dom, K, edof, fixed3)
    pdofs = _fixed_dofs(fixed3, dom.node_count)
    u_p = _EPS0 * coords.reshape(-1)
    F = np.zeros(3 * dom.node_count)
    F[ctx.free] = -(K[ctx.free][:, pdofs] @ u_p[pdofs])
    sol = solve_reduced(dom, ctx, edof, active, F, elem_E, nu_solve, **kw)
    u = sol.u.copy()
    u[pdofs] = u_p[pdofs]
    return u, edof, active


def test_poisson_ratio_classes_hydrostatic() -> None:
    """Uniform hydrostatic strain eps0*I: sigma_ii = E*eps0/(1-2*nu).

    A cell whose elements all sit in class c must recover the analytic
    stress of class c's tensor exactly — nu=0.30 vs nu=0.49 differs ~20x.
    The single-D recovery hands every element the base nu and misses it."""
    dom = _bar(4, 4, 4)
    nu0, nu1 = 0.30, 0.49
    class_D = np.stack([_D(nu0), _D(nu1)])
    elem_E = np.ones(dom.cell_count)  # unit E -> sigma = eps0/(1-2nu)

    for cls_id, nu_c in ((0, nu0), (1, nu1)):
        elem_class = np.full(dom.cell_count, cls_id, dtype=np.int64)
        u, edof, active = _hydrostatic_solve(
            dom, elem_E, elem_class=elem_class, class_D=class_D
        )
        _vm, sig = recover_elem_stress(
            dom, u, edof, active, elem_E, NU,
            elem_class=elem_class, class_D=class_D,
        )
        # the affine field is the exact solution -> every element's
        # centroid stress equals the analytic hydrostatic value
        np.testing.assert_allclose(sig[:, :3], _EPS0 / (1.0 - 2.0 * nu_c), rtol=1e-9)
        np.testing.assert_allclose(sig[:, 3:], 0.0, atol=1e-12)


def test_poisson_ratio_classes_two_zone_split() -> None:
    """Same probe on a real two-zone cell: the near-incompressible zone
    (nu=0.49) carries far higher transverse stress than the nu=0.30 zone
    — the class path must show that split; a shared-D recovery collapses
    it (the syy column under-reads by an order of magnitude)."""
    dom = _bar(12, 6, 6)
    x = np.indices(dom.dims)[0].ravel()
    elem_class = (x >= 6).astype(np.int64)  # +x half is the near-rubber zone
    nu0, nu1 = 0.30, 0.49
    class_D = np.stack([_D(nu0), _D(nu1)])
    elem_E = np.ones(dom.cell_count)

    u, edof, active = _hydrostatic_solve(
        dom, elem_E, elem_class=elem_class, class_D=class_D
    )
    _vm, sig = recover_elem_stress(
        dom, u, edof, active, elem_E, NU, elem_class=elem_class, class_D=class_D
    )
    cls_a = elem_class[active]
    # the split is mostly hydrostatic — it shows in the transverse
    # components (syy=szz by symmetry), not in von Mises
    assert sig[cls_a == 1, 1].mean() > 4.0 * sig[cls_a == 0, 1].mean()

    # regression: one shared D (base nu) collapses the zone-1 transverse
    # stress — the class recovery reports ~14x more
    _vm_l, sig_l = recover_elem_stress(dom, u, edof, active, elem_E, nu0)
    assert sig[cls_a == 1, 1].mean() > 8.0 * sig_l[cls_a == 1, 1].mean()


# --------------------------------------------------------------------------- #
# material_cell_classes                                                       #
# --------------------------------------------------------------------------- #


def test_material_cell_classes_maps_zones() -> None:
    """A keep-region with `material` becomes one constitutive class; cells
    outside stay -1 (resolved to the base class by the caller). Distinct
    (E, nu) pairs dedupe — two regions with the same material share one."""
    dom = _bar(8, 8, 8)
    zc, _zc2 = material_cell_classes(dom, [], None)
    assert zc is None  # no material zones -> no classes

    steel = RegionSpec(
        kind="box", center=[2.0, 4.0, 4.0], diameter_mm=4.0, material="steel"
    )
    pla_zone = RegionSpec(
        kind="box", center=[6.0, 4.0, 4.0], diameter_mm=4.0, material="petg"
    )
    zc, cds = material_cell_classes(dom, [steel, pla_zone], None)
    assert zc is not None and zc.shape == (dom.cell_count,)
    assert zc.dtype == np.int32
    assert set(np.unique(zc)) == {-1, 0, 1}
    assert len(cds) == 2
    np.testing.assert_allclose(cds[0], _D(0.30))  # steel
    np.testing.assert_allclose(cds[1], _D(0.38))  # petg
    # class assignment lands on the declared boxes
    centers = dom.element_centers()
    in_steel = zc == 0
    assert in_steel.any()
    assert np.all(centers[in_steel, 0] < 4.0)
    in_petg = zc == 1
    assert np.all(centers[in_petg, 0] > 4.0)

    # dedupe: two steel boxes share class 0
    steel2 = RegionSpec(
        kind="box", center=[6.0, 4.0, 4.0], diameter_mm=4.0, material="stahl"
    )  # alias of steel
    zc2, cds2 = material_cell_classes(dom, [steel, steel2], None)
    assert len(cds2) == 1
    assert set(np.unique(zc2)) == {-1, 0}


# --------------------------------------------------------------------------- #
# solve() passthrough                                                         #
# --------------------------------------------------------------------------- #


def test_solve_passthrough_matches_manual_path() -> None:
    """`solve` must forward elem_class/class_D to assemble+solve_reduced:
    the convenience wrapper's answer is identical to the explicit path."""
    dom = _bar(8, 4, 4)
    x = np.indices(dom.dims)[0].ravel()
    elem_class = (x >= 4).astype(np.int64)
    class_D = np.stack([_D(0.30), _D(0.45)])
    elem_E = np.where(x >= 4, 210_000.0, 3_500.0)
    fixed3, F = _axial_case(dom, 500.0)

    sol_api = solve(
        dom, elem_E, NU, fixed3, F, elem_class=elem_class, class_D=class_D
    )
    sol_man, _, _ = _solve_bar(
        dom, elem_E, fixed3, F, elem_class=elem_class, class_D=class_D
    )
    assert sol_api.converged
    np.testing.assert_allclose(sol_api.u, sol_man.u, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        sol_api.elem_vm, sol_man.elem_vm, rtol=1e-12, atol=1e-9
    )
