"""Helmholtz PDE density filter (Lazarov & Sigmund 2011) in the SIMP loop
(`cad/fem/topopt.py`): the cone filter's distance matrix is replaced by a
sparse SPD system L = I - r²Δ on the solid cells with homogeneous Neumann
boundary conditions — the filtered length scale stays consistent right up
to the domain boundary, where the clipped/renormalized cone filter biases
members. Tests cover the discrete operator, the finite-difference gradient
of the projection chain, boundary consistency vs the cone filter, and
combination with the Heaviside/robust formulations.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.regions import resolve_case
from cadx.cad.fem.topopt import (
    _ETA,
    RHO_MIN,
    TopOptParams,
    _build_filter_ops,
    _helmholtz_operator,
    _Projector,
    passive_solid_mask,
    simp,
)
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _block(w: float = 20.0, h: float = 8.0, d: float = 8.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[w, d, h])
    m.apply_translation([0.0, 0.0, h / 2])
    return m


def _tip_case():
    return LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -30.0],
            )
        ],
    )


def _setup(dom):
    fixed, F, _ = resolve_case(dom, _tip_case(), [])
    return fixed, F, passive_solid_mask(dom, [], [fixed], [])


def _projector(dom, kind: str, *, heaviside: bool, beta: float = 2.0, rmin_cells=3.0):
    """The same filter/projection chain simp() builds — exercises the real
    _Projector/_FilterOps code path instead of a reimplementation."""
    solid = dom.solid.ravel()
    solid_idx = np.flatnonzero(solid)
    fops = _build_filter_ops(dom, solid_idx, rmin_cells, kind)
    return _Projector(
        fops=fops,
        solid=solid,
        solid_idx=solid_idx,
        passive_solid=np.zeros(dom.cell_count, dtype=bool),
        pvoid=np.zeros(dom.cell_count, dtype=bool),
        am_layers=None,
        heaviside=heaviside,
        beta=beta,
    )


# --------------------------------------------------------------------------- #
# Operator                                                                    #
# --------------------------------------------------------------------------- #


def test_helmholtz_operator_spd_and_constant_preserving():
    """L = I - r²Δ must be symmetric, strictly diagonally dominant (SPD) and
    satisfy L·1 = 1 EXACTLY — the Neumann drop of missing-neighbour terms is
    what makes constant fields survive the filter at the domain boundary."""
    dom = voxelize(_block(w=16.0, h=10.0, d=10.0), 1.0, 200_000)
    ns = int(dom.solid.sum())
    r = 3.0 / (2.0 * np.sqrt(3.0))
    L = _helmholtz_operator(dom, r)
    assert L.shape == (ns, ns)
    assert abs(L - L.T).max() == 0.0
    diag, off = L.diagonal(), abs(L).sum(axis=1).A.ravel() - L.diagonal()
    assert (diag - off >= 1.0 - 1e-12).all()  # strictly dominant -> SPD
    # interior cells count 6 solid neighbours, boundary cells fewer
    assert diag.max() == pytest.approx(1.0 + 6 * r * r)
    assert diag.min() == pytest.approx(1.0 + 3 * r * r)  # corner cell: 3
    ones = np.ones(ns)
    np.testing.assert_allclose(L @ ones, ones, atol=1e-12)  # L·1 = 1


def test_helmholtz_apply_preserves_constants_everywhere():
    """x̃ = L⁻¹·1 = 1 to solver precision — including boundary cells, where
    the cone filter's clipped kernel has fewer neighbours to average."""
    dom = voxelize(_block(w=14.0, h=8.0, d=8.0), 1.0, 200_000)
    sidx = np.flatnonzero(dom.solid.ravel())
    fops = _build_filter_ops(dom, sidx, 3.0, "helmholtz")
    xt = fops.apply(np.ones(sidx.size))
    np.testing.assert_allclose(xt, 1.0, atol=1e-10)


def test_filter_ops_adjoint_is_self_adjoint():
    """<apply(a), b> == <a, adjoint(b)> for both filters — required for the
    sensitivity chain rule to use a single factorization."""
    dom = voxelize(_block(w=12.0, h=6.0, d=6.0), 1.0, 200_000)
    sidx = np.flatnonzero(dom.solid.ravel())
    rng = np.random.default_rng(0)
    a, b = rng.random(sidx.size), rng.random(sidx.size)
    for kind in ("cone", "helmholtz"):
        fops = _build_filter_ops(dom, sidx, 3.0, kind)
        assert float(fops.apply(a) @ b) == pytest.approx(float(a @ fops.adjoint(b)))


# --------------------------------------------------------------------------- #
# Finite-difference gradient (chain-rule regression)                          #
# --------------------------------------------------------------------------- #


def test_helmholtz_heaviside_chain_finite_difference():
    """Analytic d/dx of  f(x) = wᵀ p(x)  through the helmholtz+heaviside
    _Projector chain vs centred finite differences — the self-adjointness of
    L⁻¹ is what the chain rule relies on, so any asymmetry or missed Hs
    normalization shows up here. Relative error must stay < 1e-4."""
    dom = voxelize(_block(w=10.0, h=6.0, d=5.0), 1.0, 200_000)
    solid = dom.solid.ravel()
    sidx = np.flatnonzero(solid)
    proj = _projector(dom, "helmholtz", heaviside=True, beta=2.0)
    rng = np.random.default_rng(7)
    xb = np.full(dom.cell_count, RHO_MIN)
    xb[sidx] = rng.uniform(0.05, 0.95, sidx.size)
    w = np.zeros(dom.cell_count)
    w[sidx] = rng.uniform(0.1, 1.0, sidx.size)

    def f(v):
        return float(w @ proj.project(v)[1])

    xp, p, xt = proj.project(xb)
    ga = proj.chain(w, xp, p, xt, xb, oc_filter=False, eta=_ETA)[sidx]

    eps = 1e-6
    fd = np.empty(sidx.size)
    for j, cell in enumerate(sidx):
        vp = xb.copy()
        vp[cell] += eps
        vm = xb.copy()
        vm[cell] -= eps
        fd[j] = (f(vp) - f(vm)) / (2 * eps)
    scale = np.maximum(np.abs(ga), np.abs(fd))
    rel = np.abs(fd - ga) / np.clip(scale, 1e-3, None)
    assert float(rel.max()) < 1e-4, f"FD mismatch: max rel err {rel.max():.2e}"


def test_helmholtz_chain_vs_cone_same_structure():
    """Sanity: on a SMOOTH field cone- and helmholtz-filtered fields agree
    (the kernels share length scale; they only differ where the cone ball
    is clipped / in high-frequency content). L is an M-matrix
    (diag > 0, off-diag <= 0, dominant) → L⁻¹ >= 0 and L⁻¹·1 = 1, so the
    PDE filter can never overshoot the input range — no ringing."""
    dom = voxelize(_block(w=16.0, h=8.0, d=8.0), 1.0, 200_000)
    sidx = np.flatnonzero(dom.solid.ravel())
    fc = _build_filter_ops(dom, sidx, 3.0, "cone")
    fh = _build_filter_ops(dom, sidx, 3.0, "helmholtz")
    # smooth low-frequency field -> same filtered shape
    xs = 0.5 + 0.4 * np.sin(np.linspace(0.0, np.pi, sidx.size) * 3.0)
    assert np.corrcoef(fc.apply(xs), fh.apply(xs))[0, 1] > 0.95
    # noisy field -> range-preserving smoothing (measured: no overshoot)
    rng = np.random.default_rng(3)
    xb = rng.uniform(0.0, 1.0, sidx.size)
    xh = fh.apply(xb)
    assert xh.min() >= -1e-9 and xh.max() <= 1.0 + 1e-9
    assert xh.std() < xb.std()


# --------------------------------------------------------------------------- #
# Boundary consistency                                                        #
# --------------------------------------------------------------------------- #


def test_boundary_member_mirror_consistency():
    """Neumann consistency: a member of width w flush with the domain face
    must filter like the INNER HALF of an interior member of width 2w
    mirrored on that face — the missing-neighbour drop IS the discrete
    mirror extension. Helmholtz satisfies this to solver precision; the
    clipped/renormalized cone deviates (measured ~2.4e-2 absolute here,
    i.e. the effective filter length differs at the boundary — the
    documented boundary artefact)."""
    m = trimesh.creation.box(extents=[20.0, 12.0, 16.0])
    m.apply_translation([0.0, 0.0, 8.0])
    dom = voxelize(m, 1.0, 200_000)
    sidx = np.flatnonzero(dom.solid.ravel())
    gi = np.indices(dom.dims)
    nz, w, c = dom.dims[2], 3, 9  # 2w reference bar centered at z = c-0.5

    xb_b = np.zeros(dom.cell_count)
    flush = (gi[2] >= nz - w) & (gi[1] >= 5) & (gi[1] <= 6)
    xb_b[flush.ravel()] = 1.0  # w cells ending AT the top face
    xb_i = np.zeros(dom.cell_count)
    mid = (gi[2] >= c - w) & (gi[2] <= c + w - 1) & (gi[1] >= 5) & (gi[1] <= 6)
    xb_i[mid.ravel()] = 1.0  # 2w cells symmetric about the plane z = c-0.5

    diffs = {}
    for name in ("cone", "helmholtz"):
        fops = _build_filter_ops(dom, sidx, 4.0, name)
        xtb = np.zeros(dom.cell_count)
        xtb[sidx] = fops.apply(xb_b[sidx])
        xti = np.zeros(dom.cell_count)
        xti[sidx] = fops.apply(xb_i[sidx])
        p_b = xtb.reshape(dom.dims)[10, 5, nz - w :][::-1]  # face -> inward
        p_i = xti.reshape(dom.dims)[10, 5, c : c + w]  # mirror plane -> outward
        diffs[name] = float(np.abs(p_b - p_i).max())
    assert diffs["helmholtz"] < 1e-3  # mirror symmetry holds essentially exact
    assert diffs["cone"] > 5 * diffs["helmholtz"]  # the artefact the PDE fixes


def test_boundary_member_not_thinned():
    """Same geometry: under the cone filter the flush member reaches a
    LOWER filtered peak than its mirrored interior reference (the truncated
    kernel under-counts), so an extracted boundary member is dimensioned
    thinner than the same member in the interior. Helmholtz shows no such
    systematic thinning."""
    m = trimesh.creation.box(extents=[20.0, 12.0, 16.0])
    m.apply_translation([0.0, 0.0, 8.0])
    dom = voxelize(m, 1.0, 200_000)
    sidx = np.flatnonzero(dom.solid.ravel())
    gi = np.indices(dom.dims)
    nz, w, c = dom.dims[2], 3, 9
    xb_b = np.zeros(dom.cell_count)
    xb_b[((gi[2] >= nz - w) & (gi[1] >= 5) & (gi[1] <= 6)).ravel()] = 1.0
    xb_i = np.zeros(dom.cell_count)
    xb_i[((gi[2] >= c - w) & (gi[2] <= c + w - 1) & (gi[1] >= 5) & (gi[1] <= 6)).ravel()] = 1.0

    peaks = {}
    for name in ("cone", "helmholtz"):
        fops = _build_filter_ops(dom, sidx, 4.0, name)
        xtb = np.zeros(dom.cell_count)
        xtb[sidx] = fops.apply(xb_b[sidx])
        xti = np.zeros(dom.cell_count)
        xti[sidx] = fops.apply(xb_i[sidx])
        peaks[name] = (
            float(xtb.reshape(dom.dims)[10, 5, nz - 1]),  # face cell of w-bar
            float(xti.reshape(dom.dims)[10, 5, c]),  # mirror cell of 2w-bar
        )
    # cone under-filters the boundary member vs its interior mirror —
    # measured ~0.418 vs ~0.436 (~4% lower peak -> thinner member)
    assert peaks["cone"][0] < peaks["cone"][1] - 1e-3
    # helmholtz boundary peak is NOT below the interior reference
    assert peaks["helmholtz"][0] >= peaks["helmholtz"][1] - 1e-3


# --------------------------------------------------------------------------- #
# SIMP loop integration                                                       #
# --------------------------------------------------------------------------- #


def test_simp_helmholtz_converges():
    dom = voxelize(_block(), 1.0, 200_000)
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=40, filter="helmholtz"),
    )
    assert res.iterations > 0
    objs = [r["compliance_mj"] for r in res.history]
    assert objs[-1] < objs[0]
    assert res.history[-1]["volfrac"] == pytest.approx(0.4, abs=0.02)


def test_simp_helmholtz_heaviside_converges_crisp():
    dom = voxelize(_block(), 1.0, 200_000)
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(
            volfrac=0.4,
            rmin_mm=2.0,
            max_iter=45,
            filter="helmholtz+heaviside",
            beta_max=8.0,
        ),
    )
    assert res.iterations > 0
    assert "beta" in res.history[-1]
    solid_rho = res.density[dom.solid.ravel()]
    crisp = ((solid_rho < 0.2) | (solid_rho > 0.8)).mean()
    assert crisp > 0.8  # near-binary like the cone heaviside variant
    assert res.history[-1]["volfrac"] == pytest.approx(0.4, abs=0.03)


def test_robust_helmholtz_heaviside_combines():
    """robust=True must accept a Heaviside-bearing PDE filter directly —
    no forced filter change, one obj_per_design triple per iteration."""
    dom = voxelize(_block(), 1.0, 200_000)
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(
            volfrac=0.4,
            rmin_mm=2.0,
            max_iter=12,
            filter="helmholtz+heaviside",
            robust=True,
        ),
    )
    assert res.iterations > 0
    assert not any("forced" in w for w in res.warnings)
    objs = res.history[-1]["obj_per_design"]
    assert len(objs) == 3  # eroded / intermediate / dilated


def test_robust_pure_helmholtz_upgrades_with_warning():
    """robust needs a projection stage: filter='helmholtz' must upgrade to
    'helmholtz+heaviside' (keeping the PDE stage) with a warning."""
    dom = voxelize(_block(), 1.0, 200_000)
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=8, filter="helmholtz", robust=True),
    )
    assert res.iterations > 0
    assert any("helmholtz+heaviside" in w for w in res.warnings)
    assert "beta" in res.history[-1]  # projection actually ran
