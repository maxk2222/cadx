"""Mirror-symmetry constraint for SIMP (`cad/fem/topopt.py`
TopOptParams.symmetry): after every design update the blueprint density is
averaged with its reflection in the chosen domain mid-plane
(x = 0.5*(x + flip(x)), passive cells re-pinned), so the extracted part
comes out symmetric about "xy"/"xz"/"yz"."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.regions import resolve_case
from cadx.cad.fem.topopt import (
    RHO_MIN,
    TopOptParams,
    _mirror_index_map,
    _sym_grad,
    _symmetrize_density,
    passive_solid_mask,
    simp,
)
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, OptimizeSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _beam_domain():
    """20x8x8 voxel mini cantilever — symmetric about every mid-plane."""
    m = trimesh.creation.box(extents=[20.0, 8.0, 8.0])
    m.apply_translation([0.0, 0.0, 4.0])
    return voxelize(m, 1.0, 200_000)


def _setup(dom):
    """Cantilever: fixed -x face, downward -z tip load on the +x face.
    BCs are symmetric about xz (the y mid-plane)."""
    case = LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -30.0],
            )
        ],
    )
    fixed, F, _ = resolve_case(dom, case, [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    return fixed, F, passive


def test_symmetry_xz_produces_mirrored_density():
    """symmetry="xz" must leave the result mirror-symmetric in y — the
    blueprint bitwise (0.5*(a+b) == 0.5*(b+a) in IEEE), the filtered
    physical density within tolerance."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=15, symmetry="xz"),
    )
    assert res.iterations > 0
    g = res.density.reshape(dom.dims)
    assert np.allclose(g, np.flip(g, axis=1))
    assert res.blueprint is not None
    bg = res.blueprint.reshape(dom.dims)
    np.testing.assert_array_equal(bg, np.flip(bg, axis=1))


def test_symmetry_xy_produces_mirrored_blueprint():
    """symmetry="xy" forces a z-mirrored blueprint even though the tip
    load is NOT symmetric about the xy mid-plane — the constraint is on
    the density, not the physics."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=10, symmetry="xy"),
    )
    assert res.iterations > 0
    assert res.blueprint is not None
    bg = res.blueprint.reshape(dom.dims)
    np.testing.assert_array_equal(bg, np.flip(bg, axis=2))


def test_symmetry_off_runs_unchanged():
    """Default symmetry=None keeps the plain deterministic path — just
    check the run completes and returns a full density field."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=10),
    )
    assert res.iterations > 0
    assert res.density.shape == (dom.cell_count,)


def test_symmetry_unknown_plane_warns_and_ignores():
    """An out-of-enum plane name must not crash — warn and run plain."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=8, symmetry="diag"),
    )
    assert res.iterations > 0
    assert any("symmetry" in w for w in res.warnings)


def test_symmetry_xz_mma_path():
    """The MMA optimizer path symmetrises the unpacked blueprint every
    evaluation — the returned design is mirrored in y as well."""
    pytest.importorskip("nlopt")
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=8, solver="mma", symmetry="xz"),
    )
    assert res.iterations > 0
    assert res.blueprint is not None
    bg = res.blueprint.reshape(dom.dims)
    np.testing.assert_array_equal(bg, np.flip(bg, axis=1))


# --------------------------------------------------------------------------- #
# T5: MMA gradients through the mirror average (_sym_grad / _mirror_index_map) #
# --------------------------------------------------------------------------- #


def test_mirror_index_map_matches_flip():
    """d[mirror_idx] must reproduce np.flip along the mirrored axis, the -1
    entries must land exactly where the mirror falls outside the solid, and
    the map must be an involution on its valid entries."""
    dims = (5, 4, 3)
    axis = 1
    rng = np.random.default_rng(1)
    n = int(np.prod(dims))
    solid = rng.random(n) > 0.25
    mi = _mirror_index_map(dims, axis, solid)

    d = rng.random(n)
    ref = np.flip(d.reshape(dims), axis=axis).ravel()
    np.testing.assert_allclose(d[mi[mi >= 0]], ref[mi >= 0])
    # -1 exactly where the mirrored cell is non-solid
    np.testing.assert_array_equal(mi < 0, np.flip((~solid).reshape(dims), axis=axis).ravel())
    # involution on solid cells whose mirror is also solid (a non-solid
    # cell may mirror INTO the solid, but then its own mirror slot is -1)
    both = solid & (mi >= 0)
    np.testing.assert_array_equal(mi[mi[both]], np.flatnonzero(both))


def test_sym_grad_hand_computed():
    """The three mirror cases on a 1x4x1 strip mirrored in y: a free/free
    pair shares half sensitivities, a cell whose mirror is pinned keeps
    only its own half, and an odd-width mid-plane cell keeps all of it."""
    dims = (1, 4, 1)
    solid = np.ones(4, dtype=bool)
    mi = _mirror_index_map(dims, 1, solid)
    np.testing.assert_array_equal(mi, [3, 2, 1, 0])
    # cells 0,1,3 free — cell 1's mirror (cell 2) is pinned
    design = np.array([True, True, False, True])
    mfree = (mi >= 0) & design[mi]
    d = np.array([1.0, 2.0, 9.0, 4.0])
    g = _sym_grad(d, mi, mfree)
    np.testing.assert_allclose(g[[0, 1, 3]], [2.5, 1.0, 2.5])
    # mid-plane cell of an odd strip mirrors itself -> full sensitivity
    mi_odd = _mirror_index_map((1, 5, 1), 1, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(mi_odd, [4, 3, 2, 1, 0])
    g_odd = _sym_grad(np.arange(5.0) + 1.0, mi_odd, mi_odd >= 0)
    np.testing.assert_allclose(g_odd[2], 3.0)


def test_sym_grad_is_mirror_adjoint_finite_difference():
    """f(v) = aᵀ S(v) through the REAL _symmetrize_density (including its
    re-pinning): the _sym_grad adjoint must match centred finite
    differences of the raw variables on every free design cell — this is
    the exact Jacobian the MMA gradients were missing."""
    rng = np.random.default_rng(4)
    dims = (4, 5, 3)
    axis = 1
    n = int(np.prod(dims))
    solid = rng.random(n) > 0.2  # non-solid cells -> out-of-solid mirrors
    passive = np.zeros(n, dtype=bool)
    pvoid = np.zeros(n, dtype=bool)
    cand = np.flatnonzero(solid)
    pick = rng.choice(cand, size=cand.size // 4, replace=False)
    passive[pick[: pick.size // 2]] = True
    pvoid[pick[pick.size // 2 :]] = True
    design = solid & ~passive & ~pvoid
    des = np.flatnonzero(design)
    # same construction _run_mma uses
    mirror_idx = _mirror_index_map(dims, axis, solid)
    mirror_free = (mirror_idx >= 0) & design[np.clip(mirror_idx, 0, None)]

    a = rng.uniform(-1.0, 1.0, n)
    v = rng.uniform(0.1, 0.9, n)

    def f(vec: np.ndarray) -> float:
        # raw blueprint with the same pinning _unpack applies
        xb = vec.copy()
        xb[~solid] = RHO_MIN
        xb[passive] = 1.0
        xb[pvoid] = RHO_MIN
        xs = _symmetrize_density(xb, dims, axis, solid, passive, pvoid)
        return float(a @ xs)

    ga = _sym_grad(a, mirror_idx, mirror_free)[des]
    eps = 1e-6
    fd = np.empty(des.size)
    for j, cell in enumerate(des):
        vp, vm = v.copy(), v.copy()
        vp[cell] += eps
        vm[cell] -= eps
        fd[j] = (f(vp) - f(vm)) / (2 * eps)
    np.testing.assert_allclose(ga, fd, rtol=1e-5, atol=1e-7)


def test_symmetry_xy_mma_converges_asymmetric_load():
    """Integration smoke: MMA + xy symmetry on physics that is NOT mirror
    symmetric about the z mid-plane (-z tip load mirrors to +z). With the
    mirror adjoint the run stays stable (bounded objective), converges and
    returns a bitwise mirrored blueprint AND physical density."""
    pytest.importorskip("nlopt")
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=15, solver="mma", symmetry="xy"),
    )
    assert res.iterations > 0
    assert res.converged
    objs = [r["compliance_mj"] for r in res.history]
    assert objs[-1] <= objs[0] * 1.05  # stable — wrong-sign/magnitude grads diverge
    assert res.blueprint is not None
    bg = res.blueprint.reshape(dom.dims)
    np.testing.assert_array_equal(bg, np.flip(bg, axis=2))
    g = res.density.reshape(dom.dims)
    np.testing.assert_allclose(g, np.flip(g, axis=2), atol=1e-9)


# --------------------------------------------------------------------------- #
# T8: OptimizeSpec enum fields tolerate hallucinated values                    #
# --------------------------------------------------------------------------- #


def test_optimize_spec_enums_fall_back_to_defaults():
    """LLM-hallucinated enum values must not fail the parse — they fall
    back to the documented defaults (None for symmetry)."""
    spec = OptimizeSpec(
        solver="gradient_free",
        case_objective="worst_case",
        filter="none",
        symmetry="diagonal",
    )
    assert spec.solver == "oc"
    assert spec.case_objective == "weighted"
    assert spec.filter == "helmholtz+heaviside"
    assert spec.symmetry is None
    # None on a required enum also falls back rather than erroring
    assert OptimizeSpec(solver=None).solver == "oc"  # type: ignore[arg-type]


def test_optimize_spec_enums_accept_valid_literals():
    spec = OptimizeSpec(
        solver="mma",
        case_objective="minmax",
        filter="helmholtz+heaviside",
        symmetry="xz",
    )
    assert spec.solver == "mma"
    assert spec.case_objective == "minmax"
    assert spec.filter == "helmholtz+heaviside"
    assert spec.symmetry == "xz"
    # case-insensitive normalisation — "XY" is a plane name, not an error
    assert OptimizeSpec(symmetry="XY").symmetry == "xy"
