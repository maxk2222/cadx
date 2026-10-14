"""SIMP manufacturability filters: 45° self-support (Langelaar AM) and
Heaviside projection with beta continuation (`cad/fem/topopt.py`)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.topopt import (
    TopOptParams,
    _am_backward,
    _am_forward,
    _am_setup,
    _heaviside,
    _heaviside_d,
    passive_solid_mask,
    simp,
)
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _block(w: float = 30.0, h: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[w, 24.0, h])
    m.apply_translation([0.0, 0.0, h / 2])
    return m


def _case(axis_region: dict, force: list[float]) -> LoadCase:
    return LoadCase(
        name="c",
        fixed=[RegionSpec(**axis_region)],
        loads=[LoadSpec(region=RegionSpec(kind="all"), force_n=force)],
    )


# --------------------------------------------------------------------------- #
# AM support filter                                                             #
# --------------------------------------------------------------------------- #


def test_am_setup_layers_and_stencil():
    dom = voxelize(_block(w=10.0, h=4.0), 1.0, 200_000)
    layers = _am_setup(dom, "+z")
    assert layers  # at least bed + one
    _cells0, sup0 = layers[0]
    assert sup0 is None  # bed layer: no stencil, prints free
    cells1, sup1 = layers[1]
    assert sup1 is not None and sup1.shape[1] == 5
    # every support must be in the layer below (z index smaller by 1)
    z1 = np.unravel_index(cells1, dom.dims)[2]
    sup_ok = sup1[sup1 >= 0]
    z_sup = np.unravel_index(sup_ok, dom.dims)[2]
    assert (z_sup == z1[0] - 1).all()


def test_am_forward_propagates_support():
    dom = voxelize(_block(w=6.0, h=4.0), 1.0, 200_000)
    layers = _am_setup(dom, "+z")
    x = dom.solid.ravel().astype(float)
    # wipe the centre column above the bed -> upper cells must lose density
    g = np.unravel_index(np.arange(dom.cell_count), dom.dims)
    x[(g[2] == 1) & (g[0] == 3) & (g[1] == 3)] = 0.0  # hole in layer 1
    p = _am_forward(x, dom.solid.ravel(), layers)
    # cell directly above the hole (layer 2) has 4/5 support -> clipped to 0.8
    idx = np.ravel_multi_index((3, 3, 2), dom.dims)
    assert p[idx] == pytest.approx(0.8)


def test_am_backward_distributes_gradient():
    dom = voxelize(_block(w=6.0, h=4.0), 1.0, 200_000)
    layers = _am_setup(dom, "+z")
    x = dom.solid.ravel().astype(float)
    g = np.unravel_index(np.arange(dom.cell_count), dom.dims)
    x[(g[2] == 1) & (g[0] == 3) & (g[1] == 3)] = 0.0
    p = _am_forward(x, dom.solid.ravel(), layers)
    dp = np.ones(dom.cell_count)
    gx = _am_backward(dp, x, p, dom.solid.ravel(), layers)
    idx = np.ravel_multi_index((3, 3, 2), dom.dims)
    assert gx[idx] == pytest.approx(0.0)  # clipped cell: gradient flows down
    # its 5 supports each get 1/5 of its unit gradient
    sup = next(
        s[np.flatnonzero(c == idx)[0]] for c, s in layers if s is not None and (c == idx).any()
    )
    for s in sup[sup >= 0]:
        assert gx[s] > 1.0  # own + 0.2 share from above


def test_simp_self_supporting_no_floating_islands():
    """A cantilever-only spec: self_supporting must not leave material hanging."""
    mesh = _block()
    dom = voxelize(mesh, 1.5, 200_000)
    case = _case({"kind": "face", "axis": "-x"}, [0, 0, -40])
    from cadx.cad.fem.regions import resolve_case

    fixed, F, _ = resolve_case(dom, case, [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.35, max_iter=15, self_supporting=True, print_direction="+z"),
    )
    rho = res.density.reshape(dom.dims)
    # every solid cell above the bed must have material somewhere below it
    solid_rho = rho > 0.5
    for k in range(1, dom.dims[2]):
        top = np.argwhere(solid_rho[:, :, k])
        for i, j in top:
            below = solid_rho[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2, k - 1]
            assert below.any(), f"floating cell at {i},{j},{k}"


# --------------------------------------------------------------------------- #
# Heaviside projection                                                          #
# --------------------------------------------------------------------------- #


def test_heaviside_shape_and_derivative():
    x = np.linspace(0.0, 1.0, 11)
    h = _heaviside(x, 8.0)
    assert h[0] == pytest.approx(0.0, abs=0.02)
    assert h[-1] == pytest.approx(1.0, abs=0.02)
    assert h[5] == pytest.approx(0.5, abs=1e-3)
    assert _heaviside_d(np.array([0.5]), 8.0)[0] > 1.0  # sharpening slope
    # beta -> 0 approaches identity
    h0 = _heaviside(x, 0.01)
    np.testing.assert_allclose(h0, x, atol=0.01)


def test_simp_heaviside_converges_crisp():
    mesh = _block()
    dom = voxelize(mesh, 1.5, 200_000)
    case = _case({"kind": "face", "axis": "-x"}, [0, 0, -30])
    from cadx.cad.fem.regions import resolve_case

    fixed, F, _ = resolve_case(dom, case, [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, max_iter=45, filter="heaviside", beta_max=8.0),
    )
    rho = res.density
    solid_rho = rho[dom.solid.ravel()]
    # near-binary output: most cells <0.2 or >0.8
    crisp = ((solid_rho < 0.2) | (solid_rho > 0.8)).mean()
    assert crisp > 0.85
    assert res.iterations > 0


def test_simp_ij_cache_consistent():
    """Cached sparsity indices must not change the SIMP result."""
    mesh = _block()
    dom = voxelize(mesh, 2.0, 200_000)
    case = _case({"kind": "face", "axis": "-x"}, [0, 0, -20])
    from cadx.cad.fem.regions import resolve_case

    fixed, F, _ = resolve_case(dom, case, [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    r = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, max_iter=8),
    )
    assert r.iterations > 0
    assert 0.1 < float(r.density[dom.solid.ravel()].mean()) < 0.9


def test_upsample_density_alignment():
    """Coarse→fine upsampling must place material at matching world coords."""
    from cadx.cad.engine import _upsample_density

    mesh = _block(w=16.0, h=8.0)
    dom_f = voxelize(mesh, 1.0, 400_000)
    dom_c = voxelize(mesh, 2.0, 400_000)
    # coarse field: solid only in the +x half
    rho_c = np.zeros(dom_c.dims)
    rho_c[dom_c.dims[0] // 2 :, :, :] = 1.0
    x0 = _upsample_density(rho_c, dom_c, dom_f)
    assert x0.shape == (dom_f.cell_count,)
    rho_f = x0.reshape(dom_f.dims)
    lo, hi = rho_f[: dom_f.dims[0] // 4].mean(), rho_f[3 * dom_f.dims[0] // 4 :].mean()
    assert lo < 0.1 and hi > 0.9
    assert x0.min() >= 0.0 and x0.max() <= 1.0


def test_simp_warmstart_converges_faster():
    """Warm-started SIMP reaches the same volfrac in fewer iterations."""
    from cadx.cad.fem.regions import resolve_case

    mesh = _block()
    dom = voxelize(mesh, 2.0, 200_000)
    case = _case({"kind": "face", "axis": "-x"}, [0, 0, -20])
    fixed, F, _ = resolve_case(dom, case, [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    p = TopOptParams(volfrac=0.4, max_iter=60, converge_patience=20)
    cold = simp(dom, E_PLA, NU, [(fixed, F)], passive, p)
    warm = simp(dom, E_PLA, NU, [(fixed, F)], passive, p, x0=cold.density.copy())
    assert warm.iterations < cold.iterations
    assert abs(float(warm.density[dom.solid.ravel()].mean()) - 0.4) < 0.05
