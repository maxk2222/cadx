"""Load-path corridor field + voxelize pad_cells.

``loadpath_field`` paints a deterministic density over the solid cells —
high on the geodesic corridor between the load and the fixture cells of
each resolved case, ``base_rho`` elsewhere — usable as a SIMP warm start
or a preview-grade structure without running the optimizer.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy import ndimage

from cadx.cad.fem.loadpath import loadpath_field
from cadx.cad.fem.loadpath_cert import loadpath_certificate
from cadx.cad.fem.voxelize import VoxelDomain, voxelize


def _box_mesh(x: float = 20.0, y: float = 8.0, z: float = 8.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _l_domain() -> VoxelDomain:
    """L-bracket voxel domain: vertical leg (x<4, all z) + horizontal arm
    (z<4, all x), 4 cells thick in y — the corner carries the load path."""
    dims = (16, 4, 16)
    solid = np.zeros(dims, dtype=bool)
    solid[:4] = True
    solid[:, :, :4] = True
    return VoxelDomain(origin=np.zeros(3), pitch=1.0, dims=dims, solid=solid)


def _case_on(dom: VoxelDomain, fix_cells: np.ndarray, load_cells: np.ndarray):
    """(fixed_mask, F) for explicit cell masks — same tuple shape
    resolve_case returns: node mask + (3*nnodes,) force vector."""
    en = dom.element_nodes()
    fix_nodes = np.zeros(dom.node_count, dtype=bool)
    fix_nodes[np.unique(en[fix_cells.ravel()])] = True
    load_nodes = np.zeros(dom.node_count, dtype=bool)
    load_nodes[np.unique(en[load_cells.ravel()])] = True
    F = np.zeros(3 * dom.node_count)
    F[3 * np.flatnonzero(load_nodes) + 2] = -50.0 / max(int(load_nodes.sum()), 1)
    return fix_nodes, F


def _l_case(dom: VoxelDomain):
    """Fix the arm tip (+x end of the bottom arm), load the leg top."""
    fix_cells = np.zeros(dom.dims, dtype=bool)
    fix_cells[14:, :, :4] = True  # far end of the horizontal arm
    load_cells = np.zeros(dom.dims, dtype=bool)
    load_cells[:4, :, 13:] = True  # top of the vertical leg
    return _case_on(dom, fix_cells, load_cells), fix_cells, load_cells


# ---------------------------------------------------------------------------
# corridor field
# ---------------------------------------------------------------------------


def test_loadpath_connects_load_to_fixed() -> None:
    """The rho>0.8 corridor is one connected set reaching both the load and
    the fixed region — a path through the elbow, not a disconnected blob."""
    dom = _l_domain()
    (fixed, F), fix_cells, load_cells = _l_case(dom)
    rho = loadpath_field(dom, [(fixed, F)])
    assert rho.dtype == np.float32
    assert rho.shape == (dom.cell_count,)
    hi = (rho >= 0.8).reshape(dom.dims)
    assert hi.any()
    # the corridor touches both endpoint regions
    assert (hi & fix_cells).any()
    assert (hi & load_cells).any()
    # corridor + endpoints form a single face-connected component
    _, n = ndimage.label(
        hi | fix_cells | load_cells, structure=ndimage.generate_binary_structure(3, 1)
    )
    assert n == 1


def test_loadpath_corridor_decays_off_path() -> None:
    """Cells away from the shortest corridor stay near base_rho — the
    field is a corridor, not a uniform fill."""
    dom = _l_domain()
    (fixed, F), _fix_cells, _load_cells = _l_case(dom)
    rho = loadpath_field(dom, [(fixed, F)], base_rho=0.15).reshape(dom.dims)
    # cell under the leg at the bottom-left inner corner: off the elbow
    # corridor — both endpoints are reached via the corner, not through here
    corner = rho[1, 1, 0]
    assert corner < 0.6
    assert rho[:4, :, 13:].max() > 0.8  # load patch itself is on the path


def test_loadpath_deterministic() -> None:
    dom = _l_domain()
    (fixed, F), _fc, _lc = _l_case(dom)
    r1 = loadpath_field(dom, [(fixed, F)])
    r2 = loadpath_field(dom, [(fixed, F)])
    assert np.array_equal(r1, r2)


def test_loadpath_pins() -> None:
    """passive_solid pins to 1.0; passive_void and non-solid pin to 0."""
    dom = _l_domain()
    (fixed, F), _fc, _lc = _l_case(dom)
    psolid = np.zeros(dom.dims, dtype=bool)
    psolid[1, 1, 8] = True  # interior leg cell far from either endpoint
    pvoid = np.zeros(dom.dims, dtype=bool)
    pvoid[0, 0, 0] = True  # corner cell
    rho = loadpath_field(
        dom,
        [(fixed, F)],
        passive_solid=psolid.ravel(),
        passive_void=pvoid.ravel(),
    ).reshape(dom.dims)
    assert rho[1, 1, 8] == 1.0
    assert rho[0, 0, 0] == 0.0
    assert (rho[~dom.solid] == 0.0).all()


def test_loadpath_multicase_weights() -> None:
    """A weaker second case still paints its corridor (at lower weight);
    a zero-force case is skipped without poisoning the field."""
    dom = _l_domain()
    (fixed, F), fix_cells, _lc = _l_case(dom)
    # second case: same fixture, load applied at mid-arm instead of leg top
    load2 = np.zeros(dom.dims, dtype=bool)
    load2[7:9, :, :4] = True
    _, F2 = _case_on(dom, fix_cells, load2)
    F2 = F2 * 0.3  # weaker case
    rho = loadpath_field(dom, [(fixed, F), (fixed, F2)])
    assert rho.dtype == np.float32
    assert float(rho.max()) <= 1.0 + 1e-6
    # the weaker case's corridor shows up in the union field
    assert float((rho.reshape(dom.dims) > 0.3)[7:9, :, :4].mean()) > 0.3
    # zero-force case skipped gracefully — field stays at base level
    rho1 = loadpath_field(dom, [(fixed, np.zeros_like(F))])
    assert rho1.dtype == np.float32
    assert float(rho1.max()) <= 0.15 + 1e-6


# ---------------------------------------------------------------------------
# voxelize pad_cells
# ---------------------------------------------------------------------------


def test_voxelize_pad_cells_grows_grid() -> None:
    """pad_cells grows every side by N cells: dims +2N per axis, origin
    shifted -N*pitch, the part mask unchanged (pad ring stays void)."""
    dom0 = voxelize(_box_mesh(), 2.0, 150_000)
    dom = voxelize(_box_mesh(), 2.0, 150_000, pad_cells=2)
    assert dom.pad_cells == 2
    assert dom0.pad_cells == 0
    assert dom.pitch == pytest.approx(dom0.pitch)
    assert dom.dims == tuple(d + 4 for d in dom0.dims)
    np.testing.assert_allclose(dom.origin, dom0.origin - 2.0 * dom.pitch)
    assert dom.solid_count == dom0.solid_count
    # the pad ring is void on both extremes of every axis
    for ax in range(3):
        lo = np.moveaxis(dom.solid, ax, 0)[0]
        hi = np.moveaxis(dom.solid, ax, 0)[-1]
        assert not lo.any() and not hi.any()


def test_voxelize_pad_cells_zero_is_legacy() -> None:
    dom0 = voxelize(_box_mesh(), 2.0, 150_000)
    dom = voxelize(_box_mesh(), 2.0, 150_000, pad_cells=0)
    assert dom.dims == dom0.dims
    np.testing.assert_allclose(dom.origin, dom0.origin)
    assert (dom.solid == dom0.solid).all()


# ---------------------------------------------------------------------------
# loadpath_certificate — the deterministic min-cut export-solid check
# ---------------------------------------------------------------------------


def _bar_domain() -> VoxelDomain:
    """8x2x2 solid bar, pitch 1 — every cell is a surface cell so the
    inscribed section is exactly one cell face (1 mm²) and the transverse
    min cut is 4 mm²."""
    dims = (8, 2, 2)
    return VoxelDomain(
        origin=np.zeros(3), pitch=1.0, dims=dims, solid=np.ones(dims, dtype=bool)
    )


def test_cert_bar_axial_floor() -> None:
    dom = _bar_domain()
    fix_cells = np.zeros(dom.dims, dtype=bool)
    fix_cells[0, :, :] = True
    load_cells = np.zeros(dom.dims, dtype=bool)
    load_cells[-1, :, :] = True
    cert = loadpath_certificate(dom, [_case_on(dom, fix_cells, load_cells)], yield_mpa=25.0)
    assert cert["load_path_ok"] is True
    assert cert["bottleneck_mm2"] == pytest.approx(4.0)
    # 4 mm² * 25 MPa / 50 N applied
    assert cert["sf_floor"] == pytest.approx(2.0)
    assert cert["per_case"][0]["reachable"] is True
    assert not cert["warnings"]


def test_cert_neck_throttles_flow() -> None:
    """A 4x4 bar necked to a single cell: the cut through the neck is
    1 mm² — far below the 16 mm² ends — so the certificate reports the
    neck's section, not the part's."""
    dims = (9, 4, 4)
    solid = np.zeros(dims, dtype=bool)
    solid[:4] = True  # wide fixed end
    solid[6:] = True  # wide load end
    solid[4:6, 1, 1] = True  # 1-cell-wide neck bridging the gap
    dom = VoxelDomain(origin=np.zeros(3), pitch=1.0, dims=dims, solid=solid)
    fix_cells = np.zeros(dims, dtype=bool)
    fix_cells[0] = True
    load_cells = np.zeros(dims, dtype=bool)
    load_cells[-1] = True
    cert = loadpath_certificate(dom, [_case_on(dom, fix_cells, load_cells)], yield_mpa=50.0)
    assert cert["load_path_ok"] is True
    assert cert["bottleneck_mm2"] == pytest.approx(1.0)
    assert cert["sf_floor"] == pytest.approx(1.0)  # 1 mm² * 50 MPa / 50 N


def test_cert_disconnected_solid_fails() -> None:
    """Load and fixture on separate islands: no path, flow 0 —
    load_path_ok False, no bottleneck, and a warning names the case."""
    dims = (9, 2, 2)
    solid = np.zeros(dims, dtype=bool)
    solid[:4] = True
    solid[6:] = True  # second island, gap at x in [4,6)
    dom = VoxelDomain(origin=np.zeros(3), pitch=1.0, dims=dims, solid=solid)
    fix_cells = np.zeros(dims, dtype=bool)
    fix_cells[0] = True
    load_cells = np.zeros(dims, dtype=bool)
    load_cells[-1] = True
    cert = loadpath_certificate(dom, [_case_on(dom, fix_cells, load_cells)], yield_mpa=25.0)
    assert cert["load_path_ok"] is False
    assert cert["bottleneck_mm2"] is None
    assert cert["sf_floor"] is None
    assert cert["per_case"][0]["reachable"] is False
    assert any("cannot reach" in w for w in cert["warnings"])


def test_cert_zero_load_is_broken() -> None:
    dom = _bar_domain()
    fix_cells = np.zeros(dom.dims, dtype=bool)
    fix_cells[0, :, :] = True
    fixed, _F = _case_on(dom, fix_cells, np.zeros(dom.dims, dtype=bool))
    cert = loadpath_certificate(dom, [(fixed, np.zeros_like(_F))], yield_mpa=25.0)
    assert cert["load_path_ok"] is False
    assert cert["bottleneck_mm2"] is None
    assert any("no load or fixture" in w for w in cert["warnings"])


def test_cert_shared_cell_is_indeterminate_not_false() -> None:
    """Load and fixture on the same cells: reachable trivially — the min
    cut is indeterminate (infinite through the shared arcs), NOT the
    spurious inf_cap number."""
    dom = _bar_domain()
    cells = np.zeros(dom.dims, dtype=bool)
    cells[3:5] = True
    fixed, F = _case_on(dom, cells, cells)
    cert = loadpath_certificate(dom, [(fixed, F)], yield_mpa=25.0)
    assert cert["load_path_ok"] is True
    assert cert["per_case"][0]["reachable"] is True
    assert cert["per_case"][0]["bottleneck_mm2"] is None
    assert cert["bottleneck_mm2"] is None
    assert any("indeterminate" in w for w in cert["warnings"])


def test_cert_multicase_worst_wins() -> None:
    """Two cases — one reachable through a wide bar, one disconnected —
    fail the certificate and the reported bottleneck stays the reachable
    case's, not None-vs-number ambiguity."""
    dom = _bar_domain()
    fix_cells = np.zeros(dom.dims, dtype=bool)
    fix_cells[0, :, :] = True
    load_cells = np.zeros(dom.dims, dtype=bool)
    load_cells[-1, :, :] = True
    good = _case_on(dom, fix_cells, load_cells)
    bad = (np.zeros(dom.node_count, dtype=bool), np.zeros(3 * dom.node_count))
    cert = loadpath_certificate(dom, [good, bad], yield_mpa=25.0)
    assert cert["load_path_ok"] is False
    assert cert["bottleneck_mm2"] == pytest.approx(4.0)
    assert len(cert["per_case"]) == 2
    assert cert["per_case"][0]["reachable"] is True
    assert cert["per_case"][1]["reachable"] is False


def test_cert_no_cases_and_empty_solid_degrade() -> None:
    dom = _bar_domain()
    cert = loadpath_certificate(dom, [], yield_mpa=25.0)
    assert cert["load_path_ok"] is None
    empty = VoxelDomain(
        origin=np.zeros(3), pitch=1.0, dims=(4, 4, 4), solid=np.zeros((4, 4, 4), dtype=bool)
    )
    cert = loadpath_certificate(empty, [], yield_mpa=25.0)
    assert cert["load_path_ok"] is False
    assert cert["bottleneck_mm2"] is None


def test_cert_zero_yield_keeps_bottleneck() -> None:
    """yield_mpa=0 (unknown material) — geometry still certifies; the SF
    floor is simply not derivable."""
    dom = _bar_domain()
    fix_cells = np.zeros(dom.dims, dtype=bool)
    fix_cells[0, :, :] = True
    load_cells = np.zeros(dom.dims, dtype=bool)
    load_cells[-1, :, :] = True
    cert = loadpath_certificate(dom, [_case_on(dom, fix_cells, load_cells)], yield_mpa=0.0)
    assert cert["load_path_ok"] is True
    assert cert["bottleneck_mm2"] == pytest.approx(4.0)
    assert cert["sf_floor"] is None
