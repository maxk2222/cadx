"""kind="plane" regions — node-exact loci (fixture planes, symmetry cuts,
knife-edge supports), plus LoadSpec.spread_mode="area".

plane is the only region that resolves NODES directly instead of the
>=2-node union of a cell band — an exact x=0 symmetry cut or a single
edge line is not expressible with cell kinds. axis letters AND together
("xy" = edges, "xyz" = corners) — deliberately different from `border`
whose letters OR into a perimeter band. An empty locus reports "matched
no nodes" — it must NOT fall back to a bbox patch (that would silently
pin the wrong DOFs).
"""

from __future__ import annotations

import numpy as np

from cadx.cad.fem.regions import (
    _plane_locus,
    region_cell_mask,
    region_node_mask,
    resolve_case_dofs,
)
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

PITCH = 2.0
DIMS = (10, 8, 4)  # cells — nodes x∈{0..20}, y∈{0..16}, z∈{0..8}


def _dom() -> VoxelDomain:
    return VoxelDomain(
        origin=np.array([0.0, 0.0, 0.0]),
        pitch=PITCH,
        dims=DIMS,
        solid=np.ones(DIMS, dtype=bool),
    )


def _plane(axis: str, **kw) -> RegionSpec:
    return RegionSpec(kind="plane", axis=axis, **kw)


def test_plane_locus_extreme_planes():
    dom = _dom()
    coords = dom.node_coords()
    m = _plane_locus(dom, _plane("-x"))
    assert m.sum() == 9 * 5
    assert np.allclose(coords[m][:, 0], 0.0)
    m = _plane_locus(dom, _plane("+x"))
    assert np.allclose(coords[m][:, 0], 20.0)
    m = _plane_locus(dom, _plane("x"))  # unsigned = BOTH extremes
    assert m.sum() == 2 * 9 * 5
    assert set(np.unique(coords[m][:, 0])) == {0.0, 20.0}


def test_plane_locus_and_semantics():
    dom = _dom()
    coords = dom.node_coords()
    # "xy" = edges where the x- AND y-conditions hold — 4 vertical lines,
    # NOT a perimeter band
    m = _plane_locus(dom, _plane("xy"))
    assert m.sum() == 4 * 5
    pts = coords[m]
    assert set(np.unique(pts[:, 0])) == {0.0, 20.0}
    assert set(np.unique(pts[:, 1])) == {0.0, 16.0}
    m = _plane_locus(dom, _plane("xyz"))
    assert m.sum() == 8  # the 8 corners
    m = _plane_locus(dom, _plane("+x-y"))  # mixed signs — one edge
    assert m.sum() == 5
    pts = coords[m]
    assert np.allclose(pts[:, 0], 20.0) and np.allclose(pts[:, 1], 0.0)


def test_plane_locus_center_interior():
    dom = _dom()
    coords = dom.node_coords()
    m = _plane_locus(dom, _plane("x", center=[10.0]))  # interior cut
    assert m.sum() == 9 * 5
    assert np.allclose(coords[m][:, 0], 10.0)
    # mid-cell center is inclusive — both adjacent node planes
    m = _plane_locus(dom, _plane("x", center=[9.0]))
    assert set(np.unique(coords[m][:, 0])) == {8.0, 10.0}
    # center binds to the i-th axis LETTER, not the x/y/z component
    m = _plane_locus(dom, _plane("z", center=[4.0]))
    assert np.allclose(coords[m][:, 2], 4.0)


def test_plane_locus_range_clip_and_empty():
    dom = _dom()
    coords = dom.node_coords()
    m = _plane_locus(dom, _plane("x", center=[10.0], z_range=[4.0, 8.0]))
    assert m.sum() == 9 * 3
    assert set(np.unique(coords[m][:, 2])) == {4.0, 6.0, 8.0}
    assert not _plane_locus(dom, _plane("")).any()  # no axis -> nothing
    # out-of-part clip -> empty (and NO bbox fallback downstream)
    assert not _plane_locus(dom, _plane("x", x_range=[100.0, 200.0])).any()


def test_plane_masks():
    dom = _dom()
    nodes = region_node_mask(dom, _plane("-z"))
    coords = dom.node_coords()
    assert np.allclose(coords[nodes][:, 2], 0.0)
    cells = region_cell_mask(dom, _plane("-z"))
    assert cells.shape == DIMS and cells.sum() == 10 * 8  # the k=0 layer
    assert cells[:, :, 0].all() and not cells[:, :, 1:].any()


def test_plane_fixed_dofs():
    dom = _dom()
    case = LoadCase(
        name="sym",
        fixed=[_plane("x", center=[0.0], fix_mode="normal")],
        loads=[LoadSpec(region=RegionSpec(kind="surface", axis="+x"), force_n=[0.0, 0.0, -50.0])],
    )
    fixed3, _F, _w = resolve_case_dofs(dom, case, [], fixed_out=[])
    m = _plane_locus(dom, _plane("x", center=[0.0]))
    assert fixed3[m].tolist() == [[True, False, False]] * int(m.sum())  # ux only
    # knife-edge: "xy" pins ux+uy — rotation about z stays free
    case.fixed = [_plane("xy", fix_mode="normal")]
    fixed3, _F, _w = resolve_case_dofs(dom, case, [])
    m = _plane_locus(dom, _plane("xy"))
    assert fixed3[m].tolist() == [[True, True, False]] * int(m.sum())
    # "xyz" = point support = all three DOFs
    case.fixed = [_plane("xyz", fix_mode="normal")]
    fixed3, _F, _w = resolve_case_dofs(dom, case, [])
    m = _plane_locus(dom, _plane("xyz"))
    assert fixed3[m].all()


def test_plane_no_fallback_and_supports_record():
    dom = _dom()
    supports: list[dict] = []
    case = LoadCase(
        name="miss",
        fixed=[_plane("x", center=[10.0], x_range=[100.0, 200.0], fix_mode="normal")],
        loads=[LoadSpec(region=RegionSpec(kind="surface", axis="+x"), force_n=[0.0, 0.0, -50.0])],
    )
    _f3, _F, warnings = resolve_case_dofs(dom, case, [], fixed_out=supports)
    assert any("matched no nodes" in w for w in warnings)
    assert not any("region_fallback" in w for w in warnings)
    assert supports == []

    supports = []
    case.fixed = [_plane("-x", fix_mode="all")]
    _f3, _F, _w = resolve_case_dofs(dom, case, [], fixed_out=supports)
    assert len(supports) == 1
    rec = supports[0]
    assert rec["kind"] == "plane" and rec["fix_mode"] == "all"
    assert rec["nodes"] == 9 * 5 and rec["cells"] == 8 * 4
    assert rec["centroid_mm"] == [0.0, 8.0, 4.0]
    assert rec["dofs_pinned"] == [45, 45, 45]
    assert "fallback_ax" not in rec


def test_spread_mode_area_resultant_and_weights():
    dom = _dom()
    fixed = [RegionSpec(kind="plane", axis="-x")]
    load_kw: dict = dict(region=RegionSpec(kind="surface", axis="+x"), force_n=[100.0, 0.0, 0.0])
    case_n = LoadCase(name="n", fixed=fixed, loads=[LoadSpec(**load_kw)])
    case_a = LoadCase(name="a", fixed=fixed, loads=[LoadSpec(**load_kw, spread_mode="area")])
    app_n: list[dict] = []
    app_a: list[dict] = []
    _f, F_n, _w = resolve_case_dofs(dom, case_n, [], applied_out=app_n)
    _f, F_a, _w = resolve_case_dofs(dom, case_a, [], applied_out=app_a)
    # identical resultant either way
    assert np.isclose(F_n.sum(), F_a.sum())
    assert np.allclose(F_n.reshape(-1, 3).sum(axis=0), [100.0, 0.0, 0.0])
    # ...but not identical distribution: on the flat +x face the skin-area
    # share weights an interior node 4/3x a corner (p^2 +x-face share vs
    # 3*p^2/4 for +x+y+z exposed faces), while the even split is flat
    assert not np.allclose(F_n, F_a)
    assert app_a[0]["spread"] == "area" and app_n[0]["spread"] == "nodes"
    fa = F_a.reshape(-1, 3)[:, 0]
    idx = np.flatnonzero(fa > 0)
    pts = dom.node_coords()[idx]
    corner_local = np.flatnonzero((pts[:, 1] == 0.0) & (pts[:, 2] == 0.0))
    interior_local = np.flatnonzero((pts[:, 1] == 8.0) & (pts[:, 2] == 4.0))
    assert len(corner_local) and len(interior_local)
    assert fa[idx[interior_local[0]]] > 1.2 * fa[idx[corner_local[0]]]
    # even split: every loaded node carries the same force
    fn = F_n.reshape(-1, 3)[:, 0]
    assert np.allclose(fn[fn > 0], fn[fn > 0][0])


def test_spread_mode_area_buried_region_falls_back():
    dom = _dom()
    case = LoadCase(
        name="b",
        fixed=[RegionSpec(kind="plane", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="box", center=[10.0, 8.0, 4.0], diameter_mm=2.0),
                force_n=[10.0, 0.0, 0.0],
                spread_mode="area",
            )
        ],
    )
    _f, F, warnings = resolve_case_dofs(dom, case, [], applied_out=[])
    assert np.allclose(F.reshape(-1, 3).sum(axis=0), [10.0, 0.0, 0.0])
    assert any("spread_mode 'area' on buried region" in w for w in warnings)


def test_plane_load_resultant():
    dom = _dom()
    case = LoadCase(
        name="l",
        fixed=[RegionSpec(kind="plane", axis="-z")],
        loads=[LoadSpec(region=RegionSpec(kind="plane", axis="+z"), force_n=[0.0, 0.0, -40.0])],
    )
    app: list[dict] = []
    _f3, F, _w = resolve_case_dofs(dom, case, [], applied_out=app)
    assert np.allclose(F.reshape(-1, 3).sum(axis=0), [0.0, 0.0, -40.0])
    assert app[0]["kind"] == "plane" and app[0]["nodes"] == 9 * 11
