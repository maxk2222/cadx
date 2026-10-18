"""Voxel FEM + SIMP topology optimization: solver correctness, region mapping,
mesh extraction validity and the executor pipelines."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.analyze import FemReport, run_fem, run_fem_domain
from cadx.cad.fem.extract import _regularize_mask, _surface_mesh, density_to_mesh
from cadx.cad.fem.regions import (
    empty_region_warnings,
    region_cell_mask,
    region_node_mask,
    resolve_case,
)
from cadx.cad.fem.solver import solve
from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
from cadx.cad.fem.voxelize import VoxelDomain, voxelize
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


# --------------------------------------------------------------------------- #
# voxelization                                                                 #
# --------------------------------------------------------------------------- #


def test_voxelize_box_exact() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    assert dom.dims == (30, 12, 6)
    assert dom.solid_count == 30 * 12 * 6
    np.testing.assert_allclose(dom.origin, [-30, -12, 0])


def test_voxelize_respects_max_elements() -> None:
    dom = voxelize(_box_mesh(), 1.0, 2000)
    assert dom.solid_count <= 2000
    assert dom.pitch > 1.0


def test_voxelize_nonwatertight_stl(tmp_path: Path) -> None:
    """STL-loaded meshes arrive with split vertices — merge must restore
    containment-based filling."""
    p = tmp_path / "b.stl"
    _box_mesh().export(str(p))
    loaded = trimesh.load(str(p), force="mesh", process=False)
    assert not loaded.is_watertight
    dom = voxelize(loaded, 2.5, 150_000)
    assert dom.solid_count > 0
    # face layers must be populated for regions to resolve
    assert dom.solid[0].any() and dom.solid[-1].any()


# --------------------------------------------------------------------------- #
# regions                                                                      #
# --------------------------------------------------------------------------- #


def test_region_face_masks() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    neg = region_cell_mask(dom, RegionSpec(kind="face", axis="-x"))
    pos = region_cell_mask(dom, RegionSpec(kind="face", axis="+x"))
    assert neg[:, :, :].sum() > 0 and pos.sum() > 0
    assert not (neg & pos).any()
    assert neg[0].all() and pos[-1].all()


def test_region_face_masks_padded_grid() -> None:
    """When the bbox is not a multiple of the pitch, voxelize pads the high
    side of each axis with void cells — face regions must anchor at the solid
    extent, not the grid edge."""
    dom = voxelize(_box_mesh(61.0, 25.0, 13.0), 2.0, 150_000)
    assert not dom.solid[-1].any()  # padded high-x layer is void
    for ax in ("+x", "-x", "+y", "-y", "+z", "-z"):
        assert region_cell_mask(dom, RegionSpec(kind="face", axis=ax)).sum() > 0, ax


def test_region_all_and_box() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    allm = region_cell_mask(dom, RegionSpec(kind="all"))
    assert int(allm.sum()) == dom.solid_count
    bm = region_cell_mask(
        dom,
        RegionSpec(kind="box", center=[-30.0, -12.0, 0.0], diameter_mm=8.0),
    )
    assert bm.any()
    assert not bm[-1].any()  # far end not inside the box region


def test_resolve_case_forces() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    fixed, F, w = resolve_case(dom, _cantilever_case())
    assert fixed.sum() > 0
    assert np.isclose(F.sum(), -100.0, atol=1e-6)
    assert w == []


def test_region_surface_follows_curved_shape() -> None:
    """'surface +z' on a hemisphere must select cells BELOW the bbox top —
    the actual upper skin, not a global bbox layer."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=15)
    half = m.slice_plane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], cap=True)
    half.apply_translation([0, 0, 0])
    dom = voxelize(half, 1.5, 150_000)
    cells = region_cell_mask(dom, RegionSpec(kind="surface", axis="+z", margin_mm=1.5))
    assert cells.any()
    z_idxs = np.nonzero(cells.any(axis=(0, 1)))[0]
    # cells on the sloping flanks sit well below the top of the dome
    assert z_idxs.min() < 0.6 * (dom.dims[2] - 1)
    # face is an alias with identical coverage now
    face = region_cell_mask(dom, RegionSpec(kind="face", axis="+z", margin_mm=1.5))
    assert (cells == face).all()


def test_region_area_selects_xy_subregion() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)  # 60x24x12 centred, x in [-30,30]
    cells = region_cell_mask(
        dom, RegionSpec(kind="area", axis="+z", x_range=[0.0, 30.0], margin_mm=1.0)
    )
    assert cells.any()
    centers = dom.element_centers().reshape((*dom.dims, 3))
    xs = centers[..., 0][cells]
    assert xs.min() >= -dom.pitch  # only the +x half (voxel tolerance)
    xs_neg = centers[..., 0][~cells & dom.solid]
    assert xs_neg.min() < 0  # the -x half is not selected
    # polygon triangle selects only inside vertices
    tri = [[0.0, -20.0], [30.0, -20.0], [30.0, 20.0]]
    pcells = region_cell_mask(dom, RegionSpec(kind="area", polygon=tri, margin_mm=1.0))
    assert pcells.any()
    assert centers[..., 0][pcells].min() > -dom.pitch


def test_region_hole_anchor() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    anchors = [{"x": 10.0, "y": 0.0, "dia": 4.0}]
    cells = region_cell_mask(dom, RegionSpec(kind="hole", hole_index=1, margin_mm=2.0), anchors)
    assert cells.any()
    centers = dom.element_centers().reshape((*dom.dims, 3))
    sel = centers[cells]
    assert np.linalg.norm(sel[:, :2] - [10.0, 0.0], axis=1).max() < 2.0 + 2.0 + dom.pitch
    # out-of-range index selects nothing
    empty = region_cell_mask(dom, RegionSpec(kind="hole", hole_index=9), anchors)
    assert not empty.any()
    warnings = empty_region_warnings(dom, [RegionSpec(kind="hole", hole_index=9)], anchors)
    assert warnings and "no cells" in warnings[0]
    # 1-based contract: index 0 is rejected at the schema boundary, not dropped
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RegionSpec(kind="hole", hole_index=0)


def test_region_hole_grows_past_countersink() -> None:
    """Measured anchor dia (2.5) smaller than the real bore (5+): the region
    must grow past the void bore until it grips material."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    # bore out a Ø10 cylinder at the anchor point — the initial disc
    # (radius 1.25 + 2.0 margin) lands fully inside the void
    centers = dom.element_centers().reshape((*dom.dims, 3))
    bore = (centers[..., 0] - 10.0) ** 2 + (centers[..., 1] - 0.0) ** 2 <= 5.0**2
    solid = dom.solid & ~bore
    dom = VoxelDomain(dom.origin, dom.pitch, dom.dims, solid)
    anchors = [{"x": 10.0, "y": 0.0, "dia": 2.5}]
    cells = region_cell_mask(dom, RegionSpec(kind="hole", hole_index=1, margin_mm=2.0), anchors)
    assert cells.any()
    sel = centers[cells]
    d = np.linalg.norm(sel[:, :2] - [10.0, 0.0], axis=1)
    assert d.min() >= 5.0  # the ring, not the bore
    assert d.max() <= 5.0 + 4 * 2.0 + dom.pitch


def test_pressure_force_total_matches_area() -> None:
    """1 MPa on the +z face of a 40x20 plate = 800 N distributed over nodes."""
    mesh = _box_mesh(40.0, 20.0, 4.0)
    dom = voxelize(mesh, 2.0, 150_000)
    case = LoadCase(
        name="press",
        fixed=[RegionSpec(kind="surface", axis="-z")],
        loads=[LoadSpec(region=RegionSpec(kind="surface", axis="+z"), pressure_mpa=1.0)],
    )
    fixed, F, w = resolve_case(dom, case)
    assert fixed.any()
    total = F.reshape(-1, 3).sum(axis=0)
    assert total[2] == pytest.approx(-800.0, rel=0.05)
    assert any("800" in x or "MPa" in x for x in w)


def test_pressure_on_all_is_hydrostatic_squeeze() -> None:
    """region kind=all + pressure: every exposed face pushed inward —
    symmetric box nets ~0 resultant but carries real compressive load."""
    mesh = _box_mesh(40.0, 20.0, 4.0)
    dom = voxelize(mesh, 2.0, 150_000)
    case = LoadCase(
        name="squeeze",
        fixed=[RegionSpec(kind="surface", axis="-z")],
        loads=[LoadSpec(region=RegionSpec(kind="all"), pressure_mpa=1.0)],
    )
    fixed, F, w = resolve_case(dom, case)
    assert fixed.any()
    assert F.any()  # real nodal loads — not the old zero-force rejection
    total = F.reshape(-1, 3).sum(axis=0)
    # hydrostatic on a centred box: resultant cancels out
    assert np.linalg.norm(total) < 0.15 * np.abs(F).sum()
    # ~2080 mm² box surface → ~2080 N total compressive
    assert any("MPa over" in x for x in w)
    assert not any("pressure loads need" in x for x in w)


def test_region_points_selects_surface_patch() -> None:
    """GUI brush: solid cells within radius of painted surface points."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)  # 60x24x12, z top at 12
    cells = region_cell_mask(
        dom,
        RegionSpec(kind="points", points=[[10.0, 0.0, 12.0]], radius_mm=5.0),
    )
    assert cells.any()
    centers = dom.element_centers().reshape((*dom.dims, 3))
    sel = centers[cells]
    # all selected cells near the picked point
    assert np.linalg.norm(sel - [10.0, 0.0, 12.0], axis=1).max() < 5.0 + dom.pitch + 1e-6
    # the far corner is untouched
    assert not cells[0, 0, 0]
    # multiple points union
    cells2 = region_cell_mask(
        dom,
        RegionSpec(
            kind="points",
            points=[[10.0, 0.0, 12.0], [-20.0, 0.0, 12.0]],
            radius_mm=5.0,
        ),
    )
    assert cells2.sum() > cells.sum()


def test_region_points_empty_and_nodes() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    empty = region_cell_mask(dom, RegionSpec(kind="points"))
    assert not empty.any()
    mask = region_node_mask(dom, RegionSpec(kind="points", points=[[0.0, 0.0, 12.0]]))
    assert mask.any()


def test_points_pressure_uses_picked_normal() -> None:
    """Pressure on a painted patch: total force ≈ pressure * painted area,
    acting against the stored surface normal (here +z → -z force)."""
    mesh = _box_mesh(40.0, 20.0, 4.0)
    dom = voxelize(mesh, 2.0, 150_000)
    region = RegionSpec(
        kind="points", points=[[0.0, 0.0, 4.0]], radius_mm=5.0, normal=[0.0, 0.0, 1.0]
    )
    case = LoadCase(
        name="paint",
        fixed=[RegionSpec(kind="surface", axis="-z")],
        loads=[LoadSpec(region=region, pressure_mpa=1.0)],
    )
    fixed, F, _w = resolve_case(dom, case)
    assert fixed.any()
    total = F.reshape(-1, 3).sum(axis=0)
    assert total[2] < 0  # pushes into the part
    cells = region_cell_mask(dom, region)
    expected = float(cells.sum()) * dom.pitch**2  # rough painted area upper bound
    assert 0 < abs(total[2]) <= expected * 1.5


def test_region_points_rasterized_matches_naive() -> None:
    """Index-space rasterisation must select exactly the naive full-domain
    sphere union — the fast path must not drop boundary cells."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    pts = [[-15.0, 0.0, 12.0], [10.0, 5.0, 12.0]]
    cells = region_cell_mask(dom, RegionSpec(kind="points", points=pts, radius_mm=5.0))
    centers = dom.element_centers().reshape((*dom.dims, 3))
    radius = 5.0 + dom.pitch  # same dilation as _points_cells
    naive = np.zeros(dom.dims, dtype=bool)
    for p in pts:
        naive |= np.sum((centers - np.asarray(p)) ** 2, axis=-1) <= radius * radius
    naive &= dom.solid
    assert (cells == naive).all()


def test_region_points_dense_stroke_dedupes() -> None:
    """A 2000-point stroke of near-duplicates must not blow up cost or
    selection — it resolves to ~the same patch as a single point."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    pts = [[10.0 + i * 0.001, 0.0, 12.0] for i in range(2000)]
    cells = region_cell_mask(dom, RegionSpec(kind="points", points=pts, radius_mm=5.0))
    ref = region_cell_mask(
        dom, RegionSpec(kind="points", points=[[10.0, 0.0, 12.0]], radius_mm=5.0)
    )
    assert cells.any()
    # dedupe keeps a handful of spheres — the patch grows by ~the 2mm spread
    assert int(cells.sum()) <= int(ref.sum()) + 60


def test_resolve_case_empty_region_warns() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    case = LoadCase(
        name="bad",
        # no axis/anchor/position hint at all — nothing the bbox fallback
        # could derive a boundary patch from, so this stays unresolved
        fixed=[RegionSpec(kind="hole", hole_index=5)],
        loads=[LoadSpec(region=RegionSpec(kind="face", axis="+x"), force_n=[0, 0, -1])],
    )
    fixed, _f, w = resolve_case(dom, case)
    assert not fixed.any()
    assert any("matched no nodes" in x for x in w)


def test_resolve_case_region_fallback_rescues_declared_region() -> None:
    """A declared region that resolves to zero cells (here: a painted patch
    whose point sits off the regenerated mesh) falls back to a deterministic
    bbox boundary patch — the case stays solvable and the substitution is
    recorded as a region_fallback warning, not a silent drop."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    case = LoadCase(
        name="fb",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(
                    kind="points", points=[[0.0, 0.0, 999.0]], normal=[0.0, 0.0, 1.0]
                ),
                force_n=[0.0, 0.0, -50.0],
            )
        ],
    )
    fixed, F, w = resolve_case(dom, case)
    assert fixed.any()
    assert np.isclose(F.sum(), -50.0)
    fb = [x for x in w if "region_fallback" in x]
    assert fb and "+z" in fb[0]
    assert not any("matched no nodes" in x for x in w)


def test_resolve_case_fallback_clips_to_anchor_face() -> None:
    """hole_index region whose bore void outgrew the ring search: the
    fallback maps it onto the anchor's own axis face near the anchor, so a
    declared-but-missing bore still carries load at its intended side."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    # giant void cylinder at the anchor -> the growing ring (max r≈12 mm)
    # never reaches the solid rim
    centers = dom.element_centers().reshape((*dom.dims, 3))
    void = (centers[..., 0] - 10.0) ** 2 + (centers[..., 1] - 0.0) ** 2 <= 16.0**2
    dom = VoxelDomain(dom.origin, dom.pitch, dom.dims, dom.solid & ~void)
    anchors = [{"x": 10.0, "y": 0.0, "dia": 4.0, "axis": "+z"}]
    region = RegionSpec(kind="hole", hole_index=1, margin_mm=2.0)
    assert not region_cell_mask(dom, region, anchors).any()  # primary path fails
    case = LoadCase(
        name="bolt",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[LoadSpec(region=region, force_n=[0.0, 0.0, -50.0])],
    )
    _fixed, F, w = resolve_case(dom, case, anchors)
    assert np.isclose(F.sum(), -50.0)
    fb = [x for x in w if "region_fallback" in x]
    assert fb and "+z" in fb[0]
    # the anchor neighbourhood is void, so the patch widens to the surviving
    # +z skin — still on the anchor's declared side, never interior cells
    from cadx.cad.fem.regions import _region_fallback

    fb_cells, ax = _region_fallback(dom, region, anchors)
    assert ax == "+z" and fb_cells.any()
    sel = centers[fb_cells]
    assert sel[:, 2].min() >= 12.0 - 2 * (2.0 + dom.pitch)  # top-skin band
    assert np.linalg.norm(sel[:, :2] - [10.0, 0.0], axis=1).min() >= 16.0 - dom.pitch


# --------------------------------------------------------------------------- #
# solver                                                                       #
# --------------------------------------------------------------------------- #


def test_cantilever_euler_bernoulli() -> None:
    """60x24x12 cantilever, -100 N on the tip face: FEM within ~30% of EB."""
    mesh = _box_mesh()
    dom = voxelize(mesh, 2.0, 150_000)
    fixed, F, _ = resolve_case(dom, _cantilever_case())
    sol = solve(dom, np.full(dom.cell_count, E_PLA), NU, fixed, F)
    assert sol.converged
    inertia = 24.0 * 12.0**3 / 12.0
    eb = abs(100.0 * 60.0**3 / (3 * E_PLA * inertia))
    assert 0.6 < sol.max_disp_mm / eb < 1.4
    assert sol.max_vm_mpa > 0


def test_singular_without_fixed() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    fixed = np.zeros(dom.node_count, dtype=bool)
    F = np.zeros(3 * dom.node_count)
    sol = solve(dom, np.full(dom.cell_count, E_PLA), NU, fixed, F)
    assert sol.max_disp_mm == 0.0


# --------------------------------------------------------------------------- #
# SIMP                                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def simp_setup():
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    case = _cantilever_case()
    fixed, F, _ = resolve_case(dom, case)
    passive = passive_solid_mask(dom, [], [fixed], [region_node_mask(dom, case.loads[0].region)])
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.35, rmin_mm=3.0, max_iter=40, tol=0.01),
    )
    return dom, case, fixed, res


def test_simp_converges_to_volfrac(simp_setup) -> None:
    dom, _case, _fixed, res = simp_setup
    assert res.iterations > 0
    achieved = res.density[dom.solid.ravel()].sum() / dom.solid_count
    assert math.isclose(achieved, 0.35, abs_tol=0.05)
    assert res.history


def test_simp_preserves_passive_regions(simp_setup) -> None:
    dom, case, fixed, res = simp_setup
    passive = passive_solid_mask(dom, [], [fixed], [region_node_mask(dom, case.loads[0].region)])
    rho = res.density.reshape(dom.dims)
    assert (rho.reshape(-1)[passive] > 0.9).all()


# --------------------------------------------------------------------------- #
# extraction                                                                   #
# --------------------------------------------------------------------------- #


def test_extract_solid_box_watertight() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    mesh, _mask, _w, _nc, _info = density_to_mesh(
        dom, dom.solid.ravel().astype(float), smooth_iterations=0
    )
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert math.isclose(mesh.volume, 60 * 24 * 12, rel_tol=0.02)


def test_extract_topopt_watertight(simp_setup) -> None:
    dom, case, fixed, res = simp_setup
    passive = passive_solid_mask(dom, [], [fixed], [region_node_mask(dom, case.loads[0].region)])
    mesh, _mask, _w, _nc, _info = density_to_mesh(dom, res.density, passive_solid=passive)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    # must not grow beyond the design space bbox
    lo, hi = dom.origin, dom.origin + dom.pitch * np.asarray(dom.dims)
    assert (mesh.bounds[0] >= lo - 1e-6).all() and (mesh.bounds[1] <= hi + 1e-6).all()


def test_regularize_removes_diagonal_edges() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 1] = True  # edge-diagonal to the first cell
    reg = _regularize_mask(mask)
    dom = VoxelDomain(origin=np.zeros(3), pitch=1.0, dims=reg.shape, solid=reg)
    m = _surface_mesh(dom, reg)
    assert m.is_watertight
    assert reg.sum() > mask.sum()  # filler cell added


# --------------------------------------------------------------------------- #
# pipelines                                                                    #
# --------------------------------------------------------------------------- #


def test_run_fem_pipeline(tmp_path: Path) -> None:
    from cadx.cad.engine import run_fem_pipeline

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    out = run_fem_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case().model_dump()],
        material="pla",
        voxel_mm=3.0,
    )
    assert out["ok"]
    rep = FemReport(**out["fem"])
    assert rep.cases[0].safety_factor is not None
    assert rep.cases[0].max_disp_mm > 0
    assert (tmp_path / "o" / "fem" / "report.json").exists()


def test_run_topopt_pipeline(tmp_path: Path) -> None:
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case().model_dump()],
        optimize={"target_volume_fraction": 0.4},
        material="pla",
        voxel_mm=3.0,
    )
    assert res.ok, res.error
    assert res.topopt["converged"] or res.topopt["iterations"] > 0
    assert 0.3 <= res.topopt["volfrac_achieved"] <= 0.5
    assert res.fem["cases"][0]["safety_factor"] is not None
    # export verification + printability metrics
    assert res.fem_export
    assert res.topopt["sf_export"] is not None
    assert res.topopt["min_member_mm"] > 0
    assert 0.0 <= res.topopt["thin_fraction"] <= 1.0
    stl = tmp_path / "o" / "_run" / "model.stl"
    assert stl.exists()
    m = trimesh.load(str(stl), force="mesh", process=False)
    m.merge_vertices()
    assert m.is_watertight
    assert (tmp_path / "o" / "topopt" / "density.npz").exists()
    assert (tmp_path / "o" / "fem" / "report.json").exists()
    assert (tmp_path / "o" / "fem" / "report_export.json").exists()


def test_run_topopt_pipeline_retries_on_low_sf(tmp_path: Path) -> None:
    """sf_export below target → automatic retry at volfrac +0.1."""
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case().model_dump()],
        optimize={"target_volume_fraction": 0.4, "safety_factor_min": 1e6},
        material="pla",
        voxel_mm=3.0,
    )
    assert res.ok, res.error
    assert "retry" in res.topopt
    assert res.topopt["retry"]["volfrac_second"] == pytest.approx(0.5)


def test_run_fem_pipeline_preview_only(tmp_path: Path) -> None:
    from cadx.cad.engine import run_fem_pipeline

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    out = run_fem_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case().model_dump()],
        voxel_mm=3.0,
        preview_only=True,
    )
    assert out["ok"]
    cases = out["preview"]["cases"]
    assert cases[0]["fixed_nodes"] > 0
    assert cases[0]["loaded_nodes"] > 0
    assert cases[0]["total_force_n"] == pytest.approx(100.0)
    if out["preview"]["regions_png"]:  # requires pyvista
        assert (tmp_path / "o" / "fem" / out["preview"]["regions_png"][0]).exists()
        assert (tmp_path / "o" / "fem" / out["preview"]["regions_png"][0]).stat().st_size > 5000


def test_run_fem_pipeline_preview_cells_no_pngs(tmp_path: Path) -> None:
    """GUI region check: render_pngs=False skips renders but still returns
    the resolved cell clouds for the 3D overlay."""
    from cadx.cad.engine import run_fem_pipeline

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    out = run_fem_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case().model_dump()],
        voxel_mm=3.0,
        preview_only=True,
        render_pngs=False,
    )
    assert out["ok"]
    prev = out["preview"]
    assert prev["regions_png"] == []
    cells = prev["cells"]
    assert cells and cells[0]["case"] == "tip"
    assert len(cells[0]["fixed"]) > 0
    assert len(cells[0]["load"]) > 0
    # cell centres lie inside the part's world bounds (60x24x12, centred)
    arr = np.asarray(cells[0]["fixed"], dtype=float)
    assert arr[:, 0].min() >= -30.0 - 0.01 and arr[:, 0].max() <= 30.0 + 0.01
    assert arr[:, 2].min() >= -0.01 and arr[:, 2].max() <= 12.0 + 0.01


def test_run_fem_domain_matches_mask(simp_setup) -> None:
    """Verification on the exact voxel mask (what extraction exports)."""
    dom, case, fixed, res = simp_setup
    passive = passive_solid_mask(dom, [], [fixed], [region_node_mask(dom, case.loads[0].region)])
    _, mask, _, _nc, _info = density_to_mesh(dom, res.density, passive_solid=passive)
    rep = run_fem_domain(dom, mask, [case], "pla")
    c = rep.cases[0]
    assert c.safety_factor is not None
    assert 0 < c.max_disp_mm < 100


def test_run_fem_no_cases(tmp_path: Path) -> None:
    rep = run_fem(_box_mesh(), [], "pla", voxel_mm=4.0)
    assert "no load cases provided" in rep.warnings


def test_extract_flags_disconnected_keep_island() -> None:
    """A keep region that ends up as its own island must surface as
    n_components > 1 + a warning — a floating ring is a multi-body defect,
    not a successful optimization."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    nx, ny, nz = dom.dims
    density = np.zeros(nx * ny * nz)
    # main blob: half the bar is dense
    dense = np.zeros(dom.dims, dtype=bool)
    dense[:5] = True
    # island: a small dense cube at the far end, spatially disconnected
    dense[nx - 4 : nx - 1, 4:7, 2:5] = True
    density[dense.ravel()] = 1.0
    # passive = fix/load cells on the main blob + keep cells on the island —
    # like a mounting ring whose material bridge got optimized away
    passive = np.zeros(dom.dims, dtype=bool)
    passive[0] = True
    passive[nx - 4 : nx - 1, 4:7, 2:5] = True
    mesh, _mask, warnings, n_components, _info = density_to_mesh(
        dom, density, passive_solid=passive.ravel(), smooth_iterations=0
    )
    assert n_components == 2
    assert any("disconnected" in w for w in warnings)
    assert len(mesh.split()) == 2


def test_run_topopt_pipeline_void_regions_survive_extraction(tmp_path: Path) -> None:
    """Declared void_regions must stay empty in the extracted mask — even under
    min-member thickening (audit fix: void_cells was never passed to
    density_to_mesh)."""
    from cadx.cad.engine import run_topopt_pipeline
    from cadx.cad.fem.regions import region_cell_mask

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case().model_dump()],
        optimize={
            "target_volume_fraction": 0.4,
            # large interior void box + aggressive thickening that would
            # re-fill it if the mask weren't re-carved
            "void_regions": [{"kind": "box", "center": [0.0, 0.0, 0.0], "diameter_mm": 16.0}],
            "min_member_mm": 6.0,
        },
        material="pla",
        voxel_mm=3.0,
    )
    assert res.ok, res.error
    npz = np.load(tmp_path / "o" / "topopt" / "density.npz")
    domain_mask = np.asarray(npz["mask"], dtype=bool)
    from cadx.cad.fem.voxelize import voxelize

    dom = voxelize(trimesh.load_mesh(str(src)), 3.0, 500_000)
    void = region_cell_mask(dom, RegionSpec(kind="box", center=[0.0, 0.0, 0.0], diameter_mm=16.0))
    interior = domain_mask & np.asarray(void).reshape(dom.dims)
    assert not interior.any(), f"{int(interior.sum())} voxels grew into the void region"
    # the npz must persist the void mask so resample can re-carve
    assert "voids" in npz
    assert bool(npz["voids"].any())


def test_simp_oc_warns_on_mma_only_options(tmp_path: Path) -> None:
    """solver=oc with stress/displacement limits warns instead of silently
    ignoring them (audit risk item)."""
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case().model_dump()],
        optimize={"target_volume_fraction": 0.4, "max_stress_mpa": 50.0, "solver": "oc"},
        material="pla",
        voxel_mm=3.0,
    )
    assert res.ok, res.error
    warn_blob = " ".join(res.topopt.get("warnings") or [])
    assert "solver=oc" in warn_blob and "mma" in warn_blob
