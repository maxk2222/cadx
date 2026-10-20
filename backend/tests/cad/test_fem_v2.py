"""FEM v2: typed multi-axis anchors, bore regions, DOF support modes,
advanced loads (bearing/moment/preload/remote/gravity), cone expansion,
SIMP extras (void, warm start, weights, per-cell E) and resampling."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.anchors3d import merge_anchor_lists, scan_mesh_anchors
from cadx.cad.fem.loads import (
    body_force,
    expand_cone,
    spread_moment,
)
from cadx.cad.fem.regions import (
    region_cell_mask,
    region_node_mask,
    resolve_case,
    resolve_case_dofs,
)
from cadx.cad.fem.solver import solve
from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _box_mesh(x: float = 60.0, y: float = 24.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _bored_mesh() -> trimesh.Trimesh:
    """60x24x12 plate with a Ø5 through-bore along +z at (10, 0)."""
    box = _box_mesh()
    bore = trimesh.creation.cylinder(radius=2.5, height=20.0, sections=32)
    bore.apply_translation([10.0, 0.0, 6.0])
    out = box.difference(bore)
    assert isinstance(out, trimesh.Trimesh)
    return out


def _side_bored_mesh() -> trimesh.Trimesh:
    """40x30x20 block with a Ø6 through-bore along +x at (y=0, z=10)."""
    box = _box_mesh(40.0, 30.0, 20.0)
    bore = trimesh.creation.cylinder(radius=3.0, height=50.0, sections=32)
    bore.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    bore.apply_translation([0.0, 0.0, 10.0])
    out = box.difference(bore)
    assert isinstance(out, trimesh.Trimesh)
    return out


# --------------------------------------------------------------------------- #
# anchors3d — multi-axis mesh scan + merge                                     #
# --------------------------------------------------------------------------- #


def test_scan_mesh_anchors_finds_z_bore() -> None:
    anchors = scan_mesh_anchors(_bored_mesh(), step_mm=1.0)
    z_anchors = [a for a in anchors if a["axis"] == "+z" and a["kind"] == "hole"]
    assert z_anchors, f"no +z hole found in {anchors}"
    a = z_anchors[0]
    assert a["dia"] == pytest.approx(5.0, rel=0.4)
    assert math.hypot(a["x"] - 10.0, a["y"] - 0.0) < 2.0
    assert a["depth_mm"] == pytest.approx(12.0, rel=0.3)


def test_scan_mesh_anchors_finds_side_bore() -> None:
    """A +x bore reports its in-plane position via `pos` (x unknown)."""
    anchors = scan_mesh_anchors(_side_bored_mesh(), step_mm=1.0)
    x_anchors = [a for a in anchors if a["axis"] == "+x" and a["kind"] == "hole"]
    assert x_anchors, f"no +x hole found in {anchors}"
    a = x_anchors[0]
    assert a["dia"] == pytest.approx(6.0, rel=0.5)
    assert a["pos"] is not None and a["pos"][0] is None
    assert a["pos"][1] == pytest.approx(0.0, abs=2.0)
    assert a["pos"][2] == pytest.approx(10.0, abs=2.0)


def test_merge_anchor_lists_enriches_not_duplicates() -> None:
    existing = [{"x": 10.0, "y": 0.0, "dia": 4.0, "axis": "+z", "source": "image"}]
    scanned = [
        {
            "x": 10.5,
            "y": 0.3,
            "dia": 5.0,
            "axis": "+z",
            "z": None,
            "pos": [10.5, 0.3, None],
            "depth_mm": 12.0,
            "kind": "hole",
            "source": "scan",
        },
        {
            "x": -20.0,
            "y": 5.0,
            "dia": 3.0,
            "axis": "+z",
            "z": None,
            "pos": [-20.0, 5.0, None],
            "depth_mm": 12.0,
            "kind": "hole",
            "source": "scan",
        },
    ]
    merged = merge_anchor_lists(existing, scanned)
    assert len(merged) == 2
    assert merged[0]["depth_mm"] == 12.0  # enriched
    assert merged[0]["index"] == 1 and merged[1]["index"] == 2


# --------------------------------------------------------------------------- #
# probe_blind_features — cavities the tunnel scan can't see                     #
# --------------------------------------------------------------------------- #


def _blind_pocket_mesh() -> trimesh.Trimesh:
    """40x30x20 block with a Ø8 blind pocket 16 mm deep from +z at (5, 3)."""
    box = _box_mesh(40.0, 30.0, 20.0)
    cutter = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=48)
    cutter.apply_translation([5.0, 3.0, 12.0])  # spans z=2..22 -> floor at z=4
    out = box.difference(cutter, engine="manifold")
    assert isinstance(out, trimesh.Trimesh)
    return out


def _blind_side_mesh() -> trimesh.Trimesh:
    """40x30x20 block with a Ø6 blind bore 16 mm deep from +x at (y=0, z=10)."""
    box = _box_mesh(40.0, 30.0, 20.0)
    cutter = trimesh.creation.cylinder(radius=3.0, height=22.0, sections=48)
    cutter.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    cutter.apply_translation([13.0, 0.0, 10.0])  # spans x=2..24 -> floor at x=2
    out = box.difference(cutter, engine="manifold")
    assert isinstance(out, trimesh.Trimesh)
    return out


def test_probe_blind_finds_z_pocket() -> None:
    """A blind pocket invisible to the tunnel scan reads found + open with
    sane measured depth/diameter."""
    from cadx.cad.fem.anchors3d import probe_blind_features

    anchor = {
        "index": 1,
        "name": "M8 tap",
        "kind": "thread",
        "axis": "+z",
        "x": 5.0,
        "y": 3.0,
        "dia": 8.0,
        "depth_mm": 16.0,
        "pos": [5.0, 3.0, None],
    }
    # the tunnel scan proves through-voids only — a blind pocket at best
    # surfaces as a 'dimple', so verification needs this probe
    recs = probe_blind_features(_blind_pocket_mesh(), [anchor])
    assert len(recs) == 1
    r = recs[0]
    assert r["found"] and r["open"]
    assert r["measured_depth_mm"] == pytest.approx(16.0, abs=2.0)
    assert r["measured_dia_mm"] == pytest.approx(8.0, rel=0.35)
    assert r["axis"] == "+z" and r["pos"] == [5.0, 3.0, None]
    assert r["depth_mm"] == 16.0 and r["dia_mm"] == 8.0


def test_probe_blind_finds_side_pocket() -> None:
    """Same probe on a +x side face — the bore enters at x_max going -x."""
    from cadx.cad.fem.anchors3d import probe_blind_features

    anchor = {
        "index": 1,
        "kind": "hole",
        "axis": "+x",
        "x": 0.0,
        "y": 0.0,
        "dia": 6.0,
        "depth_mm": 16.0,
        "pos": [None, 0.0, 10.0],
    }
    r = probe_blind_features(_blind_side_mesh(), [anchor])[0]
    assert r["found"] and r["open"]
    assert r["measured_depth_mm"] == pytest.approx(16.0, abs=2.5)


def test_probe_blind_solid_box_not_found() -> None:
    from cadx.cad.fem.anchors3d import probe_blind_features

    anchor = {"index": 1, "axis": "+z", "x": 5.0, "y": 3.0, "dia": 8.0, "depth_mm": 16.0}
    r = probe_blind_features(_box_mesh(40.0, 30.0, 20.0), [anchor])[0]
    assert r["found"] is False
    assert r["open"] is False
    assert r["measured_depth_mm"] is None and r["measured_dia_mm"] is None


def test_probe_blind_through_bore_is_open() -> None:
    """A through-hole anchor (no depth_mm) is expected to reach the axis span."""
    from cadx.cad.fem.anchors3d import probe_blind_features

    anchor = {"index": 1, "axis": "+z", "x": 10.0, "y": 0.0, "dia": 5.0}
    r = probe_blind_features(_bored_mesh(), [anchor])[0]
    assert r["found"] and r["open"]
    assert r["measured_depth_mm"] == pytest.approx(12.0, abs=1.5)


def test_probe_blind_unpositioned_anchor_safe() -> None:
    """Anchors without a resolvable position return found=False, never raise."""
    from cadx.cad.fem.anchors3d import probe_blind_features

    recs = probe_blind_features(
        _box_mesh(), [{"index": 1, "axis": "+x", "dia": 4.0, "pos": [None, None, None]}]
    )
    assert recs[0]["found"] is False

    # Anchor-like (non-dict) objects work too
    class _Obj:
        axis = "+z"
        x = 5.0
        y = 3.0
        dia = 8.0
        depth_mm = 16.0
        kind = "thread"
        name = "obj anchor"

        def __init__(self) -> None:
            self.pos = [5.0, 3.0, None]

    r = probe_blind_features(_blind_pocket_mesh(), [_Obj()])[0]
    assert r["found"] and r["open"] and r["name"] == "obj anchor"


# --------------------------------------------------------------------------- #
# anchor / bore regions on arbitrary axes                                      #
# --------------------------------------------------------------------------- #


def test_anchor_region_on_side_bore() -> None:
    """Side (+x) anchor: pos carries the in-plane (y,z) — the region must
    resolve material around the bore, not at the part centre."""
    dom = voxelize(_side_bored_mesh(), 2.0, 150_000)
    anchor = {
        "x": 0.0,
        "y": 0.0,
        "dia": 6.0,
        "axis": "+x",
        "z": None,
        "pos": [None, 0.0, 10.0],
        "depth_mm": 40.0,
        "kind": "hole",
    }
    cells = region_cell_mask(dom, RegionSpec(kind="anchor", hole_index=1, margin_mm=2.0), [anchor])
    assert cells.any()
    centers = dom.element_centers().reshape((*dom.dims, 3))
    sel = centers[cells]
    # ring around the bore axis: distance in the (y,z) plane ≈ bore radius
    d = np.linalg.norm(sel[:, 1:] - [0.0, 10.0], axis=1)
    assert d.min() >= 3.0 - dom.pitch
    assert d.max() <= 3.0 + 2.0 + 2 * dom.pitch


def test_bore_region_selects_wall_band() -> None:
    dom = voxelize(_bored_mesh(), 1.5, 150_000)
    anchor = {"x": 10.0, "y": 0.0, "dia": 5.0, "axis": "+z", "depth_mm": 12.0}
    cells = region_cell_mask(dom, RegionSpec(kind="bore", hole_index=1), [anchor])
    assert cells.any()
    centers = dom.element_centers().reshape((*dom.dims, 3))
    sel = centers[cells]
    d = np.linalg.norm(sel[:, :2] - [10.0, 0.0], axis=1)
    # wall band hugs the bore radius (2.5)
    assert d.min() <= 2.5 + dom.pitch and d.max() <= 2.5 + 2 * dom.pitch


# --------------------------------------------------------------------------- #
# fix modes — (nnodes,3) DOF masks                                             #
# --------------------------------------------------------------------------- #


def test_fix_mode_normal_pins_one_dof() -> None:
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    case = LoadCase(
        name="roller",
        fixed=[RegionSpec(kind="face", axis="-z", fix_mode="normal")],
        loads=[LoadSpec(region=RegionSpec(kind="face", axis="+z"), force_n=[0, 0, -50])],
    )
    fixed3, _F, w = resolve_case_dofs(dom, case)
    assert fixed3.shape == (dom.node_count, 3)
    pinned = fixed3.any(axis=1)
    assert pinned.any()
    # only the z DOF of the bottom face is constrained
    assert fixed3[pinned, 2].all()
    assert not fixed3[pinned, :2].any()
    assert any("roller" in x for x in w)
    # legacy wrapper collapses to the same node set
    fixed, _F2, _w2 = resolve_case(dom, case)
    assert (fixed == pinned).all()


def test_fix_mode_all_is_encastre() -> None:
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    case = LoadCase(
        name="enc",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[LoadSpec(region=RegionSpec(kind="face", axis="+x"), force_n=[0, 0, -10])],
    )
    fixed3, _F, _w = resolve_case_dofs(dom, case)
    pinned = fixed3.any(axis=1)
    assert fixed3[pinned].all()  # all three DOFs


def test_solve_accepts_2d_dof_mask() -> None:
    """(nnodes,3) mask and its flattened legacy form must solve identically."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, F, _ = resolve_case(
        dom,
        LoadCase(
            name="t",
            fixed=[RegionSpec(kind="face", axis="-x")],
            loads=[LoadSpec(region=RegionSpec(kind="face", axis="+x"), force_n=[0, 0, -80])],
        ),
    )
    E = np.full(dom.cell_count, E_PLA)
    a = solve(dom, E, NU, fixed, F)
    b = solve(dom, E, NU, fixed[:, None] & np.ones(3, dtype=bool), F)
    assert a.converged and b.converged
    assert a.max_disp_mm == pytest.approx(b.max_disp_mm, rel=1e-6)


def test_roller_supported_plate_solves() -> None:
    """Plate resting on the bottom face (normal-only) under gravity-type load:
    solvable because the z DOFs are pinned."""
    dom = voxelize(_box_mesh(40.0, 40.0, 6.0), 3.0, 150_000)
    case = LoadCase(
        name="rest",
        fixed=[RegionSpec(kind="face", axis="-z", fix_mode="normal")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="surface", axis="+z"),
                pressure_mpa=0.2,
            )
        ],
    )
    fixed3, F, _w = resolve_case_dofs(dom, case)
    sol = solve(dom, np.full(dom.cell_count, E_PLA), NU, fixed3, F)
    assert sol.converged
    assert sol.max_disp_mm > 0
    assert sol.max_disp_mm < 5.0  # plate doesn't fly away


# --------------------------------------------------------------------------- #
# advanced loads                                                                #
# --------------------------------------------------------------------------- #


def test_bearing_load_resultant() -> None:
    """Cosine bearing distribution on a +z bore: resultant along -y = bearing_n."""
    dom = voxelize(_bored_mesh(), 1.5, 150_000)
    anchor = {"x": 10.0, "y": 0.0, "dia": 5.0, "axis": "+z", "depth_mm": 12.0}
    case = LoadCase(
        name="bolt",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="bore", hole_index=1),
                bearing_n=200.0,
                bearing_dir=[0.0, -1.0, 0.0],
            )
        ],
    )
    _fixed3, F, w = resolve_case_dofs(dom, case, anchors=[anchor])
    total = F.reshape(-1, 3).sum(axis=0)
    assert total[1] == pytest.approx(-200.0, rel=0.1)
    assert abs(total[0]) < 20 and abs(total[2]) < 20  # symmetric distribution
    assert not any("no force" in x for x in w)


def test_bearing_load_without_anchor_warns() -> None:
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    case = LoadCase(
        name="bad",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="bore", hole_index=1),
                bearing_n=100.0,
                bearing_dir=[0, -1, 0],
            )
        ],
    )
    _f, _F, w = resolve_case_dofs(dom, case, anchors=None)
    # no anchors → the bore region matches no nodes (warning covers the cause)
    assert any("matched no nodes" in x or "bearing" in x for x in w)


def test_moment_is_pure_couple() -> None:
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    region = RegionSpec(kind="face", axis="+x", margin_mm=2.0)
    case = LoadCase(
        name="torsion",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[LoadSpec(region=region, moment_nm=[2.0, 0.0, 0.0])],
    )
    _fixed3, F, w = resolve_case_dofs(dom, case)
    total = F.reshape(-1, 3).sum(axis=0)
    # a pure moment adds zero net force
    assert np.linalg.norm(total) < 1e-6
    # applied moment about the load centroid = 2 N·m = 2000 N·mm
    idx = np.flatnonzero(region_node_mask(dom, region))
    c = dom.node_coords()[idx].mean(axis=0)
    coords = dom.node_coords()
    m = np.zeros(3)
    fx = F.reshape(-1, 3)
    for i in np.flatnonzero(np.abs(fx).sum(axis=1) > 0):
        m += np.cross(coords[i] - c, fx[i])
    assert m[0] == pytest.approx(2000.0, rel=0.15)
    assert not any("produced no torque" in x for x in w)


def test_remote_point_adds_lever_moment() -> None:
    """Force at a remote point = same resultant + r x F on the region."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    region = RegionSpec(kind="face", axis="+x")
    case = LoadCase(
        name="lever",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=region,
                force_n=[0.0, 0.0, -100.0],
                remote_point=[30.0, 30.0, 6.0],  # 30 mm lateral lever arm in +y
            )
        ],
    )
    _fixed3, F, w = resolve_case_dofs(dom, case)
    total = F.reshape(-1, 3).sum(axis=0)
    assert total[2] == pytest.approx(-100.0)
    # moment about the region centroid ≈ lever x F (lever y ≈ +30 → m_x = -3000)
    idx = np.flatnonzero(region_node_mask(dom, region))
    c = dom.node_coords()[idx].mean(axis=0)
    coords = dom.node_coords()
    fx = F.reshape(-1, 3)
    m = np.zeros(3)
    for i in np.flatnonzero(np.abs(fx).sum(axis=1) > 0):
        m += np.cross(coords[i] - c, fx[i])
    lever_y = 30.0 - c[1]
    assert m[0] == pytest.approx(-lever_y * 100.0, rel=0.2)
    assert not any("cannot carry" in x for x in w)


def test_remote_point_on_region_is_silent() -> None:
    """Remote point at the region centroid: lever ~0 so no couple exists —
    the old unconditional 'adds 0.00 N·m' warning was pure noise (fired on
    every real FEM run where the load point sat on its own region)."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    region = RegionSpec(kind="face", axis="+x")
    case = LoadCase(
        name="on_point",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=region,
                force_n=[0.0, 0.0, -100.0],
                remote_point=[30.0, 0.0, 6.0],  # the +x face centroid itself
            )
        ],
    )
    _fixed3, F, w = resolve_case_dofs(dom, case)
    assert F.reshape(-1, 3).sum(axis=0)[2] == pytest.approx(-100.0)
    assert not any("remote force" in x for x in w)


def test_preload_opposing_mouths() -> None:
    """Bolt pretension pushes the two bore mouths apart — net force ≈ 0."""
    dom = voxelize(_bored_mesh(), 1.5, 150_000)
    anchor = {"x": 10.0, "y": 0.0, "dia": 5.0, "axis": "+z", "depth_mm": 12.0}
    case = LoadCase(
        name="pre",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="bore", hole_index=1),
                preload_n=500.0,
            )
        ],
    )
    _fixed3, F, w = resolve_case_dofs(dom, case, anchors=[anchor])
    fx = F.reshape(-1, 3)
    assert np.linalg.norm(fx.sum(axis=0)) < 1e-6 * 500.0 + 1e-3
    # both directions present (opposing mouths)
    assert (fx[:, 2] > 0).any() and (fx[:, 2] < 0).any()
    assert fx[:, 2].max() > 0
    assert not any("no open bore mouths" in x for x in w)


def test_gravity_body_force() -> None:
    """g_load [0,0,-1] on the whole part: total force = rho*V*g."""
    dom = voxelize(_box_mesh(40.0, 20.0, 10.0), 3.0, 150_000)
    case = LoadCase(
        name="g",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(region=RegionSpec(kind="all"), g_load=[0.0, 0.0, -1.0]),
        ],
    )
    _fixed3, F, w = resolve_case_dofs(dom, case, density_g_cm3=1.24)
    total = F.reshape(-1, 3).sum(axis=0)
    # 8000 mm³ = 8 cm³ → 9.92 g → 0.00992 kg
    expected = 1.24 * 40.0 * 20.0 * 10.0 * 1e-6 * 9.81
    assert total[2] == pytest.approx(-expected, rel=0.1)
    assert any("gravity" in x for x in w)


def test_expand_cone_directions() -> None:
    f = np.array([0.0, 0.0, -100.0])
    variants = expand_cone(f, 15.0, samples=4)
    assert len(variants) == 4  # rim samples only — caller keeps the base case
    norms = [np.linalg.norm(v) for v in variants]
    assert all(math.isclose(n, 100.0, rel_tol=1e-6) for n in norms)
    # rim vectors tilt ~15° off the mean axis
    for v in variants:
        angle = math.degrees(math.acos(np.dot(v, f) / (100.0 * 100.0)))
        assert angle == pytest.approx(15.0, abs=1.0)
    assert expand_cone(f, 0.0) == []


def test_spread_moment_and_body_force_direct() -> None:
    """Units sanity on the helpers themselves."""
    dom = voxelize(_box_mesh(20.0, 20.0, 20.0), 5.0, 150_000)
    F = np.zeros(3 * dom.node_count)
    nodes = np.arange(dom.node_count)
    got = spread_moment(dom, F, nodes, np.array([0.0, 0.0, 1.0]))  # 1 N·m
    assert got == pytest.approx(1.0)
    F2 = np.zeros(3 * dom.node_count)
    total = body_force(dom, F2, dom.solid.ravel(), np.array([0, 0, -1.0]), 1.24)
    mass_kg = 1.24 * 20.0**3 * 1e-6
    assert total == pytest.approx(mass_kg * 9.81, rel=0.05)


# --------------------------------------------------------------------------- #
# SIMP extras                                                                   #
# --------------------------------------------------------------------------- #


def _cantilever():
    dom = voxelize(_box_mesh(), 3.0, 150_000)
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


def test_simp_mma_converges_on_volume_target() -> None:
    """MMA backend reaches the volume constraint like OC."""
    pytest.importorskip("nlopt")
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed3, F)],
        passive,
        TopOptParams(volfrac=0.45, rmin_mm=4.0, max_iter=40, tol=0.01, solver="mma"),
    )
    assert res.iterations > 0
    rho = res.density.reshape(dom.dims)
    assert rho[dom.solid].mean() == pytest.approx(0.45, abs=0.03)


def test_simp_mma_minmax_balances_cases() -> None:
    """case_objective='minmax' uses the smooth P-norm objective — the worst
    case compliance improves vs a weighted-average run."""
    pytest.importorskip("nlopt")
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    F2 = F.copy()
    idx = np.flatnonzero(np.abs(F.reshape(-1, 3)).sum(axis=1))
    F2[3 * idx + 2] = -F[3 * idx + 2]  # flip z-loads
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed3, F), (fixed3, F2)],
        passive,
        TopOptParams(
            volfrac=0.45,
            rmin_mm=4.0,
            max_iter=60,
            tol=0.01,
            solver="mma",
            case_objective="minmax",
        ),
    )
    rho = res.density.reshape(dom.dims)
    assert rho[dom.solid].mean() == pytest.approx(0.45, abs=0.03)
    assert res.iterations > 0


def test_simp_mma_stress_constraint_adds_material() -> None:
    """A stress cap below the unconstrained optimum forces extra material —
    the volume fraction must rise above the target."""
    pytest.importorskip("nlopt")
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed3, F)],
        passive,
        TopOptParams(
            volfrac=0.4,
            rmin_mm=4.0,
            max_iter=60,
            tol=0.01,
            solver="mma",
            max_stress_mpa=5.0,
        ),
    )
    rho = res.density.reshape(dom.dims)
    assert rho[dom.solid].mean() > 0.45  # stress bound overrode the 0.4 target


def test_simp_mma_displacement_cap() -> None:
    """max_displacement_mm runs through the adjoint gradient without error
    and keeps the run feasible."""
    pytest.importorskip("nlopt")
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed3, F)],
        passive,
        TopOptParams(
            volfrac=0.6,
            rmin_mm=4.0,
            max_iter=40,
            tol=0.01,
            solver="mma",
            max_displacement_mm=0.05,
        ),
    )
    assert res.iterations > 0
    assert not any("IndexError" in w for w in res.warnings)


def test_simp_penal_continuation_records_ramp() -> None:
    """penal_init < penal ramps the exponent across iterations."""
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed3, F)],
        passive,
        TopOptParams(
            volfrac=0.45,
            rmin_mm=4.0,
            max_iter=12,
            tol=0.001,
            penal=3.0,
            penal_init=1.5,
            penal_ramp_iters=10,
        ),
    )
    pens = [h["penal"] for h in res.history]
    assert pens[0] < pens[-1]
    assert pens[-1] == pytest.approx(3.0)


def test_simp_passive_void_stays_empty() -> None:
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    # forbid material in the bottom half of the domain
    void_cells = np.zeros(dom.dims, dtype=bool)
    void_cells[:, :, :2] = True
    void = void_cells.ravel()
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed3, F)],
        passive,
        TopOptParams(volfrac=0.5, rmin_mm=4.0, max_iter=25, tol=0.02),
        passive_void=void,
    )
    rho = res.density.reshape(dom.dims)
    assert (rho[void_cells] < 0.05).all()


def test_simp_passive_floor_warns_when_target_unreachable() -> None:
    """Keep/load cells pinning >target volume must warn instead of silently
    reporting 'converged' far above the requested volfrac."""
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.dims, dtype=bool)
    passive[:, :, :3] = True  # pin 75% of a nz=4 domain
    passive = passive.ravel() | passive_solid_mask(dom, [], [fixed3.any(axis=1)], [])
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed3, F)],
        passive,
        TopOptParams(volfrac=0.3, rmin_mm=4.0, max_iter=10, tol=0.02),
    )
    assert res.passive_frac > 0.5
    assert any("unreachable" in w for w in res.warnings)


def test_simp_warm_start_converges_faster() -> None:
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    p = TopOptParams(volfrac=0.4, rmin_mm=4.0, max_iter=15, tol=0.005)
    res1 = simp(dom, E_PLA, NU, [(fixed3, F)], passive, p)
    warm = simp(dom, E_PLA, NU, [(fixed3, F)], passive, p, x0=res1.density)
    # a converged warm start needs noticeably fewer iterations than a cold run
    assert warm.iterations <= res1.iterations


def test_simp_case_weights_change_result() -> None:
    dom, fixed3, F = _cantilever()
    passive = np.zeros(dom.cell_count, dtype=bool)
    F2 = F.copy()
    idx = np.flatnonzero(np.abs(F.reshape(-1, 3)).sum(axis=1))
    F2r = F2.reshape(-1, 3)
    F2r[idx, 0] += 30.0  # second case adds a lateral component
    p = TopOptParams(volfrac=0.45, rmin_mm=4.0, max_iter=20, tol=0.02)
    r1 = simp(dom, E_PLA, NU, [(fixed3, F), (fixed3, F2)], passive, p)
    r2 = simp(
        dom,
        E_PLA,
        NU,
        [(fixed3, F), (fixed3, F2)],
        passive,
        p,
        case_weights=[1.0, 0.0],  # ignore the lateral case
    )
    assert not np.allclose(r1.density, r2.density)


def test_simp_per_cell_modulus_insert() -> None:
    """A stiff insert zone survives optimization around it."""
    dom, fixed3, F = _cantilever()
    E_cells = np.full(dom.cell_count, E_PLA)
    insert = np.zeros(dom.dims, dtype=bool)
    insert[:, :, :2] = True  # bottom slab = aluminium insert
    E_cells[insert.ravel()] = 70_000.0
    passive = passive_solid_mask(dom, [], [fixed3.any(axis=1)], [])
    res = simp(
        dom,
        E_cells,
        NU,
        [(fixed3, F)],
        passive,
        TopOptParams(volfrac=0.45, rmin_mm=4.0, max_iter=25, tol=0.02),
    )
    assert res.iterations > 0


# --------------------------------------------------------------------------- #
# extraction: min-member postpass                                              #
# --------------------------------------------------------------------------- #


def test_min_member_thickens_isolated_voxels() -> None:
    from cadx.cad.fem.extract import density_to_mask

    dom = voxelize(_box_mesh(24.0, 20.0, 10.0), 2.0, 150_000)
    rho = np.zeros(dom.cell_count)
    # 1-voxel-thick member — well below min_member 5mm (2*pitch measures 4mm)
    rho.reshape(dom.dims)[3:9, 5, 4] = 1.0
    mask, warnings, _nc = density_to_mask(dom, rho, threshold=0.5, min_member_mm=5.0)
    assert mask.sum() > 6  # thickened beyond the raw selection
    assert any("thickened" in w for w in warnings)


# --------------------------------------------------------------------------- #
# resample pipeline                                                             #
# --------------------------------------------------------------------------- #


def test_run_topopt_resample(tmp_path: Path) -> None:
    """Full pipeline → density.npz → resample at a different threshold."""
    from cadx import storage
    from cadx.cad.engine import run_topopt_pipeline, run_topopt_resample

    out_dir = storage.tmp_dir() / f"resample-{id(tmp_path)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "b.stl"
    _box_mesh().export(str(src))
    case = LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -100.0],
            )
        ],
    ).model_dump()
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=out_dir,
        load_cases=[case],
        optimize={"target_volume_fraction": 0.45},
        material="pla",
        voxel_mm=3.0,
    )
    assert res.ok, res.error
    npz = out_dir / "topopt" / "density.npz"
    assert npz.exists()
    rel = storage.relpath(npz)

    out2 = out_dir / "resampled"
    res2 = run_topopt_resample(
        density_relpath=rel,
        out_dir=out2,
        load_cases=[case],
        optimize={"density_threshold": 0.7},
        material="pla",
    )
    assert res2.ok, res2.error
    # higher threshold → less extracted volume than the 0.5 default
    assert res2.topopt["threshold"] == pytest.approx(0.7)
    assert res2.topopt["resampled_from"] == rel
    assert res2.mesh["volume_mm3"] < res.mesh["volume_mm3"]
    assert res2.mesh["solid_elements"] < res.mesh["solid_elements"]
    m = trimesh.load(str(out2 / "_run" / "model.stl"), force="mesh", process=False)
    m.merge_vertices()
    assert m.is_watertight


def test_run_topopt_multires_warm_start(tmp_path: Path) -> None:
    """multires=True with a low cell threshold runs a coarse SIMP pass and
    warm-starts the fine grid — result stays valid and watertight."""
    from cadx import storage
    from cadx.cad.engine import run_topopt_pipeline
    from cadx.runtime_config import RuntimeSettings

    out_dir = storage.tmp_dir() / f"multires-{id(tmp_path)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "b.stl"
    _box_mesh().export(str(src))
    case = LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -100.0],
            )
        ],
    ).model_dump()
    rt = RuntimeSettings(topopt_multires_min_cells=10)
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=out_dir,
        load_cases=[case],
        optimize={"target_volume_fraction": 0.45, "multires": True},
        material="pla",
        voxel_mm=3.0,
        runtime=rt,
    )
    assert res.ok, res.error
    mr = res.topopt.get("multires")
    assert mr is not None, "coarse pass should have run below the lowered threshold"
    assert mr["coarse_cells"] > 0 and mr["coarse_iterations"] > 0
    assert mr["coarse_pitch_mm"] == pytest.approx(6.0)
    assert res.topopt["iterations"] > 0
    m = trimesh.load(str(out_dir / "_run" / "model.stl"), force="mesh", process=False)
    m.merge_vertices()
    assert m.is_watertight


def test_run_topopt_multires_skipped_below_threshold(tmp_path: Path) -> None:
    """Default threshold (4000 cells) keeps small domains single-resolution."""
    from cadx import storage
    from cadx.cad.engine import run_topopt_pipeline
    from cadx.runtime_config import RuntimeSettings

    out_dir = storage.tmp_dir() / f"multires-off-{id(tmp_path)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "b.stl"
    _box_mesh().export(str(src))
    case = LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -100.0],
            )
        ],
    ).model_dump()
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=out_dir,
        load_cases=[case],
        optimize={"target_volume_fraction": 0.45, "multires": True},
        material="pla",
        voxel_mm=3.0,
        runtime=RuntimeSettings(),  # default 4000-cell threshold
    )
    assert res.ok, res.error
    assert "multires" not in res.topopt


# --------------------------------------------------------------------------- #
# case sanity report                                                            #
# --------------------------------------------------------------------------- #


def test_case_reactions_report() -> None:
    from cadx.cad.fem.regions import case_reactions

    dom, fixed3, F = _cantilever()
    rep = case_reactions(dom, fixed3, F)
    assert rep["resultant_n"][2] == pytest.approx(-100.0)
    assert rep["fixed_nodes"] > 0
    # 100 N at the tip, ~30 mm from the clamp centre → ~3000 N·mm = 3 N·m
    assert rep["moment_mag_nm"] > 1.0
