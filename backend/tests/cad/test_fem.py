"""Voxel FEM + SIMP topology optimization: solver correctness, region mapping,
mesh extraction validity and the executor pipelines."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cadagent.cad.fem.analyze import FemReport, run_fem, run_fem_domain
from cadagent.cad.fem.extract import _regularize_mask, _surface_mesh, density_to_mesh
from cadagent.cad.fem.regions import (
    region_cell_mask,
    region_node_mask,
    resolve_case,
)
from cadagent.cad.fem.solver import solve
from cadagent.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
from cadagent.cad.fem.voxelize import VoxelDomain, voxelize
from cadagent.llm.schemas import LoadCase, LoadSpec, RegionSpec

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


def test_resolve_case_empty_region_warns() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    case = LoadCase(
        name="bad",
        fixed=[RegionSpec(kind="cylinder", axis="+z", center=[999, 999], diameter_mm=2)],
        loads=[LoadSpec(region=RegionSpec(kind="face", axis="+x"), force_n=[0, 0, -1])],
    )
    fixed, _f, w = resolve_case(dom, case)
    assert not fixed.any()
    assert any("matched no nodes" in x for x in w)


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
    mesh, _mask, _w = density_to_mesh(dom, dom.solid.ravel().astype(float), smooth_iterations=0)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert math.isclose(mesh.volume, 60 * 24 * 12, rel_tol=0.02)


def test_extract_topopt_watertight(simp_setup) -> None:
    dom, case, fixed, res = simp_setup
    passive = passive_solid_mask(dom, [], [fixed], [region_node_mask(dom, case.loads[0].region)])
    mesh, _mask, _w = density_to_mesh(dom, res.density, passive_solid=passive)
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
    from cadagent.cad.engine import run_fem_pipeline

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
    from cadagent.cad.engine import run_topopt_pipeline

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
    stl = tmp_path / "o" / "_run" / "model.stl"
    assert stl.exists()
    m = trimesh.load(str(stl), force="mesh", process=False)
    m.merge_vertices()
    assert m.is_watertight
    assert (tmp_path / "o" / "topopt" / "density.npz").exists()
    assert (tmp_path / "o" / "fem" / "report.json").exists()


def test_run_fem_domain_matches_mask(simp_setup) -> None:
    """Verification on the exact voxel mask (what extraction exports)."""
    dom, case, fixed, res = simp_setup
    passive = passive_solid_mask(dom, [], [fixed], [region_node_mask(dom, case.loads[0].region)])
    _, mask, _ = density_to_mesh(dom, res.density, passive_solid=passive)
    rep = run_fem_domain(dom, mask, [case], "pla")
    c = rep.cases[0]
    assert c.safety_factor is not None
    assert 0 < c.max_disp_mm < 100


def test_run_fem_no_cases(tmp_path: Path) -> None:
    rep = run_fem(_box_mesh(), [], "pla", voxel_mm=4.0)
    assert "no load cases provided" in rep.warnings
