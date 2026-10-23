"""Regression tests for FEM rendering + 3D anchors.

- `_grid_image`: cell arrays are (nx,ny,nz) C-order — pyvista ImageData
  flattens cells x-fastest (Fortran order); a wrong ravel scrambles
  sections into confetti blocks.
- `scan_mesh_anchors`: the tunnel depth AND axial centre come from the
  pierced wall's rim-ray hit pairs — a bore through the plate of a
  stepped part reports depth=plate thickness and z=plate mid, not the
  bbox mid-plane.
- `render_regions`: bearing loads draw force arrows and the resolve
  summary reports a nonzero resultant.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.anchors3d import merge_anchor_lists, scan_mesh_anchors
from cadx.cad.fem.voxelize import VoxelDomain, voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec


def _bored_mesh() -> trimesh.Trimesh:
    """60x24x12 plate with a Ø5 through-bore along +z at (10, 0)."""
    box = trimesh.creation.box(extents=[60.0, 24.0, 12.0])
    box.apply_translation([0.0, 0.0, 6.0])
    bore = trimesh.creation.cylinder(radius=2.5, height=20.0, sections=32)
    bore.apply_translation([10.0, 0.0, 6.0])
    out = box.difference(bore)
    assert isinstance(out, trimesh.Trimesh)
    return out


def _stepped_plate_mesh() -> trimesh.Trimesh:
    """100x80x10 plate (z=0..10) + 30x30x30 flange on top (z=10..40), with
    a Ø8 through-bore in the plate at (25, 0). The bbox z-mid is 20, but
    the pierced wall is the plate: depth 10 mm, centre z=5."""
    plate = trimesh.creation.box(extents=[100.0, 80.0, 10.0])
    plate.apply_translation([0.0, 0.0, 5.0])
    flange = trimesh.creation.box(extents=[30.0, 30.0, 30.0])
    flange.apply_translation([-25.0, 0.0, 25.0])
    bore = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=32)
    bore.apply_translation([25.0, 0.0, 5.0])
    out = trimesh.util.concatenate([plate, flange]).difference(bore, engine="manifold")
    assert isinstance(out, trimesh.Trimesh)
    return out


# --------------------------------------------------------------------------- #
# _grid_image — Fortran-order cell data                                        #
# --------------------------------------------------------------------------- #


def test_grid_image_solid_respects_fortran_order() -> None:
    """Solid only at k=0 must threshold to a one-pitch slab in Z — with
    C-order flattening pyvista misreads the mask and the section view
    becomes a confetti block spanning the full height."""
    pytest.importorskip("pyvista")
    from cadx.cad.fem.analyze import _grid_image

    dims = (10, 8, 6)
    solid = np.zeros(dims, dtype=bool)
    # L-shaped footprint in the k=0 plane only
    solid[:, :3, 0] = True
    solid[:3, :, 0] = True
    dom = VoxelDomain(
        origin=np.array([1.0, -2.0, 4.0]),
        pitch=1.5,
        dims=dims,
        solid=solid,
    )
    grid = _grid_image(
        dom,
        np.zeros(dom.cell_count),
        np.zeros((dom.node_count, 3)),
    )
    body = grid.threshold(0.5, scalars="solid")
    z0, z1 = float(body.bounds[4]), float(body.bounds[5])
    assert z0 == pytest.approx(4.0, abs=1e-6)
    # one pitch tall — NOT the full 9 mm domain height
    assert z1 == pytest.approx(4.0 + 1.5, abs=0.2)
    assert z1 - z0 < 3.0


# --------------------------------------------------------------------------- #
# scan_mesh_anchors — tunnel depth + axial centre                              #
# --------------------------------------------------------------------------- #


def test_scan_mesh_anchors_axial_center_on_stepped_part() -> None:
    """The bore pierces the 10 mm plate — depth_mm ~10 and the axial
    centre z~5, not None and not the bbox mid (~20)."""
    anchors = scan_mesh_anchors(_stepped_plate_mesh(), step_mm=1.0)
    z_holes = [a for a in anchors if a["axis"] == "+z" and a["kind"] == "hole"]
    assert z_holes, f"no +z hole found in {anchors}"
    a = min(z_holes, key=lambda x: math.hypot(float(x["x"]) - 25.0, float(x["y"])))
    assert a["dia"] == pytest.approx(8.0, rel=0.5)
    assert a["depth_mm"] == pytest.approx(10.0, abs=2.5)
    assert a["z"] is not None
    assert a["z"] == pytest.approx(5.0, abs=2.0)
    assert a["pos"] is not None
    assert a["pos"][2] == pytest.approx(5.0, abs=2.0)


def test_merge_anchor_lists_fills_axial_center() -> None:
    """A spec/image anchor without axial position inherits the scanned
    tunnel centre (pos[ax] and z) when it matches a scan anchor."""
    scanned = scan_mesh_anchors(_stepped_plate_mesh(), step_mm=1.0)
    sa = next(a for a in scanned if a["axis"] == "+z" and a["kind"] == "hole")
    existing = [
        {
            "x": sa["x"],
            "y": sa["y"],
            "dia": 8.0,
            "axis": "+z",
            "z": None,
            "pos": [sa["x"], sa["y"], None],
            "kind": "hole",
            "source": "image",
        }
    ]
    merged = merge_anchor_lists(existing, scanned)
    m = next(a for a in merged if a.get("source") == "image+scan")
    assert m["z"] == pytest.approx(5.0, abs=2.0)
    assert m["pos"] is not None
    assert m["pos"][2] == pytest.approx(5.0, abs=2.0)


# --------------------------------------------------------------------------- #
# render_regions — bearing load smoke                                          #
# --------------------------------------------------------------------------- #


def test_render_regions_bearing_load(tmp_path: Path) -> None:
    """A bearing_n load on a `bore` region writes the preview PNG (when
    offscreen rendering works) and the resolved resultant is > 0."""
    pytest.importorskip("pyvista")
    from cadx.cad.fem.preview import render_regions, resolve_case_summary

    mesh = _bored_mesh()
    dom = voxelize(mesh, 2.0, 150_000)
    anchors = scan_mesh_anchors(mesh, step_mm=1.0)
    idx = next(
        i + 1 for i, a in enumerate(anchors) if a["axis"] == "+z" and a["kind"] == "hole"
    )
    case = LoadCase(
        name="bearing",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="bore", hole_index=idx),
                bearing_n=120.0,
                bearing_dir=[0.0, -1.0, 0.0],
            )
        ],
    )
    keep = [RegionSpec(kind="hole", hole_index=idx, description="keep bore")]
    out = tmp_path / "regions"
    names = render_regions(dom, [case], keep, out, anchors=anchors, source_mesh=mesh)
    # rendering may be unavailable headless — then nothing was written;
    # every reported file must exist and be non-empty
    assert names in ([], ["regions_bearing.png"])
    for name in names:
        png = out / name
        assert png.exists() and png.stat().st_size > 0
    summary = resolve_case_summary(dom, case, anchors)
    assert summary["total_force_n"] > 0


def test_render_regions_moment_ring(tmp_path: Path) -> None:
    """A pure `moment_nm` load draws the ring marker (not an arrow) and the
    summary exposes the applied moment split — moments must be VISIBLE."""
    pytest.importorskip("pyvista")
    from cadx.cad.fem.preview import _load_arrows, render_regions, resolve_case_summary

    mesh = _bored_mesh()
    dom = voxelize(mesh, 2.0, 150_000)
    anchors = scan_mesh_anchors(mesh, step_mm=1.0)
    idx = next(
        i + 1 for i, a in enumerate(anchors) if a["axis"] == "+z" and a["kind"] == "hole"
    )
    case = LoadCase(
        name="shaft_torque",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="bore", hole_index=idx),
                moment_nm=[0.0, 0.0, 1.2],
            )
        ],
    )
    centers3d = dom.element_centers().reshape((*dom.dims, 3))
    arrows, f_strs, force_n = _load_arrows(dom, case, anchors, centers3d, 20.0)
    assert arrows and len(arrows[0]) == 3  # (geometry, marker, color)
    assert any("M=" in s for s in f_strs)
    # the ring uses the moment colour, distinct from force arrows
    assert arrows[0][2] == "mediumvioletred"
    # a pure couple has ~zero net force — fallback reports the resolved
    # resultant, which stays ~0
    assert force_n < 1e-3

    out = tmp_path / "regions_m"
    names = render_regions(dom, [case], [], out, anchors=anchors, source_mesh=mesh)
    for name in names:
        png = out / name
        assert png.exists() and png.stat().st_size > 0

    summary = resolve_case_summary(dom, case, anchors)
    rec = summary["applied"][0]
    assert "moment_nm" in rec["mechanisms"]
    assert summary["reactions"]["moment_mag_direct_nm"] == pytest.approx(1.2, rel=0.25)
