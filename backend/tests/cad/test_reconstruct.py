"""Reconstruction core tests — synthetic meshes, real sandbox exec (no LLM).

The emitted code runs through ``cadx.cad.sandbox.run_user_code`` — the same
subprocess path the executor uses — and the exported B-Rep is checked for a
watertight solid with a bbox matching the input mesh.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from build123d import import_brep

from cadx.cad.reconstruct import (
    chamfer_mm,
    classify_part,
    reconstruct_mesh,
    snap_grid,
)
from cadx.cad.sandbox import run_user_code
from cadx.llm.schemas import Dim, PartSpec


def _spec(summary: str = "", L: float = 20.0, W: float = 20.0, H: float = 40.0) -> PartSpec:
    return PartSpec(
        name="recon test part",
        summary=summary,
        length_x=Dim(value=L, source="stated"),
        width_y=Dim(value=W, source="stated"),
        height_z=Dim(value=H, source="stated"),
    )


def _waisted_revolve() -> trimesh.Trimesh:
    """Lathe part around +Z: r(z) = 10 → 12 → 7 (waist) → 12 → 9, z ∈ [0,40].

    The profile touches the axis at both ends so the mesh is watertight
    (open end discs would poison the chamfer comparison).
    """
    profile = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [12.0, 8.0],
            [7.0, 20.0],
            [12.0, 32.0],
            [9.0, 40.0],
            [0.0, 40.0],
        ]
    )
    m = trimesh.creation.revolve(profile, sections=128)
    m.merge_vertices()
    return m


def _rounded_box() -> trimesh.Trimesh:
    """60x40x8 plate standing on z=0 with 4 mm rounded vertical edges."""
    from shapely.geometry import box as sh_box

    poly = sh_box(-26.0, -16.0, 26.0, 16.0).buffer(4.0, quad_segs=8)
    return trimesh.creation.extrude_polygon(poly, 8.0)


def _brep_to_mesh(brep_path) -> trimesh.Trimesh:
    shape = import_brep(brep_path)
    assert len(shape.solids()) == 1  # watertight single B-Rep solid
    assert shape.is_valid
    verts, tris = shape.tessellate(0.2)
    return trimesh.Trimesh(
        vertices=np.asarray([(v.X, v.Y, v.Z) for v in verts]),
        faces=np.asarray(tris),
        process=True,
    )


def _run_sandbox(code: str, work) -> trimesh.Trimesh:
    out = run_user_code(code, work)
    assert out.ok, out.error
    assert out.brep_path is not None
    return _brep_to_mesh(out.brep_path)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def test_snap_grid_snaps_and_passes():
    assert snap_grid(9.98) == 10.0  # 0.2 % off → snaps
    assert snap_grid(2.49) == 2.5
    assert snap_grid(0.0) == 0.0
    # 1.26 → nearest grid point 1.5 is ~19 % off → keep measured value
    assert snap_grid(1.26) == pytest.approx(1.26)
    assert snap_grid(3.31, grid=0.25) == pytest.approx(3.25)


def test_chamfer_mm_identical_and_offset():
    m = trimesh.creation.cylinder(radius=10, height=30, sections=64)
    # point-set Chamfer: identical surfaces still measure ~half the mean
    # sample spacing (~0.2 mm at n=20000 on this part), not 0
    assert chamfer_mm(m, m) < 0.5
    bigger = trimesh.creation.cylinder(radius=11, height=30, sections=64)
    d = chamfer_mm(m, bigger)
    assert 0.5 < d < 1.8  # ~1 mm radial gap everywhere


# --------------------------------------------------------------------------- #
# classification                                                              #
# --------------------------------------------------------------------------- #


def test_classify_revolve():
    mesh = _waisted_revolve()
    spec = _spec("turned round knob — cylindrical grip")
    assert classify_part(mesh, spec, {}) == "revolve"


def test_classify_prism():
    mesh = _rounded_box()
    spec = _spec("flat mounting plate", L=60.0, W=40.0, H=8.0)
    assert classify_part(mesh, spec, {}) == "prism"


def test_classify_organic():
    blob = trimesh.creation.icosphere(subdivisions=3, radius=10)
    blob.apply_scale([1.4, 0.7, 0.5])  # ellipsoid — slices are ellipses
    spec = _spec("organic sculpture", L=28.0, W=14.0, H=10.0)
    assert classify_part(blob, spec, {}) == "organic"
    assert reconstruct_mesh(blob, spec, {}) is None


# --------------------------------------------------------------------------- #
# revolve                                                                     #
# --------------------------------------------------------------------------- #


def test_reconstruct_revolve_end_to_end(tmp_path):
    mesh = _waisted_revolve()
    spec = _spec("turned round knob — cylindrical grip", L=20.0, W=20.0, H=40.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None
    assert rec.kind == "revolve"
    # stated height is authoritative; waist radius snapped to the 0.5 grid
    assert rec.params["length"] == pytest.approx(40.0)
    assert rec.params["r_waist"] == pytest.approx(7.0, abs=0.6)
    assert rec.params["r_start"] == pytest.approx(10.0, abs=0.6)
    assert all(p["name"] in rec.params for p in rec.params_schema)
    assert rec.confidence > 0.5

    recon = _run_sandbox(rec.code, tmp_path / "revolve")
    assert chamfer_mm(mesh, recon) < 0.5
    for got, want in zip(recon.extents, mesh.extents, strict=True):
        assert got == pytest.approx(want, abs=1.5)


def test_reconstruct_revolve_axis_x(tmp_path):
    """Same lathe part lying down along X — axis detection + placement."""
    mesh = _waisted_revolve()
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 1.0, 0.0])
    )
    # rest on XY plane: rotate z-axis profile to +X, then lift by max radius
    mesh.apply_translation([0.0, 0.0, 12.0])
    spec = _spec("round handle, revolved", L=40.0, W=24.0, H=24.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "revolve"
    recon = _run_sandbox(rec.code, tmp_path / "revolve_x")
    for got, want in zip(recon.extents, mesh.extents, strict=True):
        assert got == pytest.approx(want, abs=1.5)
    assert chamfer_mm(mesh, recon) < 0.6


def test_revolve_with_arc_segment(tmp_path):
    """Spherical profile → fitted circle → RadiusArc in emitted code."""
    r_sphere = 15.0
    ts = np.linspace(-r_sphere, r_sphere, 13)
    rs = np.sqrt(np.maximum(r_sphere**2 - ts**2, 0.0))
    rs[0] = rs[-1] = 0.0  # closed tips — watertight mesh
    profile = np.column_stack([rs, ts + r_sphere])
    mesh = trimesh.creation.revolve(profile, sections=128)
    mesh.merge_vertices()
    spec = _spec("round ball knob", L=30.0, W=30.0, H=30.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "revolve"
    assert "RadiusArc" in rec.code
    recon = _run_sandbox(rec.code, tmp_path / "sphere")
    assert chamfer_mm(mesh, recon) < 0.5


# --------------------------------------------------------------------------- #
# prism                                                                       #
# --------------------------------------------------------------------------- #


def test_reconstruct_prism_end_to_end(tmp_path):
    mesh = _rounded_box()
    spec = _spec("flat mounting plate", L=60.0, W=40.0, H=8.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None
    assert rec.kind == "prism"
    assert rec.params["thickness"] == pytest.approx(8.0)  # stated dim wins
    assert rec.confidence > 0.5

    recon = _run_sandbox(rec.code, tmp_path / "prism")
    for got, want in zip(recon.extents, mesh.extents, strict=True):
        assert got == pytest.approx(want, abs=1.0)
    assert chamfer_mm(mesh, recon) < 0.5


def test_reconstruct_prism_axis_x(tmp_path):
    """Extrusion along X (rail) — outline emitted on a YZ workplane."""
    from shapely.geometry import box as sh_box

    poly = sh_box(-15.0, -5.0, 15.0, 5.0).buffer(3.0, quad_segs=6)
    mesh = trimesh.creation.extrude_polygon(poly, 30.0)
    mesh.apply_transform(
        np.array(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
    )
    spec = _spec("long extruded rail", L=30.0, W=36.0, H=16.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "prism"
    recon = _run_sandbox(rec.code, tmp_path / "prism_x")
    for got, want in zip(recon.extents, mesh.extents, strict=True):
        assert got == pytest.approx(want, abs=1.0)
