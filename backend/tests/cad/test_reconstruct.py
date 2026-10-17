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
    _bore_axis_consistent,
    chamfer_mm,
    classify_part,
    reconstruct_mesh,
    snap_grid,
)
from cadx.cad.sandbox import run_user_code
from cadx.llm.schemas import Dim, Feature, PartSpec

try:
    import pyransac3d  # noqa: F401

    HAS_PYRANSAC = True
except ImportError:  # optional machinery — composite tests skip, the rest run
    HAS_PYRANSAC = False


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


def _annular_revolve() -> trimesh.Trimesh:
    """TUBE around +Z: outer r=11, bore r=5, length 40 — watertight
    trimesh annulus centred on the origin axis."""
    m = trimesh.creation.annulus(r_min=5.0, r_max=11.0, height=40.0, sections=128)
    assert m.is_watertight
    return m


def test_reconstruct_revolve_axis_x(tmp_path):
    """Same lathe part lying down along X — axis detection + placement."""
    mesh = _waisted_revolve()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 1.0, 0.0]))
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


def test_bore_axis_consistent():
    """A consistent interior ring marks the through-bore axis; a solid or
    unbored part has none."""
    assert _bore_axis_consistent(_annular_revolve(), "z")
    assert not _bore_axis_consistent(_annular_revolve(), "x")
    assert not _bore_axis_consistent(_waisted_revolve(), "z")
    # a square plate with a z-bore has a consistent ring but must NOT be
    # treated as a revolve — the circularity gate still protects it
    plate = trimesh.creation.box([40.0, 40.0, 8.0])
    bore = trimesh.creation.cylinder(radius=4.0, height=10.0, sections=48)
    bore.apply_translation([0.0, 0.0, -1.0])
    m = plate.difference(bore)
    assert _bore_axis_consistent(m, "z")
    spec = _spec("flat mounting plate", L=40.0, W=40.0, H=8.0)
    assert reconstruct_mesh(m, spec, {}).kind != "revolve"


def test_reconstruct_revolve_with_bore(tmp_path):
    """Annular profile → the emitted section must close through the bore
    radius, not down to the axis (a solid revolve misses the tunnel and
    fails chamfer — the grip measured 3.02mm against a 0.65mm limit)."""
    mesh = _annular_revolve()
    spec = _spec("cylindrical sleeve with through bore", L=20.0, W=20.0, H=40.0)
    spec.features = [
        Feature(
            name="through_bore",
            type="hole",
            description="Ø10 through bore along z",
            dims={"diameter": 10.0, "depth": 40.0},
            position="centered",
            axis="z",
            count=1,
            source="stated",
        )
    ]
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "revolve"
    # stated Ø10 wins over the measured ring radius
    assert rec.params.get("r_bore") == pytest.approx(5.0, abs=0.3)
    assert "r_bore" in rec.code

    recon = _run_sandbox(rec.code, tmp_path / "annulus")
    assert recon.is_watertight
    assert chamfer_mm(mesh, recon) < 0.4
    for got, want in zip(recon.extents, mesh.extents, strict=True):
        assert got == pytest.approx(want, abs=1.5)


def test_reconstruct_revolve_bore_measured_without_spec(tmp_path):
    """No stated feature — the measured inner ring still produces r_bore."""
    mesh = _annular_revolve()
    spec = _spec("cylindrical tube, turned", L=20.0, W=20.0, H=40.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "revolve"
    assert rec.params.get("r_bore") == pytest.approx(5.0, abs=0.6)
    recon = _run_sandbox(rec.code, tmp_path / "annulus_ns")
    assert recon.is_watertight
    assert chamfer_mm(mesh, recon) < 0.4


def test_reconstruct_solid_revolve_has_no_bore(tmp_path):
    """Regression: a solid lathe part must NOT gain an r_bore param."""
    mesh = _waisted_revolve()
    spec = _spec("turned round knob — cylindrical grip", L=20.0, W=20.0, H=40.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "revolve"
    assert "r_bore" not in rec.params
    recon = _run_sandbox(rec.code, tmp_path / "solid")
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


# --------------------------------------------------------------------------- #
# composite (primitive segmentation — optional pyransac3d dep)                  #
# --------------------------------------------------------------------------- #


def _box_boss() -> trimesh.Trimesh:
    """40x30x6 plate on z=0 with a centred Ø12x15 cylindrical boss on top.

    Sections along every axis vary (plate rect vs plate+boss vs boss disc)
    so the prism and revolve gates fail — this exercises the composite path.
    """
    base = trimesh.creation.box([40.0, 30.0, 6.0])
    base.apply_translation([0.0, 0.0, 3.0])
    boss = trimesh.creation.cylinder(radius=6.0, height=15.0, sections=64)
    boss.apply_translation([0.0, 0.0, 6.0 + 7.5])
    m = trimesh.boolean.union([base, boss])
    assert m.is_watertight
    return m


def _box_dome() -> trimesh.Trimesh:
    """Same plate with a hemisphere dome (r=8) sitting on its top face."""
    base = trimesh.creation.box([40.0, 30.0, 6.0])
    base.apply_translation([0.0, 0.0, 3.0])
    dome = trimesh.creation.icosphere(subdivisions=4, radius=8.0)
    dome.apply_translation([0.0, 0.0, 6.0])
    m = trimesh.boolean.union([base, dome])
    assert m.is_watertight
    return m


def _box_boss_bore() -> trimesh.Trimesh:
    """Plate + centred boss with a Ø6 through-bore along z (void cylinder)."""
    base = trimesh.creation.box([40.0, 30.0, 6.0])
    base.apply_translation([0.0, 0.0, 3.0])
    boss = trimesh.creation.cylinder(radius=6.0, height=10.0, sections=64)
    boss.apply_translation([0.0, 0.0, 6.0 + 5.0])
    part = trimesh.boolean.union([base, boss])
    bore = trimesh.creation.cylinder(radius=3.0, height=30.0, sections=48)
    bore.apply_translation([0.0, 0.0, 8.0])
    m = trimesh.boolean.difference([part, bore])
    assert m.is_watertight
    return m


def _freeform_blob() -> trimesh.Trimesh:
    """Perturbed, squashed icosphere — not expressible as a few primitives."""
    rng = np.random.RandomState(7)
    m = trimesh.creation.icosphere(subdivisions=3, radius=12.0)
    d = (
        1.0
        + 0.25 * np.sin(3.0 * m.vertices[:, 0]) * np.cos(2.0 * m.vertices[:, 1])
        + rng.normal(0.0, 0.12, len(m.vertices))
    )
    m.vertices = m.vertices * d[:, None] * np.array([1.2, 0.8, 0.6])
    return m


def _on_grid(v: float, grid: float = 0.5) -> bool:
    return abs(v - round(v / grid) * grid) < 1e-9


@pytest.mark.skipif(not HAS_PYRANSAC, reason="pyransac3d not installed")
def test_classify_composite():
    """Primitive-union parts classify as composite; the arbiter reaches the
    probe only after revolve and prism fail."""
    assert classify_part(_box_boss(), _spec("plate with boss"), {}) == "composite"
    assert classify_part(_box_dome(), _spec("plate with dome"), {}) == "composite"
    assert classify_part(_box_boss_bore(), _spec("bored boss plate"), {}) == "composite"


@pytest.mark.skipif(not HAS_PYRANSAC, reason="pyransac3d not installed")
def test_reconstruct_composite_boss(tmp_path):
    mesh = _box_boss()
    spec = _spec("mounting plate with cylindrical boss", L=40.0, W=30.0, H=21.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "composite"
    assert "PARAMS" in rec.code and "result" in rec.code
    assert rec.confidence > 0.5
    # fitted dims snapped to the 0.5 mm grid
    assert all(_on_grid(v) for v in rec.params.values())
    radii = [v for k, v in rec.params.items() if k.endswith("_r")]
    assert any(r == pytest.approx(6.0, abs=0.4) for r in radii)

    recon = _run_sandbox(rec.code, tmp_path / "boss")
    assert chamfer_mm(mesh, recon) < 1.5
    for got, want in zip(recon.extents, mesh.extents, strict=True):
        assert got == pytest.approx(want, abs=1.5)


@pytest.mark.skipif(not HAS_PYRANSAC, reason="pyransac3d not installed")
def test_reconstruct_composite_dome(tmp_path):
    mesh = _box_dome()
    spec = _spec("plate with spherical dome", L=40.0, W=30.0, H=14.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "composite"
    assert "Sphere" in rec.code
    radii = [v for k, v in rec.params.items() if k.endswith("_r")]
    assert any(r == pytest.approx(8.0, abs=0.6) for r in radii)

    recon = _run_sandbox(rec.code, tmp_path / "dome")
    assert chamfer_mm(mesh, recon) < 1.5


@pytest.mark.skipif(not HAS_PYRANSAC, reason="pyransac3d not installed")
def test_reconstruct_composite_bore(tmp_path):
    """The bore is an interior cylinder (wall normals point at the axis) —
    it must be emitted as a subtracted void, not unioned material."""
    mesh = _box_boss_bore()
    spec = _spec("boss plate with through bore", L=40.0, W=30.0, H=16.0)
    rec = reconstruct_mesh(mesh, spec, {})
    assert rec is not None and rec.kind == "composite"
    assert " - _p" in rec.code  # a subtracted primitive exists
    radii = [v for k, v in rec.params.items() if k.endswith("_r")]
    assert any(r == pytest.approx(3.0, abs=0.4) for r in radii)

    recon = _run_sandbox(rec.code, tmp_path / "bore")
    assert chamfer_mm(mesh, recon) < 1.5


def test_composite_missing_dependency_degrades(monkeypatch):
    """pyransac3d absent → the composite path quietly degrades: no
    exception, classify falls back to organic, reconstruct returns None."""
    import sys

    monkeypatch.setitem(sys.modules, "pyransac3d", None)
    mesh = _box_boss()
    assert classify_part(mesh, _spec("plate with boss"), {}) == "organic"
    assert reconstruct_mesh(mesh, _spec("plate with boss"), {}) is None


@pytest.mark.skipif(not HAS_PYRANSAC, reason="pyransac3d not installed")
def test_composite_organic_blob_returns_none():
    """A noisy freeform shape cannot be decomposed into a few primitives —
    the coverage/self-check gates reject it."""
    mesh = _freeform_blob()
    rec = reconstruct_mesh(mesh, _spec("organic sculpture", L=29.0, W=20.0, H=15.0), {})
    assert rec is None
