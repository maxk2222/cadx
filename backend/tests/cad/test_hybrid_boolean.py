"""Hybrid bridge: mesh -> faceted solid -> build123d boolean via the sandbox."""

import pytest
import trimesh
from build123d import export_brep

from cadx.cad.engine import run_pipeline
from cadx.cad.mesh_ops import mesh_to_solid
from cadx.db.models import ProjectSettings

HYBRID_CODE = """\
from build123d import *

PARAMS = {"hole_d": 4.2}

base = load_mesh_solid()
hole = Pos(0, 0, -100) * Cylinder(radius=PARAMS["hole_d"] / 2, height=300)
result = base - hole
"""


@pytest.fixture
def mesh_brep(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=2, radius=20)
    m.apply_translation([0, 0, 20])
    solid = mesh_to_solid(m)
    p = tmp_path / "solid.brep"
    export_brep(solid, str(p))
    return p


def test_hybrid_boolean_cut(mesh_brep, tmp_path):
    res = run_pipeline(
        code=HYBRID_CODE,
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
        mesh_solid=mesh_brep,
        mesh_bbox=[40.0, 40.0, 40.0],
        full_exports=True,
    )
    assert res.ok, res.error
    # hole cut → volume below the solid sphere (~33 500 mm^3, hole removes ~5%)
    sphere_vol = 4 / 3 * 3.14159 * 20**3
    assert sphere_vol * 0.8 < res.analysis.volume < sphere_vol * 0.99
    assert res.exports.get("step")


def test_mesh_bbox_injected(mesh_brep, tmp_path):
    code = """\
from build123d import *

PARAMS = {}
assert isinstance(MESH_BBOX, (tuple, list)) and len(MESH_BBOX) == 3
result = load_mesh_solid()
"""
    res = run_pipeline(
        code=code,
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
        mesh_solid=mesh_brep,
        mesh_bbox=[40.0, 40.0, 40.0],
    )
    assert res.ok, res.error


def test_top_z_reports_local_surface(mesh_brep, tmp_path):
    """Sphere r=20 resting on XY: top_z over centre = 40, drops off-axis,
    falls back to MESH_BBOX[2] outside the silhouette."""
    code = """\
from build123d import *

PARAMS = {}
top_mid = top_z(0.0, 0.0)
top_off = top_z(15.0, 0.0)
top_out = top_z(50.0, 0.0)
assert abs(top_mid - 40.0) < 0.6, top_mid
assert 30.0 < top_off < 37.0, top_off
assert top_out == 40.0, top_out  # fallback: MESH_BBOX[2]
result = load_mesh_solid()
"""
    res = run_pipeline(
        code=code,
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
        mesh_solid=mesh_brep,
        mesh_bbox=[40.0, 40.0, 40.0],
    )
    assert res.ok, res.error


def test_load_mesh_solid_unavailable_outside_hybrid(tmp_path):
    code = "PARAMS = {}\nresult = load_mesh_solid()\n"
    res = run_pipeline(
        code=code,
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
    )
    assert not res.ok
    assert "hybrid" in (res.error or "").lower()


def test_hybrid_runner_collapses_sliver_result(mesh_brep, tmp_path):
    """Boolean ops on faceted mesh solids shed sliver solids (observed:
    28-fragment result, largest ~94%). The runner collapses a multi-solid
    hybrid result to the dominant solid and flags a warning for improve."""
    code = """\
from build123d import *

PARAMS = {}
body = load_mesh_solid()
sliver = Pos(500, 500, 500) * Box(0.05, 0.05, 0.05)
result = Compound(children=[body, sliver])
"""
    res = run_pipeline(
        code=code,
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
        mesh_solid=mesh_brep,
        mesh_bbox=[40.0, 40.0, 40.0],
    )
    assert res.ok, res.error
    assert res.warnings and "collapsed" in res.warnings[0]
    assert res.analysis.solid_count == 1


def test_hybrid_runner_rejects_shattered_result(mesh_brep, tmp_path):
    """When booleans destroyed the mesh body (dominant fragment < 30% of the
    base volume), the runner must fail with a mesh_cut hint — a 100mm^3
    fragment is not the part."""
    code = """\
from build123d import *

PARAMS = {}
body = load_mesh_solid()
sliver = Pos(500, 500, 500) * Box(0.05, 0.05, 0.05)
result = Compound(children=[sliver, Pos(0, 0, 200) * Box(1, 1, 1)])
"""
    res = run_pipeline(
        code=code,
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
        mesh_solid=mesh_brep,
        mesh_bbox=[40.0, 40.0, 40.0],
    )
    assert not res.ok
    assert "shattered" in (res.error or "")
    assert "mesh_cut" in (res.error or "")
