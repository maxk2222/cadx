"""Hybrid bridge: mesh -> faceted solid -> build123d boolean via the sandbox."""

import pytest
import trimesh
from build123d import export_brep

from cadagent.cad.engine import run_pipeline
from cadagent.cad.mesh_ops import mesh_to_solid
from cadagent.db.models import ProjectSettings

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


def test_load_mesh_solid_unavailable_outside_hybrid(tmp_path):
    code = "PARAMS = {}\nresult = load_mesh_solid()\n"
    res = run_pipeline(
        code=code,
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
    )
    assert not res.ok
    assert "hybrid" in (res.error or "").lower()
