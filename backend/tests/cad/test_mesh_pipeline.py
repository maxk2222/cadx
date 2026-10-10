"""run_mesh_pipeline / run_mesh_candidates end-to-end on generated geometry."""

import pytest
import trimesh

from cadagent.cad.engine import run_mesh_candidates, run_mesh_pipeline
from cadagent.cad.mesh_ops import MeshPostParams
from cadagent.cad.render import render_available
from cadagent.db.models import ProjectSettings
from cadagent.llm.schemas import PartSpec


@pytest.fixture
def knob_glb(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    m.apply_scale([0.9, 0.8, 1.1])  # knob-ish, normalized units like generators
    p = tmp_path / "raw.glb"
    m.export(str(p), file_type="glb")
    return p


def _spec():
    return PartSpec(
        name="knob",
        summary="organic knob",
        units="mm",
        length_x={"value": 40.0, "unit": "mm", "source": "stated"},
        width_y={"value": 40.0, "unit": "mm", "source": "assumed"},
        height_z={"value": 30.0, "unit": "mm", "source": "assumed"},
        features=[],
        open_questions=[],
    )


def test_mesh_pipeline_basic(knob_glb, tmp_path):
    res = run_mesh_pipeline(
        mesh_path=knob_glb,
        out_dir=tmp_path / "it1",
        post=MeshPostParams(up="+z", front="+y"),
        spec=_spec(),
        settings=ProjectSettings(process="fdm"),
        full_exports=True,
    )
    assert res.ok, res.error
    # scaled so stated X = 40 mm
    assert res.analysis.bbox_size[0] == pytest.approx(40, abs=0.5)
    # frame contract: min Z == 0, centred in XY
    run_stl = tmp_path / "it1" / "_run" / "model.stl"
    assert run_stl.exists()
    m = trimesh.load(str(run_stl))
    assert m.bounds[0][2] == pytest.approx(0, abs=1e-3)
    assert res.dfm is not None
    for fmt in ("stl", "3mf", "glb"):
        assert res.exports.get(fmt), fmt
    assert res.mesh["scale_factor"] > 1.0
    assert res.mesh["input_triangles"] > 0


def test_mesh_pipeline_make_solid(knob_glb, tmp_path):
    res = run_mesh_pipeline(
        mesh_path=knob_glb,
        out_dir=tmp_path / "it1",
        spec=_spec(),
        settings=ProjectSettings(process="fdm"),
        make_solid=True,
        full_exports=True,
    )
    assert res.ok, res.error
    assert res.mesh.get("solid_ok") is True
    assert (tmp_path / "it1" / "_run" / "model.brep").exists()
    # tessellated STEP export from the faceted solid
    assert res.exports.get("step")
    assert res.mesh.get("step_tessellated") is True


def test_mesh_pipeline_post_params(knob_glb, tmp_path):
    res = run_mesh_pipeline(
        mesh_path=knob_glb,
        out_dir=tmp_path / "it1",
        post=MeshPostParams(scale=2.0, mirror_x=True, rotate_z_deg=30),
        spec=_spec(),
        settings=ProjectSettings(process="fdm"),
    )
    assert res.ok, res.error
    # stated X=40, post scale 2 → 80 mm
    assert res.analysis.bbox_size[0] == pytest.approx(80, abs=1.0)


def test_mesh_pipeline_bad_file(tmp_path):
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a mesh")
    res = run_mesh_pipeline(mesh_path=bad, out_dir=tmp_path / "it1")
    assert not res.ok
    assert res.error


def test_mesh_candidates(knob_glb, tmp_path):
    if not render_available():
        pytest.skip("offscreen rendering unavailable")
    out = run_mesh_candidates(mesh_path=knob_glb, out_dir=tmp_path / "it0", size=256)
    assert out["ok"], out.get("error")
    assert set(out["candidates"]) == {"+x", "-x", "+y", "-y", "+z", "-z"}
    assert out["sheet"] == "orient_sheet.png"
    assert (tmp_path / "it0" / "orient_sheet.png").exists()
    for png in out["candidates"].values():
        assert (tmp_path / "it0" / "orient" / png).exists()
    assert out["stats"]["watertight"] is True
