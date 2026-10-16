"""run_mesh_pipeline / run_mesh_candidates end-to-end on generated geometry."""

import pytest
import trimesh

from cadx.cad.engine import run_mesh_candidates, run_mesh_pipeline, run_mesh_preview
from cadx.cad.mesh_ops import MeshPostParams, flat_base_ratio
from cadx.cad.render import render_available
from cadx.db.models import ProjectSettings
from cadx.llm.schemas import PartSpec


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


def test_mesh_pipeline_ops_applied(knob_glb, tmp_path):
    """post.ops run after the baseline; report lands in res.mesh['ops_report']."""
    res = run_mesh_pipeline(
        mesh_path=knob_glb,
        out_dir=tmp_path / "it1",
        post=MeshPostParams(
            up="+z",
            front="+y",
            ops=[
                {"op": "flatten_bottom", "args": {"height_mm": 2.0}},
                {
                    "op": "boolean",
                    "args": {
                        "op": "union",
                        "shape": "box",
                        "size_xyz": [45, 45, 3],
                        "center": [0, 0, 1.5],
                    },
                },
            ],
        ),
        spec=_spec(),
        settings=ProjectSettings(process="fdm"),
    )
    assert res.ok, res.error
    assert [r["op"] for r in res.mesh["ops_report"]] == ["flatten_bottom", "boolean"]
    assert res.mesh["post"]["ops"][0]["op"] == "flatten_bottom"
    # union succeeded: the result is a single watertight body with a flat plate
    assert res.analysis.watertight is True
    run_stl = tmp_path / "it1" / "_run" / "model.stl"
    m = trimesh.load(str(run_stl))
    assert flat_base_ratio(m) > 0.05


def test_mesh_pipeline_recaps_faces_after_ops(knob_glb, tmp_path):
    """Ops (voxel remesh) can re-inflate the mesh far past the decimate cap —
    the pipeline must re-cap so downstream renders/exports stay cheap."""
    from cadx.runtime_config import RuntimeSettings

    res = run_mesh_pipeline(
        mesh_path=knob_glb,
        out_dir=tmp_path / "it1",
        post=MeshPostParams(ops=[{"op": "remesh", "args": {"voxel_mm": 0.4, "smooth_iters": 0}}]),
        spec=_spec(),
        settings=ProjectSettings(process="fdm"),
        runtime=RuntimeSettings(mesh_decimate_faces=8_000),
    )
    assert res.ok, res.error
    assert res.mesh["triangles"] <= 8_000
    assert res.analysis.watertight is True


def test_mesh_pipeline_bad_op_fails(knob_glb, tmp_path):
    res = run_mesh_pipeline(
        mesh_path=knob_glb,
        out_dir=tmp_path / "it1",
        post=MeshPostParams(
            ops=[{"op": "slice", "args": {"axis": "z", "at_mm": 999, "keep": "high"}}]
        ),
        spec=_spec(),
        settings=ProjectSettings(process="fdm"),
    )
    assert not res.ok
    assert "op failed" in (res.error or "")


def test_mesh_preview(knob_glb, tmp_path):
    if not render_available():
        pytest.skip("offscreen rendering unavailable")
    out = run_mesh_preview(mesh_path=knob_glb, out_dir=tmp_path / "it0")
    assert out["ok"], out.get("error")
    assert set(out["images"]) == {"iso", "front", "top"}
    for png in out["images"].values():
        assert (tmp_path / "it0" / "preview" / png).exists()
    assert out["stats"]["triangles"] > 0


def test_mesh_preview_no_render(knob_glb, tmp_path, monkeypatch):
    # render_available is imported inside run_mesh_preview — patch at source
    monkeypatch.setattr("cadx.cad.render.render_available", lambda: False)
    out = run_mesh_preview(mesh_path=knob_glb, out_dir=tmp_path / "it0")
    assert out["ok"] is False
    assert "unavailable" in (out["error"] or "")
    assert out["stats"]["triangles"] > 0


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
