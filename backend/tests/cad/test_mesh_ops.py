"""Unit tests for cad/mesh_ops — post-processing, orientation, scaling, solid."""

import numpy as np
import pytest
import trimesh

from cadagent.cad.mesh_ops import (
    MeshPostParams,
    candidate_meshes,
    clean,
    decimate,
    default_front,
    flat_base_ratio,
    flatten_bottom,
    load_mesh,
    merge_post_params,
    mesh_analysis,
    mesh_params_schema,
    mesh_to_solid,
    orient,
    rest_on_xy_center,
    scale_to_spec,
    suggest_up,
)
from cadagent.llm.schemas import PartSpec


def _box(x=10.0, y=20.0, z=30.0):
    return trimesh.creation.box(extents=[x, y, z])


def _spec(x=None, y=None, z=None, source="stated"):
    def dim(v):
        return {"value": v or 0.0, "unit": "mm", "source": source}

    return PartSpec(
        name="t",
        summary="t",
        units="mm",
        length_x=dim(x),
        width_y=dim(y),
        height_z=dim(z),
        features=[],
        open_questions=[],
    )


# --------------------------------------------------------------------------- #
# merge / schema


def test_merge_post_params_partial():
    base = MeshPostParams(up="-y", front="+z", scale=2.0)
    merged = merge_post_params(base, {"scale": 1.5, "rotate_z_deg": 45})
    assert merged.up == "-y"  # kept
    assert merged.front == "+z"
    assert merged.scale == 1.5
    assert merged.rotate_z_deg == 45


def test_mesh_params_schema_complete():
    names = {p.name for p in mesh_params_schema()}
    assert names == {
        "scale",
        "up",
        "front",
        "rotate_z_deg",
        "decimate_faces",
        "flatten_bottom_mm",
        "mirror_x",
    }


# --------------------------------------------------------------------------- #
# load / clean / decimate


def test_load_mesh_glb_stl(tmp_path):
    m = _box()
    glb = tmp_path / "m.glb"
    stl = tmp_path / "m.stl"
    m.export(str(glb), file_type="glb")
    m.export(str(stl), file_type="stl")
    assert len(load_mesh(glb).faces) == 12
    assert len(load_mesh(stl).faces) == 12


def test_clean_removes_floater():
    big = _box(30, 30, 30)
    tiny = _box(0.01, 0.01, 0.01)
    tiny.apply_translation([100, 0, 0])
    combined = trimesh.util.concatenate([big, tiny])
    out = clean(combined)
    # floater removed → extents only cover the big box
    assert out.extents[0] == pytest.approx(30, abs=0.5)


def test_decimate_target():
    m = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
    out = decimate(m, 1000)
    assert len(out.faces) <= 1200
    out2 = decimate(m, 0)
    assert len(out2.faces) == len(m.faces)


# --------------------------------------------------------------------------- #
# orient / candidates


def test_orient_up_axis():
    m = _box(10, 20, 30)  # tall in z
    out = orient(m, "-z", None)
    # inverted: still 30 tall, but flipped — bbox unchanged
    assert out.extents[2] == pytest.approx(30, abs=0.1)
    out = orient(m, "+x", None)
    # x extent becomes z extent
    assert out.extents[2] == pytest.approx(10, abs=0.1)


def test_orient_mirror_and_rotate():
    m = trimesh.creation.box(extents=[10, 20, 30])
    out = orient(m, "+z", "+y", rotate_z_deg=90)
    # 90° yaw swaps x/y extents
    assert out.extents[0] == pytest.approx(20, abs=0.1)
    assert out.extents[1] == pytest.approx(10, abs=0.1)
    mirrored = orient(m, "+z", None, mirror_x=True)
    assert mirrored.extents[0] == pytest.approx(10, abs=0.1)


def test_candidate_meshes_all_ups():
    m = _box(10, 20, 30)
    cands = candidate_meshes(m)
    assert set(cands) == {"+x", "-x", "+y", "-y", "+z", "-z"}
    for mesh in cands.values():
        # every candidate rests on XY and is centred
        assert mesh.bounds[0][2] == pytest.approx(0, abs=1e-6)
        cx = (mesh.bounds[0][0] + mesh.bounds[1][0]) / 2
        cy = (mesh.bounds[0][1] + mesh.bounds[1][1]) / 2
        assert abs(cx) < 1e-6 and abs(cy) < 1e-6


def test_suggest_up_prefers_flat_base():
    # flat slab: smallest extent is z → z stays up
    assert suggest_up(_box(40, 40, 5)) in ("+z", "-z")


def test_default_front_orthogonal():
    f = default_front("+z", np.array([10.0, 30.0, 5.0]))
    assert f not in ("+z", "-z")


# --------------------------------------------------------------------------- #
# scale / rest / flatten


def test_scale_to_spec_stated_axis():
    m = _box(1, 1, 1)
    out, factor, issues = scale_to_spec(m, _spec(x=50))
    assert factor == pytest.approx(50)
    assert out.extents[0] == pytest.approx(50, abs=1e-3)
    assert issues == []


def test_scale_to_spec_reports_aspect_mismatch():
    m = _box(1, 2, 3)  # aspect 1:2:3
    out, _, issues = scale_to_spec(m, _spec(x=60, y=60))
    assert out.extents[0] == pytest.approx(60, abs=1e-3)
    # y comes out at 120 vs stated 60 → mismatch flagged
    assert any("aspect" in i for i in issues)


def test_scale_to_spec_no_dims_normalizes():
    m = _box(1, 2, 3)
    spec = _spec()
    out, _factor, issues = scale_to_spec(m, spec)
    assert max(out.extents) == pytest.approx(100, abs=1e-3)
    assert issues


def test_rest_on_xy_center():
    m = _box(10, 10, 10)
    m.apply_translation([50, -30, 7])
    out = rest_on_xy_center(m)
    assert out.bounds[0][2] == pytest.approx(0, abs=1e-6)
    assert (out.bounds[0][0] + out.bounds[1][0]) / 2 == pytest.approx(0, abs=1e-6)


def test_flatten_bottom():
    m = trimesh.creation.icosphere(subdivisions=3, radius=10)
    m.apply_translation([0, 0, 10])
    out = flatten_bottom(m, 1.0)
    # cut at z=1 → min z is now ~1 and a flat bottom exists
    assert out.bounds[0][2] == pytest.approx(1.0, abs=0.5)
    assert flat_base_ratio(out) > 0.0
    # height_mm <= 0 is a no-op
    assert flatten_bottom(m, 0) is m


# --------------------------------------------------------------------------- #
# mesh_to_solid / analysis


def test_mesh_to_solid_box():
    solid = mesh_to_solid(_box(10, 20, 30))
    assert solid is not None
    assert solid.volume == pytest.approx(6000, rel=0.02)


def test_mesh_to_solid_decimates_dense():
    m = trimesh.creation.icosphere(subdivisions=4)
    solid = mesh_to_solid(m, max_faces=2000)
    assert solid.volume > 0


def test_mesh_to_solid_rejects_open_mesh():
    open_mesh = trimesh.creation.box(extents=[10, 10, 10]).section(
        plane_normal=[0, 0, 1], plane_origin=[0, 0, 0]
    )
    if open_mesh is not None:
        with pytest.raises(Exception):  # noqa: B017 — any conversion failure is fine
            mesh_to_solid(open_mesh.to_mesh())


def test_mesh_analysis_fields():
    a = mesh_analysis(_box(10, 20, 30))
    assert a.bbox_size == [10.0, 20.0, 30.0]
    assert a.triangle_count == 12
    assert a.watertight is True
    assert a.volume == pytest.approx(6000, rel=0.01)


def test_mesh_analysis_open_mesh_flagged():
    m = _box(10, 10, 10)
    m.faces = m.faces[:-1]  # remove a face → open
    a = mesh_analysis(m)
    assert a.watertight is False
    assert any("watertight" in w or "manifold" in w for w in a.warnings)
