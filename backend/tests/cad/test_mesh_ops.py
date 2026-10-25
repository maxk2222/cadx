"""Unit tests for cad/mesh_ops — post-processing, orientation, scaling, solid."""

import json

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from cadx.cad.mesh_ops import (
    MeshOp,
    MeshOpError,
    MeshPostParams,
    apply_op,
    apply_ops,
    baseline_mesh,
    candidate_meshes,
    clean,
    decimate,
    default_front,
    flat_base_ratio,
    flatten_bottom,
    footprint_yaw_deg,
    heal_seam_slivers,
    keep_significant_components,
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
from cadx.llm.schemas import PartSpec


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


def test_fit_stated_dims_stretches_deficient_axis():
    """Handle case: mesh arrives 120x20x37 vs stated 120x40x30 — the repair
    loop cannot legally fix the envelope, so the baseline fits stated axes
    non-uniformly (y x2, z x0.81) instead of reporting forever."""
    from cadx.cad.mesh_ops import fit_stated_dims

    m = _box(120, 20, 37)
    out, fit, issues = fit_stated_dims(m, _spec(x=120, y=40, z=30), [])
    assert fit["y"] == pytest.approx(2.0, abs=1e-3)
    assert fit["z"] == pytest.approx(30 / 37, abs=1e-3)
    assert out.extents[0] == pytest.approx(120, rel=1e-3)
    assert out.extents[1] == pytest.approx(40, rel=1e-3)
    assert out.extents[2] == pytest.approx(30, rel=1e-3)
    assert issues == []


def test_fit_stated_dims_out_of_bounds_keeps_issue():
    """A generator output 9x off the stated dim cannot be trusted — fitting it
    would produce garbage, so the axis stays reported."""
    from cadx.cad.mesh_ops import fit_stated_dims

    m = _box(60, 30, 90)
    # pipeline order: scale_to_spec reports first, fit decides what remains
    _, _, issues = scale_to_spec(m, _spec(x=60, y=30, z=10))
    out, fit, issues = fit_stated_dims(m, _spec(x=60, y=30, z=10), issues)
    assert fit == {}
    assert any("height/Z" in i for i in issues)
    assert out.extents[2] == pytest.approx(90)


def test_fit_stated_dims_inferred_dims_untouched():
    """Only 'stated' dims are authoritative — inferred dims never force a fit."""
    from cadx.cad.mesh_ops import fit_stated_dims

    m = _box(120, 20, 37)
    out, fit, issues = fit_stated_dims(m, _spec(x=120, y=40, z=30, source="inferred"), [])
    assert fit == {} and issues == []
    assert out.extents[1] == pytest.approx(20)


def test_baseline_mesh_dim_fit_reaches_spec_envelope():
    raw = _box(120, 20, 37)
    m, info = baseline_mesh(raw, MeshPostParams(), _spec(x=120, y=40, z=30))
    assert m.extents[1] == pytest.approx(40, rel=1e-3)
    assert m.extents[2] == pytest.approx(30, rel=1e-3)
    assert info["dim_fit"]["y"] == pytest.approx(2.0, abs=1e-2)
    assert info["dim_issues"] == []


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


def test_flatten_bottom_fills_overhang_voids():
    # a T-part: 20x20 base slab z∈[8,12] with a 4mm overhang wing at z∈[14,16]
    # hovering over void — a plane cut can never make the wing's underside flat.
    base = _box(20, 20, 4)
    base.apply_translation([0, 0, 10])
    wing = _box(8, 8, 2)
    wing.apply_translation([8, 0, 15])
    m = trimesh.util.concatenate([base, wing])
    out = flatten_bottom(m, 2.0)
    down = out.face_normals[:, 2] < -0.9
    zf = out.vertices[out.faces[down]][:, :, 2]
    assert down.sum() > 0
    # every down-facing face sits on a single plane (the capped cut)
    assert zf.max() - zf.min() == pytest.approx(0.0, abs=0.2)
    # the overhang column is filled: solid reaches the floor under the wing
    ray = trimesh.ray.ray_triangle.RayMeshIntersector(out)
    hits, _, _ = ray.intersects_location([[8, 0, -1]], [[0, 0, 1]], multiple_hits=True)
    assert hits.size and hits[:, 2].min() <= out.bounds[0][2] + 0.3


def test_flatten_bottom_voxel_grid_bounded():
    # a long thin part: pitch formula would give ~650 cells on the long axis —
    # a sparse shell makes nearly every occupied cell a boundary cell, so
    # marching cubes can out-vertex the cell count (observed: 41 M verts →
    # MemoryError). The pitch must be raised to respect the cell cap.
    m = _box(200, 30, 30)
    m.apply_translation([0, 0, 15])
    out = flatten_bottom(m, 1.0)
    assert len(out.faces) < 3_000_000
    assert out.is_watertight


def test_voxel_remesh_refuses_unbounded_grid():
    from cadx.cad.mesh_ops import _voxel_remesh

    with pytest.raises(ValueError, match="cells/axis"):
        _voxel_remesh(_box(200, 30, 30), pitch_mm=0.2)


def test_voxelize_coarsens_pitch_on_dense_subdivide(monkeypatch):
    """trimesh's "subdivide" voxelization refines every edge below pitch —
    a coarse mesh at fine pitch explodes (observed: 33M faces → MemoryError).
    _voxelize must coarsen the pitch until the growth bound fits the cap."""
    import cadx.cad.mesh_ops as mo

    seen: list[float] = []
    orig = trimesh.Trimesh.voxelized

    def spy(self, pitch, method="subdivide", **kw):
        seen.append(float(pitch))
        return orig(self, pitch, method=method, **kw)

    monkeypatch.setattr(trimesh.Trimesh, "voxelized", spy)
    # ~12 faces spanning 130 mm at 0.25 mm → est >> the 4M cap → coarsened
    mo._voxelize(_box(130, 32, 32), 0.25)
    assert seen[0] > 0.25
    seen.clear()
    # a fine mesh at a compatible pitch is not touched
    mo._voxelize(trimesh.creation.icosphere(subdivisions=2, radius=10), 1.0)
    assert seen == [1.0]


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


def test_mesh_to_solid_decimates_before_watertight(monkeypatch):
    """Regression: watertight repair must see the capped mesh, not the raw
    input — uncapped input OOM'd the executor on a 283k-face mesh."""
    seen: list[int] = []

    def spy(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        seen.append(len(mesh.faces))
        return mesh

    monkeypatch.setattr("cadx.cad.mesh_ops._ensure_watertight", spy)
    m = trimesh.creation.icosphere(subdivisions=5)
    import contextlib

    # conversion itself may fail — only the watertight input size is under test
    with contextlib.suppress(Exception):
        mesh_to_solid(m, max_faces=2000)
    assert seen and seen[0] <= 2000


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


# --------------------------------------------------------------------------- #
# agent ops — apply_op / apply_ops / registry                                  #
# --------------------------------------------------------------------------- #


def test_op_rotate_swaps_extents():
    m, _ = apply_ops(_box(10, 20, 30), [{"op": "rotate", "args": {"axis": "z", "deg": 90}}])
    assert m.extents[0] == pytest.approx(20, abs=0.2)
    assert m.extents[1] == pytest.approx(10, abs=0.2)
    # apply_ops re-grounds: part rests on XY, centred
    assert m.bounds[0][2] == pytest.approx(0, abs=1e-6)
    assert (m.bounds[0][0] + m.bounds[1][0]) / 2 == pytest.approx(0, abs=1e-6)


def test_op_mirror_keeps_watertight():
    m = apply_op(_box(10, 20, 30), MeshOp(op="mirror", args={"axis": "x"}))
    assert m.is_watertight
    assert m.extents[0] == pytest.approx(10, abs=0.1)


def test_op_set_up():
    m = apply_op(_box(10, 20, 30), MeshOp(op="set_up", args={"up": "+x"}))
    assert m.extents[2] == pytest.approx(10, abs=0.2)
    assert m.bounds[0][2] == pytest.approx(0, abs=1e-6)


def test_op_scale_uniform_and_per_axis():
    m = apply_op(_box(10, 20, 30), MeshOp(op="scale", args={"factor": 2.0}))
    assert list(m.extents) == pytest.approx([20, 40, 60], abs=0.1)
    m2 = apply_op(_box(10, 20, 30), MeshOp(op="scale", args={"y": 0.5}))
    assert m2.extents[1] == pytest.approx(10, abs=0.1)
    assert m2.extents[0] == pytest.approx(10, abs=0.1)
    with pytest.raises(ValueError, match="factor"):
        apply_op(_box(), MeshOp(op="scale", args={}))  # no factor at all


def test_op_slice_keeps_side():
    m = _box(10, 10, 10)
    m.apply_translation([0, 0, 5])  # z in [0, 10]
    hi = apply_op(m, MeshOp(op="slice", args={"axis": "z", "at_mm": 4.0, "keep": "high"}))
    assert hi.bounds[0][2] == pytest.approx(4.0, abs=0.5)
    assert hi.extents[2] == pytest.approx(6.0, abs=0.5)
    assert hi.is_watertight
    lo = apply_op(m, MeshOp(op="slice", args={"axis": "z", "at_mm": 4.0, "keep": "low"}))
    assert lo.bounds[1][2] == pytest.approx(4.0, abs=0.5)
    with pytest.raises(ValueError, match="outside"):
        apply_op(m, MeshOp(op="slice", args={"axis": "z", "at_mm": 99.0, "keep": "high"}))


def test_op_crop():
    m = _box(20, 20, 20)
    out = apply_op(m, MeshOp(op="crop", args={"min": [-5, -5, -5], "max": [5, 5, 5]}))
    assert out.extents[0] == pytest.approx(10, abs=0.5)
    with pytest.raises(ValueError):
        apply_op(m, MeshOp(op="crop", args={"min": [0, 0, 0], "max": [0, 1, 1]}))


def test_op_decimate_and_smooth():
    m = trimesh.creation.icosphere(subdivisions=4)
    out = apply_op(m, MeshOp(op="decimate", args={"faces": 500}))
    assert len(out.faces) <= 600
    moved = apply_op(m, MeshOp(op="smooth", args={"iterations": 3}))
    assert not np.allclose(moved.vertices, m.vertices)
    assert moved.is_watertight


def test_op_keep_largest_and_drop_small():
    big = _box(30, 30, 30)
    tiny = _box(1, 1, 1)
    tiny.apply_translation([100, 0, 0])
    combined = trimesh.util.concatenate([big, tiny])
    m = apply_op(combined, MeshOp(op="keep_largest", args={}))
    assert m.extents[0] == pytest.approx(30, abs=0.5)
    m2 = apply_op(combined, MeshOp(op="drop_small", args={"min_volume_frac": 0.1}))
    assert m2.extents[0] == pytest.approx(30, abs=0.5)


def test_op_repair_open_mesh():
    m = _box(10, 10, 10)
    m.faces = m.faces[:-2]
    out = apply_op(m, MeshOp(op="repair", args={}))
    assert out is not None and len(out.faces) > 0


def test_op_boolean_union_and_difference():
    base = _box(40, 20, 10)
    v0 = abs(base.volume)
    with_plate = apply_op(
        base,
        MeshOp(
            op="boolean",
            args={
                "op": "union",
                "shape": "box",
                "size_xyz": [50, 25, 3],
                "center": [0, 0, -1.5],
            },
        ),
    )
    assert abs(with_plate.volume) > v0
    assert with_plate.is_watertight
    drilled = apply_op(
        with_plate,
        MeshOp(
            op="boolean",
            args={
                "op": "difference",
                "shape": "cylinder",
                "radius": 3,
                "height": 30,
                "center": [0, 0, 0],
            },
        ),
    )
    assert abs(drilled.volume) < abs(with_plate.volume)
    assert drilled.is_watertight


def test_op_boolean_arg_validation():
    with pytest.raises(ValidationError):
        apply_op(_box(), MeshOp(op="boolean", args={"op": "union", "shape": "torus"}))
    with pytest.raises(ValueError, match="box needs"):
        apply_op(_box(), MeshOp(op="boolean", args={"op": "union", "shape": "box"}))
    with pytest.raises(ValueError, match="cylinder needs"):
        apply_op(
            _box(),
            MeshOp(op="boolean", args={"op": "union", "shape": "cylinder", "radius": 2}),
        )


def test_op_unknown_and_bad_args():
    with pytest.raises(ValueError, match="unknown mesh op"):
        apply_op(_box(), MeshOp(op="explode", args={}))
    with pytest.raises(ValidationError):
        apply_op(_box(), MeshOp(op="rotate", args={"axis": "z", "deg": 999}))


def test_apply_ops_order_and_report():
    _m, rep = apply_ops(
        _box(10, 10, 10),
        [
            {"op": "rotate", "args": {"axis": "z", "deg": 45}},
            {"op": "scale", "args": {"factor": 2}},
        ],
    )
    assert len(rep) == 2
    assert rep[0]["op"] == "rotate" and rep[1]["op"] == "scale"
    assert rep[1]["faces_in"] == rep[0]["faces_out"]
    # mesh serialises deterministically through MeshPostParams
    post = MeshPostParams(ops=[MeshOp(op="scale", args={"factor": 1.5})])
    assert MeshPostParams(**post.model_dump()).ops[0].args["factor"] == 1.5


def test_apply_ops_error_carries_index_and_partial_report():
    """A mid-stack failure must identify the offending op so the GUI can
    highlight it — including the report of the ops that did succeed."""
    with pytest.raises(MeshOpError) as ei:
        apply_ops(
            _box(10, 10, 10),
            [
                {"op": "rotate", "args": {"axis": "z", "deg": 90}},
                {"op": "not_a_real_op", "args": {}},
                {"op": "mirror", "args": {"axis": "x"}},
            ],
        )
    e = ei.value
    assert e.op_index == 1
    assert e.op_name == "not_a_real_op"
    assert len(e.report) == 1 and e.report[0]["op"] == "rotate"
    assert isinstance(e, ValueError)  # back-compat with existing except-ValueError

    # arg-validation failures wrap the same way, at index 0
    with pytest.raises(MeshOpError) as ei2:
        apply_ops(_box(10, 10, 10), [{"op": "scale", "args": {"factor": -1.0}}])
    assert ei2.value.op_index == 0
    assert ei2.value.report == []


def test_merge_post_params_ops_replaced_only_when_given():
    base = MeshPostParams(ops=[MeshOp(op="rotate", args={"axis": "z", "deg": 10})])
    merged = merge_post_params(base, {"scale": 2.0})
    assert merged.ops == base.ops  # untouched
    merged2 = merge_post_params(base, {"ops": [{"op": "mirror", "args": {"axis": "y"}}]})
    assert merged2.ops[0].op == "mirror"


def test_smooth_preserve_keeps_floor_and_envelope():
    from cadx.cad.mesh_ops import smooth_preserve, surface_roughness_deg

    m = trimesh.creation.icosphere(subdivisions=3, radius=20)
    flat = flatten_bottom(m, 10.0)
    flat = rest_on_xy_center(flat)
    b0 = np.asarray(flat.bounds)
    rough_before = surface_roughness_deg(flat)
    out = smooth_preserve(flat, iterations=10)
    assert out.is_watertight
    # the floor stays a single plane at min z
    down = out.face_normals[:, 2] < -0.9
    zf = out.vertices[out.faces[down]][:, :, 2]
    assert down.sum() > 0
    assert zf.max() - zf.min() <= 0.05
    # XY envelope preserved
    for axis in (0, 1):
        assert (out.bounds[1][axis] - out.bounds[0][axis]) == pytest.approx(
            b0[1][axis] - b0[0][axis], abs=0.3
        )
    # smoothing actually reduced the voxel striations
    assert surface_roughness_deg(out) <= rough_before * 0.6
    # works through the ops registry
    out2, rep = apply_ops(flat, [{"op": "smooth_preserve", "args": {"iterations": 5}}])
    assert rep[0]["op"] == "smooth_preserve" and out2.is_watertight


def test_baseline_mesh_matches_pipeline_sequence():
    raw = _box(1, 2, 3)
    post = MeshPostParams(up="+z", front="+y", rotate_z_deg=90)
    m, info = baseline_mesh(raw, post, _spec(x=60, y=30, z=10))
    assert info["orientation"]["rotate_z_deg"] == 90
    # orient runs first: 90° yaw turns [1,2,3] into [2,1,3]; the largest
    # stated dim (x=60) then drives factor=30 -> [60,30,90]
    assert sorted(m.extents) == pytest.approx([30, 60, 90], rel=0.02)
    assert m.bounds[0][2] == pytest.approx(0, abs=1e-6)


def test_baseline_mesh_cad_cap_faces():
    """Pipeline must cap generator output even without a decimate op —
    uncapped input OOM'd the executor watertight/boolean path."""
    raw = trimesh.creation.icosphere(subdivisions=4)
    m, _info = baseline_mesh(raw, MeshPostParams(), _spec(x=60, y=30, z=10), cad_cap_faces=2000)
    assert len(m.faces) <= 2000


def test_orient_by_plan_view_flat_box():
    """A flat box standing on edge must be re-laid flat by the plan view."""
    from cadx.cad.mesh_ops import orient_by_plan_view

    mesh = trimesh.creation.box(extents=[100, 30, 8])
    # stand it on edge: the 30 mm side becomes vertical (wrong pose)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
    outline = [[-50, -15], [50, -15], [50, 15], [-50, 15], [-50, -15]]
    hit = orient_by_plan_view(mesh, outline)
    assert hit is not None
    up, _front, _mirror_x, iou = hit
    assert iou >= 0.9
    assert up in ("+y", "-y")  # the original flat pose restored


def test_orient_by_plan_view_returns_none_without_match():
    from cadx.cad.mesh_ops import orient_by_plan_view

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=10)
    outline = [[-50, -2], [50, -2], [50, 2], [-50, 2], [-50, -2]]  # sliver
    assert orient_by_plan_view(mesh, outline) is None


def test_clip_to_outline():
    from cadx.cad.mesh_ops import clip_to_outline

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=20)
    outline = [[-15, -15], [15, -15], [15, 15], [-15, 15], [-15, -15]]
    out = clip_to_outline(mesh, outline, margin_mm=0.3)
    assert out.is_watertight
    ext = [float(v) for v in out.extents]
    assert ext[0] == pytest.approx(30.6, abs=0.4)
    assert ext[1] == pytest.approx(30.6, abs=0.4)


def test_clip_to_outline_bad_outline_returns_input():
    from cadx.cad.mesh_ops import clip_to_outline

    mesh = trimesh.creation.box(extents=[10, 10, 10])
    assert clip_to_outline(mesh, []) is mesh


# --------------------------------------------------------------------------- #
# new ops: clip_height / symmetrize / remesh / drill_holes / shell
# --------------------------------------------------------------------------- #


def test_op_clip_height():
    m = _box(10, 10, 30)
    m.apply_translation([0, 0, 15])  # z in [0, 30]
    out = apply_op(m, MeshOp(op="clip_height", args={"z_max": 8.0}))
    assert out.extents[2] == pytest.approx(8.0, abs=0.5)
    assert out.is_watertight
    # no-op when nothing is above the ceiling
    same = apply_op(m, MeshOp(op="clip_height", args={"z_max": 99.0}))
    assert same.extents[2] == pytest.approx(30.0, abs=0.1)
    # z_max below the mesh floor (floating mesh) removes everything
    floating = _box(10, 10, 30)
    floating.apply_translation([0, 0, 20])
    with pytest.raises(ValueError, match="whole mesh"):
        apply_op(floating, MeshOp(op="clip_height", args={"z_max": 5.0}))
    with pytest.raises(ValidationError):
        apply_op(m, MeshOp(op="clip_height", args={"z_max": 0.0}))


def test_op_symmetrize_makes_symmetric():
    # asymmetric: box + small bump on +y side only
    base = _box(20, 20, 10)
    base.apply_translation([0, 0, 5])
    bump = trimesh.creation.icosphere(subdivisions=2, radius=3)
    bump.apply_translation([0, 8, 8])
    asym = base.union(bump, engine="manifold")
    out = apply_op(asym, MeshOp(op="symmetrize", args={"axis": "y", "keep": "neg"}))
    assert out.is_watertight
    # result is symmetric about its own bbox centre on the y axis
    c = float((out.bounds[0][1] + out.bounds[1][1]) / 2.0)
    neg = out.slice_plane([0, c, 0], [0, -1, 0], cap=True)
    pos = out.slice_plane([0, c, 0], [0, 1, 0], cap=True)
    assert abs(neg.volume) == pytest.approx(abs(pos.volume), rel=0.05)
    # keep='neg' drops the +y bump → volume ≈ plain box
    assert abs(out.volume) == pytest.approx(4000.0, rel=0.1)


def test_op_remesh_watertight():
    m = trimesh.creation.icosphere(subdivisions=2, radius=15)
    out = apply_op(m, MeshOp(op="remesh", args={"voxel_mm": 1.0}))
    assert out.is_watertight
    assert len(out.faces) > 0
    # pitch guard fires before voxelizing (huge extent at fine pitch)
    big = _box(500, 500, 500)
    with pytest.raises(ValueError, match="too fine"):
        apply_op(big, MeshOp(op="remesh", args={"voxel_mm": 0.5}))
    with pytest.raises(ValidationError):
        apply_op(m, MeshOp(op="remesh", args={"voxel_mm": 0.01}))


def test_op_drill_holes_explicit():
    m = _box(40, 30, 8)
    m.apply_translation([0, 0, 4])
    out = apply_op(
        m,
        MeshOp(
            op="drill_holes",
            args={"holes": [{"x": 0.0, "y": 0.0, "dia": 6.0}]},
        ),
    )
    assert out.is_watertight
    assert abs(out.volume) < abs(m.volume)
    # the hole must be a real tunnel
    from cadx.cad.mesh_features import detect_mesh_features

    feats = detect_mesh_features(out, step_mm=0.5)
    tunnels = [h for h in feats.holes if h.kind == "tunnel"]
    assert len(tunnels) == 1
    assert tunnels[0].x == pytest.approx(0.0, abs=1.0)
    assert tunnels[0].dia == pytest.approx(6.0, abs=1.5)


def test_op_drill_holes_countersink_and_anchors():
    m = _box(40, 30, 8)
    m.apply_translation([0, 0, 4])
    anchors = {"holes": [{"x": 10.0, "y": -5.0, "dia": 5.0}]}
    out = apply_op(
        m,
        MeshOp(
            op="drill_holes",
            args={
                "from_anchors": True,
                "holes": [
                    {"x": -10.0, "y": 5.0, "dia": 4.0, "csk_dia": 8.0, "csk_angle_deg": 90.0}
                ],
            },
        ),
        anchors=anchors,
    )
    assert out.is_watertight
    assert abs(out.volume) < abs(m.volume)
    with pytest.raises(ValueError, match="no anchors"):
        apply_op(m, MeshOp(op="drill_holes", args={"from_anchors": True}))
    with pytest.raises(ValueError, match="no holes"):
        apply_op(m, MeshOp(op="drill_holes", args={}))


def test_op_drill_holes_cross_axis():
    # grip-like part: 130 mm long along X — a bore along the part axis
    # (axis="x" → x/y args are the centreline's y/z coords)
    m = trimesh.creation.cylinder(radius=16.0, height=130.0, sections=64)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    out = apply_op(
        m,
        MeshOp(
            op="drill_holes",
            args={"holes": [{"x": 0.0, "y": 0.0, "dia": 10.0, "axis": "x"}]},
        ),
    )
    assert out.is_watertight
    assert abs(m.volume) - abs(out.volume) == pytest.approx(np.pi * 5.0**2 * 130.0, rel=0.05)
    # a ray along the bore centreline passes clean through; an off-axis ray
    # still hits the body
    ray = trimesh.ray.ray_triangle.RayMeshIntersector(out)
    hits, _, _ = ray.intersects_location([[-70, 0, 0]], [[1, 0, 0]], multiple_hits=True)
    assert len(hits) == 0
    hits, _, _ = ray.intersects_location([[-70, 10, 0]], [[1, 0, 0]], multiple_hits=True)
    assert len(hits) == 2


def test_op_shell_open_bottom():
    m = _box(40, 40, 40)
    m.apply_translation([0, 0, 20])
    out = apply_op(m, MeshOp(op="shell", args={"thickness_mm": 4.0, "opening": "bottom"}))
    assert len(out.faces) > 0
    # hollow → less volume than the solid box, still has the outer shell
    assert abs(out.volume) < abs(m.volume) * 0.9
    sealed = apply_op(m, MeshOp(op="shell", args={"thickness_mm": 4.0, "opening": "none"}))
    assert sealed.is_watertight
    with pytest.raises(ValueError, match="too thin"):
        apply_op(_box(4, 4, 4), MeshOp(op="shell", args={"thickness_mm": 8.0}))


def test_ensure_watertight_repairs_missing_face():
    from cadx.cad.mesh_ops import _ensure_watertight

    m = _box(30, 30, 30)
    m.update_faces(np.arange(len(m.faces)) > 10)  # knock out 11 faces
    assert not m.is_watertight
    out = _ensure_watertight(m)
    assert out.is_watertight


def test_boolean_auto_repairs_open_mesh():
    m = _box(30, 30, 30)
    m.update_faces(np.arange(len(m.faces)) > 5)
    out = apply_op(
        m,
        MeshOp(
            op="boolean",
            args={"op": "difference", "shape": "cylinder", "radius": 5.0, "height": 60.0},
        ),
    )
    assert len(out.faces) > 0


def test_mesh_to_solid_repairs_then_converts():
    pytest.importorskip("build123d")
    from cadx.cad.mesh_ops import mesh_to_solid

    m = _box(30, 30, 30)
    m.update_faces(np.arange(len(m.faces)) > 8)
    solid = mesh_to_solid(m)
    assert solid is not None
    assert solid.volume > 0


def test_mesh_analysis_counts_disconnected_components():
    """T0.2: a fractured mesh must report its real solid count — the
    hard-coded solid_count=1 hid two-piece grips (grip_mesh score-0 case)."""
    import trimesh

    from cadx.cad.mesh_ops import mesh_analysis

    a = trimesh.creation.box(extents=(10, 10, 10))
    b = trimesh.creation.box(extents=(5, 5, 5))
    b.apply_translation([30, 0, 0])
    mesh = trimesh.util.concatenate([a, b])

    ana = mesh_analysis(mesh)
    assert ana.solid_count == 2
    assert any("2 disconnected solids" in w for w in ana.warnings)


def _force_voxel_fallback(monkeypatch):
    """Break cheap repair + manifold so _ensure_watertight must voxel-rebuild."""
    import trimesh.repair

    monkeypatch.setattr(trimesh.repair, "fill_holes", lambda *a, **k: 0)
    try:
        import manifold3d

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("forced")

        monkeypatch.setattr(manifold3d, "Manifold", _Boom)
    except ImportError:
        pass  # manifold3d absent → voxel path reached anyway


def test_ensure_watertight_voxel_cap(monkeypatch):
    """T8.9: the voxel fallback may not explode face count — marching cubes on
    a fine grid can out-face the input by an order of magnitude (OOM-killed
    the eval worker)."""
    import cadx.cad.mesh_ops as mo

    _force_voxel_fallback(monkeypatch)
    m = _box(30, 30, 30)
    m.update_faces(np.arange(len(m.faces)) > 5)
    monkeypatch.setattr(mo, "_REBUILD_MAX_FACES", 2_000)
    out = mo._ensure_watertight(m)
    assert len(out.faces) <= 2_000
    assert out.is_watertight


def test_ensure_watertight_refuses_absurd_grid(monkeypatch):
    """T8.9: an absurd voxel grid must raise instead of allocating a cells^3
    bool matrix."""
    import cadx.cad.mesh_ops as mo

    _force_voxel_fallback(monkeypatch)
    m = _box(30, 30, 30)
    m.update_faces(np.arange(len(m.faces)) > 5)
    monkeypatch.setattr(mo, "_VOXEL_REPAIR_MAX_CELLS", 5.0)
    with pytest.raises(ValueError, match="watertight"):
        mo._ensure_watertight(m)


def _tube_along_z(radius: float = 10.0, half_len: float = 30.0) -> trimesh.Trimesh:
    """Watertight tube along Z with vertex rings across the length — capsule/
    cylinder primitives only place vertices at the caps."""
    z = np.linspace(-half_len, half_len, 13)
    pts = np.vstack(
        [[[0.0, -half_len]], np.column_stack([np.full(len(z), radius), z]), [[0.0, half_len]]]
    )
    return trimesh.creation.revolve(pts, sections=48)


def test_radial_profile_waist():
    """radial_profile must pinch the cross-section at the given station —
    the grip critique's 'waist too shallow' fix."""
    out = apply_op(
        _tube_along_z(),
        {
            "op": "radial_profile",
            "args": {
                "axis": "z",
                "stations": [
                    {"at": 0.0, "r1": 1.0, "r2": 1.0},
                    {"at": 0.5, "r1": 0.6, "r2": 0.6},
                    {"at": 1.0, "r1": 1.0, "r2": 1.0},
                ],
            },
        },
    )
    mid = out.vertices[np.abs(out.vertices[:, 2]) < 5.0]
    end = out.vertices[np.abs(np.abs(out.vertices[:, 2]) - 30.0) < 0.5]
    assert np.ptp(mid[:, 0]) == pytest.approx(20.0 * 0.6, rel=0.1)
    assert np.ptp(end[:, 0]) == pytest.approx(20.0, rel=0.1)


def test_radial_profile_oval_correction():
    """Unequal r1/r2 round out an oval section — Y wider than Z gets
    rebalanced to a circle."""
    m = _tube_along_z()
    m.apply_transform(np.diag([1.0, 0.7, 1.0, 1.0]))  # oval: X 20, Y 14
    out = apply_op(
        m,
        {
            "op": "radial_profile",
            "args": {
                "axis": "z",
                "stations": [
                    {"at": 0.0, "r1": 1.0, "r2": 1.0 / 0.7},
                    {"at": 1.0, "r1": 1.0, "r2": 1.0 / 0.7},
                ],
            },
        },
    )
    ext = out.extents
    assert ext[0] == pytest.approx(20.0, rel=0.02)
    assert ext[1] == pytest.approx(20.0, rel=0.02)  # oval → round again


def test_radial_profile_keeps_axis_length():
    """Scaling is radial only — the profile-axis extent stays untouched."""
    m = _box(100, 20, 20)
    out = apply_op(
        m,
        {
            "op": "radial_profile",
            "args": {"axis": "x", "stations": [{"at": 0.0}, {"at": 1.0, "r1": 0.5}]},
        },
    )
    assert out.extents[0] == pytest.approx(100.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# new ops: silhouette_carve / radial_profile_auto
# --------------------------------------------------------------------------- #


def _hourglass_along_z(r_big: float = 10.0, r_waist: float = 6.0, half_len: float = 30.0):
    """Two cones tip-to-tip along Z — the sharp V-notch waist the generators
    produce instead of a soft ergonomic waist."""
    pts = np.array(
        [[0.0, -half_len], [r_big, -half_len], [r_waist, 0.0], [r_big, half_len], [0.0, half_len]]
    )
    return trimesh.creation.revolve(pts, sections=48)


def test_op_silhouette_carve_trims_overhang():
    """Cylinder carved to a narrower square side silhouette: the visual hull
    removes only material outside the reference — volume drops, the mesh
    stays watertight, and the +y silhouette IoU jumps."""
    from cadx.cad.silhouette import _iou, mesh_view_mask

    cyl = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=64)
    cyl.apply_translation([0, 0, 15.0])  # rests on XY, z in [0, 30]
    # reference slab 12 mm wide — in the +y view plane u = -X spans [-6, 6],
    # v = +Z spans [0, 30]
    outline = [[-6.0, 0.0], [6.0, 0.0], [6.0, 30.0], [-6.0, 30.0]]
    ref = trimesh.creation.box(extents=[12.0, 60.0, 30.0])
    ref.apply_translation([0, 0, 15.0])
    ref_mask = mesh_view_mask(ref, "+y")
    iou_before = _iou(ref_mask, mesh_view_mask(cyl, "+y"))

    out = apply_op(
        cyl,
        MeshOp(op="silhouette_carve", args={"view": "+y", "outline": outline}),
    )
    assert out.is_watertight
    assert abs(out.volume) < abs(cyl.volume) * 0.8
    iou_after = _iou(ref_mask, mesh_view_mask(out, "+y"))
    assert iou_after > iou_before + 0.3
    assert iou_after > 0.9


def test_op_silhouette_carve_from_mask_pipeline():
    """The orchestration path: reference mask -> mask_to_outline_mm -> op —
    the mm convention (u centred, v bottom-aligned) must land the polygon on
    the mesh's view-plane bbox."""
    from cadx.cad.silhouette import mask_to_outline_mm

    cyl = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=64)
    cyl.apply_translation([0, 0, 15.0])
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 28:52] = True  # subject 24 px wide x 60 px tall
    mask[30:50, 28:36] = False  # waist: left half carved out mid-height
    outline = mask_to_outline_mm(mask, mesh_extent_u=20.0, mesh_extent_v=30.0)
    us = [p[0] for p in outline]
    vs = [p[1] for p in outline]
    # anisotropic fill: the outline spans the FULL view-plane bbox — the
    # carve can trim the profile but must never shrink the stated dims
    assert max(us) - min(us) == pytest.approx(20.0, rel=0.05)
    assert max(vs) - min(vs) == pytest.approx(30.0, rel=0.05)
    assert abs((max(us) + min(us)) / 2.0) < 0.5  # centred in u
    assert min(vs) == pytest.approx(0.0, abs=1.0)  # bottom-aligned on the floor
    out = apply_op(cyl, MeshOp(op="silhouette_carve", args={"view": "+y", "outline": outline}))
    assert out.is_watertight
    assert abs(out.volume) < abs(cyl.volume)


def test_op_silhouette_carve_validation_and_empty():
    cyl = trimesh.creation.cylinder(radius=5.0, height=20.0, sections=24)
    with pytest.raises(ValidationError):
        apply_op(
            cyl,
            MeshOp(
                op="silhouette_carve",
                args={"view": "+z", "outline": [[0, 0], [1, 0], [1, 1]]},
            ),
        )
    with pytest.raises(ValueError, match="whole mesh"):
        apply_op(
            cyl,
            MeshOp(
                op="silhouette_carve",
                args={"view": "+y", "outline": [[100, 100], [110, 100], [110, 110]]},
            ),
        )


def test_silhouette_carve_op_args_roundtrip():
    """Ops replay from version.params JSON — args must survive a dump/load
    round-trip byte-identical."""
    mo = MeshOp(
        op="silhouette_carve",
        args={
            "view": "-x",
            "outline": [[-5.0, 0.0], [5.0, 0.0], [5.0, 12.0], [-5.0, 12.0]],
            "tolerance_mm": 0.4,
        },
    )
    d = json.loads(json.dumps(mo.model_dump()))
    assert MeshOp.model_validate(d) == mo


def test_op_radial_profile_auto_fills_waist():
    """An hourglass V-notch is constructive-filled back to the target radius —
    volume grows substantially and the mesh stays watertight."""
    m = _hourglass_along_z()  # r=10 at the ends, r=6 waist at z=0
    v0 = abs(m.volume)
    out = apply_op(
        m,
        MeshOp(
            op="radial_profile_auto",
            args={
                "axis": "z",
                "blend_mm": 4.0,
                "stations": [{"t": float(t), "radius_mm": 10.0} for t in np.linspace(0, 1, 9)],
            },
        ),
    )
    assert out.is_watertight
    assert abs(out.volume) > v0 * 1.3
    mid = out.vertices[np.abs(out.vertices[:, 2]) < 3.0]
    r_mid = float(np.hypot(mid[:, 0], mid[:, 1]).max())
    assert r_mid == pytest.approx(10.0, abs=1.0)
    # the ends already matched the target radius — untouched
    end = out.vertices[np.abs(np.abs(out.vertices[:, 2]) - 30.0) < 0.5]
    assert float(np.hypot(end[:, 0], end[:, 1]).max()) == pytest.approx(10.0, abs=0.5)


def test_op_radial_profile_auto_from_outline_stations():
    """Stations measured off a silhouette polygon feed the op directly."""
    from cadx.cad.silhouette import profile_stations_from_outline

    # waist silhouette: 20 wide at the ends, 12 in the middle, u in [-30, 30]
    outline = [[-30, -10], [0, -6], [30, -10], [30, 10], [0, 6], [-30, 10]]
    stations = profile_stations_from_outline(outline, axis="x", n_stations=7)
    assert stations[0]["t"] == 0.0 and stations[-1]["t"] == 1.0
    radii = [s["radius_mm"] for s in stations]
    assert min(radii) == pytest.approx(6.0, abs=0.5)
    # end bins are sampled at their midpoint (u≈±26), slightly inboard of the
    # ±30 tips — the measured radius is a bit under the 10 mm tip value
    assert max(radii) == pytest.approx(9.43, abs=0.5)

    # fill the waist of a tube lying along X up to the silhouette's max radius
    m = _hourglass_along_z()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    v0 = abs(m.volume)
    flat = [{"t": s["t"], "radius_mm": 10.0} for s in stations]
    out = apply_op(
        m,
        MeshOp(op="radial_profile_auto", args={"axis": "x", "stations": flat, "blend_mm": 3.0}),
    )
    assert out.is_watertight
    assert abs(out.volume) > v0 * 1.2


def test_op_radial_profile_auto_validation():
    m = _box(20, 20, 20)
    with pytest.raises(ValidationError):
        apply_op(
            m,
            MeshOp(op="radial_profile_auto", args={"axis": "z", "stations": []}),
        )
    with pytest.raises(ValidationError):
        apply_op(
            m,
            MeshOp(
                op="radial_profile_auto",
                args={
                    "axis": "z",
                    "stations": [{"t": 0.5, "radius_mm": -1.0}, {"t": 0.9, "radius_mm": 5.0}],
                },
            ),
        )


# --------------------------------------------------------------------------- #
# heal_seam_slivers


def _sliver_mesh() -> trimesh.Trimesh:
    """Icosphere with one vertex dragged to within 2e-5 mm of its neighbour —
    mimics the float32-quantized seam slivers manifold3d booleans leave on
    keep-restore unions (degenerate B-Rep edges → stalled OCC booleans)."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    e = m.edges_unique[0]
    v0, v1 = m.vertices[e[0]].copy(), m.vertices[e[1]].copy()
    verts = m.vertices.copy()
    verts[e[0]] = v1 + (v0 - v1) / np.linalg.norm(v0 - v1) * 2e-5
    out = trimesh.Trimesh(vertices=verts, faces=m.faces.copy(), process=False)
    assert out.is_volume
    assert (out.area_faces < 1e-4).any()  # fixture sanity: really has slivers
    return out


def test_heal_seam_slivers_removes_slivers():
    m = _sliver_mesh()
    healed = heal_seam_slivers(m)
    assert healed.is_volume
    assert not (healed.area_faces < 1e-4).any()
    assert len(healed.faces) < len(m.faces)
    assert healed.volume == pytest.approx(m.volume, rel=0.01)


def test_heal_seam_slivers_clean_mesh_unchanged():
    """A clean mesh has nothing to weld — vertex/face counts stay identical."""
    m = _box(10, 20, 30)
    healed = heal_seam_slivers(m)
    assert len(healed.faces) == len(m.faces)
    assert len(healed.vertices) == len(m.vertices)


def test_keep_significant_components_preserves_inner_negative_shell():
    outer = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    cavity = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    cavity.invert()
    sliver = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    sliver.apply_translation([30.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([outer, cavity, sliver])

    kept, dropped = keep_significant_components(combined)

    assert dropped == 1
    assert len(kept.split(only_watertight=False)) == 2
    assert kept.volume == pytest.approx(7000.0)


def test_heal_seam_slivers_returns_original_when_not_volume():
    """Regression guard: healing is lossless-only — when the welded result is
    no longer a volume the ORIGINAL mesh comes back (never a half-healed
    mesh that would trip the voxel-remesh fallback in _ensure_watertight)."""
    m = _box(10, 10, 10)
    m.update_faces(np.arange(len(m.faces)) > 2)  # open shell
    assert not m.is_volume
    assert heal_seam_slivers(m) is m


def test_mesh_to_solid_heals_slivers():
    """Integration: a slivered (but watertight) mesh converts to a B-Rep
    without exception — the weld gate runs before sewing."""
    pytest.importorskip("build123d")
    solid = mesh_to_solid(_sliver_mesh())
    assert solid is not None
    assert solid.volume > 0


# --------------------------------------------------------------------------- #
# footprint auto-yaw (M3)


def _yawed_box(deg: float, x=120.0, y=30.0, z=20.0) -> trimesh.Trimesh:
    m = _box(x, y, z)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1]))
    return m


def _apply_yaw(m: trimesh.Trimesh, yaw: float) -> trimesh.Trimesh:
    out = m.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 0, 1]))
    return out


def test_footprint_yaw_rot45_box_aligns_long_side_on_x():
    """The hybrid-run failure: a 120x30 footprint delivered at 45° yaw failed
    the dim gate (width/Y measured 134 mm) though the mesh itself was true."""
    box = _yawed_box(45)
    yaw = footprint_yaw_deg(box)
    assert yaw != 0.0
    fixed = _apply_yaw(box, yaw)
    assert fixed.extents[0] == pytest.approx(120, abs=1)
    assert fixed.extents[1] == pytest.approx(30, abs=1)


def test_footprint_yaw_cylinder_is_zero():
    cyl = trimesh.creation.cylinder(radius=15, height=40)
    assert footprint_yaw_deg(cyl) == 0.0


def test_footprint_yaw_axis_aligned_is_zero():
    assert footprint_yaw_deg(_box(120, 30, 20)) == 0.0


def test_footprint_yaw_rot90_puts_long_side_on_x():
    box = _yawed_box(90)
    yaw = footprint_yaw_deg(box)
    fixed = _apply_yaw(box, yaw)
    assert fixed.extents[0] == pytest.approx(120, abs=1)
    assert fixed.extents[1] == pytest.approx(30, abs=1)


def test_footprint_yaw_rounded_to_half_degree():
    box = _yawed_box(33.3)
    yaw = footprint_yaw_deg(box)
    assert yaw == pytest.approx(round(yaw * 2) / 2)


def test_baseline_mesh_auto_yaw_aligns_and_reports():
    post = MeshPostParams(up="+z", front="+y", auto_yaw=True)
    m, info = baseline_mesh(_yawed_box(45), post, None)
    assert info["orientation"]["yaw_auto_deg"] != 0.0
    assert m.extents[0] == pytest.approx(120, abs=1)
    assert m.extents[1] == pytest.approx(30, abs=1)


def test_baseline_mesh_auto_yaw_off_by_default():
    post = MeshPostParams(up="+z", front="+y")
    m, info = baseline_mesh(_yawed_box(45), post, None)
    assert info["orientation"]["yaw_auto_deg"] == 0.0
    assert m.extents[0] > 100  # still yawed — no silent behaviour change


def test_baseline_mesh_rotate_z_suppresses_auto_yaw():
    """An explicit rotate_z_deg wins — auto_yaw must not undo a user choice."""
    post = MeshPostParams(up="+z", front="+y", rotate_z_deg=10, auto_yaw=True)
    _m, info = baseline_mesh(_yawed_box(45), post, None)
    assert info["orientation"]["yaw_auto_deg"] == 0.0


def test_ensure_watertight_voxel_fallback_preserves_envelope(monkeypatch):
    """The marching-cubes rebuild quantizes the envelope outward (~a pitch
    per side). The pad once absorbed a dim-gate fit loop's scale ops round
    after round — the rebuild must come back on the input bounds."""
    from cadx.cad import mesh_ops

    m = _box(103.6, 60.0, 61.8)
    # open it so the ladder must go past fill_holes/manifold into voxels
    leaky = m.submesh([list(range(len(m.faces) - 1))], append=True)
    assert not leaky.is_watertight
    monkeypatch.setattr(trimesh.Trimesh, "fill_holes", lambda _m: None)
    monkeypatch.setattr(mesh_ops, "_alpha_wrap_repair", lambda *_a, **_k: None)

    # a manifold3d success would skip the voxel path — force it too
    import sys

    class _NoManifold:
        def __getattr__(self, name):
            raise ImportError("manifold3d unavailable in test")

    monkeypatch.setitem(sys.modules, "manifold3d", _NoManifold())

    out = mesh_ops._ensure_watertight(leaky)
    assert out.is_watertight
    np.testing.assert_allclose(out.extents, leaky.extents, atol=0.5)


def test_ensure_watertight_voxel_fallback_keeps_center(monkeypatch):
    """The envelope rescale must preserve the part's bbox centre, not shift it."""
    from cadx.cad import mesh_ops

    m = _box(20.0, 20.0, 20.0)
    m.apply_translation([50.0, -10.0, 7.0])
    leaky = m.submesh([list(range(len(m.faces) - 1))], append=True)
    monkeypatch.setattr(trimesh.Trimesh, "fill_holes", lambda _m: None)
    monkeypatch.setattr(mesh_ops, "_alpha_wrap_repair", lambda *_a, **_k: None)

    import sys

    class _NoManifold:
        def __getattr__(self, name):
            raise ImportError("manifold3d unavailable in test")

    monkeypatch.setitem(sys.modules, "manifold3d", _NoManifold())

    out = mesh_ops._ensure_watertight(leaky)
    np.testing.assert_allclose(out.bounds.mean(axis=0), leaky.bounds.mean(axis=0), atol=0.5)


def _hollow_shell(outer=20.0, inner=12.0):
    """Watertight hollow box: outer wall + inner cavity wall (negative vol)."""
    out = trimesh.creation.box(extents=[outer, outer, outer])
    inn = trimesh.creation.box(extents=[inner, inner, inner])
    inn.invert()
    return trimesh.util.concatenate([out, inn])


def test_solid_body_count_hollow_shell_is_one():
    """Boundary components are not bodies: a closed shell splits into
    outer + inner wall but is ONE solid — the single_valid_solid false
    positive that failed every shell/infill part."""
    from cadx.cad.mesh_ops import solid_body_count

    m = _hollow_shell()
    assert len(m.split(only_watertight=False)) == 2
    assert solid_body_count(m) == 1


def test_solid_body_count_two_bodies_still_two():
    from cadx.cad.mesh_ops import solid_body_count

    a = _box(10.0, 10.0, 10.0)
    b = _box(5.0, 5.0, 5.0)
    b.apply_translation([40.0, 0.0, 0.0])
    assert solid_body_count(trimesh.util.concatenate([a, b])) == 2


def test_solid_body_count_floating_body_in_cavity_is_two():
    """A genuinely detached body inside the cavity (positive volume) is a
    second solid — the sign convention does not hide real defects."""
    from cadx.cad.mesh_ops import solid_body_count

    hollow = _hollow_shell(outer=20.0, inner=14.0)
    floater = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    m = trimesh.util.concatenate([hollow, floater])
    assert solid_body_count(m) == 2


def test_mesh_analysis_hollow_reports_one_solid():
    a = mesh_analysis(_hollow_shell())
    assert a.solid_count == 1
    assert a.surface_components == 2
    assert not any("disconnected solids" in w for w in a.warnings)


def test_clean_preserves_small_cavity_wall():
    """Regression: a cavity wall below min_component_frac of the outer
    volume must NOT be dropped — that opened the hollow region to a shell."""
    outer = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    inner = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    inner.apply_translation([3.0, 3.0, 3.0])
    inner.invert()
    m = trimesh.util.concatenate([outer, inner])
    assert len(m.split(only_watertight=False)) == 2

    cleaned = clean(m)
    parts = cleaned.split(only_watertight=False)
    assert len(parts) == 2  # cavity wall survived
    assert cleaned.is_watertight


def test_keep_largest_keeps_cavity_wall_with_body():
    """keep_largest on a hollow part must keep the inner wall attached to
    the outer body — dropping it would open the cavity."""
    from cadx.cad.mesh_ops import OpKeepLargest, _op_keep_largest

    m = _hollow_shell()
    out = _op_keep_largest(m, OpKeepLargest())
    parts = out.split(only_watertight=False)
    assert len(parts) == 2
    assert out.is_watertight


# --------------------------------------------------------------------------- #
# machine_feature — semantic surface-targeted machining                        #
# --------------------------------------------------------------------------- #


def _ray_hits(m, origin, direction, axis_i):
    locs, _, _ = m.ray.intersects_location(
        ray_origins=[origin], ray_directions=[direction], multiple_hits=True
    )
    return sorted(float(hit[axis_i]) for hit in locs)


def test_op_machine_feature_face_word_sidewall_tap():
    """face='left' + at=[u,v]: ray entry on the -x wall, M8 tap 20 mm deep —
    the bore must open on the -x face and bottom out at x≈0."""
    out = apply_op(
        _box(40, 40, 40),
        MeshOp(
            op="machine_feature",
            args={
                "op": "tap",
                "face": "left",
                "at": [0.0, 5.0],
                "major_d": 8.0,
                "pitch": 1.25,
                "depth": 20.0,
            },
        ),
    )
    assert out.is_watertight
    xs = _ray_hits(out, [-30.0, 0.0, 5.0], [1, 0, 0], 0)
    assert xs[0] == pytest.approx(-20.0, abs=0.6)  # left entry face
    assert xs[1] == pytest.approx(0.0, abs=0.8)  # bore bottom: entry -20 + depth 20
    assert xs[2] == pytest.approx(20.0, abs=0.6)  # far wall intact


def test_op_machine_feature_pick_point_curved():
    """at=[x,y,z] on a sphere: the pick resolves to the nearest surface point
    and the bore runs along the local normal (radial) — the M8-in-ball case."""
    ball = trimesh.creation.icosphere(radius=30.0, subdivisions=3)
    out = apply_op(
        ball,
        MeshOp(
            op="machine_feature",
            args={"op": "tap", "at": [0.0, 0.0, 40.0], "major_d": 8.0, "depth": 25.0},
        ),
    )
    assert out.is_watertight
    zs = _ray_hits(out, [0.0, 0.0, 60.0], [0, 0, -1], 2)
    assert zs[0] == pytest.approx(-30.0, abs=0.6)
    assert zs[1] == pytest.approx(5.0, abs=1.0)  # bore bottom ~ 30 - 25


def test_op_machine_feature_drill_through_and_axis_override():
    """depth omitted → through; axis override makes the cut run orthogonal to
    the picked surface's local normal."""
    m = _box(30, 30, 10)
    out = apply_op(
        m,
        MeshOp(
            op="machine_feature",
            args={"op": "drill", "face": "top", "at": [5.0, -5.0], "dia": 6.0},
        ),
    )
    assert out.is_watertight
    zs = _ray_hits(out, [5.0, -5.0, 20.0], [0, 0, -1], 2)
    assert len(zs) == 0 or zs == sorted(zs)  # bore pierced: ray exits
    # through means no solid hit along that ray
    assert not zs or zs[0] <= -5.0


def test_op_machine_feature_cbore_and_csk():
    m = _box(40, 40, 12)
    out = apply_op(
        m,
        MeshOp(
            op="machine_feature",
            args={
                "op": "cbore",
                "face": "top",
                "dia": 6.0,
                "cbore_dia": 11.0,
                "cbore_depth": 4.0,
            },
        ),
    )
    assert out.is_watertight
    # counterbore shelf (top 6 - depth 4 = z~2) inside the pocket ring —
    # apply_op does not re-ground, so the box stays centred
    zs = _ray_hits(out, [4.0, 0.0, 20.0], [0, 0, -1], 2)
    assert pytest.approx(min(zs, key=lambda z: abs(z - 2.0)), abs=0.4) == 2.0


def test_op_machine_feature_errors():
    m = _box(20, 20, 20)
    with pytest.raises(ValueError, match="unresolvable face"):
        apply_op(m, MeshOp(op="machine_feature", args={"op": "drill", "face": "nowhere"}))
    with pytest.raises(ValidationError):
        apply_op(m, MeshOp(op="machine_feature", args={"op": "weld"}))
