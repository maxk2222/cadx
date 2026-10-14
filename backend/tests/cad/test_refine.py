"""Post-TopOpt feature finishing — cutter mapping, anchor/spec resolution,
keep-region restore, boolean robustness (`cad/fem/refine.py`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.refine import (
    RefineFeature,
    _cutter_mesh,
    _entry_point,
    _metric_pitch,
    _parse_spec,
    _thread_spec_from,
    apply_refine,
    resolve_refine_features,
    restore_keep_surface,
)


def _box(x: float = 60.0, y: float = 30.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _anchors() -> list[dict]:
    return [
        {
            "index": 1,
            "kind": "hole",
            "axis": "+z",
            "pos": [10.0, 0.0, 6.0],
            "x": 10.0,
            "y": 0.0,
            "dia": 5.0,
            "depth_mm": None,
        },
        {
            "index": 2,
            "kind": "hole",
            "axis": "+y",
            "pos": [0.0, 15.0, 6.0],
            "dia": 4.0,
            "depth_mm": None,
        },
    ]


# --------------------------------------------------------------------------- #
# spec parsing / cutter mapping                                                 #
# --------------------------------------------------------------------------- #


def test_parse_spec_metric():
    assert _parse_spec("M4-0.7") == (4.0, 0.7)
    assert _parse_spec("m3x0.5-6") == (3.0, 0.5)
    assert _parse_spec("M10") is None
    assert _parse_spec("ISO 4762") is None


def test_thread_spec_resolution():
    assert _thread_spec_from({"standard": "M4 ISO 10642"}, 5.0) == "M4-0.7"
    assert _thread_spec_from({"name": "M3 insert"}, 0) == "M3-0.5"
    assert _thread_spec_from({"standard": ""}, 0) == ""
    # dia-only fallback picks nearest coarse size within tolerance
    assert _thread_spec_from({}, 4.0) == "M4-0.7"


def test_metric_pitch_table():
    assert _metric_pitch(4.0) == 0.7
    assert _metric_pitch(8.0) == 1.25


def test_thread_cutter_uses_tap_drill():
    """A printed M4 thread is cut at D - pitch = 3.3 mm, not nominal 4.0."""
    feat = RefineFeature(kind="thread", dia=4.0, thread_spec="M4-0.7", depth_mm=8.0)
    c = _cutter_mesh(feat, entry=0.0, span=12.0)
    # cutter max radial extent ~ tap drill radius (3.3/2 = 1.65)
    r = np.linalg.norm(c.vertices[:, :2], axis=1).max()
    assert r == pytest.approx(1.65, abs=0.05)


def test_counterbore_cutter_two_diameters():
    feat = RefineFeature(kind="counterbore", dia=4.0, cbore_dia=8.0, cbore_depth_mm=3.0)
    c = _cutter_mesh(feat, entry=0.0, span=12.0)
    r = np.linalg.norm(c.vertices[:, :2], axis=1).max()
    assert r == pytest.approx(4.0, abs=0.05)  # counterbore radius present


def test_plain_hole_cutter():
    feat = RefineFeature(kind="hole", dia=6.0)
    c = _cutter_mesh(feat, entry=0.0, span=12.0)
    r = np.linalg.norm(c.vertices[:, :2], axis=1).max()
    assert r == pytest.approx(3.0, abs=0.05)


# --------------------------------------------------------------------------- #
# resolution: request > spec > anchors                                          #
# --------------------------------------------------------------------------- #


def test_resolve_auto_from_anchors():
    feats, warns = resolve_refine_features(_anchors(), None, None)
    assert not warns
    assert len(feats) == 2
    by_axis = {f.axis: f for f in feats}
    assert by_axis["+z"].dia == 5.0 and by_axis["+z"].u == 10.0
    assert by_axis["+y"].dia == 4.0


def test_resolve_spec_upgrades_anchor_kind():
    feats, _ = resolve_refine_features(
        _anchors(),
        [
            {
                "name": "set screw",
                "type": "thread",
                "dims": {"diameter": 4.0},
                "axis": "y",
                "standard": "M4",
            }
        ],
        None,
    )
    thr = [f for f in feats if f.kind == "thread"]
    assert len(thr) == 1
    assert thr[0].axis == "+y"
    assert thr[0].thread_spec == "M4-0.7"
    assert thr[0].source == "spec"


def test_resolve_request_anchor_index():
    feats, _ = resolve_refine_features(
        _anchors(), None, [{"anchor_index": 2, "kind": "counterbore", "cbore_dia": 8.0}]
    )
    assert len(feats) == 1
    assert feats[0].kind == "counterbore"
    assert feats[0].cbore_dia == 8.0
    assert feats[0].axis == "+y"


def test_resolve_request_missing_position_warns():
    feats, warns = resolve_refine_features([], None, [{"kind": "hole", "dia": 5.0}])
    assert not feats
    assert any("no position" in w for w in warns)


# --------------------------------------------------------------------------- #
# apply_refine end-to-end                                                       #
# --------------------------------------------------------------------------- #


def test_apply_refine_cuts_through_hole():
    mesh = _box()
    feats = [RefineFeature(name="b1", kind="hole", axis="+z", u=10.0, v=0.0, dia=6.0)]
    out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert len(rep.applied) == 1 and not rep.skipped
    v0, v1 = mesh.volume, out.volume
    expected = math.pi * 3.0**2 * 12.0
    assert (v0 - v1) == pytest.approx(expected, rel=0.08)
    assert out.is_watertight


def test_apply_refine_side_axis():
    mesh = _box()
    feats = [RefineFeature(name="s", kind="hole", axis="+y", u=0.0, v=6.0, dia=4.0)]
    out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert len(rep.applied) == 1
    # hole through the 30 mm Y wall
    expected = math.pi * 2.0**2 * 30.0
    assert (mesh.volume - out.volume) == pytest.approx(expected, rel=0.08)


def test_apply_refine_skips_out_of_bounds_feature():
    mesh = _box()
    feats = [RefineFeature(name="far", kind="hole", axis="+z", u=500.0, v=0.0, dia=4.0)]
    out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert not rep.applied
    assert rep.skipped and "misses the part" in rep.skipped[0]["reason"]
    assert out.volume == pytest.approx(mesh.volume)


def test_apply_refine_recuts_open_channel():
    """A feature over an EXISTING hole (no entry surface — ray passes through)
    must still cut: reaming the wavy voxel tunnel is refine's core job."""
    mesh = _box()
    feats = [RefineFeature(name="b", kind="bore", axis="+z", u=0.0, v=0.0, dia=8.0)]
    out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert rep.applied  # now the hole exists — second pass has no entry surface
    feats2 = [RefineFeature(name="b2", kind="bore", axis="+z", u=0.0, v=0.0, dia=6.0)]
    out2, rep2 = apply_refine(out, feats2, restore_keeps=False)
    assert rep2.applied  # falls back to the bbox face, cleans the tunnel walls
    assert any("no entry surface" in w for w in rep2.warnings)
    assert out2.is_watertight


def test_apply_refine_rejects_noncuttable():
    mesh = _box()
    feats = [RefineFeature(name="boss", kind="boss", axis="+z", u=0.0, v=0.0, dia=8.0)]
    _, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert rep.skipped[0]["reason"] == "not cuttable"


def test_entry_point_top_and_side():
    mesh = _box()
    assert _entry_point(mesh, "+z", 0.0, 0.0) == pytest.approx(12.0, abs=0.05)
    assert _entry_point(mesh, "-z", 0.0, 0.0) == pytest.approx(0.0, abs=0.05)
    assert _entry_point(mesh, "+z", 999.0, 0.0) is None


# --------------------------------------------------------------------------- #
# keep-region restore                                                           #
# --------------------------------------------------------------------------- #


def test_restore_keep_surface_unions_shrunk_wall():
    """Optimized mesh lost a corner chunk; keep region restores it from base."""
    base = _box()
    # 'optimized' = base minus a corner notch inside the keep region
    notch = trimesh.creation.box(extents=[10.0, 10.0, 12.0])
    notch.apply_translation([25.0, 10.0, 6.0])
    opt = base.difference(notch, engine="manifold")
    assert opt.volume < base.volume

    keeps = [{"kind": "box", "center": [25.0, 10.0, 6.0], "diameter_mm": 12.0}]
    out, warns, n = restore_keep_surface(opt, base, keeps, [])
    assert not warns
    assert n == 1
    # restored volume ≈ base again (overlap is idempotent)
    assert out.volume > opt.volume
    assert out.volume <= base.volume * 1.02


def test_restore_anchor_keep_box():
    base = _box()
    anchors = _anchors()
    opt = base.difference(
        trimesh.creation.cylinder(radius=4.0, height=20, sections=32).apply_translation(
            [10.0, 0.0, 10.0]
        ),
        engine="manifold",
    )
    keeps = [{"kind": "anchor", "hole_index": 1, "margin_mm": 2.0}]
    out, _, n = restore_keep_surface(opt, base, keeps, anchors)
    assert n == 1
    assert out.volume > opt.volume
