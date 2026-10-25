"""Post-TopOpt feature finishing — cutter mapping, anchor/spec resolution,
keep-region restore, boolean robustness (`cad/fem/refine.py`)."""

from __future__ import annotations

import contextlib
import math

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.anchors3d import scan_mesh_anchors
from cadx.cad.fem.refine import (
    RefineFeature,
    _bore_ring_material,
    _channel_blocked_mm,
    _cutter_mesh,
    _entry_point,
    _feature_host_main,
    _metric_pitch,
    _parse_spec,
    _spec_axis,
    _spec_uv_positions,
    _thread_spec_from,
    _through_cut_depth,
    _tunnel_open_mesh,
    _verify_expected_uv,
    _verify_uv,
    apply_refine,
    refine_verify,
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
    # dia-only fallback picks nearest coarse size within tolerance — but
    # ONLY for thread-typed features (a Ø5.5 clearance hole is the standard
    # M5 fit and must not earn an M5 spec by proximity alone)
    assert _thread_spec_from({}, 4.0, allow_dia_snap=True) == "M4-0.7"
    assert _thread_spec_from({}, 4.0) == ""
    assert _thread_spec_from({"name": "Ø5.5 Durchgangsloch"}, 5.5) == ""


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
# base-mesh tunnel gate on leftover anchors                                     #
# --------------------------------------------------------------------------- #


def _drilled_base() -> trimesh.Trimesh:
    """60x30x12 base with a real Ø5 through-bore at (+z, u=10, v=0)."""
    bore = trimesh.creation.cylinder(radius=2.5, height=20.0, sections=32)
    bore.apply_translation([10.0, 0.0, 6.0])
    return _box().difference(bore, engine="manifold")


def _dimpled_base() -> trimesh.Trimesh:
    """Base with a shallow 4 mm pocket at (+z, u=10, v=0) — a dimple, not a
    through-feature (scanners report organic micro-voids as 'hole')."""
    pocket = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=32)
    pocket.apply_translation([10.0, 0.0, 10.0])
    return _box().difference(pocket, engine="manifold")


def test_spec_matched_anchor_cuts_without_base_tunnel():
    """Spec-declared features are authoritative — a spec feature claiming a
    scanned anchor re-cuts even when the base mesh has no tunnel there."""
    feats, warns = resolve_refine_features(
        [_anchors()[0]],
        [{"name": "mount", "type": "hole", "dims": {"diameter": 5.0}, "axis": "z"}],
        None,
        base_mesh=_box(),
    )
    assert len(feats) == 1
    assert feats[0].source == "spec"
    assert not any("without a base-mesh tunnel" in w for w in warns)


def test_leftover_anchor_with_base_tunnel_kept():
    """Scan-only anchor over a REAL base-mesh bore stays in the cutter list."""
    feats, warns = resolve_refine_features([_anchors()[0]], None, None, base_mesh=_drilled_base())
    assert len(feats) == 1
    assert feats[0].u == 10.0 and feats[0].source == "anchor"
    assert not any("without a base-mesh tunnel" in w for w in warns)


def test_leftover_anchor_over_solid_base_skipped():
    """Junk scanned anchor over solid material must not be auto-cut."""
    feats, warns = resolve_refine_features([_anchors()[0]], None, None, base_mesh=_box())
    assert not feats
    assert any("without a base-mesh tunnel" in w for w in warns)


def test_leftover_anchor_over_dimple_skipped():
    """A pocket/dimple blocks nearly the whole ray span — not a tunnel."""
    feats, warns = resolve_refine_features([_anchors()[0]], None, None, base_mesh=_dimpled_base())
    assert not feats
    assert any("without a base-mesh tunnel" in w for w in warns)


def test_leftover_anchors_no_base_mesh_keeps_legacy():
    """No base mesh → every scanned anchor is re-cut (standalone refine)."""
    feats, _ = resolve_refine_features(_anchors(), None, None, base_mesh=None)
    assert len(feats) == 2


def test_refine_end_to_end_drops_junk_void():
    """Opt mesh without any feature: the tunnel-backed anchor is cut, the
    junk void anchor is dropped before a cutter is ever built."""
    junk = dict(_anchors()[0])
    junk.update({"index": 3, "pos": [-15.0, 0.0, 6.0], "x": -15.0, "dia": 4.0})
    feats, warns = resolve_refine_features(
        [_anchors()[0], junk], None, None, base_mesh=_drilled_base()
    )
    assert len(feats) == 1
    assert any("without a base-mesh tunnel" in w for w in warns)
    mesh = _box()
    out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert len(rep.applied) == 1
    assert rep.applied[0]["name"] == feats[0].name
    # only the real Ø5 bore was cut
    assert (mesh.volume - out.volume) == pytest.approx(math.pi * 2.5**2 * 12.0, rel=0.08)


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
    assert rep.skipped and "no material" in rep.skipped[0]["reason"]
    assert out.volume == pytest.approx(mesh.volume)


def test_apply_refine_recuts_open_channel():
    """A feature over an EXISTING hole (no entry surface — ray passes through)
    must still cut: reaming the wavy voxel tunnel is refine's core job. The
    phase-1 plug seals the measured channel, so the drill finds a real
    entry on the plug face and re-bores at the nominal diameter."""
    mesh = _box()
    feats = [RefineFeature(name="b", kind="bore", axis="+z", u=0.0, v=0.0, dia=8.0)]
    out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert rep.applied  # now the hole exists — second pass has no entry surface
    feats2 = [RefineFeature(name="b2", kind="bore", axis="+z", u=0.0, v=0.0, dia=6.0)]
    out2, rep2 = apply_refine(out, feats2, restore_keeps=False)
    assert rep2.applied  # plugged, then drilled at the declared position
    assert out2.is_watertight
    assert _channel_blocked_mm(out2, 0.0, 0.0, "+z") <= 0.3


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


# --------------------------------------------------------------------------- #
# keep-prism geometry — round holes restore round pads, never square blocks   #
# (the robot-arm-mount eval pasted `base ∩ AABB` — visible cubes on render)  #
# --------------------------------------------------------------------------- #


def test_anchor_keep_prism_is_cylinder_not_box():
    from cadx.cad.fem.refine import _anchor_keep_prism

    a = {"index": 1, "axis": "+z", "pos": [10.0, 0.0, 6.0], "x": 10.0, "y": 0.0, "dia": 8.5}
    p = _anchor_keep_prism(a, margin=3.0, span_axis=(0.0, 10.0))
    assert p["shape"] == "cyl"
    assert p["ax"] == 2
    assert p["r"] == pytest.approx(8.5 / 2 + 3.0)


def test_anchor_keep_prism_pad_dia_keeps_whole_boss():
    """A bore drilled through a Ø25 boss keeps the whole pad — the ring-only
    keep let TopOpt eat the boss and left a useless sleeve around the M12."""
    from cadx.cad.fem.refine import _anchor_keep_prism

    a = {
        "index": 1,
        "axis": "+z",
        "pos": [0.0, 0.0, 12.5],
        "x": 0.0,
        "y": 0.0,
        "dia": 10.2,
        "pad_dia": 25.0,
    }
    p = _anchor_keep_prism(a, margin=3.0, span_axis=(10.0, 25.0))
    assert p["r"] == pytest.approx(25.0 / 2 + 3.0)


def test_prism_mesh_side_axis_is_cylinder_along_x():
    """+x flange bores rotate the cylinder to the bore axis — extent along X
    is the span, the Y/Z cross-section stays round."""
    from cadx.cad.fem.refine import _prism_mesh

    p = {"shape": "cyl", "ax": 0, "uv": (12.0, 25.0), "r": 5.25, "span": (42.0, 50.0)}
    m = _prism_mesh(p)
    ex = m.bounds[1] - m.bounds[0]
    assert ex[0] == pytest.approx(8.0, abs=0.3)
    assert ex[1] == pytest.approx(10.5, abs=0.5)
    assert ex[2] == pytest.approx(10.5, abs=0.5)
    assert m.bounds[0][1] == pytest.approx(12.0 - 5.25, abs=0.5)
    assert m.bounds[0][2] == pytest.approx(25.0 - 5.25, abs=0.5)


def test_restore_anchor_keep_restores_round_pad_not_square():
    """End-to-end on the eval defect: a keep region over a round hole must
    restore a CYLINDER pad — the old `base ∩ AABB` pasted square columns
    around every hole (the blocks on the mount render). Probe the square
    corner (inside the old box, outside the cylinder) vs inside the pad."""
    base = _box()
    anchors = _anchors()
    opt = base.difference(
        trimesh.creation.cylinder(radius=2.0, height=20, sections=32).apply_translation(
            [10.0, 0.0, 10.0]
        ),
        engine="manifold",
    )
    keeps = [{"kind": "anchor", "hole_index": 1, "margin_mm": 2.0}]
    out, _, n = restore_keep_surface(opt, base, keeps, anchors, margin_mm=0.0)
    assert n == 1
    # corner probe: (3.4, 3.4) off-centre is at radius 4.8 > pad r 4.5 —
    # inside the old square pad, outside the cylinder. Must hold NO material.
    corner = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    corner.apply_translation([10.0 + 3.4, 0.0 + 3.4, 9.0])
    with contextlib.suppress(Exception):
        assert float(out.intersection(corner, engine="manifold").volume or 0) < 0.4
    # inside probe: 1 mm off-centre — inside the pad and inside the notch,
    # so the restore must fill it back to base material.
    inner = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    inner.apply_translation([11.0, 0.0, 9.0])
    vol = float(out.intersection(inner, engine="manifold").volume or 0)
    assert vol > 3.0


# --------------------------------------------------------------------------- #
# re-cut snap / dedup — the "double hole" defect                                #
# --------------------------------------------------------------------------- #


def _wide_box(x: float = 100.0, y: float = 30.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _bored_wide(x_center: float, dia: float) -> trimesh.Trimesh:
    """100x30x12 block with a vertical through-bore at (x_center, 0)."""
    bore = trimesh.creation.cylinder(radius=dia / 2.0, height=40.0, sections=48)
    bore.apply_translation([x_center, 0.0, 6.0])
    return _wide_box().difference(bore, engine="manifold")


def _z_tunnels_near(mesh: trimesh.Trimesh, x: float, radius: float = 4.0) -> list[dict]:
    return [
        a
        for a in scan_mesh_anchors(mesh, step_mm=0.5)
        if a["axis"] == "+z" and abs(a["x"] - x) <= radius
    ]


def test_recut_snaps_to_existing_tunnel():
    """DOUBLE-HOLE regression: the bore sits at x=-35.13 but the anchor
    measured on the smoothed mesh reports x=-35.16 — the re-cut must snap
    onto the measured tunnel centre, plug the wavy channel and drill ONE
    clean hole at the anchor diameter."""
    mesh = _bored_wide(-35.13, 6.8)
    feat = RefineFeature(name="b", kind="hole", axis="+z", u=-35.16, v=0.0, dia=6.75)
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    assert rep.applied[0].get("plugged") is True

    tunnels = _z_tunnels_near(out, -35.15)
    assert len(tunnels) == 1  # no figure-8 / sliver second void
    assert tunnels[0]["x"] == pytest.approx(-35.15, abs=0.5)
    # plugged then drilled at the anchor's 6.75 — not a wavy +0.4 ream
    assert tunnels[0]["dia"] == pytest.approx(6.75, abs=0.7)


def test_recut_snaps_via_base_mesh():
    """Same snap when the tunnel only exists on the base mesh (keep-restore
    disabled): the cut lands on the BASE tunnel centre, not the anchor."""
    base = _bored_wide(-35.13, 6.8)
    mesh = _wide_box()  # optimized part came back solid there
    feat = RefineFeature(name="b", kind="hole", axis="+z", u=-35.16, v=0.0, dia=6.75)
    out, rep = apply_refine(mesh, [feat], base_mesh=base, restore_keeps=False)
    assert len(rep.applied) == 1
    tunnels = _z_tunnels_near(out, -35.15)
    assert len(tunnels) == 1
    assert tunnels[0]["x"] == pytest.approx(-35.15, abs=0.5)


def test_recut_spec_dia_plugs_then_drills_nominal():
    """A spec-diameter bore over an oversized wavy channel comes out at the
    SPEC diameter — the measured void is plugged, then drilled nominal. A
    plain difference can't shrink a void, so without the plug the cutter
    reams +0.4 past every wall bulge (bail handle: spec 5.5 cut at 5.9)."""
    mesh = _bored_wide(-35.13, 6.8)  # oversized ~6.8 channel where the bore belongs
    feat = RefineFeature(name="m5", kind="hole", axis="+z", u=-35.16, v=0.0, dia=5.5, source="spec")
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    assert rep.applied[0].get("plugged") is True
    tunnels = _z_tunnels_near(out, -35.15)
    assert len(tunnels) == 1
    assert tunnels[0]["dia"] == pytest.approx(5.5, abs=0.7)


def test_recut_non_spec_plugs_real_mesh_tunnel():
    """Scanned anchors over a real mesh tunnel also plug+drill — the wavy
    channel gets sealed and re-drilled at the anchor diameter instead of a
    +0.4 ream that keeps degraded walls."""
    mesh = _bored_wide(-35.13, 6.8)
    feat = RefineFeature(name="b", kind="hole", axis="+z", u=-35.16, v=0.0, dia=6.75)
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    assert rep.applied[0].get("plugged") is True
    tunnels = _z_tunnels_near(out, -35.15)
    assert len(tunnels) == 1
    assert tunnels[0]["dia"] == pytest.approx(6.75, abs=0.7)


def test_recut_spec_never_snaps_to_drifted_channel():
    """A spec'd bore whose channel drifted >1mm keeps the DECLARED position:
    the drifted void is plugged at its measured centre, the drill cuts at
    (u, v) — never snapped onto the measurement."""
    mesh = _bored_wide(-34.0, 6.8)  # channel ~1.2mm off the declared x=-35.16
    feat = RefineFeature(name="m5", kind="hole", axis="+z", u=-35.16, v=0.0, dia=5.5, source="spec")
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    assert rep.applied[0].get("plugged") is True
    # the declared bore emerges at x≈-35.2 — the drifted -34 void is gone
    tunnels = _z_tunnels_near(out, -35.16)
    assert len(tunnels) == 1
    assert tunnels[0]["dia"] == pytest.approx(5.5, abs=0.7)
    assert not _z_tunnels_near(out, -34.0, radius=0.8)


def test_recut_non_spec_large_drift_plugs_channel_keeps_position():
    """Non-spec anchor with a >1mm drifted channel: plug at the measured
    centre, drill at the declared spot — same never-snap rule as spec."""
    mesh = _bored_wide(-34.0, 6.8)
    feat = RefineFeature(name="b", kind="hole", axis="+z", u=-35.16, v=0.0, dia=6.75)
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    assert rep.applied[0].get("plugged") is True
    tunnels = _z_tunnels_near(out, -35.16)
    assert len(tunnels) == 1
    assert tunnels[0]["dia"] == pytest.approx(6.75, abs=0.7)
    assert not _z_tunnels_near(out, -34.0, radius=0.8)


def test_dedup_merges_near_concentric_anchors():
    """Two anchors 0.5 mm apart at the same dia resolve to ONE cut."""
    mesh = _wide_box()
    feats = [
        RefineFeature(name="a", kind="hole", axis="+z", u=-35.0, v=0.0, dia=6.75),
        RefineFeature(name="b", kind="hole", axis="+z", u=-34.5, v=0.0, dia=6.75),
    ]
    out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert len(rep.applied) == 1
    assert any("merged into one cut" in w for w in rep.warnings)
    removed = mesh.volume - out.volume
    assert removed == pytest.approx(math.pi * (6.75 / 2) ** 2 * 12.0, rel=0.1)


def test_dedup_keeps_distant_features():
    """Bores beyond the merge radius are separate features — no over-merge."""
    mesh = _wide_box()
    feats = [
        RefineFeature(name="a", kind="hole", axis="+z", u=-35.0, v=0.0, dia=6.75),
        RefineFeature(name="b", kind="hole", axis="+z", u=-30.0, v=0.0, dia=6.75),
    ]
    _out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert len(rep.applied) == 2


# --------------------------------------------------------------------------- #
# helical threads — the "missing thread" defect                                 #
# --------------------------------------------------------------------------- #


def _groove_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    import cadx.cad.fem.refine as refine_mod

    calls: list[tuple] = []
    real = refine_mod._thread_groove_solid

    def spy(*args: object, **kwargs: object) -> trimesh.Trimesh:
        calls.append((args, kwargs))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(refine_mod, "_thread_groove_solid", spy)
    return calls


def test_helical_thread_grooves_by_default(monkeypatch: pytest.MonkeyPatch):
    """MISSING-THREAD regression: an M8-1.25 thread anchor re-cuts the
    tap-drill bore AND the helical V-groove — not a plain cylinder."""
    calls = _groove_spy(monkeypatch)
    mesh = _box()
    feat = RefineFeature(
        name="m8", kind="thread", axis="+z", u=10.0, v=0.0, dia=8.0, thread_spec="M8-1.25"
    )
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert calls, "helical groove solid was never built"
    assert len(rep.applied) == 1
    # groove removes more than the plain tap-drill cylinder (D - pitch = 6.75)
    tap_vol = math.pi * ((8.0 - 1.25) / 2) ** 2 * 12.0
    assert (mesh.volume - out.volume) > tap_vol * 1.02


def test_helical_false_explicit_keeps_plain_bore(monkeypatch: pytest.MonkeyPatch):
    """helical=False in the merged feature dict always wins — tap drill only."""
    calls = _groove_spy(monkeypatch)
    mesh = _box()
    feat = RefineFeature(
        name="m8",
        kind="thread",
        axis="+z",
        u=10.0,
        v=0.0,
        dia=8.0,
        thread_spec="M8-1.25",
        helical=False,
    )
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert not calls
    assert len(rep.applied) == 1
    tap_vol = math.pi * ((8.0 - 1.25) / 2) ** 2 * 12.0
    assert (mesh.volume - out.volume) == pytest.approx(tap_vol, rel=0.05)


def test_helical_default_gated_by_flag(monkeypatch: pytest.MonkeyPatch):
    """helical_threads=False (runtime refine_helical_threads off) → tap drill."""
    calls = _groove_spy(monkeypatch)
    mesh = _box()
    feat = RefineFeature(
        name="m8", kind="thread", axis="+z", u=10.0, v=0.0, dia=8.0, thread_spec="M8-1.25"
    )
    _out, rep = apply_refine(mesh, [feat], restore_keeps=False, helical_threads=False)
    assert not calls
    assert len(rep.applied) == 1


def test_groove_failure_keeps_tap_drill(monkeypatch: pytest.MonkeyPatch):
    """A groove-build failure must never lose the bore — warn and fall back."""
    import cadx.cad.fem.refine as refine_mod

    def boom(*args: object, **kwargs: object) -> trimesh.Trimesh:
        raise RuntimeError("groove wreckage")

    monkeypatch.setattr(refine_mod, "_thread_groove_solid", boom)
    mesh = _box()
    feat = RefineFeature(
        name="m8", kind="thread", axis="+z", u=10.0, v=0.0, dia=8.0, thread_spec="M8-1.25"
    )
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    tap_vol = math.pi * ((8.0 - 1.25) / 2) ** 2 * 12.0
    assert (mesh.volume - out.volume) == pytest.approx(tap_vol, rel=0.05)


# --------------------------------------------------------------------------- #
# auto keep-restore + spec-anchor deferred cuts — the SIMP-eaten-ring defect  #
# --------------------------------------------------------------------------- #


def _slab(x: float = 60.0, y: float = 40.0, z: float = 14.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _slab_bored(x0: float = 10.0, dia: float = 10.0) -> trimesh.Trimesh:
    """60x40x14 slab with a clean Ø`dia` through-bore at (x0, 0)."""
    bore = trimesh.creation.cylinder(radius=dia / 2.0, height=40.0, sections=48)
    bore.apply_translation([x0, 0.0, 7.0])
    return _slab().difference(bore, engine="manifold")


def _scooped_opt(base: trimesh.Trimesh, x0: float = 10.0, r: float = 15.0) -> trimesh.Trimesh:
    """'Optimized' variant of `base`: SIMP ate a huge void around (x0, 0) —
    bore walls and their keep rings included."""
    void = trimesh.creation.cylinder(radius=r, height=60.0, sections=48)
    void.apply_translation([x0, 0.0, 7.0])
    return base.difference(void, engine="manifold")


def _bore_anchor(idx: int = 1, *, source: str = "spec+scan", dia: float = 10.0) -> dict:
    return {
        "index": idx,
        "kind": "hole",
        "axis": "+z",
        "pos": [10.0, 0.0, 7.0],
        "x": 10.0,
        "y": 0.0,
        "dia": dia,
        "depth_mm": None,
        "source": source,
    }


def test_auto_keep_drops_floating_bore_ring():
    """SIMP ate the material ring around a Ø10 bore AND everything the
    keep box could bridge to — the auto keep patch must be dropped, not
    pasted back: a ring floating in a void still 'measures' a nominal
    bore but ships a loose fragment with zero load path."""
    base = _slab_bored(10.0, 10.0)
    opt = _scooped_opt(base, 10.0, 15.0)
    anchors = [_bore_anchor()]
    feats, _ = resolve_refine_features(anchors, None, None, base_mesh=base)
    assert len(feats) == 1

    out, rep = apply_refine(opt, feats, base_mesh=base, keep_regions=[], anchors=anchors)
    assert rep.restored_regions == 0
    assert any("would float" in w for w in rep.warnings)
    # nothing to drill into — the cut skips honestly instead of producing
    # a free puck with a perfect-looking Ø10 hole
    assert not rep.applied
    assert any("no material" in s["reason"] for s in rep.skipped)
    assert out.volume == pytest.approx(opt.volume)


def test_auto_keep_restores_partially_eaten_bore_ring():
    """The common case: SIMP ate the bore + inner walls but material survives
    inside the keep prism — the patch bridges and the bore re-cuts."""
    base = _slab_bored(10.0, 10.0)
    # scoop r6.5 < keep r8.8 (dia/2 + margin + grow): the prism rim overlaps
    # surviving slab — the cylindrical patch restores and bridges to the body
    opt = _scooped_opt(base, 10.0, 6.5)
    anchors = [_bore_anchor()]
    feats, _ = resolve_refine_features(anchors, None, None, base_mesh=base)
    assert len(feats) == 1

    out, rep = apply_refine(opt, feats, base_mesh=base, keep_regions=[], anchors=anchors)
    assert rep.restored_regions >= 1
    assert any("auto keep-restore" in w for w in rep.warnings)
    assert len(rep.applied) == 1 and not rep.skipped
    assert _tunnel_open_mesh(out, 10.0, 0.0, "+z")
    bores = [t for t in _z_tunnels_near(out, 10.0, radius=4.0) if t["dia"] <= 15.0]
    assert len(bores) == 1
    assert bores[0]["dia"] == pytest.approx(10.0, abs=1.5)


def test_spec_anchor_recut_after_auto_restore():
    """Spec-declared anchor over SOLID base (no base tunnel): resolve defers
    the verdict; the keep ring still bridges surviving material, restores,
    and the bore is cut — not dropped by the d1d1836 tunnel allowlist."""
    base = _slab()  # solid — the spec bore was never in the scan
    opt = _scooped_opt(base, 10.0, 6.5)
    anchors = [_bore_anchor(source="spec")]
    feats, _ = resolve_refine_features(anchors, None, None, base_mesh=base)
    assert len(feats) == 1
    assert feats[0].admit_on_material

    out, rep = apply_refine(opt, feats, base_mesh=base, keep_regions=[], anchors=anchors)
    assert rep.restored_regions >= 1
    assert len(rep.applied) == 1 and not rep.skipped
    assert _tunnel_open_mesh(out, 10.0, 0.0, "+z")
    bores = [t for t in _z_tunnels_near(out, 10.0, radius=4.0) if t["dia"] <= 15.0]
    assert len(bores) == 1
    assert bores[0]["dia"] == pytest.approx(10.0, abs=1.5)


def test_spec_anchor_no_material_when_patch_floats():
    """Same deferred spec anchor, but the whole keep box was eaten — the
    restore patch would float, so it drops and the deferred cut skips."""
    base = _slab()
    opt = _scooped_opt(base, 10.0, 15.0)
    anchors = [_bore_anchor(source="spec")]
    feats, _ = resolve_refine_features(anchors, None, None, base_mesh=base)
    assert len(feats) == 1 and feats[0].admit_on_material

    _out, rep = apply_refine(opt, feats, base_mesh=base, keep_regions=[], anchors=anchors)
    assert rep.restored_regions == 0
    assert not rep.applied
    assert any("no material" in s["reason"] for s in rep.skipped)


def test_spec_anchor_deferred_but_no_material_skipped():
    """Deferred spec anchor where NOTHING restored the material (restore
    disabled, opt mesh is a void there): the cut must be skipped — boring
    into air would be a fake 'applied'."""
    base = _slab()
    opt = _scooped_opt(base, 10.0, 15.0)
    anchors = [_bore_anchor(source="spec")]
    feats, _ = resolve_refine_features(anchors, None, None, base_mesh=base)
    assert len(feats) == 1 and feats[0].admit_on_material

    _out, rep = apply_refine(
        opt, feats, base_mesh=base, keep_regions=[], anchors=anchors, restore_keeps=False
    )
    assert not rep.applied
    assert any("no material" in s["reason"] for s in rep.skipped)
    assert any("no material" in w for w in rep.warnings)


def test_scan_anchor_without_base_tunnel_still_skipped():
    """d1d1836 regression: a mesh-scan-only anchor (no spec) over a base
    without a tunnel is still dropped at resolve time — never deferred."""
    base = _slab()
    anchors = [_bore_anchor(source="scan")]
    feats, warns = resolve_refine_features(anchors, None, None, base_mesh=base)
    assert not feats
    assert any("without a base-mesh tunnel" in w for w in warns)
    assert not any("deferred" in w for w in warns)


def test_scan_anchor_mesh_source_still_skipped():
    """Same protection with the 'mesh' source tag: excluded at resolve —
    never even admitted, no deferral."""
    anchors = [_bore_anchor(source="mesh")]
    feats, warns = resolve_refine_features(anchors, None, None, base_mesh=_slab())
    assert not feats
    assert any("without a base-mesh tunnel" in w for w in warns)


# --------------------------------------------------------------------------- #
# bore-ring material gate — the "cut into void" defect                        #
# --------------------------------------------------------------------------- #


def test_bore_ring_material_probe():
    """Ring probe: solid plate counts as material; an open bore still has
    wall material around it; a scoop that ate the whole neighbourhood does
    not — that is the no-host case the gate must catch."""
    assert _bore_ring_material(_slab(), "+z", 10.0, 0.0, 6.0)
    assert _bore_ring_material(_slab_bored(10.0, 6.8), "+z", 10.0, 0.0, 6.0)
    scooped = _scooped_opt(_slab(), 10.0, 15.0)
    assert not _bore_ring_material(scooped, "+z", 10.0, 0.0, 6.0)


def test_apply_refine_skips_feature_over_void():
    """Centre ray open AND ring empty (host region optimized away) → honest
    skip instead of a bogus bore drilled at the bbox face."""
    opt = _scooped_opt(_slab(), 10.0, 15.0)
    feat = RefineFeature(name="gone", kind="hole", axis="+z", u=10.0, v=0.0, dia=6.0)
    out, rep = apply_refine(opt, [feat], restore_keeps=False)
    assert not rep.applied
    assert rep.skipped and "no material" in rep.skipped[0]["reason"]
    assert any("no material" in w for w in rep.warnings)
    assert out.volume == pytest.approx(opt.volume)


def test_apply_refine_open_channel_still_recuts():
    """Gate must not kill the legitimate case: an existing tunnel whose
    walls survived → still re-cut from the bbox face."""
    mesh = _slab_bored(10.0, 8.0)
    feat = RefineFeature(name="b", kind="hole", axis="+z", u=10.0, v=0.0, dia=6.0)
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    assert out.is_watertight


# --------------------------------------------------------------------------- #
# spec-anchor preference + verify disambiguation                              #
# --------------------------------------------------------------------------- #


def test_spec_feature_prefers_spec_anchor_over_scanned():
    """Two same-family anchors, scanned one closer in dia: the spec-declared
    anchor carries the authoritative position and must win the feature."""
    anchors = [
        {
            "index": 1,
            "kind": "hole",
            "axis": "+z",
            "pos": [40.0, 0.0, 7.0],
            "x": 40.0,
            "y": 0.0,
            "dia": 6.7,
            "source": "scan",
        },
        {
            "index": 2,
            "kind": "hole",
            "axis": "+z",
            "pos": [-30.0, 0.0, 7.0],
            "x": -30.0,
            "y": 0.0,
            "dia": 7.0,
            "source": "spec",
        },
    ]
    spec_features = [{"type": "hole", "name": "m6", "dims": {"diameter": 6.8}}]
    feats, _ = resolve_refine_features(anchors, spec_features, None)
    f = next(f for f in feats if f.name == "m6")
    assert f.u == pytest.approx(-30.0) and f.v == pytest.approx(0.0)
    assert f.source == "spec"


def test_verify_expected_uv_disambiguates_twin_anchors():
    """Twin M6 bores share name+dia: the applied entry's recorded u/v picks
    the anchor it actually cut — verify must check THAT position."""
    anchors = [
        {"index": 1, "name": "m6", "axis": "+z", "pos": [10.0, 0.0, 7.0], "dia": 6.0},
        {"index": 2, "name": "m6", "axis": "+z", "pos": [-30.0, 0.0, 7.0], "dia": 6.0},
    ]
    entry = {"name": "m6", "dia": 6.0, "u": -30.0, "v": 0.0}
    assert _verify_expected_uv(entry, anchors, None) == pytest.approx((-30.0, 0.0))
    # without recorded u/v the first name hit is used — still deterministic
    assert _verify_expected_uv({"name": "m6", "dia": 6.0}, anchors, None) == pytest.approx(
        (10.0, 0.0)
    )


# --------------------------------------------------------------------------- #
# run_refine → refine_verify plumbing                                          #
# --------------------------------------------------------------------------- #


def _write_box_stl(tmp_path) -> object:
    p = tmp_path / "base.stl"
    _slab().export(str(p))
    return p


def test_run_refine_verify_misses_surfaced(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A cut that 'applied' but fails the post-refine rescan surfaces as
    verify_misses + a warning — the tool layer reads exactly that."""
    import cadx.cad.fem.refine as refine_mod
    from cadx.cad.engine import run_refine

    miss = [{"entry": {"name": "m6"}, "x": -30.0, "y": 0.0, "dia": 6.0}]
    monkeypatch.setattr(refine_mod, "refine_verify", lambda *a, **k: miss)
    res = run_refine(
        mesh_path=_write_box_stl(tmp_path),
        out_dir=tmp_path / "out",
        features=[{"name": "m6", "kind": "hole", "axis": "+z", "x": 10.0, "y": 0.0, "dia": 6.0}],
    )
    assert res.ok
    report = res.features["refine"]
    assert report["verify_misses"] == miss
    assert any("FAILED" in w for w in report["warnings"])


def test_run_refine_verify_passes_clean(tmp_path):
    """Happy path: the re-cut bore scans back at the requested position —
    no verify_misses key on the report."""
    from cadx.cad.engine import run_refine

    res = run_refine(
        mesh_path=_write_box_stl(tmp_path),
        out_dir=tmp_path / "out",
        features=[{"name": "m6", "kind": "hole", "axis": "+z", "x": 10.0, "y": 0.0, "dia": 6.0}],
        anchors=[
            {
                "index": 1,
                "name": "m6",
                "kind": "hole",
                "axis": "+z",
                "pos": [10.0, 0.0, 7.0],
                "x": 10.0,
                "y": 0.0,
                "dia": 6.0,
            }
        ],
    )
    assert res.ok
    assert not res.features["refine"].get("verify_misses")


def test_run_refine_fem_verify_scans_refined_mesh_anchors(tmp_path):
    """The refine FEM re-verify solves on the REFINED mesh — its anchors must
    be scanned there (run_fem_pipeline parity), not inherited from the
    parent. With anchors=None (imported STL / missing analysis) a hint-bound
    `bore` region otherwise falls back to a 1-cell boundary patch."""
    from cadx.cad.engine import run_refine

    p = tmp_path / "bored.stl"
    _slab_bored(x0=10.0, dia=10.0).export(str(p))
    res = run_refine(
        mesh_path=p,
        out_dir=tmp_path / "out",
        features=[{"name": "m10", "kind": "hole", "axis": "+z", "x": 10.0, "y": 0.0, "dia": 10.0}],
        anchors=None,
        load_cases=[
            {
                "name": "radial",
                "fixed": [{"kind": "plane", "axis": "-z"}],
                "loads": [
                    {
                        "region": {
                            "kind": "bore",
                            "axis": "+z",
                            "center": [10.0, 0.0, 7.0],
                            "diameter_mm": 10.0,
                        },
                        "bearing_n": 500.0,
                        "bearing_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ],
        voxel_mm=3.0,
    )
    assert res.ok
    case = res.fem["cases"][0]
    applied = case.get("applied") or []
    assert applied and "bearing_n" in (applied[0].get("mechanisms") or [])
    warns = case.get("warnings") or []
    assert not any("region_fallback" in w or "resolved to zero" in w for w in warns)


# --------------------------------------------------------------------------- #
# spec-declared positions — declared (u,v) beats dia-matched anchor position   #
# --------------------------------------------------------------------------- #


_ROBOT_SPEC = [
    {
        "type": "hole",
        "name": "mounting holes",
        "position": "at (x=+/-42, y=+/-32), through z=0..10",
        "dims": {"diameter": 8.5, "depth": 10.0},
        "thread_spec": "M8x1.25-6H",
    },
    {
        "type": "hole",
        "name": "M12 tapped through-bore",
        "position": "x=0, y=0, through z=0..25",
        "dims": {"diameter": 10.2, "depth": 25.0},
        "thread_spec": "M12x1.75-6H",
    },
    {
        "type": "hole",
        "name": "flange holes",
        "position": "y=+/-12, z=25, through x=42..50",
        "dims": {"diameter": 6.5, "depth": 8.0},
    },
]


def test_spec_axis_parses_through_axis():
    assert _spec_axis(_ROBOT_SPEC[0], _ROBOT_SPEC[0]["position"]) == "+z"
    assert _spec_axis(_ROBOT_SPEC[1], _ROBOT_SPEC[1]["position"]) == "+z"
    # the flange bores run across x on the +x wall
    assert _spec_axis(_ROBOT_SPEC[2], _ROBOT_SPEC[2]["position"]) == "+x"


def test_spec_uv_positions_expand_symmetric_grid():
    text = _ROBOT_SPEC[0]["position"]
    pos = _spec_uv_positions(text, "+z")
    assert sorted(pos) == sorted([(42, 32), (42, -32), (-42, 32), (-42, -32)])


def test_spec_uv_positions_single_center():
    pos = _spec_uv_positions(_ROBOT_SPEC[1]["position"], "+z")
    assert pos == [(0.0, 0.0)]


def test_spec_uv_positions_other_axis_plane():
    # +x bores: in-plane axes are y (u) and z (v)
    pos = _spec_uv_positions(_ROBOT_SPEC[2]["position"], "+x")
    assert sorted(pos) == sorted([(12.0, 25.0), (-12.0, 25.0)])


def test_spec_uv_positions_empty_when_undeclared():
    assert _spec_uv_positions("somewhere on the plate", "+z") == []


def test_resolve_spec_features_keep_declared_positions():
    """Eval regression: the M12 spec bore bound by dia+axis to a corner
    anchor 57 mm off its declared centre. Declared positions must win —
    anchors only supply tunnel evidence."""
    anchors = [
        {
            "index": 1,
            "kind": "hole",
            "axis": "+z",
            "x": 0.0,
            "y": 0.5,
            "dia": 10.4,
            "source": "scan",
            "pos": [0.0, 0.5, 12.0],
        },
        {
            "index": 2,
            "kind": "hole",
            "axis": "+z",
            "x": -42.0,
            "y": 31.9,
            "dia": 9.9,
            "source": "scan",
            "pos": [-42.0, 31.9, 5.0],
        },
        {
            "index": 3,
            "kind": "hole",
            "axis": "+z",
            "x": 42.0,
            "y": -32.0,
            "dia": 8.9,
            "source": "scan",
            "pos": [42.0, -32.0, 5.0],
        },
        {
            "index": 4,
            "kind": "hole",
            "axis": "+z",
            "x": 42.0,
            "y": 32.0,
            "dia": 8.9,
            "source": "scan",
            "pos": [42.0, 32.0, 5.0],
        },
    ]
    feats, _warns = resolve_refine_features(
        anchors, _ROBOT_SPEC, None, base_mesh=_slab_bored(10.0, 10.0)
    )
    m12 = [f for f in feats if "M12" in f.name]
    assert len(m12) == 1
    assert (m12[0].u, m12[0].v) == (0.0, 0.0)  # declared centre, NOT (-42, 32)
    assert m12[0].source == "spec" and m12[0].dia == pytest.approx(10.2)
    mounts = [f for f in feats if f.name == "mounting holes"]
    assert len(mounts) == 4
    assert sorted((f.u, f.v) for f in mounts) == sorted(
        [(42.0, 32.0), (42.0, -32.0), (-42.0, 32.0), (-42.0, -32.0)]
    )
    flange = [f for f in feats if f.name == "flange holes"]
    assert len(flange) == 2 and all(f.axis == "+x" for f in flange)


def test_resolve_spec_falls_back_to_dia_match_without_position():
    """A spec hole with no parseable position keeps the legacy dia+axis
    anchor match."""
    anchors = [
        {
            "index": 1,
            "kind": "hole",
            "axis": "+z",
            "x": 10.0,
            "y": 0.0,
            "dia": 6.1,
            "source": "scan",
            "pos": [10.0, 0.0, 6.0],
        },
    ]
    spec = [{"type": "hole", "name": "m6 bore", "position": "", "dims": {"diameter": 6.0}}]
    feats, _ = resolve_refine_features(anchors, spec, None, base_mesh=_slab_bored(10.0, 6.0))
    assert len(feats) == 1
    assert (feats[0].u, feats[0].v) == (10.0, 0.0)


def test_spec_type_bore_normalizes_to_cuttable_hole():
    """Bearing-bracket regression: spec type 'bore' was outside the spec
    allowlist, so the Ø30 seat silently vanished from the cutter list and
    the optimized web stayed shut. 'bore' must normalize to 'hole'."""
    spec = [
        {
            "type": "bore",
            "name": "bearing seat",
            "position": "x=0, z=65, through y=-35..35",
            "dims": {"diameter": 30.0},
        }
    ]
    feats, warns = resolve_refine_features([], spec, None, base_mesh=_slab())
    assert len(feats) == 1
    assert feats[0].kind == "hole" and feats[0].dia == pytest.approx(30.0)
    assert feats[0].axis == "+y" and (feats[0].u, feats[0].v) == (0.0, 65.0)
    assert feats[0].through
    assert any("normalized" in w for w in warns)


def test_spec_type_tunnel_and_bearing_alias():
    """'tunnel' maps to a through hole, 'bearing' to the seated variant —
    neither may be silently dropped."""
    spec = [
        {
            "type": "tunnel",
            "name": "through",
            "position": "x=5, y=5, through z=0..14",
            "dims": {"diameter": 8.0},
        },
        {
            "type": "bearing",
            "name": "seat",
            "position": "x=0, z=30, through y=-20..20",
            "dims": {"diameter": 30.0},
        },
    ]
    feats, warns = resolve_refine_features([], spec, None, base_mesh=_slab())
    kinds = {f.name: f.kind for f in feats}
    assert kinds == {"through": "hole", "seat": "bearing_seat"}
    assert sum("normalized" in w for w in warns) == 2


def test_feature_host_main_flags_floating_puck():
    """Bore re-cut into a keep-restored ring that floats free of the body —
    the dia still measures nominal, but the host is a fragment."""
    base = _slab_bored(10.0, 10.0)
    puck = trimesh.creation.annulus(r_min=5.0, r_max=8.0, height=14.0)
    puck.apply_translation([10.0, 0.0, 0.0])
    floating = trimesh.util.concatenate([base, puck])
    feat = RefineFeature(name="m", kind="hole", axis="+z", u=10.0, v=0.0, dia=10.0)
    assert _feature_host_main(floating, feat, entry=14.0, span=14.0) is False
    # same feature on the single-body slab → connected
    assert _feature_host_main(base, feat, entry=14.0, span=14.0) is True


# --------------------------------------------------------------------------- #
# through-intent cuts + verification                                            #
# --------------------------------------------------------------------------- #


def test_through_feature_pierces_past_declared_depth():
    """M12 eval regression: 'through z=0..25' carried depth_mm=25, but a
    keep-restored cap pushed the real span to ~28 — a cutter stopped at the
    declared depth left a sealing membrane. The effective depth must reach
    the far wall of every solid interval the reach overlaps."""
    slab = _slab(z=25.0)  # solid 60x40x25 — no bore
    feat = RefineFeature(
        name="m12",
        kind="hole",
        axis="+z",
        u=0.0,
        v=0.0,
        dia=8.0,
        depth_mm=10.0,
        through=True,
        source="spec",
    )
    out, rep = apply_refine(slab, [feat], restore_keeps=False)
    assert rep.applied and not rep.skipped
    # the cut ran deeper than declared — and the bore axis sees daylight
    assert rep.applied[0]["cut_depth_mm"] > 20.0
    assert any("through-cut extended" in w for w in rep.warnings)
    locs, _, _ = out.ray.intersects_location(
        [[0.0, 0.0, -10.0]], [[0.0, 0.0, 1.0]], multiple_hits=True
    )
    assert not len(locs)


def test_blind_feature_still_stops_at_declared_depth():
    """The counter-case: without through-intent the declared depth is a
    hard stop — a blind pocket must NOT pierce the slab."""
    slab = _slab(z=25.0)
    feat = RefineFeature(
        name="pocket",
        kind="hole",
        axis="+z",
        u=0.0,
        v=0.0,
        dia=8.0,
        depth_mm=10.0,
        through=False,
        source="spec",
    )
    out, rep = apply_refine(slab, [feat], restore_keeps=False)
    assert rep.applied
    assert rep.applied[0]["through"] is False
    locs, _, _ = out.ray.intersects_location(
        [[0.0, 0.0, -10.0]], [[0.0, 0.0, 1.0]], multiple_hits=True
    )
    assert len(locs) >= 2  # pocket floor + slab bottom still stand


def test_plugs_run_before_drills_no_mount_bore_membrane():
    """Bearing-bracket regression: the Ø30 seat's channel plug overlaps the
    Ø8.5 Z-bore radially — plugged AFTER the Z-drill it left a ~0.7 mm
    membrane sealing the fresh bore (verify caught it, gates didn't).
    All plugs must union before any cutter runs."""
    block = trimesh.creation.box(extents=[120.0, 70.0, 90.0])
    block.apply_translation([0.0, 0.0, 45.0])
    chan = trimesh.creation.cylinder(radius=16.0, height=100.0, sections=48)
    chan.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    chan.apply_translation([0.0, 0.0, 65.0])
    opt = block.difference(chan, engine="manifold")
    feats = [
        # mount bore FIRST — the order the observed run cut it in
        RefineFeature(
            name="mount", kind="hole", axis="+z", u=0.0, v=-22.0,
            dia=8.5, through=True, source="spec",
        ),
        RefineFeature(
            name="seat", kind="hole", axis="+y", u=0.0, v=65.0,
            dia=30.0, through=True, source="spec",
        ),
    ]
    out, rep = apply_refine(opt, feats, restore_keeps=False)
    names = {str(e.get("name") or "") for e in rep.applied}
    assert "mount" in names and "seat" in names
    # the mount bore's centre ray sees daylight — no plug membrane
    blocked = _channel_blocked_mm(out, 0.0, -22.0, "+z")
    assert blocked <= 0.3, f"mount bore sealed by {blocked:.2f} mm of plug"
    # and the seat itself is open at the declared position
    assert _channel_blocked_mm(out, 0.0, 65.0, "+y") <= 0.3


def test_through_cut_depth_spans_plug_extent():
    """Unit-level: entry seated on a plug face ABOVE the part — the reach
    must cover the plug's far side, not stop at the declared depth."""
    # plate z=0..10 + a cap z=24..28 (the 'plug'/restored dome scenario)
    plate = _slab(z=10.0)
    cap = trimesh.creation.box(extents=[20.0, 20.0, 4.0])
    cap.apply_translation([0.0, 0.0, 26.0])
    mesh = trimesh.util.concatenate([plate, cap])
    feat = RefineFeature(
        name="m12",
        kind="hole",
        axis="+z",
        u=0.0,
        v=0.0,
        dia=10.0,
        depth_mm=25.0,
        through=True,
    )
    depth = _through_cut_depth(mesh, feat, entry=28.0, span=28.0)
    # reach entry→entry-25 = z∈[3,28]; the plate's far wall is z=0 → ~28.3
    assert depth == pytest.approx(28.3, abs=0.6)


def test_verify_uv_reads_zero_uv_coords():
    """Regression: applied records carry ``u``/``v`` — the M12 bore at
    (0, 0) was verified against the nearest corner anchor because only
    ``pos``/``x``/``y`` were read and 0.0 looked 'missing'."""
    assert _verify_uv({"u": 0.0, "v": 0.0}, "+z") == (0.0, 0.0)
    assert _verify_uv({"u": 12.0, "v": 25.0}, "+x") == (12.0, 25.0)
    # pos triple still resolves when u/v are absent
    assert _verify_uv({"pos": [10.0, 0.0, 7.0]}, "+z") == (10.0, 0.0)


def test_verify_uses_recorded_uv_at_part_centre():
    """The M12 eval regression end-to-end: a through-bore at (0, 0) must
    verify at ITS position — not be re-seated onto a corner anchor of
    similar diameter."""
    mesh = _slab_bored(0.0, 10.0)  # Ø10 bore at the slab centre
    anchors = [
        {
            "index": 1,
            "kind": "hole",
            "axis": "+z",
            "x": -25.0,
            "y": 0.0,
            "pos": [-25.0, 0.0, 7.0],
            "dia": 9.9,
            "source": "scan",
        },
    ]
    report = {
        "applied": [
            {
                "name": "m12",
                "kind": "hole",
                "axis": "+z",
                "u": 0.0,
                "v": 0.0,
                "dia": 10.0,
                "through": True,
                "depth_mm": 25.0,
                "source": "spec",
            },
        ]
    }
    assert refine_verify(mesh, report, anchors=anchors) == []


def test_verify_through_bore_over_pocket_is_a_miss():
    """A 'through' bore that only left a blind pocket must NOT verify —
    the pocket probe cannot tell 'pierced' from 'dimple exists', so
    through-features rely on the surface scan alone."""
    plate = _slab(z=14.0)
    pocket = trimesh.creation.cylinder(radius=5.0, height=5.0, sections=48)
    pocket.apply_translation([10.0, 0.0, 11.5])
    mesh = plate.difference(pocket, engine="manifold")  # Ø10 pocket, 5 deep
    report = {
        "applied": [
            {
                "name": "m",
                "kind": "hole",
                "axis": "+z",
                "u": 10.0,
                "v": 0.0,
                "dia": 10.0,
                "through": True,
                "depth_mm": 14.0,
                "source": "spec",
            },
        ]
    }
    misses = refine_verify(mesh, report)
    assert len(misses) == 1
    # the same entry without through-intent verifies against the pocket
    report2 = {
        "applied": [
            {
                "name": "m",
                "kind": "hole",
                "axis": "+z",
                "u": 10.0,
                "v": 0.0,
                "dia": 10.0,
                "through": False,
                "depth_mm": 5.0,
                "source": "spec",
            },
        ]
    }
    assert refine_verify(mesh, report2) == []


def test_spec_position_text_marks_through_intent():
    """'through z=0..25'/'through x=42..50' in the spec's own words set
    through=True even with a numeric declared depth; a spec declaring
    only 'depth 8' stays a blind stop."""
    feats, _ = resolve_refine_features([], _ROBOT_SPEC, None, base_mesh=_slab(z=40.0))
    assert feats and all(f.through for f in feats)
    blind = [
        {
            "type": "hole",
            "name": "dowel",
            "position": "x=10, y=0",
            "dims": {"diameter": 6.0, "depth": 8.0},
        },
    ]
    feats2, _ = resolve_refine_features([], blind, None, base_mesh=_slab())
    assert len(feats2) == 1 and feats2[0].through is False


def test_post_cut_cleanup_drops_sliver_keeps_real_bodies():
    """A sub-mm³ inverted boolean sliver riding along the main body is
    dropped (it trips the shipped bodies/watertight gates), while a real
    disconnected body above the bridge floor survives."""
    from cadx.cad.fem.refine import RefineReport, _post_cut_cleanup

    main = trimesh.creation.box(extents=[20, 20, 20])
    sliver = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    sliver.apply_translation([8, 8, 8])
    sliver.invert()  # negative-volume boolean artifact
    mesh = trimesh.util.concatenate([main, sliver])
    rep = RefineReport()
    out = _post_cut_cleanup(mesh, rep)
    bodies = [p for p in out.split(only_watertight=False) if len(p.faces)]
    assert len(bodies) == 1
    assert any("sliver" in w for w in rep.warnings)

    real = trimesh.creation.box(extents=[4, 4, 4])  # 64 mm³ — meaningful
    real.apply_translation([30, 0, 0])
    mesh2 = trimesh.util.concatenate([main, real])
    out2 = _post_cut_cleanup(mesh2, RefineReport())
    assert len([p for p in out2.split(only_watertight=False) if len(p.faces)]) == 2


def test_apply_refine_drops_sliver_minted_by_postpass(monkeypatch):
    """The reskin/re-cut stage runs after `_post_cut_cleanup` — a zero-volume
    sheet it mints (observed on the robot-arm flange wall) must still be
    dropped so the shipped mesh reports one body."""
    import cadx.cad.fem.refine as rf

    base = trimesh.creation.box(extents=[20, 20, 10])

    def slivered_postpass(out, report):
        out = out.copy()
        # degenerate zero-area face pair riding the +x face — volume 0
        return trimesh.util.concatenate(
            [
                out,
                trimesh.Trimesh(
                    vertices=[
                        [12.0, 1.0, 5.0],
                        [18.0, 1.0, 5.0],
                        [10.0, 1.0, 5.0],
                        [19.0, 1.0, 5.0],
                    ],
                    faces=[[0, 1, 2], [1, 3, 2]],
                    process=False,
                ),
            ]
        )

    monkeypatch.setattr(rf, "_watertight_postpass", slivered_postpass)
    out, rep = rf.apply_refine(base, [], base_mesh=base, keep_regions=[], anchors=[])
    bodies = [p for p in out.split(only_watertight=False) if len(p.faces)]
    assert len(bodies) == 1
    assert any("sliver" in w for w in rep.warnings)
