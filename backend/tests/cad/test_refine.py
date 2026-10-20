"""Post-TopOpt feature finishing — cutter mapping, anchor/spec resolution,
keep-region restore, boolean robustness (`cad/fem/refine.py`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.anchors3d import scan_mesh_anchors
from cadx.cad.fem.refine import (
    RefineFeature,
    _cutter_mesh,
    _entry_point,
    _metric_pitch,
    _parse_spec,
    _thread_spec_from,
    _tunnel_open_mesh,
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
    onto the measured tunnel centre and ream it, leaving ONE hole."""
    mesh = _bored_wide(-35.13, 6.8)
    feat = RefineFeature(name="b", kind="hole", axis="+z", u=-35.16, v=0.0, dia=6.75)
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1

    tunnels = _z_tunnels_near(out, -35.15)
    assert len(tunnels) == 1  # no figure-8 / sliver second void
    assert tunnels[0]["x"] == pytest.approx(-35.15, abs=0.5)
    # requested 6.75 reamed ~+0.4 -> ~7.15 (scan dia reads within ~0.5·step)
    assert tunnels[0]["dia"] == pytest.approx(7.15, abs=0.7)


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
    feat = RefineFeature(
        name="m5", kind="hole", axis="+z", u=-35.16, v=0.0, dia=5.5, source="spec"
    )
    out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    assert rep.applied[0].get("plugged") is True
    tunnels = _z_tunnels_near(out, -35.15)
    assert len(tunnels) == 1
    assert tunnels[0]["dia"] == pytest.approx(5.5, abs=0.7)


def test_recut_non_spec_keeps_ream_fallback():
    """Scanned (non-spec) anchors still get the +0.4 ream — cleanliness beats
    nominal diameter when no spec pins the size."""
    mesh = _bored_wide(-35.13, 6.8)
    feat = RefineFeature(name="b", kind="hole", axis="+z", u=-35.16, v=0.0, dia=6.75)
    _out, rep = apply_refine(mesh, [feat], restore_keeps=False)
    assert len(rep.applied) == 1
    assert not rep.applied[0].get("plugged")
    assert rep.applied[0]["ream_extra_mm"] == pytest.approx(0.4)


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


def test_auto_keep_restores_eaten_bore_ring():
    """SIMP ate the material ring around a Ø10 bore and the refine job
    forwarded keep_regions=[] — an auto keep ring per anchor must restore
    the base surface first so the re-cut leaves ONE clean Ø10 tunnel."""
    base = _slab_bored(10.0, 10.0)
    opt = _scooped_opt(base, 10.0, 15.0)
    anchors = [_bore_anchor()]
    feats, _ = resolve_refine_features(anchors, None, None, base_mesh=base)
    assert len(feats) == 1

    out, rep = apply_refine(opt, feats, base_mesh=base, keep_regions=[], anchors=anchors)
    assert rep.restored_regions >= 1
    assert any("auto keep-restore" in w for w in rep.warnings)
    assert len(rep.applied) == 1 and not rep.skipped
    assert _tunnel_open_mesh(out, 10.0, 0.0, "+z")
    # the scan may also see the scooped opening itself as a tunnel — the
    # requirement is exactly one ~Ø10 bore at the anchor position
    bores = [t for t in _z_tunnels_near(out, 10.0, radius=4.0) if t["dia"] <= 15.0]
    assert len(bores) == 1
    # requested Ø10, possibly reamed +0.4 after snapping onto the restored bore
    assert bores[0]["dia"] == pytest.approx(10.0, abs=1.5)


def test_spec_anchor_recut_after_auto_restore():
    """Spec-declared anchor over SOLID base (no base tunnel): resolve defers
    the verdict, auto keep-restore pastes the ring back, and the bore is cut
    — not dropped by the d1d1836 tunnel allowlist."""
    base = _slab()  # solid — the spec bore was never in the scan
    opt = _scooped_opt(base, 10.0, 15.0)
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
