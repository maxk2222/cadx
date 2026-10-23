"""Parametric STEP (analytic feature cuts) + blind-feature refine verification.

Q3: ``exporters.export_analytic_step`` sews the optimized trimesh to a B-Rep
and re-cuts applied refine features with analytic OCC solids — true
``GeomType.CYLINDER`` bore walls, an IsoThread helical groove for threads —
instead of the all-triangle faceted STEP. Every failure path falls back to
the faceted STEP (or no STEP when sewing itself fails) and never raises.

Q4: ``refine.refine_verify`` rescans the refined mesh for the features the
report claims were cut; features the surface heightfield scan can't see are
rescued by ``anchors3d.probe_blind_features`` (mocked here — the real probe
lands with anchors3d).
"""

from __future__ import annotations

import math

import pytest
import trimesh

from cadx.cad.engine import _post_extract_finish
from cadx.cad.fem.refine import (
    RefineFeature,
    applied_refine_features,
    apply_refine,
    refine_verify,
)


def _box(x: float = 60.0, y: float = 30.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _bored(x: float = 10.0, dia: float = 6.0) -> trimesh.Trimesh:
    """60x30x12 box with a Ø`dia` through-bore at (x, 0)."""
    bore = trimesh.creation.cylinder(radius=dia / 2.0, height=40.0, sections=48)
    bore.apply_translation([x, 0.0, 6.0])
    return _box().difference(bore, engine="manifold")


# --------------------------------------------------------------------------- #
# export_analytic_step                                                          #
# --------------------------------------------------------------------------- #


def test_analytic_step_through_hole_has_cylinder_face(tmp_path):
    b3d = pytest.importorskip("build123d")
    from cadx.cad.exporters import export_analytic_step

    feats = [RefineFeature(name="b1", kind="hole", axis="+z", u=10.0, v=0.0, dia=6.0)]
    res = export_analytic_step(_box(), feats, tmp_path / "m.step")
    assert res["step"] == "m.step"
    assert res["step_features"] == [{"name": "b1", "kind": "hole", "axis": "+z", "analytic": True}]
    assert not res["warnings"]
    solid = b3d.import_step(str(tmp_path / "m.step"))
    cyl = [f for f in solid.faces() if f.geom_type == b3d.GeomType.CYLINDER]
    assert cyl, "through-hole must read back as an analytic CYLINDER face"
    assert cyl[0].radius == pytest.approx(3.0, abs=0.05)


def test_analytic_step_thread_keeps_cylinder_and_groove(tmp_path):
    b3d = pytest.importorskip("build123d")
    pytest.importorskip("bd_warehouse.thread")
    from cadx.cad.exporters import export_analytic_step

    mesh = _box()
    feats = [
        RefineFeature(
            name="m8", kind="thread", axis="+z", u=10.0, v=0.0, dia=8.0, thread_spec="M8-1.25"
        )
    ]
    res = export_analytic_step(mesh, feats, tmp_path / "t.step")
    assert res["step"] == "t.step"
    assert res["step_features"][0]["kind"] == "thread"
    solid = b3d.import_step(str(tmp_path / "t.step"))
    gtypes = {f.geom_type for f in solid.faces()}
    # tap-drill bore stays an analytic cylinder; the helical groove adds
    # non-planar faces a plain bore would never produce
    assert b3d.GeomType.CYLINDER in gtypes
    removed = mesh.volume - solid.volume
    tap = math.pi * ((8.0 - 1.25) / 2.0) ** 2 * 12.0
    assert removed > tap * 1.02, "helical groove should remove more than the tap drill"


def test_analytic_step_blind_minus_z(tmp_path):
    b3d = pytest.importorskip("build123d")
    from cadx.cad.exporters import export_analytic_step

    mesh = _box()
    feats = [RefineFeature(name="bl", kind="hole", axis="-z", u=-5.0, v=0.0, dia=6.0, depth_mm=5.0)]
    res = export_analytic_step(mesh, feats, tmp_path / "b.step")
    assert res["step"] == "b.step"
    solid = b3d.import_step(str(tmp_path / "b.step"))
    removed = mesh.volume - solid.volume
    assert removed == pytest.approx(math.pi * 3.0**2 * 5.0, rel=0.08)
    assert any(f.geom_type == b3d.GeomType.CYLINDER for f in solid.faces())


def test_analytic_step_dict_features(tmp_path):
    b3d = pytest.importorskip("build123d")
    from cadx.cad.exporters import export_analytic_step

    feats = [{"name": "d1", "kind": "hole", "axis": "+z", "x": 0.0, "y": 0.0, "dia": 5.0}]
    res = export_analytic_step(_box(), feats, tmp_path / "d.step")
    assert res["step"] == "d.step"
    assert res["step_features"][0]["name"] == "d1"
    solid = b3d.import_step(str(tmp_path / "d.step"))
    assert any(f.geom_type == b3d.GeomType.CYLINDER for f in solid.faces())


def test_analytic_step_sew_failure_never_raises(monkeypatch, tmp_path):
    pytest.importorskip("build123d")
    import cadx.cad.mesh_ops as mo
    from cadx.cad.exporters import export_analytic_step

    def boom(*args: object, **kwargs: object) -> object:
        raise ValueError("watertight rebuild diverged from input")

    monkeypatch.setattr(mo, "mesh_to_solid", boom)
    feats = [RefineFeature(name="b1", kind="hole", axis="+z", u=0.0, v=0.0, dia=6.0)]
    res = export_analytic_step(_box(), feats, tmp_path / "x.step")
    assert res["step"] is None
    assert res["step_features"] == []
    assert res["warnings"]
    assert not (tmp_path / "x.step").exists()


def test_analytic_step_bad_feature_keeps_faceted_step(tmp_path):
    pytest.importorskip("build123d")
    from cadx.cad.exporters import export_analytic_step

    # 'boss' is not a cuttable kind — skipped; the faceted sewn STEP still ships
    feats = [RefineFeature(name="boss", kind="boss", axis="+z", u=0.0, v=0.0, dia=8.0)]
    res = export_analytic_step(_box(), feats, tmp_path / "f.step")
    assert res["step"] == "f.step"
    assert res["step_features"] == []


# --------------------------------------------------------------------------- #
# refine_verify — tunnel scan + blind-probe rescue                              #
# --------------------------------------------------------------------------- #


def test_refine_verify_through_hole_passes():
    report = {
        "applied": [{"name": "b1", "kind": "hole", "axis": "+z", "x": 10.0, "y": 0.0, "dia": 6.0}]
    }
    assert refine_verify(_bored(10.0, 6.0), report) == []


def test_topopt_post_extract_recuts_real_base_bore() -> None:
    """The shared TopOpt finishing step restores/re-cuts a real base tunnel
    before verification, and reports the applied cut with no verification
    miss."""
    base = _bored(10.0, 6.0)
    degraded = _bored(10.0, 3.0)
    anchors = [
        {
            "kind": "hole",
            "axis": "+z",
            "x": 10.0,
            "y": 0.0,
            "pos": [10.0, 0.0, 6.0],
            "dia": 6.0,
            "depth_mm": 12.0,
            "source": "scan",
        }
    ]

    out, report, warnings = _post_extract_finish(
        degraded,
        base_mesh=base,
        keep_regions=[],
        anchors=anchors,
        spec_features=[],
        recut_bores=True,
        restore_keeps=True,
    )

    assert report is not None, warnings
    assert report["applied"]
    assert report["verify_misses"] == []
    assert refine_verify(out, report, anchors=anchors) == []
    from cadx.cad.mesh_features import detect_mesh_features

    holes = detect_mesh_features(out).holes
    assert any(h.axis == "z" and abs(h.dia - 6.0) < 0.6 for h in holes)


def test_topopt_post_extract_does_not_drill_base_dimple() -> None:
    """A scanned anchor without a real base-mesh tunnel is not admitted for
    re-cut; this guards against turning mesh dimples into through-holes."""
    solid = _box()
    anchor = {
        "kind": "hole",
        "axis": "+z",
        "x": 10.0,
        "y": 0.0,
        "pos": [10.0, 0.0, 6.0],
        "dia": 6.0,
        "depth_mm": 2.0,
        "source": "scan",
    }

    out, report, warnings = _post_extract_finish(
        solid,
        base_mesh=solid,
        keep_regions=[],
        anchors=[anchor],
        spec_features=[],
        recut_bores=True,
        restore_keeps=False,
    )

    assert report is not None, warnings
    assert not report["applied"]
    assert out.volume == pytest.approx(solid.volume)


def test_refine_verify_missing_feature_flagged():
    report = {
        "applied": [
            {"name": "ghost", "kind": "hole", "axis": "+z", "x": 10.0, "y": 0.0, "dia": 6.0}
        ]
    }
    misses = refine_verify(_box(), report)
    assert len(misses) == 1
    assert misses[0]["entry"]["name"] == "ghost"
    assert misses[0]["x"] == pytest.approx(10.0)


def test_refine_verify_blind_probe_verifies(monkeypatch: pytest.MonkeyPatch):
    """Scan-invisible blind pocket confirmed by probe_blind_features → no miss."""
    from cadx.cad.fem import anchors3d

    seen: list[dict] = []

    def fake_probe(mesh: object, anchors: list[dict], *, log: object = None) -> list[dict]:
        seen.extend(anchors)
        return [
            {
                "name": anchors[0]["name"],
                "found": True,
                "open": False,
                "measured_depth_mm": 6.0,
                "measured_dia_mm": 4.0,
            }
        ]

    monkeypatch.setattr(anchors3d, "probe_blind_features", fake_probe, raising=False)
    report = {
        "applied": [
            {
                "name": "p1",
                "kind": "hole",
                "axis": "+z",
                "x": 5.0,
                "y": 0.0,
                "dia": 4.0,
                "depth_mm": 6.0,
            }
        ]
    }
    assert refine_verify(_box(), report) == []
    assert seen and seen[0]["name"] == "p1" and seen[0]["depth_mm"] == 6.0


def test_refine_verify_probe_shallow_keeps_miss(monkeypatch: pytest.MonkeyPatch):
    """Probe found the pocket but only 2 mm of the requested 6 mm → FAIL."""
    from cadx.cad.fem import anchors3d

    monkeypatch.setattr(
        anchors3d,
        "probe_blind_features",
        lambda mesh, anchors, *, log=None: [
            {
                "name": anchors[0]["name"],
                "found": True,
                "open": False,
                "measured_depth_mm": 2.0,
                "measured_dia_mm": 4.0,
            }
        ],
        raising=False,
    )
    report = {
        "applied": [
            {
                "name": "p1",
                "kind": "hole",
                "axis": "+z",
                "x": 5.0,
                "y": 0.0,
                "dia": 4.0,
                "depth_mm": 6.0,
            }
        ]
    }
    assert len(refine_verify(_box(), report)) == 1


def test_refine_verify_probe_not_found_keeps_miss(monkeypatch: pytest.MonkeyPatch):
    from cadx.cad.fem import anchors3d

    monkeypatch.setattr(
        anchors3d,
        "probe_blind_features",
        lambda mesh, anchors, *, log=None: [{"name": anchors[0]["name"], "found": False}],
        raising=False,
    )
    report = {
        "applied": [{"name": "p1", "kind": "hole", "axis": "+z", "x": 5.0, "y": 0.0, "dia": 4.0}]
    }
    assert len(refine_verify(_box(), report)) == 1


def test_refine_verify_probe_crash_keeps_scan_verdict(monkeypatch: pytest.MonkeyPatch):
    """A probe error must never break verification — degrade to the scan."""
    from cadx.cad.fem import anchors3d

    def boom(*args: object, **kwargs: object) -> list[dict]:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(anchors3d, "probe_blind_features", boom, raising=False)
    report = {
        "applied": [{"name": "p1", "kind": "hole", "axis": "+z", "x": 5.0, "y": 0.0, "dia": 4.0}]
    }
    assert len(refine_verify(_box(), report)) == 1


def test_refine_verify_without_probe_keeps_miss(monkeypatch: pytest.MonkeyPatch):
    """Missing probe_blind_features (pre-merge) degrades to scan-only misses."""
    from cadx.cad.fem import anchors3d

    monkeypatch.delattr(anchors3d, "probe_blind_features", raising=False)
    report = {
        "applied": [{"name": "p1", "kind": "hole", "axis": "+z", "x": 5.0, "y": 0.0, "dia": 4.0}]
    }
    assert len(refine_verify(_box(), report)) == 1


def test_refine_verify_real_probe_blind_pocket():
    """End-to-end with the real probe: a 1.2 mm blind pocket is too shallow
    for the dimple scan (≥1.5 mm) but probe_blind_features confirms it."""
    from cadx.cad.fem import anchors3d

    if getattr(anchors3d, "probe_blind_features", None) is None:
        pytest.skip("probe_blind_features not merged yet")
    pocket = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=32)
    # cylinder spans z in [10.8, 14.8] — leaves a 1.2 mm blind pocket at the top
    pocket.apply_translation([5.0, 0.0, 12.8])
    mesh = _box().difference(pocket, engine="manifold")
    report = {
        "applied": [
            {
                "name": "p1",
                "kind": "hole",
                "axis": "+z",
                "x": 5.0,
                "y": 0.0,
                "dia": 4.0,
                "depth_mm": 1.2,
            }
        ]
    }
    assert refine_verify(mesh, report) == []


def test_refine_verify_open_probe_verifies_through(monkeypatch: pytest.MonkeyPatch):
    """Through feature missed by the scan but probed open → verified."""
    from cadx.cad.fem import anchors3d

    monkeypatch.setattr(
        anchors3d,
        "probe_blind_features",
        lambda mesh, anchors, *, log=None: [
            {
                "name": anchors[0]["name"],
                "found": True,
                "open": True,
                "measured_depth_mm": None,
                "measured_dia_mm": 6.1,
            }
        ],
        raising=False,
    )
    report = {
        "applied": [{"name": "t1", "kind": "tunnel", "axis": "+z", "x": 5.0, "y": 0.0, "dia": 6.0}]
    }
    assert refine_verify(_box(), report) == []


def test_applied_refine_features_filters_by_name():
    from cadx.cad.fem.refine import RefineReport

    feats = [
        RefineFeature(name="a", kind="hole", axis="+z", u=0.0, v=0.0, dia=5.0),
        RefineFeature(name="b", kind="hole", axis="+z", u=10.0, v=0.0, dia=4.0),
    ]
    rep = RefineReport()
    rep.applied.append({"name": "a", "kind": "hole", "axis": "+z", "dia": 5.0})
    out = applied_refine_features(feats, rep)
    assert [f.name for f in out] == ["a"]


def test_end_to_end_applied_features_feed_analytic_step(tmp_path):
    """apply_refine → applied_refine_features → export_analytic_step."""
    b3d = pytest.importorskip("build123d")
    from cadx.cad.exporters import export_analytic_step

    mesh = _box()
    feats = [RefineFeature(name="b1", kind="hole", axis="+z", u=10.0, v=0.0, dia=6.0)]
    out, rep = apply_refine(mesh, feats, restore_keeps=False)
    assert len(rep.applied) == 1
    res = export_analytic_step(out, applied_refine_features(feats, rep), tmp_path / "e.step")
    assert res["step"] == "e.step"
    assert res["step_features"][0]["analytic"] is True
    solid = b3d.import_step(str(tmp_path / "e.step"))
    assert any(f.geom_type == b3d.GeomType.CYLINDER for f in solid.faces())


# --------------------------------------------------------------------------- #
# export_analytic_step_isolated                                                #
# --------------------------------------------------------------------------- #


def test_isolated_export_applies_feature(tmp_path):
    pytest.importorskip("build123d")
    from cadx.cad.exporters import export_analytic_step_isolated

    stl = tmp_path / "base.stl"
    _bored().export(stl)
    out_step = tmp_path / "out.step"
    res = export_analytic_step_isolated(
        stl,
        [{"name": "b1", "kind": "hole", "axis": "+z", "u": 10.0, "v": 0.0, "dia": 6.0}],
        out_step,
        timeout_s=120.0,
    )
    assert res["step"] == "out.step"
    assert [f["name"] for f in res["step_features"]] == ["b1"]
    assert out_step.exists()


def test_isolated_export_timeout_keeps_prior_checkpoint(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A hanging child is killed; earlier applied features still ship."""
    pytest.importorskip("build123d")
    import multiprocessing.context as mpctx

    import cadx.cad.exporters as ex

    class _FakeHangProcess:
        """Never actually starts — always reports alive so the parent
        exercises the terminate/skip path deterministically."""

        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            pass

        @property
        def exitcode(self) -> int | None:
            return None

    def make_process(_ctx_self, *a, **kw):  # type: ignore[no-untyped-def]
        args = kw.get("args") or ()
        feature = args[2] if len(args) > 2 else None
        name = feature.get("name") if isinstance(feature, dict) else getattr(feature, "name", "")
        if name == "b2":
            return _FakeHangProcess()
        return mpctx.SpawnProcess(*a, **kw)

    monkeypatch.setattr(mpctx.SpawnContext, "Process", make_process)

    stl = tmp_path / "base.stl"
    _bored().export(stl)
    out_step = tmp_path / "out.step"
    res = ex.export_analytic_step_isolated(
        stl,
        [
            {"name": "b1", "kind": "hole", "axis": "+z", "u": 10.0, "v": 0.0, "dia": 6.0},
            {"name": "b2", "kind": "hole", "axis": "+z", "u": -10.0, "v": 0.0, "dia": 6.0},
        ],
        out_step,
        timeout_s=120.0,
        per_feature_s=15.0,
    )
    assert res["step"] == "out.step"
    assert [f["name"] for f in res["step_features"]] == ["b1"]
    assert any("b2" in w and "exceeded" in w for w in res["warnings"])
    assert out_step.exists()


def test_isolated_export_no_valid_cut_keeps_faceted(tmp_path):
    """Without a resolvable pose no checkpoint exists → no STEP promotion."""
    pytest.importorskip("build123d")
    from cadx.cad.exporters import export_analytic_step_isolated

    stl = tmp_path / "base.stl"
    _box().export(stl)
    out_step = tmp_path / "out.step"
    res = export_analytic_step_isolated(
        stl,
        [{"name": "bad", "kind": "mystery"}],
        out_step,
        timeout_s=60.0,
    )
    assert res["step"] is None
    assert res["step_features"] == []
    assert any("bad" in w for w in res["warnings"])
    assert not out_step.exists()


# --------------------------------------------------------------------------- #
# face-swap fallback (boolean-dead B-Reps)                                     #
# --------------------------------------------------------------------------- #


def _inline_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run spawn-context targets synchronously in-process.

    Real spawn children are fresh interpreters — a monkeypatched
    ``_occ_cut`` failure would not reach them. Running the worker inline
    exercises the same target/args/JSON contract deterministically."""

    import multiprocessing.context as mpctx

    class _InlineProcess:
        def __init__(self, *a: object, **kw: object) -> None:
            self._target = kw["target"]
            self._args = kw["args"]

        def start(self) -> None:
            self._target(*self._args)

        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            pass

        @property
        def exitcode(self) -> int | None:
            return 0

    monkeypatch.setattr(
        mpctx.SpawnContext,
        "Process",
        lambda _ctx, *a, **kw: _InlineProcess(*a, **kw),
    )


def test_faceswap_bore_replaces_faceted_wall():
    """Unit-level: the wall patch is swapped for a true cylinder face."""
    b3d = pytest.importorskip("build123d")
    import cadx.cad.exporters as ex
    from cadx.cad.mesh_ops import mesh_to_solid

    mesh = _bored()
    solid = mesh_to_solid(mesh, max_faces=50000)
    res, note = ex._faceswap_bore(
        solid, {"name": "b1", "kind": "hole", "axis": "+z", "u": 10.0, "v": 0.0, "dia": 6.0}, mesh
    )
    assert res is not None and note is None
    cyl = [f for f in res.faces() if f.geom_type == b3d.GeomType.CYLINDER]
    assert cyl and cyl[0].radius == pytest.approx(3.0, abs=0.1)
    assert abs(res.volume - solid.volume) < solid.volume * 0.02


def test_faceswap_bore_no_wall_returns_none():
    """No faceted tunnel (uncut box) → honest refusal, never a fake solid."""
    pytest.importorskip("build123d")
    import cadx.cad.exporters as ex
    from cadx.cad.mesh_ops import mesh_to_solid

    mesh = _box()
    solid = mesh_to_solid(mesh, max_faces=50000)
    res, note = ex._faceswap_bore(
        solid, {"name": "b1", "kind": "hole", "axis": "+z", "u": 10.0, "v": 0.0, "dia": 6.0}, mesh
    )
    assert res is None and note is None


def test_isolated_faceswap_when_boolean_dead(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """_occ_cut forced to fail → the worker falls back to the face-swap and
    the feature still lands an analytic CYLINDER in the STEP."""
    b3d = pytest.importorskip("build123d")
    import cadx.cad.exporters as ex

    monkeypatch.setattr(ex, "_occ_cut", lambda *a, **kw: None)
    _inline_process(monkeypatch)

    stl = tmp_path / "base.stl"
    _bored().export(stl)
    out_step = tmp_path / "out.step"
    res = ex.export_analytic_step_isolated(
        stl,
        [{"name": "b1", "kind": "hole", "axis": "+z", "u": 10.0, "v": 0.0, "dia": 6.0}],
        out_step,
        timeout_s=120.0,
    )
    assert res["step"] == "out.step"
    assert [f["name"] for f in res["step_features"]] == ["b1"]
    assert any("face-swap" in w for w in res["warnings"])
    solid = b3d.import_step(str(out_step))
    cyl = [f for f in solid.faces() if f.geom_type == b3d.GeomType.CYLINDER]
    assert cyl and cyl[0].radius == pytest.approx(3.0, abs=0.05)


def test_isolated_faceswap_no_wall_keeps_honest_skip(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Boolean dead AND no faceted wall to swap → the honest skip is intact."""
    pytest.importorskip("build123d")
    import cadx.cad.exporters as ex

    monkeypatch.setattr(ex, "_occ_cut", lambda *a, **kw: None)
    _inline_process(monkeypatch)

    stl = tmp_path / "base.stl"
    _box().export(stl)
    out_step = tmp_path / "out.step"
    res = ex.export_analytic_step_isolated(
        stl,
        [{"name": "b1", "kind": "hole", "axis": "+z", "u": 10.0, "v": 0.0, "dia": 6.0}],
        out_step,
        timeout_s=120.0,
    )
    assert res["step"] is None
    assert res["step_features"] == []
    assert any("b1" in w for w in res["warnings"])
    assert not out_step.exists()


def test_faceswap_uses_measured_diameter():
    """A reamed-oversize tunnel (Ø10 spec → Ø10.4 measured) still lands."""
    b3d = pytest.importorskip("build123d")
    import cadx.cad.exporters as ex
    from cadx.cad.mesh_ops import mesh_to_solid

    mesh = _bored(dia=6.4)
    solid = mesh_to_solid(mesh, max_faces=50000)
    res, note = ex._faceswap_bore(
        solid,
        {
            "name": "b1",
            "kind": "hole",
            "axis": "+z",
            "u": 10.0,
            "v": 0.0,
            "dia": 6.0,
            "measured_dia_mm": 6.4,
        },
        mesh,
    )
    assert res is not None and note is None
    cyl = [f for f in res.faces() if f.geom_type == b3d.GeomType.CYLINDER]
    assert cyl and cyl[0].radius == pytest.approx(3.2, abs=0.1)


def test_norm_for_worker_forwards_measured_dia():
    """measured_dia_mm survives normalisation → the worker's radius guess."""
    pytest.importorskip("build123d")
    import cadx.cad.exporters as ex

    feat = ex._norm_for_worker(
        {
            "name": "b1",
            "kind": "hole",
            "axis": "+z",
            "u": 10.0,
            "v": 0.0,
            "dia": 6.0,
            "measured_dia_mm": 6.4,
        }
    )
    assert feat is not None and feat.measured_dia_mm == 6.4


def test_isolated_faceswap_retries_after_cut_timeout(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A spinning boolean child is killed; the swap-only retry child (no OCC
    boolean) still lands the analytic wall on the same checkpoint."""
    b3d = pytest.importorskip("build123d")
    import multiprocessing.context as mpctx

    import cadx.cad.exporters as ex

    class _Hang:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            pass

        @property
        def exitcode(self) -> int | None:
            return None

    class _Inline:
        def __init__(self, *a: object, **kw: object) -> None:
            self._target = kw["target"]
            self._args = kw["args"]

        def start(self) -> None:
            self._target(*self._args)

        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            pass

        @property
        def exitcode(self) -> int | None:
            return 0

    def make_process(_ctx: object, *a: object, **kw: object) -> object:
        args = kw.get("args") or ()
        # feature children get 6 args (implicit swap_only=False) or 7 for the
        # swap-only retry; the base checkpoint child gets 4. Only the boolean
        # child hangs — the retry must run.
        if len(args) == 6:
            return _Hang()
        return _Inline(*a, **kw)

    monkeypatch.setattr(mpctx.SpawnContext, "Process", make_process)

    stl = tmp_path / "base.stl"
    _bored().export(stl)
    out_step = tmp_path / "out.step"
    res = ex.export_analytic_step_isolated(
        stl,
        [{"name": "b1", "kind": "hole", "axis": "+z", "u": 10.0, "v": 0.0, "dia": 6.0}],
        out_step,
        timeout_s=120.0,
        per_feature_s=5.0,
    )
    assert res["step"] == "out.step"
    assert [f["name"] for f in res["step_features"]] == ["b1"]
    assert any("b1" in w and "exceeded" in w for w in res["warnings"])
    solid = b3d.import_step(str(out_step))
    cyl = [f for f in solid.faces() if f.geom_type == b3d.GeomType.CYLINDER]
    assert cyl and cyl[0].radius == pytest.approx(3.0, abs=0.05)


def test_isolated_faceswap_thread_marks_degraded(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Thread feature swapped as pilot-wall-only → applied but degraded."""
    b3d = pytest.importorskip("build123d")
    pytest.importorskip("bd_warehouse.thread")
    import cadx.cad.exporters as ex

    monkeypatch.setattr(ex, "_occ_cut", lambda *a, **kw: None)
    _inline_process(monkeypatch)

    stl = tmp_path / "base.stl"
    _bored(dia=6.75).export(stl)  # M8 tap drill
    out_step = tmp_path / "out.step"
    res = ex.export_analytic_step_isolated(
        stl,
        [
            {
                "name": "m8",
                "kind": "thread",
                "axis": "+z",
                "u": 10.0,
                "v": 0.0,
                "dia": 8.0,
                "thread_spec": "M8-1.25",
            }
        ],
        out_step,
        timeout_s=120.0,
    )
    assert res["step"] == "out.step"
    rec = res["step_features"][0]
    assert rec["name"] == "m8" and "degraded" in rec
    solid = b3d.import_step(str(out_step))
    assert any(f.geom_type == b3d.GeomType.CYLINDER for f in solid.faces())
