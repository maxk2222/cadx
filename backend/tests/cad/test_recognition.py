"""quiddity recognition enrichment tests (optional dep, real geometry).

quiddity is an optional dependency — tests that exercise recognition results
skip when it is not installed, and the degradation tests monkeypatch it away
explicitly.
"""

from __future__ import annotations

import pytest
from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    Cylinder,
    Hole,
    Locations,
    PolarLocations,
    chamfer,
    fillet,
)

import cadx.cad.recognition as recognition
from cadx.cad.features import (
    MeasuredFeatures,
    compare_features,
    extract_features,
)
from cadx.llm.schemas import Feature

requires_quiddity = pytest.mark.skipif(
    recognition.quiddity is None, reason="quiddity not installed"
)


def _flange():
    """Flange: 6x Ø6 bolt circle on a Ø56 PCD + centre bore + hub."""
    with BuildPart() as p:
        Cylinder(radius=40, height=8)
        Cylinder(radius=15, height=20)
        with PolarLocations(radius=28, count=6):
            Hole(radius=3)
        Hole(radius=6)
    return p.part


def _bored_standoffs():
    """Plate with four bored standoffs (bored bosses)."""
    with BuildPart() as p:
        Box(80, 60, 6)
        with Locations((-25, -15), (25, -15), (-25, 15), (25, 15)):
            Cylinder(radius=6, height=12)
        with Locations((-25, -15), (25, -15), (-25, 15), (25, 15)):
            Hole(radius=2.5, depth=18)
    return p.part


@requires_quiddity
def test_bolt_circle_becomes_hole_pattern():
    rec = recognition.build_recognition_evidence(_flange())
    assert rec, "recognition returned no evidence"
    assert len(rec["hole_patterns"]) == 1
    pat = rec["hole_patterns"][0]
    assert pat["kind"] == "bolt_circle"
    assert pat["count"] == 6
    assert pat["diameter"] == pytest.approx(56.0, abs=0.1)
    assert pat["member_diameter"] == pytest.approx(6.0, abs=0.1)
    assert len(pat["members"]) == 6


@requires_quiddity
def test_extract_features_merges_patterns():
    m = extract_features(_flange())
    assert m.recognition_ran
    assert len(m.hole_patterns) == 1
    assert m.hole_patterns[0].kind == "bolt_circle"
    assert m.hole_patterns[0].count == 6
    # analytic scan still owns the hole count — no double counting
    assert m.total_holes == 7


@requires_quiddity
def test_bored_bosses_detected():
    m = extract_features(_bored_standoffs())
    assert len(m.bored_bosses) == 4
    b = m.bored_bosses[0]
    assert b.boss_diameter == pytest.approx(12.0, abs=0.2)
    assert b.bore_diameter == pytest.approx(5.0, abs=0.1)
    assert b.axis.lstrip("+-") == "z"


@requires_quiddity
def test_chamfer_and_fillet_records():
    with BuildPart() as p:
        Box(60, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), radius=3)
    m = extract_features(p.part)
    assert m.fillets, "quiddity fillet records missing"
    assert any(f.radius == pytest.approx(3.0, abs=0.05) for f in m.fillets)

    with BuildPart() as p2:
        Box(60, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
        chamfer(p2.edges().filter_by(Axis.Z), length=1)
    m2 = extract_features(p2.part)
    assert m2.chamfers, "quiddity chamfer records missing"
    assert m2.chamfer_faces >= 1


@requires_quiddity
@requires_quiddity
def test_slot_array_pattern_detected():
    """Three equidistant oblong through-slots surface as a slot_pattern."""
    from build123d import BuildSketch, Mode, Plane, SlotOverall, extrude

    with BuildPart() as p:
        Box(120, 60, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
        for y in (-20, 0, 20):
            with BuildSketch(Plane.XY.offset(8)), Locations((0, y)):
                SlotOverall(30, 6)
            extrude(amount=-10, mode=Mode.SUBTRACT)
    m = extract_features(p.part)
    assert m.slot_patterns, "quiddity slot array missing"
    pat = m.slot_patterns[0]
    assert pat.count == 3
    assert pat.pitch == pytest.approx(20.0, abs=0.1)
    assert pat.axis.lstrip("+-") == "y"


def test_recognition_does_not_double_count():
    """Union merge: the recogniser's own hole/chamfer hits must not inflate
    the analytic measurements."""
    part = _bored_standoffs()
    with_rec = extract_features(part)
    monkey = recognition.quiddity
    try:
        recognition.quiddity = None
        without_rec = extract_features(part)
    finally:
        recognition.quiddity = monkey
    assert with_rec.total_holes == without_rec.total_holes
    assert with_rec.chamfer_faces >= without_rec.chamfer_faces
    assert with_rec.countersinks >= without_rec.countersinks
    assert without_rec.recognition_ran is False


def test_missing_quiddity_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(recognition, "quiddity", None)
    assert recognition.build_recognition_evidence(_flange()) == {}
    # and the pipeline merge is a clean no-op
    m = extract_features(_flange())
    assert m.recognition_ran is False
    assert m.hole_patterns == []
    assert m.bored_bosses == []
    assert m.total_holes == 7  # analytic scan unaffected


@requires_quiddity
def test_large_shape_skipped(monkeypatch):
    monkeypatch.setattr(recognition, "_MAX_RECOGNITION_FACES", 1)
    assert recognition.build_recognition_evidence(_flange()) == {}


@requires_quiddity
def test_recognition_failure_degrades(monkeypatch):
    monkeypatch.setattr(
        recognition.quiddity,
        "build_recognition_result",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # budget_s=0 pins the in-process path — a monkeypatched quiddity cannot
    # cross the spawned child's import boundary
    assert recognition.build_recognition_evidence(_flange(), budget_s=0) == {}


@requires_quiddity
def test_recognition_budget_timeout_degrades():
    """A child that overruns its budget is hard-killed and reported as {} —
    no spawned interpreter can import + run quiddity inside 50 ms, so this
    exercises the terminate path deterministically."""
    assert recognition.build_recognition_evidence(_flange(), budget_s=0.05) == {}


@requires_quiddity
def test_measured_features_round_trip():
    m = extract_features(_flange())
    dump = m.model_dump()
    back = MeasuredFeatures(**dump)
    assert back == m
    # nested pattern record survives the dict round-trip intact
    assert back.hole_patterns[0].kind == "bolt_circle"
    assert back.hole_patterns[0].members[0] == pytest.approx([-14.0, -24.249, 4.0])


@requires_quiddity
def test_pattern_spec_check():
    m = extract_features(_flange())
    ok = compare_features([Feature(name="bolt circle", type="pattern", count=6)], m)
    assert ok == []
    short = compare_features([Feature(name="bolt circle", type="pattern", count=8)], m)
    assert any(f.severity == "major" for f in short)


def test_pattern_check_silent_without_recognition():
    """No quiddity → no unfounded 'no pattern' findings."""
    m = MeasuredFeatures(total_holes=6, recognition_ran=False)
    spec = [Feature(name="bolt circle", type="pattern", count=6)]
    assert all("pattern" not in f.feature for f in compare_features(spec, m))


def _fork_ctx_patcher(monkeypatch):
    """Force the budget child onto the FORK context so a monkeypatched
    quiddity reaches it (spawn children are fresh interpreters and can't see
    the patch). fork availability is a given on the POSIX test envs."""
    import multiprocessing as mp

    real_get_context = mp.get_context
    monkeypatch.setattr(
        mp, "get_context", lambda method=None: real_get_context("fork")
    )


def test_budget_hard_kills_slow_recognition(monkeypatch):
    """A recognition run past its budget is terminated, not awaited — the
    advisory pass may never grind the pipeline (observed: quiddity on a
    929-face thread solid burned >6 min inside the analyze band)."""
    import time
    import types

    _fork_ctx_patcher(monkeypatch)

    def slow_result(_shape):
        time.sleep(30)
        return object()

    monkeypatch.setattr(
        recognition, "quiddity", types.SimpleNamespace(build_recognition_result=slow_result)
    )
    t = time.monotonic()
    ev = recognition.build_recognition_evidence(Box(10, 10, 10), budget_s=2.0)
    assert ev == {}
    assert time.monotonic() - t < 12


@requires_quiddity
def test_budgeted_child_returns_real_evidence():
    """The fork path is lossless: a small shape returns the full evidence dict
    from the child, not an empty fallback."""
    ev = recognition.build_recognition_evidence(Box(10, 10, 10), budget_s=20)
    assert isinstance(ev, dict)
    # keys exist even when the lists are empty — the dict shape is the contract
    assert "hole_patterns" in ev and "fillet_radii" in ev


def test_cancelled_budget_returns_promptly(monkeypatch):
    """Cancel flag wins over the budget — a cooperative cancellation must not
    wait for the full budget before dropping the advisory pass."""
    import time
    import types

    _fork_ctx_patcher(monkeypatch)

    def slow_result(_shape):
        time.sleep(30)
        return object()

    monkeypatch.setattr(
        recognition, "quiddity", types.SimpleNamespace(build_recognition_result=slow_result)
    )
    t = time.monotonic()
    ev = recognition.build_recognition_evidence(
        Box(10, 10, 10), budget_s=60.0, cancelled=lambda: True
    )
    assert ev == {}
    assert time.monotonic() - t < 15
