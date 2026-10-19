"""cad/spans.py — declared from-to connections must carry material."""

from build123d import Box, Pos

from cadx.cad import cadlib
from cadx.cad.spans import declared_spans, span_checks, span_coverage


def _features(position: str) -> list[dict]:
    return [{"name": "diagonal_brace", "type": "box", "position": position}]


def test_declared_spans_finds_from_to_features():
    feats = _features(
        "from (x=-55, z=35) on the leg to (x=+15, z=95) on the arm underside, centred in Y"
    )
    spans = declared_spans(feats)
    assert len(spans) == 1
    assert spans[0]["points"] == [[-55.0, 0.0, 35.0], [15.0, 0.0, 95.0]]


def test_declared_spans_skips_voids_and_single_points():
    assert declared_spans(
        [{"name": "h", "type": "through_hole", "position": "from x=0 to x=10"}]
    ) == []
    assert declared_spans(_features("at x=5, y=0")) == []


def test_span_coverage_connected_strut():
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    part = leg + arm + cadlib.strut((-55, 0, 35), (15, 0, 95), 10, 6)
    assert span_coverage(part, [-55, 0, 35], [15, 0, 95]) >= 0.85


def test_span_coverage_detached_strut_fails():
    """The live defect: strut built from x=-5 (leg_thickness misused as the
    x coordinate) floats between the leg and the arm — the declared path
    leg→arm is mostly air."""
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    part = leg + arm + cadlib.strut((-5, 0, 35), (15, 0, 95), 10, 6)
    assert span_coverage(part, [-55, 0, 35], [15, 0, 95]) < 0.85


def test_span_checks_end_to_end():
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    good = leg + arm + cadlib.strut((-55, 0, 35), (15, 0, 95), 10, 6)
    feats = _features(
        "from (x=-55, z=35) on the leg to (x=+15, z=95) on the arm underside, centred in Y"
    )
    res = span_checks(good, feats)
    assert len(res) == 1 and res[0]["ok"] and res[0]["coverage"] >= 0.85
    bad = leg + arm + cadlib.strut((-5, 0, 35), (15, 0, 95), 10, 6)
    res2 = span_checks(bad, feats)
    assert len(res2) == 1 and not res2[0]["ok"]
