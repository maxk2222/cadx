"""M9 — verified parametric feature edits (plain through-holes)."""

import math

import pytest
from build123d import Align, Box, Cone, Cylinder, Pos

from cadx.cad.editfeature import MAX_FACES_FOR_EDIT, edit_feature, find_editable_features

quiddity = pytest.importorskip("quiddity")


def _block_with_hole(diameter: float = 6.0, box: float = 40.0, height: float = 20.0):
    """40x40x20 mm box with a centred ``diameter`` through hole along Z."""
    return Box(box, box, height) - Cylinder(diameter / 2, height + 2)


def test_find_editable_features_lists_plain_through_hole():
    part = _block_with_hole(6.0)
    feats = find_editable_features(part)
    assert len(feats) == 1
    f = feats[0]
    assert f["feature_id"] == 0
    assert f["kind"] == "through_hole"
    assert f["diameter"] == pytest.approx(6.0)
    assert f["depth"] == pytest.approx(20.0)
    assert f["editable"] is True
    assert f["has_dependents"] is False
    assert len(f["axis"]) == 3 and len(f["opening"]) == 3


def test_enlarge_through_hole_passes_gate():
    part = _block_with_hole(6.0)
    res = edit_feature(part, 0, "diameter", 8.0)
    assert res["ok"], res.get("refusal")
    edited = res["shape"]
    expected = -(math.pi / 4) * (64 - 36) * 20  # annulus removed
    assert (edited.volume - part.volume) == pytest.approx(expected, abs=0.5)
    assert res["report"]["committed"] is True
    assert res["report"]["other_holes_unchanged"] is True
    # re-recognition on the result finds Ø8
    feats = find_editable_features(edited)
    assert len(feats) == 1 and feats[0]["diameter"] == pytest.approx(8.0)


def test_shrink_through_hole_defeatures_and_passes_gate():
    part = _block_with_hole(8.0)
    res = edit_feature(part, 0, "diameter", 6.0)
    assert res["ok"], res.get("refusal")
    edited = res["shape"]
    expected = (math.pi / 4) * (64 - 36) * 20  # material restored
    assert (edited.volume - part.volume) == pytest.approx(expected, abs=0.5)
    feats = find_editable_features(edited)
    assert len(feats) == 1 and feats[0]["diameter"] == pytest.approx(6.0)


def test_other_holes_left_untouched():
    part = Box(40, 40, 20) - Pos(5, 5, 0) * Cylinder(3, 22) - Pos(-10, -8, 0) * Cylinder(2, 22)
    feats = find_editable_features(part)
    assert len(feats) == 2
    target = next(f for f in feats if f["diameter"] == pytest.approx(6.0))
    res = edit_feature(part, target["feature_id"], "diameter", 7.0)
    assert res["ok"], res.get("refusal")
    after = find_editable_features(res["shape"])
    assert len(after) == 2
    assert any(f["diameter"] == pytest.approx(7.0) for f in after)
    assert any(f["diameter"] == pytest.approx(4.0) for f in after)


def test_unknown_feature_id_refuses():
    part = _block_with_hole(6.0)
    res = edit_feature(part, 42, "diameter", 8.0)
    assert res["ok"] is False
    assert "unknown feature_id" in res["refusal"]
    assert "shape" not in res


def test_countersunk_hole_refused_as_dependent():
    part = _block_with_hole(4.0)
    part -= Pos(0, 0, 10) * Cone(4.5, 2, 2.25, align=(Align.CENTER, Align.CENTER, Align.MAX))
    feats = find_editable_features(part)
    assert len(feats) == 1
    assert feats[0]["has_dependents"] is True
    assert feats[0]["editable"] is False
    res = edit_feature(part, feats[0]["feature_id"], "diameter", 6.0)
    assert res["ok"] is False
    assert "dependent" in res["refusal"]
    assert "shape" not in res


def test_over_complex_input_refused():
    part = _block_with_hole(6.0)
    # faceted hybrids exceed the analytic cap — refuse instead of scanning
    part.faces = lambda: [None] * (MAX_FACES_FOR_EDIT + 1)
    res = edit_feature(part, 0, "diameter", 8.0)
    assert res["ok"] is False
    assert "faces" in res["refusal"]
    assert "shape" not in res


def test_unchanged_and_invalid_values_refuse():
    part = _block_with_hole(6.0)
    for bad in (6.0, 0.0, -3.0, float("nan"), float("inf"), "abc", None):
        res = edit_feature(part, 0, "diameter", bad)
        assert res["ok"] is False, bad
        assert "shape" not in res


def test_unsupported_parameter_refuses():
    part = _block_with_hole(6.0)
    res = edit_feature(part, 0, "depth", 10.0)
    assert res["ok"] is False
    assert "unsupported parameter" in res["refusal"]


def test_blind_hole_not_editable():
    part = Box(40, 40, 20) - Pos(0, 0, -10) * Cylinder(
        3, 8, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    feats = find_editable_features(part)
    if feats:  # recognised but must never be editable
        assert all(f["editable"] is False for f in feats)
        res = edit_feature(part, feats[0]["feature_id"], "diameter", 8.0)
        assert res["ok"] is False


def test_multiple_solids_refused():
    part = _block_with_hole(6.0)
    two = Pos(100, 0, 0) * part  # moved copy
    from build123d import Compound

    compound = Compound(children=[part, two])
    res = edit_feature(compound, 0, "diameter", 8.0)
    assert res["ok"] is False
    assert "shape" not in res
