"""Eval runner: expected-feature checks and benchmark loading (T5.1/T5.2)."""

from cadx.eval.runner import _features_ok, load_benchmarks


def test_features_ok_none_when_no_expectations() -> None:
    assert _features_ok([], {"total_holes": 0}) is None


def test_features_ok_hole_count_and_dia() -> None:
    feats = {
        "total_holes": 3,
        "hole_groups": [
            {"diameter": 5.5, "count": 2},
            {"diameter": 22.2, "count": 1},
        ],
    }
    assert _features_ok([{"type": "hole", "count": 2, "dia": 5.5}], feats) is True
    assert _features_ok([{"type": "hole", "count": 3, "dia": 5.5}], feats) is False
    assert _features_ok([{"type": "hole", "count": 3}], feats) is True


def test_features_ok_countersink_fillet_chamfer() -> None:
    feats = {"countersinks": 3, "counterbores": 0, "rounds": [3.0], "chamfer_faces": 0}
    assert _features_ok([{"type": "countersink", "count": 3}], feats) is True
    assert _features_ok([{"type": "counterbore", "count": 1}], feats) is False
    assert _features_ok([{"type": "fillet"}], feats) is True
    assert _features_ok([{"type": "chamfer"}], feats) is False


def test_features_ok_missing_features_dict() -> None:
    assert _features_ok([{"type": "hole", "count": 1}], None) is False


def test_load_benchmarks_parses_new_fields() -> None:
    bms = {b.name: b for b in load_benchmarks()}
    lb = bms["l_bracket_cad"]
    assert lb.gen_mode == "cad"
    assert lb.expected_features[0]["type"] == "hole"
    assert lb.expect_watertight is True
    fs = bms["firesteel_auto"]
    assert fs.max_prompt_tokens == 4_000_000
    assert len(fs.images) == 3
    assert bms["knife_scale_hybrid"].gen_mode == "hybrid"
    assert bms["grip_mesh"].expect_watertight is True
