"""Suite loading / validation: the 3 shipped cases + schema rules."""

from __future__ import annotations

import json

import pytest

from cadx.evalkit.paths import SUITE_DIR
from cadx.evalkit.suite import (
    ASSET_STAGES,
    CHILD_VERSION_STAGES,
    SAME_VERSION_STAGES,
    EvalCase,
    EvalStage,
    load_case,
    load_suite,
)


def test_shipped_suite_loads_and_validates():
    cases = load_suite()
    assert [c.name for c in cases] == [
        "l_bracket_rib",
        "m10_csk_screw",
        "m8_tapped_plate",
        "helical_pinion_shaft",
        "sprocket_520_45t",
        "robot_arm_mount",
    ]
    assert [c.difficulty for c in cases] == [1, 1, 1, 2, 2, 3]
    for c in cases:
        assert c.project.startswith("eval:")
        assert c.stages[0].type in ("generate", "meshgen")
        for s in c.stages:
            assert s.type in CHILD_VERSION_STAGES | SAME_VERSION_STAGES | ASSET_STAGES
            for g in s.gates:
                assert g.get("type"), f"gate without type in {c.name}/{s.id}"


def test_case_entities_split():
    """Every stage type maps to exactly one entity class."""
    all_types = CHILD_VERSION_STAGES | SAME_VERSION_STAGES | ASSET_STAGES
    assert not (CHILD_VERSION_STAGES & SAME_VERSION_STAGES)
    assert not (CHILD_VERSION_STAGES & ASSET_STAGES)
    # fem/reexport write onto the same version — never a child
    assert {"fem", "reexport"} <= SAME_VERSION_STAGES
    assert {"generate", "optimize", "meshgen"} <= CHILD_VERSION_STAGES
    assert {"imagegen", "imageedit"} <= ASSET_STAGES
    for c in load_suite():
        for s in c.stages:
            assert s.type in all_types


def test_enabled_stages_filter():
    case = load_case("robot_arm_mount")
    assert case is not None
    enabled = case.enabled_stages()
    assert [s.id for s in enabled] == ["codegen", "fem", "topopt", "refine"]
    refine = enabled[-1]
    assert refine.type == "refine"
    assert refine.load_cases and refine.material == "aluminum"
    assert {g.get("type") for g in refine.gates} >= {"holes", "fem"}
    only_fem = case.enabled_stages({"fem"})
    assert [s.id for s in only_fem] == ["fem"]
    by_type = case.enabled_stages({"optimize"})
    assert [s.id for s in by_type] == ["topopt"]


def test_load_case_unknown_returns_none():
    assert load_case("does_not_exist") is None


def test_stage_validator_requirements():
    with pytest.raises(ValueError, match="unknown stage type"):
        EvalStage(id="x", type="bogus")
    with pytest.raises(ValueError, match="needs a prompt"):
        EvalStage(id="x", type="generate")
    with pytest.raises(ValueError, match="needs `params`"):
        EvalStage(id="x", type="params")
    ok = EvalStage(id="x", type="generate", prompt="box")
    assert ok.enabled is True


def test_case_validator_rules(tmp_path):
    bad = {
        "name": "bad",
        "stages": [
            {"id": "a", "type": "fem"},  # first stage must be generate/meshgen
        ],
    }
    with pytest.raises(ValueError, match="first stage"):
        EvalCase.model_validate(bad)
    dup = {
        "name": "bad2",
        "stages": [
            {"id": "a", "type": "generate", "prompt": "p"},
            {"id": "a", "type": "fem"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate"):
        EvalCase.model_validate(dup)
    fwd = {
        "name": "bad3",
        "stages": [
            {"id": "a", "type": "generate", "prompt": "p", "from_stage": "later"},
            {"id": "later", "type": "fem"},
        ],
    }
    with pytest.raises(ValueError, match="earlier stage"):
        EvalCase.model_validate(fwd)


def test_suite_files_present():
    files = sorted(p.name for p in SUITE_DIR.glob("*.json"))
    assert files == [
        "helical_pinion_shaft.json",
        "l_bracket_rib.json",
        "m10_csk_screw.json",
        "m8_tapped_plate.json",
        "robot_arm_mount.json",
        "sprocket_520_45t.json",
    ]
    for p in SUITE_DIR.glob("*.json"):
        json.loads(p.read_text(encoding="utf-8"))  # valid JSON
