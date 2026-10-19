"""Unit tests for the visual hero-prompt building blocks (image_prompt.py)."""

from __future__ import annotations

from cadx.agent.image_prompt import (
    hero_prompt_parts,
    rest_pose_hint,
    visual_features,
)
from cadx.llm.schemas import Feature, PartSpec

_DIM = {"value": 80.0, "unit": "mm", "source": "stated"}


def _spec() -> PartSpec:
    return PartSpec(
        name="mounting plate",
        length_x=dict(_DIM),
        width_y=dict(_DIM),
        height_z=dict(_DIM),
        features=[
            Feature(
                name="mounting_tab_left",
                type="other",
                description="mounting tab with rounded edge",
                position="at the left end of the plate",
            ),
            Feature(
                name="mounting_tab_right",
                type="other",
                # empty description -> falls back to the code name
            ),
            Feature(
                name="corner_holes",
                type="through_hole",
                description="mounting holes",
                count=2,
                axis="z",
                position="x=-30 and x=+30, y=0",
            ),
        ],
    )


def test_visual_features_prefers_description():
    out = visual_features(_spec())
    assert "mounting tab with rounded edge" in out
    assert "at the left end of the plate" in out


def test_visual_features_name_fallback_strips_underscores():
    out = visual_features(_spec())
    assert "mounting tab right" in out
    assert "mounting_tab_right" not in out


def test_visual_features_multi_hole_axis_template():
    out = visual_features(_spec())
    assert "2 round vertical through-holes" in out
    assert "mounting holes" in out  # description still appended
    assert "x=-30" in out  # position tail


def test_visual_features_caps_at_eight():
    spec = PartSpec(
        name="p",
        length_x=dict(_DIM),
        width_y=dict(_DIM),
        height_z=dict(_DIM),
        features=[Feature(name=f"rib_{i}", type="rib") for i in range(10)],
    )
    out = visual_features(spec)
    assert "rib 7" in out
    assert "rib 8" not in out
    assert "rib 9" not in out


def test_visual_features_char_budget():
    """t2i models truncate long prompts (Z-Image: 512 tokens) — the feature
    list must stay within a hard character budget instead of silently
    dropping tail features into the truncated zone."""
    from cadx.agent.image_prompt import FEATURES_MAX_CHARS

    long_desc = "x" * 200
    spec = PartSpec(
        name="p",
        summary="s",
        units="mm",
        length_x={"value": 10},
        width_y={"value": 10},
        height_z={"value": 10},
        features=[
            {"name": f"f{i}", "type": "boss", "description": long_desc, "position": "at origin"}
            for i in range(8)
        ],
        open_questions=[],
    )
    out = visual_features(spec)
    assert len(out) <= FEATURES_MAX_CHARS + 200  # last clause may straddle
    assert len(out) < len(", ".join(long_desc for _ in range(8)))


def test_visual_features_desc_clipped():
    spec = PartSpec(
        name="p",
        summary="s",
        units="mm",
        length_x={"value": 10},
        width_y={"value": 10},
        height_z={"value": 10},
        features=[{"name": "boss", "type": "boss", "description": "y" * 300}],
        open_questions=[],
    )
    out = visual_features(spec)
    assert len(out) < 120


def test_visual_features_empty():
    assert visual_features(None) == ""
    spec = PartSpec(name="p", length_x=dict(_DIM), width_y=dict(_DIM), height_z=dict(_DIM))
    assert visual_features(spec) == ""


def test_rest_pose_hint_non_empty():
    assert rest_pose_hint(_spec()) == "lying flat on the surface"


def test_hero_prompt_parts_keys():
    parts = hero_prompt_parts(_spec(), "a wall plate with two tabs", "mounting plate")
    assert set(parts) == {"subject", "intent", "features_visual", "pose", "negative"}
    assert parts["subject"] == "mounting plate"
    assert parts["intent"] == "a wall plate with two tabs"
    assert parts["features_visual"]
    assert parts["pose"]
    assert "watermark" in parts["negative"]


def test_hero_prompt_parts_intent_clamped_at_sentence_boundary():
    long_prompt = "This is a sentence. " * 60  # ~1200 chars
    parts = hero_prompt_parts(_spec(), long_prompt, "plate")
    assert len(parts["intent"]) <= 600
    assert parts["intent"].endswith("sentence")
    # clipped at a boundary: nothing mid-sentence leaks through
    assert "This is a sent" not in parts["intent"] or parts["intent"].endswith("sentence")


def test_hero_prompt_parts_empty_features():
    spec = PartSpec(name="p", length_x=dict(_DIM), width_y=dict(_DIM), height_z=dict(_DIM))
    parts = hero_prompt_parts(spec, "plain box", "box")
    assert parts["features_visual"] == ""
