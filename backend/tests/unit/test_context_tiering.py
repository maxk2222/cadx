"""Staged reference docs: spec/plan-driven filtering shrinks the system prompt."""

from __future__ import annotations

from cadx.agent.context import codegen_system
from cadx.llm.schemas import Dim, Feature, ModelingPlan, PartSpec


def _spec(features: list[Feature] | None = None) -> PartSpec:
    d = lambda v: Dim(value=v, unit="mm", source="stated")  # noqa: E731
    return PartSpec(
        name="plate",
        summary="",
        units="mm",
        length_x=d(60),
        width_y=d(40),
        height_z=d(4),
        features=features or [],
        image_roles=[],
    )


def test_tiered_prompt_is_shorter():
    spec = _spec([Feature(name="mounting hole", type="hole", dims={}, count=2)])
    tiered = codegen_system(plan=None, spec=spec, fresh=False)
    full = codegen_system()  # no spec/plan → everything
    assert len(tiered) < 0.8 * len(full)


def test_standard_parts_include_bd_warehouse():
    spec = _spec([Feature(name="hole", type="hole", dims={}, count=1)])
    plan = ModelingPlan(
        approach="cad",
        steps=["base box"],
        standard_parts=["ClearanceHole"],
    )
    tiered = codegen_system(plan=plan, spec=spec, fresh=False)
    # unique to the bd_warehouse reference file
    assert "SingleRowDeepGrooveBallBearing" in tiered
    plain = codegen_system(plan=None, spec=spec, fresh=False)
    assert "SingleRowDeepGrooveBallBearing" not in plain


def test_cadlib_filter_keeps_relevant_helpers():
    spec = _spec([Feature(name="rib", type="rib", dims={}, count=2)])
    text = codegen_system(plan=None, spec=spec, fresh=False)
    assert "### rib" in text
    assert "### through_hole" in text  # always kept
    assert "### knurl_ring" not in text
    # ...but still discoverable through the helper index
    assert "knurl_ring(" in text
    # no spec → full reference
    assert "### knurl_ring" in codegen_system()


def test_cadlib_filter_slot_means_oblong_slot():
    """The observed L-bracket failure: a 'slot' feature must surface
    oblong_slot — hand-rolled slots were the exec failure."""
    spec = _spec([Feature(name="Langloch", type="slot", dims={}, count=1)])
    text = codegen_system(plan=None, spec=spec, fresh=False)
    assert "### oblong_slot" in text


def test_cadlib_filter_fillet_surfaces_safe_fillet():
    spec = _spec([Feature(name="Innenradius", type="fillet", dims={}, count=1)])
    text = codegen_system(plan=None, spec=spec, fresh=False)
    assert "### safe_fillet" in text


def test_cadlib_filter_uses_spec_name_and_summary():
    spec = _spec([])
    spec.name = "L-Winkel"
    spec.summary = "bracket mit rippe"
    spec.features = [Feature(name="Grundplatte", type="base", dims={}, count=1)]
    text = codegen_system(plan=None, spec=spec, fresh=False)
    assert "### l_bracket" in text
    assert "### rib" in text


def test_cadlib_filter_flange_hole_circle():
    spec = _spec([Feature(name="Lochkreis", type="hole", dims={}, count=6)])
    text = codegen_system(plan=None, spec=spec, fresh=False)
    assert "### hole_grid" in text or "### through_hole" in text


def test_measured_features_block_renders_slot_fields():
    """P8: measured slot groups carry through/elongation axes and w x l —
    render them so the patcher sees exact geometry, not a round-hole
    approximation."""
    from cadx.agent.context import measured_features_block

    block = measured_features_block(
        {
            "hole_groups": [
                {
                    "id": "s1",
                    "diameter": 8.0,
                    "count": 1,
                    "axis": "+z",
                    "depth_class": "through",
                    "centers": [[3.0, -2.0]],
                    "elongated": True,
                    "through_axis": "+z",
                    "elong_axis": "+x",
                    "width_mm": 4.0,
                    "length_mm": 12.0,
                }
            ],
            "total_holes": 0,
        }
    )
    assert "slot s1" in block
    assert "through_axis +z" in block
    assert "elongated along +x" in block
    assert "4x12 mm" in block
    assert "(3, -2)" in block


def test_measured_features_block_legacy_slot_line():
    """Slot groups without the new measured fields keep the generic
    elongated-void line."""
    from cadx.agent.context import measured_features_block

    block = measured_features_block(
        {
            "hole_groups": [
                {
                    "diameter": 8.0,
                    "count": 1,
                    "axis": "+z",
                    "depth_class": "through",
                    "elongated": True,
                }
            ],
            "total_holes": 0,
        }
    )
    assert "elongated void" in block
    assert "through_axis" not in block
