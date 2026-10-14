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
    # no spec → full reference
    assert "### knurl_ring" in codegen_system()
