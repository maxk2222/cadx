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


def _vessel_spec(with_feet: bool = True) -> PartSpec:
    spec = _spec()
    spec.name = "domed pressure tank"
    spec.summary = "closed domed vessel"
    body = Feature(
        name="sealed tank body",
        type="other",
        dims={"diameter": 80.0, "height": 120.0, "wall_thickness": 3.0},
        count=1,
    )
    feats = [body]
    if with_feet:
        feats += [
            Feature(
                name="foot lug",
                type="other",
                dims={"width": 18.0, "thickness": 6.0, "depth_total": 20.0},
                count=3,
            ),
            Feature(name="foot bolt hole", type="hole", dims={"diameter": 6.5}, count=3),
        ]
    spec.features = feats
    return spec


def test_part_family_hint_extracts_feet_params():
    """The sealed-vessel hint must carry the spec's REAL foot/bolt geometry —
    a static feet=0 template made the model disable the helper's feet and
    hand-build lugs it embeds into the cavity."""
    from cadx.agent.context import part_family_hint

    hint = part_family_hint(_vessel_spec(with_feet=True))
    assert "sealed_tank" in hint
    assert "feet=3" in hint
    assert "foot_w=18" in hint
    assert "foot_t=6" in hint
    assert "foot_h=20" in hint
    assert "foot_hole_d=6.5" in hint


def test_part_family_hint_no_feet_falls_back():
    from cadx.agent.context import part_family_hint

    hint = part_family_hint(_vessel_spec(with_feet=False))
    assert "sealed_tank" in hint
    assert "feet=0" in hint
    assert "foot_hole_d=0" in hint


def test_cadlib_filter_propeller_surfaces_radial_blade():
    """A propeller spec must surface the reviewed radial_blade primitive —
    it exists in cadlib but was invisible to the tag filter (no tags line),
    so the model hand-rolled twisted blade lofts and failed."""
    spec = _spec(
        [
            Feature(name="blade", type="airfoil", dims={}, count=5),
            Feature(name="hub", type="boss", dims={}, count=1),
        ]
    )
    spec.name = "5-blade marine propeller"
    spec.summary = "propeller with hub and blades"
    text = codegen_system(plan=None, spec=spec, fresh=False)
    assert "### radial_blade" in text


def test_cadlib_filter_arch_and_strut_tags():
    """arch_bridge and strut carry their own trigger vocabulary — a spec
    naming 'arch bridge' or 'diagonal strut' must surface them."""
    from cadx.agent.context import _cadlib_sections

    arch = _spec([Feature(name="handle", type="other", dims={}, count=1)])
    arch.name = "arched carrying handle bridge"
    assert "### arch_bridge" in _cadlib_sections(arch)

    strut = _spec([Feature(name="diagonal strut", type="rib", dims={}, count=1)])
    strut.name = "frame with diagonalstrebe"
    assert "### strut" in _cadlib_sections(strut)


def test_part_family_hint_non_vessel_empty():
    from cadx.agent.context import part_family_hint

    assert part_family_hint(_spec([Feature(name="rib", type="rib", dims={})])) == ""
