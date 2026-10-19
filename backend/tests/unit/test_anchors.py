"""Tests for cadx.agent.anchors — image/mesh hole reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

from cadx.agent.anchors import build_anchors
from cadx.agent.spechost import host_body, host_range, host_span, parse_ranges
from cadx.llm.schemas import Dim, Feature, FeatureCenter, PartSpec

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "specs"


def _bracket_spec() -> PartSpec:
    """Real CAD-intake output of the bracket E2E run (ground truth fixture)."""
    return PartSpec.model_validate(
        json.loads((_FIXTURES / "bracket_strut_cad.json").read_text(encoding="utf-8"))
    )


def _spec() -> PartSpec:
    d = lambda v: Dim(value=v, unit="mm", source="stated")  # noqa: E731
    return PartSpec(
        name="grip",
        summary="",
        units="mm",
        length_x=d(120),
        width_y=d(38),
        height_z=d(8),
        features=[
            Feature(name="pin hole", type="hole", dims={"diameter": 5.0}, count=3),
        ],
        image_roles=[],
    )


def _meas(holes, outline) -> dict:
    return {
        "A": {
            "role": "top",
            "bbox_mm": [120, 38],
            "outline_mm": outline,
            "holes": holes,
            "confidence": 0.9,
            "authoritative": True,
        }
    }


_TRAP = [[-60, -20], [60, -14], [60, 14], [-60, 20], [-60, -20]]


def test_build_anchors_mirror_and_confirm():
    # image outline is a trapezoid (wider at -x); the mirrored mesh outline
    # matches it better -> mirror_x
    img = _meas(
        [
            {"x": -45, "y": 0, "dia": 5.0},
            {"x": 0, "y": 0, "dia": 5.0},
            {"x": 45, "y": 0, "dia": 5.0},
        ],
        _TRAP,
    )
    mesh = {
        "holes": [
            {"x": 45, "y": 0.5, "dia": 5.1, "kind": "tunnel"},
            {"x": 0, "y": 0, "dia": 5.0, "kind": "tunnel"},
            {"x": -45, "y": -0.5, "dia": 4.9, "kind": "tunnel"},
        ],
        "outline_mm": [[-60, -14], [60, -20], [60, 20], [-60, 14], [-60, -14]],
    }
    a = build_anchors(_spec(), img, mesh)
    assert a.mirror_x is True
    assert len(a.holes) == 3
    assert all(h.source == "image+mesh" for h in a.holes)
    # spec Ø5.0 wins over measured 5.0/5.1 — count matches
    assert all(h.dia == 5.0 for h in a.holes)


def test_build_anchors_mesh_only():
    mesh = {
        "holes": [
            {"x": 10, "y": 1, "dia": 5.0, "kind": "tunnel"},
            {"x": -10, "y": -1, "dia": 6.0, "kind": "tunnel"},
        ],
        "outline_mm": [],
    }
    a = build_anchors(_spec(), {}, mesh)
    assert len(a.holes) == 2
    assert all(h.source == "mesh" for h in a.holes)
    assert a.mirror_x is False


def test_build_anchors_image_only():
    img = _meas([{"x": -45, "y": 2, "dia": 5.2}], _TRAP)
    a = build_anchors(_spec(), img, None)
    assert len(a.holes) == 1
    assert a.holes[0].source == "image"
    assert a.holes[0].dia == 5.0  # rounded to 0.5
    assert a.image_outline == _TRAP


# --------------------------------------------------------------------------- #
# spec-position anchors (declared intent, no image/mesh evidence needed)      #
# --------------------------------------------------------------------------- #


def _plate_spec(features) -> PartSpec:
    """90x40x14 plate centred at origin: x in [-45, 45], y in [-20, 20]."""
    d = lambda v: Dim(value=v, unit="mm", source="stated")  # noqa: E731
    return PartSpec(
        name="plate",
        summary="",
        units="mm",
        length_x=d(90),
        width_y=d(40),
        height_z=d(14),
        features=features,
        image_roles=[],
    )


def test_spec_anchors_without_image_or_mesh():
    """The hero-image-showed-no-holes case: declared end-relative positions
    must still become anchors at the correct centred coordinates."""
    spec = _plate_spec(
        [
            Feature(
                name="bore A",
                type="hole",
                dims={"diameter": 10.0},
                position="20mm from -X end, centered in Y",
            ),
            Feature(
                name="bore B",
                type="hole",
                dims={"diameter": 10.0},
                position="75mm from -X end, centered in Y",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert len(a.holes) == 2
    # -45+20 = -25, -45+75 = +30 — index == spec declaration order
    assert [(h.x, h.y, h.name) for h in a.holes] == [
        (-25.0, 0.0, "bore A"),
        (30.0, 0.0, "bore B"),
    ]
    assert all(h.source == "spec" and h.axis == "+z" for h in a.holes)
    assert [h.index for h in a.holes] == [1, 2]


def test_spec_anchors_fine_grained_hole_types():
    """Intake emits fine-grained/DE types — through_hole, threaded_bore etc.
    must still produce anchors (regression: 'Bohrung 1' type through_hole
    yielded no anchor, so no keep-ring and no re-cut)."""
    spec = _plate_spec(
        [
            Feature(
                name="Bohrung 1",
                type="through_hole",
                dims={"diameter": 10.0},
                position="20 mm from the -x end, centered in y (x=-25, y=0)",
            ),
            Feature(
                name="Bohrung 2",
                type="through_hole",
                dims={"diameter": 10.0},
                position="75 mm from the -x end, centered in y (x=30, y=0)",
            ),
            Feature(
                name="M8 Gewindebohrung",
                type="threaded_bore",
                dims={"diameter": 8.0, "depth": 16.0},
                position="8 mm from the +x end, centered in y (x=37, y=0, z=7)",
                axis="x",
                standard="M8-1.25 ISO thread, tap hole Ø6.8",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert [(h.axis, h.x, h.dia, h.kind) for h in a.holes] == [
        ("+z", -25.0, 10.0, "hole"),
        ("+z", 30.0, 10.0, "hole"),
        ("+x", 37.0, 8.0, "thread"),
    ]
    assert a.holes[2].bolt_size == "M8"


def test_spec_anchor_multi_offset_feature():
    """One count-2 feature declaring two end offsets yields both positions."""
    spec = _plate_spec(
        [
            Feature(
                name="bores",
                type="hole",
                dims={"diameter": 10.0},
                count=2,
                position="20 and 75mm from -X end, centred in Y",
            ),
            Feature(
                name="thread",
                type="thread",
                dims={"diameter": 8.0},
                position="8mm from +X end, centered in y",
                standard="M8",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert [(h.x, h.y, h.name) for h in a.holes] == [
        (-25.0, 0.0, "bores"),
        (30.0, 0.0, "bores"),
        (37.0, 0.0, "thread"),  # +45-8 — hi-side end hint
    ]
    assert a.holes[2].kind == "thread"
    assert a.holes[2].bolt_size == "M8"
    assert [h.index for h in a.holes] == [1, 2, 3]


def test_spec_anchor_side_face_threaded_hole():
    """An axis=x declared bore anchors on the y/z plane — the end offset on
    the bore's own axis becomes the axial centre (x=+37) plus a depth hint;
    the unresolved in-plane z defaults to mid-height flagged spec-est."""
    spec = _plate_spec(
        [
            Feature(
                name="M8 threaded hole",
                type="threaded_hole",
                axis="x",
                dims={"diameter": 6.8},
                position="8 mm from +X end face, centered in y",
                thread_spec="M8",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert len(a.holes) == 1
    h = a.holes[0]
    assert h.axis == "+x" and h.kind == "thread" and h.bolt_size == "M8"
    assert h.pos == [37.0, 0.0, 7.0]  # axial x, in-plane y=0, z=mid 14/2
    assert h.z == 37.0 and h.depth_mm == 16.0
    assert h.source == "spec-est"  # in-plane z was mid-defaulted


def test_spec_anchor_german_axis_phrase():
    """'in X-Richtung' declares the bore axis like 'x-axis' — German spec
    text must resolve identically (a face token '+X-Seite' works too)."""
    spec = _plate_spec(
        [
            Feature(
                name="M8 Gewindebohrung",
                type="threaded_hole",
                dims={"diameter": 6.8, "depth": 16},
                position="an der +X-Seite (x=+45 mm), zentriert in y und z, in X-Richtung",
                thread_spec="M8",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert len(a.holes) == 1
    h = a.holes[0]
    assert h.axis == "+x" and h.kind == "thread"
    assert h.pos == [45.0, 0.0, 7.0]
    assert h.depth_mm == 16.0


def test_spec_anchor_durch_z_axis_phrase():
    """'durch Z' declares a z bore in German text."""
    spec = _plate_spec(
        [
            Feature(
                name="Durchgangsbohrung",
                type="through_hole",
                dims={"diameter": 10},
                position="x=+30 mm, y=0 mm, durch Z",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert len(a.holes) == 1
    h = a.holes[0]
    assert h.axis == "+z" and (h.x, h.y) == (30.0, 0.0)


def test_spec_anchor_enriches_measured_within_3mm():
    """A measured anchor near the declared position keeps its real x/y and
    gains the spec name/kind/dia — source gains the '+spec' marker."""
    spec = _plate_spec(
        [
            Feature(
                name="bore B",
                type="hole",
                dims={"diameter": 10.0},
                position="75mm from -X end, centered in Y",
            ),
        ]
    )
    mesh = {
        "holes": [
            {"x": 29.0, "y": 0.4, "dia": 9.6, "kind": "tunnel"},  # within 3mm of (30,0)
            {"x": -10.0, "y": 5.0, "dia": 3.0, "kind": "tunnel"},
        ],
        "outline_mm": [],
    }
    a = build_anchors(spec, {}, mesh)
    assert len(a.holes) == 2
    first = a.holes[0]  # spec-anchored entries go first regardless of x order
    assert first.source == "mesh+spec"
    assert (first.x, first.y) == (29.0, 0.4)  # measured position wins
    assert first.dia == 10.0 and first.name == "bore B"  # spec dims/name win
    assert a.holes[1].source == "mesh" and a.holes[1].x == -10.0
    assert [h.index for h in a.holes] == [1, 2]


def test_spec_anchors_order_first_then_measured_tail():
    """Measured-only anchors append after all spec-anchored entries, x-sorted."""
    spec = _plate_spec(
        [
            Feature(
                name="bore",
                type="hole",
                dims={"diameter": 10.0},
                position="75mm from -X end, centered in Y",
            ),
        ]
    )
    mesh = {
        "holes": [{"x": -40.0, "y": 0.0, "dia": 4.0, "kind": "tunnel"}],
        "outline_mm": [],
    }
    a = build_anchors(spec, {}, mesh)
    assert a.holes[0].source == "spec" and a.holes[0].x == 30.0
    assert a.holes[1].source == "mesh" and a.holes[1].x == -40.0
    assert [h.index for h in a.holes] == [1, 2]


def test_spec_anchor_unresolvable_or_unusable_skipped():
    """Vague text and missing dia never become anchors; a side-face bore with
    both in-plane axes declared resolves on its own plane."""
    spec = _plate_spec(
        [
            Feature(
                name="vague",
                type="hole",
                dims={"diameter": 6.0},
                position="near the +X end",
            ),
            Feature(
                name="nodia",
                type="hole",
                position="20mm from -X end, centered in Y",
            ),
            Feature(
                name="side",
                type="hole",
                dims={"diameter": 4.0},
                axis="x",
                position="centred in y, centred in z",
            ),
        ]
    )
    a = build_anchors(spec, {}, mesh_features=None)
    assert [(h.name, h.axis, h.pos) for h in a.holes] == [
        ("side", "+x", [None, 0.0, 7.0])  # "centred in z" = mid height 14/2
    ]


def test_spec_est_anchor_from_near_end_hint():
    """'near the +X end, centered in Y' carries no numeric offset — the x
    axis resolves to an ESTIMATED inset of max(1.5*dia, 6mm) from the +x
    end and the anchor is marked spec-est, not spec."""
    spec = _plate_spec(
        [
            Feature(
                name="M8 bore",
                type="hole",
                dims={"diameter": 6.8},
                position="near the +X end, centered in Y",
                standard="M8",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert len(a.holes) == 1
    h = a.holes[0]
    assert h.source == "spec-est"
    assert (h.x, h.y) == (34.8, 0.0)  # hi 45 - max(1.5*6.8, 6)
    assert h.dia == 6.8 and h.bolt_size == "M8" and h.name == "M8 bore"


def test_spec_est_anchor_merges_like_spec():
    """A spec-est anchor merges onto a measured anchor within 3 mm exactly
    like a spec anchor — measured x/y wins, spec name/kind/dia inherit."""
    spec = _plate_spec(
        [
            Feature(
                name="M8 bore",
                type="hole",
                dims={"diameter": 6.8},
                position="near the +X end, centered in Y",
            ),
        ]
    )
    mesh = {
        "holes": [{"x": 33.5, "y": 0.2, "dia": 6.9, "kind": "tunnel"}],
        "outline_mm": [],
    }
    a = build_anchors(spec, {}, mesh)
    assert len(a.holes) == 1
    h = a.holes[0]
    assert h.source == "mesh+spec"
    assert (h.x, h.y) == (33.5, 0.2)  # measured position wins
    assert h.dia == 6.8 and h.name == "M8 bore"  # spec dims/name win


def test_spec_est_near_hint_needs_both_axes():
    """'near the +X end' alone resolves x but not y — still skipped (a
    guessed coordinate is worse than none)."""
    spec = _plate_spec(
        [
            Feature(
                name="bore",
                type="hole",
                dims={"diameter": 6.8},
                position="near the +X end",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert a.holes == []


def test_spec_est_minus_end_insets_from_lo():
    """'near the -x end' estimates lo + max(1.5*dia, 6mm)."""
    spec = _plate_spec(
        [
            Feature(
                name="bore",
                type="hole",
                dims={"diameter": 4.0},
                position="near the -x end, centred in y",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert len(a.holes) == 1
    h = a.holes[0]
    assert h.source == "spec-est"
    assert (h.x, h.y) == (-39.0, 0.0)  # lo -45 + max(1.5*4, 6)


def test_spec_anchor_contradicting_coord_dropped():
    """'75mm from -x end (x=20)' — the model-computed parenthetical
    contradicts the end offset; the stated end hint wins, no phantom
    second anchor appears."""
    spec = _plate_spec(
        [
            Feature(
                name="bore B",
                type="bore",
                dims={"diameter": 10.0},
                position="75 mm from -x end, centered in y (x=20, y=0)",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert [(h.x, h.y) for h in a.holes] == [(30.0, 0.0)]


def test_spec_anchor_agreeing_coord_dedupes():
    """'20mm from -x end (x=-25)' — consistent hints collapse to one."""
    spec = _plate_spec(
        [
            Feature(
                name="bore A",
                type="bore",
                dims={"diameter": 10.0},
                position="20 mm from -x end, centered in y (x=-25, y=0)",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert [(h.x, h.y) for h in a.holes] == [(-25.0, 0.0)]


def test_spec_anchor_tap_hole_maps_to_tap_kind():
    spec = _plate_spec(
        [
            Feature(
                name="M8 tap hole",
                type="tap_hole",
                dims={"diameter": 8.0},
                position="82 mm from -x end, centered in y",
            ),
        ]
    )
    a = build_anchors(spec, {}, None)
    assert [(h.x, h.kind, h.bolt_size) for h in a.holes] == [(37.0, "tap", "M8")]


# --------------------------------------------------------------------------- #
# multi-point spec features + host-body resolution (bracket E2E regression)
# --------------------------------------------------------------------------- #


def test_bracket_spec_yields_four_anchors():
    """The bracket E2E intake declares 2x2 holes in ONE feature each —
    'x=-25, y=0 and x=+45, y=0' produced only the LAST x before, and the
    side bore had no x at all. Now all four spec positions anchor."""
    spec = _bracket_spec()
    a = build_anchors(spec, {}, None)
    assert len(a.holes) == 4
    z_anchors = sorted(h.x for h in a.holes if h.axis == "+z")
    assert z_anchors == [-25.0, 45.0]
    x_anchors = sorted(h.pos[2] for h in a.holes if h.axis == "+x")
    assert x_anchors == [15.0, 70.0]
    for h in a.holes:
        if h.axis == "+x":
            # no x= in the text — the leg's own range pins the bore axis
            assert h.pos == [-57.5, 0.0, h.pos[2]]
            assert h.y == 0.0
    assert "4 spec-declared" in a.note


def test_side_bore_range_on_bore_axis_anchors_at_mid():
    """'x=-60 bis -55' on the bore axis is the plate span, not an endpoint —
    the flat coord parse used to surface x=-60 (the range's own lo edge),
    putting the anchor 2.5 mm off the measured bore at leg mid -57.5 and
    scoring a false major (live E2E intake declared ranges like this)."""
    d = lambda v: Dim(value=v, unit="mm", source="stated")  # noqa: E731
    spec = PartSpec(
        name="t",
        summary="t",
        length_x=d(120),
        width_y=d(30),
        height_z=d(100),
        features=[
            Feature(name="leg", type="box", position="x from -60 to -55, z from 0 to 100"),
            Feature(
                name="leg_hole",
                type="through_hole",
                description="bore through the leg, axis X",
                dims={"diameter": 5.5},
                position="x=-60 bis -55, y=0, z=15",
                axis="x",
                count=1,
            ),
        ],
    )
    a = build_anchors(spec, {}, None)
    assert len(a.holes) == 1
    h = a.holes[0]
    assert h.axis == "+x"
    assert h.pos == [-57.5, 0.0, 15.0]


def test_bracket_spec_centers_beat_text():
    """Structured `centers` on a feature replace the text parse entirely."""
    spec = _bracket_spec()
    arm = next(f for f in spec.features if f.name == "arm_holes")
    arm.centers = [FeatureCenter(x=-10.0, y=0.0), FeatureCenter(x=30.0, y=0.0)]
    a = build_anchors(spec, {}, None)
    assert sorted(h.x for h in a.holes if h.axis == "+z") == [-10.0, 30.0]


def test_parse_ranges_spellings():
    assert parse_ranges("x from -60 to -55") == {"x": (-60.0, -55.0)}
    assert parse_ranges("x=-60..-55") == {"x": (-60.0, -55.0)}
    assert parse_ranges("x: -60 bis -55") == {"x": (-60.0, -55.0)}
    assert parse_ranges("x=-5 bis x=+115") == {"x": (-5.0, 115.0)}
    assert parse_ranges("von z=0 bis z=100") == {"z": (0.0, 100.0)}
    assert parse_ranges("x zwischen -60 und -55") == {"x": (-60.0, -55.0)}
    # lo/hi are sorted and several axes coexist
    assert parse_ranges("x from -55 to -60, z from 0 to 100") == {
        "x": (-60.0, -55.0),
        "z": (0.0, 100.0),
    }


def test_host_body_and_range_bracket():
    spec = _bracket_spec()
    arm_holes = next(f for f in spec.features if f.name == "arm_holes")
    leg_holes = next(f for f in spec.features if f.name == "leg_holes")
    assert host_body(spec, arm_holes).name == "horizontal_arm"
    assert host_range(spec, arm_holes, "z") == (95.0, 100.0)
    assert host_body(spec, leg_holes).name == "vertical_leg"
    assert host_range(spec, leg_holes, "x") == (-60.0, -55.0)


def test_host_body_resolves_via_feature_name():
    """No in-phrase, but the name names the host: 'arm_holes' -> the arm.

    Intake splits multi-hole features into name-tagged singles without
    'im <host>' text — the name token still resolves the plate."""
    spec = _bracket_spec()
    arm_holes = next(f for f in spec.features if f.name == "arm_holes")
    arm_holes.description = "two through-holes, axis Z"
    arm_holes.position = "x=-25, y=0 and x=+45, y=0"
    assert host_body(spec, arm_holes).name == "horizontal_arm"
    assert host_range(spec, arm_holes, "z") == (95.0, 100.0)


def _live_rerun_spec() -> PartSpec:
    """Shape of the second live CAD intake: name-tagged single holes, hosts
    declared as '<±axis>-Ende/Oberkante + thickness' instead of axis ranges."""
    d = lambda v: Dim(value=v, unit="mm", source="stated")  # noqa: E731
    return PartSpec(
        name="winkel",
        summary="",
        units="mm",
        length_x=d(120),
        width_y=d(30),
        height_z=d(100),
        features=[
            Feature(
                name="senkrechter Schenkel",
                type="box",
                position="am -X-Ende, von z=0 bis z=100",
                dims={"thickness": 5.0},
            ),
            Feature(
                name="horizontaler Arm",
                type="box",
                position="an der Oberkante des Schenkels, von x=-5 bis x=+115",
                dims={"thickness": 5.0},
            ),
            Feature(
                name="Schenkel-Bohrung 1",
                type="hole",
                position="z=15, y=0, Achse X",
                dims={"diameter": 5.5, "depth": 5.0},
                axis="x",
                count=1,
            ),
            Feature(
                name="Arm-Bohrung 1",
                type="hole",
                position="x=-25, y=0, Achse Z",
                dims={"diameter": 5.5, "depth": 5.0},
                axis="z",
                count=1,
            ),
        ],
        image_roles=[],
    )


def test_host_span_end_anchored_plate():
    """'am -X-Ende + thickness' -> [env_lo, env_lo+t]; 'Oberkante' -> top."""
    spec = _live_rerun_spec()
    leg_hole = next(f for f in spec.features if f.name == "Schenkel-Bohrung 1")
    arm_hole = next(f for f in spec.features if f.name == "Arm-Bohrung 1")
    assert host_body(spec, leg_hole).name == "senkrechter Schenkel"
    assert host_span(spec, leg_hole, "x") == (-60.0, -55.0)
    assert host_body(spec, arm_hole).name == "horizontaler Arm"
    assert host_span(spec, arm_hole, "z") == (95.0, 100.0)
    a = build_anchors(spec, {}, None)
    leg = next(h for h in a.holes if h.axis == "+x")
    assert leg.pos == [-57.5, 0.0, 15.0]


def test_host_body_none_without_in_phrase_or_name():
    spec = _bracket_spec()
    arm_holes = next(f for f in spec.features if f.name == "arm_holes")
    arm_holes.name = "mounting bores"
    arm_holes.description = "two through-holes, axis Z"
    arm_holes.position = "x=-25, y=0 and x=+45, y=0"
    assert host_body(spec, arm_holes) is None
    assert host_range(spec, arm_holes, "z") is None
