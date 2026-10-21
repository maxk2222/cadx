"""cadx.cad.poshint — shared spec position-hint parser."""

from __future__ import annotations

import pytest

from cadx.cad.poshint import (
    parse_axis_ranges,
    parse_coord_points,
    parse_position_hints,
    resolve_end_hint,
)


def test_coord_and_midplane_hints():
    coords, faces, ends, nears = parse_position_hints("x=12, centred in Y")
    assert coords == {"x": 12.0, "y": 0.0}
    assert faces == [] and ends == [] and nears == []


def test_end_offset_list_and_masking():
    coords, faces, ends, nears = parse_position_hints("20 and 75mm from -X end, centered in y")
    assert coords == {"y": 0.0}
    assert [(e["axis"], e["end"], e["offset_mm"]) for e in ends] == [
        ("x", "-x", 20.0),
        ("x", "-x", 75.0),
    ]
    assert faces == [] and nears == []


def test_end_offset_max_side_and_cm():
    _c, _f, ends, _n = parse_position_hints("10 mm from the +x edge; 1.5cm from -y end")
    assert [(e["end"], e["offset_mm"]) for e in ends] == [("+x", 10.0), ("-y", 15.0)]


def test_range_stays_unresolved_but_face_reads():
    coords, faces, ends, nears = parse_position_hints("between 20 and 30mm from -x end")
    assert ends == [] and coords == {} and nears == []
    assert faces == ["-x"]  # unmasked — the ±axis still reads as a face hint


def test_both_face_signs_cancel():
    _c, faces, _e, _n = parse_position_hints("front and back faces")
    assert faces == []


@pytest.mark.parametrize(
    "text,end",
    [
        ("near the +x end", "+x"),
        ("near +X end", "+x"),
        ("close to the -y end", "-y"),
        ("toward the +x end", "+x"),
        ("towards the -z end", "-z"),
        ("at the -x end", "-x"),
        ("am +x Ende", "+x"),
        ("nahe dem -y Ende", "-y"),
        ("near the +x edge", "+x"),
        ("near the +x side", "+x"),
    ],
)
def test_near_end_hint_parses(text, end):
    coords, faces, ends, nears = parse_position_hints(text)
    assert ends == [] and coords == {}
    assert nears == [{"axis": end[1], "end": end}]
    # the ±axis inside a near-end phrase is masked — never a face hint
    assert faces == []


def test_near_end_combines_with_midplane():
    coords, faces, ends, nears = parse_position_hints("near the +X end, centered in Y")
    assert coords == {"y": 0.0}
    assert nears == [{"axis": "x", "end": "+x"}]
    assert ends == [] and faces == []


def test_numbered_end_stays_offset_not_near():
    """'75mm from -x end' is a point offset — never a proximity hint."""
    _c, _f, ends, nears = parse_position_hints("75mm from -x end")
    assert nears == []
    assert [(e["end"], e["offset_mm"]) for e in ends] == [("-x", 75.0)]


def test_numbered_near_phrase_is_not_a_near_hint():
    """A number immediately before the phrase ('5mm near the +x end') is an
    offset context, not a proximity hint — the ±axis degrades to a face."""
    _c, faces, ends, nears = parse_position_hints("5mm near the +x end")
    assert nears == [] and ends == []
    assert faces == ["+x"]


def test_resolve_end_hint():
    # 90mm span centred at origin: lo=-45, hi=+45
    assert resolve_end_hint("x", "-x", 20.0, -45.0, 45.0) == -25.0
    assert resolve_end_hint("x", "+x", 8.0, -45.0, 45.0) == 37.0


def test_plus_signed_coord_parses():
    """'x=+30 mm' is a legit coordinate — the regex must accept the plus
    sign (German specs write signed offsets)."""
    coords, _f, _e, _n = parse_position_hints("x=+30 mm, y=0 (zentriert in y)")
    assert coords["x"] == 30.0
    coords2, _f2, _e2, _n2 = parse_position_hints("x=+45 mm")
    assert coords2["x"] == 45.0


# --------------------------------------------------------------------------- #
# parse_coord_points — multi-point coordinate groups
# --------------------------------------------------------------------------- #


def test_coord_points_two_xy_groups():
    """'x=-25, y=0 and x=+45, y=0' declares TWO points — the per-axis dict
    keeps only the last x, so the grouped parse must keep both."""
    assert parse_coord_points("x=-25, y=0 and x=+45, y=0") == [
        {"x": -25.0, "y": 0.0},
        {"x": 45.0, "y": 0.0},
    ]


def test_coord_points_zy_groups():
    assert parse_coord_points("z=15, y=0 and z=70, y=0") == [
        {"z": 15.0, "y": 0.0},
        {"z": 70.0, "y": 0.0},
    ]


def test_coord_points_single_axis_repeats():
    assert parse_coord_points("x=-30 und x=+30") == [{"x": -30.0}, {"x": 30.0}]


def test_coord_points_single_group():
    assert parse_coord_points("x=12") == [{"x": 12.0}]


def test_coord_points_parenthesised_endpoints():
    assert parse_coord_points("(x=-55, z=35) to (x=+15, z=95)") == [
        {"x": -55.0, "z": 35.0},
        {"x": 15.0, "z": 95.0},
    ]


def test_coord_points_no_coords():
    assert parse_coord_points("near the +X end, centered in Y") == []


def test_coord_points_range_link_merges():
    """'x=-60 bis x=-55' is ONE span declaration, not two points — the
    repeated axis merges to the span's midpoint (lo/hi stay in
    parse_ranges). Live intake writes side-bore positions like this and the
    phantom second group spawned duplicate spec anchors."""
    assert parse_coord_points("z=15, y=0, x=-60 bis x=-55") == [{"z": 15.0, "y": 0.0, "x": -57.5}]
    assert parse_coord_points("x=-25, y=0, z=95 bis z=100") == [{"x": -25.0, "y": 0.0, "z": 97.5}]


def test_explanatory_ranges_are_not_second_feature_centers():
    text = (
        "x=-30..-26 (through), y=0 centered, z centered on leg height (~z=27.5 mid, i.e. z=20..35)"
    )
    assert parse_axis_ranges(text) == {"x": (-30.0, -26.0), "z": (20.0, 35.0)}
    assert parse_position_hints(text)[0] == {"x": -28.0, "y": 0.0, "z": 27.5}
    assert parse_coord_points(text) == [{"x": -28.0, "y": 0.0, "z": 27.5}]


def test_coord_points_wordy_connector_still_splits():
    """A range word inside a longer phrase is not a span — the two strut
    endpoints stay separate groups."""
    assert parse_coord_points(
        "from (x=-55, z=35) on the leg to (x=+15, z=95) on the arm underside"
    ) == [
        {"x": -55.0, "z": 35.0},
        {"x": 15.0, "z": 95.0},
    ]


# --------------------------------------------------------------------------- #
# unicode normalization (U+2212 minus, en/em dash, NBSP)                      #
# --------------------------------------------------------------------------- #


def test_unicode_minus_coords_parse():
    """LLM intake writes U+2212 verbatim — 'x=\u221230' must parse as -30."""
    coords, _f, _e, _n = parse_position_hints("x=\u221230, z=45 mm")
    assert coords == {"x": -30.0, "z": 45.0}
    assert parse_coord_points("x=\u221225, y=0 and x=+45, y=0") == [
        {"x": -25.0, "y": 0.0},
        {"x": 45.0, "y": 0.0},
    ]


def test_nbsp_and_dash_offsets_parse():
    """NBSP between value and unit, and a unicode minus before the axis of
    an end hint, both normalize away."""
    _c, _f, ends, _n = parse_position_hints("5\u00a0mm from \u2212x end")
    assert [(e["end"], e["offset_mm"]) for e in ends] == [("-x", 5.0)]
