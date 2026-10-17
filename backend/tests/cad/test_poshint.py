"""cadx.cad.poshint — shared spec position-hint parser."""

from __future__ import annotations

import pytest

from cadx.cad.poshint import parse_position_hints, resolve_end_hint


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
