"""cadx.cad.poshint — shared spec position-hint parser."""

from __future__ import annotations

from cadx.cad.poshint import parse_position_hints, resolve_end_hint


def test_coord_and_midplane_hints():
    coords, faces, ends = parse_position_hints("x=12, centred in Y")
    assert coords == {"x": 12.0, "y": 0.0}
    assert faces == [] and ends == []


def test_end_offset_list_and_masking():
    coords, faces, ends = parse_position_hints("20 and 75mm from -X end, centered in y")
    assert coords == {"y": 0.0}
    assert [(e["axis"], e["end"], e["offset_mm"]) for e in ends] == [
        ("x", "-x", 20.0),
        ("x", "-x", 75.0),
    ]
    assert faces == []


def test_end_offset_max_side_and_cm():
    _c, _f, ends = parse_position_hints("10 mm from the +x edge; 1.5cm from -y end")
    assert [(e["end"], e["offset_mm"]) for e in ends] == [("+x", 10.0), ("-y", 15.0)]


def test_range_stays_unresolved_but_face_reads():
    coords, faces, ends = parse_position_hints("between 20 and 30mm from -x end")
    assert ends == [] and coords == {}
    assert faces == ["-x"]  # unmasked — the ±axis still reads as a face hint


def test_both_face_signs_cancel():
    _c, faces, _e = parse_position_hints("front and back faces")
    assert faces == []


def test_resolve_end_hint():
    # 90mm span centred at origin: lo=-45, hi=+45
    assert resolve_end_hint("x", "-x", 20.0, -45.0, 45.0) == -25.0
    assert resolve_end_hint("x", "+x", 8.0, -45.0, 45.0) == 37.0
