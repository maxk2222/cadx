"""spechost — free-text range parsing + host-body/host-range resolution."""

import json
from pathlib import Path

import pytest

from cadx.agent.spechost import host_body, host_range, parse_ranges
from cadx.llm.schemas import Feature, PartSpec

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "specs"


@pytest.fixture()
def bracket() -> PartSpec:
    return PartSpec.model_validate(
        json.loads((_FIXTURES / "bracket_strut_cad.json").read_text(encoding="utf-8"))
    )


@pytest.mark.parametrize(
    ("text", "axis", "expected"),
    [
        ("x from -60 to -55", "x", (-60.0, -55.0)),
        ("x=-60..-55", "x", (-60.0, -55.0)),
        ("x: -60 bis -55", "x", (-60.0, -55.0)),
        ("x zwischen -60 und -55", "x", (-60.0, -55.0)),
        ("x=-55..-60", "x", (-60.0, -55.0)),  # unsorted → sorted
        ("z 0 to 100", "z", (0.0, 100.0)),
        ("y from -15 to 15", "y", (-15.0, 15.0)),
    ],
)
def test_parse_ranges(text: str, axis: str, expected: tuple[float, float]):
    ranges = parse_ranges(text)
    assert ranges[axis] == expected


def test_parse_ranges_multiple_axes():
    ranges = parse_ranges("x=-60..-55, y -15 to 15, z 0..100")
    assert ranges["x"] == (-60.0, -55.0)
    assert ranges["y"] == (-15.0, 15.0)
    assert ranges["z"] == (0.0, 100.0)


def test_parse_ranges_empty():
    assert parse_ranges("no numbers here") == {}
    assert parse_ranges("") == {}


def test_host_body_arm_holes(bracket: PartSpec):
    holes = next(f for f in bracket.features if f.name == "arm_holes")
    host = host_body(bracket, holes)
    assert host is not None and host.name == "horizontal_arm"


def test_host_body_leg_holes(bracket: PartSpec):
    holes = next(f for f in bracket.features if f.name == "leg_holes")
    host = host_body(bracket, holes)
    assert host is not None and host.name == "vertical_leg"


def test_host_range_axes(bracket: PartSpec):
    holes = next(f for f in bracket.features if f.name == "arm_holes")
    assert host_range(bracket, holes, "z") == (95.0, 100.0)
    holes = next(f for f in bracket.features if f.name == "leg_holes")
    assert host_range(bracket, holes, "x") == (-60.0, -55.0)


def test_host_body_none_without_match():
    spec = PartSpec.model_validate(
        {
            "name": "t",
            "length_x": {"value": 50},
            "width_y": {"value": 50},
            "height_z": {"value": 50},
            "features": [
                {"name": "body", "type": "box", "position": "x=0..50"},
                {
                    "name": "hole",
                    "type": "through_hole",
                    "position": "through something",
                    "dims": {"diameter": 5},
                },
            ],
        }
    )
    f = Feature(name="hole", type="through_hole", position="through something")
    assert host_body(spec, f) is None
    assert host_range(spec, f, "z") is None


def test_fixture_without_centers_roundtrips(bracket: PartSpec):
    """Older specs carry no `centers` — they must still parse and survive a
    model_dump -> model_validate round-trip (backward compatibility)."""
    assert all(f.centers == [] for f in bracket.features)
    again = PartSpec.model_validate(bracket.model_dump())
    assert again == bracket
