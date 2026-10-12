"""Tests for cadx.agent.anchors — image/mesh hole reconciliation."""

from __future__ import annotations

from cadx.agent.anchors import build_anchors
from cadx.llm.schemas import Dim, Feature, PartSpec


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
