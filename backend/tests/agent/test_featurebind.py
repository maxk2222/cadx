"""Feature binder: spec features → ready-made cadlib snippets.

The binder exists because the E2E post-mortem showed the model re-derives
axes from prose even with the exact call in the issue text — every test
below pins one resolution that must never regress.
"""

from __future__ import annotations

from cadx.agent.context import bindings_block
from cadx.agent.featurebind import bind_features
from cadx.llm.schemas import PartSpec

_DIM = {"value": 60.0, "unit": "mm", "source": "stated"}


def _spec(features, lx=60.0, wy=40.0, hz=54.0) -> PartSpec:
    return PartSpec.model_validate(
        {
            "name": "part",
            "summary": "test",
            "units": "mm",
            "length_x": {**_DIM, "value": lx},
            "width_y": {**_DIM, "value": wy},
            "height_z": {**_DIM, "value": hz},
            "image_roles": [],
            "features": features,
        }
    )


def _feat(**kw) -> dict:
    base = {
        "name": "f",
        "type": "other",
        "description": "",
        "dims": {},
        "position": "",
        "count": 1,
        "source": "stated",
    }
    base.update(kw)
    return base


def _l_bracket_spec() -> PartSpec:
    """The real intake output of the failing eval — ground truth fixture."""
    return _spec(
        [
            _feat(
                name="foot",
                type="box",
                dims={"length": 60.0, "width": 40.0, "thickness": 4.0},
                position="base, z=0..4",
            ),
            _feat(
                name="leg",
                type="box",
                dims={"height": 50.0, "width": 40.0, "thickness": 4.0},
                position="-X end, z=4..54",
            ),
            _feat(
                name="foot_hole_1",
                type="through_hole",
                dims={"diameter": 5.5},
                position="centered in Y, 30 mm apart in X (one at x=-15, one at x=+15)",
                count=2,
            ),
            _feat(
                name="leg_slot",
                type="slot",
                dims={"width": 6.5, "slot_length": 15.0},
                position="centered in leg, along Z",
            ),
            _feat(
                name="inner_fillet",
                type="fillet",
                dims={"radius": 3.0},
                position="inner corner at foot/leg junction",
            ),
            _feat(
                name="rib_1",
                type="rib",
                dims={"thickness": 3.0},
                position="outer side of leg, +Y face",
                count=2,
            ),
            _feat(
                name="rib_2",
                type="rib",
                dims={"thickness": 3.0},
                position="outer side of leg, -Y face",
            ),
        ]
    )


class TestSlotBinding:
    def test_vertical_slot_through_x_wall(self):
        """The exact observed failure: slot along Z through the X-thin leg."""
        spec = _l_bracket_spec()
        by_name = {b.name: b for b in bind_features(spec)}
        s = by_name["leg_slot"].snippet
        assert 'oblong_slot(6.5, 15, direction="+z", normal="+x")' in s
        # centred in the leg → parent centre resolves the Pos
        assert "Pos(-28" in s and "29" in s

    def test_slot_axis_field_beats_parent(self):
        spec = _spec(
            [
                _feat(
                    name="wall",
                    type="box",
                    dims={"height": 30.0, "width": 20.0, "thickness": 3.0},
                    position="+Y end",
                ),
                _feat(
                    name="s",
                    type="slot",
                    axis="z",
                    dims={"slot_width": 4.0, "slot_length": 10.0},
                    position="in wall",
                ),
            ]
        )
        s = {b.name: b for b in bind_features(spec)}["s"].snippet
        assert 'normal="+z"' in s

    def test_slot_missing_dims_degrades(self):
        spec = _spec([_feat(name="s", type="slot", dims={}, position="in wall")])
        b = {x.name: x for x in bind_features(spec)}["s"]
        assert not b.resolved

    def test_slot_unknown_wall_keeps_placeholder(self):
        spec = _spec(
            [
                _feat(
                    name="s",
                    type="slot",
                    dims={"slot_width": 5.0, "slot_length": 12.0},
                    position="somewhere",
                )
            ]
        )
        b = {x.name: x for x in bind_features(spec)}["s"]
        assert not b.resolved
        assert "<wall normal>" in b.snippet


class TestHoleBinding:
    def test_explicit_positions(self):
        spec = _l_bracket_spec()
        s = {b.name: b for b in bind_features(spec)}["foot_hole_1"].snippet
        assert "Pos(-15.0, 0.0, 0.0), Pos(15.0, 0.0, 0.0)" in s
        assert "through_hole(5.5)" in s

    def test_apart_spacing_centred(self):
        spec = _spec(
            [
                _feat(
                    name="h",
                    type="hole",
                    dims={"diameter": 5.0},
                    position="30 mm apart in X, through the top",
                    count=2,
                )
            ]
        )
        s = {b.name: b for b in bind_features(spec)}["h"].snippet
        assert "Pos(-15.0, 0.0, 0.0), Pos(15.0, 0.0, 0.0)" in s

    def test_x_axis_hole_rotates_frame(self):
        spec = _spec(
            [
                _feat(
                    name="h",
                    type="hole",
                    axis="x",
                    dims={"diameter": 8.0},
                    position="through the side",
                )
            ]
        )
        s = {b.name: b for b in bind_features(spec)}["h"].snippet
        assert "Rot(0, 90, 0)" in s and "axis +x" in s

    def test_countersink_and_blind_args(self):
        spec = _spec(
            [
                _feat(
                    name="h",
                    type="hole",
                    dims={
                        "diameter": 5.0,
                        "depth": 10.0,
                        "countersink_diameter": 9.0,
                        "countersink_angle": 90.0,
                    },
                    position="top",
                )
            ]
        )
        s = {b.name: b for b in bind_features(spec)}["h"].snippet
        assert "depth=10" in s and "csk_dia=9" in s

    def test_pcd_pattern(self):
        spec = _spec(
            [
                _feat(
                    name="flange_holes",
                    type="hole",
                    dims={"diameter": 6.6},
                    position="6 holes on PCD 62, through the flange",
                    count=6,
                )
            ]
        )
        s = {b.name: b for b in bind_features(spec)}["flange_holes"].snippet
        assert s.count("Pos(") == 6 and "PCD 62" in s
        assert "Pos(31.0, 0.0, 0.0)" in s  # first ring point


class TestRibBinding:
    def test_envelope_offset_and_sides(self):
        spec = _l_bracket_spec()
        by_name = {b.name: b for b in bind_features(spec)}
        # +Y rib only on +Y, -Y rib only on -Y, outer face flush at ±20
        assert (
            ", 18," in by_name["rib_1"].snippet
            and "-18" not in by_name["rib_1"].snippet.split("Pos")[1].split(")")[0]
        )
        assert "-18" in by_name["rib_2"].snippet
        assert "outer rib face ends at y=±20" in by_name["rib_1"].snippet

    def test_unsided_rib_emits_both(self):
        spec = _spec(
            [_feat(name="r", type="rib", dims={"thickness": 4.0}, position="outer sides", count=2)]
        )
        s = {b.name: b for b in bind_features(spec)}["r"].snippet
        assert ", 17.5," in s and "-17.5" in s  # 40/2 - 4/2 - 0.5 = 17.5


class TestOtherBindings:
    def test_inner_fillet_selector(self):
        spec = _l_bracket_spec()
        s = {b.name: b for b in bind_features(spec)}["inner_fillet"].snippet
        assert "safe_fillet(" in s and "3" in s and "Axis.Y" in s

    def test_chamfer(self):
        spec = _spec([_feat(name="c", type="chamfer", dims={"chamfer": 1.0}, position="outer rim")])
        s = {b.name: b for b in bind_features(spec)}["c"].snippet
        assert "chamfer(" in s and "1" in s

    def test_tube_and_boss(self):
        spec = _spec(
            [
                _feat(
                    name="t",
                    type="tube",
                    dims={"diameter": 20.0, "inner_diameter": 14.0, "length": 30.0},
                    position="centered",
                ),
                _feat(
                    name="b",
                    type="boss",
                    dims={"diameter": 12.0, "height": 6.0},
                    position="at x=10, y=0",
                ),
            ]
        )
        by_name = {b.name: b for b in bind_features(spec)}
        assert "tube(20" in by_name["t"].snippet and "14" in by_name["t"].snippet
        assert "boss(diameter=12" in by_name["b"].snippet

    def test_insert_and_bearing_use_standard(self):
        spec = _spec(
            [
                _feat(name="i", type="insert", standard="M4", position="top face"),
                _feat(name="bg", type="bearing_seat", standard="6204-2RS", position="centered"),
            ]
        )
        by_name = {b.name: b for b in bind_features(spec)}
        assert 'HeatSetNut("M4")' in by_name["i"].snippet
        assert "6204-2RS" in by_name["bg"].snippet

    def test_thread_tap_drill(self):
        spec = _spec([_feat(name="t", type="thread", standard="M6", position="top")])
        s = {b.name: b for b in bind_features(spec)}["t"].snippet
        assert "M6" in s and "through_hole(" in s


class TestBlock:
    def test_block_header_and_cap(self):
        spec = _l_bracket_spec()
        out = bindings_block(spec)
        assert "BOUND FEATURE SNIPPETS" in out
        # no features → empty block, no prompt noise
        assert bindings_block(_spec([])) == ""

    def test_unknown_type_skipped(self):
        spec = _spec([_feat(name="x", type="decorative_swirl")])
        assert bind_features(spec) == []

    def test_binder_exception_degrades_not_raises(self, monkeypatch):
        spec = _l_bracket_spec()

        def boom(*a, **k):
            raise RuntimeError("parser bug")

        monkeypatch.setitem(
            __import__("cadx.agent.featurebind", fromlist=["x"])._BINDERS, "slot", boom
        )
        # must not raise — the slot just doesn't get a binding
        assert all(b.name != "leg_slot" for b in bind_features(spec))
