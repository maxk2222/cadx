"""Load-case `hole_index` rebinding — `agent/steps/intake.py`.

The robot-arm-mount eval showed a weak model miscounting: `bore #8` on a
7-anchor part (dead case — the index resolved to nothing) and `bore #4` for
the M12 seat (a 15 N·m torque aimed at a corner mounting hole). The rebind
routine re-points refs at the anchors the case text actually names and
splits plural loads ("flange holes") across the matching anchors.
"""

from __future__ import annotations

from cadx.agent.steps.intake import rebind_case_anchors, sanitize_load_cases
from cadx.llm.schemas import (
    Feature,
    LoadCase,
    LoadSpec,
    OptimizeSpec,
    PartSpec,
    RegionSpec,
)


def _hole(name: str, dia: float, pos: str, axis: str = "", standard: str = "") -> Feature:
    return Feature(
        name=name,
        type="hole",
        description="",
        dims={"diameter": dia},
        position=pos,
        count=1,
        source="text",
        axis=axis,
        standard=standard,
    )


def _mount_spec() -> PartSpec:
    """The eval's 7-anchor table: #1-4 corner mounts, #5 M12 seat (+z),
    #6-7 flange bores (+x). Mirrors `anchors._spec_anchors` output."""
    return PartSpec(
        name="robot arm mount",
        summary="",
        length_x={"value": 100},
        width_y={"value": 80},
        height_z={"value": 40},
        features=[
            _hole("corner hole +x +y", 8.5, "x=42 y=32"),
            _hole("corner hole +x -y", 8.5, "x=42 y=-32"),
            _hole("corner hole -x +y", 8.5, "x=-42 y=32"),
            _hole("corner hole -x -y", 8.5, "x=-42 y=-32"),
            Feature(
                name="M12 tapped through-bore",
                type="thread",
                description="",
                dims={"diameter": 10.2},
                position="x=0 y=0",
                count=1,
                source="text",
                standard="M12",
            ),
            _hole("flange hole -y", 6.5, "z=25 y=-12", axis="x"),
            _hole("flange hole +y", 6.5, "z=25 y=12", axis="x"),
            Feature(
                name="bearing boss",
                type="boss",
                description="",
                dims={"diameter": 25, "height": 15},
                position="x=0 y=0",
                count=1,
                source="text",
            ),
        ],
        standard_parts=[],
        image_roles=[],
        verification_questions=[],
    )


def test_rebind_out_of_range_plural_splits_over_flange_pair():
    """The eval's actual failure: `bore #8` (only 7 anchors) on a case named
    'arm load on flange holes' — dead index + plural intent → one load on
    each flange bore (#6 and #7), fixtures left on the corner mounts."""
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="arm load on flange holes",
            fixed=[RegionSpec(kind="hole", hole_index=i) for i in (1, 2, 3, 4)],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="bore", hole_index=8),
                    bearing_n=200.0,
                    bearing_dir=[0.0, 0.0, -1.0],
                )
            ],
        )
    ]
    notes = rebind_case_anchors(spec)
    loads = spec.load_cases[0].loads
    assert {ld.region.hole_index for ld in loads} == {6, 7}
    # the case name describes the LOAD side — the corner fixtures stay put
    assert [r.hole_index for r in spec.load_cases[0].fixed] == [1, 2, 3, 4]
    assert notes


def test_rebind_wrong_index_rewired_by_bolt_name():
    """`bore #4` (a corner mount) carrying 'joint torque via M12 seat' —
    the M12 token only matches anchor #5, so the torque re-points there."""
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="joint torque via M12 seat",
            fixed=[RegionSpec(kind="face", axis="-z")],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="bore", hole_index=4),
                    moment_nm=[0.0, 0.0, 15.0],
                )
            ],
        )
    ]
    rebind_case_anchors(spec)
    assert spec.load_cases[0].loads[0].region.hole_index == 5


def test_rebind_leaves_unambiguous_refs_alone():
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="own weight + payload",
            fixed=[RegionSpec(kind="hole", hole_index=1)],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="bore", hole_index=5),
                    bearing_n=80.0,
                    bearing_dir=[0.0, -1.0, 0.0],
                )
            ],
        )
    ]
    notes = rebind_case_anchors(spec)
    assert spec.load_cases[0].fixed[0].hole_index == 1
    assert spec.load_cases[0].loads[0].region.hole_index == 5
    assert not notes


def test_rebind_keep_regions_use_their_own_description():
    spec = _mount_spec()
    spec.optimize = OptimizeSpec(
        enabled=True,
        keep_regions=[RegionSpec(kind="hole", hole_index=4, description="keep M12 bore")],
    )
    rebind_case_anchors(spec)
    assert spec.optimize.keep_regions[0].hole_index == 5


def test_sanitize_drops_degenerate_clamps_weight_records_notes():
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(name="", fixed=[], loads=[]),
        LoadCase(
            name="arm load on flange holes",
            fixed=[RegionSpec(kind="hole", hole_index=1)],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="bore", hole_index=8),
                    bearing_n=100.0,
                )
            ],
            weight=-2.0,
        ),
    ]
    sanitize_load_cases(spec)
    assert len(spec.load_cases) == 1
    assert spec.load_cases[0].name == "arm load on flange holes"
    assert spec.load_cases[0].weight == 1.0
    assert {ld.region.hole_index for ld in spec.load_cases[0].loads} == {6, 7}
    assert any("rebound" in a for a in spec.assumptions)


def test_rebind_g_load_on_skin_region_becomes_all():
    """g_load is a body force — a face/plane skin region is a category error
    that pins whole voxel layers as passive (97%+ of the domain)."""
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="own weight",
            fixed=[RegionSpec(kind="plane", axis="-z")],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="face", axis="+z"),
                    g_load=[0.0, 0.0, -1.0],
                )
            ],
        )
    ]
    sanitize_load_cases(spec)
    ld = spec.load_cases[0].loads[0]
    assert ld.region.kind == "all"


def test_rebind_face_encastre_replaced_by_mount_holes():
    """'plane -z' encastre on a bolted plate pins ~27% as fixture skin — the
    corner mounting holes carry the fixture, rewrite deterministically."""
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="own weight",
            fixed=[RegionSpec(kind="plane", axis="-z")],
            loads=[LoadSpec(region=RegionSpec(kind="all"), g_load=[0.0, 0.0, -1.0])],
        )
    ]
    sanitize_load_cases(spec)
    fixed = spec.load_cases[0].fixed
    kinds = {r.kind for r in fixed}
    idxs = {r.hole_index for r in fixed}
    assert kinds == {"hole"}
    assert idxs == {1, 2, 3, 4}  # corner bolts, not the M12 thread seat
    assert any("encastre" in a for a in spec.assumptions)


def test_rebind_face_encastre_kept_without_bolt_pattern():
    """A whole-face fixture stays when no matching mount-hole pattern exists
    (glued/clamped face is a real fixture then — never silently dropped)."""
    spec = _mount_spec()
    spec.features = [f for f in spec.features if str(f.position or "").startswith("z=")]
    spec.load_cases = [
        LoadCase(
            name="clamped",
            fixed=[RegionSpec(kind="face", axis="-z")],
            loads=[LoadSpec(region=RegionSpec(kind="all"), g_load=[0.0, 0.0, -1.0])],
        )
    ]
    sanitize_load_cases(spec)
    assert spec.load_cases[0].fixed[0].kind == "face"


def test_rebind_moment_on_clearance_hole_moves_to_thread():
    """A pure moment needs a seat that reacts torque — an untapped clearance
    bore can't; rewire to the largest threaded anchor."""
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="shaft torque reaction",
            fixed=[RegionSpec(kind="hole", hole_index=1)],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="bore", hole_index=7),
                    moment_nm=[0.0, 15.0, 0.0],
                )
            ],
        )
    ]
    sanitize_load_cases(spec)
    ld = spec.load_cases[0].loads[0]
    assert ld.region.hole_index == 5  # M12 thread seat
    assert any("torque" in a or "thread" in a for a in spec.assumptions)


def test_rebind_moment_on_thread_seat_untouched():
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="shaft torque",
            fixed=[RegionSpec(kind="hole", hole_index=1)],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="bore", hole_index=5),
                    moment_nm=[0.0, 15.0, 0.0],
                )
            ],
        )
    ]
    sanitize_load_cases(spec)
    assert spec.load_cases[0].loads[0].region.hole_index == 5
