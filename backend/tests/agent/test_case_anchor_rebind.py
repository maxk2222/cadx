"""Load-case `hole_index` rebinding — `agent/steps/intake.py`.

The robot-arm-mount eval showed a weak model miscounting: `bore #8` on a
7-anchor part (dead case — the index resolved to nothing) and `bore #4` for
the M12 seat (a 15 N·m torque aimed at a corner mounting hole). The rebind
routine re-points refs at the anchors the case text actually names and
splits plural loads ("flange holes") across the matching anchors.
"""

from __future__ import annotations

from cadx.agent.steps.intake import (
    lint_envelope_fit,
    materialize_anchor_hints,
    rebind_case_anchors,
    sanitize_load_cases,
)
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


def _real_mount_spec() -> PartSpec:
    """The eval spec's real feature ORDER: block, four mounts, boss, M12,
    wall, two flange bores, rib — the model's keep_regions numbered
    `hole_index` as 0-based positions into this list (#8/#9 → flange bores,
    0-based spec.features[8]/[9]) instead of anchor indices (#6/#7)."""
    return PartSpec(
        name="robot arm mount",
        summary="",
        length_x={"value": 100},
        width_y={"value": 80},
        height_z={"value": 40},
        features=[
            Feature(
                name="base plate",
                type="block",
                description="",
                dims={},
                position="",
                count=1,
                source="text",
            ),
            _hole("mounting hole -x -y", 8.5, "x=-42 y=-32"),
            _hole("mounting hole -x +y", 8.5, "x=-42 y=32"),
            _hole("mounting hole +x -y", 8.5, "x=42 y=-32"),
            _hole("mounting hole +x +y", 8.5, "x=42 y=32"),
            Feature(
                name="bearing boss",
                type="boss",
                description="",
                dims={"diameter": 25},
                position="x=0 y=0",
                count=1,
                source="text",
            ),
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
            Feature(
                name="vertical flange",
                type="wall",
                description="",
                dims={},
                position="",
                count=1,
                source="text",
            ),
            _hole("flange hole -y", 6.5, "z=25 y=-12", axis="x"),
            _hole("flange hole +y", 6.5, "z=25 y=12", axis="x"),
            Feature(
                name="gusset rib",
                type="rib",
                description="",
                dims={},
                position="",
                count=1,
                source="text",
            ),
        ],
        standard_parts=[],
        image_roles=[],
        verification_questions=[],
    )


def test_rebind_keep_region_out_of_range_via_feature_position():
    """The eval's actual keep-region failure: descriptionless regions with
    `hole_index` counted as spec.features positions — #8/#9 must rebound to
    the flange bore anchors #6/#7, not die silently in `_keep_prisms`."""
    spec = _real_mount_spec()
    spec.optimize = OptimizeSpec(
        enabled=True,
        keep_regions=[RegionSpec(kind="hole", hole_index=i) for i in (1, 2, 3, 4)]
        + [RegionSpec(kind="bore", hole_index=5)]
        + [RegionSpec(kind="hole", hole_index=i) for i in (8, 9)],
    )
    notes = rebind_case_anchors(spec)
    assert [r.hole_index for r in spec.optimize.keep_regions] == [1, 2, 3, 4, 5, 6, 7]
    assert any("spec feature position" in n for n in notes)


def test_rebind_keep_region_dead_index_without_hole_feature_stays():
    """An out-of-range index landing on a NON-hole feature stays dead —
    the rescue only re-points at hole-like features."""
    spec = _real_mount_spec()
    spec.optimize = OptimizeSpec(
        enabled=True,
        keep_regions=[RegionSpec(kind="hole", hole_index=7)],  # features[7] = wall
    )
    rebind_case_anchors(spec)
    # 7 is in range (7 anchors) — no change at all
    assert spec.optimize.keep_regions[0].hole_index == 7
    spec2 = _real_mount_spec()
    spec2.optimize = OptimizeSpec(
        enabled=True,
        keep_regions=[RegionSpec(kind="hole", hole_index=11)],  # features[10]=rib/[11] OOB
    )
    rebind_case_anchors(spec2)
    assert spec2.optimize.keep_regions[0].hole_index == 11


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
    # the rebound #6/#7 then materialize into geometric hints — the measured
    # anchor list at FEM time binds by (axis, dia, centre), not spec order
    centers = sorted(tuple(ld.region.center) for ld in spec.load_cases[0].loads)
    assert centers == [(0.0, -12.0, 25.0), (0.0, 12.0, 25.0)]
    assert all(ld.region.hole_index is None for ld in spec.load_cases[0].loads)
    assert all(ld.region.diameter_mm == 6.5 for ld in spec.load_cases[0].loads)
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
    assert kinds == {"hole"}
    # corner bolts materialize into (axis +z, dia 8.5, x/y) hints — indexes
    # are cleared so measured-anchor drift can't rebind them
    centers = sorted(tuple(r.center)[:2] for r in fixed)
    assert centers == [(-42.0, -32.0), (-42.0, 32.0), (42.0, -32.0), (42.0, 32.0)]
    assert all(r.hole_index is None for r in fixed)
    assert all(r.diameter_mm == 8.5 for r in fixed)
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
    # rebound to the M12 seat, then materialized as a geometric hint
    assert ld.region.hole_index is None
    assert ld.region.diameter_mm == 10.2
    assert list(ld.region.center) == [0.0, 0.0, 0.0]
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
    r = spec.load_cases[0].loads[0].region
    assert r.hole_index is None
    assert r.diameter_mm == 10.2


def _polluted_envelope_spec() -> PartSpec:
    """The r16 arm-mount failure shape: a SocketHeadCapScrew catalog dim
    (18.27) bled into `length_x` while the declared holes sit at x=±42 —
    the spec is self-contradictory before codegen ever runs."""
    return PartSpec(
        name="robot arm mount",
        summary="",
        length_x={"value": 18.27},
        width_y={"value": 80},
        height_z={"value": 40},
        features=[
            _hole("mounting hole +x +y", 8.5, "x=42 y=32"),
            _hole("mounting hole -x -y", 8.5, "x=-42 y=-32"),
        ],
        standard_parts=[],
        image_roles=[],
        verification_questions=[],
    )


def test_lint_envelope_fit_expands_undersized_envelope():
    spec = _polluted_envelope_spec()
    notes = lint_envelope_fit(spec)
    assert spec.length_x.value > 90.0  # 2*(42+4.25)+4 ≈ 96.5
    assert spec.width_y.value == 80  # 32+4.25 fits inside 80/2
    assert notes and "expanded" in notes[0]


def test_lint_envelope_fit_is_idempotent():
    spec = _polluted_envelope_spec()
    lint_envelope_fit(spec)
    grown = spec.length_x.value
    notes = lint_envelope_fit(spec)
    assert spec.length_x.value == grown
    assert notes == []


def test_lint_envelope_fit_never_shrinks():
    spec = _mount_spec()  # 100x80x40, holes at ±42/±32 — fits already
    notes = lint_envelope_fit(spec)
    assert spec.length_x.value == 100
    assert notes == []


def test_lint_envelope_fit_absurd_positions_only_warn():
    spec = _polluted_envelope_spec()
    spec.features[0].position = "x=800 y=0"  # implies ≥1.6 m — a parse error
    notes = lint_envelope_fit(spec)
    assert spec.length_x.value == 18.27  # kept as declared
    assert notes and ("review" in notes[0] or "kept as declared" in notes[0])


def test_materialize_clears_index_into_geometric_hint():
    """Spec-order `hole_index` binds the MEASURED list at FEM — materialize
    replaces it with the (axis, dia, centre) hint the spec anchor carries."""
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="arm load",
            fixed=[RegionSpec(kind="hole", hole_index=1)],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="bore", hole_index=6),
                    bearing_n=120.0,
                )
            ],
        )
    ]
    notes = materialize_anchor_hints(spec)
    fx = spec.load_cases[0].fixed[0]
    assert fx.hole_index is None
    assert list(fx.center) == [42.0, 32.0, 0.0]
    assert fx.diameter_mm == 8.5
    assert str(fx.axis).endswith("z")
    ld = spec.load_cases[0].loads[0].region
    assert ld.hole_index is None
    # +x side bore: in-plane evidence is (y,z), the axial slot stays 0
    assert list(ld.center) == [0.0, -12.0, 25.0]
    assert ld.diameter_mm == 6.5
    assert notes


def test_materialize_respects_existing_region_fields():
    """Region fields the model already set win — only the index is dropped
    once a usable centre exists."""
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="arm load",
            fixed=[],
            loads=[
                LoadSpec(
                    region=RegionSpec(
                        kind="bore",
                        hole_index=6,
                        axis="x",
                        diameter_mm=7.0,  # model's own estimate — kept
                        center=[],
                    ),
                    bearing_n=120.0,
                )
            ],
        )
    ]
    materialize_anchor_hints(spec)
    r = spec.load_cases[0].loads[0].region
    assert r.hole_index is None  # centre materialized → index cleared
    assert r.diameter_mm == 7.0
    assert r.axis == "x"


def test_materialize_keeps_index_on_estimated_anchor(monkeypatch):
    """An anchor whose position was estimated (no declared coordinates) is
    weaker evidence than the index itself — keep the index for the
    measured-order fallback."""
    import cadx.agent.anchors as anchors_mod
    from cadx.agent.anchors import Anchor

    est = Anchor(
        index=0,
        x=0.0,
        y=0.0,
        dia=8.5,
        axis="+z",
        pos=None,
        source="spec-est",
    )
    monkeypatch.setattr(anchors_mod, "_spec_anchors", lambda spec: [est])
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="arm load",
            fixed=[RegionSpec(kind="hole", hole_index=1)],
            loads=[LoadSpec(region=RegionSpec(kind="all"), g_load=[0.0, 0.0, -1.0])],
        )
    ]
    materialize_anchor_hints(spec)
    assert spec.load_cases[0].fixed[0].hole_index == 1


def test_materialize_leaves_out_of_range_for_drop_dead():
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="arm load",
            fixed=[RegionSpec(kind="hole", hole_index=99)],
            loads=[LoadSpec(region=RegionSpec(kind="all"), g_load=[0.0, 0.0, -1.0])],
        )
    ]
    sanitize_load_cases(spec)
    assert spec.load_cases[0].fixed == []  # dead ref dropped, honestly noted
    assert any("dropped" in a for a in spec.assumptions)


def test_sanitize_load_cases_is_idempotent():
    spec = _mount_spec()
    spec.load_cases = [
        LoadCase(
            name="arm load on flange holes",
            fixed=[RegionSpec(kind="hole", hole_index=i) for i in (1, 2, 3, 4)],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="bore", hole_index=8),
                    bearing_n=200.0,
                )
            ],
        )
    ]
    sanitize_load_cases(spec)
    first = spec.model_dump_json()
    sanitize_load_cases(spec)
    assert spec.model_dump_json() == first
