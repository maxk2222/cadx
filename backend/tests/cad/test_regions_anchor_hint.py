"""Anchor regions bound by (axis, dia, center) hint instead of `hole_index`.

Stage-declared load cases cannot name a `hole_index` — the index is a
position in the measured anchor list, which depends on scan order and the
generated topology. `_anchor_geometry` therefore falls back to matching the
nearest anchor on the declared hint; the binding must be deterministic and
fail loudly (empty region + warning) when nothing matches.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.regions import _anchor_geometry, resolve_case_dofs
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec


def _block_with_bores() -> trimesh.Trimesh:
    """120x70x90 block: Ø30 bore along Y at (0, z=65), Ø8.5 along Z at (0,-22)."""
    block = trimesh.creation.box(extents=[120.0, 70.0, 90.0])
    block.apply_translation([0.0, 0.0, 45.0])
    bore = trimesh.creation.cylinder(radius=15.0, height=100.0, sections=48)
    bore.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    bore.apply_translation([0.0, 0.0, 65.0])
    bolt = trimesh.creation.cylinder(radius=4.25, height=100.0, sections=32)
    bolt.apply_translation([0.0, -22.0, 45.0])
    out = block.difference(bore, engine="manifold")
    return out.difference(bolt, engine="manifold")


# scanned anchors in an arbitrary order — the Ø30 seat is NOT index 1
_ANCHORS = [
    {
        "kind": "hole",
        "index": 1,
        "name": "bolt",
        "x": 0.0,
        "y": -22.0,
        "z": 45.0,
        "dia": 8.5,
        "axis": "+z",
    },
    {
        "kind": "hole",
        "index": 2,
        "name": "bearing",
        "x": 0.0,
        "y": 0.0,
        "z": 65.0,
        "dia": 30.0,
        "axis": "+y",
    },
]


@pytest.fixture(scope="module")
def domain():
    return voxelize(_block_with_bores(), 2.0, 400_000)


def test_hint_binds_nearest_matching_anchor(domain) -> None:
    reg = RegionSpec(kind="bore", axis="+y", center=[0, 0, 65], diameter_mm=30.0)
    g = _anchor_geometry(domain, reg, _ANCHORS)
    assert g is not None
    center, avec, dia, _half = g
    assert dia == pytest.approx(30.0, abs=1.5)
    np.testing.assert_allclose(center[[0, 2]], [0.0, 65.0], atol=3.0)
    assert abs(avec[1]) > 0.9


def test_hint_binding_is_order_independent(domain) -> None:
    reg = RegionSpec(kind="bore", axis="+y", center=[0, 0, 65], diameter_mm=30.0)
    g1 = _anchor_geometry(domain, reg, _ANCHORS)
    g2 = _anchor_geometry(domain, reg, list(reversed(_ANCHORS)))
    assert g1 is not None and g2 is not None
    np.testing.assert_allclose(g1[0], g2[0])
    assert g1[2] == g2[2]


def test_hint_axis_and_dia_filter(domain) -> None:
    # +z axis matches the bolt anchor but Ø30 cannot be an Ø8.5 hole
    reg = RegionSpec(kind="bore", axis="+z", center=[0, -22, 45], diameter_mm=30.0)
    assert _anchor_geometry(domain, reg, _ANCHORS) is None
    # dia matches the seat but the centre is nowhere near it
    reg = RegionSpec(kind="bore", axis="+y", center=[50, 0, 65], diameter_mm=30.0)
    assert _anchor_geometry(domain, reg, _ANCHORS) is None


def test_hint_nearest_of_same_dia_axis(domain) -> None:
    # two identical bolts — the hint must pick the nearer one, not index 1
    anchors = [
        {"kind": "hole", "index": 1, "x": -46.0, "y": 22.0, "dia": 8.5, "axis": "+z"},
        {"kind": "hole", "index": 2, "x": 0.0, "y": -22.0, "dia": 8.5, "axis": "+z"},
    ]
    reg = RegionSpec(kind="bore", axis="+z", center=[1, -22, 10], diameter_mm=8.5)
    g = _anchor_geometry(domain, reg, anchors)
    assert g is not None
    np.testing.assert_allclose(g[0][:2], [0.0, -22.0], atol=0.01)


def test_bearing_n_load_via_hint(domain) -> None:
    case = LoadCase(
        name="radial",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="bore", axis="+y", center=[0, 0, 65], diameter_mm=30.0),
                bearing_n=8000.0,
                bearing_dir=[0, 0, -1],
            )
        ],
    )
    applied: list[dict] = []
    _fixed3, F, warnings = resolve_case_dofs(domain, case, _ANCHORS, applied_out=applied)
    assert applied, warnings
    rec = applied[0]
    assert "bearing_n" in rec["mechanisms"]
    assert rec["force_mag_n"] == pytest.approx(8000.0, rel=0.05)
    assert rec["centroid_mm"][0] == pytest.approx(0.0, abs=6.0)
    assert rec["centroid_mm"][2] == pytest.approx(65.0, abs=6.0)
    res = F.reshape(-1, 3).sum(axis=0)
    assert res[2] == pytest.approx(-8000.0, rel=0.05)


def test_bearing_n_hint_miss_warns_not_binds(domain) -> None:
    case = LoadCase(
        name="radial",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="bore", axis="+y", center=[0, 0, 20], diameter_mm=30.0),
                bearing_n=8000.0,
                bearing_dir=[0, 0, -1],
            )
        ],
    )
    applied: list[dict] = []
    _fixed3, _F, warnings = resolve_case_dofs(domain, case, _ANCHORS, applied_out=applied)
    assert any("bearing_n" in w for w in warnings)
    assert not applied or "bearing_n" not in applied[0]["mechanisms"]


# --- interface_regions: unified keep/void protection per fastener anchor -----

from cadx.cad.fem.regions import interface_regions  # noqa: E402


def test_interface_regions_blind_signed_axis() -> None:
    """A -z blind tap (bottom-face M8: mouth z=0, depth 10, feature mid
    z=5) must emit a keep ring, a bounded void and a floor slab — never a
    full-height shaft."""
    anchors = [
        {
            "kind": "tap",
            "index": 1,
            "name": "M8",
            "axis": "-z",
            "pos": [-46.0, 22.0, 5.0],
            "dia": 6.8,
            "depth_mm": 10.0,
            "blind": True,
            "source": "spec",
        }
    ]
    keeps, voids = interface_regions(anchors, margin_mm=3.0)
    assert len(voids) == 1
    void = voids[0]
    assert void.kind == "cylinder" and void.axis == "-z"
    assert void.diameter_mm == pytest.approx(6.8 * 0.9)
    assert sorted(void.z_range) == [pytest.approx(0.0), pytest.approx(10.0)]
    # keep ring binds the anchor by index; blind floor slab sits behind the
    # bore bottom (bottom = axc - sgn*depth/2 = 5 + 5 = 10; slab [10, 13])
    assert any(k.kind == "hole" and k.hole_index == 1 for k in keeps)
    slabs = [k for k in keeps if k.kind == "cylinder"]
    assert len(slabs) == 1
    assert sorted(slabs[0].z_range) == [pytest.approx(10.0), pytest.approx(13.0)]


def test_interface_regions_tunnel_no_slab() -> None:
    """Through tunnels get keep+void but no floor slab."""
    anchors = [
        {
            "kind": "tunnel",
            "index": 1,
            "axis": "+y",
            "pos": [0.0, 0.0, 65.0],
            "dia": 30.0,
            "depth_mm": 70.0,
            "source": "scan+spec",
        }
    ]
    keeps, voids = interface_regions(anchors, margin_mm=3.0)
    assert [k.kind for k in keeps] == ["hole"]
    assert len(voids) == 1
    # z_range bounds the CYLINDER AXIS coordinate — for a +y tunnel that is
    # the y span (pos[1]=0 ± 70/2), not literal z
    assert sorted(voids[0].z_range) == [pytest.approx(-35.0), pytest.approx(35.0)]


def test_interface_regions_spec_only_filter() -> None:
    """spec_only restricts protection to spec-sourced anchors — pure
    scan/mesh extras stay designable; unmarked anchors err on protection."""
    anchors = [
        {"kind": "hole", "index": 1, "axis": "+z", "pos": [0, 0, 5], "dia": 5.0, "source": "spec"},
        {"kind": "hole", "index": 2, "axis": "+z", "pos": [9, 9, 5], "dia": 5.0, "source": "scan"},
        {
            "kind": "pocket",
            "index": 3,
            "axis": "+z",
            "pos": [3, 3, 5],
            "dia": 9.0,
            "source": "scan",
        },
        {"kind": "hole", "index": 4, "axis": "+z", "pos": [-9, -9, 5], "dia": 4.0},
    ]
    keeps, voids = interface_regions(anchors, margin_mm=3.0, spec_only=True)
    assert all(k.hole_index in (1, 4) for k in keeps)
    assert len(voids) == 2
    # without spec_only the scan hole is protected too, the pocket never is
    keeps_all, voids_all = interface_regions(anchors, margin_mm=3.0)
    assert {k.hole_index for k in keeps_all} == {1, 2, 4}
    assert len(voids_all) == 3


# --- expand_attaches: attach/mechanism sugar → concrete regions -------------

from cadx.cad.fem.regions import expand_attaches  # noqa: E402

_ATT_ANCHORS = [
    {
        "kind": "tap",
        "index": 1,
        "name": "M8 tap left",
        "axis": "-z",
        "pos": [-46.0, 22.0, 5.0],
        "x": -46.0,
        "y": 22.0,
        "dia": 6.8,
        "depth_mm": 10.0,
        "blind": True,
        "source": "spec",
    },
    {
        "kind": "bearing_seat",
        "index": 2,
        "name": "bearing bore",
        "axis": "+y",
        "pos": [0.0, 0.0, 65.0],
        "dia": 30.0,
        "depth_mm": 20.0,
        "source": "spec+scan",
    },
]


def test_expand_attaches_fixed_hole_index() -> None:
    case = LoadCase(
        name="mounts",
        fixed=[RegionSpec(attach={"name": "m8 tap left"})],
        loads=[],
    )
    out, warnings = expand_attaches(case, _ATT_ANCHORS)
    r = out.fixed[0]
    assert r.kind == "hole" and r.hole_index == 1 and r.axis == "-z"
    assert r.attach is None
    assert any("anchor #1" in w for w in warnings)


def test_expand_attaches_bearing_mechanism() -> None:
    case = LoadCase(
        name="radial",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(attach={"name": "bearing"}),
                mechanism="bearing",
                value_n=8000.0,
                dir=[0, 0, -1],
            )
        ],
    )
    out, _warnings = expand_attaches(case, _ATT_ANCHORS)
    ld = out.loads[0]
    assert ld.region.kind == "bore" and ld.region.hole_index == 2
    assert ld.bearing_n == 8000.0 and ld.bearing_dir == [0, 0, -1]
    assert ld.mechanism is None


def test_expand_attaches_clamp_and_axial_pull() -> None:
    case = LoadCase(
        name="bolt",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(attach={"hole_index": 1}),
                mechanism="clamp",
                value_n=3500.0,
            ),
            LoadSpec(
                region=RegionSpec(attach={"hole_index": 1}),
                mechanism="axial_pull",
                value_n=1200.0,
            ),
        ],
    )
    out, _ = expand_attaches(case, _ATT_ANCHORS)
    clamp, pull = out.loads
    # clamp keeps the material-ring region and sets preload
    assert clamp.region.kind == "hole" and clamp.preload_n == 3500.0
    # pull on a -z tap: force points toward the entry face (-z)
    assert pull.region.kind == "bore"
    assert pull.force_n == pytest.approx([0.0, 0.0, -1200.0])


def test_expand_attaches_axial_push_and_torque() -> None:
    case = LoadCase(
        name="t",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(attach={"hole_index": 2}),
                mechanism="axial_push",
                value_n=900.0,
            ),
            LoadSpec(
                region=RegionSpec(attach={"hole_index": 2}),
                mechanism="torque",
                value_nm=4.0,
            ),
        ],
    )
    out, warnings = expand_attaches(case, _ATT_ANCHORS)
    push, tor = out.loads
    # push into the part on a +y bore: -y
    assert push.force_n == pytest.approx([0.0, -900.0, 0.0])
    # torque around the +y feature axis on a bearing seat — allowed kind
    assert tor.moment_nm == pytest.approx([0.0, 4.0, 0.0])
    assert not any("torque" in w for w in warnings)


def test_expand_attaches_torque_rejects_plain_hole() -> None:
    anchors = [dict(_ATT_ANCHORS[0], kind="hole")]
    case = LoadCase(
        name="t",
        fixed=[RegionSpec(kind="face", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(attach={"hole_index": 1}),
                mechanism="torque",
                value_nm=4.0,
            )
        ],
    )
    out, warnings = expand_attaches(case, anchors)
    assert out.loads[0].moment_nm == []  # not expanded
    assert any("torque" in w and "threaded" in w for w in warnings)


def test_expand_attaches_unresolvable_fails_loud() -> None:
    """An attach that matches nothing must not bind a wrong surface — the
    expanded hole_index lands out of range so the region resolves empty and
    the unresolved-region checks fire."""
    case = LoadCase(
        name="x",
        fixed=[RegionSpec(attach={"name": "no-such-feature"})],
        loads=[],
    )
    out, warnings = expand_attaches(case, _ATT_ANCHORS)
    assert any("unresolvable" in w for w in warnings)
    assert out.fixed[0].hole_index == len(_ATT_ANCHORS) + 999


def test_expand_attaches_ambiguous_name_warns() -> None:
    anchors = [
        *_ATT_ANCHORS,
        dict(_ATT_ANCHORS[0], index=3, name="M8 tap right", pos=[46.0, 22.0, 5.0]),
    ]
    case = LoadCase(
        name="x",
        fixed=[RegionSpec(attach={"name": "m8 tap"})],
        loads=[],
    )
    out, warnings = expand_attaches(case, anchors)
    assert any("ambiguous" in w for w in warnings)
    assert out.fixed[0].hole_index == len(anchors) + 999


def test_expand_attaches_pos_hint() -> None:
    case = LoadCase(
        name="x",
        fixed=[RegionSpec(attach={"pos": [-46.5, 22.3, 0], "axis": "-z", "dia": 6.8})],
        loads=[],
    )
    out, warnings = expand_attaches(case, _ATT_ANCHORS)
    assert out.fixed[0].hole_index == 1
    assert any("anchor #1" in w for w in warnings)


def test_bore_region_is_a_wall_shell_not_a_sleeve(domain) -> None:
    """The bore load region doubles as the TopOpt passive pin — a band of
    0.35·dia froze a Ø51 sleeve around a Ø30 seat (~19% of the domain). The
    shell must stay ~1.5 cells / 0.15·dia thick: cells exist on the wall but
    the radial extent stays bounded."""
    from cadx.cad.fem.regions import _bore_cells

    r = RegionSpec(kind="bore", axis="+y", center=[0, 0, 65], diameter_mm=30.0)
    cells = _bore_cells(domain, r, _ANCHORS)
    assert cells.any()
    centers = domain.element_centers().reshape((*domain.dims, 3))[cells]
    # radial distance from the +y bore axis through (x=0, z=65)
    rad = np.sqrt((centers[:, 0] - 0.0) ** 2 + (centers[:, 2] - 65.0) ** 2)
    band = max(domain.pitch * 1.5, 0.15 * 30.0, 2.0)
    assert rad.min() >= 15.0 - domain.pitch  # wall side of the channel
    assert rad.max() <= 15.0 + band + domain.pitch  # thin shell, not a sleeve
    # regression guard: old 0.35·dia band would reach r≈25.5 on this domain
    assert rad.max() < 15.0 + 0.35 * 30.0
