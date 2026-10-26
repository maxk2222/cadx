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
