"""Region kinds rim/border, annulus cylinders, z_range and normal_force_n.

- rim: outer radial band of `margin_mm` following the part silhouette —
  a gear tooth ring selects tips + root band, never the hub.
- border: margin band at the bbox outer edges along the named axes
  (letters = both sides, signed = one side).
- cylinder + inner_diameter_mm: exact annulus (no margin slop).
- z_range: literal-Z clip for area/surface picks.
- normal_force_n: total force along the region's inward skin normal.
- fixed region kind="all" is refused (it would rigid-lock every DOF).
- _anchor_geometry: an anchor without measured axial position estimates
  the pierced wall's mid instead of dropping onto the bbox mid-plane.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadx.cad.fem.regions import (
    _anchor_geometry,
    case_anchor_keep_mask,
    region_cell_mask,
    resolve_case_dofs,
)
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec


def _grid_centers(dom: VoxelDomain) -> np.ndarray:
    return dom.element_centers().reshape((*dom.dims, 3))


def _gear_domain() -> tuple[VoxelDomain, np.ndarray]:
    """32x32x4 grid @ 2 mm: 16-tooth gear around +z — root radius 25 mm,
    tip radius 29 mm. Returns (domain, radial distance per cell)."""
    pitch = 2.0
    dims = (32, 32, 4)
    origin = np.array([-32.0, -32.0, 0.0])
    ii, jj, _ = np.meshgrid(
        np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij"
    )
    cx = origin[0] + (ii + 0.5) * pitch
    cy = origin[1] + (jj + 0.5) * pitch
    r = np.hypot(cx, cy)
    ang = np.arctan2(cy, cx)
    r_max = 25.0 + 4.0 * (np.cos(16.0 * ang) > 0.0)
    solid = np.broadcast_to(r <= r_max, dims).copy()
    dom = VoxelDomain(origin=origin, pitch=pitch, dims=dims, solid=solid)
    return dom, r


def _block_domain(
    dims: tuple[int, int, int], pitch: float, origin: tuple[float, float, float]
) -> VoxelDomain:
    return VoxelDomain(
        origin=np.asarray(origin, dtype=float),
        pitch=pitch,
        dims=dims,
        solid=np.ones(dims, dtype=bool),
    )


def _disc_domain() -> tuple[VoxelDomain, np.ndarray]:
    """64x64x4 grid @ 4 mm: solid disc r <= 124 around +z."""
    pitch = 4.0
    dims = (64, 64, 4)
    origin = np.array([-128.0, -128.0, 0.0])
    ii, jj, _ = np.meshgrid(
        np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij"
    )
    cx = origin[0] + (ii + 0.5) * pitch
    cy = origin[1] + (jj + 0.5) * pitch
    r = np.hypot(cx, cy)
    solid = np.broadcast_to(r <= 124.0, dims).copy()
    dom = VoxelDomain(origin=origin, pitch=pitch, dims=dims, solid=solid)
    return dom, r


def _stepped_domain() -> VoxelDomain:
    """50x40x20 grid @ 2 mm: 10 mm plate (z 0..10) plus a 30x30x30 flange
    (z 10..40) — and a Ø8 bore at (25, 0) through the PLATE only. The bbox
    z-mid is 20 but the pierced wall mid is 5."""
    pitch = 2.0
    dims = (50, 40, 20)
    origin = np.array([-50.0, -40.0, 0.0])
    ii, jj, kk = np.meshgrid(
        np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij"
    )
    cx = origin[0] + (ii + 0.5) * pitch
    cy = origin[1] + (jj + 0.5) * pitch
    cz = origin[2] + (kk + 0.5) * pitch
    solid = cz <= 10.0
    solid |= (cx >= -40.0) & (cx <= -10.0) & (np.abs(cy) <= 15.0) & (cz > 10.0)
    solid &= np.hypot(cx - 25.0, cy) > 4.0  # Ø8 bore through the plate
    return VoxelDomain(origin=origin, pitch=pitch, dims=dims, solid=solid)


# --------------------------------------------------------------------------- #
# rim                                                                          #
# --------------------------------------------------------------------------- #


def test_rim_selects_tooth_ring_not_hub() -> None:
    """rim margin=4 on the gear: tooth tips + a shallow band under the roots,
    <40% of the cells — a plain big cylinder would swallow the whole part."""
    dom, r = _gear_domain()
    mask = region_cell_mask(dom, RegionSpec(kind="rim", axis="+z", margin_mm=4.0))
    assert mask.any()
    frac = float(mask.sum()) / dom.solid_count
    assert frac < 0.40
    # tooth tips are inside the band…
    assert (mask & (r >= 27.0)).any()
    # …the hub is not
    assert not (mask & (r <= 15.0)).any()


def test_rim_default_axis_is_plus_z() -> None:
    assert RegionSpec(kind="rim").axis == "+z"
    dom, _ = _gear_domain()
    assert region_cell_mask(dom, RegionSpec(kind="rim", margin_mm=4.0)).any()


def test_rim_exact_annulus_via_diameters() -> None:
    """diameter_mm + inner_diameter_mm -> exact ring, silhouette ignored."""
    dom, r = _disc_domain()
    mask = region_cell_mask(
        dom,
        RegionSpec(kind="rim", diameter_mm=236.0, inner_diameter_mm=208.0),
    )
    assert mask.any()
    rr = r[mask]
    assert rr.min() >= 103.9
    assert rr.max() <= 118.1


# --------------------------------------------------------------------------- #
# border                                                                       #
# --------------------------------------------------------------------------- #


def test_border_x_picks_both_side_strips() -> None:
    """80x60x6 plate, border axis='x' margin 2: strips at x<=-38 and x>=38."""
    dom = _block_domain((80, 60, 6), 1.0, (-40.0, -30.0, 0.0))
    mask = region_cell_mask(dom, RegionSpec(kind="border", axis="x", margin_mm=2.0))
    cx = _grid_centers(dom)[..., 0]
    xs = cx[mask]
    assert mask.sum() == 1440  # 2 cell-columns per side
    assert ((xs <= -38.0) | (xs >= 38.0)).all()
    assert (xs <= -38.0).any() and (xs >= 38.0).any()


def test_border_signed_axis_picks_one_side() -> None:
    dom = _block_domain((80, 60, 6), 1.0, (-40.0, -30.0, 0.0))
    mask = region_cell_mask(dom, RegionSpec(kind="border", axis="+x", margin_mm=2.0))
    cx = _grid_centers(dom)[..., 0]
    assert mask.sum() == 720
    assert (cx[mask] >= 38.0).all()


def test_border_default_axis_xy() -> None:
    assert RegionSpec(kind="border").axis == "xy"
    dom = _block_domain((80, 60, 6), 1.0, (-40.0, -30.0, 0.0))
    mask = region_cell_mask(dom, RegionSpec(kind="border", margin_mm=2.0))
    cx = _grid_centers(dom)[..., 0]
    cy = _grid_centers(dom)[..., 1]
    # lateral frame: x strips OR y strips
    sel = (cx <= -38.0) | (cx >= 38.0) | (cy <= -28.0) | (cy >= 28.0)
    np.testing.assert_array_equal(mask, sel & dom.solid)


# --------------------------------------------------------------------------- #
# cylinder annulus + z_range                                                   #
# --------------------------------------------------------------------------- #


def test_cylinder_annulus_exact_ring() -> None:
    """diameter 236 / inner 208 -> ring r in [104, 118], hub excluded."""
    dom, r = _disc_domain()
    mask = region_cell_mask(
        dom,
        RegionSpec(
            kind="cylinder",
            axis="+z",
            diameter_mm=236.0,
            inner_diameter_mm=208.0,
        ),
    )
    assert mask.any()
    rr = r[mask]
    assert rr.min() >= 103.9
    assert rr.max() <= 118.1
    # roughly the ring area fraction of the r<=124 disc
    frac = float(mask.sum()) / dom.solid_count
    assert 0.15 < frac < 0.26


def test_area_z_range_clips_cells() -> None:
    dom = _block_domain((10, 10, 50), 1.0, (-5.0, -5.0, 0.0))
    mask = region_cell_mask(
        dom,
        RegionSpec(kind="area", x_range=[0.0, 5.0], z_range=[20.0, 40.0]),
    )
    cz = _grid_centers(dom)[..., 2]
    cx = _grid_centers(dom)[..., 0]
    assert mask.sum() == 5 * 10 * 20
    assert (cz[mask] >= 20.0).all() and (cz[mask] <= 40.0).all()
    assert (cx[mask] >= 0.0).all() and (cx[mask] <= 5.0).all()


def test_surface_z_range_clips_skin() -> None:
    """surface +x on a tall block: the +x skin restricted to z in [20,40]."""
    dom = _block_domain((10, 10, 50), 1.0, (-5.0, -5.0, 0.0))
    full = region_cell_mask(dom, RegionSpec(kind="surface", axis="+x", margin_mm=2.0))
    clipped = region_cell_mask(
        dom,
        RegionSpec(kind="surface", axis="+x", margin_mm=2.0, z_range=[20.0, 40.0]),
    )
    cz = _grid_centers(dom)[..., 2]
    assert clipped.any() and clipped.sum() < full.sum()
    assert (cz[clipped] >= 20.0).all() and (cz[clipped] <= 40.0).all()


# --------------------------------------------------------------------------- #
# normal_force_n                                                               #
# --------------------------------------------------------------------------- #


def test_normal_force_pushes_into_surface() -> None:
    """normal_force_n=200 on surface +z -> resultant [0, 0, -200] N."""
    dom = _block_domain((20, 20, 2), 2.0, (-20.0, -20.0, 0.0))
    case = LoadCase(
        name="nf",
        fixed=[RegionSpec(kind="surface", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="surface", axis="+z"),
                normal_force_n=200.0,
            )
        ],
    )
    _, F, _ = resolve_case_dofs(dom, case)
    res = F.reshape(-1, 3).sum(axis=0)
    assert res == pytest.approx([0.0, 0.0, -200.0], abs=1e-6)


def test_normal_force_negative_pulls() -> None:
    dom = _block_domain((20, 20, 2), 2.0, (-20.0, -20.0, 0.0))
    case = LoadCase(
        name="nf2",
        fixed=[RegionSpec(kind="surface", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="surface", axis="+z"),
                normal_force_n=-50.0,
            )
        ],
    )
    _, F, _ = resolve_case_dofs(dom, case)
    res = F.reshape(-1, 3).sum(axis=0)
    assert res == pytest.approx([0.0, 0.0, 50.0], abs=1e-6)


# --------------------------------------------------------------------------- #
# fixed 'all' refused                                                          #
# --------------------------------------------------------------------------- #


def test_fixed_all_is_skipped_with_warning() -> None:
    dom = _block_domain((20, 20, 2), 2.0, (-20.0, -20.0, 0.0))
    case = LoadCase(
        name="bad",
        fixed=[RegionSpec(kind="all")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="surface", axis="+z"),
                force_n=[0.0, 0.0, -50.0],
            )
        ],
    )
    fixed3, F, warnings = resolve_case_dofs(dom, case)
    assert not fixed3.any()
    assert any("fixed region 'all' is invalid" in w for w in warnings)
    # the load still resolved — only the lock-all constraint was refused
    assert np.any(F)


# --------------------------------------------------------------------------- #
# anchor axial estimate + case keep mask                                       #
# --------------------------------------------------------------------------- #


def test_anchor_axial_mid_estimates_pierced_wall() -> None:
    """No z/pos on the anchor: the bore through the 10 mm plate resolves at
    z≈5 (wall mid), not at the bbox mid-plane z=20."""
    dom = _stepped_domain()
    anchors = [
        {
            "x": 25.0,
            "y": 0.0,
            "dia": 8.0,
            "axis": "+z",
            "depth_mm": 10.0,
            "kind": "hole",
        }
    ]
    g = _anchor_geometry(dom, RegionSpec(kind="hole", hole_index=1), anchors)
    assert g is not None
    assert g[0][2] == pytest.approx(5.0, abs=2.5)
    # an explicit z still wins over the estimate
    anchors[0]["z"] = 30.0
    g2 = _anchor_geometry(dom, RegionSpec(kind="hole", hole_index=1), anchors)
    assert g2 is not None and g2[0][2] == pytest.approx(30.0)


def test_case_anchor_keep_mask_unions_bore_rings() -> None:
    dom = _stepped_domain()
    anchors = [
        {
            "x": 25.0,
            "y": 0.0,
            "dia": 8.0,
            "axis": "+z",
            "depth_mm": 10.0,
            "pos": [25.0, 0.0, 5.0],
            "z": 5.0,
            "kind": "hole",
        }
    ]
    case = LoadCase(
        name="bearing",
        fixed=[RegionSpec(kind="surface", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="bore", hole_index=1),
                bearing_n=50.0,
                bearing_dir=[0.0, -1.0, 0.0],
            )
        ],
    )
    keep = case_anchor_keep_mask(dom, [case], anchors)
    assert keep.shape == dom.dims
    assert keep.any()
    # ring cells hug the bore at (25,0) and live in the plate, not the flange
    centers = _grid_centers(dom)
    rr = np.hypot(centers[..., 0] - 25.0, centers[..., 1])
    assert (rr[keep] <= 4.0 + 2.0 + 2.0).all()
    assert (centers[..., 2][keep] <= 10.0).all()
    # non-anchor cases contribute nothing
    case2 = LoadCase(
        name="plain",
        fixed=[RegionSpec(kind="surface", axis="-z")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="surface", axis="+z"),
                force_n=[0.0, 0.0, -10.0],
            )
        ],
    )
    assert not case_anchor_keep_mask(dom, [case2], anchors).any()


def _boss_on_plate_domain() -> VoxelDomain:
    """Plate 40x40x10 with a centred 10x10x15 boss — +z surfaces exist at
    z~8.75 (plate shoulder ring) and z~23.75 (boss top)."""
    pitch = 2.5
    dims = (16, 16, 10)
    origin = np.array([-20.0, -20.0, 0.0])
    ii, jj, kk = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
    cx = origin[0] + (ii + 0.5) * pitch
    cy = origin[1] + (jj + 0.5) * pitch
    cz = origin[2] + (kk + 0.5) * pitch
    plate = cz < 10.0
    boss = (np.abs(cx) <= 5.0) & (np.abs(cy) <= 5.0) & (cz < 25.0)
    return VoxelDomain(origin=origin, pitch=pitch, dims=dims, solid=plate | boss)


def test_region_fallback_prefers_hint_elevation():
    """+z fallback hinted at a boss top must not bind the plate shoulder —
    the axial band around the hint wins over the shared in-plane disc."""
    from cadx.cad.fem.regions import _region_fallback

    dom = _boss_on_plate_domain()
    r = RegionSpec(kind="surface", axis="+z", center=[0.0, 0.0, 25.0], margin_mm=2.0)
    mask, ax = _region_fallback(dom, r, None)
    assert ax == "+z" and mask.any()
    zsel = dom.element_centers().reshape((*dom.dims, 3))[..., 2][mask]
    assert zsel.min() > 15.0  # boss-top band only — plate shoulder excluded


def test_region_fallback_z_range_filters_face():
    """z_range clips the fallback face before the hint band is applied."""
    from cadx.cad.fem.regions import _region_fallback

    dom = _boss_on_plate_domain()
    r = RegionSpec(
        kind="surface",
        axis="+z",
        center=[0.0, 0.0, 25.0],
        z_range=[18.0, 30.0],
        margin_mm=2.0,
    )
    mask, _ = _region_fallback(dom, r, None)
    assert mask.any()
    zsel = dom.element_centers().reshape((*dom.dims, 3))[..., 2][mask]
    assert zsel.min() >= 18.0


def test_auto_void_regions_pins_bore_cylinders():
    """Anchored bores become empty-pinned cylinders so a grown design domain
    cannot fill their interior."""
    from cadx.cad.fem.regions import auto_void_regions

    anchors = [
        {"index": 1, "axis": "+z", "x": 10.0, "y": -5.0, "z": 4.0, "dia": 8.5},
        {"index": 2, "axis": "+x", "x": 0.0, "y": 1.0, "z": 2.0, "dia": 6.0},
    ]
    voids = auto_void_regions(anchors)
    assert len(voids) == 2
    z0 = voids[0]
    assert z0.kind == "cylinder" and z0.axis == "+z"
    assert z0.center == [10.0, -5.0, 4.0]
    assert z0.diameter_mm == pytest.approx(8.5 * 0.9)
    # hole anchors must never become void cells material — diameter stays
    # strictly below the measured bore so the wall ring survives
    x1 = voids[1]
    assert x1.axis == "+x" and x1.diameter_mm == pytest.approx(6.0 * 0.9)


def test_auto_void_regions_skips_bad_anchors():
    from cadx.cad.fem.regions import auto_void_regions

    assert auto_void_regions([]) == []
    assert auto_void_regions([{"index": 1, "axis": "+z", "x": 1.0, "y": 0.0}]) == []
    assert auto_void_regions([{"index": 1, "axis": "+z", "dia": 0.0}]) == []
    assert auto_void_regions([{"axis": "+z", "x": 1.0, "y": 0.0, "dia": 5.0}]) == []  # no index
