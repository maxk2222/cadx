from __future__ import annotations

import numpy as np

from cadx.cad.fem.extract import (
    density_to_mesh,
    fill_enclosed_voids,
    sealed_void_metrics,
    thin_walled_sealed_voids,
)
from cadx.cad.fem.voxelize import VoxelDomain


def test_sealed_void_metrics_empty_and_single_cavity():
    solid = np.ones((9, 9, 9), dtype=bool)
    assert sealed_void_metrics(solid, 2.0) == {
        "enclosed_void_mm3": 0.0,
        "n_sealed_cavities": 0,
        "largest_void_mm3": 0.0,
        "sealed_cavities_mm3": [],
    }

    solid[3:5, 3:5, 3:5] = False
    metrics = sealed_void_metrics(solid, 2.0)
    assert metrics == {
        "enclosed_void_mm3": 64.0,
        "n_sealed_cavities": 1,
        "largest_void_mm3": 64.0,
        "sealed_cavities_mm3": [64.0],
    }


def test_sealed_void_metrics_ignores_open_air_and_sorts_cavities():
    solid = np.ones((11, 11, 11), dtype=bool)
    solid[2:4, 2:4, 2:4] = False
    solid[6:9, 6:9, 6:9] = False
    metrics = sealed_void_metrics(solid, 1.0)
    assert metrics["n_sealed_cavities"] == 2
    assert metrics["enclosed_void_mm3"] == 35.0
    assert metrics["largest_void_mm3"] == 27.0
    assert metrics["sealed_cavities_mm3"] == [27.0, 8.0]

    solid[0:4, 2, 2] = False
    assert sealed_void_metrics(solid, 1.0)["n_sealed_cavities"] == 1


def test_sealed_void_metrics_requires_3d_mask_and_positive_pitch():
    with np.testing.assert_raises(ValueError):
        sealed_void_metrics(np.ones((3, 3), dtype=bool), 1.0)
    with np.testing.assert_raises(ValueError):
        sealed_void_metrics(np.ones((3, 3, 3), dtype=bool), 0.0)


def test_fill_enclosed_voids_preserves_carved_cells_and_open_channels():
    mask = np.ones((9, 9, 9), dtype=bool)
    mask[3:6, 3:6, 3:6] = False
    filled, added = fill_enclosed_voids(mask)
    assert int(added.sum()) == 27
    assert filled.all()

    carved = np.zeros_like(mask)
    carved[4, 4, 4] = True
    carved_fill, carved_added = fill_enclosed_voids(mask, carved)
    assert not carved_fill[4, 4, 4]
    assert int(carved_added.sum()) == 26

    open_mask = mask.copy()
    open_mask[0:5, 4, 4] = False
    open_fill, open_added = fill_enclosed_voids(open_mask)
    assert int(open_added.sum()) == 0
    assert np.array_equal(open_fill, open_mask)


def test_thin_walled_sealed_voids_flags_only_subminimum_walls():
    thin = np.zeros((11, 11, 11), dtype=bool)
    thin[2:9, 2:9, 2:9] = True
    thin[3:8, 3:8, 3:8] = False
    thin_pockets, thinest = thin_walled_sealed_voids(thin, 1.0, 3.0)
    assert thin_pockets.any()
    assert thinest is not None and thinest < 3.0

    thick = np.zeros((13, 13, 13), dtype=bool)
    thick[1:12, 1:12, 1:12] = True
    thick[4:9, 4:9, 4:9] = False
    thick_pockets, thickest = thin_walled_sealed_voids(thick, 1.0, 3.0)
    assert not thick_pockets.any()
    assert thickest is not None and thickest >= 3.0


def test_density_to_mesh_opt_in_void_fill_and_shell_warning():
    dims = (15, 15, 15)
    domain = VoxelDomain(
        origin=np.zeros(3),
        pitch=1.0,
        dims=dims,
        solid=np.ones(dims, dtype=bool),
    )
    mask = np.zeros(dims, dtype=bool)
    mask[2:13, 2:13, 2:13] = True
    mask[4:11, 4:11, 4:11] = False
    density = mask.astype(float).ravel()

    _mesh, normal_mask, _warnings, _n, normal_info = density_to_mesh(
        domain, density, smooth_iterations=0, extract_smooth=False
    )
    assert sealed_void_metrics(normal_mask, 1.0)["n_sealed_cavities"] == 1
    assert normal_info["sealed_voids_filled_mm3"] == 0.0

    _mesh, filled_mask, _warnings, _n, filled_info = density_to_mesh(
        domain,
        density,
        smooth_iterations=0,
        extract_smooth=False,
        fill_sealed_voids=True,
    )
    assert sealed_void_metrics(filled_mask, 1.0)["n_sealed_cavities"] == 0
    assert filled_info["sealed_voids_filled_mm3"] == 343.0

    _mesh, flagged_mask, warnings, _n, flagged_info = density_to_mesh(
        domain,
        density,
        smooth_iterations=0,
        extract_smooth=False,
        min_shell_mm=10.0,
    )
    assert sealed_void_metrics(flagged_mask, 1.0)["n_sealed_cavities"] == 1
    assert flagged_info["shell_min_wall_mm"] is not None
    assert any("below min_shell_mm" in warning for warning in warnings)
