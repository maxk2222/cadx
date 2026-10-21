"""Sub-voxel surface extraction (`cad/fem/extract.py`, quality plan WS-1).

Marching cubes on the Gaussian-smoothed density clipped to the verified
mask, with a bisected volume-preserving iso level — instead of MC at 0.5
on the binary mask (staircase surface + Taubin shrinkage).
"""

from __future__ import annotations

import numpy as np
import pytest

from cadx.cad.fem.extract import _volume_preserved_iso, density_to_mask, density_to_mesh
from cadx.cad.fem.voxelize import VoxelDomain


def _domain(n: int = 24, pitch: float = 1.0) -> VoxelDomain:
    """Fully-solid cubic domain — the design space for the density fields."""
    dims = (n, n, n)
    return VoxelDomain(origin=np.zeros(3), pitch=pitch, dims=dims, solid=np.ones(dims, bool))


def _sphere_density(dom: VoxelDomain, r_in: float = 6.0, r_out: float = 9.0) -> np.ndarray:
    """Radial density: 1 inside r_in cells, 0 outside r_out, linear ramp in
    between — a SIMP-like field with a gray transition rim."""
    idx = np.indices(dom.dims, dtype=float) + 0.5  # (3, nx, ny, nz) cell centres
    center = (np.asarray(dom.dims, dtype=float) / 2.0)[:, None, None, None]
    r = np.linalg.norm(idx - center, axis=0)
    return np.clip((r_out - r) / (r_out - r_in), 0.0, 1.0)


def test_smooth_extract_watertight_and_volume_matched() -> None:
    dom = _domain()
    rho = _sphere_density(dom)
    mesh, mask, _w, _nc, info = density_to_mesh(dom, rho.ravel(), smooth_iterations=0)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    mask_vol = float(mask.sum()) * dom.pitch**3
    err = abs(mesh.volume - mask_vol) / mask_vol
    assert err < 0.03
    assert info["smooth"] is True
    assert 0.0 < info["iso"] < 1.0
    assert abs(info["vol_error_pct"]) < 3.0
    assert info["gray_fraction"] > 0.0  # the ramp rim keeps gray cells in the mask
    # the sub-voxel surface must not leave the design space
    hi = dom.origin + dom.pitch * np.asarray(dom.dims)
    assert (mesh.bounds[0] >= dom.origin - 1e-6).all()
    assert (mesh.bounds[1] <= hi + 1e-6).all()


def test_extract_smooth_off_is_legacy_path() -> None:
    """extract_smooth=False must reproduce the binary-mask MC path — the
    returned mask is the density_to_mask result, bit-identical, and the
    smooth path shares the very same mask (FEM semantics unchanged)."""
    dom = _domain()
    rho = _sphere_density(dom)
    _mesh, mask_legacy, _w, _nc, info = density_to_mesh(
        dom, rho.ravel(), smooth_iterations=0, extract_smooth=False
    )
    mask_ref, _w_ref, _nc_ref = density_to_mask(dom, rho.ravel())
    np.testing.assert_array_equal(mask_legacy, mask_ref)
    assert info["smooth"] is False
    assert info["iso"] == 0.5
    _mesh2, mask_smooth, _w2, _nc2, _i2 = density_to_mesh(dom, rho.ravel(), smooth_iterations=0)
    np.testing.assert_array_equal(mask_smooth, mask_legacy)


def test_volume_preserve_beats_fixed_iso() -> None:
    """On a field whose mask cells sit below 1.0 (sub-unity core + gray
    rim) a fixed 0.5 iso on the smoothed field systematically under-fills;
    the bisected iso must land closer to the mask volume."""
    dom = _domain()
    idx = np.indices(dom.dims, dtype=float) + 0.5
    center = (np.asarray(dom.dims, dtype=float) / 2.0)[:, None, None, None]
    r = np.linalg.norm(idx - center, axis=0)
    rho = np.clip(0.8 * (9.0 - r) / 2.0, 0.0, 0.8)  # core at rho=0.8
    _m_vp, mask_vp, _w1, _n1, info_vp = density_to_mesh(
        dom, rho.ravel(), smooth_iterations=0, volume_preserve=True
    )
    _m_fx, mask_fx, _w2, _n2, info_fx = density_to_mesh(
        dom, rho.ravel(), smooth_iterations=0, volume_preserve=False
    )
    np.testing.assert_array_equal(mask_vp, mask_fx)  # same verified mask
    assert abs(info_vp["vol_error_pct"]) < abs(info_fx["vol_error_pct"])


def test_bisect_fallback_on_zero_field() -> None:
    """Pathological field: a pure passive voxel with rho_s == 0 makes the
    iso target unreachable -> warning + t = 0.5*max, no crash (quad
    fallback still yields a solid mesh)."""
    dom = _domain()
    passive = np.zeros(dom.dims, dtype=bool)
    passive[3, 3, 3] = True
    rho = np.zeros(dom.dims)
    mesh, mask, warnings, _nc, info = density_to_mesh(
        dom, rho.ravel(), passive_solid=passive.ravel(), smooth_iterations=0
    )
    assert int(mask.sum()) == 1
    assert any("volume-preserving iso" in w for w in warnings)
    assert mesh.is_watertight
    assert info["iso"] == 0.0  # 0.5 * field max (= 0)


def test_rank_iso_fallback_handles_tied_values() -> None:
    """When ties make the target count unreachable by threshold bisection,
    the rank fallback chooses the closest threshold instead of 0.5*max."""
    field = np.array([0.0] * 4 + [0.4] * 3 + [0.8] * 3 + [1.0] * 2)
    iso, warning = _volume_preserved_iso(field, target=6, max_iter=1)
    assert iso > 0.4
    assert int((field >= iso).sum()) == 5
    assert warning is not None and "rank threshold" in warning


def test_empty_density_raises() -> None:
    dom = _domain()
    with pytest.raises(ValueError, match="empty part"):
        density_to_mesh(dom, np.zeros(dom.cell_count))
