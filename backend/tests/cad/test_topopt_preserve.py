"""TopOpt export preservation: passive pinning + carved-void extraction.

- ``smooth_preserve(pin_mask=...)`` pins every vertex whose voxel cell is a
  passive (keep/fix/load) cell, so the surfaces the FEM verified do not
  drift under Taubin smoothing — while the bbox extent stays exact.
- ``density_to_mesh(void_cells=...)`` clamps carved cells to rho=0 before
  the Gaussian and keeps the 1-cell dilation shell out of them, so the
  iso surface cannot grow back into a bore and close it.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from cadx.cad.fem.extract import density_to_mesh
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.cad.mesh_ops import smooth_preserve


def _pin_frame(
    mesh: trimesh.Trimesh, pitch: float = 2.0
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, np.ndarray]:
    """Voxel frame covering `mesh` plus a quarter-cell pad, and the passive
    mask marking the cells of the six bbox-extreme vertices — pinning them
    freezes the extents so the bbox rescale never fires."""
    origin = mesh.bounds[0] - 0.25 * pitch
    dims = tuple(np.ceil((mesh.bounds[1] - origin) / pitch).astype(int))
    ijk = np.clip(
        np.floor((mesh.vertices - origin) / pitch).astype(int),
        0,
        np.asarray(dims) - 1,
    )
    passive = np.zeros(dims, dtype=bool)
    for ax in range(3):
        passive[tuple(ijk[int(np.argmin(mesh.vertices[:, ax]))])] = True
        passive[tuple(ijk[int(np.argmax(mesh.vertices[:, ax]))])] = True
    return origin, dims, ijk, passive


def test_smooth_preserve_pin_mask_holds_passive_cells() -> None:
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    origin, _dims, ijk, passive = _pin_frame(m)
    in_pinned_cell = passive[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    assert in_pinned_cell.any()

    out = smooth_preserve(m, iterations=5, pin_floor=False, pin_mask=(origin, 2.0, passive))
    assert out.is_watertight
    # bbox extents preserved exactly (pinned extremes -> no rescale fired)
    np.testing.assert_allclose(out.extents, m.extents, atol=1e-9)
    # every vertex inside a passive cell survived unmoved (set-based check —
    # merge_vertices may reorder the vertex table)
    d, _ = cKDTree(out.vertices).query(m.vertices[in_pinned_cell])
    assert float(d.max()) < 1e-4
    # smoothing still ran on the free surface — not a no-op pass
    free_d, _ = cKDTree(m.vertices).query(out.vertices)
    assert float(free_d.mean()) > 1e-6


def test_smooth_preserve_pin_mask_blocks_cell_shrinkage() -> None:
    """Control: without pin_mask the same vertices DO move — the mask is
    what holds them (otherwise the <1e-4 bound above is vacuous)."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    _origin, _dims, ijk, passive = _pin_frame(m)
    in_pinned_cell = passive[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    ref = smooth_preserve(m, iterations=5, pin_floor=False)
    # positions of the would-be-pinned vertices changed measurably
    d, _ = cKDTree(ref.vertices).query(m.vertices[in_pinned_cell])
    assert float(d.max()) > 1e-4


def test_density_to_mesh_keeps_void_open() -> None:
    """Block with a through-bore: the extraction must not grow the shell
    back into the carved cells — a ray down the bore centre hits nothing."""
    dims = (16, 16, 6)
    dom = VoxelDomain(origin=np.zeros(3), pitch=1.0, dims=dims, solid=np.ones(dims, bool))
    rho = np.ones(dims)
    ij = np.indices(dims, dtype=float) + 0.5  # cell centres
    cx, cy = dims[0] / 2.0, dims[1] / 2.0
    void = np.hypot(ij[0] - cx, ij[1] - cy) <= 2.2  # ~4.5-cell diameter bore

    mesh, mask, _w, _nc, _info = density_to_mesh(
        dom, rho.ravel(), void_cells=void, smooth_iterations=0
    )
    assert not mask[void].any()
    assert mesh.is_watertight

    rays = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    origins = np.array([[cx, cy, -2.0], [cx, cy, dims[2] + 2.0]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    locations, index_ray, _index_tri = rays.intersects_location(
        ray_origins=origins, ray_directions=directions
    )
    assert len(locations) == 0 and len(index_ray) == 0
