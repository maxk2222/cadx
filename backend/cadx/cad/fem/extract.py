"""Density field -> printable triangle mesh.

Threshold the SIMP density, keep only components connected to the passive
(mount/load) regions, regularize diagonal-touching cells so the surface is
manifold, then emit the exact voxel boundary as quads (watertight by
construction). Optional Taubin smoothing removes the staircase look; FEM
verification should use the returned mask, not the smoothed mesh.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

from cadx.cad.fem.voxelize import VoxelDomain, shift_mask
from cadx.logging import get_logger

log = get_logger(__name__)


def _connectivity_mask(mask: np.ndarray, required: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep components touching `required`; largest component as fallback."""
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    labels, n = ndimage.label(mask, structure=structure)
    if n <= 1:
        return mask, 0
    required_labels = np.unique(labels[required & mask])
    required_labels = required_labels[required_labels > 0]
    if len(required_labels):
        keep = np.isin(labels, required_labels)
        dropped = int((mask & ~keep).sum())
        return keep, dropped
    # nothing touches the required regions -> keep the largest component
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    return labels == biggest, int(mask.sum() - sizes[biggest - 1])


def _regularize_mask(mask: np.ndarray, max_passes: int = 6) -> np.ndarray:
    """Fill cells so no grid edge is touched by two diagonally-opposed
    occupied cells — such checkerboard edges yield a non-manifold surface.

    For each axis the 2x2 cell windows in the perpendicular plane are
    checked; a diagonal pair gets one filler cell (w01), connecting both.
    """
    m = mask.copy()
    lo, hi = slice(None, -1), slice(1, None)
    for _ in range(max_passes):
        added = 0
        for ax in range(3):
            o1, o2 = [a for a in range(3) if a != ax]
            w: dict[tuple[int, int], np.ndarray] = {}
            for a in range(2):
                for b in range(2):
                    s = [slice(None)] * 3
                    s[o1] = lo if a == 0 else hi
                    s[o2] = lo if b == 0 else hi
                    w[a, b] = m[tuple(s)]
            w00, w01, w10, w11 = w[0, 0], w[0, 1], w[1, 0], w[1, 1]
            # diag1: occupied w00+w11 -> fill empty w01
            # diag2: occupied w01+w10 -> fill empty w00
            d1 = w00 & w11 & ~w01 & ~w10
            d2 = w01 & w10 & ~w00 & ~w11
            if d1.any():
                s = [slice(None)] * 3
                s[o1], s[o2] = lo, hi
                m[tuple(s)] |= d1
                added += int(d1.sum())
            if d2.any():
                s = [slice(None)] * 3
                s[o1], s[o2] = lo, lo
                m[tuple(s)] |= d2
                added += int(d2.sum())
        if added == 0:
            break
    return m


# quad corner offsets per outward direction, CCW seen from outside
# (dx,dy,dz offsets of the 4 corners of the exposed cell face)
_FACES: list[tuple[np.ndarray, list[tuple[int, int, int]]]] = [
    (np.array([1, 0, 0]), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),  # +x
    (np.array([-1, 0, 0]), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),  # -x
    (np.array([0, 1, 0]), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),  # +y
    (np.array([0, -1, 0]), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),  # -y
    (np.array([0, 0, 1]), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),  # +z
    (np.array([0, 0, -1]), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),  # -z
]


def _surface_mesh(domain: VoxelDomain, mask: np.ndarray) -> trimesh.Trimesh:
    """Exact voxel-boundary surface: one quad per exposed cell face.

    Produces a watertight manifold mesh by construction (vertices are shared
    grid nodes). The staircase look is removed by the smoothing pass.
    """
    nx, ny, _nz = domain.dims
    nx1, ny1 = nx + 1, ny + 1

    def nidx(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return np.asarray(i + nx1 * (j + ny1 * k))

    quads: list[np.ndarray] = []
    for direction, corners in _FACES:
        shifted = shift_mask(mask, direction)
        exposed = mask & ~shifted
        i, j, k = np.nonzero(exposed)
        quad_idx = np.stack(
            [
                nidx(i + corners[0][0], j + corners[0][1], k + corners[0][2]),
                nidx(i + corners[1][0], j + corners[1][1], k + corners[1][2]),
                nidx(i + corners[2][0], j + corners[2][1], k + corners[2][2]),
                nidx(i + corners[3][0], j + corners[3][1], k + corners[3][2]),
            ],
            axis=1,
        )
        quads.append(quad_idx)
    if not quads:
        raise ValueError("iso-surface is empty")
    q = np.concatenate(quads)
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])

    used = np.unique(tris)
    remap = np.full(tris.max() + 1, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    coords = domain.node_coords()[used]
    return trimesh.Trimesh(vertices=coords, faces=remap[tris], process=False)


def density_to_mask(
    domain: VoxelDomain,
    density: np.ndarray,
    *,
    threshold: float = 0.5,
    passive_solid: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Threshold + clean the density field into a final solid mask (nx,ny,nz).

    Applies passive-region union, connectivity pruning and non-manifold-edge
    regularization. This mask — not the smoothed export mesh — is the exact
    structure FEM verification should run on.
    """
    warnings: list[str] = []
    rho = density.reshape(domain.dims)
    mask = rho >= threshold
    if passive_solid is not None:
        mask |= passive_solid.reshape(domain.dims)
    if not mask.any():
        raise ValueError("density threshold produced an empty part")

    required = (
        passive_solid.reshape(domain.dims)
        if passive_solid is not None
        else np.zeros(domain.dims, dtype=bool)
    )
    mask, dropped = _connectivity_mask(mask, required)
    if dropped:
        warnings.append(
            f"dropped {dropped} disconnected voxels "
            f"({100 * dropped / max(int(mask.sum()) + dropped, 1):.0f}% of material)"
        )
    mask = _regularize_mask(mask)
    return mask, warnings


def density_to_mesh(
    domain: VoxelDomain,
    density: np.ndarray,
    *,
    threshold: float = 0.5,
    passive_solid: np.ndarray | None = None,
    smooth_iterations: int = 3,
) -> tuple[trimesh.Trimesh, np.ndarray, list[str]]:
    """Convert the optimized density field into a trimesh.

    Returns (mesh, final_mask, warnings). The mesh is in mm in the domain
    frame; `final_mask` (nx,ny,nz) is the exact voxel structure — use it
    (via run_fem_domain) for verification instead of re-voxelizing the
    smoothed mesh, which would erode thin members.
    """
    mask, warnings = density_to_mask(
        domain, density, threshold=threshold, passive_solid=passive_solid
    )
    mesh = _surface_mesh(domain, mask)
    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iterations)
        except Exception:
            log.warning("topopt.smooth_failed", exc_info=True)
    return mesh, mask, warnings
