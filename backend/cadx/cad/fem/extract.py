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


def _surface_mesh_mc(domain: VoxelDomain, mask: np.ndarray) -> trimesh.Trimesh:
    """Marching-cubes iso-surface at 0.5 — sub-voxel smooth contours (no
    staircase aliasing on diagonal faces) compared to the boundary-quad
    fallback. Vertices land in index space; map to mm via origin+pitch."""
    from skimage.measure import marching_cubes

    padded = np.pad(mask.astype(np.float32), 1, mode="constant")
    verts, faces, _, _ = marching_cubes(padded, level=0.5)
    if not len(verts) or not len(faces):
        raise ValueError("iso-surface is empty")
    # samples are cell-centers: padded index v ↔ original v-1 ↔ center at
    # origin + (v-1+0.5)*pitch — the iso then lands exactly on cell faces
    coords = domain.origin[None, :] + (verts - 0.5) * domain.pitch
    # marching_cubes emits inward-facing windings — flip for +volume/normals
    mesh = trimesh.Trimesh(vertices=coords, faces=faces[:, ::-1], process=False)
    mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:  # belt & suspenders: never export inverted solids
        mesh.invert()
    return mesh


def _enforce_min_member(
    mask: np.ndarray, pitch: float, min_mm: float, max_passes: int = 0
) -> tuple[np.ndarray, int]:
    """Thicken members thinner than `min_mm` (local thickness via EDT).

    Cells whose local thickness 2*edt falls below min_mm are dilated by one
    voxel per pass until the deficit is covered. Dilation can only add
    material, so connectivity is preserved; callers should re-regularize.
    Returns (mask, cells_added)."""
    if min_mm <= 0:
        return mask, 0
    passes = max_passes or max(1, int(np.ceil(min_mm / (2 * pitch))))
    m = mask.copy()
    added_total = 0
    for _ in range(passes):
        edt = ndimage.distance_transform_edt(m) * pitch  # mm to the void
        thin = m & (2 * edt < min_mm - 1e-9)
        if not thin.any():
            break
        grown = ndimage.binary_dilation(thin)
        added = grown & ~m
        if not added.any():
            break
        m |= grown
        added_total += int(added.sum())
    return m, added_total


def _min_gap_warning(mask: np.ndarray, pitch: float, min_gap_mm: float) -> str | None:
    """Detect enclosed void channels narrower than `min_gap_mm` — they fuse
    shut in FDM/SLS and trap powder. Closing fills gaps < ~2r; a filled void
    cell that is NOT connected to the bounding-box border is a real enclosed
    thin gap. Returns a warning string or None."""
    if min_gap_mm <= 0:
        return None
    r = max(1, round(min_gap_mm / (2.0 * pitch)))
    structure = ndimage.iterate_structure(ndimage.generate_binary_structure(3, 1), r)
    closed = ndimage.binary_closing(mask, structure=structure)
    bridged = closed & ~mask
    if not bridged.any():
        return None
    # void components touching the bbox border are open air — they don't fuse
    void = ~mask
    labels, _ = ndimage.label(void, structure=ndimage.generate_binary_structure(3, 1))
    border = np.concatenate(
        [
            labels[0, :, :].ravel(),
            labels[-1, :, :].ravel(),
            labels[:, 0, :].ravel(),
            labels[:, -1, :].ravel(),
            labels[:, :, 0].ravel(),
            labels[:, :, -1].ravel(),
        ]
    )
    outside_ids = set(np.unique(border).tolist()) - {0}
    enclosed = bridged & ~np.isin(labels, list(outside_ids))
    n = int(enclosed.sum())
    if not n:
        return None
    return (
        f"{n} enclosed voxels sit in gaps below ~{min_gap_mm:g} mm — "
        "they will fuse when printed (raise min_member_mm or clear the gap)"
    )


def density_to_mask(
    domain: VoxelDomain,
    density: np.ndarray,
    *,
    threshold: float = 0.5,
    passive_solid: np.ndarray | None = None,
    min_member_mm: float = 0.0,
    min_gap_mm: float = 0.0,
) -> tuple[np.ndarray, list[str], int]:
    """Threshold + clean the density field into a final solid mask (nx,ny,nz).

    Applies passive-region union, optional min-member thickening,
    connectivity pruning and non-manifold-edge regularization. This mask —
    not the smoothed export mesh — is the exact structure FEM verification
    should run on.
    """
    warnings: list[str] = []
    rho = density.reshape(domain.dims)
    mask = rho >= threshold
    if passive_solid is not None:
        mask |= passive_solid.reshape(domain.dims)
    if not mask.any():
        raise ValueError("density threshold produced an empty part")

    if min_member_mm > 0:
        mask, grown = _enforce_min_member(mask, domain.pitch, min_member_mm)
        if grown:
            warnings.append(f"thickened {grown} voxels to reach min member {min_member_mm:g} mm")
    gap_warn = _min_gap_warning(mask, domain.pitch, min_gap_mm)
    if gap_warn:
        warnings.append(gap_warn)

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
    # kept components can still be disjoint: a keep/load region forming its
    # own island touches `required` trivially, so pruning cannot see it —
    # report it so the caller/agent can adjust regions or volfrac
    structure = ndimage.generate_binary_structure(3, 1)
    _labels, n_components = ndimage.label(mask, structure=structure)
    if n_components > 1:
        warnings.append(
            f"{n_components} disconnected bodies — a keep/load region is not "
            f"connected to the main structure (enlarge keep regions or raise volfrac)"
        )
    return mask, warnings, n_components


def density_to_mesh(
    domain: VoxelDomain,
    density: np.ndarray,
    *,
    threshold: float = 0.5,
    passive_solid: np.ndarray | None = None,
    min_member_mm: float = 0.0,
    min_gap_mm: float = 0.0,
    smooth_iterations: int = 3,
) -> tuple[trimesh.Trimesh, np.ndarray, list[str], int]:
    """Convert the optimized density field into a trimesh.

    Returns (mesh, final_mask, warnings, n_components). The mesh is in mm in
    the domain
    frame; `final_mask` (nx,ny,nz) is the exact voxel structure — use it
    (via run_fem_domain) for verification instead of re-voxelizing the
    smoothed mesh, which would erode thin members.
    """
    mask, warnings, n_components = density_to_mask(
        domain,
        density,
        threshold=threshold,
        passive_solid=passive_solid,
        min_member_mm=min_member_mm,
        min_gap_mm=min_gap_mm,
    )
    try:
        mesh = _surface_mesh_mc(domain, mask)
    except Exception:
        log.warning("topopt.marching_cubes_fallback", exc_info=True)
        mesh = _surface_mesh(domain, mask)
        # staircase edges need one extra smoothing pass to reach the same
        # finish as the marching-cubes surface
        smooth_iterations += 1
    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iterations)
        except Exception:
            log.warning("topopt.smooth_failed", exc_info=True)
    return mesh, mask, warnings, n_components
