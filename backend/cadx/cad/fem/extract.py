"""Density field -> printable triangle mesh.

Threshold the SIMP density, keep only components connected to the passive
(mount/load) regions, regularize diagonal-touching cells so the surface is
manifold, then extract the surface. The default path runs marching cubes
on the Gaussian-smoothed density clipped to the mask with a
volume-preserving iso level (sub-voxel surface, mesh volume matches the
FEM mask); the legacy path thresholds first and runs marching cubes on
the binary mask. Optional Taubin smoothing removes any residual
staircase look; FEM verification should use the returned mask, not the
smoothed mesh.
"""

from __future__ import annotations

from typing import Any

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


def _void_corridor(
    labels: np.ndarray,
    main: int,
    target: int,
    walkable: np.ndarray,
) -> np.ndarray | None:
    """Shortest void-cell corridor between components `target` and `main`.

    Breadth-first search through `walkable` cells seeded at every walkable
    cell face-adjacent to `target`; stops at the first cell face-adjacent to
    `main`. Uniform edge cost, so the reconstructed path is a minimum-cell
    corridor — greedy distance-descent would get trapped where the straight
    line leaves the design domain (e.g. rounded envelope at the gap).
    Returns flat indices of the corridor cells, or None when unreachable.
    """
    from collections import deque

    face = ndimage.generate_binary_structure(3, 1)
    starts = np.flatnonzero(ndimage.binary_dilation(labels == target, structure=face) & walkable)
    if not starts.size:
        return None
    nx, ny, nz = labels.shape
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    visited = np.zeros(labels.size, dtype=bool)
    parent = np.full(labels.size, -1, dtype=np.int64)
    visited[starts] = True
    queue: deque[int] = deque(int(s) for s in starts)
    last = -1
    while queue and last < 0:
        cur = queue.popleft()
        ci, cj, ck = np.unravel_index(cur, labels.shape)
        for di, dj, dk in offsets:
            i, j, k = ci + di, cj + dj, ck + dk
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                continue
            if labels[i, j, k] == main:
                last = cur
                break
            if not walkable[i, j, k]:
                continue
            fi = int(np.ravel_multi_index((i, j, k), labels.shape))
            if visited[fi]:
                continue
            visited[fi] = True
            parent[fi] = cur
            queue.append(fi)
    if last < 0:
        return None
    path = []
    while last >= 0:
        path.append(last)
        last = int(parent[last])
    return np.asarray(path, dtype=np.int64)


def _bridge_components(
    mask: np.ndarray,
    walkable: np.ndarray,
    max_added_frac: float = 0.15,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Reconnect disjoint mask components with minimal void-cell corridors.

    Keep/load-region islands each carry `required` cells, so connectivity
    pruning keeps them all — yet a printed part must be a single body.
    Merge each island nearest-first into the largest component along the
    shortest void corridor, thickened by one cell so the bridge survives
    regularization. `walkable` bounds where new material may go (inside the
    design domain, outside carved voids). Stops early when a corridor would
    add more than `max_added_frac` of the existing volume — a flagged
    multi-body defect is more honest than a bridge thicker than the
    optimized structure.

    Returns (mask, cells_added, added_mask) — the last one marks exactly the
    cells the corridors added so callers can pin them through smoothing:
    they are the part's connectivity glue and must not erode below raster
    thickness.
    """
    face = ndimage.generate_binary_structure(3, 1)
    budget = max(1, round(float(mask.sum()) * max_added_frac))
    added = 0
    added_mask = np.zeros_like(mask)
    while added < budget:
        labels, n = ndimage.label(mask, structure=face)
        if n <= 1:
            break
        sizes = np.asarray(
            ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        )
        main = int(np.argmax(sizes)) + 1
        dist = ndimage.distance_transform_edt(labels != main)
        others = [k for k in range(1, n + 1) if k != main]
        # nearest island to the main body first — star topology keeps the
        # corridor count minimal and each merge shrinks the remaining gaps
        target = min(others, key=lambda k: float(dist[labels == k].min()))
        path = _void_corridor(labels, main, target, walkable & ~mask)
        if path is None or not path.size:
            break
        path_mask = np.zeros_like(mask)
        path_mask[np.unravel_index(path, mask.shape)] = True
        corridor = (
            ndimage.binary_dilation(path_mask, structure=face) & walkable & ~mask
        ) | path_mask
        n_new = int(corridor.sum())
        if added + n_new > budget:
            break
        mask = mask | corridor
        added_mask |= corridor
        added += n_new
    return mask, added, added_mask


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


def _surface_mesh_mc(domain: VoxelDomain, field: np.ndarray, iso: float = 0.5) -> trimesh.Trimesh:
    """Marching-cubes iso-surface at `iso` — sub-voxel smooth contours (no
    staircase aliasing on diagonal faces) compared to the boundary-quad
    fallback. `field` is a (nx,ny,nz) scalar array — a binary mask for the
    legacy path, a smoothed density for sub-voxel extraction. Vertices land
    in index space; map to mm via origin+pitch."""
    from skimage.measure import marching_cubes

    padded = np.pad(field.astype(np.float32), 1, mode="constant")
    verts, faces, _, _ = marching_cubes(padded, level=iso)
    if not len(verts) or not len(faces):
        raise ValueError("iso-surface is empty")
    # samples are cell-centers: padded index v ↔ original v-1 ↔ center at
    # origin + (v-1+0.5)*pitch — the iso then lands exactly on cell faces
    coords = domain.origin[None, :] + (verts - 0.5) * domain.pitch
    # a smooth iso field can push the surface a fraction of a voxel past the
    # design-space bbox (blur interpolates toward the zero padding) — the
    # domain bounds are the part's hard physical limit, clamp to them
    np.clip(
        coords,
        domain.origin[None, :],
        (domain.origin + domain.pitch * np.asarray(domain.dims))[None, :],
        out=coords,
    )
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
    void_cells: np.ndarray | None = None,
    min_member_mm: float = 0.0,
    min_gap_mm: float = 0.0,
    bridge_islands: bool = True,
) -> tuple[np.ndarray, list[str], int, np.ndarray]:
    """Threshold + clean the density field into a final solid mask (nx,ny,nz).

    Applies passive-region union, optional min-member thickening,
    connectivity pruning and non-manifold-edge regularization. This mask —
    not the smoothed export mesh — is the exact structure FEM verification
    should run on. `void_cells` ((nx,ny,nz) bool) marks cells that must stay
    empty — a carved region wins over thickening/regularization, which would
    otherwise grow material back into it.

    `bridge_islands` rejoins surviving disjoint keep/load islands with a
    minimal void-cell corridor — a structural part must be one body. Pass
    False for purely-aesthetic density fields (organic styling) where
    disjoint bodies are valid output.

    Returns (mask, warnings, n_components, bridged_mask): the last marks the
    cells corridor-bridging added — the connectivity glue that smoothing
    must not erode below raster thickness.
    """
    warnings: list[str] = []
    void3 = void_cells if void_cells is not None else None
    rho = density.reshape(domain.dims)
    mask = rho >= threshold
    if passive_solid is not None:
        mask |= passive_solid.reshape(domain.dims)
    if void3 is not None:
        mask &= ~void3
    if not mask.any():
        raise ValueError("density threshold produced an empty part")

    if min_member_mm > 0:
        mask, grown = _enforce_min_member(mask, domain.pitch, min_member_mm)
        if void3 is not None:
            healed = int((mask & void3).sum())
            if healed:
                mask &= ~void3
                warnings.append(f"carve regions suppressed {healed} min-member regrowth voxels")
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
    # kept components can still be disjoint: a keep/load region forming its
    # own island touches `required` trivially, so pruning cannot see it. A
    # severed island is a functional feature, not debris — rejoin survivors
    # with a minimal corridor before regularization (which only adds
    # material, so it cannot break the bridge again)
    structure = ndimage.generate_binary_structure(3, 1)
    bridged_mask = np.zeros_like(mask)
    if bridge_islands and int(ndimage.label(mask, structure=structure)[1]) > 1:
        walkable = domain.solid & ~mask
        if void3 is not None:
            walkable &= ~void3
        mask, bridged, bridged_mask = _bridge_components(mask, walkable)
        if bridged:
            warnings.append(
                f"reconnected disjoint keep/load islands with {bridged} bridge "
                "voxels — a severed anchor region was rejoined to the body"
            )
    mask = _regularize_mask(mask)
    if void3 is not None:
        # diagonal-edge regularization can fill cells inside a carved void —
        # removing cells cannot create new non-manifold edges, so re-carving
        # afterwards is safe
        mask &= ~void3
    # islands the bridge pass could not rejoin inside its volume budget stay
    # disjoint — report so the caller/agent can adjust regions or volfrac
    _labels, n_components = ndimage.label(mask, structure=structure)
    if n_components > 1:
        warnings.append(
            f"{n_components} disconnected bodies — a keep/load region is not "
            f"connected to the main structure (enlarge keep regions or raise volfrac)"
        )
    return mask, warnings, n_components, bridged_mask


def _volume_preserved_iso(
    field: np.ndarray, target: int, max_iter: int = 40
) -> tuple[float, str | None]:
    """Bisect the iso level `t` in (0, max(field)] so that the number of
    cells with field >= t equals `target` — i.e. the extracted iso-surface
    encloses the same volume as the FEM-verified binary mask.

    count(field >= t) is monotone non-increasing in t, so bisection on
    (0, fmax] is exact up to ties. Returns (iso, warning): warning is set
    when no `t` hits `target` within `max_iter` halvings — e.g. cells with
    field == 0 inside the mask (min-member/passive cells the smoother
    cannot reach). The fallback is then the rank iso — the `target`-th
    largest value keeps at least `target` cells (ties can overshoot, but
    never the arbitrary 0.5*max cut that produced -36% volume errors).
    """
    vals = field.ravel()
    if not vals.size:
        raise ValueError("cannot preserve volume on an empty density field")
    fmax = float(vals.max())
    if target <= 0:
        return float(np.nextafter(fmax, np.inf)), None
    if target > len(vals):
        iso = float(vals.min())
        return iso, None if int((vals >= iso).sum()) == target else (
            f"volume-preserving iso target {target} exceeds field size {len(vals)}"
        )
    if fmax <= 0.0:
        return 0.0, (
            f"volume-preserving iso impossible (max field {fmax:g}, "
            f"target {target} voxels) — zero field contains {len(vals)} tied cells"
        )
    lo, hi = 0.0, fmax
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:  # float resolution exhausted
            break
        n = int((field >= mid).sum())
        if n == target:
            return mid, None
        if n > target:
            lo = mid
        else:
            hi = mid
    rank = min(max(len(vals) - target, 0), len(vals) - 1)
    iso = float(np.partition(vals, rank)[rank])
    n_iso = int((field >= iso).sum())
    above_iso = float(np.nextafter(iso, np.inf))
    n_above = int((field >= above_iso).sum())
    if abs(n_above - target) < abs(n_iso - target):
        iso, n_iso = above_iso, n_above
    warn = None
    if n_iso != target:
        warn = (
            f"volume-preserving iso fell back to the rank threshold "
            f"(mask {target} voxels, got {n_iso} — ties/zero cells make the "
            "exact count unattainable)"
        )
    return iso, warn


def _volume_bisect_mesh(
    domain: VoxelDomain,
    field: np.ndarray,
    mask: np.ndarray,
    iso0: float,
    target_vol: float,
    max_evals: int = 6,
) -> tuple[trimesh.Trimesh, float, str | None]:
    """Refine the iso level by bisecting on the enclosed *mesh* volume.

    The cell-count bisect only guarantees `count(field >= t) == mask.sum()`,
    but the iso surface shrinks inside thin members — measured volume runs
    well under the verified mask (observed ~35% on thin plates). The
    enclosed volume is monotone non-increasing in `iso`, so bisection
    converges; verified mask cells are floored just above the running iso
    so they stay enclosed at any level. Returns (mesh, iso, warning) —
    the mesh is already extracted at the final level; warning is set when
    the residual error stays above 5% of the target.
    """
    lo, hi = 0.0, float(field.max())
    best_mesh: trimesh.Trimesh | None = None
    best_iso = iso0
    best_err = float("inf")
    iso = iso0
    for _ in range(max_evals):
        # mask cells must stay enclosed at any iso — bridge corridors and
        # min-member regrowth carry rho≈0 in the smoothed field, so floor
        # them just above the running level; the surface still interpolates
        # sub-voxel in the 1-cell shell
        floor = iso * 1.05 + 1e-9
        f2 = np.where(mask & (field < floor), floor, field)
        try:
            m = _surface_mesh_mc(domain, f2, iso)
            vol = abs(float(m.volume))
        except Exception:
            m, vol = None, 0.0  # iso above the surface — drive it down
        err = vol - target_vol
        if abs(err) < best_err:
            best_err, best_mesh, best_iso = abs(err), m, iso
        if abs(err) <= 0.02 * max(target_vol, 1e-9):
            break
        if vol < target_vol:
            hi = min(hi, iso)
            iso = 0.5 * (lo + iso)
        else:
            lo = max(lo, iso)
            iso = 0.5 * (iso + hi)
        if hi - lo <= 1e-6:
            break
    if best_mesh is None:
        raise ValueError("iso-surface is empty at every bisected level")
    warn = None
    if best_err > 0.05 * max(target_vol, 1e-9):
        warn = (
            f"volume-preserving iso bisect left {best_err:.0f} mm^3 residual "
            f"(target {target_vol:.0f} mm^3) — shell headroom or field shape "
            "limits the attainable volume"
        )
    return best_mesh, best_iso, warn


def density_to_mesh(
    domain: VoxelDomain,
    density: np.ndarray,
    *,
    threshold: float = 0.5,
    passive_solid: np.ndarray | None = None,
    void_cells: np.ndarray | None = None,
    min_member_mm: float = 0.0,
    min_gap_mm: float = 0.0,
    smooth_iterations: int = 3,
    extract_smooth: bool = True,
    sigma_px: float = 0.8,
    volume_preserve: bool = True,
    vol_target_scale: float = 1.0,
    bridge_islands: bool = True,
) -> tuple[trimesh.Trimesh, np.ndarray, list[str], int, dict[str, Any]]:
    """Convert the optimized density field into a trimesh.

    Returns (mesh, final_mask, warnings, n_components, extract_info). The
    mesh is in mm in the domain frame; `final_mask` (nx,ny,nz) is the exact
    voxel structure — use it (via run_fem_domain) for verification instead
    of re-voxelizing the smoothed mesh, which would erode thin members.

    With `extract_smooth` (default) marching cubes runs on the
    Gaussian-smoothed density (`sigma_px` voxels) clipped to the
    one-cell-dilated mask —
    a sub-voxel iso-surface instead of the staircase boundary of the
    binary field. `volume_preserve` bisects the iso level so the enclosed
    mesh volume matches the mask volume (cell-count seed + volume
    refinement); `vol_target_scale` >1 compensates known downstream
    shrinkage (smoothing, envelope clip) by aiming above the mask volume.
    Without it the fixed `threshold` level is used. `extract_smooth=False`
    reproduces the legacy path bit-identically (binary mask -> MC at 0.5
    -> Taubin).

    `extract_info` carries extraction telemetry: `iso` (level used),
    `vol_error_pct` (mesh volume vs mask volume), `gray_fraction`
    (share of solid cells with 0.1 < rho < 0.9 — design sharpness),
    `smooth` and `sigma_px`.
    """
    mask, warnings, n_components, bridged_mask = density_to_mask(
        domain,
        density,
        threshold=threshold,
        passive_solid=passive_solid,
        void_cells=void_cells,
        min_member_mm=min_member_mm,
        min_gap_mm=min_gap_mm,
        bridge_islands=bridge_islands,
    )
    rho3 = density.reshape(domain.dims)
    solid_rho = rho3[mask]
    gray_fraction = float(((solid_rho > 0.1) & (solid_rho < 0.9)).mean())
    iso = float(threshold)
    try:
        if extract_smooth:
            # hard boundary conditions BEFORE the Gaussian: carved voids read
            # as rho=0 (the blur must pull the iso surface away from them, not
            # smear their rho=1 neighbourhood across the hole) and pinned
            # passive cells read as rho=1 so the shell keeps them enclosed.
            # Bridge-corridor cells get the same treatment: they are the
            # connectivity glue between pinned islands — a blurred rho<1
            # there pinches the iso surface off and the "connected" mask
            # exports as sub-voxel-separated bodies.
            rho_in = rho3.astype(np.float64).copy()
            if void_cells is not None:
                rho_in[void_cells] = 0.0
            if passive_solid is not None:
                rho_in[passive_solid.reshape(domain.dims)] = 1.0
            rho_in[bridged_mask] = 1.0
            rho_s = ndimage.gaussian_filter(rho_in, float(sigma_px))
            # clip the iso field to the mask, dilated by one cell: a hard
            # clip at the mask boundary would put a 0-cliff exactly where
            # the iso must interpolate, biasing the volume low by ~half a
            # boundary layer; the 1-cell shell keeps the interpolation
            # smooth while still zeroing stray material further out. The
            # shell must NOT grow back into carved voids — a blurred
            # rho_s>iso ring inside the hole would close it again
            shell = ndimage.binary_dilation(mask)
            if void_cells is not None:
                shell &= ~void_cells
            field = np.where(shell, rho_s, 0.0)
            fmax = float(rho_s.max())
            if volume_preserve:
                iso, iso_warn = _volume_preserved_iso(field, int(mask.sum()))
                if iso_warn:
                    warnings.append(iso_warn)
                    log.warning(
                        "topopt.iso_bisect_fallback",
                        target=int(mask.sum()),
                        field_max=fmax,
                    )
                # the cell-count iso only bounds count(field>=t); the iso
                # surface itself can still shrink inside thin members
                # (~35% underfill measured) — refine on enclosed volume
                target_vol = (
                    float(mask.sum()) * domain.pitch**3 * float(np.clip(vol_target_scale, 0.5, 2.0))
                )
                try:
                    mesh, iso, vb_warn = _volume_bisect_mesh(domain, field, mask, iso, target_vol)
                    if vb_warn:
                        warnings.append(vb_warn)
                except Exception:
                    log.warning("topopt.volume_bisect_fallback", exc_info=True)
                    floor = iso * 1.05 + 1e-9
                    field = np.where(mask & (field < floor), floor, field)
                    mesh = _surface_mesh_mc(domain, field, iso)
            else:
                if not (0.0 < iso < fmax):
                    iso = 0.5 * fmax
                    warnings.append(
                        f"iso threshold {threshold:g} outside the smoothed field "
                        f"range — using t={iso:g}"
                    )
                # mask-verified cells must stay inside the iso surface —
                # bridge corridors and min-member regrowth carry rho≈0 in
                # the smoothed field, so floor them just above the chosen
                # level; the surface still interpolates sub-voxel in the
                # 1-cell shell
                floor = iso * 1.05 + 1e-9
                field = np.where(mask & (field < floor), floor, field)
                mesh = _surface_mesh_mc(domain, field, iso)
        else:
            iso = 0.5  # binary mask -> fixed level
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
    mask_vol = float(mask.sum()) * domain.pitch**3
    extract_info: dict[str, Any] = {
        "iso": float(iso),
        "smooth": bool(extract_smooth),
        "sigma_px": float(sigma_px),
        "vol_error_pct": round(100.0 * (float(mesh.volume) - mask_vol) / mask_vol, 3),
        "gray_fraction": round(gray_fraction, 4),
        # cells the island-bridge added — callers pin them through the
        # export smoothing so the verified connectivity survives to print
        "bridged_mask": bridged_mask,
    }
    return mesh, mask, warnings, n_components, extract_info
