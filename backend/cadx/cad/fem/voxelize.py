"""Triangle mesh -> uniform voxel domain for the Hex8 grid.

The part's bounding box becomes the FEM/TopOpt design domain: cell (i,j,k)
is a cube of edge `pitch` mm; `solid[i,j,k]` marks cells inside the part.
Nodes are the cell corners — node (i,j,k) sits at
`origin + pitch * (i, j, k)` with i in [0..nx].
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import ndimage

# Corner ids of each cell face in element_nodes() Hex8 order
# (-,-,-) (+,-,-) (+,+,-) (-,+,-) / (-,-,+) (+,-,+) (+,+,+) (-,+,+):
# FACE_CORNERS[axis][side] — side 0 is the -axis face, 1 the +axis face.
# NOTE: the corner index does NOT decompose axis-aligned by bit — bit tests
# like `(c >> axis) & 1` pick the parity-diagonal set for x; use this table.
FACE_CORNERS: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((0, 3, 4, 7), (1, 2, 5, 6)),  # x: -face, +face
    ((0, 1, 4, 5), (2, 3, 6, 7)),  # y: -face, +face
    ((0, 1, 2, 3), (4, 5, 6, 7)),  # z: -face, +face
)


@dataclass
class VoxelDomain:
    origin: np.ndarray  # (3,) world-mm min corner of cell (0,0,0)
    pitch: float  # mm edge length
    dims: tuple[int, int, int]  # (nx, ny, nz) cells
    solid: np.ndarray  # bool (nx, ny, nz) — inside the part
    pitch_requested: float = 0.0  # before adaptive coarsening
    # number of void padding cells the grid was grown with on every side
    # (design-domain dilation for grow_mm / filter-boundary headroom) —
    # `solid` stays the part mask; the pad ring is False by construction
    pad_cells: int = 0
    # per-cell fill fraction in [0,1], flat (ncells,) in (i,j,k) ravel order
    # (same layout as element_centers/element_nodes rows) — the "thin soft
    # transition" boundary model: interior cells read 1.0, exterior 0.0 and
    # surface-crossing cells a subsampled partial fraction. `None` unless
    # voxelize(coverage=True) was requested AND the mesh admitted reliable
    # containment; consumers must apply it only on `solid` cells.
    coverage: np.ndarray | None = None
    # memo table for the pure (dims/pitch/origin-derived) geometry tables —
    # element_nodes/element_centers are recomputed per region per load case
    # otherwise (region resolution calls them constantly on big grids)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    @property
    def node_dims(self) -> tuple[int, int, int]:
        return (self.dims[0] + 1, self.dims[1] + 1, self.dims[2] + 1)

    @property
    def cell_count(self) -> int:
        return self.dims[0] * self.dims[1] * self.dims[2]

    @property
    def solid_count(self) -> int:
        return int(self.solid.sum())

    @property
    def node_count(self) -> int:
        d = self.node_dims
        return d[0] * d[1] * d[2]

    def node_index(self, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Flat node index for grid coords (i,j,k), matching node_coords order."""
        nx1, ny1, _ = self.node_dims
        return np.asarray(i + nx1 * (j + ny1 * k))

    def node_coords(self) -> np.ndarray:
        """(nnodes, 3) world coords, ordered like node_index (i fastest)."""
        nx1, ny1, _ = self.node_dims
        idx = np.arange(self.node_count)
        i = idx % nx1
        j = (idx // nx1) % ny1
        k = idx // (nx1 * ny1)
        grid = np.stack([i, j, k], axis=-1)
        return self.origin[None, :] + self.pitch * grid.astype(float)

    def cached(self, key: str, build: Callable[[], np.ndarray]) -> np.ndarray:
        """Memoize a pure function of (dims, pitch, origin) on this domain —
        the cached arrays must be treated read-only by callers."""
        out = self._cache.get(key)
        if out is None:
            out = build()
            self._cache[key] = out
        return out

    def element_nodes(self) -> np.ndarray:
        """(ncells, 8) flat node indices of every cell, Hex8 order:
        (-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+)."""

        def _build() -> np.ndarray:
            nx, ny, nz = self.dims
            ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
            i, j, k = ii.ravel(), jj.ravel(), kk.ravel()
            return np.stack(
                [
                    self.node_index(i, j, k),
                    self.node_index(i + 1, j, k),
                    self.node_index(i + 1, j + 1, k),
                    self.node_index(i, j + 1, k),
                    self.node_index(i, j, k + 1),
                    self.node_index(i + 1, j, k + 1),
                    self.node_index(i + 1, j + 1, k + 1),
                    self.node_index(i, j + 1, k + 1),
                ],
                axis=1,
            )

        return self.cached("element_nodes", _build)

    def element_centers(self) -> np.ndarray:
        """(ncells, 3) world coords of cell centers, flat (i,j,k) order."""

        def _build() -> np.ndarray:
            nx, ny, nz = self.dims
            ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
            idx = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
            return self.origin[None, :] + self.pitch * (idx.astype(float) + 0.5)

        return self.cached("element_centers", _build)


def choose_pitch(
    volume_mm3: float, target_mm: float | None, max_elements: int
) -> tuple[float, int]:
    """Pitch so estimated solid elements stay under max_elements."""
    pitch = max(target_mm or 1.5, 0.1)
    est = int(volume_mm3 / pitch**3) if volume_mm3 > 0 else 0
    if est > max_elements:
        pitch = (volume_mm3 / max_elements) ** (1.0 / 3.0)
    return pitch, est


def snapped_pitch(
    spans: np.ndarray | list[float],
    pitch_target: float,
    *,
    min_cells: int = 1,
    mesh_volume: float = 0.0,
    max_elements: int = 0,
) -> tuple[float, str | None]:
    """Snap the requested pitch so the thinnest span is an integer cell
    count — the rasterized domain then models the same thickness the part
    actually has (and the spec-envelope clip enforces on the export).
    Without it a 4 mm plate at a 2.5 mm pitch voxelizes to 2 cells = a
    5 mm design domain: the mask FEM verifies a part 25% thicker than the
    one that gets printed.

    `spans` is the effective per-axis extent (spec dim where stated, mesh
    bbox otherwise). The snap only applies when it reduces the total
    domain overhang vs the unsnapped pitch, and `min_cells` sets the floor
    for cells through the thinnest axis (thin plates need >=2-3 layers for
    meaningful bending resolution). Returns (pitch, warning): warning is
    set when the snap had to be refused (element budget would break), in
    which case the caller keeps `pitch_target` and the domain/spec
    mismatch is surfaced instead of silently accepted.
    """
    p0 = float(max(pitch_target, 1e-6))
    spans_a = np.asarray(spans, dtype=float)
    if spans_a.shape != (3,) or not np.isfinite(spans_a).all():
        return p0, None
    span_min = float(spans_a.min())
    if span_min <= 0.0:
        return p0, None
    n = max(round(span_min / p0), int(min_cells), 1)
    p_snap = span_min / n

    def _overhang(p: float) -> float:
        # total domain extent beyond the effective spans, all axes
        return float(np.sum(np.ceil(np.maximum(spans_a / p - 1e-9, 0.0)) * p - spans_a))

    if _overhang(p_snap) >= _overhang(p0) - 1e-9:
        return p0, None
    if max_elements > 0 and mesh_volume > 0 and mesh_volume / p_snap**3 > max_elements:
        return p0, (
            f"spec-fit pitch {p_snap:.3f} mm ({n} cells through "
            f"{span_min:g} mm) would exceed the element budget "
            f"({int(mesh_volume / p_snap**3)} > {max_elements} est. solid "
            f"cells) — keeping {p0:g} mm; the FEM domain thickness may "
            "differ from the spec envelope"
        )
    return float(p_snap), None


def _scanline_contains(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    n_pts: tuple[int, int, int],
    pitch: float,
    *,
    offset: float,
) -> np.ndarray | None:
    """(prod(n_pts),) containment for a regular point lattice via axis
    scanlines — one ray per lattice line along the LONGEST axis, hit
    positions sorted, inside = odd crossing count.

    ``contains()`` issues one ray per *point* against the whole mesh; a
    scanline needs only dims[other1]x dims[other2] rays total and reuses
    each hit list for every point on that line — ~dims-fold fewer
    ray/triangle tests on the same geometry (the R-tree containment path
    dominates voxelize on dense organic meshes). Points sit at
    ``origin + pitch*(index + offset)`` — 0.5 for cell centres, 0.0 for
    the node grid.

    Returns ``None`` when ray queries are unavailable — callers fall back
    to chunked ``contains()``. ``CADX_VOXEL_CONTAINS=1`` forces that
    fallback (escape hatch if a pathological mesh defeats the scanline).

    Boundary conventions (matched to contains()'s observed behavior so the
    coverage model sees the same corner masks): the ray's perpendicular
    coords are jittered by eps so it never runs *coplanar* with an
    axis-aligned face, and a query point sitting on a crossing position
    counts as outside."""
    import os

    if os.environ.get("CADX_VOXEL_CONTAINS"):
        return None
    try:
        ax = int(np.argmax(n_pts))
        oa, ob = (a for a in range(3) if a != ax)
        na, nb = n_pts[oa], n_pts[ob]
        eps = pitch * 1e-3  # same-surface collapse + face-plane jitter
        ja, jb = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
        origins = np.zeros((na * nb, 3))
        origins[:, oa] = origin[oa] + pitch * (ja.ravel() + offset) + eps
        origins[:, ob] = origin[ob] + pitch * (jb.ravel() + offset) + eps
        # start BEFORE the mesh bbox — the grid origin can sit inside the
        # part (padded/lo grids); a ray born inside only sees the exit
        # crossing and every point reads 'outside'
        origins[:, ax] = float(mesh.bounds[0, ax]) - 2.0 * pitch
        directions = np.zeros_like(origins)
        directions[:, ax] = 1.0
        q_axis = origin[ax] + pitch * (np.arange(n_pts[ax]) + offset)

        # same ~256 MB batch bound as the contains() path — rays allocate
        # per-candidate buffers against the face count too
        chunk = int(np.clip((256 * 1024 * 1024) // (64 * max(len(mesh.faces), 1)), 256, 8192))
        inside = np.zeros((len(origins), n_pts[ax]), dtype=bool)
        for s0 in range(0, len(origins), chunk):
            locs, ray_idx, _tri = mesh.ray.intersects_location(
                origins[s0 : s0 + chunk],
                directions[s0 : s0 + chunk],
                multiple_hits=True,
            )
            if not len(locs):
                continue
            order = np.lexsort((locs[:, ax], ray_idx))
            r_sorted = ray_idx[order]
            x_sorted = locs[order, ax]
            # collapse duplicate hits at (numerically) the same position —
            # an edge/vertex shared by triangles must toggle parity once
            same_ray = np.diff(r_sorted) == 0
            dup = np.concatenate([[False], same_ray & (np.diff(x_sorted) < eps)])
            r_u, x_u = r_sorted[~dup], x_sorted[~dup]
            # per-ray even/odd fill — ray_idx is local to this chunk's batch
            n_chunk = min(chunk, len(origins) - s0)
            for r_local in range(n_chunk):
                hits = x_u[
                    np.searchsorted(r_u, r_local - 0.5) : np.searchsorted(r_u, r_local + 0.5)
                ]
                if len(hits):
                    # hits strictly below the query position — a query
                    # sitting ON a crossing is boundary → outside (the
                    # convention contains() shows on surface points, and
                    # the coverage corner-count depends on it)
                    pos = np.searchsorted(hits, q_axis, side="left")
                    on_hit = pos < len(hits)
                    on_hit[on_hit] = hits[pos[on_hit]] - q_axis[on_hit] < eps
                    inside[s0 + r_local] = (pos % 2 == 1) & ~on_hit
        # reshape back to (n0, n1, n2) — lines ran over axes (oa, ob)
        lines = inside.reshape(na, nb, n_pts[ax])
        grid = np.moveaxis(lines, (0, 1, 2), (oa, ob, ax))
        return grid.ravel()
    except Exception:
        return None


def _fill_grid(
    mesh: trimesh.Trimesh, origin: np.ndarray, dims: tuple[int, int, int], pitch: float
) -> np.ndarray:
    """Solid mask over an exact-bounds grid.

    Watertight meshes use cell-center containment (no systematic growth) —
    axis scanlines first, per-point contains() as the fallback.
    Non-watertight meshes fall back to surface voxelization + hole filling.
    """
    ii, jj, kk = np.meshgrid(
        np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij"
    )
    idx = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
    centers = origin[None, :] + pitch * (idx.astype(float) + 0.5)
    try:
        m = mesh
        if not m.is_watertight:
            m = m.copy()
            m.merge_vertices(merge_norm=True)  # split STL vertices -> watertight
        if m.is_watertight:
            scan = _scanline_contains(m, origin, dims, pitch, offset=0.5)
            if scan is not None and scan.any():
                return scan.reshape(dims)
            # contains() allocates per-ray candidate buffers that scale with
            # the mesh's face count — unchunked on dense organic meshes it
            # peaks >14 GB. Bound each batch to ~256 MB of worst-case working
            # set (64 B per face per point is conservative for the ray/hit
            # arrays trimesh builds) — ~8k points on typical meshes, shrinking
            # back to 512 on very dense ones rather than unbounded.
            chunk = int(
                np.clip(
                    (256 * 1024 * 1024) // (64 * max(len(m.faces), 1)),
                    512,
                    8192,
                )
            )
            out = np.zeros(len(centers), dtype=bool)
            for s in range(0, len(centers), chunk):
                out[s : s + chunk] = np.asarray(m.contains(centers[s : s + chunk]))
            # degenerate/flipped watertight meshes can report every center as
            # outside — don't accept an empty grid, fall through to the
            # surface-distance path instead
            if out.any():
                return out.reshape(dims)
    except Exception:
        pass
    # fallback for non-watertight meshes: surface-near cells + hole fill
    try:
        dist = np.zeros(len(centers), dtype=float)
        for s in range(0, len(centers), 512):
            dist[s : s + 512] = mesh.nearest.on_surface(centers[s : s + 512])[1]
        m = (dist <= pitch * 0.75).reshape(dims)
        m = ndimage.binary_fill_holes(m)
        # border_value=1: outside-the-grid counts as solid, otherwise the
        # erosion half of the closing strips the outermost cell row on every
        # grid edge (boundary cells have no neighbour beyond the array)
        m = ndimage.binary_closing(m, iterations=1, border_value=1)
        if m.any():
            return np.asarray(m, dtype=bool)
    except Exception:
        pass
    # coarsest fallback: everything inside the bbox counts
    return np.ones(dims, dtype=bool)


def _coverage_field(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    dims: tuple[int, int, int],
    pitch: float,
    solid: np.ndarray,
) -> np.ndarray | None:
    """(ncells,) fill fraction in [0,1] — the "thin soft transition" at the
    part boundary: 1.0 inside, 0.0 outside, a partial fraction on
    surface-crossing cells.

    Cheap two-tier probe: the cell-corner node containment is sampled once
    for the whole grid, so cells with uniform corners score 0.0/1.0 for
    free and only corner-mixed cells pay a 3x3x3 interior subsample.
    `None` when the mesh admits no reliable containment (non-watertight) —
    callers then keep the binary boundary model. Non-solid cells keep their
    partial fractions too; consumers apply coverage only on solid cells.
    """
    try:
        m = mesh
        if not m.is_watertight:
            m = m.copy()
            m.merge_vertices(merge_norm=True)  # split STL vertices -> watertight
        if not m.is_watertight:
            return None
        nx1, ny1, nz1 = dims[0] + 1, dims[1] + 1, dims[2] + 1
        ii, jj, kk = np.meshgrid(np.arange(nx1), np.arange(ny1), np.arange(nz1), indexing="ij")
        pts = origin[None, :] + pitch * np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(float)
        # contains() scales with the mesh's face count — bound each batch to
        # the same ~256 MB worst-case working set as _fill_grid's chunking
        chunk = int(
            np.clip(
                (256 * 1024 * 1024) // (64 * max(len(m.faces), 1)),
                512,
                8192,
            )
        )

        def _contains(query: np.ndarray) -> np.ndarray:
            out = np.zeros(len(query), dtype=bool)
            for s0 in range(0, len(query), chunk):
                out[s0 : s0 + chunk] = np.asarray(m.contains(query[s0 : s0 + chunk]))
            return out

        # the node grid is a regular lattice too — axis scanlines amortize
        # one ray per line instead of one per node (contains() fallback for
        # meshes the scanline can't classify)
        node_scan = _scanline_contains(m, origin, (nx1, ny1, nz1), pitch, offset=0.0)
        node_in = (
            node_scan.reshape(nx1, ny1, nz1)
            if node_scan is not None
            else _contains(pts).reshape(nx1, ny1, nz1)
        )
        # count of contained corner nodes per cell (0..8) in Hex8 order
        c8 = (
            node_in[:-1, :-1, :-1].astype(np.int16)
            + node_in[1:, :-1, :-1]
            + node_in[1:, 1:, :-1]
            + node_in[:-1, 1:, :-1]
            + node_in[:-1, :-1, 1:]
            + node_in[1:, :-1, 1:]
            + node_in[1:, 1:, 1:]
            + node_in[:-1, 1:, 1:]
        )
        cov = (c8 == 8).astype(np.float64).ravel()
        # cells needing the 3x3x3 subsample: corner-mixed cells, PLUS solid
        # cells with all-outside corners — a feature thinner than one cell
        # (its centre is inside but no corner is) is invisible to the
        # corner probe; leaving it at 0.0 would give the cell zero
        # stiffness and a singular K. The subsample grid includes the
        # cell centre, so every solid cell scores >= 1/27 here.
        solid_flat = np.asarray(solid, dtype=bool).ravel()
        mixed = np.flatnonzero(
            ((c8.ravel() > 0) & (c8.ravel() < 8)) | (solid_flat & (c8.ravel() == 0))
        )
        if mixed.size:
            ci = mixed // (dims[1] * dims[2])
            rem = mixed % (dims[1] * dims[2])
            cj = rem // dims[2]
            ck = rem % dims[2]
            cell_ijk = np.stack([ci, cj, ck], axis=-1).astype(float)
            # 3x3x3 subsample at sub-cell centres (fractions 1/6, 3/6, 5/6)
            sub = (np.arange(3) + 0.5) / 3.0
            si, sj, sk = np.meshgrid(sub, sub, sub, indexing="ij")
            offs = np.stack([si, sj, sk], axis=-1).reshape(27, 3)
            sub_pts = origin[None, None, :] + pitch * (cell_ijk[:, None, :] + offs[None, :, :])
            inside = _contains(sub_pts.reshape(-1, 3)).reshape(-1, 27)
            cov[mixed] = inside.mean(axis=1)
        return cov
    except Exception:
        return None


def voxelize(
    mesh: trimesh.Trimesh,
    pitch_mm: float,
    max_elements: int,
    pad_cells: int = 0,
    *,
    coverage: bool = False,
) -> VoxelDomain:
    """Rasterize the mesh into a filled uniform grid aligned to its bbox.

    Adaptive: pitch grows until the solid element count fits `max_elements`.
    Cell-center containment keeps the voxel volume close to the real part
    (stair-stepping at boundaries, ~half a cell).

    `pad_cells` > 0 grows the grid bounds by that many cells on EVERY side
    before `dims` are computed — the padding stays void (the mesh fill does
    not cover it) so `solid` remains the part mask only. Used for
    design-domain dilation (`grow_mm`): optimizer filters and region
    masks then have void headroom instead of clipping at the part bbox.

    `coverage=True` additionally stores `domain.coverage`: the per-cell
    fill fraction (1 interior, 0 exterior, partial on surface-crossing
    cells) for the coverage-weighted boundary-stiffness verification
    model. The binary `solid` mask is identical either way; coverage
    returns `None` when the mesh admits no reliable containment.
    """
    bounds = np.asarray(mesh.bounds, dtype=float)
    lo, hi = bounds[0], bounds[1]
    volume = abs(float(mesh.volume)) if mesh.is_watertight else float(np.prod(hi - lo))
    pitch, _ = choose_pitch(volume, pitch_mm, max_elements)
    pad = int(max(0, pad_cells))

    matrix = np.zeros(1, dtype=bool)
    dims = (1, 1, 1)
    origin = lo
    for _attempt in range(12):
        # pad in cell units of the CURRENT pitch — the adaptive retry grows
        # the pitch, so the padded bounds are recomputed per attempt and the
        # pad width stays `pad` cells regardless of coarsening
        origin = lo - pad * pitch
        hi_p = hi + pad * pitch
        dims = (
            int(max(1, math.ceil(float((hi_p - origin)[0] / pitch)))),
            int(max(1, math.ceil(float((hi_p - origin)[1] / pitch)))),
            int(max(1, math.ceil(float((hi_p - origin)[2] / pitch)))),
        )
        matrix = _fill_grid(mesh, origin, dims, pitch)
        if pad:
            # the pad ring is void by definition — enforce it on the fill
            # fallbacks too (the surface-distance path can mark cells whose
            # centre sits just outside the bbox)
            inner = np.zeros(dims, dtype=bool)
            inner[pad : dims[0] - pad, pad : dims[1] - pad, pad : dims[2] - pad] = True
            matrix &= inner
        if int(matrix.sum()) <= max_elements:
            break
        pitch *= (int(matrix.sum()) / max_elements) ** (1.0 / 3.0) * 1.05
    if not matrix.any():
        raise ValueError("voxelization produced an empty grid")

    cov = _coverage_field(mesh, origin, dims, pitch, matrix) if coverage else None
    return VoxelDomain(
        origin=origin.astype(float),
        pitch=float(pitch),
        dims=dims,
        solid=matrix,
        pitch_requested=pitch_mm,
        pad_cells=pad,
        coverage=cov,
    )


def shift_mask(mask: np.ndarray, direction: tuple[int, int, int] | np.ndarray) -> np.ndarray:
    """Shift a (nx,ny,nz) bool mask by `direction` cells without wrap-around —
    cells shifted past the border are cleared. direction e.g. (0,0,1) = +z."""
    d = np.asarray(direction, dtype=int)
    shifted = np.roll(mask, -d, axis=(0, 1, 2))
    for ax in range(3):
        if d[ax] != 0:
            sl = [slice(None)] * 3
            sl[ax] = slice(-1, None) if d[ax] > 0 else slice(0, 1)
            shifted[tuple(sl)] = False
    return np.asarray(shifted, dtype=bool)


def dilate_solid(domain: VoxelDomain, cells: int = 1) -> np.ndarray:
    """Solid mask dilated by `cells` (for design-domain padding)."""
    if cells <= 0:
        return domain.solid
    try:
        return np.asarray(ndimage.binary_dilation(domain.solid, iterations=int(cells)))
    except Exception:
        return domain.solid


def grid_bounds_mm(domain: VoxelDomain) -> tuple[np.ndarray, np.ndarray]:
    lo = domain.origin
    hi = domain.origin + domain.pitch * np.asarray(domain.dims, dtype=float)
    return lo, hi


def volume_of_solid(domain: VoxelDomain) -> float:
    return float(domain.solid_count) * math.pow(domain.pitch, 3)


def estimate_fem_memory_mb(domain: VoxelDomain, n_cases: int = 1, direct: bool = True) -> float:
    """Predictive peak-memory estimate for a FEM solve on this domain (MB).

    Model: transient COO assembly (576 triplets per cell, 24 B each),
    the CSR stiffness matrix (~60 nnz per cell, 16 B each), u/F vectors
    (~3+ncases), and — for a direct solve — LU fill-in (~12x the CSR data).
    Conservative on purpose: better to refuse than to be OOM-killed
    mid-iteration (rc=137 loses the whole job)."""
    cells = domain.cell_count
    nodes = domain.node_count
    coo_mb = cells * 576 * 24 / 1e6
    csr_mb = cells * 60 * 16 / 1e6
    vec_mb = nodes * 3 * 8 * (3 + max(1, n_cases)) / 1e6
    lu_mb = (cells * 60 * 8 * 12 / 1e6) if direct else 0.0
    return coo_mb + csr_mb + vec_mb + lu_mb
