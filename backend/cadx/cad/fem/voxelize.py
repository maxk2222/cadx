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


@dataclass
class VoxelDomain:
    origin: np.ndarray  # (3,) world-mm min corner of cell (0,0,0)
    pitch: float  # mm edge length
    dims: tuple[int, int, int]  # (nx, ny, nz) cells
    solid: np.ndarray  # bool (nx, ny, nz) — inside the part
    pitch_requested: float = 0.0  # before adaptive coarsening
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


def _fill_grid(
    mesh: trimesh.Trimesh, origin: np.ndarray, dims: tuple[int, int, int], pitch: float
) -> np.ndarray:
    """Solid mask over an exact-bounds grid.

    Watertight meshes use cell-center containment (no systematic growth).
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
        m = ndimage.binary_closing(m, iterations=1)
        if m.any():
            return np.asarray(m, dtype=bool)
    except Exception:
        pass
    # coarsest fallback: everything inside the bbox counts
    return np.ones(dims, dtype=bool)


def voxelize(mesh: trimesh.Trimesh, pitch_mm: float, max_elements: int) -> VoxelDomain:
    """Rasterize the mesh into a filled uniform grid aligned to its bbox.

    Adaptive: pitch grows until the solid element count fits `max_elements`.
    Cell-center containment keeps the voxel volume close to the real part
    (stair-stepping at boundaries, ~half a cell).
    """
    bounds = np.asarray(mesh.bounds, dtype=float)
    lo, hi = bounds[0], bounds[1]
    volume = abs(float(mesh.volume)) if mesh.is_watertight else float(np.prod(hi - lo))
    pitch, _ = choose_pitch(volume, pitch_mm, max_elements)

    matrix = np.zeros(1, dtype=bool)
    dims = (1, 1, 1)
    for _attempt in range(12):
        dims = (
            int(max(1, math.ceil(float((hi - lo)[0] / pitch)))),
            int(max(1, math.ceil(float((hi - lo)[1] / pitch)))),
            int(max(1, math.ceil(float((hi - lo)[2] / pitch)))),
        )
        matrix = _fill_grid(mesh, lo, dims, pitch)
        if int(matrix.sum()) <= max_elements:
            break
        pitch *= (int(matrix.sum()) / max_elements) ** (1.0 / 3.0) * 1.05
    if not matrix.any():
        raise ValueError("voxelization produced an empty grid")

    return VoxelDomain(
        origin=lo.astype(float),
        pitch=float(pitch),
        dims=dims,
        solid=matrix,
        pitch_requested=pitch_mm,
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
