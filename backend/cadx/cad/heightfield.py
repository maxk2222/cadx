"""Deterministic heightfield mesh for flat parts.

For flat parts (grip scales, plates, covers) the measured plan-view outline +
stated height define the part almost completely. This builds a watertight
column-solid: outline footprint extruded to `height_mm`, with an optional
surface relief taken from the top-view image luminance. The result enters the
normal mesh pipeline (baseline -> ops -> scoring) like a generated mesh — its
footprint is exact by construction, so it wins the candidate score whenever
the generator produces a wrong pose/shape ("banana").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from cadx.logging import get_logger

log = get_logger(__name__)


def _lum_sampler(image_path: Path, px_per_mm: float, bounds: np.ndarray) -> Any | None:
    """Bilinear luminance sampler over the measured top-view image.

    Returns f(x, y) -> normalised luminance [0..1] for mm coords, or None."""
    try:
        from cadx.vision.measure import _mask

        mask, _filled, rgb = _mask(image_path)
        gray = np.asarray(rgb[..., :3].mean(axis=2) if rgb.ndim == 3 else rgb, dtype=float) / 255.0
        ii, jj = np.nonzero(mask)
        if len(ii) < 8:
            return None
        cy, cx = float(ii.mean()), float(jj.mean())
        step = 1.0 / px_per_mm  # mm per px — sample at native image resolution
        xs = np.arange(bounds[0], bounds[2] + step, step)
        ys = np.arange(bounds[1], bounds[3] + step, step)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        px = np.clip((cx + xx * px_per_mm).astype(int), 0, mask.shape[1] - 1)
        py = np.clip((cy - yy * px_per_mm).astype(int), 0, mask.shape[0] - 1)
        lum = gray[py, px] * mask[py, px]
        vals = lum[lum > 0]
        if len(vals) < 8 or float(vals.max() - vals.min()) < 0.05:
            return None  # flat/no usable texture
        lo, hi = float(vals.min()), float(vals.max())
        lum = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)

        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator((xs, ys), lum, bounds_error=False, fill_value=0.0)

        def sample(xy: np.ndarray) -> np.ndarray:
            return np.clip(interp(xy), 0.0, 1.0)

        return sample
    except Exception:
        log.debug("heightfield.relief_failed", exc_info=True)
        return None


def _cap_tris(poly: Any) -> tuple[np.ndarray, np.ndarray]:
    """Earcut triangulation. Returns (vert_xy, faces) indexing *its own*
    vertex list — callers map them onto the mesh arrays via _ring_index_map."""
    from trimesh.creation import triangulate_polygon

    v, f = triangulate_polygon(poly)
    return np.asarray(v, dtype=float), np.asarray(f, dtype=np.int64)


def _ring_index_map(cap_xy: np.ndarray, ring_xy: np.ndarray) -> np.ndarray:
    """Map cap-triangulation vertex indices -> indices into `ring_xy`.

    Both vertex sets describe the same polygon rings; earcut may append
    closing duplicates, so matching is coordinate-based (rounded key)."""
    from collections import defaultdict

    tab: defaultdict[tuple[int, int], int] = defaultdict(int)
    for j, p in enumerate(np.round(ring_xy / 1e-4).astype(np.int64)):
        tab[(p[0], p[1])] = j
    out = np.zeros(len(cap_xy), dtype=np.int64)
    for i, p in enumerate(np.round(cap_xy / 1e-4).astype(np.int64)):
        j = tab.get((p[0], p[1]), -1)
        if j < 0:
            raise ValueError("cap vertex not on the outline ring")
        out[i] = j
    return out


def _walls(n_ring: int, top_i0: int, bot_i0: int) -> np.ndarray:
    """Quad-strip triangles between the top and bottom copy of one ring.

    No winding flip needed for hole rings: shapely interiors are already
    CW-wound (vs CCW exterior), so the same edge order yields outward-facing
    normals on both."""
    a = top_i0 + np.arange(n_ring)
    b = top_i0 + (np.arange(n_ring) + 1) % n_ring
    c = bot_i0 + (np.arange(n_ring) + 1) % n_ring
    d = bot_i0 + np.arange(n_ring)
    tris = np.stack([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=1)
    return tris.reshape(-1, 3)


def _densify_ring(r: np.ndarray, pitch: float) -> np.ndarray:
    """Subdivide ring edges so consecutive verts are <= pitch apart — dense
    rings make Delaunay include the boundary edges (conformance check still
    guards correctness)."""
    out: list[np.ndarray] = []
    n = len(r)
    for i in range(n):
        a, b2 = r[i], r[(i + 1) % n]
        k = max(1, int(np.ceil(float(np.linalg.norm(b2 - a)) / pitch)))
        for t in range(k):
            out.append(a + (b2 - a) * (t / k))
    return np.asarray(out)


def _top_sheet_delaunay(poly: Any, top_xy: np.ndarray, n_rings: list[int]) -> np.ndarray | None:
    """Delaunay over ring + interior top verts, clipped to the polygon.

    The sheet is only usable when every ring edge appears as a triangle edge —
    otherwise the rim wouldn't meet the side walls (not watertight). Returns
    None in that case; the caller falls back to flat earcut caps."""
    from scipy.spatial import Delaunay
    from shapely import contains_xy

    try:
        tris = np.asarray(Delaunay(top_xy).simplices, dtype=np.int64)
    except Exception:
        return None
    cent = top_xy[tris].mean(axis=1)
    tris = tris[contains_xy(poly.buffer(1e-6), cent[:, 0], cent[:, 1])]
    if not len(tris):
        return None

    # boundary conformance: every ring edge must appear as a sheet edge —
    # per-ring offsets so the wrap-around edge of each ring is checked
    # within its own ring, never across rings
    used = set()
    for t in tris:
        for i in range(3):
            a, b2 = int(t[i]), int(t[(i + 1) % 3])
            used.add((min(a, b2), max(a, b2)))
    off = 0
    for n in n_rings:
        for i in range(n):
            a, b2 = off + i, off + (i + 1) % n
            if (min(a, b2), max(a, b2)) not in used:
                return None
        off += n
    return tris


def build_heightfield_mesh(
    outline_mm: list[list[float]],
    height_mm: float,
    *,
    relief_mm: float = 0.0,
    relief_image: Path | None = None,
    px_per_mm: float | None = None,
    pitch_mm: float | None = None,
) -> trimesh.Trimesh:
    """Column-solid over the measured outline, extruded to `height_mm`.

    Watertight by construction: exact polygon side walls + earcut caps —
    no voxel raster, so neither the silhouette nor the top surface is
    quantised. Frame: centred XY, resting on z=0.
    `pitch_mm` only controls the optional relief interior grid density.
    """
    from shapely.geometry import MultiPolygon, Polygon

    poly = Polygon(outline_mm)
    poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or not poly.is_valid:
        raise ValueError("outline polygon is empty/invalid")
    b = np.asarray(poly.bounds, dtype=float)  # minx, miny, maxx, maxy
    span = float(max(b[2] - b[0], b[3] - b[1]))

    # ring arrays without the duplicated closing vertex
    rings = [np.asarray(poly.exterior.coords, dtype=float)[:-1]]
    rings += [np.asarray(r.coords, dtype=float)[:-1] for r in poly.interiors]

    sampler = None
    if relief_mm > 0 and relief_image is not None and px_per_mm:
        sampler = _lum_sampler(relief_image, px_per_mm, b)

    # ---- vertices -------------------------------------------------------- #
    # layout: [top_ring (n_ring) | top_interior (m) | bottom_ring (n_ring)]
    interior = np.zeros((0, 2))
    if sampler is not None:
        pitch = float(pitch_mm or np.clip(span / 250.0, 0.2, 0.6))
        # dense ring → Delaunay follows the silhouette edge exactly
        rings = [_densify_ring(r, pitch) for r in rings]
        gx, gy = np.meshgrid(
            np.arange(b[0], b[2] + pitch, pitch),
            np.arange(b[1], b[3] + pitch, pitch),
        )
        from shapely import contains_xy

        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        grid = grid[contains_xy(poly.buffer(-pitch * 0.5), grid[:, 0], grid[:, 1])]
        if len(grid) > 60_000:
            grid = grid[np.unique(np.linspace(0, len(grid) - 1, 60_000).astype(int))]
        interior = grid

    n_rings = [len(r) for r in rings]
    ring_xy = np.vstack(rings)
    n_ring = len(ring_xy)

    # caps must tile the *actual* ring vertex set — when rings were densified
    # for the relief sheet, triangulate the densified polygon as-is (no
    # buffer/simplify — that could drop collinear verts and break the map)
    cap_poly: Any = Polygon(rings[0], [r for r in rings[1:]])

    top_xy = np.vstack([ring_xy, interior])
    top_z = np.full(len(top_xy), float(height_mm))
    if sampler is not None and len(interior):
        top_z[n_ring:] += sampler(interior) * relief_mm
    bot = np.column_stack([ring_xy, np.zeros(n_ring)])

    # ---- faces ----------------------------------------------------------- #
    top_sheet = _top_sheet_delaunay(poly, top_xy, n_rings) if sampler is not None else None
    if top_sheet is None:
        interior = np.zeros((0, 2))
        top_xy = ring_xy

    verts = np.vstack([np.column_stack([top_xy, top_z[: len(top_xy)]]), bot])
    bot_i0 = len(top_xy)

    cap_xy, cap_f = _cap_tris(cap_poly)
    remap = _ring_index_map(cap_xy, ring_xy)
    faces: list[np.ndarray] = []
    if top_sheet is not None:
        faces.append(top_sheet)
    else:
        faces.append(remap[cap_f])  # flat earcut cap onto top ring verts
    faces.append(np.fliplr(remap[cap_f]) + bot_i0)  # bottom, reversed winding

    off = 0
    for n in n_rings:
        faces.append(_walls(n, off, bot_i0 + off))
        off += n

    mesh = trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=True, validate=True)
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_watertight:
        raise ValueError("heightfield mesh is not watertight")

    # frame contract: centred XY, resting on z=0
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return mesh


def heightfield_candidate(
    spec: Any,
    outline_mm: list[list[float]] | None,
    *,
    image_path: Path | None = None,
    px_per_mm: float | None = None,
    relief_mm: float = 1.5,
) -> trimesh.Trimesh | None:
    """Build the flat-part heightfield candidate or None when not applicable.

    Applicable when: flat part (height < 30% of max plan dim) + measured
    outline + stated height. The relief is capped and centred so it can't
    violate the height spec noticeably (±relief_mm).
    """
    if not outline_mm or len(outline_mm) < 3:
        return None
    dims = [
        (spec.length_x.value or 0),
        (spec.width_y.value or 0),
        (spec.height_z.value or 0),
    ]
    L, W, Hh = (float(v) for v in dims)
    if Hh <= 0 or max(L, W) <= 0:
        return None
    if Hh >= 0.3 * max(L, W):
        return None  # not a flat part
    try:
        return build_heightfield_mesh(
            outline_mm,
            Hh,
            relief_mm=min(relief_mm, Hh * 0.25),
            relief_image=image_path,
            px_per_mm=px_per_mm,
        )
    except Exception:
        log.debug("heightfield.build_failed", exc_info=True)
        return None
