"""Deterministic mesh feature detection: through-holes, dimples, outline.

Casts a ray grid along each principal axis: cells with no hit but surrounded
by hits are through-holes ("tunnels") along that axis; local dips in the
top-down (Z) height map are dimples (countersinks, pockets). Runs in the
executor (trimesh only).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh
from pydantic import BaseModel


class MeshHole(BaseModel):
    x: float
    y: float
    dia: float
    depth: float
    kind: Literal["tunnel", "dimple"]
    axis: Literal["x", "y", "z"] = "z"  # ray/projection axis the void pierces


class MeshFeatures(BaseModel):
    holes: list[MeshHole] = []
    outline_mm: list[list[float]] = []


def _heightfield(
    mesh: trimesh.Trimesh, step_mm: float, axis: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H, xs, ys): max hit coordinate along `axis` per cell of the plane
    spanned by the other two axes, NaN where no hit."""
    i, j = (a for a in range(3) if a != axis)
    xs = np.arange(mesh.bounds[0][i], mesh.bounds[1][i] + step_mm, step_mm)
    ys = np.arange(mesh.bounds[0][j], mesh.bounds[1][j] + step_mm, step_mm)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    origins = np.zeros((xx.size, 3))
    origins[:, i] = xx.ravel()
    origins[:, j] = yy.ravel()
    origins[:, axis] = mesh.bounds[1][axis] + 10.0
    directions = np.zeros((xx.size, 3))
    directions[:, axis] = -1.0
    locs, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    H = np.full(xx.shape, np.nan)
    for r, loc in zip(ray_ids, locs, strict=True):
        z = loc[axis]
        i_, j_ = np.unravel_index(r, xx.shape)
        if np.isnan(H[i_, j_]) or z > H[i_, j_]:
            H[i_, j_] = z
    return H, xs, ys


def _hitgrid(
    mesh: trimesh.Trimesh, step_mm: float, axis: int
) -> tuple[list[list[float]], np.ndarray, np.ndarray, tuple[int, ...]]:
    """Per-cell hit coordinates along `axis` for the plane spanned by the
    other two axes — the full sightline, not just the surface height."""
    i, j = (a for a in range(3) if a != axis)
    xs = np.arange(mesh.bounds[0][i], mesh.bounds[1][i] + step_mm, step_mm)
    ys = np.arange(mesh.bounds[0][j], mesh.bounds[1][j] + step_mm, step_mm)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    origins = np.zeros((xx.size, 3))
    origins[:, i] = xx.ravel()
    origins[:, j] = yy.ravel()
    origins[:, axis] = mesh.bounds[1][axis] + 10.0
    directions = np.zeros((xx.size, 3))
    directions[:, axis] = -1.0
    locs, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    hits: list[list[float]] = [[] for _ in range(xx.size)]
    for r, loc in zip(ray_ids, locs, strict=True):
        hits[int(r)].append(float(loc[axis]))
    return hits, xs, ys, xx.shape


def _tunnels_for_axis(
    mesh: trimesh.Trimesh, step_mm: float, axis: int
) -> tuple[list[MeshHole], np.ndarray]:
    """Void cells enclosed by material on the projection plane — a ray
    passing through the void hits nothing, so it marks "outside"; enclosed
    no-hit cells are through-voids along `axis`."""
    import scipy.ndimage as ndi

    hits, xs, ys, shape = _hitgrid(mesh, step_mm, axis)
    inside = np.fromiter((bool(h) for h in hits), bool, shape[0] * shape[1]).reshape(shape)
    footprint = ndi.binary_fill_holes(inside)
    tunnel = footprint & ~inside
    span = float(mesh.bounds[1][axis] - mesh.bounds[0][axis])
    holes: list[MeshHole] = []
    lab, n = ndi.label(tunnel)
    for i in range(1, n + 1):
        cells = lab == i
        # measure the bore inside its local host band: the ring of material
        # enclosing the void fixes the wall's span along the axis, and cells
        # whose only hits lie OUTSIDE that span still pierce the wall that
        # hosts the bore. The plain no-hit test under-measures when an
        # unrelated feature grazes the bore's full-part sightline (a boss
        # rim behind a flange bore shrank a 6.4 mm bore to a 5.6 mm read).
        ring = ndi.binary_dilation(cells) & inside & ~cells
        ridx = np.nonzero(ring.ravel())[0]
        if ridx.size:
            lo = float(np.median([min(hits[r]) for r in ridx]))
            hi = float(np.median([max(hits[r]) for r in ridx]))
            margin = 0.6 * step_mm
            clear = np.zeros(shape, dtype=bool)
            for r in np.nonzero(footprint.ravel())[0]:
                if not any(lo - margin <= h <= hi + margin for h in hits[r]):
                    clear.ravel()[r] = True
            lab2, _ = ndi.label(clear & footprint)
            seed = lab2[np.nonzero(cells)[0][0], np.nonzero(cells)[1][0]]
            cells = lab2 == seed
        area = float(cells.sum()) * step_mm * step_mm
        if area < 3.0:
            continue
        ii, jj = np.where(cells)
        holes.append(
            MeshHole(
                x=round(float(xs[ii].mean()), 2),
                y=round(float(ys[jj].mean()), 2),
                dia=round(2.0 * np.sqrt(area / np.pi), 2),
                depth=round(span, 2),
                kind="tunnel",
                axis="xyz"[axis],  # type: ignore[typeddict-item]
            )
        )
    return holes, footprint


def detect_mesh_features(mesh: trimesh.Trimesh, step_mm: float = 0.5) -> MeshFeatures:
    """Detect through-holes (all 3 axes), dimples and the XY outline of a mesh.

    A bore along X (e.g. handlebar through-bore) only shows on the X-axis ray
    grid — a Z-only pass reports it as nothing, which made the refine agent
    undo correct cuts.
    """
    import scipy.ndimage as ndi
    from skimage.measure import regionprops

    holes: list[MeshHole] = []
    footprint = np.zeros((0, 0), dtype=bool)
    for axis in range(3):
        axis_holes, fp = _tunnels_for_axis(mesh, step_mm, axis)
        holes += axis_holes
        if axis == 2:
            footprint = fp

    H, xs, ys = _heightfield(mesh, step_mm, axis=2)
    inside = ~np.isnan(H)

    # dimples: local dips >= 1.5 mm below the surrounding max (top view only)
    Hf = np.nan_to_num(H, nan=0.0)
    win = max(round(7.0 / step_mm), 3)
    local_max = ndi.maximum_filter(Hf, size=win)
    dimple = ((local_max - Hf) > 1.5) & inside
    lab, _n = ndi.label(dimple)
    for prop in regionprops(lab):
        area = float(prop.area) * step_mm * step_mm
        if not (8.0 <= area <= 200.0):
            continue
        per_mm = float(prop.perimeter) * step_mm or step_mm
        if 4.0 * np.pi * (area / (per_mm * per_mm)) < 0.5:
            continue
        cells = lab == prop.label
        ii, jj = np.where(cells)
        holes.append(
            MeshHole(
                x=round(float(xs[ii].mean()), 2),
                y=round(float(ys[jj].mean()), 2),
                dia=round(2.0 * np.sqrt(area / np.pi), 2),
                depth=round(float((local_max[cells] - Hf[cells]).mean()), 2),
                kind="dimple",
            )
        )

    holes.sort(key=lambda h: (h.axis, h.x))

    outline_mm: list[list[float]] = []
    try:
        from shapely import LineString
        from skimage.measure import find_contours

        contours = find_contours(np.pad(footprint.astype(float), 1), 0.5)
        if contours:
            pts = max(contours, key=len) - 1.0  # undo the pad
            xy = np.column_stack(
                [
                    xs[np.clip(pts[:, 0].astype(int), 0, len(xs) - 1)],
                    ys[np.clip(pts[:, 1].astype(int), 0, len(ys) - 1)],
                ]
            )
            line = LineString(xy).simplify(0.5, preserve_topology=True)
            xy = np.asarray(line.coords)
            if len(xy) > 200:
                xy = xy[np.linspace(0, len(xy) - 1, 200).astype(int)]
            outline_mm = [[round(float(x), 2), round(float(y), 2)] for x, y in xy]
            if outline_mm and outline_mm[0] != outline_mm[-1]:
                outline_mm.append(list(outline_mm[0]))
    except Exception:
        outline_mm = []

    return MeshFeatures(holes=holes, outline_mm=outline_mm)
