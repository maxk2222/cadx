"""Deterministic mesh feature detection: through-holes, dimples, outline.

Casts a Z-ray grid over the mesh: cells with no hit but surrounded by hits are
through-holes ("tunnels"); local dips in the height map are dimples
(countersinks, pockets). Runs in the executor (trimesh only).
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


class MeshFeatures(BaseModel):
    holes: list[MeshHole] = []
    outline_mm: list[list[float]] = []


def _heightfield(
    mesh: trimesh.Trimesh, step_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H, xs, ys): max hit z per (x, y) cell, NaN where no hit."""
    xs = np.arange(mesh.bounds[0][0], mesh.bounds[1][0] + step_mm, step_mm)
    ys = np.arange(mesh.bounds[0][1], mesh.bounds[1][1] + step_mm, step_mm)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    origins = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, mesh.bounds[1][2] + 10.0)])
    directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    locs, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    H = np.full(xx.shape, np.nan)
    for r, z in zip(ray_ids, locs[:, 2], strict=True):
        i, j = np.unravel_index(r, xx.shape)
        if np.isnan(H[i, j]) or z > H[i, j]:
            H[i, j] = z
    return H, xs, ys


def detect_mesh_features(mesh: trimesh.Trimesh, step_mm: float = 0.5) -> MeshFeatures:
    """Detect through-holes, dimples and the XY outline of a mesh."""
    import scipy.ndimage as ndi
    from skimage.measure import regionprops

    H, xs, ys = _heightfield(mesh, step_mm)
    inside = ~np.isnan(H)
    footprint = ndi.binary_fill_holes(inside)
    holes: list[MeshHole] = []
    zspan = float(mesh.bounds[1][2] - mesh.bounds[0][2])

    # tunnels: void cells enclosed by material
    tunnel = footprint & ~inside
    lab, n = ndi.label(tunnel)
    for i in range(1, n + 1):
        cells = lab == i
        area = float(cells.sum()) * step_mm * step_mm
        if area < 3.0:
            continue
        ii, jj = np.where(cells)
        holes.append(
            MeshHole(
                x=round(float(xs[ii].mean()), 2),
                y=round(float(ys[jj].mean()), 2),
                dia=round(2.0 * np.sqrt(area / np.pi), 2),
                depth=round(zspan, 2),
                kind="tunnel",
            )
        )

    # dimples: local dips >= 1.5 mm below the surrounding max
    Hf = np.nan_to_num(H, nan=0.0)
    win = max(round(7.0 / step_mm), 3)
    local_max = ndi.maximum_filter(Hf, size=win)
    dimple = ((local_max - Hf) > 1.5) & inside
    lab, n = ndi.label(dimple)
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

    holes.sort(key=lambda h: h.x)

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
