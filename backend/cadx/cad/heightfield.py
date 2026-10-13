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


def _lum_relief(
    image_path: Path,
    px_per_mm: float,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray | None:
    """(nx, ny) normalised luminance [0..1] at the mm grid points, sampled from
    the measured top-view image. Returns None when sampling is impossible."""
    try:
        from cadx.vision.measure import _mask

        mask, _filled, rgb = _mask(image_path)
        gray = rgb[..., :3].mean(axis=2) if rgb.ndim == 3 else rgb
        # mask centroid in px maps to the outline centroid (frame-centred)
        ii, jj = np.nonzero(mask)
        cy, cx = float(ii.mean()), float(jj.mean())
        # mm -> px: image +y is down, frame +y is up
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        px = np.clip((cx + xx * px_per_mm).astype(int), 0, mask.shape[1] - 1)
        py = np.clip((cy - yy * px_per_mm).astype(int), 0, mask.shape[0] - 1)
        lum = gray[py, px] / 255.0
        inside = mask[py, px]
        vals = lum[inside]
        if len(vals) < 8 or float(vals.max() - vals.min()) < 0.05:
            return None  # flat/no usable texture
        lo, hi = float(vals.min()), float(vals.max())
        return np.clip((lum - lo) / (hi - lo), 0.0, 1.0) * inside
    except Exception:
        log.debug("heightfield.relief_failed", exc_info=True)
        return None


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

    Watertight by construction (voxel columns + marching cubes). Frame:
    centred XY, resting on z=0 — same convention as processed meshes.
    """
    from shapely.geometry import Polygon

    poly = Polygon(outline_mm)
    poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        raise ValueError("outline polygon is empty/invalid")
    b = np.asarray(poly.bounds, dtype=float)  # minx, miny, maxx, maxy
    span = float(max(b[2] - b[0], b[3] - b[1]))
    pitch = float(pitch_mm or np.clip(min(span / 150.0, height_mm / 4.0), 0.2, 0.8))
    xs = np.arange(b[0] - pitch, b[2] + pitch, pitch)
    ys = np.arange(b[1] - pitch, b[3] + pitch, pitch)
    nx, ny = len(xs), len(ys)

    # inside-mask via PIL rasterisation (fast, no per-point shapely calls)
    from PIL import Image, ImageDraw

    W = ny
    Hh = nx
    img = Image.new("L", (W, Hh), 0)

    def px(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[1] - b[1]) / pitch, (p[0] - b[0]) / pitch)

    ImageDraw.Draw(img).polygon([px((float(x), float(y))) for x, y in outline_mm], fill=1)
    # array[row, col]: rows index the image height (x axis here), cols the
    # width (y axis) — shape is already (nx, ny), no transpose needed
    inside = np.asarray(img, dtype=bool)

    top = np.full((nx, ny), float(height_mm))
    if relief_mm > 0 and relief_image is not None and px_per_mm:
        lum = _lum_relief(relief_image, px_per_mm, xs, ys)
        if lum is not None:
            top = top + lum * relief_mm

    nz = int(np.ceil(top.max() / pitch)) + 1
    k_top = np.clip(np.round(top / pitch).astype(int), 1, nz - 1)
    idx = np.arange(nz)[None, None, :]
    mat = inside[:, :, None] & (idx <= k_top[:, :, None])
    if not mat.any():
        raise ValueError("heightfield produced an empty solid")

    tf = np.eye(4)
    tf[0, 0] = tf[1, 1] = tf[2, 2] = pitch
    tf[:3, 3] = [float(b[0] - pitch), float(b[1] - pitch), 0.0]
    mesh = trimesh.voxel.VoxelGrid(mat, transform=tf).marching_cubes
    mesh.apply_transform(tf)
    if not len(mesh.faces):
        raise ValueError("heightfield marching-cubes produced no faces")
    # marching cubes offsets vertices by up to ±pitch/2 — restore the frame
    # contract exactly: centred XY, resting on z=0
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
