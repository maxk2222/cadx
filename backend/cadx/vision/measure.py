"""Deterministic plan-view measurement: outline + hole positions from an image.

Runs in the worker (no OCC). Measures an uploaded top/bottom/drawing image:
silhouette (via alpha channel or background removal), hole blobs, and calibrates
pixels -> mm from the stated part length. Output is in the Frame convention
(mm, X = length, part centred in X/Y).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel


class MeasuredHole(BaseModel):
    x: float
    y: float
    dia: float
    kind: Literal["through", "dark"] = "through"
    confidence: float = 1.0


class ImageMeasurement(BaseModel):
    asset_id: str = ""
    role: str = "top"
    px_per_mm: float
    bbox_mm: list[float]  # [length_x, width_y]
    outline_mm: list[list[float]]  # closed polygon, <= 200 points, Frame coords
    holes: list[MeasuredHole] = []
    confidence: float = 1.0
    warnings: list[str] = []


def _mask(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mask, filled, rgb): object mask from alpha or rembg; also returns the
    image pixels so surface holes can be detected on photos."""
    import scipy.ndimage as ndi

    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img)
    a = arr[..., 3]
    if float((a < 128).mean()) > 0.02:
        mask = a >= 128
    else:
        try:
            import rembg

            # u2net (~170 MB) — the 1 GB default model OOMs the worker and,
            # like prep.py, a small frame is enough for a silhouette.
            session = rembg.new_session("u2net")
            small = img.copy()
            small.thumbnail((768, 768), Image.Resampling.LANCZOS)
            cut = rembg.remove(small, session=session)
            mask = np.asarray(cut)[..., 3] >= 128
            if mask.shape != a.shape:
                m_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    img.size, Image.Resampling.LANCZOS
                )
                mask = np.asarray(m_img) >= 128
        except Exception as e:
            raise ValueError(f"background removal failed: {e}") from e
    lab, n = ndi.label(mask)
    if n == 0:
        raise ValueError("no object found in image")
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    mask = lab == int(np.argmax(sizes)) + 1
    return mask, ndi.binary_fill_holes(mask), arr[..., :3]


def _holes_px(mask: np.ndarray, filled: np.ndarray) -> list[tuple[float, float, float]]:
    """(cy, cx, dia_px) of round-ish voids inside the filled silhouette."""
    import scipy.ndimage as ndi
    from skimage.measure import regionprops

    voids = filled & ~mask
    lab, _n = ndi.label(voids)
    out: list[tuple[float, float, float]] = []
    for prop in regionprops(lab):
        area = float(prop.area)
        per = float(prop.perimeter) or 1.0
        if area < 4.0:
            continue
        roundness = 4.0 * np.pi * area / (per * per)
        if roundness < 0.6:
            continue
        cy, cx = prop.centroid
        out.append((cy, cx, 2.0 * np.sqrt(area / np.pi)))
    return out


def _surface_holes_px(
    mask: np.ndarray, filled: np.ndarray, rgb: np.ndarray, px_per_mm: float
) -> list[tuple[float, float, float]]:
    """Low-confidence holes from luminance extremes on the object's surface.

    Photos of real parts show holes as dark voids or bright metal rings, not
    alpha transparency. We take the extreme luminance tails (0.5% / 99%) of the
    object pixels, then keep compact round blobs that contrast strongly with
    their surrounding ring — that survives knurled/textured surfaces where a
    plain threshold floods with texture noise.
    """
    import scipy.ndimage as ndi
    from skimage.measure import regionprops

    inside = filled & mask
    if not inside.any():
        return []
    lum = rgb.astype(float).mean(axis=2)
    inside_lum = lum[inside]
    lo, hi = np.percentile(inside_lum, [0.5, 99.0])
    ring_iters = max(int(0.8 * px_per_mm), 2)
    out: list[tuple[float, float, float]] = []
    for tail in (inside & (lum <= lo), inside & (lum >= hi)):
        tail = ndi.binary_closing(tail, structure=np.ones((5, 5)))
        lab, _ = ndi.label(tail)
        for prop in regionprops(lab):
            area = float(prop.area)
            per = float(prop.perimeter) or 1.0
            dia_mm = 2.0 * np.sqrt(area / np.pi) / px_per_mm
            if not (2.5 <= dia_mm <= 15.0):
                continue
            # photos make hole edges ragged (JPEG ringing, speculars) —
            # looser roundness than alpha voids; contrast+size still gate
            if 4.0 * np.pi * area / (per * per) < 0.3:
                continue
            region = lab == prop.label
            ring = ndi.binary_dilation(region, iterations=ring_iters) & inside & ~region
            if not ring.any():
                continue
            contrast = abs(float(lum[region].mean()) - float(lum[ring].mean()))
            if contrast < 30.0:
                continue
            cy, cx = prop.centroid
            out.append((cy, cx, dia_mm * px_per_mm))
    return out


def _outline_px(filled: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Longest closed contour of the filled mask, simplified, <= 200 points."""
    from shapely import LineString
    from skimage.measure import find_contours

    contours = find_contours(filled.astype(float), 0.5)
    if not contours:
        raise ValueError("no outline contour found")
    pts = max(contours, key=len)  # (row, col) -> (y, x)
    xy = np.column_stack([pts[:, 1], pts[:, 0]])
    line = LineString(xy)
    line = line.simplify(max(0.3 * px_per_mm, 0.5), preserve_topology=True)
    xy = np.asarray(line.coords, dtype=float)
    if len(xy) > 200:
        keep = np.linspace(0, len(xy) - 1, 200).astype(int)
        xy = xy[keep]
    return xy


def _pca_frame(pts_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA basis + reference mean: principal axis becomes +X (right-handed)."""
    mean = pts_px.mean(axis=0)
    cov = np.cov((pts_px - mean).T)
    _, vecs = np.linalg.eigh(cov)
    rot = vecs[:, ::-1]  # principal axis first
    if np.linalg.det(rot) < 0:
        rot[:, 1] *= -1
    return rot, mean


def _to_frame(pts_px: np.ndarray, rot: np.ndarray, mean: np.ndarray, *, flip_y: bool) -> np.ndarray:
    """Apply the PCA basis, flip image-Y to CAD +Y."""
    pts = np.asarray((pts_px - mean) @ rot, dtype=float)
    pts[:, 1] *= -1.0  # image Y grows downward -> CAD +Y up
    if flip_y:
        pts[:, 1] *= -1.0
    return pts


def measure_plan_view(
    path: Path,
    *,
    length_mm: float,
    width_mm: float | None = None,
    role: str = "top",
    reference_mm: float | None = None,
) -> ImageMeasurement:
    """Measure a plan-view image: outline polygon + hole positions in Frame mm.

    `length_mm` calibrates the scale (stated part length). `width_mm` is used
    only for a consistency warning. `reference_mm` is a known hole diameter
    whose median measured diameter cross-checks the calibration.
    """
    mask, filled, rgb = _mask(Path(path))
    warnings: list[str] = []
    confidence = 1.0

    # provisional scale from the raw mask extent, then PCA-align
    extent_x_px = float(np.ptp(np.flatnonzero(filled.any(axis=0))))
    if length_mm <= 0 or extent_x_px <= 0:
        raise ValueError("cannot calibrate: no stated length or empty mask")
    px_per_mm = extent_x_px / length_mm

    outline = _outline_px(filled, px_per_mm)
    rot, mean = _pca_frame(outline)
    aligned = _to_frame(outline, rot, mean, flip_y=(role == "bottom"))
    bbox_px = aligned.max(axis=0) - aligned.min(axis=0)

    # refine: PCA extent may differ from the axis-aligned extent used above
    px_per_mm = float(bbox_px[0]) / length_mm if length_mm > 0 else px_per_mm
    bbox_mm = [float(bbox_px[0]) / px_per_mm, float(bbox_px[1]) / px_per_mm]

    # anisotropic calibration: a slightly tilted photo inflates one axis —
    # when the stated width deviates <15 %, scale Y independently so the
    # measured outline matches BOTH stated dims (holes stay proportional)
    sy_per_mm: float | None = None
    if width_mm and width_mm > 0:
        sy = float(bbox_px[1]) / width_mm
        if sy > 0 and 0.85 <= sy / px_per_mm <= 1.18:
            sy_per_mm = sy
            bbox_mm[1] = width_mm

    if width_mm:
        dev = abs(bbox_mm[1] - width_mm) / max(width_mm, 1e-9)
        if dev > 0.08:
            warnings.append(f"width mismatch: measured {bbox_mm[1]:.1f} vs stated {width_mm:.1f}")
            confidence -= 0.3

    holes_px = _holes_px(mask, filled)
    # surface luminance holes (photos) — dedupe against alpha voids
    n_void = len(holes_px)
    void_centres = [(cy, cx) for cy, cx, _ in holes_px]
    for cy, cx, dia_px in _surface_holes_px(mask, filled, rgb, px_per_mm):
        if any(
            np.hypot(cy - vy, cx - vx) < max(1.5 * px_per_mm, 0.6 * dia_px)
            for vy, vx in void_centres
        ):
            continue
        holes_px.append((cy, cx, dia_px))
    if reference_mm and holes_px:
        dias = [d for _, _, d in holes_px]
        px_per_mm_ref = float(np.median(dias)) / reference_mm
        if abs(px_per_mm_ref - px_per_mm) / px_per_mm > 0.05:
            warnings.append(
                f"hole-diameter scale {px_per_mm_ref:.2f} px/mm deviates from "
                f"length scale {px_per_mm:.2f} px/mm — trusting stated length"
            )

    # hole centres through the same alignment as the outline
    scale_xy = np.array([px_per_mm, sy_per_mm or px_per_mm])

    holes: list[MeasuredHole] = []
    if holes_px:
        centres_px = np.array([[cx, cy] for cy, cx, _ in holes_px], dtype=float)
        centres_aligned = _to_frame(centres_px, rot, mean, flip_y=(role == "bottom"))
        centres_mm = centres_aligned / scale_xy
        for (cx, cy), (hi, (_, _, dia_px)) in zip(centres_mm, enumerate(holes_px), strict=True):
            holes.append(
                MeasuredHole(
                    x=round(float(cx), 2),
                    y=round(float(cy), 2),
                    dia=round(float(dia_px) / px_per_mm, 2),
                    kind="through" if hi < n_void else "dark",
                    confidence=1.0 if hi < n_void else 0.6,
                )
            )
        holes.sort(key=lambda h: h.x)

    outline_mm = aligned / scale_xy
    outline_mm = outline_mm - (outline_mm.max(axis=0) + outline_mm.min(axis=0)) / 2.0
    # recentre holes with the same offset
    off = (aligned.max(axis=0) + aligned.min(axis=0)) / 2.0 / scale_xy
    for h in holes:
        h.x = round(h.x - float(off[0]), 2)
        h.y = round(h.y - float(off[1]), 2)

    pts = [[round(float(x), 2), round(float(y), 2)] for x, y in outline_mm]
    if pts and pts[0] != pts[-1]:
        pts.append(list(pts[0]))

    return ImageMeasurement(
        role=role,
        px_per_mm=round(px_per_mm, 4),
        bbox_mm=[round(bbox_mm[0], 2), round(bbox_mm[1], 2)],
        outline_mm=pts,
        holes=holes,
        confidence=round(max(confidence, 0.0), 2),
        warnings=warnings,
    )


def render_overlay(path: Path, m: ImageMeasurement, out_png: Path) -> None:
    """Debug overlay: measured outline (green) + numbered holes (red)."""
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    # Frame mm -> image px: the outline was PCA-aligned, so a pixel-perfect
    # back-transform isn't trivial; draw the bbox + hole markers instead.
    cx, cy = w / 2.0, h / 2.0
    scale = m.px_per_mm
    for i, hole in enumerate(m.holes, 1):
        px, py = cx + hole.x * scale, cy - hole.y * scale
        r = hole.dia * scale / 2.0
        draw.ellipse([px - r, py - r, px + r, py + r], outline="red", width=2)
        draw.text((px + r + 2, py - 6), f"{i}: d{hole.dia:.1f}", fill="red")
    bw, bh = m.bbox_mm[0] * scale, m.bbox_mm[1] * scale
    draw.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], outline="green", width=2)
    draw.text(
        (8, h - 18),
        f"{m.bbox_mm[0]:.1f} x {m.bbox_mm[1]:.1f} mm, {len(m.holes)} holes, "
        f"conf {m.confidence:.2f}",
        fill="white",
    )
    img.save(out_png)
