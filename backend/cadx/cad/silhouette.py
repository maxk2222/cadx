"""Scale-invariant silhouette IoU: reference image vs mesh side projections.

Aspect ratio and height cannot tell a knob from a blob when both fill the
same bounding box — the *outline* can. This module extracts the subject mask
of a reference image (alpha channel or a cached rembg model) and compares it
to the orthographic silhouette of an oriented mesh, normalised to a fixed
grid with uniform scale (similarity-invariant, aspect-preserving).

Used by candidate screening (score_raw_mesh), orientation ranking and the
critique calibration so subjective visual judgement is backed by a measured
shape signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cadx.logging import get_logger

log = get_logger(__name__)

_GRID = 128  # normalised raster side — IoU granularity ≈ 0.006 per pixel row

# side views a 'front' reference photo can correspond to: the mesh is already
# oriented (up=+Z, front=+Y), so a frontal photo maps to a YZ.../XZ projection
_VIEWS: dict[str, tuple[float, float, float]] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

_rembg_session: Any = None
_rembg_tried = False


def _rembg_cached() -> bool:
    """True when a rembg model is already on disk — no surprise ~1 GB
    download inside a worker job (same policy as the mesh server)."""
    models_dir = Path.home() / ".rembg" / "models"
    return models_dir.exists() and any(models_dir.iterdir())


def _rembg() -> Any:
    global _rembg_session, _rembg_tried
    if _rembg_session is None and not _rembg_tried:
        _rembg_tried = True
        try:
            from rembg import new_session  # type: ignore[import-not-found]

            _rembg_session = new_session("birefnet-general")
        except Exception:
            try:
                from rembg import new_session  # type: ignore[import-not-found]

                _rembg_session = new_session("u2net")
            except Exception:
                _rembg_session = None
    return _rembg_session


def reference_mask(path: Path | str) -> NDArray[np.bool_] | None:
    """Boolean subject mask of a reference image (True = subject pixel).

    Sources, in order: the image's own alpha channel (transparent product
    renders, generated concepts) then a rembg background removal — only when
    the model weights are already cached locally.
    """
    try:
        from PIL import Image

        img = Image.open(Path(path))
        rgba = np.asarray(img.convert("RGBA"))
        alpha = rgba[..., 3]
        # a fully-opaque alpha carries no mask information
        if alpha.min() < 128 <= alpha.max():
            return alpha > 128
        if not _rembg_cached():
            return None
        sess = _rembg()
        if sess is None:
            return None
        from rembg import remove  # type: ignore[import-not-found]

        out = np.asarray(remove(img.convert("RGB"), session=sess).convert("RGBA"))
        alpha = out[..., 3]
        if alpha.min() < 128 <= alpha.max():
            return alpha > 128
    except Exception as e:
        log.debug("silhouette.mask_failed", path=str(path), error=str(e)[:160])
        return None
    return None


def save_mask(mask: NDArray[np.bool_], path: Path | str) -> None:
    """Persist a subject mask as an L-mode PNG — lets the executor reuse the
    mask the worker extracted once (rembg need not run per process_mesh)."""
    from PIL import Image

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(p)


def load_mask(path: Path | str) -> NDArray[np.bool_] | None:
    """Load a mask PNG written by save_mask (True = subject pixel)."""
    try:
        from PIL import Image

        arr = np.asarray(Image.open(Path(path)).convert("L"))
        if arr.size == 0 or arr.max() < 128:
            return None
        return arr > 128
    except Exception:
        return None


def _mask_crop(mask: NDArray[np.bool_]) -> NDArray[np.bool_] | None:
    """Mask cropped to its subject bbox; None when empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    return mask[np.argmax(rows) : len(rows) - np.argmax(rows[::-1]) + 1][
        :, np.argmax(cols) : len(cols) - np.argmax(cols[::-1]) + 1
    ]


def _norm_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_] | None:
    """Subject resampled onto a _GRID² canvas, uniform scale, centred —
    similarity-invariant but aspect-preserving."""
    crop = _mask_crop(mask)
    if crop is None:
        return None
    from PIL import Image

    h, w = crop.shape
    scale = (_GRID - 8) / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im = Image.fromarray(crop.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)
    out = np.zeros((_GRID, _GRID), dtype=bool)
    y0, x0 = (_GRID - nh) // 2, (_GRID - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = np.asarray(im) > 128
    return out


def _projected_polygon(mesh: Any, normal: tuple[float, float, float]) -> Any:
    """Shapely polygon of the mesh's orthographic silhouette along `normal`."""
    try:
        path = mesh.projected(normal=list(normal))
        polys = [p for p in path.polygons_full if p.area > 0]
        if not polys:
            return None
        from shapely.ops import unary_union

        return unary_union(polys).buffer(0)
    except Exception:
        return None


def _rasterize_polygon(poly: Any) -> NDArray[np.bool_] | None:
    """Polygon rasterised onto the normalised grid (uniform scale, centred)."""
    if poly is None or poly.is_empty:
        return None
    from PIL import Image, ImageDraw

    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        return None
    scale = (_GRID - 8) / max(w, h)
    nw, nh = w * scale, h * scale
    x0, y0 = (_GRID - nw) / 2, (_GRID - nh) / 2

    def tx(coords: Any) -> list[tuple[float, float]]:
        return [(x0 + (float(x) - minx) * scale, y0 + (float(y) - miny) * scale) for x, y in coords]

    img = Image.new("L", (_GRID, _GRID), 0)
    draw = ImageDraw.Draw(img)
    geoms = getattr(poly, "geoms", None) or [poly]
    for g in geoms:
        if g.is_empty or g.area <= 0:
            continue
        draw.polygon(tx(g.exterior.coords), fill=255)
        for interior in g.interiors:
            draw.polygon(tx(interior.coords), fill=0)
    return np.asarray(img) > 128


def mesh_view_mask(mesh: Any, view: str = "+y") -> NDArray[np.bool_] | None:
    """Normalised silhouette of `mesh` viewed along `view` (+x/-x/+y/-y).

    Raster rows run top-to-bottom like an image — the projection is flipped
    vertically so mesh +Z maps to the image top."""
    normal = _VIEWS.get(view)
    if normal is None:
        return None
    poly = _projected_polygon(mesh, normal)
    mask = _rasterize_polygon(poly)
    return None if mask is None else np.flipud(mask)


def _iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def silhouette_iou(
    mesh: Any, ref_mask: NDArray[np.bool_], *, views: tuple[str, ...] | None = None
) -> tuple[float, str | None]:
    """Best scale-invariant IoU between `ref_mask` and the mesh's side
    silhouettes. Returns (iou, view) — (0.0, None) when unusable.

    Mirrors count too: a front photo may show the part flipped versus the
    generator's canonical front, so each view is also compared mirrored.
    Defaults to the four side views — a plan-view (top/bottom) photo feeds
    footprint IoU instead.
    """
    ref = _norm_mask(ref_mask)
    if ref is None:
        return 0.0, None
    best_iou, best_view = 0.0, None
    for view in views or ("+x", "-x", "+y", "-y"):
        m = mesh_view_mask(mesh, view)
        if m is None:
            continue
        for cand in (m, np.fliplr(m)):
            iou = _iou(ref, cand)
            if iou > best_iou:
                best_iou, best_view = iou, view
    return round(best_iou, 4), best_view


def mesh_silhouette_iou(a: Any, b: Any, *, views: tuple[str, ...] = ("+y", "+x", "+z")) -> float:
    """Mean same-view silhouette IoU between two meshes (drift metric).

    Used by the refine shape guard: an op that reshapes the working mesh away
    from the baseline (flatten on a lying part, non-uniform squash) drops this
    below ~0.8 even when dims still pass — deterministic destruction evidence.
    """
    vals: list[float] = []
    for v in views:
        ma, mb = mesh_view_mask(a, v), mesh_view_mask(b, v)
        if ma is not None and mb is not None:
            vals.append(_iou(ma, mb))
    return round(sum(vals) / len(vals), 4) if vals else 1.0
