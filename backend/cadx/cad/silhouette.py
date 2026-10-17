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

import math
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

            # u2net only: birefnet-general's onnxruntime arena allocates
            # ~9 GB during inference — OOM-kills the container (rc=137), and
            # the arena stays resident even when the run errors out.
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
        global _rembg_session
        # a failed inference keeps the onnxruntime arena resident (observed:
        # ~9 GB after a birefnet failure) — drop the session so it frees
        _rembg_session = None
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


def _deskew_mask(crop: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Rotate a subject crop so its principal axis is horizontal.

    Reference photos routinely show the subject at an angle in the image
    plane — slice-based profile measurement then reads the diagonal band's
    projected thickness (~half the true diameter at 45°, observed on the
    real grip photo: Ø32 read as ~Ø16). PCA the foreground pixels and
    rotate axis-aligned when the mask has a clear long axis; near-square
    or disc masks (eccentricity low) have no meaningful axis — and a
    near-horizontal mask needs no resampling — so those pass through."""
    ys, xs = np.nonzero(crop)
    if len(xs) < 20:
        return crop
    cov = np.cov(np.stack([xs.astype(float), ys.astype(float)]))
    evals, evecs = np.linalg.eigh(cov)
    i = int(np.argmax(evals))
    if evals[i] < 4.0 * float(evals[1 - i]):  # no dominant axis
        return crop
    ang = math.degrees(math.atan2(float(evecs[1, i]), float(evecs[0, i])))
    # the eigenvector has a 180-degree ambiguity — take the line
    # orientation in (-90, 90]; rotating BY that angle aligns the band
    # horizontally (scipy's positive rotation is CW in image coords)
    ang = ((ang + 90.0) % 180.0) - 90.0
    if abs(ang) < 4.0:
        return crop
    from scipy.ndimage import rotate

    rot = np.asarray(rotate(crop.astype(np.uint8), ang, reshape=True, order=0)) > 0
    recrop = _mask_crop(rot)
    return recrop if recrop is not None else crop


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


def _projected_faces(mesh: Any, normal: tuple[float, float, float]) -> NDArray[Any] | None:
    """2D vertex coords of every face projected orthographically along `normal`.

    trimesh.projected() silently returns degenerate paths for some meshes and
    normals (observed: a watertight 40k-face knob produced 11 zero-area
    polygons along ±y while ±x worked). Rasterising the projected triangles
    directly is exact and cheap — the union of filled triangles IS the
    silhouette, concavity included.
    """
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    # image-up = world +Z for side views, +Y for top/bottom so the normalised
    # grid stays comparable to reference photos
    up_hint = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(up_hint, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    verts = np.asarray(mesh.vertices, dtype=float)
    pts = np.stack([verts @ u, verts @ v], axis=1)
    faces = np.asarray(mesh.faces)
    if not len(pts) or not len(faces):
        return None
    return pts[faces]  # (F, 3, 2) — each face's projected triangle


def mesh_view_mask(mesh: Any, view: str = "+y") -> NDArray[np.bool_] | None:
    """Normalised silhouette of `mesh` viewed along `view` (+x/-x/+y/-y).

    Raster rows run top-to-bottom like an image — the projection is flipped
    vertically so mesh +Z maps to the image top."""
    normal = _VIEWS.get(view)
    if normal is None:
        return None
    tris = _projected_faces(mesh, normal)
    if tris is None:
        return None
    minx = float(tris[..., 0].min())
    maxx = float(tris[..., 0].max())
    miny = float(tris[..., 1].min())
    maxy = float(tris[..., 1].max())
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        return None
    from PIL import Image, ImageDraw

    scale = (_GRID - 8) / max(w, h)
    x0 = (_GRID - w * scale) / 2
    y0 = (_GRID - h * scale) / 2
    px = (tris[..., 0] - minx) * scale + x0
    py = (tris[..., 1] - miny) * scale + y0

    img = Image.new("L", (_GRID, _GRID), 0)
    draw = ImageDraw.Draw(img)
    for tri in np.stack([px, py], axis=-1):
        draw.polygon([tuple(p) for p in tri], fill=255)
    return np.flipud(np.asarray(img) > 128)


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


# --------------------------------------------------------------------------- #
# mask -> world-mm polygon (feeds the silhouette_carve / radial_profile_auto   #
# mesh ops — the ops stay image-free, conversion happens orchestration-side)   #
# --------------------------------------------------------------------------- #


def mask_to_outline_mm(
    mask: NDArray[np.bool_],
    mesh_extent_u: float,
    mesh_extent_v: float,
    *,
    fill: float = 1.0,
) -> list[list[float]]:
    """Reference subject mask -> silhouette polygon in view-plane mm.

    Companion for the `silhouette_carve` mesh op: the largest subject contour
    becomes a [[u, v], ...] polygon in the (u, v) basis `mesh_view_mask`
    projects into — u horizontal, v = world +Z for the four side views.
    The subject crop is scaled per axis to `fill` x the mesh's view-plane
    extent (NOT uniformly): the outline bbox spans the full u and v of the
    mesh box, centred horizontally on u = 0 and bottom-aligned to v = 0
    (the part rests on XY in the frame contract). Spec dims are
    authoritative — the carve prism then only trims profile overhang
    inside the fitted box, it can never shrink the stated bounding box
    (uniform min-fit scaling did exactly that: mask aspect != mesh aspect
    silently shaved off stated dims after the dim-fit). `fill` < 1 shrinks
    the outline inside the envelope — a deliberate carve margin.
    Returns [] for empty/degenerate input.
    """
    crop = _mask_crop(mask)
    if crop is None or mesh_extent_u <= 0 or mesh_extent_v <= 0:
        return []
    crop = _deskew_mask(crop)
    from skimage import measure

    padded = np.pad(crop.astype(np.float64), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return []
    pts = np.asarray(max(contours, key=len), dtype=float)  # (row, col), padded space
    if len(pts) > 400:
        # the op schema caps outlines at 500 points — marching-squares
        # contours of a 128² mask stay far below that, dense source masks
        # need a simplifying pass
        from shapely.geometry import Polygon

        poly = Polygon(pts[:, ::-1])  # shapely wants (x=col, y=row)
        tol = 0.75
        while len(poly.exterior.coords) > 400 and tol <= 8.0:
            poly = poly.simplify(tol, preserve_topology=True)
            tol *= 2.0
        if poly.geom_type == "Polygon" and poly.area > 0:
            pts = np.asarray(poly.exterior.coords, dtype=float)[:, ::-1]
    h_px, w_px = crop.shape
    scale_u = fill * mesh_extent_u / w_px
    scale_v = fill * mesh_extent_v / h_px
    return [
        [
            round(float((c - 1.0) - (w_px - 1.0) / 2.0) * scale_u, 4),
            round(float(h_px - r) * scale_v, 4),
        ]
        for r, c in pts
    ]


def _flat_coords(geom: Any) -> list[tuple[float, float]]:
    """All (x, y) coordinate pairs inside a shapely intersection result."""
    if geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing", "Point"):
        return [tuple(c[:2]) for c in geom.coords]
    out: list[tuple[float, float]] = []
    for g in getattr(geom, "geoms", []):
        out.extend(_flat_coords(g))
    return out


def profile_stations_from_outline(
    outline_mm: list[list[float]],
    axis: str = "x",
    n_stations: int = 16,
) -> list[dict[str, float]]:
    """Side-silhouette polygon -> `stations` arg for `radial_profile_auto`.

    `outline_mm` follows the (u, v) view-plane convention of
    `mask_to_outline_mm`/`mesh_view_mask` (u horizontal, v = world +Z). The
    profile axis selects the axial polygon coordinate: "x"/"y" -> u (a
    revolve lying along that horizontal axis), "z" -> v (an upright
    revolve). At `n_stations` positions across the axial extent the station
    `radius_mm` is half the polygon's perpendicular span; `t` is the
    normalised axial position. Returns [] for degenerate input.
    """
    from shapely.geometry import LineString, Polygon

    if len(outline_mm) < 3 or n_stations < 2 or axis not in ("x", "y", "z"):
        return []
    poly = Polygon(outline_mm)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or poly.area <= 0:
        return []
    a_idx = 1 if axis == "z" else 0  # axial coord: v upright, u otherwise
    r_idx = 1 - a_idx
    a_lo, a_hi = float(poly.bounds[a_idx]), float(poly.bounds[a_idx + 2])
    a_span = a_hi - a_lo
    if a_span <= 1e-9:
        return []
    r_lo, r_hi = float(poly.bounds[r_idx]), float(poly.bounds[r_idx + 2])
    r_pad = max(r_hi - r_lo, 1.0) * 2.0
    ts = np.linspace(0.5 / n_stations, 1.0 - 0.5 / n_stations, n_stations)
    radii = np.full(n_stations, np.nan)
    for i, t in enumerate(ts):
        a = a_lo + float(t) * a_span
        if a_idx == 0:
            line = LineString([(a, r_lo - r_pad), (a, r_hi + r_pad)])
        else:
            line = LineString([(r_lo - r_pad, a), (r_hi + r_pad, a)])
        vals = [c[r_idx] for c in _flat_coords(poly.intersection(line))]
        if vals:
            radii[i] = (max(vals) - min(vals)) / 2.0
    ok = np.isfinite(radii)
    if not ok.any():
        return []
    # concave outlines can miss a slice line entirely — fill gaps by
    # interpolation rather than dropping the station
    radii[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), radii[ok])
    stations = [{"t": 0.0, "radius_mm": round(float(radii[0]), 4)}]
    stations += [
        {"t": round(float(t), 4), "radius_mm": round(float(r), 4)}
        for t, r in zip(ts, radii, strict=True)
    ]
    stations.append({"t": 1.0, "radius_mm": round(float(radii[-1]), 4)})
    return stations
