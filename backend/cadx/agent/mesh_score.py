"""Deterministic scoring of generated mesh candidates.

Best-of-N needs a cheap, model-free quality signal: aspect ratio vs the spec,
footprint IoU vs the measured plan view and detected hole count vs declared
features. The same score gates processed meshes before any LLM critique —
a candidate that cannot possibly be the part is rejected without spending a
vision call on it.
"""

from __future__ import annotations

import math
import re
from typing import Any

# spec wording that implies a revolve/round body — shared with the
# orchestrator's deterministic silhouette ops (radial_profile_auto)
_ROUND_RE = re.compile(
    r"round|rund|zylindrisch|cylindrical|rotation|revolve|drehteil|drehkörper|kreisförmig",
    re.IGNORECASE,
)


def spec_suggests_revolve(spec: Any) -> bool:
    """Spec text heuristic: the part is described as round/rotational."""
    texts = [
        str(getattr(spec, "name", "") or ""),
        str(getattr(spec, "summary", "") or ""),
        str(getattr(spec, "symmetry", "") or ""),
        str(getattr(spec, "orientation_note", "") or ""),
    ]
    for f in getattr(spec, "features", None) or []:
        texts.append(f"{getattr(f, 'name', '')} {getattr(f, 'description', '')}")
    return any(_ROUND_RE.search(t) for t in texts)


def _outline_circularity(outline_mm: list[list[float]] | None) -> float | None:
    """4πA/P² of a measured footprint outline — 1.0 is a perfect circle.
    Computed on the outline the executor already measured, so no extra
    mesh probing is needed."""
    if not outline_mm or len(outline_mm) < 3:
        return None
    try:
        from shapely.geometry import Polygon

        poly = Polygon(outline_mm)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.is_empty or poly.length <= 0 or poly.area <= 0:
            return None
        return float(4.0 * math.pi * poly.area / poly.length**2)
    except Exception:
        return None


def _roundness_bonus(
    spec: Any,
    outline_mm: list[list[float]] | None = None,
    circularity: float | None = None,
) -> float:
    """+0.05 when the spec implies a round part and the footprint measures
    near-circular — separates a true revolve from a squarish blob at equal
    aspect/height. Prefers the executor's circularity when it reports one;
    otherwise reuses the already-measured XY outline (no new pipeline)."""
    if not spec_suggests_revolve(spec):
        return 0.0
    circ = circularity if circularity is not None else _outline_circularity(outline_mm)
    return 0.05 if circ is not None and float(circ) >= 0.85 else 0.0


def _height_score(z: float, spec: Any) -> float:
    """How close the part's Z extent is to the stated height (1 = exact).

    Catches the classic failure where a plan-view input becomes a standing
    volumetric object (119x113x111 instead of 119x34x8) — the X/Y aspect term
    alone lets that slip when L/W coincidentally match."""
    H = float(spec.height_z.value or 0)
    if H <= 0 or z <= 0:
        return 0.5  # no stated height — neutral
    dev = abs(z - H) / H
    return max(0.0, 1.0 - min(dev, 1.0))


def _blend_silhouette(base: float, sil_iou: float | None) -> float:
    """Blend the reference-image silhouette IoU into a base score.

    `sil_iou` is the best of the four side-view IoUs (``silhouette_iou``
    returns ``(iou, view)``). None means 'no usable reference mask' and leaves
    the score untouched; a measured 0.0 is a real miss and drags the score
    down. 25% weight — strong enough to separate two candidates with identical
    bounding boxes, weak enough not to override dims/footprint evidence."""
    if sil_iou is None:
        return base
    return round(0.75 * base + 0.25 * max(sil_iou, 0.0), 3)


def score_mesh_candidate(
    exec_result: dict[str, Any],
    spec: Any,
    plan_outline: list[list[float]] | None = None,
) -> float:
    """Score a processed-mesh exec_result in [0, 1].

    aspect ratio + footprint IoU (when a measured outline exists) + height
    agreement + detected tunnel count vs declared hole features +
    silhouette IoU vs the reference image (when the executor measured it) +
    a small footprint-circularity bonus for spec'd round parts.
    """
    analysis = exec_result.get("analysis") or {}
    bbox = analysis.get("bbox_size") or []
    mesh_meta = exec_result.get("mesh") or {}
    sil_iou = mesh_meta.get("silhouette_iou")
    # the executor reports the per-view winner as (silhouette_iou,
    # silhouette_view): an IoU of ~0 WITHOUT a matched view means the
    # reference mask was unusable — not a measured shape miss
    if sil_iou is not None and float(sil_iou) <= 0 and not mesh_meta.get("silhouette_view"):
        sil_iou = None
    elif sil_iou is not None:
        sil_iou = float(sil_iou)

    aspect = 0.5
    L, W = float(spec.length_x.value or 0), float(spec.width_y.value or 0)
    if L > 0 and W > 0 and len(bbox) == 3 and bbox[0] > 0 and bbox[1] > 0:
        dev = abs(bbox[0] / bbox[1] - L / W) / (L / W)
        aspect = max(0.0, 1.0 - min(dev, 1.0))
    height = _height_score(float(bbox[2]) if len(bbox) == 3 else 0.0, spec)

    iou = 0.0
    feats = (exec_result.get("mesh") or {}).get("features") or {}
    mesh_outline = feats.get("outline_mm") or []
    has_outline = bool(plan_outline) and len(mesh_outline) >= 3
    if has_outline:
        try:
            from shapely.geometry import Polygon

            a = Polygon(plan_outline).buffer(0)
            b = Polygon(mesh_outline).buffer(0)
            union = a.union(b).area
            iou = float(a.intersection(b).area / union) if union > 0 else 0.0
        except Exception:
            iou = 0.0

    spec_holes = sum(
        int(getattr(f, "count", 1) or 1)
        for f in (spec.features or [])
        if "hole" in str(getattr(f, "type", "")).lower()
    )
    tunnels = [h for h in feats.get("holes") or [] if h.get("kind") == "tunnel"]
    bonus = _roundness_bonus(spec, mesh_outline, circularity=analysis.get("circularity"))

    def _final(base: float) -> float:
        return round(min(1.0, _blend_silhouette(base, sil_iou) + bonus), 3)

    if has_outline:
        if spec_holes:
            holes = min(len(tunnels) / spec_holes, 1.0)
            base = round(0.3 * aspect + 0.3 * iou + 0.2 * height + 0.2 * holes, 3)
            return _final(base)
        # no declared holes — tunnels would be artefacts; weight the rest
        return _final(round(0.35 * aspect + 0.4 * iou + 0.25 * height, 3))
    if spec_holes:
        holes = min(len(tunnels) / spec_holes, 1.0)
        return _final(round(0.5 * aspect + 0.3 * height + 0.2 * holes, 3))
    return _final(round(0.6 * aspect + 0.4 * height, 3))


# scoring only needs macroscopic geometry (aspect, outline, tunnels) — a
# raw generator mesh can be >500k faces and this fn is called ~30x inside
# _orientation_ranking and per candidate, so cap faces before ray-casting
_SCORE_MAX_FACES = 25_000


def support_score(mesh: Any) -> float:
    """How stably the part rests on the build plate, in [0, 1].

    Fraction of the XY bbox footprint covered by the convex hull of the
    lowest slab of vertices (~5 % of the Z extent). A part standing on a
    proper base (knob on its skirt, scale lying flat) covers most of its
    footprint; a part lying on its side contacts on a thin strip.

    Used as an orientation-ranking term: stated dims and aspect cannot
    distinguish a near-cubic part standing up from the same part lying down.
    """
    ext = [float(v) for v in mesh.extents]
    if min(ext) <= 0:
        return 0.0
    z0 = float(mesh.bounds[0][2])
    slab_h = max(0.05 * ext[2], 0.5)
    verts = mesh.vertices[mesh.vertices[:, 2] <= z0 + slab_h]
    if len(verts) < 3:
        return 0.0
    try:
        from shapely.geometry import MultiPoint

        hull_area = float(MultiPoint(verts[:, :2]).convex_hull.area)
    except Exception:
        return 0.0
    return min(1.0, hull_area / (ext[0] * ext[1]))


def score_raw_mesh(
    mesh: Any,
    spec: Any,
    plan_outline: list[list[float]] | None = None,
    ref_mask: Any = None,
) -> float:
    """Score an oriented+scaled trimesh directly (candidate pre-selection).

    Same terms as score_mesh_candidate but computed from the mesh object —
    footprint via detect_mesh_features, holes as detected tunnels. When a
    reference-image mask is given the side-silhouette IoU blends in — this is
    the only shape signal available when no plan-view measurement exists."""
    import trimesh

    assert isinstance(mesh, trimesh.Trimesh)
    from cadx.cad.mesh_features import detect_mesh_features
    from cadx.cad.mesh_ops import decimate

    mesh = decimate(mesh, _SCORE_MAX_FACES)

    ext = [float(v) for v in mesh.extents]
    aspect = 0.5
    L, W = float(spec.length_x.value or 0), float(spec.width_y.value or 0)
    if L > 0 and W > 0 and ext[0] > 0 and ext[1] > 0:
        dev = abs(ext[0] / ext[1] - L / W) / (L / W)
        aspect = max(0.0, 1.0 - min(dev, 1.0))

    iou = 0.0
    has_outline = False
    feats = detect_mesh_features(mesh, step_mm=max(max(ext) / 120.0, 0.2))
    if plan_outline and len(feats.outline_mm) >= 3:
        try:
            from shapely.affinity import scale as sh_scale
            from shapely.affinity import translate as sh_translate
            from shapely.geometry import Polygon

            a = Polygon(plan_outline).buffer(0)
            b = Polygon(feats.outline_mm).buffer(0)
            # normalise both: centre on centroid, span matched — the raw mesh
            # is spec-scaled already, but robustness costs nothing
            ca, cb = a.centroid, b.centroid
            a = sh_translate(a, -ca.x, -ca.y)
            b = sh_translate(b, -cb.x, -cb.y)
            span_a = max(a.bounds[2] - a.bounds[0], a.bounds[3] - a.bounds[1], 1e-9)
            span_b = max(b.bounds[2] - b.bounds[0], b.bounds[3] - b.bounds[1], 1e-9)
            b = sh_scale(b, xfact=span_a / span_b, yfact=span_a / span_b, origin=(0, 0))
            union = a.union(b).area
            iou = float(a.intersection(b).area / union) if union > 0 else 0.0
            has_outline = True
        except Exception:
            iou = 0.0

    spec_holes = sum(
        int(getattr(f, "count", 1) or 1)
        for f in (spec.features or [])
        if "hole" in str(getattr(f, "type", "")).lower()
    )
    tunnels = [h for h in feats.holes if h.kind == "tunnel"]

    sil_iou: float | None = None
    if ref_mask is not None:
        try:
            from cadx.cad.silhouette import silhouette_iou

            # (iou, view): a matched view means the 0 is a measured miss;
            # view=None means the mask never rasterised — treat as unmeasured
            sil_val, sil_view = silhouette_iou(mesh, ref_mask)
            if sil_view is not None:
                sil_iou = float(sil_val)
        except Exception:
            sil_iou = None

    height = _height_score(ext[2], spec)
    bonus = _roundness_bonus(spec, feats.outline_mm)

    def _final(base: float) -> float:
        return round(min(1.0, _blend_silhouette(base, sil_iou) + bonus), 3)

    if has_outline:
        if spec_holes:
            holes = min(len(tunnels) / spec_holes, 1.0)
            base = round(0.3 * aspect + 0.3 * iou + 0.2 * height + 0.2 * holes, 3)
            return _final(base)
        return _final(round(0.35 * aspect + 0.4 * iou + 0.25 * height, 3))
    if spec_holes:
        holes = min(len(tunnels) / spec_holes, 1.0)
        return _final(round(0.5 * aspect + 0.3 * height + 0.2 * holes, 3))
    return _final(round(0.6 * aspect + 0.4 * height, 3))


def vlm_tiebreak(*_args: Any, **_kwargs: Any) -> None:
    """TODO(workstream-E): pairwise VLM tie-break for the top-2 candidates.

    `_select_candidate` currently resolves near-equal scores purely
    deterministically. A pairwise vision-model pick (two contact sheets ->
    'which is closer to the reference?') would break ties on real shape
    evidence, but this module is sync and has no LLM/vision client — wiring
    one through `_candidate_seed` is a separate change. Kept as a documented
    stub so the call site can land behind a runtime flag without feature
    creep here."""
