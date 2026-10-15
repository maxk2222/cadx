"""Deterministic scoring of generated mesh candidates.

Best-of-N needs a cheap, model-free quality signal: aspect ratio vs the spec,
footprint IoU vs the measured plan view and detected hole count vs declared
features. The same score gates processed meshes before any LLM critique —
a candidate that cannot possibly be the part is rejected without spending a
vision call on it.
"""

from __future__ import annotations

from typing import Any


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


def score_mesh_candidate(
    exec_result: dict[str, Any],
    spec: Any,
    plan_outline: list[list[float]] | None = None,
) -> float:
    """Score a processed-mesh exec_result in [0, 1].

    aspect ratio + footprint IoU (when a measured outline exists) + height
    agreement + detected tunnel count vs declared hole features.
    """
    analysis = exec_result.get("analysis") or {}
    bbox = analysis.get("bbox_size") or []

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

    if has_outline:
        if spec_holes:
            holes = min(len(tunnels) / spec_holes, 1.0)
            return round(0.3 * aspect + 0.3 * iou + 0.2 * height + 0.2 * holes, 3)
        # no declared holes — tunnels would be artefacts; weight the rest
        return round(0.35 * aspect + 0.4 * iou + 0.25 * height, 3)
    if spec_holes:
        holes = min(len(tunnels) / spec_holes, 1.0)
        return round(0.5 * aspect + 0.3 * height + 0.2 * holes, 3)
    return round(0.6 * aspect + 0.4 * height, 3)


# scoring only needs macroscopic geometry (aspect, outline, tunnels) — a
# raw generator mesh can be >500k faces and this fn is called ~30x inside
# _orientation_ranking and per candidate, so cap faces before ray-casting
_SCORE_MAX_FACES = 25_000


def score_raw_mesh(mesh: Any, spec: Any, plan_outline: list[list[float]] | None = None) -> float:
    """Score an oriented+scaled trimesh directly (candidate pre-selection).

    Same terms as score_mesh_candidate but computed from the mesh object —
    footprint via detect_mesh_features, holes as detected tunnels."""
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

    height = _height_score(ext[2], spec)
    if has_outline:
        if spec_holes:
            holes = min(len(tunnels) / spec_holes, 1.0)
            return round(0.3 * aspect + 0.3 * iou + 0.2 * height + 0.2 * holes, 3)
        return round(0.35 * aspect + 0.4 * iou + 0.25 * height, 3)
    if spec_holes:
        holes = min(len(tunnels) / spec_holes, 1.0)
        return round(0.5 * aspect + 0.3 * height + 0.2 * holes, 3)
    return round(0.6 * aspect + 0.4 * height, 3)
