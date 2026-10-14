"""Anchor reconciliation: authoritative hole positions from image + mesh.

The measured top view gives image hole positions; the generated mesh has real
tunnels/dimples. `build_anchors` pairs them (nearest neighbour <= 4 mm), flags
an X-mirror when the mesh outline matches the image outline better flipped,
and merges spec hole diameters so CAD code gets exact positions AND diameters.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from cadx.llm.schemas import PartSpec


class Anchor(BaseModel):
    """Measured hole/feature anchor in mm (part frame, origin = bbox center).

    `axis` is the bore/feature axis — "+z" for the classic top-view holes; the
    FEM mesh scan (`cad.fem.anchors3d`) also reports +x/+y side anchors. `z`
    is the centre coordinate along the bore axis (unset = part mid-depth).
    `kind`/`bolt_size` type the feature when spec.features match (thread,
    heatset insert, bearing seat...); FEM keep/bearing logic uses them.
    """

    index: int
    x: float
    y: float
    dia: float
    source: Literal["image", "mesh", "image+mesh", "scan"]
    name: str = ""
    axis: str = "+z"  # +x | +y | +z — bore/feature axis
    z: float | None = None  # centre coord along the bore axis (None = part mid)
    depth_mm: float | None = None  # bore length along `axis`
    kind: str = "hole"  # hole | thread | insert | bearing_seat | dimple
    bolt_size: str = ""  # matched spec feature standard, e.g. "M5"
    # full 3D in-plane position for non-+z anchors (axial coord may be None)
    pos: list[float | None] | None = None


class FeatureAnchors(BaseModel):
    holes: list[Anchor] = []
    image_outline: list[list[float]] = []
    mirror_x: bool = False
    note: str = ""


def _poly(pts: list[list[float]]) -> Any:
    from shapely.geometry import Polygon

    if len(pts) < 3:
        return None
    try:
        p = Polygon(pts)
        return p.buffer(0) if not p.is_valid else p
    except Exception:
        return None


def _iou(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def _flip_x(pts: list[list[float]]) -> list[list[float]]:
    return [[-x, y] for x, y in pts]


def _spec_hole_dias(spec: PartSpec) -> list[tuple[float, str]]:
    """(diameter, name) of spec features that look like holes, sorted by name."""
    import re

    out = []
    for f in spec.features:
        looks_hole = f.type in ("hole", "insert", "bearing_seat", "thread") or bool(
            re.search(r"hole|bore|loch", f.name or "", re.IGNORECASE)
        )
        if not looks_hole:
            continue
        dia = (
            (f.dims or {}).get("diameter")
            or (f.dims or {}).get("dia")
            or (f.dims or {}).get("bore_diameter")
        )
        if dia:
            out.append((float(dia), f.name or ""))
    return out


def _bolt_size(text: str) -> str:
    """Extract a metric fastener designation ("M5", "M4x0.7") from a string."""
    import re

    m = re.search(r"\bM\d+(?:[.x]\d+)?\b", text or "")
    return m.group(0) if m else ""


def _type_anchors(spec: PartSpec, anchors: list[Anchor]) -> None:
    """Assign kind/bolt_size to anchors from spec.features (diameter-matched).

    Only upgrades "hole" anchors: a spec feature of type thread/insert/
    bearing_seat whose diameter (or bolt clearance) is within 15% of the
    anchor bore sets the anchor's kind and bolt size — the FEM side then
    knows this anchor takes a bearing/preload load, not just a pin.
    """
    feats = [
        f
        for f in spec.features
        if f.type in ("hole", "thread", "insert", "bearing_seat")
        or _bolt_size(f.standard or f.name or "")
    ]
    for a in anchors:
        if a.kind != "hole" or not feats:
            continue
        best = None
        best_rel = 0.15
        for f in feats:
            dia = (
                (f.dims or {}).get("diameter")
                or (f.dims or {}).get("dia")
                or (f.dims or {}).get("bore_diameter")
                or (f.dims or {}).get("counterbore_diameter")
            )
            if not dia or a.dia <= 0:
                continue
            rel = abs(float(dia) - a.dia) / max(a.dia, float(dia))
            if rel <= best_rel:
                best_rel, best = rel, f
        if best is None:
            continue
        a.kind = {
            "thread": "thread",
            "insert": "insert",
            "bearing_seat": "bearing_seat",
        }.get(best.type, "hole")
        a.bolt_size = _bolt_size(best.standard or "") or _bolt_size(best.name or "")
        if not a.name:
            a.name = best.name or ""


def build_anchors(  # noqa: C901
    spec: PartSpec,
    measurements: dict[str, dict[str, Any]],
    mesh_features: dict[str, Any] | None,
) -> FeatureAnchors:
    """Reconcile image holes with mesh features into authoritative anchors."""
    # pick the best plan-view measurement: top role preferred, then confidence
    best: dict[str, Any] | None = None
    for m in measurements.values():
        if best is None:
            best = m
            continue
        rank = (m.get("role") == "top", m.get("confidence", 0))
        brank = (best.get("role") == "top", best.get("confidence", 0))
        if rank > brank:
            best = m
    img_holes = (best or {}).get("holes") or []
    img_outline = (best or {}).get("outline_mm") or []

    mesh_holes = [
        h
        for h in (mesh_features or {}).get("holes") or []
        if h.get("kind") == "tunnel" or float(h.get("dia", 99)) <= 12.0
    ]
    mesh_outline = (mesh_features or {}).get("outline_mm") or []

    # mirror check: mesh outline vs image outline, direct and x-flipped
    mirror_x = False
    if img_outline and mesh_outline:
        p_img, p_mesh = _poly(img_outline), _poly(mesh_outline)
        direct = _iou(p_img, p_mesh)
        flipped = _iou(p_img, _poly(_flip_x(mesh_outline)))
        mirror_x = flipped > direct + 0.05

    if mirror_x:
        mesh_holes = [{**h, "x": -h["x"]} for h in mesh_holes]

    anchors: list[Anchor] = []
    used: set[int] = set()
    n_conf = 0
    for ih in img_holes:
        match = -1
        best_d = 4.0
        for j, mh in enumerate(mesh_holes):
            if j in used:
                continue
            d = float(np.hypot(ih["x"] - mh["x"], ih["y"] - mh["y"]))
            if d <= best_d:
                best_d, match = d, j
        if match >= 0:
            used.add(match)
            n_conf += 1
        anchors.append(
            Anchor(
                index=len(anchors) + 1,
                x=round(float(ih["x"]), 2),
                y=round(float(ih["y"]), 2),
                dia=round(float(ih["dia"]), 2),
                source="image+mesh" if match >= 0 else "image",
            )
        )
    for j, mh in enumerate(mesh_holes):
        if j in used:
            continue
        anchors.append(
            Anchor(
                index=len(anchors) + 1,
                x=round(float(mh["x"]), 2),
                y=round(float(mh["y"]), 2),
                dia=round(float(mh["dia"]), 2),
                source="mesh",
            )
        )
    anchors.sort(key=lambda a: a.x)
    for i, a in enumerate(anchors):
        a.index = i + 1

    # spec hole diameters win when counts line up
    spec_dias = _spec_hole_dias(spec)
    if anchors and len(spec_dias) == len(anchors):
        spec_dias.sort(key=lambda t: t[1])
        for a, (dia, name) in zip(anchors, spec_dias, strict=True):
            a.dia = round(dia, 2)
            a.name = name
    else:
        for a in anchors:
            if a.source != "mesh":
                a.dia = round(round(a.dia * 2) / 2, 2)

    _type_anchors(spec, anchors)

    note = (
        f"{len(img_holes)} image holes, {n_conf} confirmed by mesh, "
        f"{sum(1 for a in anchors if a.source == 'mesh')} mesh-only, "
        f"mirror_x={mirror_x}"
    )
    return FeatureAnchors(holes=anchors, image_outline=img_outline, mirror_x=mirror_x, note=note)
