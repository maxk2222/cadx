"""Anchor reconciliation: authoritative hole positions from image + mesh + spec.

The measured top view gives image hole positions; the generated mesh has real
tunnels/dimples. `build_anchors` pairs them (nearest neighbour <= 4 mm), flags
an X-mirror when the mesh outline matches the image outline better flipped,
and merges spec hole diameters so CAD code gets exact positions AND diameters.

Spec features with resolvable position hints ("75mm from -X end, centred in
Y") are synthesized into `source="spec"` anchors even when image/mesh
evidence is empty — declared intent, not a scan — so FEM `hole_index`
regions, topopt auto-keep rings and the deterministic re-cut all work from
the declared positions. A measured anchor within 3 mm wins the x/y (real
geometry beats declared) but inherits the spec name/kind/dia/bolt_size.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from cadx.cad.poshint import parse_position_hints, resolve_end_hint
from cadx.llm.schemas import Feature, PartSpec

# spec feature types that are (or carry) a bore
_HOLE_TYPES = ("hole", "thread", "insert", "bearing_seat")
# declared bore axis in free text ("x-axis", "along z") — same narrow shape
# as verifyspec's so "x = 12" offsets never read as an axis declaration
_AXIS_TEXT_RE = re.compile(
    r"\b([xyz])[\s-]*axis\b|\balong\s+(?:the\s+)?([xyz])(?:[\s-]*axis)?\b", re.I
)
# spec-vs-measured same-position merge radius (mm, in-plane)
_SPEC_MERGE_MM = 3.0


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
    # "image-approx" marks anchors from a NON-authoritative measurement
    # (aspect-fallback perspective shot) — downstream gates treat them like
    # mesh hints, not exact positions. "spec" marks declared-intent anchors
    # (authoritative position) — appended as "+spec" when a spec anchor
    # merges onto a measured one.
    source: Literal[
        "image",
        "mesh",
        "image+mesh",
        "image-approx",
        "image-approx+mesh",
        "scan",
        "spec",
        "image+spec",
        "mesh+spec",
        "image+mesh+spec",
        "image-approx+spec",
        "image-approx+mesh+spec",
        "scan+spec",
    ]
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


def _bolt_size(text: str) -> str:
    """Extract a metric fastener designation ("M5", "M4x0.7") from a string."""
    m = re.search(r"\bM\d+(?:[.x]\d+)?\b", text or "")
    return m.group(0) if m else ""


def _hole_like(f: Feature) -> bool:
    """Spec feature that is (or carries) a bore: hole-ish type, a hole-ish
    name ('hole'/'bore'/'loch'), or a metric bolt designation."""
    return (
        f.type in _HOLE_TYPES
        or bool(re.search(r"hole|bore|loch", f.name or "", re.IGNORECASE))
        or bool(_bolt_size(f.standard or f.name or ""))
    )


def _hole_dia(f: Feature) -> float | None:
    """Declared bore diameter of a spec feature (diameter|dia|bore_diameter)."""
    dia = (
        (f.dims or {}).get("diameter")
        or (f.dims or {}).get("dia")
        or (f.dims or {}).get("bore_diameter")
    )
    try:
        return float(dia) if dia else None
    except (TypeError, ValueError):
        return None


def _spec_hole_dias(spec: PartSpec) -> list[tuple[float, str]]:
    """(diameter, name) of spec features that look like holes, sorted by name."""
    out = []
    for f in spec.features:
        if not _hole_like(f):
            continue
        dia = _hole_dia(f)
        if dia:
            out.append((dia, f.name or ""))
    return out


def _declared_bore_axis(f: Feature, text: str) -> str:
    """'x'|'y'|'z' bore axis declared by the feature — the ``axis`` field
    first, then a narrow axis phrase in the position/description text."""
    axis = str(f.axis or "").strip().lower().lstrip("+-")
    if not axis:
        m = _AXIS_TEXT_RE.search(text)
        if m:
            axis = next(g for g in m.groups() if g).lower()
    return axis


def _spec_xy(f: Feature, bounds: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    """Resolved [x, y] positions of one spec feature on the centred envelope.

    Explicit coords, end-relative hints and mid-plane hints combine per
    axis; the cross product covers multi-offset declarations ("20 and 75mm
    from -X end" yields both x positions). Empty when either in-plane axis
    fails to resolve — a guessed coordinate is worse than none.
    """
    text = str(f.position or "").strip() or str(f.description or "")
    coords, _faces, ends = parse_position_hints(text)
    vals: dict[str, list[float]] = {"x": [], "y": []}
    for ax in ("x", "y"):
        if ax in coords:  # explicit "x=12" + "centred in y" (-> 0.0) hints
            vals[ax].append(coords[ax])
    for e in ends:
        ax = str(e["axis"])
        if ax in vals:
            v = resolve_end_hint(ax, str(e["end"]), float(e["offset_mm"]), *bounds[ax])
            if v not in vals[ax]:
                vals[ax].append(v)
    if not vals["x"] or not vals["y"]:
        return []
    return [(x, y) for x in vals["x"] for y in vals["y"]]


def _spec_anchors(spec: PartSpec) -> list[Anchor]:
    """Anchors synthesized from spec features with resolvable positions.

    A spec feature with a machine-readable position ("75mm from -X end,
    centred in Y", "x=12 y=-5") is declared intent — it must exist as an
    anchor even when the image/mesh evidence is empty, so FEM `hole_index`
    regions, topopt auto-keeps and the deterministic re-cut all see it.
    Hints resolve against the spec envelope centred at origin
    (x ∈ [-length/2, +length/2]); only +z bores are synthesized — a feature
    declaring another bore axis stays for the mesh scan. Both in-plane axes
    must resolve or the feature is skipped (a guessed coordinate is worse
    than none). One feature may emit several anchors: "20 and 75mm from
    -X end" resolves to both x positions.
    """
    lx = getattr(spec, "length_x", None)
    wy = getattr(spec, "width_y", None)
    try:
        lx_v, wy_v = float(lx.value), float(wy.value)  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError):
        return []  # no declared envelope — end-relative hints unresolvable
    if lx_v <= 0 or wy_v <= 0:
        return []
    bounds = {"x": (-lx_v / 2, lx_v / 2), "y": (-wy_v / 2, wy_v / 2)}
    out: list[Anchor] = []
    for f in spec.features:
        if not _hole_like(f):
            continue
        text = str(f.position or "").strip() or str(f.description or "")
        axis = _declared_bore_axis(f, text)
        if axis and axis != "z":
            continue  # conservative: only plan-view (+z) bores get spec anchors
        dia = _hole_dia(f)
        if dia is None or dia <= 0:
            continue
        kind = {"thread": "thread", "insert": "insert", "bearing_seat": "bearing_seat"}.get(
            str(f.type).lower(), "hole"
        )
        bolt = (
            _bolt_size(f.standard or "")
            or _bolt_size(f.thread_spec or "")
            or _bolt_size(f.name or "")
        )
        for x, y in _spec_xy(f, bounds):
            out.append(
                Anchor(
                    index=0,  # re-indexed by build_anchors
                    x=round(x, 2),
                    y=round(y, 2),
                    dia=round(dia, 2),
                    source="spec",
                    name=f.name or "",
                    kind=kind,
                    bolt_size=bolt,
                )
            )
    return out


def _type_anchors(spec: PartSpec, anchors: list[Anchor]) -> None:
    """Assign kind/bolt_size to anchors from spec.features (diameter-matched).

    Only upgrades "hole" anchors: a spec feature of type thread/insert/
    bearing_seat whose diameter (or bolt clearance) is within 15% of the
    anchor bore sets the anchor's kind and bolt size — the FEM side then
    knows this anchor takes a bearing/preload load, not just a pin.
    """
    feats = [f for f in spec.features if _hole_like(f)]
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
    # a non-authoritative measurement (aspect-fallback on a perspective shot)
    # must not produce 'image' anchors that downstream gates treat as exact
    img_src = "image" if (best or {}).get("authoritative", False) else "image-approx"

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
                source=f"{img_src}+mesh" if match >= 0 else img_src,
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

    # spec-declared positions are authoritative intent: each merges onto a
    # measured anchor within _SPEC_MERGE_MM (same axis — the measured x/y
    # wins, real geometry beats declared) or appends as a pure-spec anchor.
    # Spec-anchored entries go FIRST in spec declaration order so
    # hole_index == spec order; measured-only anchors keep x-sort behind.
    spec_anchors = _spec_anchors(spec)
    n_spec = 0
    if spec_anchors:
        claimed: set[int] = set()
        ordered: list[Anchor] = []
        for sa in spec_anchors:
            hit = next(
                (
                    (i, a)
                    for i, a in enumerate(anchors)
                    if i not in claimed
                    and a.axis == sa.axis
                    and float(np.hypot(a.x - sa.x, a.y - sa.y)) <= _SPEC_MERGE_MM
                ),
                None,
            )
            if hit is None:
                ordered.append(sa)
            else:
                i, a = hit
                claimed.add(i)
                a.name = sa.name or a.name
                a.kind = sa.kind
                a.dia = sa.dia
                a.bolt_size = sa.bolt_size or a.bolt_size
                if "spec" not in a.source:
                    a.source = f"{a.source}+spec"  # type: ignore[assignment]
                ordered.append(a)
            n_spec += 1
        anchors = ordered + [a for i, a in enumerate(anchors) if i not in claimed]
    for i, a in enumerate(anchors):
        a.index = i + 1

    note = (
        f"{len(img_holes)} image holes, {n_conf} confirmed by mesh, "
        f"{sum(1 for a in anchors if a.source == 'mesh')} mesh-only, "
        f"{n_spec} spec-declared, "
        f"mirror_x={mirror_x}"
    )
    return FeatureAnchors(holes=anchors, image_outline=img_outline, mirror_x=mirror_x, note=note)
