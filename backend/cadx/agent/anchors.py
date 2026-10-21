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

from cadx.agent.spechost import host_span, parse_ranges
from cadx.cad.poshint import parse_coord_points, parse_position_hints, resolve_end_hint
from cadx.llm.schemas import Feature, PartSpec

# axis -> (axial index, in-plane u index, in-plane v index) — same layout as
# cad.fem.anchors3d._AXES (kept local: agent/ never imports executor modules)
_AXES: dict[str, tuple[int, int, int]] = {
    "+x": (0, 1, 2),
    "+y": (1, 0, 2),
    "+z": (2, 0, 1),
}

# spec feature types that are (or carry) a bore
# mirrors features._HOLE_TYPES — intake emits fine-grained hole types
_HOLE_TYPES = (
    "hole",
    "thread",
    "insert",
    "bearing_seat",
    "through_hole",
    "blind_hole",
    "countersink_hole",
    "counterbore_hole",
    "through_hole_with_countersink",
    "clearance_hole",
    "tapped_hole",
    "threaded_hole",
    "threaded_bore",
    "tap_hole",
    "bore",
)
# declared bore axis in free text ("x-axis", "along z", "in X-Richtung",
# "durch Z") — narrow shapes only so "x = 12" offsets never read as an
# axis declaration
_AXIS_TEXT_RE = re.compile(
    r"\b([xyz])[\s-]*(?:axis|achse)\b"
    r"|\balong\s+(?:the\s+)?([xyz])(?:[\s-]*axis)?\b"
    r"|\bin\s+(?:der\s+)?([xyz])[\s-]*richtung\b"
    r"|\bdurch\s+([xyz])\b",
    re.I,
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
        "spec-est",
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
    # True only when the spec DECLARES a bottomed bore ("blind", "Sackloch",
    # dims.depth). Host-span depth bounds (the plate the bore pierces) are
    # NOT blind intent — a through-hole through a 6 mm lug also carries 6 mm.
    blind: bool = False
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
    name ('hole'/'bore'/'loch'/'bohrung' — intake writes EN and DE names), or
    a metric bolt designation."""
    blob = f"{f.type or ''} {f.name or ''} {f.description or ''}".lower()
    return (
        f.type in _HOLE_TYPES
        or bool(re.search(r"hole|bore|loch|bohr|thread|gewinde|tap|durchgang|senk", blob))
        or bool(_bolt_size(f.standard or f.name or ""))
    )


_BLIND_RE = re.compile(
    r"\bblind\b|sackloch|sackbohr|grundloch|\btiefe\b|\bdepth\b|\bdeep\b|"
    r"nicht\s+durchgehend|partial\s+depth",
    re.I,
)


def _is_blind(f: Feature) -> bool:
    """Spec feature declares a bottomed bore — explicit 'blind'/'Sackloch'
    text/type or a depth dimension on the hole itself."""
    blob = f"{f.type or ''} {f.name or ''} {f.position or ''} {f.description or ''}".lower()
    if _BLIND_RE.search(blob):
        return True
    dims = f.dims or {}
    return any(k in dims and dims[k] for k in ("depth", "depth_mm", "tiefe", "bore_depth"))


def _hole_dia(f: Feature) -> float | None:
    """Declared bore diameter of a spec feature (diameter|dia|bore_diameter).

    Tapped/threaded holes usually declare the fastener ("M8-1.25") instead of
    the drilled diameter — fall back to the norm table so the feature still
    emits an anchor (the physical bore is the tap drill; a plain named
    clearance hole gets the clearance diameter)."""
    dia = (
        (f.dims or {}).get("diameter")
        or (f.dims or {}).get("dia")
        or (f.dims or {}).get("bore_diameter")
    )
    try:
        if dia:
            return float(dia)
    except (TypeError, ValueError):
        return None
    bolt = _bolt_size(f"{f.standard or ''} {f.thread_spec or ''} {f.name or ''}")
    m = re.search(r"\d+", bolt)
    if not m:
        return None
    try:
        from cadx.cad.catalog import nominal_hole_diameter

        blob = f"{f.type or ''} {f.name or ''} {f.description or ''}".lower()
        kind = "tap" if re.search(r"tap|thread|gewinde", blob) else "clearance"
        return nominal_hole_diameter(int(m.group(0)), kind)
    except Exception:
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
    first, then a narrow axis phrase in the position/description text, then
    a single face token ("an der +X-Seite") — a bore enters a face along its
    normal."""
    axis = str(f.axis or "").strip().lower().lstrip("+-")
    if not axis:
        m = _AXIS_TEXT_RE.search(text)
        if m:
            axis = next(g for g in m.groups() if g).lower()
    if not axis:
        from cadx.cad.poshint import parse_position_hints

        _c, faces, _e, _n = parse_position_hints(text)
        if len(faces) == 1:
            axis = faces[0].lstrip("+-")
    if not axis:
        # a through-bore's axial coordinate is declared as the pierced
        # plate's SPAN ("x=-60 bis x=-55" -> bore along x) — a single
        # declared range identifies the axis even without an axis word
        rng = parse_ranges(text)
        if len(rng) == 1:
            axis = next(iter(rng))
    return axis


def _axis_vals(
    text: str,
    bounds: dict[str, tuple[float, float]],
    dia: float,
    in_plane: tuple[str, str],
) -> dict[str, list[tuple[float, bool]]]:
    """Per-axis resolved positions for the given in-plane axes.

    Each entry is ``(value, est)`` — ``est`` marks near-end estimates. End
    offsets carry the user's stated semantics; a contradicting model-computed
    coord ("75mm from -x end (x=20)") would spawn a phantom position — the
    end hint wins, agreeing values dedupe."""
    coords, _faces, ends, nears = parse_position_hints(text)
    vals: dict[str, list[tuple[float, bool]]] = {ax: [] for ax in in_plane}
    for ax in in_plane:
        end_vals = [
            resolve_end_hint(ax, str(e["end"]), float(e["offset_mm"]), *bounds[ax])
            for e in ends
            if str(e["axis"]) == ax
        ]
        coord_v = coords.get(ax)  # "x=12" + "centred in y" (-> 0.0) hints
        # "centred in z"/"mid z" folds to 0.0 in poshint, but the part frame
        # rests on XY — the true mid is h/2. Only remap when a mid-word is
        # present so an explicit "z=0" (bottom face) is still honoured.
        if ax == "z" and coord_v == 0.0 and re.search(r"\bcent|mid", text):
            coord_v = (bounds[ax][0] + bounds[ax][1]) / 2
        if coord_v is not None and end_vals and all(abs(coord_v - v) > 1.0 for v in end_vals):
            coord_v = None
        if coord_v is not None:
            vals[ax].append((coord_v, False))
        for v in end_vals:
            if all(v != ex for ex, _ in vals[ax]):
                vals[ax].append((v, False))
    inset = max(1.5 * dia, 6.0)
    for n in nears:
        ax = str(n["axis"])
        if ax in vals and not vals[ax]:
            lo, hi = bounds[ax]
            vals[ax].append((hi - inset if str(n["end"]).startswith("+") else lo + inset, True))
    return vals


def _centers_on_plane(
    f: Feature, in_plane: tuple[str, str]
) -> list[tuple[float, float, bool]] | None:
    """Structured ``Feature.centers`` mapped onto the in-plane axes — the
    intake's own per-instance coordinates beat any text re-parse. ``None``
    when no usable centres are declared (caller falls back to text)."""
    centers = getattr(f, "centers", None) or []
    if not centers:
        return None
    u_ax, v_ax = in_plane
    out: list[tuple[float, float, bool]] = []
    for c in centers:
        u = getattr(c, u_ax, None)
        v = getattr(c, v_ax, None)
        if u is None or v is None:
            continue  # incomplete row — never guess one axis of a centre
        out.append((float(u), float(v), False))
    return out or None


def _pairwise_pts(
    groups: list[dict[str, float]],
    vals: dict[str, list[tuple[float, bool]]],
    in_plane: tuple[str, str],
) -> list[tuple[float, float, bool]]:
    """Pairwise (u, v) positions from multi-point coord groups.

    "x=-25, y=0 and x=+45, y=0" must pair (-25, 0) and (45, 0) — a cross
    product of the per-axis lists would work here but collapses whenever
    the flat coords dict dropped a repeated axis. A group missing an
    in-plane axis takes it from ``vals`` only when that axis resolved to
    exactly one value; otherwise the group is skipped (a guessed
    coordinate is worse than none)."""
    out: list[tuple[float, float, bool]] = []
    for g in groups:
        uv: list[float] = []
        est = False
        ok = True
        for ax in in_plane:
            if ax in g:
                uv.append(g[ax])
            elif len(vals.get(ax) or []) == 1:
                v, e = vals[ax][0]
                uv.append(v)
                est = est or e
            else:
                ok = False
        if ok:
            out.append((uv[0], uv[1], est))
    return out


def _spec_xy(
    f: Feature, bounds: dict[str, tuple[float, float]], dia: float
) -> list[tuple[float, float, bool]]:
    """Resolved [x, y] positions of one spec feature on the centred envelope.

    Returns ``(x, y, est)`` triples — ``est`` flags positions that came from
    a ``near the +x end`` proximity hint instead of an exact offset, so the
    caller can mark the anchor ``spec-est``. Explicit coords, end-relative
    hints and mid-plane hints combine per axis; the cross product covers
    multi-offset declarations ("20 and 75mm from -X end" yields both x
    positions). A near-end hint on an otherwise-unresolved in-plane axis
    resolves to an estimate inset ``max(1.5*dia, 6mm)`` from the declared
    end. Empty when either in-plane axis fails to resolve — a guessed
    coordinate is worse than none.
    """
    text = str(f.position or "").strip() or str(f.description or "")
    vals = _axis_vals(text, bounds, dia, ("x", "y"))
    centers = _centers_on_plane(f, ("x", "y"))
    if centers is not None:
        return centers
    groups = parse_coord_points(text)
    if len(groups) >= 2:
        return _pairwise_pts(groups, vals, ("x", "y"))
    if not vals["x"] or not vals["y"]:
        return []
    return [(x, y, ex or ey) for x, ex in vals["x"] for y, ey in vals["y"]]


def _spec_plane(
    spec: PartSpec,
    f: Feature,
    bounds: dict[str, tuple[float, float]],
    dia: float,
    in_plane: tuple[str, str],
) -> tuple[list[tuple[float, float, bool]], float | None, float | None]:
    """In-plane (u, v) positions of a non-z bore + axial centre hint.

    Same hint rules as :func:`_spec_xy` but on the two axes perpendicular to
    the bore axis (an x-axis bore positions in the y/z plane). An unresolved
    in-plane axis defaults to the plane's mid (0.0 for centred x/y, h/2 for
    the rest-on-XY z) marked ``est`` — end-face bores sit at mid height unless
    declared otherwise. An end/near hint on the bore's OWN axis is not
    in-plane — it resolves to the axial centre coordinate (a "8mm from +x end
    face" tap sits at x=+37), returned as second value (None when absent);
    the third value is a bore-depth hint (2x the axial end offset) for
    keep-region slab sizing — end-face bores only keep the entry region.
    """
    text = str(f.position or "").strip() or str(f.description or "")
    coords, _faces, ends, nears = parse_position_hints(text)
    vals = _axis_vals(text, bounds, dia, in_plane)
    u_ax, v_ax = in_plane
    pts = _centers_on_plane(f, in_plane)
    if pts is None:
        groups = parse_coord_points(text)
        if len(groups) >= 2:
            pts = _pairwise_pts(groups, vals, in_plane)
        else:
            # unresolved in-plane defaults to the plane mid (est); z rests on
            # XY so its mid is h/2, not 0.0
            for ax in in_plane:
                if not vals[ax]:
                    lo, hi = bounds[ax]
                    vals[ax].append(((lo + hi) / 2 if ax == "z" else 0.0, True))
            pts = [(u, v, eu or ev) for u, eu in vals[u_ax] for v, ev in vals[v_ax]]
    bore_ax = next(a for a in "xyz" if a not in in_plane)
    axial, axial_est = _axial_hint(spec, f, bounds, bore_ax, dia, coords, ends, nears)
    axial_depth = _axial_depth(spec, f, bounds, bore_ax, ends, text)
    return (
        [(u, v, est or (axial is not None and axial_est)) for u, v, est in pts],
        axial,
        axial_depth,
    )


def _axial_hint(
    spec: PartSpec,
    f: Feature,
    bounds: dict[str, tuple[float, float]],
    bore_ax: str,
    dia: float,
    coords: dict[str, float],
    ends: list[dict],
    nears: list[dict],
) -> tuple[float | None, bool]:
    """Axial centre coordinate of a side-bore + whether it is an estimate.

    Order: explicit end offset > near-end proximity (est) > declared range
    mid (own text, then host body) > bare coordinate — a bore through ONE
    declared plate sits at that plate's mid when nothing else resolves it."""
    inset = max(1.5 * dia, 6.0)
    for e in ends:
        if str(e["axis"]) == bore_ax:
            return (
                resolve_end_hint(bore_ax, str(e["end"]), float(e["offset_mm"]), *bounds[bore_ax]),
                False,
            )
    for n in nears:
        if str(n["axis"]) == bore_ax:
            lo, hi = bounds[bore_ax]
            return (hi - inset if str(n["end"]).startswith("+") else lo + inset), True
    # a declared span on the bore axis ("x=-60 bis -55") is the plate the
    # bore pierces — its MID is the right axial centre; the flat coord parse
    # would surface the range's own lo endpoint instead (observed: anchor at
    # x=-60 while the measured bore sits at leg mid -57.5 -> false major)
    text = f"{f.position or ''} {f.description or ''}"
    rng = parse_ranges(text).get(bore_ax) or host_span(spec, f, bore_ax)
    if rng is not None:
        return (rng[0] + rng[1]) / 2.0, False
    if coords.get(bore_ax) is not None:
        return coords[bore_ax], False
    return None, False


def _axial_depth(
    spec: PartSpec,
    f: Feature,
    bounds: dict[str, tuple[float, float]],
    bore_ax: str,
    ends: list[dict],
    text: str = "",
) -> float | None:
    """Bore-depth hint for keep-region slab sizing and the deterministic
    re-cut — a declared depth wins over the 2x-offset heuristic (end-face
    bores only keep the entry region). Falling back to the declared axial
    range or the host body's span bounds the re-cut to its plate: a
    full-length cutter drilled through a strut sharing the bore's ray
    (hybrid bracket run)."""
    declared = (f.dims or {}).get("depth") or (f.dims or {}).get("bore_depth")
    try:
        if declared:
            return float(declared)
    except (TypeError, ValueError):
        pass
    for e in ends:
        if str(e["axis"]) == bore_ax:
            return min(2.0 * float(e["offset_mm"]), bounds[bore_ax][1] - bounds[bore_ax][0])
    rng = parse_ranges(text).get(bore_ax) or host_span(spec, f, bore_ax)
    if rng is not None:
        return rng[1] - rng[0]
    return None


def _spec_bounds(spec: PartSpec) -> dict[str, tuple[float, float]] | None:
    """Spec envelope bounds for hint resolution: x/y centred at origin,
    z resting on 0 (the coordinate contract). ``None`` when the spec
    declares no usable envelope."""
    lx = getattr(spec, "length_x", None)
    wy = getattr(spec, "width_y", None)
    hz = getattr(spec, "height_z", None)
    try:
        lx_v, wy_v = float(lx.value), float(wy.value)  # type: ignore[union-attr]
        hz_v = float(hz.value)  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError):
        return None
    if lx_v <= 0 or wy_v <= 0:
        return None
    return {
        "x": (-lx_v / 2, lx_v / 2),
        "y": (-wy_v / 2, wy_v / 2),
        "z": (0.0, max(hz_v, 0.0)),
    }


def _spec_anchors(spec: PartSpec) -> list[Anchor]:
    """Anchors synthesized from spec features with resolvable positions.

    A spec feature with a machine-readable position ("75mm from -X end,
    centred in Y", "x=12 y=-5") is declared intent — it must exist as an
    anchor even when the image/mesh evidence is empty, so FEM `hole_index`
    regions, topopt auto-keeps and the deterministic re-cut all see it.
    Hints resolve against the spec envelope centred at origin
    (x ∈ [-length/2, +length/2]); +z bores resolve on the x/y plane,
    side-face bores (``axis="x"``/``"y"``) on their perpendicular plane with
    an unresolved in-plane axis defaulting to the plane mid (``spec-est``) —
    an end offset on the bore's own axis becomes the axial centre and a
    2x-offset depth hint for the keep slab. For +z bores both in-plane axes
    must resolve or the feature is skipped (a guessed coordinate is worse
    than none) — a ``near the +x end`` proximity hint resolves an axis to an
    estimated inset position, emitted with ``source="spec-est"`` so it is
    distinguishable from exact spec anchors. One feature may emit several
    anchors: "20 and 75mm from -X end" resolves to both x positions.
    """
    bounds = _spec_bounds(spec)
    if bounds is None:
        return []  # no declared envelope — end-relative hints unresolvable
    out: list[Anchor] = []
    for f in spec.features:
        if not _hole_like(f):
            continue
        text = str(f.position or "").strip() or str(f.description or "")
        axis = _declared_bore_axis(f, text)
        dia = _hole_dia(f)
        if dia is None or dia <= 0:
            continue
        ftype = str(f.type or "").lower()
        fblob = f"{ftype} {f.name or ''} {f.standard or ''} {f.thread_spec or ''}".lower()
        kind = {
            "thread": "thread",
            "threaded_hole": "thread",
            "threaded_bore": "thread",
            "tapped_hole": "tap",
            "tap_hole": "tap",
            "insert": "insert",
            "bearing_seat": "bearing_seat",
        }.get(ftype, "hole")
        if kind == "hole" and re.search(r"thread|gewinde|tap", fblob):
            kind = "tap" if "tap" in ftype else "thread"
        bolt = (
            _bolt_size(f.standard or "")
            or _bolt_size(f.thread_spec or "")
            or _bolt_size(f.name or "")
        )
        if axis in ("x", "y"):
            # side-face bore — resolve on the perpendicular plane; scan-format
            # pos[] carries in-plane u/v at their coordinate slots
            in_plane = ("y", "z") if axis == "x" else ("x", "z")
            pts, axial, axial_depth = _spec_plane(spec, f, bounds, dia, in_plane)
            _ax, au, av = _AXES[f"+{axis}"]
            for u, v, est in pts:
                pos: list[float | None] = [None, None, None]
                pos[au], pos[av] = round(u, 2), round(v, 2)
                if axial is not None:
                    pos[_ax] = round(axial, 2)
                out.append(
                    Anchor(
                        index=0,  # re-indexed by build_anchors
                        x=round(pos[0] or 0.0, 2),
                        y=round(pos[1] or 0.0, 2),
                        dia=round(dia, 2),
                        axis=f"+{axis}",
                        z=round(axial, 2) if axial is not None else None,
                        depth_mm=round(axial_depth, 2) if axial_depth else None,
                        pos=pos,
                        source="spec-est" if est else "spec",
                        name=f.name or "",
                        kind=kind,
                        blind=_is_blind(f),
                        bolt_size=bolt,
                    )
                )
            continue
        blind = _is_blind(f)
        for x, y, est in _spec_xy(f, bounds, dia):
            # a +z bore bound to ONE declared plate carries the plate's axial
            # centre + span so the deterministic re-cut drills the plate only —
            # a full-length cutter would pierce a strut/rib on the same ray
            rng = parse_ranges(text).get("z") or host_span(spec, f, "z")
            z_mid = (rng[0] + rng[1]) / 2.0 if rng is not None else None
            depth_mm = rng[1] - rng[0] if rng is not None else None
            if blind and depth_mm is None:
                # declared blind depth (dims.depth) — never fall through to a
                # full-length cut for a bottomed bore
                for dk in ("depth", "depth_mm", "tiefe", "bore_depth"):
                    if (f.dims or {}).get(dk):
                        depth_mm = float(f.dims[dk])
                        break
            out.append(
                Anchor(
                    index=0,  # re-indexed by build_anchors
                    x=round(x, 2),
                    y=round(y, 2),
                    dia=round(dia, 2),
                    z=round(z_mid, 2) if z_mid is not None else None,
                    depth_mm=round(depth_mm, 2) if depth_mm else None,
                    pos=(
                        [round(x, 2), round(y, 2), round(z_mid, 2)] if z_mid is not None else None
                    ),
                    source="spec-est" if est else "spec",
                    name=f.name or "",
                    kind=kind,
                    blind=blind,
                    bolt_size=bolt,
                )
            )
    return out


def _anchor_plane_uv(a: Anchor) -> tuple[float, float] | None:
    """In-plane (u, v) of an anchor on its own declared axis — the ``pos``
    slots first, then the legacy x/y pair for +z anchors. ``None`` when the
    anchor carries no usable in-plane position."""
    _ax, au, av = _AXES.get(str(a.axis or "+z").lower(), _AXES["+z"])
    pos = a.pos
    if isinstance(pos, list) and len(pos) == 3 and pos[au] is not None and pos[av] is not None:
        return float(pos[au]), float(pos[av])
    if (au, av) == (0, 1):
        return float(a.x), float(a.y)
    return None


def _feature_plane_pts(
    spec: PartSpec,
    f: Feature,
    bounds: dict[str, tuple[float, float]] | None,
    axis: str,
    dia: float,
) -> list[tuple[float, float]]:
    """Exact (non-estimated) in-plane positions a spec feature resolves to —
    the same machinery ``_spec_anchors`` uses, filtered to ``est=False``
    points so a mid-plane default never poses as a measured position."""
    if bounds is None:
        return []
    if axis in ("x", "y"):
        in_plane = ("y", "z") if axis == "x" else ("x", "z")
        pts, _axial, _depth = _spec_plane(spec, f, bounds, dia, in_plane)
    else:
        pts = _spec_xy(f, bounds, dia)
    return [(u, v) for u, v, est in pts if not est]


def _typed_bore_geo(
    spec: PartSpec,
) -> list[tuple[Feature, str, float, list[tuple[float, float]]]]:
    """(feature, bore axis letter, dia, exact in-plane pts) per hole-like
    spec feature whose bore axis resolves — the matching table for
    ``_type_anchors``. Features with an unresolvable axis contribute
    nothing: without a declared direction, diameter-only matching let a +z
    phantom absorb a +y bore's name."""
    out: list[tuple[Feature, str, float, list[tuple[float, float]]]] = []
    bounds = _spec_bounds(spec)
    for f in spec.features:
        if not _hole_like(f):
            continue
        text = str(f.position or "").strip() or str(f.description or "")
        axis = _declared_bore_axis(f, text)
        if axis not in ("x", "y", "z"):
            continue
        dia = (
            (f.dims or {}).get("diameter")
            or (f.dims or {}).get("dia")
            or (f.dims or {}).get("bore_diameter")
            or (f.dims or {}).get("counterbore_diameter")
        )
        try:
            f_dia = float(dia)
        except (TypeError, ValueError):
            continue
        if f_dia <= 0:
            continue
        out.append((f, axis, f_dia, _feature_plane_pts(spec, f, bounds, axis, f_dia)))
    return out


def _type_anchors(spec: PartSpec, anchors: list[Anchor]) -> None:
    """Assign kind/bolt_size/name to measured anchors from spec.features.

    Runs AFTER the spec-anchor merge in ``build_anchors`` so the declared
    anchors already exist. A measured anchor inherits a feature's
    name/bolt_size/kind ONLY when the feature's declared bore axis equals
    the anchor's axis AND a resolvable in-plane position lies within
    ``max(2*dia, 6mm)`` of the anchor — diameter alone once let a +z
    phantom pocket absorb the M6 name meant for a +y bore. Axis resolvable
    but position not -> ``kind`` upgrade only (never name/bolt_size); axis
    unresolvable -> the feature transfers nothing at all. The +/-15%
    diameter gate applies to every transfer. Spec-sourced anchors are
    already authoritative and are never re-typed here.
    """
    geo = _typed_bore_geo(spec)
    if not geo:
        return
    for a in anchors:
        if a.kind != "hole" or "spec" in a.source or a.dia <= 0:
            continue
        a_axis = str(a.axis or "+z").strip().lower().lstrip("+-")
        uv = _anchor_plane_uv(a)
        best_full: tuple[float, Feature] | None = None
        best_axis: tuple[float, Feature] | None = None
        for f, f_axis, f_dia, pts in geo:
            if f_axis != a_axis:
                continue
            rel = abs(f_dia - a.dia) / max(a.dia, f_dia)
            if rel > 0.15:
                continue
            full = uv is not None and any(
                float(np.hypot(uv[0] - u, uv[1] - v)) <= max(2.0 * f_dia, 6.0) for u, v in pts
            )
            if full:
                if best_full is None or rel < best_full[0]:
                    best_full = (rel, f)
            elif best_axis is None or rel < best_axis[0]:
                best_axis = (rel, f)
        best = best_full or best_axis
        if best is None:
            continue
        f = best[1]
        a.kind = {
            "thread": "thread",
            "insert": "insert",
            "bearing_seat": "bearing_seat",
        }.get(f.type, "hole")
        if best_full is not None:
            a.bolt_size = _bolt_size(f.standard or "") or _bolt_size(f.name or "")
            if not a.name:
                a.name = f.name or ""


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
            sa_uv = _anchor_plane_uv(sa)
            hit = next(
                (
                    (i, a)
                    for i, a in enumerate(anchors)
                    if i not in claimed
                    and a.axis == sa.axis
                    and sa_uv is not None
                    and (a_uv := _anchor_plane_uv(a)) is not None
                    and float(np.hypot(a_uv[0] - sa_uv[0], a_uv[1] - sa_uv[1])) <= _SPEC_MERGE_MM
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

    # typing runs AFTER the spec merge: a measured anchor may inherit a
    # feature's name/bolt_size/kind only on an axis+position match — never
    # on diameter alone (a phantom must not absorb a spec bore's name)
    _type_anchors(spec, anchors)

    note = (
        f"{len(img_holes)} image holes, {n_conf} confirmed by mesh, "
        f"{sum(1 for a in anchors if a.source == 'mesh')} mesh-only, "
        f"{n_spec} spec-declared, "
        f"mirror_x={mirror_x}"
    )
    return FeatureAnchors(holes=anchors, image_outline=img_outline, mirror_x=mirror_x, note=note)
