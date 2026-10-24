"""Deterministic feature verification: extract measurable features from a
generated shape and compare them against the spec's feature list.

Feedback-only: findings flow into the repair/critique feedback but never block
`succeeded` on their own — geometric heuristics are approximate by nature,
especially on faceted mesh solids.

Two measurement layers: the analytic face scan (holes/countersinks/rounds/
chamfers/bosses) plus a rule-based face-adjacency-graph pass (pockets,
grooves, notches — see the AAG section below).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

from build123d import GeomType, Shape
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepTools import BRepTools
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec
from OCP.TopAbs import TopAbs_Orientation, TopAbs_State
from pydantic import BaseModel, Field

from cadx.cad.analysis import _FACET_FACE_CAP


class MeasuredHoleGroup(BaseModel):
    id: str = ""  # stable selector id (h1… round bore, s1… slot) — assign_feature_ids
    diameter: float  # mm
    count: int
    axis: str = ""  # dominant axis direction: +x|-x|+y|-y|+z|-z
    depth_class: str = "through"  # through | blind | unknown
    centers: list[list[float]] = Field(default_factory=list)  # legacy per-hole [x, y]
    # Axis-aware center coordinates: one [x, y, z] row per member; None means
    # that coordinate is not observable (notably the through-axis position of
    # a ray-detected tunnel). Never substitute the global bbox midpoint.
    centers_xyz: list[list[float | None]] = Field(default_factory=list)
    blocked: int = 0  # bores in this group sealed by a membrane (tunnel-verified)
    elongated: bool = False  # slot-like void (aspect ratio > threshold), not a round hole
    # through_axis is the tunnel/pierce direction — same value as `axis`,
    # restated so slot consumers never misread `axis` as the elongation
    # direction (a leg slot is THROUGH +x but elongated ALONG +z).
    through_axis: str = ""
    # elong_axis: the in-plane direction the void is stretched along,
    # perpendicular to through_axis — populated for elongated slots only
    # ("" for round bores). The "+" sign is a fixed placeholder: a
    # symmetric span has no orientation, only the axis letter is meaningful.
    elong_axis: str = ""
    # width_mm/length_mm: minor/major in-plane spans of the void's
    # tunnel-grid footprint — a 6.5x15 slot reports 6.5/15; a round bore
    # spans ~diameter on both. 0.0 when the spans were never measured
    # (the analytic B-Rep path measures diameter only).
    width_mm: float = 0.0
    length_mm: float = 0.0


class MeasuredBoss(BaseModel):
    """Exterior (protruding) full-wrap cylinder — a boss/tube, not a fillet."""

    id: str = ""  # stable selector id (b1…) — assign_feature_ids
    axis: str = ""  # dominant axis direction: +x|-x|+y|-y|+z|-z
    diameter: float = 0.0
    height: float = 0.0


class MeasuredPattern(BaseModel):
    """A recognised repetition of features — bolt circle, linear array, rect
    grid, slot array/grid (quiddity). Plain records only; members carry the
    per-member centres when the recogniser reports them."""

    id: str = ""  # stable selector id (p1…) — assign_feature_ids
    kind: str = ""  # bolt_circle|linear_array|rect_grid|rect_set|slot_array|slot_grid|…
    count: int = 0
    axis: str = ""  # member bore axis (holes) or array direction (slots)
    center: list[float] = Field(default_factory=list)  # pattern centre [x, y, z]
    diameter: float = 0.0  # pitch-circle diameter (bolt circles only)
    width: float = 0.0
    height: float = 0.0
    pitch: float = 0.0  # member pitch (linear arrays / grids)
    rows: int = 0
    cols: int = 0
    angle: float = 0.0
    member_diameter: float = 0.0  # mean member bore Ø (or slot width)
    members: list[list[float]] = Field(default_factory=list)  # member centres


class MeasuredBoredBoss(BaseModel):
    """A protruding boss with a coaxial bore (standoff with a screw hole)."""

    id: str = ""  # stable selector id (bb1…) — assign_feature_ids
    axis: str = ""
    location: list[float] = Field(default_factory=list)
    boss_diameter: float = 0.0
    bore_diameter: float = 0.0
    boss_height: float = 0.0
    bore_depth: float = 0.0


class MeasuredChamfer(BaseModel):
    """A recognised chamfer edge (quiddity) — richer than a 45°-plane count."""

    id: str = ""  # stable selector id (c1…) — assign_feature_ids
    axis: str = ""  # edge direction letter: x|y|z
    leg1: float = 0.0
    leg2: float = 0.0
    angle: float = 0.0
    at: list[float] = Field(default_factory=list)


class MeasuredFillet(BaseModel):
    """A recognised fillet edge (quiddity): radius + convex/concave side."""

    id: str = ""  # stable selector id (f1…) — assign_feature_ids
    axis: str = ""  # edge direction letter: x|y|z
    radius: float = 0.0
    at: list[float] = Field(default_factory=list)
    side: str = ""  # convex | concave


class MeasuredPocket(BaseModel):
    """A recess whose floor face is ringed by concave adjacency edges and
    opens through a single outer face (structural/AAG-derived — the class a
    hole/boss scan cannot see: a pocket is neither a hole_group nor a boss)."""

    id: str = ""  # stable selector id (pk1…) — assign_feature_ids
    axis: str = ""  # dominant axis of the opening (floor-normal) direction
    depth: float = 0.0  # floor to opening-plane distance (mm)
    size: list[float] = Field(default_factory=list)  # plan dims ⊥ axis, desc
    center: list[float] = Field(default_factory=list)  # floor centre [x, y, z]
    centers: list[list[float]] = Field(default_factory=list)  # [[x, y]] plan-view


class MeasuredGroove(BaseModel):
    """An open channel: a bottom face bridged by a pair of parallel opposing
    walls joined by concave edges. ``open_ends`` is 2 for a through slot cut
    edge-to-edge, 1 for a blind channel (closed by an end cap face)."""

    id: str = ""  # stable selector id (gr1…) — assign_feature_ids
    axis: str = ""  # dominant axis of the channel (through-) direction
    opening: str = ""  # dominant axis of the opening (bottom-normal) direction
    width: float = 0.0  # distance between the two wall planes (mm)
    length: float = 0.0  # channel extent along `axis` (mm)
    depth: float = 0.0  # bottom to opening plane (mm)
    open_ends: int = 0
    center: list[float] = Field(default_factory=list)  # bottom centre [x, y, z]
    centers: list[list[float]] = Field(default_factory=list)  # [[x, y]] plan-view


class MeasuredNotch(BaseModel):
    """A step cut: one ledge face meeting a single riser face along a concave
    edge, open on every other side (a notch, not a channel)."""

    id: str = ""  # stable selector id (nt1…) — assign_feature_ids
    axis: str = ""  # dominant axis of the ledge (opening) normal
    width: float = 0.0  # ledge extent along the riser normal (mm)
    depth: float = 0.0  # riser height along the ledge normal (mm)
    length: float = 0.0  # ledge extent along the shared edge (mm)
    center: list[float] = Field(default_factory=list)  # ledge centre [x, y, z]
    centers: list[list[float]] = Field(default_factory=list)  # [[x, y]] plan-view


class MeasuredFeatures(BaseModel):
    hole_groups: list[MeasuredHoleGroup] = Field(default_factory=list)
    total_holes: int = 0
    slots: int = 0  # elongated through-voids (aspect > ~1.8) — not round holes
    countersinks: int = 0  # cones coaxial to a hole
    counterbores: int = 0  # stepped coaxial hole cylinders
    rounds: list[float] = Field(default_factory=list)  # fillet radii (edge cylinders)
    chamfer_faces: int = 0  # 45° planes + cones too wide to be a countersink
    bosses: list[MeasuredBoss] = Field(default_factory=list)  # protruding cylinders
    approximate: bool = False  # faceted/mesh source — treat findings loosely
    # quiddity recognition enrichment (optional dep): additive pattern /
    # relationship evidence — never double-counts the analytic measurements
    recognition_ran: bool = False  # quiddity produced evidence for this shape
    hole_patterns: list[MeasuredPattern] = Field(default_factory=list)
    slot_patterns: list[MeasuredPattern] = Field(default_factory=list)
    bored_bosses: list[MeasuredBoredBoss] = Field(default_factory=list)
    chamfers: list[MeasuredChamfer] = Field(default_factory=list)
    fillets: list[MeasuredFillet] = Field(default_factory=list)
    # structural features from the face-adjacency graph (occwl/AAG pattern —
    # rule-based on convexity-labelled face adjacency, no learned weights)
    pockets: list[MeasuredPocket] = Field(default_factory=list)
    grooves: list[MeasuredGroove] = Field(default_factory=list)
    notches: list[MeasuredNotch] = Field(default_factory=list)


class FeatureFinding(BaseModel):
    feature: str = ""
    expected: str = ""
    actual: str = ""
    severity: Literal["major", "minor"] = "minor"
    fix_hint: str = ""


def _pt(v: Any) -> tuple[float, ...]:
    """JSON point/list -> a comparable tuple for deterministic id sorting."""
    return tuple(round(float(c), 3) for c in (v or []))


def assign_feature_ids(measured: MeasuredFeatures) -> MeasuredFeatures:
    """Stamp deterministic selector ids on every measured feature entry.

    Ids are assigned per kind bucket — ``h`` round hole groups, ``s``
    elongated slot voids, ``p`` recognised patterns, ``b`` bosses,
    ``bb`` bored bosses, ``c`` chamfers, ``f`` fillets — after sorting on
    (kind, axis, center, size), so the same geometry always yields the same
    ids across runs. verify_spec evidence, inspect-tool output and issue
    texts can cite them ("hole h1 on +x") instead of vague prose. Additive
    only: entries simply gain an ``id`` field.
    """
    holes = sorted(
        (g for g in measured.hole_groups if not g.elongated),
        key=lambda g: (g.axis, sorted(_pt(c) for c in g.centers), g.diameter),
    )
    for n, g in enumerate(holes, 1):
        g.id = f"h{n}"
    slots = sorted(
        (g for g in measured.hole_groups if g.elongated),
        key=lambda g: (g.axis, sorted(_pt(c) for c in g.centers), g.diameter),
    )
    for n, g in enumerate(slots, 1):
        g.id = f"s{n}"
    pats = sorted(
        [*measured.hole_patterns, *measured.slot_patterns],
        key=lambda p: (p.kind, p.axis, _pt(p.center), p.member_diameter, p.count),
    )
    for n, p in enumerate(pats, 1):
        p.id = f"p{n}"
    bosses = sorted(measured.bosses, key=lambda b: (b.axis, b.diameter, b.height))
    for n, b in enumerate(bosses, 1):
        b.id = f"b{n}"
    bored = sorted(
        measured.bored_bosses,
        key=lambda b: (b.axis, _pt(b.location), b.boss_diameter, b.bore_diameter),
    )
    for n, b in enumerate(bored, 1):
        b.id = f"bb{n}"
    chams = sorted(measured.chamfers, key=lambda c: (c.axis, _pt(c.at), c.leg1, c.leg2, c.angle))
    for n, c in enumerate(chams, 1):
        c.id = f"c{n}"
    fils = sorted(measured.fillets, key=lambda f: (f.axis, _pt(f.at), f.radius))
    for n, f_ in enumerate(fils, 1):
        f_.id = f"f{n}"
    pockets = sorted(measured.pockets, key=lambda p: (p.axis, _pt(p.center), p.depth, p.size))
    for n, p in enumerate(pockets, 1):
        p.id = f"pk{n}"
    grooves = sorted(
        measured.grooves,
        key=lambda g: (g.axis, g.opening, _pt(g.center), g.width, g.length),
    )
    for n, g in enumerate(grooves, 1):
        g.id = f"gr{n}"
    notches = sorted(measured.notches, key=lambda nt: (nt.axis, _pt(nt.center), nt.width, nt.depth))
    for n, nt in enumerate(notches, 1):
        nt.id = f"nt{n}"
    return measured


_AXIS_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")
_SAMPLE_DIRS = 8
_INSIDE_FRACTION_HOLE = 0.7  # radial samples inside material -> hole, else round
# u-span (radians): edge fillets wrap ~90° (π/2); a protruding cylinder wraps
# at least a half-circle — even when a union clips it (embedded tube ≈ π,
# free-standing boss ≈ 2π)
_BOSS_WRAP_MIN = 2.4
_THROUGH_FRACTION = 0.75  # cylinder axial extent vs part extent -> "through"
_GROUP_DIA_TOL = 0.3
_COAX_TOL = 0.5  # mm between axes to count as coaxial

# expected bore diameter (mm) per metric screw size — normal clearance fit.
# Fallback tables only; primary source is the bd_warehouse catalog
# (`catalog.nominal_hole_diameter`). Used as a plausibility band (+-20%).
_CLEARANCE_DIA = {
    3: 3.4,
    4: 4.5,
    5: 5.5,
    6: 6.6,
    8: 8.4,
    10: 10.5,
    12: 13.0,
    16: 17.0,
    20: 21.0,
}
_TAP_DIA = {3: 2.5, 4: 3.3, 5: 4.2, 6: 5.0, 8: 6.8, 10: 8.5, 12: 10.2}
_HOLE_TYPES = {
    "hole",
    "insert",
    "thread",
    "bearing_seat",
    # intake emits fine-grained hole types — they all must reach the check
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
}
_CSK_RE = re.compile(r"countersunk|csk|senk(?!.*zylinder)|flat.?head", re.I)
_CBORE_RE = re.compile(r"counterbore|cbore|zylindersenk|socket.?head", re.I)
# spec vocabulary for repeated-hole arrangements — a "hole" feature described
# as a bolt circle is also a pattern expectation, not just a count
_PATTERN_RE = re.compile(
    r"bolt.?circle|pitch.?circle|\bpcd\b|pattern|array|grid|loch(?:bild|kreis|reihe)",
    re.I,
)
# pattern expectations only apply to hole-ish features — a gear's tooth ring
# legitimately says "pitch circle Ø40" without describing a hole pattern
_HOLEISH_RE = re.compile(r"hole|bore|bohrung|loch", re.I)
# protruding cylinders (tube, boss, standoff …) — the feature's own axis is
# meaningful, unlike a fillet radius. 'dome' is deliberately absent: a dome
# is a curved cap, never measurable as a protruding cylinder.
_BOSS_TYPES = {"boss", "tube", "pin", "post", "stud", "standoff"}
_BOSS_RE = re.compile(r"\b(tube|pipe|boss|rohr|stutzen|buchse|nabe)\b", re.I)
# declared classes that can never be a protruding cylinder — a "boss" word in
# their prose ("rib joining the bearing boss") must not conjure a tube check
_NON_BOSS_TYPES = {
    "hole",
    "bore",
    "slot",
    "pocket",
    "cutout",
    "groove",
    "notch",
    "thread",
    "insert",
    "nut",
    "washer",
    "tap",
    "fillet",
    "round",
    "chamfer",
    "bevel",
    "rib",
    "gusset",
    "wall",
    "web",
    "strut",
    "brace",
    "frame",
    "plate",
    "block",
    "flange",
    "base",
    "bed",
    "rim",
    "lip",
    "shell",
    "bearing_seat",
    "gear",
    "text",
    "pattern",
    "knurl",
    "cut",
}
_TAP_RE = re.compile(r"\btap(ped)?\b|gewinde|thread", re.I)
_M_RE = re.compile(r"\bM\s*(\d+)", re.I)
_AXIS_LETTER_RE = re.compile(
    r"(?:along|entlang(?:\s+(?:der|die|den))?|längs|um)\s+([xyz])(?:[\s-]*(?:axis|achse))?"
    r"|\b([xyz])[\s-]*(?:axis|achse)\b",
    re.I,
)
_AXIS_HINT_RE = re.compile(
    r"(top|upper|deck|\bz\b|\+z|oben)|bottom|unter|front|back|left|right|side"
    r"|\bx\b|\by\b",
    re.I,
)


def _dominant_axis(v: list[float]) -> str:
    ax = max(range(3), key=lambda i: abs(v[i]))
    sign = "+" if v[ax] >= 0 else "-"
    return f"{sign}{'xyz'[ax]}"


def _vec3(v: Any) -> list[float]:
    out = []
    for attr in ("X", "Y", "Z"):
        c = getattr(v, attr)
        out.append(float(c() if callable(c) else c))
    return out


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(c * c for c in v))


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _unit(v: list[float]) -> list[float]:
    n = _norm(v) or 1.0
    return [c / n for c in v]


def _mesh_contains_fn(mesh: Any):
    """Vectorised point-containment over the analysis tessellation.

    trimesh `contains` (ray parity) costs ~50 µs/point; OCC `is_inside` costs
    ~200 ms/point on curved B-Reps — on helical thread solids the per-face
    sampling below used to grind the analyze band for minutes (observed live:
    77 cylinder faces x 8 samples x 0.19 s ≈ 120 s on a screw)."""
    import numpy as np

    def contains(pts: list[list[float]]):
        try:
            return np.asarray(mesh.contains(np.asarray(pts)), dtype=bool)
        except Exception:
            return None

    return contains


def _inside_fraction_pts(center: list[float], axis: list[float], r: float) -> list[list[float]]:
    """Radial sample points on a circle of radius ~1.35r around `center`."""
    a = axis
    ref = [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = [
        a[1] * ref[2] - a[2] * ref[1],
        a[2] * ref[0] - a[0] * ref[2],
        a[0] * ref[1] - a[1] * ref[0],
    ]
    un = _norm(u) or 1.0
    u = [c / un for c in u]
    w = [
        a[1] * u[2] - a[2] * u[1],
        a[2] * u[0] - a[0] * u[2],
        a[0] * u[1] - a[1] * u[0],
    ]
    rr = r * 1.35
    out = []
    for i in range(_SAMPLE_DIRS):
        ang = 2.0 * math.pi * i / _SAMPLE_DIRS
        out.append(
            [center[j] + rr * (math.cos(ang) * u[j] + math.sin(ang) * w[j]) for j in range(3)]
        )
    return out


def _inside_fraction(
    shape: Shape,
    center: list[float],
    axis: list[float],
    r: float,
    *,
    contains: Any = None,
    cancelled: Any = None,
) -> float:
    """Fraction of radial samples (on a circle of radius ~1.35r around the axis
    point) that lie inside material. Holes score ~1.0 (wall surrounded by
    material), edge fillets score ~0.25-0.5 (open to air on most sides)."""
    pts = _inside_fraction_pts(center, axis, r)
    if contains is not None:
        try:
            flags = contains(pts)
            if flags is not None:
                return float(sum(bool(f) for f in flags)) / _SAMPLE_DIRS
        except Exception:
            pass
    inside = 0
    for pt in pts:
        if cancelled is not None and cancelled():
            break
        try:
            if shape.is_inside(pt):
                inside += 1
        except Exception:
            continue
    return inside / _SAMPLE_DIRS


def _coaxial(
    a_center: list[float], a_dir: list[float], b_center: list[float], b_dir: list[float]
) -> bool:
    """True when two axes are (anti-)parallel and their lines nearly coincide."""
    dot = abs(sum(a_dir[i] * b_dir[i] for i in range(3)))
    if dot < 0.995:
        return False
    d = _sub(b_center, a_center)
    # perpendicular distance from b_center to line(a_center, a_dir)
    par = sum(d[i] * a_dir[i] for i in range(3))
    perp = [d[i] - par * a_dir[i] for i in range(3)]
    return _norm(perp) <= _COAX_TOL


def _scan_faces(
    shape: Shape,
) -> tuple[
    list[tuple[float, list[float], list[float], float, float]],
    list[tuple[list[float], list[float], float, float]],
    list[float],
    int,
]:
    """Walk all faces once; collect cylinders, cones, fillet tori and
    chamfer-like planes. Cone tuples carry the face-centre distance to the
    cone axis so a rim chamfer coaxial to a bore is not mistaken for a
    countersink; torus entries are the minor radius of partial wraps."""
    cyl_faces: list[tuple[float, list[float], list[float], float, float]] = []
    cones: list[tuple[list[float], list[float], float, float]] = []
    tori: list[float] = []
    chamfer_planes = 0
    # faceted mesh solid: every face is a PLANE triangle — a BRepAdaptor scan
    # over tens of thousands of faces costs minutes and can only find noise;
    # mesh-level probes (ray sampling) below carry the real signal
    if len(shape.faces()) > _FACET_FACE_CAP:
        return cyl_faces, cones, tori, chamfer_planes
    for f in shape.faces():
        try:
            gt = f.geom_type
            ad = BRepAdaptor_Surface(f.wrapped)
            if gt == GeomType.CYLINDER:
                cyl = ad.Cylinder()
                axis = _vec3(cyl.Axis().Direction())
                an = _norm(axis) or 1.0
                axis = [c / an for c in axis]
                # project the face centre onto the axis line: the sample point for
                # the inside-fraction test must sit ON the axis, not on the wall
                loc = _vec3(cyl.Axis().Location())
                fc = _vec3(f.center())
                d = _sub(fc, loc)
                par = sum(d[i] * axis[i] for i in range(3))
                on_axis = [loc[i] + par * axis[i] for i in range(3)]
                u1, u2, v1, v2 = BRepTools.UVBounds_s(f.wrapped)
                cyl_faces.append((float(cyl.Radius()), axis, on_axis, abs(v2 - v1), abs(u2 - u1)))
            elif gt == GeomType.CONE:
                cone = ad.Cone()
                ax = _vec3(cone.Axis().Direction())
                an = _norm(ax) or 1.0
                ax = [c / an for c in ax]
                loc = _vec3(cone.Axis().Location())
                fc = _vec3(f.center())
                d = _sub(fc, loc)
                par = sum(d[i] * ax[i] for i in range(3))
                on_axis = [loc[i] + par * ax[i] for i in range(3)]
                cones.append(
                    (
                        ax,
                        loc,
                        abs(math.degrees(cone.SemiAngle())),
                        _norm(_sub(fc, on_axis)),
                    )
                )
            elif gt == GeomType.TORUS:
                # edge fillet on a curved edge: minor circle wraps ~90-120°.
                # A full torus solid wraps 2π and is a shape, not a round.
                u1, u2, v1, v2 = BRepTools.UVBounds_s(f.wrapped)
                if abs(v2 - v1) < 2.8:
                    tori.append(float(ad.Torus().MinorRadius()))
            elif gt == GeomType.PLANE:
                n = _vec3(f.normal_at())
                comps = sorted(abs(c) for c in n)
                # chamfer plane: two normal components ~0.707, third ~0
                if comps[2] > 0.6 and comps[1] > 0.6 and comps[0] < 0.15:
                    chamfer_planes += 1
        except Exception:
            continue
    return cyl_faces, cones, tori, chamfer_planes


def _load_watertight_mesh(stl_path: Path):
    """Best-effort tessellation for tunnel probes. Watertight gives the
    cleanest signal; cracked exports still carry usable hit maps — the
    caller flags the result `approximate` instead of losing slot/depth
    detection entirely (observed: a degenerate 7 mm sliver under an
    l_bracket made the STL leak and hid a real slot)."""
    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    mesh.merge_vertices()
    if not mesh.is_watertight:
        mesh.fill_holes()
    return mesh if len(mesh.faces) else None


def _ray_tunnel(mesh, center: list[float], axis: int, r: float) -> bool:
    """True when a small cluster of rays along `axis` through `center` crosses
    the whole part without a hit — the bore is genuinely through even when the
    local wall thickness caps the measured vspan (organic dips)."""
    import numpy as np

    a1, a2 = (i for i in range(3) if i != axis)
    mn, _ = mesh.bounds
    offs = [(0.0, 0.0), (0.4 * r, 0.0), (-0.4 * r, 0.0), (0.0, 0.4 * r), (0.0, -0.4 * r)]
    origins = np.zeros((len(offs), 3))
    for k, (o1, o2) in enumerate(offs):
        origins[k, a1] = center[a1] + o1
        origins[k, a2] = center[a2] + o2
        origins[k, axis] = mn[axis] - 1.0
    dirs = np.zeros((len(offs), 3))
    dirs[:, axis] = 1.0
    hits = mesh.ray.intersects_any(origins, dirs)
    return int((~np.asarray(hits)).sum()) >= 3


def _bore_open_local(mesh, center: list[float], axis: int, r: float, vspan: float) -> bool:
    """True when the bore is open through its HOST material: walking outward
    from both bore ends, the axis stays void for as long as the ring around it
    is still inside material. A body further along the same line (strut,
    opposite wall) after an air gap does not count as a membrane."""
    import numpy as np

    a1, a2 = (i for i in range(3) if i != axis)
    step = max(0.25, min(0.5, r / 2))
    s0 = center[axis] - vspan / 2
    s1 = center[axis] + vspan / 2
    ring_off = [(1.6 * r, 0.0), (-1.6 * r, 0.0), (0.0, 1.6 * r), (0.0, -1.6 * r)]
    for direction in (1, -1):
        start = s1 if direction > 0 else s0
        limit = mesh.bounds[1][axis] + step if direction > 0 else mesh.bounds[0][axis] - step
        ts: list[float] = []
        t = start
        while (t <= limit) if direction > 0 else (t >= limit):
            ts.append(t)
            t += direction * step
        if not ts:
            continue
        # one bundled contains() per direction: row i = [core, ring x4]
        pts = np.zeros((len(ts), 5, 3))
        pts[:, :, a1] = center[a1]
        pts[:, :, a2] = center[a2]
        pts[:, :, axis] = np.asarray(ts)[:, None]
        for k, (o1, o2) in enumerate(ring_off):
            pts[:, k + 1, a1] += o1
            pts[:, k + 1, a2] += o2
        inside = np.asarray(mesh.contains(pts.reshape(-1, 3))).reshape(len(ts), 5)
        for row in inside:
            ring_in = bool(row[1:].any())
            if not ring_in:
                break  # left the host material — this side is open to air
            if bool(row[0]):
                return False  # core sealed while still inside the host — membrane
        # ran past the part bound without finding material: open
    return True


def _bore_fit_diameter(
    mesh: Any, axis: int, a1: int, a2: int, cu: float, cv: float, dia_est: float
) -> float:
    """Fitted bore diameter around a tunnel centre.

    The zero-hit column underreports bores whose void column is shadowed by
    a tighter interior cavity or whose mouth blends into a wider bay (the
    centroid then drifts off the bore axis). The binding diameter for a
    fastener is the bore rim: cast one dense raster of rays along the axis
    per side — a ray is rim-blocked when its FIRST surface hit lands on the
    face band (entry mouth level), while zero-hit and deep-entry rays pass
    through the bore. For each candidate centre the largest all-clear ring
    radius is the inscribed bore radius; the bore's true centre maximises
    it (max over candidates) and the mouth side always finds the true rim
    (min over sides).
    """
    import numpy as np

    mn, mx = mesh.bounds
    r_max = dia_est / 2.0 + 4.0
    step = 0.3
    ext = r_max + 2.0
    g1 = np.arange(cu - ext, cu + ext + step, step)
    g2 = np.arange(cv - ext, cv + ext + step, step)
    if not len(g1) or not len(g2):
        return 0.0
    uu, vv = np.meshgrid(g1, g2, indexing="ij")
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    radii = np.arange(0.7, r_max, step)
    side_best: list[float] = []
    for side in (0, 1):  # 0: rays +axis from mn side; 1: rays -axis from mx side
        origins = np.zeros((uu.size, 3))
        origins[:, a1] = uu.ravel()
        origins[:, a2] = vv.ravel()
        origins[:, axis] = (mn[axis] - 1.0) if side == 0 else (mx[axis] + 1.0)
        d = np.zeros(3)
        d[axis] = 1.0 if side == 0 else -1.0
        locs, ray_ids, _ = mesh.ray.intersects_location(origins, np.tile(d, (len(origins), 1)))
        first = np.full(uu.size, np.nan)
        for rid in np.unique(ray_ids):
            hit_ax = locs[ray_ids == rid][:, axis]
            first[rid] = hit_ax.min() if side == 0 else hit_ax.max()
        first = first.reshape(uu.shape)
        finite = first[np.isfinite(first)]
        if len(finite) < 6:
            continue  # no face on this side — nothing to fit against
        face = float(np.percentile(finite, 10) if side == 0 else np.percentile(finite, 90))
        # the face band only makes sense near the part's outer surface — on
        # an open side (bay/cutaway) first hits land deep in the interior
        # and would fake a face: recessed real faces are < ~20 mm inside
        bbox_face = float(mn[axis]) if side == 0 else float(mx[axis])
        if abs(face - bbox_face) > min(20.0, 0.3 * float(mx[axis] - mn[axis])):
            continue
        blocked = np.isfinite(first) & (
            (first <= face + 0.8) if side == 0 else (first >= face - 0.8)
        )
        cand_best = 0.0
        for du in np.arange(-2.0, 2.0 + step, step):
            for dv in np.arange(-2.0, 2.0 + step, step):
                for r in radii:
                    iu = np.clip(
                        np.round((cu + du + r * np.cos(ang) - g1[0]) / step).astype(int),
                        0,
                        uu.shape[0] - 1,
                    )
                    iv = np.clip(
                        np.round((cv + dv + r * np.sin(ang) - g2[0]) / step).astype(int),
                        0,
                        uu.shape[1] - 1,
                    )
                    # rim = radius where a solid fraction of ring directions
                    # blocks; a single grazing ray (thread groove lip, seam)
                    # must not cut the fit short, and a mouth blending into
                    # a bay leaves part of the ring unconstrained
                    if float(blocked[iu, iv].mean()) >= 0.3:
                        break
                    cand_best = max(cand_best, float(r))
        side_best.append(cand_best)
    r_fit = min(side_best, default=0.0)
    return float(2.0 * r_fit) if r_fit > 0 else 0.0


def _mesh_through_holes(stl_path: Path, mesh=None) -> tuple[list[MeasuredHoleGroup], int, int]:
    """Detect through-holes on a faceted solid via ray casting.

    A ray that crosses the whole part without a single hit passes through a
    tunnel. Clustered zero-hit grid cells give count, diameter and axis;
    a ring of probe rays around each tunnel detects a flared (countersunk)
    mouth opening toward +axis. Used as fallback when the B-Rep shows no
    analytic cylinders — mesh-derived solids triangulate every cut.
    """
    import numpy as np
    from scipy import ndimage

    if mesh is None:
        mesh = _load_watertight_mesh(stl_path)
    if mesh is None:
        return [], 0, 0
    mn, mx = mesh.bounds
    step = 0.7  # mm grid pitch — resolves Ø~1.6mm tunnels and larger
    groups: list[MeasuredHoleGroup] = []
    csk_hits = 0
    slot_count = 0

    for axis in range(3):
        a1, a2 = (i for i in range(3) if i != axis)
        g1 = np.arange(mn[a1] + step / 2, mx[a1], step)
        g2 = np.arange(mn[a2] + step / 2, mx[a2], step)
        if not len(g1) or not len(g2):
            continue
        uu, vv = np.meshgrid(g1, g2, indexing="ij")
        origins = np.zeros((uu.size, 3))
        origins[:, a1] = uu.ravel()
        origins[:, a2] = vv.ravel()
        origins[:, axis] = mn[axis] - 1.0
        direction = np.zeros(3)
        direction[axis] = 1.0
        directions = np.tile(direction, (len(origins), 1))
        hits_any = mesh.ray.intersects_any(origins, directions)
        hits = hits_any.reshape(uu.shape)
        # a tunnel is a zero-hit ray INSIDE the part's projected footprint:
        # rays that miss the silhouette entirely are not holes. The footprint
        # is the hit mask with its interior gaps filled — enclosed voids are
        # the tunnels, open silhouette bays stay excluded.
        footprint = ndimage.binary_fill_holes(hits)
        free = (~hits) & footprint
        if not free.any():
            continue

        labels, n = ndimage.label(free, structure=np.ones((3, 3)))
        for lab in range(1, n + 1):
            cells = np.argwhere(labels == lab)
            area = float(len(cells)) * step * step
            dia = 2.0 * math.sqrt(area / math.pi)
            if dia < 1.6:  # tessellation gaps can fake tiny tunnels
                continue
            # elongated voids are slots, not round holes — report them
            # separately so spec checks don't match a 20x6 slot against a
            # Ø16 hole expectation (aspect of the component's bounding box)
            span_u = (cells[:, 0].max() - cells[:, 0].min() + 1) * step
            span_v = (cells[:, 1].max() - cells[:, 1].min() + 1) * step
            aspect = max(span_u, span_v) / max(min(span_u, span_v), step)
            is_slot = aspect > 1.8
            cu = float(cells[:, 0].mean())
            cv = float(cells[:, 1].mean())
            center = np.zeros(3)
            center[a1] = mn[a1] + step / 2 + cu * step
            center[a2] = mn[a2] + step / 2 + cv * step
            center[axis] = (mn[axis] + mx[axis]) / 2
            if not is_slot:
                # the zero-hit column can be narrower than the actual bore —
                # a mouth blending into an exterior bay or a tighter interior
                # cavity shrinks the mask. The tunnel's narrowest section is
                # the binding diameter and can only lift an underreport.
                sec = _bore_fit_diameter(
                    mesh, axis, a1, a2, float(center[a1]), float(center[a2]), dia
                )
                if sec > dia:
                    dia = sec
            if is_slot:
                slot_count += 1
            center_xyz: list[float | None] = [None, None, None]
            center_xyz[a1] = round(float(center[a1]), 2)
            center_xyz[a2] = round(float(center[a2]), 2)
            # the elongation direction is the in-plane axis (a1/a2 — the
            # two non-tunnel axes) with the larger span; sign-less on
            # purpose — a symmetric span has no orientation
            elong_axis = f"+{'xyz'[a1 if span_u >= span_v else a2]}" if is_slot else ""
            groups.append(
                MeasuredHoleGroup(
                    diameter=round(dia, 3),
                    count=1,
                    axis=f"+{'xyz'[axis]}",
                    depth_class="through",
                    centers=[[round(float(center[0]), 2), round(float(center[1]), 2)]],
                    centers_xyz=[center_xyz],
                    elongated=is_slot,
                    through_axis=f"+{'xyz'[axis]}",
                    elong_axis=elong_axis,
                    width_mm=round(min(span_u, span_v), 2),
                    length_mm=round(max(span_u, span_v), 2),
                )
            )

            # countersink probe: ring rays just outside the bore shot downward
            # from above the +axis face — when they land below the top surface
            # a flared mouth (cone) is present
            top_z = mx[axis]
            probe_r = dia * 0.75
            ring = np.zeros((8, 3))
            ang = np.linspace(0, 2 * math.pi, 8, endpoint=False)
            ring[:, a1] = center[a1] + probe_r * np.cos(ang)
            ring[:, a2] = center[a2] + probe_r * np.sin(ang)
            ring[:, axis] = top_z + 1.0
            down = np.zeros(3)
            down[axis] = -1.0
            locs, ray_ids, _ = mesh.ray.intersects_location(ring, np.tile(down, (8, 1)))
            # first surface each downward ray meets = its highest hit
            first_hit = np.array([locs[ray_ids == r][:, axis].max() for r in np.unique(ray_ids)])
            if len(first_hit) >= 6 and int((first_hit < top_z - 0.3).sum()) >= 5:
                csk_hits += 1
    return groups, csk_hits, slot_count


# ---------------------------------------------------------------------------
# Face-adjacency-graph (AAG) structural features — occwl pattern implemented
# against OCP/build123d directly (license-clean, rule-based, no learned
# weights). Faces are graph nodes; every interior edge shared by exactly two
# faces is a graph edge labelled convex/concave/tangent. Pockets, grooves and
# notches are then recognised by simple structural rules on that graph.
# ---------------------------------------------------------------------------

_AAG_MAX_FACES = 600  # faceted/mesh solids triangulate every face — skip
_AAG_PERP_DOT = 0.35  # |n1·n2| below this counts the faces as perpendicular
_AAG_ANTI_DOT = -0.9  # n1·n2 below this counts the faces as opposing walls
_AAG_PARALLEL_DOT = 0.95  # opening-face normal ≈ floor normal
_AAG_TANGENT_BISECTOR = 0.45  # |a1+a2| below this is a tangent/smooth join
_AAG_OPEN_END_DOT = 0.6  # bottom-face interior direction along ±channel dir
_AAG_MIN_DEPTH = 0.2  # mm — shallower recesses are texture, not pockets


def _project_uv(face: Any, p: list[float]) -> tuple[float, float] | None:
    """Parameters of the closest surface point to ``p`` on ``face``."""
    try:
        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*p), BRep_Tool.Surface_s(face.wrapped))
        if proj.NbPoints() < 1:
            return None
        return proj.LowerDistanceParameters()
    except Exception:
        return None


def _normal_at_uv(face: Any, u: float, v: float) -> list[float] | None:
    """Outward surface normal of ``face`` at (u, v), orientation-aware."""
    try:
        surf = BRepAdaptor_Surface(face.wrapped)
        pnt, d1u, d1v = gp_Pnt(), gp_Vec(), gp_Vec()
        surf.D1(u, v, pnt, d1u, d1v)
        n = _vec3(d1u.Crossed(d1v))
        if face.wrapped.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
            n = [-c for c in n]
        nn = _norm(n)
        return [c / nn for c in n] if nn > 1e-9 else None
    except Exception:
        return None


def _point_on_face(face: Any, u: float, v: float) -> bool:
    try:
        cls = BRepClass_FaceClassifier(face.wrapped, gp_Pnt2d(u, v), 1e-6)
        return cls.State() in (TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON)
    except Exception:
        return False


def _face_interior_dir(face: Any, pe: list[float], tang: list[float]) -> list[float] | None:
    """Direction perpendicular to the edge tangent ``tang`` at point ``pe``,
    lying in the face's tangent plane and pointing INTO the face interior.

    Found without edge-orientation bookkeeping: the two candidates ±(tang x n)
    are offset a small step, projected back onto the surface and classified
    against the face's 2D domain — the one that lands inside wins. Using the
    face centroid instead fails on annular/U-shaped faces (its projection can
    point across a void)."""
    uv = _project_uv(face, pe)
    if uv is None:
        return None
    n = _normal_at_uv(face, *uv)
    if n is None:
        return None
    e = [
        tang[1] * n[2] - tang[2] * n[1],
        tang[2] * n[0] - tang[0] * n[2],
        tang[0] * n[1] - tang[1] * n[0],
    ]
    en = _norm(e)
    if en < 1e-9:
        return None
    e = [c / en for c in e]
    eps = max(0.05, min(0.3, 0.05 * math.sqrt(max(float(face.area), 1e-3))))
    for d in (e, [-c for c in e]):
        q = [pe[i] + eps * d[i] for i in range(3)]
        uv2 = _project_uv(face, q)
        if uv2 is not None and _point_on_face(face, *uv2):
            return d
    return None


def _edge_convexity(
    shape: Shape, edge: Any, f1: Any, f2: Any, *, contains: Any = None
) -> tuple[str, dict[str, Any]]:
    """Convexity of the adjacency edge between two faces.

    Method (documented, simplest reliable): compute each face's interior
    direction a1/a2 at the edge midpoint (perpendicular to the edge, into the
    face). If a1 and a2 are antiparallel the join is tangent-continuous —
    "tangent". Otherwise probe the point halfway along the interior bisector:
    ``shape.is_inside(edge_mid + δ·(a1+a2)/|a1+a2|)`` — inside material means
    the solid occupies the wedge between the faces (convex, e.g. a box
    corner); outside means the wedge is air (concave, e.g. a pocket corner).
    ``δ`` scales with the smaller face area so the probe stays local."""
    try:
        pe = _vec3(edge.position_at(0.5))
        t = _vec3(edge.tangent_at(0.5))
        tn = _norm(t)
        if tn < 1e-9:
            return "unknown", {}
        tang = [c / tn for c in t]
    except Exception:
        return "unknown", {}
    a1 = _face_interior_dir(f1, pe, tang)
    a2 = _face_interior_dir(f2, pe, tang)
    if a1 is None or a2 is None:
        return "unknown", {}
    s = [a1[i] + a2[i] for i in range(3)]
    sn = _norm(s)
    if sn < _AAG_TANGENT_BISECTOR:
        return "tangent", {"a1": a1, "a2": a2}
    bdir = [c / sn for c in s]
    delta = max(0.15, min(1.0, 0.15 * math.sqrt(min(float(f1.area), float(f2.area)))))
    q = [pe[i] + delta * bdir[i] for i in range(3)]
    try:
        if contains is not None:
            flags = contains([q])
            if flags is None:
                return "unknown", {}
            inside = bool(flags[0])
        else:
            inside = shape.is_inside(q)
    except Exception:
        return "unknown", {}
    return ("convex" if inside else "concave"), {"a1": a1, "a2": a2}


def _build_face_adjacency(shape: Shape, *, contains: Any = None):
    """Face-adjacency graph: nodes = faces (surface type, area, normal,
    centre); edges = interior brep edges shared by exactly 2 faces, labelled
    by convexity. OCP's TopExp_MapShapesAndAncestors map type is not bound in
    this OCP version, so edge→faces adjacency is rebuilt by bucketing the
    per-face edges on their shared TShape hash."""
    import networkx as nx  # lazy — the executor image may lag pyproject

    faces = shape.faces()
    graph = nx.Graph()
    edge_faces: dict[int, list[Any]] = {}
    for fi, f in enumerate(faces):
        try:
            fedges = f.edges()
        except Exception:
            fedges = []
        for e in fedges:
            bucket = edge_faces.setdefault(hash(e.wrapped), [e, []])
            if fi not in bucket[1]:
                bucket[1].append(fi)
        try:
            geom = str(f.geom_type)
        except Exception:
            geom = "OTHER"
        normal = None
        try:
            if f.geom_type == GeomType.PLANE:
                normal = _vec3(f.normal_at())
                nn = _norm(normal)
                normal = [c / nn for c in normal] if nn > 1e-9 else None
        except Exception:
            normal = None
        try:
            center = _vec3(f.center())
        except Exception:
            center = [0.0, 0.0, 0.0]
        graph.add_node(
            fi,
            geom=geom,
            area=float(getattr(f, "area", 0.0) or 0.0),
            normal=normal,
            center=center,
            face=f,
        )
    for edge, fidx in (b for b in edge_faces.values() if len(b[1]) == 2):
        i, j = fidx
        conv, adir = _edge_convexity(shape, edge, faces[i], faces[j], contains=contains)
        adir["lo"] = i  # a1 always belongs to the lower of the two indices
        graph.add_edge(i, j, conv=conv, adir=adir)
    return graph


def _concave_neighbors(graph: Any, i: int) -> list[int]:
    return [j for j in graph.neighbors(i) if graph.edges[i, j]["conv"] == "concave"]


def _subgraph_bbox(graph: Any, nodes: set[int] | frozenset[int]) -> tuple[list, list]:
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for k in nodes:
        try:
            bb = graph.nodes[k]["face"].bounding_box()
        except Exception:
            continue
        for a, (mn, mx) in enumerate(
            ((bb.min.X, bb.max.X), (bb.min.Y, bb.max.Y), (bb.min.Z, bb.max.Z))
        ):
            lo[a] = min(lo[a], float(mn))
            hi[a] = max(hi[a], float(mx))
    return lo, hi


def _opening_coord(
    graph: Any, faces: set[int] | frozenset[int], wall: int, normal: list[float]
) -> float | None:
    """Coordinate along ``normal`` of the outer face a wall opens through —
    the convex neighbour outside the subgraph with normal ≈ ``normal``."""
    best = None
    for o in graph.neighbors(wall):
        if o in faces or graph.edges[wall, o]["conv"] != "convex":
            continue
        on = graph.nodes[o]["normal"]
        if on is None or abs(sum(on[i] * normal[i] for i in range(3))) <= _AAG_PARALLEL_DOT:
            continue
        coord = sum(graph.nodes[o]["center"][i] * normal[i] for i in range(3))
        best = coord if best is None else max(best, coord)
    return best


def _detect_structural_features(  # noqa: C901 — graph pattern matching
    shape: Shape,
    *,
    contains: Any = None,
) -> tuple[list[MeasuredPocket], list[MeasuredGroove], list[MeasuredNotch]]:
    """Recognise pockets, grooves and notches on the face-adjacency graph.

    Rules (deliberately conservative — nothing emitted when the structure is
    ambiguous):
    - POCKET: a planar floor whose EVERY adjacency edge is concave; flood-fill
      through concave edges gives the recess subgraph (floor + walls), which
      must be bounded by convex edges reaching an outer face parallel to the
      floor (the opening plane). Requires ≥1 planar wall — a pure cylindrical
      wall is a blind hole, already covered by the hole scan.
    - GROOVE: a planar bottom joined by concave edges to exactly one pair of
      antiparallel planar walls (⊥ the bottom); any further concave neighbour
      must be an end cap (⊥ both). Every remaining edge of the bottom must be
      an open end (convex/tangent, interior direction along the channel).
    - NOTCH (step cut): exactly one concave planar wall ⊥ a ledge face, all
      other edges open.
    """
    pockets: list[MeasuredPocket] = []
    grooves: list[MeasuredGroove] = []
    notches: list[MeasuredNotch] = []
    faces = shape.faces()
    if not faces or len(faces) > _AAG_MAX_FACES:
        return pockets, grooves, notches
    graph = _build_face_adjacency(shape, contains=contains)
    nodes = graph.nodes

    def is_planar(i: int) -> bool:
        return nodes[i]["geom"] == "GeomType.PLANE" and nodes[i]["normal"] is not None

    def d(a: list[float], b: list[float]) -> float:
        return sum(a[i] * b[i] for i in range(3))

    claimed: set[int] = set()
    seeds = [
        i
        for i in graph.nodes
        if is_planar(i)
        and graph.degree[i] > 0
        and all(graph.edges[i, j]["conv"] == "concave" for j in graph.neighbors(i))
    ]
    for seed in seeds:
        if seed in claimed:
            continue
        sub: set[int] = set()
        frontier = [seed]
        while frontier:
            k = frontier.pop()
            if k in sub:
                continue
            sub.add(k)
            frontier += [j for j in _concave_neighbors(graph, k) if j not in sub]
        nb = nodes[seed]["normal"]
        # opening plane: an outer face parallel to the floor that walls reach
        # through a convex edge. A STEPPED recess finds an intermediate ledge
        # first — keep climbing: extend the subgraph through each such ledge
        # face (and the walls it feeds via concave edges) until the opening
        # stops rising, so a two-level pocket reports its full depth and its
        # shallow tier cannot double-report as a groove.
        opening: float | None = None
        while True:
            merged = False
            candidates: set[int] = set()
            for w in sub - {seed}:
                for o in graph.neighbors(w):
                    if o in sub or graph.edges[w, o]["conv"] != "convex":
                        continue
                    on = nodes[o]["normal"]
                    if on is not None and abs(d(on, nb)) > _AAG_PARALLEL_DOT:
                        candidates.add(o)
            for o in candidates:
                coord_o = d(nodes[o]["center"], nb)
                if opening is None or coord_o > opening:
                    opening = coord_o
                # try growing through this opening face into a higher tier
                ext = {o}
                ext_frontier = [o]
                while ext_frontier:
                    k = ext_frontier.pop()
                    for j in _concave_neighbors(graph, k):
                        if j not in sub and j not in ext:
                            ext.add(j)
                            ext_frontier.append(j)
                higher = [
                    c
                    for w in ext
                    for c in [_opening_coord(graph, sub | ext, w, nb)]
                    if c is not None
                ]
                if higher and max(higher) > coord_o + 1e-6:
                    sub |= ext
                    opening = max(higher)
                    merged = True
                    break
            if not merged:
                break
        walls = sub - {seed}
        depth = (opening if opening is not None else 0.0) - d(nodes[seed]["center"], nb)
        if depth < _AAG_MIN_DEPTH:
            continue
        if not any(is_planar(w) for w in walls):
            continue  # pure cylindrical/conical wall → blind hole, not a pocket
        claimed |= sub
        lo, hi = _subgraph_bbox(graph, sub)
        ax = _dominant_axis(nb)
        axi = "xyz".index(ax[-1])
        dims = sorted((round(hi[a] - lo[a], 3) for a in range(3) if a != axi), reverse=True)
        center = [round(c, 3) for c in nodes[seed]["center"]]
        pockets.append(
            MeasuredPocket(
                axis=ax,
                depth=round(depth, 3),
                size=dims,
                center=center,
                centers=[[center[0], center[1]]],
            )
        )

    # groove/notch bottoms: a planar face whose concave neighbours are planar
    # walls perpendicular to it; all other edges must be open (convex/tangent)
    cand_g: list[dict[str, Any]] = []
    cand_n: list[dict[str, Any]] = []
    seen: dict[frozenset[int], int] = {}
    for b in sorted(graph.nodes):
        if b in claimed or not is_planar(b):
            continue
        nb = nodes[b]["normal"]
        bedges = list(graph.edges(b, data=True))
        if not bedges:
            continue
        walls: list[int] = []
        ambiguous = False
        for _, j, data in bedges:
            if data["conv"] != "concave":
                continue
            if is_planar(j) and abs(d(nodes[j]["normal"], nb)) < _AAG_PERP_DOT:
                walls.append(j)
            else:
                ambiguous = True  # concave join to non-planar/non-⊥ face
        if ambiguous or not walls:
            continue
        channel: list[float] | None = None
        pair: tuple[int, int] | None = None
        kind: str | None = None
        if len(walls) == 1:
            kind = "notch"
        elif (
            len(walls) == 2
            and d(nodes[walls[0]]["normal"], nodes[walls[1]]["normal"]) < _AAG_ANTI_DOT
        ):
            kind = "groove"
            pair = (walls[0], walls[1])
        elif len(walls) == 3:
            antip = [
                (k, m)
                for k in range(3)
                for m in range(k + 1, 3)
                if d(nodes[walls[k]]["normal"], nodes[walls[m]]["normal"]) < _AAG_ANTI_DOT
            ]
            if len(antip) == 1:
                k, m = antip[0]
                other = ({0, 1, 2} - {k, m}).pop()
                channel = _unit(_cross_v(nodes[walls[k]]["normal"], nb))
                if abs(d(nodes[walls[other]]["normal"], channel)) >= 0.5:
                    kind = "groove"
                    pair = (walls[k], walls[m])
        if kind is None:
            continue
        if kind == "groove" and channel is None:
            channel = _unit(_cross_v(nodes[pair[0]]["normal"], nb))
        wallset = set(walls)
        end_signs: set[int] = set()
        ok = True
        for _, j, data in bedges:
            if j in wallset:
                continue
            if data["conv"] not in ("convex", "tangent"):
                ok = False  # recess edge that is not a wall — not a clean cut
                break
            if kind == "groove":
                slot = "a1" if data["adir"].get("lo") == b else "a2"
                a_dir = data["adir"].get(slot)
                if a_dir is None or abs(d(a_dir, channel)) < _AAG_OPEN_END_DOT:
                    ok = False  # boundary edge is not along the channel
                    break
                end_signs.add(1 if d(a_dir, channel) > 0 else -1)
        if not ok:
            continue
        key = frozenset({b, *wallset})
        cand = {
            "bottom": b,
            "key": key,
            "walls": walls,
            "pair": pair,
            "channel": channel,
            "end_signs": end_signs,
        }
        pool = cand_g if kind == "groove" else cand_n
        if key in seen:
            # the same recess recognised from a different face (a U-channel
            # reads as a groove from its floor AND from its end wall) — keep
            # the reading whose "bottom" is the largest face (the floor)
            prev = pool[seen[key]]
            if nodes[b]["area"] > nodes[prev["bottom"]]["area"]:
                pool[seen[key]] = cand
            continue
        seen[key] = len(pool)
        pool.append(cand)

    groove_faces: set[int] = set()
    for c in cand_g:
        b = c["bottom"]
        nb = nodes[b]["normal"]
        w1, w2 = c["pair"]
        n1 = nodes[w1]["normal"]
        width = abs(d(_sub(nodes[w1]["center"], nodes[w2]["center"]), n1))
        lo, hi = _subgraph_bbox(graph, c["key"])
        cax = max(range(3), key=lambda i: abs(c["channel"][i]))
        openings = [
            v for w in c["walls"] if (v := _opening_coord(graph, c["key"], w, nb)) is not None
        ]
        if not openings:
            continue  # walls reach no outer face ∥ the bottom — not an open channel
        depth = max(openings) - d(nodes[b]["center"], nb)
        center = [round(v, 3) for v in nodes[b]["center"]]
        grooves.append(
            MeasuredGroove(
                axis=_dominant_axis(c["channel"]),
                opening=_dominant_axis(nb),
                width=round(width, 3),
                length=round(hi[cax] - lo[cax], 3),
                depth=round(depth, 3),
                open_ends=len(c["end_signs"]),
                center=center,
                centers=[[center[0], center[1]]],
            )
        )
        groove_faces |= set(c["key"])

    covered = claimed | groove_faces
    for c in cand_n:
        # a groove wall on its own reads as a single-wall notch — drop
        # notches overlapping an already-claimed recess
        if set(c["key"]) & covered:
            continue
        b, w = c["bottom"], c["walls"][0]
        nb, nw = nodes[b]["normal"], nodes[w]["normal"]
        lo, hi = _subgraph_bbox(graph, c["key"])
        wax = max(range(3), key=lambda i: abs(nw[i]))
        eax = max(range(3), key=lambda i: abs(_cross_v(nb, nw)[i]))
        opening = _opening_coord(graph, c["key"], w, nb)
        if opening is None:
            continue  # riser reaches no higher face ∥ the ledge — not a step
        depth = opening - d(nodes[b]["center"], nb)
        center = [round(v, 3) for v in nodes[b]["center"]]
        notches.append(
            MeasuredNotch(
                axis=_dominant_axis(nb),
                width=round(hi[wax] - lo[wax], 3),
                depth=round(depth, 3),
                length=round(hi[eax] - lo[eax], 3),
                center=center,
                centers=[[center[0], center[1]]],
            )
        )
    return pockets, grooves, notches


def _cross_v(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _extract_structural(shape: Shape, out: MeasuredFeatures, *, contains: Any = None) -> None:
    """Run the AAG pass best-effort: any OCP/networkx failure simply leaves
    the structural fields empty (the analytic measurements already stand)."""
    try:
        pockets, grooves, notches = _detect_structural_features(shape, contains=contains)
    except Exception:
        return
    out.pockets = pockets
    out.grooves = grooves
    out.notches = notches


def extract_features(  # noqa: C901 — sequential measurement passes
    shape: Shape,
    stl_path: Path | None = None,
    *,
    mesh: Any | None = None,
    cancelled: Any = None,
    recognition: bool = True,
) -> MeasuredFeatures:
    """Measure hole groups, countersinks, counterbores, fillet rounds and
    chamfer-like faces from a solid. Faceted mesh solids degrade gracefully —
    cylinders only survive if the mesh kept them near-circular.

    `mesh` is the pipeline's analysis tessellation (one load, shared by every
    ray-parity consumer); `cancelled` aborts the OCC fallbacks promptly —
    the executor's timeout cannot kill a worker thread, so the expensive
    passes must yield cooperatively."""
    out = MeasuredFeatures(approximate=False)
    bb = shape.bounding_box()
    extent = [
        float(bb.max.X - bb.min.X),
        float(bb.max.Y - bb.min.Y),
        float(bb.max.Z - bb.min.Z),
    ]

    if mesh is None and stl_path is not None:
        try:
            mesh = _load_watertight_mesh(stl_path)
        except Exception:
            mesh = None
    if mesh is not None and not mesh.is_watertight:
        out.approximate = True
    contains = _mesh_contains_fn(mesh) if mesh is not None else None

    cyl_faces, cones, tori, chamfer_planes = _scan_faces(shape)
    if len(shape.faces()) > _FACET_FACE_CAP:
        out.approximate = True

    # classify cylinders: hole (axis void surrounded by material) vs round
    # (edge fillet) vs boss (protruding full-wrap cylinder — a tube/handle).
    # Fillets only wrap ~90° of the circle; a boss wraps the full 2π.
    holes: list[tuple[float, list[float], list[float], float]] = []
    for r, axis, center, vspan, uspan in cyl_faces:
        if cancelled is not None and cancelled():
            return out
        try:
            frac = _inside_fraction(shape, center, axis, r, contains=contains, cancelled=cancelled)
        except Exception:
            frac = 0.0
        if frac >= _INSIDE_FRACTION_HOLE:
            holes.append((r, axis, center, vspan))
        elif uspan >= _BOSS_WRAP_MIN:
            out.bosses.append(
                MeasuredBoss(
                    axis=_dominant_axis(axis), diameter=round(2.0 * r, 3), height=round(vspan, 3)
                )
            )
        else:
            out.rounds.append(round(r, 3))
    for r in tori:
        out.rounds.append(round(r, 3))

    # merge coaxial same-diameter segments: a countersink/counterbore step
    # splits one physical bore into several cylinder faces (bore below the
    # cone + remnant above) — they are ONE hole, not two.
    merged: list[list[Any]] = []
    for r, axis, center, vspan in holes:
        for m in merged:
            if abs(m[0] - r) <= _GROUP_DIA_TOL and _coaxial(m[2], m[1], center, axis):
                m[3] += vspan
                break
        else:
            merged.append([r, axis, center, vspan])
    holes = [(m[0], m[1], m[2], m[3]) for m in merged]

    # cones coaxial to a hole and sized like a bore flare -> countersink;
    # a 45° cone far outside the bore is a rim chamfer (same axis, big radius)
    csk = 0
    cbore = 0
    chamfer_cones = 0
    for ax, ccenter, semi, face_r in cones:
        hole_r = next(
            (r for r, h_axis, h_center, _ in holes if _coaxial(h_center, h_axis, ccenter, ax)),
            None,
        )
        if hole_r is not None and face_r <= hole_r * 2.2 + 2.0:
            csk += 1
        elif abs(semi - 45.0) < 10.0:
            chamfer_cones += 1

    # stepped coaxial hole cylinders (different radii, same axis) -> counterbore
    for i, (r1, ax1, c1, _) in enumerate(holes):
        if any(
            _coaxial(c1, ax1, c2, ax2) and abs(r1 - r2) > _GROUP_DIA_TOL
            for r2, ax2, c2, _ in holes[i + 1 :]
        ):
            cbore += 1

    # group holes by (dominant axis, diameter cluster)
    groups: list[MeasuredHoleGroup] = []
    for r, axis, center, vspan in holes:
        dom = _dominant_axis(axis)
        ax_idx = "xyz".index(dom[-1])
        depth = "through" if vspan >= _THROUGH_FRACTION * extent[ax_idx] else "blind"
        # vspan is measured on the CUT face's parameter range — it can claim
        # "through" even when a membrane seals the bore, and it is capped by
        # LOCAL thickness so a real through bore in a dipped region always
        # looks "blind" against the global extent. When the tessellation is
        # available the local bore-open walk decides instead: it only asks
        # whether the bore exits its HOST material, so a body further along
        # the same line (strut, opposite wall) behind an air gap no longer
        # counts as a blockage (observed: bracket arm bore marked "blocked"
        # because a strut crossed the axis below the plate).
        if mesh is not None:
            depth = "through" if _bore_open_local(mesh, center, ax_idx, r, vspan) else "blind"
        dia = 2.0 * r
        cxy = [round(float(center[0]), 2), round(float(center[1]), 2)]
        aligned = abs(abs(axis[ax_idx]) - 1.0) <= 0.01
        center_xyz: list[float | None] = [
            None if i == ax_idx or not aligned else round(float(center[i]), 2) for i in range(3)
        ]
        for g in groups:
            if g.axis == dom and abs(g.diameter - dia) <= _GROUP_DIA_TOL:
                g.count += 1
                g.diameter = round((g.diameter * (g.count - 1) + dia) / g.count, 3)
                g.centers.append(cxy)
                g.centers_xyz.append(center_xyz)
                if g.depth_class != depth:
                    g.depth_class = "unknown"
                if depth == "blind":
                    g.blocked += 1
                break
        else:
            groups.append(
                MeasuredHoleGroup(
                    diameter=round(dia, 3),
                    count=1,
                    axis=dom,
                    depth_class=depth,
                    centers=[cxy],
                    centers_xyz=[center_xyz],
                    blocked=1 if depth == "blind" else 0,
                    through_axis=dom,
                )
            )

    out.hole_groups = groups
    out.total_holes = len(holes)
    out.countersinks = csk
    out.counterbores = cbore
    out.chamfer_faces = chamfer_planes + chamfer_cones
    out.rounds.sort()

    # mesh-derived solids triangulate every cut — no analytic cylinders exist
    # even when holes were really subtracted. Fall back to ray-tunnel detection
    # on the tessellation when the B-Rep scan found nothing — and ALWAYS when
    # slots are still unproven: an oblong slot is not an analytic cylinder, so
    # `holes` being non-empty used to skip the scan that could see it (observed:
    # l_bracket leg slot measured as 0 elongated voids on a correct part).
    if stl_path is not None and (not holes or not out.slots):
        _mesh_fallback(out, stl_path, mesh)

    # structural features on the face-adjacency graph (pockets/grooves/
    # notches — the classes a hole/boss scan cannot see). B-Rep only;
    # skipped entirely on heavily faceted solids
    _extract_structural(shape, out, contains=contains)

    # quiddity feature recognition (optional dep): enriches with pattern and
    # relationship evidence the face scan cannot express. Additive only —
    # plain bores/bosses stay owned by the analytic scan above. `recognition`
    # lets the caller skip it where it cannot add evidence (declared thread
    # export solids — the helical faces make it crawl without adding anything
    # the declaration did not already state).
    if recognition:
        try:
            from cadx.cad.recognition import build_recognition_evidence

            _merge_recognition(out, build_recognition_evidence(shape, cancelled=cancelled))
        except Exception:
            pass
    # stable selector ids — must run last so every detector's entries get one
    assign_feature_ids(out)
    return out


def _merge_recognition(out: MeasuredFeatures, rec: dict[str, Any]) -> None:
    """Union quiddity recognition evidence into measured features.

    The additive pattern/boss records are quiddity-only evidence. For the
    shared counters (countersinks, counterbores, chamfer_faces, rounds) the
    analytic `_scan_faces` measurement stays authoritative — the recogniser's
    count only rescues features the face heuristics under-detected (max() of
    two detectors over the SAME geometry is a union, not a double count).
    """
    if not rec:
        return
    out.recognition_ran = True
    field_models = (
        ("hole_patterns", MeasuredPattern),
        ("slot_patterns", MeasuredPattern),
        ("bored_bosses", MeasuredBoredBoss),
        ("chamfers", MeasuredChamfer),
        ("fillets", MeasuredFillet),
    )
    for key, model in field_models:
        parsed = []
        for row in rec.get(key) or []:
            try:
                parsed.append(model.model_validate(row))
            except Exception:
                continue
        setattr(out, key, parsed)
    out.countersinks = max(out.countersinks, int(rec.get("countersink_count") or 0))
    out.counterbores = max(out.counterbores, int(rec.get("counterbore_count") or 0))
    out.chamfer_faces = max(out.chamfer_faces, int(rec.get("chamfer_face_count") or 0))
    # rounds is a multiset of radii measured per-face; merging partial lists
    # would double-count same-radius fillets, so the recogniser only rescues
    # when the face scan found none at all
    if not out.rounds:
        out.rounds = sorted(r for r in (rec.get("fillet_radii") or []) if r > 0)


def _inplane_uv(group: MeasuredHoleGroup) -> list[tuple[float, float]]:
    """(u, v) centre pairs on the plane perpendicular to the group's through
    axis — the only coordinates comparable between the B-Rep scan and the
    mesh tunnel scan.

    The legacy ``centers`` field keeps a raw ``[x, y]`` projection for both:
    for a ``+x`` bore that is ``[through_axis_x, y]`` on the B-Rep side but
    ``[bbox_mid_x, y]`` on the mesh side — same axis, never matching. The
    axis-aware ``centers_xyz`` rows mark the unobservable through-axis
    coordinate ``None`` on both paths, so the two non-None in-plane values
    are the truth. ``centers`` stays the fallback for groups without
    axis-aware rows.
    """
    ax_idx = {"x": 0, "y": 1, "z": 2}.get(str(group.axis or "+z").lstrip("+-").lower(), 2)
    others = [i for i in range(3) if i != ax_idx]
    uvs: list[tuple[float, float]] = []
    for row in group.centers_xyz or []:
        if not isinstance(row, list | tuple) or len(row) != 3:
            continue
        u, v = row[others[0]], row[others[1]]
        if u is None or v is None:
            continue
        uvs.append((float(u), float(v)))
    if uvs:
        return uvs
    for row in group.centers or []:
        if isinstance(row, list | tuple) and len(row) >= 2:
            uvs.append((float(row[0]), float(row[1])))
    return uvs


def _mesh_fallback(out: MeasuredFeatures, stl_path: Path, mesh=None) -> None:
    try:
        m_groups, m_csk, m_slots = _mesh_through_holes(stl_path, mesh)
    except Exception:
        return
    if not m_groups:
        return
    if out.hole_groups:
        # the B-Rep scan already measured the round bores analytically —
        # merge only what it cannot see: elongated tunnels (slots) plus any
        # round tunnels it missed entirely. Round bores re-detected by the
        # mesh scan (same axis, a center within ~2 mm) would double-count —
        # compared on the IN-PLANE coordinates only: the through-axis entry
        # of a projected center pair is a fabricated bbox midpoint on the
        # mesh side and the cylinder-face position on the B-Rep side, so it
        # can never dedup (every side hole counted twice).
        new = []
        for g in m_groups:
            if g.elongated:
                new.append(g)
                continue
            g_uvs = _inplane_uv(g)
            dupe = any(
                _axis_letter(g.axis) == _axis_letter(h.axis)
                and any(
                    abs(gu - hu) <= 2.0 and abs(gv - hv) <= 2.0
                    for hu, hv in _inplane_uv(h)
                    for gu, gv in g_uvs
                )
                for h in out.hole_groups
            )
            if not dupe:
                new.append(g)
        if new:
            out.hole_groups = out.hole_groups + new
            out.total_holes += len([g for g in new if not g.elongated])
        out.slots = m_slots
        # NOT approximate: the analytic B-Rep measurements (round bores,
        # rounds/fillets, chamfers) stay authoritative — the mesh scan only
        # ADDED slots/tunnels it could see that the face scan cannot. Setting
        # approximate here used to downgrade every part with holes-but-no-
        # slots, turning _fillet_check FAILs into harmless UNVERIFIEDs.
        return
    out.hole_groups = m_groups
    out.total_holes = len(m_groups)
    out.slots = m_slots
    out.countersinks = m_csk
    out.approximate = True


def _expected_dia(feature: Any) -> float | None:
    """Expected bore diameter for a spec feature, or None."""
    for key in ("diameter", "dia", "d", "bore", "hole_d"):
        v = feature.dims.get(key) if feature.dims else None
        if isinstance(v, int | float) and v > 0:
            return float(v)
    for key in ("radius", "r"):
        v = feature.dims.get(key) if feature.dims else None
        if isinstance(v, int | float) and v > 0:
            return float(v) * 2.0
    m = _M_RE.search(feature.standard or "")
    if m:
        size = int(m.group(1))
        std = feature.standard or ""
        from cadx.cad import catalog

        if _TAP_RE.search(std):
            return catalog.nominal_hole_diameter(size, "tap") or _TAP_DIA.get(size)
        return catalog.nominal_hole_diameter(size, "clearance") or _CLEARANCE_DIA.get(size)
    return None


def _feature_axis_hint(feature: Any) -> str | None:
    # structured field first ("x"|"y"|"z"|"+z"…) — intake emits it explicitly;
    # the text fallback below only covers top/bottom/through vocabulary
    ax = str(getattr(feature, "axis", "") or "").strip().lower()
    if ax and ax.lstrip("+-") in ("x", "y", "z"):
        return ax if ax.startswith(("+", "-")) else f"+{ax}"
    pos = f"{feature.position or ''} {feature.description or ''}".lower()
    if not _AXIS_HINT_RE.search(pos):
        return None
    # explicit axis-letter phrases beat direction words — a bore described as
    # "along X, through both ends" runs on X; the word "through" alone says
    # nothing about direction (a through-bore can run along any axis)
    m = _AXIS_LETTER_RE.search(pos)
    if m:
        letter = next(g for g in m.groups() if g)
        return f"+{letter.lower()}"
    if any(k in pos for k in ("top", "upper", "deck", "+z", "oben", " through", "durch")):
        return "+z"
    if "bottom" in pos or "unter" in pos or "-z" in pos:
        return "-z"
    return None


def _axis_letter(axis: str) -> str:
    """'x'|'y'|'z' — a through-bore on '-y' has the same orientation as '+y'."""
    return (axis or "").lstrip("+-").lower()


def _match_hole_group(
    groups: list[MeasuredHoleGroup], want_dia: float | None, want_axis: str | None
) -> MeasuredHoleGroup | None:
    """Best matching hole group: diameter(+axis) first, then axis, then largest."""
    want_ax = _axis_letter(want_axis or "")
    for g in groups:
        if (
            want_dia
            and abs(g.diameter - want_dia) <= max(0.2 * want_dia, 0.5)
            and (not want_ax or _axis_letter(g.axis) == want_ax)
        ):
            return g
    if want_ax:
        for g in groups:
            if _axis_letter(g.axis) == want_ax:
                return g
    if groups:
        return max(groups, key=lambda g: g.count)
    return None


def _check_hole_feature(
    f: Any, measured: MeasuredFeatures
) -> tuple[list[FeatureFinding], int, int]:
    """Compare one hole-like spec feature with measured holes.

    Returns (findings, csk_needed, cbore_needed) for the aggregate checks.
    """
    findings: list[FeatureFinding] = []
    name = f.name or f.standard or f.type
    want_n = max(1, int(f.count or 1))
    want_dia = _expected_dia(f)
    want_axis = _feature_axis_hint(f)
    std_desc = f"{f.standard or ''} {f.description or ''}"
    wants_csk = bool(_CSK_RE.search(std_desc))
    wants_cbore = bool(_CBORE_RE.search(std_desc))
    csk_needed = want_n if wants_csk else 0
    cbore_needed = want_n if wants_cbore else 0

    # elongated voids are slots, not round bores — they must not satisfy a
    # circular-hole expectation (a 20x6 slot would otherwise count as Ø12)
    round_groups = [g for g in measured.hole_groups if not g.elongated]

    # definitive check before matching: spec wants a bore along a specific
    # axis and measured holes exist only on other axes → report the axis
    # mismatch explicitly (e.g. boss bore +z instead of spec +y), not a
    # generic "no holes" that hides orientation errors
    if want_axis and round_groups:
        want_ax = _axis_letter(want_axis)
        if not any(_axis_letter(g.axis) == want_ax for g in round_groups):
            actual_axes = ", ".join(sorted({_axis_letter(g.axis) for g in round_groups}))
            findings.append(
                FeatureFinding(
                    feature=name,
                    expected=f"{want_n}x hole along the {want_ax} axis",
                    actual=f"holes detected only on the {actual_axes} axis",
                    severity="major",
                    fix_hint=(
                        f"the bore axis is wrong — rotate the cutter so the hole "
                        f"runs along {want_ax} (spec axis '{want_axis}')"
                    ),
                )
            )
            return findings, csk_needed, cbore_needed

    group = _match_hole_group(round_groups, want_dia, want_axis)
    if group is None:
        findings.append(
            FeatureFinding(
                feature=name,
                expected=f"{want_n}x hole" + (f" Ø{want_dia}mm" if want_dia else ""),
                actual="no cylindrical holes detected",
                severity="major",
                fix_hint="add the missing holes (e.g. Hole()/ClearanceHole at the spec positions)",
            )
        )
        return findings, csk_needed, cbore_needed

    if want_n > group.count:
        findings.append(
            FeatureFinding(
                feature=name,
                expected=f"{want_n} holes",
                actual=f"{group.count} matching holes",
                severity="major",
                fix_hint=f"increase the hole count to {want_n} (check the pattern)",
            )
        )
    if want_dia and abs(group.diameter - want_dia) > max(0.2 * want_dia, 0.5):
        findings.append(
            FeatureFinding(
                feature=name,
                expected=f"hole Ø~{want_dia}mm",
                actual=f"measured Ø{group.diameter}mm",
                severity="minor" if measured.approximate else "major",
                fix_hint=(
                    f"set hole diameter to {want_dia}mm — for a standard "
                    f"({f.standard or 'metric'}) use the matching bd_warehouse hole"
                ),
            )
        )

    # a bore that stops short leaves a membrane sealing the hole — it LOOKS
    # like a hole in renders but nothing goes through. Spec holes default to
    # through unless they are explicitly blind (insert pockets, depth-limited).
    dims = f.dims or {}
    desc = f"{std_desc} {f.name or ''}".lower()
    wants_blind = (
        f.type == "insert"
        or "blind" in desc
        or "sackloch" in desc
        or any(k in dims for k in ("depth", "tiefe"))
    )
    if (group.depth_class == "blind" or group.blocked > 0) and not wants_blind:
        findings.append(
            FeatureFinding(
                feature=name,
                expected="through hole(s)",
                actual=(
                    f"{max(group.blocked, 1)}/{group.count} bore(s) blocked — "
                    "the hole does not go through (a membrane seals it)"
                ),
                severity="major",
                fix_hint=(
                    "the cutting cylinder is too short or mis-centred — use "
                    "cadlib.through_hole() or a cutter whose base is below the "
                    "part, e.g. Pos(x, y, -1) * Cylinder(..., "
                    "align=(Align.CENTER, Align.CENTER, Align.MIN))"
                ),
            )
        )
    return findings, csk_needed, cbore_needed


def _check_edge_features(
    spec_features: list[Any], measured: MeasuredFeatures
) -> list[FeatureFinding]:
    """Chamfer/fillet presence + type checks (chamfer requested but fillets found)."""
    findings: list[FeatureFinding] = []
    for f in spec_features:
        if f.type == "chamfer":
            if measured.chamfer_faces == 0 and measured.rounds:
                findings.append(
                    FeatureFinding(
                        feature=f.name or "chamfer",
                        expected="chamfer (flat 45° edge break)",
                        actual=f"rounded edges (fillets r≈{measured.rounds[0]:g}mm)",
                        severity="minor",
                        fix_hint="use chamfer() on the edges instead of fillet()",
                    )
                )
            elif measured.chamfer_faces == 0:
                findings.append(
                    FeatureFinding(
                        feature=f.name or "chamfer",
                        expected="chamfer present",
                        actual="no 45° edge faces detected",
                        severity="minor",
                        fix_hint="add chamfer() to the specified edges",
                    )
                )
        elif f.type == "fillet":
            r_want = f.dims.get("radius") or f.dims.get("r") if f.dims else None
            measured_radii = [
                *measured.rounds,
                *(f.radius for f in measured.fillets if f.radius > 0),
            ]
            if not measured_radii and not measured.approximate:
                findings.append(
                    FeatureFinding(
                        feature=f.name or "fillet",
                        expected=(
                            f"fillet r≈{float(r_want):g}mm"
                            if isinstance(r_want, int | float)
                            else "fillet present"
                        ),
                        actual="no analytic or recognized fillet surface detected",
                        severity="major",
                        fix_hint=(
                            "apply the requested fillet to the intended edge and verify "
                            "that the resulting Shape, not only the applied radius, is retained"
                        ),
                    )
                )
            elif isinstance(r_want, int | float) and measured_radii:
                r_act = min(measured_radii, key=lambda r: abs(r - float(r_want)))
                if abs(r_act - float(r_want)) > max(0.5, 0.2 * float(r_want)):
                    findings.append(
                        FeatureFinding(
                            feature=f.name or "fillet",
                            expected=f"fillet r≈{r_want}mm",
                            actual=f"nearest fillet r≈{r_act}mm",
                            severity="minor",
                            fix_hint=f"set fillet radius to {r_want}mm",
                        )
                    )
    return findings


def _check_boss_features(
    spec_features: list[Any], measured: MeasuredFeatures
) -> list[FeatureFinding]:
    """Boss/tube axis check — mirrors the hole-axis check for protrusions.

    A tube lying along +x when the spec wants +z is the same class of defect
    as a bore on the wrong axis, but it never produced a finding because
    exterior cylinders were all lumped into `rounds` (fillets).
    """
    findings: list[FeatureFinding] = []
    # mesh-derived measurements never populate `bosses` — every check would
    # be a guaranteed false-major ("no protruding cylinder detected" on a
    # dome-topped knob, refuted by the critique but still capped the score)
    if measured.approximate:
        return findings
    for f in spec_features:
        ftype = (f.type or "").lower()
        text = f"{f.name or ''} {f.description or ''}".lower()
        # a declared non-cylinder class wins over stray "boss" words in the
        # prose — the regex only rescues untyped/cylinder-adjacent features
        if ftype not in _BOSS_TYPES and (ftype in _NON_BOSS_TYPES or not _BOSS_RE.search(text)):
            continue
        want_axis = _feature_axis_hint(f)
        if not want_axis:
            continue
        want_ax = _axis_letter(want_axis)
        if any(_axis_letter(b.axis) == want_ax for b in measured.bosses):
            continue
        actual_axes = ", ".join(sorted({_axis_letter(b.axis) for b in measured.bosses}))
        findings.append(
            FeatureFinding(
                feature=f.name or ftype or "boss",
                expected=f"boss/tube along the {want_ax} axis",
                actual=(
                    f"bosses detected only on the {actual_axes} axis"
                    if measured.bosses
                    else "no protruding cylinder detected"
                ),
                severity="major",
                fix_hint=(
                    f"the boss/tube axis is wrong — orient the cylinder along "
                    f"{want_ax} (spec axis '{want_axis}')"
                    if measured.bosses
                    else "the boss/tube protrusion is missing entirely"
                ),
            )
        )
    return findings


def _check_slot_features(
    spec_features: list[Any], measured: MeasuredFeatures
) -> list[FeatureFinding]:
    """Elongated through-voids are measured separately from round holes —
    a spec 'slot' feature checks the slot count, not hole groups. Open
    channels measured structurally (grooves) are slot evidence too."""
    findings: list[FeatureFinding] = []
    slot_feats = [f for f in spec_features if "slot" in (f.type or "").lower()]
    wanted = sum(max(1, int(f.count or 1)) for f in slot_feats)
    measured_slots = measured.slots + len(measured.grooves)
    if wanted and measured_slots < wanted:
        findings.append(
            FeatureFinding(
                feature=slot_feats[0].name or "slot",
                expected=f"{wanted} slot(s)",
                actual=(
                    f"{measured_slots} elongated void(s)/channel(s) detected"
                    f" ({measured.slots} through-void, {len(measured.grooves)} groove)"
                ),
                severity="major",
                fix_hint=(
                    "cut the slot as an elongated through-void, not a round "
                    "hole — cadlib.oblong_slot(width, length, direction=<slot "
                    "axis>, normal=<wall normal>) inside a Locations scope; "
                    "a slot along Z through an X-thin wall: "
                    'oblong_slot(6.5, 15, direction="+z", normal="+x")'
                ),
            )
        )
    return findings


def _check_pattern_features(
    spec_features: list[Any], measured: MeasuredFeatures
) -> list[FeatureFinding]:
    """Hole-pattern expectations (bolt circle, array, grid) checked against
    quiddity pattern evidence. Only runs when the recogniser actually ran —
    without it, "no pattern detected" would be an unfounded guess."""
    if not measured.recognition_ran:
        return []
    findings: list[FeatureFinding] = []
    for f in spec_features:
        ftype = (f.type or "").lower()
        text = f"{f.type or ''} {f.name or ''} {f.description or ''}"
        # a non-hole feature whose description merely mentions a circle term
        # (e.g. a gear tooth ring's "pitch circle") is not a hole pattern
        holeish = ftype in _HOLE_TYPES or bool(_HOLEISH_RE.search(text))
        if ftype != "pattern" and not (holeish and _PATTERN_RE.search(text)):
            continue
        want_n = max(1, int(f.count or 1))
        if measured.hole_patterns:
            best = max(measured.hole_patterns, key=lambda p: p.count)
            if want_n > 1 and best.count < want_n:
                findings.append(
                    FeatureFinding(
                        feature=f.name or "pattern",
                        expected=f"{want_n} holes in the pattern",
                        actual=(f"largest detected pattern has {best.count} members ({best.kind})"),
                        severity="major",
                        fix_hint=(
                            f"the pattern needs {want_n} members — check the "
                            "repetition count (e.g. PolarLocations count)"
                        ),
                    )
                )
        elif measured.total_holes >= want_n:
            # holes exist but form no recognisable arrangement
            findings.append(
                FeatureFinding(
                    feature=f.name or "pattern",
                    expected=f"hole pattern ({want_n}x)",
                    actual="holes detected but no repeating pattern recognised",
                    severity="minor",
                    fix_hint=(
                        "arrange the holes as the specified pattern — "
                        "PolarLocations for a bolt circle, GridLocations for a "
                        "rectangular array"
                    ),
                )
            )
    return findings


def compare_features(spec_features: list[Any], measured: MeasuredFeatures) -> list[FeatureFinding]:
    """Compare spec features against measured geometry. Feedback-only findings."""
    findings: list[FeatureFinding] = []

    hole_feats = [
        f
        for f in spec_features
        if f.type in _HOLE_TYPES or (f.standard and _M_RE.search(f.standard))
    ]
    csk_needed = 0
    cbore_needed = 0

    for f in hole_feats:
        f_findings, csk_n, cbore_n = _check_hole_feature(f, measured)
        findings += f_findings
        csk_needed += csk_n
        cbore_needed += cbore_n

    findings += _check_slot_features(spec_features, measured)
    findings += _check_boss_features(spec_features, measured)
    findings += _check_edge_features(spec_features, measured)
    findings += _check_pattern_features(spec_features, measured)

    csk_total = measured.countersinks
    cbore_total = measured.counterbores
    if csk_needed > csk_total:
        findings.append(
            FeatureFinding(
                feature="countersinks",
                expected=f"{csk_needed} countersunk holes",
                actual=f"{csk_total} countersink cones detected",
                severity="major",
                fix_hint=(
                    "add countersinks (bd_warehouse CounterSinkHole / "
                    "Hole(depth=None) + countersink cone)"
                ),
            )
        )
    if cbore_needed > cbore_total:
        findings.append(
            FeatureFinding(
                feature="counterbores",
                expected=f"{cbore_needed} counterbored holes",
                actual=f"{cbore_total} stepped bores detected",
                severity="major",
                fix_hint="add counterbores (bd_warehouse CounterBoreHole)",
            )
        )

    return findings


def mesh_features_to_measured(mesh_feats: dict[str, Any]) -> MeasuredFeatures:
    """Adapt detect_mesh_features output (trimesh ray-grid tunnels/dimples)
    into MeasuredFeatures so the spec/anchor comparators can run on the pure
    mesh path too. `centers` preserves the legacy projection pair; `centers_xyz`
    maps that pair to its real world axes and leaves the tunnel axis unknown.
    """
    groups: list[MeasuredHoleGroup] = []
    n_dimples = 0
    for h in mesh_feats.get("holes") or []:
        kind = str(h.get("kind") or "")
        if kind == "dimple":
            n_dimples += 1
            continue
        if kind != "tunnel":
            continue
        ax = str(h.get("axis") or "z")
        x, y = float(h.get("x") or 0), float(h.get("y") or 0)
        through_idx = "xyz".index(ax) if ax in "xyz" else 2
        in_plane = [idx for idx in range(3) if idx != through_idx]
        center_xyz: list[float | None] = [None, None, None]
        center_xyz[in_plane[0]] = x
        center_xyz[in_plane[1]] = y
        groups.append(
            MeasuredHoleGroup(
                diameter=float(h.get("dia") or 0),
                count=1,
                axis=f"+{ax}",
                depth_class="through",
                centers=[[x, y]],
                centers_xyz=[center_xyz],
                through_axis=f"+{ax}",
            )
        )
    return assign_feature_ids(
        MeasuredFeatures(
            hole_groups=groups,
            total_holes=len(groups),
            countersinks=n_dimples,  # top-down dips — the only mesh csk signal
            approximate=True,
        )
    )


def compare_anchor_positions(
    anchors: dict[str, Any] | None, measured: MeasuredFeatures
) -> list[FeatureFinding]:
    """Compare measured hole centres against the plan-view anchors.

    Anchors carry the authoritative XY positions (image ± mesh-reconciled);
    a hole off by more than 2 mm — or missing entirely — is a major finding
    so the critique asks for the anchored coordinates, not a vague fix.
    """
    findings: list[FeatureFinding] = []
    holes = (anchors or {}).get("holes") or []
    if not holes:
        return findings
    centres: list[tuple[float, float]] = []
    wrong_axis_centres: list[tuple[float, float, str]] = []
    for g in measured.hole_groups:
        if g.axis in ("+z", "-z", ""):
            centres += [(c[0], c[1]) for c in g.centers]
        else:
            wrong_axis_centres += [(c[0], c[1], g.axis) for c in g.centers]
    any_axis = centres + [(c0, c1) for c0, c1, _ in wrong_axis_centres]
    for a in holes:
        ax, ay = float(a["x"]), float(a["y"])
        # position exactness only makes sense for anchors backed by an
        # authoritative plan-view image; mesh-only / approximate anchors are
        # self-referential (the mesh IS the part) so a deviation is minor
        src = str(a.get("source", "image"))
        # spec-anchored positions are declared intent — just as exact as a
        # measured plan-view position for deviation reporting
        exact = "spec" in src or src in ("image", "image+mesh")
        # when the anchor carries a measured drill axis (plan-view holes are
        # "+z"), a hole bored along a different axis never matches it — a
        # sideways pin through the width is not a mounting hole.
        want = str(a.get("axis") or "")
        want_z = want.lstrip("+-") in ("z", "")
        use_centres = centres if want_z else any_axis
        if not use_centres and not want_z:
            use_centres = any_axis
        d = min((math.hypot(ax - cx, ay - cy) for cx, cy in use_centres), default=None)
        if want_z and (d is None or d > (2.0 if exact else 3.0)) and wrong_axis_centres:
            dw = min(
                (math.hypot(ax - cx, ay - cy) for cx, cy, _ in wrong_axis_centres),
                default=None,
            )
            if dw is not None and dw <= 3.0:
                bad_axis = min(wrong_axis_centres, key=lambda t: math.hypot(ax - t[0], ay - t[1]))[
                    2
                ]
                findings.append(
                    FeatureFinding(
                        feature=f"hole {a.get('index', '?')}",
                        expected=f"at ({ax:.1f}, {ay:.1f}) drilled along {want or '+z'}",
                        actual=f"hole near that position drilled along {bad_axis}",
                        severity="major" if exact else "minor",
                        fix_hint=(
                            "the hole is bored through the wrong face — cut it "
                            f"along {want or '+z'} (through the flat top face), "
                            "not through a side edge"
                        ),
                    )
                )
                continue

        if d is None:
            findings.append(
                FeatureFinding(
                    feature=f"hole {a.get('index', '?')}",
                    expected=f"at ({ax:.1f}, {ay:.1f}) Ø{a.get('dia', 0):.1f}",
                    actual="missing",
                    severity="major",
                    fix_hint=(
                        f"cut the hole at the anchored position "
                        f"({ax:.1f}, {ay:.1f}) — use anchor_holes()"
                    ),
                )
            )
        elif d > (2.0 if exact else 3.0):
            findings.append(
                FeatureFinding(
                    feature=f"hole {a.get('index', '?')}",
                    expected=f"at ({ax:.1f}, {ay:.1f})",
                    actual=f"off by {d:.1f} mm",
                    severity="major" if exact else "minor",
                    fix_hint=(
                        f"move the hole to ({ax:.1f}, {ay:.1f}) — "
                        "use anchor_holes() coordinates verbatim"
                        if exact
                        else "nudge the hole toward the anchored position"
                    ),
                )
            )
        elif d > 1.0:
            findings.append(
                FeatureFinding(
                    feature=f"hole {a.get('index', '?')}",
                    expected=f"at ({ax:.1f}, {ay:.1f})",
                    actual=f"off by {d:.1f} mm",
                    severity="minor",
                    fix_hint="nudge the hole to the anchored position",
                )
            )
    return findings
