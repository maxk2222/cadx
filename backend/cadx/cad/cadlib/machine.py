"""cadlib machine — deterministic machining ops targeted at real faces.

Instead of hand-deriving ``Pos(...) * Rot(...)`` for a tapped hole on a side
wall or a sphere pole, the agent names the surface semantically:

    result, feat = machine_on_face(part, face="+z", op="tap",
                                   major_d=8, depth=25)
    result, feat = machine_on_face(part, face="left", op="cbore",
                                   dia=5, cbore_dia=9, cbore_depth=3)

``resolve_face`` accepts a Face, an index into ``describe_faces`` order, an
axis/word hint ("+z", "top", "front"), a surface kind ("planar",
"cylindrical", "spherical"), or a 3D pick point (nearest face wins).
``face_frame`` builds a local frame whose origin sits ON the surface at the
picked point and whose +Z is the outward normal — identical semantics on
planar, sloped, cylindrical and spherical faces.

Cutters are built mouth-at-z0 extending -Z (matching ``tapped_hole``), then
placed with the frame. ``machine_on_face`` returns ``(part, feature_record)``
where the record is CADX_FEATURES-compatible — register it so previews draw
the thread overlay and the runner counts real threads.
"""

import math
from typing import Any

from build123d import (
    Axis,
    Box,
    Cone,
    Cylinder,
    Face,
    GeomType,
    Part,
    Plane,
    Pos,
    ShapeList,
    Vector,
)

from .cutouts import tapped_hole
from .ops import detail_mode as _detail_mode

_AXIS_VECS: dict[str, tuple[float, float, float]] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
# frame contract: +Y front (AGENTS.md coordinate contract)
_WORD_TO_AXIS: dict[str, str] = {
    "top": "+z",
    "bottom": "-z",
    "front": "+y",
    "back": "-y",
    "left": "-x",
    "right": "+x",
}
_GEOM_KIND: dict[GeomType, str] = {
    GeomType.PLANE: "planar",
    GeomType.CYLINDER: "cylindrical",
    GeomType.SPHERE: "spherical",
    GeomType.CONE: "conical",
    GeomType.TORUS: "toroidal",
}
_KIND_WORDS = {"planar", "flat", "cylindrical", "spherical", "conical", "toroidal"}


def _vec(v: Any) -> Vector:
    if isinstance(v, Vector):
        return v
    if isinstance(v, tuple | list) and len(v) == 3:
        return Vector(float(v[0]), float(v[1]), float(v[2]))
    raise ValueError(f"expected a 3-vector, got {v!r}")


def _sorted_faces(part: Any) -> ShapeList[Face]:
    faces = ShapeList(part.faces())
    return ShapeList(
        sorted(
            faces,
            key=lambda f: (
                -f.area,
                round(f.center().X, 3),
                round(f.center().Y, 3),
                round(f.center().Z, 3),
            ),
        )
    )


def describe_faces(part: Any) -> list[dict[str, Any]]:
    """Canonical face inventory for a solid — indices are stable for a given
    shape (area desc, center coords tiebreak) so ``resolve_face(part, 0)``
    and this listing agree."""
    out: list[dict[str, Any]] = []
    for i, f in enumerate(_sorted_faces(part)):
        c = f.center()
        try:
            n = f.normal_at(c)
        except Exception:
            n = Vector(0, 0, 0)
        out.append(
            {
                "index": i,
                "kind": _GEOM_KIND.get(f.geom_type, "other"),
                "area_mm2": round(float(f.area), 3),
                "center": [round(c.X, 3), round(c.Y, 3), round(c.Z, 3)],
                "normal": [round(n.X, 4), round(n.Y, 4), round(n.Z, 4)],
            }
        )
    return out


def _face_by_axis(faces: ShapeList[Face], spec: str) -> Face:
    a = Vector(*_AXIS_VECS[_WORD_TO_AXIS.get(spec, spec)])
    best, best_score = None, -2.0
    for f in faces:
        try:
            score = f.normal_at(f.center()).dot(a)
        except Exception:
            score = -2.0
        # deterministic tiebreak: alignment first, then area
        if score > best_score + 1e-9 or (
            abs(score - best_score) <= 1e-9 and best is not None and f.area > best.area
        ):
            best, best_score = f, score
    if best is None:
        raise ValueError(f"no face resolved for axis hint {spec!r}")
    return best


def _face_by_word(faces: ShapeList[Face], spec: str) -> Face:
    word = spec.strip().lower()
    if _WORD_TO_AXIS.get(word, word) in _AXIS_VECS:
        return _face_by_axis(faces, word)
    if word in ("largest", "biggest"):
        return faces[0]
    if word == "smallest":
        return faces[-1]
    if word in _KIND_WORDS:
        want = "planar" if word == "flat" else word
        hits = [f for f in faces if _GEOM_KIND.get(f.geom_type) == want]
        if not hits:
            raise ValueError(f"no {want} face on this part")
        return hits[0]
    raise ValueError(
        f"unknown face descriptor {spec!r} — use an index, +x/+y/+z/-x/-y/-z, "
        "top|bottom|front|back|left|right, largest|smallest, a kind "
        "(planar|cylindrical|spherical|conical|toroidal), or a pick point"
    )


def _face_by_point(faces: ShapeList[Face], point: Vector) -> Face:
    best, best_dist = None, float("inf")
    for f in faces:
        try:
            surf, _n = project_to_face(f, point)
            dist = (surf - point).length
        except Exception:
            dist = (f.center() - point).length
        if dist < best_dist - 1e-9 or (
            abs(dist - best_dist) <= 1e-9 and best is not None and f.area > best.area
        ):
            best, best_dist = f, dist
    if best is None:
        raise ValueError("no face resolved near the pick point")
    return best


def resolve_face(part: Any, spec: Any) -> Face:
    """Pick a single Face off `part` from a semantic descriptor.

    Accepted: a Face (passthrough), an int into `describe_faces` order,
    "+x|+y|+z|-x|-y|-z" or top/bottom/front/back/left/right (face whose
    centre-normal best matches the axis; largest area wins ties),
    "largest"/"smallest", a surface kind ("planar"/"cylindrical"/"spherical"/
    "conical"/"toroidal" → largest of that kind), or a 3D pick point /
    (x, y, z) — the face whose surface is nearest to the point wins.
    """
    if isinstance(spec, Face):
        return spec
    faces = _sorted_faces(part)
    if isinstance(spec, int) and not isinstance(spec, bool):
        if not 0 <= spec < len(faces):
            raise ValueError(
                f"face index {spec} out of range — describe_faces lists {len(faces)}"
            )
        return faces[spec]
    if isinstance(spec, str):
        return _face_by_word(faces, spec)
    return _face_by_point(faces, _vec(spec))


def project_to_face(face: Face, point: Any) -> tuple[Vector, Vector]:
    """Nearest on-surface point to `point` + the outward normal there.

    Casts a ray through the pick point along both directions of the
    closest-UV normal and keeps the nearest face intersection; falls back to
    the face centre when the ray misses the patch (degenerate trims).
    """
    p = _vec(point)
    try:
        n0 = face.normal_at(p)
    except Exception:
        c = face.center()
        return c, face.normal_at(c)
    candidates: list[tuple[Vector, Vector]] = []
    for direction in (-n0, n0):
        try:
            for hit in face.find_intersection_points(Axis(p, direction)):
                candidates.append((hit[0], hit[1]))
        except Exception:
            continue
    if candidates:
        # normal_at already picked the closest patch region; a long ray can
        # wrap a closed surface (sphere far side) — keep the nearest hit
        surf, n = min(candidates, key=lambda h: (h[0] - p).length)
        return surf, n
    c = face.center()
    return c, face.normal_at(c)


def face_frame(face: Face, point: Any = None, at: tuple[float, float] | None = None) -> Plane:
    """Local work frame on `face`: origin on the surface, +Z = outward normal.

    `point` is a 3D pick (default: face centre); `at=(u, v)` additionally
    offsets u/v mm along two tangent directions and re-projects onto the
    surface — on curved faces the offset follows the skin instead of
    drilling from a tangent plane."""
    pick = _vec(point) if point is not None else face.center()
    surf, n = project_to_face(face, pick)
    if n.length < 1e-9:
        n = Vector(0, 0, 1)
    # natural tangent frame: x_dir = the world axis least parallel to the
    # normal (projected onto the tangent plane) — on a +z face `at=(u, v)`
    # maps u → +x, v → +y, which is what a drawing-style callout means
    ref = min(
        (Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)),
        key=lambda r: abs(n.dot(r)),
    )
    x_dir = (ref - n * n.dot(ref)).normalized()
    y_dir = n.cross(x_dir).normalized()
    if at is not None:
        u, v = float(at[0]), float(at[1])
        surf, n = project_to_face(face, surf + x_dir * u + y_dir * v)
        x_dir = (ref - n * n.dot(ref)).normalized()
        y_dir = n.cross(x_dir).normalized()
    return Plane(origin=surf, x_dir=x_dir, z_dir=n)


_CUT_OPS = {"drill", "hole", "clearance", "tap", "tapped_hole", "cbore", "counterbore",
            "csk", "countersink", "spotface", "pocket", "slot_pocket"}
_ADD_OPS = {"boss"}


def _drill_cutter(
    op: str,
    *,
    dia: float | None,
    depth: float | None,
    span: float,
    cbore_dia: float | None,
    cbore_depth: float | None,
    csk_dia: float | None,
    csk_angle: float,
) -> Part:
    if dia is None:
        raise ValueError(f"{op} needs dia")
    r = float(dia) / 2
    if depth is None:
        height = span + 2.0
        z_mid = (1.0 - span) / 2  # overshoots 1 mm above the mouth
    else:
        height = float(depth) + 1.0
        z_mid = (1.0 - float(depth)) / 2
    cutter: Part = Pos(0, 0, z_mid) * Cylinder(radius=r, height=height)
    if op in ("cbore", "counterbore", "spotface"):
        if cbore_dia is None:
            raise ValueError(f"{op} needs cbore_dia")
        cb = float(cbore_dia) / 2
        cd = float(cbore_depth) if cbore_depth else max(1.0, (cb - r))
        cutter += Pos(0, 0, -cd / 2 + 0.5) * Cylinder(radius=cb, height=cd + 1.0)
    elif op in ("csk", "countersink"):
        if csk_dia is None:
            raise ValueError("countersink needs csk_dia")
        cr = float(csk_dia) / 2
        cd = (cr - r) / math.tan(math.radians(float(csk_angle)) / 2)
        cutter += Pos(0, 0, -cd / 2 + 0.5) * Cone(bottom_radius=r, top_radius=cr, height=cd + 1.0)
    return cutter


def cutter_for(
    op: str,
    *,
    span: float,
    dia: float | None = None,
    depth: float | None = None,
    major_d: float | None = None,
    pitch: float | None = None,
    hand: str = "right",
    drill_dia: float | None = None,
    cbore_dia: float | None = None,
    cbore_depth: float | None = None,
    csk_dia: float | None = None,
    csk_angle: float = 90.0,
    width: float | None = None,
    length: float | None = None,
    entry_chamfer: float = 0.6,
) -> Part:
    """Machining cutter in the canonical frame: mouth at z=0, cut along -Z.

    `span` is the part's clearance length (bbox diagonal + margin) used for
    through cuts when `depth` is None — the cutter overshoots both sides so
    no membrane seals the opening. Returns a Part/Compound to subtract
    (op in cut-ops) or add (op="boss")."""
    op = op.strip().lower()
    if op in ("tap", "tapped_hole"):
        if major_d is None:
            raise ValueError("tap needs major_d (e.g. 8 for M8)")
        depth = float(depth if depth is not None else span)
        cutter = tapped_hole(
            major_d, pitch, depth + 0.5, hand=hand, drill_dia=drill_dia
        )
        # mouth chamfer: a real tapped hole opens with a small countersink;
        # it also keeps the groove ridge from ending in ragged concentric
        # steps where a curved entry surface clips the helix
        if entry_chamfer > 0:
            drill = float(drill_dia) if drill_dia is not None else (
                float(major_d) - float(pitch or 1.0)
            )
            ch = float(entry_chamfer)
            cutter += Pos(0, 0, -ch / 2) * Cone(
                bottom_radius=drill / 2, top_radius=drill / 2 + ch, height=ch
            )
        return cutter
    if op in ("drill", "hole", "clearance", "cbore", "counterbore", "csk",
              "countersink", "spotface"):
        return _drill_cutter(
            op, dia=dia, depth=depth, span=span, cbore_dia=cbore_dia,
            cbore_depth=cbore_depth, csk_dia=csk_dia, csk_angle=csk_angle,
        )
    if op in ("pocket", "slot_pocket"):
        if depth is None:
            raise ValueError("pocket needs a depth")
        d = float(depth)
        if width is not None and length is not None:
            return Pos(0, 0, (1.0 - d) / 2) * Box(
                float(width), float(length), d + 1.0
            )
        if dia is not None:
            return Pos(0, 0, (1.0 - d) / 2) * Cylinder(
                radius=float(dia) / 2, height=d + 1.0
            )
        raise ValueError("pocket needs width+length or dia")
    if op == "boss":
        if dia is None:
            raise ValueError("boss needs dia")
        h = float(depth if depth is not None else 10.0)
        # additive: base sits flush on the surface plane, extends +Z OUT
        return Pos(0, 0, h / 2 - 0.5) * Cylinder(radius=float(dia) / 2, height=h + 0.5)
    raise ValueError(
        f"unknown op {op!r} — drill|tap|cbore|csk|spotface|pocket|boss"
    )


def _span(part: Any) -> float:
    s = part.bounding_box().size
    return math.sqrt(s.X * s.X + s.Y * s.Y + s.Z * s.Z) + 10.0


def machine_on_face(
    part: Any,
    face: Any,
    op: str,
    *,
    at: Any = None,
    axis: Any = None,
    depth: float | None = None,
    **dims: Any,
) -> tuple[Any, dict[str, Any]]:
    """Resolve the face, project the pick point, place the canonical cutter.

    `face`: descriptor for `resolve_face`. `at`: None → face centre; a
    (u, v) 2-tuple → tangent-mm offsets re-projected onto the skin; an
    (x, y, z) 3-tuple → world pick point. `axis`: overrides the outward
    normal as the cutter's approach direction (string "+x"/.../Vector) —
    leave None for "orthogonal zur Fläche". `depth` mm → blind; None →
    through (only for through-capable ops).

    Returns ``(new_part, feature_record)`` — the record carries the
    CADX_FEATURES thread fields (kind/thread_spec/major_diameter/pitch/
    length/start/external/hand/axis/position_mm) for taps so preview
    overlays and the export gate see the real thread intent.
    """
    f = resolve_face(part, face)
    point = None
    uv = None
    if at is not None:
        seq = tuple(at) if isinstance(at, list | tuple) else None
        if seq is not None and len(seq) == 2:
            uv = (float(seq[0]), float(seq[1]))
        elif seq is not None and len(seq) == 3:
            point = Vector(*[float(c) for c in seq])
        else:
            raise ValueError("at must be (u, v) tangent offsets or an (x, y, z) point")
    elif not isinstance(face, Face | int | str):
        # a 3D pick descriptor selects the face AND marks the spot —
        # "fläche hier, dort bohren" in one point
        point = _vec(face)
    frame = face_frame(f, point=point, at=uv)
    if axis is not None:
        a = _vec(_AXIS_VECS.get(str(axis).lower(), axis) if isinstance(axis, str) else axis)
        n = a.normalized()
        ref = min(
            (Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)),
            key=lambda r: abs(n.dot(r)),
        )
        frame = Plane(
            origin=frame.location.position,
            x_dir=(ref - n * n.dot(ref)).normalized(),
            z_dir=n,
        )

    op_name = op.strip().lower()
    cutter = cutter_for(op_name, span=_span(part), depth=depth, **dims)
    placed = frame.location * cutter
    is_cut = op_name not in _ADD_OPS
    result = part - placed if is_cut else part + placed

    # honesty check: a cutter that barely touches or misses the body is a
    # silent no-op — surface it instead of shipping an unmachined part
    try:
        delta = abs(result.volume - part.volume)
        if delta < 1e-3:
            raise ValueError(
                f"{op_name} at {face!r} removed no material — the cutter "
                "missed the body (check face descriptor and pick point)"
            )
    except (AttributeError, TypeError):
        pass

    surf = frame.location.position
    n = frame.z_dir
    record: dict[str, Any] = {
        "kind": _RECORD_KIND.get(op_name, op_name),
        "op": op_name,
        "face": str(face)[:80],
        "position_mm": [round(surf.X, 3), round(surf.Y, 3), round(surf.Z, 3)],
        "normal": [round(n.X, 4), round(n.Y, 4), round(n.Z, 4)],
    }
    record.update({k: v for k, v in dims.items() if v is not None})
    if depth is not None:
        record["depth"] = float(depth)
    if op_name in ("tap", "tapped_hole"):
        major = float(dims.get("major_d") or dims.get("dia") or 0.0)
        record.update(
            {
                "kind": "thread",
                "major_diameter": major,
                "pitch": float(dims["pitch"]) if dims.get("pitch") else None,
                "length": float(depth) if depth is not None else None,
                "start": -float(depth or 0.0),
                "external": False,
                "hand": dims.get("hand", "right"),
                "detail": _detail_mode(),
            }
        )
    return result, record


_RECORD_KIND = {
    "drill": "hole",
    "hole": "hole",
    "clearance": "hole",
    "cbore": "counterbore",
    "counterbore": "counterbore",
    "spotface": "counterbore",
    "csk": "countersink",
    "countersink": "countersink",
    "pocket": "pocket",
    "slot_pocket": "pocket",
    "boss": "boss",
}


def machine_on_point(
    part: Any,
    point: Any,
    op: str,
    *,
    axis: Any = None,
    depth: float | None = None,
    **dims: Any,
) -> tuple[Any, dict[str, Any]]:
    """Same as `machine_on_face` but the target IS the world pick point —
    the face is resolved by proximity and the frame built there."""
    f = resolve_face(part, point)
    return machine_on_face(part, f, op, at=point, axis=axis, depth=depth, **dims)
