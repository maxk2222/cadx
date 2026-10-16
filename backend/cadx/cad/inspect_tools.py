"""In-namespace measurement primitives for generated model code.

Trusted runner-side helper — NOT part of cadlib's allowlisted import surface:
``runner_entry`` imports it (runner frames bypass the sandbox import guard)
and injects the public functions as bare names into the model.py exec
namespace, so generated code can self-verify what it just built::

    result = base - cutter
    print(measure(result))        # bbox / volume / face inventory
    print(find_holes(result))     # did the bore survive the boolean?

Deliberately no ``cadx`` imports and no lazy imports inside functions — any
deferred import would hit the runner's _ImportGuard once a model.py frame is
on the stack, so every dependency is resolved at module load time (the import
itself runs from a runner_entry frame, which the guard allowlists).

All public functions are defensive: bad input returns ``{"error": ...}``
instead of raising, and faceted hybrid solids (more than ``_FACE_CAP`` faces)
short-circuit the face-walking finders with ``{"skipped": "faceted solid"}``
because a per-face analytic scan on a 15k-triangle sewn solid is pointless.
"""

from __future__ import annotations

import math
from typing import Any

from build123d import GeomType, Shape
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.TopAbs import TopAbs_EDGE, TopAbs_IN, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

# OCP 8 (build123d 0.13) moved the NCollection sequences out of OCP.TopTools
# and dropped the ``_s`` suffix on the static TopoDS downcasts.
try:
    from OCP.TopTools import TopTools_HSequenceOfShape as _HSequenceOfShape
except ImportError:  # OCP 8+
    from OCP.collections import HSequence_TopoDS_Shape as _HSequenceOfShape  # type: ignore[no-redef]

_to_wire = getattr(TopoDS, "Wire_s", None) or TopoDS.Wire
_to_vertex = getattr(TopoDS, "Vertex_s", None) or TopoDS.Vertex

# above ~5000 faces the shape is a sewn faceted mesh solid: analytic face
# scanning finds thousands of triangles, not features — skip instead of
# burning seconds returning garbage
_FACE_CAP = 5000
_INSIDE_FRACTION_HOLE = 0.7  # radial samples in material -> bore, else boss/round
# u-span (radians): edge fillets wrap ~90deg; a protruding cylinder wraps at
# least a half-circle (embedded tube ~pi, free boss ~2pi)
_BOSS_WRAP_MIN = 2.4
_THROUGH_FRACTION = 0.75  # cylinder axial extent vs part extent -> "through"
_GROUP_DIA_TOL = 0.3  # mm — same-radius coaxial segments merge into one bore
_COAX_TOL = 0.5  # mm between axes to count as coaxial
_AXIS_ALIGN_MIN = 0.98  # |dominant component| for a named axis, else "other"
_SAMPLE_DIRS = 8
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _err(e: Exception) -> dict[str, Any]:
    return {"error": f"{type(e).__name__}: {e}"}


def _as_shape(obj: Any) -> Shape | None:
    """Best-effort unwrap: generated code may pass a Part/Compound/Solid, a
    BuildPart/BuildSketch builder (``.part``/``.sketch``/``.obj``) or a bare
    Shape. Returns None when nothing shape-like was found."""
    if isinstance(obj, Shape):
        return obj
    for attr in ("part", "sketch", "obj", "shape"):
        inner = getattr(obj, attr, None)
        if isinstance(inner, Shape):
            return inner
    wrapped = getattr(obj, "wrapped", None)
    if wrapped is not None:
        try:
            return Shape(wrapped)
        except Exception:
            return None
    return None


def _r3(v: float) -> float:
    return round(float(v), 3)


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


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _axis_label(axis: list[float]) -> str:
    """'+x'/'-y'/'+z' when the direction is near-axis-aligned, else 'other'."""
    i = max(range(3), key=lambda k: abs(axis[k]))
    if abs(axis[i]) < _AXIS_ALIGN_MIN:
        return "other"
    return f"{'+' if axis[i] >= 0 else '-'}{'xyz'[i]}"


def _extent_along(shape: Shape, axis: list[float]) -> float:
    """Part extent measured along an arbitrary direction: project the eight
    bbox corners — robust for off-axis bores where the bbox extents lie."""
    try:
        bb = shape.bounding_box()
        proj = [
            _dot([x, y, z], axis)
            for x in (bb.min.X, bb.max.X)
            for y in (bb.min.Y, bb.max.Y)
            for z in (bb.min.Z, bb.max.Z)
        ]
        return max(proj) - min(proj)
    except Exception:
        return 0.0


def _inside_fraction(shape: Shape, center: list[float], axis: list[float], r: float) -> float:
    """Fraction of radial samples (on a circle of radius ~1.35r around the axis
    point) that lie inside material. Bores score ~1.0 (wall surrounded by
    material), edge fillets ~0.25-0.5 (open to air on most sides). Ported from
    features._inside_fraction."""
    ref = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = [
        axis[1] * ref[2] - axis[2] * ref[1],
        axis[2] * ref[0] - axis[0] * ref[2],
        axis[0] * ref[1] - axis[1] * ref[0],
    ]
    un = _norm(u) or 1.0
    u = [c / un for c in u]
    w = [
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
    ]
    rr = r * 1.35
    inside = 0
    for i in range(_SAMPLE_DIRS):
        ang = 2.0 * math.pi * i / _SAMPLE_DIRS
        pt = [center[j] + rr * (math.cos(ang) * u[j] + math.sin(ang) * w[j]) for j in range(3)]
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
    if abs(_dot(a_dir, b_dir)) < 0.995:
        return False
    d = _sub(b_center, a_center)
    par = _dot(d, a_dir)
    perp = [d[i] - par * a_dir[i] for i in range(3)]
    return _norm(perp) <= _COAX_TOL


def _scan_faces(  # ported from features._scan_faces (cylinder + cone passes)
    faces: list[Any],
) -> tuple[
    list[tuple[float, list[float], list[float], float, float]],
    list[tuple[list[float], list[float], float, float, float, float, float]],
]:
    """Walk faces once; collect analytic cylinders and cones.

    Cylinder tuples: (radius, unit_axis, on_axis_center, vspan, uspan) — the
    centre is the face centre projected ONTO the axis so inside-fraction
    sampling happens on the bore axis, not the wall. Cone tuples:
    (unit_axis, axis_location, semi_angle_rad_signed, face_radius, ref_radius,
    v1, v2) — face_radius is the face centre's distance to the cone axis (a
    rim chamfer coaxial to a bore must not read as a countersink); the cone
    surface's radius at parameter v is ``ref_radius + v*sin(semi)`` and its
    rim centre ``loc + v*cos(semi)*axis`` — the SIGNED semi-angle is kept
    because the adaptor may orient the cone axis against the widening
    direction.
    """
    cyl_faces: list[tuple[float, list[float], list[float], float, float]] = []
    cones: list[tuple[list[float], list[float], float, float, float, float, float]] = []
    for f in faces:
        try:
            gt = f.geom_type
            ad = BRepAdaptor_Surface(f.wrapped)
            if gt == GeomType.CYLINDER:
                cyl = ad.Cylinder()
                axis = _vec3(cyl.Axis().Direction())
                an = _norm(axis) or 1.0
                axis = [c / an for c in axis]
                loc = _vec3(cyl.Axis().Location())
                fc = _vec3(f.center())
                d = _sub(fc, loc)
                par = _dot(d, axis)
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
                par = _dot(d, ax)
                on_axis = [loc[i] + par * ax[i] for i in range(3)]
                u1, u2, v1, v2 = BRepTools.UVBounds_s(f.wrapped)
                cones.append(
                    (
                        ax,
                        loc,
                        float(cone.SemiAngle()),
                        _norm(_sub(fc, on_axis)),
                        float(cone.RefRadius()),
                        float(v1),
                        float(v2),
                    )
                )
        except Exception:
            continue
    return cyl_faces, cones


def _find_hole_cyls(
    shape: Shape, cyl_faces: list[tuple[float, list[float], list[float], float, float]]
) -> list[tuple[float, list[float], list[float], float]]:
    """Cylinders whose axis sits inside material -> bores; coaxial same-radius
    segments (a countersink/counterbore step splits one physical bore into
    several faces) merge into a single hole with summed depth."""
    holes: list[tuple[float, list[float], list[float], float]] = []
    for r, axis, center, vspan, _uspan in cyl_faces:
        try:
            if _inside_fraction(shape, center, axis, r) >= _INSIDE_FRACTION_HOLE:
                holes.append((r, axis, center, vspan))
        except Exception:
            continue
    merged: list[list[Any]] = []
    for r, axis, center, vspan in holes:
        for m in merged:
            if abs(m[0] - r) <= _GROUP_DIA_TOL and _coaxial(m[2], m[1], center, axis):
                m[3] += vspan
                break
        else:
            merged.append([r, axis, center, vspan])
    return [(m[0], m[1], m[2], m[3]) for m in merged]


def measure(shape: Any) -> dict[str, Any]:
    """Bounding box, volume, area, topology counts, face-type inventory and
    the OCC validity flag. Cheap enough to print after every op."""
    s = _as_shape(shape)
    if s is None:
        return {"error": f"not a shape: {type(shape).__name__}"}
    try:
        bb = s.bounding_box()
        faces = s.faces()
        inventory: dict[str, int] = {}
        for f in faces:
            try:
                gt = f.geom_type
                name = getattr(gt, "name", None) or str(gt).split(".")[-1]
            except Exception:
                name = "UNKNOWN"
            inventory[name] = inventory.get(name, 0) + 1
        valid_attr = getattr(s, "is_valid", None)
        is_valid = bool(valid_attr() if callable(valid_attr) else valid_attr)
        return {
            "bbox_min": [_r3(bb.min.X), _r3(bb.min.Y), _r3(bb.min.Z)],
            "bbox_max": [_r3(bb.max.X), _r3(bb.max.Y), _r3(bb.max.Z)],
            "dims": {
                "x": _r3(bb.max.X - bb.min.X),
                "y": _r3(bb.max.Y - bb.min.Y),
                "z": _r3(bb.max.Z - bb.min.Z),
            },
            "volume_mm3": _r3(float(s.volume)),
            "area_mm2": _r3(float(s.area)),
            "solids": len(s.solids()),
            "faces": len(faces),
            "edges": len(s.edges()),
            "face_types": inventory,
            "is_valid": is_valid,
        }
    except Exception as e:
        return _err(e)


def bbox_dims(shape: Any) -> dict[str, Any]:
    """Just the bounding-box size: {"x": ..., "y": ..., "z": ...} in mm."""
    s = _as_shape(shape)
    if s is None:
        return {"error": f"not a shape: {type(shape).__name__}"}
    try:
        bb = s.bounding_box()
        return {
            "x": _r3(bb.max.X - bb.min.X),
            "y": _r3(bb.max.Y - bb.min.Y),
            "z": _r3(bb.max.Z - bb.min.Z),
        }
    except Exception as e:
        return _err(e)


def find_holes(shape: Any) -> list[dict[str, Any]] | dict[str, Any]:
    """Analytic cylindrical inner faces -> one dict per physical bore:
    {diameter, axis, direction, center, through, depth}. ``through`` means the
    bore spans >= 75% of the part extent along its axis (the vspan heuristic
    from features.py — no tunnel-ray verification inside the sandbox)."""
    s = _as_shape(shape)
    if s is None:
        return {"error": f"not a shape: {type(shape).__name__}"}
    try:
        faces = s.faces()
    except Exception as e:
        return _err(e)
    if len(faces) > _FACE_CAP:
        return {"skipped": "faceted solid", "faces": len(faces)}
    try:
        cyl_faces, _cones = _scan_faces(faces)
        out: list[dict[str, Any]] = []
        for r, axis, center, vspan in _find_hole_cyls(s, cyl_faces):
            extent = _extent_along(s, axis)
            out.append(
                {
                    "diameter": _r3(2.0 * r),
                    "axis": _axis_label(axis),
                    "direction": [_r3(c) for c in axis],
                    "center": [_r3(c) for c in center],
                    "through": bool(vspan >= _THROUGH_FRACTION * extent),
                    "depth": _r3(vspan),
                }
            )
        return out
    except Exception as e:
        return _err(e)


def find_bosses(shape: Any) -> list[dict[str, Any]] | dict[str, Any]:
    """Outer cylindrical faces wrapping >= ~140deg that are NOT surrounded by
    material -> protruding bosses/tubes/pins (fillets wrap ~90deg and are
    excluded). One dict per boss: {diameter, axis, direction, center, height}."""
    s = _as_shape(shape)
    if s is None:
        return {"error": f"not a shape: {type(shape).__name__}"}
    try:
        faces = s.faces()
    except Exception as e:
        return _err(e)
    if len(faces) > _FACE_CAP:
        return {"skipped": "faceted solid", "faces": len(faces)}
    try:
        cyl_faces, _cones = _scan_faces(faces)
        out: list[dict[str, Any]] = []
        for r, axis, center, vspan, uspan in cyl_faces:
            if uspan < _BOSS_WRAP_MIN:
                continue
            try:
                if _inside_fraction(s, center, axis, r) >= _INSIDE_FRACTION_HOLE:
                    continue  # a bore, not a boss
            except Exception:
                pass
            out.append(
                {
                    "diameter": _r3(2.0 * r),
                    "axis": _axis_label(axis),
                    "direction": [_r3(c) for c in axis],
                    "center": [_r3(c) for c in center],
                    "height": _r3(vspan),
                }
            )
        return out
    except Exception as e:
        return _err(e)


def find_countersinks(shape: Any) -> list[dict[str, Any]] | dict[str, Any]:
    """Conical faces coaxial to a detected bore -> {location, axis, top_dia,
    angle_deg}. ``location`` is the centre of the cone's wide rim; ``top_dia``
    its diameter; ``angle_deg`` the semi-angle between flank and axis (a 90deg
    countersink reads ~45)."""
    s = _as_shape(shape)
    if s is None:
        return {"error": f"not a shape: {type(shape).__name__}"}
    try:
        faces = s.faces()
    except Exception as e:
        return _err(e)
    if len(faces) > _FACE_CAP:
        return {"skipped": "faceted solid", "faces": len(faces)}
    try:
        cyl_faces, cones = _scan_faces(faces)
        holes = _find_hole_cyls(s, cyl_faces)
        out: list[dict[str, Any]] = []
        for ax, loc, semi, face_r, ref_r, v1, v2 in cones:
            hole_r = next(
                (r for r, h_ax, h_c, _ in holes if _coaxial(h_c, h_ax, loc, ax)),
                None,
            )
            if hole_r is None or face_r > hole_r * 2.2 + 2.0:
                continue  # no coaxial bore, or a rim chamfer far outside it
            # radius/height at each axial bound of the cone face; the wide
            # end is the countersink mouth
            ends = sorted(
                ((v, ref_r + v * math.sin(semi)) for v in (v1, v2)),
                key=lambda e: e[1],
                reverse=True,
            )
            v_wide, r_wide = ends[0]
            rim = [loc[i] + v_wide * math.cos(semi) * ax[i] for i in range(3)]
            out.append(
                {
                    "location": [_r3(c) for c in rim],
                    "axis": _axis_label(ax),
                    "direction": [_r3(c) for c in ax],
                    "top_dia": _r3(2.0 * r_wide),
                    "angle_deg": _r3(abs(math.degrees(semi))),
                }
            )
        return out
    except Exception as e:
        return _err(e)


def cross_sections(
    shape: Any,
    axis: str = "z",
    positions: list[float] | None = None,
    count: int = 5,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Material cross-section area at positions along an axis. ``positions``
    None -> ``count`` slices evenly spaced inside the bbox (1% inset so the
    extreme faces don't produce degenerate zero-area cuts). Each entry:
    {position, area_mm2}; unresolvable slices carry area_uncertain/error."""
    s = _as_shape(shape)
    if s is None:
        return {"error": f"not a shape: {type(shape).__name__}"}
    ax = str(axis or "z").strip().lower().lstrip("+-")
    if ax not in _AXIS_INDEX:
        return {"error": f"bad axis '{axis}' (expected x, y or z)"}
    ai = _AXIS_INDEX[ax]
    try:
        bb = s.bounding_box()
        lo, hi = ((bb.min.X, bb.max.X), (bb.min.Y, bb.max.Y), (bb.min.Z, bb.max.Z))[ai]
        span = hi - lo
        if positions is None:
            count = max(int(count), 2)
            lo_s, hi_s = lo + span * 0.01, hi - span * 0.01
            step = (hi_s - lo_s) / (count - 1) if count > 1 else 0.0
            positions = [lo_s + i * step for i in range(count)]
        positions = [float(p) for p in positions][:64]
    except Exception as e:
        return _err(e)

    origin = [0.0, 0.0, 0.0]
    d = [0.0, 0.0, 0.0]
    d[ai] = 1.0
    out: list[dict[str, Any]] = []
    for pos in positions:
        origin[ai] = pos
        try:
            plane = gp_Pln(gp_Pnt(*origin), gp_Dir(*d))
            section = BRepAlgoAPI_Section(s.wrapped, plane, False)
            section.Build()
            area, resolved = _section_area(plane, section.Shape())
            rec: dict[str, Any] = {"position": _r3(pos), "area_mm2": _r3(area)}
            if not resolved:
                rec["area_uncertain"] = True
            out.append(rec)
        except Exception as e:
            out.append({"position": _r3(pos), "error": f"{type(e).__name__}: {e}"})
    return out


def _section_area(plane: gp_Pln, section_shape: Any) -> tuple[float, bool]:
    """Net material area of one slice: section wires -> planar faces, then
    even-odd nesting (a loop inside an odd number of larger loops is a void
    and subtracts; islands inside voids add back). Returns (area, resolved)."""
    edges = _HSequenceOfShape()
    exp = TopExp_Explorer(section_shape, TopAbs_EDGE)
    while exp.More():
        edges.Append(exp.Current())
        exp.Next()
    wires = _HSequenceOfShape()
    ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(edges, 1e-7, False, wires)

    loops: list[tuple[float, Any, gp_Pnt, Bnd_Box]] = []
    for j in range(1, wires.Length() + 1):
        wire = _to_wire(wires.Value(j))
        try:
            maker = BRepBuilderAPI_MakeFace(plane, wire)
            if not maker.IsDone():
                continue
            face = maker.Face()
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area = abs(props.Mass())
            if area <= 0.0:
                continue
            verts = TopExp_Explorer(wire, TopAbs_VERTEX)
            if not verts.More():
                continue
            point = BRep_Tool.Pnt_s(_to_vertex(verts.Current()))
            box = Bnd_Box()
            BRepBndLib.Add_s(face, box, False)
            loops.append((area, face, point, box))
        except Exception:
            continue

    # descending area: a loop's possible containers always come earlier
    loops.sort(key=lambda lp: lp[0], reverse=True)
    total = 0.0
    resolved = True
    for i in range(len(loops)):
        area, _face, point, box = loops[i]
        depth = 0
        for j in range(i):
            _a, outer_face, _p, outer_box = loops[j]
            if outer_box.IsOut(box):
                continue
            try:
                if BRepClass_FaceClassifier(outer_face, point, 1e-7).State() == TopAbs_IN:
                    depth += 1
            except Exception:
                resolved = False
        total += area if depth % 2 == 0 else -area
    if total < 0.0:
        return 0.0, False
    return total, resolved


def clearance(shape_a: Any, shape_b: Any) -> dict[str, Any]:
    """Minimum surface-to-surface distance between two shapes, plus a status:
    ``apart`` (gap = distance), ``touching`` (~0 distance, no overlap) or
    ``interpenetrating`` (positive overlap volume)."""
    a = _as_shape(shape_a)
    b = _as_shape(shape_b)
    if a is None or b is None:
        bad = type(shape_a).__name__ if a is None else type(shape_b).__name__
        return {"error": f"not a shape: {bad}"}
    try:
        dist = float(a.distance_to(b))
    except Exception as e:
        return _err(e)
    out: dict[str, Any] = {"distance_mm": _r3(dist)}
    if dist <= 1e-6:
        try:
            inter = a.intersect(b) or []
            overlap = float(sum(float(getattr(x, "volume", 0.0) or 0.0) for x in inter))
            out["overlap_mm3"] = _r3(overlap)
            out["status"] = "interpenetrating" if overlap > 1e-6 else "touching"
        except Exception:
            out["status"] = "touching_or_overlapping"
    else:
        out["status"] = "apart"
    return out
