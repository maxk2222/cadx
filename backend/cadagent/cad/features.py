"""Deterministic feature verification: extract measurable features from a
generated shape and compare them against the spec's feature list.

Feedback-only: findings flow into the repair/critique feedback but never block
`succeeded` on their own — geometric heuristics are approximate by nature,
especially on faceted mesh solids.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from build123d import GeomType, Shape
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepTools import BRepTools
from pydantic import BaseModel, Field


class MeasuredHoleGroup(BaseModel):
    diameter: float  # mm
    count: int
    axis: str = ""  # dominant axis direction: +x|-x|+y|-y|+z|-z
    depth_class: str = "through"  # through | blind | unknown


class MeasuredFeatures(BaseModel):
    hole_groups: list[MeasuredHoleGroup] = Field(default_factory=list)
    total_holes: int = 0
    countersinks: int = 0  # cones coaxial to a hole
    counterbores: int = 0  # stepped coaxial hole cylinders
    rounds: list[float] = Field(default_factory=list)  # fillet radii (edge cylinders)
    chamfer_faces: int = 0  # 45° planes + non-coaxial cones
    approximate: bool = False  # faceted/mesh source — treat findings loosely


class FeatureFinding(BaseModel):
    feature: str = ""
    expected: str = ""
    actual: str = ""
    severity: Literal["major", "minor"] = "minor"
    fix_hint: str = ""


_AXIS_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")
_SAMPLE_DIRS = 8
_INSIDE_FRACTION_HOLE = 0.7  # radial samples inside material -> hole, else round
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
_HOLE_TYPES = {"hole", "insert", "thread", "bearing_seat"}
_CSK_RE = re.compile(r"countersunk|csk|senk(?!.*zylinder)|flat.?head", re.I)
_CBORE_RE = re.compile(r"counterbore|cbore|zylindersenk|socket.?head", re.I)
_TAP_RE = re.compile(r"\btap(ped)?\b|gewinde|thread", re.I)
_M_RE = re.compile(r"\bM\s*(\d+)", re.I)
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


def _inside_fraction(shape: Shape, center: list[float], axis: list[float], r: float) -> float:
    """Fraction of radial samples (on a circle of radius ~1.35r around the axis
    point) that lie inside material. Holes score ~1.0 (wall surrounded by
    material), edge fillets score ~0.25-0.5 (open to air on most sides)."""
    # build an orthonormal basis in the plane perpendicular to the axis
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
    list[tuple[float, list[float], list[float], float]],
    list[tuple[list[float], list[float], float]],
    int,
]:
    """Walk all faces once; collect cylinders, cones and chamfer-like planes."""
    cyl_faces: list[tuple[float, list[float], list[float], float]] = []
    cones: list[tuple[list[float], list[float], float]] = []
    chamfer_planes = 0
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
                _, _, v1, v2 = BRepTools.UVBounds_s(f.wrapped)
                cyl_faces.append((float(cyl.Radius()), axis, on_axis, abs(v2 - v1)))
            elif gt == GeomType.CONE:
                cone = ad.Cone()
                ax = _vec3(cone.Axis().Direction())
                an = _norm(ax) or 1.0
                ax = [c / an for c in ax]
                cones.append(
                    (ax, _vec3(cone.Axis().Location()), abs(math.degrees(cone.SemiAngle())))
                )
            elif gt == GeomType.PLANE:
                n = _vec3(f.normal_at())
                comps = sorted(abs(c) for c in n)
                # chamfer plane: two normal components ~0.707, third ~0
                if comps[2] > 0.6 and comps[1] > 0.6 and comps[0] < 0.15:
                    chamfer_planes += 1
        except Exception:
            continue
    return cyl_faces, cones, chamfer_planes


def extract_features(shape: Shape) -> MeasuredFeatures:
    """Measure hole groups, countersinks, counterbores, fillet rounds and
    chamfer-like faces from a solid. Faceted mesh solids degrade gracefully —
    cylinders only survive if the mesh kept them near-circular."""
    out = MeasuredFeatures(approximate=False)
    bb = shape.bounding_box()
    extent = [
        float(bb.max.X - bb.min.X),
        float(bb.max.Y - bb.min.Y),
        float(bb.max.Z - bb.min.Z),
    ]

    cyl_faces, cones, chamfer_planes = _scan_faces(shape)

    # classify cylinders: hole (axis void surrounded by material) vs round (edge fillet)
    holes: list[tuple[float, list[float], list[float], float]] = []
    for r, axis, center, vspan in cyl_faces:
        try:
            frac = _inside_fraction(shape, center, axis, r)
        except Exception:
            frac = 0.0
        if frac >= _INSIDE_FRACTION_HOLE:
            holes.append((r, axis, center, vspan))
        else:
            out.rounds.append(round(r, 3))

    # cones coaxial to a hole -> countersink; others count as chamfer faces
    csk = 0
    cbore = 0
    chamfer_cones = 0
    for ax, ccenter, semi in cones:
        if any(_coaxial(h_center, h_axis, ccenter, ax) for _, h_axis, h_center, _ in holes):
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
    for r, axis, _, vspan in holes:
        dom = _dominant_axis(axis)
        ax_idx = "xyz".index(dom[-1])
        depth = "through" if vspan >= _THROUGH_FRACTION * extent[ax_idx] else "blind"
        dia = 2.0 * r
        for g in groups:
            if g.axis == dom and abs(g.diameter - dia) <= _GROUP_DIA_TOL:
                g.count += 1
                g.diameter = round((g.diameter * (g.count - 1) + dia) / g.count, 3)
                if g.depth_class != depth:
                    g.depth_class = "unknown"
                break
        else:
            groups.append(
                MeasuredHoleGroup(diameter=round(dia, 3), count=1, axis=dom, depth_class=depth)
            )

    out.hole_groups = groups
    out.total_holes = len(holes)
    out.countersinks = csk
    out.counterbores = cbore
    out.chamfer_faces = chamfer_planes + chamfer_cones
    out.rounds.sort()
    return out


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
        from cadagent.cad import catalog

        if _TAP_RE.search(std):
            return catalog.nominal_hole_diameter(size, "tap") or _TAP_DIA.get(size)
        return catalog.nominal_hole_diameter(size, "clearance") or _CLEARANCE_DIA.get(size)
    return None


def _feature_axis_hint(feature: Any) -> str | None:
    pos = f"{feature.position or ''} {feature.description or ''}".lower()
    if not _AXIS_HINT_RE.search(pos):
        return None
    if any(k in pos for k in ("top", "upper", "deck", "+z", "oben", " through", "durch")):
        return "+z"
    if "bottom" in pos or "unter" in pos or "-z" in pos:
        return "-z"
    return None


def _match_hole_group(
    groups: list[MeasuredHoleGroup], want_dia: float | None, want_axis: str | None
) -> MeasuredHoleGroup | None:
    """Best matching hole group: diameter(+axis) first, then axis, then largest."""
    for g in groups:
        if (
            want_dia
            and abs(g.diameter - want_dia) <= max(0.2 * want_dia, 0.5)
            and (want_axis is None or g.axis == want_axis)
        ):
            return g
    if want_axis:
        for g in groups:
            if g.axis == want_axis:
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

    group = _match_hole_group(measured.hole_groups, want_dia, want_axis)
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
            if isinstance(r_want, int | float) and measured.rounds:
                r_act = min(measured.rounds, key=lambda r: abs(r - float(r_want)))
                if abs(r_act - float(r_want)) > 0.5 * float(r_want):
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

    findings += _check_edge_features(spec_features, measured)

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
