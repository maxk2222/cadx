"""Deterministic feature verification: extract measurable features from a
generated shape and compare them against the spec's feature list.

Feedback-only: findings flow into the repair/critique feedback but never block
`succeeded` on their own — geometric heuristics are approximate by nature,
especially on faceted mesh solids.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
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
    centers: list[list[float]] = Field(default_factory=list)  # per-hole [x, y]
    blocked: int = 0  # bores in this group sealed by a membrane (tunnel-verified)


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
}
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


def _load_watertight_mesh(stl_path: Path):
    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    mesh.merge_vertices()
    return mesh if mesh.is_watertight else None


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


def _mesh_through_holes(stl_path: Path, mesh=None) -> tuple[list[MeasuredHoleGroup], int]:
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
        return [], 0
    mn, mx = mesh.bounds
    step = 0.7  # mm grid pitch — resolves Ø~1.6mm tunnels and larger
    groups: list[MeasuredHoleGroup] = []
    csk_hits = 0

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
            cu = float(cells[:, 0].mean())
            cv = float(cells[:, 1].mean())
            center = np.zeros(3)
            center[a1] = mn[a1] + step / 2 + cu * step
            center[a2] = mn[a2] + step / 2 + cv * step
            center[axis] = (mn[axis] + mx[axis]) / 2
            groups.append(
                MeasuredHoleGroup(
                    diameter=round(dia, 3),
                    count=1,
                    axis=f"+{'xyz'[axis]}",
                    depth_class="through",
                    centers=[[round(float(center[0]), 2), round(float(center[1]), 2)]],
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
    return groups, csk_hits


def extract_features(  # noqa: C901 — sequential measurement passes
    shape: Shape, stl_path: Path | None = None
) -> MeasuredFeatures:
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

    mesh = None
    if stl_path is not None:
        try:
            mesh = _load_watertight_mesh(stl_path)
        except Exception:
            mesh = None

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
    for r, axis, center, vspan in holes:
        dom = _dominant_axis(axis)
        ax_idx = "xyz".index(dom[-1])
        depth = "through" if vspan >= _THROUGH_FRACTION * extent[ax_idx] else "blind"
        # vspan is measured on the CUT face's parameter range — it can claim
        # "through" even when a membrane seals the bore, and it is capped by
        # LOCAL thickness so a real through bore in a dipped region always
        # looks "blind" against the global extent. Verify both directions with
        # a tunnel ray when the tessellation is available.
        if mesh is not None:
            tunnel = _ray_tunnel(mesh, center, ax_idx, r)
            if depth == "through" and not tunnel:
                depth = "blind"
            elif depth == "blind" and tunnel:
                depth = "through"
        dia = 2.0 * r
        cxy = [round(float(center[0]), 2), round(float(center[1]), 2)]
        for g in groups:
            if g.axis == dom and abs(g.diameter - dia) <= _GROUP_DIA_TOL:
                g.count += 1
                g.diameter = round((g.diameter * (g.count - 1) + dia) / g.count, 3)
                g.centers.append(cxy)
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
                    blocked=1 if depth == "blind" else 0,
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
    # on the tessellation when the B-Rep scan found nothing.
    if not holes and stl_path is not None:
        _mesh_fallback(out, stl_path, mesh)
    return out


def _mesh_fallback(out: MeasuredFeatures, stl_path: Path, mesh=None) -> None:
    try:
        m_groups, m_csk = _mesh_through_holes(stl_path, mesh)
    except Exception:
        return
    if m_groups:
        out.hole_groups = m_groups
        out.total_holes = len(m_groups)
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

    # definitive check before matching: spec wants a bore along a specific
    # axis and measured holes exist only on other axes → report the axis
    # mismatch explicitly (e.g. boss bore +z instead of spec +y), not a
    # generic "no holes" that hides orientation errors
    if want_axis and measured.hole_groups:
        want_ax = _axis_letter(want_axis)
        if not any(_axis_letter(g.axis) == want_ax for g in measured.hole_groups):
            actual_axes = ", ".join(sorted({_axis_letter(g.axis) for g in measured.hole_groups}))
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
    for g in measured.hole_groups:
        if g.axis in ("+z", "-z", ""):
            centres += [(c[0], c[1]) for c in g.centers]
    if not centres:  # any axis is better than nothing
        centres += [(c[0], c[1]) for g in measured.hole_groups for c in g.centers]
    for a in holes:
        ax, ay = float(a["x"]), float(a["y"])
        # position exactness only makes sense for anchors backed by an
        # authoritative plan-view image; mesh-only / approximate anchors are
        # self-referential (the mesh IS the part) so a deviation is minor
        src = str(a.get("source", "image"))
        exact = src in ("image", "image+mesh")
        d = min((math.hypot(ax - cx, ay - cy) for cx, cy in centres), default=None)
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
