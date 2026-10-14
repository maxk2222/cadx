"""Post-TopOpt feature finishing — re-cut exact functional geometry.

SIMP keeps fastener/keep regions pinned, but the extracted surface is
voxel-staircased and Taubin-smoothed: an M4 bore comes back wavy, flange
planes lose their flatness. This module re-applies precise cutter solids at
measured anchor positions via manifold booleans, and restores the ORIGINAL
surface inside keep regions by unioning back (base_mesh ∩ keep-box).

Runs in the executor (trimesh + manifold3d + numpy only — no OCC). All
booleans are best-effort per feature: a failed cut warns, never fails the
job.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from cadx.cad.fem.anchors3d import is_fastener_anchor
from cadx.logging import get_logger

log = get_logger(__name__)

# axis -> (axial index, in-plane u index, in-plane v index), same layout as
# anchors3d._AXES
_AXES: dict[str, tuple[int, int, int]] = {
    "+x": (0, 1, 2),
    "-x": (0, 1, 2),
    "+y": (1, 0, 2),
    "-y": (1, 0, 2),
    "+z": (2, 0, 1),
    "-z": (2, 0, 1),
}
_AXIS_VEC: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}

# ISO metric coarse pitch — thread_spec is preferred, this is the fallback
_COARSE_PITCH: dict[float, float] = {
    1.6: 0.35,
    2.0: 0.4,
    2.5: 0.45,
    3.0: 0.5,
    4.0: 0.7,
    5.0: 0.8,
    6.0: 1.0,
    8.0: 1.25,
    10.0: 1.5,
    12.0: 1.75,
}

# feature kinds we can re-cut subtractively (boss = additive, skipped)
_CUTABLE = {
    "hole",
    "bore",
    "tunnel",
    "thread",
    "tap",
    "insert",
    "bearing_seat",
    "bearing",
    "countersink",
    "counterbore",
}


@dataclass
class RefineFeature:
    """One feature to re-cut/restore on the optimized part."""

    name: str = ""
    kind: str = "hole"  # hole|tunnel|thread|tap|insert|bearing_seat|countersink|counterbore
    axis: str = "+z"  # bore axis; the feature opens on the `axis`-signed surface
    u: float = 0.0  # in-plane coords (see _AXES)
    v: float = 0.0
    dia: float = 4.0
    depth_mm: float | None = None  # None = through
    csk_dia: float | None = None
    csk_angle_deg: float = 90.0
    cbore_dia: float | None = None
    cbore_depth_mm: float | None = None
    thread_spec: str = ""  # "M4-0.7" — overrides dia for tap/thread cutters
    helical: bool = False  # cut a real helical thread groove (experimental, FDM)
    source: str = "anchor"  # anchor|spec|request


# --------------------------------------------------------------------------- #
# feature resolution                                                            #
# --------------------------------------------------------------------------- #


def _metric_pitch(dia: float) -> float:
    """Nearest standard metric coarse pitch for a nominal diameter."""
    best = min(_COARSE_PITCH, key=lambda d: abs(d - dia))
    return _COARSE_PITCH[best]


def _parse_spec(spec: str) -> tuple[float, float] | None:
    """'M4-0.7' -> (4.0, 0.7); 'M3-0.5-6' -> (3.0, 0.5)."""
    import re

    m = re.search(r"M(\d+(?:\.\d+)?)[-x](\d+(?:\.\d+)?)", spec.strip(), re.I)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _thread_spec_from(sf: dict[str, Any], dia: float) -> str:
    """Resolve a fastener spec string from Feature.standard/name/description.

    Accepts 'M4-0.7' verbatim; a bare 'M4' gets the coarse pitch for its
    diameter. Returns '' when nothing looks like a metric size."""
    import re

    for f_name in ("standard", "name", "description"):
        txt = str(sf.get(f_name) or "")
        parsed = _parse_spec(txt)
        if parsed:
            return f"M{parsed[0]:g}-{parsed[1]:g}"
        m = re.search(r"\bM(\d+(?:\.\d+)?)\b", txt, re.I)
        if m:
            d = float(m.group(1))
            return f"M{d:g}-{_metric_pitch(d):g}"
    if dia > 0:
        nearest = min(_COARSE_PITCH, key=lambda d: abs(d - dia))
        if abs(nearest - dia) <= 0.6:
            return f"M{nearest:g}-{_COARSE_PITCH[nearest]:g}"
    return ""


def _feature_axis(f: dict[str, Any]) -> str:
    """Normalize a spec Feature.axis ('x'|'+x'|'z'...) to signed axis."""
    a = str(f.get("axis") or "").strip().lower()
    if a in ("x", "y", "z"):
        a = "+" + a
    return a if a in _AXES else "+z"


def _anchor_uv(a: dict[str, Any]) -> tuple[float, float] | None:
    """In-plane (u, v) of an anchor on its own axis."""
    pos = a.get("pos") or []
    axis = _feature_axis(a)
    _ax, au, av = _AXES[axis]
    try:
        u = pos[au] if len(pos) > au else None
        v = pos[av] if len(pos) > av else None
    except (TypeError, IndexError):
        return None
    if u is None or v is None:
        # legacy 2D anchors only carry x/y
        if axis in ("+z", "-z"):
            u, v = a.get("x"), a.get("y")
        if u is None or v is None:
            return None
    return float(u), float(v)


def _upgrade_kind(anchor_kind: str, spec_type: str | None, spec_dims: dict[str, Any]) -> str:
    """Spec feature type wins over the scanned 'hole' kind."""
    t = (spec_type or "").lower()
    if t in ("thread", "tap", "insert", "bearing_seat", "countersink", "counterbore"):
        return t
    if spec_dims.get("countersink_diameter"):
        return "countersink"
    if spec_dims.get("counterbore_diameter"):
        return "counterbore"
    return anchor_kind if anchor_kind in _CUTABLE else "hole"


def resolve_refine_features(  # noqa: C901
    anchors: list[dict[str, Any]] | None,
    spec_features: list[dict[str, Any]] | None,
    request_features: list[dict[str, Any]] | None,
    *,
    match_tol_mm: float = 3.0,
) -> tuple[list[RefineFeature], list[str]]:
    """Merge request > spec > scanned anchors into a cutter list.

    Request entries may carry `anchor_index` (1-based into `anchors`) or a
    full geometric description. Spec features upgrade a nearby anchor's kind
    and contribute depth/counter/thread dims. Unmatched fastener anchors are
    re-cut as plain holes at their measured diameter.
    """
    feats: list[RefineFeature] = []
    warnings: list[str] = []
    anchors = [a for a in (anchors or []) if is_fastener_anchor(a)]
    spec_features = spec_features or []

    def _nearest_anchor(u: float, v: float, axis: str) -> dict[str, Any] | None:
        best, best_d = None, match_tol_mm
        for a in anchors:
            if _feature_axis(a) != axis:
                continue
            uv = _anchor_uv(a)
            if uv is None:
                continue
            d = math.hypot(uv[0] - u, uv[1] - v)
            if d < best_d:
                best, best_d = a, d
        return best

    def _from_anchor(
        a: dict[str, Any], kind: str | None = None, dims: dict | None = None
    ) -> RefineFeature | None:
        uv = _anchor_uv(a)
        if uv is None:
            return None
        dims = dims or {}
        dia = float(dims.get("diameter") or dims.get("bore_diameter") or a.get("dia") or 0)
        if dia <= 0:
            return None
        return RefineFeature(
            name=str(a.get("name") or f"anchor {a.get('index', '?')}"),
            kind=_upgrade_kind(kind or str(a.get("kind") or "hole"), None, dims),
            axis=_feature_axis(a),
            u=uv[0],
            v=uv[1],
            dia=dia,
            depth_mm=dims.get("depth") or (a.get("depth_mm") or None),
            csk_dia=dims.get("countersink_diameter"),
            csk_angle_deg=float(dims.get("countersink_angle") or 90.0),
            cbore_dia=dims.get("counterbore_diameter"),
            cbore_depth_mm=dims.get("counterbore_depth"),
            thread_spec=str(dims.get("thread_spec") or ""),
        )

    # --- explicit request selection -----------------------------------------
    if request_features:
        for r in request_features:
            r = dict(r)
            idx = r.pop("anchor_index", None)
            base: dict[str, Any] = {}
            if idx is not None and 1 <= int(idx) <= len(anchors):
                base = dict(anchors[int(idx) - 1])
            merged = {**base, **{k: v for k, v in r.items() if v is not None}}
            axis = _feature_axis(merged)
            uv = _anchor_uv(merged) or (None, None)
            if uv[0] is None:
                # request may carry raw in-plane coords
                _ax, au, av = _AXES[axis]
                p = merged.get("pos") or []
                if len(p) > max(au, av) and p[au] is not None and p[av] is not None:
                    uv = (float(p[au]), float(p[av]))
                elif axis in ("+z", "-z") and merged.get("x") is not None:
                    uv = (float(merged["x"]), float(merged["y"]))
            if uv[0] is None:
                warnings.append(
                    f"refine feature '{merged.get('name') or idx}' has no position — skipped"
                )
                continue
            dia = float(
                merged.get("dia") or merged.get("diameter") or merged.get("bore_diameter") or 0
            )
            if dia <= 0:
                warnings.append(
                    f"refine feature '{merged.get('name') or idx}' has no diameter — skipped"
                )
                continue
            feats.append(
                RefineFeature(
                    name=str(merged.get("name") or f"feature {idx or len(feats) + 1}"),
                    kind=str(merged.get("kind") or merged.get("type") or "hole"),
                    axis=axis,
                    u=float(uv[0]),
                    v=float(uv[1]),
                    dia=dia,
                    depth_mm=merged.get("depth_mm") or merged.get("depth"),
                    csk_dia=merged.get("csk_dia") or merged.get("countersink_diameter"),
                    csk_angle_deg=float(
                        merged.get("csk_angle_deg") or merged.get("countersink_angle") or 90.0
                    ),
                    cbore_dia=merged.get("cbore_dia") or merged.get("counterbore_diameter"),
                    cbore_depth_mm=merged.get("cbore_depth_mm") or merged.get("counterbore_depth"),
                    thread_spec=str(merged.get("thread_spec") or merged.get("standard") or ""),
                    helical=bool(merged.get("helical") or merged.get("helical_thread")),
                    source="request",
                )
            )
        return feats, warnings

    # --- auto: anchors upgraded by nearby spec features ----------------------
    used_anchors: set[int] = set()
    for sf in spec_features:
        stype = str(sf.get("type") or "").lower()
        if (
            stype not in ("hole", "thread", "insert", "bearing_seat")
            and not (sf.get("dims") or {}).get("countersink_diameter")
            and not (sf.get("dims") or {}).get("counterbore_diameter")
        ):
            continue
        dims = dict(sf.get("dims") or {})
        dia = float(dims.get("diameter") or dims.get("bore_diameter") or 0)
        axis = _feature_axis(sf)
        anchor = None
        # spec features have no numeric position — match by diameter+axis
        # against unclaimed anchors (closest dia wins)
        cands = [a for a in anchors if id(a) not in used_anchors and _feature_axis(a) == axis]
        if not cands:
            cands = [a for a in anchors if id(a) not in used_anchors]
        if cands and dia > 0:
            anchor = min(cands, key=lambda a: abs(float(a.get("dia") or 0) - dia))
        elif cands:
            anchor = cands[0]
        if anchor is None:
            continue
        used_anchors.add(id(anchor))
        dims["thread_spec"] = _thread_spec_from(sf, dia)
        f = _from_anchor(anchor, kind=stype, dims=dims)
        if f:
            f.name = str(sf.get("name") or f.name)
            f.source = "spec"
            feats.append(f)

    for a in anchors:
        if id(a) in used_anchors:
            continue
        f = _from_anchor(a)
        if f:
            feats.append(f)

    return feats, warnings


# --------------------------------------------------------------------------- #
# cutter construction                                                           #
# --------------------------------------------------------------------------- #


def _entry_point(mesh: trimesh.Trimesh, axis: str, u: float, v: float) -> float | None:
    """Surface coordinate along `axis` where a ray from outside first hits.

    The feature opens on the signed side of the axis; we cast from beyond the
    bounds inward and take the first hit — that is the entry plane."""
    ax, au, av = _AXES[axis]
    dvec = -_AXIS_VEC[axis]  # inward
    origin = np.zeros(3)
    origin[au], origin[av] = u, v
    origin[ax] = mesh.bounds[1][ax] + 5.0 if axis[0] == "+" else mesh.bounds[0][ax] - 5.0
    locs, _, _ = mesh.ray.intersects_location(
        ray_origins=np.array([origin]),
        ray_directions=np.array([dvec]),
        multiple_hits=True,
    )
    if not len(locs):
        return None
    return float(locs[:, ax].max() if axis[0] == "+" else locs[:, ax].min())


def _effective_dia(feat: RefineFeature) -> float:
    """Diameter actually cut — tap-drill (D - pitch) for threads."""
    if feat.kind in ("thread", "tap"):
        parsed = _parse_spec(feat.thread_spec)
        return (parsed[0] - parsed[1]) if parsed else max(feat.dia * 0.85, feat.dia - 1.0)
    return feat.dia


def _thread_groove_solid(r_in: float, r_out: float, pitch: float, depth: float) -> trimesh.Trimesh:
    """Helical V-groove solid for a metric internal thread.

    Meridian cross-section (r-z plane at each helix station): triangle with
    apex at the major radius and base spanning ~0.55·pitch at the minor
    radius. Stations step ~15°; caps close the tube. Built along +Z starting
    at local z=0, going -Z (same convention as _cutter_mesh)."""
    turns = depth / pitch + 1.5  # overshoot both ends
    n_stat = max(24, int(turns * 24))
    thetas = np.linspace(0, -2 * np.pi * turns, n_stat)
    half_w = pitch * 0.28  # 30° flank half-width
    r_in = max(r_in, 0.05)
    r_out = max(r_out, r_in + 0.1)
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for th in thetas:
        c, s = math.cos(th), math.sin(th)
        z_c = th / (2 * np.pi) * pitch
        verts += [
            [r_in * c, r_in * s, z_c - half_w],
            [r_in * c, r_in * s, z_c + half_w],
            [r_out * c, r_out * s, z_c],
        ]
    for i in range(n_stat - 1):
        a, b = 3 * i, 3 * (i + 1)
        faces += [[a, b, b + 1], [a, b + 1, a + 1]]  # inner wall
        faces += [[a + 1, b + 1, b + 2], [a + 1, b + 2, a + 2]]  # flank up
        faces += [[a + 2, b + 2, b], [a + 2, b, a]]  # flank down
    faces.append([0, 1, 2])  # caps
    li = 3 * (n_stat - 1)
    faces.append([li + 2, li + 1, li])
    m = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
    if m.volume < 0:
        m.invert()
    return m


def _cutter_mesh(feat: RefineFeature, entry: float, span: float) -> trimesh.Trimesh:
    """Cutting solid for one feature, built with bore axis along +Z, entry at
    local z=0, bore going -Z. Caller rotates/translates into place."""
    over_out = 2.0  # overshoot above the entry face
    depth = feat.depth_mm if feat.depth_mm is not None else span + 2.0
    dia = _effective_dia(feat)
    tools: list[trimesh.Trimesh] = []
    cyl = trimesh.creation.cylinder(radius=dia / 2.0, height=depth + over_out, sections=48)
    cyl.apply_translation([0, 0, -depth / 2.0 + over_out / 2.0])
    tools.append(cyl)
    if feat.kind in ("thread", "tap") and feat.helical:
        # real helical thread: pilot bore (tap drill) + helical V-groove out
        # to the major diameter — printable FDM thread, no tapping needed
        parsed = _parse_spec(feat.thread_spec)
        d_maj, pitch = parsed if parsed else (feat.dia, _metric_pitch(feat.dia))
        groove = _thread_groove_solid(
            r_in=dia / 2.0 - 0.05,
            r_out=d_maj / 2.0 + 0.15,  # small FDM clearance on the major dia
            pitch=pitch,
            depth=depth,
        )
        groove.apply_translation([0, 0, over_out * 0.5])
        tools.append(groove)
    if feat.kind == "insert":
        # heat-set pocket: slight taper, a bit larger than the pilot bore —
        # anchor dia for inserts is the INSERT's outer pocket dia
        pass  # the straight cylinder already matches InsertHole intent
    if feat.kind == "countersink" and feat.csk_dia and feat.csk_dia > dia:
        h = (feat.csk_dia - dia) / 2.0 / math.tan(math.radians(feat.csk_angle_deg) / 2.0)
        if h > 0.01:
            cone = trimesh.creation.cone(radius=feat.csk_dia / 2.0 + 0.3, height=h + 0.5)
            cone.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
            cone.apply_translation([0, 0, (h + 0.5) / 2.0 - h])
            tools.append(cone)
    if feat.kind == "counterbore" and feat.cbore_dia and feat.cbore_dia > dia:
        cd = feat.cbore_depth_mm or (feat.cbore_dia - dia) / 2.0
        bore = trimesh.creation.cylinder(
            radius=feat.cbore_dia / 2.0, height=cd + over_out, sections=48
        )
        bore.apply_translation([0, 0, -cd / 2.0 + over_out / 2.0])
        tools.append(bore)
    return trimesh.util.concatenate(tools)


def _measure_feature_dia(
    mesh: trimesh.Trimesh, feat: RefineFeature, entry: float, span: float
) -> float | None:
    """Measured bore diameter after the cut — cross-section rays through the
    axis point at ~half the cut depth. Returns None when the rays escape
    (e.g. the cut merged into a larger void or channel)."""
    ax, au, av = _AXES[feat.axis]
    inward = -1.0 if feat.axis[0] == "+" else 1.0
    depth = min(feat.depth_mm or span, span)
    origin = np.zeros(3)
    origin[au], origin[av] = feat.u, feat.v
    origin[ax] = entry + inward * depth * 0.5
    diams: list[float] = []
    for d_ax in (au, av):
        total = 0.0
        for sgn in (1.0, -1.0):
            d = np.zeros(3)
            d[d_ax] = sgn
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins=np.array([origin]),
                ray_directions=np.array([d]),
                multiple_hits=False,
            )
            if not len(locs):
                total = np.inf
                break
            total += float(np.linalg.norm(locs[0] - origin))
        diams.append(total)
    out = min(diams)
    return out if np.isfinite(out) else None


def _orient_cutter(cutter: trimesh.Trimesh, axis: str, u: float, v: float, entry: float) -> None:
    """Map local +Z onto the outward axis direction and seat entry at z=0."""
    ax, au, av = _AXES[axis]
    dvec = _AXIS_VEC[axis]  # outward = entry side
    R = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), dvec)
    if R is not None:
        cutter.apply_transform(R)
    target = np.zeros(3)
    target[au], target[av] = u, v
    target[ax] = entry
    cutter.apply_translation(target - np.zeros(3))


# --------------------------------------------------------------------------- #
# keep-region restore                                                           #
# --------------------------------------------------------------------------- #


def detect_plane_regions(
    base_mesh: trimesh.Trimesh,
    *,
    min_area_frac: float = 0.08,
    margin_mm: float = 1.6,
    bin_mm: float = 1.0,
) -> tuple[list[np.ndarray], list[str]]:
    """Auto-detect planar faces worth restoring (C5).

    Axis-aligned faces dominate printed parts (walls, decks, flanges). For
    each of the 6 axis directions, cluster face centroids along that axis in
    `bin_mm` bins; bins holding ≥ `min_area_frac` of the total surface area
    become thin keep-slabs restored from the base mesh. Returns
    (boxes, warnings)."""
    warnings: list[str] = []
    boxes: list[np.ndarray] = []
    if not len(base_mesh.faces):
        return boxes, warnings
    normals = base_mesh.face_normals
    centers = base_mesh.triangles_center
    areas = np.asarray(base_mesh.area_faces, dtype=float)
    total = float(areas.sum())
    if total <= 0:
        return boxes, warnings
    lo, hi = np.asarray(base_mesh.bounds, dtype=float)
    for ax in range(3):
        for sgn in (1.0, -1.0):
            sel = normals[:, ax] * sgn > 0.95
            if not sel.any() or areas[sel].sum() / total < min_area_frac:
                continue
            pos = centers[sel, ax]
            w = areas[sel]
            edges = np.arange(math.floor(pos.min()), pos.max() + 2 * bin_mm, bin_mm)
            hist, eb = np.histogram(pos, bins=edges, weights=w)
            for i, h in enumerate(hist):
                if h / total < min_area_frac:
                    continue
                in_bin = sel.copy()
                in_bin[sel] = (pos >= eb[i]) & (pos < eb[i + 1])
                plane_at = float(np.average(centers[in_bin, ax], weights=areas[in_bin]))
                b = np.stack([lo - 0.5, hi + 0.5])
                b[0][ax] = plane_at - margin_mm
                b[1][ax] = plane_at + margin_mm
                boxes.append(b)
    if boxes:
        warnings.append(f"auto plane-restore: {len(boxes)} dominant planar face(s) restored")
    return boxes, warnings


def _region_box(region: dict[str, Any], bounds: np.ndarray) -> np.ndarray | None:
    """World-space [min,max] box for a keep region (conservative superset)."""
    lo = np.array(bounds[0], dtype=float) - 0.5
    hi = np.array(bounds[1], dtype=float) + 0.5
    kind = str(region.get("kind") or "face")
    axis = str(region.get("axis") or "")
    margin = float(region.get("margin_mm") or 2.0)

    if kind == "all":
        return np.stack([lo, hi])
    if kind in ("face", "surface") and axis in _AXES:
        ax, _, _ = _AXES[axis]
        b = np.stack([lo.copy(), hi.copy()])
        if axis[0] == "+":
            b[0][ax] = bounds[1][ax] - margin
        else:
            b[1][ax] = bounds[0][ax] + margin
        return b
    if kind == "area" and axis in ("+z", "-z"):
        b = np.stack([lo.copy(), hi.copy()])
        if region.get("x_range"):
            b[0][0], b[1][0] = region["x_range"][0] - margin, region["x_range"][1] + margin
        if region.get("y_range"):
            b[0][1], b[1][1] = region["y_range"][0] - margin, region["y_range"][1] + margin
        if region.get("polygon"):
            poly = np.asarray(region["polygon"], dtype=float)
            b[0][0], b[0][1] = poly[:, 0].min() - margin, poly[:, 1].min() - margin
            b[1][0], b[1][1] = poly[:, 0].max() + margin, poly[:, 1].max() + margin
        return b
    if kind == "cylinder" and axis in _AXES:
        ax, au, av = _AXES[axis]
        c = region.get("center") or []
        d = float(region.get("diameter_mm") or 0) / 2.0 + margin
        b = np.stack([lo.copy(), hi.copy()])
        if len(c) >= 3 and d > 0:
            b[0][au], b[1][au] = c[au] - d, c[au] + d
            b[0][av], b[1][av] = c[av] - d, c[av] + d
            return b
        return None
    if kind == "box" and region.get("center"):
        c = np.asarray(region["center"], dtype=float)[:3]
        d = float(region.get("diameter_mm") or 2 * margin) / 2.0 + margin
        return np.stack([c - d, c + d])
    if kind in ("hole", "anchor", "bore") and region.get("hole_index") is not None:
        return None  # resolved per-anchor by caller
    return None


def _anchor_keep_box(
    anchor: dict[str, Any], margin: float, span_axis: tuple[float, float]
) -> np.ndarray | None:
    """Box around an anchor: the bore channel plus `margin` around it."""
    uv = _anchor_uv(anchor)
    if uv is None:
        return None
    axis = _feature_axis(anchor)
    ax, au, av = _AXES[axis]
    r = float(anchor.get("dia") or 0) / 2.0 + margin
    lo = np.full(3, -np.inf)
    hi = np.full(3, np.inf)
    lo[au], hi[au] = uv[0] - r, uv[0] + r
    lo[av], hi[av] = uv[1] - r, uv[1] + r
    lo[ax], hi[ax] = span_axis
    return np.stack([lo, hi])


def _box_mesh(box: np.ndarray) -> trimesh.Trimesh:
    ext = np.asarray(box[1] - box[0], dtype=float)
    m = trimesh.creation.box(extents=np.maximum(ext, 0.05))
    m.apply_translation((box[0] + box[1]) / 2.0)
    return m


def restore_keep_surface(
    opt_mesh: trimesh.Trimesh,
    base_mesh: trimesh.Trimesh,
    keep_regions: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    *,
    margin_mm: float = 1.6,
) -> tuple[trimesh.Trimesh, list[str], int]:
    """Union back the ORIGINAL surface inside each keep region.

    `opt_mesh union (base_mesh ∩ grown_keep_box)` — Taubin smoothing shrinks kept
    material; this restores exact bore walls and planar faces. Returns
    (mesh, warnings, restored_count).

    Robustness: both operands must be manifold volumes for the booleans. A
    non-volume base or a sliver patch is repaired once via _ensure_watertight;
    if it still is not a volume the region is skipped with a warning instead
    of aborting the whole restore."""
    from cadx.cad.mesh_ops import _ensure_watertight

    warnings: list[str] = []
    out = opt_mesh
    restored = 0
    base = base_mesh
    if not base.is_volume:
        with contextlib.suppress(Exception):
            base = _ensure_watertight(base)
    if not base.is_volume:
        warnings.append("keep-region restore skipped — base mesh is not a volume")
        return out, warnings, 0
    span = (float(base.bounds[0].min()) - 1, float(base.bounds[1].max()) + 1)
    for b in _keep_boxes(keep_regions or [], anchors, base.bounds, span, margin_mm):
        try:
            patch = base.intersection(_box_mesh(b), engine="manifold")
            if patch is None or not len(getattr(patch, "faces", [])):
                continue
            if not patch.is_volume:
                # thin sliver / open shell — try one repair, else skip
                with contextlib.suppress(Exception):
                    patch = _ensure_watertight(patch)
            if not patch.is_volume:
                warnings.append("keep-region patch is not a volume — region left smoothed")
                continue
            if not out.is_volume:
                warnings.append("optimized mesh lost volume-ness — restore aborted")
                break
            merged = out.union(patch, engine="manifold")
            if merged is None or not len(getattr(merged, "faces", [])):
                warnings.append("keep-region union produced empty mesh — region left smoothed")
                continue
            out = merged
            restored += 1
        except Exception as e:
            warnings.append(f"keep-region restore failed ({e}) — region left smoothed")
    return out, warnings, restored


def _keep_boxes(
    keep_regions: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    bounds: np.ndarray,
    span: tuple[float, float],
    margin_mm: float,
) -> list[np.ndarray]:
    """Resolve keep regions (incl. `_plane_box` auto-restore) to grown AABBs."""
    boxes: list[np.ndarray] = []
    for r in keep_regions:
        if r.get("kind") == "_plane_box" and r.get("box") is not None:
            boxes.append(np.asarray(r["box"], dtype=float))
            continue
        hi_idx = r.get("hole_index")
        if str(r.get("kind")) in ("hole", "anchor", "bore") and hi_idx is not None:
            idx = int(hi_idx)
            if 1 <= idx <= len(anchors):
                b = _anchor_keep_box(anchors[idx - 1], float(r.get("margin_mm") or margin_mm), span)
                if b is not None:
                    boxes.append(b)
            continue
        b = _region_box(r, bounds)
        if b is not None:
            boxes.append(b)
    for b in boxes:
        b[0] -= margin_mm * 0.5
        b[1] += margin_mm * 0.5
    return boxes


# --------------------------------------------------------------------------- #
# main entry                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class RefineReport:
    applied: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    restored_regions: int = 0
    warnings: list[str] = field(default_factory=list)


def apply_refine(
    opt_mesh: trimesh.Trimesh,
    features: list[RefineFeature],
    *,
    base_mesh: trimesh.Trimesh | None = None,
    keep_regions: list[dict[str, Any]] | None = None,
    anchors: list[dict[str, Any]] | None = None,
    restore_keeps: bool = True,
) -> tuple[trimesh.Trimesh, RefineReport]:
    """Re-cut `features` on `opt_mesh` via manifold difference; restore keep
    surfaces from `base_mesh`. Never raises on a single bad feature."""
    report = RefineReport()
    from cadx.cad.mesh_ops import _ensure_watertight

    out = opt_mesh
    if not (out.is_watertight and out.is_winding_consistent):
        try:
            out = _ensure_watertight(out)
        except Exception as e:
            report.warnings.append(f"mesh not watertight — refine skipped: {e}")
            return out, report

    # restore FIRST so re-cuts stay crisp — a restored keep patch may overlap
    # a feature footprint; the cutter must win (exact bore > base surface)
    if restore_keeps and base_mesh is not None:
        try:
            out, w, n = restore_keep_surface(out, base_mesh, keep_regions or [], anchors or [])
            report.warnings += w
            report.restored_regions = n
        except Exception as e:
            report.warnings.append(f"keep-region restore failed: {e}")

    for feat in features:
        if feat.kind not in _CUTABLE:
            report.skipped.append({"name": feat.name, "kind": feat.kind, "reason": "not cuttable"})
            continue
        try:
            ax, _, _ = _AXES[feat.axis]
            entry = _entry_point(out, feat.axis, feat.u, feat.v)
            if entry is None:
                # ray at the bore centre hits nothing — the channel is already
                # open (tunnel through the part) OR the region was fully
                # optimized away. Re-cutting the wavy channel walls is still
                # the point of refine, so seat the cutter at the bbox face and
                # let the boolean decide; warn so the report stays honest.
                entry = float(out.bounds[1][ax] if feat.axis[0] == "+" else out.bounds[0][ax])
                report.warnings.append(
                    f"feature '{feat.name}': no entry surface — channel already "
                    "open or optimized away; cutting anyway"
                )
            span = float(out.bounds[1][ax] - out.bounds[0][ax])
            cutter = _cutter_mesh(feat, entry, span)
            _orient_cutter(cutter, feat.axis, feat.u, feat.v, entry)
            # overlap sanity: AABBs must intersect (cheap) — a fully disjoint
            # cutter would boolean fine but remove nothing
            cb = cutter.bounds
            if cb is None or np.any(cb[0] >= out.bounds[1]) or np.any(cb[1] <= out.bounds[0]):
                report.skipped.append(
                    {"name": feat.name, "kind": feat.kind, "reason": "cutter misses the part"}
                )
                continue
            res = out.difference(cutter, engine="manifold")
            if res is None or not len(getattr(res, "faces", [])):
                report.skipped.append(
                    {"name": feat.name, "kind": feat.kind, "reason": "boolean produced empty mesh"}
                )
                continue
            out = res
            entry_rec: dict[str, Any] = {
                "name": feat.name,
                "kind": feat.kind,
                "axis": feat.axis,
                "dia": round(feat.dia, 2),
                "depth_mm": feat.depth_mm,
                "source": feat.source,
            }
            try:
                requested = _effective_dia(feat)
                measured = _measure_feature_dia(out, feat, entry, span)
                entry_rec["requested_dia_mm"] = round(requested, 3)
                if measured is not None:
                    entry_rec["measured_dia_mm"] = round(measured, 3)
                    dev = abs(measured - requested)
                    entry_rec["dia_deviation_mm"] = round(dev, 3)
                    if dev > max(0.25, requested * 0.05):
                        report.warnings.append(
                            f"feature '{feat.name}': measured dia {measured:.2f} mm "
                            f"deviates {dev:.2f} mm from requested {requested:.2f} mm"
                        )
            except Exception:
                pass  # measurement is informational — never fail a cut on it
            report.applied.append(entry_rec)
        except Exception as e:
            report.skipped.append(
                {"name": feat.name, "kind": feat.kind, "reason": f"cut failed: {e}"}
            )
            report.warnings.append(f"feature '{feat.name}' cut failed: {e}")

    # cleanup pass — booleans can leave slivers
    try:
        out.update_faces(out.nondegenerate_faces())
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out, report
