"""Reconstruction core: watertight trimesh → classified parametric build123d source.

A generated/processed mesh that is provably a lathe part or a prism is
rebuilt as contract-conformant build123d code (``PARAMS`` dict + ``result``
assignment), yielding a clean parametric B-Rep instead of a faceted solid.
``"organic"`` shapes return ``None`` — they stay on the mesh path.

Public interface (mocked by the executor/orchestrator wiring)::

    Reconstruction(kind, code, params, params_schema, confidence)
    reconstruct_mesh(mesh, spec, analysis) -> Reconstruction | None
    chamfer_mm(a, b, n=20000) -> float
    snap_grid(v, grid=0.5, tol_rel=0.06) -> float

``spec`` is a ``cadx.llm.schemas.PartSpec``-shaped object (duck-typed — only
``.name/.summary/.symmetry/.orientation_note/.features`` and the three
``.length_x/.width_y/.height_z`` dims are read). ``analysis`` may carry the
executor's ``bbox_size``/``circularity`` hints; both are optional, the mesh
is re-measured either way.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel

from cadx.cad import mesh_to_code
from cadx.cad.mesh_to_code import ProfileSeg, ProfileVertex
from cadx.llm.schemas import ParamDef
from cadx.logging import get_logger

log = get_logger(__name__)

Kind = Literal["revolve", "prism", "composite", "organic"]

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
_RADIAL_IDX = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
_SPEC_DIM = {"x": "length_x", "y": "width_y", "z": "height_z"}

# spec wording implying a rotational body — regex ported from
# cadx.agent.mesh_score.spec_suggests_revolve (importing the agent layer from
# cad/ would be a layering violation / import cycle; keep them in sync).
_ROUND_RE = re.compile(
    r"round|rund|zylindrisch|cylindrical|rotation|revolve|dreh|turned|lathe|kreisförmig",
    re.IGNORECASE,
)

_N_STATIONS = 24  # profile stations on the revolve path
_N_CHECK = 13  # slices for classify_part's circularity gate (>10 per contract)
_CIRC_MIN = 0.9  # per-station circularity threshold (4πA/P²)
_ARC_MAX_RESID_MM = 0.15  # circle-fit residual budget for profile arcs
_EPS_R = 0.05  # profile radii below this collapse onto the rotation axis
_MAX_BBOX_PAIR_DEV = 0.2  # radial bbox extents of a revolve must match (±20%)


class Reconstruction(BaseModel):
    """A classified, parametric rebuild of a mesh."""

    kind: Kind
    code: str  # contract-conformant build123d source (PARAMS + result)
    params: dict[str, float]
    params_schema: list[dict[str, Any]]  # ParamDef.model_dump() form
    confidence: float  # 0..1
    meta: dict[str, Any] = {}  # measured section stats for acceptance gating


# --------------------------------------------------------------------------- #
# small public helpers                                                        #
# --------------------------------------------------------------------------- #


def snap_grid(v: float, grid: float = 0.5, tol_rel: float = 0.06) -> float:
    """Snap ``v`` to the nearest ``grid`` multiple when within ``tol_rel``.

    9.98 → 10.0 (error 0.2 %), 1.26 stays 1.26 (nearest grid point 1.5 is
    19 % off). Non-snapping values pass through unchanged — a param that
    cannot snap keeps its measured value rather than distorting the part.
    """
    if grid <= 0 or v == 0:
        return float(v)
    snapped = round(v / grid) * grid
    return float(snapped) if abs(v - snapped) <= tol_rel * abs(v) else float(v)


def chamfer_mm(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 20000) -> float:
    """Symmetric mean surface distance in mm: max(mean a→b, mean b→a)."""
    return _chamfer(a, b, n, seed=None)


def _chamfer(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int, seed: int | None) -> float:
    """Point-set Chamfer distance; ``seed`` makes internal comparisons
    deterministic (axis pick must not flip on sampling noise)."""
    pts_a, _ = trimesh.sample.sample_surface(a, n, seed=seed)
    pts_b, _ = trimesh.sample.sample_surface(b, n, seed=None if seed is None else seed + 1)
    try:
        from scipy.spatial import cKDTree

        d_ab = float(np.mean(cKDTree(pts_b).query(pts_a)[0]))
        d_ba = float(np.mean(cKDTree(pts_a).query(pts_b)[0]))
    except Exception:  # scipy is only a transitive dep — trimesh fallback
        d_ab = float(np.mean(trimesh.proximity.closest_point(b, pts_a)[1]))
        d_ba = float(np.mean(trimesh.proximity.closest_point(a, pts_b)[1]))
    return max(d_ab, d_ba)


# --------------------------------------------------------------------------- #
# sections                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class _Section:
    polygon: Any | None  # largest shapely polygon of the slice (plane-local)
    n_polys: int
    centroid: tuple[float, float] | None  # polygon centroid, radial world coords
    uv: np.ndarray  # (N,2) radial world coords of all slice vertices
    to_3d: np.ndarray | None = None  # trimesh transform: plane-local 2D → 3D
    inner_uv: tuple[float, float] | None = None  # inner-ring centroid, radial world


def _section_at(mesh: trimesh.Trimesh, axis: str, t: float) -> _Section | None:
    """Slice ``mesh`` with a plane ⟂ ``axis`` at ``t``."""
    i = _AXIS_IDX[axis]
    j, k = _RADIAL_IDX[axis]
    origin = np.zeros(3)
    origin[i] = t
    normal = np.zeros(3)
    normal[i] = 1.0
    try:
        path = mesh.section(plane_normal=normal, plane_origin=origin)
    except Exception:
        return None
    if path is None or len(path.vertices) < 3:
        return None
    polys: list[Any] = []
    to_3d = None
    try:
        p2, to_3d = path.to_2D()
        polys = [g for g in p2.polygons_full if g.area > 1e-9]
    except Exception:
        pass
    poly = max(polys, key=lambda g: g.area) if polys else None
    centroid: tuple[float, float] | None = None
    inner_uv: tuple[float, float] | None = None
    if poly is not None and to_3d is not None:
        # trimesh's to_2D centres the slice on its own origin — map the 2D
        # centroid back to 3D to get the station's axis point in world coords
        c = poly.centroid
        c3 = (np.asarray(to_3d) @ np.array([c.x, c.y, 0.0, 1.0]))[:3]
        centroid = (float(c3[j]), float(c3[k]))
        try:
            if poly.interiors:
                from shapely.geometry import Polygon

                inner = max(poly.interiors, key=lambda r: abs(Polygon(r).area))
                ic = Polygon(inner).centroid
                c3i = (np.asarray(to_3d) @ np.array([ic.x, ic.y, 0.0, 1.0]))[:3]
                inner_uv = (float(c3i[j]), float(c3i[k]))
        except Exception:
            pass
    uv = np.asarray(path.vertices, dtype=float)[:, (j, k)]
    return _Section(poly, len(polys), centroid, uv, to_3d, inner_uv)


def _circularity(poly: Any) -> float:
    """4πA/P² of the polygon's EXTERIOR ring (an axial bore stays a revolve)."""
    try:
        from shapely.geometry import Polygon

        ext = Polygon(poly.exterior.coords)
        per = float(ext.length)
        if per <= 0 or ext.area <= 0:
            return 0.0
        return float(4.0 * math.pi * ext.area / per**2)
    except Exception:
        return 0.0


def _station_circularity(
    mesh: trimesh.Trimesh, axis: str, n: int = _N_CHECK
) -> tuple[float, float]:
    """(fraction of stations with circularity > 0.9, mean circularity).

    Stations need exactly ONE outline loop — a torus slice produces two
    separate loops and must not pass the revolve gate.
    """
    i = _AXIS_IDX[axis]
    lo, hi = float(mesh.bounds[0][i]), float(mesh.bounds[1][i])
    span = hi - lo
    if span <= 0:
        return 0.0, 0.0
    got = ok = 0
    circs: list[float] = []
    for t in np.linspace(lo + 0.12 * span, hi - 0.12 * span, n):
        s = _section_at(mesh, axis, float(t))
        if s is None or s.polygon is None:
            continue
        got += 1
        if s.n_polys != 1:
            continue
        c = _circularity(s.polygon)
        circs.append(c)
        if c > _CIRC_MIN:
            ok += 1
    if got == 0:
        return 0.0, 0.0
    return ok / got, (float(np.mean(circs)) if circs else 0.0)


# --------------------------------------------------------------------------- #
# spec / analysis hints (ported from orchestrator._revolve_axis + mesh_score — #
# importing the agent layer from cad/ would be an import cycle)               #
# --------------------------------------------------------------------------- #


def _spec_suggests_revolve(spec: Any) -> bool:
    """Spec text heuristic: the part is described as round/rotational."""
    if spec is None:
        return False
    texts = [
        str(getattr(spec, "name", "") or ""),
        str(getattr(spec, "summary", "") or ""),
        str(getattr(spec, "symmetry", "") or ""),
        str(getattr(spec, "orientation_note", "") or ""),
    ]
    for f in getattr(spec, "features", None) or []:
        texts.append(f"{getattr(f, 'name', '')} {getattr(f, 'description', '')}")
    return any(_ROUND_RE.search(t) for t in texts)


def _measured_round(analysis: dict[str, Any] | None) -> bool:
    """The executor measured near-circular cross-sections already."""
    try:
        return float((analysis or {}).get("circularity")) >= 0.85
    except (TypeError, ValueError):
        return False


def _stated_dim(spec: Any, axis: str) -> float | None:
    """Stated (user-given) dimension along ``axis`` — authoritative over mesh."""
    if spec is None:
        return None
    d = getattr(spec, _SPEC_DIM[axis], None)
    v = getattr(d, "value", None)
    if getattr(d, "source", "") == "stated" and v and float(v) > 0:
        return float(v)
    return None


def _revolve_candidates(mesh: trimesh.Trimesh) -> list[str]:
    """Candidate rotation axes: radial bbox extents must agree within 20%
    (a revolve is radially symmetric in its bbox). Ordered by deviation."""
    ext = np.asarray(mesh.extents, dtype=float)
    if len(ext) != 3 or ext.min() <= 0:
        return []
    pairs = {"z": (ext[0], ext[1]), "x": (ext[1], ext[2]), "y": (ext[0], ext[2])}
    cands = []
    for ax, (a, b) in pairs.items():
        hi = max(a, b)
        if hi > 0:
            dev = abs(a - b) / hi
            if dev <= _MAX_BBOX_PAIR_DEV:
                cands.append((dev, ax))
    cands.sort()
    return [ax for _, ax in cands]


def _bore_axis_consistent(mesh: trimesh.Trimesh, axis: str, n: int = _N_CHECK) -> bool:
    """Through-bore along ``axis``: most check sections show an interior
    ring whose centres are collinear within 15 % of the radial extent.
    A drilled grip is axi-symmetric in INTENT even when its outer form is
    asymmetric — the bore axis is the natural parametric revolve axis."""
    i = _AXIS_IDX[axis]
    j, k = _RADIAL_IDX[axis]
    lo, hi = float(mesh.bounds[0][i]), float(mesh.bounds[1][i])
    span = hi - lo
    if span <= 0:
        return False
    centres: list[tuple[float, float]] = []
    for t in np.linspace(lo + 0.12 * span, hi - 0.12 * span, n):
        s = _section_at(mesh, axis, float(t))
        if s is not None and s.inner_uv is not None:
            centres.append(s.inner_uv)
    if len(centres) < max(4, n // 2):
        return False
    spread = float(np.ptp(np.asarray(centres), axis=0).max())
    ext = max(float(mesh.extents[j]), float(mesh.extents[k]), 1e-9)
    return spread <= 0.15 * ext


def _pick_revolve_axis(
    mesh: trimesh.Trimesh, spec: Any, analysis: dict[str, Any] | None
) -> str | None:
    """Rotation axis when the part is a provable revolve, else None.

    Gate: spec wording / measured circularity (hint) lowers the required
    station fraction to 70 %; without a hint pure geometry must carry it
    (>85 % of stations circular AND mean circularity > 0.93). A consistent
    through-bore along a candidate axis is a revolve signal of its own —
    the outer form may be asymmetric while the part is clearly lathe-like.
    """
    hint = _spec_suggests_revolve(spec) or _measured_round(analysis)
    need = 0.7 if hint else 0.85
    for ax in _revolve_candidates(mesh):
        frac, mean = _station_circularity(mesh, ax)
        if frac >= need and (hint or mean >= 0.93):
            return ax
        if _bore_axis_consistent(mesh, ax) and frac >= 0.5:
            return ax
    return None


def _prism_candidates(mesh: trimesh.Trimesh) -> list[tuple[float, str, float]]:
    """Extrusion axes where the section profile is constant over >80 % of
    the thickness (IoU > 0.85 between centroid-aligned slices).

    Several axes can qualify (a rounded-edge plate is a prism along z AND a
    coarser rectangle-prism along x/y) — they produce DIFFERENT solids, so
    return all candidates sorted by slice-agreement; the caller rebuilds
    each and keeps the best chamfer fit.
    """
    try:
        from shapely.affinity import translate
    except Exception:
        return []

    def _centred(p: Any) -> Any:
        return translate(p, -p.centroid.x, -p.centroid.y)

    cands: list[tuple[float, str, float]] = []  # (score, axis, mean_iou)
    for axis in ("z", "y", "x"):
        i = _AXIS_IDX[axis]
        lo, hi = float(mesh.bounds[0][i]), float(mesh.bounds[1][i])
        span = hi - lo
        if span <= 0:
            continue
        polys: list[Any | None] = []
        for t in np.linspace(lo + 0.12 * span, hi - 0.12 * span, 7):
            s = _section_at(mesh, axis, float(t))
            polys.append(s.polygon if s is not None else None)
        ref = polys[len(polys) // 2]
        if ref is None or ref.is_empty or ref.area <= 0:
            continue
        try:
            a = _centred(ref)
        except Exception:
            continue
        ious: list[float] = []
        for p in polys:
            if p is None or p.is_empty or p.area <= 0:
                continue
            try:
                b = _centred(p)
                u = a.union(b).area
                ious.append(a.intersection(b).area / u if u > 0 else 0.0)
            except Exception:
                continue
        if len(ious) < 5:
            continue
        frac = float(np.mean([v > 0.85 for v in ious]))
        mean_iou = float(np.mean(ious))
        if frac >= 0.8:
            cands.append((frac * mean_iou, axis, mean_iou))
    cands.sort(reverse=True)
    return cands


# --------------------------------------------------------------------------- #
# classification                                                              #
# --------------------------------------------------------------------------- #


def classify_part(
    mesh: trimesh.Trimesh, spec: Any, analysis: dict[str, Any] | None
) -> Literal["revolve", "prism", "organic"]:
    """Classify a mesh: ``"revolve"`` | ``"prism"`` | ``"organic"``.

    Revolve gate: spec/analysis hint + >70 % circular stations (or >85 %
    pure-geometry) on a radially-symmetric bbox axis. Prism gate: constant
    section profile (IoU > 0.85) over >80 % of one axis. Else organic.
    """
    if _pick_revolve_axis(mesh, spec, analysis) is not None:
        return "revolve"
    if _prism_candidates(mesh):
        return "prism"
    return "organic"


# --------------------------------------------------------------------------- #
# revolve profile fitting                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class _FitSeg:
    """Internal segment fit: endpoints in (axial, radial) profile coords."""

    kind: Literal["line", "arc"]
    p0: tuple[float, float]
    p1: tuple[float, float]
    center: tuple[float, float] | None = None
    radius: float | None = None
    mid: tuple[float, float] | None = None  # raw mid point (arc side / sweep)
    major: bool = False  # arc spans > 180°
    sign: int = 1  # RadiusArc side flag


def _fit_circle(pts: np.ndarray) -> tuple[float, float, float, float] | None:
    """Kasa least-squares circle fit → (cx, cy, R, max_residual_mm)."""
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([2 * x, 2 * y, np.ones(len(pts))])
    try:
        sol, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    except Exception:
        return None
    cx, cy, c = (float(v) for v in sol)
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    radius = math.sqrt(r2)
    max_res = float(np.max(np.abs(np.hypot(x - cx, y - cy) - radius)))
    return cx, cy, radius, max_res


def _sweep(a0: float, a1: float, amid: float) -> float:
    """Signed angular sweep from a0 to a1 passing through amid (radians)."""
    d = (a1 - a0) % (2 * math.pi)
    if (amid - a0) % (2 * math.pi) <= d + 1e-9:
        return d  # counter-clockwise contains the mid point
    return -(2 * math.pi - d)


def _classify_segments(
    raw: list[tuple[float, float]], dp: list[tuple[float, float]]
) -> list[_FitSeg]:
    """Split the Douglas-Peucker polyline into lines and circle arcs.

    A straight profile region simplifies to a single DP edge; curvature
    produces ≥2 consecutive short edges. Runs of ≥3 DP vertices whose raw
    stations fit a circle within ``_ARC_MAX_RESID_MM`` become RadiusArcs;
    everything else stays a Line.
    """
    raw_t = np.array([p[0] for p in raw])
    didx = [int(np.argmin(np.abs(raw_t - p[0]))) for p in dp]
    out: list[_FitSeg] = []
    i = 0
    while i < len(dp) - 1:
        best_j, best_fit = i + 1, None
        for j in range(i + 2, len(dp)):
            sub = np.asarray(raw[didx[i] : didx[j] + 1], dtype=float)
            fit = _fit_circle(sub)
            if fit is None:
                continue
            cx, cy, radius, max_res = fit
            chord = math.dist(dp[i], dp[j])
            sagitta = radius - math.sqrt(max(radius * radius - (chord / 2) ** 2, 0.0))
            if max_res < _ARC_MAX_RESID_MM and radius <= 25 * chord and sagitta > 0.1:
                best_j, best_fit = j, (cx, cy, radius)
        if best_fit is not None:
            cx, cy, radius = best_fit
            p0, p1 = dp[i], dp[best_j]
            mid = raw[(didx[i] + didx[best_j]) // 2]
            a0 = math.atan2(p0[1] - cy, p0[0] - cx)
            a1 = math.atan2(p1[1] - cy, p1[0] - cx)
            am = math.atan2(mid[1] - cy, mid[0] - cx)
            sweep = _sweep(a0, a1, am)
            # RadiusArc: positive radius → centre right of travel direction
            cross = (p1[0] - p0[0]) * (cy - p0[1]) - (p1[1] - p0[1]) * (cx - p0[0])
            out.append(
                _FitSeg(
                    kind="arc",
                    p0=p0,
                    p1=p1,
                    center=(cx, cy),
                    radius=radius,
                    mid=mid,
                    major=abs(sweep) > math.pi,
                    sign=1 if cross < 0 else -1,
                )
            )
            i = best_j
        else:
            out.append(_FitSeg(kind="line", p0=dp[i], p1=dp[i + 1]))
            i += 1
    return out


def _arc_points(seg: _FitSeg, n: int = 9) -> list[tuple[float, float]]:
    """Sample a fitted arc (t, r) on the fitted side for the recon mesh."""
    assert seg.center is not None and seg.radius is not None and seg.mid is not None
    cx, cy = seg.center
    a0 = math.atan2(seg.p0[1] - cy, seg.p0[0] - cx)
    a1 = math.atan2(seg.p1[1] - cy, seg.p1[0] - cx)
    am = math.atan2(seg.mid[1] - cy, seg.mid[0] - cx)
    sweep = _sweep(a0, a1, am)
    return [
        (
            cx + seg.radius * math.cos(a0 + sweep * k / (n - 1)),
            cy + seg.radius * math.sin(a0 + sweep * k / (n - 1)),
        )
        for k in range(n)
    ]


# --------------------------------------------------------------------------- #
# revolve reconstruction                                                      #
# --------------------------------------------------------------------------- #


def _vertex_names(radii: list[float]) -> list[str]:
    """Param name per profile vertex: r_start/r_end bookend; a distinct
    interior extremum earns the semantic r_waist / r_crown, else r_<i>."""
    n = len(radii)
    i_min = int(np.argmin(radii[1:-1])) + 1 if n > 2 else -1
    i_max = int(np.argmax(radii[1:-1])) + 1 if n > 2 else -1
    names: list[str] = []
    for vi in range(n):
        if vi == 0:
            names.append("r_start")
        elif vi == n - 1:
            names.append("r_end")
        elif vi == i_min and radii[vi] < 0.9 * max(radii):
            names.append("r_waist")
        elif vi == i_max and radii[vi] > 1.1 * min(radii):
            names.append("r_crown")
        else:
            names.append(f"r_{vi}")
    return names


def _emit_segments(
    segs: list[_FitSeg],
    names: list[str],
    params: dict[str, float],
    lo: float,
    extent: float,
    *,
    swap_uv: bool,
) -> list[ProfileSeg]:
    """Fit segments → emitted segments; arcs whose snapped radius undercuts
    the half-chord (RadiusArc would raise at exec time) degrade to lines.

    ``swap_uv``: the z-axis profile plane emits points as (radial, axial) —
    a reflection of the (axial, radial) fit frame — so the RadiusArc side
    sign must flip.
    """
    emit_segs: list[ProfileSeg] = []
    for si, s in enumerate(segs):
        if s.kind != "arc" or s.radius is None:
            emit_segs.append(ProfileSeg(kind="line", i0=si, i1=si + 1))
            continue
        p0 = (params["length"] * (s.p0[0] - lo) / extent, params[names[si]])
        p1 = (params["length"] * (s.p1[0] - lo) / extent, params[names[si + 1]])
        arc_r = snap_grid(s.radius)
        if arc_r < 0.505 * math.dist(p0, p1):
            emit_segs.append(ProfileSeg(kind="line", i0=si, i1=si + 1))
            continue
        pname = f"arc_r_{si}"
        params[pname] = arc_r
        emit_segs.append(
            ProfileSeg(
                kind="arc",
                i0=si,
                i1=si + 1,
                radius_param=pname,
                radius_sign=-s.sign if swap_uv else s.sign,
                major=s.major,
            )
        )
    return emit_segs


def _recon_profile(
    segs: list[_FitSeg],
    emit_segs: list[ProfileSeg],
    names: list[str],
    params: dict[str, float],
    lo: float,
    extent: float,
) -> list[tuple[float, float]]:
    """(axial, radial) polyline of the EMITTED profile (snapped params +
    fitted arcs) — the same geometry the generated code produces."""
    length = params["length"]
    prof: list[tuple[float, float]] = []
    for si, s in enumerate(segs):
        seg_pts: list[tuple[float, float]]
        if emit_segs[si].kind == "arc" and s.center is not None and s.mid is not None:
            seg_pts = [(lo + (t - lo) / extent * length, r) for t, r in _arc_points(s)]
            seg_pts[0] = (lo + (s.p0[0] - lo) / extent * length, params[names[si]])
            seg_pts[-1] = (lo + (s.p1[0] - lo) / extent * length, params[names[si + 1]])
        else:
            seg_pts = [
                (lo + (s.p0[0] - lo) / extent * length, params[names[si]]),
                (lo + (s.p1[0] - lo) / extent * length, params[names[si + 1]]),
            ]
        prof.extend(seg_pts[:-1] if si < len(segs) - 1 else seg_pts)
    # close the half-profile: through the inner radius for a bored part
    # (annular section) or down to the rotation axis and back
    bore_r = float(params.get("r_bore") or 0.0)
    inner = bore_r if bore_r > 0.0 else 0.0
    prof.append((prof[-1][0], inner))
    prof.append((prof[0][0], inner))
    return prof


def _bore_radius_mm(stations: list[tuple[float, Any]]) -> float | None:
    """Median equivalent radius of the interior ring across stations.

    A through-bore along the revolve axis appears as an interior loop in
    the section polygon at (nearly) every station. Equivalent radius from
    ring area — robust to polygon vertex density. Returns None when fewer
    than half the stations show a consistent inner loop."""
    rs: list[float] = []
    for _, s in stations:
        p = s.polygon
        if p is None or not getattr(p, "interiors", None):
            continue
        try:
            from shapely.geometry import Polygon

            inner = max(p.interiors, key=lambda r: abs(Polygon(r).area))
            a = float(abs(Polygon(inner).area))
            if a > 1e-6:
                rs.append(math.sqrt(a / math.pi))
        except Exception:
            continue
    if len(rs) < max(3, len(stations) // 2):
        return None
    return float(np.median(rs))


def _axis_centre(stations: list[tuple[float, Any]], rb: float | None) -> tuple[float, float]:
    """Rotation-axis position: bore-ring centres when a through-bore
    exists, median section centroids otherwise."""
    inner = [s.inner_uv for _, s in stations if s.inner_uv is not None]
    if rb is not None and len(inner) >= max(3, len(stations) // 2):
        return (
            float(np.median([c[0] for c in inner])),
            float(np.median([c[1] for c in inner])),
        )
    return (
        float(np.median([s.centroid[0] for _, s in stations])),
        float(np.median([s.centroid[1] for _, s in stations])),
    )


def _spec_bore_dia(spec: Any, axis: str) -> float | None:
    """Stated bore diameter along ``axis`` — spec dims stay authoritative
    over the mesh measurement (Ø22.2 stated, Ø22.0 measured+snapped)."""
    for f in getattr(spec, "features", None) or []:
        t = str(getattr(f, "type", "") or "").lower()
        ax = str(getattr(f, "axis", "") or "").lower().lstrip("+-")
        if ax and ax != axis:
            continue
        if not any(k in t for k in ("hole", "tunnel", "bore", "through")):
            continue
        dims = getattr(f, "dims", None) or {}
        d = dims.get("diameter") or dims.get("bore_diameter")
        if d and float(d) > 0:
            return float(d)
    return None


def _param_schema(params: dict[str, float]) -> list[dict[str, Any]]:
    """ParamDef.model_dump() entries for the emitted PARAMS dict."""
    return [
        ParamDef(
            name=name,
            value=val,
            unit="mm",
            min=(0.0 if name.startswith("r_") or name.startswith("arc_") else round(0.5 * val, 4)),
            max=round(3.0 * val, 4) if val > 0 else None,
            description="reconstructed from mesh section profile",
        ).model_dump()
        for name, val in params.items()
    ]


def _reconstruct_revolve(mesh: trimesh.Trimesh, axis: str, spec: Any) -> Reconstruction | None:
    i = _AXIS_IDX[axis]
    j, k = _RADIAL_IDX[axis]
    lo, hi = float(mesh.bounds[0][i]), float(mesh.bounds[1][i])
    span = hi - lo
    if span <= 0:
        return None

    stations: list[tuple[float, _Section]] = []
    for t in np.linspace(lo + 0.03 * span, hi - 0.03 * span, _N_STATIONS):
        s = _section_at(mesh, axis, float(t))
        if s is not None and s.centroid is not None:
            stations.append((float(t), s))
    if len(stations) < 8:
        return None

    # rotation-axis position in the radial plane: a consistent interior
    # ring means a through-bore whose centre IS the part axis (the annulus
    # centroid is shifted by asymmetric wall mass — the grip measured
    # z≈19 vs the true axis at z≈16 -> Ø38 reconstruction of a Ø32 part)
    rb = _bore_radius_mm(stations)
    u0, v0 = _axis_centre(stations, rb)

    raw: list[tuple[float, float]] = []
    for t, s in stations:
        r = float(np.percentile(np.hypot(s.uv[:, 0] - u0, s.uv[:, 1] - v0), 98))
        raw.append((t, r))
    r_max = max(r for _, r in raw)

    # extend the profile to the true mesh ends: a nearly-zero end radius is
    # a tip (land on the axis), a substantial one is a flat cap (hold r)
    if lo < raw[0][0] - 1e-9:
        raw.insert(0, (lo, 0.0 if raw[0][1] < 0.15 * r_max else raw[0][1]))
    if hi > raw[-1][0] + 1e-9:
        raw.append((hi, 0.0 if raw[-1][1] < 0.15 * r_max else raw[-1][1]))

    try:
        from shapely.geometry import LineString

        tol = max(0.05, 0.004 * span)
        dp = [
            (float(x), float(y))
            for x, y in LineString(raw).simplify(tol, preserve_topology=False).coords
        ]
    except Exception:
        return None
    if len(dp) < 2:
        dp = [raw[0], raw[-1]]
    segs = _classify_segments(raw, dp)

    verts_raw = [segs[0].p0] + [s.p1 for s in segs]
    extent = raw[-1][0] - raw[0][0]
    names = _vertex_names([r for _, r in verts_raw])

    stated = _stated_dim(spec, axis)
    length = (
        float(stated) if stated and 0.65 * extent <= stated <= 1.35 * extent else snap_grid(extent)
    )
    params: dict[str, float] = {"length": length}
    for vi, (_, r) in enumerate(verts_raw):
        params[names[vi]] = 0.0 if r < _EPS_R else snap_grid(r)

    emit_segs = _emit_segments(segs, names, params, lo, extent, swap_uv=axis == "z")
    verts = [
        ProfileVertex(frac=(t - raw[0][0]) / extent, param=names[vi])
        for vi, (t, _) in enumerate(verts_raw)
    ]

    # through-bore along the axis: prefer the stated spec diameter over
    # the measured inner ring — a solid-to-axis profile filled the bore
    # and blew the chamfer (grip: 3.02mm > 0.65mm limit)
    bore_r: float | None = None
    if rb is not None and rb < 0.8 * r_max:
        spec_d = _spec_bore_dia(spec, axis)
        bore_r = round(spec_d / 2.0, 4) if spec_d else snap_grid(rb)
        params["r_bore"] = bore_r

    offset = [0.0, 0.0, 0.0]
    offset[i] = lo
    offset[j] = u0
    offset[k] = v0
    code = mesh_to_code.emit_revolve_code(
        axis,
        verts,
        emit_segs,
        (offset[0], offset[1], offset[2]),
        params,
        bore_param="r_bore" if bore_r else None,
    )

    # --- confidence: rebuild the emitted profile (snapped values, fitted
    # arcs) as a trimesh and measure the symmetric surface distance back to
    # the input; plus a small penalty for stations that produced no slice.
    try:
        prof = _recon_profile(segs, emit_segs, names, params, lo, extent)
        recon = trimesh.creation.revolve(np.array([[r, t] for t, r in prof]), sections=72, cap=True)
        if axis == "x":
            recon.apply_transform(
                trimesh.transformations.rotation_matrix(math.pi / 2, [0.0, 1.0, 0.0])
            )
        elif axis == "y":
            recon.apply_transform(
                trimesh.transformations.rotation_matrix(-math.pi / 2, [1.0, 0.0, 0.0])
            )
        # (axis "z": trimesh.creation.revolve already revolves about +Z)
        radial_off = [0.0, 0.0, 0.0]
        radial_off[j], radial_off[k] = u0, v0
        recon.apply_translation(radial_off)
        d = _chamfer(mesh, recon, n=8000, seed=7)
        rel = d / max(float(np.max(mesh.extents)), 1e-9)
    except Exception as e:
        log.debug("reconstruct.recon_failed", error=str(e)[:160])
        rel = 0.15  # recon failed — neutral-ish penalty
    coverage = len(stations) / _N_STATIONS
    confidence = float(min(0.98, max(0.05, 1.0 - (4.0 * rel + 0.5 * (1.0 - coverage)))))

    # organic-vs-technical signal for the acceptance gate: an asymmetric
    # grip (scallops, mean circ ~0.85) cannot be chamfer-tight as a revolve
    circs = [_circularity(s.polygon) for _, s in stations if s.polygon is not None]
    meta = {"section_circularity_mean": round(float(np.mean(circs)), 4)} if circs else {}

    return Reconstruction(
        kind="revolve",
        code=code,
        params=params,
        params_schema=_param_schema(params),
        confidence=round(confidence, 3),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# prism reconstruction                                                        #
# --------------------------------------------------------------------------- #


def _reconstruct_prism(
    mesh: trimesh.Trimesh, axis: str, spec: Any, mean_iou: float
) -> Reconstruction | None:
    i = _AXIS_IDX[axis]
    j, k = _RADIAL_IDX[axis]
    lo, hi = float(mesh.bounds[0][i]), float(mesh.bounds[1][i])
    span = hi - lo
    if span <= 0:
        return None
    t_mid = (lo + hi) / 2
    s = _section_at(mesh, axis, t_mid)
    if s is None or s.polygon is None or s.polygon.area <= 0:
        return None
    try:
        simp = s.polygon.simplify(max(0.12, 0.002 * span), preserve_topology=True)
        if simp.is_empty or simp.area <= 0:
            simp = s.polygon
        coords2d = np.asarray(simp.exterior.coords)[:-1]  # drop closing dup
        if len(coords2d) < 3 or s.to_3d is None:
            return None
        # map the plane-local 2D coords back to world radial (j,k) coords
        hom = np.column_stack([coords2d, np.zeros(len(coords2d)), np.ones(len(coords2d))])
        w3 = (np.asarray(s.to_3d) @ hom.T).T
        wuv = w3[:, (j, k)]
    except Exception:
        return None

    stated = _stated_dim(spec, axis)
    thickness = (
        float(stated) if stated and 0.65 * span <= stated <= 1.35 * span else snap_grid(span)
    )
    params: dict[str, float] = {"thickness": float(thickness)}

    # emit outline in the workplane's local (u, v) coords — see mesh_to_code
    if axis == "z":
        outline = [(float(p[0]), float(p[1])) for p in wuv]
    elif axis == "x":
        outline = [(float(p[1]), -float(p[0])) for p in wuv]  # (z, -y)
    else:
        outline = [(float(p[1]), float(p[0])) for p in wuv]  # (z, x)
    code = mesh_to_code.emit_prism_code(axis, outline, lo, params)

    # confidence: extrude the simplified outline back to a mesh and measure
    rel = 0.15
    try:
        from shapely.geometry import Polygon

        recon = trimesh.creation.extrude_polygon(Polygon(wuv), span)
        # local extrusion axis is +Z — rotate it onto the measured axis
        # (the y-axis map is a reflection; harmless for distance sampling)
        if axis == "x":
            m4 = np.eye(4)
            m4[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
            recon.apply_transform(m4)
            recon.apply_translation([lo, 0.0, 0.0])
        elif axis == "y":
            m4 = np.eye(4)
            m4[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
            recon.apply_transform(m4)
            recon.apply_translation([0.0, lo, 0.0])
        else:
            recon.apply_translation([0.0, 0.0, lo])
        d = _chamfer(mesh, recon, n=8000, seed=11)
        rel = d / max(float(np.max(mesh.extents)), 1e-9)
    except Exception as e:
        log.debug("reconstruct.prism_recon_failed", error=str(e)[:160])
    confidence = float(min(0.98, max(0.05, 1.0 - (4.0 * rel + 0.5 * (1.0 - mean_iou)))))

    return Reconstruction(
        kind="prism",
        code=code,
        params=params,
        params_schema=[
            ParamDef(
                name="thickness",
                value=params["thickness"],
                unit="mm",
                min=round(0.5 * thickness, 4),
                max=round(2.0 * thickness, 4),
                description="extrusion thickness (reconstructed from mesh)",
            ).model_dump()
        ],
        confidence=round(confidence, 3),
    )


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #


def reconstruct_mesh(
    mesh: trimesh.Trimesh, spec: Any, analysis: dict[str, Any] | None
) -> Reconstruction | None:
    """Classify ``mesh`` and rebuild it as parametric code; None for organic."""
    if mesh is None or len(mesh.vertices) < 4:
        return None
    try:
        axis = _pick_revolve_axis(mesh, spec, analysis)
        if axis is not None:
            rec = _reconstruct_revolve(mesh, axis, spec)
            if rec is not None:
                log.info(
                    "reconstruct.revolve",
                    axis=axis,
                    n_params=len(rec.params),
                    confidence=rec.confidence,
                )
                return rec
        best: Reconstruction | None = None
        for _score, pax, mean_iou in _prism_candidates(mesh):
            rec = _reconstruct_prism(mesh, pax, spec, mean_iou)
            if rec is not None and (best is None or rec.confidence > best.confidence):
                best = rec
        if best is not None:
            log.info(
                "reconstruct.prism",
                thickness=best.params.get("thickness"),
                confidence=best.confidence,
            )
            return best
    except Exception as e:
        log.warning("reconstruct.failed", error=str(e)[:200])
        return None
    return None
