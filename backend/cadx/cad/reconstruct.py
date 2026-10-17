"""Reconstruction core: watertight trimesh → classified parametric build123d source.

A generated/processed mesh that is provably a lathe part, a prism or a
primitive composite (box + boss + dome + bores) is rebuilt as
contract-conformant build123d code (``PARAMS`` dict + ``result``
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
import random
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

# composite path (pyransac3d primitive segmentation — optional dependency)
_COMP_SAMPLE = 5000  # surface samples fed to the RANSAC fits
_COMP_MAX_PRIMS = 6  # cap on unioned/cut primitives
_COMP_MAX_ITER = 8  # greedy segmentation rounds (RANSAC retries included)
_COMP_MAX_STALL = 3  # consecutive rounds with no qualifying primitive → stop
_COMP_CUBOID_TRIES = 2  # cuboid RANSAC is stochastic — retry per round
_COMP_MIN_INLIER_FRAC = 0.03  # a primitive must explain ≥3 % of the samples
_COMP_MIN_COVERAGE = 0.55  # total explained-surface fraction required to emit
_COMP_PROBE_FRAC = 0.3  # classify_part hint: best single fit must explain ≥30 %
_COMP_MAX_REL_CHAMFER = 0.02  # self-check: reject if recon chamfer > 2 % of max dim


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
) -> Literal["revolve", "prism", "composite", "organic"]:
    """Classify a mesh: ``"revolve"`` | ``"prism"`` | ``"composite"`` | ``"organic"``.

    Revolve gate: spec/analysis hint + >70 % circular stations (or >85 %
    pure-geometry) on a radially-symmetric bbox axis. Prism gate: constant
    section profile (IoU > 0.85) over >80 % of one axis. Composite probe: a
    RANSAC primitive (cuboid/cylinder/sphere) explains ≥30 % of the sampled
    surface — only consulted after revolve and prism fail. Else organic.
    """
    if _pick_revolve_axis(mesh, spec, analysis) is not None:
        return "revolve"
    if _prism_candidates(mesh):
        return "prism"
    if _composite_hint(mesh):
        return "composite"
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
# composite reconstruction — pyransac3d primitive segmentation                #
# --------------------------------------------------------------------------- #
#
# A part that is neither a revolve nor a prism (box base + cylinder boss +
# sphere dome, plate with a blind pocket, ...) is decomposed greedily into
# RANSAC primitives on the sampled surface: fit cuboid/cylinder/sphere, take
# the candidate that explains the most samples WITHOUT burying surface inside
# its solid, remove those inliers, repeat. Cylinders whose surface normals
# point towards the axis are voids (bores) and are emitted as cuts; fitted
# bounding planes become half-space trims — only when they actually remove
# material (a bad cut is worse than no reconstruction). pyransac3d is an
# optional dependency: it is lazy-imported here and only here — without it the
# composite path returns None and the part stays on the mesh pipeline.


@dataclass
class _Prim:
    """One fitted primitive (internal — converted to CompositePrim at emit)."""

    shape: Literal["box", "cylinder", "sphere", "trim"]
    op: Literal["add", "cut"]
    center: np.ndarray  # (3,) world point
    rot: np.ndarray  # (3,3) — columns are the local basis vectors in world
    dims: tuple[float, ...]  # box (l,w,h) | cyl (r,t0,t1 span) | sphere (r) | trim (l,w,h)
    clip: tuple[float, float] | None = None  # sphere: keep local z ∈ [z0,z1]
    n_inliers: int = 0
    axis: np.ndarray | None = None  # cylinder axis (pre-orthonormalisation)
    normal_dot: float = 0.0  # mean normal·outward-radial of cylinder inliers
    drop: bool = False  # fitted but un-emittable (e.g. fully enclosed void)


@dataclass
class _Seg:
    """Segmentation result on the sampled surface."""

    pts: np.ndarray  # (N,3) samples
    nrm: np.ndarray  # (N,3) face normals at the samples
    remain: np.ndarray  # indices of unexplained samples
    prims: list[_Prim]
    thresh: float
    max_dim: float
    coverage: float


def _pyransac() -> Any | None:
    """Optional dependency: pyransac3d or None (degrades to organic)."""
    try:
        import pyransac3d

        return pyransac3d
    except Exception:
        return None


def _ortho_with_z(z: np.ndarray) -> np.ndarray:
    """Rotation matrix whose columns are an orthonormal basis with +Z ∥ ``z``."""
    z = np.asarray(z, dtype=float)
    zn = float(np.linalg.norm(z))
    if zn <= 0:
        return np.eye(3)
    z = z / zn
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, a)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _box_sdf(pts: np.ndarray, c: np.ndarray, rot: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """Signed distance to a box surface (negative inside)."""
    local = (pts - c) @ rot
    q = np.abs(local) - ext / 2.0
    return np.asarray(
        np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
    )


def _cuboid_supported(
    pts: np.ndarray, c: np.ndarray, rot: np.ndarray, ext: np.ndarray, thresh: float
) -> bool:
    """Reject "bounding boxes" enclosing curved geometry: every face of a
    real cuboid carries surface samples — RANSAC also accepts boxes whose
    top face is only a boss cap / dome silhouette. At most ONE face may be
    sparse (the attachment face of a boss is internal and has no surface):
    the second-worst face still needs ≥35 % of the best inlier density and
    ≥55 % of its 4x4 bins occupied."""
    local = (pts - c) @ rot
    dens: list[float] = []
    occs: list[float] = []
    for k in range(3):
        other = [j for j in range(3) if j != k]
        area = max(float(np.prod(ext[other])), 1e-9)
        for sgn in (-1.0, 1.0):
            sel = np.abs(local[:, k] - sgn * ext[k] / 2.0) < thresh
            dens.append(float(sel.sum()) / area)
            occ = 0.0
            sub = local[sel][:, other]
            if len(sub):
                ij = np.floor((sub + ext[other] / 2.0) / ext[other] * 4).astype(int).clip(0, 3)
                occ = len({(int(a), int(b)) for a, b in ij}) / 16.0
            occs.append(occ)
    d = np.sort(dens)
    o = np.sort(occs)
    return bool(d[1] >= 0.35 * d[5] and o[1] >= 0.55)


def _angular_coverage(pts_inl: np.ndarray, c: np.ndarray, rot: np.ndarray) -> float:
    """Fraction of 30° sectors around the local +Z axis occupied by inlier
    points. A real cylinder/bore wall wraps the full shell (≥0.75); a
    bogus big-radius fit that hugs a flat face covers only a small arc."""
    uv = (pts_inl - c) @ rot[:, :2]
    ang = np.arctan2(uv[:, 1], uv[:, 0])
    bins = np.floor((ang + np.pi) / (2 * np.pi) * 12).astype(int).clip(0, 11)
    return len(np.unique(bins)) / 12.0


def _fit_candidates(
    pyrsc: Any,
    pts: np.ndarray,
    nrm: np.ndarray,
    thresh: float,
    max_dim: float,
) -> list[tuple[int, np.ndarray, _Prim]]:
    """RANSAC-fit all solid primitive types on ``pts``; returns
    (score, own-inlier-mask, prim) candidates. Score = explained surface
    points minus HALF the residual points the solid would bury inside
    itself — a bounding box that swallows the plate-top surface loses to
    the real base box even when its raw inlier count is higher, while a
    legitimate boss containing a bore wall is only mildly penalised (the
    buried points get explained by the subsequent void fit)."""
    cands: list[tuple[int, np.ndarray, _Prim]] = []
    with np.errstate(all="ignore"):
        for _ in range(_COMP_CUBOID_TRIES):
            try:
                r = pyrsc.Cuboid().fit(pts, thresh=thresh, maxIteration=1500)
            except Exception:
                continue
            c = np.asarray(r.center, dtype=float)
            ext = np.asarray(r.extents, dtype=float)
            rot = np.asarray(r.axes, dtype=float).T  # rows→column basis
            if not len(r.inliers) or not np.isfinite(c).all() or not np.isfinite(rot).all():
                continue
            if not np.all(ext > 2 * thresh) or not np.all(ext < 3 * max_dim):
                continue
            if not _cuboid_supported(pts, c, rot, ext, thresh):
                continue
            sdf = _box_sdf(pts, c, rot, ext)
            mask = np.abs(sdf) < thresh
            score = int(mask.sum() - 0.5 * (sdf < -2 * thresh).sum())
            cands.append((score, mask, _Prim("box", "add", c, rot, tuple(float(v) for v in ext))))
        try:
            r = pyrsc.Sphere().fit(pts, thresh=thresh, maxIteration=800)
            c = np.asarray(r.center, dtype=float)
            if len(r.inliers) and np.isfinite(c).all() and thresh < float(r.radius) < 2 * max_dim:
                dist = np.linalg.norm(pts - c, axis=1)
                mask = np.abs(dist - float(r.radius)) < thresh
                inl = np.asarray(r.inliers)
                cap_dir = pts[inl].mean(axis=0) - c
                dl = float(np.linalg.norm(cap_dir))
                rot = np.eye(3)
                clip = None
                if dl > 0.25 * float(r.radius):
                    rot = _ortho_with_z(cap_dir / dl)
                    zs = (pts[inl] - c) @ rot[:, 2]
                    clip = (float(zs.min() - thresh), float(zs.max() + thresh))
                score = int(mask.sum() - 0.5 * (dist < float(r.radius) - 2 * thresh).sum())
                cands.append(
                    (
                        score,
                        mask,
                        _Prim("sphere", "add", c, rot, (float(r.radius),), clip=clip),
                    )
                )
        except Exception:
            pass
        # cylinders: big-radius near-planar wraps dominate the inlier
        # count, so retry a few times and keep only fits that wrap a real
        # shell (angular coverage) with a plausible radius
        for _ in range(4):
            try:
                r = pyrsc.Cylinder().fit(pts, thresh=thresh, maxIteration=1200)
            except Exception:
                break
            c = np.asarray(r.center, dtype=float)
            ax = np.asarray(r.axis, dtype=float)
            nax = float(np.linalg.norm(ax))
            inl = np.asarray(r.inliers)
            if (
                not len(inl)
                or not np.isfinite(c).all()
                or not np.isfinite(ax).all()
                or nax <= 0
                or not thresh < float(r.radius) < 0.4 * max_dim
            ):
                continue
            ax = ax / nax
            t = (pts[inl] - c) @ ax
            t0, t1 = float(np.min(t)), float(np.max(t))
            span = t1 - t0
            if not (2 * thresh < span < 1.5 * max_dim):
                continue
            rot = _ortho_with_z(ax)
            # void test: bore-wall normals point TOWARD the axis
            radv = pts[inl] - (c + np.outer(t, ax))
            rl = np.linalg.norm(radv, axis=1, keepdims=True)
            rl[rl == 0] = 1.0
            dot = float(np.mean(np.sum(nrm[inl] * (radv / rl), axis=1)))
            op: Literal["add", "cut"] = "cut" if dot < -0.3 else "add"
            cov = _angular_coverage(pts[inl], c, rot)
            if cov < (0.75 if op == "cut" else 0.5):
                continue
            rho = np.linalg.norm((pts - c) @ rot[:, :2], axis=1)
            tt = (pts - c) @ ax
            mask = (
                (np.abs(rho - float(r.radius)) < thresh)
                & (tt >= t0 - 2 * thresh)
                & (tt <= t1 + 2 * thresh)
            )
            inside = (rho < float(r.radius) - 2 * thresh) & (tt >= t0) & (tt <= t1)
            cands.append(
                (
                    int(mask.sum() - 0.5 * inside.sum()),
                    mask,
                    _Prim(
                        "cylinder",
                        op,
                        c,
                        rot,
                        (float(r.radius), t0, t1),
                        axis=ax,
                        normal_dot=dot,
                    ),
                )
            )
            break  # first valid cylinder wins — retry loop is for junk fits
    return cands


def _segment_primitives(mesh: trimesh.Trimesh, pyrsc: Any) -> _Seg | None:
    """Greedy RANSAC primitive segmentation of the mesh surface."""
    n_samples = _COMP_SAMPLE
    try:
        pts, fidx, _ = trimesh.sample.sample_surface(
            mesh, n_samples, seed=0, return_barycentric=True
        )
    except Exception:
        return None
    nrm = np.asarray(mesh.face_normals[np.asarray(fidx)], dtype=float)
    max_dim = max(float(np.max(mesh.extents)), 1e-9)
    thresh = max(0.2, 0.01 * max_dim)
    min_inliers = max(60, int(_COMP_MIN_INLIER_FRAC * n_samples))
    remain = np.arange(n_samples)
    prims: list[_Prim] = []
    stall = 0
    state = random.getstate()  # pyransac3d samples via `random` — seed for
    random.seed(0)  # determinism, restore the caller's state afterwards
    try:
        for _ in range(_COMP_MAX_ITER):
            if (
                len(remain) < min_inliers
                or len(prims) >= _COMP_MAX_PRIMS
                or stall >= _COMP_MAX_STALL
            ):
                break
            sub = pts[remain]
            cands = [
                (score, mask, prim)
                for score, mask, prim in _fit_candidates(pyrsc, sub, nrm[remain], thresh, max_dim)
                if score >= min_inliers
            ]
            if not cands:
                stall += 1
                continue
            stall = 0
            cands.sort(key=lambda c: -c[0])
            _, mask, prim = cands[0]
            prim.n_inliers = int(mask.sum())
            remain = remain[~mask]
            prims.append(prim)
    finally:
        random.setstate(state)
    coverage = 1.0 - len(remain) / n_samples
    return _Seg(pts, nrm, remain, prims, thresh, max_dim, coverage)


def _extend_inside(
    mesh: trimesh.Trimesh, base: np.ndarray, direction: np.ndarray, cap: float, step: float = 0.6
) -> float:
    """March from ``base`` along ``direction`` while inside the watertight
    mesh — lets a fitted boss cylinder extend into the base it sits on
    (union overlap) without poking out the far side."""
    t = 0.0
    try:
        while t < cap:
            cand = t + step
            if not bool(mesh.contains(np.asarray([base + cand * direction]))[0]):
                break
            t = cand
        # bisect the exit boundary so the extension stops on the surface,
        # not a full step past it
        hi = t + step
        for _ in range(4):
            mid = (t + hi) / 2.0
            if bool(mesh.contains(np.asarray([base + mid * direction]))[0]):
                t = mid
            else:
                hi = mid
    except Exception:
        pass
    return t


def _refine_cylinder(mesh: trimesh.Trimesh, seg: _Seg, prim: _Prim) -> None:
    """Resolve a fitted cylinder's span: bosses extend into their support,
    voids extend through the part (through-bore) or to a detected blind
    bottom (surface disc on the axis past the inlier span)."""
    assert prim.axis is not None
    r, t0, t1 = prim.dims
    ax, c = prim.axis, prim.center
    if prim.op == "add":
        cap = 0.6 * seg.max_dim
        e0 = _extend_inside(mesh, c + ax * t0, -ax, cap)
        e1 = _extend_inside(mesh, c + ax * t1, ax, cap)
        prim.dims = (r, t0 - e0, t1 + e1)
        return
    # void: an open end runs out past the part bbox (full cut through);
    # a blind end stops at the measured bottom disc (surface samples on
    # the hole floor, just past the wall inlier span).
    corners = np.asarray(mesh.bounds).reshape(-1, 3)
    lo, hi = float(((corners - c) @ ax).min()) - 1.0, float(((corners - c) @ ax).max()) + 1.0
    tt = (seg.pts - c) @ ax
    rho = np.linalg.norm((seg.pts - c) @ prim.rot[:, :2], axis=1)
    open_ends = 0
    for side in (-1.0, 1.0):
        end = t0 if side < 0 else t1
        past = side * (tt - end)  # >0 beyond this end of the wall span
        disc = (rho < 0.85 * r) & (past > 0.0) & (past < r)
        if int(disc.sum()) >= 15:
            # blind bottom: the disc sits at the hole floor — cut to the
            # measured face (+a fraction of thresh into it: a sliver of
            # leftover wall is worse than a fraction-mm deeper pocket)
            t_face = float(tt[disc].min() if side < 0 else tt[disc].max())
            if side < 0:
                t0 = t_face + 0.25 * seg.thresh
            else:
                t1 = t_face - 0.25 * seg.thresh
        else:
            open_ends += 1
            if side < 0:
                t0 = lo
            else:
                t1 = hi
    if open_ends == 0:
        # fully enclosed void — cannot be cut from inside; drop it
        prim.drop = True
        return
    prim.dims = (r, t0, t1)


def _dedup_prims(prims: list[_Prim]) -> list[_Prim]:
    """Drop near-duplicate fits of the same surface (a second RANSAC pass
    can re-find the same sphere/box within sampling noise)."""

    def _same(a: _Prim, b: _Prim) -> bool:
        if a.shape != b.shape or a.op != b.op:
            return False
        if a.shape == "sphere":
            return bool(
                np.linalg.norm(a.center - b.center) < 0.3 * min(a.dims[0], b.dims[0])
                and 0.8 < a.dims[0] / b.dims[0] < 1.25
            )
        if a.shape == "box":
            same_axes = abs(float(np.trace(a.rot.T @ b.rot))) > 2.8  # ~identity
            return bool(
                same_axes
                and np.linalg.norm(a.center - b.center) < 0.3 * float(np.min(b.dims))
                and np.max(np.abs(np.asarray(a.dims) - np.asarray(b.dims)) / np.asarray(b.dims))
                < 0.2
            )
        if a.shape == "cylinder":
            assert a.axis is not None and b.axis is not None
            if abs(float(a.axis @ b.axis)) < 0.98:
                return False
            dc = b.center - a.center
            axial = abs(float(dc @ a.axis))
            radial = float(np.linalg.norm(dc - (dc @ a.axis) * a.axis))
            return bool(
                axial < 0.5 * abs(a.dims[2] - a.dims[1])
                and radial < 0.3 * a.dims[0]
                and 0.8 < a.dims[0] / b.dims[0] < 1.25
            )
        return False

    keep: list[_Prim] = []
    for p in sorted(prims, key=lambda q: -q.n_inliers):
        if not any(_same(p, k) for k in keep):
            keep.append(p)
    return keep[:_COMP_MAX_PRIMS]


def _prim_trimesh(prim: _Prim) -> trimesh.Trimesh:
    """Trimesh mirror of the emitted solid (for the chamfer self-check)."""
    m4 = np.eye(4)
    m4[:3, :3] = prim.rot
    m4[:3, 3] = prim.center
    if prim.shape == "box" or prim.shape == "trim":
        m: trimesh.Trimesh = trimesh.creation.box(list(prim.dims))
        m.apply_transform(m4)
        return m
    if prim.shape == "cylinder":
        r, t0, t1 = prim.dims
        m = trimesh.creation.cylinder(radius=r, height=t1 - t0, sections=64)
        tmid = np.eye(4)
        tmid[:3, 3] = [0.0, 0.0, (t0 + t1) / 2.0]
        m.apply_transform(m4 @ tmid)
        return m
    # sphere (+ optional clip to the fitted cap extent)
    m = trimesh.creation.icosphere(subdivisions=4, radius=prim.dims[0])
    if prim.clip is not None:
        z0, z1 = prim.clip
        box: trimesh.Trimesh = trimesh.creation.box([4 * prim.dims[0], 4 * prim.dims[0], z1 - z0])
        tb = np.eye(4)
        tb[:3, 3] = [0.0, 0.0, (z0 + z1) / 2.0]
        m.apply_transform(m4)
        box.apply_transform(m4 @ tb)
        try:
            out: trimesh.Trimesh = trimesh.boolean.intersection([m, box])
            return out
        except Exception:
            return m
    m.apply_transform(m4)
    return m


def _void_fits(mesh: trimesh.Trimesh, seg: _Seg, recon: trimesh.Trimesh, pyrsc: Any) -> list[_Prim]:
    """Find voids: residual surface samples buried INSIDE the unioned
    solids can only be interior walls — bores and pockets. RANSAC on the
    outer residual is dominated by big-radius wraps of flat faces; on the
    interior subset the true bore wall is the dominant structure.
    A void is emitted only when clearly a void (normals point at the
    axis, full shell coverage) — else union, per the conservative rule."""
    if len(seg.remain) == 0:
        return []
    pts_r = seg.pts[seg.remain]
    try:
        # deeper than ~0.6 mm inside the unioned solids — fitted prims
        # overshoot slightly, so plain contains() marks almost everything.
        # |signed_distance| is the depth (its sign convention is not stable
        # across trimesh versions — use contains() for the inside test).
        inside = np.asarray(recon.contains(pts_r), dtype=bool)
        inside &= np.abs(trimesh.proximity.signed_distance(recon, pts_r)) > max(
            0.6, 1.5 * seg.thresh
        )
    except Exception:
        return []
    sub_pts = pts_r[inside]
    sub_nrm = seg.nrm[seg.remain][inside]
    pool = np.arange(len(sub_pts))
    found: list[_Prim] = []
    while len(found) < 3 and len(pool) >= 60:
        sub = sub_pts[pool]
        best: tuple[np.ndarray, _Prim] | None = None
        for _ in range(4):
            try:
                with np.errstate(all="ignore"):
                    r = pyrsc.Cylinder().fit(sub, thresh=seg.thresh, maxIteration=800)
            except Exception:
                break
            c = np.asarray(r.center, dtype=float)
            ax = np.asarray(r.axis, dtype=float)
            nax = float(np.linalg.norm(ax))
            inl = np.asarray(r.inliers)
            if (
                not len(inl)
                or not np.isfinite(c).all()
                or not np.isfinite(ax).all()
                or nax <= 0
                or not seg.thresh < float(r.radius) < 0.4 * seg.max_dim
            ):
                continue
            ax = ax / nax
            t = (sub[inl] - c) @ ax
            t0, t1 = float(np.min(t)), float(np.max(t))
            if t1 - t0 <= 2 * seg.thresh:
                continue
            rot = _ortho_with_z(ax)
            if _angular_coverage(sub[inl], c, rot) < 0.7:
                continue
            radv = sub[inl] - (c + np.outer(t, ax))
            rl = np.linalg.norm(radv, axis=1, keepdims=True)
            rl[rl == 0] = 1.0
            dot = float(np.mean(np.sum(sub_nrm[pool][inl] * (radv / rl), axis=1)))
            if dot > -0.4:
                continue  # not clearly a void
            rho = np.linalg.norm((sub - c) @ rot[:, :2], axis=1)
            tt = (sub - c) @ ax
            mask = (
                (np.abs(rho - float(r.radius)) < seg.thresh)
                & (tt >= t0 - 2 * seg.thresh)
                & (tt <= t1 + 2 * seg.thresh)
            )
            best = (
                mask,
                _Prim(
                    "cylinder",
                    "cut",
                    c,
                    rot,
                    (float(r.radius), t0, t1),
                    n_inliers=int(mask.sum()),
                    axis=ax,
                    normal_dot=dot,
                ),
            )
            break
        if best is None:
            break
        mask, prim = best
        _refine_cylinder(mesh, seg, prim)
        if prim.drop:
            break
        found.append(prim)
        pool = pool[~mask]
    return found


def _composite_trimesh(prims: list[_Prim]) -> trimesh.Trimesh | None:
    """Union adds, subtract cuts — the emitted boolean order, in trimesh."""
    adds = [p for p in prims if p.op == "add"]
    if not adds:
        return None
    try:
        meshes = [_prim_trimesh(p) for p in adds]
        recon = meshes[0] if len(meshes) == 1 else trimesh.boolean.union(meshes)
        for p in prims:
            if p.op == "cut":
                recon = trimesh.boolean.difference([recon, _prim_trimesh(p)])
        return recon
    except Exception as e:
        log.debug("reconstruct.composite_recon_failed", error=str(e)[:160])
        return None


def _plane_trims(
    mesh: trimesh.Trimesh, seg: _Seg, recon: trimesh.Trimesh, pyrsc: Any
) -> list[_Prim]:
    """Fit planes on the residual; keep only true bounding planes that
    actually trim overshoot (all surface on one side AND the unioned
    solids poke past the plane). Conservative: a bad cut is worse than
    no reconstruction — anything ambiguous is skipped."""
    trims: list[_Prim] = []
    remain = seg.remain.copy()
    pts = seg.pts
    for _ in range(3):
        if len(remain) < 60:
            break
        sub = pts[remain]
        try:
            with np.errstate(all="ignore"):
                r = pyrsc.Plane().fit(sub, thresh=seg.thresh, maxIteration=400)
        except Exception:
            break
        inl = np.asarray(r.inliers)
        if len(inl) < 60:
            break
        eq = np.asarray(r.equation, dtype=float)
        n, d = eq[:3], float(eq[3])
        nn = float(np.linalg.norm(n))
        if nn <= 0:
            break
        n, d = n / nn, d / nn
        sd = pts @ n + d  # signed distance of ALL samples to the plane
        frac_in = float(np.mean(sd <= seg.thresh))
        frac_out = float(np.mean(sd >= -seg.thresh))
        drop = np.abs(sub @ n + d) >= seg.thresh
        if frac_in < 0.995 and frac_out < 0.995:
            remain = remain[drop]
            continue  # not a bounding plane — slice of the interior
        if frac_in < frac_out:
            n, d, sd = -n, -d, -sd  # orient so the part is on the -n side
        try:
            smp, _ = trimesh.sample.sample_surface(recon, 2000, seed=1)
            over = smp @ n + d
            if float(np.mean(over > 1.5 * seg.thresh)) < 0.02:
                remain = remain[drop]
                continue  # nothing to trim
        except Exception:
            break
        p0 = -d * n  # closest point on the plane to the origin
        b = 4 * seg.max_dim
        trims.append(
            _Prim(
                "trim",
                "cut",
                p0 + n * (b / 2.0 + 0.75 * seg.thresh),
                _ortho_with_z(n),
                (b, b, b),
            )
        )
        remain = remain[drop]
    return trims


def _rot_axis_angle(rot: np.ndarray) -> tuple[tuple[float, float, float], float] | None:
    """3x3 rotation → build123d (axis, angle_deg); None for ~identity."""
    m4 = np.eye(4)
    m4[:3, :3] = rot
    angle, direc, _ = trimesh.transformations.rotation_from_matrix(m4)
    deg = math.degrees(float(angle))
    if abs(deg) < 0.5 or not np.isfinite(direc).all():
        return None
    dn = float(np.linalg.norm(direc))
    if dn <= 0:
        return None
    ax = np.asarray(direc) / dn
    return ((float(ax[0]), float(ax[1]), float(ax[2])), round(deg, 3))


def _t3(v: Any) -> tuple[float, float, float]:
    """3-vector → rounded literal tuple for emitted code."""
    return (round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4))


def _emit_composite_prims(prims: list[_Prim], params: dict[str, float]) -> list[Any]:
    """Internal prims → mesh_to_code.CompositePrim list + PARAMS entries."""
    out: list[Any] = []
    for i, p in enumerate(prims):
        rot = _rot_axis_angle(p.rot)
        center = _t3(p.center)
        if p.shape == "box":
            names = ("length", "width", "height")
            dim_params = {}
            for k, v in zip(names, p.dims, strict=True):
                key = f"p{i}_{k[:3]}"
                params[key] = snap_grid(float(v))
                dim_params[k] = key
            out.append(
                mesh_to_code.CompositePrim(
                    op="add", shape="box", center=center, rot=rot, dim_params=dim_params
                )
            )
        elif p.shape == "cylinder":
            r, t0, t1 = (float(v) for v in p.dims)
            params[f"p{i}_r"] = snap_grid(r)
            params[f"p{i}_h"] = snap_grid(t1 - t0)
            # canonical cylinder is centred — shift along its axis to the
            # fitted span midpoint
            assert p.axis is not None
            wc = p.center + p.axis * (t0 + t1) / 2.0
            out.append(
                mesh_to_code.CompositePrim(
                    op=p.op,
                    shape="cylinder",
                    center=_t3(wc),
                    rot=rot,
                    dim_params={"radius": f"p{i}_r", "height": f"p{i}_h"},
                )
            )
        elif p.shape == "sphere":
            params[f"p{i}_r"] = snap_grid(float(p.dims[0]))
            clip_center = clip_rot = clip_dims = None
            if p.clip is not None:
                z0, z1 = p.clip
                r = float(p.dims[0])
                lc = p.center + p.rot[:, 2] * (z0 + z1) / 2.0
                clip_center = _t3(lc)
                clip_rot = rot
                clip_dims = (4 * r, 4 * r, z1 - z0)
            out.append(
                mesh_to_code.CompositePrim(
                    op="add",
                    shape="sphere",
                    center=center,
                    rot=None,
                    dim_params={"radius": f"p{i}_r"},
                    clip_center=clip_center,
                    clip_rot=clip_rot,
                    clip_dims=clip_dims,
                )
            )
        else:  # trim: oversized literal box on the far side of the plane
            lx, wy, hz = (float(v) for v in p.dims)
            out.append(
                mesh_to_code.CompositePrim(
                    op="cut",
                    shape="trim",
                    center=center,
                    rot=rot,
                    dim_literals={"length": lx, "width": wy, "height": hz},
                )
            )
    return out


def _composite_hint(mesh: trimesh.Trimesh) -> bool:
    """Cheap probe for classify_part: does a RANSAC primitive explain a
    substantial share of the surface? Lazy-imports pyransac3d; any failure
    (missing dep, fit crash) is a quiet False — arbiter stays safe."""
    pyrsc = _pyransac()
    if pyrsc is None:
        return False
    try:
        n_probe = 1500
        pts, fidx, _ = trimesh.sample.sample_surface(mesh, n_probe, seed=0, return_barycentric=True)
        nrm = np.asarray(mesh.face_normals[np.asarray(fidx)], dtype=float)
        max_dim = max(float(np.max(mesh.extents)), 1e-9)
        thresh = max(0.2, 0.01 * max_dim)
        state = random.getstate()
        best = 0
        try:
            random.seed(0)
            for _ in range(3):  # RANSAC is stochastic — a few draws
                cands = _fit_candidates(pyrsc, np.asarray(pts), nrm, thresh, max_dim)
                best = max(best, max((int(m.sum()) for _, m, _ in cands), default=0))
                if best >= _COMP_PROBE_FRAC * n_probe:
                    return True
        finally:
            random.setstate(state)
        return best >= _COMP_PROBE_FRAC * n_probe
    except Exception:
        return False


def _reconstruct_composite(mesh: trimesh.Trimesh, spec: Any) -> Reconstruction | None:
    """Segment the mesh into RANSAC primitives and emit their boolean as
    build123d code. Self-checks: ≥55 % explained coverage AND a trimesh
    rebuild of the emitted solids within 2 % of the max dimension —
    otherwise None (the part stays on the mesh path)."""
    pyrsc = _pyransac()
    if pyrsc is None:
        return None
    seg = _segment_primitives(mesh, pyrsc)
    if seg is None or not seg.prims or seg.coverage < _COMP_MIN_COVERAGE:
        return None
    prims = _dedup_prims(seg.prims)
    for p in prims:
        if p.shape == "cylinder":
            _refine_cylinder(mesh, seg, p)
    # fully-enclosed voids cannot be cut — an un-emittable primitive drops
    prims = [p for p in prims if not p.drop]
    if not any(p.op == "add" for p in prims):
        return None

    recon = _composite_trimesh(prims)
    if recon is None or recon.is_empty:
        return None
    # void surfaces the union swallowed (bore walls, pockets) — fitted on
    # the interior residual; seeded like the main loop for determinism
    state = random.getstate()
    random.seed(1)
    try:
        voids = _void_fits(mesh, seg, recon, pyrsc)
        if voids:
            prims = prims + voids
            recon = _composite_trimesh(prims)
            if recon is None or recon.is_empty:
                return None
        trims = _plane_trims(mesh, seg, recon, pyrsc)
    finally:
        random.setstate(state)
    coverage = min(1.0, seg.coverage + sum(v.n_inliers for v in voids) / len(seg.pts))
    if trims:
        prims = prims + trims
        recon = _composite_trimesh(prims)
        if recon is None or recon.is_empty:
            return None

    # self-check: chamfer of the emitted-solid mirror vs the source mesh —
    # the downstream parametrize gate re-checks on real B-Rep output; this
    # pre-filter rejects garbage fits before they ever reach the sandbox
    try:
        d = _chamfer(mesh, recon, n=8000, seed=13)
        rel = d / seg.max_dim
    except Exception as e:
        log.debug("reconstruct.composite_chamfer_failed", error=str(e)[:160])
        return None
    if rel > _COMP_MAX_REL_CHAMFER:
        log.info(
            "reconstruct.composite_rejected",
            chamfer_mm=round(d, 3),
            rel=round(rel, 4),
            n_prims=len(prims),
            coverage=round(coverage, 3),
        )
        return None

    params: dict[str, float] = {}
    emit_prims = _emit_composite_prims(prims, params)
    code = mesh_to_code.emit_composite_code(emit_prims, params)
    meta = {
        "chamfer_rel": round(rel, 4),
        "coverage": round(coverage, 4),
        "primitives": [
            {
                "shape": p.shape,
                "op": p.op,
                "dims": [round(float(v), 4) for v in p.dims],
                "inliers": p.n_inliers,
            }
            for p in prims
        ],
    }
    return Reconstruction(
        kind="composite",
        code=code,
        params=params,
        params_schema=_param_schema(params),
        confidence=round(min(0.98, coverage), 3),
        meta=meta,
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
        rec = _reconstruct_composite(mesh, spec)
        if rec is not None:
            log.info(
                "reconstruct.composite",
                n_prims=len(rec.meta.get("primitives", [])),
                confidence=rec.confidence,
            )
            return rec
    except Exception as e:
        log.warning("reconstruct.failed", error=str(e)[:200])
        return None
    return None
