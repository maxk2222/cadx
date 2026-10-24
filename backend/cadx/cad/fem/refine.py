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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from cadx.cad.fem import anchors3d
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

# restored keep-patch must overlap the optimized body by at least this
# much real volume — less means it lands in an emptied region and floats
_MIN_BRIDGE_MM3 = 1.0

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
    # through-intent: the bore pierces every solid interval within the
    # declared reach (a restored cap or a plugged channel above/below the
    # original surfaces must never seal the cut). False = genuine blind
    # depth (insert pockets, blind bores declared without "through").
    through: bool = True
    csk_dia: float | None = None
    csk_angle_deg: float = 90.0
    cbore_dia: float | None = None
    cbore_depth_mm: float | None = None
    thread_spec: str = ""  # "M4-0.7" — overrides dia for tap/thread cutters
    # tri-state: None = auto — apply_refine grooves thread/tap features whose
    # pitch is derivable (helical_threads flag permitting); True/False is an
    # explicit per-feature choice that always wins
    helical: bool | None = None
    source: str = "anchor"  # anchor|spec|request (+ scanned source tags like "spec+scan")
    dia_extra: float = 0.0  # internal: ream bump after snapping onto an existing tunnel
    # internal: spec-sourced anchor admitted WITHOUT a base-mesh tunnel —
    # apply_refine re-verifies real material at the anchor AFTER keep-restore
    # before cutting (a scoop whose ring was eaten and never restored leaves
    # nothing to bore into)
    admit_on_material: bool = False
    # internal: snapped spec feature — fill the measured wavy channel first,
    # then drill at the exact spec diameter. A plain difference can't shrink
    # a void, so without the plug the cutter reams +0.4 past every wall
    # bulge; the plug restores nominal-diameter walls
    plug_first: bool = False
    # internal: plug centre override — when the drill must land at the
    # DECLARED position but the wavy channel drifted (topopt keep-rings hold
    # the ring, the channel itself wanders), the plug seals the channel at
    # its measured centre and the drill cuts at (u, v)
    plug_uv: tuple[float, float] | None = None


# --------------------------------------------------------------------------- #
# feature resolution                                                            #
# --------------------------------------------------------------------------- #


def _metric_pitch(dia: float) -> float:
    """Nearest standard metric coarse pitch for a nominal diameter."""
    best = min(_COARSE_PITCH, key=lambda d: abs(d - dia))
    return _COARSE_PITCH[best]


def _opt_bool(v: Any) -> bool | None:
    """Tri-state flag from a merged feature dict — None means 'unset'."""
    return None if v is None else bool(v)


def _thread_pitch(feat: RefineFeature) -> float | None:
    """Thread pitch in mm when a metric spec is derivable — thread_spec
    ('M8-1.25' / 'M8x1.25') first, else the nominal diameter snapped to the
    nearest ISO coarse size. None means 'cannot groove this thread'."""
    parsed = _parse_spec(feat.thread_spec)
    if parsed:
        return parsed[1]
    if feat.dia > 0:
        nearest = min(_COARSE_PITCH, key=lambda d: abs(d - feat.dia))
        if abs(nearest - feat.dia) <= 0.6:
            return _COARSE_PITCH[nearest]
    return None


def _parse_spec(spec: str) -> tuple[float, float] | None:
    """'M4-0.7' -> (4.0, 0.7); 'M3-0.5-6' -> (3.0, 0.5)."""
    import re

    m = re.search(r"M(\d+(?:\.\d+)?)[-x](\d+(?:\.\d+)?)", spec.strip(), re.I)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _thread_spec_from(sf: dict[str, Any], dia: float, *, allow_dia_snap: bool = False) -> str:
    """Resolve a fastener spec string from Feature.standard/name/description.

    Accepts 'M4-0.7' verbatim; a bare 'M4' gets the coarse pitch for its
    diameter. Returns '' when nothing looks like a metric size.

    `allow_dia_snap` additionally infers the spec from the bare diameter
    (tap-drill sizes) — only for thread-typed features: a Ø5.5 *clearance*
    hole is the standard M5 fit and must not earn an M5 thread spec by
    diameter proximity alone."""
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
    if allow_dia_snap and dia > 0:
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


def _spec_feature_text(sf: dict[str, Any]) -> str:
    """Free-text fields that may carry machine-readable position hints."""
    return " ".join(
        str(sf.get(k) or "") for k in ("position", "description", "note", "name")
    )


def _spec_axis(sf: dict[str, Any], text: str) -> str | None:
    """Drill axis declared by a spec feature: an explicit ``axis`` field,
    else the ``through <axis>`` word of its position text ('through
    x=42..50' bores across x). The through-range midpoint picks the entry
    side (parts are centred — mid > 0 drills from +). None = undeclared;
    callers fall back to the feature's default axis."""
    import re

    a = str(sf.get("axis") or "").strip().lower()
    if a in ("x", "y", "z"):
        a = "+" + a
    if a in _AXES:
        return a
    m = re.search(r"\bthrough\s+([xyz])\b", text, re.I)
    if not m:
        return None
    letter = m.group(1)
    from cadx.cad.poshint import parse_axis_ranges

    rng = parse_axis_ranges(text).get(letter)
    mid = (rng[0] + rng[1]) / 2.0 if rng else 0.0
    return ("-" if mid < 0 else "+") + letter


def _spec_uv_positions(text: str, axis: str) -> list[tuple[float, float]]:
    """Declared in-plane ``(u, v)`` bore positions from spec position text.

    Point hints ('x=0, y=0' and multi-point lists) come from
    ``parse_coord_points``; ``x=+/-42`` symmetric pairs expand to both
    members; a point missing an in-plane axis borrows that axis' symmetric
    values (mounting-hole grids declare +/-42 x +/-32 once for four bores).
    ``[]`` when the text declares no in-plane coordinates — the caller
    then keeps dia+axis anchor matching."""
    from cadx.cad.poshint import (
        parse_coord_points,
        parse_position_hints,
        parse_symmetric_coords,
    )

    _ax, au, av = _AXES[axis]
    aul, avl = "xyz"[au], "xyz"[av]
    sym = parse_symmetric_coords(text)
    points = parse_coord_points(text) or [parse_position_hints(text)[0]]
    out: list[tuple[float, float]] = []
    for p in points:
        uvals = [p[aul]] if aul in p else sym.get(aul, [])
        vvals = [p[avl]] if avl in p else sym.get(avl, [])
        for u in uvals:
            for v in vvals:
                uv = (float(u), float(v))
                if uv not in out:
                    out.append(uv)
    return out


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


def _tunnel_open_mesh(
    mesh: trimesh.Trimesh, u: float, v: float, axis: str, *, open_frac: float = 0.2
) -> bool:
    """trimesh analogue of `mesh_boolean.tunnel_open` — True when a ray
    through (u, v) along `axis` travels mostly through open space, i.e. the
    part already has a through-feature there.

    Hits are paired into inside-material segments; the tunnel counts as open
    when the blocked length is ≤ max(1 mm, `open_frac`·span) — enough slack
    for tessellation wobble on bore walls, far below the ~full span a
    dimple/pocket or solid junk position blocks."""
    ax, au, av = _AXES[axis]
    lo, hi = float(mesh.bounds[0][ax]), float(mesh.bounds[1][ax])
    span = hi - lo
    if span <= 0:
        return False
    origin = np.zeros(3)
    origin[au], origin[av] = u, v
    origin[ax] = hi + 5.0 if axis[0] == "+" else lo - 5.0
    locs, _, _ = mesh.ray.intersects_location(
        ray_origins=np.array([origin]),
        ray_directions=np.array([-_AXIS_VEC[axis]]),
        multiple_hits=True,
    )
    if not len(locs):
        return True
    zc = np.sort(locs[:, ax])
    if len(zc) % 2:
        # odd count (non-watertight or tangent graze) — count the dangling
        # stretch to the far bound as blocked, erring on "not a tunnel"
        zc = np.append(zc, lo if axis[0] == "+" else hi)
    blocked = float(zc[1::2].sum() - zc[0::2].sum())
    return blocked <= max(1.0, open_frac * span)


def _anchor_material(mesh: trimesh.Trimesh, u: float, v: float, axis: str) -> bool:
    """True when `mesh` holds real solid material along the anchor's
    through-ray — the inverse of `_tunnel_open_mesh`. Used for spec anchors
    admitted without a base tunnel: after keep-restore a pasted ring (or
    surviving opt material) must exist or the cut would remove nothing."""
    return not _tunnel_open_mesh(mesh, u, v, axis)


def _bore_ring_material(mesh: trimesh.Trimesh, axis: str, u: float, v: float, dia: float) -> bool:
    """True when solid material rings the (u, v) position on `axis` — the
    discriminating test for an entry-less feature: an open channel that
    refine should re-cut still has wall material around it; a SIMP-eaten
    host region (wall optimized away) has none, and cutting there drills a
    bogus hole at the bbox face (observed: Ø6.8 spec bores landing as
    Ø3.8 mm nicks 40+ mm off-anchor on a lattice field).

    Casts rays along `axis` on a ring just outside the bore radius — a ray
    counts "material" when its blocked span exceeds the same threshold
    `_tunnel_open_mesh` uses (short grazes are noise); >=40% material rays
    means enough host survives to cut into."""
    ax, au, av = _AXES[axis]
    lo, hi = float(mesh.bounds[0][ax]), float(mesh.bounds[1][ax])
    span = hi - lo
    if span <= 0:
        return False
    r = dia / 2.0 + max(1.5, 0.15 * dia)
    n = 10
    origins = np.zeros((n, 3))
    dirs = np.zeros((n, 3))
    for i in range(n):
        a = 2.0 * math.pi * i / n
        origins[i, au] = u + r * math.cos(a)
        origins[i, av] = v + r * math.sin(a)
    sgn = 1.0 if axis[0] == "+" else -1.0
    origins[:, ax] = hi + 5.0 if sgn > 0 else lo - 5.0
    dirs[:, ax] = -sgn
    locs, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True
    )
    hits: list[list[float]] = [[] for _ in range(n)]
    for r_id, loc in zip(ray_ids, locs, strict=True):
        hits[int(r_id)].append(float(loc[ax]))
    blocked_min = max(1.0, 0.2 * span)
    far = lo if sgn > 0 else hi
    material = 0
    for z in hits:
        zc = np.sort(np.asarray(z, dtype=float))
        if len(zc) % 2:
            zc = np.append(zc, far)
        blocked = float(zc[1::2].sum() - zc[0::2].sum()) if len(zc) else 0.0
        if blocked > blocked_min:
            material += 1
    return material >= math.ceil(0.4 * n)


def resolve_refine_features(  # noqa: C901
    anchors: list[dict[str, Any]] | None,
    spec_features: list[dict[str, Any]] | None,
    request_features: list[dict[str, Any]] | None,
    *,
    match_tol_mm: float = 3.0,
    base_mesh: trimesh.Trimesh | None = None,
) -> tuple[list[RefineFeature], list[str]]:
    """Merge request > spec > scanned anchors into a cutter list.

    Request entries may carry `anchor_index` (1-based into `anchors`) or a
    full geometric description. Spec features upgrade a nearby anchor's kind
    and contribute depth/counter/thread dims. Unmatched fastener anchors are
    re-cut as plain holes at their measured diameter — but ONLY when the
    pre-optimization `base_mesh` shows a real through-tunnel at the anchor
    position (scanners report organic surface dimples as "hole" anchors;
    drilling those cuts junk bores). Spec-DECLARED anchors (anchor source
    contains "spec") are authoritative: without a base tunnel they are
    admitted anyway with `admit_on_material` — apply_refine then requires
    real material at the anchor on the post-restore mesh before cutting.
    Without `base_mesh` every leftover anchor is kept (legacy behaviour).
    """
    feats: list[RefineFeature] = []
    warnings: list[str] = []
    anchors = [a for a in (anchors or []) if is_fastener_anchor(a)]
    spec_features = spec_features or []

    def _nearest_anchor(
        u: float, v: float, axis: str, used: set[int] | None = None
    ) -> dict[str, Any] | None:
        best, best_d = None, match_tol_mm
        for a in anchors:
            if used is not None and id(a) in used:
                continue
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
            # a scanned hole/tunnel is a real through-void; a dimple/pocket
            # or an explicit depth override is a genuine blind stop
            through=(
                str(a.get("kind") or "hole") not in ("dimple", "pocket", "insert")
                and dims.get("depth") is None
            ),
            csk_dia=dims.get("countersink_diameter"),
            csk_angle_deg=float(dims.get("countersink_angle") or 90.0),
            cbore_dia=dims.get("counterbore_diameter"),
            cbore_depth_mm=dims.get("counterbore_depth"),
            thread_spec=str(dims.get("thread_spec") or a.get("thread_spec") or ""),
            helical=_opt_bool(dims.get("helical", a.get("helical"))),
            source=str(a.get("source") or "anchor"),
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
                    through=bool(merged["through"])
                    if merged.get("through") is not None
                    else (merged.get("depth_mm") or merged.get("depth")) is None,
                    csk_dia=merged.get("csk_dia") or merged.get("countersink_diameter"),
                    csk_angle_deg=float(
                        merged.get("csk_angle_deg") or merged.get("countersink_angle") or 90.0
                    ),
                    cbore_dia=merged.get("cbore_dia") or merged.get("counterbore_diameter"),
                    cbore_depth_mm=merged.get("cbore_depth_mm") or merged.get("counterbore_depth"),
                    thread_spec=str(merged.get("thread_spec") or merged.get("standard") or ""),
                    helical=_opt_bool(
                        merged["helical"]
                        if merged.get("helical") is not None
                        else merged.get("helical_thread")
                    ),
                    source="request",
                )
            )
        return feats, warnings

    # --- auto: spec features with declared positions -------------------------
    used_anchors: set[int] = set()

    def _from_spec(sf: dict[str, Any], stype: str, dims: dict[str, Any], u: float, v: float,
                   axis: str, anchor: dict[str, Any] | None) -> RefineFeature | None:
        dia = float(
            dims.get("diameter")
            or dims.get("bore_diameter")
            or (anchor or {}).get("dia")
            or 0
        )
        if dia <= 0:
            return None
        # dia-snap only when the feature actually means a thread — otherwise
        # a Ø5.5 M5-clearance hole silently picks up an 'M5' spec
        threadish = stype in ("thread", "tap", "insert") or bool(dims.get("thread_spec"))
        spec_txt = _thread_spec_from(sf, dia, allow_dia_snap=threadish)
        # through-intent: the spec's own words ('through holes', 'through
        # x=42..50') mark it; a scanned through-anchor does too; a declared
        # depth WITHOUT 'through' is a genuine blind stop
        low_text = " ".join(
            str(sf.get(k) or "") for k in ("position", "description", "name")
        ).lower()
        through = (
            "through" in low_text
            or str((anchor or {}).get("kind") or "") in ("hole", "tunnel")
        ) and stype != "insert"
        return RefineFeature(
            name=str(sf.get("name") or (anchor or {}).get("name") or "spec feature"),
            kind=_upgrade_kind("hole", stype, dims),
            axis=axis,
            u=u,
            v=v,
            dia=dia,
            depth_mm=dims.get("depth") or (anchor or {}).get("depth_mm") or None,
            through=through,
            csk_dia=dims.get("countersink_diameter"),
            csk_angle_deg=float(dims.get("countersink_angle") or 90.0),
            cbore_dia=dims.get("counterbore_diameter"),
            cbore_depth_mm=dims.get("counterbore_depth"),
            thread_spec=spec_txt,
            helical=_opt_bool(dims.get("helical", (anchor or {}).get("helical"))),
            source="spec",
        )

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
        text = _spec_feature_text(sf)
        axis = _spec_axis(sf, text) or _feature_axis(sf)
        positions = _spec_uv_positions(text, axis)
        if positions:
            # declared positions are authoritative — a nearby anchor only
            # supplies tunnel evidence (plug target) and a dia fallback;
            # the bore keeps the declared (u, v), never the measured one.
            # Without this, dia+axis matching once bound an M12 spec bore
            # to a corner anchor 57 mm off its declared centre.
            for pu, pv in positions:
                anchor = _nearest_anchor(pu, pv, axis, used_anchors)
                if anchor is not None:
                    used_anchors.add(id(anchor))
                f = _from_spec(sf, stype, dims, pu, pv, axis, anchor)
                if f is None:
                    warnings.append(
                        f"refine feature '{sf.get('name')}' has no diameter — skipped"
                    )
                    continue
                feats.append(f)
            continue
        # no parseable declared position — match by diameter+axis
        anchor = None
        cands = [a for a in anchors if id(a) not in used_anchors and _feature_axis(a) == axis]
        if not cands:
            cands = [a for a in anchors if id(a) not in used_anchors]
        if cands and dia > 0:
            # spec-declared anchors carry the authoritative position —
            # a scanned anchor at a lattice opening/scan artifact with a
            # near diameter must not win the feature (observed: spec bore
            # bound to a scaffold tunnel 43 mm off its declared anchor)
            anchor = min(
                cands,
                key=lambda a: (
                    "spec" not in str(a.get("source") or ""),
                    abs(float(a.get("dia") or 0) - dia),
                ),
            )
        elif cands:
            anchor = min(cands, key=lambda a: "spec" not in str(a.get("source") or ""))
        if anchor is None:
            continue
        used_anchors.add(id(anchor))
        dims["thread_spec"] = _thread_spec_from(
            sf, dia, allow_dia_snap=stype in ("thread", "tap", "insert") or bool(dims.get("thread_spec"))
        )
        f = _from_anchor(anchor, kind=stype, dims=dims)
        if f:
            f.name = str(sf.get("name") or f.name)
            f.source = "spec"
            feats.append(f)

    no_tunnel: list[str] = []
    deferred: list[str] = []
    for a in anchors:
        if id(a) in used_anchors:
            continue
        f = _from_anchor(a)
        if f is None:
            continue
        if base_mesh is not None and not _tunnel_open_mesh(base_mesh, f.u, f.v, f.axis):
            if "spec" in str(f.source):
                # spec-declared anchor over non-tunnel base — the spec is
                # authoritative (FEM regions resolve onto it). Keep-restore
                # pastes the base ring back first, so admit it and let
                # apply_refine verify real material at the anchor on the
                # post-restore mesh before cutting.
                f.admit_on_material = True
                deferred.append(f.name)
                feats.append(f)
                continue
            # scanned anchor with no real through-feature in the
            # pre-optimization part — a surface dimple / noise void, not a
            # bore worth drilling
            no_tunnel.append(f.name)
            continue
        feats.append(f)
    if no_tunnel:
        warnings.append(
            f"skipped {len(no_tunnel)} anchor(s) without a base-mesh tunnel "
            f"({', '.join(no_tunnel[:4])}{'…' if len(no_tunnel) > 4 else ''})"
        )
    if deferred:
        warnings.append(
            f"{len(deferred)} spec anchor(s) without a base-mesh tunnel deferred "
            f"to keep-restore ({', '.join(deferred[:4])}"
            f"{'…' if len(deferred) > 4 else ''})"
        )

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


def _cutter_mesh(
    feat: RefineFeature, entry: float, span: float, depth_override: float | None = None
) -> trimesh.Trimesh:
    """Cutting solid for one feature, built with bore axis along +Z, entry at
    local z=0, bore going -Z. Caller rotates/translates into place."""
    over_out = 2.0  # overshoot above the entry face
    depth = (
        depth_override
        if depth_override is not None
        else feat.depth_mm if feat.depth_mm is not None else span + 2.0
    )
    dia = _effective_dia(feat) + feat.dia_extra
    tools: list[trimesh.Trimesh] = []
    cyl = trimesh.creation.cylinder(radius=dia / 2.0, height=depth + over_out, sections=48)
    cyl.apply_translation([0, 0, -depth / 2.0 + over_out / 2.0])
    tools.append(cyl)
    if feat.kind in ("thread", "tap") and feat.helical:
        # real helical thread: pilot bore (tap drill) + helical V-groove out
        # to the major diameter — printable FDM thread, no tapping needed.
        # A groove-build failure must never cost the bore: warn, keep the
        # plain tap-drill cylinder.
        parsed = _parse_spec(feat.thread_spec)
        d_maj, pitch = parsed if parsed else (feat.dia, _metric_pitch(feat.dia))
        try:
            groove = _thread_groove_solid(
                r_in=dia / 2.0 - 0.05,
                r_out=d_maj / 2.0 + 0.15 + feat.dia_extra / 2.0,  # FDM clearance
                pitch=pitch,
                depth=depth,
            )
            groove.apply_translation([0, 0, over_out * 0.5])
            tools.append(groove)
        except Exception as e:
            log.warning(
                "refine: helical groove build failed — tap-drill bore only",
                feature=feat.name,
                error=str(e),
            )
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


def _through_cut_depth(
    mesh: trimesh.Trimesh, feat: RefineFeature, entry: float, span: float
) -> float:
    """Effective cutter depth for a through-intent feature.

    The declared ``depth_mm`` is the *intended* material span — the bore must
    still pierce every solid interval that reach crosses: a keep-restored cap
    can sit above the channel, a plugged channel fills it to its far mouth,
    and the far wall's inner face can land a fraction of a mm past the
    declared depth. Any of those left standing seals the 'through' bore.

    The ray at (u, v) from the entry face inward yields solid intervals; the
    target is the far side of the deepest interval overlapping the declared
    reach (``entry → entry + inward·declared``), plus a small margin so the
    inner face can't survive as a membrane. Intervals beyond the reach
    (unrelated structure deeper along the line — a web behind a drilled
    flange wall) stay untouched. ``declared`` unchanged when nothing overlaps.
    """
    declared = feat.depth_mm if feat.depth_mm is not None else span + 2.0
    ax, au, av = _AXES[feat.axis]
    inward = -1.0 if feat.axis[0] == "+" else 1.0
    origin = np.zeros(3)
    origin[au], origin[av], origin[ax] = feat.u, feat.v, entry + inward * 0.02
    d = np.zeros(3)
    d[ax] = inward
    try:
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=np.array([origin]),
            ray_directions=np.array([d]),
            multiple_hits=True,
        )
    except Exception:
        return declared
    if not len(locs):
        return declared
    hits = sorted(float(loc[ax]) for loc in locs)
    # the ray starts ε inside the entry face — when that face is material
    # (normal seat, plug surface) the hit count is odd and the first solid
    # interval runs from the outermost hit back to the origin. Insert the
    # origin at the ray-facing end so pairing sees that interval too —
    # otherwise the plate under a plug is invisible and the declared depth
    # survives as a sealing membrane.
    if len(hits) % 2 == 1:
        if inward < 0:
            hits.append(float(origin[ax]))
        else:
            hits.insert(0, float(origin[ax]))
    r_lo = min(entry, entry + inward * declared)
    r_hi = max(entry, entry + inward * declared)
    fars: list[float] = []
    for i in range(0, len(hits) - 1, 2):
        lo, hi = hits[i], hits[i + 1]  # solid segment the ray crosses
        if hi >= r_lo and lo <= r_hi:
            fars.append(lo if inward < 0 else hi)
    if not fars:
        return declared
    far = min(fars) if inward < 0 else max(fars)
    need = (entry - far) * (-inward) + 0.3  # past the far wall, plus margin
    return max(declared, need)


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
# re-cut snapping / dedup — subsuming existing tunnels                          #
# --------------------------------------------------------------------------- #


def _merge_radius(dia: float) -> float:
    """Re-cuts this far apart on the same axis are the same bore."""
    return max(2.0, 0.3 * dia)


def _tunnel_center(
    mesh: trimesh.Trimesh, axis: str, u: float, v: float, r_max: float, dia: float
) -> tuple[float, float] | None:
    """Measured centre of an open through-tunnel near (u, v), else None.

    Anchors are measured on the smoothed optimized mesh — the real bore can
    sit a few hundredths of a mm off, and re-cutting at the raw position
    leaves a near-concentric figure-8 void (the 'double hole' defect). A ray
    grid in the feature plane finds the open cells (same blocked-length rule
    as `_tunnel_open_mesh`); the cluster nearest (u, v) seeds the centre,
    then in-plane rays to the bore walls at mid-depth refine it. Returns
    None when the measured centre lands outside `r_max` — the opening found
    is a different feature, not this bore."""
    ax, au, av = _AXES[axis]
    lo, hi = float(mesh.bounds[0][ax]), float(mesh.bounds[1][ax])
    span = hi - lo
    if span <= 0 or r_max <= 0:
        return None
    # window wide enough to see the bore rim, not just its interior — a
    # window fully inside the tunnel would centroid back onto the anchor
    w = r_max + 0.6 * dia
    step = min(0.5, max(w / 8.0, 0.2))
    n_side = max(5, int(2 * w / step) + 1)
    us = np.linspace(u - w, u + w, n_side)
    vs = np.linspace(v - w, v + w, n_side)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    n = uu.size
    origins = np.zeros((n, 3))
    origins[:, au] = uu.ravel()
    origins[:, av] = vv.ravel()
    sgn = 1.0 if axis[0] == "+" else -1.0
    origins[:, ax] = hi + 5.0 if sgn > 0 else lo - 5.0
    directions = np.zeros((n, 3))
    directions[:, ax] = -sgn
    locs, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    hits: list[list[float]] = [[] for _ in range(n)]
    for r_id, loc in zip(ray_ids, locs, strict=True):
        hits[int(r_id)].append(float(loc[ax]))
    blocked_max = max(1.0, 0.2 * span)
    far = lo if sgn > 0 else hi
    open_mask = np.zeros(uu.shape, dtype=bool)
    for flat in range(n):
        z = np.sort(np.asarray(hits[flat], dtype=float))
        if not len(z):
            open_mask.flat[flat] = True
            continue
        if len(z) % 2:
            z = np.append(z, far)
        open_mask.flat[flat] = float(z[1::2].sum() - z[0::2].sum()) <= blocked_max
    if not open_mask.any():
        return None
    import scipy.ndimage as ndi

    lab, _ = ndi.label(open_mask)
    dist = np.hypot(uu - u, vv - v)
    nearest = int(np.argmin(np.where(open_mask, dist, np.inf)))
    cells = lab == lab.flat[nearest]
    cu, cv = float(uu[cells].mean()), float(vv[cells].mean())

    # refine: from the open point at mid-depth, wall distances on both
    # in-plane directions give the true bore centre (and confirm an
    # enclosed bore rather than an open channel)
    origin = np.zeros(3)
    origin[au], origin[av] = cu, cv
    origin[ax] = (lo + hi) / 2.0
    center = [cu, cv]
    for k, d_ax in enumerate((au, av)):
        dists: list[float] = []
        for sgn2 in (1.0, -1.0):
            d = np.zeros(3)
            d[d_ax] = sgn2
            locs2, _, _ = mesh.ray.intersects_location(
                ray_origins=np.array([origin]),
                ray_directions=np.array([d]),
                multiple_hits=False,
            )
            dists.append(float(np.linalg.norm(locs2[0] - origin)) if len(locs2) else float("inf"))
        if math.isfinite(dists[0]) and math.isfinite(dists[1]):
            center[k] = (cu, cv)[k] + (dists[0] - dists[1]) / 2.0
    if math.hypot(center[0] - u, center[1] - v) > r_max:
        return None
    return float(center[0]), float(center[1])


def _plug_cylinder(mesh: trimesh.Trimesh, feat: RefineFeature) -> trimesh.Trimesh | None:
    """Filler cylinder that swallows the measured wavy channel at a snapped
    bore — unioned in BEFORE the nominal-diameter drill so the re-cut walls
    are exact instead of 'channel + ream margin'.

    A boolean difference can't shrink a void: a Ø5.5 cutter inside a ~Ø5.9
    wavy channel removes nothing and the ragged walls survive. Plug radius
    covers the widest measured wall bulge (first-hit radius over 24
    directions) plus weld margin, capped by the part silhouette (last-hit
    radius) so the plug can never stick out through an open channel mouth.
    Axial span covers the two part surfaces at the bore centre."""
    ax, au, av = _AXES[feat.axis]
    opp = ("-" if feat.axis[0] == "+" else "+") + feat.axis[1]
    pu, pv = feat.plug_uv or (feat.u, feat.v)
    # channel span along the axis — the centre ray of a through-channel hits
    # nothing, so surface coords are sampled on rings around the bore
    side_a: list[float] = []
    side_b: list[float] = []
    for r_probe in (feat.dia * 0.5 + 1.2, feat.dia * 0.5 + 2.5):
        for ang in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            uu = pu + r_probe * math.cos(ang)
            vv = pv + r_probe * math.sin(ang)
            e = _entry_point(mesh, feat.axis, uu, vv)
            if e is not None:
                side_a.append(e)
            e = _entry_point(mesh, opp, uu, vv)
            if e is not None:
                side_b.append(e)
        if side_a and side_b:
            break
    if not side_a or not side_b:
        return None
    entry = float(np.median(side_a))
    exit_ = float(np.median(side_b))
    lo, hi = min(entry, exit_), max(entry, exit_)
    span_ax = hi - lo
    if span_ax <= 0.5:
        return None
    mid = (lo + hi) / 2.0
    n_rays = 24
    angs = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    origin = np.zeros(3)
    origin[au], origin[av], origin[ax] = pu, pv, mid
    origins = np.tile(origin, (n_rays, 1))
    dirs = np.zeros((n_rays, 3))
    dirs[:, au] = np.cos(angs)
    dirs[:, av] = np.sin(angs)
    locs, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True
    )
    first = np.full(n_rays, np.inf)
    last = np.zeros(n_rays)
    origin_uv = np.zeros(3)  # in-plane distance only — no axial component
    origin_uv[au], origin_uv[av] = pu, pv
    for loc, ri in zip(locs, ray_ids, strict=True):
        p = np.zeros(3)
        p[au], p[av] = loc[au], loc[av]
        d = float(np.linalg.norm(p - origin_uv))
        first[int(ri)] = min(first[int(ri)], d)
        last[int(ri)] = max(last[int(ri)], d)
    finite = np.isfinite(first)
    if not finite.any():
        return None
    # channel-scale gate: when the NEAREST surface is already far beyond the
    # nominal bore, the "channel" is really an open interior cavity (hollow
    # SIMP shells) — plugging it would re-solidify the part core, and the
    # +0.4 weld margin on `first.max()` would poke past the exterior
    # silhouette the probe mistook for a channel wall. Refine only re-cuts
    # the nominal bore there — nothing to seal.
    if float(first[finite].min()) > feat.dia * 0.9:
        return None
    r_void = float(first[finite].max()) + 0.4  # widest bulge + weld margin
    # silhouette bound: `last` is the farthest surface along each ray — but a
    # ray with NO hits means that direction is open void to the outside
    # (sparse/cage parts: the channel gap connects straight to the exterior).
    # Such directions bound the plug to the nominal bore radius — anything
    # past the missing wall would protrude out of the part silhouette.
    limit = np.where(finite, last, feat.dia / 2.0)
    r_part = float(limit.min()) - 0.3  # never protrude the silhouette
    plug_r = min(r_void, r_part)
    if plug_r <= feat.dia / 2.0 + 0.05:
        return None
    # axial span: cover the measured channel mouth-to-mouth — inset 0.05mm
    # on both faces so the plug can never protrude past the silhouette (a
    # spec-envelope clip already ran upstream; sticking out would
    # re-introduce the overshoot it removed)
    plug = trimesh.creation.cylinder(radius=plug_r, height=span_ax - 0.1, sections=48)
    plug.apply_translation([0, 0, -span_ax / 2.0])
    _orient_cutter(plug, feat.axis, pu, pv, entry)
    return plug


def _resolve_snap(
    feat: RefineFeature,
    mesh: trimesh.Trimesh,
    base_mesh: trimesh.Trimesh | None,
    report: RefineReport,
) -> tuple[float, float] | None:
    """Snap/plug decision for one feature over a measured tunnel.

    - the wavy channel is real (mesh tunnel) → plug_first seals it before the
      nominal drill — for spec'd features AND scanned anchors whose diameter
      was measured on the base mesh (a plain difference can't shrink a void)
    - spec positions are authoritative: never snap onto a drifted channel —
      the plug seals it at its measured centre, the drill cuts at (u, v)
    - drift >1mm likewise: plug at the channel, drill at the declared spot
    - small drift (<1mm): snap onto the tunnel centre outright
    """
    r = _merge_radius(feat.dia)
    c = _tunnel_center(mesh, feat.axis, feat.u, feat.v, r, feat.dia)
    mesh_tunnel = c is not None
    if c is None and base_mesh is not None and base_mesh is not mesh:
        c = _tunnel_center(base_mesh, feat.axis, feat.u, feat.v, r, feat.dia)
    if c is None:
        return None
    spec_dia = "spec" in str(feat.source)
    drift = math.hypot(c[0] - feat.u, c[1] - feat.v)
    if spec_dia or mesh_tunnel:
        feat.plug_first = True
    if spec_dia or drift > 1.0:
        if drift > 0.02:
            report.warnings.append(
                f"feature '{feat.name}': channel drifted {drift:.1f}mm to "
                f"({c[0]:.2f}, {c[1]:.2f}) — plugging it there, drilling at "
                f"the declared position ({feat.u:.2f}, {feat.v:.2f})"
            )
        feat.plug_uv = c
    else:
        if drift > 0.02:
            report.warnings.append(
                f"feature '{feat.name}': re-cut snapped to measured tunnel "
                f"centre ({c[0]:.2f}, {c[1]:.2f}), plugged then drilled"
            )
        feat.u, feat.v = c
    if not feat.plug_first:
        feat.dia_extra = max(feat.dia_extra, 0.4)
    return c


def _snap_dedup_features(
    mesh: trimesh.Trimesh,
    features: list[RefineFeature],
    *,
    base_mesh: trimesh.Trimesh | None,
    report: RefineReport,
) -> list[RefineFeature]:
    """Snap re-cuts onto measured tunnel centres and merge near-concentric
    duplicates into a single cut.

    Runs AFTER keep-region restore so `mesh` already holds the original bore
    walls. A feature over an existing tunnel is centred on the MEASURED
    tunnel and its cutter reamed +0.4 mm — the old bore is fully swallowed
    instead of becoming a sliver void next to the new cut. Same-axis
    features within the merge radius are one bore: cut once at the merged
    centre with the larger spec diameter."""
    snapped: dict[int, tuple[float, float]] = {}
    resolved: list[RefineFeature] = []
    for feat in features:
        if feat.kind not in _CUTABLE:
            resolved.append(feat)
            continue
        c = _resolve_snap(feat, mesh, base_mesh, report)
        if c is not None:
            snapped[id(feat)] = c
        resolved.append(feat)

    merged: list[RefineFeature] = []
    for feat in resolved:
        if feat.kind not in _CUTABLE:
            merged.append(feat)
            continue
        target_i = -1
        for i, m in enumerate(merged):
            if m.kind not in _CUTABLE or m.axis != feat.axis:
                continue
            if math.hypot(m.u - feat.u, m.v - feat.v) <= max(
                _merge_radius(m.dia), _merge_radius(feat.dia)
            ):
                target_i = i
                break
        if target_i < 0:
            merged.append(feat)
            continue
        target = merged[target_i]
        keep, drop = target, feat
        if _effective_dia(feat) > _effective_dia(target):
            keep, drop = feat, target
        # spec-sourced keeps drill at their declared position — only measured
        # centres merge; the channel seal tracks whichever tunnel was found
        keep_declared = "spec" in str(keep.source)
        uv = snapped.get(id(keep)) or snapped.get(id(drop)) or (
            (keep.u + drop.u) / 2.0,
            (keep.v + drop.v) / 2.0,
        )
        if keep_declared:
            keep.plug_uv = uv
        else:
            keep.u, keep.v = uv
        keep.dia_extra = max(keep.dia_extra, drop.dia_extra)
        keep.plug_first = keep.plug_first or drop.plug_first
        if keep.plug_uv is None:
            keep.plug_uv = drop.plug_uv
        if keep.plug_first:
            keep.dia_extra = 0.0  # plugged bores drill nominal — no ream
        merged[target_i] = keep
        report.warnings.append(
            f"features '{target.name}' and '{feat.name}' resolve to the same "
            f"bore — merged into one cut ('{keep.name}')"
        )
    return merged


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


def _auto_keep_margin_mm() -> float:
    """Auto keep-ring margin — mirrors
    RuntimeSettings.topopt_auto_keep_margin_mm (default 3.0 mm). Read lazily
    so the executor path stays free of the runtime-config import; falls back
    to the hard default when unavailable."""
    try:
        from cadx.runtime_config import RuntimeSettings

        return float(RuntimeSettings.model_fields["topopt_auto_keep_margin_mm"].default or 3.0)
    except Exception:
        return 3.0


def _auto_anchor_keep_regions(anchors: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """One keep ring per anchor with a resolvable position — mirrors the
    TopOpt auto-keep (`regions.auto_keep_regions`) so keep-restore still
    finds the bore rings when the caller forwarded no explicit keep_regions
    (the in-job refine payload only carries user-declared regions)."""
    out: list[dict[str, Any]] = []
    margin = _auto_keep_margin_mm()
    for i, a in enumerate(anchors or []):
        if _anchor_uv(a) is None:
            continue
        out.append({"kind": "hole", "hole_index": i + 1, "margin_mm": margin})
    return out


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

    When `keep_regions` is empty, one AUTO keep ring per anchor (mirroring
    the TopOpt auto-keep, `_auto_anchor_keep_regions`) is synthesized —
    the in-job refine payload only forwards user-declared regions, and
    without the rings a SIMP-eaten bore leaves nothing to re-cut. Auto
    restores are labelled "auto keep-restore" in the warnings.

    Robustness: both operands must be manifold volumes for the booleans. A
    non-volume base or a sliver patch is repaired once via _ensure_watertight;
    if it still is not a volume the region is skipped with a warning instead
    of aborting the whole restore."""
    from cadx.cad.mesh_ops import _ensure_watertight, heal_seam_slivers

    warnings: list[str] = []
    out = opt_mesh
    restored = 0
    auto_restored = 0
    base = base_mesh
    if not base.is_volume:
        with contextlib.suppress(Exception):
            base = _ensure_watertight(base)
    if not base.is_volume:
        warnings.append("keep-region restore skipped — base mesh is not a volume")
        return out, warnings, 0
    span = (float(base.bounds[0].min()) - 1, float(base.bounds[1].max()) + 1)
    # `_plane_box` restores do NOT go through the boolean union: a thin slab
    # covering the whole part footprint unions into coincident-face seams
    # (float32-quantized manifold output) that trimesh reports as watertight
    # but that sew into boolean-dead B-Reps — the analytic STEP cuts then
    # return empty solids. Vertex snapping restores the flat plane without
    # touching topology.
    union_regions = [r for r in (keep_regions or []) if r.get("kind") != "_plane_box"]
    plane_boxes = [r["box"] for r in (keep_regions or []) if r.get("kind") == "_plane_box"]
    if plane_boxes and out.is_volume:
        snapped = _snap_plane_verts(out, plane_boxes, base.bounds)
        if snapped:
            restored += len(plane_boxes)
            warnings.append(f"plane-snap keep-restore: {snapped} vertices restored")
    tagged: list[tuple[np.ndarray, bool]] = [
        (b, False) for b in _keep_boxes(union_regions, anchors, base.bounds, span, margin_mm)
    ]
    if not keep_regions:
        tagged += [
            (b, True)
            for b in _keep_boxes(
                _auto_anchor_keep_regions(anchors), anchors, base.bounds, span, margin_mm
            )
        ]
    # Build each keep patch once — reused by the merged union and the
    # per-patch fallback.
    tagged_patches = _bridging_keep_patches(out, base, tagged, warnings)
    # Merge all surviving patches into a SINGLE body before unioning:
    # adjacent regions (e.g. a bore ring overlapping a thread ring)
    # otherwise create one coincident-face seam per union — float32-quantized
    # manifold output leaves T-junction slivers there that sew into
    # boolean-dead B-Reps. One merged patch + one union = exactly one seam.
    merged_done = False
    if len(tagged_patches) > 1 and out.is_volume:
        try:
            merged_patch = tagged_patches[0][0]
            for p, _ in tagged_patches[1:]:
                merged_patch = merged_patch.union(p, engine="manifold")
            merged = out.union(merged_patch, engine="manifold")
            if merged is not None and len(getattr(merged, "faces", [])):
                out = merged
                restored += len(tagged_patches)
                auto_restored += sum(1 for _, a in tagged_patches if a)
                merged_done = True
        except Exception as e:
            warnings.append(f"merged keep-restore failed ({e}) — trying per-region")
    if not merged_done:
        out, warnings2, restored2, auto2 = _union_keep_patches(out, tagged_patches)
        warnings += warnings2
        restored += restored2
        auto_restored += auto2
    if auto_restored:
        warnings.append(f"auto keep-restore: {auto_restored} anchor ring(s) restored")
    # every union above ran through manifold3d — its float32 vertex
    # quantization leaves sliver triangles on the seam that sew into
    # degenerate B-Rep edges and stall the later OCC re-cut. Weld them now
    # (lossless-only: returns `out` untouched when the weld isn't clean).
    out = heal_seam_slivers(out)
    return out, warnings, restored


def _bridging_keep_patches(
    out: trimesh.Trimesh,
    base: trimesh.Trimesh,
    tagged: list[tuple[np.ndarray, bool]],
    warnings: list[str],
) -> list[tuple[trimesh.Trimesh, bool]]:
    """`base ∩ box` per keep box — but only patches that overlap `out` by
    real volume. A keep-ring restored into a region the optimizer emptied
    completely floats in air and ships a loose fragment with a
    nominal-looking bore (observed: corner mounting pads restored onto
    nothing became free pucks the verify still reported as Ø8.5-true)."""
    from cadx.cad.mesh_ops import _ensure_watertight

    patches: list[tuple[trimesh.Trimesh, bool]] = []
    for b, is_auto in tagged:
        tag = "auto keep-restore" if is_auto else "keep-region"
        try:
            patch = base.intersection(_box_mesh(b), engine="manifold")
        except Exception:
            patch = None
        if patch is None or not len(getattr(patch, "faces", [])):
            continue
        if not patch.is_volume:
            with contextlib.suppress(Exception):
                patch = _ensure_watertight(patch)
        if not patch.is_volume:
            warnings.append(f"{tag} patch is not a volume — region left smoothed")
            continue
        if out.is_volume:
            overlap_vol = None
            with contextlib.suppress(Exception):
                overlap_vol = float(
                    getattr(out.intersection(patch, engine="manifold"), "volume", 0) or 0
                )
            if overlap_vol is not None and overlap_vol < _MIN_BRIDGE_MM3:
                cx = float((b[0][0] + b[1][0]) / 2)
                cy = float((b[0][1] + b[1][1]) / 2)
                cz = float((b[0][2] + b[1][2]) / 2)
                warnings.append(
                    f"{tag} at ({cx:.0f}, {cy:.0f}, {cz:.0f}) would float — "
                    "region left smoothed"
                )
                continue
        patches.append((patch, is_auto))
    return patches


def _union_keep_patches(
    out: trimesh.Trimesh,
    tagged_patches: list[tuple[trimesh.Trimesh, bool]],
) -> tuple[trimesh.Trimesh, list[str], int, int]:
    """Per-region fallback: `out union patch` per precomputed keep patch."""
    warnings: list[str] = []
    restored = 0
    auto_restored = 0
    for patch, is_auto in tagged_patches:
        tag = "auto keep-restore" if is_auto else "keep-region"
        try:
            if not out.is_volume:
                warnings.append("optimized mesh lost volume-ness — restore aborted")
                break
            merged = out.union(patch, engine="manifold")
            if merged is None or not len(getattr(merged, "faces", [])):
                warnings.append(f"{tag} union produced empty mesh — region left smoothed")
                continue
            out = merged
            restored += 1
            auto_restored += int(is_auto)
        except Exception as e:
            warnings.append(f"{tag} failed ({e}) — region left smoothed")
    return out, warnings, restored, auto_restored


def _snap_plane_verts(out: trimesh.Trimesh, plane_boxes: list[Any], base_bounds: np.ndarray) -> int:
    """Snap outward-facing vertices inside each plane slab onto the base
    plane. Restores flattened top/bottom skins after smoothing shrank them —
    no boolean, so no coincident-face seams. Returns snapped vertex count."""
    lo_b = np.asarray(base_bounds[0], dtype=float)
    hi_b = np.asarray(base_bounds[1], dtype=float)
    normals = out.vertex_normals
    verts = out.vertices
    snapped = 0
    for raw in plane_boxes:
        b = np.asarray(raw, dtype=float)
        lo, hi = np.asarray(b[0], dtype=float), np.asarray(b[1], dtype=float)
        ext = hi - lo
        axis = int(np.argmin(ext))
        if ext[axis] > 5.0 or ext[axis] <= 0:
            continue  # not a thin slab — leave it to the union path
        top = (lo[axis] + hi[axis]) * 0.5 > (lo_b[axis] + hi_b[axis]) * 0.5
        plane = float(hi_b[axis] if top else lo_b[axis])
        in_box = np.all((verts >= lo) & (verts <= hi), axis=1)
        facing = normals[:, axis] > 0.6 if top else normals[:, axis] < -0.6
        idx = np.nonzero(in_box & facing)[0]
        if not len(idx):
            continue
        verts[idx, axis] = plane
        snapped += len(idx)
    return snapped


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
            continue  # handled by _snap_plane_verts — no boolean union

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


def _try_difference(
    mesh: trimesh.Trimesh, cutter: trimesh.Trimesh
) -> tuple[trimesh.Trimesh | None, str]:
    """manifold difference → (result, error); (None, err) on empty/failure."""
    try:
        res = mesh.difference(cutter, engine="manifold")
    except Exception as e:
        return None, str(e)
    if res is None or not len(getattr(res, "faces", [])):
        return None, ""
    return res, ""


def _feature_host_main(mesh: trimesh.Trimesh, feat: RefineFeature, entry: float, span: float) -> bool | None:
    """True when the material ringing the bore belongs to the mesh's
    largest body — a keep-restored ring floating in an emptied region
    still 'measures' a perfect bore but carries no load. None = could not
    tell (split/proximity failure) — callers then keep the benefit of the
    doubt."""
    try:
        parts = [p for p in mesh.split(only_watertight=False) if len(getattr(p, "faces", []))]
    except Exception:
        return None
    if len(parts) <= 1:
        return True
    ax, au, av = _AXES[feat.axis]
    inward = -1.0 if feat.axis[0] == "+" else 1.0
    depth = min(feat.depth_mm or span, span)
    centre = np.zeros(3)
    centre[au], centre[av] = feat.u, feat.v
    centre[ax] = entry + inward * depth * 0.5
    ring_r = feat.dia / 2.0 + 0.75
    probes = []
    for d_ax in (au, av):
        for sgn in (1.0, -1.0):
            pt = centre.copy()
            pt[d_ax] += sgn * ring_r
            probes.append(pt)
    probe_arr = np.array(probes)
    main = max(parts, key=lambda p: float(getattr(p, "volume", 0) or 0))
    best_d, host = np.inf, None
    for p in parts:
        try:
            dists = np.abs(trimesh.proximity.signed_distance(p, probe_arr))
        except Exception:
            continue
        d = float(dists.min())
        if d < best_d:
            best_d, host = d, p
    return None if host is None else bool(host is main)


def _record_applied(
    report: RefineReport,
    feat: RefineFeature,
    mesh: trimesh.Trimesh,
    entry: float,
    span: float,
    cut_depth_mm: float | None = None,
) -> None:
    """Applied-feature report entry + post-cut diameter sanity measurement."""
    entry_rec: dict[str, Any] = {
        "name": feat.name,
        "kind": feat.kind,
        "axis": feat.axis,
        "u": round(feat.u, 3),
        "v": round(feat.v, 3),
        "dia": round(feat.dia, 2),
        "depth_mm": feat.depth_mm,
        "through": bool(feat.through),
        "source": feat.source,
    }
    if cut_depth_mm is not None and (
        feat.depth_mm is None or abs(cut_depth_mm - feat.depth_mm) > 0.05
    ):
        entry_rec["cut_depth_mm"] = round(cut_depth_mm, 2)
    if feat.thread_spec:
        entry_rec["thread_spec"] = feat.thread_spec
    if feat.helical is not None:
        entry_rec["helical"] = feat.helical
    if feat.dia_extra:
        entry_rec["ream_extra_mm"] = round(feat.dia_extra, 2)
    if feat.plug_first:
        entry_rec["plugged"] = True
    try:
        requested = _effective_dia(feat) + feat.dia_extra
        measured = _measure_feature_dia(mesh, feat, entry, span)
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
    host = _feature_host_main(mesh, feat, entry, span)
    if host is False:
        entry_rec["host_fragmented"] = True
        report.warnings.append(
            f"feature '{feat.name}': bore sits in a disconnected fragment — "
            "the region was optimized away and the restored ring floats free"
        )
    report.applied.append(entry_rec)


def _plug_channel(
    out: trimesh.Trimesh,
    feat: RefineFeature,
    entry: float,
    span: float,
    report: RefineReport,
) -> tuple[trimesh.Trimesh, float]:
    """spec'd bore over a wavy channel: fill the measured void first so the
    nominal drill leaves exact walls — a plain cut can only widen, never
    shrink, an existing tunnel. On plug failure fall back to the +0.4
    ream (oversized but clean) rather than cutting ragged walls."""
    ax = _AXES[feat.axis][0]
    plug = _plug_cylinder(out, feat)
    if plug is None:
        feat.plug_first = False
        feat.dia_extra = max(feat.dia_extra, 0.4)
        report.warnings.append(
            f"feature '{feat.name}': no plug geometry at measured tunnel — "
            "reamed +0.4 mm instead of drilling at spec dia"
        )
        return out, entry
    try:
        filled = out.union(plug, engine="manifold")
        if filled is None or not len(getattr(filled, "faces", [])):
            filled = None
    except Exception:
        filled = None
    if filled is None:
        feat.plug_first = False
        feat.dia_extra = max(feat.dia_extra, 0.4)
        report.warnings.append(
            f"feature '{feat.name}': channel plug union failed — "
            "reamed +0.4 mm instead of drilling at spec dia"
        )
        return out, entry
    out = filled
    # seat the drill on the plug surface — the channel surface
    # entry measured above is gone now
    new_entry = _entry_point(out, feat.axis, feat.u, feat.v)
    if new_entry is not None:
        entry = new_entry
    else:
        entry = float(out.bounds[1][ax] if feat.axis[0] == "+" else out.bounds[0][ax])
    return out, entry


def _apply_cut(out: trimesh.Trimesh, feat: RefineFeature, report: RefineReport) -> trimesh.Trimesh:
    """One re-cut: build the cutter, difference it, retry a failed helical
    groove as a plain tap-drill bore (the bore is never lost), and record the
    outcome. Returns the (possibly unchanged) mesh."""
    ax, _, _ = _AXES[feat.axis]
    entry = _entry_point(out, feat.axis, feat.u, feat.v)
    entry_reseated = False
    if entry is None:
        # ray at the bore centre hits nothing — the channel is already
        # open (tunnel through the part) OR the region was fully
        # optimized away. The two cases look identical on the centre ray;
        # the bore-ring test separates them: no wall material around the
        # position means there is nothing to drill INTO — seating the
        # cutter at the bbox face would shave a nick off an unrelated
        # surface and report it as the spec's bore (the eval's Ø3.8 mm
        # hole 43 mm off-anchor). Skip honestly instead.
        if not _bore_ring_material(out, feat.axis, feat.u, feat.v, feat.dia):
            report.skipped.append(
                {
                    "name": feat.name,
                    "kind": feat.kind,
                    "reason": "no material at feature position — region optimized away",
                }
            )
            report.warnings.append(
                f"feature '{feat.name}': no entry surface and no material "
                "ring at the anchor — the optimized part removed the host "
                "region; the cut is skipped, not re-seated"
            )
            return out
        # open channel with surviving walls — re-cutting them is still
        # the point of refine, so seat the cutter at the bbox face and
        # let the boolean decide; warn so the report stays honest.
        entry = float(out.bounds[1][ax] if feat.axis[0] == "+" else out.bounds[0][ax])
        entry_reseated = True
        report.warnings.append(
            f"feature '{feat.name}': no entry surface — channel already "
            "open; re-cutting from the bbox face"
        )
    span = float(out.bounds[1][ax] - out.bounds[0][ax])
    if feat.plug_first:
        out, entry = _plug_channel(out, feat, entry, span, report)
    depth_override: float | None = None
    if feat.through:
        if entry_reseated:
            # open channel on the centre ray — only the walls carry
            # material; a full-span cutter recuts them wherever they are
            depth_override = span + 2.0
        else:
            declared = feat.depth_mm if feat.depth_mm is not None else span + 2.0
            needed = _through_cut_depth(out, feat, entry, span)
            if needed > declared + 0.05:
                depth_override = needed
                report.warnings.append(
                    f"feature '{feat.name}': through-cut extended to "
                    f"{needed:.1f} mm — material below the declared span "
                    "would seal the bore"
                )
    cutter = _cutter_mesh(feat, entry, span, depth_override=depth_override)
    _orient_cutter(cutter, feat.axis, feat.u, feat.v, entry)
    # overlap sanity: AABBs must intersect (cheap) — a fully disjoint
    # cutter would boolean fine but remove nothing
    cb = cutter.bounds
    if cb is None or np.any(cb[0] >= out.bounds[1]) or np.any(cb[1] <= out.bounds[0]):
        report.skipped.append(
            {"name": feat.name, "kind": feat.kind, "reason": "cutter misses the part"}
        )
        return out
    res, bool_err = _try_difference(out, cutter)
    if res is None and feat.helical:
        # the helical groove wrecked the boolean — never lose the bore:
        # retry with the plain tap-drill cutter
        report.warnings.append(
            f"feature '{feat.name}': helical groove boolean failed "
            f"({bool_err or 'empty result'}) — retried as a plain tap-drill bore"
        )
        cutter = _cutter_mesh(
            replace(feat, helical=False), entry, span, depth_override=depth_override
        )
        _orient_cutter(cutter, feat.axis, feat.u, feat.v, entry)
        res, bool_err = _try_difference(out, cutter)
    if res is None:
        reason = "boolean produced empty mesh"
        if bool_err:
            reason += f": {bool_err}"
        report.skipped.append({"name": feat.name, "kind": feat.kind, "reason": reason})
        return out
    out = res
    _record_applied(
        report,
        feat,
        out,
        entry,
        span,
        cut_depth_mm=depth_override if depth_override is not None else feat.depth_mm,
    )
    return out


def _repaired_input_mesh(opt_mesh: trimesh.Trimesh, report: RefineReport) -> trimesh.Trimesh | None:
    """Watertight/winding repair of the refine input — returns None when the
    input can't be repaired into a volume (refine skipped).

    Voxel-fill repairs can silently inflate an open mesh (a leaking part is
    closed by padding the voxel hull) — a materially different "refined"
    part is worse than a warning, so volume/bbox drift is reported."""
    out = opt_mesh
    if out.is_watertight and out.is_winding_consistent:
        return out
    from cadx.cad.mesh_ops import _ensure_watertight

    try:
        repaired = _ensure_watertight(out)
    except Exception as e:
        report.warnings.append(f"mesh not watertight — refine skipped: {e}")
        return None
    if abs(repaired.volume - out.volume) > 0.05 * abs(out.volume) or np.any(
        repaired.bounds[1] - repaired.bounds[0] > (out.bounds[1] - out.bounds[0]) + 1.0
    ):
        report.warnings.append(
            "watertight repair changed geometry "
            f"(vol {out.volume:.0f}->{repaired.volume:.0f} mm3) — verify result"
        )
    return repaired


def apply_refine(
    opt_mesh: trimesh.Trimesh,
    features: list[RefineFeature],
    *,
    base_mesh: trimesh.Trimesh | None = None,
    keep_regions: list[dict[str, Any]] | None = None,
    anchors: list[dict[str, Any]] | None = None,
    restore_keeps: bool = True,
    helical_threads: bool = True,
) -> tuple[trimesh.Trimesh, RefineReport]:
    """Re-cut `features` on `opt_mesh` via manifold difference; restore keep
    surfaces from `base_mesh`. Never raises on a single bad feature.

    Order is fixed and load-bearing: keep-surface RESTORE runs FIRST (unions
    `base_mesh ∩ keep_box` back; with no explicit `keep_regions` one auto
    ring per anchor is synthesized), then re-cuts are SNAPPED onto measured
    post-restore tunnel centres, then each feature is CUT. Snapping before
    restore would miss bore walls SIMP ate; cutting before restore would be
    buried by the restored patch.

    `helical_threads` is the auto-default for thread/tap features whose
    `helical` field is unset: when a pitch is derivable the re-cut adds the
    real helical V-groove (tap-drill bore is always kept as fallback).
    Explicit per-feature `helical=True/False` always wins."""
    report = RefineReport()
    out = _repaired_input_mesh(opt_mesh, report)
    if out is None:
        return opt_mesh, report

    # restore FIRST so re-cuts stay crisp — a restored keep patch may overlap
    # a feature footprint; the cutter must win (exact bore > base surface)
    if restore_keeps and base_mesh is not None:
        try:
            out, w, n = restore_keep_surface(out, base_mesh, keep_regions or [], anchors or [])
            report.warnings += w
            report.restored_regions = n
        except Exception as e:
            report.warnings.append(f"keep-region restore failed: {e}")

    # deferred spec anchors: resolve admitted them without a base-mesh
    # tunnel on the condition that keep-restore (or surviving opt material)
    # leaves real material at the anchor — verify on the POST-restore mesh.
    # No material → the bore would cut into a void; skip honestly.
    if features:
        gated: list[RefineFeature] = []
        for feat in features:
            if feat.admit_on_material and not _anchor_material(out, feat.u, feat.v, feat.axis):
                report.skipped.append(
                    {
                        "name": feat.name,
                        "kind": feat.kind,
                        "reason": "no material at anchor after keep-restore",
                    }
                )
                report.warnings.append(
                    f"feature '{feat.name}': spec anchor admitted pending "
                    "keep-restore found no material — cut skipped"
                )
                continue
            gated.append(feat)
        features = gated

    # snap re-cuts onto the measured tunnel centre and merge near-concentric
    # duplicates — restore ran first, so `out` already holds the original
    # bore walls; cutting beside them leaves a figure-8 sliver void
    if features:
        features = _snap_dedup_features(out, features, base_mesh=base_mesh, report=report)

    # helical default: groove every thread/tap whose pitch is derivable —
    # an M8 spec that degrades to a plain Ø6.8 cylinder is the missing-thread
    # defect; explicit helical=False on the feature always wins
    for feat in features:
        if feat.kind in ("thread", "tap") and feat.helical is None:
            feat.helical = bool(helical_threads) and _thread_pitch(feat) is not None

    for feat in features:
        if feat.kind not in _CUTABLE:
            report.skipped.append({"name": feat.name, "kind": feat.kind, "reason": "not cuttable"})
            continue
        try:
            out = _apply_cut(out, feat, report)
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
    return _watertight_postpass(out, report), report


def _watertight_postpass(out: trimesh.Trimesh, report: RefineReport) -> trimesh.Trimesh:
    """In-memory watertight repair of the refined mesh.

    Keep-region unions and manifold cuts can leave hairline leaks in `out`.
    The file-level exporter repairs the STL on disk but not this in-memory
    mesh — a leaky mesh sews into a B-Rep that reports is_valid yet returns
    empty solids on OCC booleans, silently degrading the analytic STEP.
    Repair in memory so every downstream consumer (analysis, sew, verify
    probes, analytic cuts) sees the same topology. Same ladder as
    ``repair_mesh_file``, drift-guarded so a voxel rebuild can never
    balloon the lattice."""
    if out.is_watertight:
        return out
    try:
        out.merge_vertices()
        out.update_faces(out.nondegenerate_faces())
        out.remove_unreferenced_vertices()
        out.fill_holes()
        out.fix_normals()
    except Exception:
        pass
    if not out.is_watertight:
        try:
            from cadx.cad.mesh_ops import _ensure_watertight, _rebuild_drift

            rebuilt = _ensure_watertight(out)
            drift = _rebuild_drift(out, rebuilt)
            if drift is None:
                out = rebuilt
            else:
                report.warnings.append(f"watertight repair skipped — {drift}")
        except Exception as e:
            report.warnings.append(f"watertight repair failed: {e}")
    if not out.is_watertight:
        report.warnings.append("refined mesh is not watertight — STEP/analytic cuts may degrade")
    return out


# --------------------------------------------------------------------------- #
# export / verification helpers                                                 #
# --------------------------------------------------------------------------- #

# applied refine kinds that must leave a real through-feature in the mesh
_VERIFY_KINDS = {"hole", "tunnel", "thread", "tap"}


def applied_refine_features(
    features: list[RefineFeature], report: RefineReport
) -> list[RefineFeature]:
    """The ``RefineFeature`` objects that were actually applied — the
    ``report.applied`` records carry no pose (u/v), so analytic STEP export
    needs the feature objects filtered by the applied names."""
    names = {str(e.get("name") or "") for e in report.applied}
    return [f for f in features if f.name in names]


def _verify_uv(a: dict[str, Any], axis: str) -> tuple[float, float] | None:
    """In-plane (u, v) coords of an applied-entry/anchor/request dict for `axis`.

    Applied refine records carry the cut position as ``u``/``v`` — those are
    already the in-plane components for `axis` and win over any ``pos``
    triple (whose components the dict may not even fill)."""
    _ax, au, av = _AXES.get(axis, _AXES["+z"])
    if a.get("u") is not None and a.get("v") is not None:
        try:
            return float(a["u"]), float(a["v"])
        except (TypeError, ValueError):
            pass
    pos = a.get("pos") or []
    try:
        if len(pos) > max(au, av) and pos[au] is not None and pos[av] is not None:
            return float(pos[au]), float(pos[av])
    except (TypeError, ValueError):
        pass
    if axis in ("+z", "-z") and a.get("x") is not None and a.get("y") is not None:
        try:
            return float(a["x"]), float(a["y"])
        except (TypeError, ValueError):
            return None
    return None


def _verify_expected_uv(
    entry: dict[str, Any],
    anchors: list[dict[str, Any]] | None,
    request_features: list[dict[str, Any]] | None,
    axis: str = "+z",
) -> tuple[float, float] | None:
    """Expected in-plane (u, v) of an applied refine feature on `axis`.

    ``applied`` records usually carry no position — resolve it from the
    request feature / anchor the cut was derived from. Anchor-derived names
    are ``<name>`` or ``anchor <index>``; spec-renamed features fall back to
    the same-axis anchor closest in diameter."""
    direct = _verify_uv(entry, axis)
    if direct is not None:
        return direct
    name = str(entry.get("name") or "")
    for r in request_features or []:
        if str(r.get("name") or "") == name:
            uv = _verify_uv(r, axis)
            if uv is not None:
                return uv
    dia = float(entry.get("requested_dia_mm") or entry.get("dia") or 0.0)
    best: tuple[float, float] | None = None
    best_dev: float | None = None
    name_hits: list[tuple[float, float]] = []
    for a in anchors or []:
        if str(a.get("axis") or "+z") != axis:
            continue
        uv = _verify_uv(a, axis)
        if uv is None:
            continue
        if str(a.get("name") or f"anchor {a.get('index', '?')}") == name:
            name_hits.append(uv)
        if dia > 0:
            dev = abs(float(a.get("dia") or 0.0) - dia)
            if best_dev is None or dev < best_dev:
                best, best_dev = uv, dev
    if name_hits:
        # twin features share name+dia (e.g. two M6 bores) — the applied
        # entry's own recorded u/v disambiguates which anchor is expected
        eu, ev = entry.get("u"), entry.get("v")
        if isinstance(eu, int | float) and isinstance(ev, int | float):
            return min(name_hits, key=lambda uv: math.hypot(uv[0] - eu, uv[1] - ev))
        return name_hits[0]
    return best


def _probe_blind_feature(
    mesh: trimesh.Trimesh,
    entry: dict[str, Any],
    axis: str,
    uv: tuple[float, float],
    dia: float,
    *,
    depth_frac: float,
    dia_tol_frac: float,
    log: Any,
) -> tuple[bool | None, dict[str, Any] | None]:
    """Blind-pocket rescue verdict via ``anchors3d.probe_blind_features``.

    The surface scan can't see a shallow blind bore on some axes — probe the
    refined mesh for just this feature. Returns ``(verdict, record)``:
    verdict True when the probe confirms the feature (found; ≥`depth_frac` of
    requested depth — or open tunnel when no depth was requested or the bore
    went deeper; measured diameter inside the scan tolerance), False on a
    probe miss, None when the probe is unavailable or crashed (caller keeps
    the scan verdict). ``record`` is the raw probe dict for reporting."""
    try:
        from cadx.cad.fem import anchors3d
    except Exception:
        return None, None
    probe = getattr(anchors3d, "probe_blind_features", None)
    if not callable(probe):
        return None, None
    _ax, au, av = _AXES[axis]
    pos: list[Any] = [None, None, None]
    pos[au], pos[av] = uv[0], uv[1]
    depth_req = entry.get("depth_mm")
    try:
        recs = probe(
            mesh,
            [
                {
                    "name": entry.get("name"),
                    "kind": entry.get("kind") or "hole",
                    "axis": axis,
                    "pos": pos,
                    "x": uv[0],
                    "y": uv[1],
                    "dia_mm": dia,
                    "dia": dia,
                    "depth_mm": depth_req,
                }
            ],
            log=log,
        )
    except Exception as e:
        log.warning("refine.verify_probe_failed", feature=str(entry.get("name")), error=str(e))
        return None, None
    rec = recs[0] if recs else None
    if rec is None:
        return None, None
    if not rec.get("found"):
        return False, rec
    measured_depth = rec.get("measured_depth_mm")
    if depth_req:
        ok_depth = bool(rec.get("open")) or (
            measured_depth is not None and float(measured_depth) >= depth_frac * float(depth_req)
        )
    else:
        ok_depth = bool(rec.get("open"))
    measured_dia = rec.get("measured_dia_mm")
    ok_dia = (
        measured_dia is None or dia <= 0 or abs(float(measured_dia) - dia) <= dia_tol_frac * dia
    )
    return bool(ok_depth and ok_dia), rec


def _scan_has_feature(
    scanned: list[dict[str, Any]],
    uv: tuple[float, float],
    dia: float,
    want: set[str],
    xy_tol_mm: float,
    dia_tol_frac: float,
    axis: str = "+z",
) -> bool:
    """A scanned anchor of a wanted kind sits within tolerance of (u, v)."""
    _ax, au, av = _AXES.get(axis, _AXES["+z"])
    for t in scanned:
        if str(t.get("axis") or "+z") != axis or str(t.get("kind") or "") not in want:
            continue
        tpos = t.get("pos") or []
        try:
            if len(tpos) > max(au, av) and tpos[au] is not None and tpos[av] is not None:
                du, dv = float(tpos[au]) - uv[0], float(tpos[av]) - uv[1]
            elif axis in ("+z", "-z"):
                if t.get("x") is None or t.get("y") is None:
                    continue
                du = float(t["x"]) - uv[0]
                dv = float(t["y"]) - uv[1]
            else:
                continue
        except (TypeError, ValueError):
            continue
        if math.hypot(du, dv) > xy_tol_mm:
            continue
        if dia <= 0 or abs(float(t.get("dia") or 0.0) - dia) <= dia_tol_frac * dia:
            return True
    return False


def refine_verify(
    mesh: trimesh.Trimesh | str | Path,
    refine_report: dict[str, Any],
    *,
    anchors: list[dict[str, Any]] | None = None,
    request_features: list[dict[str, Any]] | None = None,
    xy_tol_mm: float = 3.0,
    dia_tol_frac: float = 0.25,
    depth_frac: float = 0.8,
    log: Any = None,
) -> list[dict[str, Any]]:
    """Applied cuttable features that left NO matching feature in the mesh.

    The refine report's ``applied`` list is self-reported — a cut landing in
    a SIMP void (``dia_deviation_mm`` in the tens of mm) still counts as
    applied. Rescan the refined mesh and require a scanned ``hole`` (through
    cuts) or ``dimple`` (blind cuts, ``depth_mm`` set) within `xy_tol_mm` xy
    / ±`dia_tol_frac` dia of the expected position. Features the surface
    scan misses get a second verdict from
    ``anchors3d.probe_blind_features`` (blind pockets the heightfield scan
    can't see); a missing function or a probe error degrades gracefully to
    the scan-only verdict.

    Returns miss records ``{"entry", "x", "y", "dia"}`` — empty means every
    applied feature verified."""
    logger = log if log is not None else get_logger(__name__)
    if not isinstance(mesh, trimesh.Trimesh):
        try:
            p = Path(mesh)
            if not p.exists():
                return []
            loaded = trimesh.load(str(p), force="mesh")
        except Exception:
            return []
        if not isinstance(loaded, trimesh.Trimesh):
            return []
        mesh = loaded
    try:
        scanned = anchors3d.scan_mesh_anchors(mesh)
    except Exception as e:
        logger.warning("refine.verify_scan_failed", error=str(e))
        return []
    misses: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for entry in refine_report.get("applied") or []:
        if str(entry.get("kind") or "hole") not in _VERIFY_KINDS:
            continue
        axis = str(entry.get("axis") or "+z")
        if axis not in _AXES:
            continue
        uv = _verify_expected_uv(entry, anchors, request_features, axis)
        if uv is None:
            continue  # no position to check against — cannot verify
        dia = float(entry.get("requested_dia_mm") or entry.get("dia") or 0.0)
        if entry.get("host_fragmented"):
            # a hole measured on a floating fragment is not a feature of
            # the part — count it as a miss even when the surface scan
            # sees the bore
            misses.append(
                {"entry": entry, "x": uv[0], "y": uv[1], "dia": dia, "reason": "host_fragmented"}
            )
            continue
        # through-features verify on a real through-tunnel — a cavity probe
        # cannot distinguish 'bore pierced the part' from 'pocket exists',
        # so the surface scan is authoritative for them. Blind features get
        # the pocket probe rescue the scan can't see.
        through = entry.get("through")
        if through is None:
            through = not entry.get("depth_mm")
        found, probe_rec = _verify_entry(
            mesh,
            scanned,
            entry,
            axis,
            uv,
            dia,
            through,
            xy_tol_mm=xy_tol_mm,
            dia_tol_frac=dia_tol_frac,
            depth_frac=depth_frac,
            log=logger,
        )
        if probe_rec is not None:
            probes.append({**probe_rec, "verified": found})
        if not found:
            misses.append({"entry": entry, "x": uv[0], "y": uv[1], "dia": dia})
    if probes:
        refine_report["probes"] = probes
    return misses


def _verify_entry(
    mesh: trimesh.Trimesh,
    scanned: list[dict[str, Any]],
    entry: dict[str, Any],
    axis: str,
    uv: tuple[float, float],
    dia: float,
    through: bool,
    *,
    xy_tol_mm: float,
    dia_tol_frac: float,
    depth_frac: float,
    log: Any,
) -> tuple[bool, dict[str, Any] | None]:
    """One applied entry → (verified, probe record). The scan is
    authoritative for through-features; blind ones get the pocket probe."""
    want = {"hole"} if through else {"dimple"}
    if _scan_has_feature(scanned, uv, dia, want, xy_tol_mm, dia_tol_frac, axis):
        return True, None
    if through:
        return False, None
    verdict, probe_rec = _probe_blind_feature(
        mesh,
        entry,
        axis,
        uv,
        dia,
        depth_frac=depth_frac,
        dia_tol_frac=dia_tol_frac,
        log=log,
    )
    return bool(verdict), probe_rec  # None (probe unavailable) keeps the miss
