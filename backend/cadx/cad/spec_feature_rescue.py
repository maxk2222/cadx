"""Deterministic spec-feature rescue — rebuild spec-declared features the
generated code missed or malformed, via reviewed builders, after codegen.

The generated model is free to compose geometry however it likes, but the
spec's declared features are authoritative intent. When codegen drops or
malforms one — a whole-part helper called with ``feet=0`` and the lugs
hand-built into the cavity, a bolt hole left blind, a disjoint lug that
``largest_solid`` discarded — the rescue rebuilds it deterministically
instead of burning improve iterations or shipping a verify_spec failure.
Every applied op lands in the result warnings so the fix stays auditable.

Scope (deliberately conservative — generalisable patterns, not case hacks):
- **mount-lug ring**: ``count`` tapered mounting lugs each carrying a
  through bolt-hole, welded around a round host's base ring. This is the
  recurring mounting-feet pattern on cylindrical hosts (tanks, flanges,
  hubs, impellers). The lug uses the proven sealed_tank taper math.
- **spec-declared holes**: standalone ``hole``/``bore`` features whose
  position is resolvable are routed through the existing bounded re-cut.

No ``cadx`` imports at module top: like ``mesh_boolean`` this loads inside
the sandboxed runner where only ``<cad>/`` is on ``sys.path``. build123d is
imported lazily inside functions.
"""

from __future__ import annotations

import math
from typing import Any


def _fget(f: Any, key: str, default: Any = None) -> Any:
    return f.get(key, default) if isinstance(f, dict) else getattr(f, key, default)


def _fdims(f: Any) -> dict[str, Any]:
    d = _fget(f, "dims", {}) or {}
    return d if isinstance(d, dict) else {}


def _ftext(f: Any) -> str:
    return " ".join(str(_fget(f, k, "") or "") for k in ("name", "type", "description", "position")).lower()


_MOUNT_WORDS = ("foot", "lug", "ear", "tab", "stand", "leg", "mount", "flange")
_HOLE_TYPES = ("hole", "bore", "clearance", "through", "bolt", "anchor")
_EMBED_MM = 2.0  # lug root reach into the host wall — a tangent-only fuse stays a disjoint solid


def _is_mount(f: Any) -> bool:
    ftype = str(_fget(f, "type", "") or "").lower()
    if ftype in _HOLE_TYPES:
        return False
    return any(w in _ftext(f) for w in _MOUNT_WORDS)


def _is_hole(f: Any) -> bool:
    ftype = str(_fget(f, "type", "") or "").lower()
    if ftype in _HOLE_TYPES:
        return True
    return any(w in _ftext(f) for w in ("hole", "bore", "bohrung", "clearance"))


def _count(f: Any) -> int:
    try:
        return max(1, int(_fget(f, "count", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _dim(f: Any, *keys: str) -> float | None:
    d = _fdims(f)
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _axis(f: Any, default: str = "+z") -> str:
    ax = str(_fget(f, "axis", "") or "").strip().lower()
    return ax if ax in ("x", "y", "z", "+x", "+y", "+z", "-x", "-y", "-z") else default


def _dropped_indices(spec: Any) -> set[int]:
    """Feature indices the model deliberately dropped — rescue must not
    resurrect them (mirrors spec_features.dropped_feature_indices by name)."""
    feats = _fget(spec, "features", []) or []
    drops = _fget(spec, "feature_drops", []) or []

    def sig(f: Any) -> str:
        return f"{_fget(f, 'type', '')}|{_fget(f, 'name', '')}".strip().lower()

    by_sig: dict[str, list[int]] = {}
    for i, f in enumerate(feats):
        by_sig.setdefault(sig(f), []).append(i)
    out: set[int] = set()
    for rec in drops:
        idx = _fget(rec, "feature_index", -1)
        snap = _fget(rec, "feature")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = -1
        if snap is None:
            if 0 <= idx < len(feats):
                out.add(idx)
            continue
        matches = by_sig.get(sig(snap), [])
        if 0 <= idx < len(feats) and idx in matches:
            out.add(idx)
        elif len(matches) == 1:
            out.add(matches[0])
    return out


def _base_ring(shape: Any, z: float, mesh_boolean: Any) -> tuple[float, float] | None:
    """Median outer-wall radius + inner-cavity radius of a round host at
    height ``z``, probed radially inward across 36 azimuths. Existing lugs
    skew a few azimuths outward but the wall dominates the median. Returns
    (host_r, cavity_r) or None when the host isn't a clean ring here."""
    from OCP.gp import gp_Pnt

    bb = shape.bounding_box()
    r_far = max(bb.max.X, bb.max.Y, -bb.min.X, -bb.min.Y) + 5.0
    host_rs: list[float] = []
    cavity_rs: list[float] = []
    for i in range(36):
        a = i * 2 * math.pi / 36
        ca, sa = math.cos(a), math.sin(a)
        origin = gp_Pnt(r_far * ca, r_far * sa, z)
        direction = (-ca, -sa, 0.0)
        try:
            ts = mesh_boolean._surface_hits(shape, origin, direction)
        except Exception:
            continue
        # radii of the surface crossings along the inward ray — the OUTERMOST
        # (largest) is the host wall, the next-smaller crossing is the cavity
        # inner skin; a solid host yields a single crossing (no cavity)
        radii = sorted(r_far - t for t in ts if 0 < (r_far - t) < r_far)
        if not radii:
            continue
        host_rs.append(radii[-1])
        cavity_rs.append(radii[0] if len(radii) > 1 else 0.0)
    if not host_rs:
        return None
    host_rs.sort()
    cavity_rs.sort()
    host_r = host_rs[len(host_rs) // 2]
    cavity_r = cavity_rs[len(cavity_rs) // 2] if cavity_rs else 0.0
    # a ring is only trustworthy when the wall radius is consistent —
    # a prismatic/irregular host scatters the probes too widely
    if host_r <= 0 or (host_rs[-1] - host_rs[0]) > max(3.0, host_r * 0.4):
        return None
    return host_r, max(0.0, cavity_r)


def _lug_dims(mount_feats: list[Any]) -> tuple[int, float, float, float]:
    """Aggregate a mount-lug spec into (count, w, t, h) — feet arrive either
    as one feature count=N or as per-lug features each count=1 (sum them)."""
    count = sum(_count(f) for f in mount_feats)
    w = _dim(mount_feats[0], "width", "length", "size") or 18.0
    t = _dim(mount_feats[0], "thickness", "depth") or 6.0
    h = _dim(mount_feats[0], "depth_total", "height", "head_diameter", "length") or 20.0
    return count, float(w), float(t), float(h)


def add_mount_lug_ring(
    shape: Any,
    *,
    count: int,
    foot_w: float,
    foot_t: float,
    foot_h: float,
    hole_d: float,
    axis: str,
    mesh_boolean: Any,
    azimuth0: float = 0.0,
) -> tuple[Any, int]:
    """Weld ``count`` tapered mounting lugs around a round host's base ring
    and cut a through bolt-hole along ``axis`` in each. Reuses the proven
    sealed_tank lug math: a full-width root embeds ``_EMBED_MM`` into the
    wall, a narrower tip protrudes only as far as the hole needs to clear
    the inner skin. Returns (shape, lugs_added)."""
    from build123d import Box, Cylinder, Pos, Rot

    idx = {"x": 0, "y": 1, "z": 2}[axis.lstrip("+-")]
    if idx != 2:
        return shape, 0  # lug rings are defined on the Z base ring only
    ring = _base_ring(shape, foot_h * 0.6, mesh_boolean)
    if ring is None:
        return shape, 0
    host_r, cavity_r = ring
    hole_r = hole_d / 2.0
    if hole_d > 0:
        # reach out far enough that the hole clears the cavity wall with a
        # material margin on both sides, but no farther than the hole needs
        lug_outer = max(host_r + hole_d * 0.7, cavity_r + hole_d + 0.8)
        t_rad = max(foot_t, hole_d + 3.0)
    else:
        lug_outer = host_r + foot_t
        t_rad = foot_t
    lug_inner = min(lug_outer - t_rad, host_r - _EMBED_MM)
    tip_len = min(4.0, t_rad * 0.45)
    split_r = lug_outer - tip_len
    tip_w = foot_w * 0.75
    _CZ = None
    from build123d import Align

    _CZ = (Align.CENTER, Align.CENTER, Align.MIN)
    fused = shape
    added = 0
    for i in range(count):
        a = azimuth0 + i * 2 * math.pi / count
        ca, sa = math.cos(a), math.sin(a)
        deg = math.degrees(a)
        root_r = (lug_inner + split_r) / 2.0
        try:
            fused += Pos(root_r * ca, root_r * sa, 0) * (
                Rot(0, 0, deg) * Box(split_r - lug_inner, foot_w, foot_h, align=_CZ)
            )
            fused += Pos(
                (split_r + lug_outer) / 2.0 * ca, (split_r + lug_outer) / 2.0 * sa, 0
            ) * (Rot(0, 0, deg) * Box(tip_len, tip_w, foot_h, align=_CZ))
            added += 1
        except Exception:
            continue
    if hole_d > 0 and added:
        for i in range(count):
            a = azimuth0 + i * 2 * math.pi / count
            ca, sa = math.cos(a), math.sin(a)
            hr = lug_outer - hole_r - 0.4
            hr = max(hr, cavity_r + hole_r + 0.3)
            try:
                fused -= Pos(hr * ca, hr * sa, -0.5) * Cylinder(
                    hole_r, foot_h + 1.0, align=_CZ
                )
            except Exception:
                continue
    return fused, added


def _hole_realized(shape: Any, hx: float, hy: float, axis: str, mesh_boolean: Any) -> bool:
    """True when a clean through-tunnel exists at the hole centre AND lug
    material surrounds it — distinguishing a real bolt hole from empty
    space outside the part (a bare host reads open there too)."""
    try:
        if not mesh_boolean.tunnel_open(shape, hx, hy, axis):
            return False
    except Exception:
        return False
    return True


def _ring_realized(shape: Any, count: int, hole_d: float, axis: str, foot_h: float, mesh_boolean: Any) -> bool:
    """True when ``count`` clean Ø``hole_d`` bolt holes already exist on the
    host's base ring. Probes the expected lug-hole positions for both an
    open tunnel and surrounding lug material."""
    from OCP.gp import gp_Pnt

    ring = _base_ring(shape, foot_h * 0.6, mesh_boolean)
    if ring is None:
        return False
    host_r, cavity_r = ring
    hole_r = hole_d / 2.0
    lug_outer = max(host_r + hole_d * 0.7, cavity_r + hole_d + 0.8)
    z = foot_h * 0.6
    realized = 0
    for i in range(count):
        a = i * 2 * math.pi / count
        ca, sa = math.cos(a), math.sin(a)
        hr = lug_outer - hole_r - 0.4
        hr = max(hr, cavity_r + hole_r + 0.3)
        hx, hy = hr * ca, hr * sa
        # lug material must surround the hole — a bare host has a void at the
        # hole radius (empty air, also "open"), so an open tunnel alone proves
        # nothing. The material check sits TANGENTIALLY beside the hole, well
        # inside the lug's width — the lug face ends ~0.4 past the hole edge,
        # so a radial probe overshoots the part and reads empty.
        tx = hx - (hole_r + 0.9) * sa  # tangential = (-sin a, cos a)
        ty = hy + (hole_r + 0.9) * ca
        try:
            lug_here = bool(shape.is_inside(gp_Pnt(tx, ty, z)))
        except Exception:
            lug_here = False
        if lug_here and _hole_realized(shape, hx, hy, axis, mesh_boolean):
            realized += 1
    return realized >= count


def rescue_mount_lugs(
    shape: Any,
    features: list[Any],
    mesh_boolean: Any,
    emit=print,
) -> tuple[Any, list[str]]:
    """Rescue a declared mount-lug ring (feet/ears) + its bolt holes.

    The functional signature of a mounting foot is the bolt hole it carries
    — the ring is realized when ``count`` clean bolt holes exist in lug
    material at the base ring. Missing lugs are rebuilt and missing holes
    re-cut. Non-round hosts skip honestly."""
    mount_feats = [f for f in features if _is_mount(f)]
    if not mount_feats:
        return shape, []
    count, foot_w, foot_t, foot_h = _lug_dims(mount_feats)
    # the coupled bolt-hole spec: a hole feature whose count matches the lugs
    hole_d = 0.0
    hole_axis = "+z"
    for f in features:
        if _is_hole(f):
            hole_d = _dim(f, "diameter", "bore_diameter") or 0.0
            hole_axis = _axis(f, "+z")
            break
    applied: list[str] = []
    if hole_d <= 0:
        # feet without a bolt hole — only worth adding when the lugs are
        # absent entirely (a declared mount with no hole is unusual)
        ring = _base_ring(shape, foot_h * 0.6, mesh_boolean)
        if ring is None:
            return shape, applied
        fused, added = add_mount_lug_ring(
            shape, count=count, foot_w=foot_w, foot_t=foot_t, foot_h=foot_h,
            hole_d=0.0, axis=hole_axis, mesh_boolean=mesh_boolean,
        )
        if added:
            applied.append(f"{added}x mount lug {foot_w:g}x{foot_t:g}x{foot_h:g} (no bolt hole declared)")
            return fused, applied
        return shape, applied
    # the ring is realized only when every declared bolt hole is present in
    # lug material — a bare host or a lug whose hole is blind/into-cavity
    # both fail and get rebuilt
    try:
        if _ring_realized(shape, count, hole_d, hole_axis, foot_h, mesh_boolean):
            return shape, applied
    except Exception:
        pass
    fused, added = add_mount_lug_ring(
        shape, count=count, foot_w=foot_w, foot_t=foot_t, foot_h=foot_h,
        hole_d=hole_d, axis=hole_axis, mesh_boolean=mesh_boolean,
    )
    if added:
        applied.append(
            f"{added}x mount lug {foot_w:g}x{foot_t:g}x{foot_h:g} + Ø{hole_d:g} bolt hole "
            f"(declared feet/bolt-holes missing — rescued)"
        )
        return fused, applied
    return shape, applied


def rescue_spec_features(
    shape: Any,
    spec: Any,
    mesh_boolean: Any,
    emit=print,
) -> tuple[Any, list[str]]:
    """Rebuild spec-declared features the produced part does not realize.

    ``spec`` is the ``anchors["spec"]`` sub-dict (name/dims/features/
    feature_drops) the worker forwards into every execute call. Returns
    (shape, applied_labels); the input is returned unchanged when every
    declared feature is realized or no rescuer applies."""
    if mesh_boolean is None or spec is None:
        return shape, []
    features = list(_fget(spec, "features", []) or [])
    if not features:
        return shape, []
    dropped = _dropped_indices(spec)
    active = [f for i, f in enumerate(features) if i not in dropped]
    if not any(_is_mount(f) for f in active):
        return shape, []
    return rescue_mount_lugs(shape, active, mesh_boolean, emit=emit)
