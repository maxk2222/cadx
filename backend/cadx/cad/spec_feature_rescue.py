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
import re
from typing import Any


def _fget(f: Any, key: str, default: Any = None) -> Any:
    return f.get(key, default) if isinstance(f, dict) else getattr(f, key, default)


def _fdims(f: Any) -> dict[str, Any]:
    d = _fget(f, "dims", {}) or {}
    return d if isinstance(d, dict) else {}


def _fname_type(f: Any) -> str:
    return f"{_fget(f, 'name', '')} {_fget(f, 'type', '')}".lower()


_MOUNT_WORDS = ("foot", "feet", "lug", "ear", "tab", "stand", "leg", "mount", "flange")
_MOUNT_RE = re.compile(r"\b(?:" + "|".join(_MOUNT_WORDS) + r")\b")
_HOLE_RE = re.compile(r"\b(?:hole|bore|bohrung|clearance)\b")
_HOLE_TYPES = ("hole", "bore", "clearance", "through", "bolt", "anchor")
_EMBED_MM = 2.0  # lug root reach into the host wall — a tangent-only fuse stays a disjoint solid


def _is_mount(f: Any) -> bool:
    ftype = str(_fget(f, "type", "") or "").lower()
    if ftype in _HOLE_TYPES:
        return False
    # mount words are matched on name+type only — the description/position
    # text of the host itself routinely mentions the mounting plane
    # ("from z=0 (foot base) to z=120"), and counting the shell as a lug
    # inflates the ring and poisons lug dims with the part envelope
    return bool(_MOUNT_RE.search(_fname_type(f)))


def _is_hole(f: Any) -> bool:
    ftype = str(_fget(f, "type", "") or "").lower()
    if ftype in _HOLE_TYPES:
        return True
    return bool(_HOLE_RE.search(_fname_type(f)))


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


def _shell_seam_z(shape: Any, shell_r: float, mesh_boolean: Any) -> float:
    """Lowest z where the host wall reaches its full straight-shell radius —
    the bottom-dome -> shell equator. A mounting lug whose wall-embedding
    face straddles that seam fuses onto a surface that STEP export serialises
    as unsewn shells, so lugs must sit above it. Returns the seam height
    (base z when the host is a straight shell all the way down)."""
    bb = shape.bounding_box()
    z = bb.min.Z + 0.25
    top = bb.min.Z + 20.0
    while z < top:
        ring = _base_ring(shape, z, mesh_boolean)
        if ring is not None and ring[0] >= shell_r - 0.15:
            return z
        z += 0.25
    return bb.min.Z


def _lug_dims(mount_feats: list[Any]) -> tuple[int, float, float, float]:
    """Aggregate a mount-lug spec into (count, w, t, h) — feet arrive either
    as one feature count=N or as per-lug features each count=1 (sum them).
    Dims are the per-key lower-median across the mount features so a stray
    host-level feature that slipped the name filter cannot poison the lug
    height with the part's envelope."""
    count = sum(_count(f) for f in mount_feats)

    def pick(*keys: str) -> float | None:
        vals = sorted(v for f in mount_feats if (v := _dim(f, *keys)) is not None)
        return vals[(len(vals) - 1) // 2] if vals else None

    w = pick("width", "length", "size") or 18.0
    t = pick("thickness", "depth") or 6.0
    h = pick("depth_total", "height", "head_diameter", "length") or 20.0
    return count, float(w), float(t), float(h)


def _skirted_base(shape: Any, host_r: float, cavity_r: float, mesh_boolean: Any) -> Any:
    """Rebuild a convex-domed bottom as a straight skirt + recessed head so
    grounded feet can attach to clean cylinder wall — a lug crossing a convex
    dome's equator fuse-seam serialises as unsewn shells on STEP export, so
    the dome is recessed inside the skirt instead (same construction
    sealed_tank uses when it is given feet). Returns the host unchanged when
    its wall is already straight down to the base plane."""
    from build123d import Box, Cylinder, Pos, Sphere

    from cadx.cad.cadlib.shapes import _CZ

    bb = shape.bounding_box()
    base_z = bb.min.Z
    seam = _shell_seam_z(shape, host_r, mesh_boolean)
    if seam <= base_z + 1.0:
        return shape  # straight/skirt wall already — feet can sit at the base
    wall = max(0.5, host_r - cavity_r)
    dome_h = seam - base_z  # convex dome depth ≈ equator height off the base
    rim_z = base_z + dome_h + 1.0  # recessed head's rim this far up the skirt
    floor_z = rim_z + wall  # new interior floor — solid base cap re-seals here
    # keep the host above the new interior floor, then cap it with a solid
    # skirted base whose top face re-seals the opened cavity
    cutter = Pos(0, 0, floor_z) * Box(
        host_r * 4.0, host_r * 4.0, (bb.max.Z - floor_z) + 10.0, align=_CZ
    )
    host_above = shape & cutter
    rs = (cavity_r * cavity_r + dome_h * dome_h) / (2.0 * dome_h)
    head = Pos(0, 0, rim_z + rs - dome_h) * Sphere(rs)
    punt = Pos(0, 0, base_z) * Cylinder(cavity_r, rim_z - base_z, align=_CZ) - head
    base = Pos(0, 0, base_z) * Cylinder(host_r, floor_z - base_z, align=_CZ) - punt
    return host_above + base


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
    from build123d import Cylinder, Pos, Rot

    from cadx.cad.cadlib.shapes import mount_lug

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
    tip_w = foot_w * 0.75
    from build123d import Align

    _CZ = (Align.CENTER, Align.CENTER, Align.MIN)
    # feet must reach the mounting plane on clean cylinder wall — a convex
    # bottom dome can't take grounded feet (its equator fuse-seam breaks STEP
    # export) and elevated lugs miss the FEM -z fixed-support band, so the
    # bottom is rebuilt as a skirt + recessed head first (no-op for an
    # already-straight wall).
    shape = _skirted_base(shape, host_r, cavity_r, mesh_boolean)
    foot_z = 0.0
    fused = shape
    added = 0
    for i in range(count):
        a = azimuth0 + i * 2 * math.pi / count
        ca, sa = math.cos(a), math.sin(a)
        deg = math.degrees(a)
        try:
            fused += Pos(0, 0, foot_z + foot_h / 2.0) * (
                Rot(0, 0, deg) * mount_lug(lug_inner, lug_outer, foot_w, tip_w, foot_h)
            )
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
                fused -= Pos(hr * ca, hr * sa, foot_z - 0.5) * Cylinder(
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


def _ring_realized(
    shape: Any, count: int, hole_d: float, axis: str, foot_h: float, mesh_boolean: Any
) -> bool:
    """True when ``count`` clean Ø``hole_d`` bolt holes already exist on the
    host's base ring — at ANY azimuth. The generated code may lay out the
    ring on a rotated offset; probing only canonical positions would fail a
    realized-but-rotated ring and fuse a duplicate rescue ring on top."""
    from OCP.gp import gp_Pnt

    ring = _base_ring(shape, foot_h * 0.6, mesh_boolean)
    if ring is None:
        return False
    host_r, cavity_r = ring
    hole_r = hole_d / 2.0
    lug_outer = max(host_r + hole_d * 0.7, cavity_r + hole_d + 0.8)
    z = foot_h * 0.6
    hr = max(lug_outer - hole_r - 0.4, cavity_r + hole_r + 0.3)
    # scan the ring for lug material beside each candidate hole centre —
    # the material check sits TANGENTIALLY beside the hole, well inside the
    # lug's width (the lug face ends ~0.4 past the hole edge, so a radial
    # probe overshoots the part and reads empty)
    n_scan = 72
    hits: list[float] = []
    for i in range(n_scan):
        a = i * 2 * math.pi / n_scan
        ca, sa = math.cos(a), math.sin(a)
        hx, hy = hr * ca, hr * sa
        tx = hx - (hole_r + 0.9) * sa  # tangential = (-sin a, cos a)
        ty = hy + (hole_r + 0.9) * ca
        try:
            if shape.is_inside(gp_Pnt(tx, ty, z)):
                hits.append(a)
        except Exception:
            continue
    # group adjacent azimuth hits into lug clusters (a lug spans several
    # scan steps; gaps split neighbouring lugs), then verify a clean
    # through-tunnel at each cluster centre
    clusters: list[list[float]] = []
    for a in hits:
        if clusters and a - clusters[-1][-1] <= 1.5 * (2 * math.pi / n_scan):
            clusters[-1].append(a)
        else:
            clusters.append([a])
    # wrap-around: first and last clusters may be the same lug across 0
    if len(clusters) > 1 and (clusters[0][0] + 2 * math.pi) - clusters[-1][-1] <= 1.5 * (
        2 * math.pi / n_scan
    ):
        clusters[0] = clusters.pop() + clusters[0]
    realized = 0
    for cl in clusters:
        # circular mean — angles straddling the 2π seam must not average to ~π
        a = math.atan2(
            sum(math.sin(x) for x in cl) / len(cl), sum(math.cos(x) for x in cl) / len(cl)
        )
        ca, sa = math.cos(a), math.sin(a)
        # the generated hole need not sit at the rescue's canonical radius —
        # scan inward across the lug span: a candidate point counts when lug
        # material flanks it AND a clean through-tunnel exists there
        r = hr
        while r >= cavity_r + hole_r + 0.3:
            px, py = r * ca, r * sa
            try:
                solid = bool(
                    shape.is_inside(gp_Pnt(px - (hole_r + 0.9) * sa, py + (hole_r + 0.9) * ca, z))
                    or shape.is_inside(
                        gp_Pnt(px + (hole_r + 0.9) * sa, py - (hole_r + 0.9) * ca, z)
                    )
                )
            except Exception:
                solid = False
            if solid and _hole_realized(shape, px, py, axis, mesh_boolean):
                realized += 1
                break
            r -= max(1.0, hole_r)
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
    # the coupled bolt-hole spec: prefer the hole feature tied to the mount
    # (a mount word or "bolt" in its name, or a count matching the lugs) —
    # a spec can carry several unrelated holes and the first one is not
    # necessarily the lug bolt hole
    hole_feats = [f for f in features if _is_hole(f)]

    def _hole_rank(f: Any) -> tuple[bool, bool]:
        text = _fname_type(f)
        return (bool(_MOUNT_RE.search(text)) or "bolt" in text, _count(f) == count)

    hole_feat = max(hole_feats, key=_hole_rank, default=None)
    hole_d = (
        _dim(hole_feat, "diameter", "bore_diameter") if hole_feat is not None else None
    ) or 0.0
    hole_axis = _axis(hole_feat, "+z") if hole_feat is not None else "+z"
    applied: list[str] = []
    if hole_d <= 0:
        # feet without a bolt hole — only worth adding when the lugs are
        # absent entirely (a declared mount with no hole is unusual)
        ring = _base_ring(shape, foot_h * 0.6, mesh_boolean)
        if ring is None:
            return shape, applied
        fused, added = add_mount_lug_ring(
            shape,
            count=count,
            foot_w=foot_w,
            foot_t=foot_t,
            foot_h=foot_h,
            hole_d=0.0,
            axis=hole_axis,
            mesh_boolean=mesh_boolean,
        )
        if added:
            applied.append(
                f"{added}x mount lug {foot_w:g}x{foot_t:g}x{foot_h:g} (no bolt hole declared)"
            )
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
        shape,
        count=count,
        foot_w=foot_w,
        foot_t=foot_t,
        foot_h=foot_h,
        hole_d=hole_d,
        axis=hole_axis,
        mesh_boolean=mesh_boolean,
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
