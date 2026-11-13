"""Deterministic placement solver (port of KIARA p5_placement/solver.py).

All math in **mm**, CAD frame (Z-up, RH). Unit- and schema-adapted port —
KIARA works on dict-shaped port defs in cm/Y-up; here inputs are the
pydantic ``PortInfo``/``GuidePart`` of ``guide/state.py``. The core
algorithms are the same:

- orthonormal port frames (Z = outward normal) with a symmetric
  roll-reference convention (bbox-center projection, world fallback)
- transform = T_target_port_frame @ inv(T_source_port_frame)
- roll snaps pivoted at the target port's world position
- hard gates: joint gap < 0.05 mm, normal_dot ≤ -0.999 (antiparallel mate)
- minimal-rotation alignment for orientation hints + free placement
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadx.guide.state import GuidePart, PortInfo

GAP_GATE_MM = 0.05
NORMAL_DOT_GATE = -0.999
DEFAULT_SNAPS = (0.0, 90.0, 180.0, 270.0)

_IDENTITY = np.eye(4).tolist()


# --------------------------------------------------------------------------- #
# small linear algebra (ported from KIARA p5_placement/geometry.py)
# --------------------------------------------------------------------------- #


def _v(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(3)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def axis_angle_rotation_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    a = _v(axis)
    norm = float(np.linalg.norm(a))
    if norm < 1e-9:
        return np.eye(3)
    ux, uy, uz = (a / norm).tolist()
    th = np.deg2rad(float(degrees))
    c, s = float(np.cos(th)), float(np.sin(th))
    return np.array(
        [
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ]
    )


def delta_about_pivot(
    axis_world: np.ndarray, degrees: float, pivot_world: np.ndarray
) -> np.ndarray:
    """4x4 delta rotating about a world-space pivot point."""
    r = axis_angle_rotation_matrix(axis_world, degrees)
    pivot = _v(pivot_world)
    d = np.eye(4)
    d[0:3, 0:3] = r
    d[0:3, 3] = pivot - r @ pivot
    return d


def _frame_matrix(pos: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """4x4 orthonormal frame at ``pos`` with +Z = ``normal``.

    Hysteresis-free world fallback for X/Y (KIARA _make_frame_matrix):
    pick the world axis least parallel to the normal as the X reference.
    """
    z = _unit(_v(normal))
    refs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    ref = min(refs, key=lambda r: abs(float(np.dot(r, z))))
    x = _unit(np.cross(ref, z))
    if float(np.linalg.norm(x)) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    m = np.eye(4)
    m[0:3, 0:3] = np.column_stack([x, y, z])
    m[0:3, 3] = _v(pos)
    return m


def _frame_matrix_with_hint(
    pos: np.ndarray, normal: np.ndarray, roll_ref: np.ndarray
) -> np.ndarray:
    """Frame with +Z = normal and +X = roll_ref projected onto the tangent
    plane (KIARA _make_frame_matrix_with_hint)."""
    z = _unit(_v(normal))
    r = _v(roll_ref)
    x = r - z * float(np.dot(r, z))
    if float(np.linalg.norm(x)) < 1e-6:
        return _frame_matrix(pos, normal)
    x = _unit(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[0:3, 0:3] = np.column_stack([x, y, z])
    m[0:3, 3] = _v(pos)
    return m


def _part_center(part: GuidePart) -> np.ndarray:
    return (_v(part.bbox_min) + _v(part.bbox_max)) / 2.0


def _port_roll_reference(port: PortInfo, part: GuidePart) -> np.ndarray | None:
    """Bbox-center direction projected onto the port tangent plane — the
    deterministic roll reference (KIARA _port_roll_reference)."""
    z = _unit(_v(port.axis))
    ref = _part_center(part) - _v(port.pos)
    x = ref - z * float(np.dot(ref, z))
    return x if float(np.linalg.norm(x)) > 1e-6 else None


def _engage_mm(moving_port: PortInfo, fixed_port: PortInfo) -> float:
    """How deep the plug enters the socket along the joint axis — the
    RECEIVING port's measured seat only. Two refinements over the naive
    'deepest bore wins' model:

    - using the PLUG's own bore depth sank the moving part past the seat
      into the host body (a union-nut adapter vanished ~20mm into the
      nut cavity)
    - a LOOSE fit (collar bore ≫ plug — union nuts, flange bosses) means
      the plug rests at the mouth/seat face, not buried to full seat
      depth; only snug fits travel to the seat (spigot into muffe)

    0 for unmeasured ports keeps flush mating."""
    sock, plug = _receiver(moving_port, fixed_port)
    if sock is None:
        return 0.0
    d = _socket_depth(sock)
    if d <= 0.0:
        return 0.0
    deep = min(d * 0.85, 40.0)
    margin = _inner_r(sock) - _outer_r(plug)
    # margin ≤2mm → true socket fit (deep); ≥10mm → collar/nut mate (shallow)
    t = min(max((margin - 2.0) / 8.0, 0.0), 1.0)
    return deep * (1.0 - t) + min(deep, 6.0) * t


def transform_port_to_target(
    moving: GuidePart,
    moving_port: PortInfo,
    fixed_transform: np.ndarray,
    fixed_port: PortInfo,
    fixed_part: GuidePart,
    rotation_degrees: float = 0.0,
    engage_mm: float | None = None,
) -> np.ndarray:
    """4x4 world transform placing ``moving`` so that ``moving_port`` mates
    ``fixed_port`` (KIARA _compute_transform_from_port_to_target).

    Both frames are proper rotations (det=+1); the target frame's Z is the
    FLIPPED normal (mates face each other). Symmetric roll convention:
    if either side lacks a roll reference both fall back to the world
    convention — a hint on only one side was KIARA's RC-2 bug.

    ``engage_mm`` shifts the mate point INTO the fixed port along its
    inward axis — a spigot lands at the bottom of the socket, not
    floating at its rim.
    """
    if engage_mm is None:
        engage_mm = _engage_mm(moving_port, fixed_port)
    t_r = fixed_transform[0:3, 0:3]
    t_t = fixed_transform[0:3, 3]
    tp_pos = t_r @ (_v(fixed_port.pos) - _v(fixed_port.axis) * engage_mm) + t_t
    tp_norm = t_r @ _v(fixed_port.axis)

    new_roll = _port_roll_reference(moving_port, moving)
    tgt_roll = _port_roll_reference(fixed_port, fixed_part)
    tgt_roll_w = t_r @ tgt_roll if tgt_roll is not None else None

    tp_flipped = -_unit(tp_norm)
    if new_roll is None or tgt_roll_w is None or float(np.linalg.norm(tgt_roll_w)) < 1e-6:
        p_mat = _frame_matrix(_v(moving_port.pos), _v(moving_port.axis))
        t_mat = _frame_matrix(tp_pos, tp_flipped)
    else:
        p_mat = _frame_matrix_with_hint(_v(moving_port.pos), _v(moving_port.axis), new_roll)
        t_mat = _frame_matrix_with_hint(tp_pos, tp_flipped, tgt_roll_w)

    try:
        m = t_mat @ np.linalg.inv(p_mat)
    except np.linalg.LinAlgError:
        m = t_mat @ np.linalg.pinv(p_mat)

    if abs(rotation_degrees) > 0.01:
        m = delta_about_pivot(tp_norm, rotation_degrees, tp_pos) @ m
    return m


def evaluate_mate(
    moving_port: PortInfo,
    moving_transform: np.ndarray,
    fixed_port: PortInfo,
    fixed_transform: np.ndarray,
    *,
    engage_mm: float | None = None,
) -> tuple[float, float]:
    """(joint_gap_mm, normal_dot) of a mated port pair. normal_dot = -1 is a
    perfect antiparallel mate. Gap is measured to the ENGAGED mate point —
    a plug seated inside a socket still counts as closed."""
    if engage_mm is None:
        engage_mm = _engage_mm(moving_port, fixed_port)
    t = np.asarray(moving_transform, dtype=float).reshape(4, 4)
    tp = np.asarray(fixed_transform, dtype=float).reshape(4, 4)
    mp_world = t[0:3, 0:3] @ _v(moving_port.pos) + t[0:3, 3]
    mn_world = t[0:3, 0:3] @ _v(moving_port.axis)
    fp_world = tp[0:3, 0:3] @ (_v(fixed_port.pos) - _v(fixed_port.axis) * engage_mm) + tp[0:3, 3]
    fn_world = tp[0:3, 0:3] @ _v(fixed_port.axis)
    gap = float(np.linalg.norm(mp_world - fp_world))
    normal_dot = float(np.dot(_unit(mn_world), _unit(fn_world)))
    return gap, normal_dot


# --------------------------------------------------------------------------- #
# port matching + attachment candidates
# --------------------------------------------------------------------------- #


def _gender_ok(a: PortInfo, b: PortInfo) -> bool:
    # two bare spigots cannot mate — everything else stays a candidate:
    # rim extraction cannot tell a socket mouth from a pipe's open end
    # (both surface as "female"), so f-f pairs are allowed and
    # solve_attachment + the collision check decide whether they fit
    return {a.gender, b.gender} != {"male"}


def _socket_depth(p: PortInfo) -> float:
    """Effective grip depth that makes a port a SOCKET — either a measured
    seat (boss, nut, shoulder) or a clear through-bore (push-fit coupler:
    nominal grip ≈ 2x clear radius). A solid face (inner_r=0, seat<4mm)
    receives nothing."""
    d = p.depth
    if 4.0 <= d < 400.0:
        return d
    if d >= 400.0 and p.inner_radius > 0.0:
        return min(2.0 * p.inner_radius, 30.0)
    return 0.0


def _inner_r(p: PortInfo) -> float:
    """Clear radius the socket accepts (bore), else the rim radius."""
    return p.inner_radius or p.radius


def _outer_r(p: PortInfo) -> float:
    """Outer envelope radius the port presents to a socket (tube OD)."""
    return p.outer_radius or p.radius


def _measured(p: PortInfo) -> bool:
    """Ports carrying bore measurements get the physical socket model;
    unmeasured (BRep/executor ports, legacy states) keep the soft rules."""
    return p.depth > 0.0 or p.inner_radius > 0.0


def _receiver(a: PortInfo, b: PortInfo) -> tuple[PortInfo | None, PortInfo | None]:
    """(socket, plug) — which side physically receives the other."""
    sa, sb = _socket_depth(a) > 0.0, _socket_depth(b) > 0.0
    if not sa and not sb:
        return None, None
    if sa and sb:  # two sockets: the wider bore swallows the smaller feature
        return (a, b) if _inner_r(a) >= _inner_r(b) else (b, a)
    return (a, b) if sa else (b, a)


def _score_pair(a: PortInfo, b: PortInfo, *, include_mismatch: bool) -> float:
    # different port KINDS can never mate — a flat face can't accept a
    # pipe, a rim can't seat on a plane. Mixed pairs only ever produced
    # geometrically legal but physically meaningless frame matches
    # (adapter face_0 "mating" a casing rim)
    if a.kind != b.kind:
        return -1.0
    score = 0.1
    if a.kind == "cylinder" and b.kind == "cylinder" and min(a.radius, b.radius) > 0:
        if _measured(a) and _measured(b):
            # physical connection model: a joint needs a SOCKET receiving a
            # plug whose outer envelope fits its bore — open end on open
            # end (pipe-on-pipe), or a socket narrower than the plug, is
            # physically impossible. Threads are just small-clearance
            # versions of the same fit — no thread detection needed.
            sock, plug = _receiver(a, b)
            if sock is None or plug is None:
                return -1.0
            # clear radius is ray-measured ~10% low — accept with slack,
            # still hard-reject a plug that can't physically enter
            if _inner_r(sock) * 1.15 + 0.5 < _outer_r(plug):
                return -1.0  # socket can't swallow the plug
            # SNUGNESS: the right interface is the TIGHTEST legal fit —
            # a union nut's clear bore and a recessed shoulder are both
            # legal receivers but the fitting's actual seat has the
            # smallest margin. Reward the smallest inner-outer gap
            # (threads = near-zero margins score best), so a plug never
            # lands in a much larger opening that "also fits".
            margin = _inner_r(sock) - _outer_r(plug)
            score += max(0.0, 1.0 - max(margin, 0.0) / max(_outer_r(plug), 8.0))
        else:
            ratio = min(a.radius, b.radius) / max(a.radius, b.radius)
            if ratio < 0.5:
                if not include_mismatch:
                    return -1.0  # reject: radii can't mate
                # radius mismatch stays a LAST-RESORT candidate, not a reject:
                # rim extraction can measure a collar instead of the stub, and
                # instruction kits are representative — the manual may call a
                # joint the measured radii can't prove
                return -0.5
            score += ratio
    g = {a.gender, b.gender}
    if g == {"male", "female"}:
        score += 0.5
    elif g == {"female"}:
        score -= 0.2  # rim ambiguity: socket-over-pipe is plausible but not preferred
    return score


def _is_collar(port: PortInfo, part: GuidePart, *, moving_side: bool) -> bool:
    """A male rim coincident with a female mouth on the same part is the
    fitting's collar EDGE, not a mate target — the mouth is the real
    interface. On the moving side only a RECESS-SHADOWED male counts
    (a genuine spigot protrudes past its coincident bore lip)."""
    if port.gender != "male":
        return False
    pa = _v(port.axis)
    pp = _v(port.pos)
    for q in part.ports:
        if q is port or q.gender != "female":
            continue
        if float(np.linalg.norm(_v(q.pos) - pp)) >= 2.5:
            continue
        if abs(float(np.dot(_v(q.axis), pa))) <= 0.9:
            continue
        if not moving_side:
            return True
        if float(np.dot(_v(q.pos), pa)) > float(np.dot(pp, pa)) + 0.5:
            return True  # the female mouth reaches further — this male is recessed
    return False


def _port_exposure(part: GuidePart) -> dict[str, float]:
    """Recession depth of each port's mouth below the part's silhouette
    along its own axis. A shoulder INSIDE a through-bore (adapter rim_3)
    scores high — the real openings sit at the part's extremes. Reaching
    a recessed mouth requires tunneling through the part's other
    openings, which is physically wrong for a fresh mate."""
    mn, mx = _v(part.bbox_min), _v(part.bbox_max)
    corners = np.array(
        [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    )
    out: dict[str, float] = {}
    for p in part.ports:
        a = _unit(_v(p.axis))
        extreme = float((corners @ a).max())
        out[p.name] = max(extreme - float(np.dot(_v(p.pos), a)), 0.0)
    return out


def find_matching_ports(
    moving: GuidePart,
    fixed: GuidePart,
    used_fixed_ports: set[str] | None = None,
    include_mismatch: bool = False,
) -> list[dict[str, Any]]:
    """Deterministic ranked port-pair candidates (no transforms yet)."""
    exp_m = _port_exposure(moving)
    exp_f = _port_exposure(fixed)
    out = []
    for mp in moving.ports:
        mp_collar = _is_collar(mp, moving, moving_side=True)
        for tp in fixed.ports:
            if used_fixed_ports and tp.name in used_fixed_ports:
                continue
            if not _gender_ok(mp, tp):
                continue
            sc = _score_pair(mp, tp, include_mismatch=include_mismatch)
            if sc <= -1.0:
                continue
            if mp_collar or _is_collar(tp, fixed, moving_side=False):
                sc = -0.6  # collar edges rank below every real interface
            elif sc > 0.0:
                # recessed mouths lose to exposed ones among REAL
                # candidates — approaching an interior shoulder means
                # tunneling through the part's own openings; an exposed
                # alternative (the fitting's other end) is almost always
                # the intended interface. Last-resort/collar scores are
                # informational — don't sink them below the gate.
                sc -= min((exp_m.get(mp.name, 0.0) + exp_f.get(tp.name, 0.0)) / 25.0, 0.55)
            out.append(
                {
                    "moving_port": mp.name,
                    "fixed_port": tp.name,
                    "score": round(sc, 4),
                    "moving_gender": mp.gender,
                    "fixed_gender": tp.gender,
                    "radius_mm": [mp.radius, tp.radius],
                    "engage_mm": round(_engage_mm(mp, tp), 2),
                }
            )
    out.sort(key=lambda c: (-c["score"], c["moving_port"], c["fixed_port"]))
    return out


def _port(part: GuidePart, name: str) -> PortInfo | None:
    for p in part.ports:
        if p.name == name:
            return p
    return None


# --------------------------------------------------------------------------- #
# burial / containment audit
# --------------------------------------------------------------------------- #
#
# Mesh-boolean collision is structurally blind to parts hidden inside
# hollow cavities (a pipe slid through a union nut into the casing wall
# intersects NO solid — it lives inside the bore). The general test is
# volumetric: how much of a placed part's world AABB sits inside the
# union of every other part's AABB. A correctly mated part overlaps its
# host only by the engagement length; a swallowed part scores near 1.0.
# --------------------------------------------------------------------------- #


def world_bbox(
    bbox_min: Any, bbox_max: Any, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World-space AABB of a local bbox under a 4x4 transform."""
    mn, mx = _v(bbox_min), _v(bbox_max)
    corners = np.array(
        [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    )
    t = np.asarray(transform, dtype=float).reshape(4, 4)
    w = corners @ t[0:3, 0:3].T + t[0:3, 3]
    return w.min(axis=0), w.max(axis=0)


def union_containment(
    mn: np.ndarray,
    mx: np.ndarray,
    others: list[tuple[np.ndarray, np.ndarray]],
    *,
    grid: int = 8,
) -> float:
    """Fraction of the part's AABB volume covered by the UNION of other
    AABBs — sampled on a coarse grid (exact AABB∩AABB per cell)."""
    if not others:
        return 0.0
    span = mx - mn
    if float(np.prod(span)) <= 1e-9:
        return 0.0
    # cell centers over the moving bbox
    qs = [np.linspace(mn[i] + span[i] / (2 * grid), mx[i] - span[i] / (2 * grid), grid)
          for i in range(3)]
    inside = 0
    for x in qs[0]:
        for y in qs[1]:
            for z in qs[2]:
                p = np.array([x, y, z])
                for omn, omx in others:
                    if bool(np.all(p >= omn) and np.all(p <= omx)):
                        inside += 1
                        break
    return inside / (grid ** 3)


def solve_attachment(
    moving: GuidePart,
    moving_port_name: str,
    fixed: GuidePart,
    fixed_transform: list[list[float]],
    fixed_port_name: str,
    *,
    rotation_snaps: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Candidate transforms for a port pair across roll snaps — the LLM
    picks one via place_instance (KIARA solve_attachment shape)."""
    mp = _port(moving, moving_port_name)
    tp = _port(fixed, fixed_port_name)
    if mp is None or tp is None:
        return []
    tf = np.asarray(fixed_transform, dtype=float).reshape(4, 4)
    out = []
    for rot in rotation_snaps or DEFAULT_SNAPS:
        m = transform_port_to_target(moving, mp, tf, tp, fixed, rotation_degrees=rot)
        gap, ndot = evaluate_mate(mp, m, tp, tf)
        out.append(
            {
                "transform": [[round(float(v), 8) for v in row] for row in m.tolist()],
                "rotation_degrees": rot,
                "joint_gap_mm": round(gap, 6),
                "normal_dot": round(ndot, 6),
                "geometry_ok": bool(gap < GAP_GATE_MM and ndot <= NORMAL_DOT_GATE),
                "moving_port": mp.name,
                "fixed_port": tp.name,
            }
        )
    out.sort(key=lambda c: (not c["geometry_ok"], c["joint_gap_mm"], c["rotation_degrees"]))
    return out


def solve_multi_attachment(
    moving: GuidePart,
    fixed: GuidePart,
    fixed_transform: list[list[float]],
    port_pairs: list[tuple[str, str]],
) -> list[list[float]] | None:
    """Multi-port mate: primary pair fixes the frame; the roll angle is
    chosen to minimise residual distance of the remaining pairs."""
    if not port_pairs:
        return None
    tf = np.asarray(fixed_transform, dtype=float).reshape(4, 4)
    primary = solve_attachment(
        moving,
        port_pairs[0][0],
        fixed,
        fixed_transform,
        port_pairs[0][1],
        rotation_snaps=[0.0],
    )
    if not primary:
        return None
    base = np.asarray(primary[0]["transform"], dtype=float).reshape(4, 4)
    if len(port_pairs) == 1:
        return base.tolist()
    # pivot = primary target port world position
    tp = _port(fixed, port_pairs[0][1])
    mp = _port(moving, port_pairs[0][0])
    if tp is None or mp is None:
        return base.tolist()
    pivot = tf[0:3, 0:3] @ (_v(tp.pos) - _v(tp.axis) * _engage_mm(mp, tp)) + tf[0:3, 3]
    axis = tf[0:3, 0:3] @ _v(tp.axis)
    best, best_res = base, float("inf")
    for deg in np.linspace(0.0, 360.0, 72, endpoint=False):
        m = delta_about_pivot(axis, float(deg), pivot) @ base
        res = 0.0
        for mpn, tpn in port_pairs[1:]:
            sp = _port(moving, mpn)
            tpp = _port(fixed, tpn)
            if sp is None or tpp is None:
                continue
            res += evaluate_mate(sp, m, tpp, tf)[0] ** 2
        if res < best_res:
            best, best_res = m, res
    return [[float(v) for v in row] for row in best.tolist()]


def solve_bridge_attachment(
    moving: GuidePart,
    pairs: list[tuple[str, GuidePart, list[list[float]], str]],
) -> list[list[float]] | None:
    """Multi-mate across DIFFERENT fixed instances: each tuple is
    (moving_port, fixed_part, fixed_transform, fixed_port). The primary
    pair fixes the frame; the roll angle around the primary axis minimises
    the residual gap of every remaining pair — bridging parts (an elbow
    joining a vertical pipe to a riser) need both ends mated, not one."""
    if len(pairs) < 2:
        return None
    mp0_name, fx0, tf0_list, fp0_name = pairs[0]
    primary = solve_attachment(
        moving, mp0_name, fx0, tf0_list, fp0_name, rotation_snaps=[0.0]
    )
    if not primary:
        return None
    base = np.asarray(primary[0]["transform"], dtype=float).reshape(4, 4)
    tf0 = np.asarray(tf0_list, dtype=float).reshape(4, 4)
    tp0 = _port(fx0, fp0_name)
    mp0 = _port(moving, mp0_name)
    if tp0 is None or mp0 is None:
        return base.tolist()
    pivot = tf0[0:3, 0:3] @ (_v(tp0.pos) - _v(tp0.axis) * _engage_mm(mp0, tp0)) + tf0[0:3, 3]
    axis = tf0[0:3, 0:3] @ _v(tp0.axis)
    best, best_res = base, float("inf")
    for deg in np.linspace(0.0, 360.0, 72, endpoint=False):
        m = delta_about_pivot(axis, float(deg), pivot) @ base
        res = 0.0
        for mpn, fxn, tfn_list, fpn in pairs[1:]:
            sp = _port(moving, mpn)
            tpp = _port(fxn, fpn)
            if sp is None or tpp is None:
                continue
            res += evaluate_mate(
                sp, m, tpp, np.asarray(tfn_list, dtype=float).reshape(4, 4)
            )[0] ** 2
        if res < best_res:
            best, best_res = m, res
    return [[float(v) for v in row] for row in best.tolist()]


# --------------------------------------------------------------------------- #
# free / environment placement (bbox heuristics — KIARA solve_free/env)
# --------------------------------------------------------------------------- #

# CADX frame contract (AGENTS.md): +Y is FRONT, +X right, +Z up
_HINT_DIRS = {
    "left": np.array([-1.0, 0.0, 0.0]),
    "right": np.array([1.0, 0.0, 0.0]),
    "back": np.array([0.0, -1.0, 0.0]),
    "front": np.array([0.0, 1.0, 0.0]),
    "top": np.array([0.0, 0.0, 1.0]),
    "up": np.array([0.0, 0.0, 1.0]),
    "bottom": np.array([0.0, 0.0, -1.0]),
    "down": np.array([0.0, 0.0, -1.0]),
}


def solve_free_placement(
    part: GuidePart,
    placed: list[dict[str, Any]],
    hint: str = "",
    *,
    table_z: float = 0.0,
    gap_mm: float = 2.0,
) -> list[list[float]]:
    """Identity-orientation placement honouring a spatial hint.

    ``placed`` entries: {"bbox_min_world", "bbox_max_world"} of existing
    parts (post-transform). Hints: "on table"/"ground", "on top of X" is
    resolved by the caller into hint="top" + placed list.
    """
    mn, mx = _v(part.bbox_min), _v(part.bbox_max)
    size = mx - mn
    center = (mn + mx) / 2.0
    t = np.eye(4)
    h = (hint or "").strip().lower()
    if not placed or h in ("on table", "table", "ground", "base", "first"):
        t[0:3, 3] = [-center[0], -center[1], -mn[2] + table_z]
        return [[float(v) for v in row] for row in t.tolist()]

    # stack on top of the union bbox of placed parts
    pmn = np.min([_v(p["bbox_min_world"]) for p in placed], axis=0)
    pmx = np.max([_v(p["bbox_max_world"]) for p in placed], axis=0)
    pctr = (pmn + pmx) / 2.0
    d = _HINT_DIRS.get(h)
    if h in ("top", "up", "on top", "above"):
        off = np.array([pctr[0] - center[0], pctr[1] - center[1], pmx[2] + gap_mm - mn[2]])
    elif d is not None:
        axis_i = int(np.argmax(np.abs(d)))
        sign = float(np.sign(d[axis_i]))
        extent = (pmx[axis_i] - pmn[axis_i]) / 2 + size[axis_i] / 2 + gap_mm
        off = np.array([pctr[0] - center[0], pctr[1] - center[1], pctr[2] - center[2]])
        off[axis_i] = pctr[axis_i] + sign * extent - center[axis_i]
        off[2] = -mn[2] + max(table_z, pmn[2])  # rest on the same level
    else:
        # no hint: spread out BESIDE the placed parts (+X row, same base
        # level) — never stack vertically. A tower of coincident parts
        # reads as a broken pile in renders and defeats collision/gap
        # checks; a row is a sane bill-of-materials-style starting layout
        # the refine pass can rearrange.
        off = np.array([pmx[0] + size[0] / 2 + gap_mm - center[0],
                        pctr[1] - center[1],
                        -mn[2] + max(table_z, pmn[2])])
    t[0:3, 3] = off
    return [[float(v) for v in row] for row in t.tolist()]


def solve_env_placement(
    part: GuidePart,
    placed: list[dict[str, Any]],
    step_text: str,
) -> list[list[float]]:
    """Keyword-scan of the step text → solve_free_placement hint."""
    import re

    t = (step_text or "").lower()
    for word, hint in (
        ("on top", "top"),
        ("above", "top"),
        ("oben", "top"),
        ("auf", "top"),
        ("left", "left"),
        ("links", "left"),
        ("right", "right"),
        ("rechts", "right"),
        ("front", "front"),
        ("vorne", "front"),
        ("back", "back"),
        ("hinten", "back"),
        ("table", "table"),
        ("ground", "table"),
        ("boden", "table"),
    ):
        # word boundaries — bare substring matches hit "auf" inside
        # "laufen"/"verkauf" and misplace parts
        if re.search(rf"\b{re.escape(word)}\b", t):
            return solve_free_placement(part, placed, hint)
    return solve_free_placement(part, placed, "")


def identity_transform() -> list[list[float]]:
    return [list(r) for r in _IDENTITY]


def transform_bbox(
    part: GuidePart, transform: list[list[float]]
) -> tuple[list[float], list[float]]:
    """World-space AABB of a part under a transform (8-corner hull)."""
    m = np.asarray(transform, dtype=float).reshape(4, 4)
    mn, mx = _v(part.bbox_min), _v(part.bbox_max)
    corners = np.array(
        [[x, y, z, 1.0] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    )
    w = (m @ corners.T).T[:, :3]
    return w.min(axis=0).tolist(), w.max(axis=0).tolist()
