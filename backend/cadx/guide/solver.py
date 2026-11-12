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


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal 3x3 rotation mapping unit vector a onto unit vector b."""
    a_u, b_u = _unit(_v(a)), _unit(_v(b))
    d = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    if d > 0.999999:
        return np.eye(3)
    if d < -0.999999:
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a_u, ortho))
        return axis_angle_rotation_matrix(axis, 180.0)
    axis = np.cross(a_u, b_u)
    deg = float(np.degrees(np.arccos(d)))
    return axis_angle_rotation_matrix(axis, deg)


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


def transform_port_to_target(
    moving: GuidePart,
    moving_port: PortInfo,
    fixed_transform: np.ndarray,
    fixed_port: PortInfo,
    fixed_part: GuidePart,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    """4x4 world transform placing ``moving`` so that ``moving_port`` mates
    ``fixed_port`` (KIARA _compute_transform_from_port_to_target).

    Both frames are proper rotations (det=+1); the target frame's Z is the
    FLIPPED normal (mates face each other). Symmetric roll convention:
    if either side lacks a roll reference both fall back to the world
    convention — a hint on only one side was KIARA's RC-2 bug.
    """
    t_r = fixed_transform[0:3, 0:3]
    t_t = fixed_transform[0:3, 3]
    tp_pos = t_r @ _v(fixed_port.pos) + t_t
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
) -> tuple[float, float]:
    """(joint_gap_mm, normal_dot) of a mated port pair. normal_dot = -1 is a
    perfect antiparallel mate."""
    t = np.asarray(moving_transform, dtype=float).reshape(4, 4)
    tp = np.asarray(fixed_transform, dtype=float).reshape(4, 4)
    mp_world = t[0:3, 0:3] @ _v(moving_port.pos) + t[0:3, 3]
    mn_world = t[0:3, 0:3] @ _v(moving_port.axis)
    fp_world = tp[0:3, 0:3] @ _v(fixed_port.pos) + tp[0:3, 3]
    fn_world = tp[0:3, 0:3] @ _v(fixed_port.axis)
    gap = float(np.linalg.norm(mp_world - fp_world))
    normal_dot = float(np.dot(_unit(mn_world), _unit(fn_world)))
    return gap, normal_dot


# --------------------------------------------------------------------------- #
# port matching + attachment candidates
# --------------------------------------------------------------------------- #


def _gender_ok(a: PortInfo, b: PortInfo) -> bool:
    g = {a.gender, b.gender}
    if g == {"male", "female"}:
        return True
    return "flat" in g or "unknown" in g  # uncertain pairs stay candidates


def _score_pair(a: PortInfo, b: PortInfo) -> float:
    score = 0.0
    if a.kind == "cylinder" and b.kind == "cylinder" and min(a.radius, b.radius) > 0:
        ratio = min(a.radius, b.radius) / max(a.radius, b.radius)
        if ratio < 0.5:
            return -1.0  # reject: radii can't mate
        score += ratio
    if {a.gender, b.gender} == {"male", "female"}:
        score += 0.5
    if a.kind == b.kind:
        score += 0.1
    return score


def find_matching_ports(
    moving: GuidePart,
    fixed: GuidePart,
    used_fixed_ports: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic ranked port-pair candidates (no transforms yet)."""
    out = []
    for mp in moving.ports:
        for tp in fixed.ports:
            if used_fixed_ports and tp.name in used_fixed_ports:
                continue
            if not _gender_ok(mp, tp):
                continue
            sc = _score_pair(mp, tp)
            if sc < 0:
                continue
            out.append(
                {
                    "moving_port": mp.name,
                    "fixed_port": tp.name,
                    "score": round(sc, 4),
                    "moving_gender": mp.gender,
                    "fixed_gender": tp.gender,
                    "radius_mm": [mp.radius, tp.radius],
                }
            )
    out.sort(key=lambda c: (-c["score"], c["moving_port"], c["fixed_port"]))
    return out


def _port(part: GuidePart, name: str) -> PortInfo | None:
    for p in part.ports:
        if p.name == name:
            return p
    return None


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
    pivot = tf[0:3, 0:3] @ _v(tp.pos) + tf[0:3, 3]
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


# --------------------------------------------------------------------------- #
# free / environment placement (bbox heuristics — KIARA solve_free/env)
# --------------------------------------------------------------------------- #

_HINT_DIRS = {
    "left": np.array([-1.0, 0.0, 0.0]),
    "right": np.array([1.0, 0.0, 0.0]),
    "back": np.array([0.0, 1.0, 0.0]),
    "front": np.array([0.0, -1.0, 0.0]),
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
    if not placed or h in ("", "on table", "table", "ground", "base", "first"):
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
        off = np.array([pctr[0] - center[0], pctr[1] - center[1], pmx[2] + gap_mm - mn[2]])
    t[0:3, 3] = off
    return [[float(v) for v in row] for row in t.tolist()]


def solve_env_placement(
    part: GuidePart,
    placed: list[dict[str, Any]],
    step_text: str,
) -> list[list[float]]:
    """Keyword-scan of the step text → solve_free_placement hint."""
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
        if word in t:
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
