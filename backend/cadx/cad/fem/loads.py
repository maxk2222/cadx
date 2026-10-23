"""Physically-shaped load distributions on the voxel grid.

The solver only understands nodal forces; this module turns engineering load
concepts into consistent nodal force sets:

- spread_force:   total force split evenly over the region nodes (legacy)
- spread_moment:  pure couple — tangential nodal forces satisfying
                  sum(r_i x t_i) = M exactly via the rigidity tensor J
- bearing_load:   cosine-distributed radial force on a bore wall — a bolt
                  pin presses only on the half of the wall facing the load
- bolt_preload:   axial compression pair on the two bore mouths (head/nut
                  squeeze the flange between them)
- body_force:     gravity / acceleration — per-cell mass lumped to its nodes

All positions are world mm; forces in Newton, moments in N·m (converted to
N·mm internally since node coords are mm).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadx.cad.fem.voxelize import FACE_CORNERS, VoxelDomain

G_MM_S2 = 9810.0  # gravity in mm/s² for kg·mm/s² -> mN handling below


def spread_force(F: np.ndarray, node_idx: np.ndarray, force_n: np.ndarray) -> None:
    """Split total force evenly over the selected nodes (legacy behaviour)."""
    if not len(node_idx):
        return
    share = force_n.astype(float) / len(node_idx)
    for c in range(3):
        F[3 * node_idx + c] += share[c]


def _couple_solve(
    P: np.ndarray, M: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the couple field over node positions P for moment M (N·mm).

    Returns (omega, r, w_used): the angular field omega satisfying
    J_w omega = M, the node offsets r measured from the WEIGHTED centroid
    (so sum_i w_i r_i = 0 -> sum t_i = 0 for t_i = w_i (omega x r_i)), and
    the normalised weight vector actually used. J_w is the weighted
    rigidity tensor sum w_i (|r_i|² I - r_i r_i^T). `w=None`, a
    length-mismatched or all-zero weight vector falls back to uniform —
    a volumetric couple on every node instead of a skin traction.
    """
    n = len(P)
    if w is None or len(w) != n or float(np.sum(w)) <= 1e-12:
        wv = np.full(n, 1.0 / n)
    else:
        wv = np.asarray(w, dtype=float) / float(np.sum(w))
    r = P - (P * wv[:, None]).sum(axis=0)
    J = np.einsum("i,ij,ij->", wv, r, r) * np.eye(3) - (r * wv[:, None]).T @ r
    try:
        omega = np.linalg.solve(J + 1e-9 * np.eye(3), M)
    except np.linalg.LinAlgError:
        omega = np.zeros(3)
    return omega, r, wv


def spread_moment(
    domain: VoxelDomain,
    F: np.ndarray,
    node_idx: np.ndarray,
    moment_nm: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a pure couple on the region — returns the ACHIEVED moment (N·m).

    Nodal forces t_i = w_i (omega x r_i) with omega = J_w^-1 M; J_w the
    weighted rigidity tensor sum w_i(|r_i|² I - r_i r_i^T). With per-node
    `weights` (e.g. each node's share of the region's exposed surface)
    the tangential traction lands on the skin — a surface couple instead
    of a body moment smeared through the volume. Then sum(r_i x t_i) =
    J_w omega = M exactly and sum(t_i) = 0.

    The returned vector is sum(r_i x t_i)/1e3 — what actually landed in F.
    On degenerate node sets (single node, collinear nodes) the requested
    couple has components outside J_w's range; those are silently dropped
    by the regularised solve, so callers must compare achieved vs
    requested — reporting |requested| here would hide a partial load.
    """
    if not len(node_idx):
        return np.zeros(3)
    M = np.asarray(moment_nm, dtype=float) * 1e3  # N·m -> N·mm
    if float(np.linalg.norm(M)) < 1e-9:
        return np.zeros(3)
    P = domain.node_coords()[node_idx]
    omega, r, w = _couple_solve(P, M, weights)
    if not np.any(omega):
        return np.zeros(3)
    t = w[:, None] * np.cross(np.broadcast_to(omega, r.shape), r)
    F[3 * node_idx[:, None] + np.arange(3)[None, :]] += t
    return np.cross(r, t).sum(axis=0) / 1e3  # N·mm -> N·m


def moment_deficit(
    domain: VoxelDomain,
    node_idx: np.ndarray,
    moment_nm: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """(3,) N·m of `moment_nm` the node set CANNOT carry — M - achieved.

    Mirrors spread_moment's solve without touching F: the unachievable
    remainder names the axis a region fails on (e.g. the collinear axis
    of a line of nodes). ~zero for any healthy cell-derived patch — kept
    as a guard for degenerate/synthetic node sets and as the structured
    couple_deficit_nm report field.
    """
    if not len(node_idx):
        return np.asarray(moment_nm, dtype=float)
    M = np.asarray(moment_nm, dtype=float) * 1e3  # N·m -> N·mm
    if float(np.linalg.norm(M)) < 1e-9:
        return np.zeros(3)
    P = domain.node_coords()[node_idx]
    omega, r, w = _couple_solve(P, M, weights)
    t = w[:, None] * np.cross(np.broadcast_to(omega, r.shape), r)
    return (M - np.cross(r, t).sum(axis=0)) / 1e3  # N·mm -> N·m


def bearing_load(
    domain: VoxelDomain,
    F: np.ndarray,
    node_idx: np.ndarray,
    axis: np.ndarray,
    center_mm: np.ndarray,
    direction_n: np.ndarray,
    total_n: float,
) -> float:
    """Cosine bearing distribution on a bore wall — returns |F| applied (N).

    Each wall node's radial direction u_i (projected onto the plane perp to
    the bore axis) gets force w_i * u_i * s with w_i = max(0, u_i . d):
    the pin presses into the half of the wall facing the load direction.
    The scale s is solved so the resultant equals `total_n` along `d`.
    """
    if not len(node_idx) or abs(total_n) < 1e-9:
        return 0.0
    P = domain.node_coords()[node_idx]
    q = P - np.asarray(center_mm, dtype=float)
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a) + 1e-12
    rad = q - np.outer(q @ a, a)
    norm = np.linalg.norm(rad, axis=1)
    ok = norm > 1e-9
    u = np.zeros_like(rad)
    u[ok] = rad[ok] / norm[ok, None]
    d = np.asarray(direction_n, dtype=float)
    d = d - (d @ a) * a  # only the in-plane component presses the wall
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        return 0.0
    d /= dn
    w = np.clip(u @ d, 0.0, None)
    denom = float(w @ (u @ d))
    if denom < 1e-9:
        return 0.0
    s = total_n / denom
    t = w[:, None] * u * s
    F[3 * node_idx[:, None] + np.arange(3)[None, :]] += t
    return float(np.linalg.norm(t.sum(axis=0)))


def bolt_preload(
    domain: VoxelDomain,
    F: np.ndarray,
    cell_mask_flat: np.ndarray,
    axis: np.ndarray,
    preload_n: float,
) -> float:
    """Bolt pretension: compressive axial pair on the two bore mouths.

    The region's cells are split by axial coordinate; on each end, the cells
    whose outward neighbour along the axis is void get an inward-pointing
    share of `preload_n` — the head and nut squeezing the flange.
    """
    if abs(preload_n) < 1e-9 or not cell_mask_flat.any():
        return 0.0
    ax = int(np.argmax(np.abs(axis)))
    sgn = 1.0 if axis[ax] >= 0 else -1.0
    cells = cell_mask_flat.reshape(domain.dims)
    d = np.zeros(3, dtype=int)
    d[ax] = sgn
    from cadx.cad.fem.voxelize import shift_mask

    pos_face = cells & ~shift_mask(domain.solid, d)  # +axis surface of the region
    neg_face = cells & ~shift_mask(domain.solid, -d)  # -axis surface
    en = domain.element_nodes()
    # corner nodes on the outward faces of each mouth cell — FACE_CORNERS
    # maps (axis, hi/lo) onto the Hex8 corner ordering; a bit test on the
    # corner id picks the wrong (parity-diagonal) node set for the x axis
    pos_corners = list(FACE_CORNERS[ax][int(sgn > 0)])
    neg_corners = list(FACE_CORNERS[ax][int(sgn < 0)])

    def _apply(face: np.ndarray, corners: list[int], sign: float) -> int:
        sel = np.flatnonzero(face.ravel())
        if not len(sel):
            return 0
        nodes = np.unique(en[sel][:, corners].ravel())
        F[3 * nodes + ax] += sign * sgn * preload_n / len(nodes)
        return len(nodes)

    n_pos = _apply(pos_face, pos_corners, -1.0)  # head pushes inward (-axis)
    n_neg = _apply(neg_face, neg_corners, +1.0)  # nut pushes inward (+axis)
    return float(preload_n) if (n_pos and n_neg) else 0.0


def body_force(
    domain: VoxelDomain,
    F: np.ndarray,
    cell_mask_flat: np.ndarray,
    accel_g: np.ndarray,
    density_g_cm3: float,
) -> float:
    """Gravity / inertial load: per-cell mass lumped to its 8 corner nodes.

    accel_g is in units of g ([0,0,-1] = normal gravity). Density in g/cm³;
    a voxel of pitch p weighs p³·rho g = p³·rho·1e-3 kg → F = m·a in mN,
    converted to N. Returns total applied force magnitude (N).
    """
    if not cell_mask_flat.any():
        return 0.0
    a = np.asarray(accel_g, dtype=float)
    if float(np.linalg.norm(a)) < 1e-9:
        return 0.0
    cell_mass_kg = domain.pitch**3 * density_g_cm3 * 1e-6  # mm³->cm³ /1e3, g->kg /1e3
    node_force = cell_mass_kg * 9.81 * a / 8.0  # N per node-corner share
    sel = np.flatnonzero(cell_mask_flat)
    en = domain.element_nodes()[sel].ravel()
    for c in range(3):
        np.add.at(F, 3 * en + c, node_force[c])
    return float(np.linalg.norm(node_force)) * 8.0 * len(sel)


def expand_cone(force_n: np.ndarray, cone_deg: float, samples: int = 4) -> list[np.ndarray]:
    """Direction variants of `force_n` tilted `cone_deg` around the mean.

    Returns `samples` vectors on the cone rim at evenly spaced azimuths plus
    does NOT include the nominal direction (the caller keeps the base case).
    """
    f = np.asarray(force_n, dtype=float)
    fn = float(np.linalg.norm(f))
    if fn < 1e-9 or cone_deg <= 0:
        return []
    u = f / fn
    # any orthonormal basis (t1, t2) around u
    t1 = np.cross(u, [1.0, 0.0, 0.0])
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(u, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(u, t1)
    theta = np.radians(cone_deg)
    return [
        fn * (np.cos(theta) * u + np.sin(theta) * (np.cos(phi) * t1 + np.sin(phi) * t2))
        for phi in np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
    ]


def rot_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector `u` to unit vector `v` (Rodrigues).

    Used to rotate every directional field of a load (force_n, moment_nm,
    bearing_dir) by the SAME rotation a direction-cone sample applies to the
    primary direction — combined force+couple loads tilt coherently.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u /= np.linalg.norm(u) + 1e-12
    v /= np.linalg.norm(v) + 1e-12
    k = np.cross(u, v)
    s = float(np.linalg.norm(k))
    c = float(np.dot(u, v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: 180° about any axis perpendicular to u
        axis = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / s**2)


def primary_dir(load: Any) -> np.ndarray | None:
    """Reference direction for cone expansion: the first non-zero directional
    vector of the load — force_n, else moment_nm (pure torque axis), else
    bearing_dir. None when the load carries only scalar/geometry directions."""
    for attr in ("force_n", "moment_nm", "bearing_dir"):
        v = getattr(load, attr, None)
        if v and len(v) >= 3:
            a = np.asarray(v[:3], dtype=float)
            if np.linalg.norm(a) > 1e-9:
                return a
    return None
