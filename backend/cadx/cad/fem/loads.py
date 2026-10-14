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

import numpy as np

from cadx.cad.fem.voxelize import VoxelDomain

G_MM_S2 = 9810.0  # gravity in mm/s² for kg·mm/s² -> mN handling below


def spread_force(F: np.ndarray, node_idx: np.ndarray, force_n: np.ndarray) -> None:
    """Split total force evenly over the selected nodes (legacy behaviour)."""
    if not len(node_idx):
        return
    share = force_n.astype(float) / len(node_idx)
    for c in range(3):
        F[3 * node_idx + c] += share[c]


def spread_moment(
    domain: VoxelDomain, F: np.ndarray, node_idx: np.ndarray, moment_nm: np.ndarray
) -> float:
    """Apply a pure couple on the region — returns |M| actually applied (N·m).

    Nodal forces t_i = omega x r_i with omega = J^-1 M, J the rigidity tensor
    sum(|r_i|² I - r_i r_i^T). Then sum(r_i x t_i) = J omega = M exactly and
    sum(t_i) = omega x sum(r_i) = 0 (r measured from the node centroid).
    """
    if not len(node_idx):
        return 0.0
    M = np.asarray(moment_nm, dtype=float) * 1e3  # N·m -> N·mm
    if float(np.linalg.norm(M)) < 1e-9:
        return 0.0
    P = domain.node_coords()[node_idx]
    r = P - P.mean(axis=0)
    J = np.einsum("ij,ij->", r, r) * np.eye(3) - r.T @ r
    try:
        omega = np.linalg.solve(J + 1e-9 * np.eye(3), M)
    except np.linalg.LinAlgError:
        return 0.0
    t = np.cross(np.broadcast_to(omega, r.shape), r)
    F[3 * node_idx[:, None] + np.arange(3)[None, :]] += t
    return float(np.linalg.norm(M)) / 1e3


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
    # corner nodes on the outward faces of each mouth cell
    pos_corners = [c for c in range(8) if ((c >> ax) & 1) == (sgn > 0)]
    neg_corners = [c for c in range(8) if ((c >> ax) & 1) != (sgn > 0)]

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
