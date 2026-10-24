"""Method of Manufactured Solutions — code verification via convergence order.

A sinusoidal displacement field

    ux = U sin(a x) sin(a y) sin(a z)
    uy = U sin(2a x) sin(a y) sin(a z)
    uz = U sin(a x) sin(2a y) sin(a z)        a = pi/L

vanishes on every face of the cube [0, L]³, so pinning all boundary nodes to
zero IS the exact Dirichlet condition — no approximation at the boundary.
The Navier equation b = -mu lap(u) - (lam+mu) grad(div u) then gives an
analytic body force, applied via apply_body_field (same lumping as
body_force). The L2 error of the nodal displacement must shrink ~O(h^2)
for trilinear hexes; we assert rate >= 1.5 and a bounded finest error.
"""

from __future__ import annotations

import numpy as np
import trimesh

from cadx.cad.fem.solver import assemble, reduce_case, solve_reduced
from cadx.cad.fem.voxelize import voxelize

from . import harness

L = 20.0  # mm cube edge
U = 1e-3  # mm displacement amplitude
E_MPA = 3500.0
NU = 0.36
A = np.pi / L

MU = E_MPA / (2.0 * (1.0 + NU))
LAM = E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

PITCHES = (4.0, 2.5, 2.0)  # h = 5, 8, 10 elements per edge


def _u_exact(pts: np.ndarray) -> np.ndarray:
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return U * np.stack(
        [
            np.sin(A * x) * np.sin(A * y) * np.sin(A * z),
            np.sin(2 * A * x) * np.sin(A * y) * np.sin(A * z),
            np.sin(A * x) * np.sin(2 * A * y) * np.sin(A * z),
        ],
        axis=1,
    )


def _b_exact(x: float, y: float, z: float) -> tuple[float, float, float]:
    sx, sy, sz = np.sin(A * x), np.sin(A * y), np.sin(A * z)
    cx, cy, cz = np.cos(A * x), np.cos(A * y), np.cos(A * z)
    s2x, s2y = np.sin(2 * A * x), np.sin(2 * A * y)
    c2x, c2y = np.cos(2 * A * x), np.cos(2 * A * y)
    ux, uy, uz = U * sx * sy * sz, U * s2x * sy * sz, U * sx * s2y * sz
    a2 = A * A
    # div u = Ua[cx sy sz + s2x cy sz + sx s2y cz]; grad applied componentwise
    ddiv_dx = U * a2 * (-sx * sy * sz + 2 * c2x * cy * sz + cx * s2y * cz)
    ddiv_dy = U * a2 * (cx * cy * sz - s2x * sy * sz + 2 * sx * c2y * cz)
    ddiv_dz = U * a2 * (cx * sy * cz + s2x * cy * cz - sx * s2y * sz)
    bx = 3 * MU * a2 * ux - (LAM + MU) * ddiv_dx
    by = 6 * MU * a2 * uy - (LAM + MU) * ddiv_dy
    bz = 6 * MU * a2 * uz - (LAM + MU) * ddiv_dz
    return float(bx), float(by), float(bz)


def _l2_error(pitch: float) -> float:
    mesh = trimesh.creation.box(extents=[L, L, L])
    mesh.apply_translation([L / 2, L / 2, L / 2])
    dom = voxelize(mesh, pitch, 400_000)
    solid = dom.solid.ravel()
    elem_E = np.full(dom.cell_count, E_MPA)
    K, edof, active = assemble(dom, elem_E, NU, solid)

    coords = dom.node_coords()
    en = dom.element_nodes()
    used = np.unique(en[solid])
    u = coords[used]
    on_bnd = (
        (u[:, 0] < 1e-9)
        | (u[:, 0] > L - 1e-9)
        | (u[:, 1] < 1e-9)
        | (u[:, 1] > L - 1e-9)
        | (u[:, 2] < 1e-9)
        | (u[:, 2] > L - 1e-9)
    )
    fixed3 = np.zeros((len(coords), 3), dtype=bool)
    fixed3[used[on_bnd]] = True

    F = np.zeros(len(coords) * 3)
    harness.apply_body_field(dom, F, _b_exact)

    ctx = reduce_case(dom, K, edof, fixed3)
    sol = solve_reduced(dom, ctx, edof, active, F, elem_E, NU)
    got = sol.node_disp[used]
    ref = _u_exact(u)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def test_mms_convergence_order():
    """Manufactured-solution L2 error must converge ~O(h^2)."""
    errs = [_l2_error(p) for p in PITCHES]
    rates = [
        np.log(errs[i] / errs[i + 1]) / np.log(PITCHES[i] / PITCHES[i + 1])
        for i in range(len(PITCHES) - 1)
    ]
    assert errs[-1] < 0.05, f"MMS error too large at finest pitch: {errs}"
    assert min(rates) >= 1.5, f"convergence below O(h^1.5): errs={errs} rates={rates}"
