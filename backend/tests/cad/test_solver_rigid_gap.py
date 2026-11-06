"""Rigid-mode gap detection: under-constrained cases must skip the solve.

A fixed region that matches nodes but leaves rigid-body modes free (roller
support, disconnected loaded island) makes K_ff singular — CG previously
burned its full maxiter on it before reporting converged=False. The
`_rigid_gaps` rank check in reduce_case flags the deficient DOFs and
solve_reduced returns the unsolvable verdict before touching a solver.
"""

from __future__ import annotations

import numpy as np
import trimesh

from cadx.cad.fem.solver import assemble, reduce_case, solve_reduced
from cadx.cad.fem.voxelize import VoxelDomain, voxelize

E_PLA = 3500.0
NU = 0.36


def _box_mesh(x: float = 60.0, y: float = 24.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def test_roller_support_skips_solve():
    """Beam clamped on one face in y only — tx/tz/rotation-y stay free, so
    the reduced system is singular and the solve must be skipped, not
    attempted for thousands of CG iterations."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    coords = dom.node_coords()
    lo_x = coords[:, 0].min()
    on_face = np.isclose(coords[:, 0], lo_x)
    fixed3 = np.zeros((dom.node_count, 3), dtype=bool)
    fixed3[on_face, 1] = True  # y only — a roller, not a clamp
    F = np.zeros(3 * dom.node_count)
    hi_x = np.isclose(coords[:, 0], coords[:, 0].max())
    F.reshape(-1, 3)[hi_x, 2] = -100.0 / float(hi_x.sum())
    elem_E = np.where(dom.solid.ravel(), E_PLA, 0.0)
    K, edof, active = assemble(dom, elem_E, NU)
    ctx = reduce_case(dom, K, edof, fixed3, spsolve_dofs=0, active=active)
    assert ctx.deficient_dofs.size > 0
    sol = solve_reduced(dom, ctx, edof, active, F, elem_E, NU)
    assert not sol.converged and sol.solver == "skipped"
    assert any("rigid" in w for w in sol.warnings)


def test_fully_clamped_face_not_flagged():
    """All-3-DOF clamp on the same face pins every rigid mode — no gap."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    coords = dom.node_coords()
    on_face = np.isclose(coords[:, 0], coords[:, 0].min())
    fixed3 = np.repeat(on_face[:, None], 3, axis=1)
    elem_E = np.where(dom.solid.ravel(), E_PLA, 0.0)
    K, edof, active = assemble(dom, elem_E, NU)
    ctx = reduce_case(dom, K, edof, fixed3, spsolve_dofs=0, active=active)
    assert ctx.deficient_dofs.size == 0


def _two_block_domain() -> VoxelDomain:
    """Two 3x4x4 solid blocks separated by a 2-cell void gap in x."""
    dims = (8, 4, 4)
    solid = np.zeros(dims, dtype=bool)
    solid[:3] = True
    solid[5:] = True
    return VoxelDomain(origin=np.zeros(3), pitch=1.0, dims=dims, solid=solid, pitch_requested=1.0)


def test_loaded_island_skips_unloaded_island_solves():
    """Component-wise check: a load on the floating block is unsolvable;
    the same fixture with the load on the clamped block still runs."""
    dom = _two_block_domain()
    en = dom.element_nodes()
    a_cells = np.flatnonzero(dom.solid.ravel()[: 3 * 4 * 4] != 0)
    a_nodes = np.unique(en[a_cells])
    fixed3 = np.repeat(np.isin(np.arange(dom.node_count), a_nodes)[:, None], 3, axis=1)
    solid = dom.solid.ravel()
    elem_E = np.where(solid, E_PLA, 0.0)
    K, edof, active = assemble(dom, elem_E, NU, solid)
    ctx = reduce_case(dom, K, edof, fixed3, spsolve_dofs=0, active=active)
    # the floating block's dofs are all deficient
    assert ctx.deficient_dofs.size > 0

    b_nodes = np.setdiff1d(np.unique(en[active]), a_nodes)
    F = np.zeros(3 * dom.node_count)
    F.reshape(-1, 3)[b_nodes, 0] = -10.0
    sol = solve_reduced(dom, ctx, edof, active, F, elem_E, NU)
    assert sol.solver == "skipped" and not sol.converged

    F2 = np.zeros(3 * dom.node_count)
    F2.reshape(-1, 3)[a_nodes, 0] = -10.0
    sol2 = solve_reduced(dom, ctx, edof, active, F2, elem_E, NU)
    assert sol2.solver != "skipped"
