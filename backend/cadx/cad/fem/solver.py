"""Hex8 linear-elasticity solver on the uniform VoxelDomain grid.

8-node brick elements, 2x2x2 Gauss integration, uniform element edge `pitch`
so a single element stiffness matrix is reused for the whole grid. Assembly
is the classic top3d vectorised edof/kron trick (Liu & Tovar 2014).

Units: mm, N, MPa (E). Displacements in mm.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from cadx.cad.fem.voxelize import VoxelDomain

# natural coords of the 8 hex nodes, matching voxelize.element_nodes order
_NODES_NAT = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)

_G = 1.0 / math.sqrt(3.0)
_GAUSS = np.array([[a, b, c] for a in (-_G, _G) for b in (-_G, _G) for c in (-_G, _G)], dtype=float)


def _dshape(xi: float, eta: float, zeta: float) -> np.ndarray:
    """dN/d(xi,eta,zeta) at a natural point -> (8,3)."""
    x, y, z = _NODES_NAT[:, 0], _NODES_NAT[:, 1], _NODES_NAT[:, 2]
    dxi = 0.125 * x * (1 + y * eta) * (1 + z * zeta)
    deta = 0.125 * y * (1 + x * xi) * (1 + z * zeta)
    dzeta = 0.125 * z * (1 + x * xi) * (1 + y * eta)
    return np.stack([dxi, deta, dzeta], axis=1)


def _B_matrix(dn_dx: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix (6, 24) from physical shape derivs (8,3)."""
    B = np.zeros((6, 24))
    for a in range(8):
        dx, dy, dz = dn_dx[a]
        c = 3 * a
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c] = dy
        B[3, c + 1] = dx
        B[4, c + 1] = dz
        B[4, c + 2] = dy
        B[5, c] = dz
        B[5, c + 2] = dx
    return B


def _D(nu: float) -> np.ndarray:
    """Isotropic elasticity matrix for E = 1 (scale later)."""
    c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    s = (1.0 - 2.0 * nu) / 2.0
    return c * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, s, 0, 0],
            [0, 0, 0, 0, s, 0],
            [0, 0, 0, 0, 0, s],
        ]
    )


@lru_cache(maxsize=32)
def _ke_unit(nu: float, pitch: float) -> np.ndarray:
    """Element stiffness for E=1, uniform cube of edge `pitch`.

    Memoized — pure function of (nu, pitch); the returned matrix is shared,
    callers must treat it as read-only."""
    D = _D(nu)
    detJ = (pitch / 2.0) ** 3
    ke = np.zeros((24, 24))
    for xi, eta, zeta in _GAUSS:
        B = _B_matrix(_dshape(xi, eta, zeta) * (2.0 / pitch))
        ke += B.T @ D @ B * detJ
    return ke


def _B0(pitch: float) -> np.ndarray:
    """B at the element centre (xi=eta=zeta=0)."""
    return _B_matrix(_dshape(0.0, 0.0, 0.0) * (2.0 / pitch))


@dataclass
class FemSolution:
    u: np.ndarray  # (3*nnodes,) global displacement vector (mm)
    elem_vm: np.ndarray  # (ncells,) von Mises stress per cell (MPa), 0 outside active
    node_disp: np.ndarray  # (nnodes, 3) displacement per node
    max_vm_mpa: float
    max_disp_mm: float
    compliance_j: float  # u^T F in N*mm == mJ
    dofs: int
    converged: bool
    solver: str = "spsolve"
    warnings: list[str] = field(default_factory=list)


def element_dofs(elem_nodes: np.ndarray) -> np.ndarray:
    """(nelem, 24) dof indices: [3n0,3n0+1,3n0+2, 3n1, ...]."""
    d = np.empty((elem_nodes.shape[0], 24), dtype=np.int64)
    for a in range(8):
        d[:, 3 * a : 3 * a + 3] = 3 * elem_nodes[:, a : a + 1] + np.arange(3)
    return d


def domain_edof(domain: VoxelDomain) -> np.ndarray:
    """(ncells, 24) dof table of the full grid — memoized on the domain
    (pure function of the dims; the shared array is read-only)."""
    return domain.cached("element_dofs", lambda: element_dofs(domain.element_nodes()))


def assemble(
    domain: VoxelDomain,
    elem_E: np.ndarray,
    nu: float,
    cell_mask: np.ndarray | None = None,
    ij: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble K over selected cells.

    elem_E: (ncells,) per-cell Young's modulus (only cells in cell_mask are
    assembled). Returns (K, edof, active_elem_idx): K is (3*nnodes, 3*nnodes)
    csr, edof is (nelem,24), active_elem_idx maps rows to flat cell ids.

    `ij` may carry a precomputed (iK, jK) sparsity pattern — valid only when
    `cell_mask` selects the same cells it was built for (SIMP assembles ALL
    cells every iteration, so the pattern is constant and caching it skips an
    O(nelem·576) index build per solve).
    """
    mask = domain.solid.ravel() if cell_mask is None else np.asarray(cell_mask).ravel()
    active = np.flatnonzero(mask)
    # element_nodes/element_dofs/_ke_unit are pure functions of the domain —
    # memoized, so repeated assembles (one per SIMP iteration, one per FEM
    # load case) only pay the COO value fill
    edof = domain_edof(domain)[active]
    ke = _ke_unit(nu, domain.pitch)

    if ij is None:
        iK = np.repeat(edof, 24, axis=1).ravel()
        jK = np.tile(edof, (1, 24)).ravel()
    else:
        iK, jK = ij
    sK = (elem_E[active][:, None, None] * ke[None]).reshape(-1)
    K = sparse.coo_matrix(
        (sK, (iK, jK)), shape=(3 * domain.node_count, 3 * domain.node_count)
    ).tocsr()
    K.sum_duplicates()
    return K, edof, active


def _fixed_dofs(fixed: np.ndarray, nnodes: int) -> np.ndarray:
    """Flat dof indices of the constraint mask — accepts (nnodes,) node masks
    (legacy: all 3 DOFs pinned) or (nnodes, 3) per-DOF masks (fix_mode)."""
    fixed = np.asarray(fixed, dtype=bool)
    if fixed.ndim == 1:
        return np.unique(np.concatenate([3 * np.flatnonzero(fixed) + c for c in range(3)]))
    # (nnodes, 3) C-order ravel -> flat index IS the dof index (3*node + comp)
    _ = nnodes
    return np.flatnonzero(fixed.ravel())


def _ssor_preconditioner(Kf: sparse.csr_matrix) -> spla.LinearOperator | None:
    """SSOR preconditioner M ~ (D+L) D^-1 (D+L)^T via two triangular solves —
    much better conditioned than Jacobi for voxel elasticity, still O(nnz).
    Returns None on any problem (caller falls back to diagonal scaling)."""
    try:
        L = sparse.tril(Kf, format="csr")  # includes the diagonal
        diag = L.diagonal()
        if np.any(diag <= 0):
            return None
        D = sparse.diags(diag)

        def _mv(r: np.ndarray) -> np.ndarray:
            y = spla.spsolve_triangular(L, r, lower=True)
            return np.asarray(spla.spsolve_triangular(L.T.tocsr(), D @ y, lower=False))

        return spla.LinearOperator(Kf.shape, matvec=_mv, dtype=np.float64)
    except Exception:
        return None


def factor_symmetric(Kf: sparse.csr_matrix) -> Callable[[np.ndarray], np.ndarray] | None:
    """Direct-solve callable ``b -> x`` for the reduced stiffness matrix.

    Kf is symmetric (structural elasticity), so SuperLU's SymmetricMode gets
    a symmetric permutation + half the numeric work of the generic path —
    measurably cheaper than ``factorized``/``spsolve``'s default unsymmetric
    symbolic+numeric factorization on the same constant sparsity pattern.
    Falls back to ``factorized``, then None (caller degrades to CG)."""
    try:
        return spla.splu(Kf.tocsc(), options={"SymmetricMode": True}).solve
    except Exception:
        pass
    try:
        return spla.factorized(Kf.tocsc())
    except Exception:
        return None


@dataclass
class CaseSolve:
    """Reduced system for one fixed-DOF signature on an assembled K.

    `free` are the unconstrained DOFs (sorted), `Kf = K[free][:, free]`,
    `factor` the direct-solve callable (None above the spsolve_dofs cutoff),
    `M` the CG preconditioner. Multi-case verifies with per-case fixtures
    (or SIMP cases sharing one fixture) reuse one CaseSolve per distinct
    `free` set instead of re-factorizing per case.
    """

    free: np.ndarray
    Kf: sparse.csr_matrix
    factor: Callable[[np.ndarray], np.ndarray] | None
    M: Any
    warnings: list[str] = field(default_factory=list)


def reduce_case(
    domain: VoxelDomain,
    K: sparse.csr_matrix,
    edof: np.ndarray,
    fixed_nodes: np.ndarray,
    *,
    spsolve_dofs: int = 120_000,
) -> CaseSolve:
    """Extract the reduced system Kf = K[free][:, free] for one constraint
    mask and prepare its solver (symmetric SuperLU under the cutoff, SSOR-CG
    above). `fixed_nodes` accepts a (nnodes,) node mask or a (nnodes,3)
    per-DOF mask — cases with identical fixed DOFs produce identical `free`
    signatures and can share the returned context."""
    warnings: list[str] = []
    used_dofs = np.unique(edof.ravel())
    fixed_dofs = np.unique(_fixed_dofs(fixed_nodes, domain.node_count))
    if np.asarray(fixed_nodes).ndim == 2:
        fixed3 = np.asarray(fixed_nodes, dtype=bool)
        constrained_dirs = int(np.count_nonzero(fixed3.any(axis=0)))
        if constrained_dirs < 3:
            warnings.append(
                f"only {constrained_dirs}/3 translation directions are constrained — "
                "check for unconstrained rigid-body modes"
            )
    free = used_dofs[~np.isin(used_dofs, fixed_dofs, assume_unique=True)]
    if free.size == 0:
        Kf = sparse.csr_matrix((0, 0))
        return CaseSolve(free=free, Kf=Kf, factor=None, M=None, warnings=warnings)

    Kf = K[free][:, free]
    factor: Callable[[np.ndarray], np.ndarray] | None = None
    M: Any = None
    if free.size <= spsolve_dofs:
        factor = factor_symmetric(Kf)
        if factor is None:
            warnings.append("direct factorization failed; falling back to CG")
    if factor is None:
        M = _ssor_preconditioner(Kf.tocsr())
        if M is None:
            diag = Kf.diagonal()
            diag[diag == 0] = 1.0
            M = spla.diags(1.0 / diag)
    return CaseSolve(free=free, Kf=Kf, factor=factor, M=M, warnings=warnings)


def solve_reduced(
    domain: VoxelDomain,
    ctx: CaseSolve,
    edof: np.ndarray,
    active: np.ndarray,
    F: np.ndarray,
    elem_E: np.ndarray,
    nu: float,
    *,
    u0: np.ndarray | None = None,
    E_ref: float | None = None,
) -> FemSolution:
    """Solve + post-stress for a case on its `reduce_case` context."""
    warnings = list(ctx.warnings)
    free = ctx.free
    Kf = ctx.Kf
    used_nodes = np.unique(domain.element_nodes()[active])
    if free.size == 0:
        return FemSolution(
            u=np.zeros(3 * domain.node_count),
            elem_vm=np.zeros(domain.cell_count),
            node_disp=np.zeros((domain.node_count, 3)),
            max_vm_mpa=0.0,
            max_disp_mm=0.0,
            compliance_j=0.0,
            dofs=0,
            converged=False,
            warnings=["no free DOFs — everything is constrained"],
        )

    Ff = F[free]
    u = np.zeros(3 * domain.node_count)
    solver_name = "spsolve"
    converged = True
    if ctx.factor is not None:
        try:
            u[free] = np.asarray(ctx.factor(Ff), dtype=float)
        except Exception as e:
            warnings.append(f"direct solve failed ({e}); falling back to CG")
            solver_name = "cg"
            converged = False
    else:
        solver_name = "cg"
        converged = False
    if solver_name == "cg":
        M = ctx.M
        if M is None:
            M = _ssor_preconditioner(Kf.tocsr())
        if M is None:
            diag = Kf.diagonal()
            diag[diag == 0] = 1.0
            M = spla.diags(1.0 / diag)
        x0 = u0[free] if u0 is not None else None
        x, info = spla.cg(Kf, Ff, x0=x0, rtol=1e-6, atol=0.0, maxiter=5000, M=M)
        u[free] = x
        converged = info == 0
        if info > 0:
            warnings.append(f"CG did not converge in {info} iterations")

    if not np.all(np.isfinite(u)):
        warnings.append("solution contains NaN/inf — likely unconstrained rigid-body modes")

    # element von Mises at cell centres (only for cells carrying real stiffness)
    D = _D(nu) * (E_ref if E_ref is not None else float(elem_E[active].max(initial=1.0)))
    ue = u[edof]
    eps = ue @ _B0(domain.pitch).T  # (nelem, 6)
    sig = eps @ D.T  # (nelem, 6)
    sx, sy, sz, txy, tyz, tzx = sig.T
    vm = np.sqrt(
        np.clip(
            sx**2 + sy**2 + sz**2 - sx * sy - sy * sz - sz * sx + 3 * (txy**2 + tyz**2 + tzx**2),
            0.0,
            None,
        )
    )
    elem_vm = np.zeros(domain.cell_count)
    elem_vm[active] = vm

    node_disp = u.reshape(-1, 3)
    disp_mag = np.linalg.norm(node_disp[used_nodes], axis=1) if used_nodes.size else [0.0]
    return FemSolution(
        u=u,
        elem_vm=elem_vm,
        node_disp=node_disp,
        max_vm_mpa=float(vm.max(initial=0.0)),
        max_disp_mm=float(np.max(disp_mag)),
        compliance_j=float(F[free] @ u[free]),
        dofs=int(free.size),
        converged=converged,
        solver=solver_name,
        warnings=warnings,
    )


def solve(
    domain: VoxelDomain,
    elem_E: np.ndarray,
    nu: float,
    fixed_nodes: np.ndarray,
    F: np.ndarray,
    *,
    cell_mask: np.ndarray | None = None,
    u0: np.ndarray | None = None,
    spsolve_dofs: int = 120_000,
    E_ref: float | None = None,
) -> FemSolution:
    """Solve K u = F with Dirichlet BCs on `fixed_nodes`.

    `fixed_nodes` accepts a (nnodes,) node mask (all 3 DOFs pinned — legacy)
    or a (nnodes, 3) per-DOF mask for roller/symmetry supports.
    elem_E is per-cell Young's modulus; only `cell_mask` cells (default:
    domain.solid) carry stiffness and are solved for stress.

    Single-case convenience wrapper over assemble + reduce_case +
    solve_reduced; multi-case callers should hoist `assemble` and cache
    `reduce_case` per fixed-DOF signature themselves (see
    ``analyze._solve_cases``).
    """
    mask = domain.solid.ravel() if cell_mask is None else np.asarray(cell_mask).ravel()
    K, edof, active = assemble(domain, elem_E, nu, mask)
    ctx = reduce_case(domain, K, edof, fixed_nodes, spsolve_dofs=spsolve_dofs)
    return solve_reduced(domain, ctx, edof, active, F, elem_E, nu, u0=u0, E_ref=E_ref)
