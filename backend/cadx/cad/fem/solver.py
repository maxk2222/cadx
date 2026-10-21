"""Hex8 linear-elasticity solver on the uniform VoxelDomain grid.

8-node brick elements, 2x2x2 Gauss integration, uniform element edge `pitch`
so a single element stiffness matrix is reused for the whole grid. Assembly
is the classic top3d vectorised edof/kron trick (Liu & Tovar 2014).

Units: mm, N, MPa (E). Displacements in mm.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from cadx.cad.fem.voxelize import VoxelDomain
from cadx.logging import get_logger

log = get_logger(__name__)

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


def _D(nu: float, aniso_z: float = 1.0, axis: int = 2) -> np.ndarray:
    """Constitutive matrix for E = 1 (scale later), Voigt order
    [xx, yy, zz, xy, yz, zx].

    ``aniso_z != 1`` builds an engineering model of FDM layer weakening —
    not a full orthotropic tensor. The congruence ``S @ D_iso @ S`` with
    ``S = diag(sqrt(k) * indicator)`` scales the ``axis``-normal stiffness
    and the two shear terms containing ``axis`` by ~k: pure-axis modulus
    drops to ~k·E, the axis shears to ~k·G, the in-plane response is
    untouched and the result stays SPD for any k > 0."""
    c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    s = (1.0 - 2.0 * nu) / 2.0
    D = c * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, s, 0, 0],
            [0, 0, 0, 0, s, 0],
            [0, 0, 0, 0, 0, s],
        ]
    )
    if aniso_z == 1.0:
        return D
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"aniso axis must be 0, 1 or 2, got {axis}")
    if aniso_z <= 0.0:
        raise ValueError(f"aniso factor must be > 0, got {aniso_z}")
    # the shear not touching `axis` is the one spanned by the two in-plane
    # directions: axis=x -> yz(4), axis=y -> zx(5), axis=z -> xy(3)
    shear_clear = {0: 4, 1: 5, 2: 3}[axis]
    idx = [axis] + [i for i in (3, 4, 5) if i != shear_clear]
    sv = np.ones(6)
    sv[idx] = math.sqrt(aniso_z)
    S = np.diag(sv)
    return S @ D @ S


@lru_cache(maxsize=64)
def _ke_unit(nu: float, pitch: float, aniso_z: float = 1.0, axis: int = 2) -> np.ndarray:
    """Element stiffness for E=1, uniform cube of edge `pitch`.

    Memoized — pure function of (nu, pitch, aniso_z, axis); the returned
    matrix is shared, callers must treat it as read-only."""
    D = _D(nu, aniso_z, axis)
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


class _CgDiverged(Exception):
    def __init__(self, it: int, ratio: float):
        self.it = it
        self.ratio = ratio
        super().__init__(f"iter {it}, ||r||/||b|| {ratio:g}")


def _cg_watchdog(
    Kf: sparse.csr_matrix,
    Ff: np.ndarray,
    x0: np.ndarray | None,
    M: Any,
    *,
    maxiter: int,
) -> tuple[np.ndarray | None, int | None, _CgDiverged | None]:
    """spla.cg with a divergence watchdog.

    An unconstrained/under-constrained K (region_fallback patch that fails
    to kill rigid-body modes, load components in the nullspace) makes CG
    explode rather than converge: observed 5000-iter burn ≈ 13 min on
    132k DOFs producing e14-mm garbage the caller then marks
    converged=False anyway. Three abort signals, sampled every 40
    iterations (~2% overhead): (a) |x| beyond any physically sane mm
    displacement, (b) residual blowing >5x past best-seen for 3 straight
    samples, (c) residual stuck above ||b|| with no improvement for 10
    samples (400 iters — plateaus in a converging run still trend down).
    """
    bnorm = float(np.linalg.norm(Ff)) or 1.0
    state = {"it": 0, "best": np.inf, "bad": 0, "flat": 0}

    def callback(xk: np.ndarray) -> None:
        state["it"] += 1
        if state["it"] % 40:
            return
        xnorm = float(np.max(np.abs(xk)))
        ratio = float(np.linalg.norm(Ff - Kf @ xk)) / bnorm
        if xnorm > 1e6 or not np.isfinite(ratio):
            raise _CgDiverged(state["it"], ratio)
        if ratio < state["best"] * 0.9:
            state["best"] = ratio
            state["bad"] = 0
            state["flat"] = 0
            return
        state["flat"] += 1
        if ratio > max(5.0 * state["best"], 5.0):
            state["bad"] += 1
        if state["bad"] >= 3 or (state["flat"] >= 10 and ratio > 1.0):
            raise _CgDiverged(state["it"], ratio)

    try:
        x, info = spla.cg(
            Kf, Ff, x0=x0, rtol=1e-6, atol=0.0, maxiter=maxiter, M=M, callback=callback
        )
    except _CgDiverged as e:
        return None, None, e
    return x, info, None


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
    *,
    aniso: tuple[float, float] | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble K over selected cells.

    elem_E: (ncells,) per-cell Young's modulus (only cells in cell_mask are
    assembled). Returns (K, edof, active_elem_idx): K is (3*nnodes, 3*nnodes)
    csr, edof is (nelem,24), active_elem_idx maps rows to flat cell ids.

    `ij` may carry a precomputed (iK, jK) sparsity pattern — valid only when
    `cell_mask` selects the same cells it was built for (SIMP assembles ALL
    cells every iteration, so the pattern is constant and caching it skips an
    O(nelem·576) index build per solve).

    `aniso` = (k, axis) applies the FDM layer-weakening factor to the
    element stiffness (see _D); None assembles the isotropic ke.
    """
    mask = domain.solid.ravel() if cell_mask is None else np.asarray(cell_mask).ravel()
    active = np.flatnonzero(mask)
    # element_nodes/element_dofs/_ke_unit are pure functions of the domain —
    # memoized, so repeated assembles (one per SIMP iteration, one per FEM
    # load case) only pay the COO value fill
    edof = domain_edof(domain)[active]
    ke = (
        _ke_unit(nu, domain.pitch)
        if aniso is None
        else _ke_unit(nu, domain.pitch, float(aniso[0]), int(aniso[1]))
    )

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


def _free_node_coords(domain: VoxelDomain, free: np.ndarray) -> np.ndarray:
    """(n_free_nodes, 3) world-mm coords of the nodes carrying `free` DOFs.

    Node flat index is ``i + nx1*(j + ny1*k)`` on the (dims+1) node grid
    (VoxelDomain.node_index), so ``free // 3`` recovers the node ids —
    sorted, matching `free`'s ordering — and coordinates follow
    ``origin + pitch * (i, j, k)`` exactly like domain.node_coords."""
    nodes = np.unique(np.asarray(free, dtype=np.int64) // 3)
    nx1, ny1, _ = domain.node_dims
    i = nodes % nx1
    j = (nodes // nx1) % ny1
    k = nodes // (nx1 * ny1)
    grid = np.stack([i, j, k], axis=-1)
    return domain.origin[None, :] + domain.pitch * grid.astype(float)


_PYAMG_AVAILABLE: bool | None = None


def _pyamg_available() -> bool:
    """Import-probe pyamg once — it is an optional dependency."""
    global _PYAMG_AVAILABLE
    if _PYAMG_AVAILABLE is None:
        try:
            import pyamg  # noqa: F401
        except Exception:
            _PYAMG_AVAILABLE = False
        else:
            _PYAMG_AVAILABLE = True
    return _PYAMG_AVAILABLE


def _amg_preconditioner(
    Kf: sparse.csr_matrix, coords_mm: np.ndarray
) -> spla.LinearOperator | None:
    """Smoothed-aggregation AMG preconditioner for 3D elasticity.

    `B` carries the 6 rigid-body modes (3 translations + 3 infinitesimal
    rotations) per node — the classic SA-AMG near-nullspace for elasticity,
    so aggregation keeps the coarse level well-conditioned. Requires every
    free node to own all 3 DOFs (checked via shape); returns None when
    pyamg is missing, the shapes disagree, or the hierarchy build fails."""
    if not _pyamg_available():
        return None
    n = int(np.asarray(coords_mm).shape[0])
    if n == 0 or Kf.shape[0] != 3 * n:
        return None
    try:
        import pyamg

        x, y, z = np.asarray(coords_mm, dtype=float).T
        B = np.zeros((3 * n, 6))
        B[0::3, 0] = 1.0
        B[0::3, 4] = -z
        B[0::3, 5] = y
        B[1::3, 1] = 1.0
        B[1::3, 3] = z
        B[1::3, 5] = -x
        B[2::3, 2] = 1.0
        B[2::3, 3] = -y
        B[2::3, 4] = x
        ml = pyamg.smoothed_aggregation_solver(Kf, B=B)
        return ml.aspreconditioner()
    except Exception:
        return None


def iterative_preconditioner(
    Kf: sparse.csr_matrix,
    backend: str = "auto",
    domain: VoxelDomain | None = None,
    free: np.ndarray | None = None,
) -> spla.LinearOperator:
    """CG preconditioner for a reduced elastic stiffness matrix.

    `backend` (auto|amg|ssor) overrides CADX_FEM_CG_PRECONDITIONER when not
    "auto". Smoothed-aggregation AMG is tried first — it needs pyamg plus
    `domain`+`free` to place the rigid-body near-nullspace vectors — then
    SSOR, then plain diagonal scaling. Always returns a usable M."""
    want = str(backend).strip().lower()
    if want in ("", "auto"):
        want = os.environ.get("CADX_FEM_CG_PRECONDITIONER", "auto").strip().lower()
    if want != "ssor" and domain is not None and free is not None:
        M = _amg_preconditioner(Kf, _free_node_coords(domain, free))
        if M is not None:
            return M
    M = _ssor_preconditioner(Kf)
    if M is not None:
        return M
    diag = Kf.diagonal()
    diag[diag == 0] = 1.0
    return spla.aslinearoperator(sparse.diags(1.0 / diag))


def _elastic_preconditioner(
    Kf: sparse.csr_matrix,
    domain: VoxelDomain | None = None,
    free: np.ndarray | None = None,
    backend: str = "auto",
) -> spla.LinearOperator:
    """Auto-tier alias for `iterative_preconditioner` — the env var and the
    amg->ssor->diagonal chain apply unchanged (reduce_case/solve_reduced
    call sites)."""
    return iterative_preconditioner(Kf, backend=backend, domain=domain, free=free)


_CHOLMOD_AVAILABLE: bool | None = None
_TIER_LOGGED = False


def _cholmod_available() -> bool:
    """Import-probe scikit-sparse CHOLMOD once — it is an optional dep."""
    global _CHOLMOD_AVAILABLE
    if _CHOLMOD_AVAILABLE is None:
        try:
            import sksparse.cholmod  # noqa: F401
        except Exception:
            _CHOLMOD_AVAILABLE = False
        else:
            _CHOLMOD_AVAILABLE = True
    return _CHOLMOD_AVAILABLE


def solver_tier(backend: str = "auto") -> str:
    """Resolve the direct-solve backend: "cholmod" or "splu".

    `backend` (auto|cholmod|splu|superlu) overrides the
    CADX_FEM_DIRECT_SOLVER env var when not "auto" — topopt's
    `solver_backend` benchmark knob shares this vocabulary. An unavailable
    or unrecognized name resolves to the auto pick: "cholmod" when
    scikit-sparse imports cleanly, else "splu". The import probe is cached
    and the resolved backend is logged once."""
    global _TIER_LOGGED
    want = backend if backend != "auto" else os.environ.get("CADX_FEM_DIRECT_SOLVER", "auto")
    want = str(want).strip().lower()
    # auto / cholmod / unknown names prefer cholmod when importable
    tier = (
        "splu"
        if want in ("splu", "superlu") or not _cholmod_available()
        else "cholmod"
    )
    if not _TIER_LOGGED:
        _TIER_LOGGED = True
        log.info("fem.solver_tier", backend=tier)
    return tier


class CholmodFactor:
    """CHOLMOD Cholesky factor wrapped as the ``b -> x`` solve callable.

    `refactorize` refreshes the numeric factorization on a matrix with the
    same sparsity pattern (SIMP density updates rescale values, never the
    pattern) via ``cholesky_inplace`` — one symbolic analysis amortized
    across many solves.
    """

    def __init__(self, factor: Any, pattern: sparse.spmatrix) -> None:
        self._f = factor
        self._shape = pattern.shape
        self._nnz = pattern.nnz
        self._dtype = pattern.dtype

    def __call__(self, b: np.ndarray) -> np.ndarray:
        return np.asarray(self._f(b), dtype=float)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Alias for __call__ — cholmod factors are callable directly."""
        return self(b)

    def refactorize(self, K_new: sparse.spmatrix) -> bool:
        """Recompute the numeric factorization in place for `K_new`, which
        must carry the stored sparsity pattern (shape/nnz/dtype checked).
        Returns False on any mismatch or CHOLMOD failure — the caller then
        re-factorizes fresh."""
        if K_new.shape != self._shape or K_new.nnz != self._nnz or K_new.dtype != self._dtype:
            return False
        try:
            A = K_new if sparse.isspmatrix_csc(K_new) else K_new.tocsc()
            self._f.cholesky_inplace(A)
        except Exception:
            return False
        return True


def factor_symmetric(
    Kf: sparse.csr_matrix, backend: str = "auto"
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Direct-solve callable ``b -> x`` for the reduced stiffness matrix.

    `backend` overrides CADX_FEM_DIRECT_SOLVER (auto|cholmod|splu|superlu);
    the iterative names (amg|ssor) return None so a benchmarking caller
    lands on the CG path. Kf is SPD, so the priority is CHOLMOD Cholesky
    (scikit-sparse, when the tier resolves to cholmod), then SuperLU's
    SymmetricMode — a symmetric permutation + half the numeric work of the
    generic path — then ``factorized``, then None (caller degrades to CG)."""
    want = str(backend).strip().lower()
    if want in ("amg", "ssor"):
        return None
    if solver_tier(want if want in ("cholmod", "splu", "superlu") else "auto") == "cholmod":
        try:
            from sksparse import cholmod

            kf = Kf.tocsc()
            return CholmodFactor(cholmod.cholesky(kf), kf)
        except Exception:
            pass
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
    aniso: tuple[float, float] | None = None,
    solver_backend: str = "auto",
) -> CaseSolve:
    """Extract the reduced system Kf = K[free][:, free] for one constraint
    mask and prepare its solver (CHOLMOD/symmetric SuperLU under the cutoff,
    AMG/SSOR-CG above). `fixed_nodes` accepts a (nnodes,) node mask or a
    (nnodes,3) per-DOF mask — cases with identical fixed DOFs produce
    identical `free` signatures and can share the returned context.

    `aniso` is accepted for call-site uniformity — it only changes the
    assembled K upstream (already reflected in Kf) and the stress recovery
    in solve_reduced; the reduction itself is material-blind."""
    _ = aniso
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
        factor = factor_symmetric(Kf, backend=solver_backend)
        if factor is None:
            warnings.append("direct factorization failed; falling back to CG")
    if factor is None:
        M = _elastic_preconditioner(Kf.tocsr(), domain, free, backend=solver_backend)
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
    aniso: tuple[float, float] | None = None,
    solver_backend: str = "auto",
) -> FemSolution:
    """Solve + post-stress for a case on its `reduce_case` context.

    `aniso` = (k, axis) must match the value K was assembled with — it
    rescales the constitutive matrix used for von Mises recovery (see _D)."""
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
            M = _elastic_preconditioner(Kf.tocsr(), domain, free, backend=solver_backend)
        x0 = u0[free] if u0 is not None else None
        x, info, diverged = _cg_watchdog(Kf, Ff, x0, M, maxiter=5000)
        if diverged is not None:
            # divergent iterate is numerically meaningless — return zeros so
            # downstream stress fields read as a failed solve, not e14-mm junk
            converged = False
            warnings.append(
                f"CG diverging — aborted at iteration {diverged.it} "
                f"(||r||/||b|| {diverged.ratio:g}); check fixed/load regions"
            )
        else:
            u[free] = x
            converged = info == 0
            if info is not None and info > 0:
                warnings.append(f"CG did not converge in {info} iterations")

    if not np.all(np.isfinite(u)):
        warnings.append("solution contains NaN/inf — likely unconstrained rigid-body modes")

    # element von Mises at cell centres (only for cells carrying real stiffness)
    k_aniso, a_axis = (1.0, 2) if aniso is None else (float(aniso[0]), int(aniso[1]))
    D = _D(nu, k_aniso, a_axis) * (
        E_ref if E_ref is not None else float(elem_E[active].max(initial=1.0))
    )
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
    aniso: tuple[float, float] | None = None,
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
    K, edof, active = assemble(domain, elem_E, nu, mask, aniso=aniso)
    ctx = reduce_case(domain, K, edof, fixed_nodes, spsolve_dofs=spsolve_dofs, aniso=aniso)
    return solve_reduced(
        domain, ctx, edof, active, F, elem_E, nu, u0=u0, E_ref=E_ref, aniso=aniso
    )
