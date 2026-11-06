"""Native-vs-scipy parity for the FEM kernels (CSR assembly, SpMV)."""

from __future__ import annotations

import numpy as np
import pytest

cadx_native = pytest.importorskip("cadx_native")

from cadx.cad import native  # noqa: E402
from cadx.cad.fem.solver import _ke_unit, assemble  # noqa: E402
from cadx.cad.fem.voxelize import VoxelDomain  # noqa: E402


def _domain(dims=(4, 3, 2), pitch=1.0, fill=0.8, seed=1):
    rng = np.random.default_rng(seed)
    solid = rng.random(dims) < fill
    solid[0, 0, 0] = solid[-1, -1, -1] = True  # never an empty domain
    return VoxelDomain(
        origin=np.array([0.0, 0.0, 0.0]), pitch=pitch, dims=dims, solid=solid
    )


def _ij(domain, active):
    from cadx.cad.fem.solver import domain_edof

    edof = domain_edof(domain)[active]
    return np.repeat(edof, 24, axis=1).ravel(), np.tile(edof, (1, 24)).ravel()


def test_assemble_parity():
    domain = _domain()
    rng = np.random.default_rng(2)
    elem_E = np.where(domain.solid.ravel(), 2e5, 0.0) * (0.5 + rng.random(domain.cell_count))
    K, edof, active = assemble(domain, elem_E, 0.3)

    iK, jK = _ij(domain, active)
    asm = native.assembler_for(iK, jK, 3 * domain.node_count)
    assert asm is not None
    ke = _ke_unit(0.3, domain.pitch)
    sK = native.assemble_values(asm, elem_E[active], ke)
    data = native.assemble_fill(asm, sK)
    assert data is not None

    from scipy import sparse

    K2 = sparse.csr_matrix(
        (data, np.asarray(asm.indices), np.asarray(asm.indptr)), shape=K.shape
    )
    # canonical structure identical, values within summation-order noise
    assert np.array_equal(np.asarray(K.indptr), np.asarray(asm.indptr))
    assert np.array_equal(np.asarray(K.indices), np.asarray(asm.indices))
    assert np.allclose(K.data, K2.data, rtol=1e-9, atol=1e-8)


def test_assemble_parity_class_table():
    """class_D path — per-class stiffness table."""
    domain = _domain()
    rng = np.random.default_rng(3)
    ncls = 2
    elem_class = (rng.random(domain.cell_count) * ncls).astype(np.int64)
    from cadx.cad.fem.solver import _D, _ke_from_D

    class_D = np.stack([_D(0.3), _D(0.45)])
    elem_E = np.where(domain.solid.ravel(), 1e5, 0.0)
    K, edof, active = assemble(
        domain, elem_E, 0.3, elem_class=elem_class, class_D=class_D
    )

    iK, jK = _ij(domain, active)
    asm = native.assembler_for(iK, jK, 3 * domain.node_count)
    assert asm is not None
    ke_table = np.stack([_ke_from_D(Dk, domain.pitch).ravel() for Dk in class_D])
    sK = native.assemble_values(asm, elem_E[active], ke_table, elem_class[active])
    data = native.assemble_fill(asm, sK)
    from scipy import sparse

    K2 = sparse.csr_matrix(
        (data, np.asarray(asm.indices), np.asarray(asm.indptr)), shape=K.shape
    )
    # atol relative to the matrix scale (~2e4): 1e-8 ≈ fp summation-order
    # noise on cancelling near-zero off-diagonals
    assert np.allclose(K.data, K2.data, rtol=1e-9, atol=1e-8)


def test_spmv_parity():
    domain = _domain()
    K, _edof, _act = assemble(domain, np.where(domain.solid.ravel(), 2e5, 0.0), 0.3)
    Kf = K.tocsr()
    op = native.spmv_op(Kf)
    assert op is not None
    rng = np.random.default_rng(4)
    x = rng.random(Kf.shape[0])
    ref = Kf @ x
    got = np.asarray(op @ x)
    assert np.allclose(ref, got, rtol=1e-12, atol=0)


def test_spmv_in_cg_solve_parity():
    """CG with the native matvec converges to the same u as the scipy A."""
    from scipy.sparse import linalg as spla

    # solid cantilever: fully dense block, one face clamped — SPD system
    # (a sparse random domain with 3 pinned dofs is singular → CG diverges
    # legitimately and parity of the iterates is meaningless)
    dims = (5, 4, 3)
    domain = VoxelDomain(
        origin=np.zeros(3), pitch=1.0, dims=dims,
        solid=np.ones(dims, dtype=bool),
    )
    K, _edof, _act = assemble(domain, np.full(domain.cell_count, 2e5), 0.3)
    nx, ny, _nz = domain.node_dims
    n0 = np.arange(nx * ny)  # k=0 face — i fastest layout per node_index
    fixed = np.concatenate([3 * n0 + c for c in range(3)])
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    Kf = K[free][:, free].tocsr()
    rng = np.random.default_rng(5)
    b = rng.random(len(free))
    ref, _ = spla.cg(Kf, b, rtol=1e-10, atol=0.0, maxiter=2000)
    op = native.spmv_op(Kf)
    got, _ = spla.cg(op, b, rtol=1e-10, atol=0.0, maxiter=2000)
    assert np.allclose(ref, got, rtol=1e-5)
