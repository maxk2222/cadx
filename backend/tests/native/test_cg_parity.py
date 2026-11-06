"""Native-vs-SciPy parity for the Rust PCG loop.

`native.cg` must reproduce `scipy.sparse.linalg.cg` within float noise and
must surface watchdog divergence the same way `_cg_watchdog` does — the
solver treats diverged systems as unusable, never as results. Skips cleanly
when cadx_native is absent."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

cadx_native = pytest.importorskip("cadx_native")

from cadx.cad import native  # noqa: E402


def _spd(n: int = 400, seed: int = 7) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=0.02, random_state=rng, format="csr")
    return (a.T @ a + 4.0 * sp.identity(n)).tocsr()


def test_native_cg_matches_scipy():
    k = _spd()
    rng = np.random.default_rng(3)
    b = rng.standard_normal(k.shape[0])
    x_ref, info = spla.cg(k, b, rtol=1e-10, maxiter=2000)
    assert info == 0
    out = native.cg(k, b, None, None, rtol=1e-10, maxiter=2000)
    assert out is not None
    x, iters, status, ratio = out
    assert status == "converged"
    assert ratio < 1e-10
    assert np.max(np.abs(x - x_ref)) < 1e-8 * np.max(np.abs(x_ref))
    assert iters > 0


def test_native_cg_preconditioner_matches():
    k = _spd()
    rng = np.random.default_rng(11)
    b = rng.standard_normal(k.shape[0])
    diag = k.diagonal()
    m = spla.LinearOperator(k.shape, matvec=lambda v: v / diag)
    x_ref, info = spla.cg(k, b, M=m, rtol=1e-10, maxiter=2000)
    assert info == 0
    out = native.cg(k, b, None, m, rtol=1e-10, maxiter=2000)
    assert out is not None
    x, iters, status, _ = out
    assert status == "converged"
    assert np.max(np.abs(x - x_ref)) < 1e-8 * np.max(np.abs(x_ref))
    assert iters > 0


def test_native_cg_warm_start():
    k = _spd()
    rng = np.random.default_rng(5)
    b = rng.standard_normal(k.shape[0])
    x_exact = spla.spsolve(k, b)
    out = native.cg(k, b, x_exact, None, rtol=1e-10, maxiter=100)
    assert out is not None
    x, iters, status, _ = out
    assert status == "converged"
    assert iters <= 1
    assert np.max(np.abs(x - x_exact)) < 1e-10


def test_native_cg_flags_singular_system():
    # a free rigid mode: row/col of zeros on an otherwise grounded block
    k = sp.csr_matrix(np.diag([4.0, 4.0, 0.0, 4.0]))
    b = np.array([1.0, 1.0, 1.0, 1.0])
    out = native.cg(k, b, None, None, rtol=1e-10, maxiter=200)
    assert out is not None
    _, iters, status, ratio = out
    assert status == "diverged"
    assert iters <= 200
    assert np.isnan(ratio) or ratio > 1.0
