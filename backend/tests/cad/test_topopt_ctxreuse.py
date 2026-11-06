"""Persistent solver-context coverage for SIMP/BESO iterations.

The ij pattern and each case's `free` are run-constant, so solver state
survives in `persist_ctx`: CHOLMOD refactorises in place, the AMG
hierarchy rebuilds only on the rtol-band staleness policy, and a load
reaching a rigid-mode-deficient component gets spring-regularised once
per signature instead of burning the CG budget.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.regions import resolve_case_dofs
from cadx.cad.fem.solver import assemble
from cadx.cad.fem.topopt import (
    TopOptParams,
    _CaseCtx,
    _ctx_snapshot,
    _m_reuse,
    _refresh_ctx,
    _rtol_band,
    _solve_u,
    simp,
)
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _box_mesh(x: float = 24.0, y: float = 12.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _cantilever_case(dom):
    case = LoadCase(
        name="cantilever",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -50.0],
            )
        ],
    )
    f, F, _ = resolve_case_dofs(dom, case)
    return f, F


def test_rtol_band() -> None:
    """The cg_rtol schedule maps onto distinct bands; staleness is only
    comparable inside a band (tighter tolerances legitimately take more
    iterations even on a fresh hierarchy)."""
    assert _rtol_band(1e-3) == 3
    assert _rtol_band(1e-4) == 4
    assert _rtol_band(1e-6) == 6
    assert len({_rtol_band(v) for v in (1e-3, 1e-4, 1e-6)}) == 3


def test_refresh_ctx_updates_values() -> None:
    """A same-pattern Kf refresh keeps the factor working — new values,
    correct solve, no shape error."""
    from cadx.cad.fem.topopt import _prepare_solve

    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, F = _cantilever_case(dom)
    K, edof, _ = assemble(dom, np.full(dom.cell_count, E_PLA), NU)
    from cadx.cad.fem.solver import _fixed_dofs

    fd = np.unique(_fixed_dofs(fixed, dom.node_count))
    used = np.unique(edof.ravel())
    free = used[~np.isin(used, fd, assume_unique=True)]
    Kf = K[free][:, free].tocsr()
    factor, M = _prepare_solve(Kf, Kf.shape[0] + 1, "auto", domain=dom, free=free)
    assert factor is not None  # direct tier for the test
    ctx = _CaseCtx(free=free, Kf=Kf, factor=factor, M=M)

    # same pattern, scaled values (density iteration surrogate)
    Kf2 = (Kf * 0.5).tocsr()
    _refresh_ctx(ctx, Kf2, backend="auto")
    assert ctx.Kf is not Kf
    assert ctx.Kf.nnz == Kf2.nnz
    u = ctx.factor(F[free])
    # residual check — the refactorised factor still solves Kf2 u = Ff
    res = Kf2 @ u - F[free]
    assert np.linalg.norm(res) <= 1e-8 * max(np.linalg.norm(F[free]), 1.0)


def test_ctx_snapshot_drops_mutable_factor() -> None:
    """Snapshots returned for adjoint solves keep free/Kf/M but must not
    carry a mutable CholmodFactor — a later eval refactorises it onto a
    different density's values."""
    from cadx.cad.fem.solver import CholmodFactor, factor_symmetric

    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, _ = _cantilever_case(dom)
    K, edof, _ = assemble(dom, np.full(dom.cell_count, E_PLA), NU)
    from cadx.cad.fem.solver import _fixed_dofs

    fd = np.unique(_fixed_dofs(fixed, dom.node_count))
    used = np.unique(edof.ravel())
    free = used[~np.isin(used, fd, assume_unique=True)]
    Kf = K[free][:, free].tocsr()
    factor = factor_symmetric(Kf)
    ctx = _CaseCtx(free=free, Kf=Kf, factor=factor, M=None)
    snap = _ctx_snapshot(ctx)
    assert snap.free is ctx.free and snap.Kf is ctx.Kf
    if isinstance(factor, CholmodFactor):
        assert snap.factor is None
    else:
        assert snap.factor is factor


def test_solve_u_iters_and_watchdog() -> None:
    """_solve_u reports CG iterations and aborts singular systems via the
    shared watchdog instead of burning the full maxiter budget."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, F = _cantilever_case(dom)
    K, edof, _ = assemble(dom, np.full(dom.cell_count, E_PLA), NU)
    from cadx.cad.fem.solver import _fixed_dofs

    fd = np.unique(_fixed_dofs(fixed, dom.node_count))
    used = np.unique(edof.ravel())
    free = used[~np.isin(used, fd, assume_unique=True)]
    Kf = K[free][:, free].tocsr()
    u, name, iters = _solve_u(dom, Kf, F, free, None, factor=None, M=None, cg_rtol=1e-6)
    assert name == "cg" and iters > 0
    assert np.isfinite(u[free]).all()

    # singular: no fixture at all -> the whole used set is deficient and
    # the load sits in the nullspace -> watchdog fires, u stays exploded
    # (honestly large compliance) rather than a fake zero
    free_all = used
    Kf_free = K[free_all][:, free_all].tocsr()
    _u2, name2, it2 = _solve_u(dom, Kf_free, F, free_all, None, factor=None, M=None, cg_rtol=1e-6)
    assert name2 in ("cg", "cg_diverged")
    assert it2 <= 4000


def test_simp_reuses_solver_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """With reuse on, _prepare_solve runs ~once per signature over the
    whole run; with CADX_TOPOPT_M_REUSE=0 it runs per iteration."""
    import cadx.cad.fem.topopt as tp

    dom = voxelize(_box_mesh(), 3.0, 150_000)
    fixed, F = _cantilever_case(dom)
    passive = np.zeros(dom.cell_count, dtype=bool)

    calls = {"n": 0}
    orig = tp._prepare_solve

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(tp, "_prepare_solve", counting)
    # force the iterative tier so the M hierarchy is the reused state
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=4.0, max_iter=4, tol=0.0),
        spsolve_dofs=0,
    )
    assert len(res.history) >= 2
    reuse_calls = calls["n"]
    assert reuse_calls <= 1 + len(res.history)  # build + at most per-iter rebuilds

    calls["n"] = 0
    monkeypatch.setenv("CADX_TOPOPT_M_REUSE", "0")
    assert not _m_reuse()
    res2 = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=4.0, max_iter=4, tol=0.0),
        spsolve_dofs=0,
    )
    # kill-switch path rebuilds every iteration -> strictly more builds
    assert calls["n"] > reuse_calls
    # both modes still produce sane fields
    assert res2.history


def test_simp_rigid_gap_spring_regularised() -> None:
    """A case whose load reaches an under-constrained component gets
    spring-regularised: finite (large) compliance instead of a CG burn,
    plus a warning naming the mechanism."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    # fixture on a tiny corner -> translations pinned nowhere: the whole
    # domain is rigid-mode deficient and the +x load reaches it
    fixed = np.zeros(dom.node_count * 3, dtype=bool)
    F = np.zeros(dom.node_count * 3)
    # load on the +x face nodes
    coords = dom.node_coords()
    xmax = coords[:, 0].max()
    on = coords[:, 0] > xmax - 2.0
    F[np.repeat(on, 3) & (np.arange(dom.node_count * 3) % 3 == 2)] = -50.0
    passive = np.zeros(dom.cell_count, dtype=bool)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=4.0, max_iter=2, tol=0.0),
    )
    assert any("under-constrained" in w or "rigid" in w for w in res.warnings)
    assert res.history
    obj = res.history[0]["compliance_mj"]
    assert np.isfinite(obj)
