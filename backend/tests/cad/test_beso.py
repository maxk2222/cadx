"""BESO topology optimization — the soft-kill binary sibling of simp().

Same contract as the SIMP tests: a cantilever resolved through
``resolve_case`` must converge to the target volfrac with an essentially
0/1 density field, pinned passive/void regions and the shared
``TopOptResult.status`` vocabulary.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import trimesh

import cadx.cad.fem.beso as beso_mod
from cadx.cad.fem.beso import beso
from cadx.cad.fem.regions import region_node_mask, resolve_case
from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _box_mesh(x: float = 24.0, y: float = 8.0, z: float = 8.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _cantilever() -> LoadCase:
    return LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -50.0],
            )
        ],
    )


def _case(pitch: float = 1.5):
    dom = voxelize(_box_mesh(), pitch, 200_000)
    fixed, F, _ = resolve_case(dom, _cantilever(), [])
    return dom, fixed, F


def _params(**kw: Any) -> TopOptParams:
    p = TopOptParams(volfrac=0.45, rmin_mm=2.0, max_iter=80, tol=0.01)
    for k, v in kw.items():
        setattr(p, k, v)
    p.beso_er = 0.05  # extension point: beso() reads it via getattr
    return p


# ---------------------------------------------------------------------------
# convergence / field shape
# ---------------------------------------------------------------------------


def test_beso_converges_to_volfrac() -> None:
    dom, fixed, F = _case()
    res = beso(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        _params(),
    )
    assert res.status == "converged"
    assert res.converged
    assert res.iterations > 0
    assert res.history
    achieved = res.density[dom.solid.ravel()].sum() / dom.solid_count
    assert math.isclose(achieved, 0.45, abs_tol=0.05)


def test_beso_density_is_binary() -> None:
    """BESO produces a 0/1 design — >=99% of design cells at the floor or 1."""
    dom, fixed, F = _case()
    passive = passive_solid_mask(dom, [], [fixed], [region_node_mask(dom, _cantilever().loads[0].region)])
    res = beso(dom, E_PLA, NU, [(fixed, F)], passive, _params())
    design = dom.solid.ravel() & ~passive
    rho = res.density[design]
    binary = (rho <= 0.01) | (rho >= 0.99)
    assert float(binary.mean()) >= 0.99


def test_beso_preserves_passive_and_void() -> None:
    """keep/load/fix cells stay at 1.0; passive_void cells stay at rho_min."""
    dom, fixed, F = _case()
    case = _cantilever()
    passive = passive_solid_mask(dom, [], [fixed], [region_node_mask(dom, case.loads[0].region)])
    centers = dom.element_centers().reshape((*dom.dims, 3))
    pvoid = ((centers[..., 0] > -4.0) & (centers[..., 0] < 4.0)).ravel() & dom.solid.ravel()
    res = beso(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        _params(),
        passive_void=pvoid,
    )
    assert res.status in (
        "converged",
        "infeasible_target",
        "limit_cycle",
        "degenerate",
        "budget_exhausted",
    )
    assert (res.density[passive] == 1.0).all()
    assert (res.density[pvoid] < 0.01).all()
    assert (res.density[~dom.solid.ravel()] < 0.01).all()


def test_beso_deterministic() -> None:
    """Two identical runs must produce identical fields — the keep ranking
    is a stable argsort, no randomness anywhere."""
    dom, fixed, F = _case()
    kw: dict[str, Any] = {}
    r1 = beso(dom, E_PLA, NU, [(fixed, F)], np.zeros(dom.cell_count, dtype=bool), _params(), **kw)
    r2 = beso(dom, E_PLA, NU, [(fixed, F)], np.zeros(dom.cell_count, dtype=bool), _params(), **kw)
    assert np.array_equal(r1.density, r2.density)
    assert r1.iterations == r2.iterations
    assert r1.status == r2.status


# ---------------------------------------------------------------------------
# status vocabulary
# ---------------------------------------------------------------------------


def test_beso_infeasible_target() -> None:
    """Passive cells above the target pin the floor — all design cells get
    removed and the run reports infeasible_target, not convergence."""
    dom, fixed, F = _case(pitch=2.0)
    passive = np.zeros(dom.cell_count, dtype=bool)
    sidx = np.flatnonzero(dom.solid.ravel())
    passive[sidx[: int(0.62 * sidx.size)]] = True
    res = beso(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        _params(volfrac=0.3, max_iter=60),
    )
    assert res.status == "infeasible_target"
    assert res.converged is False
    assert res.passive_frac == pytest.approx(0.62, abs=0.01)
    assert any("passive" in w and "floor" in w for w in res.warnings)


def test_beso_no_design_cells_infeasible() -> None:
    dom, fixed, F = _case()
    passive = dom.solid.ravel().copy()
    res = beso(dom, E_PLA, NU, [(fixed, F)], passive, _params(volfrac=0.4))
    assert res.status == "infeasible_target"
    assert res.converged is False
    assert res.iterations == 0


def test_beso_budget_exhausted() -> None:
    """max_iter reached mid-ramp (tol=0 keeps flips > 0 -> never settles)."""
    dom, fixed, F = _case()
    res = beso(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        _params(max_iter=4, tol=0.0, converge_patience=50),
    )
    assert res.iterations == 4
    assert res.status == "budget_exhausted"
    assert res.converged is False


def test_beso_degenerate_on_solver_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside the per-iteration assembly/solve returns the last
    field flagged 'degenerate' — mirroring simp()'s honest outcome."""

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("boom — singular stiffness")

    monkeypatch.setattr(beso_mod, "assemble", _boom)
    dom, fixed, F = _case()
    res = beso(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        _params(max_iter=10),
    )
    assert res.status == "degenerate"
    assert res.converged is False
    assert any("FEM solve failed" in w and "boom" in w for w in res.warnings)


def test_beso_warns_on_simp_only_params() -> None:
    """Projection/robust/constraint params are accepted but reported."""
    dom, fixed, F = _case()
    res = beso(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        _params(filter="heaviside", robust=True, max_stress_mpa=50.0, max_iter=5),
    )
    blob = " ".join(res.warnings)
    assert "beso ignores" in blob
    assert "heaviside" in blob.lower() or "Heaviside" in blob
    assert "robust" in blob
    assert "max_stress_mpa" in blob


def test_beso_history_and_blueprint() -> None:
    dom, fixed, F = _case()
    res = beso(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        _params(max_iter=6),
    )
    assert res.history
    rec = res.history[-1]
    for key in ("iteration", "compliance_mj", "volfrac", "change", "solve_s"):
        assert key in rec, key
    assert res.blueprint is not None
    assert np.array_equal(res.blueprint, res.density)
