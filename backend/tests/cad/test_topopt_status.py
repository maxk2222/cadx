"""TopOptResult.status vocabulary + export-degeneracy gates + surfaced
passive metrics (milestones B17/B18/B19 + O2/O6).

B18 — ``TopOptResult.status`` reports HOW a SIMP run ended:
``converged | infeasible_target | limit_cycle | degenerate |
budget_exhausted``; ``engine.py`` propagates it to ``res.topopt["status"]``
and toolspecs re-exports it as ``out["status"]``.

O2 — the limit-cycle break no longer waits for the beta cap: a density field
saturating >=0.9*move for 10+ consecutive iterations with <1% compliance
improvement over the window is extracted early with status ``limit_cycle``.

B19 — ``_export_degeneracy`` gates sf_export on displacement fractions of the
part span and a 10x-yield von-Mises bound; ``verification`` carries
``meets_safety_factor`` separately from ``valid``.

B17 — preview_regions/check_load emit structured cell counts
(``n_fixed_cells``/``n_load_cells``/``n_keep_cells`` per case) plus top-level
``passive_fraction``/``min_feasible_volfrac``; topopt_run re-exports them as
``passive_fraction``/``free_fraction``/``min_feasible_volfrac``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

import cadx.cad.fem.topopt as topopt_mod
from cadx.cad.fem.regions import resolve_case
from cadx.cad.fem.topopt import (
    TopOptParams,
    _windowed_compliance_improvement,
    simp,
)
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec
from cadx.runtime_config import RuntimeSettings

E_PLA = 3500.0
NU = 0.36


def _box_mesh(x: float = 20.0, y: float = 8.0, z: float = 8.0) -> trimesh.Trimesh:
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


def _case() -> tuple[np.ndarray, np.ndarray]:
    dom = voxelize(_box_mesh(), 1.0, 200_000)
    fixed, F, _ = resolve_case(dom, _cantilever(), [])
    return dom, fixed, F


# ---------------------------------------------------------------------------
# B18 — status vocabulary
# ---------------------------------------------------------------------------


def test_status_converged() -> None:
    """A normal run that settles under `tol` reports status 'converged'."""
    dom = voxelize(_box_mesh(12.0, 6.0, 6.0), 1.5, 100_000)
    fixed, F, _ = resolve_case(dom, _cantilever(), [])
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=60),
    )
    assert res.converged
    assert res.status == "converged"


def test_status_infeasible_target() -> None:
    """Passive cells covering more than the requested volfrac pin the floor
    above the target: the optimizer drains every free cell and settles —
    that is infeasible_target, not a converged optimum."""
    dom, fixed, F = _case()
    passive = np.zeros(dom.cell_count, dtype=bool)
    sidx = np.flatnonzero(dom.solid.ravel())
    passive[sidx[: int(0.62 * sidx.size)]] = True
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.3, rmin_mm=2.0, max_iter=30),
    )
    assert res.status == "infeasible_target"
    assert res.converged is False
    assert res.passive_frac == pytest.approx(0.62, abs=0.01)
    assert any("passive" in w and "floor" in w for w in res.warnings)


def test_status_no_design_cells_infeasible() -> None:
    """Everything pinned -> no free cells at all: the passive floor is the
    whole domain, so the request is infeasible by construction."""
    dom, fixed, F = _case()
    passive = dom.solid.ravel().copy()
    res = simp(dom, E_PLA, NU, [(fixed, F)], passive, TopOptParams(volfrac=0.4))
    assert res.status == "infeasible_target"
    assert res.converged is False


def test_status_budget_exhausted() -> None:
    """max_iter reached without settling -> budget_exhausted (tol=0 keeps the
    loop alive so the cap is what ends it)."""
    dom, fixed, F = _case()
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=4, tol=0.0, converge_patience=50),
    )
    assert res.iterations == 4
    assert res.status == "budget_exhausted"
    assert res.converged is False


def test_status_degenerate_solver_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside the per-iteration FEM eval is not a lost run —
    the last density field is returned flagged 'degenerate' with a warning."""
    dom, fixed, F = _case()

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("boom — singular stiffness")

    monkeypatch.setattr(topopt_mod, "assemble", _boom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=10),
    )
    assert res.status == "degenerate"
    assert res.converged is False
    assert any("FEM solve failed" in w and "boom" in w for w in res.warnings)


def test_min_primary_iters_runs_at_least_three() -> None:
    """P5: a warm-started field already at the optimum can satisfy
    change<=tol at it=0-2 — the OC loop owes at least a few evaluations
    before trusting that early exit (a multires-warmed density otherwise
    'converges' having done no work at the fine resolution)."""
    dom, fixed, F = _case()
    p = TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=10)
    passive = np.zeros(dom.cell_count, dtype=bool)
    res1 = simp(dom, E_PLA, NU, [(fixed, F)], passive, p)
    warm = simp(dom, E_PLA, NU, [(fixed, F)], passive, p, x0=res1.density)
    assert warm.iterations >= 3


# ---------------------------------------------------------------------------
# O2 — early limit-cycle detection
# ---------------------------------------------------------------------------


def test_windowed_compliance_improvement() -> None:
    """The window helper returns the relative drop over the last `window`
    records; short history or a non-positive baseline is None (unknown)."""
    hist = [{"compliance_mj": v} for v in [100.0] * 8 + [99.2]]
    assert _windowed_compliance_improvement(hist, 10) is None  # window not full
    hist = [{"compliance_mj": v} for v in [100.0] + [100.0 + (i % 2) * 0.2 for i in range(9)]]
    imp = _windowed_compliance_improvement(hist, 10)
    assert imp is not None and imp < 0.01  # oscillating, not improving
    hist = [{"compliance_mj": 100.0}] + [{"compliance_mj": 90.0}] * 9
    imp = _windowed_compliance_improvement(hist, 10)
    assert imp is not None and imp == pytest.approx(0.10)
    assert _windowed_compliance_improvement([{"compliance_mj": -1.0}] * 12, 10) is None
    # a beta step inside the window redefines the objective surface — only
    # the trailing same-beta segment is judged, and its first record (a
    # stale objective under the previous projection) is dropped, so a fresh
    # ramp reads 'unknown' until the segment fills
    hist = [{"compliance_mj": 50.0, "beta": 1.0}] * 5 + [
        {"compliance_mj": 80.0, "beta": 2.0}
    ] * 7
    assert _windowed_compliance_improvement(hist, 10) is None
    hist += [{"compliance_mj": 76.0, "beta": 2.0}] * 4
    imp = _windowed_compliance_improvement(hist, 10)
    assert imp is not None and imp == pytest.approx(0.05)


def test_limit_cycle_early_break_before_beta_cap() -> None:
    """A real period-2 limit cycle: this plate under Heaviside continuation
    saturates the move cap and bounces compliance between two layouts at a
    fixed beta forever — the early break must fire before beta_max/max_iter
    and extract the crisper of the two alternating phases."""
    m = trimesh.creation.box(extents=[30.0, 30.0, 6.0])
    m.apply_translation([0.0, 0.0, 3.0])
    dom = voxelize(m, 1.5, 300_000)
    case = LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.5),
                force_n=[0.0, 0.0, -30.0],
            )
        ],
    )
    fixed, F, _ = resolve_case(dom, case, [])
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(
            volfrac=0.5,
            rmin_mm=3.0,
            max_iter=55,
            tol=0.01,
            converge_patience=5,
            filter="heaviside",
            beta_max=8.0,
        ),
    )
    assert res.status == "limit_cycle"
    assert res.converged is False
    assert res.iterations < 55
    assert res.iterations < 45 or all(  # broke before a useless beta_cap tail
        "beta cap" not in w for w in res.warnings
    )
    assert any("limit cycle" in w for w in res.warnings)
    # the extracted phase is the crisper one of the two alternating fields —
    # a mid-cycle break must not ship the blurry mirror
    solid_rho = res.density[dom.solid.ravel()]
    crisp = float(((solid_rho < 0.2) | (solid_rho > 0.8)).mean())
    assert crisp > 0.85


def test_limit_cycle_still_improving_runs_on() -> None:
    """Move-cap saturation alone is not a limit cycle — without sustained
    mask churn and a stalled compliance window the loop keeps running."""
    dom, fixed, F = _case()
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(
            volfrac=0.4,
            rmin_mm=2.0,
            max_iter=12,
            move=0.001,
            tol=0.0,
            converge_patience=50,
        ),
    )
    assert res.iterations == 12  # never broke early
    assert res.status == "budget_exhausted"


# ---------------------------------------------------------------------------
# B19 — export degeneracy gates + meets_safety_factor
# ---------------------------------------------------------------------------


def test_export_degeneracy_thresholds() -> None:
    from cadx.cad.engine import _export_degeneracy

    # sane numbers -> valid, silent
    valid, warns = _export_degeneracy(0.01 * 100.0, 20.0, 100.0, 50.0)
    assert valid and not warns

    # 0.15..0.3 span -> warning, still valid (the 27mm-on-130mm case)
    valid, warns = _export_degeneracy(0.2 * 100.0, 20.0, 100.0, 50.0)
    assert valid
    assert any("unusually large displacement" in w for w in warns)

    # 0.3..0.5 span + vm over 10x yield -> degenerate, invalid
    valid, warns = _export_degeneracy(0.4 * 100.0, 600.0, 100.0, 50.0)
    assert not valid
    assert any("degenerate" in w for w in warns)
    assert any("10x material yield" in w for w in warns)

    # same displacement under the yield gate -> stays valid
    valid, warns = _export_degeneracy(0.4 * 100.0, 100.0, 100.0, 50.0)
    assert valid
    assert any("unusually large" in w for w in warns)

    # > half the span -> degenerate, invalid
    valid, warns = _export_degeneracy(0.6 * 100.0, 10.0, 100.0, 50.0)
    assert not valid
    assert any("half the part span" in w for w in warns)

    # vm alone > 10x yield -> warning, still valid
    valid, warns = _export_degeneracy(5.0, 600.0, 100.0, 50.0)
    assert valid
    assert any("far outside validity" in w for w in warns)

    # no measurements -> valid, silent
    valid, warns = _export_degeneracy(None, None, 100.0, 50.0)
    assert valid and not warns


def test_pipeline_emits_status_and_meets_safety_factor(tmp_path: Path) -> None:
    """res.topopt carries the run status and the verification block reports
    meets_safety_factor against the requested floor — decoupled from the
    measurement-validity flag."""
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box_mesh(24.0, 24.0, 24.0).export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever().model_dump()],
        optimize={
            "target_volume_fraction": 0.5,
            "max_iterations": 4,
            "safety_factor_min": 1.5,
            "keep_mode": "manual",
        },
        material="pla",
        voxel_mm=1.5,
        scan_anchors=False,
        runtime=RuntimeSettings(),
    )
    assert res.ok, res.error
    assert res.topopt["status"] in (
        "converged",
        "infeasible_target",
        "limit_cycle",
        "degenerate",
        "budget_exhausted",
    )
    ver = res.topopt["verification"]
    if ver.get("sf_export") is not None:
        assert ver["meets_safety_factor"] == (ver["sf_export"] >= 1.5)


# ---------------------------------------------------------------------------
# B17 — structured preview metrics + tool-result surfacing
# ---------------------------------------------------------------------------


def test_fem_preview_carries_cell_metrics(tmp_path: Path) -> None:
    """preview_only resolution emits n_fixed/n_load/n_keep cells per case and
    the passive union as passive_fraction == min_feasible_volfrac."""
    from cadx.cad.engine import run_fem_pipeline

    src = tmp_path / "b.stl"
    _box_mesh(24.0, 24.0, 24.0).export(str(src))
    res = run_fem_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever().model_dump()],
        keep_regions=[
            RegionSpec(kind="box", center=[0.0, 0.0, 3.0], diameter_mm=6.0).model_dump()
        ],
        voxel_mm=1.5,
        preview_only=True,
        render_pngs=False,
        scan_anchors=False,
        runtime=RuntimeSettings(),
    )
    assert res["ok"], res.get("error")
    prev = res["preview"]
    assert prev["pitch_mm"] == pytest.approx(1.5)
    assert prev["passive_fraction"] > 0.0
    assert prev["min_feasible_volfrac"] == prev["passive_fraction"]
    assert prev["n_keep_cells"] > 0
    case = prev["cases"][0]
    assert case["n_fixed_cells"] > 0
    assert case["n_load_cells"] > 0
    assert case["n_keep_cells"] > 0
    # the union bound: passive fraction covers keep+fixed+load cells
    assert prev["passive_fraction"] >= (
        case["n_fixed_cells"] + case["n_keep_cells"]
    ) / prev["solid_elements"] * 0.9


def _opt_session(**kw: Any):
    """ToolSession for the optimize kind with canned executor callbacks."""
    from cadx.agent.toolspecs import ToolSession
    from cadx.llm.fake import FakeLLM
    from tests.unit.test_session_condense import _ctx

    return ToolSession(_ctx(FakeLLM([{"content": "x"}])), kind="optimize", **kw)


async def test_topopt_run_tool_surfaces_status_fields() -> None:
    """topopt_run re-exports the run status, passive metrics and
    meets_safety_factor so the agent sees feasibility, not just SF."""

    async def fake_topopt(a: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "topopt": {
                "status": "infeasible_target",
                "converged": False,
                "passive_frac": 0.62,
                "volfrac_achieved": 0.62,
                "iterations": 3,
                "sf_export": 1.1,
                "verification": {"valid": True, "meets_safety_factor": False},
                "warnings": ["passive floor above target"],
            },
            "fem": {"cases": [{"safety_factor": 1.1}]},
        }

    session = _opt_session(topopt_fn=fake_topopt)
    out = await session.execute("topopt_run", {"load_cases": [{"name": "c"}]})
    assert out["ok"], out
    assert out["status"] == "infeasible_target"
    assert out["passive_fraction"] == pytest.approx(0.62)
    assert out["free_fraction"] == pytest.approx(0.38)
    assert out["min_feasible_volfrac"] == pytest.approx(0.62)
    assert out["meets_safety_factor"] is False


async def test_preview_and_check_load_surface_passive_metrics() -> None:
    """preview_regions/check_load forward the structured region metrics."""
    preview = {
        "pitch_mm": 1.5,
        "solid_elements": 4000,
        "n_keep_cells": 40,
        "passive_fraction": 0.25,
        "min_feasible_volfrac": 0.25,
        "cases": [
            {
                "case": "c",
                "n_fixed_cells": 100,
                "n_load_cells": 30,
                "n_keep_cells": 40,
            }
        ],
    }

    async def fake_preview(a: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "preview": preview}

    session = _opt_session(preview_fn=fake_preview)
    out = await session.execute("preview_regions", {"load_cases": [{"name": "c"}]})
    assert out["ok"], out
    assert out["passive_fraction"] == pytest.approx(0.25)
    assert out["free_fraction"] == pytest.approx(0.75)
    assert out["min_feasible_volfrac"] == pytest.approx(0.25)
    assert out["cases"][0]["n_fixed_cells"] == 100
    assert out["cases"][0]["n_load_cells"] == 30
    assert out["cases"][0]["n_keep_cells"] == 40

    out = await session.execute("check_load", {"load_cases": [{"name": "c"}]})
    assert out["ok"], out
    assert out["passive_fraction"] == pytest.approx(0.25)
    assert out["min_feasible_volfrac"] == pytest.approx(0.25)
    assert out["cases"][0]["n_fixed_cells"] == 100
