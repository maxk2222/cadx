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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import trimesh

import cadx.cad.fem.topopt as topopt_mod
from cadx.cad.fem.regions import resolve_case
from cadx.cad.fem.topopt import (
    TopOptParams,
    _same_flip_run,
    _span_compliance_improvement,
    _windowed_compliance_improvement,
    simp,
)
from cadx.cad.fem.voxelize import VoxelDomain, voxelize
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
    above the target — unreachable by construction, so the run refuses up
    front (0 iterations) rather than draining every free cell first."""
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
    assert res.iterations == 0
    assert res.passive_frac == pytest.approx(0.62, abs=0.01)
    assert any("passive" in w and "floor" in w for w in res.warnings)


def test_simp_can_pin_detected_sealed_void_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    dims = (8, 4, 4)
    dom = VoxelDomain(
        origin=np.zeros(3),
        pitch=1.0,
        dims=dims,
        solid=np.ones(dims, dtype=bool),
    )
    nodes = np.arange(dom.node_count).reshape(tuple(n + 1 for n in dims))
    fixed = np.zeros(dom.node_count, dtype=bool)
    fixed[nodes[0, :, :].ravel()] = True
    force = np.zeros(3 * dom.node_count)
    load_node = int(nodes[-1, dims[1] // 2, dims[2] // 2])
    force[3 * load_node + 2] = -10.0
    pocket = np.zeros(dims, dtype=bool)
    pocket[4, 2, 2] = True
    monkeypatch.setattr(topopt_mod, "_sealed_pocket_mask", lambda _mask: pocket.copy())

    result = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, force)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(
            volfrac=0.6,
            rmin_mm=2.0,
            max_iter=8,
            prevent_sealed_voids=True,
            sealed_void_check_iters=1,
        ),
    )
    assert result.void_pinned_cells_total == 1
    assert result.void_pin_iters == 1
    assert result.void_pinning_applied is True
    assert any("pinned 1 cells" in warning for warning in result.warnings)


def test_simp_void_pins_mirror_across_symmetry_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    """An asymmetric detected cavity must pin BOTH sides of the symmetry
    plane — pinning only the detected half would break the very symmetry
    the run is asked to hold."""
    dims = (8, 4, 4)
    dom = VoxelDomain(
        origin=np.zeros(3),
        pitch=1.0,
        dims=dims,
        solid=np.ones(dims, dtype=bool),
    )
    nodes = np.arange(dom.node_count).reshape(tuple(n + 1 for n in dims))
    fixed = np.zeros(dom.node_count, dtype=bool)
    fixed[nodes[0, :, :].ravel()] = True
    force = np.zeros(3 * dom.node_count)
    load_node = int(nodes[-1, dims[1] // 2, dims[2] // 2])
    force[3 * load_node + 2] = -10.0
    pocket = np.zeros(dims, dtype=bool)
    pocket[4, 2, 2] = True
    monkeypatch.setattr(topopt_mod, "_sealed_pocket_mask", lambda _mask: pocket.copy())

    result = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, force)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(
            volfrac=0.6,
            rmin_mm=2.0,
            max_iter=8,
            prevent_sealed_voids=True,
            sealed_void_check_iters=1,
            symmetry="yz",  # flips axis 0: cell x=4 mirrors to x=3
        ),
    )
    assert result.void_pinned_cells_total == 2
    assert result.void_pin_sym_mirrored == 1


def test_simp_mma_void_prevention_pins_like_oc(monkeypatch: pytest.MonkeyPatch) -> None:
    """prevent_sealed_voids now runs inside the MMA loop too: sealed pockets
    detected on the thresholded physical density (xb never drops below
    RHO_MIN, so a bool cast would see all-solid) get pinned via the shared
    passive mask — _unpack re-clamps them, the nlopt variable vector is
    never re-indexed."""
    dims = (8, 4, 4)
    dom = VoxelDomain(
        origin=np.zeros(3),
        pitch=1.0,
        dims=dims,
        solid=np.ones(dims, dtype=bool),
    )
    nodes = np.arange(dom.node_count).reshape(tuple(n + 1 for n in dims))
    fixed = np.zeros(dom.node_count, dtype=bool)
    fixed[nodes[0, :, :].ravel()] = True
    force = np.zeros(3 * dom.node_count)
    force[3 * int(nodes[-1, dims[1] // 2, dims[2] // 2]) + 2] = -10.0
    pocket = np.zeros(dims, dtype=bool)
    pocket[4, 2, 2] = True
    monkeypatch.setattr(topopt_mod, "_sealed_pocket_mask", lambda _mask: pocket.copy())

    result = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, force)],
        np.zeros(dom.cell_count, dtype=bool),
        TopOptParams(
            volfrac=0.6,
            rmin_mm=2.0,
            max_iter=6,
            solver="mma",
            prevent_sealed_voids=True,
            sealed_void_check_iters=1,
        ),
    )
    assert result.void_pinning_applied is True
    assert result.void_pinned_cells_total >= 1
    assert result.void_pin_iters >= 1
    assert any("pinned" in w for w in result.warnings)


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


def test_status_infeasible_target_at_boundary() -> None:
    """floor == target is degenerate but NOT refused (strict '>'): the free
    cells can still choose — only a floor strictly above volfrac is
    unreachable."""
    dom, fixed, F = _case()
    passive = np.zeros(dom.cell_count, dtype=bool)
    sidx = np.flatnonzero(dom.solid.ravel())
    n_pass = int(0.40 * sidx.size)
    passive[sidx[:n_pass]] = True
    floor = n_pass / sidx.size
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=floor + 0.001, rmin_mm=2.0, max_iter=6),
    )
    assert res.status != "infeasible_target" or res.iterations > 0


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
    hist = [{"compliance_mj": 50.0, "beta": 1.0}] * 5 + [{"compliance_mj": 80.0, "beta": 2.0}] * 7
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
    from cadx.cad.engine import _export_degeneracy, _realization_case_details, _sf_basis

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

    # The realization audit distinguishes a mechanism verdict from a
    # conventional stress-limited safety factor without changing the SF.
    assert (
        _sf_basis(
            {"nominal": 0.01, "eroded": 0.2},
            {
                "nominal": [{"sf": 0.01, "degenerate": True}],
                "eroded": [{"sf": 0.2, "degenerate": False}],
            },
            0.01,
            True,
        )
        == "mechanism"
    )
    assert _sf_basis({"nominal": 0.01}, {}, 0.01, False) == "loadpath"
    assert (
        _sf_basis(
            {"nominal": 1.5},
            {"nominal": [{"sf": 1.5, "degenerate": False}]},
            1.5,
            True,
        )
        == "stress"
    )
    report = SimpleNamespace(
        cases=[
            SimpleNamespace(
                name="overturn",
                safety_factor=0.01,
                max_disp_mm=2204.0,
                max_vm_mpa=157.0,
                converged=True,
                warnings=[],
            )
        ]
    )
    detail = _realization_case_details(report, span_mm=100.0, yield_mpa=250.0)
    assert detail[0]["sf"] == 0.01
    assert detail[0]["degenerate"] is True
    assert any("degenerate" in warning for warning in detail[0]["warnings"])


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
        keep_regions=[RegionSpec(kind="box", center=[0.0, 0.0, 3.0], diameter_mm=6.0).model_dump()],
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
    assert (
        prev["passive_fraction"]
        >= (case["n_fixed_cells"] + case["n_keep_cells"]) / prev["solid_elements"] * 0.9
    )


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


# ---------------------------------------------------------------------------
# N5 — period-1 limit cycle below the churn threshold
# ---------------------------------------------------------------------------


def test_same_flip_run_counts_identical_tail() -> None:
    """Identical nonzero flips > _SAME_FLIP_FLOOR count up from the tail;
    a settling tail (few cells) and zero-flip records never count."""
    hist = [{"flips": v} for v in (0.01, 0.00245, 0.00245, 0.00245, 0.00245)]
    assert _same_flip_run(hist) == 4  # the eval pattern: fires at the 4th repeat
    assert _same_flip_run([{"flips": 0.00028}] * 5) == 0  # settling-tail flicker
    assert _same_flip_run([{"flips": 0.00245}] * 3 + [{"flips": 0.004}]) == 1
    assert _same_flip_run([{"flips": 0.0}] * 4) == 0
    assert _same_flip_run([]) == 0
    assert _same_flip_run([{"flips": 0.00245}]) == 1


def test_span_compliance_improvement() -> None:
    """Span-local relative improvement — works on segments too short for
    the same-beta window (the capped-beta eval case)."""
    hist = [{"compliance_mj": 100.0}] * 3 + [{"compliance_mj": 99.0}]
    assert _span_compliance_improvement(hist, 4) == pytest.approx(0.01)
    # oscillating ~0 -> clearly stalled
    hist = [{"compliance_mj": v} for v in (100.0, 99.98, 100.01, 99.99)]
    imp = _span_compliance_improvement(hist, 4)
    assert imp is not None and imp < 0.01
    assert _span_compliance_improvement([{"compliance_mj": 100.0}] * 2, 4) is None
    assert _span_compliance_improvement([{"compliance_mj": 0.0}] * 5, 5) is None


# ---------------------------------------------------------------------------
# N3 — passive-floor preflight refuse
# ---------------------------------------------------------------------------


def test_pipeline_refuses_floor_above_target(tmp_path: Path) -> None:
    """keep_regions covering the whole part make the requested volfrac
    unreachable — the pipeline refuses before any SIMP iteration instead
    of burning a run (the eval's 0.62-floor vs 0.40-target case)."""
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box_mesh(24.0, 24.0, 24.0).export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever().model_dump()],
        optimize={
            "target_volume_fraction": 0.4,
            "keep_regions": [{"kind": "all"}],
            "keep_mode": "manual",
            "max_iterations": 4,
        },
        material="pla",
        voxel_mm=1.5,
        scan_anchors=False,
        runtime=RuntimeSettings(topopt_auto_keep=False),
    )
    assert not res.ok
    assert res.error and "unreachable" in res.error
    assert res.topopt["infeasible_preflight"] is True
    assert res.topopt["passive_frac"] > 0.9
    assert res.topopt["min_feasible_volfrac"] == res.topopt["passive_frac"]


def test_pipeline_warns_high_floor_when_feasible(tmp_path: Path) -> None:
    """A floor that still fits under the target is feasible — the run
    proceeds (no preflight refuse) and exposes the floor metrics."""
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box_mesh(24.0, 24.0, 24.0).export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever().model_dump()],
        optimize={
            "target_volume_fraction": 0.6,
            "keep_regions": [{"kind": "box", "center": [0.0, 0.0, 12.0], "diameter_mm": 12.0}],
            "keep_mode": "manual",
            "max_iterations": 4,
        },
        material="pla",
        voxel_mm=1.5,
        scan_anchors=False,
        runtime=RuntimeSettings(topopt_auto_keep=False),
    )
    assert res.topopt.get("infeasible_preflight") is not True
    assert 0.0 < res.topopt["passive_frac"] <= 0.6


def test_grow_preflight_uses_design_domain_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """grow_mm enlarges the volume denominator; the passive-floor preflight
    must use the same design domain as SIMP rather than the input mesh."""
    from cadx.cad.engine import run_topopt_pipeline

    def quarter_passive(domain, extra, fixed, loads):
        del extra, fixed, loads
        cells = np.flatnonzero(domain.solid.ravel())
        mask = np.zeros(domain.cell_count, dtype=bool)
        mask[cells[: round(0.25 * len(cells))]] = True
        return mask

    monkeypatch.setattr(topopt_mod, "passive_solid_mask", quarter_passive)
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "grow",
        load_cases=[_cantilever().model_dump()],
        optimize={
            "target_volume_fraction": 0.3,
            "grow_mm": 3.0,
            "max_iterations": 1,
        },
        material="pla",
        voxel_mm=1.0,
        scan_anchors=False,
        runtime=RuntimeSettings(topopt_multires_min_cells=10_000_000),
    )
    assert res.topopt.get("infeasible_preflight") is not True
    assert res.topopt["passive_frac"] == pytest.approx(0.25)


async def test_topopt_run_infeasible_preflight_refunds_budget() -> None:
    """A preflight-refused run never executed SIMP — the call budget is
    refunded and the floor fields reach the agent for re-targeting."""

    async def fake_topopt(a: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": False,
            "error": "passive keep/load/fixed cells cover 62% — unreachable",
            "topopt": {
                "infeasible_preflight": True,
                "passive_frac": 0.62,
                "min_feasible_volfrac": 0.62,
            },
        }

    session = _opt_session(topopt_fn=fake_topopt)
    before = session.topopt_left
    out = await session.execute("topopt_run", {"load_cases": [{"name": "c"}]})
    assert out["ok"] is False
    assert out["infeasible_target"] is True
    assert out["min_feasible_volfrac"] == pytest.approx(0.62)
    assert out["passive_fraction"] == pytest.approx(0.62)
    assert session.topopt_left == before  # refunded — the run never ran


async def test_preview_and_check_load_warn_floor_above_target() -> None:
    """With a session target below the keep/load/fix floor, both free
    preview tools flag infeasible_target before a run is wasted."""
    preview = {
        "pitch_mm": 1.5,
        "solid_elements": 4000,
        "n_keep_cells": 2500,
        "passive_fraction": 0.62,
        "min_feasible_volfrac": 0.62,
        "cases": [{"case": "c", "n_fixed_cells": 100, "n_load_cells": 30}],
    }

    async def fake_preview(a: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "preview": preview}

    session = _opt_session(preview_fn=fake_preview, target_volfrac=0.4)
    out = await session.execute("preview_regions", {"load_cases": [{"name": "c"}]})
    assert out["ok"], out
    assert out["infeasible_target"] is True
    assert out["suggested_min_target"] == pytest.approx(0.64)
    assert "floor" in out["warning"]

    out = await session.execute("check_load", {"load_cases": [{"name": "c"}]})
    assert out["infeasible_target"] is True

    # same preview without a session target carries no verdict
    session = _opt_session(preview_fn=fake_preview)
    out = await session.execute("preview_regions", {"load_cases": [{"name": "c"}]})
    assert "infeasible_target" not in out

    # and a floor under the target stays silent
    session = _opt_session(preview_fn=fake_preview, target_volfrac=0.8)
    out = await session.execute("preview_regions", {"load_cases": [{"name": "c"}]})
    assert "infeasible_target" not in out


# ---------------------------------------------------------------------------
# early load-path certificate: provably disconnected exports skip the sweep
# ---------------------------------------------------------------------------


def test_early_loadpath_cert_skips_fem_on_disconnected_export(
    tmp_path: Path,
) -> None:
    """Two disjoint bodies: the load sits on one, the fixture on the other.
    The min-cut certificate proves the path broken before any FEM solve —
    _verify_export_structure must return load_path_ok=False with an empty
    report instead of burning the realization sweep on a doomed part."""
    from cadx.cad.engine import _verify_export_structure
    from cadx.cad.fem.materials import resolve_material

    a = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    a.apply_translation([-8.0, 0.0, 5.0])
    b = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    b.apply_translation([8.0, 0.0, 5.0])
    candidate = trimesh.util.concatenate([a, b])
    case = LoadCase(
        name="broken",
        fixed=[RegionSpec(kind="cylinder", axis="z", center=[8.0, 0.0, 5.0], diameter_mm=4.0)],
        loads=[
            LoadSpec(
                region=RegionSpec(
                    kind="cylinder", axis="z", center=[-8.0, 0.0, 5.0], diameter_mm=4.0
                ),
                force_n=[0.0, 0.0, -50.0],
            )
        ],
    )
    s = RuntimeSettings(fem_loadpath_cert=True)
    info = _verify_export_structure(
        candidate_mesh=candidate,
        cases=[case],
        mat_name="pla",
        mat=resolve_material("pla"),
        design_pitch=2.0,
        max_elements=200_000,
        keep_list=[],
        anchors=[],
        out_dir=tmp_path,
        suffix="_t",
        s=s,
        solver_backend="auto",
        cancelled=None,
        verify_robust=True,
    )
    assert info["load_path_ok"] is False
    assert info["sf_verified"] is None
    assert info["n_bodies_export"] == 2
    assert info["n_bodies_voxel"] == 2
    # no FEM case ran — the sweep is skipped on a provably broken part
    assert info["report"].cases == []
    assert any("cannot reach" in w or "skipped" in w for w in info["warnings"])


def test_realization_detail_records_per_case_audit(tmp_path: Path) -> None:
    """sf_verified alone cannot say WHICH case collapsed on the eroded
    realization — the guided-infill failure showed eroded=0.27 with no
    audit trail. The detail map must carry per-case sf/disp/warnings for
    every solved realization."""
    from cadx.cad.engine import _verify_export_structure
    from cadx.cad.fem.materials import resolve_material

    candidate = _box_mesh(24.0, 12.0, 12.0)
    s = RuntimeSettings(fem_spsolve_dofs=60_000)
    info = _verify_export_structure(
        candidate_mesh=candidate,
        cases=[_cantilever()],
        mat_name="pla",
        mat=resolve_material("pla"),
        design_pitch=3.0,
        max_elements=200_000,
        keep_list=[],
        anchors=[],
        out_dir=tmp_path,
        suffix="_t",
        s=s,
        solver_backend="auto",
        cancelled=None,
        verify_robust=True,
    )
    detail = info["sf_realization_detail"]
    assert set(info["sf_realizations"]) == set(detail)
    for _name, cases in detail.items():
        assert cases and all("case" in c and "sf" in c for c in cases)
        assert all("disp_mm" in c and "vm_mpa" in c and "warnings" in c for c in cases)


def test_early_loadpath_cert_runs_fem_on_connected_export(tmp_path: Path) -> None:
    """A connected single body passes the certificate and the nominal solve
    still runs — the early check must not disable verification wholesale."""
    from cadx.cad.engine import _verify_export_structure
    from cadx.cad.fem.materials import resolve_material

    candidate = _box_mesh(24.0, 12.0, 12.0)
    s = RuntimeSettings(fem_loadpath_cert=True, fem_spsolve_dofs=60_000)
    info = _verify_export_structure(
        candidate_mesh=candidate,
        cases=[_cantilever()],
        mat_name="pla",
        mat=resolve_material("pla"),
        design_pitch=3.0,
        max_elements=200_000,
        keep_list=[],
        anchors=[],
        out_dir=tmp_path,
        suffix="_t",
        s=s,
        solver_backend="auto",
        cancelled=None,
        verify_robust=False,
    )
    assert info["load_path_ok"] is not False
    assert info["report"].cases, "nominal solve must run on a connected part"


# ---------------------------------------------------------------------------
# Feature floor — anchored bores cap the design-grid pitch (M1)
# ---------------------------------------------------------------------------


def test_feature_floor_pitch_helper() -> None:
    from cadx.cad.engine import _feature_floor_pitch

    assert _feature_floor_pitch(None) is None
    assert _feature_floor_pitch([]) is None
    assert _feature_floor_pitch([{"dia": 0.0}, {}]) is None
    assert _feature_floor_pitch([{"dia": 6.5}, {"dia": 13.0}]) == pytest.approx(6.5 / 3.5)


def test_snap_pitch_under_floor_clamps_round_down_overshoot() -> None:
    """snapped_pitch can round the thin-axis cell count DOWN and land above
    the floor — the clamp must return a pitch <= floor either way."""
    from cadx.cad.engine import _snap_pitch_under_floor

    # span_min 7.2 / floor 3.0 -> n=round(2.4)=2 -> p_snap 3.6 > floor
    p, _warn = _snap_pitch_under_floor(
        np.asarray([7.2, 40.0, 40.0]), 3.0, min_cells=1, mesh_volume=0.0, max_elements=0
    )
    assert p <= 3.0


def test_topopt_pipeline_clamps_pitch_to_feature_floor(tmp_path: Path) -> None:
    """A requested coarse pitch must be clamped to min_anchor_dia/3.5 so
    anchored bores still span ~3.5 cells on the design grid."""
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box_mesh(24.0, 24.0, 24.0).export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever().model_dump()],
        optimize={
            "target_volume_fraction": 0.5,
            "max_iterations": 2,
            "safety_factor_min": 1.5,
            "keep_mode": "manual",
        },
        material="pla",
        voxel_mm=4.0,  # coarse request: snapped 4.0 >> floor 1.857
        anchors=[{"x": 0.0, "y": 0.0, "z": 12.0, "axis": "+z", "dia": 6.5}],
        scan_anchors=False,
        runtime=RuntimeSettings(),
    )
    assert res.ok, res.error
    pitch = res.topopt["domain_pitch"]
    assert pitch["feature_floor_mm"] == pytest.approx(6.5 / 3.5, abs=1e-3)
    assert pitch["feature_floor_clamped"] is True
    assert pitch["design_mm"] <= 6.5 / 3.5 + 1e-6


def test_topopt_coarsen_holds_feature_floor_over_dof_cap(tmp_path: Path) -> None:
    """The DOF cap must NOT coarsen past the feature floor — the grid holds
    at min_dia/3.5 and reports feature_floor_hit instead of silently
    under-resolving the anchored bore (the mount kept-base regression)."""
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box_mesh(24.0, 12.0, 12.0).export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever().model_dump()],
        optimize={
            "target_volume_fraction": 0.5,
            "max_iterations": 1,
            "safety_factor_min": 1.5,
            "keep_mode": "manual",
        },
        material="pla",
        voxel_mm=1.0,
        anchors=[{"x": 0.0, "y": 0.0, "z": 6.0, "axis": "+z", "dia": 6.5}],
        scan_anchors=False,
        runtime=RuntimeSettings(fem_spsolve_dofs=1_000),  # tiny cap forces coarsening
    )
    assert res.ok, res.error
    ac = res.topopt["auto_coarsen"]
    assert ac["feature_floor_hit"] is True
    assert ac["feature_floor_mm"] == pytest.approx(6.5 / 3.5, abs=1e-3)
    assert ac["design_pitch_mm"] == pytest.approx(6.5 / 3.5, abs=1e-3)
    # DOFs stay above the cap — surfaced via flag+warning, not silently coarsened
    assert ac["dofs"] > 1_000
    assert any("feature floor" in w for w in res.topopt["warnings"])


def test_infill_pipeline_clamps_pitch_to_feature_floor(tmp_path: Path) -> None:
    """run_infill_pipeline shares the same snapped_pitch exposure — the same
    floor clamp must apply there (no coarsen loop, single solve)."""
    from cadx.cad.engine import run_infill_pipeline

    src = tmp_path / "b.stl"
    _box_mesh(24.0, 24.0, 24.0).export(str(src))
    res = run_infill_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever().model_dump()],
        optimize={
            "target_volume_fraction": 0.5,
            "safety_factor_min": 1.5,
            "keep_mode": "manual",
        },
        material="pla",
        voxel_mm=4.0,
        anchors=[{"x": 0.0, "y": 0.0, "z": 12.0, "axis": "+z", "dia": 6.5}],
        runtime=RuntimeSettings(),
    )
    assert res.ok, res.error
    pitch = res.topopt["domain_pitch"]
    assert pitch["feature_floor_clamped"] is True
    assert pitch["design_mm"] <= 6.5 / 3.5 + 1e-6


async def test_topopt_run_tool_surfaces_recovery_hints() -> None:
    """recovery_hints translate the run status into next-run knobs — a
    floor-hit + limit_cycle run must name both remedies."""

    async def fake_topopt(a: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "topopt": {
                "status": "limit_cycle",
                "converged": False,
                "passive_frac": 0.3,
                "volfrac_achieved": 0.5,
                "iterations": 40,
                "sf_export": 0.2,
                "auto_coarsen": {"feature_floor_hit": True, "feature_floor_mm": 1.86},
                "verification": {"valid": False, "meets_safety_factor": False},
                "warnings": ["mask still churning at beta cap"],
            },
            "fem": {"cases": [{"safety_factor": 0.2}]},
        }

    session = _opt_session(topopt_fn=fake_topopt)
    out = await session.execute("topopt_run", {"load_cases": [{"name": "c"}]})
    hints = out.get("recovery_hints") or []
    assert any("feature floor" in h for h in hints)
    assert any("limit_cycle" in h for h in hints)
