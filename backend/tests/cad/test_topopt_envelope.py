"""TopOpt input normalization to the spec envelope + compact history (Q1/S1).

Q1 — a base mesh overshooting the declared spec dims (>2 mm or >5% on a
constrained axis) is clipped to the spec envelope before voxelization: the
voxel FEM domain IS the mesh bbox, so an oversize input silently shifted the
whole FEM problem out of envelope. The normalized STL lands in
``out_dir/_run/topopt_input.stl`` and ``res.topopt`` carries
base_mesh_relpath / input_bbox / spec_bbox / input_normalized / domain_bbox.

S1 — ``res.topopt["history"]`` is the compact frontend chart series
({it, compliance, volfrac, change, solve_s}, ≤200 entries) and every simp()
history record carries per-phase timings (assemble_s/factorize_s/solve_s/
filter_s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import trimesh

from cadx.cad.engine import normalize_mesh_to_spec_envelope
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec
from cadx.runtime_config import RuntimeSettings

VOXEL_MM = 1.5


def _box(x: float, y: float, z: float) -> trimesh.Trimesh:
    """Contract frame: XY centred on the origin, resting on the XY plane."""
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


def _cantilever_case() -> dict[str, Any]:
    return _cantilever().model_dump()


def _pipeline(src: Path, out_dir: Path, **opt: Any):
    from cadx.cad.engine import run_topopt_pipeline

    return run_topopt_pipeline(
        mesh_path=src,
        out_dir=out_dir,
        load_cases=[_cantilever_case()],
        optimize={
            "target_volume_fraction": 0.5,
            "max_iterations": 4,
            "keep_mode": "manual",
            **opt,
        },
        material="pla",
        voxel_mm=VOXEL_MM,
        scan_anchors=False,
        runtime=RuntimeSettings(),
    )


def test_clip_mesh_to_spec_envelope() -> None:
    """Helper level: overshoot on X triggers a boolean clip; all constrained
    axes end up inside the declared envelope and the report is populated."""
    mesh = _box(94.9, 35.7, 16.4)  # observed bug: 90x40x14 spec
    out, rep = normalize_mesh_to_spec_envelope(mesh, [90.0, 40.0, 14.0])
    assert rep["input_normalized"] is True
    assert rep["method"] == "clip"
    assert rep["input_bbox"] == pytest.approx([94.9, 35.7, 16.4], abs=0.05)
    assert rep["spec_bbox"] == [90.0, 40.0, 14.0]
    ext = out.bounds[1] - out.bounds[0]
    assert ext[0] <= 90.0 + 1e-3
    assert ext[2] <= 14.0 + 1e-3
    # the resting base survives: clip keeps z>=0, XY centred at origin
    assert out.bounds[0][2] == pytest.approx(0.0, abs=1e-3)
    assert out.is_volume


def test_no_clip_within_slack() -> None:
    """Sub-tolerance overshoot (<2 mm and <5%) and unconstrained axes leave
    the input untouched."""
    mesh = _box(21.0, 12.0, 8.0)
    out, rep = normalize_mesh_to_spec_envelope(mesh, [20.0, 12.0, 8.0])
    assert out is mesh
    assert rep["input_normalized"] is False
    assert rep["spec_bbox"] == [20.0, 12.0, 8.0]

    # X overshoots but is unconstrained (None) -> no clip
    out, rep = normalize_mesh_to_spec_envelope(mesh, [None, 12.0, 8.0])
    assert out is mesh
    assert rep["input_normalized"] is False


def test_pipeline_normalizes_oversize_input(tmp_path: Path) -> None:
    """Q1 end-to-end: 24x12x8 mesh vs a stated 20x12x8 spec is clipped before
    voxelize; res.topopt carries the normalization report + domain_bbox."""
    src = tmp_path / "b.stl"
    _box(24.0, 12.0, 8.0).export(str(src))
    res = _pipeline(src, tmp_path / "o", spec_dims_mm=[20.0, 12.0, 8.0])
    assert res.ok, res.error
    top = res.topopt
    assert top["input_normalized"] is True
    assert top["input_bbox"] == pytest.approx([24.0, 12.0, 8.0], abs=0.05)
    assert top["spec_bbox"] == [20.0, 12.0, 8.0]
    assert top["base_mesh_relpath"]
    norm_stl = tmp_path / "o" / "_run" / "topopt_input.stl"
    assert norm_stl.exists()
    # the voxel domain must now fit inside the spec envelope (+1 cell of
    # stair-stepping slack) — before the fix it tracked the raw 24 mm bbox
    lo, hi = top["domain_bbox"]
    assert hi[0] - lo[0] <= 20.0 + VOXEL_MM
    assert hi[2] - lo[2] <= 8.0 + VOXEL_MM


def test_pipeline_skips_normalization_without_spec(tmp_path: Path) -> None:
    """Mesh-only runs (no spec dims) normalize nothing — the keys exist but
    spec_bbox stays None and input_normalized is False."""
    src = tmp_path / "b.stl"
    _box(24.0, 12.0, 8.0).export(str(src))
    res = _pipeline(src, tmp_path / "o")
    assert res.ok, res.error
    assert res.topopt["input_normalized"] is False
    assert res.topopt["spec_bbox"] is None
    assert res.topopt["input_bbox"] == pytest.approx([24.0, 12.0, 8.0], abs=0.05)
    assert not (tmp_path / "o" / "_run" / "topopt_input.stl").exists()


def test_pipeline_history_contract(tmp_path: Path) -> None:
    """S1: res.topopt['history'] is the compact chart series; the raw
    progress-file entries carry the per-phase timing fields."""
    src = tmp_path / "b.stl"
    _box(24.0, 12.0, 8.0).export(str(src))
    res = _pipeline(src, tmp_path / "o")
    assert res.ok, res.error
    hist = res.topopt["history"]
    assert 0 < len(hist) <= 200
    for rec in hist:
        assert set(rec) == {"it", "compliance", "volfrac", "change", "solve_s"}
    assert hist[-1]["it"] == res.topopt["iterations"]
    # compact series is also persisted where topopt['history_relpath'] points
    hist_file = tmp_path / "o" / "topopt" / "history.json"
    assert hist_file.exists()


def test_simp_history_phase_timings() -> None:
    """Every OC history record carries assemble_s/factorize_s/solve_s/
    filter_s — measured around the real calls, non-negative."""
    from cadx.cad.fem.regions import resolve_case
    from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
    from cadx.cad.fem.voxelize import voxelize

    dom = voxelize(_box(20.0, 8.0, 8.0), 1.0, 200_000)
    fixed, F, _ = resolve_case(dom, _cantilever(), [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    res = simp(
        dom,
        3500.0,
        0.36,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=4),
    )
    assert res.iterations > 0
    for rec in res.history:
        assert rec["assemble_s"] >= 0.0
        assert rec["factorize_s"] >= 0.0
        assert rec["solve_s"] >= 0.0
        assert rec["filter_s"] >= 0.0
    # a real FEM iteration must have spent some time in assemble+solve
    assert sum(r["assemble_s"] + r["solve_s"] for r in res.history) > 0.0


def test_move_limit_oscillation_warning() -> None:
    """A run whose OC update stays saturated at the move cap for 15+
    consecutive iterations is a limit cycle — it must surface a warning so
    the caller can react (symmetry plane, different filter/threshold)."""
    from cadx.cad.fem.regions import resolve_case
    from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
    from cadx.cad.fem.voxelize import voxelize

    dom = voxelize(_box(20.0, 8.0, 8.0), 1.0, 200_000)
    fixed, F, _ = resolve_case(dom, _cantilever(), [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    res = simp(
        dom,
        3500.0,
        0.36,
        [(fixed, F)],
        passive,
        # move far below any real density step: every iteration where the
        # field still evolves reports change == move — a saturated streak;
        # tol=0 + high patience keep the run alive to the iteration cap
        TopOptParams(
            volfrac=0.4,
            rmin_mm=2.0,
            max_iter=20,
            move=0.001,
            tol=0.0,
            converge_patience=50,
        ),
    )
    assert res.iterations == 20
    assert any("limit cycle" in w for w in res.warnings)


def test_limit_cycle_break_at_beta_cap() -> None:
    """Once the Heaviside continuation sits at beta_max and the density still
    saturates the move cap, no mechanism remains to settle the field — the
    loop must break and extract rather than burn solver minutes to the cap."""
    from cadx.cad.fem.regions import resolve_case
    from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
    from cadx.cad.fem.voxelize import voxelize

    dom = voxelize(_box(20.0, 8.0, 8.0), 1.0, 200_000)
    fixed, F, _ = resolve_case(dom, _cantilever(), [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    res = simp(
        dom,
        3500.0,
        0.36,
        [(fixed, F)],
        passive,
        # beta already at the cap + a move far below any real step: the
        # streak counter hits 15 with no settle mechanism left — break
        TopOptParams(
            volfrac=0.4,
            rmin_mm=2.0,
            max_iter=30,
            move=0.001,
            tol=0.0,
            converge_patience=50,
            filter="heaviside",
            beta0=8.0,
            beta_max=8.0,
        ),
    )
    assert res.iterations <= 20
    assert any("persisted past the beta cap" in w for w in res.warnings)


def test_multires_default_on_with_opt_out(tmp_path: Path) -> None:
    """S1: multires runs by default once the domain clears
    topopt_multires_min_cells; an explicit multires=false opts out."""
    from cadx.cad.engine import run_topopt_pipeline

    src = tmp_path / "b.stl"
    _box(24.0, 24.0, 24.0).export(str(src))  # 16^3 = 4096 cells > 4000 gate
    kw: dict[str, Any] = {
        "mesh_path": src,
        "load_cases": [_cantilever_case()],
        "optimize": {"target_volume_fraction": 0.5, "max_iterations": 3},
        "material": "pla",
        "voxel_mm": VOXEL_MM,
        "scan_anchors": False,
        "runtime": RuntimeSettings(),
    }
    res_on = run_topopt_pipeline(out_dir=tmp_path / "on", **kw)
    assert res_on.ok, res_on.error
    assert res_on.topopt.get("multires"), "default-on multires did not engage"

    res_off = run_topopt_pipeline(
        out_dir=tmp_path / "off", **{**kw, "optimize": {**kw["optimize"], "multires": False}}
    )
    assert res_off.ok, res_off.error
    assert "multires" not in res_off.topopt
