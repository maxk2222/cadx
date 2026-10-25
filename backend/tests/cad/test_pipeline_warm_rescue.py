"""Unit tests for the multi-stage structural pipeline pieces:

- ``drop_island_components`` — mesh-level island rescue (P3)
- ``_auto_warm_density`` — provenance-aware warm-start candidates (P0a/P4)
- ``_guide_beso_er`` — adaptive BESO guide ramp (P2b)
- ``_recommend_structure_mode`` — routing hint (P1a)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from cadx.cad.engine import (
    _auto_warm_density,
    _guide_beso_er,
    _npz_kind,
    _recommend_structure_mode,
)
from cadx.cad.fem.extract import drop_island_components


def _sphere(center=(0, 0, 0), r=10.0) -> trimesh.Trimesh:
    s = trimesh.creation.icosphere(subdivisions=2, radius=r)
    s.apply_translation(center)
    return s


# ---------------------------------------------------------------------------
# drop_island_components
# ---------------------------------------------------------------------------


def test_drop_island_removes_small_and_sliver() -> None:
    main = _sphere(r=10.0)  # ~4189 mm³
    island = _sphere(center=(50, 0, 0), r=2.0)  # ~33 mm³ ≈ 0.8%
    sliver = _sphere(center=(80, 0, 0), r=0.05)  # ~5e-4 mm³
    mesh = trimesh.util.concatenate([main, island, sliver])
    out, rep = drop_island_components(mesh, max_island_frac=0.02)
    assert rep["islands_found"] == 2
    assert rep["dropped"] == 2
    assert len(out.split(only_watertight=False)) == 1
    assert rep["dropped_vol_frac"] < 0.01


def test_drop_island_keeps_large_island() -> None:
    main = _sphere(r=10.0)
    island = _sphere(center=(50, 0, 0), r=6.0)  # ~905 mm³ ≈ 21%
    mesh = trimesh.util.concatenate([main, island])
    out, rep = drop_island_components(mesh, max_island_frac=0.02)
    assert rep["dropped"] == 0
    assert rep["islands_found"] == 1
    assert rep["kept_islands"]
    assert len(out.split(only_watertight=False)) == 2


def test_drop_island_pinned_cell_survives() -> None:
    """An island below the fraction cutoff that carries a pinned
    (keep/load/fix) cell must not be dropped — it is declared material."""
    main = _sphere(r=10.0)
    island = _sphere(center=(50, 0, 0), r=2.0)
    mesh = trimesh.util.concatenate([main, island])
    # 1-voxel-per-mm grid pinned around the island's surface — world x in
    # [48,50) maps to indices 98..99 under the (-50,-50,-50) origin
    dims = (120, 120, 120)
    pin3 = np.zeros(dims, dtype=bool)
    pin3[98:100, 49:52, 49:52] = True
    out, rep = drop_island_components(
        mesh,
        max_island_frac=0.02,
        pin_mask=(np.array([-50.0, -50.0, -50.0]), 1.0, pin3),
    )
    assert rep["dropped"] == 0
    assert len(out.split(only_watertight=False)) == 2


def test_drop_island_single_body_noop() -> None:
    mesh = _sphere(r=10.0)
    out, rep = drop_island_components(mesh)
    assert rep["islands_found"] == 0
    assert rep["dropped"] == 0
    assert out is mesh


# ---------------------------------------------------------------------------
# _npz_kind / _auto_warm_density
# ---------------------------------------------------------------------------


def _write_npz(
    path,
    *,
    density,
    dims=(4, 4, 4),
    pitch=1.0,
    origin=(0.0, 0.0, 0.0),
    kind=None,
):
    path = Path(path)
    kw = {
        "density": np.asarray(density, dtype=np.float32),
        "dims": np.asarray(dims),
        "pitch": np.float64(pitch),
        "origin": np.asarray(origin, dtype=float),
    }
    if kind is not None:
        kw["kind"] = np.str_(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kw)


def test_npz_kind_missing_is_legacy() -> None:
    p = None
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = f"{td}/d.npz"
        _write_npz(p, density=np.ones(64))
        with np.load(p) as npz:
            assert _npz_kind(npz) == ""
        _write_npz(p, density=np.ones(64), kind="infill")
        with np.load(p) as npz:
            assert _npz_kind(npz) == "infill"


def test_auto_warm_skips_infill_kind(tmp_path) -> None:
    """An infill lattice field must never seed a SIMP x0 — binary occupancy
    pins hollow cells at the density floor instead of regrowing them."""
    it1 = tmp_path / "1"
    it2 = tmp_path / "2"
    domain = SimpleNamespace(dims=(4, 4, 4), origin=np.zeros(3), pitch=1.0, cell_count=64)
    _write_npz(it1 / "topopt" / "density.npz", density=np.ones(64) * 0.5, kind="infill")
    x0, src = _auto_warm_density(it2, domain)
    assert x0 is None and src is None
    # same field tagged simp IS a valid warm start
    _write_npz(it1 / "topopt" / "density.npz", density=np.ones(64) * 0.5, kind="simp")
    x0, src = _auto_warm_density(it2, domain)
    assert x0 is not None and src is not None and src.endswith("density.npz")
    assert np.allclose(x0, 0.5)
    # legacy untagged file is accepted (pre-provenance runs)
    _write_npz(it1 / "topopt" / "density.npz", density=np.ones(64) * 0.7)
    x0, _src = _auto_warm_density(it2, domain)
    assert x0 is not None and np.allclose(x0, 0.7)


def test_auto_warm_guide_upsample(tmp_path) -> None:
    """A guide_density.npz at ~2x pitch upsamples into the fine grid."""
    it1 = tmp_path / "1"
    it2 = tmp_path / "2"
    fine = SimpleNamespace(dims=(8, 8, 8), origin=np.zeros(3), pitch=1.0, cell_count=512)
    _write_npz(
        it1 / "topopt" / "guide_density.npz",
        density=np.ones(64) * 0.8,
        dims=(4, 4, 4),
        pitch=2.0,
        kind="beso",
    )
    x0, src = _auto_warm_density(it2, fine)
    assert x0 is not None
    assert src is not None and src.startswith("guide@")
    assert x0.shape == (512,)
    assert np.allclose(x0, 0.8)


def test_auto_warm_guide_resamples_near_pitch(tmp_path) -> None:
    """Guide pitch ~1.02x the domain pitch (the infill coarse grid is only
    ~2x the INFILL domain — unrelated to a SIMP domain's pitch) still
    resamples instead of being dropped."""
    it1 = tmp_path / "1"
    it2 = tmp_path / "2"
    fine = SimpleNamespace(dims=(8, 8, 8), origin=np.zeros(3), pitch=1.0, cell_count=512)
    _write_npz(
        it1 / "topopt" / "guide_density.npz",
        density=np.ones(2197) * 0.7,
        dims=(13, 13, 13),
        pitch=1.02,
        kind="beso",
    )
    x0, src = _auto_warm_density(it2, fine)
    assert x0 is not None and src is not None and src.startswith("guide@")
    assert x0.shape == (512,)


def test_auto_warm_same_grid_beats_guide(tmp_path) -> None:
    """A converged same-grid field outranks the coarse guide."""
    it1 = tmp_path / "1"
    it2 = tmp_path / "2"
    fine = SimpleNamespace(dims=(8, 8, 8), origin=np.zeros(3), pitch=1.0, cell_count=512)
    _write_npz(
        it1 / "topopt" / "density.npz",
        density=np.ones(512) * 0.3,
        dims=(8, 8, 8),
        kind="simp",
    )
    _write_npz(
        it1 / "topopt" / "guide_density.npz",
        density=np.ones(64) * 0.9,
        dims=(4, 4, 4),
        pitch=2.0,
    )
    x0, src = _auto_warm_density(it2, fine)
    assert x0 is not None and np.allclose(x0, 0.3)
    assert src is not None and not src.startswith("guide@2x")


# ---------------------------------------------------------------------------
# _guide_beso_er
# ---------------------------------------------------------------------------


def test_guide_beso_er_reaches_target() -> None:
    """The scaled ER must let the ramp arrive at volfrac inside the budget:
    V_iters = (1-ER)^iters >= volfrac, i.e. the ramp is not truncated early."""
    vf, iters = 0.4, 15
    er = _guide_beso_er(vf, iters)
    assert er > 0.02  # the classic step cannot reach 0.4 in 15 iters
    assert (1.0 - er) ** iters <= vf + 1e-9
    # never below the floor
    assert _guide_beso_er(0.95, 1000) == pytest.approx(0.02)
    # edge cases keep the floor
    assert _guide_beso_er(0.0, 10) == pytest.approx(0.02)
    assert _guide_beso_er(0.4, 0) == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# _recommend_structure_mode
# ---------------------------------------------------------------------------


def _report(*, fill: float, sf: float | None) -> SimpleNamespace:
    # pitch 1mm, 100³ grid → bbox 1e6 mm³
    vol = fill * 1_000_000.0
    grid = {"pitch_mm": 1.0, "dims": [100, 100, 100], "volume_mm3": vol}
    cases = [SimpleNamespace(safety_factor=sf)] if sf is not None else []
    return SimpleNamespace(grid=grid, cases=cases)


def test_recommend_thin_walled() -> None:
    mode, reasons = _recommend_structure_mode(
        _report(fill=0.25, sf=2.0), keep_regions=[], anchors=[]
    )
    assert mode == "infill"
    assert reasons


def test_recommend_headroom() -> None:
    mode, _ = _recommend_structure_mode(_report(fill=0.6, sf=4.0), keep_regions=[], anchors=[])
    assert mode == "infill"


def test_recommend_guided_with_keeps() -> None:
    mode, _ = _recommend_structure_mode(
        _report(fill=0.6, sf=2.0), keep_regions=[object()], anchors=[]
    )
    assert mode == "guided"


def test_recommend_topopt_open_freedom() -> None:
    mode, _ = _recommend_structure_mode(_report(fill=0.7, sf=2.0), keep_regions=[], anchors=[])
    assert mode == "topopt"


def test_recommend_marginal_baseline() -> None:
    mode, reasons = _recommend_structure_mode(
        _report(fill=0.25, sf=1.1), keep_regions=[], anchors=[]
    )
    assert mode == "topopt"
    assert "below" in reasons[0]


# ---------------------------------------------------------------------------
# _latest_parent_density — escalation warm-start resolver (P4)
# ---------------------------------------------------------------------------


def test_latest_parent_density_prefers_guide_over_infill(tmp_path, monkeypatch) -> None:
    """Infill → SIMP escalation: the parent's binary rho (kind=infill) is not
    a SIMP seed — its persisted guide_density.npz is picked instead."""
    import cadx.agent.worker_jobs as wj

    vdir = tmp_path / "v1"
    for it in (1, 2):
        _write_npz(
            vdir / str(it) / "topopt" / "density.npz",
            density=np.ones(64) * 0.5,
            kind="infill",
        )
    _write_npz(
        vdir / "2" / "topopt" / "guide_density.npz",
        density=np.ones(64) * 0.8,
        pitch=2.0,
        kind="beso",
    )
    parent = SimpleNamespace(project_id="p", id="v1")
    monkeypatch.setattr(wj.storage, "iteration_dir", lambda p, v, it: vdir / str(it))
    monkeypatch.setattr(wj.storage, "relpath", lambda p: str(p))
    rel = wj._latest_parent_density(parent)
    assert rel is not None and rel.endswith("guide_density.npz")


def test_latest_parent_density_picks_simp_field(tmp_path, monkeypatch) -> None:
    """A converged simp/beso field on the newest iteration wins outright."""
    import cadx.agent.worker_jobs as wj

    vdir = tmp_path / "v2"
    _write_npz(vdir / "1" / "topopt" / "density.npz", density=np.ones(64) * 0.5, kind="simp")
    _write_npz(vdir / "2" / "topopt" / "density.npz", density=np.ones(64) * 0.9, kind="beso")
    _write_npz(
        vdir / "2" / "topopt" / "guide_density.npz",
        density=np.ones(64) * 0.8,
        pitch=2.0,
        kind="beso",
    )
    parent = SimpleNamespace(project_id="p", id="v2")
    monkeypatch.setattr(wj.storage, "iteration_dir", lambda p, v, it: vdir / str(it))
    monkeypatch.setattr(wj.storage, "relpath", lambda p: str(p))
    rel = wj._latest_parent_density(parent)
    assert rel is not None and rel.endswith("2/topopt/density.npz")


def test_latest_parent_density_none_when_empty(tmp_path, monkeypatch) -> None:
    import cadx.agent.worker_jobs as wj

    vdir = tmp_path / "v3"
    vdir.mkdir()
    parent = SimpleNamespace(project_id="p", id="v3")
    monkeypatch.setattr(wj.storage, "iteration_dir", lambda p, v, it: vdir / str(it))
    assert wj._latest_parent_density(parent) is None
