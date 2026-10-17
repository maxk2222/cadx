"""End-to-end regression coverage for the topopt pipeline audit fixes in
`cad/engine.py`.

T1 — the runtime `topopt_auto_keep` flag (default on) now fires auto
     keep-regions when keep_mode="manual" (previously a dead setting) and the
     run reports `topopt["auto_keep_applied"]`.
T4 — topopt-family exports attempt a faceted STEP via mesh_to_solid sew.
T6 — FEM verification renders are pass-indexed so a volfrac retry no longer
     overwrites pass-1 evidence in out_dir/fem.
T7 — res.topopt["verification"] carries the extra telemetry fields.
T8 — a fixed/load region that resolves to zero nodes is a hard error:
     optimizing under constraints that are absent from the geometry produces
     a meaningless result (previously it was a buried warning while the run
     shipped a blob).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.regions import region_cell_mask
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec
from cadx.runtime_config import RuntimeSettings

VOXEL_MM = 1.5  # 24 mm cube -> 16^3 voxel domain — small enough for a fast SIMP

# synthetic fastener anchor in the part centre (bore axis +z, dia 4 mm)
_ANCHOR: dict[str, Any] = {
    "index": 1,
    "kind": "hole",
    "axis": "+z",
    "x": 0.0,
    "y": 0.0,
    "dia": 4.0,
}
# auto_keep_regions() resolves the margin from opt.auto_keep_margin_mm or
# RuntimeSettings.topopt_auto_keep_margin_mm — 3.0 in all runs below
_KEEP_MARGIN_MM = 3.0


def _box_mesh(size: float = 24.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[size, size, size])
    m.apply_translation([0.0, 0.0, size / 2])  # contract: rests on XY, centred
    return m


def _cantilever_case() -> dict[str, Any]:
    return LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -50.0],
            )
        ],
    ).model_dump()


def _run(src: Path, out_dir: Path, rt: RuntimeSettings, **opt: Any):
    from cadx.cad.engine import run_topopt_pipeline

    return run_topopt_pipeline(
        mesh_path=src,
        out_dir=out_dir,
        load_cases=[_cantilever_case()],
        optimize={
            "target_volume_fraction": 0.5,
            "max_iterations": 6,
            "keep_mode": "manual",
            **opt,
        },
        material="pla",
        voxel_mm=VOXEL_MM,
        anchors=[dict(_ANCHOR)],
        scan_anchors=False,  # keep the synthetic anchor table as declared
        runtime=rt,
    )


def _anchor_ring() -> np.ndarray:
    """Cells the auto-keep hole region pins — recomputed on an identical
    voxelization of the same cube (voxelize is deterministic)."""
    dom = voxelize(_box_mesh(), VOXEL_MM, 150_000)
    ring = np.asarray(
        region_cell_mask(
            dom,
            RegionSpec(kind="hole", hole_index=1, margin_mm=_KEEP_MARGIN_MM),
            [dict(_ANCHOR)],
        ),
        dtype=bool,
    )
    assert ring.any()
    return ring


def test_auto_keep_flag_fires_in_manual_mode(tmp_path: Path) -> None:
    """T1: keep_mode='manual' + topopt_auto_keep=True must pin a ring of
    passive cells around the declared anchor, same as keep_mode='auto'."""
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = _run(src, tmp_path / "on", RuntimeSettings(topopt_auto_keep=True))
    assert res.ok, res.error
    assert res.topopt["auto_keep_applied"] is True

    npz = np.load(tmp_path / "on" / "topopt" / "density.npz")
    passive = np.asarray(npz["passive"], dtype=bool)
    ring = _anchor_ring()
    assert passive.shape == ring.shape
    assert not (ring & ~passive).any()  # every ring cell is pinned solid
    # and the pinned ring survives density extraction into the shipped mask
    assert np.asarray(npz["mask"], dtype=bool)[ring].all()

    # T7 telemetry — emitted only for keys whose sources were computed
    ver = res.topopt["verification"]
    assert ver["volfrac_mask"] > 0
    assert ver["solver_used"] == "oc"
    assert ver["dofs"] > 0
    assert len(ver["case_sfs_mask"]) == 1
    assert len(ver["case_sfs_export"]) == 1
    if ver["sf_mask"] is not None and ver["sf_export"] is not None:
        assert ver["sf_gap"] == pytest.approx(ver["sf_mask"] - ver["sf_export"])


def test_auto_keep_flag_off_disables_manual_keeps(tmp_path: Path) -> None:
    """T1: the flag is the opt-out — keep_mode='manual' + flag False adds no
    auto keeps, so the anchor ring stays unpinned."""
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = _run(src, tmp_path / "off", RuntimeSettings(topopt_auto_keep=False))
    assert res.ok, res.error
    assert res.topopt["auto_keep_applied"] is False

    npz = np.load(tmp_path / "off" / "topopt" / "density.npz")
    passive = np.asarray(npz["passive"], dtype=bool)
    ring = _anchor_ring()
    # no declared keep regions -> passive only covers fix/load cells, which
    # live on the +-x faces and never reach the centred anchor ring
    assert not (ring & passive).any()


def test_export_mesh_all_writes_step(tmp_path: Path) -> None:
    """T4: try_step=True sews the mesh into a faceted B-rep and ships a STEP;
    failure degrades to step=None, never raises."""
    from build123d import import_step

    from cadx.cad.exporters import export_mesh_all

    res = export_mesh_all(_box_mesh(), tmp_path / "e", name="m", try_step=True)
    assert res["stl"] == "m.stl" and res["3mf"] == "m.3mf" and res["glb"] == "m.glb"
    if res["step"] is None:
        pytest.skip("OCC STEP export (mesh_to_solid sew) unavailable in this env")
    step = tmp_path / "e" / res["step"]
    assert step.exists() and step.stat().st_size > 100
    solid = import_step(str(step))
    assert abs(solid.volume - 24.0**3) / 24.0**3 < 0.05


def test_export_mesh_all_try_step_off(tmp_path: Path) -> None:
    """try_step=False keeps the old contract — step key present but None."""
    from cadx.cad.exporters import export_mesh_all

    res = export_mesh_all(_box_mesh(), tmp_path / "e", name="m", try_step=False)
    assert res["step"] is None
    assert res["stl"] == "m.stl"


def test_fem_renders_are_pass_indexed(monkeypatch, tmp_path: Path) -> None:
    """T6: a forced volfrac retry must re-run FEM verification under '_p2'
    suffixes instead of overwriting the pass-1 renders."""
    import cadx.cad.fem.analyze as analyze_mod
    from cadx.cad.engine import run_topopt_pipeline

    suffixes: list[str] = []
    real = analyze_mod.run_fem_domain

    def spy(domain, solid_mask, load_cases, material_name="", **kw):
        suffixes.append(str(kw.get("suffix", "")))
        return real(domain, solid_mask, load_cases, material_name, **kw)

    monkeypatch.setattr(analyze_mod, "run_fem_domain", spy)

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case()],
        optimize={
            "target_volume_fraction": 0.5,
            "max_iterations": 4,
            # huge SF target -> sf_export always below -> retry pass runs
            "safety_factor_min": 1e6,
        },
        material="pla",
        voxel_mm=VOXEL_MM,
        scan_anchors=False,
        runtime=RuntimeSettings(),
    )
    assert res.ok, res.error
    assert "retry" in res.topopt
    assert suffixes == ["", "_export", "_p2", "_p2_export"]

    # when offscreen rendering is available the files must exist on disk too
    p2 = sorted(p.name for p in (tmp_path / "o" / "fem").glob("*_p2_*"))
    if p2:
        assert any(n.startswith("vonmises_p2_") for n in p2)
        assert any(n.startswith("deformed_p2_export_") for n in p2)


def test_unresolved_fixed_region_is_a_hard_error(tmp_path: Path) -> None:
    """T8: the spec's 'bore #2' fixed region cannot resolve when the geometry
    only contains one scanned bore — the run must refuse instead of
    optimizing a part clamped at the wrong place."""
    from cadx.cad.engine import run_topopt_pipeline

    case = LoadCase(
        name="op",
        fixed=[
            RegionSpec(kind="hole", hole_index=1),
            RegionSpec(kind="hole", hole_index=2),
        ],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -50.0],
            )
        ],
    ).model_dump()
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[case],
        optimize={"target_volume_fraction": 0.5, "max_iterations": 4},
        material="pla",
        voxel_mm=VOXEL_MM,
        anchors=[dict(_ANCHOR)],  # one anchor only — hole_index=2 misses
        scan_anchors=False,
        runtime=RuntimeSettings(),
    )
    assert not res.ok
    assert "did not resolve" in (res.error or "")
    assert "hole" in res.error
