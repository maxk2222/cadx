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
    """Cube with a REAL dia-4 bore through +z at (0,0) — matching `_ANCHOR`.

    regions.py now requires the anchor's interior cylinder to sit on actual
    void cells: a phantom anchor inside solid material resolves to an empty
    ring by design (that was the phantom-fixture bug), so the fixture needs
    the declared hole to exist in the geometry."""
    m = trimesh.creation.box(extents=[size, size, size])
    m.apply_translation([0.0, 0.0, size / 2])  # contract: rests on XY, centred
    bore = trimesh.creation.cylinder(radius=float(_ANCHOR["dia"]) / 2, height=size + 2.0)
    bore.apply_translation([0.0, 0.0, size / 2])
    try:
        holed = m.difference(bore)
    except Exception:  # boolean backend missing — phantom fixture again
        return m
    return holed if isinstance(holed, trimesh.Trimesh) else m


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


def _run(src: Path, out_dir: Path, rt: RuntimeSettings, spec: dict | None = None, **opt: Any):
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
        spec=spec,
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
    auto keeps, so passive covers ONLY the cells the fix/load regions touch.
    (The bore wall legitimately belongs to the -x face region — surface
    normals, not just the outer skin — so a ring-disjoint check no longer
    works once the anchor hole is real; checking the passive mask is a pure
    fix+load expansion asserts the same thing directly.)"""
    from cadx.cad.fem.regions import region_node_mask, resolve_case_dofs
    from cadx.cad.fem.topopt import passive_solid_mask

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = _run(src, tmp_path / "off", RuntimeSettings(topopt_auto_keep=False))
    assert res.ok, res.error
    assert res.topopt["auto_keep_applied"] is False

    npz = np.load(tmp_path / "off" / "topopt" / "density.npz")
    passive = np.asarray(npz["passive"], dtype=bool)
    dom = voxelize(_box_mesh(), VOXEL_MM, 150_000)
    case = LoadCase(**_cantilever_case())
    fixed3, _, _ = resolve_case_dofs(dom, case, [dict(_ANCHOR)])
    load_nm = region_node_mask(dom, case.loads[0].region, [dict(_ANCHOR)])
    expected = passive_solid_mask(dom, [], [fixed3.any(axis=1)], [load_nm])
    # no keep regions -> nothing pinned beyond the fix+load expansion
    assert not (passive.ravel() & ~expected).any()


def test_design_domain_auto_coarsens_to_direct_solver(tmp_path: Path) -> None:
    """SIMP re-solves the design system every iteration — when the grid's
    DOFs exceed fem_spsolve_dofs the SSOR-CG fallback turns a minutes-long
    run into hours. The pipeline must coarsen the design pitch until the
    system fits the direct solver and report the choice."""
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = _run(src, tmp_path / "c", RuntimeSettings(fem_spsolve_dofs=1000))
    assert res.ok, res.error
    ac = res.topopt.get("auto_coarsen")
    assert ac is not None, "grid over the DOF cap must record the coarsen"
    assert ac["design_pitch_mm"] > ac["requested_pitch_mm"]
    assert ac["dofs"] <= 1000
    # the density field must live on the coarsened grid, matching the report
    npz = np.load(tmp_path / "c" / "topopt" / "density.npz")
    assert float(npz["pitch"]) == pytest.approx(ac["design_pitch_mm"], abs=1e-6)


def test_topopt_result_carries_bound_verify_spec(tmp_path: Path) -> None:
    """The optimize payload's spec must reach run_topopt_pipeline, which
    attaches the canonical verify_spec report with *bound* envelope
    semantics: stated dims are an upper bound the optimized part fits
    inside — a shrunken SIMP result passes, exact-match would flag drift."""
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    spec = {
        "name": "b",
        "length_x": {"value": 24.0, "source": "stated"},
        "width_y": {"value": 24.0, "source": "stated"},
        "height_z": {"value": 24.0, "source": "stated"},
        "features": [],
    }
    res = _run(src, tmp_path / "vs", RuntimeSettings(), spec=spec)
    assert res.ok, res.error
    vs = (res.to_dict() or {}).get("verify_spec")
    assert vs is not None, "topopt result must carry the canonical verify_spec report"
    env = {c["name"]: c for c in vs["checks"] if c["name"].startswith("envelope_")}
    assert env, "envelope checks missing from the report"
    for c in env.values():
        # the SIMP'd part is smaller than the 24 mm box — under exact-match
        # semantics that is a FAIL; bound semantics must read PASS
        assert c["evidence"].get("bound") == "upper"
        assert c["status"] == "PASS", c
    shrunk = [
        c
        for c in env.values()
        if c["evidence"]["actual_mm"] < c["evidence"]["expected_mm"] - c["evidence"]["tol_mm"]
    ]
    assert shrunk, "test premise broken — the optimized part shrank on no axis"
    # a watertight single-body optimized part has no legitimately FAILing tier
    # — anything else would flag false negatives on every topopt result
    fails = [c["name"] for c in vs["checks"] if c["status"] == "FAIL"]
    assert not fails, f"spurious FAIL tiers on a valid topopt result: {fails}"


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
