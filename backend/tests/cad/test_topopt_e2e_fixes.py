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
M4 — res.topopt["iterations_total"] counts the multires coarse pass too:
     `iterations` stays the fine-phase count for backward compat.
M8 — an sf_mask >> sf_export divergence (>3x) means smoothing deleted
     load-bearing members; it must retry at the SAME volfrac with a
     thicker min-member floor, not bump the volume fraction.
M2 — a shell+infill run degenerating to fully solid surfaces an
     actionable warning instead of silently shipping the heavier part.
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
    # a coarse grid inflates the pinned keep/fix/load fraction — with the
    # anchor ring at ~4 cells pitch that floor legitimately tops the default
    # 0.5 target and the pipeline rightly refuses as infeasible; a 0.8 target
    # keeps the (still coarsened) run solvable so the coarsen is exercised
    res = _run(
        src,
        tmp_path / "c",
        RuntimeSettings(fem_spsolve_dofs=5000),
        target_volume_fraction=0.8,
    )
    assert res.ok, res.error
    ac = res.topopt.get("auto_coarsen")
    assert ac is not None, "grid over the DOF cap must record the coarsen"
    assert ac["design_pitch_mm"] > ac["requested_pitch_mm"]
    assert ac["dofs"] <= 5000
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
    # each pass solves the mask plus the nominal/eroded/dilated export
    # realizations — every run gets its own suffix, so pass-2 renders can
    # never clobber pass-1 output
    assert suffixes == [
        "",
        "_export",
        "_export_ero",
        "_export_dil",
        "_p2",
        "_p2_export",
        "_p2_export_ero",
        "_p2_export_dil",
    ]

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


def test_multires_reports_iterations_total(tmp_path: Path) -> None:
    """M4: under multires `iterations` counts only the fine phase — a run
    that reports "converged, 2 iters" after a 10-iter coarse pass misleads
    the agent about what the optimization cost. `iterations_total` must
    report coarse + fine while `iterations` keeps the fine-phase count."""
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = _run(
        src,
        tmp_path / "mr",
        # force the coarse pass below the default 4000-cell payoff floor
        RuntimeSettings(topopt_multires_min_cells=100),
    )
    assert res.ok, res.error
    mr = res.topopt.get("multires")
    assert mr is not None, "multires coarse pass did not run"
    assert mr["coarse_iterations"] > 0
    assert res.topopt["iterations_total"] == mr["coarse_iterations"] + res.topopt["iterations"]


def test_staged_volfrac_continuation_preserves_export_sf(tmp_path: Path) -> None:
    """The plateau-held schedule should avoid the old order-of-magnitude
    SF collapse; it remains an opt-in when its topology basin is weaker."""
    from cadx.cad.engine import run_topopt_pipeline

    mesh = trimesh.creation.box(extents=[60.0, 40.0, 6.0])
    mesh.apply_translation([0.0, 0.0, 3.0])
    src = tmp_path / "plate.stl"
    mesh.export(str(src))
    cases = [
        LoadCase(
            name="bend",
            fixed=[RegionSpec(kind="face", axis="-x")],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="face", axis="+x", margin_mm=2.0),
                    force_n=[0.0, 0.0, -50.0],
                )
            ],
        ).model_dump(),
        LoadCase(
            name="shear",
            fixed=[RegionSpec(kind="face", axis="-x")],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="face", axis="+x", margin_mm=2.0),
                    force_n=[0.0, 40.0, 0.0],
                )
            ],
        ).model_dump(),
    ]

    def run(name: str, continuation: bool):
        optimize = {
            "target_volume_fraction": 0.4,
            "max_iterations": 55,
            "min_member_mm": 1.5,
            "solver": "oc",
            "filter": "cone",
        }
        if continuation:
            optimize.update(volfrac_init=0.9, volfrac_ramp_iters=40)
        return run_topopt_pipeline(
            mesh_path=src,
            out_dir=tmp_path / name,
            load_cases=cases,
            optimize=optimize,
            material="pla",
            voxel_mm=1.5,
            scan_anchors=False,
            runtime=RuntimeSettings(),
        )

    baseline = run("baseline", False)
    staged = run("staged", True)
    sf_base = baseline.topopt["sf_export"]
    sf_staged = staged.topopt["sf_export"]
    assert baseline.ok and staged.ok
    assert sf_base is not None and sf_staged is not None
    assert staged.topopt["verification"]["valid"] is True
    assert staged.topopt["n_bodies_export"] == 1
    assert abs(staged.topopt["verification"]["vol_error_pp"]) <= 5.0
    # volume-matched extraction ships ~full-volume exports now (the old
    # ~30% underfill shrank BOTH candidates), so the honest mask/export
    # picture spreads wider than on the artifact meshes this bound was
    # calibrated on — the guard is against an order-of-magnitude collapse,
    # not parity
    assert sf_staged >= 0.5 * sf_base


def test_sf_gap_divergence_triggers_thick_member_retry(monkeypatch, tmp_path: Path) -> None:
    """M8: sf_mask >> sf_export means extraction smoothing deleted
    load-bearing members (the mask solve verified the voxel structure, the
    export solve the smoothed printed mesh). A >3x gap must retry at the
    SAME volfrac with a thicker min-member floor — more volume is the wrong
    cure — and the retry record must not be shadowed by a second warning."""
    import cadx.cad.fem.analyze as analyze_mod
    from cadx.cad.engine import run_topopt_pipeline

    real = analyze_mod.run_fem_domain

    def spy(domain, solid_mask, load_cases, material_name="", **kw):
        rep = real(domain, solid_mask, load_cases, material_name, **kw)
        # deterministic 5x mask/export gap, safely above the 1.5 sf floor so
        # the safety (volfrac) retry cannot steal the trigger
        sf = 10.0 if str(kw.get("suffix", "")).endswith("_export") else 50.0
        for c in rep.cases:
            c.safety_factor = sf
        return rep

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
            "safety_factor_min": 1.5,
        },
        material="pla",
        voxel_mm=VOXEL_MM,
        scan_anchors=False,
        runtime=RuntimeSettings(),
    )
    assert res.ok, res.error
    retry = res.topopt.get("retry")
    assert retry is not None, "sf_gap retry did not fire on a 5x divergence"
    assert retry["trigger"] == "sf_gap"
    assert retry["crisp_extraction"] is True
    # the divergence retry holds the volume target — it thickens members
    # instead of adding material
    assert retry["volfrac_second"] == retry["volfrac_first"]
    assert retry["sf_ratio_first"] == pytest.approx(5.0)
    assert retry["min_member_mm"] >= 2 * res.mesh["pitch_mm"]
    # the retry record documents the divergence — no duplicate warning
    assert not any("smoothing removed load-bearing material" in w for w in res.topopt["warnings"])


@pytest.mark.parametrize(
    ("interior_cells", "expected"),
    [
        (0, "no interior cells — the part is too thin for shell+infill entirely"),
        (
            5,
            "infill degenerated to solid — the eligible core is too thin for infill_period_mm=8.0",
        ),
    ],
)
def test_infill_degenerate_to_solid_warns(
    monkeypatch, tmp_path: Path, interior_cells: int, expected: str
) -> None:
    """M2: a part too thin for shell+infill used to ship fully solid with no
    hint. infill.py flags the degeneration in its meta (core_cells==0 means
    no room for a shell+core split at all); the pipeline must surface
    an actionable warning either way."""
    import cadx.cad.fem.infill as infill_mod
    from cadx.cad.engine import run_infill_pipeline

    real = infill_mod.infill_field

    def spy(domain, p, energies, dirs, passive, void, **kw):
        rho, mask, meta = real(domain, p, energies, dirs, passive, void, **kw)
        meta["degenerated_to_solid"] = True
        meta["interior_cells"] = interior_cells
        meta["core_cells"] = interior_cells
        return rho, mask, meta

    monkeypatch.setattr(infill_mod, "infill_field", spy)

    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_infill_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        load_cases=[_cantilever_case()],
        optimize={"infill_scope": "full"},  # skip the guided coarse SIMP
        material="pla",
        voxel_mm=VOXEL_MM,
        runtime=RuntimeSettings(),
    )
    assert res.ok, res.error
    assert res.topopt["infill"]["degenerated_to_solid"] is True
    assert any(expected in w for w in res.topopt["warnings"])


def test_thick_infill_export_preserves_nested_lattice_shells(tmp_path: Path) -> None:
    """A substantial negative-volume inner shell is the gyroid cavity, not a
    marching-cubes sliver; preserve it while counting physical export bodies."""
    from cadx.cad.engine import run_infill_pipeline

    mesh = trimesh.creation.box(extents=[60.0, 40.0, 24.0])
    mesh.apply_translation([0.0, 0.0, 12.0])
    src = tmp_path / "plate24.stl"
    mesh.export(str(src))
    res = run_infill_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "infill24",
        load_cases=[_cantilever_case()],
        optimize={
            "infill_scope": "full",
            "infill_style": "gyroid_graded",
            "target_volume_fraction": 0.7,
            "infill_shell_mm": 2.0,
            "infill_period_mm": 8.0,
        },
        material="pla",
        voxel_mm=1.5,
        runtime=RuntimeSettings(),
    )
    assert res.ok, res.error
    part_fraction = res.mesh["volume_mm3"] / (60.0 * 40.0 * 24.0)
    mask_fraction = res.topopt["verification"]["volfrac_mask"]
    assert res.topopt["n_surface_components"] >= 2
    assert res.topopt["n_bodies_export"] == 1
    ver = res.topopt["verification"]
    # the fixture region pins ~5x more nodes on the hollow structure than on
    # the solid baseline — a porous part cannot physically be stiffer than
    # the full part, so the inflated SF is honestly flagged invalid
    if ver["sf_baseline"] and ver["sf_export"] and ver["sf_export"] >= 2 * ver["sf_baseline"]:
        assert ver["valid"] is False
        assert any("baseline" in w for w in res.topopt["warnings"])
    else:
        assert ver["valid"] is True
    assert abs(part_fraction - mask_fraction) <= 0.05
    assert part_fraction < 0.9


def test_infill_target_fraction_calibrates_full_and_guided_stl(tmp_path: Path) -> None:
    """Full and Guided target_volume_fraction refers to exported STL mass,
    not only the coarse guide or pre-extraction occupancy estimate."""
    from cadx.cad.engine import run_infill_pipeline

    mesh = trimesh.creation.box(extents=[60.0, 40.0, 24.0])
    mesh.apply_translation([0.0, 0.0, 12.0])
    src = tmp_path / "plate24.stl"
    mesh.export(str(src))

    def run(scope: str, target: float, out_name: str):
        return run_infill_pipeline(
            mesh_path=src,
            out_dir=tmp_path / out_name,
            load_cases=[_cantilever_case()],
            optimize={
                "infill_scope": scope,
                "infill_style": "gyroid_graded",
                "target_volume_fraction": target,
                "infill_shell_mm": 2.0,
                "infill_period_mm": 8.0,
            },
            material="pla",
            voxel_mm=1.5,
            runtime=RuntimeSettings(),
        )

    full_60 = run("full", 0.60, "full60")
    full_75 = run("full", 0.75, "full75")
    guided_40 = run("guided", 0.40, "guided40")
    for result, target in ((full_60, 0.60), (full_75, 0.75), (guided_40, 0.40)):
        assert result.ok, result.error
        assert result.topopt["verification"]["volfrac_export"] == pytest.approx(target, abs=0.05)
        assert result.topopt["verification"]["meets_target_volume"] is True
    assert abs(full_60.topopt["volfrac_achieved"] - full_75.topopt["volfrac_achieved"]) > 0.1


def test_infill_target_below_shell_floor_is_reported_infeasible(tmp_path: Path) -> None:
    """Do not silently return a shell whose material floor is far above the
    requested full-infill fraction."""
    from cadx.cad.engine import run_infill_pipeline

    mesh = trimesh.creation.box(extents=[60.0, 40.0, 24.0])
    mesh.apply_translation([0.0, 0.0, 12.0])
    src = tmp_path / "plate24.stl"
    mesh.export(str(src))
    res = run_infill_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "infeasible",
        load_cases=[_cantilever_case()],
        optimize={
            "infill_scope": "full",
            "target_volume_fraction": 0.05,
            "infill_shell_mm": 2.0,
            "infill_period_mm": 8.0,
        },
        material="pla",
        voxel_mm=1.5,
        runtime=RuntimeSettings(),
    )
    assert res.ok, res.error
    assert res.topopt["status"] == "infeasible_target"
    assert res.topopt["verification"]["target_infeasible"] is True
    assert res.topopt["volfrac_achieved"] > 0.05


def test_topopt_smoothed_export_is_clipped_to_spec_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Surface smoothing may grow beyond a thin stated envelope; clip the
    print/export candidate before FEM and the final conformance report."""
    import cadx.cad.engine as engine_mod

    real_smooth = engine_mod.smooth_preserve

    def overshoot(mesh, *args, **kwargs):
        out = real_smooth(mesh, *args, **kwargs)
        zmax = float(out.bounds[1][2])
        out.vertices[:, 2] *= 24.8 / max(zmax, 1e-6)
        return out

    monkeypatch.setattr(engine_mod, "smooth_preserve", overshoot)
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = _run(
        src,
        tmp_path / "envelope",
        RuntimeSettings(topopt_auto_keep=False),
        spec_dims_mm=[24.0, 24.0, 24.0],
        target_volume_fraction=0.7,
        max_iterations=3,
    )

    assert res.ok, res.error
    assert res.topopt["output_envelope_ok"] is True
    assert res.analysis.bbox_size[2] <= 24.0 + 0.05


def test_infill_smoothed_export_is_clipped_to_spec_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cadx.cad.engine as engine_mod
    from cadx.cad.engine import run_infill_pipeline

    real_smooth = engine_mod.smooth_preserve

    def overshoot(mesh, *args, **kwargs):
        out = real_smooth(mesh, *args, **kwargs)
        zmax = float(out.bounds[1][2])
        out.vertices[out.vertices[:, 2] >= zmax - 1e-6, 2] += 0.8
        return out

    monkeypatch.setattr(engine_mod, "smooth_preserve", overshoot)
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_infill_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "infill-envelope",
        load_cases=[_cantilever_case()],
        optimize={
            "infill_scope": "full",
            "target_volume_fraction": 0.7,
            "infill_shell_mm": 2.0,
            "infill_period_mm": 8.0,
            "spec_dims_mm": [24.0, 24.0, 24.0],
        },
        material="pla",
        voxel_mm=VOXEL_MM,
        runtime=RuntimeSettings(),
    )

    assert res.ok, res.error
    assert res.topopt["output_envelope"]["method"] == "clip"
    assert res.topopt["output_envelope_ok"] is True
    assert res.analysis.bbox_size[2] <= 24.0 + 0.05


def test_infill_export_body_count_stays_unknown_when_revoxelization_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cadx.cad.fem.analyze as analyze_mod
    from cadx.cad.engine import run_infill_pipeline

    real_solve = analyze_mod.run_fem_domain

    def fail_export(domain, solid_mask, load_cases, material_name="", **kwargs):
        if str(kwargs.get("suffix", "")).endswith("export"):
            raise RuntimeError("injected export re-voxel solve failure")
        return real_solve(domain, solid_mask, load_cases, material_name, **kwargs)

    monkeypatch.setattr(analyze_mod, "run_fem_domain", fail_export)
    src = tmp_path / "b.stl"
    _box_mesh().export(str(src))
    res = run_infill_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "body-count-unknown",
        load_cases=[_cantilever_case()],
        optimize={
            "infill_scope": "full",
            "target_volume_fraction": 0.7,
            "infill_shell_mm": 2.0,
            "infill_period_mm": 8.0,
        },
        material="pla",
        voxel_mm=VOXEL_MM,
        runtime=RuntimeSettings(),
    )

    assert res.ok, res.error
    assert res.topopt["n_surface_components"] >= 1
    assert res.topopt["n_bodies_export"] is None
    assert res.topopt["verification"]["valid"] is False


def test_infill_export_divergence_gets_one_crisp_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A >3x mask/export SF gap retries once with crisp extraction; the
    final verification must describe the selected retry candidate."""
    import cadx.cad.fem.analyze as analyze_mod
    from cadx.cad.engine import run_infill_pipeline

    real = analyze_mod.run_fem_domain

    def spy(domain, solid_mask, load_cases, material_name="", **kw):
        report = real(domain, solid_mask, load_cases, material_name, **kw)
        suffix = str(kw.get("suffix", ""))
        safety = 30.0 if suffix == "_retry_export" else 10.0 if suffix == "_export" else 50.0
        for case in report.cases:
            case.safety_factor = safety
            case.compliance_j = 1e-3
            case.max_disp_mm = 0.1
            case.max_vm_mpa = 1.0
        return report

    monkeypatch.setattr(analyze_mod, "run_fem_domain", spy)
    mesh = trimesh.creation.box(extents=[60.0, 40.0, 24.0])
    mesh.apply_translation([0.0, 0.0, 12.0])
    src = tmp_path / "plate24.stl"
    mesh.export(str(src))
    res = run_infill_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "retry",
        load_cases=[_cantilever_case()],
        optimize={
            "infill_scope": "full",
            "target_volume_fraction": 0.75,
            "infill_shell_mm": 2.0,
            "infill_period_mm": 8.0,
        },
        material="pla",
        voxel_mm=1.5,
        runtime=RuntimeSettings(),
    )

    assert res.ok, res.error
    assert res.topopt["retry"]["trigger"] == ["sf_divergence"]
    assert res.topopt["verification"]["sf_export"] == pytest.approx(30.0)
    assert res.topopt["verification"]["sf_ratio"] == pytest.approx(50.0 / 30.0)
    assert res.topopt["verification"]["valid"] is True, (
        res.topopt["warnings"],
        res.topopt["verification"],
    )


def test_spec_dims_snap_design_pitch_and_domain(tmp_path: Path) -> None:
    """The reference mismatch: a 40x40x4 plate at a 2.5 mm requested pitch
    rasterized to a 5 mm domain (2 cells), so the mask FEM verified a part
    25% thicker than the 4 mm export the envelope clip produced. The
    spec-aware snap must land the thin axis on an integer cell count and
    report ~zero per-axis overshoot."""
    from cadx.cad.engine import run_topopt_pipeline

    plate = trimesh.creation.box(extents=[40.0, 40.0, 4.0])
    plate.apply_translation([0.0, 0.0, 2.0])
    src = tmp_path / "plate40x4.stl"
    plate.export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "plate-snap",
        load_cases=[_cantilever_case()],
        optimize={
            "target_volume_fraction": 0.5,
            "max_iterations": 3,
            "keep_mode": "manual",
            "spec_dims_mm": [40.0, 40.0, 4.0],
        },
        material="pla",
        voxel_mm=2.5,
        anchors=[],
        scan_anchors=False,
        runtime=RuntimeSettings(topopt_auto_keep=False),
    )

    assert res.ok, res.error
    dp = res.topopt["domain_pitch"]
    assert dp["snapped"] is True
    assert dp["requested_mm"] == pytest.approx(2.5)
    assert dp["design_mm"] == pytest.approx(4.0 / 3.0, abs=1e-3)  # min_cells_thinnest=3
    assert res.topopt["domain_spec_overshoot_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=0.05)
    lo, hi = res.topopt["domain_bbox"]
    assert hi[2] - lo[2] == pytest.approx(4.0, abs=0.05)
    # the extracted export must keep most of the verified mask volume —
    # the old pipeline dropped ~32 pp here
    assert abs(float(res.topopt["verification"]["vol_error_pp"])) < 15.0


def test_spec_dims_snap_keeps_requested_pitch_when_it_fits(tmp_path: Path) -> None:
    """Dims already divisible by the requested pitch stay untouched."""
    from cadx.cad.engine import run_topopt_pipeline

    mesh = trimesh.creation.box(extents=[60.0, 40.0, 6.0])
    mesh.apply_translation([0.0, 0.0, 3.0])
    src = tmp_path / "plate60x6.stl"
    mesh.export(str(src))
    res = run_topopt_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "plate-nosnap",
        load_cases=[_cantilever_case()],
        optimize={
            "target_volume_fraction": 0.5,
            "max_iterations": 3,
            "keep_mode": "manual",
            "spec_dims_mm": [60.0, 40.0, 6.0],
        },
        material="pla",
        voxel_mm=2.0,
        anchors=[],
        scan_anchors=False,
        runtime=RuntimeSettings(topopt_auto_keep=False),
    )

    assert res.ok, res.error
    assert res.topopt["domain_pitch"]["design_mm"] == pytest.approx(2.0)
    assert res.topopt["domain_spec_overshoot_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=0.05)
