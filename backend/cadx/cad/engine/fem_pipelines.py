"""FEM orchestration pipelines — topopt, infill, resample, refine,
organic, fem-verify. Extracted from the former `engine.py` monolith;
all public entries are re-exported by `cadx.cad.engine`."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cadx.cad import resources
from cadx.cad.contact_sheet import make_contact_sheet
from cadx.cad.engine.shared import (
    ExecResult,
    _attach_verify_spec,
    _audit_refined_features,
    _auto_keep_dicts,
    _auto_warm_density,
    _cancelled_pipeline_result,
    _clip_export_to_spec_envelope,
    _compact_history,
    _export_degeneracy,
    _feature_floor_pitch,
    _gate_sf,
    _GridRef,
    _guard_export_grooves,
    _guide_beso_er,
    _is_cancelled,
    _load_pins_nodes,
    _mark_cancelled,
    _opt_cancel_pair,
    _pitch_feature_warn,
    _post_extract_finish,
    _recommend_structure_mode,
    _record_lost_load_regions,
    _refine_analytic_step,
    _render_quality,
    _repair_analysis_mesh,
    _repair_export_meshes,
    _resolve_topopt_cases,
    _scan_anchors,
    _snap_pitch_under_floor,
    _solver_tier_gate,
    _spec_dims_from_dict,
    _tier_warning,
    _uniform_baseline_check,
    _upsample_density,
    _verify_export_structure,
    normalize_mesh_to_spec_envelope,
)
from cadx.cad.exporters import (
    export_mesh_all,
)
from cadx.cad.mesh_ops import (
    export_stl,
    is_flat_bottom,
    keep_significant_components,
    mesh_analysis,
    smooth_preserve,
)
from cadx.cad.mesh_ops import (
    load_mesh as load_triangle_mesh,
)
from cadx.cad.render import render_thumbnail, render_views
from cadx.cad.spec_features import active_features
from cadx.db.models import ProjectSettings
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

if TYPE_CHECKING:
    from cadx.cad.fem.topopt import TopOptParams

log = get_logger(__name__)

def run_fem_pipeline(
    *,
    mesh_path: Path,
    out_dir: Path,
    load_cases: list[dict[str, Any]],
    material: str = "",
    voxel_mm: float | None = None,
    max_elements: int | None = None,
    keep_regions: list[dict[str, Any]] | None = None,
    anchors: list[dict] | None = None,
    preview_only: bool = False,
    render_pngs: bool = True,
    scan_anchors: bool = True,
    scan_only: bool = False,
    runtime: RuntimeSettings | None = None,
    cancelled: Callable[[], bool] | None = None,
    target_volfrac: float | None = None,
) -> dict[str, Any]:
    """FEM verification on an existing mesh artifact. Writes fem/report.json +
    stress renders into <out_dir>/fem/. Returns the report as a dict.

    preview_only=True skips the solve and only renders which cells the
    declared regions select (fixed=blue load=red keep=green) — lets the
    agent sanity-check region definitions before running TopOpt.
    scan_only=True only scans/merges feature anchors (the `list_anchors`
    agent tool) — no voxelization, no solve."""
    from cadx.cad.fem.analyze import report_to_json, run_fem
    from cadx.llm.schemas import LoadCase, RegionSpec

    s = runtime or RuntimeSettings()
    cases = [LoadCase(**c) for c in load_cases]
    keep = [RegionSpec(**r) for r in (keep_regions or [])]
    try:
        mesh = load_triangle_mesh(mesh_path)
    except Exception as e:
        return {"ok": False, "error": f"failed to load mesh for FEM: {e}"}

    anchors = _scan_anchors(mesh, anchors, enabled=scan_anchors)
    if scan_only:
        return {"ok": True, "anchors": anchors}

    if preview_only:
        from cadx.cad.fem.preview import (
            case_region_masks,
            cells_preview,
            render_regions,
            resolve_case_summary,
        )
        from cadx.cad.fem.voxelize import voxelize

        try:
            domain = voxelize(mesh, voxel_mm or s.fem_voxel_mm, max_elements or s.fem_max_elements)
        except Exception as e:
            return {"ok": False, "error": f"voxelization failed: {e}"}
        pngs = (
            render_regions(domain, cases, keep, out_dir / "fem", anchors, source_mesh=mesh)
            if render_pngs
            else []
        )
        # structured region coverage per case — the values the PNG captions
        # already show — plus the passive-cell union across all cases: the
        # fraction of the design space a SIMP run cannot touch, i.e. the
        # minimum feasible volfrac
        case_summaries: list[dict[str, Any]] = []
        passive_cells = np.zeros(domain.dims, dtype=bool)
        keep_union = np.zeros(domain.dims, dtype=bool)
        fixed_union = np.zeros(domain.dims, dtype=bool)
        load_union = np.zeros(domain.dims, dtype=bool)
        for c in cases:
            summary = resolve_case_summary(domain, c, anchors)
            fixed_m, load_m, keep_m = case_region_masks(domain, c, keep, anchors)
            summary["n_fixed_cells"] = int(fixed_m.sum())
            summary["n_load_cells"] = int(load_m.sum())
            summary["n_keep_cells"] = int(keep_m.sum())
            passive_cells |= fixed_m | load_m | keep_m
            keep_union |= keep_m
            fixed_union |= fixed_m
            load_union |= load_m
            case_summaries.append(summary)
        passive_fraction = round(
            float((passive_cells & domain.solid).sum()) / max(int(domain.solid.sum()), 1),
            4,
        )
        return {
            "ok": True,
            "preview": {
                "pitch_mm": round(domain.pitch, 4),
                "dims": list(domain.dims),
                "solid_elements": domain.solid_count,
                "n_keep_cells": int(keep_union.sum()),
                "n_fixed_cells": int(fixed_union.sum()),
                "n_load_cells": int(load_union.sum()),
                "passive_fraction": passive_fraction,
                "min_feasible_volfrac": passive_fraction,
                "cases": case_summaries,
                "cells": cells_preview(domain, cases, keep, anchors),
                "regions_png": pngs,
                "anchors": anchors,
            },
        }

    report = run_fem(
        mesh,
        cases,
        material,
        voxel_mm=voxel_mm or s.fem_voxel_mm,
        max_elements=max_elements or s.fem_max_elements,
        out_dir=out_dir,
        anchors=anchors,
        keep_regions=keep,
        spsolve_dofs=s.fem_spsolve_dofs,
        mem_budget_mb=float(resources.effective_exec_mem_mb(s)),
        pool_workers=s.exec_pool_workers,
        cancelled=cancelled,
    )
    report_to_json(report, out_dir / "fem" / "report.json")
    fem_d = report.model_dump()
    mode, reasons = _recommend_structure_mode(
        report, keep_regions=keep, anchors=anchors, target_volfrac=target_volfrac
    )
    fem_d["recommended_mode"] = mode
    fem_d["recommended_reasons"] = reasons
    if target_volfrac is not None:
        fem_d["recommended_target_volfrac"] = target_volfrac
    return {"ok": True, "fem": fem_d}
def run_infill_pipeline(  # noqa: C901
    *,
    mesh_path: Path,
    out_dir: Path,
    load_cases: list[dict[str, Any]],
    optimize: dict[str, Any] | None = None,
    material: str = "",
    voxel_mm: float | None = None,
    max_elements: int | None = None,
    anchors: list[dict] | None = None,
    name: str = "model",
    cancelled: Any = None,
    runtime: RuntimeSettings | None = None,
    spec: dict[str, Any] | None = None,
) -> ExecResult:
    """Shell+Infill structure method — the FDM-native alternative to SIMP.

    One FEM solve per load case on the FULL part supplies the element
    strain energies; a closed outer skin plus a density-graded gyroid /
    strut lattice inside (fem/infill.py) then produces a watertight
    structure in ~2 solves instead of 60+ iterations. scope="guided"
    additionally restricts the lattice to the topology a cheap coarse
    SIMP retains. Verified through the same mask + export FEM path as
    run_topopt_pipeline — sf_export still describes the printed mesh,
    and sf_verified carries the worst case across the eroded/nominal/
    dilated export realizations plus the homogenized lattice model
    (sf_homog / sf_micro).
    """
    from scipy import ndimage

    from cadx import storage
    from cadx.cad.fem.analyze import report_to_json, run_fem_domain
    from cadx.cad.fem.beso import beso
    from cadx.cad.fem.extract import density_to_mesh, drop_island_components
    from cadx.cad.fem.infill import (
        InfillParams,
        element_strain_energy,
        infill_field,
    )
    from cadx.cad.fem.materials import resolve_material
    from cadx.cad.fem.preview import render_regions
    from cadx.cad.fem.regions import (
        anchor_entry_warnings,
        empty_region_warnings,
        keep_element_mask,
        material_cell_factors,
    )
    from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
    from cadx.cad.fem.voxelize import grid_bounds_mm, snapped_pitch, voxelize
    from cadx.llm.schemas import LoadCase, OptimizeSpec, RegionSpec

    res = ExecResult()
    s = runtime or RuntimeSettings()
    opt = OptimizeSpec(**(optimize or {}))
    t0 = time.monotonic()

    # same optimizer wall-clock bound as run_topopt_pipeline: the guide
    # solve (BESO/SIMP on the coarse grid) must stop inside
    # topopt_timeout_s so the candidate builds + verification land inside
    # the caller's wait + tail allowance. Reuses the cancelled mechanism —
    # a stopped guide still yields a usable density field.
    _opt_deadline = t0 + float(s.topopt_timeout_s) if s.topopt_timeout_s else None
    _opt_cancelled, _deadline_hit = _opt_cancel_pair(cancelled, _opt_deadline)

    # per-run overrides follow the runtime settings when unset — the same
    # tri-state pattern the other optimize flags use
    eff_octet = opt.infill_octet if opt.infill_octet is not None else bool(s.infill_octet)
    eff_robust = (
        opt.verify_robust if opt.verify_robust is not None else bool(s.fem_verify_realizations)
    )
    eff_homog = (
        opt.verify_homogenized
        if opt.verify_homogenized is not None
        else bool(s.fem_homogenized_infill)
    )

    try:
        mesh = load_triangle_mesh(mesh_path)
    except Exception as e:
        res.error = f"failed to load mesh for optimization: {e}"
        return res

    cases = [LoadCase(**c) for c in load_cases]
    if not cases:
        res.error = "topology optimization requires at least one load case"
        return res

    mat_name = material or ""
    aniso_axis = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}.get(
        str(getattr(opt, "print_direction", "+z") or "+z"), 2
    )
    mat = resolve_material(
        mat_name,
        aniso_z=(float(opt.aniso_factor) if opt.aniso_factor else 0.6)
        if opt.aniso_material
        else None,
        aniso_axis=aniso_axis if opt.aniso_material else None,
    )

    try:
        res.topopt["base_mesh_relpath"] = storage.relpath(mesh_path)
    except Exception:
        res.topopt["base_mesh_relpath"] = str(mesh_path)
    dims_spec = (optimize or {}).get("spec_dims_mm") or _spec_dims_from_dict(spec)
    if dims_spec:
        mesh, norm_rep = normalize_mesh_to_spec_envelope(mesh, list(dims_spec))
        res.topopt["input_bbox"] = norm_rep["input_bbox"]
        res.topopt["spec_bbox"] = norm_rep["spec_bbox"]
        res.topopt["input_normalized"] = norm_rep["input_normalized"]
        if norm_rep.get("warning"):
            res.topopt.setdefault("warnings", []).append(norm_rep["warning"])

    # scan anchors BEFORE the design pitch is chosen — same feature-floor
    # contract as run_topopt_pipeline (the smallest anchored bore caps the
    # grid resolution)
    anchors = _scan_anchors(mesh, anchors, enabled=True)
    res.topopt["anchors"] = anchors

    req_voxel = voxel_mm or s.fem_voxel_mm
    # same spec-aware pitch snap as run_topopt_pipeline — the domain must
    # model the thickness the envelope clip enforces on the export
    mesh_span = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
    eff_span = np.asarray(
        [
            float(d) if d else float(mesh_span[i])
            for i, d in enumerate(list(dims_spec or [None, None, None])[:3])
        ],
        dtype=float,
    )
    design_pitch, snap_warn = snapped_pitch(
        eff_span,
        req_voxel,
        min_cells=int(opt.min_cells_thinnest or 3),
        mesh_volume=abs(float(mesh.volume)) if mesh.is_watertight else 0.0,
        max_elements=max_elements or s.fem_max_elements,
    )
    if snap_warn:
        res.topopt.setdefault("warnings", []).append(snap_warn)
    # feature floor — same contract as run_topopt_pipeline: the snapped
    # pitch may overshoot the smallest anchored bore's resolution bound
    feature_floor = _feature_floor_pitch(anchors)
    floor_clamped = False
    if feature_floor is not None and design_pitch > feature_floor:
        design_pitch, floor_warn = _snap_pitch_under_floor(
            eff_span,
            feature_floor,
            min_cells=int(opt.min_cells_thinnest or 3),
            mesh_volume=abs(float(mesh.volume)) if mesh.is_watertight else 0.0,
            max_elements=max_elements or s.fem_max_elements,
        )
        floor_clamped = True
        if floor_warn:
            res.topopt.setdefault("warnings", []).append(floor_warn)
    res.topopt["domain_pitch"] = {
        "requested_mm": round(req_voxel, 4),
        "design_mm": round(design_pitch, 4),
        "snapped": abs(design_pitch - req_voxel) > 1e-6,
        "feature_floor_mm": round(feature_floor, 4) if feature_floor is not None else None,
        "feature_floor_clamped": floor_clamped,
    }
    try:
        domain = voxelize(mesh, design_pitch, max_elements or s.fem_max_elements)
    except Exception as e:
        res.error = f"voxelization failed: {e}"
        return res
    d_lo, d_hi = grid_bounds_mm(domain)
    res.topopt["domain_bbox"] = [
        [round(float(v), 3) for v in d_lo],
        [round(float(v), 3) for v in d_hi],
    ]
    span_cells = np.asarray(domain.dims, dtype=float) - 2 * domain.pad_cells
    overshoot = domain.pitch * span_cells - eff_span
    res.topopt["domain_spec_overshoot_mm"] = [round(float(v), 3) for v in overshoot]
    if dims_spec:
        bad_axes = [
            "xyz"[i] for i, d in enumerate(dims_spec) if d and overshoot[i] > 0.5 * domain.pitch
        ]
        if bad_axes:
            res.topopt.setdefault("warnings", []).append(
                f"design domain exceeds the spec envelope on "
                f"{','.join(bad_axes)} by more than half a cell — the export "
                "clip removes material the mask FEM verified"
            )

    # predictive memory budget + solver-tier gate — same policy as the
    # topopt pipeline: direct only when the predicted factorization fits,
    # CG/AMG otherwise, hard error only when even CG cannot fit
    from cadx.cad.fem.voxelize import estimate_fem_memory_mb

    mem_budget = float(resources.effective_exec_mem_mb(s))
    n_eff = min(len(cases) * 2, s.fem_max_effective_cases)
    # simultaneous factor count = load cases (ctx_cache), not effective cases
    spsolve_eff, tier_info = _solver_tier_gate(domain, len(cases), s, mem_budget)
    res.topopt["solver_tier"] = tier_info
    if warn := _tier_warning(tier_info):
        res.topopt.setdefault("warnings", []).append(warn)
    if mem_budget:
        est_mb = estimate_fem_memory_mb(
            domain, n_eff, direct=tier_info["tier"] == "direct", n_factors=len(cases)
        )
        if est_mb > mem_budget * 0.75:
            res.error = (
                f"predicted FEM memory ~{est_mb:.0f} MB exceeds the executor "
                f"budget ({mem_budget:.0f} MB) — raise fem_voxel_mm or "
                "exec_mem_limit_mb"
            )
            return res

    resolved, _cw, fixed_node_masks, load_node_masks, unresolved = _resolve_topopt_cases(
        domain,
        cases,
        anchors,
        mat,
        s.fem_max_effective_cases,
        warnings_out=res.topopt.setdefault("warnings", []),
    )
    if unresolved:
        res.error = (
            "load-case regions did not resolve on the geometry — "
            + "; ".join(unresolved[:4])
            + ". The declared fix/load feature is missing from the solid or its "
            "anchor is unresolvable; fix the geometry (feature placement) before "
            "optimizing — a result optimized under wrong constraints is unusable."
        )
        return res
    if not resolved:
        res.error = "no solvable load case — check fixed/load regions"
        return res

    keep_list = list(opt.keep_regions)
    iface_voids: list[RegionSpec] = []
    if anchors and (opt.keep_mode == "auto" or s.topopt_auto_keep):
        from cadx.cad.fem.regions import interface_regions

        margin = opt.auto_keep_margin_mm or s.topopt_auto_keep_margin_mm
        ik, iface_voids = interface_regions(anchors, margin, spec_only=(opt.keep_mode != "auto"))
        keep_list += ik
    res.topopt.setdefault("warnings", []).extend(empty_region_warnings(domain, keep_list, anchors))
    res.topopt.setdefault("warnings", []).extend(
        empty_region_warnings(domain, list(opt.void_regions), anchors)
    )
    res.topopt.setdefault("warnings", []).extend(anchor_entry_warnings(domain, anchors))
    keep_cells = keep_element_mask(domain, keep_list, anchors)
    void_cells = keep_element_mask(domain, list(opt.void_regions) + iface_voids, anchors)
    passive = passive_solid_mask(domain, [keep_cells], fixed_node_masks, load_node_masks)

    res.topopt["regions_png"] = render_regions(
        domain, cases, keep_list, out_dir / "fem", anchors, source_mesh=mesh
    )
    if _is_cancelled(cancelled):
        return _cancelled_pipeline_result(res, "after_region_render")

    min_member = float(opt.min_member_mm or s.topopt_min_member_mm or 0.0)
    min_gap = float(opt.min_gap_mm or min_member or 0.0)
    solver_backend = str(getattr(s, "fem_solver_backend", "auto") or "auto")

    # --- stage 1: full-solid FEM — the baseline report AND the energy field
    # that grades the lattice (fields_out returns the per-case solves)
    t = time.monotonic()
    fo: dict[str, Any] = {}
    verify_base = run_fem_domain(
        domain,
        domain.solid,
        cases,
        mat_name,
        material=mat,
        out_dir=out_dir,
        anchors=anchors,
        keep_regions=keep_list,
        spsolve_dofs=spsolve_eff,
        pool_workers=s.exec_pool_workers,
        mem_budget_mb=mem_budget,
        suffix="_baseline",
        source_mesh=mesh,
        cancelled=cancelled,
        solver_backend=solver_backend,
        fields_out=fo,
    )
    res.fem = verify_base.model_dump()
    report_to_json(verify_base, out_dir / "fem" / "report_baseline.json")
    res.timings["baseline_fem_s"] = round(time.monotonic() - t, 2)
    if _is_cancelled(cancelled):
        return _cancelled_pipeline_result(res, "after_baseline_fem")

    # solutions index the RAW case list (run_fem_domain resolves each
    # declared case once — cone expansion lives in the optimizer's resolved
    # set, not here); weights come from the cases themselves
    sols_pairs = [(i, sol) for i, sol in enumerate(fo.get("solutions") or []) if sol is not None]
    energies: list[np.ndarray] = []
    energy_w: list[float] = []
    for i, sol in sols_pairs:
        energies.append(
            element_strain_energy(
                domain,
                sol.u,
                fo["edof"],
                fo["active"],
                fo["elem_E"],
                fo["nu"],
                aniso=fo.get("aniso"),
                elem_class=fo.get("elem_class"),
                class_D=fo.get("class_D"),
            )
        )
        energy_w.append(float(cases[i].weight) if i < len(cases) else 1.0)
    if not energies:
        res.error = "no load case produced a displacement field — cannot grade infill"
        return res

    dirs = None
    frames = None
    if opt.infill_style == "struts_aligned":
        try:
            # per-case full stress frames -> a single energy-weighted
            # (evals, evecs) pair per cell; infill_field orients the strut
            # families from the combined tensor (octet diagonals included)
            from cadx.cad.fem.infill import (
                combine_stress_frames,
                element_principal_frames,
            )

            frames_list = [
                element_principal_frames(
                    domain,
                    sol.u,
                    fo["edof"],
                    fo["active"],
                    fo["elem_E"],
                    fo["nu"],
                    aniso=fo.get("aniso"),
                    elem_class=fo.get("elem_class"),
                    class_D=fo.get("class_D"),
                )
                for _i, sol in sols_pairs
            ]
            frames = combine_stress_frames(frames_list, energy_w)
            if frames is not None:
                # compatibility dirs for the axis-strut path: the dominant
                # eigenvector of the combined stress ellipsoid
                _evals, _evecs = frames
                _dom = np.argmax(np.abs(_evals), axis=1)
                dirs = _evecs[np.arange(len(_evecs)), :, _dom]
        except Exception as e:
            frames = None
            dirs = None
            res.topopt["warnings"].append(
                f"principal-frame solve failed — struts fall back to gyroid: {e}"
            )

    # --- stage 2 (optional): guided scope — a cheap coarse optimizer picks
    # the retained topology and the lattice only fills what it keeps.
    # BESO is the default guide solver: its natively binary output maps
    # straight onto the binary guide_mask without a thresholded gray zone.
    guide_mask: np.ndarray | None = None
    guide_field: dict[str, Any] | None = None  # persisted for warm starts
    if opt.infill_scope == "guided":
        try:
            coarse = voxelize(mesh, domain.pitch * 2.0, max_elements or s.fem_max_elements)
            res_c, w_c, fix_c, load_c, _un = _resolve_topopt_cases(
                coarse, cases, anchors, mat, s.fem_max_effective_cases, warn=False
            )
            if res_c:
                keep_c = keep_element_mask(coarse, keep_list, anchors)
                void_c = keep_element_mask(coarse, list(opt.void_regions), anchors)
                passive_c = passive_solid_mask(coarse, [keep_c], fix_c, load_c)
                E0_c = mat.elastic_modulus_mpa * material_cell_factors(
                    coarse, keep_list, anchors, mat.elastic_modulus_mpa
                )
                iters_c = max(10, (opt.max_iterations or s.topopt_max_iterations) // 3)
                pc = TopOptParams(
                    volfrac=opt.target_volume_fraction or s.topopt_default_volfrac,
                    rmin_mm=max(s.topopt_default_rmin_mm, min_member / 2),
                    max_iter=iters_c,
                    filter="cone",
                    solver_backend=solver_backend,
                    aniso=(mat.aniso_z, float(mat.aniso_axis)) if mat.aniso_z != 1.0 else None,
                )
                guide_solver = getattr(opt, "infill_guide_solver", "beso") or "beso"
                if guide_solver == "beso":
                    # the V_k·(1-ER) volume ramp must actually REACH volfrac
                    # inside the coarse iteration budget — the classic 2%
                    # step needs ~45 iterations to get from a full field to
                    # 0.4, so the guide ER scales up to the budget
                    pc.beso_er = _guide_beso_er(pc.volfrac, iters_c, pc.beso_er)
                    guide_fn = beso
                else:
                    guide_fn = simp
                guide_r = guide_fn(
                    coarse,
                    E0_c,
                    mat.poisson,
                    res_c,
                    passive_c,
                    pc,
                    passive_void=void_c,
                    case_weights=np.asarray(w_c),
                    spsolve_dofs=spsolve_eff,
                    cancelled=_opt_cancelled,
                )
                if _is_cancelled(cancelled):
                    res.topopt["guide"] = {
                        "solver": guide_solver,
                        "status": guide_r.status,
                        "iterations": guide_r.iterations,
                        "warnings": guide_r.warnings,
                    }
                    return _cancelled_pipeline_result(res, "after_guided_optimizer")
                guide_mask = (
                    _upsample_density(guide_r.density.reshape(coarse.dims), coarse, domain).reshape(
                        domain.dims
                    )
                    >= 0.5
                )
                res.topopt["guide"] = {
                    "solver": guide_solver,
                    "coarse_pitch_mm": round(float(coarse.pitch), 4),
                    "coarse_iterations": guide_r.iterations,
                    "kept_cells": int(guide_mask.sum()),
                }
                if _deadline_hit():
                    res.topopt["guide"]["deadline_stop"] = True
                # flat scalar so eval gates/reports can see the guide solver
                res.topopt["guide_solver"] = guide_solver
                if guide_solver == "beso":
                    res.topopt["guide"]["er_eff"] = round(pc.beso_er, 4)
                # keep the coarse layout field: a later topopt_run on the
                # same version warm-starts SIMP from this macro topology
                guide_field = {
                    "density": guide_r.density.reshape(coarse.dims).astype(np.float32),
                    "origin": np.asarray(coarse.origin, dtype=float),
                    "pitch": np.float64(coarse.pitch),
                    "dims": np.asarray(coarse.dims),
                    "kind": np.str_(guide_solver),
                }
            else:
                res.topopt["warnings"].append(
                    "guided scope: no solvable case on the coarse grid — full infill"
                )
        except Exception as e:
            res.topopt["warnings"].append(f"guided scope coarse pass failed: {e} — full infill")

    p_infill = InfillParams(
        shell_mm=float(opt.infill_shell_mm),
        infill_cell_mm=float(opt.infill_period_mm),
        dmin=float(opt.infill_min_density),
        dmax=float(opt.infill_max_density),
        alpha=float(opt.infill_alpha),
        style=opt.infill_style,
        scope="guided" if guide_mask is not None else "full",
        seed=int(opt.infill_seed),
        octet=eff_octet,
    )
    opt_dir = out_dir / "topopt"
    opt_dir.mkdir(parents=True, exist_ok=True)
    base_flat = is_flat_bottom(mesh)

    def _cancelled_candidate(candidate: dict[str, Any], phase: str) -> ExecResult:
        _mark_cancelled(res)
        res.topopt["cancelled_phase"] = phase
        rho_part = candidate.get("rho")
        if rho_part is not None:
            density_path = opt_dir / "partial_infill_density.npz"
            np.savez_compressed(density_path, density=np.asarray(rho_part, dtype=np.float32))
            res.topopt["partial_density_relpath"] = storage.relpath(density_path)
        candidate_mesh = candidate.get("mesh")
        if candidate_mesh is not None:
            run_dir = out_dir / "_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            stl_path = run_dir / "model.stl"
            with contextlib.suppress(Exception):
                candidate_mesh = export_stl(candidate_mesh, stl_path)
                res.mesh = {
                    "pipeline": "infill",
                    "partial": True,
                    "stl_relpath": storage.relpath(stl_path),
                    "triangles": len(candidate_mesh.faces),
                    "volume_mm3": round(float(candidate_mesh.volume), 1),
                }
        return res

    if guide_field is not None:
        try:
            np.savez_compressed(opt_dir / "guide_density.npz", **guide_field)
            res.topopt["guide"]["guide_density_relpath"] = storage.relpath(
                opt_dir / "guide_density.npz"
            )
        except Exception as e:
            res.topopt["warnings"].append(f"guide density persist failed: {e}")

    part_volume = (
        abs(float(mesh.volume))
        if mesh.is_watertight and abs(float(mesh.volume)) > 0
        else float(domain.solid_count) * domain.pitch**3
    )
    target_volfrac = float(opt.target_volume_fraction or s.topopt_default_volfrac)

    def _build_candidate(density_offset: float, *, crisp: bool = False) -> dict[str, Any]:
        # fields_out keeps the internal target-density/interior/occupancy
        # fields — the homogenized verification model re-uses them instead
        # of re-deriving the lattice layout
        cand_fields: dict[str, Any] = {}
        rho, _mask, meta = infill_field(
            domain,
            p_infill,
            energies,
            dirs,
            passive,
            void_cells,
            guide_mask=guide_mask,
            weights=energy_w,
            density_offset=density_offset,
            frames=frames,
            fields_out=cand_fields,
        )
        if _is_cancelled(cancelled):
            return {"cancelled_phase": "after_infill_field", "rho": rho}
        candidate_mesh, mask, extract_warnings, components, extract = density_to_mesh(
            domain,
            rho,
            threshold=0.5,
            passive_solid=passive,
            void_cells=void_cells.reshape(domain.dims) | ~domain.solid,
            min_member_mm=min_member,
            min_gap_mm=min_gap,
            smooth_iterations=0 if crisp else s.topopt_smooth_iterations,
            extract_smooth=opt.extract_smooth and not crisp,
            sigma_px=0.0 if crisp else opt.extract_sigma_px,
            volume_preserve=opt.volume_preserve,
            # the lattice occupancy is FEM-verified structure — its strut
            # cross-sections and the skin bond must not erode under the
            # internal Taubin pass
            pin_cells=cand_fields.get("occ3"),
            min_shell_mm=opt.min_shell_mm,
        )
        if _is_cancelled(cancelled):
            return {
                "cancelled_phase": "after_density_extraction",
                "rho": rho,
                "mask": mask,
                "mesh": candidate_mesh,
            }
        # ndarray telemetry must not reach the persisted extract dict —
        # it kills the version-row JSON flush
        bridged3 = extract.pop("bridged_mask", None)
        pin3 = passive.reshape(domain.dims).copy()
        if bridged3 is not None:
            # corridor cells added by island-bridging pin like passive —
            # the connectivity glue must not erode below raster
            # thickness between mask and export
            pin3 |= np.asarray(bridged3, dtype=bool).reshape(domain.dims)
        occ3 = cand_fields.get("occ3")
        if occ3 is not None:
            # same for the lattice occupancy — verified strut material must
            # reach the shipped mesh, not just the voxel mask
            pin3 |= np.asarray(occ3, dtype=bool).reshape(domain.dims)
        if not crisp:
            thin_cells = float(eff_span.min()) / max(domain.pitch, 1e-6)
            candidate_mesh = smooth_preserve(
                candidate_mesh,
                iterations=2 if thin_cells < 4.0 else 4,
                pin_floor=base_flat,
                pin_mask=(domain.origin, domain.pitch, pin3),
            )
        candidate_mesh, dropped_slivers = keep_significant_components(candidate_mesh)
        warnings = list(extract_warnings)
        if dropped_slivers:
            warnings.append(f"dropped {dropped_slivers} near-zero-volume mesh sliver components")
        candidate_mesh, island_rescue = drop_island_components(
            candidate_mesh,
            max_island_frac=0.02,
            pin_mask=(domain.origin, domain.pitch, pin3),
        )
        if island_rescue["dropped"]:
            warnings.append(
                f"dropped {island_rescue['dropped']} disconnected island(s) "
                f"({island_rescue['dropped_vol_frac']:.4%} of export volume) "
                "— sub-voxel extraction pinch-off"
            )
        candidate_mesh, output_envelope, output_envelope_ok, envelope_warnings = (
            _clip_export_to_spec_envelope(
                candidate_mesh,
                list(dims_spec) if dims_spec else None,
                label="smoothed infill export",
            )
        )
        warnings.extend(envelope_warnings)
        mask_volume = float(mask.sum()) * domain.pitch**3
        export_fraction = abs(float(candidate_mesh.volume)) / part_volume
        extract["vol_error_pct"] = round(
            100.0 * (abs(float(candidate_mesh.volume)) - mask_volume) / max(mask_volume, 1e-9),
            3,
        )
        meta["density_offset"] = round(float(density_offset), 6)
        meta["volfrac_export_estimate"] = round(export_fraction, 4)
        meta["target_volume_fraction"] = target_volfrac
        meta["target_volume_error_pp"] = round(100.0 * (export_fraction - target_volfrac), 2)
        meta["scope_requested"] = opt.infill_scope
        if meta.get("degenerated_to_solid"):
            if int(meta.get("core_cells") or 0) == 0:
                warnings.append(
                    "no interior cells — the part is too thin for shell+infill entirely"
                )
            else:
                warnings.append(
                    "infill degenerated to solid — the eligible core is too thin for "
                    f"infill_period_mm={opt.infill_period_mm}; increase the period "
                    "or thicken the part"
                )
        elif (
            meta.get("volfrac_estimate") is not None
            and export_fraction - float(meta["volfrac_estimate"]) > 0.10
        ):
            # the designed lattice existed in the voxel mask but collapsed to
            # solid at mesh export — period too coarse vs local wall thickness
            # (the arm/bracket failure: 0.71 mask vs ~1.0 exported solid)
            warnings.append(
                f"lattice lost at export — mask volfrac {meta['volfrac_estimate']:.2f} "
                f"but the shipped solid measures {export_fraction:.2f}; the "
                f"{opt.infill_period_mm}mm period cannot resolve inside this "
                "wall — shrink infill_period_mm or switch to topopt material "
                "removal for deep volume targets"
            )
        elif opt.infill_scope == "guided" and int(meta.get("interior_cells") or 0) == 0:
            warnings.append(
                "guided SIMP retained no interior cells for lattice — the export is shell-only; "
                "use infill_scope='full' or revise the guided topology"
            )
        meta["island_rescue"] = island_rescue
        return {
            "rho": rho,
            "mask": mask,
            "mesh": candidate_mesh,
            "warnings": warnings,
            "components": components,
            "extract": extract,
            "meta": meta,
            "export_fraction": export_fraction,
            "density_offset": density_offset,
            "output_envelope": output_envelope,
            "output_envelope_ok": output_envelope_ok,
            "fields": cand_fields,
        }

    t_field = time.monotonic()
    best = _build_candidate(0.0)
    if best.get("cancelled_phase"):
        return _cancelled_candidate(best, best["cancelled_phase"])
    target_min: float | None = None
    target_max: float | None = None
    target_infeasible = False
    volume_tolerance = 0.05
    if abs(float(best["export_fraction"]) - target_volfrac) > volume_tolerance:
        offset_min = p_infill.dmin - p_infill.dmax
        offset_max = p_infill.dmax - p_infill.dmin
        low = _build_candidate(offset_min)
        if low.get("cancelled_phase"):
            return _cancelled_candidate(low, low["cancelled_phase"])
        high = _build_candidate(offset_max)
        if high.get("cancelled_phase"):
            return _cancelled_candidate(high, high["cancelled_phase"])
        low_fraction = float(low["export_fraction"])
        high_fraction = float(high["export_fraction"])
        target_min, target_max = sorted((low_fraction, high_fraction))
        increasing = low_fraction <= high_fraction
        best = min(
            (best, low, high),
            key=lambda row: abs(float(row["export_fraction"]) - target_volfrac),
        )
        target_infeasible = not (
            target_min - volume_tolerance <= target_volfrac <= target_max + volume_tolerance
        )
        lo, hi = offset_min, offset_max
        if not target_infeasible:
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                trial = _build_candidate(mid)
                if trial.get("cancelled_phase"):
                    return _cancelled_candidate(best, trial["cancelled_phase"])
                trial_fraction = float(trial["export_fraction"])
                if abs(trial_fraction - target_volfrac) < abs(
                    float(best["export_fraction"]) - target_volfrac
                ):
                    best = trial
                if abs(trial_fraction - target_volfrac) <= 0.01:
                    best = trial
                    break
                if (trial_fraction < target_volfrac) == increasing:
                    lo = mid
                else:
                    hi = mid
            if abs(float(best["export_fraction"]) - target_volfrac) > volume_tolerance:
                target_infeasible = True

    meta = best["meta"]
    if target_min is not None and target_max is not None:
        meta["volfrac_min_feasible"] = round(target_min, 4)
        meta["volfrac_max_feasible"] = round(target_max, 4)
    meta["target_infeasible"] = target_infeasible
    res.topopt["infill"] = meta
    res.topopt["output_envelope"] = best.get("output_envelope")
    res.topopt["output_envelope_ok"] = best.get("output_envelope_ok", True)
    if target_infeasible and target_min is not None and target_max is not None:
        res.topopt["warnings"].append(
            f"requested final STL volume fraction {target_volfrac:.0%} is outside the "
            f"reachable interval {target_min:.0%} to {target_max:.0%} with the current "
            "shell, guide, passive regions and minimum infill density"
        )
    res.timings["infill_field_s"] = round(time.monotonic() - t_field, 2)
    rho = best["rho"]
    final_mask = best["mask"]
    opt_mesh = best["mesh"]
    n_components = best["components"]
    extract_info = best["extract"]
    n_surface_components = int(opt_mesh.body_count)
    n_bodies_export: int | None = None

    edt = ndimage.distance_transform_edt(final_mask)
    inside = edt[final_mask]
    thin_fraction = float((inside <= 1.0).mean()) if len(inside) else 0.0
    min_member_mm = (
        round(2 * domain.pitch * float(np.percentile(inside, 5)), 2) if len(inside) else 0.0
    )

    # --- verification: same two-level evidence as SIMP — mask (exact voxel
    # structure) and re-voxelized export mesh (what actually gets printed)
    t = time.monotonic()
    verify = run_fem_domain(
        domain,
        final_mask,
        cases,
        mat_name,
        material=mat,
        out_dir=out_dir,
        anchors=anchors,
        keep_regions=keep_list,
        spsolve_dofs=spsolve_eff,
        pool_workers=s.exec_pool_workers,
        mem_budget_mb=mem_budget,
        source_mesh=opt_mesh,
        cancelled=cancelled,
        solver_backend=solver_backend,
    )
    res.fem = verify.model_dump()
    report_to_json(verify, out_dir / "fem" / "report.json")
    if _is_cancelled(cancelled):
        return _cancelled_candidate(best, "after_mask_verify")

    # the homogenized infill model (class-tensor lattice elements) — solved
    # once `best` settles below; None when disabled or unsolvable
    verify_h: Any = None

    export_solve_ok = True
    n_bodies_voxel: int | None = None
    exp_info: dict[str, Any] | None = None
    try:
        # same hardened export verification as SIMP — coverage-weighted
        # boundary cells, realization sweep (sf_verified), optional
        # mesh-shift probe and load-path certificate
        exp_info = _verify_export_structure(
            candidate_mesh=best["mesh"],
            cases=cases,
            mat_name=mat_name,
            mat=mat,
            design_pitch=design_pitch,
            max_elements=max_elements or s.fem_max_elements,
            keep_list=keep_list,
            anchors=anchors,
            out_dir=out_dir,
            suffix="_export",
            s=s,
            solver_backend=solver_backend,
            cancelled=cancelled,
            verify_robust=eff_robust,
        )
        if exp_info.get("cancelled"):
            if exp_info.get("report") is not None:
                res.fem_export = exp_info["report"].model_dump()
                report_to_json(exp_info["report"], out_dir / "fem" / "report_export.json")
            return _cancelled_candidate(best, exp_info.get("phase", "export_verify"))
        verify_exp = exp_info["report"]
        n_bodies_export = exp_info["n_bodies_export"]
        n_bodies_voxel = exp_info["n_bodies_voxel"]
        res.topopt.setdefault("warnings", []).extend(exp_info["warnings"])
    except Exception as e:
        if _is_cancelled(cancelled):
            return _cancelled_candidate(best, "during_export_verify")
        log.warning("infill.verify_export.failed", error=str(e)[:200])
        exp_info = None
        verify_exp = verify
        export_solve_ok = False
        res.topopt["warnings"].append(
            f"export FEM solve failed ({str(e)[:80]}) — sf_export "
            "duplicates the voxel-mask result, not the printed mesh"
        )

    def _export_metrics(
        candidate: dict[str, Any], report: Any, body_count: int | None
    ) -> tuple[float | None, float | None, bool]:
        sf_mask = verify.worst_safety_factor
        sf_export = report.worst_safety_factor
        ratio = (
            max(sf_mask, sf_export) / max(min(sf_mask, sf_export), 1e-9)
            if sf_mask is not None and sf_export is not None
            else None
        )
        export_fraction = float(candidate["export_fraction"])
        meets_target = abs(export_fraction - target_volfrac) <= volume_tolerance
        gate_failed = (
            candidate.get("output_envelope_ok") is False
            or body_count is None
            or body_count > 1
            or (ratio is not None and ratio > 3.0)
            or (not target_infeasible and not meets_target)
        )
        return ratio, export_fraction, gate_failed

    sf_ratio, volfrac_export, initial_gate_failed = _export_metrics(
        best, verify_exp, n_bodies_export
    )
    sf_export_first = verify_exp.worst_safety_factor
    sf_verified_first = (exp_info or {}).get("sf_verified")
    volfrac_first = volfrac_export
    ratio_first = sf_ratio
    bodies_first = n_bodies_export
    sf_min = float(opt.safety_factor_min)
    # the verified worst-case realization gates before the nominal export SF
    sf_eff = _gate_sf(exp_info, verify_exp)
    retry_needed = (
        initial_gate_failed
        or best.get("output_envelope_ok") is False
        or n_bodies_export is None
        or n_bodies_export > 1
        or not export_solve_ok
        or sf_eff is None
        or (sf_ratio is not None and sf_ratio > 3.0)
        or (sf_eff is not None and sf_eff < sf_min)
    )
    if retry_needed and (cancelled is None or not cancelled()):
        retry_reasons = []
        if n_bodies_export is None:
            retry_reasons.append("body_count_unverified")
        elif n_bodies_export > 1:
            retry_reasons.append("multi_body")
        if best.get("output_envelope_ok") is False:
            retry_reasons.append("output_envelope")
        if not export_solve_ok or sf_eff is None:
            retry_reasons.append("export_solve")
        if sf_ratio is not None and sf_ratio > 3.0:
            retry_reasons.append("sf_divergence")
        if (
            not target_infeasible
            and abs(float(best["export_fraction"]) - target_volfrac) > volume_tolerance
        ):
            retry_reasons.append("stl_volume")
        if sf_eff is not None and sf_eff < sf_min:
            retry_reasons.append("safety_factor")
        retry_started = time.monotonic()
        retry_candidate = _build_candidate(float(best["density_offset"]), crisp=True)
        if retry_candidate.get("cancelled_phase"):
            return _cancelled_candidate(best, retry_candidate["cancelled_phase"])
        retry_info: dict[str, Any] | None = None
        retry_voxel_bodies: int | None = None
        try:
            retry_info = _verify_export_structure(
                candidate_mesh=retry_candidate["mesh"],
                cases=cases,
                mat_name=mat_name,
                mat=mat,
                design_pitch=design_pitch,
                max_elements=max_elements or s.fem_max_elements,
                keep_list=keep_list,
                anchors=anchors,
                out_dir=out_dir,
                suffix="_retry_export",
                s=s,
                solver_backend=solver_backend,
                cancelled=cancelled,
                verify_robust=eff_robust,
            )
            if retry_info.get("cancelled"):
                if retry_info.get("report") is not None:
                    res.fem_export = retry_info["report"].model_dump()
                    report_to_json(
                        retry_info["report"], out_dir / "fem" / "report_export_retry.json"
                    )
                return _cancelled_candidate(best, retry_info.get("phase", "retry_export_verify"))
            retry_report = retry_info["report"]
            retry_bodies = retry_info["n_bodies_export"]
            retry_voxel_bodies = retry_info["n_bodies_voxel"]
            retry_export_ok = True
            res.topopt.setdefault("warnings", []).extend(retry_info["warnings"])
        except Exception as e:
            if _is_cancelled(cancelled):
                return _cancelled_candidate(best, "during_retry_export_verify")
            log.warning("infill.verify_export_retry.failed", error=str(e)[:200])
            retry_info = None
            retry_report, retry_bodies = verify, None
            retry_export_ok = False
            retry_candidate["warnings"].append(f"crisp export retry FEM failed ({str(e)[:80]})")
        retry_ratio, retry_fraction, _retry_gate_failed = _export_metrics(
            retry_candidate, retry_report, retry_bodies
        )

        def _retry_score(
            sf: float | None,
            bodies: int | None,
            ratio: float | None,
            fraction: float,
            solved: bool,
            envelope_ok: bool,
        ) -> tuple[int, int, int, int, int, float]:
            return (
                int(solved),
                int(envelope_ok),
                int(bodies is not None and bodies <= 1),
                int(ratio is None or ratio <= 3.0),
                int(target_infeasible or abs(fraction - target_volfrac) <= volume_tolerance),
                float(sf or 0.0),
            )

        if _retry_score(
            _gate_sf(retry_info, retry_report),
            retry_bodies,
            retry_ratio,
            retry_fraction,
            retry_export_ok,
            retry_candidate.get("output_envelope_ok") is not False,
        ) > _retry_score(
            sf_eff,
            n_bodies_export,
            sf_ratio,
            volfrac_export,
            export_solve_ok,
            best.get("output_envelope_ok") is not False,
        ):
            best = retry_candidate
            opt_mesh = best["mesh"]
            final_mask = best["mask"]
            rho = best["rho"]
            n_components = best["components"]
            extract_info = best["extract"]
            meta = best["meta"]
            n_surface_components = int(opt_mesh.body_count)
            verify_exp, n_bodies_export = retry_report, retry_bodies
            n_bodies_voxel = retry_voxel_bodies
            export_solve_ok = retry_export_ok
            exp_info = retry_info
            sf_eff = _gate_sf(exp_info, verify_exp)
            sf_ratio, volfrac_export = retry_ratio, retry_fraction
        res.topopt["retry"] = {
            "trigger": retry_reasons,
            "mode": "crisp_extraction",
            "volfrac_requested": target_volfrac,
            "volfrac_first": volfrac_first,
            "volfrac_second": volfrac_export,
            "sf_export_first": sf_export_first,
            "sf_export_second": retry_report.worst_safety_factor,
            "sf_verified_first": sf_verified_first,
            "sf_verified_second": (retry_info or {}).get("sf_verified"),
            "sf_ratio_first": ratio_first,
            "sf_ratio_second": sf_ratio,
            "n_bodies_export_first": bodies_first,
            "n_bodies_export_second": n_bodies_export,
        }
        res.timings["verify_retry_s"] = round(time.monotonic() - retry_started, 2)
    meta = best["meta"]
    meta["volfrac_min_feasible"] = round(target_min, 4) if target_min is not None else None
    meta["volfrac_max_feasible"] = round(target_max, 4) if target_max is not None else None
    meta["target_infeasible"] = target_infeasible
    res.topopt["infill"] = meta
    res.topopt["warnings"] += best["warnings"]
    res.timings["verify_s"] = round(time.monotonic() - t, 2)

    # --- homogenized infill verification: the same lattice layout solved
    # on class tensors — each interior cell carries C_H(rho)/A(rho) from the
    # unit-cell table instead of resolved strut geometry. Runs alongside
    # the resolved mask solve; a failure degrades to a warning only.
    if eff_homog and isinstance(best.get("fields"), dict) and best["fields"].get("d") is not None:
        try:
            from cadx.cad.fem.homogenize import infill_homog_elems, lattice_table
            from cadx.cad.fem.solver import _D

            tbl = lattice_table(
                str(meta.get("style", "gyroid_graded") or "gyroid_graded"),
                float(mat.poisson),
                cell_voxels=int(s.fem_homog_cell_voxels),
                octet=bool(eff_octet),
            )
            base_D = _D(mat.poisson, mat.aniso_z, mat.aniso_axis) if mat.aniso_z != 1.0 else None
            elem_class, class_D, micro_amp = infill_homog_elems(
                domain,
                np.asarray(best["fields"]["d"], dtype=float).ravel(),
                np.asarray(best["fields"]["interior"], dtype=bool),
                tbl,
                base_D=base_D,
            )
            homog_mask = domain.solid.ravel() & (
                np.asarray(best["fields"]["d"], dtype=float).ravel() > 0.0
            )
            if not homog_mask.any():
                raise ValueError("homogenized mask is empty")
            verify_h = run_fem_domain(
                domain,
                homog_mask,
                cases,
                mat_name,
                material=mat,
                out_dir=out_dir,
                anchors=anchors,
                keep_regions=keep_list,
                spsolve_dofs=spsolve_eff,
                pool_workers=s.exec_pool_workers,
                mem_budget_mb=mem_budget,
                suffix="_homog",
                source_mesh=opt_mesh,
                cancelled=cancelled,
                solver_backend=solver_backend,
                elem_class=elem_class,
                class_D=class_D,
                micro_amp=micro_amp,
                model="homogenized",
            )
            if _is_cancelled(cancelled):
                return _cancelled_candidate(best, "after_homogenized_verify")
            res.fem_homog = verify_h.model_dump()
            report_to_json(verify_h, out_dir / "fem" / "report_homog.json")
        except Exception as e:
            verify_h = None
            log.warning("infill.homog_verify.failed", error=str(e)[:200])
            res.topopt["warnings"].append(f"homogenized infill verification failed: {str(e)[:80]}")

    verification: dict[str, Any] = {
        "compliance_mask_j": [c.compliance_j for c in verify.cases],
        "compliance_export_j": [c.compliance_j for c in verify_exp.cases],
        "sf_mask": verify.worst_safety_factor,
        "sf_export": verify_exp.worst_safety_factor,
        # headline: worst case over the nominal/eroded/dilated export
        # realizations (the nominal export SF when the sweep was off)
        "sf_verified": (
            (exp_info or {}).get("sf_verified")
            if (exp_info or {}).get("sf_verified") is not None
            else verify_exp.worst_safety_factor
        ),
        "sf_realizations": (exp_info or {}).get("sf_realizations"),
        "sf_realization_detail": (exp_info or {}).get("sf_realization_detail"),
        "sf_verified_basis": (exp_info or {}).get("sf_verified_basis"),
        "enclosed_void_mm3": (exp_info or {}).get("enclosed_void_mm3"),
        "void_frac_of_solid": (exp_info or {}).get("void_frac_of_solid"),
        "n_sealed_cavities": (exp_info or {}).get("n_sealed_cavities"),
        "largest_void_mm3": (exp_info or {}).get("largest_void_mm3"),
        "sealed_cavities_mm3": (exp_info or {}).get("sealed_cavities_mm3"),
        "export_grid": (exp_info or {}).get("export_grid"),
        "bottleneck_mm2": (exp_info or {}).get("bottleneck_mm2"),
        "sf_floor": (exp_info or {}).get("sf_floor"),
        "load_path_ok": (exp_info or {}).get("load_path_ok"),
        "load_path_cases": (exp_info or {}).get("load_path_cases"),
        "sf_mesh_sensitive": (exp_info or {}).get("sf_mesh_sensitive"),
        # homogenized lattice model: member-scale SF on C_H(rho) class
        # tensors (sf_homog) and its micro stress-localized floor
        # (sf_micro) — None when the homogenized solve did not run
        "sf_homog": verify_h.worst_safety_factor if verify_h is not None else None,
        "sf_micro": getattr(verify_h, "worst_safety_factor_micro", None),
        "strut_align_deg": meta.get("strut_align_deg"),
        "sf_gap": (
            round(verify.worst_safety_factor - verify_exp.worst_safety_factor, 4)
            if verify.worst_safety_factor is not None and verify_exp.worst_safety_factor is not None
            else None
        ),
        "sf_ratio_mask_export": (
            round(verify.worst_safety_factor / max(verify_exp.worst_safety_factor, 1e-9), 4)
            if verify.worst_safety_factor is not None and verify_exp.worst_safety_factor is not None
            else None
        ),
        "sf_ratio": sf_ratio,
        "max_disp_export_mm": max((c.max_disp_mm for c in verify_exp.cases), default=None),
        "max_vm_export_mpa": max((c.max_vm_mpa for c in verify_exp.cases), default=None),
        "volfrac_mask": round(float(final_mask.sum()) / max(int(domain.solid.sum()), 1), 4),
        "volfrac_export": round(float(best["export_fraction"]), 4),
        "target_volume_error_pp": round(
            100.0 * (float(best["export_fraction"]) - target_volfrac), 2
        ),
        "meets_target_volume": abs(float(best["export_fraction"]) - target_volfrac)
        <= volume_tolerance,
        "target_infeasible": target_infeasible,
        "output_envelope_ok": best.get("output_envelope_ok", True),
        # physical bodies in the printed mesh (positive-volume components)
        "n_bodies_export": n_bodies_export,
        "shell_min_wall_mm": extract_info.get("shell_min_wall_mm"),
        "sealed_voids_filled_mm3": extract_info.get("sealed_voids_filled_mm3", 0.0),
        # body count of the re-voxelized FEM raster — a fusion below this
        # number means the export solve verified a fused phantom
        "n_bodies_voxel": n_bodies_voxel,
        # boundary surface components incl. inner cavity walls — diagnostic
        # only, >n_bodies_export is normal for hollow parts
        "n_surface_components": n_surface_components,
        "solver_used": "infill",
        "sf_baseline": verify_base.worst_safety_factor,
        "fixed_nodes_baseline": max(
            (c.reactions.get("fixed_nodes") for c in verify_base.cases), default=None
        ),
        "fixed_nodes_export": max(
            (c.reactions.get("fixed_nodes") for c in verify_exp.cases), default=None
        ),
    }
    span = float(np.linalg.norm(domain.pitch * np.asarray(domain.dims)))
    comps = [c for c in verification["compliance_export_j"] if c is not None]
    deg_valid, deg_warnings = _export_degeneracy(
        disp_mm=verification["max_disp_export_mm"],
        max_vm_mpa=verification["max_vm_export_mpa"],
        span_mm=span,
        yield_mpa=getattr(mat, "yield_mpa", None),
    )
    res.topopt["warnings"].extend(deg_warnings)
    nonphysical = bool(comps) and any(c <= 0 for c in comps)
    if nonphysical or not deg_valid:
        verification["valid"] = False
        if nonphysical:
            res.topopt["warnings"].append(
                "export FEM solve degenerate (unconstrained part or unresolved "
                "load regions) — sf_export not trustworthy"
            )
    elif not export_solve_ok:
        verification["valid"] = False
    elif best.get("output_envelope_ok") is False:
        verification["valid"] = False
        res.topopt["warnings"].append(
            "infill export remains outside the stated spec envelope — "
            "manufacturing verification is invalid"
        )
    elif verify_exp.worst_safety_factor is None:
        verification["valid"] = False
        res.topopt["warnings"].append(
            "no export FEM case produced a safety factor — sf_export is unverified, not safe"
        )
    elif max(int(n_components), int(n_bodies_export)) > 1:
        verification["valid"] = False
        res.topopt["warnings"].append(
            f"optimized part has {n_bodies_export} disconnected bodies "
            "in the export mesh — sf_export describes only the loaded island, "
            "not a single part"
        )
    elif (
        n_bodies_voxel is not None
        and n_bodies_export is not None
        and int(n_bodies_voxel) < int(n_bodies_export)
    ):
        # the FEM raster fused physically separate bodies into one component
        # — the solve describes a phantom that does not exist in the print
        verification["valid"] = False
        res.topopt["warnings"].append(
            f"voxel raster fused {n_bodies_export} physical bodies into "
            f"{n_bodies_voxel} FEM components — sf_export is not trustworthy"
        )
    elif (
        n_bodies_voxel is not None
        and n_bodies_export is not None
        and int(n_bodies_voxel) > int(n_bodies_export) + 1
    ):
        # the raster fragmented the one real body into many components —
        # thin members dropped out; the solve sees only the dominant island.
        # a warning, not a failure: the dominant island still carries the
        # load path honestly
        verification["valid"] = True
        res.topopt["warnings"].append(
            f"export re-voxelization fragmented into {n_bodies_voxel} "
            "components — thin lattice members fall below the FEM pitch; "
            "sf_export describes the dominant island only"
        )
    else:
        verification["valid"] = True
    if (exp_info or {}).get("load_path_ok") is False:
        # the geometric certificate found a case whose loads cannot reach
        # their fixtures through the export solid — structurally broken
        verification["valid"] = False
        failed_cases = [
            str(c.get("case_name") or c.get("case"))
            for c in ((exp_info or {}).get("load_path_cases") or [])
            if c.get("reachable") is False
        ]
        res.topopt["warnings"].append(
            "load-path certificate: a load case cannot reach its fixtures "
            "through the export solid — the part is structurally disconnected"
            + (f" (failing: {', '.join(failed_cases)})" if failed_cases else "")
        )
    if sf_ratio is not None and sf_ratio > 3.0:
        verification["valid"] = False
        res.topopt["warnings"].append(
            f"mask/export safety-factor ratio remains {sf_ratio:.1f}x after the "
            "bounded retry — the STL verification does not reproduce the mask result"
        )
    elif sf_ratio is not None and sf_ratio > 2.0:
        res.topopt["warnings"].append(
            f"mask/export safety-factor ratio is {sf_ratio:.1f}x — inspect the "
            "load path on the final STL; sf_export governs"
        )
    # sanity: removing material cannot add stiffness — a porous infill that
    # beats its solid baseline signals over-constrained fixtures or fused
    # phantom connectivity, not a stronger part
    sf_base = verify_base.worst_safety_factor
    if sf_base is not None and verify_exp.worst_safety_factor is not None:
        if verify_exp.worst_safety_factor >= 2.0 * sf_base:
            verification["valid"] = False
            res.topopt["warnings"].append(
                f"export safety factor {verify_exp.worst_safety_factor:.1f} is "
                f"{verify_exp.worst_safety_factor / max(sf_base, 1e-9):.1f}x the "
                f"solid baseline {sf_base:.1f} — fixture over-constraint "
                "suspected; verification invalid"
            )
        elif verify_exp.worst_safety_factor > sf_base:
            res.topopt["warnings"].append(
                f"export safety factor {verify_exp.worst_safety_factor:.1f} exceeds "
                f"the solid baseline {sf_base:.1f} — inspect the fixture "
                "constraint density before trusting it"
            )
    if not verification["meets_target_volume"] and not target_infeasible:
        verification["valid"] = False
        res.topopt["warnings"].append(
            f"final STL material fraction {best['export_fraction']:.1%} remains outside "
            f"the requested {target_volfrac:.1%} ±{volume_tolerance:.0%} after the "
            "bounded retry"
        )
    sf_min = getattr(opt, "safety_factor_min", None)
    # the requested safety floor is judged against the verified worst-case
    # realization (sf_verified -> sf_export -> sf_mask fallback chain)
    sf_gate = verification["sf_verified"]
    if sf_gate is None:
        sf_gate = verify_exp.worst_safety_factor
    if sf_gate is None:
        sf_gate = verify.worst_safety_factor
    if sf_min is not None and sf_gate is not None:
        verification["meets_safety_factor"] = bool(sf_gate >= sf_min)
    if verify.grid.get("dofs"):
        verification["dofs"] = verify.grid["dofs"]
    if verify.cases:
        verification["case_sfs_mask"] = [c.safety_factor for c in verify.cases]
    if verify_exp.cases:
        verification["case_sfs_export"] = [c.safety_factor for c in verify_exp.cases]

    from cadx.cad.dfm.fdm import overhang_ratio

    overhang = round(
        overhang_ratio(opt_mesh, str(getattr(opt, "print_direction", "+z") or "+z")) * 100, 1
    )
    verification["overhang_pct"] = overhang

    res.topopt.update(
        {
            "method": "infill",
            "solver": "infill",
            "iterations": 0,
            # the run has no optimizer loop — "converged" reports the
            # pipeline completed with a valid structure; target infeasibility
            # and failed export gates remain separate, explicit outcomes
            "status": (
                "infeasible_target"
                if target_infeasible
                else "degenerate"
                if not verification["valid"]
                else "converged"
            ),
            "converged": bool(verification["valid"] and not target_infeasible),
            "volfrac_requested": opt.target_volume_fraction,
            "volfrac_achieved": verification["volfrac_export"],
            "volfrac_mask": verification["volfrac_mask"],
            "volfrac_export": verification["volfrac_export"],
            "passive_frac": round(float(passive.sum()) / max(int(domain.solid.sum()), 1), 4),
            "material": mat.name,
            "sf_mask": verify.worst_safety_factor,
            "sf_export": verify_exp.worst_safety_factor,
            "sf_verified": verification["sf_verified"],
            "sf_realizations": verification["sf_realizations"],
            "sf_realization_detail": verification.get("sf_realization_detail"),
            "sf_verified_basis": verification.get("sf_verified_basis"),
            "enclosed_void_mm3": verification.get("enclosed_void_mm3"),
            "void_frac_of_solid": verification.get("void_frac_of_solid"),
            "n_sealed_cavities": verification.get("n_sealed_cavities"),
            "largest_void_mm3": verification.get("largest_void_mm3"),
            "sealed_cavities_mm3": verification.get("sealed_cavities_mm3"),
            "export_grid": verification.get("export_grid"),
            "sealed_voids_filled_mm3": extract_info.get("sealed_voids_filled_mm3", 0.0),
            "shell_min_wall_mm": extract_info.get("shell_min_wall_mm"),
            "void_pinned_cells_total": 0,
            "void_pin_iters": 0,
            "void_pin_sym_mirrored": 0,
            "void_pinning_applied": False,
            "void_pinned_mm3": 0.0,
            "bottleneck_mm2": verification["bottleneck_mm2"],
            "sf_floor": verification["sf_floor"],
            "load_path_ok": verification["load_path_ok"],
            "load_path_cases": verification.get("load_path_cases"),
            "sf_mesh_sensitive": verification["sf_mesh_sensitive"],
            "sf_homog": verification["sf_homog"],
            "sf_micro": verification["sf_micro"],
            "strut_align_deg": verification["strut_align_deg"],
            "min_member_mm": min_member_mm,
            "thin_fraction": round(thin_fraction, 4),
            "n_components": n_components,
            "n_bodies_export": n_bodies_export,
            "n_surface_components": n_surface_components,
            "extract": extract_info,
            "verification": verification,
            "rescue_islands": (meta.get("island_rescue") or {}).get("islands_found"),
            "rescue_dropped": (meta.get("island_rescue") or {}).get("dropped"),
            "rescue_dropped_frac": (meta.get("island_rescue") or {}).get("dropped_vol_frac"),
        }
    )

    if _is_cancelled(cancelled):
        return _cancelled_candidate(best, "before_final_density_persist")
    np.savez_compressed(
        opt_dir / "density.npz",
        density=rho.astype(np.float32),
        mask=final_mask,
        solid=domain.solid,
        passive=passive.reshape(domain.dims),
        voids=void_cells.reshape(domain.dims),
        origin=domain.origin,
        pitch=np.float64(domain.pitch),
        dims=np.asarray(domain.dims),
        flat_bottom=np.bool_(base_flat),
        kind=np.str_("infill"),
    )
    with contextlib.suppress(Exception):
        (opt_dir / "infill_meta.json").write_text(
            json.dumps(meta, indent=1, default=str), encoding="utf-8"
        )
        (opt_dir / "history.json").write_text(
            json.dumps({"history": []}, indent=1), encoding="utf-8"
        )
    report_to_json(verify, out_dir / "fem" / "report.json")
    report_to_json(verify_exp, out_dir / "fem" / "report_export.json")
    res.fem = verify.model_dump()
    res.fem_export = verify_exp.model_dump()

    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    if _is_cancelled(cancelled):
        return _cancelled_candidate(best, "before_final_exports")
    try:
        opt_mesh = export_stl(opt_mesh, stl_path)
    except Exception as e:
        res.error = f"failed to write optimized STL: {e}"
        return res
    res.exports = export_mesh_all(
        opt_mesh,
        out_dir / "exports",
        name=name,
        try_step=True,
        max_faces=s.mesh_max_faces_for_cad,
    )
    res.exports["dxf"] = None
    res.exports["svg"] = None
    res.analysis = mesh_analysis(opt_mesh)
    _repair_analysis_mesh(res.analysis, stl_path)
    # unconditional: in-memory watertight does not guarantee the shipped
    # files — flap/sliver debris can split off as phantom bodies; heavy
    # repair stays conditional inside repair_mesh_file
    _repair_export_meshes(res.exports, out_dir / "exports")
    res.mesh = {
        "pipeline": "infill",
        "triangles": len(opt_mesh.faces),
        "volume_mm3": round(float(opt_mesh.volume), 1),
        "solid_elements": int(final_mask.sum()),
        "pitch_mm": domain.pitch,
    }

    if _is_cancelled(cancelled):
        return _cancelled_pipeline_result(res, "before_render")
    t = time.monotonic()
    render_dir = out_dir / "renders"
    _rq_size, _rq_ghosts = _render_quality(res.detail_mode)
    render_paths = render_views(
        stl_path, render_dir, res.analysis.bbox_size, size=_rq_size, ghosts=_rq_ghosts
    )
    res.renders = {v: p.name for v, p in render_paths.items()}
    thumb = render_thumbnail(stl_path, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
    stats = [
        f"bbox: {res.analysis.bbox_size[0]:.1f} x {res.analysis.bbox_size[1]:.1f} "
        f"x {res.analysis.bbox_size[2]:.1f} mm",
        f"volume: {res.analysis.volume:.0f} mm^3",
        f"triangles: {res.analysis.triangle_count}",
        f"watertight: {res.analysis.watertight_after_repair}",
        f"SF mask/export: {verify.worst_safety_factor} / {verify_exp.worst_safety_factor}",
        f"infill: {meta.get('style')} / {meta.get('scope')} (occ {meta.get('infill_occupancy')})",
    ]
    sheet = make_contact_sheet(render_paths, out_dir / "contact_sheet.png", stats)
    res.contact_sheet = sheet.name if sheet else None
    res.timings["render_s"] = round(time.monotonic() - t, 2)
    if _is_cancelled(cancelled):
        return _cancelled_pipeline_result(res, "after_render")

    if not opt_mesh.is_watertight:
        res.topopt["warnings"].append(
            "optimized mesh is not watertight — inspect before manufacturing"
        )
    res.ok = True
    _attach_verify_spec(res, spec, ProjectSettings(), s, envelope_bound_only=True)
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res
def run_organic_pipeline(
    *,
    mesh_path: Path,
    out_dir: Path,
    organic: dict[str, Any] | None = None,
    voxel_mm: float | None = None,
    max_elements: int | None = None,
    anchors: list[dict] | None = None,
    name: str = "model",
    runtime: RuntimeSettings | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ExecResult:
    """Pseudo-topology-optimization: procedural organic restyling.

    No FEM/SIMP — a deterministic density field (OrganicParams.style/seed)
    is generated inside the part's voxel design domain and extracted through
    the shared TopOpt machinery (density_to_mesh: watertight, connectivity-
    pruned, min-member safe). Purely aesthetic — NOT structurally verified.

    Writes the density field in the same npz layout as the topopt pipeline
    so the threshold-resample endpoint can re-extract it.
    """
    from cadx.cad.fem.organic import organic_transform
    from cadx.llm.schemas import OrganicParams

    res = ExecResult()
    s = runtime or RuntimeSettings()
    t0 = time.monotonic()

    try:
        params = OrganicParams(**(organic or {}))
    except Exception as e:
        res.error = f"invalid organic params: {e}"
        return res
    try:
        mesh = load_triangle_mesh(mesh_path)
    except Exception as e:
        res.error = f"failed to load mesh for organic restyle: {e}"
        return res

    anchors = _scan_anchors(mesh, anchors)
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return res
    try:
        out = organic_transform(
            mesh,
            params,
            anchors=anchors,
            voxel_mm=voxel_mm,
            max_elements=max_elements,
            runtime=s,
        )
    except Exception as e:
        res.error = str(e)
        return res

    opt_mesh = smooth_preserve(out.mesh, iterations=4, pin_floor=is_flat_bottom(mesh))

    opt_dir = out_dir / "organic"
    opt_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        opt_dir / "density.npz",
        density=out.density.astype(np.float32),
        mask=out.mask,
        solid=out.domain.solid,
        passive=out.passive.reshape(out.domain.dims),
        voids=out.void.reshape(out.domain.dims),
        origin=out.domain.origin,
        pitch=np.float64(out.domain.pitch),
        dims=np.asarray(out.domain.dims),
        flat_bottom=np.bool_(is_flat_bottom(mesh)),
        kind=np.str_("organic"),
    )

    # --- artifacts: STL/GLB/3MF + STL copy used for renders/analysis -------
    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    try:
        opt_mesh = export_stl(opt_mesh, stl_path)
    except Exception as e:
        res.error = f"failed to write organic STL: {e}"
        return res
    res.exports = export_mesh_all(opt_mesh, out_dir / "exports", name=name, try_step=False)
    res.exports["dxf"] = None
    res.exports["svg"] = None
    res.analysis = mesh_analysis(opt_mesh)
    _repair_analysis_mesh(res.analysis, stl_path)
    # unconditional: in-memory watertight does not guarantee the shipped
    # files — flap/sliver debris can split off as phantom bodies; heavy
    # repair stays conditional inside repair_mesh_file
    _repair_export_meshes(res.exports, out_dir / "exports")
    res.mesh = {
        "pipeline": "organic",
        "triangles": len(opt_mesh.faces),
        "volume_mm3": round(abs(float(opt_mesh.volume)), 1),
        "solid_elements": int(out.mask.sum()),
        "pitch_mm": round(float(out.domain.pitch), 4),
        "organic": out.report,
    }

    # renders
    t = time.monotonic()
    render_dir = out_dir / "renders"
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return res
    _rq_size, _rq_ghosts = _render_quality(res.detail_mode)
    render_paths = render_views(
        stl_path, render_dir, res.analysis.bbox_size, size=_rq_size, ghosts=_rq_ghosts
    )
    res.renders = {v: p.name for v, p in render_paths.items()}
    thumb = render_thumbnail(stl_path, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
    stats = [
        f"bbox: {res.analysis.bbox_size[0]:.1f} x {res.analysis.bbox_size[1]:.1f} "
        f"x {res.analysis.bbox_size[2]:.1f} mm",
        f"volume: {res.analysis.volume:.0f} mm^3 (target {params.volfrac:.0%})",
        f"style: {params.style}  seed: {params.seed}  period: {params.period_mm:g} mm",
        f"triangles: {res.analysis.triangle_count}",
        f"watertight: {res.analysis.watertight_after_repair}",
        "aesthetic restyle — not structurally verified",
    ]
    sheet = make_contact_sheet(render_paths, out_dir / "contact_sheet.png", stats)
    res.contact_sheet = sheet.name if sheet else None
    res.timings["render_s"] = round(time.monotonic() - t, 2)

    if not opt_mesh.is_watertight:
        res.mesh["organic"]["warnings"] = [
            *res.mesh["organic"].get("warnings", []),
            "organic mesh is not watertight — inspect before manufacturing",
        ]
    res.ok = True
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res
def run_refine(
    *,
    mesh_path: Path,
    out_dir: Path,
    features: list[dict[str, Any]] | None = None,
    spec_features: list[dict[str, Any]] | None = None,
    anchors: list[dict[str, Any]] | None = None,
    keep_regions: list[dict[str, Any]] | None = None,
    load_cases: list[dict[str, Any]] | None = None,
    base_mesh_path: Path | None = None,
    restore_planes: bool = True,
    helical_threads: bool | None = None,
    auto_plane_restore: bool = False,
    auto_keep_anchors: bool = False,
    material: str = "",
    voxel_mm: float | None = None,
    max_elements: int | None = None,
    name: str = "model",
    runtime: RuntimeSettings | None = None,
    cancelled: Callable[[], bool] | None = None,
    spec: dict[str, Any] | None = None,
) -> ExecResult:
    """Post-TopOpt feature finishing: re-cut exact bores/threads/inserts and
    restore keep-region surfaces on a (usually optimized) mesh.

    Subtractive cuts use tessellated cutter solids + manifold booleans — the
    optimized surface is voxel-smoothed, so exact functional geometry is
    re-applied at measured anchors. Keep-region restore unions the ORIGINAL
    base surface back (Taubin shrinks pinned regions by ~a pitch).

    `features`   explicit selection (UI/agent) — overrides auto-resolution
    `spec_features` + `anchors` drive the deterministic auto pass
    `base_mesh_path` the pre-optimization part (keep restore source)
    `auto_keep_anchors` when keep_regions is empty, synthesize one anchor
        keep-region per fastener anchor (topopt auto-keep parity)
    """
    from cadx.cad.fem.analyze import report_to_json, run_fem
    from cadx.cad.fem.refine import (
        apply_refine,
        rebind_keep_hole_indices,
        refine_thread_render_features,
        refine_verify,
        resolve_refine_features,
    )
    from cadx.llm.schemas import LoadCase

    res = ExecResult()
    s = runtime or RuntimeSettings()
    t0 = time.monotonic()

    try:
        mesh = load_triangle_mesh(mesh_path)
    except Exception as e:
        res.error = f"failed to load mesh for refine: {e}"
        return res

    base_mesh = None
    if base_mesh_path is not None:
        try:
            base_mesh = load_triangle_mesh(base_mesh_path)
        except Exception as e:
            res.features.setdefault("warnings", []).append(
                f"base mesh unreadable — keep restore disabled: {e}"
            )

    feats, res_warnings = resolve_refine_features(
        anchors, spec_features, features, base_mesh=base_mesh
    )
    keep_regions = list(keep_regions or [])
    keep_regions = rebind_keep_hole_indices(
        keep_regions, spec_features or [], anchors or [], res_warnings
    )
    if auto_keep_anchors and not keep_regions and anchors:
        # worker-declared gap: TopOpt ran without explicit keep regions so
        # fastener islands may have been eaten — restore each anchor's
        # material ring from the base mesh
        keep_regions += _auto_keep_dicts(anchors, s.topopt_auto_keep_margin_mm)
    if auto_plane_restore and base_mesh is not None:
        from cadx.cad.fem.refine import detect_plane_regions

        plane_boxes, plane_warn = detect_plane_regions(base_mesh)
        res_warnings += plane_warn
        keep_regions += [{"kind": "_plane_box", "box": b.tolist()} for b in plane_boxes]
    out, rep = apply_refine(
        mesh,
        feats,
        base_mesh=base_mesh,
        keep_regions=keep_regions,
        anchors=anchors or [],
        restore_keeps=restore_planes and base_mesh is not None,
        # explicit job/spec value wins; only an omitted (None) request
        # falls back to the runtime default — per-feature helical=False
        # still wins inside apply_refine
        helical_threads=(
            helical_threads if helical_threads is not None else s.refine_helical_threads
        ),
    )
    warnings = res_warnings + rep.warnings

    if not rep.applied and not rep.restored_regions:
        res.error = "nothing to refine — no cuttable features or keep regions matched"
        res.features["refine"] = {
            "applied": [],
            "skipped": rep.skipped,
            "warnings": warnings,
        }
        return res

    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    try:
        out = export_stl(out, stl_path)
    except Exception as e:
        res.error = f"failed to write refined STL: {e}"
        return res

    res.features["refine"] = {
        "applied": rep.applied,
        "skipped": rep.skipped,
        "restored_regions": rep.restored_regions,
        "warnings": warnings,
    }

    # post-cut verification: an "applied" entry only means the boolean ran —
    # rescan the mesh and require a tunnel/dimple within tolerance of the
    # expected position+diameter (the standalone worker job already does
    # this; the optimize-session path previously trusted the self-report).
    # Passing the report dict itself records blind-probe records on it.
    verify_misses = refine_verify(
        out,
        res.features["refine"],
        anchors=anchors,
        request_features=features,
    )
    if verify_misses:
        res.features["refine"]["verify_misses"] = verify_misses
        for m in verify_misses:
            entry = m.get("entry") or {}
            warnings.append(
                f"feature '{entry.get('name')}': cut verified FAILED — no "
                f"tunnel/dimple within 3 mm / ±25% dia of expected position "
                f"({m.get('x')}, {m.get('y')}) Ø{m.get('dia')} mm"
            )
        res.features["refine"]["warnings"] = warnings

    _audit_refined_features(res, out)

    # FEM re-verify — the cuts remove material, the SF must be re-issued
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return res
    cases = [LoadCase(**c) for c in (load_cases or [])]
    if cases:
        # the verify solve runs on `out` — scan ITS anchors so bore/anchor
        # regions bind even when the parent carried none (imported STL,
        # missing analysis); passed-in anchors merge in, like run_fem_pipeline
        verify = run_fem(
            out,
            cases,
            material,
            voxel_mm=voxel_mm or s.fem_voxel_mm,
            max_elements=max_elements or s.fem_max_elements,
            out_dir=out_dir,
            anchors=_scan_anchors(out, anchors),
            spsolve_dofs=s.fem_spsolve_dofs,
            mem_budget_mb=float(resources.effective_exec_mem_mb(s)),
            pool_workers=s.exec_pool_workers,
            cancelled=cancelled,
        )
        report_to_json(verify, out_dir / "fem" / "report.json")
        res.fem = verify.model_dump()

    res.exports = export_mesh_all(
        out,
        out_dir / "exports",
        name=name,
        try_step=True,
        max_faces=s.mesh_max_faces_for_cad,
    )
    res.exports["dxf"] = None
    res.exports["svg"] = None
    res.analysis = mesh_analysis(out)
    _repair_analysis_mesh(res.analysis, stl_path)
    # unconditional: in-memory watertight does not guarantee the shipped
    # files — flap/sliver debris can split off as phantom bodies; heavy
    # repair stays conditional inside repair_mesh_file
    _repair_export_meshes(res.exports, out_dir / "exports")
    # but a heavy file repair reskins — erasing helical grooves the in-memory
    # postpass restored; guard re-probes + re-cuts them on the shipped files
    _guard_export_grooves(res.exports, out_dir / "exports", feats, rep)
    res.features["refine"]["warnings"] = res_warnings + rep.warnings
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return res
    _refine_analytic_step(res, out, feats, rep, out_dir, name, warnings, s, cancelled=cancelled)
    warnings = res.features["refine"]["warnings"]
    res.mesh = {
        "pipeline": "refine",
        "triangles": len(out.faces),
        "volume_mm3": round(float(out.volume), 1),
    }
    render_dir = out_dir / "renders"
    _rq_size, _rq_ghosts = _render_quality(res.detail_mode)
    render_paths = render_views(
        out,
        render_dir,
        res.analysis.bbox_size,
        size=_rq_size,
        thread_detail_features=refine_thread_render_features(out, rep.applied),
        ghosts=_rq_ghosts,
    )
    res.renders = {v: p.name for v, p in render_paths.items()}
    thumb = render_thumbnail(out, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
    if not out.is_watertight:
        warnings.append("refined mesh is not watertight — inspect before manufacturing")
    res.ok = True
    # canonical conformance report on the finished part — bound semantics:
    # the refined body is still the optimized envelope, not a CAD target
    _attach_verify_spec(res, spec, ProjectSettings(), s, envelope_bound_only=True)
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res

def _pad_cells(pitch: float,
    *,
    grow_mm,
)-> int:
    return math.ceil(grow_mm / max(float(pitch), 1e-6)) if grow_mm > 0 else 0


def _resolve_cases_on(dom: Any, warn: bool = True,
    *,
    anchors,
    cases,
    mat,
    max_cases,
    res,
):
    """Re-resolve the load cases + passive-pin node masks on `dom` —
    needed again on the coarse multiresolution grid (masks are
    domain-specific)."""
    return _resolve_topopt_cases(
        dom,
        cases,
        anchors,
        mat,
        max_cases,
        warn=warn,
        warnings_out=res.topopt.setdefault("warnings", []),
    )


def _cancelled_pass(out: dict[str, Any], phase: str,
    *,
    opt_dir,
    out_dir,
    res,
    storage,
)-> ExecResult:
    _mark_cancelled(res)
    res.topopt["cancelled_phase"] = phase
    result = out.get("result")
    if result is not None:
        density_path = opt_dir / f"partial_density_pass{out.get('pass_idx', 1)}.npz"
        np.savez_compressed(density_path, density=result.density.astype(np.float32))
        res.topopt["partial_density_relpath"] = storage.relpath(density_path)
        res.topopt["partial_optimizer"] = {
            "status": result.status,
            "iterations": result.iterations,
            "history": _compact_history(result.history),
            "warnings": result.warnings,
        }
    if out.get("verify") is not None:
        res.fem = out["verify"].model_dump()
    export_report = out.get("cancelled_report") or out.get("verify_export")
    if export_report is not None:
        res.fem_export = export_report.model_dump()
    opt_mesh = out.get("opt_mesh")
    if opt_mesh is not None:
        run_dir = out_dir / "_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        stl_path = run_dir / "model.stl"
        with contextlib.suppress(Exception):
            opt_mesh = export_stl(opt_mesh, stl_path)
            res.mesh = {
                "pipeline": "topopt",
                "partial": True,
                "stl_relpath": storage.relpath(stl_path),
                "triangles": len(opt_mesh.faces),
                "volume_mm3": round(float(opt_mesh.volume), 1),
            }
    return res


def _verify_one_pass_export(
    opt_mesh: Any, fem_suffix: str, p: TopOptParams, verify: Any,
    *,
    anchors,
    cancelled,
    cases,
    design_pitch,
    eff_robust,
    keep_list,
    mat,
    mat_name,
    max_elements,
    out_dir,
    res,
    s,
)-> tuple[dict[str, Any] | None, Any]:
    try:
        info = _verify_export_structure(
            candidate_mesh=opt_mesh,
            cases=cases,
            mat_name=mat_name,
            mat=mat,
            design_pitch=design_pitch,
            max_elements=max_elements or s.fem_max_elements,
            keep_list=keep_list,
            anchors=anchors,
            out_dir=out_dir,
            suffix=f"{fem_suffix}_export",
            s=s,
            solver_backend=p.solver_backend,
            cancelled=cancelled,
            verify_robust=eff_robust,
        )
        if not info.get("cancelled"):
            res.topopt.setdefault("warnings", []).extend(info["warnings"])
        return info, info.get("report", verify)
    except Exception as e:
        if _is_cancelled(cancelled):
            return {"cancelled": True, "phase": "during_export_verify"}, verify
        log.warning("topopt.verify_export.failed", error=str(e)[:200])
        res.topopt.setdefault("warnings", []).append(
            f"export FEM solve failed ({str(e)[:80]}) — sf_export "
            "duplicates the voxel-mask result, not the printed mesh"
        )
        return None, verify


def _one_pass(
    volfrac: float,
    pass_idx: int = 1,
    warm: np.ndarray | None = None,
    member_floor_mm: float | None = None,
    crisp_extraction: bool = False,
    vol_target_scale: float = 1.0,
    *,
    E0_arr,
    _opt_cancelled,
    anchors,
    base_flat,
    beso,
    cancelled,
    case_weights,
    cases,
    density_to_mesh,
    design_domain,
    dims_spec,
    domain,
    drop_island_components,
    eff_span,
    keep_list,
    mat,
    mat_name,
    mesh,
    min_gap,
    min_member,
    ndimage,
    opt,
    opt_dir,
    out_dir,
    params,
    passive,
    resolved,
    run_fem_domain,
    s,
    simp,
    spec,
    spsolve_eff,
    void_cells,
    x0,
    design_pitch,
    eff_robust,
    max_elements,
    res,
)-> dict[str, Any]:
    """One SIMP run at `volfrac` + extraction, smoothing and both FEM
    verifications. Returns the pieces needed for artifacts/retry.
    `warm` overrides the run-level x0 (retry passes warm-start from the
    first pass's converged field)."""
    out: dict[str, Any] = {"volfrac": volfrac}
    # pass-indexed render names: pass 1 keeps the historical names,
    # retries must not overwrite its evidence in out_dir/fem
    fem_suffix = "" if pass_idx <= 1 else f"_p{pass_idx}"
    effective_member = min_member if member_floor_mm is None else member_floor_mm
    p = dataclasses.replace(
        params,
        volfrac=volfrac,
        rmin_mm=max(params.rmin_mm, effective_member / 2),
    )
    out["min_member_mm"] = effective_member
    t = time.monotonic()
    # solver="beso" routes to the binary evolutionary optimizer — same
    # contract (domain, E0, nu, cases, passive, params) as simp(), but
    # the returned density field is natively 0/1
    opt_fn = beso if p.solver == "beso" else simp
    result = opt_fn(
        design_domain,
        E0_arr,
        mat.poisson,
        resolved,
        passive,
        p,
        passive_void=void_cells,
        case_weights=case_weights,
        x0=x0 if warm is None else warm,
        spsolve_dofs=spsolve_eff,
        cancelled=_opt_cancelled,
        progress_file=opt_dir / f"history_{volfrac:.2f}.json",
    )
    out["simp_s"] = round(time.monotonic() - t, 2)
    out["result"] = result
    out["params"] = p
    out["pass_idx"] = pass_idx
    if _is_cancelled(cancelled):
        out["cancelled_phase"] = "after_optimizer"
        return out
    # hard-kill ESO can settle on a local optimum worse than spreading
    # the mass evenly — the baseline check flags that verdict (a no-op
    # for non-BESO solvers; costs one K assembly + case solves)
    out.update(
        _uniform_baseline_check(
            p.solver,
            design_domain,
            E0_arr,
            mat.poisson,
            resolved,
            passive,
            p,
            case_weights=case_weights,
            spsolve_dofs=spsolve_eff,
            history=result.history,
        )
    )

    opt_mesh, final_mask, extract_warnings, n_components, extract_info = density_to_mesh(
        design_domain,
        result.density,
        threshold=p.threshold,
        passive_solid=passive,
        # declared void regions + outside-domain cells must stay empty —
        # min-member dilation / regularization would regrow them otherwise
        void_cells=void_cells.reshape(design_domain.dims) | ~design_domain.solid,
        min_member_mm=effective_member,
        min_gap_mm=min_gap,
        smooth_iterations=0 if crisp_extraction else s.topopt_smooth_iterations,
        extract_smooth=opt.extract_smooth and not crisp_extraction,
        sigma_px=0.0 if crisp_extraction else opt.extract_sigma_px,
        volume_preserve=opt.volume_preserve,
        vol_target_scale=vol_target_scale,
        fill_sealed_voids=opt.fill_sealed_voids,
        min_shell_mm=opt.min_shell_mm,
        fill_thin_walled_voids=True,
    )
    out["opt_mesh"] = opt_mesh
    out["final_mask"] = final_mask
    if _is_cancelled(cancelled):
        out["cancelled_phase"] = "after_density_extraction"
        return out
    # ndarray telemetry must not reach the persisted extract dict —
    # it kills the version-row JSON flush
    bridged3 = extract_info.pop("bridged_mask", None)
    out["n_components"] = n_components
    out["extract"] = extract_info
    # marching-cubes can leave zero-volume face-plane slivers — they
    # count as extra bodies downstream even though they carry no mass.
    # min_fraction mirrors the refine sliver floor (~1e-4 of the main
    # body): the default 0.5% would drop real sealed cavities
    opt_mesh, dropped_slivers = keep_significant_components(opt_mesh, min_fraction=1e-4)
    if dropped_slivers:
        extract_warnings.append(
            f"dropped {dropped_slivers} near-zero-volume mesh sliver components"
        )
    # pinned cells for smoothing + island rescue: keep/load/fix cells
    # plus any bridge-corridor cells the mask builder added — the
    # connectivity glue must not erode below raster thickness between
    # mask and export, and an island carrying pinned material is
    # structurally declared (never silently dropped)
    pin3 = passive.reshape(design_domain.dims).copy()
    if bridged3 is not None:
        pin3 |= np.asarray(bridged3, dtype=bool).reshape(design_domain.dims)
    # floor-pinned smoothing: keeps a flat manufacturing base flat.
    # Taubin shrinks thin members measurably per pass — on parts only a
    # few cells thick, cap iterations so the export keeps the volume
    # the mask FEM verified. Measured on the part span, not the
    # extracted extent — the candidate-to-candidate jitter around the
    # threshold must not flip the smoothing regime between passes.
    if not crisp_extraction:
        thin_cells = float(eff_span.min()) / max(design_domain.pitch, 1e-6)
        opt_mesh = smooth_preserve(
            opt_mesh,
            iterations=2 if thin_cells < 4.0 else 4,
            pin_floor=base_flat,
            # pin every vertex inside a passive (keep/fix/load) cell —
            # the surfaces the FEM verified must not drift under Taubin
            pin_mask=(
                design_domain.origin,
                design_domain.pitch,
                pin3,
            ),
        )
    opt_mesh, output_envelope, output_envelope_ok, envelope_warnings = (
        _clip_export_to_spec_envelope(
            opt_mesh, list(dims_spec) if dims_spec else None, label="smoothed export"
        )
    )
    extract_warnings.extend(envelope_warnings)
    # Restore + re-cut before FEM/export verification: sf_verified,
    # load_path_ok and feature evidence must describe the shipped mesh.
    opt_mesh, recut_report, finish_warnings = _post_extract_finish(
        opt_mesh,
        base_mesh=mesh,
        keep_regions=keep_list,
        anchors=anchors,
        spec_features=active_features(spec),
        recut_bores=bool(opt.recut_bores),
        restore_keeps=bool(opt.restore_planes),
        helical_threads=False,
    )
    out["opt_mesh"] = opt_mesh
    if _is_cancelled(cancelled):
        out["cancelled_phase"] = "after_post_extract_finish"
        return out
    extract_warnings.extend(finish_warnings)
    # last-mile sliver drop: the finish path (restore unions, reskin,
    # re-cut booleans) can mint zero-volume sheets that survive as a
    # phantom second body and trip the bodies gate on the shipped mesh
    opt_mesh, dropped_slivers = keep_significant_components(opt_mesh, min_fraction=1e-4)
    if dropped_slivers:
        extract_warnings.append(
            f"dropped {dropped_slivers} near-zero-volume post-finish sliver components"
        )
    # structural island rescue: the mask is connectivity-normalized, so
    # extra bodies here are extraction artifacts — sub-voxel pinch-offs
    # on thin corridors (the p16 export measured 2 bodies where the
    # mask verified 1). Drop islands below the fraction cutoff unless
    # they carry pinned keep/load/fix material; bigger islands stay
    # and keep the bodies-gate verdict honest
    opt_mesh, island_rescue = drop_island_components(
        opt_mesh,
        max_island_frac=0.02,
        pin_mask=(design_domain.origin, design_domain.pitch, pin3),
    )
    if island_rescue["dropped"]:
        extract_warnings.append(
            f"dropped {island_rescue['dropped']} disconnected island(s) "
            f"({island_rescue['dropped_vol_frac']:.4%} of export volume) "
            "— sub-voxel extraction pinch-off"
        )
    out["island_rescue"] = island_rescue
    if recut_report is not None:
        out["recut"] = recut_report
    if output_envelope is not None:
        out["output_envelope"] = output_envelope
    out["output_envelope_ok"] = output_envelope_ok
    out["extract_warnings"] = extract_warnings
    mask_volume = float(final_mask.sum()) * design_domain.pitch**3
    volfrac_mask = float(final_mask.sum()) / max(int(design_domain.solid.sum()), 1)
    extract_info["vol_error_pct"] = round(
        100.0 * (abs(float(opt_mesh.volume)) - mask_volume) / max(mask_volume, 1e-9), 3
    )
    out["volfrac_export"] = abs(float(opt_mesh.volume)) / max(
        float(design_domain.solid_count) * design_domain.pitch**3, 1e-9
    )
    out["volfrac_mask"] = volfrac_mask
    extract_info["vol_error_pp"] = round(100.0 * (out["volfrac_export"] - volfrac_mask), 3)
    out["crisp_extraction"] = crisp_extraction
    out["opt_mesh"] = opt_mesh
    out["final_mask"] = final_mask
    # Post-extraction region survival: a case region whose cells the
    # optimizer removed entirely can never carry its load — flag it now
    # so the agent sees the honest verdict early instead of reading a
    # min-cut failure after the FEM sweep. Pinned cells disappearing is
    # a separate tell — passive material is not designable.
    _record_lost_load_regions(design_domain, cases, anchors, final_mask, out, extract_warnings)
    passive_lost = int((passive.reshape(design_domain.dims) & ~final_mask).sum())
    if passive_lost:
        out["passive_cells_lost"] = passive_lost
        extract_warnings.append(
            f"{passive_lost} pinned keep/fix/load cell(s) removed in extraction "
            "— passive material is not designable"
        )
    # Surface components include nested cavity walls. The physical body
    # count is updated from the re-voxelized export solid below.
    out["n_surface_components"] = int(opt_mesh.body_count)
    out["n_bodies_export"] = None

    # thin-member metrics (printability feedback for the agent)
    edt = ndimage.distance_transform_edt(final_mask)
    inside = edt[final_mask]
    out["thin_fraction"] = float((inside <= 1.0).mean()) if len(inside) else 0.0
    out["min_member_mm"] = (
        round(2 * domain.pitch * float(np.percentile(inside, 5)), 2) if len(inside) else 0.0
    )

    t = time.monotonic()
    # opt_mesh is the extraction of this mask — same coordinate frame, so
    # the renderer can draw the smooth true surface instead of the voxel
    # hull
    verify = run_fem_domain(
        design_domain,
        final_mask,
        cases,
        mat_name,
        material=mat,
        out_dir=out_dir,
        anchors=anchors,
        keep_regions=keep_list,
        spsolve_dofs=spsolve_eff,
        pool_workers=s.exec_pool_workers,
        mem_budget_mb=float(resources.effective_exec_mem_mb(s)),
        suffix=fem_suffix,
        source_mesh=opt_mesh,
        cancelled=cancelled,
        solver_backend=params.solver_backend,
    )
    out["verify"] = verify
    out["verify_s"] = round(time.monotonic() - t, 2)

    # export verification: re-voxelize the smoothed mesh that gets printed —
    # at the requested (spec-snapped) pitch, not the possibly
    # auto-coarsened design pitch: the mesh is smooth, so verification
    # fidelity stays nominal and mask/export share one grid convention.
    # The shared helper adds the eroded/dilated realization sweep
    # (sf_verified), optional mesh-shift probe and load-path
    # certificate on top of the nominal export solve.
    info, verify_exp = _verify_one_pass_export(opt_mesh, fem_suffix, params, verify, anchors=anchors, cancelled=cancelled, cases=cases, design_pitch=design_pitch, eff_robust=eff_robust, keep_list=keep_list, mat=mat, mat_name=mat_name, max_elements=max_elements, out_dir=out_dir, res=res, s=s)
    if info and info.get("cancelled"):
        out["cancelled_phase"] = info.get("phase", "export_verify")
        out["cancelled_report"] = info.get("report")
        return out
    if info is not None:
        out["n_bodies_export"] = info["n_bodies_export"]
    else:
        out["export_solve_ok"] = False
    out["verify_export"] = verify_exp
    out["export_verify_info"] = info
    return out

def run_topopt_pipeline(  # noqa: C901
    *,
    mesh_path: Path,
    out_dir: Path,
    load_cases: list[dict[str, Any]],
    optimize: dict[str, Any] | None = None,
    material: str = "",
    voxel_mm: float | None = None,
    max_elements: int | None = None,
    anchors: list[dict] | None = None,
    name: str = "model",
    cancelled: Any = None,
    scan_anchors: bool = True,
    runtime: RuntimeSettings | None = None,
    spec_dims_mm: list[float | None] | None = None,
    spec: dict[str, Any] | None = None,
) -> ExecResult:
    """SIMP topology optimization on the part's voxelized design domain.

    Input mesh defines the design space (its solid voxels). keep/load/fix
    regions are pinned solid, void_regions pinned empty; everything else is
    optimized down to `volfrac`. Writes optimized mesh exports + density
    field + history + FEM verification into out_dir.

    Two verifications per pass: `sf_mask` on the exact voxel structure and
    `sf_export` on the re-voxelized smoothed export mesh — the latter is the
    structure that actually gets printed. When `sf_export` misses the target
    the pass is retried once at volfrac +0.1 and the better result wins.
    """
    from scipy import ndimage

    from cadx import storage
    from cadx.cad.fem.analyze import report_to_json, run_fem_domain
    from cadx.cad.fem.beso import beso
    from cadx.cad.fem.extract import density_to_mesh, drop_island_components
    from cadx.cad.fem.loadpath import loadpath_field, loadpath_spine_mask
    from cadx.cad.fem.preview import render_regions
    from cadx.cad.fem.regions import (
        anchor_entry_warnings,
        empty_region_warnings,
        keep_element_mask,
        material_cell_factors,
    )
    from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
    from cadx.cad.fem.voxelize import dilate_solid, grid_bounds_mm, snapped_pitch, voxelize
    from cadx.llm.schemas import LoadCase, OptimizeSpec, RegionSpec

    res = ExecResult()
    s = runtime or RuntimeSettings()
    opt = OptimizeSpec(**(optimize or {}))
    t0 = time.monotonic()

    # topopt_timeout_s bounds the OPTIMIZER wall-clock across all passes
    # (warm-start, SIMP/BESO/MMA and the safety retry). The agent-side
    # caller adds a tail allowance for extraction/export/verification on
    # top — without an internal bound a single run-away pass eats the
    # whole caller wait and the finished artifact lands past the wait
    # deadline (observed: a verified result discarded ~3min after the
    # caller timed out). The stop reuses the cancelled path: an honest
    # 'budget_exhausted' status while the partial density is still
    # extracted, verified and committed. Verification solves keep the raw
    # flag — they belong to the post-optimizer tail, not the budget.
    _opt_deadline = t0 + float(s.topopt_timeout_s) if s.topopt_timeout_s else None
    _opt_cancelled, _deadline_hit = _opt_cancel_pair(cancelled, _opt_deadline)

    try:
        mesh = load_triangle_mesh(mesh_path)
    except Exception as e:
        res.error = f"failed to load mesh for optimization: {e}"
        return res

    cases = [LoadCase(**c) for c in load_cases]
    if not cases:
        res.error = "topology optimization requires at least one load case"
        return res

    from cadx.cad.fem.materials import resolve_material

    mat_name = material or ""
    # aniso_material: opt-in FDM layer weakening — the layer-normal
    # stiffness factor (default 0.6, typical interlayer bond) acts along
    # the print direction's axis; runs through the material so BOTH the
    # optimizer and the verification solves see the same weakened D
    aniso_axis = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}.get(
        str(getattr(opt, "print_direction", "+z") or "+z"), 2
    )
    mat = resolve_material(
        mat_name,
        aniso_z=(float(opt.aniso_factor) if opt.aniso_factor else 0.6)
        if opt.aniso_material
        else None,
        aniso_axis=aniso_axis if opt.aniso_material else None,
    )

    # --- input normalization -------------------------------------------------
    # `spec_dims_mm` ([length_x, width_y, height_z], None = unconstrained
    # axis) travels inside the `optimize` payload dict (OptimizeSpec ignores
    # the extra key); mesh-only runs carry no spec and skip silently.
    try:
        res.topopt["base_mesh_relpath"] = storage.relpath(mesh_path)
    except Exception:
        res.topopt["base_mesh_relpath"] = str(mesh_path)
    res.topopt["input_bbox"] = [round(float(v), 3) for v in (mesh.bounds[1] - mesh.bounds[0])]
    res.topopt["spec_bbox"] = None
    res.topopt["input_normalized"] = False
    dims_spec = (
        spec_dims_mm
        if spec_dims_mm is not None
        else (optimize or {}).get("spec_dims_mm") or _spec_dims_from_dict(spec)
    )
    if dims_spec:
        mesh, norm_rep = normalize_mesh_to_spec_envelope(mesh, list(dims_spec))
        res.topopt["spec_bbox"] = norm_rep["spec_bbox"]
        res.topopt["input_normalized"] = norm_rep["input_normalized"]
        if norm_rep.get("warning"):
            res.topopt.setdefault("warnings", []).append(norm_rep["warning"])
        if norm_rep["input_normalized"]:
            norm_stl = out_dir / "_run" / "topopt_input.stl"
            try:
                norm_stl.parent.mkdir(parents=True, exist_ok=True)
                export_stl(mesh, norm_stl)
                res.topopt["normalized_input_relpath"] = storage.relpath(norm_stl)
            except Exception as e:
                log.warning("topopt.normalized_export_failed", error=str(e)[:200])
            log.info(
                "topopt.input_normalized",
                method=norm_rep.get("method"),
                input_bbox=norm_rep["input_bbox"],
                spec_bbox=norm_rep["spec_bbox"],
            )

    # scan anchors BEFORE the design pitch is chosen — the smallest anchored
    # bore caps the grid resolution (feature floor below); running the scan
    # only after voxelization could warn but never prevent the damage
    anchors = _scan_anchors(mesh, anchors, enabled=scan_anchors)
    res.topopt["anchors"] = anchors

    # --- design domain -------------------------------------------------------
    req_voxel = voxel_mm or s.fem_voxel_mm
    # spec-aware pitch snap: rasterize on a pitch that lands on an integer
    # cell count through the thinnest effective span — a 4 mm plate at a
    # 2.5 mm pitch otherwise becomes a 5 mm domain, so the mask FEM would
    # verify a part 25% thicker than the export the envelope clip produces
    # (observed ~30 pp volume loss / 5x sf_ratio divergence downstream)
    mesh_span = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
    eff_span = np.asarray(
        [
            float(d) if d else float(mesh_span[i])
            for i, d in enumerate(list(dims_spec or [None, None, None])[:3])
        ],
        dtype=float,
    )
    design_pitch, snap_warn = snapped_pitch(
        eff_span,
        req_voxel,
        min_cells=int(opt.min_cells_thinnest or 3),
        mesh_volume=abs(float(mesh.volume)) if mesh.is_watertight else 0.0,
        max_elements=max_elements or s.fem_max_elements,
    )
    if snap_warn:
        res.topopt.setdefault("warnings", []).append(snap_warn)
    # feature floor: an anchored bore must span ~3.5 cells or the export
    # extracts it undersized — snapped_pitch may land ABOVE the floor (its
    # thin-axis snap can round the cell count down), so clamp here AND in
    # the auto-coarsen loop below
    feature_floor = _feature_floor_pitch(anchors)
    floor_clamped = False
    if feature_floor is not None and design_pitch > feature_floor:
        design_pitch, floor_warn = _snap_pitch_under_floor(
            eff_span,
            feature_floor,
            min_cells=int(opt.min_cells_thinnest or 3),
            mesh_volume=abs(float(mesh.volume)) if mesh.is_watertight else 0.0,
            max_elements=max_elements or s.fem_max_elements,
        )
        floor_clamped = True
        if floor_warn:
            res.topopt.setdefault("warnings", []).append(floor_warn)
    res.topopt["domain_pitch"] = {
        "requested_mm": round(req_voxel, 4),
        "design_mm": round(design_pitch, 4),
        "snapped": abs(design_pitch - req_voxel) > 1e-6,
        "feature_floor_mm": round(feature_floor, 4) if feature_floor is not None else None,
        "feature_floor_clamped": floor_clamped,
    }
    # grow_mm: pad the grid so the optimizer can also ADD material beyond the
    # part silhouette — the pad ring stays void in `domain.solid` and becomes
    # designable space only in `design_domain` below
    grow_mm = float(getattr(opt, "grow_mm", 0.0) or 0.0)


    try:
        domain = voxelize(
            mesh,
            design_pitch,
            max_elements or s.fem_max_elements,
            pad_cells=_pad_cells(design_pitch, grow_mm=grow_mm),
        )
    except Exception as e:
        res.error = f"voxelization failed: {e}"
        return res
    # auto-coarsen the design grid until its system fits the direct solver —
    # SIMP re-solves it every iteration, so the SSOR-CG fallback above
    # fem_spsolve_dofs turns a minutes-long run into hours (observed ~300s/it
    # on a 143k-DOF bail handle vs ~2s/it direct). The printed part is the
    # smoothed *mesh*, not this grid: export verification below re-voxelizes
    # at the requested fine pitch, and the mask FEM stays consistent with
    # the grid it was optimized on.
    dofs_cap = int(getattr(s, "fem_spsolve_dofs", 120_000) or 120_000)
    coarsen_steps = 0
    floor_hit = False
    while 3 * domain.node_count > dofs_cap and coarsen_steps < 4:
        new_pitch = domain.pitch * 1.4
        if feature_floor is not None and new_pitch > feature_floor:
            if domain.pitch >= feature_floor - 1e-9:
                # already on the floor — coarsening further would rasterize
                # the smallest anchored bore away; hold the grid and let the
                # solver fall back to CG/AMG on the large system instead
                floor_hit = True
                break
            new_pitch = feature_floor  # last allowed step, right onto the floor
        coarsen_steps += 1
        try:
            domain = voxelize(
                mesh,
                new_pitch,
                max_elements or s.fem_max_elements,
                pad_cells=_pad_cells(new_pitch, grow_mm=grow_mm),
            )
        except Exception:
            break
    if coarsen_steps or floor_hit:
        res.topopt["auto_coarsen"] = {
            "requested_pitch_mm": round(req_voxel, 4),
            "design_pitch_mm": round(float(domain.pitch), 4),
            "dofs": int(3 * domain.node_count),
            "coarsen_steps": coarsen_steps,
            "feature_floor_mm": round(feature_floor, 4) if feature_floor is not None else None,
            "feature_floor_hit": floor_hit,
        }
        log.info(
            "topopt.auto_coarsen",
            pitch=round(float(domain.pitch), 4),
            dofs=int(3 * domain.node_count),
            feature_floor_hit=floor_hit,
        )
    if 3 * domain.node_count > dofs_cap:
        if floor_hit:
            res.topopt.setdefault("warnings", []).append(
                f"design grid held at the anchor feature floor "
                f"{float(domain.pitch):.2f}mm — {int(3 * domain.node_count)} DOFs "
                f"stay above the direct-solve cutoff {dofs_cap}; the solver "
                "engages CG/AMG on the large system, expect slow iterations "
                "(raise fem_spsolve_dofs for a direct factorization)"
            )
        else:
            res.topopt.setdefault("warnings", []).append(
                f"design grid {int(3 * domain.node_count)} DOFs exceeds the direct-solve "
                f"cutoff {dofs_cap} after coarsening — CG fallback, expect slow iterations"
            )
    d_lo, d_hi = grid_bounds_mm(domain)
    res.topopt["domain_bbox"] = [
        [round(float(v), 3) for v in d_lo],
        [round(float(v), 3) for v in d_hi],
    ]
    # domain vs the effective (spec-or-mesh) spans — positive means the
    # rasterized domain sticks out beyond the part; on a stated spec axis
    # the export clip then removes material the mask FEM verified
    span_cells = np.asarray(domain.dims, dtype=float) - 2 * domain.pad_cells
    overshoot = domain.pitch * span_cells - eff_span
    res.topopt["domain_spec_overshoot_mm"] = [round(float(v), 3) for v in overshoot]
    if dims_spec:
        bad_axes = [
            "xyz"[i] for i, d in enumerate(dims_spec) if d and overshoot[i] > 0.5 * domain.pitch
        ]
        if bad_axes:
            res.topopt.setdefault("warnings", []).append(
                f"design domain exceeds the spec envelope on "
                f"{','.join(bad_axes)} by more than half a cell — the export "
                "clip removes material the mask FEM verified"
            )

    # two domains on ONE shared grid when grow_mm > 0: `part_domain.solid`
    # is the physical part (regions/anchors/passive masks resolve on it — a
    # face region must match the real surface, never padded or dilated
    # cells), `design_domain.solid` additionally holds the dilated ring the
    # optimizer may fill. Identical dims/origin/pitch → every mask and DOF
    # vector indexes both domains without remapping.
    part_domain = domain
    design_domain = domain
    if grow_mm > 0:
        grow_cells = math.ceil(grow_mm / domain.pitch)
        if grow_cells > 0:
            design_domain = dataclasses.replace(domain, solid=dilate_solid(domain, grow_cells))
        res.topopt["design_domain"] = {
            "grow_mm": round(grow_mm, 3),
            "part_cells": int(part_domain.solid_count),
            "design_cells": int(design_domain.solid_count),
        }
        log.info(
            "topopt.design_domain",
            grow_mm=grow_mm,
            part_cells=int(part_domain.solid_count),
            design_cells=int(design_domain.solid_count),
        )

    # predictive memory budget + solver-tier gate — an OOM kill mid-SIMP
    # loses the whole job, so the direct tier only runs when the predicted
    # factorization fits; otherwise the loop degrades to CG/AMG and only a
    # CG solve that still cannot fit is refused outright
    from cadx.cad.fem.voxelize import estimate_fem_memory_mb

    mem_budget = float(resources.effective_exec_mem_mb(s))
    n_eff = min(len(cases) * 2, s.fem_max_effective_cases)
    # the tier gate counts SIMULTANEOUS factors — the case-solve ctx_cache
    # holds one per load case, not per effective case
    spsolve_eff, tier_info = _solver_tier_gate(domain, len(cases), s, mem_budget)
    res.topopt["solver_tier"] = tier_info
    if warn := _tier_warning(tier_info):
        res.topopt.setdefault("warnings", []).append(warn)
    if mem_budget:
        est_mb = estimate_fem_memory_mb(
            domain, n_eff, direct=tier_info["tier"] == "direct", n_factors=len(cases)
        )
        if est_mb > mem_budget * 0.75:
            res.error = (
                f"predicted FEM memory ~{est_mb:.0f} MB exceeds the executor "
                f"budget ({mem_budget:.0f} MB) — raise fem_voxel_mm or "
                "exec_mem_limit_mb"
            )
            return res

    # --- resolve load cases (with direction-cone expansion for robustness) ---
    max_cases = s.fem_max_effective_cases


    resolved, case_weights, fixed_node_masks, load_node_masks, unresolved = _resolve_cases_on(
        domain,
        anchors=anchors,
            cases=cases,
            mat=mat,
            max_cases=max_cases,
            res=res,
    )
    if unresolved:
        res.error = (
            "load-case regions did not resolve on the geometry — "
            + "; ".join(unresolved[:4])
            + ". The declared fix/load feature is missing from the solid or its "
            "anchor is unresolvable; fix the geometry (feature placement) before "
            "optimizing — a result optimized under wrong constraints is unusable."
        )
        return res
    if not resolved:
        res.error = "no solvable load case — check fixed/load regions"
        return res

    # --- non-design zones ----------------------------------------------------
    # interface protection: every fastener anchor (declared bore/tap/insert/
    # bearing seat) gets a bounded keep ring + channel void + blind-floor
    # slab via interface_regions. keep_mode="auto" protects all of them;
    # "manual" + the runtime flag still protects the SPEC-DECLARED ones
    # (scan extras stay designable — the declared interface contract is not
    # negotiable).
    keep_list = list(opt.keep_regions)
    n_auto_keeps = 0
    auto_interface_voids: list[RegionSpec] = []
    if anchors and (opt.keep_mode == "auto" or s.topopt_auto_keep):
        from cadx.cad.fem.regions import interface_regions

        margin = opt.auto_keep_margin_mm or s.topopt_auto_keep_margin_mm
        ik, iv = interface_regions(anchors, margin, spec_only=(opt.keep_mode != "auto"))
        keep_list += ik
        auto_interface_voids = iv
        n_auto_keeps = len(ik)
    res.topopt["auto_keep_applied"] = bool(n_auto_keeps)
    res.topopt.setdefault("warnings", []).extend(
        empty_region_warnings(part_domain, keep_list, anchors)
    )
    res.topopt.setdefault("warnings", []).extend(
        empty_region_warnings(part_domain, list(opt.void_regions), anchors)
    )
    res.topopt.setdefault("warnings", []).extend(anchor_entry_warnings(part_domain, anchors))
    keep_cells = keep_element_mask(part_domain, keep_list, anchors)
    # anchor keep rings around every bore a case's load/fix region resolves
    # on — a `bore`/`hole` load region must keep its material ring even when
    _pitch_feature_warn(res, domain, anchors)
    # the agent forgot an explicit keep. Lazy: the helper may not exist yet
    # on this branch — degrade to the declared keeps only.
    try:
        from cadx.cad.fem.regions import case_anchor_keep_mask

        keep_cells |= np.asarray(
            case_anchor_keep_mask(part_domain, cases, anchors), dtype=bool
        ).ravel()
    except Exception:
        pass
    void_list = list(opt.void_regions) + auto_interface_voids
    # voids resolve on the DESIGN domain — grown ring cells inside a bore
    # channel must stay empty too (part_domain only sees original solid)
    void_cells = keep_element_mask(design_domain, void_list, anchors)
    # passive pins run on the DESIGN domain: grown cells sharing a
    # fix/load node stay material — the load-introduction patch can use
    # the extra design space (masks share the grid, resolved on the part)
    passive = passive_solid_mask(design_domain, [keep_cells], fixed_node_masks, load_node_masks)

    # preflight: the pinned keep/load/fixed floor alone exceeding the
    # requested volfrac makes the target unreachable regardless of what
    # SIMP finds — refuse before voxel FEM setup instead of burning a run
    # (observed: agent's keeps grew 0.22 -> 0.62 while the target stayed
    # 0.40; every pass still ran ~20-27 iterations)
    # volfrac is measured on the optimization domain, which includes the
    # grow_mm dilation ring; keep the preflight denominator consistent with
    # simp() or a feasible grown design can be refused against the smaller
    # original-part volume.
    nsolid = int(design_domain.solid.sum())
    passive_frac = float(passive.sum()) / nsolid if nsolid else 0.0
    volfrac = opt.target_volume_fraction or s.topopt_default_volfrac
    res.topopt["passive_frac"] = round(passive_frac, 4)
    res.topopt["min_feasible_volfrac"] = round(passive_frac, 4)
    # per-source passive breakdown: which keep region / fix-load patch owns
    # the pinned floor — the first thing to check when the preflight refuses
    keep_in_passive = int((keep_cells & passive).sum())
    res.topopt["passive_breakdown"] = {
        "keep_cells": keep_in_passive,
        "fixed_load_cells": int(passive.sum()) - keep_in_passive,
        "void_cells": int(void_cells.sum()),
        "design_cells": nsolid,
        # cells the auto interface protection pinned — the answer to "did
        # the declared interfaces get their ring", distinct from the case
        # regions and hand-declared keeps
        "auto_interface_cells": int(
            keep_element_mask(
                part_domain,
                [r for r in keep_list if str(r.description or "").startswith("auto-keep")],
                anchors,
            ).sum()
        )
        if n_auto_keeps
        else 0,
        "per_keep_region": [
            {
                "kind": r.kind,
                "axis": r.axis or None,
                "hole_index": r.hole_index,
                "cells": int(keep_element_mask(part_domain, [r], anchors).sum()),
            }
            for r in keep_list
        ],
    }
    if passive_frac > volfrac:
        fl_cells = res.topopt["passive_breakdown"]["fixed_load_cells"]
        if fl_cells > keep_in_passive:
            hint = (
                f" Fixed/load case regions own {fl_cells} of {int(passive.sum())}"
                " pinned cells — check for whole-face plane/face/skin fixtures"
                " or loads pinning voxel layers (a g_load belongs on 'all'; a"
                " bolted mount belongs on its hole anchors, not a face encastre)."
            )
        else:
            tops = sorted(
                res.topopt["passive_breakdown"]["per_keep_region"],
                key=lambda d: -d["cells"],
            )[:3]
            hint = (
                " Largest keep regions: "
                + ", ".join(
                    f"{d['kind']}#{d['hole_index'] or d['axis']}={d['cells']}" for d in tops
                )
                + "."
            )
        res.error = (
            f"passive keep/load/fixed cells cover {passive_frac:.0%} of the optimization domain — "
            f"above the requested volume fraction {volfrac:.0%}; the target is "
            "unreachable regardless of optimization. Reduce keep/auto-keep "
            "regions or set target_volume_fraction >= "
            f"{min(0.95, passive_frac + 0.02):.2f}"
        ) + hint
        res.topopt["infeasible_preflight"] = True
        return res
    if passive_frac > 0.3:
        res.topopt.setdefault("warnings", []).append(
            f"keep/load/fixed regions cover {passive_frac:.0%} of the optimization domain — "
            f"only {1.0 - passive_frac:.0%} is free for optimization"
        )
    ndesign = int((design_domain.solid.ravel() & ~passive & ~void_cells.ravel()).sum())
    if nsolid and ndesign < 0.05 * nsolid:
        res.topopt.setdefault("warnings", []).append(
            "very few free design cells (<5%) — keep regions cover most of the part"
        )

    # region-selection previews so the agent/user can see where forces land
    res.topopt["regions_png"] = render_regions(
        part_domain, cases, keep_list, out_dir / "fem", anchors, source_mesh=mesh
    )
    if _is_cancelled(cancelled):
        return _cancelled_pipeline_result(res, "after_region_render")

    min_member = float(opt.min_member_mm or s.topopt_min_member_mm or 0.0)
    min_gap = float(opt.min_gap_mm or min_member or 0.0)
    params = TopOptParams(
        volfrac=opt.target_volume_fraction or s.topopt_default_volfrac,
        # the filter radius also acts as the printable-feature floor
        rmin_mm=max(s.topopt_default_rmin_mm, min_member / 2),
        max_iter=opt.max_iterations or s.topopt_max_iterations,
        threshold=float(opt.density_threshold or 0.5),
        filter=str(getattr(opt, "filter", None) or "helmholtz+heaviside"),
        robust=bool(getattr(opt, "robust", False)),
        robust_delta_eta=float(getattr(opt, "robust_delta_eta", 0.25) or 0.25),
        robust_designs=str(getattr(opt, "robust_designs", "edi") or "edi"),
        symmetry=getattr(opt, "symmetry", None) or None,
        self_supporting=bool(getattr(opt, "self_supporting", False)),
        print_direction=str(getattr(opt, "print_direction", "+z") or "+z"),
        penal=float(opt.penal or 3.0),
        penal_init=opt.penal_init,
        volfrac_init=opt.volfrac_init,
        solver=str(getattr(opt, "solver", "oc") or "oc"),
        solver_backend=str(getattr(s, "fem_solver_backend", "auto") or "auto"),
        beso_er=float(getattr(opt, "beso_er", 0.02) or 0.02),
        mma_final_beta=opt.mma_final_beta,
        loadpath_corridor_mm=opt.loadpath_corridor_mm,
        volfrac_ramp_iters=opt.volfrac_ramp_iters or 40,
        aniso=(mat.aniso_z, float(mat.aniso_axis)) if mat.aniso_z != 1.0 else None,
        case_objective=str(getattr(opt, "case_objective", "weighted") or "weighted"),
        max_displacement_mm=opt.max_displacement_mm,
        max_stress_mpa=opt.max_stress_mpa,
        prevent_sealed_voids=bool(opt.prevent_sealed_voids),
        sealed_void_check_iters=opt.sealed_void_check_iters,
    )
    # hardened export verification: eroded/nominal/dilated realization
    # sweep whose worst case is reported as sf_verified (see
    # _verify_export_structure); the per-run override follows the runtime
    # setting when unset
    eff_robust = (
        opt.verify_robust if opt.verify_robust is not None else bool(s.fem_verify_realizations)
    )

    # robust formulation multiplies the FEM solves per iteration (one per
    # eta realization per load case) — warn when the effective per-iteration
    # solve count blows past the case budget rather than refusing outright
    if params.robust:
        n_designs = 1 if params.robust_designs == "e" else 3
        n_solves = len(resolved) * n_designs
        if n_solves > s.fem_max_effective_cases:
            res.topopt["warnings"].append(
                f"robust formulation runs {n_solves} FEM solves per iteration "
                f"({len(resolved)} cases x {n_designs} designs > "
                f"fem_max_effective_cases={s.fem_max_effective_cases}) — "
                "expect ~3x runtime; consider robust_designs='e' or fewer cases"
            )

    # insert/material zones: keep-regions with `material` change E per cell
    E0_arr = mat.elastic_modulus_mpa * material_cell_factors(
        domain, keep_list, anchors, mat.elastic_modulus_mpa
    )

    # optional warm start from a previous run's density field
    x0 = None
    warm_src: str | None = None
    if opt.warm_density_relpath:
        try:
            with np.load(storage.resolve(opt.warm_density_relpath)) as warm:
                x0 = np.asarray(warm["density"], dtype=float)
                if x0.size != domain.cell_count:
                    # grid mismatch — resample the stored field onto the
                    # design grid instead of dropping it. Covers guided-infill
                    # guide_density.npz (~2x the INFILL pitch, which is not a
                    # fixed ratio vs this domain), multires coarse fields and
                    # same-pitch grids with shifted origins/dims. A
                    # coarse binary (beso) field resamples into a gray seed —
                    # still a useful macro layout.
                    x0 = None
                    try:
                        pitch_w = float(warm["pitch"])
                        ratio = pitch_w / float(domain.pitch)
                        if 0.25 <= ratio <= 8.0:
                            dom_w = _GridRef(
                                origin=np.asarray(warm["origin"], dtype=float),
                                pitch=pitch_w,
                                dims=tuple(int(v) for v in warm["dims"]),
                            )
                            x0 = _upsample_density(
                                np.asarray(warm["density"], dtype=float).reshape(dom_w.dims),
                                dom_w,
                                design_domain,
                            )
                            warm_src = f"resampled@{ratio:.2f}x:{opt.warm_density_relpath}"
                    except Exception:
                        pass
                    if x0 is None:
                        res.topopt["warnings"].append(
                            "warm-start density has a different grid — ignored"
                        )
                else:
                    warm_src = opt.warm_density_relpath
        except Exception as e:
            res.topopt["warnings"].append(f"warm-start density unreadable: {e}")
    elif str(getattr(opt, "warmstart", "") or "") == "loadpath":
        # deterministic Dijkstra corridor field (load centroids -> fixed
        # regions) — skips SIMP's broad-search phase; on the DESIGN domain
        # so corridors may route through grow_mm space too
        try:
            x0 = loadpath_field(
                design_domain,
                resolved,
                passive_solid=passive,
                passive_void=void_cells,
                corridor_mm=params.loadpath_corridor_mm,
            )
            warm_src = "loadpath"
        except Exception as e:
            res.topopt["warnings"].append(f"loadpath warm start failed: {e}")
        # the corridor field is only an x0 hint — SIMP can re-thin seeded
        # cells and sever the declared load path (arm r12: flange eaten to
        # floating pin tabs → load_path_ok=false). Pin each case's
        # near-geodesic spine as passive material so the resolved path
        # stays connected by construction.
        try:
            spine = loadpath_spine_mask(
                design_domain,
                resolved,
                corridor_mm=params.loadpath_corridor_mm,
            )
            spine_new = spine & ~passive
            if spine_new.any():
                passive |= spine
                res.topopt["loadpath_spine_cells"] = int(spine.sum())
                res.topopt["passive_breakdown"]["loadpath_spine_cells"] = int(spine_new.sum())
                nsolid = int(design_domain.solid.sum())
                new_frac = float(passive.sum()) / nsolid if nsolid else 0.0
                res.topopt["passive_frac"] = round(new_frac, 4)
                if volfrac and new_frac >= volfrac:
                    res.topopt["warnings"].append(
                        f"loadpath spine pins {new_frac:.0%} of the domain — "
                        f"above the {volfrac:.0%} volume target; the run is "
                        "feasible only near the pinned floor"
                    )
        except Exception as e:
            res.topopt["warnings"].append(f"loadpath spine pinning failed: {e}")
    else:
        # no explicit warm start — reuse a previous run's density.npz when its
        # grid is identical (agent sessions chain SIMP runs on one base mesh)
        x0, warm_src = _auto_warm_density(out_dir, domain)
    if warm_src:
        # report a storage-relative path — absolute paths leak DATA_DIR layout
        if Path(warm_src).is_absolute():
            with contextlib.suppress(Exception):
                warm_src = storage.relpath(warm_src)
        res.topopt["warm_start"] = warm_src

    # multiresolution: run a coarse SIMP at 2x pitch first, upsample the
    # density and warm-start the fine grid — converges ~3x faster on big parts.
    # Default-on (runtime `topopt_multires` flag): OptimizeSpec.multires is
    # schema-False even when unset, so a serialized spec dump would read as
    # an opt-out — the worker pops falsy values and only an *explicit*
    # multires=false in the request (tool arg / API payload) disables it here.
    multires_min = int(getattr(s, "topopt_multires_min_cells", 4000) or 0)
    multires_on = bool(opt.multires) or (optimize or {}).get("multires") is None
    if (
        getattr(s, "topopt_multires", True)
        and multires_on
        and x0 is None
        and domain.solid_count > multires_min
    ):
        t_mr = time.monotonic()
        try:
            coarse = voxelize(
                mesh,
                domain.pitch * 2.0,
                max_elements or s.fem_max_elements,
                pad_cells=_pad_cells(domain.pitch * 2.0, grow_mm=grow_mm),
            )
            # same two-domain split on the coarse grid when grow_mm is on
            coarse_design = coarse
            if design_domain is not part_domain:
                gcells_c = math.ceil(grow_mm / coarse.pitch)
                if gcells_c > 0:
                    coarse_design = dataclasses.replace(
                        coarse, solid=dilate_solid(coarse, gcells_c)
                    )
            if coarse.cell_count >= domain.cell_count // 2:
                res.topopt["warnings"].append("multires: coarse grid not much coarser — skipped")
            else:
                res_c, w_c, fix_c, load_nm_c, _unres_c = _resolve_cases_on(coarse, warn=False, anchors=anchors, cases=cases, mat=mat, max_cases=max_cases, res=res)
                if not res_c:
                    res.topopt["warnings"].append(
                        "multires: no solvable case on the coarse grid — skipped"
                    )
                else:
                    keep_c = keep_element_mask(coarse, keep_list, anchors)
                    void_c = keep_element_mask(coarse, void_list, anchors)
                    passive_c = passive_solid_mask(coarse_design, [keep_c], fix_c, load_nm_c)
                    E0_c = mat.elastic_modulus_mpa * material_cell_factors(
                        coarse, keep_list, anchors, mat.elastic_modulus_mpa
                    )
                    pc = TopOptParams(
                        volfrac=params.volfrac,
                        rmin_mm=params.rmin_mm,
                        max_iter=max(10, params.max_iter // 3),  # coarse pass is cheap
                        threshold=params.threshold,
                        filter="cone",  # no projection on the coarse pass
                        penal=params.penal,
                        solver_backend=params.solver_backend,
                        aniso=params.aniso,
                        self_supporting=params.self_supporting,
                        print_direction=params.print_direction,
                        symmetry=params.symmetry,  # keep the warm-start symmetric too
                    )
                    warm_r = simp(
                        coarse_design,
                        E0_c,
                        mat.poisson,
                        res_c,
                        passive_c,
                        pc,
                        passive_void=void_c,
                        case_weights=np.asarray(w_c),
                        spsolve_dofs=spsolve_eff,
                        cancelled=_opt_cancelled,
                    )
                    x0 = _upsample_density(
                        warm_r.density.reshape(coarse.dims), coarse, design_domain
                    )
                    res.topopt["multires"] = {
                        "coarse_pitch_mm": coarse.pitch,
                        "coarse_cells": int(coarse.solid_count),
                        "coarse_iterations": warm_r.iterations,
                        "seconds": round(time.monotonic() - t_mr, 2),
                    }
        except Exception as e:
            res.topopt["warnings"].append(f"multires coarse pass failed: {e}")

    opt_dir = out_dir / "topopt"
    opt_dir.mkdir(parents=True, exist_ok=True)
    base_flat = is_flat_bottom(mesh)




    p1 = _one_pass(
        params.volfrac,
        pass_idx=1,
        E0_arr=E0_arr,
        _opt_cancelled=_opt_cancelled,
        anchors=anchors,
        base_flat=base_flat,
        beso=beso,
        cancelled=cancelled,
        case_weights=case_weights,
        cases=cases,
        density_to_mesh=density_to_mesh,
        design_domain=design_domain,
        dims_spec=dims_spec,
        domain=domain,
        drop_island_components=drop_island_components,
        eff_span=eff_span,
        keep_list=keep_list,
        mat=mat,
        mat_name=mat_name,
        mesh=mesh,
        min_gap=min_gap,
        min_member=min_member,
        ndimage=ndimage,
        opt=opt,
        opt_dir=opt_dir,
        out_dir=out_dir,
        params=params,
        passive=passive,
        resolved=resolved,
        run_fem_domain=run_fem_domain,
        s=s,
        simp=simp,
        spec=spec,
        spsolve_eff=spsolve_eff,
        void_cells=void_cells,
        x0=x0,
        design_pitch=design_pitch,
        eff_robust=eff_robust,
        max_elements=max_elements,
        res=res,
    )
    if p1.get("cancelled_phase"):
        return _cancelled_pass(p1, p1["cancelled_phase"], opt_dir=opt_dir, out_dir=out_dir, res=res, storage=storage)
    res.timings["simp_s"] = p1["simp_s"]
    res.timings["verify_s"] = p1["verify_s"]
    best = p1

    sf_exp = p1["verify_export"].worst_safety_factor
    sf_gate1 = _gate_sf(p1.get("export_verify_info"), p1["verify_export"])
    sf_mask_first = p1["verify"].worst_safety_factor
    sf_ratio = (
        max(sf_mask_first, sf_exp) / max(min(sf_mask_first, sf_exp), 1e-9)
        if sf_mask_first is not None and sf_exp is not None
        else None
    )
    # A degenerate export or missed safety floor needs more volume. A large
    # mask/export ratio instead points to thin members erased by smoothing;
    # retry that case at the same volume with a thicker member floor.
    exp_span = float(np.linalg.norm(domain.pitch * np.asarray(domain.dims)))
    exp_degenerate = (
        p1.get("export_solve_ok") is False
        or p1.get("output_envelope_ok") is False
        or any((c.compliance_j or 1.0) <= 0 for c in p1["verify_export"].cases)
        or any((c.max_disp_mm or 0.0) > 10 * exp_span for c in p1["verify_export"].cases)
    )
    exp_info1 = p1.get("export_verify_info") or {}
    loadpath_retry = exp_info1.get("load_path_ok") is False
    safety_retry = (
        exp_degenerate
        or (sf_gate1 is not None and sf_gate1 < opt.safety_factor_min)
        or bool(p1.get("regions_lost"))
        or loadpath_retry
    )
    gap_retry = sf_ratio is not None and sf_ratio > 3.0
    connectivity_retry = (
        p1.get("n_bodies_export") is None
        or int(p1["n_bodies_export"]) > 1
        # sub-voxel bridges: the mesh reads as one body but the FEM raster
        # splits it — only a thicker member floor can re-join the grid
        or int(exp_info1.get("n_bodies_voxel") or 1) > 1
        or loadpath_retry
    )
    # Stated spec envelopes intentionally allow a shrunken export; avoid
    # replacing that geometry with a crisp full-boundary retry. A real
    # volume *deficit* on a spec'd part instead retries at a raised
    # extraction volume target — the iso moves outward, the boundary does
    # not get filled.
    p1_vol_err_pp = float(p1["extract"].get("vol_error_pp") or 0.0)
    volume_retry = p1_vol_err_pp < -5.0 if dims_spec else abs(p1_vol_err_pp) > 5.0
    retry_needed = safety_retry or gap_retry or connectivity_retry or volume_retry
    # a retry only helps when the optimizer window still fits a meaningful
    # warm-start pass — under ~2min it just re-enters the same dead end and
    # burns the tail reserve the committed artifact needs
    retry_room = _opt_deadline is None or _opt_deadline - time.monotonic() > 120.0
    if retry_needed and retry_room and not _opt_cancelled():
        retry_volfrac = params.volfrac + 0.1 if safety_retry else params.volfrac
        regions_retry = bool(p1.get("regions_lost"))
        retry_member = (
            max(min_member, 2.0 * design_domain.pitch)
            if gap_retry or connectivity_retry or regions_retry
            else min_member
        )
        # crisp extraction also when a case region vanished: Taubin can
        # dissolve the thin bridge a lost region needs to stay reachable
        retry_crisp = not dims_spec and (
            gap_retry or connectivity_retry or volume_retry or regions_retry
        )
        # compensate the measured deficit: vol_error_pp is relative to the
        # design-space volume, so scale the mask-volume target by that
        # fraction of the mask plus a small margin, bounded at 1.5x
        retry_vol_scale = 1.0
        if dims_spec and p1_vol_err_pp < -5.0:
            mask_vol = max(float(p1["final_mask"].sum()) * design_domain.pitch**3, 1e-9)
            design_vol = float(design_domain.solid_count) * design_domain.pitch**3
            retry_vol_scale = min(1.5, 1.0 + (-p1_vol_err_pp / 100.0) * design_vol / mask_vol * 1.1)
        if retry_volfrac <= 0.9:
            p2 = _one_pass(
                retry_volfrac,
                pass_idx=2,
                warm=p1["result"].density,
                member_floor_mm=retry_member,
                crisp_extraction=retry_crisp,
                vol_target_scale=retry_vol_scale,
                E0_arr=E0_arr,
                _opt_cancelled=_opt_cancelled,
                anchors=anchors,
                base_flat=base_flat,
                beso=beso,
                cancelled=cancelled,
                case_weights=case_weights,
                cases=cases,
                density_to_mesh=density_to_mesh,
                design_domain=design_domain,
                dims_spec=dims_spec,
                domain=domain,
                drop_island_components=drop_island_components,
                eff_span=eff_span,
                keep_list=keep_list,
                mat=mat,
                mat_name=mat_name,
                mesh=mesh,
                min_gap=min_gap,
                min_member=min_member,
                ndimage=ndimage,
                opt=opt,
                opt_dir=opt_dir,
                out_dir=out_dir,
                params=params,
                passive=passive,
                resolved=resolved,
                run_fem_domain=run_fem_domain,
                s=s,
                simp=simp,
                spec=spec,
                spsolve_eff=spsolve_eff,
                void_cells=void_cells,
                x0=x0,
                design_pitch=design_pitch,
                eff_robust=eff_robust,
                max_elements=max_elements,
                res=res,
            )
            if p2.get("cancelled_phase"):
                return _cancelled_pass(best, p2["cancelled_phase"], opt_dir=opt_dir, out_dir=out_dir, res=res, storage=storage)
            sf_exp2 = p2["verify_export"].worst_safety_factor
            sf_gate2 = _gate_sf(p2.get("export_verify_info"), p2["verify_export"])
            trigger_reasons = [
                reason
                for enabled, reason in (
                    (regions_retry, "regions_lost"),
                    (loadpath_retry, "load_path"),
                    (
                        exp_degenerate
                        or (sf_gate1 is not None and sf_gate1 < opt.safety_factor_min),
                        "safety",
                    ),
                    (gap_retry, "sf_gap"),
                    (
                        connectivity_retry and not loadpath_retry,
                        "connectivity",
                    ),
                    (volume_retry, "volume_preserve"),
                )
                if enabled
            ]
            res.topopt["retry"] = {
                "trigger": trigger_reasons[0],
                "triggers": trigger_reasons,
                "volfrac_first": params.volfrac,
                "volfrac_second": retry_volfrac,
                "sf_export_first": sf_exp,
                "sf_export_second": sf_exp2,
                "sf_verified_first": sf_gate1,
                "sf_verified_second": sf_gate2,
                "sf_ratio_first": sf_ratio,
                "min_member_mm": retry_member,
                "crisp_extraction": retry_crisp,
                "vol_target_scale": retry_vol_scale,
                "warm_start": "pass1_density",
                "first_export_degenerate": exp_degenerate or None,
            }
            res.timings["simp_s"] += p2["simp_s"]
            res.timings["verify_s"] += p2["verify_s"]
            p1_ok = (
                p1.get("output_envelope_ok") is not False
                and p1.get("n_bodies_export") is not None
                and int(p1["n_bodies_export"]) <= 1
            )
            p2_ok = (
                p2.get("output_envelope_ok") is not False
                and p2.get("n_bodies_export") is not None
                and int(p2["n_bodies_export"]) <= 1
            )
            p2_sf_mask = p2["verify"].worst_safety_factor
            p2_ratio = (
                max(p2_sf_mask, sf_exp2) / max(min(p2_sf_mask, sf_exp2), 1e-9)
                if p2_sf_mask is not None and sf_exp2 is not None
                else None
            )
            p1_quality = (
                p1_ok,
                abs(float(p1["extract"].get("vol_error_pp") or 0.0)) <= 5.0,
                sf_ratio is None or sf_ratio <= 3.0,
                float(sf_gate1 or 0.0),
            )
            p2_quality = (
                p2_ok,
                abs(float(p2["extract"].get("vol_error_pp") or 0.0)) <= 5.0,
                p2_ratio is None or p2_ratio <= 3.0,
                float(sf_gate2 or 0.0),
            )
            if p2_quality > p1_quality:
                best = p2

    if _deadline_hit():
        res.topopt.setdefault("warnings", []).append(
            f"optimizer reached the {s.topopt_timeout_s}s run budget — the "
            "partial density was still extracted, verified and committed as "
            "'budget_exhausted'"
        )
    result = best["result"]
    opt_mesh = best["opt_mesh"]
    final_mask = best["final_mask"]
    verify = best["verify"]
    verify_export = best["verify_export"]
    res.topopt["warnings"] += best["extract_warnings"]

    # how well does the printed geometry realize the SIMP optimum — the
    # numbers the agent should actually judge a run by. Flat dict; extra
    # telemetry keys are only emitted when their source was computed.
    exp_info = best.get("export_verify_info") or {}
    verification: dict[str, Any] = {
        "compliance_pred_mj": (result.history[-1].get("compliance_mj") if result.history else None),
        "compliance_mask_j": [c.compliance_j for c in verify.cases],
        "compliance_export_j": [c.compliance_j for c in verify_export.cases],
        "sf_mask": verify.worst_safety_factor,
        "sf_export": verify_export.worst_safety_factor,
        # headline: worst safety factor over the nominal/eroded/dilated
        # export realizations (the nominal export SF when the sweep was
        # off); the mask result is the last resort when even the nominal
        # export solve could not run
        "sf_verified": (
            exp_info.get("sf_verified")
            if exp_info.get("sf_verified") is not None
            else verify_export.worst_safety_factor
        ),
        "sf_realizations": exp_info.get("sf_realizations"),
        "sf_realization_detail": exp_info.get("sf_realization_detail"),
        "sf_verified_basis": exp_info.get("sf_verified_basis"),
        "enclosed_void_mm3": exp_info.get("enclosed_void_mm3"),
        "void_frac_of_solid": exp_info.get("void_frac_of_solid"),
        "n_sealed_cavities": exp_info.get("n_sealed_cavities"),
        "largest_void_mm3": exp_info.get("largest_void_mm3"),
        "sealed_cavities_mm3": exp_info.get("sealed_cavities_mm3"),
        "export_grid": exp_info.get("export_grid"),
        "bottleneck_mm2": exp_info.get("bottleneck_mm2"),
        "sf_floor": exp_info.get("sf_floor"),
        "load_path_ok": exp_info.get("load_path_ok"),
        "load_path_cases": exp_info.get("load_path_cases"),
        # post-extraction early flags: regions the optimizer emptied
        # entirely (their load path can never verify) and pinned cells the
        # extraction dropped (a bug signal — passive is not designable)
        "regions_lost": best.get("regions_lost"),
        "passive_cells_lost": best.get("passive_cells_lost"),
        "sf_mesh_sensitive": exp_info.get("sf_mesh_sensitive"),
        "sf_ratio": (
            max(verify.worst_safety_factor, verify_export.worst_safety_factor)
            / max(min(verify.worst_safety_factor, verify_export.worst_safety_factor), 1e-9)
            if verify.worst_safety_factor is not None
            and verify_export.worst_safety_factor is not None
            else None
        ),
        "volfrac_export": best["volfrac_export"],
        "vol_error_pp": best["extract"].get("vol_error_pp"),
        "max_disp_export_mm": max((c.max_disp_mm for c in verify_export.cases), default=None),
        "max_vm_export_mpa": max((c.max_vm_mpa for c in verify_export.cases), default=None),
        "gray_fraction": best["extract"].get("gray_fraction"),
        "vol_error_pct": best["extract"].get("vol_error_pct"),
        "sealed_voids_filled_mm3": best["extract"].get("sealed_voids_filled_mm3", 0.0),
        "shell_min_wall_mm": best["extract"].get("shell_min_wall_mm"),
        # solid fraction of the design space the extracted mask actually keeps
        "volfrac_mask": round(float(final_mask.sum()) / max(int(design_domain.solid.sum()), 1), 4),
        "solver_used": params.solver,
    }
    if result.history and result.history[-1].get("beta") is not None:
        verification["beta_final"] = result.history[-1]["beta"]
    # degenerate export solve — load regions missed the extracted mesh, the
    # part flies away (disp on the order of the part size), stress sits far
    # outside the linear model, or compliance goes non-physical. The SF
    # number is then meaningless, flag instead of storing as valid.
    # A multi-body export is also not a verified single part: voxel FEM can
    # solve on one island while the others carry no load.
    span = float(np.linalg.norm(domain.pitch * np.asarray(domain.dims)))
    disp = verification["max_disp_export_mm"]
    vm_max = verification["max_vm_export_mpa"]
    comps = [c for c in verification["compliance_export_j"] if c is not None]
    deg_valid, deg_warnings = _export_degeneracy(
        disp_mm=disp,
        max_vm_mpa=vm_max,
        span_mm=span,
        yield_mpa=getattr(mat, "yield_mpa", None),
    )
    res.topopt["warnings"].extend(deg_warnings)
    nonphysical = bool(comps) and any(c <= 0 for c in comps)
    if nonphysical or not deg_valid:
        verification["valid"] = False
        if nonphysical:
            res.topopt["warnings"].append(
                "export FEM solve degenerate (unconstrained part or unresolved "
                "load regions) — sf_export not trustworthy"
            )
    elif best.get("export_solve_ok") is False:
        # the export solve crashed and the mask report was substituted —
        # sf_export is a label, not a measurement
        verification["valid"] = False
    elif best.get("output_envelope_ok") is False:
        verification["valid"] = False
        res.topopt["warnings"].append(
            "smoothed export remains outside the stated spec envelope — "
            "manufacturing verification is invalid"
        )
    elif best.get("n_bodies_export") is None:
        verification["valid"] = False
        res.topopt["warnings"].append(
            "physical export body count is unverified because the export could not be re-voxelized"
        )
    elif verify_export.worst_safety_factor is None:
        # every export case was skipped (e.g. load region missed the
        # smoothed mesh) — no export evidence exists at all
        verification["valid"] = False
        res.topopt["warnings"].append(
            "no export FEM case produced a safety factor — sf_export is unverified, not safe"
        )
    elif int(best["n_bodies_export"]) > 1:
        verification["valid"] = False
        res.topopt["warnings"].append(
            f"optimized part has {best['n_bodies_export']} disconnected bodies "
            "in the export mesh — sf_export describes only the loaded island, "
            "not a single part"
        )
    else:
        verification["valid"] = True
    if verification.get("regions_lost"):
        # a case region vanished entirely in extraction — the load-path
        # certificate will agree, but flag it here with the region names
        verification["valid"] = False
        res.topopt["warnings"].append(
            "optimizer removed region(s) outright: "
            + ", ".join(verification["regions_lost"])
            + " — load path cannot verify"
        )
    if exp_info.get("load_path_ok") is False:
        # the geometric certificate found a case whose loads cannot reach
        # their fixtures through the export solid — structurally broken
        verification["valid"] = False
        failed_cases = [
            str(c.get("case_name") or c.get("case"))
            for c in (exp_info.get("load_path_cases") or [])
            if c.get("reachable") is False
        ]
        res.topopt["warnings"].append(
            "load-path certificate: a load case cannot reach its fixtures "
            "through the export solid — the part is structurally disconnected"
            + (f" (failing: {', '.join(failed_cases)})" if failed_cases else "")
        )
    if verification["sf_ratio"] is not None and verification["sf_ratio"] > 3.0:
        verification["valid"] = False
        res.topopt["warnings"].append(
            f"mask/export safety-factor ratio remains {verification['sf_ratio']:.1f}x "
            "after the bounded retry — the STL FEM result does not reproduce the mask"
        )
    elif verification["sf_ratio"] is not None and verification["sf_ratio"] > 2.0:
        res.topopt["warnings"].append(
            f"mask/export safety-factor ratio is {verification['sf_ratio']:.1f}x — "
            "inspect the final STL load path; sf_export governs"
        )
    if abs(float(verification["vol_error_pp"] or 0.0)) > 5.0:
        verification["valid"] = False
        res.topopt["warnings"].append(
            f"final STL volume differs from the FEM mask by "
            f"{verification['vol_error_pp']:.1f} percentage points after the bounded retry"
        )
    # "is the measurement valid" (above) is a different question from "does
    # the printed part meet the requested safety floor" — report both; the
    # field is omitted when no export SF was measured
    sf_min = getattr(opt, "safety_factor_min", None)
    # the requested safety floor is judged against the verified worst-case
    # realization (sf_verified -> sf_export -> sf_mask fallback chain)
    sf_gate = verification["sf_verified"]
    if sf_gate is None:
        sf_gate = verify.worst_safety_factor
    if sf_min is not None and sf_gate is not None:
        verification["meets_safety_factor"] = bool(sf_gate >= sf_min)
    if verify.worst_safety_factor is not None and verify_export.worst_safety_factor is not None:
        # >0 means the smoothed export mesh is stiffer than the mask suggests
        verification["sf_gap"] = round(
            verify.worst_safety_factor - verify_export.worst_safety_factor, 4
        )
        # a >=2x divergence between the two verifications means one of them is
        # modelling a different structure than the other: usually the discrete
        # mask solve hits a one-voxel connection the smoothing fills in (mask
        # pessimistic) or the smooth skin bridges a severed section (export
        # optimistic). sf_export governs the printed part, but the agent
        # should inspect the thin junction instead of trusting either blind.
        sf_ratio = verify.worst_safety_factor / max(verify_export.worst_safety_factor, 1e-9)
        if sf_ratio > 2.0:
            retry_fired = res.topopt.get("retry", {}).get("trigger") == "sf_gap"
            advice = (
                "inspect the thin junction; sf_export governs the printed part"
                if retry_fired
                else "smoothing removed load-bearing material; increase min_member_mm "
                "or lower extract_sigma_px to reduce loss"
            )
            res.topopt["warnings"].append(f"sf_mask is {sf_ratio:.1f}x sf_export — {advice}")
        elif sf_ratio < 0.5:
            res.topopt["warnings"].append(
                f"sf_export is {1.0 / max(sf_ratio, 1e-9):.1f}x sf_mask — the smoothed "
                "mesh and voxel mask model different load paths; inspect the geometry"
            )
    if verify.grid.get("dofs"):
        verification["dofs"] = verify.grid["dofs"]
    if verify.cases:
        verification["case_sfs_mask"] = [c.safety_factor for c in verify.cases]
    if verify_export.cases:
        verification["case_sfs_export"] = [c.safety_factor for c in verify_export.cases]

    # manufacturing signal for candidate selection — the FDM overhang metric
    # on the extracted part, parameterized on the run's print direction.
    # Reported per-run so the agent can steer: a support-free variant is one
    # self_supporting=True rerun away.
    from cadx.cad.dfm.fdm import overhang_ratio

    overhang = round(overhang_ratio(opt_mesh, params.print_direction) * 100, 1)
    verification["overhang_pct"] = overhang
    if overhang > 20.0 and not params.self_supporting:
        res.topopt["warnings"].append(
            f"{overhang:.0f}% of the surface overhangs >50° from "
            f"{params.print_direction} — rerun with self_supporting=True "
            "for a support-free variant, or accept support material"
        )
    if best.get("worse_than_uniform"):
        res.topopt["warnings"].append(
            "BESO converged to a layout less stiff than a uniform field of "
            "the same mass — treat it as a warm-start heuristic and prefer "
            "OC/MMA for the final topology"
        )

    res.topopt.update(
        {
            "iterations": result.iterations,
            "iterations_total": int((res.topopt.get("multires") or {}).get("coarse_iterations", 0))
            + result.iterations,
            "converged": result.converged,
            # how the run ended — converged | infeasible_target | limit_cycle
            # | degenerate | budget_exhausted (topopt.py TopOptResult.status)
            "status": result.status,
            # the caller's ask vs the pass the sweep actually committed —
            # `volfrac_target` alone reported the silent +0.1 retry as the
            # target and misled the agent about what ran
            "volfrac_requested": params.volfrac,
            "volfrac_target": best["volfrac"],
            "volfrac_achieved": round(
                float(result.density[design_domain.solid.ravel()].sum())
                / max(int(design_domain.solid.sum()), 1),
                4,
            ),
            "volfrac_export": verification["volfrac_export"],
            "history": _compact_history(result.history),
            "passive_frac": round(result.passive_frac, 4),
            "warnings": res.topopt.get("warnings", []) + result.warnings,
            "material": mat.name,
            "sf_mask": verify.worst_safety_factor,
            "sf_export": verify_export.worst_safety_factor,
            "sf_verified": verification["sf_verified"],
            "sf_realizations": verification["sf_realizations"],
            "sf_realization_detail": verification.get("sf_realization_detail"),
            "sf_verified_basis": exp_info.get("sf_verified_basis"),
            "enclosed_void_mm3": verification.get("enclosed_void_mm3"),
            "void_frac_of_solid": verification.get("void_frac_of_solid"),
            "n_sealed_cavities": verification.get("n_sealed_cavities"),
            "largest_void_mm3": verification.get("largest_void_mm3"),
            "sealed_cavities_mm3": verification.get("sealed_cavities_mm3"),
            "export_grid": verification.get("export_grid"),
            "sealed_voids_filled_mm3": best["extract"].get("sealed_voids_filled_mm3", 0.0),
            "shell_min_wall_mm": best["extract"].get("shell_min_wall_mm"),
            "void_pinned_cells_total": result.void_pinned_cells_total,
            "void_pin_iters": result.void_pin_iters,
            "void_pin_sym_mirrored": result.void_pin_sym_mirrored,
            "void_pinning_applied": result.void_pinning_applied,
            "void_pinned_mm3": round(result.void_pinned_cells_total * design_domain.pitch**3, 3),
            "bottleneck_mm2": verification["bottleneck_mm2"],
            "sf_floor": verification["sf_floor"],
            "load_path_ok": verification["load_path_ok"],
            "load_path_cases": exp_info.get("load_path_cases"),
            "sf_mesh_sensitive": verification["sf_mesh_sensitive"],
            # BESO honesty check — top-level so _metrics_hint/compaction
            # keep it: a converged binary field that loses to the uniform
            # baseline is a verdict, not a detail
            "uniform_baseline_compliance_mj": best.get("uniform_baseline_compliance_mj"),
            "worse_than_uniform": best.get("worse_than_uniform"),
            "min_member_mm": best["min_member_mm"],
            "thin_fraction": round(best["thin_fraction"], 4),
            "n_components": best["n_components"],
            "n_bodies_export": best["n_bodies_export"],
            "regions_lost": verification["regions_lost"],
            "passive_cells_lost": verification["passive_cells_lost"],
            "n_bodies_voxel": exp_info.get("n_bodies_voxel"),
            "n_surface_components": best["n_surface_components"],
            "output_envelope": best.get("output_envelope"),
            "output_envelope_ok": best.get("output_envelope_ok", True),
            "extract": best["extract"],
            "verification": verification,
            "recut": best.get("recut"),
            # flat rescue telemetry so gates can assert on the numbers
            # (dropping a big island silently would hide a real defect)
            "rescue": best.get("island_rescue"),
            "rescue_islands": (best.get("island_rescue") or {}).get("islands_found"),
            "rescue_dropped": (best.get("island_rescue") or {}).get("dropped"),
            "rescue_dropped_frac": (best.get("island_rescue") or {}).get("dropped_vol_frac"),
        }
    )

    if _is_cancelled(cancelled):
        return _cancelled_pass(best, "before_final_density_persist", opt_dir=opt_dir, out_dir=out_dir, res=res, storage=storage)
    np.savez_compressed(
        opt_dir / "density.npz",
        density=result.density.astype(np.float32),
        mask=final_mask,
        solid=design_domain.solid,
        part_solid=part_domain.solid,
        passive=passive.reshape(design_domain.dims),
        # persisted so topopt_resample can re-carve declared void regions
        # (kept cells alone cannot express them)
        voids=void_cells.reshape(design_domain.dims),
        origin=design_domain.origin,
        pitch=np.float64(design_domain.pitch),
        dims=np.asarray(design_domain.dims),
        flat_bottom=np.bool_(base_flat),
        # field provenance: _auto_warm_density only seeds SIMP from
        # continuous/binary optimizer fields — an infill lattice rho would
        # pin cavities at the density floor instead of regrowing them
        kind=np.str_("beso" if params.solver == "beso" else "simp"),
    )
    # compact convergence trace alongside the per-iteration progress files —
    # tasks.py points topopt["history_relpath"] here
    with contextlib.suppress(Exception):
        (opt_dir / "history.json").write_text(
            json.dumps({"history": res.topopt["history"]}, indent=1), encoding="utf-8"
        )
    report_to_json(verify, out_dir / "fem" / "report.json")
    report_to_json(verify_export, out_dir / "fem" / "report_export.json")
    res.fem = verify.model_dump()
    res.fem_export = verify_export.model_dump()

    # --- artifacts: STL/GLB/3MF + STL copy used for renders/analysis ----------
    if _is_cancelled(cancelled):
        return _cancelled_pass(best, "before_final_exports", opt_dir=opt_dir, out_dir=out_dir, res=res, storage=storage)
    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    try:
        opt_mesh = export_stl(opt_mesh, stl_path)
    except Exception as e:
        res.error = f"failed to write optimized STL: {e}"
        return res
    # faceted STEP alongside the mesh formats — mesh_to_solid decimates
    # beyond the cad face cap and export_mesh_all falls back to step=None
    res.exports = export_mesh_all(
        opt_mesh,
        out_dir / "exports",
        name=name,
        try_step=True,
        max_faces=s.mesh_max_faces_for_cad,
    )
    res.exports["dxf"] = None
    res.exports["svg"] = None
    res.analysis = mesh_analysis(opt_mesh)
    # feature evidence for verify_spec / critique / salvage: the topopt path
    # has no B-Rep, so measure the shipped mesh — hole groups get real
    # evidence (approximate=True is correct: mesh ray-grid can't see fillets)
    try:
        from cadx.cad.features import mesh_features_to_measured
        from cadx.cad.mesh_features import detect_mesh_features

        mf = mesh_features_to_measured(detect_mesh_features(opt_mesh).model_dump(mode="json"))
        res.features = {**mf.model_dump(mode="json")}
        if isinstance(best.get("recut"), dict):
            res.features["refine"] = best["recut"]
    except Exception as e:
        log.warning("features.mesh_measure_failed", error=str(e)[:160])
    _repair_analysis_mesh(res.analysis, stl_path)
    # unconditional: the in-memory watertight flag does not guarantee the
    # round-tripped files are clean — non-manifold flap faces / zero-volume
    # slivers can split off as phantom bodies; heavy repair stays
    # conditional inside repair_mesh_file
    _repair_export_meshes(res.exports, out_dir / "exports")
    res.mesh = {
        "pipeline": "topopt",
        "triangles": len(opt_mesh.faces),
        "volume_mm3": round(float(opt_mesh.volume), 1),
        "solid_elements": int(final_mask.sum()),
        "pitch_mm": domain.pitch,
    }

    # renders
    if _is_cancelled(cancelled):
        return _cancelled_pipeline_result(res, "before_render")
    t = time.monotonic()
    render_dir = out_dir / "renders"
    _rq_size, _rq_ghosts = _render_quality(res.detail_mode)
    render_paths = render_views(
        stl_path, render_dir, res.analysis.bbox_size, size=_rq_size, ghosts=_rq_ghosts
    )
    res.renders = {v: p.name for v, p in render_paths.items()}
    thumb = render_thumbnail(stl_path, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
    sf_mask_s = verify.worst_safety_factor
    sf_exp_s = verify_export.worst_safety_factor
    stats = [
        f"bbox: {res.analysis.bbox_size[0]:.1f} x {res.analysis.bbox_size[1]:.1f} "
        f"x {res.analysis.bbox_size[2]:.1f} mm",
        f"volume: {res.analysis.volume:.0f} mm^3 (target {best['volfrac']:.0%})",
        f"triangles: {res.analysis.triangle_count}",
        f"watertight: {res.analysis.watertight_after_repair}",
        f"SF mask/export: {sf_mask_s} / {sf_exp_s}",
        f"min member: {best['min_member_mm']} mm",
    ]
    sheet = make_contact_sheet(render_paths, out_dir / "contact_sheet.png", stats)
    res.contact_sheet = sheet.name if sheet else None
    res.timings["render_s"] = round(time.monotonic() - t, 2)
    if _is_cancelled(cancelled):
        return _cancelled_pipeline_result(res, "after_render")

    if not opt_mesh.is_watertight:
        res.topopt["warnings"].append(
            "optimized mesh is not watertight — inspect before manufacturing"
        )
    res.ok = True
    # canonical conformance report — bound semantics: the stated dims are an
    # envelope the optimized part must fit inside, not targets to match
    _attach_verify_spec(res, spec, ProjectSettings(), s, envelope_bound_only=True)
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res
def run_topopt_resample(  # noqa: C901
    *,
    density_relpath: str,
    out_dir: Path,
    load_cases: list[dict[str, Any]],
    base_mesh_path: Path | None = None,
    optimize: dict[str, Any] | None = None,
    material: str = "",
    max_elements: int | None = None,
    anchors: list[dict] | None = None,
    name: str = "model",
    runtime: RuntimeSettings | None = None,
    cancelled: Callable[[], bool] | None = None,
    spec: dict[str, Any] | None = None,
) -> ExecResult:
    """Re-extract a stored SIMP density field at a different threshold /
    min-member without re-running SIMP — the threshold-sweep endpoint.

    Loads density.npz from a previous optimize run, rebuilds the voxel
    domain from the stored grid meta, extracts + verifies both the exact
    mask and the smoothed export mesh, writes exports + reports.
    """
    from cadx import storage
    from cadx.cad.fem.analyze import report_to_json, run_fem_domain
    from cadx.cad.fem.extract import density_to_mesh
    from cadx.cad.fem.materials import resolve_material
    from cadx.cad.fem.regions import (
        keep_element_mask,
        region_node_mask,
        resolve_case_dofs,
    )
    from cadx.cad.fem.topopt import passive_solid_mask
    from cadx.cad.fem.voxelize import VoxelDomain
    from cadx.llm.schemas import LoadCase, OptimizeSpec, RegionSpec

    res = ExecResult()
    s = runtime or RuntimeSettings()
    opt = OptimizeSpec(**(optimize or {}))
    t0 = time.monotonic()

    try:
        npz = np.load(storage.resolve(density_relpath))
        density = np.asarray(npz["density"], dtype=float)
        origin = np.asarray(npz["origin"], dtype=float)
        pitch = float(npz["pitch"])
        _d = [int(v) for v in npz["dims"]]
        dims = (_d[0], _d[1], _d[2])
        if "solid" in npz:
            solid = np.asarray(npz["solid"], dtype=bool).reshape(dims)
        else:  # legacy npz without the stored design space
            solid = (density >= 1e-2).reshape(dims) | np.asarray(npz["mask"], dtype=bool).reshape(
                dims
            )
    except Exception as e:
        res.error = f"density field unreadable: {e}"
        return res

    domain = VoxelDomain(origin=origin, pitch=pitch, dims=dims, solid=solid)
    mat = resolve_material(material or "")
    cases = [LoadCase(**c) for c in load_cases]
    threshold = float(opt.density_threshold or 0.5)
    # a hard-kill (BESO/crisp) field has no gray zone to re-threshold —
    # different cut levels carve the same part; say so instead of silently
    # returning an identical mesh (the contract stays valid, the knob does
    # nothing)
    rho_solid = density[solid.ravel()] if density.size == solid.size else density
    frac_gray = float(np.mean((rho_solid > 1e-3) & (rho_solid < 1.0 - 1e-3)))
    if rho_solid.size and frac_gray < 0.01:
        res.topopt.setdefault("warnings", []).append(
            f"density field is binary ({frac_gray:.1%} gray cells) — a "
            "different threshold re-extracts the same part; threshold "
            "sweeps only matter on continuous (OC/MMA) fields"
        )
    min_member = float(opt.min_member_mm or s.topopt_min_member_mm or 0.0)
    min_gap = float(opt.min_gap_mm or min_member or 0.0)

    # rebuild the pinned-solid mask from the declared regions
    fixed_node_masks: list[Any] = []
    load_node_masks: list[Any] = []
    for c in cases:
        fixed3, _F, _w = resolve_case_dofs(domain, c, anchors, density_g_cm3=mat.density_g_cm3)
        res.topopt.setdefault("warnings", []).extend(_w)
        fixed_node_masks.append(np.asarray(fixed3).any(axis=1))
        for ld in c.loads:
            if _load_pins_nodes(ld):
                load_node_masks.append(region_node_mask(domain, ld.region, anchors))
    keep_list = list(opt.keep_regions)
    # same interface-protection rule as run_topopt_pipeline: keep_mode="auto"
    # covers every fastener anchor, "manual" protects spec-declared ones when
    # the runtime flag is on; scan extras stay designable
    n_auto_keeps = 0
    iface_voids: list[RegionSpec] = []
    if anchors and (opt.keep_mode == "auto" or s.topopt_auto_keep):
        from cadx.cad.fem.regions import interface_regions

        margin = opt.auto_keep_margin_mm or s.topopt_auto_keep_margin_mm
        ik, iface_voids = interface_regions(anchors, margin, spec_only=(opt.keep_mode != "auto"))
        keep_list += ik
        n_auto_keeps = len(ik)
    res.topopt["auto_keep_applied"] = bool(n_auto_keeps)
    keep_cells = keep_element_mask(domain, keep_list, anchors)
    passive = passive_solid_mask(domain, [keep_cells], fixed_node_masks, load_node_masks)
    if "passive" in npz:
        # stored pin mask — carries organic floor/shell/auto-keep cells that
        # region-based rebuild cannot express; harmless union for topopt
        # fields where it coincides with the rebuilt mask
        passive |= np.asarray(npz["passive"], dtype=bool).ravel()

    # void mask for the extraction: prefer the persisted `voids` key (organic
    # npz and topopt npz written since the void-persistence fix); otherwise
    # re-resolve declared void_regions — older npz files simply lack the key
    void_flat: np.ndarray | None = None
    if "voids" in npz:
        void_flat = np.asarray(npz["voids"], dtype=bool).ravel()
    elif opt.void_regions or iface_voids:
        void_flat = keep_element_mask(domain, list(opt.void_regions) + iface_voids, anchors)

    try:
        opt_mesh, final_mask, extract_warnings, n_components, extract_info = density_to_mesh(
            domain,
            density,
            threshold=threshold,
            passive_solid=passive,
            # organic npz persists carved cells — resample must re-carve them
            # after min-member dilation, same as the original extraction.
            # plus ~solid: cells outside the stored design domain must stay
            # empty too, or dilation grows material past the part surface
            void_cells=(
                (void_flat.reshape(domain.dims) | ~domain.solid)
                if void_flat is not None
                else ~domain.solid
            ),
            min_member_mm=min_member,
            min_gap_mm=min_gap,
            smooth_iterations=s.topopt_smooth_iterations,
            extract_smooth=opt.extract_smooth,
            sigma_px=opt.extract_sigma_px,
            volume_preserve=opt.volume_preserve,
            fill_sealed_voids=opt.fill_sealed_voids,
            min_shell_mm=opt.min_shell_mm,
            fill_thin_walled_voids=True,
        )
    except Exception as e:
        res.error = f"re-extraction failed: {e}"
        return res
    res.topopt["warnings"] = res.topopt.get("warnings", []) + extract_warnings
    res.topopt.update(
        {
            "threshold": threshold,
            "min_member_mm": min_member,
            "n_components": n_components,
            "resampled_from": density_relpath,
            "extract": extract_info,
            "sealed_voids_filled_mm3": extract_info.get("sealed_voids_filled_mm3", 0.0),
            "shell_min_wall_mm": extract_info.get("shell_min_wall_mm"),
        }
    )

    # same smoothing stage as the original run — floor pinning follows the
    # flag stored in the density field
    from cadx.cad.mesh_ops import smooth_preserve

    # ndarray telemetry must not reach the persisted extract dict —
    # it kills the version-row JSON flush (already stored in res.topopt
    # above, so pop it off the shared dict)
    bridged3 = extract_info.pop("bridged_mask", None)
    pin3 = passive.reshape(domain.dims).copy()
    if bridged3 is not None:
        # corridor cells added by island-bridging pin like passive — the
        # connectivity glue must not erode below raster thickness
        pin3 |= np.asarray(bridged3, dtype=bool).reshape(domain.dims)
    opt_mesh = smooth_preserve(
        opt_mesh,
        iterations=4,
        pin_floor=bool(npz.get("flat_bottom", False)),
        pin_mask=(domain.origin, domain.pitch, pin3),
    )
    dims_spec = (optimize or {}).get("spec_dims_mm") or _spec_dims_from_dict(spec)
    opt_mesh, output_envelope, output_envelope_ok, envelope_warnings = (
        _clip_export_to_spec_envelope(
            opt_mesh,
            list(dims_spec) if dims_spec else None,
            label="smoothed resampled export",
        )
    )
    res.topopt.setdefault("warnings", []).extend(envelope_warnings)
    res.topopt["output_envelope"] = output_envelope
    res.topopt["output_envelope_ok"] = output_envelope_ok
    res.topopt["n_surface_components"] = int(opt_mesh.body_count)
    res.topopt["n_bodies_export"] = None
    base_mesh = None
    if base_mesh_path is not None:
        try:
            base_mesh = load_triangle_mesh(base_mesh_path)
        except Exception as e:
            res.topopt.setdefault("warnings", []).append(
                f"resample base mesh unavailable for restore/re-cut: {str(e)[:140]}"
            )
    opt_mesh, recut_report, finish_warnings = _post_extract_finish(
        opt_mesh,
        base_mesh=base_mesh,
        keep_regions=keep_list,
        anchors=anchors,
        spec_features=active_features(spec),
        recut_bores=bool(opt.recut_bores),
        restore_keeps=bool(opt.restore_planes),
        helical_threads=False,
    )
    res.topopt.setdefault("warnings", []).extend(finish_warnings)
    if recut_report is not None:
        res.topopt["recut"] = recut_report

    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    try:
        opt_mesh = export_stl(opt_mesh, stl_path)
    except Exception as e:
        res.error = f"failed to write resampled STL: {e}"
        return res

    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return res
    if cases:
        # opt_mesh shares the mask's frame / IS the export surface — the FEM
        # renders draw its smooth skin instead of a blocky voxel hull
        verify = run_fem_domain(
            domain,
            final_mask,
            cases,
            material or "",
            out_dir=out_dir,
            anchors=anchors,
            keep_regions=keep_list,
            pool_workers=s.exec_pool_workers,
            mem_budget_mb=float(resources.effective_exec_mem_mb(s)),
            source_mesh=opt_mesh,
            cancelled=cancelled,
        )
        report_to_json(verify, out_dir / "fem" / "report.json")
        res.fem = verify.model_dump()
        try:
            # same hardened export verification as the optimize passes —
            # sf_verified carries the realization sweep when enabled
            info = _verify_export_structure(
                candidate_mesh=opt_mesh,
                cases=cases,
                mat_name=material or "",
                mat=mat,
                design_pitch=pitch,
                max_elements=max_elements or s.fem_max_elements,
                keep_list=keep_list,
                anchors=anchors,
                out_dir=out_dir,
                suffix="_export",
                s=s,
                solver_backend=str(getattr(s, "fem_solver_backend", "auto") or "auto"),
                cancelled=cancelled,
                verify_robust=(
                    opt.verify_robust
                    if opt.verify_robust is not None
                    else bool(s.fem_verify_realizations)
                ),
            )
            if info.get("cancelled"):
                return _cancelled_pipeline_result(res, info.get("phase", "resample_export_verify"))
            verify_exp = info["report"]
            report_to_json(verify_exp, out_dir / "fem" / "report_export.json")
            res.fem_export = verify_exp.model_dump()
            res.topopt["n_bodies_export"] = info["n_bodies_export"]
            res.topopt["n_bodies_voxel"] = info["n_bodies_voxel"]
            res.topopt["sf_mask"] = verify.worst_safety_factor
            res.topopt["sf_export"] = verify_exp.worst_safety_factor
            res.topopt["sf_verified"] = info["sf_verified"]
            res.topopt["sf_realizations"] = info["sf_realizations"]
            res.topopt["sf_realization_detail"] = info.get("sf_realization_detail")
            res.topopt["sf_verified_basis"] = info.get("sf_verified_basis")
            res.topopt["enclosed_void_mm3"] = info.get("enclosed_void_mm3")
            res.topopt["void_frac_of_solid"] = info.get("void_frac_of_solid")
            res.topopt["n_sealed_cavities"] = info.get("n_sealed_cavities")
            res.topopt["largest_void_mm3"] = info.get("largest_void_mm3")
            res.topopt["sealed_cavities_mm3"] = info.get("sealed_cavities_mm3")
            res.topopt["export_grid"] = info.get("export_grid")
            res.topopt["bottleneck_mm2"] = info["bottleneck_mm2"]
            res.topopt["sf_floor"] = info["sf_floor"]
            res.topopt["load_path_ok"] = info["load_path_ok"]
            res.topopt["load_path_cases"] = info.get("load_path_cases")
            res.topopt["sf_mesh_sensitive"] = info["sf_mesh_sensitive"]
            res.topopt.setdefault("warnings", []).extend(info["warnings"])
        except Exception as e:
            if _is_cancelled(cancelled):
                return _cancelled_pipeline_result(res, "during_resample_export_verify")
            log.warning("topopt.resample_verify_failed", error=str(e)[:200])
            res.topopt["sf_mask"] = verify.worst_safety_factor

    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return res
    res.exports = export_mesh_all(
        opt_mesh,
        out_dir / "exports",
        name=name,
        try_step=True,
        max_faces=s.mesh_max_faces_for_cad,
    )
    res.exports["dxf"] = None
    res.exports["svg"] = None
    res.analysis = mesh_analysis(opt_mesh)
    try:
        from cadx.cad.features import mesh_features_to_measured
        from cadx.cad.mesh_features import detect_mesh_features

        mf = mesh_features_to_measured(detect_mesh_features(opt_mesh).model_dump(mode="json"))
        res.features = mf.model_dump(mode="json")
        if recut_report is not None:
            res.features["refine"] = recut_report
    except Exception as e:
        log.warning("features.mesh_measure_failed", error=str(e)[:160])
    res.mesh = {
        "pipeline": "topopt_resample",
        "triangles": len(opt_mesh.faces),
        "volume_mm3": round(float(opt_mesh.volume), 1),
        "solid_elements": int(final_mask.sum()),
        "pitch_mm": pitch,
    }
    render_dir = out_dir / "renders"
    _rq_size, _rq_ghosts = _render_quality(res.detail_mode)
    render_paths = render_views(
        opt_mesh, render_dir, res.analysis.bbox_size, size=_rq_size, ghosts=_rq_ghosts
    )
    res.renders = {v: p.name for v, p in render_paths.items()}
    thumb = render_thumbnail(opt_mesh, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
    res.ok = True
    _attach_verify_spec(res, spec, ProjectSettings(), s, envelope_bound_only=True)
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res
