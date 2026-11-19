"""Shared substrate for cad/engine — ExecResult plus the helper set every
pipeline stage consumes (verify, repair, cancel, render-quality, spec
envelope, topopt case resolution). Split out of the former monolith
`engine.py`; `cadx.cad.engine` re-exports every name, so import paths
stay stable."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import import_brep

from cadx.cad import resources
from cadx.cad.analysis import Analysis
from cadx.cad.dfm import DfmReport
from cadx.cad.exporters import (
    export_2d,
    export_all_3d,
    export_analytic_step_isolated,
    export_mesh_all,
    repair_mesh_file,
)
from cadx.cad.mesh_ops import (
    _ensure_watertight,
    solid_body_count,
)
from cadx.cad.mesh_ops import (
    load_mesh as load_triangle_mesh,
)
from cadx.cad.spec_features import active_features, dropped_feature_indices
from cadx.db.models import ProjectSettings
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)
_STEP_GATE_BUDGET_FRACTION = 0.8
_STEP_GATE_MESH_MAX_FACES = 4_000
_PIPELINE_STEPS: dict[str, tuple[str, float]] = {}
_ENVELOPE_CLIP_SLACK_MM = 2.0
_ENVELOPE_CLIP_SLACK_FRAC = 0.05
_NO_WARM_KINDS = frozenset({"infill", "organic"})
_AUX_CG_MAXITER = 1200
_FEATURE_CELLS_PER_DIA = 3.5
_INFILL_SHELL_MM = 2.0
_INFILL_LATTICE_MIN_OCCUPANCY = 0.25
_THIN_PART_INFILL_FLOOR = 0.8

class ExecResult:
    def __init__(self) -> None:
        self.ok: bool = False
        self.error: str | None = None
        self.traceback: str | None = None
        self.stdout: str = ""
        self.params_used: dict[str, Any] = {}
        self.analysis: Analysis | None = None
        self.renders: dict[str, str] = {}  # view -> relpath from out_dir
        self.contact_sheet: str | None = None
        self.thumbnail: str | None = None
        self.exports: dict[str, str | None] = {}
        self.dfm: DfmReport | None = None
        self.mesh: dict[str, Any] = {}
        self.features: dict[str, Any] = {}
        # canonical M6 conformance report (PASS/FAIL/UNVERIFIED per tier)
        self.verify_spec: dict[str, Any] | None = None
        self.fem: dict[str, Any] = {}
        self.fem_export: dict[str, Any] = {}
        self.fem_homog: dict[str, Any] = {}
        self.topopt: dict[str, Any] = {}
        self.thread_count: int = 0
        self.preview_thread_count: int = 0
        self.detail_mode: str = "preview"
        self.design_features: list[dict[str, Any]] = []
        self.bom: list[dict[str, Any]] = []
        self.timings: dict[str, float] = {}
        self.warnings: list[str] = []
        # geometry inventory of the failed sandbox namespace (failed runs only)
        self.geometry_at_failure: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "traceback": self.traceback,
            "stdout": self.stdout,
            "params_used": self.params_used,
            "analysis": self.analysis.model_dump() if self.analysis else None,
            "renders": self.renders,
            "contact_sheet": self.contact_sheet,
            "thumbnail": self.thumbnail,
            "exports": self.exports,
            "dfm": self.dfm.model_dump() if self.dfm else None,
            "mesh": self.mesh,
            "features": self.features,
            "verify_spec": self.verify_spec,
            "fem": self.fem,
            "fem_export": self.fem_export,
            "fem_homog": self.fem_homog,
            "topopt": self.topopt,
            "thread_count": self.thread_count,
            "preview_thread_count": self.preview_thread_count,
            "detail_mode": self.detail_mode,
            "design_features": self.design_features,
            "bom": self.bom,
            "timings": self.timings,
            "warnings": self.warnings,
            "geometry_at_failure": self.geometry_at_failure,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExecResult:
        r = ExecResult()
        r.ok = bool(d.get("ok"))
        r.error = d.get("error")
        r.traceback = d.get("traceback")
        r.stdout = d.get("stdout", "")
        r.params_used = d.get("params_used") or {}
        r.renders = d.get("renders") or {}
        r.contact_sheet = d.get("contact_sheet")
        r.thumbnail = d.get("thumbnail")
        r.exports = d.get("exports") or {}
        r.thread_count = int(d.get("thread_count") or 0)
        r.preview_thread_count = int(d.get("preview_thread_count") or 0)
        r.detail_mode = str(d.get("detail_mode") or "preview")
        r.design_features = list(d.get("design_features") or [])
        r.bom = list(d.get("bom") or [])
        r.timings = d.get("timings") or {}
        if d.get("analysis"):
            r.analysis = Analysis(**d["analysis"])
        if d.get("dfm"):
            r.dfm = DfmReport(**d["dfm"])
        r.mesh = d.get("mesh") or {}
        r.features = d.get("features") or {}
        r.verify_spec = d.get("verify_spec")
        r.fem = d.get("fem") or {}
        r.fem_export = d.get("fem_export") or {}
        r.fem_homog = d.get("fem_homog") or {}
        r.topopt = d.get("topopt") or {}
        r.warnings = list(d.get("warnings") or [])
        r.geometry_at_failure = list(d.get("geometry_at_failure") or [])
        return r
@dataclasses.dataclass
class _GridRef:
    """Minimal VoxelDomain-shaped grid descriptor for resampling a persisted
    density field — only the attributes `_upsample_density` reads."""

    origin: np.ndarray
    pitch: float
    dims: tuple[int, ...]
def _attach_verify_spec(
    res: ExecResult,
    spec_dict: dict[str, Any] | None,
    settings: ProjectSettings,
    rt: RuntimeSettings,
    *,
    measured_features: dict[str, Any] | None = None,
    feature_findings: list[dict[str, Any]] | None = None,
    envelope_bound_only: bool = False,
) -> ExecResult:
    """Compute the canonical M6 conformance report over the measured evidence
    already in ``res`` and store it at ``res.verify_spec`` (shipped as
    ``res["verify_spec"]`` in the exec result dict). Never raises — a report
    failure must not fail the pipeline; the worker simply gets no verdict."""
    try:
        from cadx.cad.features import MeasuredFeatures, compare_features
        from cadx.cad.verifyspec import verify_spec
        from cadx.llm.schemas import Feature

        tol_pct = (
            settings.dim_tolerance_pct
            if settings.dim_tolerance_pct is not None
            else rt.agent_dim_tolerance_pct
        )
        tol_mm = (
            settings.dim_tolerance_mm
            if settings.dim_tolerance_mm is not None
            else rt.agent_dim_tolerance_mm
        )
        measured = measured_features if measured_features is not None else res.features
        if spec_dict is not None and spec_dict.get("min_wall_mm") is None:
            # project default supplies the robustness tier's wall floor when
            # the spec itself does not declare one
            spec_dict = {**spec_dict, "min_wall_mm": settings.min_wall_mm}
        if (
            feature_findings is None
            and spec_dict
            and active_features(spec_dict)
            and measured
            and rt.agent_feature_feedback
        ):
            try:
                from cadx.cad.standard_parts import covered_feature_indices

                covered = covered_feature_indices(spec_dict, keep_host_voids=True)
                dropped = dropped_feature_indices(spec_dict)
                feats = [
                    Feature.model_validate(f)
                    for i, f in enumerate(spec_dict["features"])
                    if i not in covered and i not in dropped
                ]
                feature_findings = [
                    f.model_dump() for f in compare_features(feats, MeasuredFeatures(**measured))
                ]
            except Exception as e:
                log.warning("verify_spec.compare_failed", error=str(e)[:200])
        _tvs = time.monotonic()
        res.verify_spec = verify_spec(
            spec_dict,
            res.to_dict(),
            feature_findings=feature_findings,
            measured_features=measured,
            dim_tolerance_pct=tol_pct,
            dim_tolerance_mm=tol_mm,
            envelope_bound_only=envelope_bound_only,
        )
        res.timings["verify_spec_s"] = round(time.monotonic() - _tvs, 2)
    except Exception as e:
        log.warning("verify_spec.failed", error=str(e)[:200])
    return res
def _audit_refined_features(res: ExecResult, out: trimesh.Trimesh) -> None:
    """Feature audit on the shipped refined mesh: tunnels from the mesh
    scan plus the verified blind pockets only the refine probes can see —
    without this a verified blind tap is invisible in version.mesh.features
    and every consumer (spec gate, UI, evalkit) reports total_holes short.
    """
    try:
        from cadx.cad.features import mesh_features_to_measured
        from cadx.cad.fem.refine import merge_probe_features
        from cadx.cad.mesh_features import detect_mesh_features

        mf = mesh_features_to_measured(
            detect_mesh_features(out).model_dump(mode="json")
        ).model_dump(mode="json")
        for k, v in mf.items():
            res.features.setdefault(k, v)
        res.features["hole_groups"] = merge_probe_features(
            res.features.get("hole_groups") or [],
            (res.features.get("refine") or {}).get("probes") or [],
        )
        res.features["total_holes"] = len(
            [g for g in res.features["hole_groups"] if not g.get("elongated")]
        )
    except Exception as e:
        log.warning("refine.features_measure_failed", error=str(e)[:160])
def _auto_keep_dicts(
    anchors: list[dict[str, Any]] | None, margin_mm: float
) -> list[dict[str, Any]]:
    """keep-region dicts (the apply_refine shape) around every fastener
    anchor — the refine analogue of OptimizeSpec.keep_mode="auto" /
    fem.regions.auto_keep_regions. ``hole_index`` is the 1-based list
    position: apply_refine resolves it via ``anchors[hole_index - 1]``."""
    from cadx.cad.fem.anchors3d import is_fastener_anchor

    return [
        {"kind": "hole", "hole_index": i + 1, "margin_mm": margin_mm}
        for i, a in enumerate(anchors or [])
        if is_fastener_anchor(a)
    ]
def _auto_warm_density(out_dir: Path, domain: Any) -> tuple[np.ndarray | None, str | None]:
    """Opportunistic SIMP warm start, three candidate tiers in order:

    1. the previous iteration dir's ``topopt/density.npz`` (agent optimize
       sessions chain SIMP runs on the same base mesh, so run N benefits
       from run N-1's converged field),
    2. the version-level ``topopt/density.npz`` copied up from an earlier
       optimize on the same version,
    3. a ``guide_density.npz`` written by a guided-infill pass — the coarse
       layout field resampled onto the fine grid (the macro topology the
       coarse solve already verified; the guide pitch is ~2x the *infill*
       grid's pitch, so the ratio vs this domain is measured, not assumed).

    Tiers 1-2 require an *identical* voxel grid (same origin, pitch and
    dims) — a size-only match on a shifted or differently-pitched domain
    would misalign the field; tier 3 resamples through
    ``_upsample_density`` whenever the stored pitch is within 0.25-8x of
    the design pitch.
    Infill/organic density fields are skipped via the ``kind`` tag —
    binary occupancy is not a continuous SIMP seed.
    """
    same_grid: list[Path] = []
    guide_grid: list[Path] = []
    if out_dir.name.isdigit() and int(out_dir.name) > 1:
        prev = out_dir.parent / str(int(out_dir.name) - 1) / "topopt"
        same_grid.append(prev / "density.npz")
        guide_grid.append(prev / "guide_density.npz")
    if len(out_dir.parents) > 1:
        ver = out_dir.parents[1] / "topopt"
        same_grid.append(ver / "density.npz")
        guide_grid.append(ver / "guide_density.npz")
    for p in same_grid:
        if not p.exists():
            continue
        try:
            with np.load(p) as npz:
                if _npz_kind(npz) in _NO_WARM_KINDS:
                    continue
                dims = tuple(int(v) for v in npz["dims"])
                same = (
                    dims == tuple(domain.dims)
                    and np.allclose(np.asarray(npz["origin"], dtype=float), domain.origin)
                    and abs(float(npz["pitch"]) - float(domain.pitch)) < 1e-9
                )
                if same:
                    return np.asarray(npz["density"], dtype=float), str(p)
        except Exception:
            continue
    for p in guide_grid:
        if not p.exists():
            continue
        try:
            with np.load(p) as npz:
                pitch_c = float(npz["pitch"])
                ratio = pitch_c / float(domain.pitch)
                # the guide pitch is ~2x the INFILL grid's pitch — against
                # this domain the ratio is unconstrained, so accept any
                # plausible resample scale and report the measured factor
                if not (0.25 <= ratio <= 8.0):
                    continue
                dom_c = _GridRef(
                    origin=np.asarray(npz["origin"], dtype=float),
                    pitch=pitch_c,
                    dims=tuple(int(v) for v in npz["dims"]),
                )
                x0 = _upsample_density(
                    np.asarray(npz["density"], dtype=float).reshape(dom_c.dims),
                    dom_c,
                    domain,
                )
                return x0, f"guide@{ratio:.2f}x:{p}"
        except Exception:
            continue
    return None, None
def _cancelled_pipeline_result(res: ExecResult, phase: str) -> ExecResult:
    _mark_cancelled(res)
    res.topopt["cancelled_phase"] = phase
    return res
def _clip_export_to_spec_envelope(
    mesh: trimesh.Trimesh,
    spec_dims_mm: list[float | None] | None,
    *,
    label: str,
) -> tuple[trimesh.Trimesh, dict[str, Any] | None, bool, list[str]]:
    """Strictly clip an optimized export without scaling failed candidates."""
    if not spec_dims_mm:
        return mesh, None, True, []
    clipped, report = normalize_mesh_to_spec_envelope(
        mesh,
        spec_dims_mm,
        scale_fallback=False,
        slack_mm=0.25,
        slack_frac=0.01,
    )
    warnings = []
    if report.get("input_normalized"):
        warnings.append(f"{label} clipped to the stated spec envelope")
    if report.get("warning"):
        warnings.append(f"{label} could not be clipped to the spec envelope: {report['warning']}")
    return clipped, report, not bool(report.get("warning")), warnings
def _compact_history(history: list[dict[str, Any]], cap: int = 200) -> list[dict[str, Any]]:
    """Frontend chart series for a SIMP run: one ``{it, compliance, volfrac,
    change, solve_s}`` dict per iteration, decimated to `cap` entries (first
    and last always kept). JSON-serializable — the TopOptPanel contract."""
    if len(history) > cap:
        idx = np.unique(np.linspace(0, len(history) - 1, cap).astype(int))
        history = [history[int(i)] for i in idx]
    return [
        {
            "it": int(h.get("iteration") or 0),
            "compliance": h.get("compliance_mj"),
            "volfrac": h.get("volfrac"),
            "change": h.get("change"),
            "solve_s": h.get("solve_s"),
        }
        for h in history
    ]
def _cone_expand_case(c: Any, budget: int, warn: bool, warn_list: list[str]) -> list[Any]:
    """Tilted copies of case `c`: for every load with direction_cone_deg,
    each rim sample rotates ALL its directional fields (force_n, moment_nm,
    bearing_dir) by the same rotation — a combined force+couple load
    (e.g. a wrench) stays physically coherent. `budget` caps variants."""
    from cadx.cad.fem.loads import expand_cone, primary_dir, rot_between

    out: list[Any] = []
    for li, ld in enumerate(c.loads):
        ref = primary_dir(ld)
        if (ld.direction_cone_deg or 0) <= 0 or ref is None:
            continue
        for fvar in expand_cone(ref, float(ld.direction_cone_deg)):
            if len(out) >= budget:
                if warn:
                    warn_list.append("direction-cone expansion capped by max_cases")
                return out
            vc = c.model_copy(deep=True)
            vc.name = f"{c.name}_cone{len(out) + 1}"
            R = rot_between(ref, np.asarray(fvar, dtype=float))
            for attr in ("force_n", "moment_nm", "bearing_dir"):
                v = getattr(vc.loads[li], attr, None)
                if v and len(v) == 3 and np.linalg.norm(v) > 1e-9:
                    setattr(
                        vc.loads[li],
                        attr,
                        [float(x) for x in R @ np.asarray(v, dtype=float)],
                    )
            vc.loads[li].direction_cone_deg = None
            out.append(vc)
    return out
def _early_loadpath_cert(
    export_domain: Any,
    cases: list[Any],
    anchors: list[dict] | None,
    mat: Any,
    s: Any,
    warnings: list[str],
) -> tuple[float | None, float | None, bool | None, list[dict] | None]:
    """Run the min-cut load-path certificate before any FEM solve.

    Returns ``(bottleneck_mm2, sf_floor, load_path_ok, per_case)`` — the
    last one ``False`` when a case's load cannot reach its fixtures
    through the voxelized export, which no solve can rescue."""
    if not getattr(s, "fem_loadpath_cert", False):
        return None, None, None, None
    from cadx.cad.fem.loadpath_cert import loadpath_certificate
    from cadx.cad.fem.regions import resolve_case_dofs

    try:
        cases_res = []
        for c in cases:
            fixed3, F, w = resolve_case_dofs(
                export_domain,
                c,
                anchors,
                density_g_cm3=getattr(mat, "density_g_cm3", 1.24),
            )
            cases_res.append((fixed3, F))
            # region-resolution warnings explain WHY an endpoint came back
            # empty (fallback pinned a void patch) — surface them next to the
            # certificate's own reachability verdicts
            warnings.extend(w[:4])
        cert = loadpath_certificate(
            export_domain,
            cases_res,
            yield_mpa=float(getattr(mat, "yield_mpa", 0.0) or 0.0),
        )
        warnings.extend(cert.get("warnings") or [])
        per_case = cert.get("per_case")
        if isinstance(per_case, list):
            for rec, case in zip(per_case, cases, strict=False):
                rec["case_name"] = getattr(case, "name", None) or str(rec.get("case"))
        return (
            cert.get("bottleneck_mm2"),
            cert.get("sf_floor"),
            cert.get("load_path_ok"),
            per_case,
        )
    except Exception as e:
        warnings.append(f"load-path certificate failed: {str(e)[:80]}")
        return None, None, None, None
def _export_degeneracy(
    disp_mm: float | None,
    max_vm_mpa: float | None,
    span_mm: float,
    yield_mpa: float | None,
) -> tuple[bool, list[str]]:
    """Degeneracy gates on the export-verification numbers (B19).

    A linear voxel solve reporting a displacement on the order of the part
    span is not a weaker-but-valid part — it is an unconstrained or
    noodle-thin load path whose safety factor is meaningless; and a
    von Mises orders of magnitude past yield sits far outside the linear
    model's validity. Returns ``(valid, warnings)`` — ``valid=False`` marks
    sf_export untrustworthy. All thresholds are fractions of the domain
    diagonal `span_mm`; `yield_mpa` None/0 disables the stress gate.
    """
    warnings: list[str] = []
    valid = True
    vm_over = max_vm_mpa is not None and bool(yield_mpa) and max_vm_mpa > 10.0 * float(yield_mpa)
    if vm_over and max_vm_mpa is not None:
        warnings.append(
            f"max von Mises {max_vm_mpa:.0f} MPa exceeds 10x material yield — "
            "linear model far outside validity"
        )
    if disp_mm is None:
        return valid, warnings
    if disp_mm > 0.5 * span_mm:
        valid = False
        warnings.append(
            f"export FEM solve degenerate — displacement {disp_mm:.0f}mm exceeds "
            "half the part span; unconstrained or noodle-thin load path"
        )
    elif disp_mm > 0.15 * span_mm:
        warnings.append(
            f"unusually large displacement {disp_mm:.0f}mm — linear FEM "
            "questionable, inspect load paths"
        )
    if disp_mm > 0.3 * span_mm and vm_over:
        valid = False
        if not any("degenerate" in w for w in warnings):
            warnings.append(
                f"export FEM solve degenerate — displacement {disp_mm:.0f}mm "
                "combined with von Mises far past yield; unconstrained or "
                "noodle-thin load path"
            )
    if disp_mm > 10.0 * span_mm:
        # the historical hard gate — always implied by the 0.5-span rule
        # above, kept so the validity contract stays explicit
        valid = False
        if not any("degenerate" in w for w in warnings):
            warnings.append(
                "export FEM solve degenerate (unconstrained part or unresolved "
                "load regions) — sf_export not trustworthy"
            )
    return valid, warnings
def _feature_floor_pitch(anchors: list[dict] | None) -> float | None:
    """Upper bound the design pitch must keep so the smallest anchored bore
    still spans ~3.5 cells. None when no anchor carries a measured diameter
    — the pitch is then feature-agnostic (requested / span-snap / DOF cap
    only)."""
    dias = [float(a.get("dia") or 0.0) for a in (anchors or []) if float(a.get("dia") or 0.0) > 0]
    return min(dias) / _FEATURE_CELLS_PER_DIA if dias else None
def _gate_sf(info: dict[str, Any] | None, report: Any) -> float | None:
    """Effective export safety factor for gating: the verified worst-case
    realization when the hardened export sweep produced one, else the
    report's own worst safety factor (which is the mask-substituted
    fallback report when the export solve crashed). None-checked, not
    or-chained — a solved SF of 0.0 IS a verdict."""
    sf = (info or {}).get("sf_verified")
    return sf if sf is not None else report.worst_safety_factor
def _guard_export_grooves(
    exports: dict[str, str | None],
    exp_dir: Path,
    feats: list[Any],
    rep: Any,
) -> None:
    """Re-probe + re-cut helical grooves on the shipped mesh files.

    A heavy file repair reskins (alpha-wrap/voxel ≈ 0.5-0.7 mm granularity)
    — erasing the grooves the in-memory postpass already restored once."""
    from cadx.cad.fem.refine import recut_exported_threads

    for fmt in ("stl", "3mf", "glb"):
        fname = exports.get(fmt)
        if fname:
            recut_exported_threads(exp_dir / fname, feats, rep)
def _guide_beso_er(volfrac: float, iters: int, er_min: float = 0.02) -> float:
    """Adaptive BESO evolutionary ratio for a guide pass.

    The kept-volume ramp is ``V_{k+1} = max(volfrac, V_k·(1-ER))`` starting
    from a full field — the classic 2% step needs ~45 iterations to reach
    volfrac 0.4, so under a coarse-pass budget (~10-20) the guide would
    still sit near 80% fill and provide almost no guidance. Scale ER so
    the ramp can actually arrive at the target inside `iters`.
    """
    if iters <= 0 or volfrac <= 0:
        return er_min
    return max(float(er_min), 1.0 - float(volfrac) ** (1.0 / iters))
def _infill_volfrac_floor(dims: Any, pitch: float, fill_ratio: float | None) -> float | None:
    """Optimistic lower bound on the exported volfrac a shell+lattice run can
    reach: the closed skin preserves the silhouette and only the core deeper
    than the shell thickness can lattice down (bottoming out at the lattice
    occupancy floor). None when the grid gives no extents to reason from."""
    if fill_ratio is not None and fill_ratio <= 0.35:
        return _THIN_PART_INFILL_FLOOR
    try:
        extents = [float(d) * pitch for d in dims]
    except (TypeError, ValueError):
        extents = []
    if len(extents) != 3 or any(e <= 0 for e in extents):
        return None
    core_frac = 1.0
    for e in extents:
        core_frac *= max(0.0, (e - 2.0 * _INFILL_SHELL_MM) / e)
    return 1.0 - core_frac * (1.0 - _INFILL_LATTICE_MIN_OCCUPANCY)
def _is_cancelled(cancelled: Callable[[], bool] | None) -> bool:
    """Poll the optional cancellation checker — never raises (a flaky
    checker must not fail the pipeline)."""
    try:
        return bool(cancelled is not None and cancelled())
    except Exception:
        return False
def _load_pins_nodes(load: Any) -> bool:
    """Whether a load's region must be pinned in the passive mask — a
    g_load-only body force travels WITH the design (pinning it would freeze
    the whole domain), and an 'all' region is the domain itself. Only
    concentrated loads need their attachment nodes preserved.
    `normal_force_n` via getattr: the field lands with the schema update
    and may not exist on older payloads."""
    concentrated = bool(
        load.force_n
        or load.bearing_n
        or load.pressure_mpa
        or load.moment_nm
        or load.preload_n
        or getattr(load, "normal_force_n", None)
        or load.remote_point
    )
    if load.g_load and not concentrated:
        return False
    return getattr(load.region, "kind", "") != "all"
def _mark_cancelled(res: ExecResult) -> None:
    """Same 'cancelled' error shape the task-level checks produce."""
    res.ok = False
    res.error = "cancelled"
def _mark_step(key: str | None, name: str) -> None:
    if key is not None:
        _PIPELINE_STEPS[key] = (name, time.monotonic())
def _npz_kind(npz: Any) -> str:
    try:
        return str(np.asarray(npz["kind"]).item())
    except Exception:
        return ""
def _opt_cancel_pair(cancelled: Any, deadline: float | None):
    """Optimizer stop predicate + 'it was the deadline, not a cancel' flag.
    TopOpt tails treat a deadline stop as 'budget_exhausted' (still a
    usable partial density) while a cancel discards the run."""

    hit = [False]

    def _opt_cancelled() -> bool:
        if _is_cancelled(cancelled):
            return True
        if deadline is not None and time.monotonic() > deadline:
            hit[0] = True
            return True
        return False

    return _opt_cancelled, lambda: hit[0]
def _pitch_feature_warn(res: Any, domain: Any, anchors: list[dict] | None) -> None:
    """Feature-vs-pitch check: a bore only survives the design grid when its
    diameter spans a few cells — auto-coarsening past ~dia/3.5 rasterizes
    anchored holes down to ~3 cells and the extracted mesh comes out
    undersized no matter how good the optimizer is."""
    anchor_dias = [
        float(a.get("dia") or 0.0) for a in (anchors or []) if float(a.get("dia") or 0.0) > 0
    ]
    if not anchor_dias:
        return
    min_anchor_dia = min(anchor_dias)
    if domain.pitch * _FEATURE_CELLS_PER_DIA <= min_anchor_dia:
        return
    res.topopt["pitch_feature_warn"] = {
        "pitch_mm": round(float(domain.pitch), 4),
        "min_anchor_dia_mm": round(min_anchor_dia, 2),
    }
    res.topopt.setdefault("warnings", []).append(
        f"design pitch {domain.pitch:.2f}mm under-resolves the smallest anchored "
        f"bore Ø{min_anchor_dia:.1f}mm (needs ~{min_anchor_dia / 3.5:.1f}mm) — "
        "anchored holes may extract undersized; raise fem_spsolve_dofs or "
        "request a coarser min_member for a finer grid"
    )
def _post_extract_finish(
    opt_mesh: trimesh.Trimesh,
    *,
    base_mesh: trimesh.Trimesh | None,
    keep_regions: list[Any],
    anchors: list[dict[str, Any]] | None,
    spec_features: list[dict[str, Any]] | None,
    recut_bores: bool,
    restore_keeps: bool,
    helical_threads: bool = False,
) -> tuple[trimesh.Trimesh, dict[str, Any] | None, list[str]]:
    """Restore kept surfaces and deterministically re-cut real base bores.

    Keep this after extraction/smoothing but before every structural
    verification/export so the reported SF and connectivity certify the
    mesh that will ship. `resolve_refine_features` requires base-mesh tunnel
    evidence for scanned anchors, preventing dimples from becoming bores.
    """
    if not (recut_bores or restore_keeps):
        return opt_mesh, None, []
    if base_mesh is None:
        return opt_mesh, None, ["post-extraction restore/re-cut skipped: base mesh unavailable"]

    try:
        from cadx.cad.fem.refine import (
            apply_refine,
            rebind_keep_hole_indices,
            refine_verify,
            resolve_refine_features,
        )

        features: list[Any] = []
        warnings: list[str] = []
        if recut_bores:
            features, resolve_warnings = resolve_refine_features(
                anchors,
                spec_features,
                None,
                base_mesh=base_mesh,
            )
            warnings.extend(resolve_warnings)
        keep_dicts = [
            r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r) for r in keep_regions
        ]
        keep_dicts = rebind_keep_hole_indices(
            keep_dicts, spec_features or [], anchors or [], warnings
        )
        refined, report = apply_refine(
            opt_mesh,
            features,
            base_mesh=base_mesh,
            keep_regions=keep_dicts,
            anchors=anchors or [],
            restore_keeps=restore_keeps,
            helical_threads=helical_threads,
        )
        report_dict = dataclasses.asdict(report)
        warnings.extend(report.warnings)
        if recut_bores and report.applied:
            report_dict["verify_misses"] = refine_verify(
                refined,
                report_dict,
                anchors=anchors,
            )
        return refined, report_dict, warnings
    except Exception as e:
        return opt_mesh, None, [f"post-extraction restore/re-cut failed: {str(e)[:180]}"]
def _realization_case_details(
    report: Any, *, span_mm: float, yield_mpa: float | None
) -> list[dict[str, Any]]:
    """Serialize FEM cases and annotate solves outside the linear regime."""
    details = []
    for case in report.cases:
        valid, deg_warnings = _export_degeneracy(
            disp_mm=case.max_disp_mm,
            max_vm_mpa=case.max_vm_mpa,
            span_mm=span_mm,
            yield_mpa=yield_mpa,
        )
        details.append(
            {
                "case": case.name,
                "sf": case.safety_factor,
                "disp_mm": case.max_disp_mm,
                "vm_mpa": case.max_vm_mpa,
                "converged": case.converged,
                "degenerate": not valid,
                "warnings": [w for w in (case.warnings or [])][:6] + deg_warnings,
            }
        )
    return details
def _recommend_structure_mode(
    report: Any,
    *,
    keep_regions: list[Any] | None,
    anchors: list[Any] | None,
    target_volfrac: float | None = None,
) -> tuple[str, list[str]]:
    """Routing hint for the agent — which structure_mode a follow-up
    optimize run should start with.

    Mechanical inputs only: the baseline solve's fill ratio (solid volume
    over bbox volume), its worst safety factor, the presence of pinned
    features (keep regions / feature anchors), and — when the session's
    volume target is known — whether a silhouette-preserving construction
    could reach it at all. The agent combines the hint with user intent —
    it is a recommendation, never a routing decision.
    """
    grid = getattr(report, "grid", None) or {}
    dims = grid.get("dims") or []
    pitch = float(grid.get("pitch_mm") or 0.0)
    vol = float(grid.get("volume_mm3") or 0.0)
    bbox_vol = float(np.prod(dims)) * pitch**3 if dims and pitch > 0 else 0.0
    fill_ratio = vol / bbox_vol if bbox_vol > 0 else None
    fr = f"{fill_ratio:.2f}" if fill_ratio is not None else "unknown"
    sfs = [
        float(c.safety_factor)
        for c in (getattr(report, "cases", None) or [])
        if getattr(c, "safety_factor", None) is not None
    ]
    sf_min = min(sfs) if sfs else None
    n_pinned = len(keep_regions or []) + len(anchors or [])
    if sf_min is not None and sf_min < 1.5:
        mode, reasons = (
            "topopt",
            [
                f"baseline worst SF {sf_min:.2f} below the 1.5 default target — "
                "removing material cannot fix a failing baseline; full topology "
                "work (or a geometry edit first) is required"
            ],
        )
    elif fill_ratio is not None and fill_ratio <= 0.35:
        mode, reasons = (
            "infill",
            [
                f"thin-walled part (fill ratio {fr} <= 0.35) — the silhouette is "
                "already near-optimal; shell+infill costs ~2 solves and is "
                "watertight by construction"
            ],
        )
    elif sf_min is not None and sf_min >= 3.0:
        mode, reasons = (
            "infill",
            [
                f"large SF headroom ({sf_min:.1f} >= 2x the 1.5 target) — graded "
                "infill captures most of the benefit without a SIMP run"
            ],
        )
    elif n_pinned:
        mode, reasons = (
            "guided",
            [
                f"bulky part (fill ratio {fr}) with {n_pinned} pinned "
                "feature(s) — a coarse guide picks the retained topology and "
                "the lattice preserves the features"
            ],
        )
    else:
        mode, reasons = (
            "topopt",
            [
                f"bulky part (fill ratio {fr}) with open design freedom — "
                "full SIMP redistribution pays here"
            ],
        )
    if target_volfrac is not None and mode in {"infill", "guided"}:
        floor = _infill_volfrac_floor(dims, pitch, fill_ratio)
        if floor is not None and target_volfrac < floor:
            mode = "topopt"
            reasons.append(
                f"requested volfrac {target_volfrac:.2f} sits below what "
                f"shell+lattice can reach on this geometry (optimistic floor "
                f"~{floor:.2f}) — silhouette-preserving infill cannot lighten "
                "the part that far; SIMP removes real material, or the "
                "target itself is infeasible for the structure"
            )
    return mode, reasons
def _record_lost_load_regions(
    domain: Any,
    cases: list[Any],
    anchors: list[dict] | None,
    final_mask: np.ndarray,
    out: dict[str, Any],
    warnings: list[str],
) -> None:
    from cadx.cad.fem.regions import region_cell_mask

    lost: list[tuple[int, str, str]] = []
    for case_i, case in enumerate(cases):
        regions = [("fixed", region) for region in (case.fixed or [])]
        regions.extend(("load", load.region) for load in (case.loads or []))
        for role, region in regions:
            try:
                cells = region_cell_mask(domain, region, anchors).reshape(domain.dims)
            except Exception:
                continue
            if cells.any() and not bool((cells & final_mask).any()):
                label = getattr(region, "description", None) or getattr(region, "kind", "?")
                lost.append((case_i, role, label))
    if lost:
        out["regions_lost"] = [f"case {case_i} {role} '{label}'" for case_i, role, label in lost]
        for case_i, role, label in lost:
            warnings.append(
                f"case {case_i}: {role} region '{label}' fully removed by "
                "the optimizer — its load path cannot verify"
            )
def _refine_analytic_step(
    res: ExecResult,
    out: trimesh.Trimesh,
    feats: list[Any],
    rep: Any,
    out_dir: Path,
    name: str,
    warnings: list[str],
    s: RuntimeSettings,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Analytic STEP for refine — cut real Cylinder/Helix faces into the sewn
    solid for each applied feature; keeps the faceted STEP as fallback.

    rep.applied records are self-describing (u/v/dia/depth/thread) so the
    export does not depend on re-joining feature objects by name. Source
    mesh: the repaired on-disk STL — the exact artefact shipped to the user;
    the STL roundtrip + repair normalise duplicated vertices into a boolean-
    safe topology the leaky in-memory mesh lacks. The export runs isolated
    in a subprocess (``export_analytic_step_isolated``) because OCC booleans
    on sewn solids are unbounded — on timeout the faceted STEP stays."""
    from cadx.cad.fem.refine import applied_refine_features

    applied_feats = applied_refine_features(feats, rep)
    if rep.applied and not applied_feats:
        log.warning(
            "export.analytic_step_name_join_empty",
            applied=len(rep.applied),
            feats=len(feats),
        )
    analytic_src: list[Any] = applied_feats or list(rep.applied)
    if not analytic_src:
        return
    stl_name = res.exports.get("stl")
    if not stl_name:
        warnings.append("analytic STEP skipped — no STL export to sew")
        res.features["refine"]["warnings"] = warnings
        return
    # self-describing dicts for the subprocess boundary — feature objects
    # may not survive pickling; rep.applied records already carry u/v/dia
    feat_dicts = [f if isinstance(f, dict) else dataclasses.asdict(f) for f in analytic_src]
    try:
        step_res = export_analytic_step_isolated(
            out_dir / "exports" / stl_name,
            feat_dicts,
            out_dir / "exports" / f"{name}.step",
            timeout_s=float(s.analytic_step_timeout_s),
            max_faces=s.mesh_max_faces_for_cad,
            log=log,
            cancelled=cancelled,
        )
    except Exception as e:
        # the isolated export (spawn, sew, per-feature children) can raise —
        # a spawn failure must not kill a refine job that already produced a
        # good mesh + faceted STEP
        log.warning("export.analytic_step_isolated_failed", error=str(e)[:300])
        warnings.append(f"analytic STEP skipped — isolated export failed: {str(e)[:200]}")
        res.features["refine"]["warnings"] = warnings
        res.features["refine"]["step_features"] = []
        return
    if step_res.get("step"):
        res.exports["step"] = step_res["step"]
    res.features["refine"]["step_features"] = step_res.get("step_features") or []
    warnings += step_res.get("warnings") or []
    res.features["refine"]["warnings"] = warnings
def _render_quality(detail_mode: str) -> tuple[int, bool]:
    """(size, ghosts) for attempt vs export renders — preview attempts are
    judged on the contact sheet (512px tiles anyway) and internal x-ray
    shots only pay off on the winning export re-render."""
    return (1024, True) if detail_mode == "export" else (512, False)
def _repair_analysis_mesh(analysis: Analysis, stl_path: Path) -> None:
    """A leaky tessellation must not be handed to the slicer as-is — repair the
    analysis STL in place so downstream consumers (renders, DfM, critique)
    see the post-repair state. Runs unconditionally: the STL roundtrip can
    degrade a watertight in-memory mesh (non-manifold flap faces splitting
    into phantom bodies) — repair_mesh_file's heavy steps stay conditional
    on the file-level state it actually finds."""
    if analysis.watertight is not False:
        analysis.watertight_after_repair = analysis.watertight
    try:
        analysis.watertight_after_repair = repair_mesh_file(stl_path)
    except Exception as e:
        log.warning("export.repair_failed", error=str(e)[:200])
        analysis.watertight_after_repair = False
def _repair_export_meshes(exports: dict[str, str | None], exp_dir: Path) -> None:
    """Apply the same in-place repair to the shipped mesh formats."""
    for fmt in ("stl", "3mf", "glb"):
        fname = exports.get(fmt)
        if fname:
            try:
                repair_mesh_file(exp_dir / fname)
            except Exception as e:
                log.warning("export.repair_failed", fmt=fmt, error=str(e)[:200])
def _resolve_topopt_cases(
    domain: Any,
    cases: list[Any],
    anchors: list[dict] | None,
    mat: Any,
    max_cases: int,
    *,
    warn: bool = True,
    warnings_out: list[str] | None = None,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[float],
    list[np.ndarray],
    list[np.ndarray],
    list[str],
]:
    """Resolve load cases + passive-pin node masks on `domain` — shared by
    the SIMP/BESO pipeline (which calls it again on the coarse multires
    grid, warn=False) and the shell+infill pipeline.

    Returns (resolved (fixed3, F) tuples, case weights, fixed node masks,
    load node masks, unresolved-region messages). Direction-cone loads
    expand into extra case variants capped at `max_cases`; load regions
    that carry only body forces or match the whole part are not pinned —
    they would freeze the design domain."""
    from cadx.cad.fem.regions import region_node_mask, resolve_case_dofs

    warn_list = warnings_out if warnings_out is not None else []
    r_resolved: list[tuple[np.ndarray, np.ndarray]] = []
    r_weights: list[float] = []
    r_fixed_nm: list[np.ndarray] = []
    r_load_nm: list[np.ndarray] = []
    r_unresolved: list[str] = []

    def _add(c: Any, weight: float) -> None:
        fixed3, F, w = resolve_case_dofs(domain, c, anchors, density_g_cm3=mat.density_g_cm3)
        if warn:
            warn_list.extend(w)
            # a fixed/load region that matched nothing means the constraint
            # the spec declared is absent from the geometry — optimizing
            # anyway produces a structurally meaningless result
            r_unresolved.extend(
                m for m in w if re.search(r"(fixed|load) region .*matched no nodes", m)
            )
        if not fixed3.any() or not np.any(F):
            if warn:
                warn_list.append(f"case '{c.name}' skipped — no fixed region or no effective load")
            return
        r_resolved.append((fixed3, F))
        r_weights.append(weight)

    for c in cases:
        _add(c, float(c.weight))
        for vc in _cone_expand_case(c, max_cases - len(r_resolved), warn, warn_list):
            _add(vc, float(c.weight))
        for load in c.loads:
            # don't pin body-load or whole-part regions — gravity acts on
            # whatever material survives and `all` re-matches the
            # extracted mesh; pinning them would freeze the domain.
            if _load_pins_nodes(load):
                r_load_nm.append(region_node_mask(domain, load.region, anchors))
    for f3, _ in r_resolved:
        r_fixed_nm.append(np.asarray(f3).any(axis=1) if np.ndim(f3) == 2 else f3)
    return r_resolved, r_weights, r_fixed_nm, r_load_nm, r_unresolved
def _scan_anchors(mesh: Any, anchors: list[dict] | None, *, enabled: bool = True) -> list[dict]:
    """Merge mesh-scanned feature anchors (+x/+y/+z) into the declared list.

    `anchors` is the measured/declared list (may be None); the scan adds
    side holes the top view never saw and enriches known ones with real
    depth — both keyed into a re-indexed anchor table."""
    from cadx.cad.fem.anchors3d import merge_anchor_lists, scan_mesh_anchors

    if not enabled:
        return [dict(a) for a in anchors or []]
    try:
        scanned = scan_mesh_anchors(mesh)
    except Exception as e:
        log.warning("fem.anchor_scan_failed", error=str(e)[:200])
        scanned = []
    return merge_anchor_lists(list(anchors or []), scanned)
def _sf_basis(
    sf_realizations: dict[str, float | None],
    sf_realization_detail: dict[str, list[dict[str, Any]]],
    sf_verified: float | None,
    load_path_ok: bool | None,
) -> str:
    if load_path_ok is False:
        return "loadpath"
    if sf_verified is not None and any(
        sf_realizations.get(realization) == sf_verified
        and any(
            case.get("degenerate")
            and case.get("sf") is not None
            and float(case["sf"]) == sf_verified
            for case in cases
        )
        for realization, cases in sf_realization_detail.items()
    ):
        return "mechanism"
    return "stress"
def _snap_pitch_under_floor(
    eff_span: np.ndarray,
    floor: float,
    *,
    min_cells: int,
    mesh_volume: float,
    max_elements: int,
) -> tuple[float, str | None]:
    """Re-snap a design pitch that overshot the anchor feature floor.
    snapped_pitch may round the thin-axis cell count DOWN (p_snap > p0),
    landing above the floor. Re-snapping AT the floor keeps integer cells
    where possible; the result is still clamped to the floor — bore
    resolution wins over the exact integer-cell fit."""
    from cadx.cad.fem.voxelize import snapped_pitch

    p, warn = snapped_pitch(
        eff_span,
        floor,
        min_cells=min_cells,
        mesh_volume=mesh_volume,
        max_elements=max_elements,
    )
    return min(float(p), float(floor)), warn
def _solver_tier_gate(
    domain: Any, n_cases: int, s: RuntimeSettings, mem_budget: float
) -> tuple[int, dict[str, Any]]:
    """Effective direct-solve DOF cutoff honoring the memory budget.

    A kernel OOM kill mid-factorization is uncatchable (rc=137 loses the
    whole job and the worker process), so the direct tier is allowed only
    when the predicted factorization memory fits the budget AND the live
    cgroup headroom. Otherwise returns 0: ``_prepare_solve`` then takes the
    CG/AMG path for every case solve — a slow legitimate solve always beats
    a killed worker. Returns ``(spsolve_dofs_eff, info)`` where ``info``
    feeds ``solver_tier`` telemetry + warnings."""
    from cadx.cad.fem.voxelize import estimate_fem_memory_mb

    cap = int(getattr(s, "fem_spsolve_dofs", 120_000) or 120_000)
    dofs = 3 * domain.node_count
    info: dict[str, Any] = {
        "dofs": int(dofs),
        "cap": cap,
        "budget_mb": round(mem_budget, 1) if mem_budget else None,
    }
    if dofs > cap:
        info.update(tier="iterative", reason="dofs_cap")
        return 0, info
    est = estimate_fem_memory_mb(domain, n_cases, direct=True)
    headroom = resources.mem_headroom_mb()
    info["est_direct_mb"] = round(est, 1)
    info["headroom_mb"] = round(headroom, 1) if headroom is not None else None
    if mem_budget and est > mem_budget * 0.75:
        info.update(tier="iterative", reason="mem_budget")
        return 0, info
    if headroom is not None and est > headroom * 0.8:
        info.update(tier="iterative", reason="cgroup_headroom")
        return 0, info
    info["tier"] = "direct"
    return cap, info
def _spec_dims_from_dict(spec: dict[str, Any] | None) -> list[float | None] | None:
    """Stated [X,Y,Z] envelope dimensions from an executor spec payload."""
    if not isinstance(spec, dict):
        return None
    dims: list[float | None] = []
    for key in ("length_x", "width_y", "height_z"):
        row = spec.get(key)
        value = row.get("value") if isinstance(row, dict) else None
        source = row.get("source") if isinstance(row, dict) else None
        dims.append(float(value) if source == "stated" and isinstance(value, int | float) else None)
    return dims if any(d is not None for d in dims) else None
def _spec_envelope_bounds(
    spec_dims_mm: list[float | None], mesh_bounds: np.ndarray
) -> np.ndarray | None:
    """(2,3) axis-aligned bounds of the spec envelope in model space — X/Y
    centred on the origin, Z resting on the XY plane (the
    ``rest_on_xy_center`` contract). Axes with ``None``/non-positive dims are
    unconstrained: the box spans the mesh's own bounds there so the boolean
    clip only trims declared axes. None when nothing is constrained."""
    if len(spec_dims_mm) < 3:
        return None
    lo = np.asarray(mesh_bounds[0], dtype=float).copy()
    hi = np.asarray(mesh_bounds[1], dtype=float).copy()
    constrained = False
    for ax in range(3):
        d = spec_dims_mm[ax]
        if d is None or float(d) <= 0:
            continue
        constrained = True
        if ax == 2:
            lo[ax], hi[ax] = 0.0, float(d)
        else:
            lo[ax], hi[ax] = -float(d) / 2.0, float(d) / 2.0
    return np.stack([lo, hi]) if constrained else None
def _tier_warning(info: dict[str, Any]) -> str | None:
    """Human-readable reason the direct tier was skipped (None when direct)."""
    if info.get("tier") != "iterative" or info.get("reason") == "dofs_cap":
        return None
    return (
        f"direct solve skipped — predicted ~{info.get('est_direct_mb', 0):.0f} MB "
        f"factorization exceeds {info['reason']} "
        f"(budget {info.get('budget_mb')} MB, headroom {info.get('headroom_mb')} MB) — "
        "CG/AMG tier engages, expect slower iterations"
    )
def _uniform_baseline_check(
    solver: str,
    design_domain: Any,
    E0_arr: np.ndarray,
    nu: float,
    resolved: list[Any],
    passive: np.ndarray,
    p: Any,
    *,
    case_weights: Any = None,
    spsolve_dofs: int = 120_000,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """BESO honesty check — compare the converged binary field's compliance
    against the uniform-density baseline on the same domain/cases. Hard-kill
    ESO can settle on a worse local optimum than spreading the mass evenly;
    the flag surfaces that verdict instead of a bare 'converged'."""
    out: dict[str, Any] = {}
    if solver != "beso":
        return out
    from cadx.cad.fem.topopt import uniform_compliance

    base_c = uniform_compliance(
        design_domain,
        E0_arr,
        nu,
        resolved,
        passive,
        p,
        case_weights=case_weights,
        spsolve_dofs=spsolve_dofs,
    )
    if base_c is None:
        return out
    out["uniform_baseline_compliance_mj"] = round(float(base_c), 4)
    achieved_c = next(
        (
            float(h["compliance_mj"])
            for h in reversed(history or [])
            if isinstance(h, dict) and h.get("compliance_mj") is not None
        ),
        None,
    )
    if achieved_c is not None:
        out["worse_than_uniform"] = bool(achieved_c > base_c * 1.02)
    return out
def _upsample_density(
    rho_c: np.ndarray,
    dom_c: Any,
    dom_f: Any,
) -> np.ndarray:
    """Trilinearly resample a coarse SIMP density field onto the fine domain's
    cell grid (multiresolution warm start). Both grids are bbox-aligned so
    index coords map through world mm."""
    from scipy import ndimage

    coords = []
    for a in range(3):
        centers = dom_f.origin[a] + (np.arange(dom_f.dims[a]) + 0.5) * dom_f.pitch
        coords.append((centers - dom_c.origin[a]) / dom_c.pitch - 0.5)
    gi, gj, gk = np.meshgrid(*coords, indexing="ij")
    # "nearest" so boundary cells past the last coarse centre inherit the
    # edge cell's value instead of attenuating towards 0
    vals = ndimage.map_coordinates(rho_c.astype(float), [gi, gj, gk], order=1, mode="nearest")
    return np.clip(vals.ravel(), 0.0, 1.0)
def _verify_export_structure(  # noqa: C901
    *,
    candidate_mesh: Any,
    cases: list[Any],
    mat_name: str,
    mat: Any,
    design_pitch: float,
    max_elements: int,
    keep_list: list[Any],
    anchors: list[dict] | None,
    out_dir: Path,
    suffix: str,
    s: RuntimeSettings,
    solver_backend: str,
    cancelled: Callable[[], bool] | None,
    verify_robust: bool,
    source_mesh: Any = None,
) -> dict[str, Any]:
    """Shared hardened export verification for the topopt + infill pipelines.

    Re-voxelizes the export candidate at ``design_pitch`` (with
    coverage-weighted boundary cells when ``fem_verify_coverage_e``), runs
    the nominal export solve and — when ``verify_robust`` — the eroded /
    dilated mask realizations on the same grid. ``sf_verified`` is the
    worst safety factor across the solvable realizations and becomes the
    headline number downstream (falling back to the nominal export SF when
    the sweep is off or unproductive). ``fem_verify_mesh_shift`` adds a
    half-pitch-translated probe whose drift is reported as
    ``sf_mesh_sensitive``; ``fem_loadpath_cert`` attaches the
    FEM-independent min-cut certificate.

    The optional steps degrade to ``warnings`` — only a nominal-solve
    failure propagates, preserving the callers' ``verify_exp = verify``
    fallback. Returns::

        {report, n_bodies_export, n_bodies_voxel, sf_realizations,
         sf_verified, sf_realization_detail, bottleneck_mm2, sf_floor,
         load_path_ok, sf_mesh_sensitive, warnings}
    """
    from scipy import ndimage

    from cadx.cad.fem.analyze import FemReport, run_fem_domain
    from cadx.cad.fem.extract import sealed_void_metrics
    from cadx.cad.fem.voxelize import voxelize

    warnings: list[str] = []
    mem_budget = float(resources.effective_exec_mem_mb(s))
    skin = candidate_mesh if source_mesh is None else source_mesh
    use_coverage = bool(getattr(s, "fem_verify_coverage_e", False))
    if _is_cancelled(cancelled):
        return {"cancelled": True, "phase": "before_export_voxelize", "warnings": warnings}
    export_domain = voxelize(candidate_mesh, design_pitch, max_elements, coverage=use_coverage)
    if _is_cancelled(cancelled):
        return {"cancelled": True, "phase": "after_export_voxelize", "warnings": warnings}
    # same memory-aware tier policy as the pipeline guard: the verification
    # solves run per-case factorizations too — a too-large direct solve here
    # is just as OOM-prone as inside the SIMP loop
    spsolve_eff_v, tier_info_v = _solver_tier_gate(export_domain, len(cases) or 1, s, mem_budget)
    if warn := _tier_warning(tier_info_v):
        warnings.append(warn)
    void_metrics = sealed_void_metrics(export_domain.solid, export_domain.pitch)
    _solid_vol = float(export_domain.solid.sum() * export_domain.pitch**3)
    if _solid_vol > 0:
        # material-relative share — max_frac (grid norm) alone is not
        # comparable across open/thin parts where the bounding grid is
        # mostly air
        void_metrics["void_frac_of_solid"] = round(
            float(void_metrics["enclosed_void_mm3"]) / _solid_vol, 6
        )
    else:
        void_metrics["void_frac_of_solid"] = None
    export_grid = {
        "pitch": float(export_domain.pitch),
        "dims": list(export_domain.dims),
        "elements": int(export_domain.solid.sum()),
        "domain_volume_mm3": float(np.prod(export_domain.dims) * export_domain.pitch**3),
        "solid_volume_mm3": float(export_domain.solid.sum() * export_domain.pitch**3),
    }
    # the voxel raster is what the FEM *solves* — nested-but-disconnected
    # bodies can fuse into one component here; keep it separate from the
    # physical mesh body count so a fusion is measurable
    n_bodies_voxel = int(
        ndimage.label(
            export_domain.solid,
            structure=ndimage.generate_binary_structure(3, 1),
        )[1]
    )
    # the printed truth: positive-volume mesh components (inner cavity
    # walls carry negative volume and are not bodies)
    n_bodies_export = solid_body_count(candidate_mesh)
    if n_bodies_voxel > n_bodies_export:
        warnings.append(
            f"export rasterizes into {n_bodies_voxel} disconnected bodies on the "
            f"FEM grid though the mesh is {n_bodies_export} — connections thinner "
            "than the voxel pitch cannot carry load"
        )

    # cheap load-path certificate FIRST: when a case's resolved load and
    # fixture cells provably share no component, no FEM solve can produce a
    # meaningful safety factor — skip the nominal+realization sweep (each
    # costs a full solve) and report the broken path instead of burning the
    # structural budget on a doomed part.
    bottleneck_mm2, sf_floor, load_path_ok, cert_per_case = _early_loadpath_cert(
        export_domain, cases, anchors, mat, s, warnings
    )
    if _is_cancelled(cancelled):
        return {"cancelled": True, "phase": "after_loadpath_certificate", "warnings": warnings}
    if load_path_ok is False:
        warnings.append(
            "load-path certificate: load(s) cannot reach fixtures on the export "
            "grid — FEM sweep skipped on a provably disconnected part"
        )
        return {
            "report": FemReport(
                grid={
                    "pitch": design_pitch,
                    "dims": list(export_domain.dims),
                    "elements": int(export_domain.solid.sum()),
                }
            ),
            "n_bodies_export": n_bodies_export,
            "n_bodies_voxel": n_bodies_voxel,
            "sf_realizations": {},
            "sf_realization_detail": {},
            "sf_verified": None,
            "sf_verified_basis": "loadpath",
            **void_metrics,
            "export_grid": export_grid,
            "bottleneck_mm2": bottleneck_mm2,
            "sf_floor": sf_floor,
            "load_path_ok": load_path_ok,
            "load_path_cases": cert_per_case,
            "sf_mesh_sensitive": None,
            "warnings": warnings,
        }

    def _solve(
        mask: np.ndarray,
        sfx: str,
        od: Path | None = out_dir,
        dom: Any = None,
        workers: int | None = None,
        cg_maxiter: int = 5000,
    ) -> Any:
        # od=None for the auxiliary realizations — their reports feed
        # sf_verified only, so skip the unreferenced per-case PNG renders
        return run_fem_domain(
            export_domain if dom is None else dom,
            mask,
            cases,
            mat_name,
            material=mat,
            out_dir=od,
            anchors=anchors,
            keep_regions=keep_list,
            spsolve_dofs=spsolve_eff_v,
            pool_workers=s.exec_pool_workers if workers is None else workers,
            mem_budget_mb=mem_budget,
            suffix=sfx,
            source_mesh=skin,
            cancelled=cancelled,
            solver_backend=solver_backend,
            cg_maxiter=cg_maxiter,
        )

    report = _solve(export_domain.solid, suffix)
    if _is_cancelled(cancelled):
        return {
            "cancelled": True,
            "phase": "after_nominal_export_verify",
            "report": report,
            "warnings": warnings,
        }

    span = float(np.linalg.norm(export_domain.pitch * np.asarray(export_domain.dims)))
    yield_mpa = getattr(mat, "yield_mpa", None)

    def _realization_detail(rep: Any) -> list[dict[str, Any]]:
        """Keep per-case SFs auditable, including solves outside linear validity."""
        return _realization_case_details(rep, span_mm=span, yield_mpa=yield_mpa)

    sf_realizations: dict[str, float | None] = {"nominal": report.worst_safety_factor}
    sf_realization_detail: dict[str, Any] = {"nominal": _realization_detail(report)}
    if verify_robust:
        from concurrent.futures import ThreadPoolExecutor

        from cadx.cad import native as _native
        from cadx.cad.fem.voxelize import estimate_fem_memory_mb

        n_shift = int(getattr(s, "fem_verify_realization_cells", 1) or 1)
        n_shift = max(1, min(n_shift, 4))
        # mask morphs + coverage doms are cheap deterministic prep on the
        # caller thread — the solves below are what run long
        jobs: list[tuple[str, Any, Any]] = []
        for name, mask in (
            ("eroded", ndimage.binary_erosion(export_domain.solid, iterations=n_shift)),
            ("dilated", ndimage.binary_dilation(export_domain.solid, iterations=n_shift)),
        ):
            if not np.asarray(mask).any():
                sf_realizations[name] = None
                warnings.append(f"export realization '{name}' is empty — skipped")
                continue
            # coverage measures the NOMINAL mesh's fill fractions — on the
            # dilated mask the newly-solid cells model over-extrusion, so
            # they print FULL (coverage 1.0), not the ~0 fraction their
            # nominal-boundary position suggests
            dom = None
            if name == "dilated" and use_coverage and export_domain.coverage is not None:
                cov_d = np.asarray(export_domain.coverage, dtype=float).copy()
                cov_d[(np.asarray(mask) & ~export_domain.solid).ravel()] = 1.0
                dom = dataclasses.replace(export_domain, coverage=cov_d)
            jobs.append((name, mask, dom))
        if _is_cancelled(cancelled):
            return {
                "cancelled": True,
                "phase": "before_eroded_export_verify",
                "report": report,
                "warnings": warnings,
            }
        # two realizations overlap meaningfully only when the heavy kernels
        # release the GIL (native fem on) AND two solves fit the memory
        # budget — same gate vocabulary as the tier check above. Inner
        # pool_workers=0: forking a case pool from a worker thread is unsafe.
        est_par = estimate_fem_memory_mb(export_domain, len(cases) or 1, direct=bool(spsolve_eff_v))
        headroom_par = resources.mem_headroom_mb()
        parallel_realizations = (
            len(jobs) > 1
            and _native.native_enabled("fem")
            and (not mem_budget or 2 * est_par <= mem_budget * 0.75)
            and (headroom_par is None or 2 * est_par <= headroom_par * 0.8)
        )
        results: dict[str, Any] = {}
        if parallel_realizations:
            with ThreadPoolExecutor(max_workers=2) as tp:
                futs = {
                    name: tp.submit(
                        _solve,
                        mask,
                        f"{suffix}_{'ero' if name == 'eroded' else 'dil'}",
                        None,
                        dom,
                        0,
                        _AUX_CG_MAXITER,
                    )
                    for name, mask, dom in jobs
                }
                for name in futs:
                    try:
                        results[name] = futs[name].result()
                    except Exception as e:
                        results[name] = e
        else:
            for name, mask, dom in jobs:
                try:
                    results[name] = _solve(
                        mask,
                        f"{suffix}_{'ero' if name == 'eroded' else 'dil'}",
                        None,
                        dom,
                        cg_maxiter=_AUX_CG_MAXITER,
                    )
                except Exception as e:
                    results[name] = e
                if _is_cancelled(cancelled):
                    break
        for name, _mask, _dom in jobs:
            rep = results.get(name)
            if isinstance(rep, Exception):
                sf_realizations[name] = None
                warnings.append(f"export realization '{name}' solve failed: {str(rep)[:80]}")
            elif rep is not None:
                sf_realizations[name] = rep.worst_safety_factor
                sf_realization_detail[name] = _realization_detail(rep)
            else:
                # serial path broke on cancellation — the cancelled return
                # below fires before this None is ever read as a verdict
                sf_realizations[name] = None
            if _is_cancelled(cancelled):
                return {
                    "cancelled": True,
                    "phase": f"after_{name}_export_verify",
                    "report": report,
                    "warnings": warnings,
                }
            if sf_realizations[name] is None and name in results:
                warnings.append(f"export realization '{name}' produced no safety factor")
    # sf_verified only exists when the NOMINAL realization solved — the
    # headline structure itself being unsolvable IS the verdict; a
    # dilated-only success must never read as "verified". When nominal is
    # None the gate falls back to the (equally None) report SF and the
    # structure is flagged for review.
    solvable = [v for v in sf_realizations.values() if v is not None]
    sf_verified = min(solvable) if sf_realizations["nominal"] is not None else None
    sf_verified_basis = _sf_basis(sf_realizations, sf_realization_detail, sf_verified, load_path_ok)

    sf_mesh_sensitive: bool | None = None
    if getattr(s, "fem_verify_mesh_shift", False):
        if _is_cancelled(cancelled):
            return {
                "cancelled": True,
                "phase": "before_mesh_shift_verify",
                "report": report,
                "warnings": warnings,
            }
        try:
            shifted = candidate_mesh.copy()
            shifted.apply_translation(np.full(3, design_pitch / 2.0))
            dom_sh = voxelize(shifted, design_pitch, max_elements, coverage=use_coverage)
            rep_sh = run_fem_domain(
                dom_sh,
                dom_sh.solid,
                cases,
                mat_name,
                material=mat,
                out_dir=None,
                anchors=anchors,
                keep_regions=keep_list,
                spsolve_dofs=spsolve_eff_v,
                pool_workers=s.exec_pool_workers,
                mem_budget_mb=mem_budget,
                suffix="",
                source_mesh=None,
                cancelled=cancelled,
                solver_backend=solver_backend,
            )
            if _is_cancelled(cancelled):
                return {
                    "cancelled": True,
                    "phase": "after_mesh_shift_verify",
                    "report": report,
                    "warnings": warnings,
                }
            sf_sh = rep_sh.worst_safety_factor
            if sf_sh is not None and sf_verified:
                delta = abs(sf_sh - sf_verified) / max(sf_verified, 1e-9)
                sf_mesh_sensitive = bool(delta > 0.2)
                if sf_mesh_sensitive:
                    warnings.append(
                        f"export SF shifts {delta:.0%} under a half-pitch mesh "
                        f"translation ({sf_verified:.2f} -> {sf_sh:.2f}) — the "
                        "result is voxel-grid sensitive"
                    )
            elif sf_sh is None:
                warnings.append("mesh-shift verification produced no safety factor")
        except Exception as e:
            warnings.append(f"mesh-shift verification failed: {str(e)[:80]}")
        if _is_cancelled(cancelled):
            return {
                "cancelled": True,
                "phase": "after_mesh_shift_verify",
                "report": report,
                "warnings": warnings,
            }

    return {
        "report": report,
        "n_bodies_export": n_bodies_export,
        "n_bodies_voxel": n_bodies_voxel,
        "sf_realizations": sf_realizations,
        "sf_realization_detail": sf_realization_detail,
        "sf_verified": sf_verified,
        "sf_verified_basis": sf_verified_basis,
        **void_metrics,
        "export_grid": export_grid,
        "bottleneck_mm2": bottleneck_mm2,
        "sf_floor": sf_floor,
        "load_path_ok": load_path_ok,
        "load_path_cases": cert_per_case,
        "sf_mesh_sensitive": sf_mesh_sensitive,
        "warnings": warnings,
    }
def copy_artifacts(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        s = src / n
        if s.exists():
            shutil.copy2(s, dst / n)
def export_from_brep(
    brep_path: Path, out_dir: Path, name: str = "model", flat_brep: Path | None = None
) -> dict[str, str | None]:
    """Full export set from an existing BREP (used by FINALIZE / reexport)."""
    shape = import_brep(str(brep_path))
    out = export_all_3d(shape, out_dir, name=name)
    out.update(export_2d(shape, out_dir, name, flat_brep))
    # finalize/reexport bypasses the run_code fan-out repair — the shipped
    # STL/3MF/GLB must not leak a tessellation seam to the slicer
    _repair_export_meshes(out, out_dir)
    return out
def export_from_mesh(
    stl_path: Path, out_dir: Path, name: str = "model", *, max_faces: int | None = None
) -> dict[str, str | None]:
    """Export set for mesh-mode versions from a stored STL (used by FINALIZE / reexport)."""
    mesh = load_triangle_mesh(stl_path)
    out = export_mesh_all(mesh, out_dir, name=name, max_faces=max_faces)
    out["dxf"] = None
    out["svg"] = None
    _repair_export_meshes(out, out_dir)
    return out
def normalize_mesh_to_spec_envelope(
    mesh: trimesh.Trimesh,
    spec_dims_mm: list[float | None],
    *,
    scale_fallback: bool = True,
    slack_mm: float | None = None,
    slack_frac: float | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Clip an out-of-envelope topopt input back to the declared spec dims.

    The clip is a manifold boolean intersection with the spec-envelope box
    (same engine as the rest of the codebase). A failed/empty boolean may
    fall back to uniform scale-to-fit on a watertight input. Export callers
    can disable that fallback so a failed trim is surfaced instead of
    silently rescaling optimized geometry.

    Returns ``(mesh, report)``; report keys: ``input_bbox``/``spec_bbox``
    (flat [x, y, z] mm extents — the frontend's TopOptPanel contract),
    ``input_normalized``, ``method`` ("clip" | "scale") and ``warning``.
    """
    report: dict[str, Any] = {
        "input_bbox": [round(float(v), 3) for v in (mesh.bounds[1] - mesh.bounds[0])],
        "spec_bbox": None,
        "input_normalized": False,
    }
    bounds = np.asarray(mesh.bounds, dtype=float)
    ext = bounds[1] - bounds[0]
    env = _spec_envelope_bounds(spec_dims_mm, bounds)
    if env is None:
        return mesh, report
    if all(d and d > 0 for d in spec_dims_mm):
        report["spec_bbox"] = [round(float(d), 3) for d in spec_dims_mm]
    abs_slack = _ENVELOPE_CLIP_SLACK_MM if slack_mm is None else max(0.0, float(slack_mm))
    rel_slack = _ENVELOPE_CLIP_SLACK_FRAC if slack_frac is None else max(0.0, float(slack_frac))
    over = any(
        ext[ax] > float(d) + abs_slack
        or ext[ax] > float(d) * (1.0 + rel_slack)
        or bounds[0, ax] < env[0, ax] - abs_slack
        or bounds[1, ax] > env[1, ax] + abs_slack
        for ax, d in enumerate(spec_dims_mm)
        if d and d > 0
    )
    if not over:
        return mesh, report

    box = trimesh.creation.box(extents=env[1] - env[0])
    box.apply_translation((env[0] + env[1]) / 2.0)
    clipped: trimesh.Trimesh | None = None
    try:
        cand = mesh if mesh.is_watertight else _ensure_watertight(mesh)
        clipped = cand.intersection(box, engine="manifold")
    except Exception as e:
        report["warning"] = f"spec-envelope clip failed: {e} — input left unnormalized"
    if clipped is not None and not clipped.is_empty and clipped.is_volume and clipped.volume > 0:
        report["input_normalized"] = True
        report["method"] = "clip"
        return clipped, report
    if "warning" not in report:
        report["warning"] = "spec-envelope clip produced no usable volume — input left as-is"

    if not scale_fallback:
        report["warning"] += " — scale fallback disabled"
        return mesh, report

    # bounds-based fallback: a watertight solid stays watertight under uniform
    # scaling about the origin (XY-centred/on-plane inputs keep the contract)
    if mesh.is_watertight:
        factors = [
            float(d) / float(ext[ax])
            for ax, d in enumerate(spec_dims_mm)
            if d and d > 0 and ext[ax] > d
        ]
        if factors:
            out = mesh.copy()
            out.apply_scale(min(factors))
            report["input_normalized"] = True
            report["method"] = "scale"
            report["warning"] += " — fell back to uniform scale-to-fit"
            return out, report
    return mesh, report
def pipeline_step(key: str) -> tuple[str, float] | None:
    """Last heartbeat of the run identified by `key` (its out_dir) or None."""
    return _PIPELINE_STEPS.get(key)
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
