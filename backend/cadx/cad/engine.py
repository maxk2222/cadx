"""Pipeline executed inside the cad-executor: sandbox -> analysis -> renders -> DfM -> exports.

Pure function (no DB, no queue) so it can be reused by the arq task, the CLI and tests.
All inputs are file paths + plain settings dicts; output is an ExecResult.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
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
from cadx.cad.analysis import _FACET_FACE_CAP, Analysis, analyze
from cadx.cad.contact_sheet import make_contact_sheet
from cadx.cad.dfm import DfmReport
from cadx.cad.dfm.cnc import cnc_checks, cnc_mesh_checks
from cadx.cad.dfm.common import common_checks, load_mesh
from cadx.cad.dfm.fdm import fdm_checks, fdm_mesh_checks
from cadx.cad.dfm.report import DfmWarning
from cadx.cad.exporters import (
    _premesh,
    export_2d,
    export_all_3d,
    export_analytic_step_isolated,
    export_mesh_all,
    quick_stl,
    repair_mesh_file,
)
from cadx.cad.mesh_ops import (
    MESH_OPS_VERSION,
    MeshPostParams,
    _ensure_watertight,
    apply_ops,
    baseline_mesh,
    candidate_meshes,
    clean,
    decimate,
    flat_base_ratio,
    is_flat_bottom,
    keep_significant_components,
    mesh_analysis,
    mesh_to_solid,
    rest_on_xy_center,
    smooth_preserve,
    solid_body_count,
)
from cadx.cad.mesh_ops import (
    load_mesh as load_triangle_mesh,
)
from cadx.cad.render import render_thumbnail, render_views
from cadx.cad.sandbox import Reference, RunnerOutput, run_user_code
from cadx.db.models import ProjectSettings
from cadx.llm.schemas import PartSpec
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)

# STEP roundtrip gate: skip when the pipeline has already burned most of the
# per-execution budget — the gate adds a full export+import roundtrip
_STEP_GATE_BUDGET_FRACTION = 0.8

# Mesh-derived (faceted) solids: their STEP export carries every triangle as
# an analytic face (~18k faces → 58 MB, minutes of OCC time). A full
# roundtrip inside an iteration run is not worth that cost — the
# deterministic mesh checks carry the evidence instead.
_STEP_GATE_MESH_MAX_FACES = 4_000

# per-run step heartbeat: the executor thread cannot be killed on a hang —
# this map at least tells the task-level timeout WHICH step is grinding
# (observed: pipeline_hang with no clue whether sandbox, OCC boolean,
# tessellation or the STEP gate was stuck)
_PIPELINE_STEPS: dict[str, tuple[str, float]] = {}


def _mark_step(key: str | None, name: str) -> None:
    if key is not None:
        _PIPELINE_STEPS[key] = (name, time.monotonic())


def pipeline_step(key: str) -> tuple[str, float] | None:
    """Last heartbeat of the run identified by `key` (its out_dir) or None."""
    return _PIPELINE_STEPS.get(key)


def _is_cancelled(cancelled: Callable[[], bool] | None) -> bool:
    """Poll the optional cancellation checker — never raises (a flaky
    checker must not fail the pipeline)."""
    try:
        return bool(cancelled is not None and cancelled())
    except Exception:
        return False


def _mark_cancelled(res: ExecResult) -> None:
    """Same 'cancelled' error shape the task-level checks produce."""
    res.ok = False
    res.error = "cancelled"


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


def _repair_analysis_mesh(analysis: Analysis, stl_path: Path) -> None:
    """A leaky tessellation must not be handed to the slicer as-is — repair the
    analysis STL in place so downstream consumers (renders, DfM, critique)
    see the post-repair state."""
    if analysis.watertight is False:
        try:
            analysis.watertight_after_repair = repair_mesh_file(stl_path)
        except Exception as e:
            log.warning("export.repair_failed", error=str(e)[:200])
            analysis.watertight_after_repair = False
    else:
        analysis.watertight_after_repair = analysis.watertight


def _repair_export_meshes(exports: dict[str, str | None], exp_dir: Path) -> None:
    """Apply the same in-place repair to the shipped mesh formats."""
    for fmt in ("stl", "3mf", "glb"):
        fname = exports.get(fmt)
        if fname:
            try:
                repair_mesh_file(exp_dir / fname)
            except Exception as e:
                log.warning("export.repair_failed", fmt=fmt, error=str(e)[:200])


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
            and spec_dict.get("features")
            and measured
            and rt.agent_feature_feedback
        ):
            try:
                feats = [Feature.model_validate(f) for f in spec_dict["features"]]
                feature_findings = [
                    f.model_dump() for f in compare_features(feats, MeasuredFeatures(**measured))
                ]
            except Exception as e:
                log.warning("verify_spec.compare_failed", error=str(e)[:200])
        res.verify_spec = verify_spec(
            spec_dict,
            res.to_dict(),
            feature_findings=feature_findings,
            measured_features=measured,
            dim_tolerance_pct=tol_pct,
            dim_tolerance_mm=tol_mm,
            envelope_bound_only=envelope_bound_only,
        )
    except Exception as e:
        log.warning("verify_spec.failed", error=str(e)[:200])
    return res


def run_pipeline(  # noqa: C901
    *,
    code: str,
    out_dir: Path,
    params_override: dict[str, object] | None = None,
    references: list[Reference] | None = None,
    settings: ProjectSettings | None = None,
    runtime: RuntimeSettings | None = None,
    full_exports: bool = False,
    name: str = "model",
    mesh_solid: Path | None = None,
    mesh_bbox: list[float] | None = None,
    anchors: dict[str, Any] | None = None,
    light: bool = False,
    spec: dict[str, Any] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ExecResult:
    """Full executor pipeline for one candidate. out_dir receives all artifacts.

    `light=True` skips renders/thumbnail/contact-sheet and DfM — used for
    tool-session verifies where only geometry numbers matter (~2 s cheaper
    per sandbox verify).
    """
    res = ExecResult()
    settings = settings or ProjectSettings()
    rt = runtime or RuntimeSettings()
    t0 = time.monotonic()
    # design-intent spec for the verify_spec report — the execute payload
    # carries it inside the free-form anchors dict (cad/tasks.py only forwards
    # known keys); the explicit kwarg covers direct callers
    spec_dict: dict[str, Any] | None = spec
    if spec_dict is None and isinstance(anchors, dict):
        embedded = anchors.get("spec")
        spec_dict = embedded if isinstance(embedded, dict) else None

    step_key = str(out_dir)
    _mark_step(step_key, "sandbox")
    # 1. sandboxed execution
    workdir = out_dir / "_run"
    ro: RunnerOutput = run_user_code(
        code,
        workdir,
        params_override=params_override,
        references=references,
        mesh_solid=mesh_solid,
        mesh_bbox=mesh_bbox,
        anchors=anchors,
        timeout_s=rt.exec_timeout_s,
        mem_mb=rt.exec_mem_limit_mb,
        warm=rt.exec_warm_runner,
        threads=rt.exec_threads,
    )
    res.stdout = ro.stdout
    res.params_used = ro.params_used
    res.thread_count = ro.thread_count
    res.bom = ro.bom
    res.warnings = list(ro.warnings)
    res.geometry_at_failure = list(ro.geometry_at_failure)
    res.timings["execute_s"] = round(ro.duration_s, 2)
    if not ro.ok or ro.brep_path is None:
        res.error = ro.error or "execution failed"
        res.traceback = ro.traceback
        return _attach_verify_spec(res, spec_dict, settings, rt)

    # 2. load shape
    try:
        shape = import_brep(str(ro.brep_path))
    except Exception as e:
        res.error = f"failed to load generated shape: {e}"
        return _attach_verify_spec(res, spec_dict, settings, rt)

    # hybrid: the runner ships the result's manifold mesh next to the brep —
    # re-attach it so tessellation/analysis skip re-meshing the faceted solid
    _res_glb = ro.brep_path.parent / "result_mesh.glb"
    if _res_glb.exists():
        try:
            from cadx.cad.mesh_boolean import attach_mesh

            attach_mesh(shape, load_triangle_mesh(_res_glb))
        except Exception:
            pass

    # 3. quick STL for mesh analysis + renders
    _mark_step(step_key, "tessellate")
    t = time.monotonic()
    mesh_dir = out_dir / "renders"
    try:
        stl_path = quick_stl(shape, mesh_dir, name="mesh")
    except Exception as e:
        res.error = f"tessellation failed: {e}"
        return _attach_verify_spec(res, spec_dict, settings, rt)
    res.timings["tessellate_s"] = round(time.monotonic() - t, 2)

    # 4. analysis
    _mark_step(step_key, "analyze")
    res.analysis = analyze(shape, stl_path)
    _repair_analysis_mesh(res.analysis, stl_path)
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return _attach_verify_spec(res, spec_dict, settings, rt)

    # 4b. deterministic feature extraction (feedback-only, never blocks)
    try:
        from cadx.cad.features import extract_features

        res.features = extract_features(shape, stl_path).model_dump()
    except Exception as e:
        log.warning("features.extract_failed", error=str(e)[:200])

    # 4b1. declared connection spans: a "strut from A to B" must carry
    # material along the whole path — a detached bar reads as correctly
    # connected to the visual critic, so the centreline is sampled here
    try:
        spec_feats = (spec_dict or {}).get("features") or []
        if spec_feats:
            from cadx.cad.spans import span_checks

            if not isinstance(res.features, dict):
                res.features = {}
            res.features["spans"] = span_checks(shape, spec_feats)
            from cadx.cad.spans import plate_checks

            res.features["plates"] = plate_checks(shape, spec_feats, spec_dict=spec_dict)
    except Exception as e:
        log.warning("features.span_checks_failed", error=str(e)[:200])

    # 4b2. invalid B-Rep → locate defects so repair feedback gets coordinates
    # (WHERE geometry is broken), not just "is_valid: false". On faceted
    # solids BRepCheck localization is a multi-minute per-face grind with no
    # actionable output — the faceted fast path already gates on it
    if not res.analysis.is_valid and res.analysis.face_count <= _FACET_FACE_CAP:
        try:
            from cadx.cad.defects import locate_defects, summarize_defects

            defects = locate_defects(shape)
            if defects and not all("error" in d for d in defects):
                res.features["defects"] = defects[:5]
                res.warnings.append(f"invalid B-Rep: {summarize_defects(defects)}")
        except Exception as e:
            log.warning("defects.locate_failed", error=str(e)[:200])

    # 4c. footprint agreement with the measured plan view (image outline)
    if anchors and anchors.get("image_outline"):
        try:
            import trimesh
            from shapely.geometry import Polygon

            from cadx.cad.mesh_features import detect_mesh_features

            tm = trimesh.load(str(stl_path), force="mesh", process=False)
            if not isinstance(tm, trimesh.Trimesh):
                raise TypeError("not a single mesh")
            out_pts = detect_mesh_features(tm, step_mm=1.0).outline_mm
            if len(out_pts) >= 3:
                a = Polygon(anchors["image_outline"]).buffer(0)
                b = Polygon(out_pts).buffer(0)
                union = a.union(b).area
                if union > 0:
                    res.analysis.footprint_iou = round(float(a.intersection(b).area / union), 3)
        except Exception as e:
            log.warning("features.footprint_iou_failed", error=str(e)[:200])

    exp_dir = out_dir / "exports"

    def _do_renders() -> float:
        # 5. renders + contact sheet + thumbnail
        t = time.monotonic()
        render_paths = render_views(stl_path, mesh_dir, res.analysis.bbox_size)
        res.renders = {v: p.name for v, p in render_paths.items()}
        thumb = render_thumbnail(stl_path, out_dir / "thumbnail.png")
        res.thumbnail = thumb.name if thumb else None
        stats = [
            f"bbox: {res.analysis.bbox_size[0]:.1f} x {res.analysis.bbox_size[1]:.1f} x {res.analysis.bbox_size[2]:.1f} mm",
            f"volume: {res.analysis.volume:.0f} mm^3",
            f"solids: {res.analysis.solid_count}",
            f"valid: {res.analysis.is_valid}",
            f"watertight: {res.analysis.watertight_after_repair}",
        ]
        sheet = make_contact_sheet(render_paths, out_dir / "contact_sheet.png", stats)
        res.contact_sheet = sheet.name if sheet else None
        return time.monotonic() - t

    def _do_dfm() -> float:
        # 6. DfM
        t = time.monotonic()
        mesh = load_mesh(stl_path)
        warnings = common_checks(res.analysis, mesh, settings.build_volume)
        if settings.process in ("fdm", "sla"):
            warnings += fdm_checks(
                shape,
                mesh,
                res.analysis,
                min_wall_mm=settings.min_wall_mm,
                nozzle_mm=settings.nozzle_mm,
                thread_count=res.thread_count,
                thread_compensation_mm=settings.thread_compensation_mm,
            )
        if settings.process in ("cnc3", "cnc2d"):
            warnings += cnc_checks(
                shape,
                mesh,
                res.analysis,
                tool_radius_mm=settings.tool_radius_mm,
                thread_count=res.thread_count,
            )
        res.dfm = DfmReport(process=settings.process, warnings=warnings).sort()
        return time.monotonic() - t

    def _do_exports() -> float:
        # 7. exports
        t = time.monotonic()
        if full_exports:
            res.exports = export_all_3d(shape, exp_dir, name=name, parallel=rt.exec_pool_workers)
            res.exports.update(export_2d(shape, exp_dir, name, ro.flat_brep_path))
            if res.analysis.watertight is False:
                # same repair the analysis STL got — the shipped mesh formats
                # must not leak non-watertight geometry to the slicer
                _repair_export_meshes(res.exports, exp_dir)
        else:
            # iterations only need the STL (already exported for renders)
            res.exports = {"stl": None}
        return time.monotonic() - t

    fanout = resources.pool_size(rt.exec_pool_workers) > 1 and not light
    if fanout:
        # one shared OCC triangulation up front — after this every consumer
        # below (writers, DfM face reads) is read-only on the shape, which
        # makes the thread fan-out safe (no concurrent re-mesh writes)
        _premesh(shape, 0.01, 0.1)

    _mark_step(step_key, "render" if not light else "export")
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return _attach_verify_spec(res, spec_dict, settings, rt)

    if fanout:
        from concurrent.futures import ThreadPoolExecutor

        _mark_step(step_key, "export")
        t = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                f_ren = pool.submit(_do_renders)
                f_dfm = pool.submit(_do_dfm)
                f_exp = pool.submit(_do_exports)
                render_s, dfm_s, export_s = f_ren.result(), f_dfm.result(), f_exp.result()
        except Exception as e:  # a crashed fan-out retries serially
            log.warning("pipeline.fanout_failed", error=str(e)[:200])
            render_s, dfm_s, export_s = _do_renders(), _do_dfm(), _do_exports()
        res.timings["render_s"] = round(render_s, 2)
        res.timings["dfm_s"] = round(dfm_s, 2)
        res.timings["export_s"] = round(export_s, 2)
    else:
        if not light:
            res.timings["render_s"] = round(_do_renders(), 2)
            res.timings["dfm_s"] = round(_do_dfm(), 2)
        else:
            res.timings["render_s"] = 0.0
            res.timings["dfm_s"] = 0.0
        _mark_step(step_key, "export")
        if _is_cancelled(cancelled):
            _mark_cancelled(res)
            return _attach_verify_spec(res, spec_dict, settings, rt)
        res.timings["export_s"] = round(_do_exports(), 2)

    # 8. STEP roundtrip gate — a shape can be valid in memory yet break on
    # serialisation; bounded by face count (inside the check) and by the
    # remaining exec budget here. Success-path only, never fatal.
    # Light runs (tool-session verifies) ship no STEP at all — a full
    # export+import roundtrip on an ~18k-face mesh solid is a >400 s OCC
    # grind that was observed to trigger pipeline_hang; the gate is skipped
    # outright. Mesh-derived (faceted) solids get a much lower cap for the
    # same reason.
    t = time.monotonic()
    if not light and t - t0 <= rt.exec_timeout_s * _STEP_GATE_BUDGET_FRACTION:
        try:
            from cadx.cad.defects import STEP_GATE_MAX_FACES, step_roundtrip_check

            gate_faces = (
                _STEP_GATE_MESH_MAX_FACES if mesh_solid is not None else STEP_GATE_MAX_FACES
            )
            shipped = res.exports.get("step")
            if shipped:
                gate = step_roundtrip_check(
                    shape, exp_dir / shipped, write=False, max_faces=gate_faces
                )
            else:
                gate = step_roundtrip_check(shape, workdir / "step_gate.step", max_faces=gate_faces)
            # store the gate result unconditionally — verify_spec reads it as
            # structural evidence (ok→PASS, warning→FAIL, skipped→UNVERIFIED)
            res.features["step_roundtrip"] = gate
            if gate.get("warning"):
                res.warnings.append(str(gate["warning"]))
            elif gate.get("skipped"):
                log.info("step_gate.skipped", reason=gate["skipped"])
            elif gate.get("error"):
                log.warning("step_gate.error", error=str(gate["error"])[:200])
        except Exception as e:
            log.warning("step_gate.failed", error=str(e)[:200])
        res.timings["step_gate_s"] = round(time.monotonic() - t, 2)

    res.ok = True
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    # 9. canonical conformance report — reads only evidence already in res
    return _attach_verify_spec(res, spec_dict, settings, rt)


def export_from_brep(
    brep_path: Path, out_dir: Path, name: str = "model", flat_brep: Path | None = None
) -> dict[str, str | None]:
    """Full export set from an existing BREP (used by FINALIZE / reexport)."""
    shape = import_brep(str(brep_path))
    out = export_all_3d(shape, out_dir, name=name)
    out.update(export_2d(shape, out_dir, name, flat_brep))
    return out


def export_from_mesh(
    stl_path: Path, out_dir: Path, name: str = "model", *, max_faces: int | None = None
) -> dict[str, str | None]:
    """Export set for mesh-mode versions from a stored STL (used by FINALIZE / reexport)."""
    mesh = load_triangle_mesh(stl_path)
    out = export_mesh_all(mesh, out_dir, name=name, max_faces=max_faces)
    out["dxf"] = None
    out["svg"] = None
    return out


# --------------------------------------------------------------------------- #
# mesh pipeline (mesh / hybrid modes)                                          #
# --------------------------------------------------------------------------- #


def run_mesh_pipeline(  # noqa: C901
    *,
    mesh_path: Path,
    out_dir: Path,
    post: MeshPostParams | None = None,
    spec: PartSpec | None = None,
    settings: ProjectSettings | None = None,
    runtime: RuntimeSettings | None = None,
    full_exports: bool = False,
    name: str = "model",
    make_solid: bool = False,
    skip_baseline: bool = False,
    anchors: dict[str, Any] | None = None,
    solid_max_faces: int | None = None,
    ref_mask_rel: str | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ExecResult:
    """Mesh-mode pipeline: clean -> orient -> scale -> analyse -> render -> DfM -> export.

    Input is the raw generator output (GLB/STL/...) in arbitrary normalized
    space; the result follows the frame contract (mm, +Z up, on XY, centred).
    With `skip_baseline` the input is already a finished mm-frame model
    (GUI mesh edit on a produced STL): only a light clean runs before the
    recorded ops — no decimate/orient/scale/flatten.
    """
    res = ExecResult()
    settings = settings or ProjectSettings()
    rt = runtime or RuntimeSettings()
    post = post or MeshPostParams()
    t0 = time.monotonic()
    workdir = out_dir / "_run"
    workdir.mkdir(parents=True, exist_ok=True)

    def _done() -> ExecResult:
        """Attach the canonical verify_spec report, then return ``res``.

        Mesh path: the measured-features evidence lives under
        ``res.mesh["features"]`` (ray-grid tunnel/dimple detection) and must be
        adapted into MeasuredFeatures before compare_features can run on it.
        """
        measured = findings = None
        if spec is not None:
            try:
                from cadx.cad.features import (
                    compare_features,
                    mesh_features_to_measured,
                )

                mf = res.mesh.get("features") or {}
                m = mesh_features_to_measured(mf) if mf else None
                if m is not None:
                    measured = m.model_dump()
                    if spec.features and rt.agent_feature_feedback:
                        findings = [f.model_dump() for f in compare_features(spec.features, m)]
            except Exception as e:
                log.warning("verify_spec.mesh_features_failed", error=str(e)[:200])
        spec_dict = spec.model_dump() if spec is not None else None
        if spec_dict is not None:
            spec_dict.setdefault("min_wall_mm", settings.min_wall_mm)
        return _attach_verify_spec(
            res,
            spec_dict,
            settings,
            rt,
            measured_features=measured,
            feature_findings=findings,
        )

    # 1-5. baseline: clean -> decimate -> orient -> scale -> flatten -> rest
    try:
        raw = load_triangle_mesh(mesh_path)
    except Exception as e:
        res.error = f"failed to load generated mesh: {e}"
        return _done()
    if skip_baseline:
        mesh = clean(raw)
        res.mesh["ops_only"] = True
        res.mesh["input_triangles"] = len(raw.faces)
    else:
        # cap at mesh_decimate_faces like the workbench/candidates paths do —
        # uncapped generator output (observed 283k faces) OOMs downstream
        # watertight repair and boolean ops inside the executor cgroup
        mesh, base_info = baseline_mesh(raw, post, spec, cad_cap_faces=rt.mesh_decimate_faces)
        res.mesh.update(base_info)

    # 5b. agent transform ops (validated MeshOp list inside post.params)
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return _done()
    if post.ops:
        if post.ops_version and post.ops_version != MESH_OPS_VERSION:
            log.warning(
                "mesh.ops_version_mismatch",
                recorded=post.ops_version,
                current=MESH_OPS_VERSION,
            )
            res.mesh["ops_version_warning"] = (
                f"ops recorded with version {post.ops_version}, replayed with "
                f"{MESH_OPS_VERSION} — semantics may differ"
            )
        t = time.monotonic()
        try:
            mesh, op_report = apply_ops(mesh, post.ops, anchors=anchors, runtime=rt)
        except Exception as e:
            res.error = f"mesh op failed: {e}"
            return _done()
        res.mesh["ops_report"] = op_report
        res.timings["mesh_ops_s"] = round(time.monotonic() - t, 2)
        mesh = rest_on_xy_center(mesh)

    # ops re-inflate the mesh (remesh caps, voxel rebuilds — observed 318k
    # faces after a flatten on a 60k-capped baseline) and can break
    # watertightness. Repair + re-cap + a light top-up fix so exports stay
    # slicer-usable; failures are non-fatal — downstream analysis reports it.
    cap = rt.mesh_decimate_faces
    if cap and len(mesh.faces) > cap:
        mesh = decimate(mesh, cap)
    if not mesh.is_watertight:
        with contextlib.suppress(Exception):
            repaired = _ensure_watertight(mesh)
            # a repair that shatters the part into more shells than it
            # started with is worse than the coherent open input — the
            # critique reads shell count as a 'not a single solid' major
            if len(repaired.split(only_watertight=False)) <= len(mesh.split(only_watertight=False)):
                mesh = repaired
    if cap and len(mesh.faces) > cap:
        # _ensure_watertight's voxel fallback can itself out-face the cap
        mesh = decimate(mesh, cap)
        if not mesh.is_watertight:
            # decimation can nick watertightness — cheap top-up only (no
            # voxel rebuild, which would blow the cap a second time)
            try:
                import trimesh

                mesh.merge_vertices()
                mesh.fix_normals()
                trimesh.repair.fill_holes(mesh)
            except Exception:
                pass
    mesh = rest_on_xy_center(mesh)

    res.mesh["post"] = post.model_dump()
    res.mesh["triangles"] = len(mesh.faces)

    # 6. STL into _run (drives renders, analysis, exports)
    stl_path = workdir / "model.stl"
    try:
        mesh.export(str(stl_path), file_type="stl")
    except Exception as e:
        res.error = f"failed to write STL: {e}"
        return _done()

    # 7. analysis
    res.analysis = mesh_analysis(mesh)
    _repair_analysis_mesh(res.analysis, stl_path)

    # 7b. deterministic feature detection (holes, dimples, XY outline) —
    # feeds worker-side anchor reconciliation with the measured top view.
    # Runs for every mesh result, not just hybrid solids: pure mesh mode
    # needs the same deterministic evidence (a 22 mm X-axis bore must be
    # verifiable in eval instead of relying on LLM render judgement)
    try:
        from cadx.cad.mesh_features import detect_mesh_features

        res.mesh["features"] = detect_mesh_features(mesh).model_dump()
    except Exception as e:
        log.warning("mesh_features.failed", error=str(e)[:200])

    # 7c. silhouette IoU vs the reference image — the deterministic shape
    # signal for critique calibration when no plan-view measurement exists.
    # The worker extracts the subject mask once (ref_mask_rel is a PNG in
    # DATA_DIR); the comparison runs on this authoritative processed mesh.
    if ref_mask_rel:
        try:
            from cadx import storage as _storage
            from cadx.cad.silhouette import load_mask, silhouette_iou

            ref_mask = load_mask(_storage.resolve(ref_mask_rel))
            if ref_mask is not None:
                sil, view = silhouette_iou(mesh, ref_mask)
                res.mesh["silhouette_iou"] = sil
                res.mesh["silhouette_view"] = view
        except Exception as e:
            log.warning("mesh.silhouette_iou_failed", error=str(e)[:200])

    # 8. optional solid conversion (hybrid bridge + tessellated STEP)
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return _done()
    if make_solid:
        t = time.monotonic()
        try:
            # decimates a copy when over the cap — `mesh` keeps full detail
            solid = mesh_to_solid(mesh, max_faces=solid_max_faces or rt.mesh_max_faces_for_cad)
            from build123d import export_brep

            brep = workdir / "model.brep"
            export_brep(solid, str(brep))
            res.mesh["solid_ok"] = True
            res.mesh["solid_faces"] = len(mesh.faces)
        except Exception as e:
            res.mesh["solid_ok"] = False
            res.mesh["solid_error"] = str(e)[:500]
            log.warning("mesh.solid_failed", error=str(e)[:300])
        res.timings["mesh_to_solid_s"] = round(time.monotonic() - t, 2)

    # 9. renders + contact sheet + thumbnail
    t = time.monotonic()
    render_dir = out_dir / "renders"
    render_paths = render_views(stl_path, render_dir, res.analysis.bbox_size)
    res.renders = {v: p.name for v, p in render_paths.items()}
    thumb = render_thumbnail(stl_path, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
    stats = [
        f"bbox: {res.analysis.bbox_size[0]:.1f} x {res.analysis.bbox_size[1]:.1f} x {res.analysis.bbox_size[2]:.1f} mm",
        f"volume: {res.analysis.volume:.0f} mm^3",
        f"triangles: {res.analysis.triangle_count}",
        f"watertight: {res.analysis.watertight_after_repair}",
    ]
    sheet = make_contact_sheet(render_paths, out_dir / "contact_sheet.png", stats)
    res.contact_sheet = sheet.name if sheet else None
    res.timings["render_s"] = round(time.monotonic() - t, 2)

    # 10. DfM (mesh-based checks only — no analytic B-Rep)
    t = time.monotonic()
    warnings: list[DfmWarning] = common_checks(res.analysis, mesh, settings.build_volume)
    if res.analysis.triangle_count > 500_000:
        warnings.append(
            DfmWarning(
                code="high_triangle_count",
                severity="low",
                message=f"mesh has {res.analysis.triangle_count} triangles — consider decimating",
                value=float(res.analysis.triangle_count),
                threshold=500_000.0,
            )
        )
    base_ratio = flat_base_ratio(mesh)
    if base_ratio < 0.02:
        warnings.append(
            DfmWarning(
                code="no_flat_base",
                severity="medium",
                message="almost no flat contact surface with the build plate — "
                "use brim/raft or flatten_bottom",
                value=round(base_ratio * 100, 1),
                threshold=2.0,
            )
        )
    if settings.process in ("fdm", "sla"):
        warnings += fdm_mesh_checks(
            mesh,
            res.analysis,
            min_wall_mm=settings.min_wall_mm,
            nozzle_mm=settings.nozzle_mm,
        )
    if settings.process in ("cnc3", "cnc2d"):
        warnings += cnc_mesh_checks(mesh, res.analysis)
    res.dfm = DfmReport(process=settings.process, warnings=warnings).sort()
    res.timings["dfm_s"] = round(time.monotonic() - t, 2)

    # 11. exports
    t = time.monotonic()
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return _done()
    if full_exports:
        # STEP comes from the same solid built above when available
        res.exports = export_mesh_all(mesh, out_dir / "exports", name=name, try_step=False)
        brep = workdir / "model.brep"
        if brep.exists():
            try:
                shape = import_brep(str(brep))
                from build123d import export_step

                export_step(shape, str(out_dir / "exports" / f"{name}.step"))
                res.exports["step"] = f"{name}.step"
                res.mesh["step_tessellated"] = True
            except Exception:
                res.exports["step"] = None
        else:
            try:
                res.exports["step"] = export_mesh_all(
                    mesh, out_dir / "exports", name=name, try_step=True
                )["step"]
                if res.exports["step"]:
                    res.mesh["step_tessellated"] = True
            except Exception:
                res.exports["step"] = None
        res.exports["dxf"] = None
        res.exports["svg"] = None
        if res.analysis.watertight is False:
            _repair_export_meshes(res.exports, out_dir / "exports")
    else:
        res.exports = {"stl": None}
    res.timings["export_s"] = round(time.monotonic() - t, 2)

    res.ok = True
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    # canonical conformance report — reads only evidence already in res
    return _done()


def run_mesh_candidates(*, mesh_path: Path, out_dir: Path, size: int = 512) -> dict[str, Any]:
    """Render the 6 up-axis orientation candidates into a labelled contact sheet.

    Returns {ok, candidates: {up: relpath}, sheet: relpath|None, stats: {...}}.
    Used once per version by the ORIENT step (vision LLM picks up/front).
    """
    from cadx.cad.render import render_available

    out: dict[str, Any] = {"ok": False, "candidates": {}, "sheet": None, "stats": {}}
    try:
        raw = load_triangle_mesh(mesh_path)
    except Exception as e:
        out["error"] = f"failed to load mesh: {e}"
        return out
    mesh = clean(raw)
    mesh = decimate(mesh, 60_000)
    out["stats"] = {
        "triangles": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "extents": [round(float(v), 3) for v in mesh.extents],
    }
    if not render_available():
        out["error"] = "offscreen rendering unavailable"
        return out

    import tempfile

    import pyvista as pv
    from PIL import Image, ImageDraw

    cand_dir = out_dir / "orient"
    cand_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[tuple[str, Path]] = []
    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "cand.stl"
        for up, m in candidate_meshes(mesh).items():
            m.export(str(stl), file_type="stl")
            pvm = pv.read(str(stl))
            pl: Any = pv.Plotter(off_screen=True, window_size=[size, size])
            pl.set_background("white")
            pl.add_mesh(pvm, color="#9aa3ad", smooth_shading=True, split_sharp_edges=True)
            fedges = pvm.extract_feature_edges(30)
            if fedges.n_points:
                pl.add_mesh(fedges, color="#222222", line_width=1.0)
            pl.add_mesh(pvm.outline(), color="#888888", line_width=2)
            pl.camera_position = "iso"
            pl.reset_camera()
            pl.camera.zoom(1.15)
            pl.add_axes(color="black")
            png = cand_dir / f"up_{up.replace('+', 'p').replace('-', 'm')}.png"
            pl.screenshot(str(png))
            pl.close()
            tiles.append((up, png))
            out["candidates"][up] = png.name

    # labelled contact sheet: 3x2
    cols, rows = 3, 2
    label_h = 40
    sheet = Image.new("RGB", (cols * size, rows * (size + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (up, png) in enumerate(tiles):
        x, y = (i % cols) * size, (i // cols) * (size + label_h)
        sheet.paste(Image.open(png), (x, y + label_h))
        draw.text((x + 10, y + 12), f"option {i + 1}: up = {up}", fill="black")
    sheet_path = out_dir / "orient_sheet.png"
    sheet.save(sheet_path)
    out["sheet"] = sheet_path.name
    out["ok"] = True
    return out


def run_mesh_preview(
    *, mesh_path: Path, out_dir: Path, views: tuple[str, ...] = ("iso", "front", "top")
) -> dict[str, Any]:
    """Cheap render-only pass for the mesh-refine session's preview tool.

    Renders the mesh as stored (no baseline/ops — the session writes the
    current working state itself). Returns {ok, images: {view: relname}, stats}.
    """
    from cadx.cad.render import render_available

    out: dict[str, Any] = {"ok": False, "images": {}, "stats": {}}
    try:
        mesh = load_triangle_mesh(mesh_path)
    except Exception as e:
        out["error"] = f"failed to load mesh: {e}"
        return out
    out["stats"] = {
        "triangles": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "extents": [round(float(v), 3) for v in mesh.extents],
    }
    if not render_available():
        out["error"] = "offscreen rendering unavailable"
        return out

    import tempfile

    prev_dir = out_dir / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as td:
            stl = Path(td) / "preview.stl"
            mesh.export(str(stl), file_type="stl")
            render_paths = render_views(stl, prev_dir, [float(v) for v in mesh.extents], size=512)
        out["images"] = {v: p.name for v, p in render_paths.items() if v in views}
        out["ok"] = bool(out["images"])
    except Exception as e:
        out["error"] = f"preview render failed: {e}"
    return out


# --------------------------------------------------------------------------- #
# FEM / topology optimization                                                  #
# --------------------------------------------------------------------------- #


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


# --- spec-envelope input normalization (topopt) ------------------------------
#
# The declared spec dims are the hard part envelope, but the FEM/TopOpt design
# domain is simply the input mesh's bbox — an oversize base (a proud rib, a
# scale drift, 94.9 mm input for a 90 mm spec) silently shifts the whole FEM
# problem out of envelope. Clip the input back to the envelope before
# voxelizing. Normalization fires only past this slack — sub-tolerance
# overshoot is voxel-resolution noise, not a geometry fault.
_ENVELOPE_CLIP_SLACK_MM = 2.0
_ENVELOPE_CLIP_SLACK_FRAC = 0.05


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


def _auto_warm_density(out_dir: Path, domain: Any) -> tuple[np.ndarray | None, str | None]:
    """Opportunistic SIMP warm start: reuse a previous optimize run's
    ``density.npz`` when its voxel grid is *identical* (same origin, pitch
    and dims) to the current domain — a size-only match on a shifted or
    differently-pitched domain would misalign the field.

    Candidates: the previous iteration dir of this version (agent optimize
    sessions chain SIMP runs on the same base mesh, so run N benefits from
    run N-1's converged field), then the version-level ``topopt/density.npz``
    copied up from an earlier optimize on the same version."""
    cands: list[Path] = []
    if out_dir.name.isdigit() and int(out_dir.name) > 1:
        cands.append(out_dir.parent / str(int(out_dir.name) - 1) / "topopt" / "density.npz")
    if len(out_dir.parents) > 1:
        cands.append(out_dir.parents[1] / "topopt" / "density.npz")
    for p in cands:
        if not p.exists():
            continue
        try:
            with np.load(p) as npz:
                dims = tuple(int(v) for v in npz["dims"])
                same_grid = (
                    dims == tuple(domain.dims)
                    and np.allclose(np.asarray(npz["origin"], dtype=float), domain.origin)
                    and abs(float(npz["pitch"]) - float(domain.pitch)) < 1e-9
                )
                if same_grid:
                    return np.asarray(npz["density"], dtype=float), str(p)
        except Exception:
            continue
    return None, None


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
    from cadx.cad.fem.loads import expand_cone
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
        cone_idx = [
            li
            for li, ld in enumerate(c.loads)
            if (ld.direction_cone_deg or 0) > 0 and ld.force_n and len(ld.force_n) == 3
        ]
        _add(c, float(c.weight))
        for li in cone_idx:
            ld = c.loads[li]
            variants = expand_cone(
                np.asarray(ld.force_n, dtype=float), float(ld.direction_cone_deg or 0.0)
            )
            for vi, fvar in enumerate(variants):
                if len(r_resolved) >= max_cases:
                    if warn:
                        warn_list.append(f"direction-cone expansion capped at {max_cases} cases")
                    break
                vc = c.model_copy(deep=True)
                vc.name = f"{c.name}_cone{vi + 1}"
                vc.loads[li].force_n = [float(v) for v in fvar]
                if vc.loads[li].bearing_dir:
                    vc.loads[li].bearing_dir = [float(v) for v in fvar]
                vc.loads[li].direction_cone_deg = None
                _add(vc, float(c.weight))
        for load in c.loads:
            # don't pin body-load or whole-part regions — gravity acts on
            # whatever material survives and `all` re-matches the
            # extracted mesh; pinning them would freeze the domain.
            concentrated = bool(
                load.force_n
                or load.pressure_mpa
                or load.moment_nm
                or load.preload_n
                or load.remote_point
            )
            if load.g_load and not concentrated:
                continue
            if getattr(load.region, "kind", "") == "all":
                continue
            r_load_nm.append(region_node_mask(domain, load.region, anchors))
    for f3, _ in r_resolved:
        r_fixed_nm.append(np.asarray(f3).any(axis=1) if np.ndim(f3) == 2 else f3)
    return r_resolved, r_weights, r_fixed_nm, r_load_nm, r_unresolved


def _verify_export_structure(
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
         sf_verified, bottleneck_mm2, sf_floor, load_path_ok,
         sf_mesh_sensitive, warnings}
    """
    from scipy import ndimage

    from cadx.cad.fem.analyze import run_fem_domain
    from cadx.cad.fem.loadpath_cert import loadpath_certificate
    from cadx.cad.fem.regions import resolve_case_dofs
    from cadx.cad.fem.voxelize import voxelize

    warnings: list[str] = []
    mem_budget = float(resources.effective_exec_mem_mb(s))
    skin = candidate_mesh if source_mesh is None else source_mesh
    use_coverage = bool(getattr(s, "fem_verify_coverage_e", False))
    export_domain = voxelize(
        candidate_mesh, design_pitch, max_elements, coverage=use_coverage
    )
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

    def _solve(
        mask: np.ndarray,
        sfx: str,
        od: Path | None = out_dir,
        dom: Any = None,
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
            spsolve_dofs=s.fem_spsolve_dofs,
            pool_workers=s.exec_pool_workers,
            mem_budget_mb=mem_budget,
            suffix=sfx,
            source_mesh=skin,
            cancelled=cancelled,
            solver_backend=solver_backend,
        )

    report = _solve(export_domain.solid, suffix)
    sf_realizations: dict[str, float | None] = {
        "nominal": report.worst_safety_factor
    }
    if verify_robust:
        n_shift = int(getattr(s, "fem_verify_realization_cells", 1) or 1)
        n_shift = max(1, min(n_shift, 4))
        for name, mask in (
            ("eroded", ndimage.binary_erosion(export_domain.solid, iterations=n_shift)),
            ("dilated", ndimage.binary_dilation(export_domain.solid, iterations=n_shift)),
        ):
            sfx = f"{suffix}_ero" if name == "eroded" else f"{suffix}_dil"
            if not np.asarray(mask).any():
                sf_realizations[name] = None
                warnings.append(f"export realization '{name}' is empty — skipped")
                continue
            # coverage measures the NOMINAL mesh's fill fractions — on the
            # dilated mask the newly-solid cells model over-extrusion, so
            # they print FULL (coverage 1.0), not the ~0 fraction their
            # nominal-boundary position suggests
            dom = None
            if (
                name == "dilated"
                and use_coverage
                and export_domain.coverage is not None
            ):
                cov_d = np.asarray(export_domain.coverage, dtype=float).copy()
                cov_d[(np.asarray(mask) & ~export_domain.solid).ravel()] = 1.0
                dom = dataclasses.replace(export_domain, coverage=cov_d)
            try:
                rep = _solve(mask, sfx, None, dom)
                sf_realizations[name] = rep.worst_safety_factor
            except Exception as e:
                sf_realizations[name] = None
                warnings.append(f"export realization '{name}' solve failed: {str(e)[:80]}")
            if sf_realizations[name] is None:
                warnings.append(
                    f"export realization '{name}' produced no safety factor"
                )
    # sf_verified only exists when the NOMINAL realization solved — the
    # headline structure itself being unsolvable IS the verdict; a
    # dilated-only success must never read as "verified". When nominal is
    # None the gate falls back to the (equally None) report SF and the
    # structure is flagged for review.
    solvable = [v for v in sf_realizations.values() if v is not None]
    sf_verified = min(solvable) if sf_realizations["nominal"] is not None else None

    sf_mesh_sensitive: bool | None = None
    if getattr(s, "fem_verify_mesh_shift", False):
        try:
            shifted = candidate_mesh.copy()
            shifted.apply_translation(np.full(3, design_pitch / 2.0))
            dom_sh = voxelize(
                shifted, design_pitch, max_elements, coverage=use_coverage
            )
            rep_sh = run_fem_domain(
                dom_sh,
                dom_sh.solid,
                cases,
                mat_name,
                material=mat,
                out_dir=None,
                anchors=anchors,
                keep_regions=keep_list,
                spsolve_dofs=s.fem_spsolve_dofs,
                pool_workers=s.exec_pool_workers,
                mem_budget_mb=mem_budget,
                suffix="",
                source_mesh=None,
                cancelled=cancelled,
                solver_backend=solver_backend,
            )
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

    bottleneck_mm2: float | None = None
    sf_floor: float | None = None
    load_path_ok: bool | None = None
    if getattr(s, "fem_loadpath_cert", False):
        try:
            cases_res = [
                resolve_case_dofs(
                    export_domain,
                    c,
                    anchors,
                    density_g_cm3=getattr(mat, "density_g_cm3", 1.24),
                )[:2]
                for c in cases
            ]
            cert = loadpath_certificate(
                export_domain,
                cases_res,
                yield_mpa=float(getattr(mat, "yield_mpa", 0.0) or 0.0),
            )
            bottleneck_mm2 = cert.get("bottleneck_mm2")
            sf_floor = cert.get("sf_floor")
            load_path_ok = cert.get("load_path_ok")
            warnings.extend(cert.get("warnings") or [])
        except Exception as e:
            warnings.append(f"load-path certificate failed: {str(e)[:80]}")

    return {
        "report": report,
        "n_bodies_export": n_bodies_export,
        "n_bodies_voxel": n_bodies_voxel,
        "sf_realizations": sf_realizations,
        "sf_verified": sf_verified,
        "bottleneck_mm2": bottleneck_mm2,
        "sf_floor": sf_floor,
        "load_path_ok": load_path_ok,
        "sf_mesh_sensitive": sf_mesh_sensitive,
        "warnings": warnings,
    }


def _gate_sf(info: dict[str, Any] | None, report: Any) -> float | None:
    """Effective export safety factor for gating: the verified worst-case
    realization when the hardened export sweep produced one, else the
    report's own worst safety factor (which is the mask-substituted
    fallback report when the export solve crashed). None-checked, not
    or-chained — a solved SF of 0.0 IS a verdict."""
    sf = (info or {}).get("sf_verified")
    return sf if sf is not None else report.worst_safety_factor


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
    return {"ok": True, "fem": report.model_dump()}


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
    from cadx.cad.fem.extract import density_to_mesh
    from cadx.cad.fem.loadpath import loadpath_field
    from cadx.cad.fem.preview import render_regions
    from cadx.cad.fem.regions import (
        auto_keep_regions,
        empty_region_warnings,
        keep_element_mask,
        material_cell_factors,
    )
    from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
    from cadx.cad.fem.voxelize import dilate_solid, grid_bounds_mm, snapped_pitch, voxelize
    from cadx.llm.schemas import LoadCase, OptimizeSpec

    res = ExecResult()
    s = runtime or RuntimeSettings()
    opt = OptimizeSpec(**(optimize or {}))
    t0 = time.monotonic()

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
                mesh.export(str(norm_stl), file_type="stl")
                res.topopt["normalized_input_relpath"] = storage.relpath(norm_stl)
            except Exception as e:
                log.warning("topopt.normalized_export_failed", error=str(e)[:200])
            log.info(
                "topopt.input_normalized",
                method=norm_rep.get("method"),
                input_bbox=norm_rep["input_bbox"],
                spec_bbox=norm_rep["spec_bbox"],
            )

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
    res.topopt["domain_pitch"] = {
        "requested_mm": round(req_voxel, 4),
        "design_mm": round(design_pitch, 4),
        "snapped": abs(design_pitch - req_voxel) > 1e-6,
    }
    # grow_mm: pad the grid so the optimizer can also ADD material beyond the
    # part silhouette — the pad ring stays void in `domain.solid` and becomes
    # designable space only in `design_domain` below
    grow_mm = float(getattr(opt, "grow_mm", 0.0) or 0.0)

    def _pad_cells(pitch: float) -> int:
        return math.ceil(grow_mm / max(float(pitch), 1e-6)) if grow_mm > 0 else 0

    try:
        domain = voxelize(
            mesh,
            design_pitch,
            max_elements or s.fem_max_elements,
            pad_cells=_pad_cells(design_pitch),
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
    while 3 * domain.node_count > dofs_cap and coarsen_steps < 4:
        coarsen_steps += 1
        try:
            domain = voxelize(
                mesh,
                domain.pitch * 1.4,
                max_elements or s.fem_max_elements,
                pad_cells=_pad_cells(domain.pitch * 1.4),
            )
        except Exception:
            break
    if coarsen_steps:
        res.topopt["auto_coarsen"] = {
            "requested_pitch_mm": round(req_voxel, 4),
            "design_pitch_mm": round(float(domain.pitch), 4),
            "dofs": int(3 * domain.node_count),
        }
        log.info(
            "topopt.auto_coarsen",
            pitch=round(float(domain.pitch), 4),
            dofs=int(3 * domain.node_count),
        )
    if 3 * domain.node_count > dofs_cap:
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

    # predictive memory budget — an OOM kill mid-SIMP loses the whole job;
    # better to refuse early with a pointer to the knobs that fix it
    from cadx.cad.fem.voxelize import estimate_fem_memory_mb

    mem_budget = float(resources.effective_exec_mem_mb(s))
    if mem_budget:
        n_eff = min(len(cases) * 2, s.fem_max_effective_cases)
        est_mb = estimate_fem_memory_mb(
            domain, n_eff, direct=3 * domain.node_count <= s.fem_spsolve_dofs
        )
        if est_mb > mem_budget * 0.75:
            res.error = (
                f"predicted FEM memory ~{est_mb:.0f} MB exceeds the executor "
                f"budget ({mem_budget:.0f} MB) — raise fem_voxel_mm or "
                "exec_mem_limit_mb"
            )
            return res

    anchors = _scan_anchors(mesh, anchors, enabled=scan_anchors)
    res.topopt["anchors"] = anchors

    # --- resolve load cases (with direction-cone expansion for robustness) ---
    max_cases = s.fem_max_effective_cases

    def _resolve_cases_on(dom: Any, warn: bool = True):
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

    resolved, case_weights, fixed_node_masks, load_node_masks, unresolved = _resolve_cases_on(
        domain
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
    keep_list = list(opt.keep_regions)
    # the runtime `topopt_auto_keep` flag (default on) is the opt-out:
    # keep_mode="manual" still gets anchor keeps on top of declared regions
    # unless the flag disables them; "auto" requests them explicitly
    n_auto_keeps = 0
    if anchors and (opt.keep_mode == "auto" or (opt.keep_mode == "manual" and s.topopt_auto_keep)):
        margin = opt.auto_keep_margin_mm or s.topopt_auto_keep_margin_mm
        auto_keeps = auto_keep_regions(anchors, margin)
        keep_list += auto_keeps
        n_auto_keeps = len(auto_keeps)
    res.topopt["auto_keep_applied"] = bool(n_auto_keeps)
    res.topopt.setdefault("warnings", []).extend(
        empty_region_warnings(part_domain, keep_list, anchors)
    )
    res.topopt.setdefault("warnings", []).extend(
        empty_region_warnings(part_domain, list(opt.void_regions), anchors)
    )
    keep_cells = keep_element_mask(part_domain, keep_list, anchors)
    void_cells = keep_element_mask(part_domain, list(opt.void_regions), anchors)
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
    if passive_frac > volfrac:
        res.error = (
            f"passive keep/load/fixed cells cover {passive_frac:.0%} of the optimization domain — "
            f"above the requested volume fraction {volfrac:.0%}; the target is "
            "unreachable regardless of optimization. Reduce keep/auto-keep "
            "regions or set target_volume_fraction >= "
            f"{min(0.95, passive_frac + 0.02):.2f}"
        )
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

    min_member = float(opt.min_member_mm or s.topopt_min_member_mm or 0.0)
    min_gap = float(opt.min_gap_mm or min_member or 0.0)
    params = TopOptParams(
        volfrac=opt.target_volume_fraction or s.topopt_default_volfrac,
        # the filter radius also acts as the printable-feature floor
        rmin_mm=max(s.topopt_default_rmin_mm, min_member / 2),
        max_iter=opt.max_iterations or s.topopt_max_iterations,
        threshold=float(opt.density_threshold or 0.5),
        filter=str(getattr(opt, "filter", "cone") or "cone"),
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
    )
    # hardened export verification: eroded/nominal/dilated realization
    # sweep whose worst case is reported as sf_verified (see
    # _verify_export_structure); the per-run override follows the runtime
    # setting when unset
    eff_robust = (
        opt.verify_robust
        if opt.verify_robust is not None
        else bool(s.fem_verify_realizations)
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
            warm = np.load(storage.resolve(opt.warm_density_relpath))
            x0 = np.asarray(warm["density"], dtype=float)
            if x0.size != domain.cell_count:
                res.topopt["warnings"].append("warm-start density has a different grid — ignored")
                x0 = None
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
                pad_cells=_pad_cells(domain.pitch * 2.0),
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
                res_c, w_c, fix_c, load_nm_c, _unres_c = _resolve_cases_on(coarse, warn=False)
                if not res_c:
                    res.topopt["warnings"].append(
                        "multires: no solvable case on the coarse grid — skipped"
                    )
                else:
                    keep_c = keep_element_mask(coarse, keep_list, anchors)
                    void_c = keep_element_mask(coarse, list(opt.void_regions), anchors)
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
                        spsolve_dofs=s.fem_spsolve_dofs,
                        cancelled=cancelled,
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

    def _one_pass(
        volfrac: float,
        pass_idx: int = 1,
        warm: np.ndarray | None = None,
        member_floor_mm: float | None = None,
        crisp_extraction: bool = False,
        vol_target_scale: float = 1.0,
    ) -> dict[str, Any]:
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
            spsolve_dofs=s.fem_spsolve_dofs,
            cancelled=cancelled,
            progress_file=opt_dir / f"history_{volfrac:.2f}.json",
        )
        out["simp_s"] = round(time.monotonic() - t, 2)
        out["result"] = result
        out["params"] = p

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
        )
        out["n_components"] = n_components
        out["extract"] = extract_info
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
            )
        opt_mesh, output_envelope, output_envelope_ok, envelope_warnings = (
            _clip_export_to_spec_envelope(
                opt_mesh, list(dims_spec) if dims_spec else None, label="smoothed export"
            )
        )
        extract_warnings.extend(envelope_warnings)
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
            spsolve_dofs=s.fem_spsolve_dofs,
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
                solver_backend=params.solver_backend,
                cancelled=cancelled,
                verify_robust=eff_robust,
            )
            res.topopt.setdefault("warnings", []).extend(info["warnings"])
            out["n_bodies_export"] = info["n_bodies_export"]
            verify_exp = info["report"]
        except Exception as e:
            info = None
            log.warning("topopt.verify_export.failed", error=str(e)[:200])
            # falling back to the MASK report makes sf_export indistinguishable
            # from a real export solve — record it so downstream consumers and
            # the agent see the substitution instead of trusting the number
            verify_exp = verify
            res.topopt.setdefault("warnings", []).append(
                f"export FEM solve failed ({str(e)[:80]}) — sf_export "
                "duplicates the voxel-mask result, not the printed mesh"
            )
            out["export_solve_ok"] = False
        out["verify_export"] = verify_exp
        out["export_verify_info"] = info
        return out

    p1 = _one_pass(params.volfrac, pass_idx=1)
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
    safety_retry = exp_degenerate or (sf_gate1 is not None and sf_gate1 < opt.safety_factor_min)
    gap_retry = sf_ratio is not None and sf_ratio > 3.0
    connectivity_retry = p1.get("n_bodies_export") is None or int(p1["n_bodies_export"]) > 1
    # Stated spec envelopes intentionally allow a shrunken export; avoid
    # replacing that geometry with a crisp full-boundary retry. A real
    # volume *deficit* on a spec'd part instead retries at a raised
    # extraction volume target — the iso moves outward, the boundary does
    # not get filled.
    p1_vol_err_pp = float(p1["extract"].get("vol_error_pp") or 0.0)
    volume_retry = p1_vol_err_pp < -5.0 if dims_spec else abs(p1_vol_err_pp) > 5.0
    retry_needed = safety_retry or gap_retry or connectivity_retry or volume_retry
    if retry_needed and (cancelled is None or not cancelled()):
        retry_volfrac = params.volfrac + 0.1 if safety_retry else params.volfrac
        retry_member = (
            max(min_member, 2.0 * design_domain.pitch)
            if gap_retry or connectivity_retry
            else min_member
        )
        retry_crisp = not dims_spec and (gap_retry or connectivity_retry or volume_retry)
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
            )
            sf_exp2 = p2["verify_export"].worst_safety_factor
            sf_gate2 = _gate_sf(p2.get("export_verify_info"), p2["verify_export"])
            trigger_reasons = [
                reason
                for enabled, reason in (
                    (safety_retry, "safety"),
                    (gap_retry, "sf_gap"),
                    (connectivity_retry, "connectivity"),
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
        "bottleneck_mm2": exp_info.get("bottleneck_mm2"),
        "sf_floor": exp_info.get("sf_floor"),
        "load_path_ok": exp_info.get("load_path_ok"),
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
    if exp_info.get("load_path_ok") is False:
        # the geometric certificate found a case whose loads cannot reach
        # their fixtures through the export solid — structurally broken
        verification["valid"] = False
        res.topopt["warnings"].append(
            "load-path certificate: a load case cannot reach its fixtures "
            "through the export solid — the part is structurally disconnected"
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
            "bottleneck_mm2": verification["bottleneck_mm2"],
            "sf_floor": verification["sf_floor"],
            "load_path_ok": verification["load_path_ok"],
            "sf_mesh_sensitive": verification["sf_mesh_sensitive"],
            "min_member_mm": best["min_member_mm"],
            "thin_fraction": round(best["thin_fraction"], 4),
            "n_components": best["n_components"],
            "n_bodies_export": best["n_bodies_export"],
            "n_bodies_voxel": exp_info.get("n_bodies_voxel"),
            "n_surface_components": best["n_surface_components"],
            "output_envelope": best.get("output_envelope"),
            "output_envelope_ok": best.get("output_envelope_ok", True),
            "extract": best["extract"],
            "verification": verification,
        }
    )

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
    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    try:
        opt_mesh.export(str(stl_path), file_type="stl")
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
    _repair_analysis_mesh(res.analysis, stl_path)
    if res.analysis.watertight is False:
        _repair_export_meshes(res.exports, out_dir / "exports")
    res.mesh = {
        "pipeline": "topopt",
        "triangles": len(opt_mesh.faces),
        "volume_mm3": round(float(opt_mesh.volume), 1),
        "solid_elements": int(final_mask.sum()),
        "pitch_mm": domain.pitch,
    }

    # renders
    t = time.monotonic()
    render_dir = out_dir / "renders"
    render_paths = render_views(stl_path, render_dir, res.analysis.bbox_size)
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
    from cadx.cad.fem.extract import density_to_mesh
    from cadx.cad.fem.infill import (
        InfillParams,
        element_strain_energy,
        infill_field,
    )
    from cadx.cad.fem.materials import resolve_material
    from cadx.cad.fem.preview import render_regions
    from cadx.cad.fem.regions import (
        auto_keep_regions,
        empty_region_warnings,
        keep_element_mask,
        material_cell_factors,
    )
    from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
    from cadx.cad.fem.voxelize import grid_bounds_mm, snapped_pitch, voxelize
    from cadx.llm.schemas import LoadCase, OptimizeSpec

    res = ExecResult()
    s = runtime or RuntimeSettings()
    opt = OptimizeSpec(**(optimize or {}))
    t0 = time.monotonic()

    # per-run overrides follow the runtime settings when unset — the same
    # tri-state pattern the other optimize flags use
    eff_octet = (
        opt.infill_octet if opt.infill_octet is not None else bool(s.infill_octet)
    )
    eff_robust = (
        opt.verify_robust
        if opt.verify_robust is not None
        else bool(s.fem_verify_realizations)
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
    res.topopt["domain_pitch"] = {
        "requested_mm": round(req_voxel, 4),
        "design_mm": round(design_pitch, 4),
        "snapped": abs(design_pitch - req_voxel) > 1e-6,
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

    from cadx.cad.fem.voxelize import estimate_fem_memory_mb

    mem_budget = float(resources.effective_exec_mem_mb(s))
    if mem_budget:
        n_eff = min(len(cases) * 2, s.fem_max_effective_cases)
        est_mb = estimate_fem_memory_mb(
            domain, n_eff, direct=3 * domain.node_count <= s.fem_spsolve_dofs
        )
        if est_mb > mem_budget * 0.75:
            res.error = (
                f"predicted FEM memory ~{est_mb:.0f} MB exceeds the executor "
                f"budget ({mem_budget:.0f} MB) — raise fem_voxel_mm or "
                "exec_mem_limit_mb"
            )
            return res

    anchors = _scan_anchors(mesh, anchors, enabled=True)
    res.topopt["anchors"] = anchors

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
    if anchors and (opt.keep_mode == "auto" or (opt.keep_mode == "manual" and s.topopt_auto_keep)):
        margin = opt.auto_keep_margin_mm or s.topopt_auto_keep_margin_mm
        keep_list += auto_keep_regions(anchors, margin)
    res.topopt.setdefault("warnings", []).extend(empty_region_warnings(domain, keep_list, anchors))
    res.topopt.setdefault("warnings", []).extend(
        empty_region_warnings(domain, list(opt.void_regions), anchors)
    )
    keep_cells = keep_element_mask(domain, keep_list, anchors)
    void_cells = keep_element_mask(domain, list(opt.void_regions), anchors)
    passive = passive_solid_mask(domain, [keep_cells], fixed_node_masks, load_node_masks)

    res.topopt["regions_png"] = render_regions(
        domain, cases, keep_list, out_dir / "fem", anchors, source_mesh=mesh
    )

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
        spsolve_dofs=s.fem_spsolve_dofs,
        pool_workers=s.exec_pool_workers,
        mem_budget_mb=mem_budget,
        suffix="_baseline",
        source_mesh=mesh,
        cancelled=cancelled,
        solver_backend=solver_backend,
        fields_out=fo,
    )
    res.timings["baseline_fem_s"] = round(time.monotonic() - t, 2)
    report_to_json(verify_base, out_dir / "fem" / "report_baseline.json")

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

    # --- stage 2 (optional): guided scope — a cheap coarse SIMP picks the
    # retained topology and the lattice only fills what it keeps
    guide_mask: np.ndarray | None = None
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
                pc = TopOptParams(
                    volfrac=opt.target_volume_fraction or s.topopt_default_volfrac,
                    rmin_mm=max(s.topopt_default_rmin_mm, min_member / 2),
                    max_iter=max(10, (opt.max_iterations or s.topopt_max_iterations) // 3),
                    filter="cone",
                    solver_backend=solver_backend,
                    aniso=(mat.aniso_z, float(mat.aniso_axis)) if mat.aniso_z != 1.0 else None,
                )
                guide_r = simp(
                    coarse,
                    E0_c,
                    mat.poisson,
                    res_c,
                    passive_c,
                    pc,
                    passive_void=void_c,
                    case_weights=np.asarray(w_c),
                    spsolve_dofs=s.fem_spsolve_dofs,
                    cancelled=cancelled,
                )
                guide_mask = (
                    _upsample_density(guide_r.density.reshape(coarse.dims), coarse, domain).reshape(
                        domain.dims
                    )
                    >= 0.5
                )
                res.topopt["guide"] = {
                    "coarse_pitch_mm": round(float(coarse.pitch), 4),
                    "coarse_iterations": guide_r.iterations,
                    "kept_cells": int(guide_mask.sum()),
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
        )
        if not crisp:
            thin_cells = float(eff_span.min()) / max(domain.pitch, 1e-6)
            candidate_mesh = smooth_preserve(
                candidate_mesh,
                iterations=2 if thin_cells < 4.0 else 4,
                pin_floor=base_flat,
            )
        candidate_mesh, dropped_slivers = keep_significant_components(candidate_mesh)
        warnings = list(extract_warnings)
        if dropped_slivers:
            warnings.append(f"dropped {dropped_slivers} near-zero-volume mesh sliver components")
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
        elif opt.infill_scope == "guided" and int(meta.get("interior_cells") or 0) == 0:
            warnings.append(
                "guided SIMP retained no interior cells for lattice — the export is shell-only; "
                "use infill_scope='full' or revise the guided topology"
            )
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
    target_min: float | None = None
    target_max: float | None = None
    target_infeasible = False
    volume_tolerance = 0.05
    if abs(float(best["export_fraction"]) - target_volfrac) > volume_tolerance:
        offset_min = p_infill.dmin - p_infill.dmax
        offset_max = p_infill.dmax - p_infill.dmin
        low = _build_candidate(offset_min)
        high = _build_candidate(offset_max)
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
        spsolve_dofs=s.fem_spsolve_dofs,
        pool_workers=s.exec_pool_workers,
        mem_budget_mb=mem_budget,
        source_mesh=opt_mesh,
        cancelled=cancelled,
        solver_backend=solver_backend,
    )

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
        verify_exp = exp_info["report"]
        n_bodies_export = exp_info["n_bodies_export"]
        n_bodies_voxel = exp_info["n_bodies_voxel"]
        res.topopt.setdefault("warnings", []).extend(exp_info["warnings"])
    except Exception as e:
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
            retry_report = retry_info["report"]
            retry_bodies = retry_info["n_bodies_export"]
            retry_voxel_bodies = retry_info["n_bodies_voxel"]
            retry_export_ok = True
            res.topopt.setdefault("warnings", []).extend(retry_info["warnings"])
        except Exception as e:
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
            base_D = (
                _D(mat.poisson, mat.aniso_z, mat.aniso_axis) if mat.aniso_z != 1.0 else None
            )
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
                spsolve_dofs=s.fem_spsolve_dofs,
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
            res.fem_homog = verify_h.model_dump()
            report_to_json(verify_h, out_dir / "fem" / "report_homog.json")
        except Exception as e:
            verify_h = None
            log.warning("infill.homog_verify.failed", error=str(e)[:200])
            res.topopt["warnings"].append(
                f"homogenized infill verification failed: {str(e)[:80]}"
            )

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
        "bottleneck_mm2": (exp_info or {}).get("bottleneck_mm2"),
        "sf_floor": (exp_info or {}).get("sf_floor"),
        "load_path_ok": (exp_info or {}).get("load_path_ok"),
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
        res.topopt["warnings"].append(
            "load-path certificate: a load case cannot reach its fixtures "
            "through the export solid — the part is structurally disconnected"
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
            "bottleneck_mm2": verification["bottleneck_mm2"],
            "sf_floor": verification["sf_floor"],
            "load_path_ok": verification["load_path_ok"],
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
        }
    )

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
    try:
        opt_mesh.export(str(stl_path), file_type="stl")
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
    if res.analysis.watertight is False:
        _repair_export_meshes(res.exports, out_dir / "exports")
    res.mesh = {
        "pipeline": "infill",
        "triangles": len(opt_mesh.faces),
        "volume_mm3": round(float(opt_mesh.volume), 1),
        "solid_elements": int(final_mask.sum()),
        "pitch_mm": domain.pitch,
    }

    t = time.monotonic()
    render_dir = out_dir / "renders"
    render_paths = render_views(stl_path, render_dir, res.analysis.bbox_size)
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

    if not opt_mesh.is_watertight:
        res.topopt["warnings"].append(
            "optimized mesh is not watertight — inspect before manufacturing"
        )
    res.ok = True
    _attach_verify_spec(res, spec, ProjectSettings(), s, envelope_bound_only=True)
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res


def run_topopt_resample(
    *,
    density_relpath: str,
    out_dir: Path,
    load_cases: list[dict[str, Any]],
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
        auto_keep_regions,
        keep_element_mask,
        region_node_mask,
        resolve_case_dofs,
    )
    from cadx.cad.fem.topopt import passive_solid_mask
    from cadx.cad.fem.voxelize import VoxelDomain
    from cadx.llm.schemas import LoadCase, OptimizeSpec

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
            load_node_masks.append(region_node_mask(domain, ld.region, anchors))
    keep_list = list(opt.keep_regions)
    # same auto-keep rule as run_topopt_pipeline: keep_mode="manual" only
    # opts out via the runtime `topopt_auto_keep` flag
    n_auto_keeps = 0
    if anchors and (opt.keep_mode == "auto" or (opt.keep_mode == "manual" and s.topopt_auto_keep)):
        margin = opt.auto_keep_margin_mm or s.topopt_auto_keep_margin_mm
        auto_keeps = auto_keep_regions(anchors, margin)
        keep_list += auto_keeps
        n_auto_keeps = len(auto_keeps)
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
    elif opt.void_regions:
        void_flat = keep_element_mask(domain, list(opt.void_regions), anchors)

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
        }
    )

    # same smoothing stage as the original run — floor pinning follows the
    # flag stored in the density field
    from cadx.cad.mesh_ops import smooth_preserve

    opt_mesh = smooth_preserve(
        opt_mesh, iterations=4, pin_floor=bool(npz.get("flat_bottom", False))
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

    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    try:
        opt_mesh.export(str(stl_path), file_type="stl")
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
            verify_exp = info["report"]
            report_to_json(verify_exp, out_dir / "fem" / "report_export.json")
            res.fem_export = verify_exp.model_dump()
            res.topopt["n_bodies_export"] = info["n_bodies_export"]
            res.topopt["n_bodies_voxel"] = info["n_bodies_voxel"]
            res.topopt["sf_mask"] = verify.worst_safety_factor
            res.topopt["sf_export"] = verify_exp.worst_safety_factor
            res.topopt["sf_verified"] = info["sf_verified"]
            res.topopt["sf_realizations"] = info["sf_realizations"]
            res.topopt["bottleneck_mm2"] = info["bottleneck_mm2"]
            res.topopt["sf_floor"] = info["sf_floor"]
            res.topopt["load_path_ok"] = info["load_path_ok"]
            res.topopt["sf_mesh_sensitive"] = info["sf_mesh_sensitive"]
            res.topopt.setdefault("warnings", []).extend(info["warnings"])
        except Exception as e:
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
    res.mesh = {
        "pipeline": "topopt_resample",
        "triangles": len(opt_mesh.faces),
        "volume_mm3": round(float(opt_mesh.volume), 1),
        "solid_elements": int(final_mask.sum()),
        "pitch_mm": pitch,
    }
    render_dir = out_dir / "renders"
    render_paths = render_views(stl_path, render_dir, res.analysis.bbox_size)
    res.renders = {v: p.name for v, p in render_paths.items()}
    thumb = render_thumbnail(stl_path, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
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
    )

    # --- artifacts: STL/GLB/3MF + STL copy used for renders/analysis -------
    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    try:
        opt_mesh.export(str(stl_path), file_type="stl")
    except Exception as e:
        res.error = f"failed to write organic STL: {e}"
        return res
    res.exports = export_mesh_all(opt_mesh, out_dir / "exports", name=name, try_step=False)
    res.exports["dxf"] = None
    res.exports["svg"] = None
    res.analysis = mesh_analysis(opt_mesh)
    _repair_analysis_mesh(res.analysis, stl_path)
    if res.analysis.watertight is False:
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
    render_paths = render_views(stl_path, render_dir, res.analysis.bbox_size)
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
    helical_threads: bool = False,
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
        # job/spec-level request OR the runtime default — an explicit
        # per-feature helical=False still wins inside apply_refine
        helical_threads=bool(helical_threads) or s.refine_helical_threads,
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
        out.export(str(stl_path), file_type="stl")
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

    # FEM re-verify — the cuts remove material, the SF must be re-issued
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return res
    cases = [LoadCase(**c) for c in (load_cases or [])]
    if cases:
        verify = run_fem(
            out,
            cases,
            material,
            voxel_mm=voxel_mm or s.fem_voxel_mm,
            max_elements=max_elements or s.fem_max_elements,
            out_dir=out_dir,
            anchors=anchors,
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
    if res.analysis.watertight is False:
        _repair_export_meshes(res.exports, out_dir / "exports")
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
    render_paths = render_views(stl_path, render_dir, res.analysis.bbox_size)
    res.renders = {v: p.name for v, p in render_paths.items()}
    thumb = render_thumbnail(stl_path, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
    if not out.is_watertight:
        warnings.append("refined mesh is not watertight — inspect before manufacturing")
    res.ok = True
    # canonical conformance report on the finished part — bound semantics:
    # the refined body is still the optimized envelope, not a CAD target
    _attach_verify_spec(res, spec, ProjectSettings(), s, envelope_bound_only=True)
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res


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


def copy_artifacts(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        s = src / n
        if s.exists():
            shutil.copy2(s, dst / n)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
