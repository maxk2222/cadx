"""Pipeline executed inside the cad-executor: sandbox -> analysis -> renders -> DfM -> exports.

Pure function (no DB, no queue) so it can be reused by the arq task, the CLI and tests.
All inputs are file paths + plain settings dicts; output is an ExecResult.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import import_brep

from cadx.cad.analysis import Analysis, analyze
from cadx.cad.contact_sheet import make_contact_sheet
from cadx.cad.dfm import DfmReport
from cadx.cad.dfm.cnc import cnc_checks, cnc_mesh_checks
from cadx.cad.dfm.common import common_checks, load_mesh
from cadx.cad.dfm.fdm import fdm_checks, fdm_mesh_checks
from cadx.cad.dfm.report import DfmWarning
from cadx.cad.exporters import (
    export_2d,
    export_all_3d,
    export_analytic_step,
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
    mesh_analysis,
    mesh_to_solid,
    rest_on_xy_center,
    smooth_preserve,
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

    # 3. quick STL for mesh analysis + renders
    t = time.monotonic()
    mesh_dir = out_dir / "renders"
    try:
        stl_path = quick_stl(shape, mesh_dir, name="mesh")
    except Exception as e:
        res.error = f"tessellation failed: {e}"
        return _attach_verify_spec(res, spec_dict, settings, rt)
    res.timings["tessellate_s"] = round(time.monotonic() - t, 2)

    # 4. analysis
    res.analysis = analyze(shape, stl_path)
    _repair_analysis_mesh(res.analysis, stl_path)

    # 4b. deterministic feature extraction (feedback-only, never blocks)
    try:
        from cadx.cad.features import extract_features

        res.features = extract_features(shape, stl_path).model_dump()
    except Exception as e:
        log.warning("features.extract_failed", error=str(e)[:200])

    # 4b2. invalid B-Rep → locate defects so repair feedback gets coordinates
    # (WHERE geometry is broken), not just "is_valid: false"
    if not res.analysis.is_valid:
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

    if not light:
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
        res.timings["render_s"] = round(time.monotonic() - t, 2)

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
        res.timings["dfm_s"] = round(time.monotonic() - t, 2)
    else:
        res.timings["render_s"] = 0.0
        res.timings["dfm_s"] = 0.0

    # 7. exports
    t = time.monotonic()
    exp_dir = out_dir / "exports"
    if full_exports:
        res.exports = export_all_3d(shape, exp_dir, name=name)
        res.exports.update(export_2d(shape, exp_dir, name, ro.flat_brep_path))
        if res.analysis.watertight is False:
            # same repair the analysis STL got — the shipped mesh formats
            # must not leak non-watertight geometry to the slicer
            _repair_export_meshes(res.exports, exp_dir)
    else:
        # iterations only need the STL (already exported for renders)
        res.exports = {"stl": None}
    res.timings["export_s"] = round(time.monotonic() - t, 2)

    # 8. STEP roundtrip gate — a shape can be valid in memory yet break on
    # serialisation; bounded by face count (inside the check) and by the
    # remaining exec budget here. Success-path only, never fatal.
    t = time.monotonic()
    if t - t0 <= rt.exec_timeout_s * _STEP_GATE_BUDGET_FRACTION:
        try:
            from cadx.cad.defects import step_roundtrip_check

            shipped = res.exports.get("step")
            if shipped:
                gate = step_roundtrip_check(shape, exp_dir / shipped, write=False)
            else:
                gate = step_roundtrip_check(shape, workdir / "step_gate.step")
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
            pl.add_mesh(pvm, color="#9aa3ad", smooth_shading=True)
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


def normalize_mesh_to_spec_envelope(
    mesh: trimesh.Trimesh, spec_dims_mm: list[float | None]
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Clip an out-of-envelope topopt input back to the declared spec dims.

    The clip is a manifold boolean intersection with the spec-envelope box
    (same engine as the rest of the codebase). A failed/empty boolean falls
    back to uniform scale-to-fit — only on a watertight input, which scaling
    keeps watertight-plausible; leaky geometry passes through unchanged with
    a warning. Never raises, never blocks the run.

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
    over = any(
        ext[ax] > d + _ENVELOPE_CLIP_SLACK_MM or ext[ax] > d * (1.0 + _ENVELOPE_CLIP_SLACK_FRAC)
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
            cells_preview,
            render_regions,
            resolve_case_summary,
        )
        from cadx.cad.fem.voxelize import voxelize

        try:
            domain = voxelize(mesh, voxel_mm or s.fem_voxel_mm, max_elements or s.fem_max_elements)
        except Exception as e:
            return {"ok": False, "error": f"voxelization failed: {e}"}
        pngs = render_regions(domain, cases, keep, out_dir / "fem", anchors) if render_pngs else []
        return {
            "ok": True,
            "preview": {
                "pitch_mm": round(domain.pitch, 4),
                "dims": list(domain.dims),
                "solid_elements": domain.solid_count,
                "cases": [resolve_case_summary(domain, c, anchors) for c in cases],
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
        mem_budget_mb=float(s.exec_mem_limit_mb),
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
    from cadx.cad.fem.extract import density_to_mesh
    from cadx.cad.fem.loads import expand_cone
    from cadx.cad.fem.preview import render_regions
    from cadx.cad.fem.regions import (
        auto_keep_regions,
        empty_region_warnings,
        keep_element_mask,
        material_cell_factors,
        region_node_mask,
        resolve_case_dofs,
    )
    from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
    from cadx.cad.fem.voxelize import grid_bounds_mm, voxelize
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
    mat = resolve_material(mat_name)

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
    dims_spec = spec_dims_mm if spec_dims_mm is not None else (optimize or {}).get("spec_dims_mm")
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
    try:
        domain = voxelize(mesh, voxel_mm or s.fem_voxel_mm, max_elements or s.fem_max_elements)
    except Exception as e:
        res.error = f"voxelization failed: {e}"
        return res
    d_lo, d_hi = grid_bounds_mm(domain)
    res.topopt["domain_bbox"] = [
        [round(float(v), 3) for v in d_lo],
        [round(float(v), 3) for v in d_hi],
    ]

    # predictive memory budget — an OOM kill mid-SIMP loses the whole job;
    # better to refuse early with a pointer to the knobs that fix it
    from cadx.cad.fem.voxelize import estimate_fem_memory_mb

    mem_budget = float(getattr(s, "exec_mem_limit_mb", 0) or 0)
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
        r_resolved: list[tuple[np.ndarray, np.ndarray]] = []
        r_weights: list[float] = []
        r_fixed_nm: list[np.ndarray] = []
        r_load_nm: list[np.ndarray] = []
        r_unresolved: list[str] = []

        def _add(c: LoadCase, weight: float) -> None:
            fixed3, F, w = resolve_case_dofs(dom, c, anchors, density_g_cm3=mat.density_g_cm3)
            if warn:
                res.topopt.setdefault("warnings", []).extend(w)
                # a fixed/load region that matched nothing means the constraint
                # the spec declared is absent from the geometry — optimizing
                # anyway produces a structurally meaningless result
                r_unresolved.extend(
                    m for m in w if re.search(r"(fixed|load) region .*matched no nodes", m)
                )
            if not fixed3.any() or not np.any(F):
                if warn:
                    res.topopt["warnings"].append(
                        f"case '{c.name}' skipped — no fixed region or no effective load"
                    )
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
                            res.topopt["warnings"].append(
                                f"direction-cone expansion capped at {max_cases} cases"
                            )
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
                r_load_nm.append(region_node_mask(dom, load.region, anchors))
        for f3, _ in r_resolved:
            r_fixed_nm.append(np.asarray(f3).any(axis=1) if np.ndim(f3) == 2 else f3)
        return r_resolved, r_weights, r_fixed_nm, r_load_nm, r_unresolved

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
    res.topopt.setdefault("warnings", []).extend(empty_region_warnings(domain, keep_list, anchors))
    res.topopt.setdefault("warnings", []).extend(
        empty_region_warnings(domain, list(opt.void_regions), anchors)
    )
    keep_cells = keep_element_mask(domain, keep_list, anchors)
    void_cells = keep_element_mask(domain, list(opt.void_regions), anchors)
    passive = passive_solid_mask(domain, [keep_cells], fixed_node_masks, load_node_masks)

    # region-selection previews so the agent/user can see where forces land
    res.topopt["regions_png"] = render_regions(domain, cases, keep_list, out_dir / "fem", anchors)

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
        solver=str(getattr(opt, "solver", "oc") or "oc"),
        case_objective=str(getattr(opt, "case_objective", "weighted") or "weighted"),
        max_displacement_mm=opt.max_displacement_mm,
        max_stress_mpa=opt.max_stress_mpa,
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
            coarse = voxelize(mesh, domain.pitch * 2.0, max_elements or s.fem_max_elements)
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
                    passive_c = passive_solid_mask(coarse, [keep_c], fix_c, load_nm_c)
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
                        self_supporting=params.self_supporting,
                        print_direction=params.print_direction,
                        symmetry=params.symmetry,  # keep the warm-start symmetric too
                    )
                    warm_r = simp(
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
                    x0 = _upsample_density(warm_r.density.reshape(coarse.dims), coarse, domain)
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
        volfrac: float, pass_idx: int = 1, warm: np.ndarray | None = None
    ) -> dict[str, Any]:
        """One SIMP run at `volfrac` + extraction, smoothing and both FEM
        verifications. Returns the pieces needed for artifacts/retry.
        `warm` overrides the run-level x0 (retry passes warm-start from the
        first pass's converged field)."""
        out: dict[str, Any] = {"volfrac": volfrac}
        # pass-indexed render names: pass 1 keeps the historical names,
        # retries must not overwrite its evidence in out_dir/fem
        fem_suffix = "" if pass_idx <= 1 else f"_p{pass_idx}"
        p = dataclasses.replace(params, volfrac=volfrac)
        t = time.monotonic()
        result = simp(
            domain,
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
            domain,
            result.density,
            threshold=p.threshold,
            passive_solid=passive,
            # declared void regions + outside-domain cells must stay empty —
            # min-member dilation / regularization would regrow them otherwise
            void_cells=void_cells.reshape(domain.dims) | ~domain.solid,
            min_member_mm=min_member,
            min_gap_mm=min_gap,
            smooth_iterations=s.topopt_smooth_iterations,
            extract_smooth=opt.extract_smooth,
            sigma_px=opt.extract_sigma_px,
            volume_preserve=opt.volume_preserve,
        )
        out["extract_warnings"] = extract_warnings
        out["n_components"] = n_components
        out["extract"] = extract_info
        # floor-pinned smoothing: keeps a flat manufacturing base flat
        opt_mesh = smooth_preserve(opt_mesh, iterations=4, pin_floor=base_flat)
        out["opt_mesh"] = opt_mesh
        out["final_mask"] = final_mask

        # thin-member metrics (printability feedback for the agent)
        edt = ndimage.distance_transform_edt(final_mask)
        inside = edt[final_mask]
        out["thin_fraction"] = float((inside <= 1.0).mean()) if len(inside) else 0.0
        out["min_member_mm"] = (
            round(2 * domain.pitch * float(np.percentile(inside, 5)), 2) if len(inside) else 0.0
        )

        t = time.monotonic()
        verify = run_fem_domain(
            domain,
            final_mask,
            cases,
            mat_name,
            out_dir=out_dir,
            anchors=anchors,
            keep_regions=keep_list,
            spsolve_dofs=s.fem_spsolve_dofs,
            suffix=fem_suffix,
        )
        out["verify"] = verify
        out["verify_s"] = round(time.monotonic() - t, 2)

        # export verification: re-voxelize the smoothed mesh that gets printed
        try:
            domain_exp = voxelize(opt_mesh, domain.pitch, max_elements or s.fem_max_elements)
            verify_exp = run_fem_domain(
                domain_exp,
                domain_exp.solid,
                cases,
                mat_name,
                out_dir=out_dir,
                anchors=anchors,
                keep_regions=keep_list,
                spsolve_dofs=s.fem_spsolve_dofs,
                suffix=f"{fem_suffix}_export",
            )
        except Exception as e:
            log.warning("topopt.verify_export.failed", error=str(e)[:200])
            verify_exp = verify
        out["verify_export"] = verify_exp
        return out

    p1 = _one_pass(params.volfrac, pass_idx=1)
    res.timings["simp_s"] = p1["simp_s"]
    res.timings["verify_s"] = p1["verify_s"]
    best = p1

    sf_exp = p1["verify_export"].worst_safety_factor
    if (
        sf_exp is not None
        and sf_exp < opt.safety_factor_min
        and params.volfrac + 0.1 <= 0.9
        and (cancelled is None or not cancelled())
    ):
        # warm-start the retry from pass 1's converged field — the extra
        # volfrac material grows into the existing design instead of
        # re-optimizing from a uniform start
        p2 = _one_pass(params.volfrac + 0.1, pass_idx=2, warm=p1["result"].density)
        sf_exp2 = p2["verify_export"].worst_safety_factor
        res.topopt["retry"] = {
            "volfrac_first": params.volfrac,
            "volfrac_second": params.volfrac + 0.1,
            "sf_export_first": sf_exp,
            "sf_export_second": sf_exp2,
            "warm_start": "pass1_density",
        }
        res.timings["simp_s"] += p2["simp_s"]
        res.timings["verify_s"] += p2["verify_s"]
        if sf_exp2 is not None and sf_exp2 > sf_exp:
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
    verification: dict[str, Any] = {
        "compliance_pred_mj": (result.history[-1].get("compliance_mj") if result.history else None),
        "compliance_mask_j": [c.compliance_j for c in verify.cases],
        "compliance_export_j": [c.compliance_j for c in verify_export.cases],
        "sf_mask": verify.worst_safety_factor,
        "sf_export": verify_export.worst_safety_factor,
        "max_disp_export_mm": max((c.max_disp_mm for c in verify_export.cases), default=None),
        "max_vm_export_mpa": max((c.max_vm_mpa for c in verify_export.cases), default=None),
        "gray_fraction": best["extract"].get("gray_fraction"),
        "vol_error_pct": best["extract"].get("vol_error_pct"),
        # solid fraction of the design space the extracted mask actually keeps
        "volfrac_mask": round(float(final_mask.sum()) / max(int(domain.solid.sum()), 1), 4),
        "solver_used": params.solver,
    }
    if result.history and result.history[-1].get("beta") is not None:
        verification["beta_final"] = result.history[-1]["beta"]
    # degenerate export solve — load regions missed the extracted mesh, the
    # part flies away (disp >> part size) or compliance goes non-physical.
    # The SF number is then meaningless, flag instead of storing as valid.
    span = float(np.linalg.norm(domain.pitch * np.asarray(domain.dims)))
    disp = verification["max_disp_export_mm"]
    comps = [c for c in verification["compliance_export_j"] if c is not None]
    if (comps and any(c <= 0 for c in comps)) or (disp is not None and disp > 10 * span):
        verification["valid"] = False
        res.topopt["warnings"].append(
            "export FEM solve degenerate (unconstrained part or unresolved "
            "load regions) — sf_export not trustworthy"
        )
    else:
        verification["valid"] = True
    if verify.worst_safety_factor is not None and verify_export.worst_safety_factor is not None:
        # >0 means the smoothed export mesh is stiffer than the mask suggests
        verification["sf_gap"] = round(
            verify.worst_safety_factor - verify_export.worst_safety_factor, 4
        )
    if verify.grid.get("dofs"):
        verification["dofs"] = verify.grid["dofs"]
    if verify.cases:
        verification["case_sfs_mask"] = [c.safety_factor for c in verify.cases]
    if verify_export.cases:
        verification["case_sfs_export"] = [c.safety_factor for c in verify_export.cases]

    res.topopt.update(
        {
            "iterations": result.iterations,
            "converged": result.converged,
            "volfrac_target": best["volfrac"],
            "volfrac_achieved": round(
                float(result.density[domain.solid.ravel()].sum()) / max(int(domain.solid.sum()), 1),
                4,
            ),
            "history": _compact_history(result.history),
            "passive_frac": round(result.passive_frac, 4),
            "warnings": res.topopt.get("warnings", []) + result.warnings,
            "material": mat.name,
            "sf_mask": verify.worst_safety_factor,
            "sf_export": verify_export.worst_safety_factor,
            "min_member_mm": best["min_member_mm"],
            "thin_fraction": round(best["thin_fraction"], 4),
            "n_components": best["n_components"],
            "extract": best["extract"],
            "verification": verification,
        }
    )

    np.savez_compressed(
        opt_dir / "density.npz",
        density=result.density.astype(np.float32),
        mask=final_mask,
        solid=domain.solid,
        passive=passive.reshape(domain.dims),
        # persisted so topopt_resample can re-carve declared void regions
        # (kept cells alone cannot express them)
        voids=void_cells.reshape(domain.dims),
        origin=domain.origin,
        pitch=np.float64(domain.pitch),
        dims=np.asarray(domain.dims),
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
    from cadx.cad.fem.voxelize import VoxelDomain, voxelize
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

    run_dir = out_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    stl_path = run_dir / "model.stl"
    try:
        opt_mesh.export(str(stl_path), file_type="stl")
    except Exception as e:
        res.error = f"failed to write resampled STL: {e}"
        return res

    if cases:
        verify = run_fem_domain(
            domain,
            final_mask,
            cases,
            material or "",
            out_dir=out_dir,
            anchors=anchors,
            keep_regions=keep_list,
        )
        report_to_json(verify, out_dir / "fem" / "report.json")
        res.fem = verify.model_dump()
        try:
            domain_exp = voxelize(opt_mesh, pitch, max_elements or s.fem_max_elements)
            verify_exp = run_fem_domain(
                domain_exp,
                domain_exp.solid,
                cases,
                material or "",
                out_dir=out_dir,
                anchors=anchors,
                keep_regions=keep_list,
                suffix="_export",
            )
            report_to_json(verify_exp, out_dir / "fem" / "report_export.json")
            res.fem_export = verify_exp.model_dump()
            res.topopt["sf_mask"] = verify.worst_safety_factor
            res.topopt["sf_export"] = verify_exp.worst_safety_factor
        except Exception as e:
            log.warning("topopt.resample_verify_failed", error=str(e)[:200])
            res.topopt["sf_mask"] = verify.worst_safety_factor

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
        applied_refine_features,
        apply_refine,
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

    # FEM re-verify — the cuts remove material, the SF must be re-issued
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
    # analytic STEP: cut real Cylinder/Helix faces into the sewn solid for
    # the applied features — keeps the faceted STEP as fallback. rep.applied
    # records are self-describing (u/v/dia/depth/thread) so the export does
    # not depend on re-joining feature objects by name.
    applied_feats = applied_refine_features(feats, rep)
    if rep.applied and not applied_feats:
        log.warning(
            "export.analytic_step_name_join_empty",
            applied=len(rep.applied),
            feats=len(feats),
        )
    analytic_src: list[Any] = applied_feats or list(rep.applied)
    if analytic_src:
        step_res = export_analytic_step(
            out,
            analytic_src,
            out_dir / "exports" / f"{name}.step",
            max_faces=s.mesh_max_faces_for_cad,
            log=log,
        )
        if step_res.get("step"):
            res.exports["step"] = step_res["step"]
        res.features["refine"]["step_features"] = step_res.get("step_features") or []
        warnings += step_res.get("warnings") or []
        res.features["refine"]["warnings"] = warnings
    res.exports["dxf"] = None
    res.exports["svg"] = None
    res.analysis = mesh_analysis(out)
    _repair_analysis_mesh(res.analysis, stl_path)
    if res.analysis.watertight is False:
        _repair_export_meshes(res.exports, out_dir / "exports")
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
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res


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
