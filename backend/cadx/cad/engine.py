"""Pipeline executed inside the cad-executor: sandbox -> analysis -> renders -> DfM -> exports.

Pure function (no DB, no queue) so it can be reused by the arq task, the CLI and tests.
All inputs are file paths + plain settings dicts; output is an ExecResult.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
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


def run_pipeline(
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
        return res

    # 2. load shape
    try:
        shape = import_brep(str(ro.brep_path))
    except Exception as e:
        res.error = f"failed to load generated shape: {e}"
        return res

    # 3. quick STL for mesh analysis + renders
    t = time.monotonic()
    mesh_dir = out_dir / "renders"
    try:
        stl_path = quick_stl(shape, mesh_dir, name="mesh")
    except Exception as e:
        res.error = f"tessellation failed: {e}"
        return res
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

    res.ok = True
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res


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

    # 1-5. baseline: clean -> decimate -> orient -> scale -> flatten -> rest
    try:
        raw = load_triangle_mesh(mesh_path)
    except Exception as e:
        res.error = f"failed to load generated mesh: {e}"
        return res
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
            return res
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
            mesh = _ensure_watertight(mesh)
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
        return res

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
    return res


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
    from cadx.cad.fem.voxelize import voxelize
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

    # --- design domain -------------------------------------------------------
    try:
        domain = voxelize(mesh, voxel_mm or s.fem_voxel_mm, max_elements or s.fem_max_elements)
    except Exception as e:
        res.error = f"voxelization failed: {e}"
        return res

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

        def _add(c: LoadCase, weight: float) -> None:
            fixed3, F, w = resolve_case_dofs(dom, c, anchors, density_g_cm3=mat.density_g_cm3)
            if warn:
                res.topopt.setdefault("warnings", []).extend(w)
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
        return r_resolved, r_weights, r_fixed_nm, r_load_nm

    resolved, case_weights, fixed_node_masks, load_node_masks = _resolve_cases_on(domain)
    if not resolved:
        res.error = "no solvable load case — check fixed/load regions"
        return res

    # --- non-design zones ----------------------------------------------------
    keep_list = list(opt.keep_regions)
    if opt.keep_mode == "auto" and anchors:
        margin = opt.auto_keep_margin_mm or s.topopt_auto_keep_margin_mm
        keep_list += auto_keep_regions(anchors, margin)
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
        self_supporting=bool(getattr(opt, "self_supporting", False)),
        print_direction=str(getattr(opt, "print_direction", "+z") or "+z"),
        penal=float(opt.penal or 3.0),
        penal_init=opt.penal_init,
        solver=str(getattr(opt, "solver", "oc") or "oc"),
        case_objective=str(getattr(opt, "case_objective", "weighted") or "weighted"),
        max_displacement_mm=opt.max_displacement_mm,
        max_stress_mpa=opt.max_stress_mpa,
    )

    # insert/material zones: keep-regions with `material` change E per cell
    E0_arr = mat.elastic_modulus_mpa * material_cell_factors(
        domain, keep_list, anchors, mat.elastic_modulus_mpa
    )

    # optional warm start from a previous run's density field
    x0 = None
    if opt.warm_density_relpath:
        try:
            from cadx import storage

            warm = np.load(storage.resolve(opt.warm_density_relpath))
            x0 = np.asarray(warm["density"], dtype=float)
            if x0.size != domain.cell_count:
                res.topopt["warnings"].append("warm-start density has a different grid — ignored")
                x0 = None
        except Exception as e:
            res.topopt["warnings"].append(f"warm-start density unreadable: {e}")

    # multiresolution: run a coarse SIMP at 2x pitch first, upsample the
    # density and warm-start the fine grid — converges ~3x faster on big parts
    multires_min = int(getattr(s, "topopt_multires_min_cells", 4000) or 0)
    if getattr(opt, "multires", False) and x0 is None and domain.solid_count > multires_min:
        t_mr = time.monotonic()
        try:
            coarse = voxelize(mesh, domain.pitch * 2.0, max_elements or s.fem_max_elements)
            if coarse.cell_count >= domain.cell_count // 2:
                res.topopt["warnings"].append("multires: coarse grid not much coarser — skipped")
            else:
                res_c, w_c, fix_c, load_nm_c = _resolve_cases_on(coarse, warn=False)
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

    def _one_pass(volfrac: float) -> dict[str, Any]:
        """One SIMP run at `volfrac` + extraction, smoothing and both FEM
        verifications. Returns the pieces needed for artifacts/retry."""
        out: dict[str, Any] = {"volfrac": volfrac}
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
            x0=x0,
            spsolve_dofs=s.fem_spsolve_dofs,
            cancelled=cancelled,
            progress_file=opt_dir / f"history_{volfrac:.2f}.json",
        )
        out["simp_s"] = round(time.monotonic() - t, 2)
        out["result"] = result
        out["params"] = p

        opt_mesh, final_mask, extract_warnings, n_components = density_to_mesh(
            domain,
            result.density,
            threshold=p.threshold,
            passive_solid=passive,
            min_member_mm=min_member,
            min_gap_mm=min_gap,
            smooth_iterations=s.topopt_smooth_iterations,
        )
        out["extract_warnings"] = extract_warnings
        out["n_components"] = n_components
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
                suffix="_export",
            )
        except Exception as e:
            log.warning("topopt.verify_export.failed", error=str(e)[:200])
            verify_exp = verify
        out["verify_export"] = verify_exp
        return out

    p1 = _one_pass(params.volfrac)
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
        p2 = _one_pass(params.volfrac + 0.1)
        sf_exp2 = p2["verify_export"].worst_safety_factor
        res.topopt["retry"] = {
            "volfrac_first": params.volfrac,
            "volfrac_second": params.volfrac + 0.1,
            "sf_export_first": sf_exp,
            "sf_export_second": sf_exp2,
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
    res.topopt.update(
        {
            "iterations": result.iterations,
            "converged": result.converged,
            "volfrac_target": best["volfrac"],
            "volfrac_achieved": round(
                float(result.density[domain.solid.ravel()].sum()) / max(int(domain.solid.sum()), 1),
                4,
            ),
            "history": result.history[-50:],
            "passive_frac": round(result.passive_frac, 4),
            "warnings": res.topopt.get("warnings", []) + result.warnings,
            "material": mat.name,
            "sf_mask": verify.worst_safety_factor,
            "sf_export": verify_export.worst_safety_factor,
            "min_member_mm": best["min_member_mm"],
            "thin_fraction": round(best["thin_fraction"], 4),
            "n_components": best["n_components"],
        }
    )

    np.savez_compressed(
        opt_dir / "density.npz",
        density=result.density.astype(np.float32),
        mask=final_mask,
        solid=domain.solid,
        passive=passive.reshape(domain.dims),
        origin=domain.origin,
        pitch=np.float64(domain.pitch),
        dims=np.asarray(domain.dims),
        flat_bottom=np.bool_(base_flat),
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
    res.exports = export_mesh_all(opt_mesh, out_dir / "exports", name=name, try_step=False)
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
    if opt.keep_mode == "auto" and anchors:
        margin = opt.auto_keep_margin_mm or s.topopt_auto_keep_margin_mm
        keep_list += auto_keep_regions(anchors, margin)
    keep_cells = keep_element_mask(domain, keep_list, anchors)
    passive = passive_solid_mask(domain, [keep_cells], fixed_node_masks, load_node_masks)
    if "passive" in npz:
        # stored pin mask — carries organic floor/shell/auto-keep cells that
        # region-based rebuild cannot express; harmless union for topopt
        # fields where it coincides with the rebuilt mask
        passive |= np.asarray(npz["passive"], dtype=bool).ravel()

    try:
        opt_mesh, final_mask, extract_warnings, n_components = density_to_mesh(
            domain,
            density,
            threshold=threshold,
            passive_solid=passive,
            min_member_mm=min_member,
            min_gap_mm=min_gap,
            smooth_iterations=s.topopt_smooth_iterations,
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

    res.exports = export_mesh_all(opt_mesh, out_dir / "exports", name=name, try_step=False)
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
    """
    from cadx.cad.fem.analyze import report_to_json, run_fem
    from cadx.cad.fem.refine import apply_refine, resolve_refine_features
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

    feats, res_warnings = resolve_refine_features(anchors, spec_features, features)
    if helical_threads:
        for f in feats:
            if f.kind in ("thread", "tap"):
                f.helical = True
    keep_regions = list(keep_regions or [])
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

    res.exports = export_mesh_all(out, out_dir / "exports", name=name, try_step=False)
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
