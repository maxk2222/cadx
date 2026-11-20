"""Pipeline executed inside the cad-executor: sandbox -> analysis -> renders -> DfM -> exports.

Pure function (no DB, no queue) so it can be reused by the arq task, the CLI and tests.
All inputs are file paths + plain settings dicts; output is an ExecResult.

The FEM orchestration lives in `engine/fem_pipelines.py`, the shared
substrate in `engine/shared.py` — both re-exported here so every
`from kiara.cad.engine import ...` call site keeps working unchanged.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from build123d import import_brep

from kiara.cad import resources
from kiara.cad.analysis import _FACET_FACE_CAP, Analysis, analyze
from kiara.cad.contact_sheet import make_contact_sheet
from kiara.cad.dfm import DfmReport
from kiara.cad.dfm.cnc import cnc_checks, cnc_mesh_checks
from kiara.cad.dfm.common import common_checks, load_mesh
from kiara.cad.dfm.fdm import fdm_checks, fdm_mesh_checks
from kiara.cad.dfm.report import DfmWarning
from kiara.cad.exporters import (
    _premesh,
    export_2d,
    export_all_3d,
    export_mesh_all,
    quick_stl,
)
from kiara.cad.mesh_ops import (
    MESH_OPS_VERSION,
    MeshPostParams,
    _ensure_watertight,
    apply_ops,
    baseline_mesh,
    candidate_meshes,
    clean,
    decimate,
    export_stl,
    flat_base_ratio,
    mesh_analysis,
    mesh_to_solid,
    orient_nested_shells,
    rest_on_xy_center,
)
from kiara.cad.mesh_ops import (
    load_mesh as load_triangle_mesh,
)
from kiara.cad.render import render_thumbnail, render_views
from kiara.cad.sandbox import Reference, RunnerOutput, run_user_code
from kiara.cad.spec_features import active_features
from kiara.db.models import ProjectSettings
from kiara.llm.schemas import PartSpec
from kiara.logging import get_logger
from kiara.runtime_config import RuntimeSettings

log = get_logger(__name__)


# --- re-exports: keep every `from kiara.cad.engine import ...` path stable ---
from kiara.cad.engine.fem_pipelines import (  # noqa: E402,F401
    _cancelled_pass,
    _one_pass,
    _pad_cells,
    _resolve_cases_on,
    _verify_one_pass_export,
    run_fem_pipeline,
    run_infill_pipeline,
    run_organic_pipeline,
    run_refine,
    run_topopt_pipeline,
    run_topopt_resample,
)
from kiara.cad.engine.shared import (  # noqa: E402,F401
    _AUX_CG_MAXITER,
    _ENVELOPE_CLIP_SLACK_FRAC,
    _ENVELOPE_CLIP_SLACK_MM,
    _FEATURE_CELLS_PER_DIA,
    _INFILL_LATTICE_MIN_OCCUPANCY,
    _INFILL_SHELL_MM,
    _NO_WARM_KINDS,
    _PIPELINE_STEPS,
    _STEP_GATE_BUDGET_FRACTION,
    _STEP_GATE_MESH_MAX_FACES,
    _THIN_PART_INFILL_FLOOR,
    ExecResult,
    _attach_verify_spec,
    _audit_refined_features,
    _auto_keep_dicts,
    _auto_warm_density,
    _cancelled_pipeline_result,
    _clip_export_to_spec_envelope,
    _compact_history,
    _cone_expand_case,
    _early_loadpath_cert,
    _export_degeneracy,
    _feature_floor_pitch,
    _gate_sf,
    _GridRef,
    _guard_export_grooves,
    _guide_beso_er,
    _infill_volfrac_floor,
    _is_cancelled,
    _load_pins_nodes,
    _mark_cancelled,
    _mark_step,
    _npz_kind,
    _opt_cancel_pair,
    _pitch_feature_warn,
    _post_extract_finish,
    _realization_case_details,
    _recommend_structure_mode,
    _record_lost_load_regions,
    _refine_analytic_step,
    _render_quality,
    _repair_analysis_mesh,
    _repair_export_meshes,
    _resolve_topopt_cases,
    _scan_anchors,
    _sf_basis,
    _snap_pitch_under_floor,
    _solver_tier_gate,
    _spec_dims_from_dict,
    _spec_envelope_bounds,
    _tier_warning,
    _uniform_baseline_check,
    _upsample_density,
    _verify_export_structure,
    copy_artifacts,
    export_from_brep,
    export_from_mesh,
    normalize_mesh_to_spec_envelope,
    pipeline_step,
    write_json,
)


def render_existing(
    source_dir: Path,
    out_dir: Path,
    *,
    detail_mode: str = "preview",
) -> ExecResult:
    """Render-only pipeline — reuses the ``renders/mesh.stl`` + runner
    metadata an earlier attempt already produced instead of re-running the
    sandbox + tessellation + analysis (~50s full pipeline -> a few seconds).

    ``source_dir`` is an attempt dir (``iterations/N/aM``); PNGs land in
    ``out_dir`` like a normal run's artifacts. Geometry facts come from the
    STL itself — no OCC analysis pass."""
    res = ExecResult()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    stl_path = source_dir / "renders" / "mesh.stl"
    if not stl_path.exists():
        res.error = f"render_existing: no mesh.stl under {source_dir}"
        return res
    runner_json = source_dir / "_run" / "result.json"
    if runner_json.exists():
        with contextlib.suppress(Exception):
            runner = json.loads(runner_json.read_text())
            res.design_features = list(runner.get("design_features") or [])
            res.thread_count = int(runner.get("thread_count") or 0)
            res.params_used = dict(runner.get("params_used") or {})
            res.stdout = str(runner.get("stdout") or "")
    # the caller's requested mode wins — a stored "preview" must not veto a
    # deliberate export re-render of the same attempt dir
    res.detail_mode = detail_mode
    try:
        mesh = load_triangle_mesh(stl_path)
    except Exception as e:
        res.error = f"render_existing: failed to load {stl_path}: {e}"
        return res
    merged = mesh.copy()
    merged.merge_vertices()
    res.analysis = Analysis(
        bbox_min=[float(v) for v in mesh.bounds[0]],
        bbox_max=[float(v) for v in mesh.bounds[1]],
        bbox_size=[float(v) for v in mesh.extents],
        volume=float(abs(mesh.volume)),
        surface_area=float(mesh.area),
        solid_count=len(merged.split(only_watertight=False)),
        watertight=bool(mesh.is_watertight),
        watertight_after_repair=bool(merged.is_watertight),
        triangle_count=len(mesh.faces),
        source="mesh",
    )
    thread_preview = (
        [f for f in res.design_features if f.get("kind") == "thread"]
        if res.detail_mode == "preview"
        else []
    )
    render_dir = out_dir / "renders"
    rsize, rghosts = _render_quality(res.detail_mode)
    try:
        render_paths = render_views(
            merged,
            render_dir,
            res.analysis.bbox_size,
            size=rsize,
            thread_features=thread_preview,
            ghosts=rghosts,
        )
    except Exception as e:
        log.warning("render_existing.thread_overlay_failed", error=str(e)[:200])
        render_paths = render_views(
            merged, render_dir, res.analysis.bbox_size, size=rsize, ghosts=rghosts
        )
    res.renders = {v: p.name for v, p in render_paths.items()}
    try:
        thumb = render_thumbnail(merged, out_dir / "thumbnail.png", thread_features=thread_preview)
    except Exception as e:
        log.warning("render_existing.thumbnail_failed", error=str(e)[:200])
        thumb = render_thumbnail(stl_path, out_dir / "thumbnail.png")
    res.thumbnail = thumb.name if thumb else None
    stats = [
        f"bbox: {res.analysis.bbox_size[0]:.1f} x {res.analysis.bbox_size[1]:.1f} x "
        f"{res.analysis.bbox_size[2]:.1f} mm",
        f"volume: {res.analysis.volume:.0f} mm^3",
        f"triangles: {res.analysis.triangle_count}",
        f"watertight: {res.analysis.watertight_after_repair}",
    ]
    sheet = make_contact_sheet(render_paths, out_dir / "contact_sheet.png", stats)
    res.contact_sheet = sheet.name if sheet else None
    res.ok = True
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    return res
def run_mesh_candidates(*, mesh_path: Path, out_dir: Path, size: int = 512) -> dict[str, Any]:
    """Render the 6 up-axis orientation candidates into a labelled contact sheet.

    Returns {ok, candidates: {up: relpath}, sheet: relpath|None, stats: {...}}.
    Used once per version by the ORIENT step (vision LLM picks up/front).
    """
    from kiara.cad.render import render_available

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

    import pyvista as pv
    from PIL import Image, ImageDraw

    cand_dir = out_dir / "orient"
    cand_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[tuple[str, Path]] = []
    # one GL context for all six candidates; the per-iteration Plotter +
    # STL roundtrip cost dominated this helper (~1 s Plotter init each)
    pl: Any = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.set_background("white")
    try:
        for up, m in candidate_meshes(mesh).items():
            faces = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel()
            # weld the facet soup so smooth shading/feature edges behave
            pvm = pv.PolyData(np.asarray(m.vertices, dtype=float), faces).clean()
            pl.clear()
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
            tiles.append((up, png))
            out["candidates"][up] = png.name
    finally:
        pl.close()

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
                from kiara.cad.features import (
                    compare_features,
                    mesh_features_to_measured,
                )

                mf = res.mesh.get("features") or {}
                m = mesh_features_to_measured(mf) if mf else None
                if m is not None:
                    measured = m.model_dump()
                    current = active_features(spec)
                    if current and rt.agent_feature_feedback:
                        findings = [f.model_dump() for f in compare_features(current, m)]
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
                mesh = orient_nested_shells(mesh)
            except Exception:
                pass
    mesh = rest_on_xy_center(mesh)

    res.mesh["post"] = post.model_dump()
    res.mesh["triangles"] = len(mesh.faces)

    # 6. STL into _run (drives renders, analysis, exports)
    stl_path = workdir / "model.stl"
    try:
        mesh = export_stl(mesh, stl_path)
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
        from kiara.cad.mesh_features import detect_mesh_features

        res.mesh["features"] = detect_mesh_features(mesh).model_dump()
    except Exception as e:
        log.warning("mesh_features.failed", error=str(e)[:200])

    # 7c. silhouette IoU vs the reference image — the deterministic shape
    # signal for critique calibration when no plan-view measurement exists.
    # The worker extracts the subject mask once (ref_mask_rel is a PNG in
    # DATA_DIR); the comparison runs on this authoritative processed mesh.
    if ref_mask_rel:
        try:
            from kiara import storage as _storage
            from kiara.cad.silhouette import load_mask, silhouette_iou

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
    _rq_size, _rq_ghosts = _render_quality(res.detail_mode)
    render_paths = render_views(
        stl_path, render_dir, res.analysis.bbox_size, size=_rq_size, ghosts=_rq_ghosts
    )
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
        # unconditional: the in-memory watertight flag does not guarantee
        # the round-tripped files are clean — slivers/flaps can split off
        # as phantom bodies; heavy repair stays conditional inside
        _repair_export_meshes(res.exports, out_dir / "exports")
    else:
        res.exports = {"stl": None}
    res.timings["export_s"] = round(time.monotonic() - t, 2)

    res.ok = True
    res.timings["total_s"] = round(time.monotonic() - t0, 2)
    # canonical conformance report — reads only evidence already in res
    return _done()
def run_mesh_preview(
    *, mesh_path: Path, out_dir: Path, views: tuple[str, ...] = ("iso", "front", "top")
) -> dict[str, Any]:
    """Cheap render-only pass for the mesh-refine session's preview tool.

    Renders the mesh as stored (no baseline/ops — the session writes the
    current working state itself). Returns {ok, images: {view: relname}, stats}.
    """
    from kiara.cad.render import render_available

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
    detail_mode: str | None = None,
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
    detail_mode = detail_mode or ("export" if full_exports else "preview")
    if detail_mode not in {"preview", "export"}:
        invalid = ExecResult()
        invalid.error = f"invalid detail_mode '{detail_mode}'"
        return invalid
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
        detail_mode=detail_mode,
        timeout_s=rt.exec_timeout_s,
        mem_mb=rt.exec_mem_limit_mb,
        warm=rt.exec_warm_runner,
        threads=rt.exec_threads,
    )
    res.stdout = ro.stdout
    res.params_used = ro.params_used
    res.thread_count = ro.thread_count
    res.preview_thread_count = ro.preview_thread_count
    res.detail_mode = ro.detail_mode
    res.design_features = ro.design_features
    res.bom = ro.bom
    res.warnings = list(ro.warnings)
    res.geometry_at_failure = list(ro.geometry_at_failure)
    res.timings["execute_s"] = round(ro.duration_s, 2)
    if not ro.ok or ro.brep_path is None:
        res.error = ro.error or "execution failed"
        res.traceback = ro.traceback
        return _attach_verify_spec(res, spec_dict, settings, rt)

    # 2. load shape
    t = time.monotonic()
    try:
        shape = import_brep(str(ro.brep_path))
    except Exception as e:
        res.error = f"failed to load generated shape: {e}"
        return _attach_verify_spec(res, spec_dict, settings, rt)
    res.timings["brep_s"] = round(time.monotonic() - t, 2)

    # hybrid: the runner ships the result's manifold mesh next to the brep —
    # re-attach it so tessellation/analysis skip re-meshing the faceted solid
    _res_glb = ro.brep_path.parent / "result_mesh.glb"
    if _res_glb.exists():
        try:
            from kiara.cad.mesh_boolean import attach_mesh

            attach_mesh(shape, load_triangle_mesh(_res_glb))
        except Exception:
            pass

    # 3. quick STL for mesh analysis + renders
    _mark_step(step_key, "tessellate")
    t = time.monotonic()
    mesh_dir = out_dir / "renders"
    try:
        # a real thread in export mode tessellated at the coarse render
        # settings reads as disconnected "fins" — match the export STL
        # tolerances so the helix renders as a helix (preview keeps the
        # cheap mesh + KIARA_FEATURES helix overlay)
        fine_thread_mesh = detail_mode == "export" and any(
            feature.get("kind") == "thread" for feature in res.design_features
        )
        stl_path = quick_stl(
            shape,
            mesh_dir,
            name="mesh",
            tolerance=0.01 if fine_thread_mesh else None,
            angular_tolerance=0.1 if fine_thread_mesh else None,
        )
    except Exception as e:
        res.error = f"tessellation failed: {e}"
        return _attach_verify_spec(res, spec_dict, settings, rt)
    res.timings["tessellate_s"] = round(time.monotonic() - t, 2)

    # 4. analysis
    _mark_step(step_key, "analyze")
    t = time.monotonic()
    res.analysis = analyze(shape, stl_path)
    _repair_analysis_mesh(res.analysis, stl_path)
    res.timings["analyze_s"] = round(time.monotonic() - t, 2)
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return _attach_verify_spec(res, spec_dict, settings, rt)

    # the analysis tessellation doubles as the ray-parity containment oracle
    # for every point-in-solid probe below — OCC is_inside costs ~200 ms/pt on
    # curved B-Reps (helical thread solids ground this band for minutes;
    # pipeline_hang step=analyze). Load once, share by reference.
    analysis_mesh = None
    try:
        from kiara.cad.features import _load_watertight_mesh

        analysis_mesh = _load_watertight_mesh(stl_path) if stl_path is not None else None
    except Exception:
        analysis_mesh = None

    # 4b. deterministic feature extraction (feedback-only, never blocks)
    _mark_step(step_key, "features")
    try:
        from kiara.cad.features import extract_features

        # quiddity adds nothing on a declared thread's real-helix rebuild —
        # the helical faces only make it crawl (its evidence stays advisory,
        # the declaration already answers what the part is)
        run_recognition = not (
            detail_mode == "export"
            and any(f.get("kind") == "thread" for f in (res.design_features or []))
        )
        t = time.monotonic()
        res.features = extract_features(
            shape,
            stl_path,
            mesh=analysis_mesh,
            cancelled=cancelled,
            recognition=run_recognition,
        ).model_dump()
        res.timings["features_s"] = round(time.monotonic() - t, 2)
    except Exception as e:
        log.warning("features.extract_failed", error=str(e)[:200])
    if res.design_features:
        res.features["design"] = {
            "detail_mode": detail_mode,
            "thread_count": res.thread_count,
            "preview_thread_count": res.preview_thread_count,
            "records": res.design_features,
        }
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return _attach_verify_spec(res, spec_dict, settings, rt)

    # 4b1. declared connection spans: a "strut from A to B" must carry
    # material along the whole path — a detached bar reads as correctly
    # connected to the visual critic, so the centreline is sampled here
    _mark_step(step_key, "spans")
    try:
        spec_feats = (spec_dict or {}).get("features") or []
        t = time.monotonic()
        if spec_feats:
            from kiara.cad.spans import span_checks

            if not isinstance(res.features, dict):
                res.features = {}
            res.features["spans"] = span_checks(
                shape, spec_feats, mesh=analysis_mesh, cancelled=cancelled
            )
            from kiara.cad.spans import plate_checks

            res.features["plates"] = plate_checks(
                shape,
                spec_feats,
                spec_dict=spec_dict,
                mesh=analysis_mesh,
                cancelled=cancelled,
            )
        res.timings["spans_s"] = round(time.monotonic() - t, 2)
    except Exception as e:
        log.warning("features.span_checks_failed", error=str(e)[:200])
    if _is_cancelled(cancelled):
        _mark_cancelled(res)
        return _attach_verify_spec(res, spec_dict, settings, rt)

    # 4b2. invalid B-Rep → locate defects so repair feedback gets coordinates
    # (WHERE geometry is broken), not just "is_valid: false". On faceted
    # solids BRepCheck localization is a multi-minute per-face grind with no
    # actionable output — the faceted fast path already gates on it
    if not res.analysis.is_valid and res.analysis.face_count <= _FACET_FACE_CAP:
        try:
            from kiara.cad.defects import locate_defects, summarize_defects

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

            from kiara.cad.mesh_features import detect_mesh_features

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
        thread_preview = (
            [feature for feature in res.design_features if feature.get("kind") == "thread"]
            if detail_mode == "preview"
            else []
        )
        rsize, rghosts = _render_quality(detail_mode)
        # the watertight-merged analysis mesh doubles as the render mesh —
        # skips re-parsing the STL soup twice (views + thumbnail)
        rsrc = analysis_mesh if analysis_mesh is not None else stl_path
        try:
            render_paths = render_views(
                rsrc,
                mesh_dir,
                res.analysis.bbox_size,
                size=rsize,
                thread_features=thread_preview,
                ghosts=rghosts,
            )
        except Exception as e:
            # a malformed KIARA_FEATURES record must never kill the exec —
            # re-render bare; the candidate keeps its geometry verdict
            log.warning("pipeline.thread_overlay_failed", error=str(e)[:200])
            render_paths = render_views(
                rsrc, mesh_dir, res.analysis.bbox_size, size=rsize, ghosts=rghosts
            )
        res.renders = {v: p.name for v, p in render_paths.items()}
        try:
            thumb = render_thumbnail(
                rsrc,
                out_dir / "thumbnail.png",
                thread_features=thread_preview,
            )
        except Exception as e:
            log.warning("pipeline.thumbnail_overlay_failed", error=str(e)[:200])
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
            with ThreadPoolExecutor(max_workers=2) as pool:
                # VTK/GL contexts are not thread-safe — the render probe ran
                # on this thread, so renders must stay here; only the pure
                # python/CGAL-side tasks (dfm, exports) go to pool workers.
                f_dfm = pool.submit(_do_dfm)
                f_exp = pool.submit(_do_exports)
                render_s = _do_renders()
                dfm_s, export_s = f_dfm.result(), f_exp.result()
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
            from kiara.cad.defects import STEP_GATE_MAX_FACES, step_roundtrip_check

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
