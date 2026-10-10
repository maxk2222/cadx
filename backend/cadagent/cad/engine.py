"""Pipeline executed inside the cad-executor: sandbox -> analysis -> renders -> DfM -> exports.

Pure function (no DB, no queue) so it can be reused by the arq task, the CLI and tests.
All inputs are file paths + plain settings dicts; output is an ExecResult.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from build123d import import_brep

from cadagent.cad.analysis import Analysis, analyze
from cadagent.cad.contact_sheet import make_contact_sheet
from cadagent.cad.dfm import DfmReport
from cadagent.cad.dfm.cnc import cnc_checks, cnc_mesh_checks
from cadagent.cad.dfm.common import common_checks, load_mesh
from cadagent.cad.dfm.fdm import fdm_checks, fdm_mesh_checks
from cadagent.cad.dfm.report import DfmWarning
from cadagent.cad.exporters import export_2d, export_all_3d, export_mesh_all, quick_stl
from cadagent.cad.mesh_ops import (
    MeshPostParams,
    candidate_meshes,
    clean,
    decimate,
    flat_base_ratio,
    flatten_bottom,
    mesh_analysis,
    mesh_to_solid,
    orient,
    rest_on_xy_center,
    scale_to_spec,
)
from cadagent.cad.mesh_ops import (
    load_mesh as load_triangle_mesh,
)
from cadagent.cad.render import render_thumbnail, render_views
from cadagent.cad.sandbox import Reference, RunnerOutput, run_user_code
from cadagent.config import get_settings
from cadagent.db.models import ProjectSettings
from cadagent.llm.schemas import PartSpec
from cadagent.logging import get_logger

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
        self.thread_count: int = 0
        self.timings: dict[str, float] = {}

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
            "thread_count": self.thread_count,
            "timings": self.timings,
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
        r.timings = d.get("timings") or {}
        if d.get("analysis"):
            r.analysis = Analysis(**d["analysis"])
        if d.get("dfm"):
            r.dfm = DfmReport(**d["dfm"])
        r.mesh = d.get("mesh") or {}
        return r


def run_pipeline(
    *,
    code: str,
    out_dir: Path,
    params_override: dict[str, object] | None = None,
    references: list[Reference] | None = None,
    settings: ProjectSettings | None = None,
    full_exports: bool = False,
    name: str = "model",
    mesh_solid: Path | None = None,
    mesh_bbox: list[float] | None = None,
) -> ExecResult:
    """Full executor pipeline for one candidate. out_dir receives all artifacts."""
    res = ExecResult()
    settings = settings or ProjectSettings()
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
    )
    res.stdout = ro.stdout
    res.params_used = ro.params_used
    res.thread_count = ro.thread_count
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
        f"watertight: {res.analysis.watertight}",
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

    # 7. exports
    t = time.monotonic()
    exp_dir = out_dir / "exports"
    if full_exports:
        res.exports = export_all_3d(shape, exp_dir, name=name)
        res.exports.update(export_2d(shape, exp_dir, name, ro.flat_brep_path))
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


def export_from_mesh(stl_path: Path, out_dir: Path, name: str = "model") -> dict[str, str | None]:
    """Export set for mesh-mode versions from a stored STL (used by FINALIZE / reexport)."""
    mesh = load_triangle_mesh(stl_path)
    out = export_mesh_all(mesh, out_dir, name=name)
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
    full_exports: bool = False,
    name: str = "model",
    make_solid: bool = False,
) -> ExecResult:
    """Mesh-mode pipeline: clean -> orient -> scale -> analyse -> render -> DfM -> export.

    Input is the raw generator output (GLB/STL/...) in arbitrary normalized
    space; the result follows the frame contract (mm, +Z up, on XY, centred).
    """
    res = ExecResult()
    settings = settings or ProjectSettings()
    post = post or MeshPostParams()
    t0 = time.monotonic()
    workdir = out_dir / "_run"
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. load + clean
    try:
        raw = load_triangle_mesh(mesh_path)
    except Exception as e:
        res.error = f"failed to load generated mesh: {e}"
        return res
    res.mesh["input_triangles"] = len(raw.faces)
    mesh = clean(raw)

    # 2. decimate (explicit target, or CAD cap when a solid is needed)
    target = post.decimate_faces or 0
    if make_solid:
        target = target or get_settings().mesh_max_faces_for_cad
    if target:
        mesh = decimate(mesh, target)

    # 3. orient / mirror / yaw
    mesh = orient(mesh, post.up, post.front, post.rotate_z_deg, post.mirror_x)
    res.mesh["orientation"] = {
        "up": post.up,
        "front": post.front,
        "rotate_z_deg": post.rotate_z_deg,
        "mirror_x": post.mirror_x,
    }

    # 4. scale to spec (uniform) + user multiplier
    mesh, factor, dim_issues = scale_to_spec(mesh, spec)
    if post.scale and abs(post.scale - 1.0) > 1e-9:
        mesh.apply_scale(post.scale)
        factor *= post.scale
    res.mesh["scale_factor"] = round(factor, 6)
    res.mesh["dim_issues"] = dim_issues

    # 5. flatten bottom (optional), then rest on XY / centre
    if post.flatten_bottom_mm > 0:
        mesh = flatten_bottom(mesh, post.flatten_bottom_mm)
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

    # 8. optional solid conversion (hybrid bridge + tessellated STEP)
    if make_solid:
        t = time.monotonic()
        try:
            solid = mesh_to_solid(mesh)
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
        f"watertight: {res.analysis.watertight}",
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
    from cadagent.cad.render import render_available

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


def copy_artifacts(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        s = src / n
        if s.exists():
            shutil.copy2(s, dst / n)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
