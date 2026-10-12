"""arq tasks running inside the cad-executor (queue name "cad")."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cadx import storage
from cadx.cad.engine import (
    ExecResult,
    export_from_brep,
    export_from_mesh,
    run_fem_pipeline,
    run_mesh_candidates,
    run_mesh_pipeline,
    run_mesh_preview,
    run_pipeline,
    run_topopt_pipeline,
)
from cadx.cad.mesh_ops import MeshPostParams
from cadx.cad.sandbox import Reference
from cadx.db.models import ProjectSettings
from cadx.llm.schemas import PartSpec
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)


def _runtime(payload: dict[str, Any]) -> RuntimeSettings:
    """Runtime snapshot attached by the worker; defaults when absent."""
    return RuntimeSettings.model_validate(payload.get("runtime_settings") or {})


async def _cancelled(ctx: dict[str, Any], key: str | None) -> bool:
    if not key:
        return False
    try:
        return bool(await ctx["redis"].exists(key))
    except Exception:
        return False


def run_execute_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Synchronous part of execute_version — shared with InProcessExecutor."""
    project_id = uuid.UUID(payload["project_id"])
    version_id = uuid.UUID(payload["version_id"])
    iteration = int(payload.get("iteration", 1))

    out_dir = storage.iteration_dir(project_id, version_id, iteration)
    settings = ProjectSettings(**(payload.get("settings") or {}))
    references = [
        Reference(asset_id=r["asset_id"], path=r["path"]) for r in payload.get("references", [])
    ]

    result = run_pipeline(
        code=payload["code"],
        out_dir=out_dir,
        params_override=payload.get("params_override"),
        references=references,
        settings=settings,
        runtime=_runtime(payload),
        full_exports=bool(payload.get("full_exports")),
        name=payload.get("name", "model"),
        mesh_solid=Path(payload["mesh_solid"]) if payload.get("mesh_solid") else None,
        mesh_bbox=payload.get("mesh_bbox"),
        anchors=payload.get("anchors"),
    )

    d = result.to_dict()
    it_rel = storage.relpath(out_dir)
    d["renders"] = {v: f"{it_rel}/renders/{p}" for v, p in result.renders.items()}
    if result.contact_sheet:
        d["contact_sheet"] = f"{it_rel}/contact_sheet.png"
    if result.thumbnail:
        d["thumbnail"] = f"{it_rel}/thumbnail.png"
    d["exports"] = {k: (f"{it_rel}/exports/{v}" if v else None) for k, v in result.exports.items()}
    log.info("execute_version", version=str(version_id), ok=d["ok"], timings=d["timings"])
    return d


async def execute_version(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): run sandbox + analysis + renders + dfm (+exports)."""
    cancel_key = f"cancel:{payload['job_id']}" if payload.get("job_id") else None
    if await _cancelled(ctx, cancel_key):
        r = ExecResult()
        r.error = "cancelled"
        return r.to_dict()

    d = run_execute_payload(payload)

    if await _cancelled(ctx, cancel_key):
        d["ok"] = False
        d["error"] = "cancelled"
    return d


async def export_all(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Export all formats from a stored iteration BREP into the version exports dir."""
    project_id = uuid.UUID(payload["project_id"])
    version_id = uuid.UUID(payload["version_id"])
    iteration = int(payload["iteration"])
    name = payload.get("name", "model")

    it_dir = storage.iteration_dir(project_id, version_id, iteration)
    brep = it_dir / "_run" / "model.brep"
    flat = it_dir / "_run" / "flat_profile.brep"
    out_dir = storage.export_dir(project_id, version_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if brep.exists():
        exports = export_from_brep(
            brep, out_dir, name=name, flat_brep=flat if flat.exists() else None
        )
    elif (it_dir / "_run" / "model.stl").exists():
        # mesh-mode version: export from the stored STL
        exports = export_from_mesh(
            it_dir / "_run" / "model.stl",
            out_dir,
            name=name,
            max_faces=_runtime(payload).mesh_max_faces_for_cad,
        )
    else:
        return {"ok": False, "error": "no model.brep or model.stl found for this iteration"}
    rel = storage.relpath(out_dir)
    return {"ok": True, "exports": {k: (f"{rel}/{v}" if v else None) for k, v in exports.items()}}


def run_mesh_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Synchronous part of process_mesh — shared with InProcessExecutor.

    mode="candidates": render orientation candidates (returns sheet + per-up pngs).
    mode="final":      full mesh pipeline (clean/orient/scale/analysis/renders/dfm/exports).
    """
    project_id = uuid.UUID(payload["project_id"])
    version_id = uuid.UUID(payload["version_id"])
    iteration = int(payload.get("iteration", 1))
    mode = payload.get("mode", "final")

    out_dir = storage.iteration_dir(project_id, version_id, iteration)
    mesh_rel = payload["mesh_rel"]
    mesh_path = storage.resolve(mesh_rel)

    if mode == "candidates":
        d = run_mesh_candidates(mesh_path=mesh_path, out_dir=out_dir)
        it_rel = storage.relpath(out_dir)
        d["candidates"] = {k: f"{it_rel}/orient/{v}" for k, v in d.get("candidates", {}).items()}
        if d.get("sheet"):
            d["sheet"] = f"{it_rel}/{d['sheet']}"
        return d

    if mode == "preview":
        d = run_mesh_preview(mesh_path=mesh_path, out_dir=out_dir)
        it_rel = storage.relpath(out_dir)
        d["images"] = {k: f"{it_rel}/preview/{v}" for k, v in d.get("images", {}).items()}
        return d

    settings = ProjectSettings(**(payload.get("settings") or {}))
    spec = PartSpec(**payload["spec"]) if payload.get("spec") else None
    post = MeshPostParams(**(payload.get("post") or {}))

    result = run_mesh_pipeline(
        mesh_path=mesh_path,
        out_dir=out_dir,
        post=post,
        spec=spec,
        settings=settings,
        runtime=_runtime(payload),
        full_exports=bool(payload.get("full_exports")),
        name=payload.get("name", "model"),
        make_solid=bool(payload.get("make_solid")),
    )

    d = result.to_dict()
    it_rel = storage.relpath(out_dir)
    d["renders"] = {v: f"{it_rel}/renders/{p}" for v, p in result.renders.items()}
    if result.contact_sheet:
        d["contact_sheet"] = f"{it_rel}/contact_sheet.png"
    if result.thumbnail:
        d["thumbnail"] = f"{it_rel}/thumbnail.png"
    d["exports"] = {k: (f"{it_rel}/exports/{v}" if v else None) for k, v in result.exports.items()}
    if (out_dir / "_run" / "model.brep").exists():
        d["mesh"]["solid_relpath"] = f"{it_rel}/_run/model.brep"
    d["mesh"]["stl_relpath"] = f"{it_rel}/_run/model.stl"
    log.info("process_mesh", version=str(version_id), ok=d["ok"], timings=d["timings"])
    return d


async def process_mesh(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): mesh post-processing + optional solid conversion."""
    cancel_key = f"cancel:{payload['job_id']}" if payload.get("job_id") else None
    if await _cancelled(ctx, cancel_key):
        r = ExecResult()
        r.error = "cancelled"
        return r.to_dict()

    d = run_mesh_payload(payload)

    if await _cancelled(ctx, cancel_key):
        d["ok"] = False
        d["error"] = "cancelled"
    return d


def _relpath_exports(d: dict[str, Any], result: ExecResult, out_dir: Path) -> dict[str, Any]:
    it_rel = storage.relpath(out_dir)
    d["renders"] = {v: f"{it_rel}/renders/{p}" for v, p in result.renders.items()}
    if result.contact_sheet:
        d["contact_sheet"] = f"{it_rel}/contact_sheet.png"
    if result.thumbnail:
        d["thumbnail"] = f"{it_rel}/thumbnail.png"
    d["exports"] = {k: (f"{it_rel}/exports/{v}" if v else None) for k, v in result.exports.items()}
    return d


def run_fem_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Synchronous part of fem_analyze — shared with InProcessExecutor."""
    out_dir = storage.iteration_dir(
        uuid.UUID(payload["project_id"]),
        uuid.UUID(payload["version_id"]),
        int(payload.get("iteration", 1)),
    )
    mesh_path = storage.resolve(payload["mesh_rel"])
    d = run_fem_pipeline(
        mesh_path=mesh_path,
        out_dir=out_dir,
        load_cases=payload.get("load_cases") or [],
        material=payload.get("material") or "",
        voxel_mm=payload.get("voxel_mm"),
        max_elements=payload.get("max_elements"),
        keep_regions=payload.get("keep_regions"),
        anchors=payload.get("anchors"),
        preview_only=bool(payload.get("preview_only")),
        runtime=_runtime(payload),
    )
    preview = d.get("preview") or {}
    if preview.get("regions_png"):
        it_rel = storage.relpath(out_dir)
        preview["regions_png"] = [f"{it_rel}/fem/{n}" for n in preview["regions_png"]]
    return d


async def fem_analyze(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): FEM verification of a stored mesh artifact."""
    if await _cancelled(ctx, f"cancel:{payload.get('job_id')}" if payload.get("job_id") else None):
        return {"ok": False, "error": "cancelled"}
    d = run_fem_payload(payload)
    fem = d.get("fem") or {}
    log.info(
        "fem_analyze",
        version=payload.get("version_id"),
        ok=d.get("ok"),
        worst_sf=(fem or {}).get("cases", [{}])[0].get("safety_factor") if fem else None,
    )
    return d


def _sync_cancel_checker(cancel_key: str | None) -> Callable[[], bool] | None:
    """Polling checker usable inside synchronous SIMP/FEM loops.

    The arq redis client is async and can't be awaited mid-solve, so we open
    a separate sync connection for the cancel key the API sets.
    """
    if not cancel_key:
        return None
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        import redis as sync_redis

        client = sync_redis.Redis.from_url(url)
    except Exception:
        return None

    def check() -> bool:
        try:
            return bool(client.exists(cancel_key))
        except Exception:
            return False

    return check


def run_optimize_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of optimize — shared with InProcessExecutor."""
    out_dir = storage.iteration_dir(
        uuid.UUID(payload["project_id"]),
        uuid.UUID(payload["version_id"]),
        int(payload.get("iteration", 1)),
    )
    mesh_path = storage.resolve(payload["mesh_rel"])
    result = run_topopt_pipeline(
        mesh_path=mesh_path,
        out_dir=out_dir,
        load_cases=payload.get("load_cases") or [],
        optimize=payload.get("optimize"),
        material=payload.get("material") or "",
        voxel_mm=payload.get("voxel_mm"),
        max_elements=payload.get("max_elements"),
        anchors=payload.get("anchors"),
        name=payload.get("name", "model"),
        runtime=_runtime(payload),
        cancelled=cancelled,
    )
    d = result.to_dict()
    d = _relpath_exports(d, result, out_dir)
    it_rel = storage.relpath(out_dir)
    d.setdefault("topopt", {})
    d["topopt"]["density_relpath"] = f"{it_rel}/topopt/density.npz"
    d["topopt"]["history_relpath"] = f"{it_rel}/topopt/history.json"
    d["topopt"]["regions_png"] = [f"{it_rel}/fem/{n}" for n in result.topopt.get("regions_png", [])]
    d["fem_report_relpath"] = f"{it_rel}/fem/report.json" if result.fem else None
    d["fem_report_export_relpath"] = (
        f"{it_rel}/fem/report_export.json" if result.fem_export else None
    )
    d.setdefault("mesh", {})["stl_relpath"] = f"{it_rel}/_run/model.stl"
    log.info(
        "optimize",
        version=payload.get("version_id"),
        ok=d["ok"],
        iters=result.topopt.get("iterations"),
        timings=d["timings"],
    )
    return d


async def optimize(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): SIMP topology optimization + FEM verification."""
    cancel_key = f"cancel:{payload['job_id']}" if payload.get("job_id") else None
    if await _cancelled(ctx, cancel_key):
        r = ExecResult()
        r.error = "cancelled"
        return r.to_dict()
    d = run_optimize_payload(payload, cancelled=_sync_cancel_checker(cancel_key))
    if await _cancelled(ctx, cancel_key):
        d["ok"] = False
        d["error"] = "cancelled"
    return d
