"""arq tasks running inside the cad-executor (queue name "cad")."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from cadagent import storage
from cadagent.cad.engine import (
    ExecResult,
    export_from_brep,
    export_from_mesh,
    run_mesh_candidates,
    run_mesh_pipeline,
    run_pipeline,
)
from cadagent.cad.mesh_ops import MeshPostParams
from cadagent.cad.sandbox import Reference
from cadagent.db.models import ProjectSettings
from cadagent.llm.schemas import PartSpec
from cadagent.logging import get_logger

log = get_logger(__name__)


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
        full_exports=bool(payload.get("full_exports")),
        name=payload.get("name", "model"),
        mesh_solid=Path(payload["mesh_solid"]) if payload.get("mesh_solid") else None,
        mesh_bbox=payload.get("mesh_bbox"),
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
        exports = export_from_mesh(it_dir / "_run" / "model.stl", out_dir, name=name)
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

    settings = ProjectSettings(**(payload.get("settings") or {}))
    spec = PartSpec(**payload["spec"]) if payload.get("spec") else None
    post = MeshPostParams(**(payload.get("post") or {}))

    result = run_mesh_pipeline(
        mesh_path=mesh_path,
        out_dir=out_dir,
        post=post,
        spec=spec,
        settings=settings,
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
