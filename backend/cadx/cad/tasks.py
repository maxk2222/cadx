"""arq tasks running inside the cad-executor (queue name "cad")."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import pickle
import resource
import shutil
import signal
import subprocess
import sys
import time
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


def _cancel_keys(payload: dict[str, Any]) -> list[str]:
    """Cancel keys a cad task must honour: the JOB-level flag (user pressed
    cancel — kills every task of the job) plus the TASK-level flag the
    worker sets when THIS task ran past its wait deadline. The two must
    stay distinct — a task timeout setting the job key cancelled every
    later task of that job for an hour (observed live: one hung execute
    turned all following iterations into instant 'cancelled' results)."""
    keys: list[str] = []
    if payload.get("job_id"):
        keys.append(f"cancel:{payload['job_id']}")
    if payload.get("_task_id"):
        keys.append(f"cancel:{payload['_task_id']}")
    return keys


async def _cancelled(ctx: dict[str, Any], key: str | list[str] | None) -> bool:
    keys = [key] if isinstance(key, str) else (key or [])
    if not keys:
        return False
    try:
        return bool(await ctx["redis"].exists(*keys))
    except Exception:
        return False


def _cancel_kwargs(fn: Callable[..., Any], cancelled: Callable[[], bool] | None) -> dict[str, Any]:
    """Kwargs bridge for the engine ``cancelled=`` contract: forwards the
    checker only to entry points that already accept it — the kwarg is being
    added across engine.py, and a blind pass would TypeError meanwhile.
    Once every engine function carries the kwarg this can go away."""
    if cancelled is None:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {"cancelled": cancelled} if "cancelled" in params else {}


def _artifact_dir(payload: dict[str, Any]) -> Path:
    """Iteration dir, namespaced by attempt — several pipeline runs share one
    iteration number (repair verifies, autofix, mesh refine rounds) and would
    silently overwrite each other's artifacts. The winning candidate's score
    must match the files on disk, so every write gets its own subdir."""
    it_dir = storage.iteration_dir(
        uuid.UUID(payload["project_id"]),
        uuid.UUID(payload["version_id"]),
        int(payload.get("iteration", 1)),
    )
    sub = str(payload.get("subdir") or "")
    return it_dir / sub if sub else it_dir


def run_execute_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of execute_version — shared with InProcessExecutor."""
    version_id = uuid.UUID(payload["version_id"])

    out_dir = _artifact_dir(payload)
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
        light=bool(payload.get("light")),
        **_cancel_kwargs(run_pipeline, cancelled),
    )

    d = result.to_dict()
    it_rel = storage.relpath(out_dir)
    d["art_dir"] = it_rel
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
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        r = ExecResult()
        r.error = "cancelled"
        return r.to_dict()

    # CPU-blocking pipeline — must run in a thread so the arq event loop
    # keeps dequeuing other jobs, health checks and cancel flags. The thread
    # cannot be killed on timeout, but failing the task releases the arq job
    # slot — an unbounded hang otherwise wedges the worker permanently
    # (observed: a stuck job starved the loop so jobs queued 480 s unclaimed).
    deadline = _runtime(payload).exec_timeout_s * 2 + 120
    try:
        d = await asyncio.wait_for(
            asyncio.to_thread(
                run_execute_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
            ),
            deadline,
        )
    except TimeoutError:
        from cadx.cad.engine import pipeline_step

        hb = pipeline_step(str(_artifact_dir(payload)))
        log.error(
            "execute_version.pipeline_hang",
            version=payload["version_id"],
            step=hb[0] if hb else None,
            step_s=round(time.monotonic() - hb[1], 1) if hb else None,
        )
        r = ExecResult()
        r.error = f"executor pipeline exceeded {deadline}s — worker thread may be stuck"
        if hb:
            r.error += f" (last step: {hb[0]})"
        return r.to_dict()

    if await _cancelled(ctx, cancel_keys):
        d["ok"] = False
        d["error"] = "cancelled"
    return d


def run_audit_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of audit_params — shared with InProcessExecutor.

    Perturbs each numeric PARAMS value ±epsilon and re-runs the SAME code in
    light pipeline mode (sandbox + analysis only). Runs land in the version's
    scratch dir and are removed when the audit finishes — the report is the
    only artifact."""
    from cadx.cad.design_audit import audit_params, make_pipeline_runner

    project_id = uuid.UUID(payload["project_id"])
    version_id = uuid.UUID(payload["version_id"])
    settings = ProjectSettings(**(payload.get("settings") or {}))
    references = [
        Reference(asset_id=r["asset_id"], path=r["path"]) for r in payload.get("references", [])
    ]

    scratch = storage.ensure(storage.scratch_dir(project_id, version_id) / "audit")
    runner = make_pipeline_runner(
        payload["code"],
        scratch,
        references=references,
        settings=settings,
        runtime=_runtime(payload),
        mesh_solid=Path(payload["mesh_solid"]) if payload.get("mesh_solid") else None,
        mesh_bbox=payload.get("mesh_bbox"),
        anchors=payload.get("anchors"),
        cancelled=cancelled,
    )
    try:
        report = audit_params(
            payload["code"],
            payload.get("params") or {},
            epsilon=float(payload.get("epsilon") or 0.05),
            max_params=int(payload.get("max_params") or 8),
            runner=runner,
            **_cancel_kwargs(audit_params, cancelled),
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    report["ok"] = True
    log.info(
        "audit_params",
        version=str(version_id),
        status=report.get("status"),
        audited=report.get("audited"),
    )
    return report


async def audit_params(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): design-robustness audit of a version's PARAMS."""
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        return {"ok": False, "status": "fails", "error": "cancelled"}

    # one baseline + up to 2 runs per audited param, each bounded by the
    # sandbox timeout inside the thread — the wait_for deadline releases the
    # arq job slot if a run wedges anyway (same pattern as execute_version)
    runs = 2 * int(payload.get("max_params") or 8) + 1
    deadline = _runtime(payload).exec_timeout_s * runs + 120
    try:
        d = await asyncio.wait_for(
            asyncio.to_thread(
                run_audit_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
            ),
            deadline,
        )
    except TimeoutError:
        log.error("audit_params.pipeline_hang", version=payload.get("version_id"))
        return {
            "ok": False,
            "status": "fails",
            "error": f"audit pipeline exceeded {deadline}s — worker thread may be stuck",
        }

    if await _cancelled(ctx, cancel_keys):
        d["ok"] = False
        d["error"] = "cancelled"
    return d


async def export_all(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Export all formats from a stored iteration BREP into the version exports dir."""
    project_id = uuid.UUID(payload["project_id"])
    version_id = uuid.UUID(payload["version_id"])
    iteration = int(payload["iteration"])
    name = payload.get("name", "model")

    # explicit artifact dir wins — a candidate's exports must come from the
    # same attempt dir its score was computed on (refine rounds overwrite
    # the shared iteration dir otherwise)
    art_dir = str(payload.get("art_dir") or "")
    it_dir = (
        storage.resolve(art_dir)
        if art_dir
        else storage.iteration_dir(project_id, version_id, iteration)
    )
    brep = it_dir / "_run" / "model.brep"
    flat = it_dir / "_run" / "flat_profile.brep"
    out_dir = storage.export_dir(project_id, version_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if brep.exists():
        exports = await asyncio.to_thread(
            export_from_brep, brep, out_dir, name=name, flat_brep=flat if flat.exists() else None
        )
    elif (it_dir / "_run" / "model.stl").exists():
        # mesh-mode version: export from the stored STL
        exports = await asyncio.to_thread(
            export_from_mesh,
            it_dir / "_run" / "model.stl",
            out_dir,
            name=name,
            max_faces=_runtime(payload).mesh_max_faces_for_cad,
        )
    else:
        return {"ok": False, "error": "no model.brep or model.stl found for this iteration"}
    rel = storage.relpath(out_dir)
    return {"ok": True, "exports": {k: (f"{rel}/{v}" if v else None) for k, v in exports.items()}}


def run_mesh_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of process_mesh — shared with InProcessExecutor.

    mode="candidates": render orientation candidates (returns sheet + per-up pngs).
    mode="final":      full mesh pipeline (clean/orient/scale/analysis/renders/dfm/exports).
    """
    if cancelled is None:
        # the queue path runs this in a subprocess (mesh_payload_entry) — a
        # callable can't cross the pickle boundary, so rebuild the redis
        # checker from the payload's job_id in whichever process runs us
        cancelled = _sync_cancel_checker(_cancel_keys(payload))
    project_id = uuid.UUID(payload["project_id"])
    version_id = uuid.UUID(payload["version_id"])
    mode = payload.get("mode", "final")

    out_dir = _artifact_dir(payload)
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
        views = payload.get("views")
        d = run_mesh_preview(
            mesh_path=mesh_path,
            out_dir=out_dir,
            views=tuple(views) if views else ("iso", "front", "top"),
        )
        it_rel = storage.relpath(out_dir)
        d["images"] = {k: f"{it_rel}/preview/{v}" for k, v in d.get("images", {}).items()}
        return d

    if mode == "ops_preview":
        return _ops_preview(payload, mesh_path, project_id, version_id)

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
        skip_baseline=bool(payload.get("ops_only")),
        anchors=payload.get("anchors"),
        solid_max_faces=payload.get("solid_max_faces"),
        ref_mask_rel=payload.get("ref_mask_rel"),
        **_cancel_kwargs(run_mesh_pipeline, cancelled),
    )

    d = result.to_dict()
    it_rel = storage.relpath(out_dir)
    d["art_dir"] = it_rel
    d["renders"] = {v: f"{it_rel}/renders/{p}" for v, p in result.renders.items()}
    if result.contact_sheet:
        d["contact_sheet"] = f"{it_rel}/contact_sheet.png"
    if result.thumbnail:
        d["thumbnail"] = f"{it_rel}/thumbnail.png"
    d["exports"] = {k: (f"{it_rel}/exports/{v}" if v else None) for k, v in result.exports.items()}
    if (out_dir / "_run" / "model.brep").exists():
        d["mesh"]["solid_relpath"] = f"{it_rel}/_run/model.brep"
    d["mesh"]["stl_relpath"] = f"{it_rel}/_run/model.stl"
    # echo the executor's op-semantics version so the worker can detect a
    # worker/executor image drift even when no ops were replayed
    from cadx.cad.mesh_ops import MESH_OPS_VERSION

    d["mesh"]["mesh_ops_version"] = MESH_OPS_VERSION
    log.info("process_mesh", version=str(version_id), ok=d["ok"], timings=d["timings"])
    return d


def _preview_source_cache(
    sdir: Path,
    payload: dict[str, Any],
    mesh_path: Path,
    post: MeshPostParams,
    spec: PartSpec | None,
) -> Path | None:
    """Cache path for the processed preview source (baseline or cleaned mesh).

    Keyed by source file content + the inputs that change the result — a hit
    skips the load/clean/decimate/orient/scale baseline on repeat previews."""
    try:
        st = mesh_path.stat()
        key = hashlib.sha1(
            json.dumps(
                {
                    "rel": payload.get("mesh_rel"),
                    "mtime": st.st_mtime_ns,
                    "size": st.st_size,
                    "post": post.model_dump(mode="json", exclude={"ops"}),
                    "spec": spec.model_dump(mode="json") if spec else None,
                    "ops_only": bool(payload.get("ops_only")),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        return sdir / f"preview_base_{key}.glb"
    except Exception:
        return None


def _ops_preview(
    payload: dict[str, Any], mesh_path: Path, project_id: uuid.UUID, version_id: uuid.UUID
) -> dict[str, Any]:
    """GUI mesh-editor preview: baseline (mesh parents) or ops-only (others),
    then apply the draft op list and export a GLB into _scratch/."""
    from cadx.cad.mesh_ops import (
        MeshOpError,
        apply_ops,
        baseline_mesh,
        clean,
        load_mesh,
        rest_on_xy_center,
    )

    sdir = storage.ensure(storage.scratch_dir(project_id, version_id))
    post = MeshPostParams(**(payload.get("post") or {}))
    spec = PartSpec(**payload["spec"]) if payload.get("spec") else None
    cache = _preview_source_cache(sdir, payload, mesh_path, post, spec)

    try:
        mesh = None
        if cache is not None and cache.exists():
            with contextlib.suppress(Exception):
                mesh = load_mesh(cache)
            if mesh is None:
                cache.unlink(missing_ok=True)  # corrupt cache — recompute
        if mesh is None:
            try:
                mesh = load_mesh(mesh_path)
            except Exception as e:
                return {"ok": False, "error": f"failed to load mesh: {e}"}
            if payload.get("ops_only"):
                mesh = clean(mesh)
            else:
                mesh, _ = baseline_mesh(
                    mesh,
                    post,
                    spec,
                    cad_cap_faces=_runtime(payload).mesh_decimate_faces,
                )
            if cache is not None:
                with contextlib.suppress(Exception):
                    mesh.export(str(cache))
        ops = payload.get("ops") or post.ops
        if ops:
            mesh, report = apply_ops(
                mesh, ops, anchors=payload.get("anchors"), runtime=_runtime(payload)
            )
        else:
            report = []
        mesh = rest_on_xy_center(mesh)
    except MeshOpError as e:
        return {
            "ok": False,
            "error": str(e)[:400],
            "failed_op": e.op_index,
            "report": e.report,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:400]}

    out = sdir / f"preview_{uuid.uuid4().hex[:8]}.glb"
    try:
        mesh.export(str(out))
    except Exception as e:
        return {"ok": False, "error": f"preview export failed: {e}"}
    b = mesh.bounds
    return {
        "ok": True,
        "glb": storage.relpath(out),
        "stats": {
            "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight),
            "bbox_size": [round(float(v), 3) for v in (b[1] - b[0])],
            "ops_applied": len(report),
            "report": report,
        },
    }


def _mesh_child_limits(mem_mb: int) -> None:
    """preexec_fn for the mesh payload child: address-space cap first so an OOM
    raises MemoryError in the child instead of tripping the container cgroup
    (which SIGKILLs the worker too). VTK/trimesh imports need real NOFILE room."""
    for res, lim in (
        (resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024)),
        (resource.RLIMIT_NOFILE, (1024, 1024)),
    ):
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(res, lim)
    os.setsid()


def _run_mesh_payload_subprocess(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Run run_mesh_payload in a child process (see mesh_payload_entry.py for the
    why). Timeout mirrors the client budget; a missing result.pkl means the child
    died (OOM/segfault) which we surface as a normal error result."""
    rt = _runtime(payload)
    timeout_s = rt.exec_timeout_s + rt.mesh_timeout_s
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "input.pkl").write_bytes(pickle.dumps(payload))

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    # inherit stdout/stderr so the child's structlog lines keep landing in the
    # executor's docker logs
    proc = subprocess.Popen(
        [sys.executable, "-m", "cadx.cad.mesh_payload_entry", str(workdir)],
        cwd=str(workdir),
        env=env,
        preexec_fn=lambda: _mesh_child_limits(rt.exec_mem_limit_mb),
    )
    try:
        proc.wait(timeout=timeout_s + 15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return {"ok": False, "error": f"mesh processing timed out after {timeout_s + 15}s (killed)"}

    result_path = workdir / "result.pkl"
    if result_path.exists():
        try:
            d = pickle.loads(result_path.read_bytes())
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    rc = proc.returncode
    detail = "memory limit" if rc in (-signal.SIGKILL, -signal.SIGSEGV) else f"exit {rc}"
    return {"ok": False, "error": f"mesh processing subprocess died ({detail})"}


async def process_mesh(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): mesh post-processing + optional solid conversion."""
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        r = ExecResult()
        r.error = "cancelled"
        return r.to_dict()

    workdir = _artifact_dir(payload) / "_meshjob"
    # the pipeline runs in a child process — the cancel callable can't cross
    # the pickle boundary, so run_mesh_payload rebuilds it from payload's
    # job_id inside the child instead
    d = await asyncio.to_thread(_run_mesh_payload_subprocess, payload, workdir)

    if await _cancelled(ctx, cancel_keys):
        d["ok"] = False
        d["error"] = "cancelled"
    return d


def run_parametrize_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of parametrize_mesh — shared with InProcessExecutor.

    Reproduces the final mesh (the stored STL the winning process_mesh run
    wrote when ``stl_rel`` is passed, otherwise the same baseline_mesh +
    apply_ops replay run_mesh_pipeline performs), asks cad/reconstruct for a
    parametric rebuild, executes the emitted code through the standard
    sandbox pipeline and measures the chamfer distance between source mesh
    and reconstruction.
    """
    version_id = uuid.UUID(payload["version_id"])
    rt = _runtime(payload)
    settings = ProjectSettings(**(payload.get("settings") or {}))
    spec = PartSpec(**payload["spec"]) if payload.get("spec") else None

    from cadx.cad.mesh_ops import (
        apply_ops,
        baseline_mesh,
        clean,
        rest_on_xy_center,
    )
    from cadx.cad.mesh_ops import (
        load_mesh as load_triangle_mesh,
    )

    # 1. final mesh — prefer the STL the winning process_mesh run already
    # wrote (the exact artifact the candidate was scored on); fall back to
    # replaying baseline + recorded ops on the raw generator output
    try:
        stl_rel = str(payload.get("stl_rel") or "")
        if stl_rel and storage.resolve(stl_rel).exists():
            mesh = load_triangle_mesh(storage.resolve(stl_rel))
        else:
            raw = load_triangle_mesh(storage.resolve(payload["mesh_rel"]))
            post = MeshPostParams(**(payload.get("post") or {}))
            if payload.get("ops_only"):
                mesh = clean(raw)
            else:
                mesh, _base = baseline_mesh(raw, post, spec, cad_cap_faces=rt.mesh_decimate_faces)
            ops = payload.get("ops") or post.ops
            if ops:
                mesh, _report = apply_ops(mesh, ops, anchors=payload.get("anchors"), runtime=rt)
            mesh = rest_on_xy_center(mesh)
    except Exception as e:
        return {"ok": False, "error": f"failed to load final mesh: {e}"}

    # 2. mesh -> parametric build123d code (None = shape is not
    # reconstructible — organic blobs land here, a normal outcome)
    try:
        from cadx.cad.reconstruct import chamfer_mm, reconstruct_mesh

        recon = reconstruct_mesh(mesh, spec, dict(payload.get("analysis") or {}))
    except Exception as e:
        log.warning("parametrize.reconstruct_failed", error=str(e)[:200])
        return {"ok": False, "reason": "not reconstructible", "error": str(e)[:300]}
    if recon is None:
        return {"ok": False, "reason": "not reconstructible"}

    # 3. execute + analyse the reconstruction through the standard pipeline —
    # the exact sandbox/analysis environment run_execute_payload uses
    out_dir = _artifact_dir(payload)
    result = run_pipeline(
        code=recon.code,
        out_dir=out_dir,
        settings=settings,
        runtime=rt,
        full_exports=False,
        name=payload.get("name", "model"),
        anchors=payload.get("anchors"),
        spec=payload.get("spec"),
        light=bool(payload.get("light")),
        **_cancel_kwargs(run_pipeline, cancelled),
    )
    exec_result = {
        "ok": result.ok,
        "error": result.error,
        "analysis": result.analysis.model_dump() if result.analysis else None,
        "features": result.features,
        "dfm": result.dfm.model_dump() if result.dfm else None,
        "params_used": result.params_used,
    }
    if not result.ok:
        return {
            "ok": False,
            "reason": "reconstruction did not execute",
            "error": result.error,
            "kind": recon.kind,
            "code": recon.code,
            "exec_result": exec_result,
        }

    # 4. shape agreement: chamfer distance source mesh <-> reconstruction
    try:
        recon_mesh = load_triangle_mesh(out_dir / "renders" / "mesh.stl")
        dist = float(chamfer_mm(mesh, recon_mesh))
    except Exception as e:
        return {
            "ok": False,
            "reason": "chamfer measurement failed",
            "error": str(e)[:300],
            "kind": recon.kind,
            "code": recon.code,
            "exec_result": exec_result,
        }

    d = {
        "ok": True,
        "kind": recon.kind,
        "code": recon.code,
        "params": dict(recon.params),
        "params_schema": list(recon.params_schema),
        "chamfer_mm": dist,
        "confidence": float(recon.confidence),
        "meta": dict(recon.meta),
        "exec_result": exec_result,
        "art_dir": storage.relpath(out_dir),
    }
    log.info(
        "parametrize_mesh",
        version=str(version_id),
        kind=recon.kind,
        chamfer_mm=round(dist, 3),
        confidence=round(float(recon.confidence), 3),
    )
    return d


async def parametrize_mesh(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): best-effort parametric reconstruction of a
    finished mesh — dispatched once per job when the mesh loop finalises."""
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        return {"ok": False, "error": "cancelled"}

    # mesh replay + reconstruction + one full sandbox pipeline — same
    # envelope as execute_version; the wait_for deadline releases the arq
    # job slot when the thread wedges (same pattern as execute_version)
    deadline = _runtime(payload).exec_timeout_s * 2 + 120
    try:
        d = await asyncio.wait_for(
            asyncio.to_thread(
                run_parametrize_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
            ),
            deadline,
        )
    except TimeoutError:
        log.error("parametrize_mesh.pipeline_hang", version=payload.get("version_id"))
        return {
            "ok": False,
            "error": f"parametrize pipeline exceeded {deadline}s — worker thread may be stuck",
        }

    if await _cancelled(ctx, cancel_keys):
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


def run_fem_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of fem_analyze — shared with InProcessExecutor."""
    if payload.get("scratch"):
        # GUI region preview — write renders into the transient _scratch dir
        out_dir = storage.scratch_dir(
            uuid.UUID(payload["project_id"]), uuid.UUID(payload["version_id"])
        )
    else:
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
        render_pngs=bool(payload.get("render_pngs", True)),
        scan_anchors=bool(payload.get("scan_anchors", True)),
        scan_only=bool(payload.get("scan_only")),
        runtime=_runtime(payload),
        **_cancel_kwargs(run_fem_pipeline, cancelled),
    )
    preview = d.get("preview") or {}
    if preview.get("regions_png"):
        it_rel = storage.relpath(out_dir)
        preview["regions_png"] = [f"{it_rel}/fem/{n}" for n in preview["regions_png"]]
    return d


async def fem_analyze(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): FEM verification of a stored mesh artifact."""
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        return {"ok": False, "error": "cancelled"}
    d = await asyncio.to_thread(
        run_fem_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
    )
    fem = d.get("fem") or {}
    log.info(
        "fem_analyze",
        version=payload.get("version_id"),
        ok=d.get("ok"),
        worst_sf=(fem or {}).get("cases", [{}])[0].get("safety_factor") if fem else None,
    )
    return d


def _sync_cancel_checker(
    cancel_keys: str | list[str] | None,
) -> Callable[[], bool] | None:
    """Polling checker usable inside synchronous SIMP/FEM loops.

    The arq redis client is async and can't be awaited mid-solve, so we open
    a separate sync connection for the cancel keys (job + task level).
    """
    keys = [cancel_keys] if isinstance(cancel_keys, str) else (cancel_keys or [])
    if not keys:
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
            return bool(client.exists(*keys))
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
        scan_anchors=bool(payload.get("scan_anchors", True)),
        spec=payload.get("spec"),
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
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        r = ExecResult()
        r.error = "cancelled"
        return r.to_dict()
    d = await asyncio.to_thread(
        run_optimize_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
    )
    if await _cancelled(ctx, cancel_keys):
        d["ok"] = False
        d["error"] = "cancelled"
    return d


def run_topopt_resample_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of topopt_resample — re-extract a stored density
    field at a new threshold without re-running SIMP."""
    from cadx.cad.engine import run_topopt_resample

    out_dir = storage.iteration_dir(
        uuid.UUID(payload["project_id"]),
        uuid.UUID(payload["version_id"]),
        int(payload.get("iteration", 1)),
    )
    result = run_topopt_resample(
        density_relpath=payload["density_relpath"],
        out_dir=out_dir,
        load_cases=payload.get("load_cases") or [],
        optimize=payload.get("optimize"),
        material=payload.get("material") or "",
        max_elements=payload.get("max_elements"),
        anchors=payload.get("anchors"),
        name=payload.get("name", "model"),
        runtime=_runtime(payload),
        spec=payload.get("spec"),
        **_cancel_kwargs(run_topopt_resample, cancelled),
    )
    d = result.to_dict()
    d = _relpath_exports(d, result, out_dir)
    d.setdefault("mesh", {})["stl_relpath"] = f"{storage.relpath(out_dir)}/_run/model.stl"
    log.info(
        "topopt_resample",
        version=payload.get("version_id"),
        ok=d["ok"],
        threshold=(payload.get("optimize") or {}).get("density_threshold"),
    )
    return d


async def topopt_resample(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): threshold resample of a stored density field."""
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        return {"ok": False, "error": "cancelled"}
    return await asyncio.to_thread(
        run_topopt_resample_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
    )


def run_refine_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of refine — post-TopOpt feature finishing (exact
    bores/threads/inserts, keep-region surface restore)."""
    from cadx.cad.engine import run_refine

    out_dir = storage.iteration_dir(
        uuid.UUID(payload["project_id"]),
        uuid.UUID(payload["version_id"]),
        int(payload.get("iteration", 1)),
    )
    result = run_refine(
        mesh_path=storage.resolve(payload["mesh_rel"]),
        out_dir=out_dir,
        features=payload.get("features"),
        spec_features=payload.get("spec_features"),
        anchors=payload.get("anchors"),
        keep_regions=payload.get("keep_regions"),
        load_cases=payload.get("load_cases"),
        base_mesh_path=(
            storage.resolve(payload["base_mesh_rel"]) if payload.get("base_mesh_rel") else None
        ),
        restore_planes=bool(payload.get("restore_planes", True)),
        helical_threads=bool(payload.get("helical_threads", False)),
        auto_plane_restore=bool(payload.get("auto_plane_restore", False)),
        auto_keep_anchors=bool(payload.get("auto_keep_anchors", False)),
        material=payload.get("material") or "",
        voxel_mm=payload.get("voxel_mm"),
        max_elements=payload.get("max_elements"),
        name=payload.get("name", "model"),
        runtime=_runtime(payload),
        spec=payload.get("spec"),
        **_cancel_kwargs(run_refine, cancelled),
    )
    d = result.to_dict()
    d = _relpath_exports(d, result, out_dir)
    d.setdefault("mesh", {})["stl_relpath"] = f"{storage.relpath(out_dir)}/_run/model.stl"
    log.info(
        "refine",
        version=payload.get("version_id"),
        ok=d["ok"],
        applied=len((result.features.get("refine") or {}).get("applied") or []),
    )
    return d


async def refine(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): post-TopOpt feature finishing."""
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        return {"ok": False, "error": "cancelled"}
    return await asyncio.to_thread(
        run_refine_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
    )


def run_organic_payload(
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Synchronous part of organic — procedural pseudo-topology restyle."""
    from cadx.cad.engine import run_organic_pipeline

    out_dir = storage.iteration_dir(
        uuid.UUID(payload["project_id"]),
        uuid.UUID(payload["version_id"]),
        int(payload.get("iteration", 1)),
    )
    result = run_organic_pipeline(
        mesh_path=storage.resolve(payload["mesh_rel"]),
        out_dir=out_dir,
        organic=payload.get("organic"),
        voxel_mm=payload.get("voxel_mm"),
        max_elements=payload.get("max_elements"),
        anchors=payload.get("anchors"),
        name=payload.get("name", "model"),
        runtime=_runtime(payload),
        **_cancel_kwargs(run_organic_pipeline, cancelled),
    )
    d = result.to_dict()
    d = _relpath_exports(d, result, out_dir)
    it_rel = storage.relpath(out_dir)
    d.setdefault("mesh", {})["stl_relpath"] = f"{it_rel}/_run/model.stl"
    if (out_dir / "organic" / "density.npz").exists():
        d["mesh"].setdefault("organic", {})["density_relpath"] = f"{it_rel}/organic/density.npz"
    log.info(
        "organic",
        version=payload.get("version_id"),
        ok=d["ok"],
        style=(payload.get("organic") or {}).get("style"),
        timings=d["timings"],
    )
    return d


async def organic(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): procedural organic restyle — no FEM/SIMP."""
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        return {"ok": False, "error": "cancelled"}
    return await asyncio.to_thread(
        run_organic_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
    )


def run_edit_feature_payload(  # noqa: C901
    payload: dict[str, Any], cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Verified parametric feature edit (M9) — synchronous executor body.

    Loads the PARENT's canonical B-Rep (``source_brep`` relpath), runs
    quiddity recognition + the transactional gate in ``cad/editfeature.py``
    and — only when the gate passes — writes the edited solid plus the
    standard artifact set (brep/step/stl, renders, contact sheet, exports)
    under the CHILD version's artifact dir. A refusal returns the gate report
    without writing a byte.
    """
    source_rel = str(payload.get("source_brep") or "")
    if not source_rel:
        return {"ok": False, "error": "no source model supplied"}
    src = storage.resolve(source_rel)
    if not src.exists():
        return {"ok": False, "error": f"source model missing: {source_rel}"}

    from cadx.cad.editfeature import edit_feature, find_editable_features

    settings = ProjectSettings.model_validate(payload.get("settings") or {})

    try:
        if src.suffix.lower() in {".step", ".stp"}:
            from build123d import import_step

            shape = import_step(str(src))
        else:
            from build123d import import_brep

            shape = import_brep(str(src))
        solids = shape.solids()
        base = solids[0] if len(solids) == 1 else shape
    except Exception as exc:
        return {"ok": False, "error": f"import: {type(exc).__name__}: {exc}"}
    if base is None:
        return {"ok": False, "error": "source model contains no solid"}

    features = find_editable_features(base)
    feature_id = payload.get("feature_id")
    if feature_id is None:
        # discovery mode — the agent/UI lists candidates, nothing is edited
        return {"ok": True, "features": features}

    started = time.time()
    res = edit_feature(
        base,
        feature_id,
        str(payload.get("parameter") or "diameter"),
        payload.get("value"),
        **_cancel_kwargs(edit_feature, cancelled),
    )
    report = res.get("report") or {}
    if not res.get("ok"):
        return {
            "ok": False,
            "error": res.get("refusal") or "edit refused",
            "report": report,
            "features": features,
        }
    edited = res["shape"]
    if payload.get("dry_run"):
        return {
            "ok": True,
            "dry_run": True,
            "report": report,
            "features": features,
            "elapsed_s": round(time.time() - started, 3),
        }

    art_dir = _artifact_dir(payload)
    run_dir = art_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    writes: dict[Path, float] = {}

    try:
        from build123d import export_brep, export_step, export_stl

        model_brep = run_dir / "model.brep"
        model_step = run_dir / "model.step"
        model_stl = run_dir / "model.stl"
        export_brep(edited, str(model_brep))
        export_step(edited, str(model_step))
        export_stl(edited, str(model_stl))
        writes[model_brep] = writes[model_step] = writes[model_stl] = time.time()
    except Exception as exc:
        return {
            "ok": False,
            "error": f"write model: {type(exc).__name__}: {exc}",
            "report": report,
            "features": features,
        }

    # analysis first — stats/renders and DfM feed off it (same checks as the
    # param pipeline's post-verify steps)
    analysis: dict[str, Any] = {}
    dfm: dict[str, Any] = {}
    try:
        from cadx.cad.analysis import analyze
        from cadx.cad.dfm import DfmReport
        from cadx.cad.dfm.cnc import cnc_checks
        from cadx.cad.dfm.common import common_checks, load_mesh
        from cadx.cad.dfm.fdm import fdm_checks

        ana = analyze(edited, model_stl)
        analysis = ana.model_dump()
        mesh = load_mesh(model_stl)
        warnings = common_checks(ana, mesh, settings.build_volume)
        if settings.process in ("fdm", "sla"):
            warnings += fdm_checks(
                edited,
                mesh,
                ana,
                min_wall_mm=settings.min_wall_mm,
                nozzle_mm=settings.nozzle_mm,
                thread_compensation_mm=settings.thread_compensation_mm,
            )
        if settings.process in ("cnc3", "cnc2d"):
            warnings += cnc_checks(edited, mesh, ana, tool_radius_mm=settings.tool_radius_mm)
        dfm = DfmReport(process=settings.process, warnings=warnings).sort().model_dump()
    except Exception as exc:
        log.warning(
            "edit_feature: analysis/dfm failed",
            error=str(exc),
            job_id=job_id_or_none(payload),
        )

    # renders + contact sheet + thumbnail (best-effort, as in run_pipeline)
    renders_rel: dict[str, str] = {}
    try:
        from cadx.cad.contact_sheet import make_contact_sheet
        from cadx.cad.render import render_available, render_thumbnail, render_views

        if render_available():
            renders_dir = art_dir / "renders"
            bbox = (analysis or {}).get("bbox_size") or [0.0, 0.0, 0.0]
            renders = render_views(model_stl, renders_dir, [float(v) for v in bbox])
            stats = [
                f"bbox: {bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f} mm",
                f"volume: {analysis.get('volume', 0):.0f} mm^3",
                f"solids: {analysis.get('solid_count', 0)}",
                f"valid: {analysis.get('is_valid')}",
                f"feature edit: {report.get('old_diameter')} -> {report.get('new_diameter')} mm",
            ]
            sheet = make_contact_sheet(renders, art_dir / "contact_sheet.png", stats)
            thumb = render_thumbnail(model_stl, art_dir / "thumbnail.png")
            renders_rel = {view: str(p.relative_to(art_dir)) for view, p in renders.items()}
            if sheet:
                renders_rel["contact_sheet"] = str(sheet.relative_to(art_dir))
            if thumb:
                renders_rel["thumbnail"] = str(thumb.relative_to(art_dir))
            stats_path = renders_dir / "stats.txt"
            stats_path.write_text("\n".join(stats))
            writes[stats_path] = time.time()
    except Exception as exc:
        log.warning("edit_feature: render failed", error=str(exc), job_id=job_id_or_none(payload))

    # standard 3D+2D export set, from the committed brep
    exports: dict[str, str | None] = {}
    try:
        exp_dir = art_dir / "exports"
        exp_dir.mkdir(parents=True, exist_ok=True)
        exports = export_from_brep(model_brep, exp_dir, name="model")
    except Exception as exc:
        log.warning("edit_feature: exports failed", error=str(exc), job_id=job_id_or_none(payload))

    log.info(
        "edit_feature committed",
        job_id=job_id_or_none(payload),
        feature_id=feature_id,
        old_diameter=report.get("old_diameter"),
        new_diameter=report.get("new_diameter"),
        elapsed_s=round(time.time() - started, 3),
    )
    art_rel = storage.relpath(art_dir)
    return {
        "ok": True,
        "report": report,
        "features": features,
        # all paths are storage relpaths, matching execute_payload's contract
        "model_brep": f"{art_rel}/_run/model.brep",
        "model_step": f"{art_rel}/_run/model.step",
        "model_stl": f"{art_rel}/_run/model.stl",
        "renders": renders_rel,
        "exports": {k: (f"{art_rel}/exports/{v}" if v else None) for k, v in exports.items()},
        "analysis": analysis,
        "dfm": dfm,
        "elapsed_s": round(time.time() - started, 3),
    }


def job_id_or_none(payload: dict[str, Any]) -> str | None:
    job_id = payload.get("job_id")
    return str(job_id) if job_id else None


async def edit_feature(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq task (queue "cad"): transactional feature edit on a parent B-Rep.

    Discovery when ``feature_id`` is absent, dry-run gate check when
    ``dry_run`` is set — only a full run writes artifacts.
    """
    cancel_keys = _cancel_keys(payload)
    if await _cancelled(ctx, cancel_keys):
        return {"ok": False, "error": "cancelled"}
    return await asyncio.to_thread(
        run_edit_feature_payload, payload, cancelled=_sync_cancel_checker(cancel_keys)
    )
