"""MCP tool catalog: schemas + handlers for the LLM-free CADX capability set.

`MCP_TOOL_SCHEMAS` are `_fn()`-format dicts (same shape as the agent tool
specs) — `cadx.mcpserver.server` exposes them to MCP clients as tools 1:1.
`MCP_HANDLERS` maps tool names to async `(session, args) -> dict` callables;
`McpToolSession.execute` consults this table BEFORE the inherited `_tool_*`
dispatch, so a handler registered here shadows any same-named session tool.
`view_image`/`generate_image` deliberately have handlers but NO schemas —
`ProjectToolSession.tools()` already exposes those (gated by project binding
and image budget); re-declaring them here would duplicate the wire names.

STRICTLY LLM-FREE: no handler may touch `session.ctx.llm` or spawn an agent
(`run_agent_job`) job. The external agent writes build123d code itself; CADX
only verifies (`check_code`, `execute_cad`, `audit_params`), persists
(`submit_code`), runs deterministic jobs (mesh gen/edit, FEM, feature edits)
and serves assets/catalog data.

Project-bound tools resolve the session's bound project via
`session._project(db)` — an unbound session surfaces `{"ok": False, ...}`
(the `SpawnError` message tells the caller to `use_project`/`create_project`
first). `execute_cad`/`audit_params` are the exception: they also run
project-less "scratch" executions under the all-zero project id.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from sqlmodel import col, func, select

from cadx import storage
from cadx.agent.toolspecs import _fn, _i, _s
from cadx.cad.allowlist import check_code
from cadx.cad.mesh_ops import MeshOpError
from cadx.cad.partspec import check_standard_part_calls
from cadx.db.models import (
    IMAGE_ASSET_KINDS,
    Asset,
    AssetKind,
    Job,
    JobStatus,
    JobType,
    Part,
    Version,
    VersionStatus,
)
from cadx.db.session import session_scope
from cadx.logging import get_logger
from cadx.spawn import SpawnError, feature_edit_brep_path, spawn_edit_feature, validate_ops

log = get_logger(__name__)

# scratch executions (no bound project) write under the all-zero project —
# same convention as `cadx exec` and the CLI's default project id
_SCRATCH_PROJECT = "00000000-0000-0000-0000-000000000000"

# MCP uploads accept files too big for the browser drag&drop path (the REST
# endpoint caps at 25MB) but still bounded so a rogue client can't fill the
# data volume in one call
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "agent" / "prompts"

# codegen_contract doc selector -> prompt/reference file. These are the exact
# documents the internal codegen LLM is conditioned on — deterministic source
# material for the external agent writing the code.
_DOC_FILES = {
    "contract": "code_contract.md",
    "build123d": "build123d_reference.md",
    "cadlib": "cadlib_reference.md",
    "bd_warehouse": "bd_warehouse_reference.md",
    "examples": "code_examples.md",
    "dfm": "dfm_rules.md",
}

_TERMINAL_JOB_STATES = (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled)


# --------------------------------------------------------------------------- #
# tool schemas                                                                 #
# --------------------------------------------------------------------------- #

_VER_REF = {"version_id": _s("version id — see list_versions")}
_JOB_REF = {"job_id": _s("job id returned by a spawn tool or list_jobs")}

# spawn tools return immediately by default — the external agent polls
# job_status / wait_job itself (unlike the chat tools which block by default)
_WAIT = {
    "wait": {
        "type": "boolean",
        "description": "block until the spawned job reaches a terminal state "
        "(default false — returns job_id for polling via job_status)",
    }
}

_CODE = {
    "code": _s(
        "complete build123d code file — top-level PARAMS dict, "
        "assigns `result`; see codegen_contract"
    )
}

MCP_TOOL_SCHEMAS: list[dict[str, Any]] = [
    # -- code loop ------------------------------------------------------- #
    _fn(
        "codegen_contract",
        "Read the CADX code contract and reference docs the sandbox enforces: "
        "coordinate frame (X=length, Y=width, Z=height, part on XY centred at "
        "origin, mm), the PARAMS dict + `result` assignment rules, allowed "
        "imports (build123d, bd_warehouse.*, cadlib, math, numpy), the "
        "build123d/cadlib/bd_warehouse API references, worked code examples "
        "and DfM rules. READ THIS FIRST — code that violates the contract "
        "fails check_code/execute_cad.",
        {
            "doc": {
                "type": "string",
                "enum": sorted(_DOC_FILES),
                "description": "which document to return (default 'contract')",
            }
        },
        [],
    ),
    _fn(
        "check_code",
        "Static contract check on a code string — imports, PARAMS dict, "
        "`result` assignment, standard-part call validity. Pure static "
        "analysis, no execution; ALWAYS run this before execute_cad or "
        "submit_code.",
        {
            **_CODE,
            "sigcheck": {
                "type": "boolean",
                "description": "also run the introspected API-signature gate "
                "(unknown attrs / arity mismatches)",
            },
        },
        ["code"],
    ),
    _fn(
        "execute_cad",
        "Run build123d code through the sandboxed executor pipeline and "
        "return the result: error diagnosis on failure; measured geometry "
        "(bbox, volume, watertight, features), DfM report and artifact "
        "relpaths on success. NOTHING is persisted — iterate here, then "
        "submit_code to keep a version. With version_id the code runs "
        "against that version's context (hybrid parents get their mesh "
        "solid); without it the run is a scratch execution.",
        {
            **_CODE,
            "params": {
                "type": "object",
                "description": "PARAMS overrides — keys must exist in the code's PARAMS dict",
            },
            **_VER_REF,
            "light": {
                "type": "boolean",
                "description": "skip renders/DfM — geometry numbers only, much faster",
            },
            "full_exports": {
                "type": "boolean",
                "description": "write the full export set (STEP/STL/3MF/GLB/DXF/SVG)",
            },
        },
        ["code"],
    ),
    _fn(
        "audit_params",
        "Design-robustness pre-flight: perturb each numeric PARAMS value ±ε "
        "and re-run the pipeline per parameter. Reports which parameters are "
        "brittle (fail or change geometry unexpectedly) — run before "
        "submit_code when callers will sweep PARAMS.",
        {
            **_CODE,
            "params": {
                "type": "object",
                "description": "base PARAMS values — defaults to the declared literals",
            },
        },
        ["code"],
    ),
    _fn(
        "submit_code",
        "Persist build123d code as a new version in the bound project and "
        "enqueue its execution job — this is how a model enters the version "
        "tree (renders, exports, DfM, FEM-eligible). Verify first with "
        "check_code/execute_cad; a rejected submission fails the job, not "
        "the tool call.",
        {
            **_CODE,
            "params": {
                "type": "object",
                "description": "PARAMS overrides to bake into the version",
            },
            "parent_version_id": _s("optional parent — creates a child version"),
            "part_id": _s(
                "target part id from list_parts (root submissions only — a "
                "child always inherits its parent's part)"
            ),
            "part_name": _s(
                "name of a NEW part, or exact name of an existing part (root submissions only)"
            ),
            "prompt": _s("version label / provenance note"),
            **_WAIT,
        },
        ["code"],
    ),
    _fn(
        "parametrize_mesh",
        "Best-effort parametric reconstruction of a mesh version: replay the "
        "stored mesh, reconstruct a build123d solid (revolve/extrude/box "
        "primitives), execute it and report the chamfer agreement. Returns "
        "reconstructed code+params on success, a reason string when the "
        "shape is not reconstructible (organic blobs aren't).",
        _VER_REF,
        ["version_id"],
    ),
    # -- mesh ------------------------------------------------------------ #
    _fn(
        "mesh_generate",
        "Image→3D mesh generation on the mesh-server sidecar (deterministic "
        "adapter call, no LLM): creates a mesh version + job from reference "
        "image assets. Upload images first with upload_asset.",
        {
            "asset_ids": {
                "type": "array",
                "items": _s("asset id"),
                "description": "image/generated_image asset ids from upload_asset or list_assets",
            },
            "subject": _s("what the images show — guides adapter selection"),
            "model": _s("adapter id — empty = runtime default (see mesh_health)"),
            "seed": _i("generator seed, default 0"),
            "part_id": _s("target part id from list_parts — default: a new part"),
            "part_name": _s("name of a NEW part, or exact name of an existing part"),
            "post": {
                "type": "object",
                "description": "post-processing overrides (up/front/scale/"
                "flatten_bottom_mm/decimate_faces/…)",
            },
            **_WAIT,
        },
        ["asset_ids"],
    ),
    _fn(
        "mesh_preview",
        "Preview deterministic mesh ops on a version's mesh IN-PROCESS — "
        "returns bbox/faces/watertight + a per-op report without spawning a "
        "job. Ops come from the OPS_REGISTRY (mesh_scale→scale, "
        "mesh_rotate→rotate, flatten_bottom, drill_holes, boolean, …). "
        "Commit with mesh_edit using the same ops list.",
        {
            **_VER_REF,
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "description": "{op: name, args: {...}}",
                },
                "description": "ordered mesh ops to preview",
            },
        },
        ["version_id", "ops"],
    ),
    _fn(
        "editable_features",
        "List recognised hole features on a version's best B-Rep: "
        "feature_id, kind, diameter, axis/opening/depth and whether the hole "
        "is editable (plain through-holes only). Run before edit_feature.",
        _VER_REF,
        ["version_id"],
    ),
    _fn(
        "edit_feature",
        "Verified parametric edit of a recognised feature — resize a plain "
        "through-hole to a new diameter. Spawns a child version; the "
        "executor gate (volume annulus, envelope, sibling holes, "
        "re-recognition) decides success — a refusal fails the child with "
        "the reason.",
        {
            **_VER_REF,
            "feature_id": {
                "type": ["integer", "string"],
                "description": "feature_id from editable_features",
            },
            "params": {
                "type": "object",
                "description": "{diameter: mm} — only 'diameter' is editable",
            },
            **_WAIT,
        },
        ["version_id", "feature_id", "params"],
    ),
    # -- jobs ------------------------------------------------------------ #
    _fn(
        "list_jobs",
        "List the bound project's jobs (newest first): id, type, status, "
        "step, progress, version_id. Poll spawned work here or via "
        "job_status/wait_job.",
        {"limit": _i("max results, default 20")},
        [],
    ),
    _fn(
        "task_status",
        "Status of a cad-queue executor task (the arq task_id inside "
        "execute_cad/parametrize_mesh-style dispatches — NOT a DB job_id): "
        "queued/in_progress/complete plus the result dict when complete.",
        {"task_id": _s("arq task id, e.g. 'execute_version-<uuid>'")},
        ["task_id"],
    ),
    _fn(
        "cancel_job",
        "Request cancellation of a queued/running job in the bound project "
        "(same semantics as the GUI cancel button).",
        _JOB_REF,
        ["job_id"],
    ),
    _fn(
        "finish_job",
        "Ask a RUNNING job to stop iterating early and finalise its best "
        "candidate instead of running the full budget.",
        _JOB_REF,
        ["job_id"],
    ),
    _fn(
        "restart_job",
        "Re-run a finished/failed/cancelled job with the same inputs — "
        "creates a sibling version + job like the GUI restart. Only "
        "deterministic jobs (params/code/meshgen/optimize) — LLM-generated "
        "versions can't be restarted over MCP.",
        {**_JOB_REF, **_WAIT},
        ["job_id"],
    ),
    # -- assets / images -------------------------------------------------- #
    _fn(
        "list_assets",
        "List the bound project's assets (images, generated images, "
        "STEP/STL/DXF references): id, kind, filename, caption, dimensions.",
        {
            "kind": _s(
                "optional filter: image | generated_image | reference_step "
                "| reference_stl | reference_dxf"
            )
        },
        [],
    ),
    _fn(
        "upload_asset",
        "Upload a file into the bound project as an asset — pass EITHER a "
        "server-local path OR base64 data. Images are normalised to JPEG; "
        "STEP/STL/DXF become reference assets loadable via load_reference() "
        "in generated code. Required before mesh_generate / img2img tools.",
        {
            "path": _s("absolute path of a file readable by the CADX host"),
            "data_base64": _s("base64-encoded file bytes (XOR with path)"),
            "kind": {
                "type": "string",
                "enum": [k.value for k in AssetKind],
                "description": "asset kind (default image)",
            },
            "name": _s("filename override — defaults to the path basename"),
            "caption": _s("short description stored on the asset"),
        },
        [],
    ),
    # NOTE: view_image/generate_image are NOT declared here — the inherited
    # session schemas expose them (bound-project + image-budget gated); the
    # handlers below still shadow the impls on MCP sessions.
    _fn(
        "remove_bg",
        "Cut out an image asset's subject via rembg (u2net) — stores the "
        "RGBA result as a new derived asset and returns it as image "
        "content. First call downloads the model (~170MB).",
        {
            "asset_id": _s("image asset id"),
            "wait": {
                "type": "boolean",
                "description": "accepted for symmetry — removal always runs synchronously",
            },
        },
        ["asset_id"],
    ),
    # -- system ----------------------------------------------------------- #
    _fn(
        "settings_get",
        "Read the CADX runtime settings (models, budgets, timeouts, mesh/"
        "image sidecar config). Secrets are never included — llm_api_key "
        "only appears as the boolean llm_api_key_set.",
        {},
        [],
    ),
    _fn(
        "mesh_health",
        "Probe the mesh-server sidecar: ok, device, backend and the "
        "installed mesh/image adapters. Check this before mesh_generate or "
        "generate_image.",
        {},
        [],
    ),
    _fn(
        "image_models",
        "List the installed text-to-image adapters on the mesh-server "
        "sidecar (empty when the sidecar is not configured).",
        {},
        [],
    ),
    _fn(
        "stats",
        "Usage summary: project/version/job/asset totals plus per-status "
        "breakdowns — and project-scoped counts when a project is bound.",
        {},
        [],
    ),
    _fn(
        "use_project",
        "Bind the session to an existing project (alias for attach_project) "
        "— most tools require a bound project; list_projects finds ids.",
        {"project_id": _s("project id from list_projects")},
        ["project_id"],
    ),
]

MCP_TOOL_NAMES = {t["function"]["name"] for t in MCP_TOOL_SCHEMAS}


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _enqueue_fn(session: Any) -> Any:
    """The session's enqueue hook (McpToolSession wires cadx.api.deps.enqueue)."""
    return getattr(session, "_enqueue_fn", None)


def _settings_dump(session: Any) -> dict[str, Any]:
    settings = getattr(session.ctx, "settings", None)
    return settings.model_dump() if hasattr(settings, "model_dump") else {}


def _runtime_values(session: Any) -> dict[str, Any]:
    rt = getattr(session.ctx, "runtime_settings", None)
    return rt.executor_values() if hasattr(rt, "executor_values") else {}


def _session_pid(session: Any) -> uuid.UUID | None:
    return getattr(session, "_pid", None) or getattr(session.ctx, "project_id", None)


def _executor(session: Any) -> dict[str, Any] | Any:
    executor = getattr(session.ctx, "executor", None)
    if executor is None:
        return {"ok": False, "error": "no cad executor configured for this session"}
    return executor


async def _bound_version(session: Any, args: dict[str, Any]) -> Version | dict[str, Any]:
    """Resolve args["version_id"] inside the bound project (SpawnError → dict)."""
    try:
        async with session_scope() as db:
            return await session._version(db, str(args.get("version_id", "")))
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"version lookup failed: {e}"}


def _references(session: Any) -> list[dict[str, str]]:
    """CAD reference bodies visible to the session — mirrors _exec_payload's
    filter (only STEP/STL references are loadable inside the sandbox)."""
    return [
        {"asset_id": a.asset_id, "path": str(a.path)}
        for a in getattr(session.ctx, "assets", None) or []
        if a.kind in ("reference_step", "reference_stl")
    ]


def _exec_payload(
    session: Any,
    code: str,
    args: dict[str, Any],
    version: Version | None,
) -> dict[str, Any]:
    """execute_version task payload — mirrors run_params_job's execute call."""
    pid = version.project_id if version is not None else _session_pid(session)
    vid = version.id if version is not None else uuid.uuid4()
    payload: dict[str, Any] = {
        "job_id": str(uuid.uuid4()),
        "project_id": str(pid) if pid else _SCRATCH_PROJECT,
        "version_id": str(vid),
        "iteration": 1,
        "code": code,
        "params_override": args.get("params") or {},
        "references": _references(session),
        "settings": _settings_dump(session),
        "runtime_settings": _runtime_values(session),
        "full_exports": bool(args.get("full_exports")),
        "light": bool(args.get("light")),
        "name": "model",
    }
    if version is not None:
        # hybrid parents: the mesh-derived solid + its measured bbox ride the
        # payload exactly like run_params_job's re-execution
        solid = storage.version_dir(version.project_id, version.id) / "mesh" / "solid.brep"
        if solid.exists() and (version.resolved_mode or "") == "hybrid":
            payload["mesh_solid"] = str(solid)
            payload["mesh_bbox"] = (version.mesh or {}).get("bbox_size")
    return payload


async def _bound_job(
    session: Any, db: Any, job_id: str, *, version_required: bool = True
) -> Job | dict[str, Any]:
    """Load a Job inside the bound project (SpawnError → error dict)."""
    try:
        jid = uuid.UUID(str(job_id))
    except ValueError:
        return {"ok": False, "error": "invalid job id"}
    await session._project(db)  # resolves/refreshes session._pid
    job = await db.get(Job, jid)
    if job is None:
        return {"ok": False, "error": "job not found"}
    if version_required:
        v = await db.get(Version, job.version_id)
        if v is None or v.project_id != _session_pid(session):
            return {"ok": False, "error": "job not found in this project"}
    return job


# --------------------------------------------------------------------------- #
# handlers — code loop                                                         #
# --------------------------------------------------------------------------- #


async def _h_codegen_contract(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    name = str(args.get("doc") or "contract").strip().lower()
    filename = _DOC_FILES.get(name)
    if filename is None:
        return {
            "ok": False,
            "error": f"unknown doc '{name}' — valid: {', '.join(sorted(_DOC_FILES))}",
        }
    path = _PROMPTS_DIR / filename
    if not path.exists():
        return {"ok": False, "error": f"doc file missing on server: {filename}"}
    return {"ok": True, "doc": name, "file": filename, "text": path.read_text(encoding="utf-8")}


async def _h_check_code(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    code = str(args.get("code") or "")
    if not code.strip():
        return {"ok": False, "error": "code is required"}
    res = check_code(code, sigcheck=bool(args.get("sigcheck")))
    return {
        "ok": res.ok,
        "errors": res.errors,
        "warnings": res.warnings,
        "params": res.params,
        "sig_errors": res.sig_errors,
        "sig_warnings": res.sig_warnings,
        "standard_part_errors": check_standard_part_calls(code),
    }


async def _h_execute_cad(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    code = str(args.get("code") or "")
    if not code.strip():
        return {"ok": False, "error": "code is required"}
    executor = _executor(session)
    if isinstance(executor, dict):
        return executor
    version: Version | None = None
    if args.get("version_id"):
        resolved = await _bound_version(session, args)
        if isinstance(resolved, dict):
            return resolved
        version = resolved
    payload = _exec_payload(session, code, args, version)
    return await executor.execute(payload)


async def _h_audit_params(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    code = str(args.get("code") or "")
    if not code.strip():
        return {"ok": False, "error": "code is required"}
    executor = _executor(session)
    if isinstance(executor, dict):
        return executor
    if getattr(executor, "audit_params", None) is None:
        return {"ok": False, "error": "executor does not support audit_params"}
    # baseline = declared PARAMS merged with the caller's overrides — same
    # convention as run_params_job's design-audit pre-flight
    base_params = {**check_code(code).params, **(args.get("params") or {})}
    pid = _session_pid(session)
    payload = {
        "job_id": str(uuid.uuid4()),
        "project_id": str(pid) if pid else _SCRATCH_PROJECT,
        "version_id": str(uuid.uuid4()),
        "iteration": 1,
        "code": code,
        "params": base_params,
        "references": _references(session),
        "settings": _settings_dump(session),
        "runtime_settings": _runtime_values(session),
    }
    return await executor.audit_params(payload)


async def _h_submit_code(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    code = str(args.get("code") or "")
    if not code.strip():
        return {"ok": False, "error": "code is required"}

    async def spawn(db: Any) -> tuple[Version, Job]:
        from cadx.spawn import spawn_code

        project = await session._project(db)
        parent = None
        if args.get("parent_version_id"):
            parent = await session._version(db, str(args["parent_version_id"]))
        return await spawn_code(
            db,
            project,
            code,
            params=args.get("params") or None,
            parent=parent,
            part_id=str(args.get("part_id") or "") or None,
            part_name=str(args.get("part_name") or "") or None,
            prompt=str(args.get("prompt") or ""),
            enqueue_fn=_enqueue_fn(session),
        )

    return await session._spawn_and_wait("submit_code", spawn, bool(args.get("wait")))


async def _h_parametrize_mesh(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    executor = _executor(session)
    if isinstance(executor, dict):
        return executor
    if getattr(executor, "parametrize_mesh", None) is None:
        return {"ok": False, "error": "executor does not support parametrize_mesh"}
    resolved = await _bound_version(session, args)
    if isinstance(resolved, dict):
        return resolved
    v = resolved
    # payload mirrors cadx.cad.tasks.run_parametrize_payload — stl_rel (the
    # exact artifact the version was scored on) preferred, mesh_raw.glb +
    # stored post ops as the replay fallback
    stl_rel = storage.best_mesh_relpath(v.project_id, v.id, v.iterations or 1)
    raw = storage.version_dir(v.project_id, v.id) / "mesh" / "mesh_raw.glb"
    mesh_rel = storage.relpath(raw) if raw.exists() else None
    if stl_rel is None and mesh_rel is None:
        return {"ok": False, "error": "version has no mesh artifact to parametrize"}
    payload = {
        "job_id": str(uuid.uuid4()),
        "project_id": str(v.project_id),
        "version_id": str(v.id),
        "iteration": v.iterations or 1,
        "subdir": "parametrize",
        "mode": "final",
        "mesh_rel": mesh_rel,
        "stl_rel": stl_rel,
        "ref_mask_rel": None,
        "post": (v.mesh or {}).get("post") or v.params or {},
        "anchors": (v.analysis or {}).get("anchors"),
        "spec": v.spec,
        "analysis": v.analysis,
        "settings": _settings_dump(session),
        "runtime_settings": _runtime_values(session),
        "full_exports": False,
        "light": False,
        "name": "model",
    }
    return await executor.parametrize_mesh(payload)


# --------------------------------------------------------------------------- #
# handlers — mesh                                                              #
# --------------------------------------------------------------------------- #


async def _h_mesh_generate(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    asset_ids = args.get("asset_ids")
    if not isinstance(asset_ids, list) or not asset_ids:
        return {"ok": False, "error": "asset_ids list required (upload images first)"}

    async def spawn(db: Any) -> tuple[Version, Job]:
        from cadx.spawn import spawn_meshgen

        project = await session._project(db)
        return await spawn_meshgen(
            db,
            project,
            asset_ids=[str(a) for a in asset_ids],
            subject=str(args.get("subject") or ""),
            model=str(args.get("model") or ""),
            seed=int(args.get("seed") or 0),
            post=args.get("post") or None,
            part_id=str(args.get("part_id") or "") or None,
            part_name=str(args.get("part_name") or "") or None,
            prompt=str(args.get("prompt") or args.get("subject") or "mcp mesh generation"),
            enqueue_fn=_enqueue_fn(session),
        )

    return await session._spawn_and_wait("mesh_generate", spawn, bool(args.get("wait")))


async def _h_mesh_preview(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    ops = args.get("ops")
    if not isinstance(ops, list) or not ops:
        return {"ok": False, "error": "ops list required"}
    try:
        validate_ops(ops)
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    resolved = await _bound_version(session, args)
    if isinstance(resolved, dict):
        return resolved
    v = resolved
    rel = storage.best_mesh_relpath(v.project_id, v.id, v.iterations or 1)
    if rel is None:
        raw = storage.version_dir(v.project_id, v.id) / "mesh" / "mesh_raw.glb"
        rel = storage.relpath(raw) if raw.exists() else None
    if rel is None:
        return {"ok": False, "error": "version has no mesh artifact to preview"}
    runtime = getattr(session.ctx, "runtime_settings", None)
    anchors = (v.analysis or {}).get("anchors")

    def _run() -> dict[str, Any]:
        from cadx.cad.mesh_ops import apply_ops, load_mesh

        mesh = load_mesh(storage.resolve(rel))
        work, report = apply_ops(mesh, ops, anchors=anchors, runtime=runtime)
        b = work.bounds
        return {
            "faces": len(work.faces),
            "watertight": bool(work.is_watertight),
            "bbox_size": [round(float(x), 3) for x in (b[1] - b[0])],
            "ops_report": report,
        }

    try:
        out = await asyncio.to_thread(_run)
    except MeshOpError as e:
        return {"ok": False, "error": str(e), "op_index": e.op_index, "ops_report": e.report}
    except Exception as e:
        return {"ok": False, "error": f"preview failed: {e}"}
    return {
        "ok": True,
        "version_id": str(v.id),
        "note": "preview only — commit the same ops with mesh_edit",
        **out,
    }


async def _h_editable_features(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    resolved = await _bound_version(session, args)
    if isinstance(resolved, dict):
        return resolved
    v = resolved
    src = feature_edit_brep_path(v)
    if src is None:
        return {
            "ok": True,
            "features": [],
            "reason": "version has no B-Rep solid to feature-edit",
        }

    def _scan() -> list[dict[str, Any]]:
        if src.suffix == ".step":
            from build123d import import_step

            shape = import_step(str(src))
        else:
            from build123d import import_brep

            shape = import_brep(str(src))
        from cadx.cad import editfeature

        return editfeature.find_editable_features(shape)

    try:
        features = await asyncio.to_thread(_scan)
    except Exception as e:
        log.warning("mcp.editable_features_failed", version=str(v.id), error=str(e))
        return {"ok": False, "features": [], "error": f"feature scan failed: {e}"}
    return {"ok": True, "features": features}


async def _h_edit_feature(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    feature_id = args.get("feature_id")
    if feature_id is None or (isinstance(feature_id, str) and not feature_id.strip()):
        return {"ok": False, "error": "feature_id required — see editable_features"}
    params = args.get("params") or {}
    value = params.get("diameter")
    if value is None:
        return {
            "ok": False,
            "error": "params.diameter required — only 'diameter' is editable",
        }
    try:
        diameter = float(value)
    except (TypeError, ValueError):
        return {"ok": False, "error": "params.diameter must be a number (mm)"}
    if isinstance(feature_id, str):
        feature_id = feature_id.strip()
        feature_id = int(feature_id) if feature_id.isdigit() else feature_id

    async def spawn(db: Any) -> tuple[Version, Job]:
        parent = await session._version(db, str(args.get("version_id", "")))
        return await spawn_edit_feature(
            db,
            parent,
            feature_id=feature_id,
            parameter="diameter",
            value=diameter,
            enqueue_fn=_enqueue_fn(session),
        )

    return await session._spawn_and_wait("edit_feature", spawn, bool(args.get("wait")))


# --------------------------------------------------------------------------- #
# handlers — jobs                                                              #
# --------------------------------------------------------------------------- #


async def _h_list_jobs(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    limit = max(1, min(100, int(args.get("limit", 20) or 20)))
    try:
        async with session_scope() as db:
            await session._project(db)
            pid = _session_pid(session)
            rows = await db.execute(
                select(Job, Version, Part)
                .join(Version, Job.version_id == Version.id)
                .outerjoin(Part, Version.part_id == Part.id)
                .where(Version.project_id == pid)
                .order_by(col(Job.created_at).desc())
                .limit(limit)
            )
            triples = rows.all()
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "jobs": [
            {
                "id": str(j.id),
                "type": j.type.value,
                "status": j.status.value,
                "step": j.step,
                "progress": j.progress,
                "version_id": str(j.version_id),
                "version_number": v.number,
                "part_id": str(v.part_id),
                "part": (
                    f"P{pt.number}{f' · {pt.name}' if pt.name else ''}" if pt is not None else None
                ),
                "created_at": j.created_at.isoformat(),
            }
            for j, v, pt in triples
        ],
    }


async def _h_task_status(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    task_id = str(args.get("task_id") or "").strip()
    if not task_id:
        return {"ok": False, "error": "task_id is required"}
    try:
        from arq.jobs import Job as ArqJob
        from arq.jobs import JobStatus as ArqStatus

        from cadx.api.deps import get_pool

        job = ArqJob(task_id, await get_pool())
        status = await job.status()
        out: dict[str, Any] = {"ok": True, "task_id": task_id, "status": status.value}
        if status == ArqStatus.complete:
            try:
                out["result"] = await job.result()
            except Exception as e:
                out["result_error"] = str(e)
        return out
    except Exception as e:
        return {"ok": False, "error": f"task status unavailable: {e}"}


async def _h_cancel_job(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    try:
        async with session_scope() as db:
            resolved = await _bound_job(session, db, str(args.get("job_id", "")))
            if isinstance(resolved, dict):
                return resolved
            job = resolved
            if job.status in _TERMINAL_JOB_STATES:
                return {"ok": False, "error": f"job already {job.status.value}"}
            # same semantics as routers/jobs.py cancel()
            job.cancel_requested = True
            if job.status == JobStatus.queued:
                job.status = JobStatus.cancelled
                v = await db.get(Version, job.version_id)
                if v is not None:
                    v.status = VersionStatus.cancelled
            await db.commit()
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    # wake in-flight cad executor tasks (topopt/fem poll this key in-loop)
    try:
        from cadx.api.deps import get_pool

        pool = await get_pool()
        await pool.setex(f"cancel:{job.id}", 3600, b"1")
    except Exception:
        pass
    return {"ok": True, "job_id": str(job.id)}


async def _h_finish_job(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    try:
        async with session_scope() as db:
            resolved = await _bound_job(session, db, str(args.get("job_id", "")))
            if isinstance(resolved, dict):
                return resolved
            job = resolved
            if job.status != JobStatus.running:
                return {"ok": False, "error": f"job is {job.status.value}, not running"}
            job.finish_requested = True
            await db.commit()
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "job_id": str(job.id)}


async def _h_restart_job(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    async def spawn(db: Any) -> Any:
        from cadx.spawn import spawn_restart

        resolved = await _bound_job(session, db, str(args.get("job_id", "")))
        if isinstance(resolved, dict):
            raise SpawnError(resolved["error"], status=422)
        job = resolved
        if job.status not in _TERMINAL_JOB_STATES:
            raise SpawnError(f"job is {job.status.value} — cancel it before restarting")
        # HARD LLM-free guard: generate/edit/resume jobs restart through
        # run_agent_job (the LLM-driven agent) — never expose those on MCP.
        # Deterministic job types (fem/reexport/refine/resample/organic/
        # mesh_edit/edit_feature/params/optimize/code/meshgen) replay their
        # stored requests without an LLM — including on LLM-kind versions.
        if job.type in (JobType.generate, JobType.edit, JobType.resume):
            raise SpawnError(
                "restart is only exposed for deterministic jobs — this "
                "version is LLM-generated; write new code and submit_code "
                "instead",
                status=409,
            )
        return await spawn_restart(db, job, enqueue_fn=_enqueue_fn(session))

    return await session._spawn_and_wait("restart_job", spawn, bool(args.get("wait")))


# --------------------------------------------------------------------------- #
# handlers — assets / images                                                   #
# --------------------------------------------------------------------------- #


def _asset_dict(a: Asset) -> dict[str, Any]:
    return {
        "id": str(a.id),
        "kind": a.kind.value,
        "filename": a.filename,
        "caption": a.caption,
        "width": a.width,
        "height": a.height,
        "created_at": a.created_at.isoformat(),
        "url": f"/api/assets/{a.id}/file",
    }


async def _bound_asset(session: Any, db: Any, asset_id: str) -> Asset | dict[str, Any]:
    try:
        aid = uuid.UUID(str(asset_id))
    except ValueError:
        return {"ok": False, "error": "invalid asset id"}
    await session._project(db)
    a = await db.get(Asset, aid)
    if a is None or a.project_id != _session_pid(session):
        return {"ok": False, "error": "asset not found in this project"}
    return a


async def _h_list_assets(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    kind = str(args.get("kind") or "").strip()
    if kind:
        try:
            kind_filter = AssetKind(kind)
        except ValueError:
            return {
                "ok": False,
                "error": f"unknown kind '{kind}' — valid: {', '.join(k.value for k in AssetKind)}",
            }
    else:
        kind_filter = None
    try:
        async with session_scope() as db:
            await session._project(db)
            pid = _session_pid(session)
            q = select(Asset).where(Asset.project_id == pid).order_by(col(Asset.created_at))
            if kind_filter is not None:
                q = q.where(Asset.kind == kind_filter)
            rows = await db.execute(q)
            return {"ok": True, "assets": [_asset_dict(a) for a in rows.scalars().all()]}
    except SpawnError as e:
        return {"ok": False, "error": str(e)}


def _read_upload(path_arg: str, b64_arg: Any) -> tuple[bytes | None, str, str | None]:
    """Sync part of upload_asset: read the file bytes or decode the base64
    payload. Returns (data, default_filename, error)."""
    if path_arg:
        src = Path(path_arg).expanduser()
        if not src.is_file():
            return None, "", f"path is not a readable file: {path_arg}"
        if src.stat().st_size > _MAX_UPLOAD_BYTES:
            return None, "", f"file exceeds {_MAX_UPLOAD_BYTES // 2**20}MB limit"
        return src.read_bytes(), src.name, None
    try:
        return base64.b64decode(str(b64_arg), validate=True), "upload.bin", None
    except Exception:
        return None, "", "data_base64 is not valid base64"


async def _h_upload_asset(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    path_arg = str(args.get("path") or "").strip()
    b64_arg = args.get("data_base64")
    if bool(path_arg) == bool(b64_arg):
        return {"ok": False, "error": "exactly one of path or data_base64 is required"}
    data, default_name, err = await asyncio.to_thread(_read_upload, path_arg, b64_arg)
    if err is not None or data is None:
        return {"ok": False, "error": err or "upload failed"}
    filename = str(args.get("name") or "").strip() or default_name
    if len(data) > _MAX_UPLOAD_BYTES:
        return {"ok": False, "error": f"data exceeds {_MAX_UPLOAD_BYTES // 2**20}MB limit"}
    try:
        kind = AssetKind(str(args.get("kind") or "image"))
    except ValueError:
        return {
            "ok": False,
            "error": f"unknown kind — valid: {', '.join(k.value for k in AssetKind)}",
        }
    filename = storage.safe_filename(filename or "upload.bin")
    # image assets are normalised to JPEG (EXIF rotation, size cap) — same as
    # the REST upload path; undecodable bytes are stored as-is
    mime = ""
    width: int | None = None
    height: int | None = None
    if kind == AssetKind.image:
        from cadx.llm.images import normalize

        norm = normalize(data)
        if norm is not None:
            data, width, height = norm
            filename = filename.rsplit(".", 1)[0] + ".jpg"
            mime = "image/jpeg"
        else:
            log.warning("mcp.upload.normalize_failed", filename=filename)
            try:
                import io

                from PIL import Image

                with Image.open(io.BytesIO(data)) as im:
                    width, height = im.size
            except Exception:
                pass
    try:
        async with session_scope() as db:
            await session._project(db)
            pid = _session_pid(session)
            asset = Asset(
                project_id=pid,
                kind=kind,
                filename=filename,
                path="",  # filled below
                mime=mime,
                width=width,
                height=height,
                caption=str(args.get("caption") or "")[:400],
            )
            path = storage.asset_path(pid, asset.id, asset.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            asset.path = storage.relpath(path)
            db.add(asset)
            await db.commit()
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    # keep ctx.assets in sync so view_image/mesh_generate see the upload
    # without waiting for the next rebind
    try:
        from cadx.agent.types import InputAsset

        session.ctx.assets.append(
            InputAsset(
                asset_id=str(asset.id),
                kind=asset.kind.value,
                path=path,
                caption=asset.caption,
                filename=asset.filename,
            )
        )
    except Exception:
        pass
    return {"ok": True, "asset": _asset_dict(asset), "asset_id": str(asset.id)}


async def _h_view_image(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    try:
        async with session_scope() as db:
            resolved = await _bound_asset(session, db, str(args.get("asset_id", "")))
            if isinstance(resolved, dict):
                return resolved
            asset = resolved
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    try:
        path = storage.resolve(asset.path)
    except ValueError:
        return {"ok": False, "error": "invalid asset path"}
    if not path.exists():
        return {"ok": False, "error": "asset file missing on disk"}
    return {
        "ok": True,
        "asset_id": str(asset.id),
        "filename": asset.filename,
        "_images": [str(path)],
    }


async def _h_remove_bg(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    try:
        async with session_scope() as db:
            resolved = await _bound_asset(session, db, str(args.get("asset_id", "")))
            if isinstance(resolved, dict):
                return resolved
            src = resolved
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    if src.kind not in IMAGE_ASSET_KINDS:
        return {"ok": False, "error": "asset is not an image"}
    try:
        data = storage.resolve(src.path).read_bytes()
    except (ValueError, FileNotFoundError):
        return {"ok": False, "error": "asset file missing"}
    try:
        import io

        import rembg
        from PIL import Image

        from cadx.api.routers.assets import _rembg_session, _store_derivative
    except ImportError:
        return {
            "ok": False,
            "error": "rembg is not installed — install the backend with rembg support",
        }
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        out = await asyncio.to_thread(lambda: rembg.remove(img, session=_rembg_session()))
        if isinstance(out, bytes | bytearray):
            png = bytes(out)
        else:
            buf = io.BytesIO()
            out.convert("RGBA").save(buf, format="PNG")
            png = buf.getvalue()
    except Exception as e:
        log.warning("mcp.remove_bg_failed", asset=str(src.id), error=str(e))
        return {"ok": False, "error": f"background removal failed: {e}"}
    async with session_scope() as db:
        src_db = await db.get(Asset, src.id)
        asset = _store_derivative(src_db, png, "nobg") if src_db is not None else None
        if asset is None:
            return {"ok": False, "error": "background removal produced no PNG"}
        db.add(asset)
        await db.commit()
    return {
        "ok": True,
        "asset": _asset_dict(asset),
        "asset_id": str(asset.id),
        "_images": [str(storage.resolve(asset.path))],
    }


async def _h_generate_image(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    # the session's own implementation wins when it exists (image budget,
    # bound-project resolution live there post-refactor)
    handler = getattr(session, "_tool_generate_image", None)
    if handler is not None:
        return await handler(args)
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt is required"}
    ctx = session.ctx
    if getattr(ctx, "image", None) is None:
        return {"ok": False, "error": "image generation not configured (no image adapter)"}
    try:
        async with session_scope() as db:
            await session._project(db)
    except SpawnError as e:
        return {"ok": False, "error": str(e)}
    ref = None
    ref_id = str(args.get("reference_asset_id") or "").strip()
    if ref_id:
        ref = next(
            (a for a in ctx.assets if a.asset_id == ref_id),
            None,
        )
        if ref is None:
            return {"ok": False, "error": f"unknown image asset '{ref_id}'"}
    try:
        from cadx.agent.concept_image import generate_concept_images

        out = await generate_concept_images(
            ctx,
            prompt,
            count=max(1, min(2, int(args.get("count", 1) or 1))),
            seed=int(args["seed"]) if args.get("seed") is not None else None,
            ref_asset=ref,
            strength=float(args["strength"]) if args.get("strength") is not None else None,
            mark_sketch=bool(args.get("mark_sketch")),
            source="mcp",
        )
        return {
            "ok": True,
            "assets": [{"asset_id": x.asset_id, "filename": x.filename} for x in out],
            "_images": [str(x.path) for x in out],
        }
    except Exception as e:
        log.warning("mcp.generate_image_failed", error=str(e))
        return {"ok": False, "error": f"image generation failed: {e}"}


# --------------------------------------------------------------------------- #
# handlers — system                                                            #
# --------------------------------------------------------------------------- #


async def _h_settings_get(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    from cadx.config import get_settings
    from cadx.runtime_config import load_runtime_settings

    async with session_scope() as db:
        runtime = await load_runtime_settings(db)
    # public_values() NEVER contains llm_api_key — only the llm_api_key_set
    # boolean; mirrors routers/settings.py GET
    out = runtime.public_values()
    out["data_dir"] = get_settings().data_dir
    return {"ok": True, "settings": out}


async def _h_mesh_health(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    from cadx.agent.mesh_client import make_mesh_client
    from cadx.runtime_config import load_runtime_settings

    try:
        async with session_scope() as db:
            runtime = await load_runtime_settings(db)
        client = make_mesh_client(runtime_settings=runtime)
        if client is None:
            return {"ok": False, "error": "mesh server not configured (MESH_BASE_URL)"}
        h = await client.health()
    except Exception as e:
        return {"ok": False, "error": f"mesh health probe failed: {e}"}
    return {
        "ok": bool(h.ok),
        "device": h.device,
        "backend": h.backend,
        "error": h.error,
        "models": [m.model_dump(mode="json") for m in h.models],
    }


async def _h_image_models(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    from cadx.agent.image_client import make_image_client
    from cadx.runtime_config import load_runtime_settings

    try:
        async with session_scope() as db:
            runtime = await load_runtime_settings(db)
        client = make_image_client(runtime_settings=runtime)
        if client is None:
            return {"ok": True, "models": [], "note": "mesh server not configured"}
        models = await client.models()
    except Exception as e:
        return {"ok": False, "error": f"image model probe failed: {e}"}
    return {"ok": True, "models": [m.model_dump(mode="json") for m in models]}


async def _h_stats(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    from cadx.db.models import Project

    async def _count(db: Any, model: Any) -> int:
        return int((await db.execute(select(func.count()).select_from(model))).scalar_one())

    async def _by_status(db: Any, model: Any, col_: Any) -> dict[str, int]:
        rows = await db.execute(select(col_, func.count()).group_by(col_))
        return {str(s.value if hasattr(s, "value") else s): int(c) for s, c in rows.all()}

    async with session_scope() as db:
        out: dict[str, Any] = {
            "ok": True,
            "totals": {
                "projects": await _count(db, Project),
                "versions": await _count(db, Version),
                "jobs": await _count(db, Job),
                "assets": await _count(db, Asset),
            },
            "versions_by_status": await _by_status(db, Version, Version.status),
            "jobs_by_status": await _by_status(db, Job, Job.status),
        }
        pid = _session_pid(session)
        if pid is not None:
            n_versions = await db.execute(
                select(func.count()).select_from(Version).where(Version.project_id == pid)
            )
            n_jobs = await db.execute(
                select(func.count())
                .select_from(Job)
                .join(Version, Job.version_id == Version.id)
                .where(Version.project_id == pid)
            )
            n_assets = await db.execute(
                select(func.count()).select_from(Asset).where(Asset.project_id == pid)
            )
            out["project"] = {
                "project_id": str(pid),
                "versions": int(n_versions.scalar_one()),
                "jobs": int(n_jobs.scalar_one()),
                "assets": int(n_assets.scalar_one()),
            }
    return out


async def _h_use_project(session: Any, args: dict[str, Any]) -> dict[str, Any]:
    pid = str(args.get("project_id") or "").strip()
    if not pid:
        return {"ok": False, "error": "project_id is required"}
    handler = getattr(session, "_tool_attach_project", None)
    if handler is not None:
        return await handler({"project_id": pid})
    # fallback for sessions without the bound method — attach_project is a
    # session tool, not an MCP_HANDLER, so this cannot recurse
    return await session.execute("attach_project", {"project_id": pid})


MCP_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    "codegen_contract": _h_codegen_contract,
    "check_code": _h_check_code,
    "execute_cad": _h_execute_cad,
    "audit_params": _h_audit_params,
    "submit_code": _h_submit_code,
    "parametrize_mesh": _h_parametrize_mesh,
    "mesh_generate": _h_mesh_generate,
    "mesh_preview": _h_mesh_preview,
    "editable_features": _h_editable_features,
    "edit_feature": _h_edit_feature,
    "list_jobs": _h_list_jobs,
    "task_status": _h_task_status,
    "cancel_job": _h_cancel_job,
    "finish_job": _h_finish_job,
    "restart_job": _h_restart_job,
    "list_assets": _h_list_assets,
    "upload_asset": _h_upload_asset,
    "view_image": _h_view_image,
    "remove_bg": _h_remove_bg,
    "generate_image": _h_generate_image,
    "settings_get": _h_settings_get,
    "mesh_health": _h_mesh_health,
    "image_models": _h_image_models,
    "stats": _h_stats,
    "use_project": _h_use_project,
}
