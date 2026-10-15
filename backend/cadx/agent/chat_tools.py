"""Chat-agent tool session — the full CADX capability set as LLM tools.

`ChatToolSession` subclasses the generic `ToolSession` so the catalog /
library / image-inspection handlers are reused verbatim; chat adds
project/version inspection plus ACTION tools that spawn real versions and
jobs through `cadx.spawn` (the identical code path the REST routers take)
and block until the job reaches a terminal state. The agent therefore sees
each result — including contact-sheet renders via `_images` on the vision
role — and can relay outcomes, clarify, or chain follow-up work.

Safety boundaries: no arbitrary code execution (every CAD/mesh run goes
through the sandboxed executor pipeline as a normal job), no filesystem
access, no deletion tools. Everything the agent does is visible in the
normal version tree and jobs dashboard.
"""

from __future__ import annotations

import contextlib
import inspect
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from sqlmodel import col, select

from cadx import storage
from cadx.agent.toolspecs import (
    INFO_TOOLS,
    LIBRARY_INFO_TOOLS,
    ToolSession,
    _fn,
    _i,
    _s,
    _truncate,
    _truncate_args,
)
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx
from cadx.db.models import (
    Artifact,
    ArtifactKind,
    Job,
    JobStatus,
    LibraryPart,
    Project,
    ProjectSettings,
    Version,
)
from cadx.db.session import session_scope
from cadx.llm.schemas import OrganicParams
from cadx.logging import get_logger
from cadx.spawn import (
    SpawnError,
    enqueue_resume,
    spawn_edit,
    spawn_fem,
    spawn_generate,
    spawn_mesh_edit,
    spawn_optimize,
    spawn_organic,
    spawn_params,
    spawn_refine,
    spawn_resample,
    wait_job_terminal,
)

log = get_logger(__name__)

# project.settings keys the chat agent may change (validated through
# ProjectSettings before persisting — unknown/unsafe keys are refused)
_SETTINGS_WHITELIST = {
    "material",
    "process",
    "gen_mode",
    "fem",
    "fem_note",
    "fem_material",
    "topopt",
    "topopt_volfrac",
    "topopt_iterations",
    "max_iterations",
    "min_wall_mm",
    "nozzle_mm",
    "include_fasteners",
    "mesh_model",
    "mesh_resolution",
    "mesh_seed",
    "clarification_mode",
    "build_volume",
}

# action tools share this result envelope: spawned ids + terminal summary
_WAIT_NOTE = (
    "the job keeps running in the background — the user sees its live "
    "progress card; check later with wait_job"
)


# --------------------------------------------------------------------------- #
# schemas                                                                     #
# --------------------------------------------------------------------------- #

_JOB_REF = {"job_id": _s("job id returned by a spawn tool or job_status")}
_VER_REF = {"version_id": _s("version id — see list_versions")}

CHAT_INFO_TOOLS = [
    _fn(
        "list_versions",
        "List this project's versions (newest first): id, number, kind, "
        "status, prompt, score. Call before editing so you target a real "
        "version id.",
        {"limit": _i("max results, default 20")},
        [],
    ),
    _fn(
        "version_details",
        "Inspect one version: params, measured analysis (bbox, anchors), "
        "issues, FEM summary, and optionally the generated code.",
        {
            **(_VER_REF),
            "include_code": {
                "type": "boolean",
                "description": "include the full build123d source (default false)",
            },
        },
        ["version_id"],
    ),
    _fn(
        "job_status",
        "Current state of a job: status, step, progress, questions when it is waiting_for_input.",
        _JOB_REF,
        ["job_id"],
    ),
    _fn(
        "view_render",
        "Look at a finished version: attaches its contact-sheet/render "
        "images so you can visually judge the geometry.",
        _VER_REF,
        ["version_id"],
    ),
    _fn(
        "project_settings",
        "Read the project's current settings (material, process, gen_mode, "
        "fem/topopt defaults, build volume).",
        {},
        [],
    ),
]

# every spawn tool blocks until the job is terminal by default —
# `wait=false` returns immediately for "fire and continue" flows
_WAIT = {
    "wait": {
        "type": "boolean",
        "description": "block until the job finishes (default true) — "
        "recommended so you can report the real outcome",
    }
}

CHAT_ACTION_TOOLS = [
    _fn(
        "generate_part",
        "Create a NEW part: spawns a generate job (LLM designs + verifies a "
        "parametric model, or image→3D mesh generation). mode: auto|cad|mesh|"
        "hybrid|topopt. Pass asset_ids of uploaded images when the request "
        "references them.",
        {
            "prompt": _s("complete part description — fold in everything the user said"),
            "mode": {
                "type": "string",
                "enum": ["auto", "cad", "mesh", "hybrid", "topopt"],
                "description": "generation mode, default auto",
            },
            "asset_ids": {
                "type": "array",
                "items": _s("asset id"),
                "description": "image/reference asset ids from the conversation",
            },
            **_WAIT,
        },
        ["prompt"],
    ),
    _fn(
        "edit_version",
        "Modify an existing version with a natural-language edit request — "
        "spawns an edit job (LLM reworks the code/mesh).",
        {
            **_VER_REF,
            "prompt": _s("what to change"),
            "mode": {
                "type": "string",
                "enum": ["auto", "cad", "mesh", "hybrid", "topopt"],
                "description": "force a mode; default inherits the parent",
            },
            **_WAIT,
        },
        ["version_id", "prompt"],
    ),
    _fn(
        "set_params",
        "Change PARAMS values of a parametric version and re-execute — cheap "
        "dimension tweaks without an LLM round. Pass only the keys to change.",
        {
            **_VER_REF,
            "params": {
                "type": "object",
                "description": "PARAMS key -> new value",
            },
            **_WAIT,
        },
        ["version_id", "params"],
    ),
    _fn(
        "mesh_edit",
        "Apply deterministic mesh operations (mesh_scale, mesh_rotate, "
        "mesh_flatten_bottom, mesh_drill_holes, mesh_boolean_add/cut, …) to a "
        "version — creates a mesh_edit child. Use list from version_details.",
        {
            **_VER_REF,
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "description": "{op: name, args: {...}}",
                },
                "description": "ordered mesh ops to apply",
            },
            "note": _s("short description of the edit"),
            **_WAIT,
        },
        ["version_id", "ops"],
    ),
    _fn(
        "fem_check",
        "Run a structural FEM analysis on a version's mesh with the given "
        "load cases — returns safety factors per case.",
        {
            **_VER_REF,
            "load_cases": {
                "type": "array",
                "items": {"type": "object"},
                "description": "load cases (region/load schema like optimize)",
            },
            "material": _s("material key, e.g. pla|petg|aluminum — empty = default"),
            **_WAIT,
        },
        ["version_id", "load_cases"],
    ),
    _fn(
        "optimize_version",
        "Topology-optimize a version: SIMP on the mesh with the given load "
        "cases — creates an optimized child version, FEM-verified.",
        {
            **_VER_REF,
            "load_cases": {
                "type": "array",
                "items": {"type": "object"},
                "description": "load cases (fixed regions + loads)",
            },
            "optimize": {
                "type": "object",
                "description": "SIMP options, e.g. {target_volume_fraction: 0.4}",
            },
            "material": _s("material key — empty = spec default"),
            **_WAIT,
        },
        ["version_id"],
    ),
    _fn(
        "topopt_resample",
        "Re-extract a stored density field at a new threshold/min-member "
        "without re-running SIMP — only on optimize/organic versions.",
        {
            **_VER_REF,
            "density_threshold": {"type": "number", "description": "0.2-0.9"},
            "min_member_mm": {"type": "number", "description": "minimum member size"},
            **_WAIT,
        },
        ["version_id"],
    ),
    _fn(
        "refine_version",
        "Re-cut exact bores/threads/inserts at measured anchors on an "
        "optimized mesh and restore shrunk surfaces — refine child version.",
        {
            **_VER_REF,
            "features": {
                "type": "array",
                "items": {"type": "object"},
                "description": "explicit features; omit for auto-resolution",
            },
            **_WAIT,
        },
        ["version_id"],
    ),
    _fn(
        "organic_restyle",
        "Aesthetic pseudo-topology restyle of a version — gyroid / "
        "trabecular / voronoi / branches infill structure. Purely visual, "
        "NOT a structural optimization.",
        {
            **_VER_REF,
            "style": {
                "type": "string",
                "enum": ["gyroid", "trabecular", "voronoi", "branches"],
            },
            "volfrac": {"type": "number", "description": "solid fraction 0.05-0.9"},
            "period_mm": {"type": "number", "description": "feature size in mm"},
            "shell_mm": {"type": "number", "description": "solid skin thickness (0=off)"},
            **_WAIT,
        },
        ["version_id", "style"],
    ),
    _fn(
        "generate_image",
        "Generate a design-concept image with the local image model and view "
        "it — ideation, style reference, or img2img variation of an existing "
        "asset (reference_asset_id). Stored as a generated_image asset usable "
        "for mesh generation.",
        {
            "prompt": _s("what to render — subject, view, style"),
            "count": _i("variants, 1 or 2"),
            "reference_asset_id": _s("optional existing image asset to vary/edit"),
            "strength": {"type": "number", "description": "img2img strength 0-1"},
        },
        ["prompt"],
    ),
    _fn(
        "answer_questions",
        "Submit the user's answers to a job that is waiting_for_input — "
        "resumes the job. Relay the questions verbatim first.",
        {
            **_JOB_REF,
            "answers": {
                "type": "array",
                "items": _s("answer text, in question order"),
                "description": "the user's answers",
            },
            **_WAIT,
        },
        ["job_id", "answers"],
    ),
    _fn(
        "wait_job",
        "Wait for a running spawned job to finish and return its outcome — "
        "use after wait=false or when a previous wait was interrupted.",
        _JOB_REF,
        ["job_id"],
    ),
    _fn(
        "update_project_settings",
        "Update whitelisted project settings (material, process, gen_mode, "
        "fem/topopt defaults, nozzle, build_volume, …) — e.g. 'print in "
        "PETG' sets material='PETG'.",
        {
            "settings": {
                "type": "object",
                "description": "setting key -> value; only whitelisted keys apply",
            }
        },
        ["settings"],
    ),
    _fn(
        "save_library_part",
        "Save a successful version to the reusable part library — as a "
        "parametric code template (as_template=true, needs generated code) "
        "or as a STEP body.",
        {
            **_VER_REF,
            "name": _s("library part name"),
            "caption": _s("short description"),
            "as_template": {
                "type": "boolean",
                "description": "save the parametric code instead of the STEP",
            },
        },
        ["version_id", "name"],
    ),
]


# --------------------------------------------------------------------------- #
# session                                                                     #
# --------------------------------------------------------------------------- #


class ChatToolSession(ToolSession):
    """kind="chat" — project inspection + job-spawning actions."""

    def __init__(
        self,
        ctx: AgentCtx,
        *,
        chat_id: uuid.UUID,
        attach_images: bool = True,
        cancel_fn: Callable[[], Awaitable[bool]] | None = None,
        interrupt_fn: Callable[[], Awaitable[bool]] | None = None,
        enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        publish_fn: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(ctx, kind="info")  # placeholder — tools() overridden below
        self.kind = "chat"
        self.chat_id = chat_id
        self.attach_images = attach_images
        self._cancel_fn = cancel_fn
        self._interrupt_fn = interrupt_fn
        self._enqueue_fn = enqueue_fn
        self._publish_fn = publish_fn
        # every spawned job — persisted into the chat message meta so the UI
        # renders progress cards and deep links
        self.spawned: list[dict[str, Any]] = []

    # -- schema selection --------------------------------------------------- #

    def tools(self) -> list[dict[str, Any]]:
        out = [
            *CHAT_INFO_TOOLS,
            *CHAT_ACTION_TOOLS,
            *INFO_TOOLS,
            *LIBRARY_INFO_TOOLS,
        ]
        # image inspection only makes sense with a vision-capable role
        if self.attach_images:
            from cadx.agent.toolspecs import IMAGE_TOOLS

            out += IMAGE_TOOLS
        if self.image_left <= 0 or not self.attach_images:
            out = [t for t in out if t["function"]["name"] != "generate_image"]
        return out

    # -- dispatch: strip images for text-only roles -------------------------- #

    async def _publish(self, event: str, data: dict[str, Any]) -> None:
        if self._publish_fn is not None:
            with contextlib.suppress(Exception):
                await self._publish_fn(event, data)

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in self.allowed_names():
            result = {"ok": False, "error": f"tool '{name}' is not available in this session"}
            self.tool_log.append({"name": name, "args": _truncate_args(args), "result": result})
            return result
        await self._publish("tool_start", {"name": name, "args": _truncate_args(args)})
        try:
            result = await self._dispatch(name, args)
        except SpawnError as e:
            result = {"ok": False, "error": str(e)}
        except Exception as e:
            log.warning("chat_tool.error", tool=name, error=str(e))
            result = {"ok": False, "error": f"tool '{name}' failed: {e}"}
        if not self.attach_images and result.pop("_images", None):
            result["images_suppressed"] = "image result hidden — text-only model role"
        self.tool_log.append(
            {"name": name, "args": _truncate_args(args), "result": _truncate(result)}
        )
        await self._publish(
            "tool_result",
            {
                "name": name,
                "ok": bool(result.get("ok", True)),
                "result": _truncate({k: v for k, v in result.items() if k != "_images"}),
            },
        )
        return result

    async def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown tool '{name}'"}
        if inspect.iscoroutinefunction(handler):
            res: dict[str, Any] = await handler(args)
        else:
            res = handler(args)
        return res

    # -- shared spawn+wait path ---------------------------------------------- #

    async def _wait(self, job_id: uuid.UUID) -> dict[str, Any]:
        res = await wait_job_terminal(
            job_id,
            cancel_check=self._cancel_fn,
            interrupt_check=self._interrupt_fn,
            timeout_s=float(self.ctx.runtime_settings.chat_spawn_timeout_s),
        )
        await self._attach_result_images(res)
        return res

    async def _spawn_and_wait(
        self,
        name: str,
        spawn: Callable[[Any], Awaitable[tuple[Version, Job] | Job]],
        wait: bool = True,
    ) -> dict[str, Any]:
        """Run a spawn helper in a short transaction, record it, then wait."""
        try:
            async with session_scope() as db:
                out = await spawn(db)
        except SpawnError as e:
            return {"ok": False, "error": str(e)}
        job = out[1] if isinstance(out, tuple) else out
        version_id = out[0].id if isinstance(out, tuple) else job.version_id
        self.spawned.append(
            {
                "tool": name,
                "job_id": str(job.id),
                "version_id": str(version_id),
            }
        )
        if not wait:
            return {
                "ok": True,
                "status": "queued",
                "job_id": str(job.id),
                "version_id": str(version_id),
                "note": _WAIT_NOTE,
            }
        return await self._wait(job.id)

    async def _attach_result_images(self, res: dict[str, Any]) -> None:
        """Attach the finished version's contact sheet so the vision role can
        see — and judge — the produced geometry."""
        if not self.attach_images or not res.get("ok") or res.get("status") != "succeeded":
            return
        vid = res.get("version_id")
        if not vid:
            return
        try:
            images: list[str] = []
            sheet = storage.version_dir(self.ctx.project_id, vid) / "contact_sheet.png"
            if sheet.exists():
                images.append(str(sheet))
            if images:
                res["_images"] = images
                res["render"] = "contact sheet attached — verify it matches the request"
        except Exception:
            pass

    async def _project(self, db: Any) -> Project:
        p = await db.get(Project, self.ctx.project_id)
        if p is None:
            raise SpawnError("project not found", status=404)
        return p

    async def _version(self, db: Any, version_id: str) -> Version:
        try:
            vid = uuid.UUID(str(version_id))
        except ValueError:
            raise SpawnError("invalid version id", status=422) from None
        v = await db.get(Version, vid)
        if v is None or v.project_id != self.ctx.project_id:
            raise SpawnError("version not found in this project", status=404)
        return v

    # -- read tools ----------------------------------------------------------- #

    async def _tool_list_versions(self, a: dict[str, Any]) -> dict[str, Any]:
        limit = max(1, min(50, int(a.get("limit", 20) or 20)))
        async with session_scope() as db:
            rows = await db.execute(
                select(Version)
                .where(Version.project_id == self.ctx.project_id)
                .order_by(col(Version.number).desc())
                .limit(limit)
            )
            versions = rows.scalars().all()
            job_rows = await db.execute(
                select(Job)
                .where(col(Job.version_id).in_([v.id for v in versions]))
                .order_by(col(Job.created_at).desc())
            )
            jobs = list(job_rows.scalars().all())
        active = {
            str(j.version_id): str(j.id)
            for j in jobs
            if j.status in (JobStatus.queued, JobStatus.running, JobStatus.waiting_for_input)
        }
        return {
            "ok": True,
            "versions": [
                {
                    "id": str(v.id),
                    "number": v.number,
                    "kind": v.kind.value,
                    "status": v.status.value,
                    "prompt": (v.prompt or "")[:120],
                    "score": v.score,
                    "resolved_mode": v.resolved_mode,
                    "active_job": active.get(str(v.id)),
                }
                for v in versions
            ],
        }

    async def _tool_version_details(self, a: dict[str, Any]) -> dict[str, Any]:
        async with session_scope() as db:
            v = await self._version(db, str(a.get("version_id", "")))
            out: dict[str, Any] = {
                "ok": True,
                "id": str(v.id),
                "number": v.number,
                "kind": v.kind.value,
                "status": v.status.value,
                "resolved_mode": v.resolved_mode,
                "params": v.params or {},
                "analysis": {
                    k: (v.analysis or {}).get(k)
                    for k in ("bbox_size", "volume", "watertight", "anchors", "warnings")
                    if (v.analysis or {}).get(k) is not None
                },
                "fem": {
                    "cases": [
                        {
                            "name": c.get("name"),
                            "safety_factor": c.get("safety_factor"),
                            "max_vm_mpa": c.get("max_vm_mpa"),
                        }
                        for c in (v.fem or {}).get("cases") or []
                    ]
                },
                "issues": [i.get("what", "") for i in (v.issues or [])[:8]],
                "has_code": bool(v.code),
            }
            if a.get("include_code") and v.code:
                out["code"] = v.code[:12000]
                if len(v.code) > 12000:
                    out["code_truncated"] = True
            return out

    async def _tool_job_status(self, a: dict[str, Any]) -> dict[str, Any]:
        async with session_scope() as db:
            try:
                jid = uuid.UUID(str(a.get("job_id", "")))
            except ValueError:
                return {"ok": False, "error": "invalid job id"}
            j = await db.get(Job, jid)
            if j is None:
                return {"ok": False, "error": "job not found"}
            out: dict[str, Any] = {
                "ok": True,
                "job_id": str(j.id),
                "version_id": str(j.version_id),
                "type": j.type.value,
                "status": j.status.value,
                "step": j.step,
                "progress": j.progress,
            }
            if j.status == JobStatus.waiting_for_input:
                out["questions"] = j.questions or []
            if j.status == JobStatus.failed:
                out["log_tail"] = (j.log or "")[-1500:]
            return out

    async def _tool_view_render(self, a: dict[str, Any]) -> dict[str, Any]:
        async with session_scope() as db:
            v = await self._version(db, str(a.get("version_id", "")))
            rows = await db.execute(
                select(Artifact).where(
                    Artifact.version_id == v.id,
                    col(Artifact.kind).in_([ArtifactKind.render, ArtifactKind.contact_sheet]),
                )
            )
            arts = list(rows.scalars().all())
        images: list[str] = []
        sheet = storage.version_dir(v.project_id, v.id) / "contact_sheet.png"
        if sheet.exists():
            images.append(str(sheet))
        for art in arts:
            if len(images) >= 4:
                break
            try:
                p = storage.resolve(art.path)
            except ValueError:
                continue
            if p.exists() and str(p) not in images:
                images.append(str(p))
        if not images:
            return {"ok": False, "error": "no renders for this version yet"}
        return {
            "ok": True,
            "version_id": str(v.id),
            "version_number": v.number,
            "_images": images,
        }

    async def _tool_project_settings(self, a: dict[str, Any]) -> dict[str, Any]:
        async with session_scope() as db:
            p = await self._project(db)
            return {
                "ok": True,
                "name": p.name,
                "settings": p.settings or {},
                "settable": sorted(_SETTINGS_WHITELIST),
            }

    async def _tool_update_project_settings(self, a: dict[str, Any]) -> dict[str, Any]:
        raw = a.get("settings")
        if not isinstance(raw, dict) or not raw:
            return {"ok": False, "error": "settings object required"}
        bad = sorted(k for k in raw if k not in _SETTINGS_WHITELIST)
        if bad:
            return {
                "ok": False,
                "error": f"settings not settable via chat: {', '.join(bad)}",
                "settable": sorted(_SETTINGS_WHITELIST),
            }
        async with session_scope() as db:
            p = await self._project(db)
            merged = {**(p.settings or {}), **raw}
            try:
                validated = ProjectSettings.model_validate(merged)
            except Exception as e:
                return {"ok": False, "error": f"invalid settings: {e}"}
            p.settings = validated.model_dump(mode="json")
            await db.commit()
            return {"ok": True, "settings": p.settings}

    # -- action tools --------------------------------------------------------- #

    async def _tool_generate_part(self, a: dict[str, Any]) -> dict[str, Any]:
        prompt = str(a.get("prompt", "")).strip()

        async def spawn(db: Any) -> tuple[Version, Job]:
            project = await self._project(db)
            return await spawn_generate(
                db,
                project,
                prompt,
                gen_mode=str(a.get("mode") or ""),
                asset_ids=[str(x) for x in a.get("asset_ids") or []],
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("generate_part", spawn, bool(a.get("wait", True)))

    async def _tool_edit_version(self, a: dict[str, Any]) -> dict[str, Any]:
        prompt = str(a.get("prompt", "")).strip()

        async def spawn(db: Any) -> tuple[Version, Job]:
            parent = await self._version(db, str(a.get("version_id", "")))
            project = await self._project(db)
            return await spawn_edit(
                db,
                project,
                parent,
                prompt,
                gen_mode=str(a.get("mode") or ""),
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("edit_version", spawn, bool(a.get("wait", True)))

    async def _tool_set_params(self, a: dict[str, Any]) -> dict[str, Any]:
        params = a.get("params")
        if not isinstance(params, dict) or not params:
            return {"ok": False, "error": "params object required"}

        async def spawn(db: Any) -> tuple[Version, Job]:
            parent = await self._version(db, str(a.get("version_id", "")))
            return await spawn_params(db, parent, params, enqueue_fn=self._enqueue_fn)

        return await self._spawn_and_wait("set_params", spawn, bool(a.get("wait", True)))

    async def _tool_mesh_edit(self, a: dict[str, Any]) -> dict[str, Any]:
        ops = a.get("ops")
        if not isinstance(ops, list) or not ops:
            return {"ok": False, "error": "ops list required"}

        async def spawn(db: Any) -> tuple[Version, Job]:
            parent = await self._version(db, str(a.get("version_id", "")))
            return await spawn_mesh_edit(
                db,
                parent,
                ops=ops,
                note=str(a.get("note") or "chat mesh edit"),
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("mesh_edit", spawn, bool(a.get("wait", True)))

    async def _tool_fem_check(self, a: dict[str, Any]) -> dict[str, Any]:
        async def spawn(db: Any) -> Job:
            v = await self._version(db, str(a.get("version_id", "")))
            return await spawn_fem(
                db,
                v,
                load_cases=a.get("load_cases") or [],
                material=str(a.get("material") or ""),
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("fem_check", spawn, bool(a.get("wait", True)))

    async def _tool_optimize_version(self, a: dict[str, Any]) -> dict[str, Any]:
        async def spawn(db: Any) -> tuple[Version, Job]:
            parent = await self._version(db, str(a.get("version_id", "")))
            return await spawn_optimize(
                db,
                parent,
                load_cases=a.get("load_cases") or [],
                optimize=a.get("optimize") or {},
                material=str(a.get("material") or ""),
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("optimize_version", spawn, bool(a.get("wait", True)))

    async def _tool_topopt_resample(self, a: dict[str, Any]) -> dict[str, Any]:
        async def spawn(db: Any) -> tuple[Version, Job]:
            parent = await self._version(db, str(a.get("version_id", "")))
            return await spawn_resample(
                db,
                parent,
                density_threshold=a.get("density_threshold"),
                min_member_mm=a.get("min_member_mm"),
                material=str(a.get("material") or ""),
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("topopt_resample", spawn, bool(a.get("wait", True)))

    async def _tool_refine_version(self, a: dict[str, Any]) -> dict[str, Any]:
        async def spawn(db: Any) -> tuple[Version, Job]:
            parent = await self._version(db, str(a.get("version_id", "")))
            return await spawn_refine(
                db,
                parent,
                features=a.get("features") or [],
                load_cases=a.get("load_cases") or [],
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("refine_version", spawn, bool(a.get("wait", True)))

    async def _tool_organic_restyle(self, a: dict[str, Any]) -> dict[str, Any]:
        try:
            organic = OrganicParams(
                style=str(a.get("style") or "gyroid"),
                volfrac=float(a["volfrac"]) if a.get("volfrac") is not None else 0.35,
                period_mm=float(a["period_mm"]) if a.get("period_mm") is not None else 12.0,
                shell_mm=float(a["shell_mm"]) if a.get("shell_mm") is not None else 0.0,
            ).model_dump(mode="json", exclude={"anchors"})
        except Exception as e:
            return {"ok": False, "error": f"invalid organic params: {e}"}

        async def spawn(db: Any) -> tuple[Version, Job]:
            parent = await self._version(db, str(a.get("version_id", "")))
            return await spawn_organic(db, parent, organic, enqueue_fn=self._enqueue_fn)

        return await self._spawn_and_wait("organic_restyle", spawn, bool(a.get("wait", True)))

    async def _tool_generate_image(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.image_left <= 0:
            return {"ok": False, "error": "image budget exhausted"}
        prompt = str(a.get("prompt", "")).strip()
        if not prompt:
            return {"ok": False, "error": "prompt is required"}
        count = max(1, min(2, int(a.get("count", 1) or 1)))
        ref_id = str(a.get("reference_asset_id") or "").strip()
        ref = None
        if ref_id:
            ref = next(
                (
                    x
                    for x in self.ctx.assets
                    if x.asset_id == ref_id and x.kind in IMAGE_INPUT_KINDS
                ),
                None,
            )
            if ref is None:
                return {"ok": False, "error": f"unknown image asset '{ref_id}'"}
        self.image_left -= 1
        try:
            from cadx.agent.concept_image import generate_concept_images

            out = await generate_concept_images(
                self.ctx,
                prompt,
                count=count,
                ref_asset=ref,
                strength=float(a["strength"]) if a.get("strength") is not None else None,
                source="chat",
            )
            return {
                "ok": True,
                "assets": [{"asset_id": x.asset_id, "filename": x.filename} for x in out],
                "note": "stored as generated_image assets — pass their ids to "
                "generate_part(asset_ids=...) for image→3D",
                "_images": [str(x.path) for x in out],
            }
        except Exception as e:
            log.warning("chat_tool.image_failed", error=str(e))
            return {"ok": False, "error": f"image generation failed: {e}"}

    async def _tool_answer_questions(self, a: dict[str, Any]) -> dict[str, Any]:
        answers = a.get("answers") or []
        if not isinstance(answers, list) or not answers:
            return {"ok": False, "error": "answers list required"}
        answers_text = "\n".join(
            f"A{i + 1}: {x.get('answer') if isinstance(x, dict) else x}"
            for i, x in enumerate(answers)
        )

        async def spawn(db: Any) -> tuple[Version, Job]:
            try:
                jid = uuid.UUID(str(a.get("job_id", "")))
            except ValueError:
                raise SpawnError("invalid job id", status=422) from None
            j = await db.get(Job, jid)
            if j is None:
                raise SpawnError("job not found", status=404)
            return await enqueue_resume(db, j, answers_text, enqueue_fn=self._enqueue_fn)

        return await self._spawn_and_wait("answer_questions", spawn, bool(a.get("wait", True)))

    async def _tool_wait_job(self, a: dict[str, Any]) -> dict[str, Any]:
        try:
            jid = uuid.UUID(str(a.get("job_id", "")))
        except ValueError:
            return {"ok": False, "error": "invalid job id"}
        return await self._wait(jid)

    async def _tool_save_library_part(self, a: dict[str, Any]) -> dict[str, Any]:
        name = str(a.get("name", "")).strip()
        if not name:
            return {"ok": False, "error": "name required"}
        async with session_scope() as db:
            v = await self._version(db, str(a.get("version_id", "")))
            if v.status.value not in ("succeeded", "needs_review"):
                return {"ok": False, "error": f"version is {v.status.value} — save a finished one"}
            caption = str(a.get("caption") or "")[:400]
            if a.get("as_template"):
                if not v.code:
                    return {"ok": False, "error": "version has no generated code"}
                part = LibraryPart(
                    name=name[:120],
                    kind="template",
                    code=v.code,
                    params=v.params,
                    caption=caption,
                    source_version_id=v.id,
                )
            else:
                rows = await db.execute(
                    select(Artifact).where(
                        Artifact.version_id == v.id, Artifact.kind == ArtifactKind.step
                    )
                )
                step = rows.scalars().first()
                if step is None:
                    return {"ok": False, "error": "version has no STEP export"}
                try:
                    src = storage.resolve(step.path)
                except ValueError:
                    return {"ok": False, "error": "invalid artifact path"}
                if not src.exists():
                    return {"ok": False, "error": "STEP artifact file missing"}
                import shutil

                part = LibraryPart(
                    name=name[:120],
                    kind="step",
                    filename=f"{storage.safe_filename(name)}.step",
                    caption=caption,
                    source_version_id=v.id,
                )
                dst = storage.library_part_path(part.id, part.filename)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                part.path = storage.relpath(dst)
            db.add(part)
            await db.commit()
            return {
                "ok": True,
                "part_id": str(part.id),
                "name": part.name,
                "kind": part.kind,
            }
