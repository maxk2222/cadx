"""Worker-side job runners: wire AgentCtx -> orchestrator -> DB.

Runs in the `worker` service (queue "agent"). CAD work is delegated to the
`cad` queue via QueueExecutor.
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadagent import events, storage
from cadagent.agent.executor_client import QueueExecutor
from cadagent.agent.mesh_client import MeshPort, make_mesh_client
from cadagent.agent.orchestrator import RunOutcome, run_generation
from cadagent.agent.trace import MemoryTrace, sanitize_messages
from cadagent.agent.types import AgentCtx, InputAsset, ParentContext
from cadagent.cad.mesh_ops import MeshPostParams, merge_post_params
from cadagent.db.models import (
    AgentTrace,
    Artifact,
    ArtifactKind,
    Asset,
    Job,
    JobStatus,
    JobType,
    LibraryPart,
    Project,
    ProjectSettings,
    Version,
    VersionStatus,
)
from cadagent.db.session import session_scope
from cadagent.llm.client import LLMClient
from cadagent.llm.schemas import PartSpec
from cadagent.logging import get_logger
from cadagent.runtime_config import RuntimeSettings, load_runtime_settings

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _u(s: str | uuid.UUID) -> uuid.UUID:
    return s if isinstance(s, uuid.UUID) else uuid.UUID(str(s))


async def _publish(job_id: str, event: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        await events.publish(job_id, event)


class DbTrace(MemoryTrace):
    """Memory trace that also persists each record as an AgentTrace row."""

    def __init__(self, session: AsyncSession, job_id: uuid.UUID) -> None:
        super().__init__()
        self._db = session
        self._job_id = job_id

    async def record(self, **kwargs: Any) -> None:
        await super().record(**kwargs)
        self._db.add(
            AgentTrace(
                job_id=self._job_id,
                step=kwargs["step"],
                iteration=kwargs["iteration"],
                model=kwargs["model"],
                messages=sanitize_messages(kwargs["messages"]),
                response=kwargs["response_text"][:20_000],
                parsed=kwargs["parsed"],
                prompt_tokens=kwargs["prompt_tokens"],
                completion_tokens=kwargs["completion_tokens"],
                latency_ms=kwargs["latency_ms"],
            )
        )
        await self._db.commit()


def register_artifacts(
    session: AsyncSession,
    version_id: uuid.UUID,
    artifacts: dict[str, str],
    *,
    tessellated_step: bool = False,
) -> None:
    """artifacts: {"render:front": relpath, "export:step": relpath, "contact_sheet": rel, ...}"""
    for key, rel in artifacts.items():
        p = storage.resolve(rel)
        size = p.stat().st_size if p.exists() else 0
        if key.startswith("render:"):
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind.render,
                    view=key.split(":", 1)[1],
                    path=rel,
                    size=size,
                )
            )
        elif key.startswith("export:"):
            fmt = key.split(":", 1)[1]
            meta = {"tessellated": True} if fmt == "step" and tessellated_step else {}
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind(fmt),
                    path=rel,
                    size=size,
                    meta=meta,
                )
            )
        elif key == "mesh_raw":
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind.mesh_raw,
                    path=rel,
                    size=size,
                )
            )
        elif key.startswith("fem:"):
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind.fem,
                    view=key.split(":", 1)[1],
                    path=rel,
                    size=size,
                )
            )
        elif key in ("contact_sheet", "thumbnail"):
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind(key),
                    path=rel,
                    size=size,
                )
            )


def _input_assets(assets: list[Asset]) -> list[InputAsset]:
    return [
        InputAsset(
            asset_id=str(a.id),
            kind=a.kind.value,
            path=storage.resolve(a.path),
            caption=a.caption,
            filename=a.filename,
        )
        for a in assets
    ]


async def _load_library(
    db: AsyncSession, version: Version
) -> tuple[list[InputAsset], list[dict[str, Any]]]:
    """Resolve version.library_part_ids into extra reference assets + templates."""
    extra_assets: list[InputAsset] = []
    templates: list[dict[str, Any]] = []
    if not version.library_part_ids:
        return extra_assets, templates
    rows = await db.execute(
        select(LibraryPart).where(
            col(LibraryPart.id).in_([_u(p) for p in version.library_part_ids])
        )
    )
    for part in rows.scalars().all():
        if part.kind == "template":
            templates.append(
                {
                    "name": part.name,
                    "caption": part.caption,
                    "code": part.code or "",
                    "params": part.params or {},
                }
            )
        elif part.path:
            extra_assets.append(
                InputAsset(
                    asset_id=str(part.id),
                    kind="reference_step" if part.kind == "step" else "reference_stl",
                    path=storage.resolve(part.path),
                    caption=f"{part.name} — {part.caption}".rstrip(" —"),
                    filename=part.filename,
                )
            )
    return extra_assets, templates


def _parent_ctx(version: Version, parent: Version | None) -> ParentContext:
    if parent is None:
        return ParentContext()
    v_dir = storage.version_dir(parent.project_id, parent.id)
    sheet = v_dir / "contact_sheet.png"
    solid = v_dir / "mesh" / "solid.brep"
    return ParentContext(
        code=parent.code or "",
        spec=PartSpec(**parent.spec) if parent.spec else None,
        contact_sheet=sheet if sheet.exists() else None,
        number=parent.number,
        resolved_mode=parent.resolved_mode,
        mesh=parent.mesh,
        mesh_solid=solid if solid.exists() else None,
    )


# --------------------------------------------------------------------------- #
# job runners                                                                  #
# --------------------------------------------------------------------------- #


async def run_agent_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq function on queue 'agent' for types generate|edit|resume."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False, "error": "job or version not found"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        async def emit(step: str, iteration: int, message: str) -> None:
            async with session_scope() as s2:
                j = await s2.get(Job, job_id)
                if j is not None:
                    j.step = step
                    j.iteration = iteration
                    j.progress = min(0.95, j.progress + 0.05)
                    j.log = (j.log or "") + f"[{step}:{iteration}] {message}\n"
                    await s2.commit()
            with contextlib.suppress(Exception):
                await events.publish(
                    str(job_id),
                    {"type": "step", "step": step, "iteration": iteration, "message": message},
                )

        async def check_cancel() -> bool:
            async with session_scope() as s3:
                j = await s3.get(Job, job_id)
                return bool(j and j.cancel_requested)

        runtime_settings = await load_runtime_settings(db)
        llm = _make_llm(runtime_settings)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        trace = DbTrace(db, job_id)
        mesh = _make_mesh(runtime_settings)

        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            _merge_settings((project.settings or {}) if project else {}, version.options or {}),
            runtime_settings,
        )

        # assets
        asset_ids = version.asset_ids or []
        assets: list[Asset] = []
        if asset_ids:
            rows = await db.execute(
                select(Asset).where(col(Asset.id).in_([_u(a) for a in asset_ids]))
            )
            assets = list(rows.scalars().all())
        else:
            rows = await db.execute(select(Asset).where(Asset.project_id == version.project_id))
            assets = list(rows.scalars().all())

        # reusable library parts selected for this version: file parts become
        # reference assets (loadable via load_reference(part_id)), code
        # templates are surfaced in the codegen context only
        input_assets = _input_assets(assets)
        extra_assets, library_templates = await _load_library(db, version)
        input_assets += extra_assets

        parent = await db.get(Version, version.parent_id) if version.parent_id else None
        parent_ctx = _parent_ctx(version, parent)

        mode = "edit" if job.type == JobType.edit else "generate"
        gen_mode = version.gen_mode or settings.gen_mode or runtime_settings.agent_default_gen_mode
        # hybrid edit keeps the parent's solid unless the user asked for regeneration
        regenerate_mesh = bool(job.resume_state and job.resume_state.get("regenerate_mesh"))
        if mode == "edit" and gen_mode in ("auto", "mesh", "hybrid"):
            effective = gen_mode if gen_mode != "auto" else (parent_ctx.resolved_mode or "cad")
            if effective == "hybrid" and parent_ctx.resolved_mode == "hybrid":
                gen_mode = "hybrid"
                if regenerate_mesh:
                    parent_ctx.mesh_solid = None
            elif effective == "mesh" and parent_ctx.resolved_mode == "mesh":
                gen_mode = "mesh"

        mesh_client: MeshPort | None = mesh
        mesh_available = False
        if mesh_client is not None:
            try:
                mesh_available = (await mesh_client.health()).ok
            except Exception:
                mesh_available = False

        actx = AgentCtx(
            llm=llm,
            executor=executor,
            trace=trace,
            emit=emit,
            check_cancel=check_cancel,
            job_id=str(job_id),
            project_id=version.project_id,
            version_id=version.id,
            mode=mode,
            prompt=version.prompt,
            settings=settings,
            runtime_settings=runtime_settings,
            assets=input_assets,
            library_templates=library_templates,
            parent=parent_ctx,
            gen_mode=gen_mode,
            mesh=mesh_client,
            mesh_available=mesh_available,
            budget_tokens=runtime_settings.agent_max_tokens_per_job,
        )
        if job.type == JobType.resume and job.resume_state and job.resume_state.get("spec"):
            actx.spec = PartSpec(**job.resume_state["spec"])

        try:
            outcome = await run_generation(actx)
        except Exception as e:
            log.exception("agent.job_crashed", job=str(job_id))
            outcome = RunOutcome(status="failed", error=f"agent crashed: {e}")

        if actx.library_attached:
            # record library parts the agent selected itself via tools
            version.library_part_ids = sorted(
                set(version.library_part_ids) | set(actx.library_attached)
            )

        await _persist_outcome(db, job, version, outcome, emit)
        return {"ok": outcome.status in ("succeeded", "needs_review"), "status": outcome.status}


async def run_params_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Re-execute parent with new params (no LLM). Child version already exists.

    CAD/hybrid parents: code re-run with PARAMS overrides.
    Mesh parents: mesh pipeline re-run with merged post-parameters.
    """
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = await db.get(Version, version.parent_id) if version and version.parent_id else None
        parent_mode = (parent.resolved_mode if parent else None) or "cad"
        if (
            job is None
            or version is None
            or parent is None
            or (parent_mode != "mesh" and not parent.code)
        ):
            if job:
                job.status = JobStatus.failed
                job.log = (job.log or "") + "params job: missing parent code/mesh\n"
                await db.commit()
            return {"ok": False, "error": "missing parent code/mesh"}

        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        async def emit(step: str, iteration: int, message: str) -> None:
            async with session_scope() as s2:
                j = await s2.get(Job, job_id)
                if j is not None:
                    j.step = step
                    j.log = (j.log or "") + f"[{step}] {message}\n"
                    await s2.commit()
            with contextlib.suppress(Exception):
                await events.publish(
                    str(job_id),
                    {"type": "step", "step": step, "iteration": iteration, "message": message},
                )

        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            _merge_settings(
                (project.settings or {}) if project else {},
                version.options or (parent.options if parent else {}) or {},
            ),
            runtime_settings,
        )

        if parent_mode == "mesh":
            exec_result = await _run_mesh_params(
                executor,
                job_id,
                version,
                parent,
                settings,
                runtime_settings,
                payload.get("params") or {},
                emit,
            )
        else:
            await emit("execute", 1, "re-executing with new parameters")
            parent_v_dir = storage.version_dir(parent.project_id, parent.id)
            solid = parent_v_dir / "mesh" / "solid.brep"
            exec_result = await executor.execute(
                {
                    "job_id": str(job_id),
                    "project_id": str(version.project_id),
                    "version_id": str(version.id),
                    "iteration": 1,
                    "code": parent.code,
                    "params_override": payload.get("params") or {},
                    "references": [],
                    "settings": settings.model_dump(),
                    "runtime_settings": runtime_settings.executor_values(),
                    "full_exports": True,
                    "name": "model",
                    "mesh_solid": str(solid)
                    if solid.exists() and parent_mode == "hybrid"
                    else None,
                    "mesh_bbox": (parent.mesh or {}).get("bbox_size"),
                }
            )

        if not exec_result.get("ok"):
            version.status = VersionStatus.failed
            version.error = exec_result.get("error") or "execution failed"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            job.log = (job.log or "") + f"failed: {version.error}\n"
            await db.commit()
            return {"ok": False, "error": version.error}

        # copy artifacts to version root

        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, exec_result)
        for fmt, rel in (exec_result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        tessellated = parent_mode == "mesh"

        version.gen_mode = parent_mode
        version.resolved_mode = parent_mode
        if parent_mode == "mesh":
            # merged post params live under mesh.post in the exec result
            version.params = (exec_result.get("mesh") or {}).get("post") or version.params
        else:
            version.params = exec_result.get("params_used") or version.params
        version.params_schema = parent.params_schema
        version.code = parent.code
        version.analysis = exec_result.get("analysis")
        version.dfm = exec_result.get("dfm")
        version.iterations = 1
        version.status = VersionStatus.succeeded

        import json

        if parent_mode == "mesh":
            # mesh params child: inherit mesh provenance + copy intermediates;
            # overlay the fresh run's mesh info (post, triangles, ops_report)
            version.mesh = dict(parent.mesh or {})
            version.mesh.update(exec_result.get("mesh") or {})
            version.mesh["post"] = version.params
            version.mesh["from_params_of"] = str(parent.id)
            artifacts.update(
                _copy_mesh_artifacts_ctxless(version.project_id, version.id, 1, parent)
            )
            m_dir = storage.ensure(storage.version_dir(version.project_id, version.id) / "mesh")
            (m_dir / "mesh.json").write_text(json.dumps(version.mesh, indent=2), encoding="utf-8")
        register_artifacts(db, version.id, artifacts, tessellated_step=tessellated)

        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        if version.code:
            (v_dir / "model.py").write_text(version.code, encoding="utf-8")
        (v_dir / "params.json").write_text(
            json.dumps(version.params or {}, indent=2), encoding="utf-8"
        )
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        job.log = (job.log or "") + "done\n"
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True}


async def _run_mesh_params(
    executor: QueueExecutor,
    job_id: uuid.UUID,
    version: Version,
    parent: Version,
    settings: ProjectSettings,
    runtime_settings: RuntimeSettings,
    overrides: dict[str, Any],
    emit: Any,
) -> dict[str, Any]:
    """Mesh-mode params job: re-run the post-pipeline on the parent's raw mesh."""
    parent_mesh_dir = storage.version_dir(parent.project_id, parent.id) / "mesh"
    raw = parent_mesh_dir / "mesh_raw.glb"
    if not raw.exists():
        return {"ok": False, "error": "parent mesh_raw.glb not found"}

    base = MeshPostParams(**(parent.params or {}))
    post = merge_post_params(base, overrides)
    await emit("meshprocess", 1, f"re-processing mesh ({overrides})")

    return await executor.process_mesh(
        {
            "job_id": str(job_id),
            "project_id": str(version.project_id),
            "version_id": str(version.id),
            "iteration": 1,
            "mode": "final",
            "mesh_rel": storage.relpath(raw),
            "post": post.model_dump(),
            "spec": parent.spec,
            "settings": settings.model_dump(),
            "runtime_settings": runtime_settings.executor_values(),
            "full_exports": True,
            "name": "model",
        }
    )


def _parent_stl(parent: Version) -> str | None:
    """Best available triangle-mesh artifact of a version (relpath)."""
    v_dir = storage.version_dir(parent.project_id, parent.id)
    for cand in (
        v_dir / "exports" / "model.stl",
        v_dir / "mesh" / "mesh_clean.stl",
    ):
        if cand.exists():
            return storage.relpath(cand)
    # last iteration's working STL (cad: renders/mesh.stl, mesh: _run/model.stl)
    it_dir = storage.iteration_dir(parent.project_id, parent.id, parent.iterations or 1)
    for cand in (
        it_dir / "_run" / "model.stl",
        it_dir / "renders" / "mesh.stl",
        it_dir / "_run" / "mesh.stl",
    ):
        if cand.exists():
            return storage.relpath(cand)
    return None


async def run_optimize_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Manual topology optimization (LLM-free, like params): child version gets
    the optimized mesh + FEM report + artifacts."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = await db.get(Version, version.parent_id) if version and version.parent_id else None
        if job is None or version is None or parent is None:
            if job:
                job.status = JobStatus.failed
                job.log = (job.log or "") + "optimize job: missing parent\n"
                await db.commit()
            return {"ok": False, "error": "missing parent version"}

        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        async def emit(step: str, iteration: int, message: str) -> None:
            async with session_scope() as s2:
                j = await s2.get(Job, job_id)
                if j is not None:
                    j.step = step
                    j.progress = min(0.95, j.progress + 0.03)
                    j.log = (j.log or "") + f"[{step}] {message}\n"
                    await s2.commit()
            with contextlib.suppress(Exception):
                await events.publish(
                    str(job_id),
                    {"type": "step", "step": step, "iteration": iteration, "message": message},
                )

        mesh_rel = _parent_stl(parent)
        if mesh_rel is None:
            version.status = VersionStatus.failed
            version.error = "parent has no mesh artifact to optimize"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        # load cases: request payload -> parent spec -> none
        spec = PartSpec(**parent.spec) if parent.spec else None
        load_cases = payload.get("load_cases") or [
            c.model_dump() for c in (spec.load_cases if spec else [])
        ]
        if not load_cases:
            version.status = VersionStatus.failed
            version.error = "no load cases — declare loads in the prompt or the optimize dialog"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        runtime_settings = await load_runtime_settings(db)
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            ProjectSettings(**(project.settings or {})) if project else ProjectSettings(),
            runtime_settings,
        )
        optimize = {
            **(spec.optimize.model_dump() if spec and spec.optimize else {}),
            **(
                {"target_volume_fraction": settings.topopt_volfrac}
                if settings.topopt_volfrac
                else {}
            ),
            "max_iterations": settings.topopt_iterations,
            **(payload.get("optimize") or {}),
        }
        material = (
            payload.get("material") or settings.fem_material or (spec.material if spec else "")
        )

        await emit("optimize", 1, f"SIMP on {mesh_rel} ({len(load_cases)} load case(s))")
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        result = await executor.optimize(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": 1,
                "mesh_rel": mesh_rel,
                "load_cases": load_cases,
                "optimize": optimize,
                "material": material,
                "voxel_mm": settings.fem_voxel_mm,
                "runtime_settings": runtime_settings.executor_values(),
                "name": "model",
            }
        )

        if not result.get("ok"):
            version.status = VersionStatus.failed
            version.error = result.get("error") or "optimization failed"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            job.log = (job.log or "") + f"failed: {version.error}\n"
            await db.commit()
            return {"ok": False, "error": version.error}

        # persist: artifacts + fem report + topopt metadata on the child version
        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, result)
        for fmt, rel in (result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        it_dir = storage.iteration_dir(version.project_id, version.id, 1)
        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        fem_dir = storage.ensure(v_dir / "fem")
        fem_art = 0
        import shutil

        for png in (it_dir / "fem").glob("*.png") if (it_dir / "fem").exists() else []:
            shutil.copy2(png, fem_dir / png.name)
            artifacts[f"fem:{png.stem}"] = storage.relpath(fem_dir / png.name)
            fem_art += 1
        for extra in ("topopt/density.npz", "topopt/history.json", "fem/report.json"):
            src = it_dir / extra
            if src.exists():
                dst = (
                    fem_dir / src.name
                    if src.parent.name == "fem"
                    else storage.ensure(v_dir / "topopt") / src.name
                )
                shutil.copy2(src, dst)
                artifacts[f"fem:{src.stem}"] = storage.relpath(dst)
                fem_art += 1

        topopt = result.get("topopt") or {}
        fem = result.get("fem") or {}
        version.gen_mode = "topopt"
        version.resolved_mode = "topopt"
        version.spec = parent.spec
        version.fem = fem
        version.mesh = {
            **(result.get("mesh") or {}),
            "topopt": topopt,
            "from_optimize_of": str(parent.id),
        }
        version.analysis = result.get("analysis")
        version.params = {"optimize": optimize, "load_cases": load_cases, "material": material}
        version.iterations = 1
        # structural review: flag unsafe results instead of silent success
        sfs = [c.get("safety_factor") for c in fem.get("cases", []) if c.get("safety_factor")]
        worst_sf = min(sfs) if sfs else None
        version.status = (
            VersionStatus.needs_review
            if (worst_sf is not None and worst_sf < 1.0)
            else VersionStatus.succeeded
        )
        register_artifacts(db, version.id, artifacts, tessellated_step=False)
        await emit(
            "optimize",
            1,
            f"done: volfrac {topopt.get('volfrac_achieved')}, SF {worst_sf}, "
            f"{topopt.get('iterations')} iterations",
        )
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        job.log = (job.log or "") + "done\n"
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True}


async def run_reexport_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Re-export formats from the best iteration BREP."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False}
        job.status = JobStatus.running
        await db.commit()
        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        result = await executor.export_all(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": payload.get("iteration") or 1,
                "name": "model",
                "runtime_settings": runtime_settings.executor_values(),
            }
        )
        artifacts = {}
        for fmt, rel in (result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        register_artifacts(db, version.id, artifacts)
        job.status = JobStatus.succeeded if result.get("ok") else JobStatus.failed
        job.finished_at = datetime.now(UTC)
        await db.commit()
        return result


# --------------------------------------------------------------------------- #
# internals                                                                    #
# --------------------------------------------------------------------------- #


def _merge_settings(
    project_settings: dict[str, Any], version_options: dict[str, Any]
) -> ProjectSettings:
    """project defaults < per-run options (GUI options panel)."""
    s = ProjectSettings(**{**project_settings, **version_options})
    if s.fem and not s.fem_material and s.material:
        s.fem_material = s.material
    return s


def _fill_settings(settings: ProjectSettings, runtime: RuntimeSettings) -> ProjectSettings:
    for key, default in (
        ("clarification_mode", runtime.agent_clarification_mode),
        ("max_iterations", runtime.agent_max_iterations),
        ("max_repairs", runtime.agent_max_repairs),
        ("pass_score", runtime.agent_pass_score),
        ("dim_tolerance_pct", runtime.agent_dim_tolerance_pct),
        ("dim_tolerance_mm", runtime.agent_dim_tolerance_mm),
        ("fem_voxel_mm", runtime.fem_voxel_mm),
        ("topopt_volfrac", runtime.topopt_default_volfrac),
        ("topopt_iterations", runtime.topopt_max_iterations),
    ):
        if getattr(settings, key) is None:
            setattr(settings, key, default)
    if not settings.mesh_model:
        settings.mesh_model = runtime.mesh_model
    return settings


def _make_mesh(runtime: RuntimeSettings) -> MeshPort | None:
    return make_mesh_client(
        base_url=runtime.mesh_base_url,
        timeout_s=runtime.mesh_timeout_s,
    )


def _make_llm(runtime: RuntimeSettings) -> LLMClient:
    return LLMClient(runtime_settings=runtime)


def _copy_best_artifacts_ctxless(
    project_id: uuid.UUID, version_id: uuid.UUID, iteration: int, exec_result: dict[str, Any]
) -> dict[str, str]:
    """Same as orchestrator._copy_best_artifacts but without an AgentCtx."""
    import shutil
    from pathlib import Path

    it_dir = storage.iteration_dir(project_id, version_id, iteration)
    v_dir = storage.version_dir(project_id, version_id)
    out: dict[str, str] = {}
    r_dir = storage.ensure(v_dir / "renders")
    for view, rel in (exec_result.get("renders") or {}).items():
        src = it_dir / "renders" / Path(rel).name
        if src.exists():
            dst = r_dir / f"{view}.png"
            shutil.copy2(src, dst)
            out[f"render:{view}"] = storage.relpath(dst)
    for name, kind in (("contact_sheet.png", "contact_sheet"), ("thumbnail.png", "thumbnail")):
        src = it_dir / name
        if src.exists():
            dst = v_dir / name
            shutil.copy2(src, dst)
            out[kind] = storage.relpath(dst)
    if exec_result.get("dfm"):
        import json

        (v_dir / "dfm.json").write_text(json.dumps(exec_result["dfm"], indent=2), encoding="utf-8")
    return out


def _copy_mesh_artifacts_ctxless(
    project_id: uuid.UUID, version_id: uuid.UUID, iteration: int, parent: Version | None = None
) -> dict[str, str]:
    """Version-level mesh intermediates for params children (no AgentCtx)."""
    import shutil

    it_dir = storage.iteration_dir(project_id, version_id, iteration)
    mesh_dir = storage.ensure(storage.version_dir(project_id, version_id) / "mesh")
    out: dict[str, str] = {}
    parent_mesh = storage.version_dir(parent.project_id, parent.id) / "mesh" if parent else None
    for src, dst_name, key in (
        (it_dir / "mesh_raw.glb", "mesh_raw.glb", "mesh_raw"),
        (it_dir / "_run" / "model.stl", "mesh_clean.stl", "mesh_clean"),
        (it_dir / "_run" / "model.brep", "solid.brep", "mesh_solid"),
    ):
        src_path = src
        if not src_path.exists() and parent_mesh is not None:
            # params children don't regenerate the raw mesh — inherit from parent
            candidate = parent_mesh / dst_name
            if candidate.exists() and key == "mesh_raw":
                src_path = candidate
        if src_path.exists():
            dst = mesh_dir / dst_name
            shutil.copy2(src_path, dst)
            out[key] = storage.relpath(dst)
    return out


async def _persist_outcome(
    db: AsyncSession, job: Job, version: Version, outcome: RunOutcome, emit: Any
) -> None:
    job.finished_at = datetime.now(UTC)
    if outcome.status == "waiting_for_input":
        job.status = JobStatus.waiting_for_input
        job.questions = outcome.questions
        job.resume_state = outcome.resume_state
        version.status = VersionStatus.waiting_for_input
        version.spec = outcome.spec
        await db.commit()
        await _publish(str(job.id), {"type": "question", "questions": outcome.questions})
        return

    version.spec = outcome.spec
    version.plan = outcome.plan
    version.code = outcome.code
    version.params = outcome.params
    version.params_schema = outcome.params_schema
    version.analysis = outcome.analysis
    version.critique = outcome.critique
    version.dfm = outcome.dfm
    version.bom = outcome.bom
    version.issues = outcome.issues or None
    version.mesh = outcome.mesh
    version.fem = outcome.fem
    version.resolved_mode = outcome.resolved_mode
    version.score = outcome.score
    version.iterations = outcome.iterations
    version.usage = outcome.usage
    version.error = outcome.error
    tessellated = outcome.resolved_mode == "mesh" or bool(
        outcome.mesh and outcome.mesh.get("step_tessellated")
    )
    register_artifacts(db, version.id, outcome.artifacts, tessellated_step=tessellated)
    if outcome.mode_note:
        job.log = (job.log or "") + f"mode: {outcome.mode_note}\n"

    if outcome.code or outcome.params is not None:
        import json

        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        if outcome.code:
            (v_dir / "model.py").write_text(outcome.code, encoding="utf-8")
        (v_dir / "params.json").write_text(
            json.dumps(outcome.params or {}, indent=2), encoding="utf-8"
        )

    mapping = {
        "succeeded": (VersionStatus.succeeded, JobStatus.succeeded),
        "needs_review": (VersionStatus.needs_review, JobStatus.succeeded),
        "failed": (VersionStatus.failed, JobStatus.failed),
        "cancelled": (VersionStatus.cancelled, JobStatus.cancelled),
    }
    v_status, j_status = mapping.get(outcome.status, (VersionStatus.failed, JobStatus.failed))
    version.status = v_status
    job.status = j_status
    job.progress = 1.0
    job.log = (job.log or "") + f"final: {outcome.status}\n"
    await db.commit()
    await _publish(str(job.id), {"type": "done", "status": outcome.status, "score": outcome.score})
