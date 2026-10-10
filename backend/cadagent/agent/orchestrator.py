"""Agent orchestrator: the bounded generate-verify-revise state machine.

INTAKE -> (CLARIFY?) -> PLAN -> mode resolution -> one of:
  cad:    [CODEGEN -> EXECUTE -> (REPAIR x N) -> ANALYZE -> CRITIQUE] x K -> FINALIZE
  mesh:   [MESH_GENERATE -> ORIENT(once) -> PROCESS_MESH -> ANALYZE -> CRITIQUE] x K -> FINALIZE
  hybrid: MESH_GENERATE -> ORIENT -> PROCESS_MESH(+solid) -> cad loop with load_mesh_solid()
"""

from __future__ import annotations

import itertools
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cadagent import storage
from cadagent.agent.mesh_client import MeshGenError, MeshGenRequest
from cadagent.agent.steps import codegen as codegen_step
from cadagent.agent.steps import critique as critique_step
from cadagent.agent.steps import intake as intake_step
from cadagent.agent.steps import mesh_orient as mesh_orient_step
from cadagent.agent.steps import plan as plan_step
from cadagent.agent.types import AgentCtx
from cadagent.cad.mesh_ops import MeshPostParams, merge_post_params, mesh_params_schema
from cadagent.config import get_settings
from cadagent.llm.schemas import CodeResponse, Critique, PartSpec
from cadagent.logging import get_logger

log = get_logger(__name__)


@dataclass
class Candidate:
    iteration: int
    code_response: CodeResponse | None
    exec_result: dict[str, Any]
    critique: Critique | None
    dim_issues: list[str]
    mesh_post: dict[str, Any] | None = None  # MeshPostParams used (mesh/hybrid modes)

    @property
    def rank(self) -> tuple[int, int, int]:
        major = (
            sum(1 for d in self.critique.discrepancies if d.severity == "major")
            if self.critique
            else 99
        )
        score = self.critique.score if self.critique else 0
        return (0 if self.dim_issues else 1, -major, score)


@dataclass
class RunOutcome:
    status: str = "failed"  # succeeded | needs_review | failed | waiting_for_input | cancelled
    spec: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    code: str | None = None
    params: dict[str, Any] | None = None
    params_schema: list[dict[str, Any]] | None = None
    analysis: dict[str, Any] | None = None
    critique: dict[str, Any] | None = None
    dfm: dict[str, Any] | None = None
    mesh: dict[str, Any] | None = None
    resolved_mode: str | None = None
    mode_note: str = ""
    score: int | None = None
    iterations: int = 0
    best_iteration: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # kind|view -> relpath
    questions: list[dict[str, Any]] | None = None
    resume_state: dict[str, Any] | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def compare_dims(
    spec: PartSpec, analysis: dict[str, Any], tol_pct: float, tol_mm: float
) -> list[str]:
    """Numeric axis-wise dimension check + swapped-axes detection."""
    issues: list[str] = []
    actual = [float(v) for v in analysis["bbox_size"]]
    stated: list[tuple[Any, int, str]] = [
        (spec.length_x, 0, "length/X"),
        (spec.width_y, 1, "width/Y"),
        (spec.height_z, 2, "height/Z"),
    ]
    stated_dims = [d for d, _, _ in stated if d.source == "stated"]
    for dim, idx, name in stated:
        if dim.source != "stated":
            continue
        tol = max(dim.value * tol_pct / 100.0, tol_mm)
        if abs(actual[idx] - dim.value) > tol:
            issues.append(
                f"{name}: spec {dim.value:g} mm vs actual {actual[idx]:.2f} mm (tol {tol:.2f})"
            )
    if issues and len(stated_dims) >= 2:
        tol = lambda v: max(v * tol_pct / 100.0, tol_mm)  # noqa: E731
        for perm in itertools.permutations(actual):
            if all(abs(perm[i] - d.value) <= tol(d.value) for i, d in enumerate(stated_dims)):
                issues.append(
                    "note: dimensions match under a different axis order — "
                    "check axis assignment (X/Y possibly swapped)"
                )
                break
    return issues


def build_feedback(
    critique: Critique | None, dim_issues: list[str], dfm: dict[str, Any] | None
) -> str:
    items: list[str] = []
    items += [f"dimension mismatch — {i}" for i in dim_issues]
    if critique:
        for d in critique.discrepancies:
            hint = f" ({d.fix_hint})" if d.fix_hint else ""
            items.append(f"[{d.severity}] {d.what} @ {d.where}{hint}")
    if dfm:
        for w in dfm.get("warnings", []):
            if w.get("severity") == "high":
                items.append(f"dfm: {w['message']}")
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items[:12]))


def resolve_mode(ctx: AgentCtx) -> tuple[str, str]:
    """Decide the effective generation mode (cad | mesh | hybrid) + reason.

    Requested mode comes from ctx.gen_mode ("auto" defers to the planner's
    plan.approach). Guards: mesh/hybrid need image input AND a reachable mesh
    server — otherwise we fall back to CAD.
    """
    requested = ctx.gen_mode or "auto"
    has_images = any(a.kind == "image" for a in ctx.assets)
    approach = ctx.plan.approach if ctx.plan and ctx.plan.approach else "cad"
    mesh_ok = ctx.mesh_available

    if requested == "cad":
        return "cad", "CAD mode selected"
    if requested in ("mesh", "hybrid"):
        if not has_images:
            return "cad", f"mode '{requested}' needs image input — falling back to CAD"
        if not mesh_ok:
            return (
                "cad",
                f"mode '{requested}' selected but mesh server is unreachable — falling back to CAD",
            )
        return requested, f"{requested} mode selected"
    # auto
    if not has_images:
        return "cad", "auto: no image input — CAD path"
    if approach in ("mesh", "hybrid") and not mesh_ok:
        return (
            "cad",
            f"auto: planner suggested '{approach}' but mesh server is unreachable — CAD fallback",
        )
    if approach in ("mesh", "hybrid"):
        reason = ctx.plan.approach_reason if ctx.plan else ""
        return approach, f"auto: planner chose '{approach}'" + (f" — {reason}" if reason else "")
    return "cad", "auto: planner chose CAD"


def _exec_payload(
    ctx: AgentCtx, code: str, iteration: int, *, full_exports: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": ctx.job_id,
        "project_id": str(ctx.project_id),
        "version_id": str(ctx.version_id),
        "iteration": iteration,
        "code": code,
        "params_override": None,
        "references": [
            {"asset_id": a.asset_id, "path": str(a.path)}
            for a in ctx.assets
            if a.kind in ("reference_step", "reference_stl")
        ],
        "settings": ctx.settings.model_dump(),
        "full_exports": full_exports,
        "name": "model",
    }
    if ctx.hybrid_solid is not None:
        payload["mesh_solid"] = str(ctx.hybrid_solid)
        payload["mesh_bbox"] = ctx.hybrid_mesh_info.get("bbox_size")
    return payload


def _mesh_payload(
    ctx: AgentCtx,
    iteration: int,
    *,
    mode: str,
    mesh_rel: str | None = None,
    post: MeshPostParams | None = None,
    spec: PartSpec | None = None,
    make_solid: bool = False,
    full_exports: bool = False,
) -> dict[str, Any]:
    return {
        "job_id": ctx.job_id,
        "project_id": str(ctx.project_id),
        "version_id": str(ctx.version_id),
        "iteration": iteration,
        "mode": mode,
        "mesh_rel": mesh_rel,
        "post": (post or MeshPostParams()).model_dump(),
        "spec": spec.model_dump() if spec else None,
        "settings": ctx.settings.model_dump(),
        "make_solid": make_solid,
        "full_exports": full_exports,
        "name": "model",
    }


def _copy_best_artifacts(
    ctx: AgentCtx, iteration: int, exec_result: dict[str, Any]
) -> dict[str, str]:
    """Copy renders/contact sheet/thumbnail from the winning iteration to version root."""
    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration)
    v_dir = storage.version_dir(ctx.project_id, ctx.version_id)
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
        dfm_path = v_dir / "dfm.json"
        dfm_path.write_text(json.dumps(exec_result["dfm"], indent=2), encoding="utf-8")
        out["dfm_json"] = storage.relpath(dfm_path)
    return out


def _copy_mesh_artifacts(project_id: Any, version_id: Any, iteration: int) -> dict[str, str]:
    """Copy the version-level mesh intermediates (raw GLB, cleaned STL, solid BREP).

    Files already present in versions/<vid>/mesh/ (e.g. inherited from a parent
    on hybrid reuse) are registered as artifacts too.
    """
    it_dir = storage.iteration_dir(project_id, version_id, iteration)
    mesh_dir = storage.ensure(storage.version_dir(project_id, version_id) / "mesh")
    out: dict[str, str] = {}
    for src, dst_name, key in (
        (it_dir / "mesh_raw.glb", "mesh_raw.glb", "mesh_raw"),
        (it_dir / "_run" / "model.stl", "mesh_clean.stl", "mesh_clean"),
        (it_dir / "_run" / "model.brep", "solid.brep", "mesh_solid"),
    ):
        dst = mesh_dir / dst_name
        if src.exists():
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            out[key] = storage.relpath(dst)
        elif dst.exists():
            out[key] = storage.relpath(dst)
    return out


def _write_mesh_json(project_id: Any, version_id: Any, info: dict[str, Any]) -> None:
    mesh_dir = storage.ensure(storage.version_dir(project_id, version_id) / "mesh")
    clean_info = {k: v for k, v in info.items() if not k.startswith("_")}
    (mesh_dir / "mesh.json").write_text(json.dumps(clean_info, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# mesh generation helpers                                                      #
# --------------------------------------------------------------------------- #


def _mesh_source_images(ctx: AgentCtx, iteration: int) -> tuple[list[Any], list[str]]:
    """Pick input images for the generator: planner's mesh_source_images order,
    hero first; later iterations rotate the hero among image assets."""
    images = [a for a in ctx.assets if a.kind == "image"]
    if not images:
        return [], []
    by_id = {a.asset_id: a for a in images}
    ordered: list[Any] = []
    if ctx.plan:
        for aid in ctx.plan.mesh_source_images:
            if aid in by_id and by_id[aid] not in ordered:
                ordered.append(by_id[aid])
    ordered += [a for a in images if a not in ordered]
    if iteration > 1 and len(ordered) > 1:
        ordered = ordered[1:] + ordered[:1]  # rotate hero
    roles = ["front"] + ["other"] * (len(ordered) - 1)
    return ordered, roles


async def _generate_mesh(ctx: AgentCtx, iteration: int, spec: PartSpec, plan: Any) -> str | None:
    """MESH_GENERATE step. Returns the relpath of the raw GLB or None."""
    if ctx.mesh is None:
        ctx.mesh_info["gen_error"] = "mesh server not configured (MESH_BASE_URL)"
        return None

    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration)
    out_path = it_dir / "mesh_raw.glb"
    rel = storage.relpath(out_path)

    # reuse an already-generated raw mesh (hybrid->mesh fallback, edit re-run)
    if iteration == 1 and ctx.mesh_raw_reuse is not None and ctx.mesh_raw_reuse.exists():
        it_dir.mkdir(parents=True, exist_ok=True)
        if ctx.mesh_raw_reuse.resolve() != out_path.resolve():
            shutil.copy2(ctx.mesh_raw_reuse, out_path)
        return rel

    images, roles = _mesh_source_images(ctx, iteration)
    if not images:
        ctx.mesh_info["gen_error"] = "no image input for mesh generation"
        return None

    s = get_settings()
    base_seed = ctx.settings.mesh_seed
    if base_seed is None:
        base_seed = int(str(ctx.version_id).replace("-", "")[:8], 16) % 100_000
    seed = base_seed + iteration - 1
    model = ctx.settings.mesh_model or s.mesh_model
    subject = (plan.mesh_subject if plan else "") or spec.name or ctx.prompt[:80]

    req = MeshGenRequest(
        images=[a.path for a in images],
        image_roles=roles[: len(images)],
        subject=subject,
        model=model,
        seed=seed,
        steps=ctx.settings.mesh_steps,
        guidance=ctx.settings.mesh_guidance,
        resolution=ctx.settings.mesh_resolution,
        out_path=out_path,
        job_id=ctx.job_id,
    )

    async def on_progress(msg: str, p: float) -> None:
        await ctx.emit("meshgen", iteration, f"{msg} ({int(p * 100)}%)")

    try:
        result = await ctx.mesh.generate(req, on_progress)
    except MeshGenError as e:
        await ctx.emit("meshgen", iteration, f"generation failed: {e}")
        log.warning("mesh.generate_failed", iteration=iteration, error=str(e))
        ctx.mesh_info["gen_error"] = str(e)
        return None
    except Exception as e:
        log.warning("mesh.generate_crashed", iteration=iteration, error=str(e))
        ctx.mesh_info["gen_error"] = str(e)
        return None

    gen = result.model_dump(mode="json")
    gen["mesh_rel"] = storage.relpath(result.mesh_path)
    gen.pop("mesh_path", None)
    gen["images"] = [a.asset_id for a in images]
    ctx.mesh_info["generation"] = gen
    return rel


async def _orient_mesh(ctx: AgentCtx, iteration: int, mesh_rel: str) -> dict[str, str]:
    """ORIENT step (once per job). Returns {up, front}. LLM choice, heuristic fallback."""
    if ctx.mesh_orientation:
        return ctx.mesh_orientation
    # inherit the parent version's orientation on plain edits (same subject)
    if ctx.parent.mesh and ctx.parent.mesh.get("orientation") and ctx.mode == "edit":
        o = ctx.parent.mesh["orientation"]
        if o.get("up") and o.get("front"):
            ctx.mesh_orientation = {"up": o["up"], "front": o["front"]}
            ctx.mesh_info["orientation_source"] = "parent"
            await ctx.emit("orient", iteration, f"reusing orientation {o['up']}/{o['front']}")
            return ctx.mesh_orientation
    choice = await mesh_orient_step.run(ctx, iteration, mesh_rel)
    ctx.mesh_orientation = choice
    ctx.mesh_info["orientation_source"] = ctx.mesh_info.get("orientation_source") or "llm"
    return ctx.mesh_orientation


async def _process_mesh(
    ctx: AgentCtx,
    iteration: int,
    mesh_rel: str,
    spec: PartSpec,
    post: MeshPostParams,
    *,
    make_solid: bool = False,
) -> dict[str, Any]:
    return await ctx.executor.process_mesh(
        _mesh_payload(
            ctx,
            iteration,
            mode="final",
            mesh_rel=mesh_rel,
            post=post,
            spec=spec,
            make_solid=make_solid,
        )
    )


# --------------------------------------------------------------------------- #
# loops                                                                        #
# --------------------------------------------------------------------------- #


async def _run_cad(  # noqa: C901
    ctx: AgentCtx,
    outcome: RunOutcome,
    spec: PartSpec,
    plan: Any,
    t0: float,
    cancelled: Any,
    *,
    max_iterations: int,
    max_repairs: int,
    pass_score: int,
    tol_pct: float,
    tol_mm: float,
) -> RunOutcome:
    s = get_settings()
    best: Candidate | None = None
    feedback: str | None = None
    iterations_done = 0

    for it in range(1, max_iterations + 1):
        iterations_done = it
        if await cancelled():
            return outcome
        if ctx.tokens_used() > ctx.budget_tokens or time.monotonic() - t0 > s.agent_max_wall_s:
            await ctx.emit("budget", it, "budget exceeded — finalising best candidate")
            break

        await ctx.emit("codegen", it, f"writing CAD code (iteration {it}/{max_iterations})")
        try:
            code_resp = await codegen_step.run(ctx, spec, plan, iteration=it, feedback=feedback)
        except Exception as e:
            feedback = f"code generation failed: {e}"
            log.warning("agent.codegen_failed", iteration=it, error=str(e))
            continue

        # execute with repairs
        exec_result: dict[str, Any] = {}
        for repair_n in range(max_repairs + 1):
            if await cancelled():
                return outcome
            await ctx.emit(
                "execute",
                it,
                "executing in sandbox"
                if repair_n == 0
                else f"repair attempt {repair_n}/{max_repairs}",
            )
            exec_result = await ctx.executor.execute(
                _exec_payload(ctx, code_resp.code, it, full_exports=False)
            )
            if exec_result.get("ok"):
                break
            if repair_n < max_repairs:
                await ctx.emit("repair", it, "fixing execution error")
                try:
                    code_resp = await codegen_step.repair(
                        ctx,
                        code_resp,
                        exec_result.get("error"),
                        exec_result.get("traceback"),
                        exec_result.get("stdout"),
                        iteration=it,
                        repair_n=repair_n + 1,
                    )
                except Exception as e:
                    log.warning("agent.repair_failed", iteration=it, error=str(e))
                    break

        if not exec_result.get("ok"):
            feedback = (
                "code failed to execute after repairs — rewrite more conservatively.\n"
                + (exec_result.get("error") or "")[:1500]
            )
            continue

        # numeric checks
        analysis = exec_result.get("analysis") or {}
        dim_issues = compare_dims(spec, analysis, tol_pct, tol_mm)

        # vision critique
        await ctx.emit("critique", it, "comparing renders with your images")
        sheet_rel = exec_result.get("contact_sheet")
        sheet = storage.resolve(sheet_rel) if sheet_rel else None
        try:
            critique = await critique_step.run(
                ctx,
                spec,
                sheet,
                analysis,
                dim_issues,
                exec_result.get("dfm"),
                iteration=it,
            )
        except Exception as e:
            log.warning("agent.critique_failed", iteration=it, error=str(e))
            critique = Critique(score=0, passed=False, confidence=0.0)

        cand = Candidate(it, code_resp, exec_result, critique, dim_issues)
        if best is None or cand.rank > best.rank:
            best = cand

        major = [d for d in critique.discrepancies if d.severity == "major"]
        await ctx.emit(
            "critique",
            it,
            f"score {critique.score}/100 — {len(major)} major, "
            f"{len(critique.discrepancies) - len(major)} minor issues",
        )

        if critique.passed and critique.score >= pass_score and not dim_issues and not major:
            await ctx.emit("done", it, "all checks passed")
            break
        feedback = build_feedback(critique, dim_issues, exec_result.get("dfm"))

    outcome.iterations = iterations_done

    # ---------- FINALIZE ----------
    if best is None:
        outcome.error = "all iterations failed to produce executable code"
        return outcome

    await ctx.emit("finalize", best.iteration, "exporting all formats")
    exports = await ctx.executor.export_all(
        {
            "job_id": ctx.job_id,
            "project_id": str(ctx.project_id),
            "version_id": str(ctx.version_id),
            "iteration": best.iteration,
            "name": "model",
        }
    )
    artifacts = _copy_best_artifacts(ctx, best.iteration, best.exec_result)
    for fmt, rel in (exports.get("exports") or {}).items():
        if rel:
            artifacts[f"export:{fmt}"] = rel

    outcome.status = (
        "succeeded"
        if (best.critique and best.critique.passed and not best.dim_issues)
        else "needs_review"
    )
    outcome.code = best.code_response.code if best.code_response else None
    outcome.params = dict(best.exec_result.get("params_used") or {})
    outcome.params_schema = (
        [p.model_dump() for p in best.code_response.params] if best.code_response else None
    )
    outcome.analysis = best.exec_result.get("analysis")
    outcome.critique = best.critique.model_dump() if best.critique else None
    outcome.dfm = best.exec_result.get("dfm")
    outcome.score = best.critique.score if best.critique else None
    outcome.best_iteration = best.iteration
    if ctx.resolved_mode == "hybrid" and ctx.mesh_info:
        prep_it = int(ctx.mesh_info.get("_prep_it", 1))
        artifacts.update(_copy_mesh_artifacts(ctx.project_id, ctx.version_id, prep_it))
        outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
        _write_mesh_json(ctx.project_id, ctx.version_id, ctx.mesh_info)
    outcome.artifacts = artifacts
    outcome.usage = dict(getattr(ctx.llm, "total_usage", {}))
    return outcome


async def _run_mesh(
    ctx: AgentCtx,
    outcome: RunOutcome,
    spec: PartSpec,
    plan: Any,
    t0: float,
    cancelled: Any,
    *,
    max_iterations: int,
    pass_score: int,
    tol_pct: float,
    tol_mm: float,
) -> RunOutcome:
    """Mesh-mode loop: generate (varying seed) -> process -> analyse -> critique."""
    s = get_settings()
    best: Candidate | None = None
    iterations_done = 0

    for it in range(1, max_iterations + 1):
        iterations_done = it
        if await cancelled():
            return outcome
        if ctx.tokens_used() > ctx.budget_tokens or time.monotonic() - t0 > s.agent_max_wall_s:
            await ctx.emit("budget", it, "budget exceeded — finalising best candidate")
            break

        # reuse a precomputed result (hybrid->mesh fallback)
        exec_result: dict[str, Any] | None = None
        mesh_rel: str | None = None
        if it == 1 and ctx.mesh_info.pop("_pre_result", None):
            exec_result = ctx.mesh_info["_pre_result_exec"]
            mesh_rel = ctx.mesh_info["_pre_result_mesh_rel"]
        else:
            # MESH_GENERATE
            await ctx.emit("meshgen", it, f"generating mesh (iteration {it}/{max_iterations})")
            mesh_rel = await _generate_mesh(ctx, it, spec, plan)
            if mesh_rel is None:
                continue

            # ORIENT (once per job)
            orientation = await _orient_mesh(ctx, it, mesh_rel)
            post = MeshPostParams(
                up=orientation.get("up"),  # type: ignore[arg-type]
                front=orientation.get("front"),  # type: ignore[arg-type]
            )
            post = merge_post_params(post, {})

            # PROCESS_MESH
            await ctx.emit("meshprocess", it, "cleaning, orienting and scaling mesh")
            exec_result = await _process_mesh(ctx, it, mesh_rel, spec, post)
            ctx.mesh_info["post"] = post.model_dump()
            ctx.mesh_info["orientation"] = {"up": post.up, "front": post.front}

        if not exec_result or not exec_result.get("ok"):
            log.warning("mesh.process_failed", iteration=it, error=(exec_result or {}).get("error"))
            continue

        analysis = exec_result.get("analysis") or {}
        dim_issues = compare_dims(spec, analysis, tol_pct, tol_mm)
        dim_issues += list((exec_result.get("mesh") or {}).get("dim_issues") or [])

        # vision critique
        await ctx.emit("critique", it, "comparing renders with your images")
        sheet_rel = exec_result.get("contact_sheet")
        sheet = storage.resolve(sheet_rel) if sheet_rel else None
        try:
            critique = await critique_step.run(
                ctx,
                spec,
                sheet,
                analysis,
                dim_issues,
                exec_result.get("dfm"),
                iteration=it,
            )
        except Exception as e:
            log.warning("agent.critique_failed", iteration=it, error=str(e))
            critique = Critique(score=0, passed=False, confidence=0.0)

        cand = Candidate(
            it,
            None,
            exec_result,
            critique,
            dim_issues,
            mesh_post=(exec_result.get("mesh") or {}).get("post"),
        )
        if best is None or cand.rank > best.rank:
            best = cand

        major = [d for d in critique.discrepancies if d.severity == "major"]
        await ctx.emit(
            "critique",
            it,
            f"score {critique.score}/100 — {len(major)} major, "
            f"{len(critique.discrepancies) - len(major)} minor issues",
        )

        if critique.passed and critique.score >= pass_score and not dim_issues and not major:
            await ctx.emit("done", it, "all checks passed")
            break
        # mesh-mode "feedback" is limited: next iteration uses another seed

    outcome.iterations = iterations_done

    # ---------- FINALIZE ----------
    if best is None:
        outcome.error = ctx.mesh_info.get("gen_error") or "all mesh iterations failed"
        return outcome

    await ctx.emit("finalize", best.iteration, "exporting mesh formats")
    exports = await ctx.executor.export_all(
        {
            "job_id": ctx.job_id,
            "project_id": str(ctx.project_id),
            "version_id": str(ctx.version_id),
            "iteration": best.iteration,
            "name": "model",
        }
    )
    artifacts = _copy_best_artifacts(ctx, best.iteration, best.exec_result)
    for fmt, rel in (exports.get("exports") or {}).items():
        if rel:
            artifacts[f"export:{fmt}"] = rel
    artifacts.update(_copy_mesh_artifacts(ctx.project_id, ctx.version_id, best.iteration))

    outcome.status = (
        "succeeded"
        if (best.critique and best.critique.passed and not best.dim_issues)
        else "needs_review"
    )
    outcome.code = None
    post_used = best.mesh_post or {}
    outcome.params = post_used or dict(best.exec_result.get("params_used") or {})
    outcome.params_schema = [p.model_dump() for p in mesh_params_schema()]
    outcome.analysis = best.exec_result.get("analysis")
    outcome.critique = best.critique.model_dump() if best.critique else None
    outcome.dfm = best.exec_result.get("dfm")
    outcome.score = best.critique.score if best.critique else None
    outcome.best_iteration = best.iteration
    outcome.artifacts = artifacts
    outcome.usage = dict(getattr(ctx.llm, "total_usage", {}))
    outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
    outcome.mesh["best_iteration"] = best.iteration
    _write_mesh_json(ctx.project_id, ctx.version_id, outcome.mesh)
    return outcome


async def _prepare_hybrid_mesh(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: Any,
    cancelled: Any,
) -> tuple[bool, str]:
    """Hybrid phase 1: generate + orient + process mesh, build the faceted solid.

    Returns (ready, note). On success ctx.hybrid_solid/_mesh_info are filled.
    On failure the caller falls back to mesh mode (mesh already processed) or
    CAD (nothing usable).

    Mesh prep runs in iterations/0 so its files (mesh solid/model.brep/model.stl)
    are not overwritten by CAD executions starting at iteration 1.
    """
    it = 0
    ctx.mesh_info["_prep_it"] = it
    # hybrid edit without regeneration: reuse the parent's solid directly
    if ctx.parent.mesh_solid is not None and ctx.parent.mesh:
        # copy the parent's mesh intermediates so the child version is self-contained
        src_dir = ctx.parent.mesh_solid.parent
        mesh_dir = storage.ensure(storage.version_dir(ctx.project_id, ctx.version_id) / "mesh")
        for name in ("mesh_raw.glb", "mesh_clean.stl", "solid.brep", "mesh.json"):
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, mesh_dir / name)
        ctx.hybrid_solid = mesh_dir / "solid.brep"
        ctx.mesh_info = dict(ctx.parent.mesh)
        ctx.mesh_info["_prep_it"] = it
        ctx.mesh_info["reused_solid"] = True
        ctx.hybrid_mesh_info = {
            "bbox_size": ctx.parent.mesh.get("bbox_size"),
            "triangles": ctx.parent.mesh.get("triangles"),
        }
        await ctx.emit("meshgen", it, "reusing parent mesh solid")
        return True, "reused parent mesh solid"

    await ctx.emit("meshgen", it, "generating organic base mesh")
    mesh_rel = await _generate_mesh(ctx, it, spec, plan)
    if mesh_rel is None:
        return False, "mesh generation failed"
    ctx.mesh_raw_reuse = storage.resolve(mesh_rel)

    orientation = await _orient_mesh(ctx, it, mesh_rel)
    post = MeshPostParams(
        up=orientation.get("up"),  # type: ignore[arg-type]
        front=orientation.get("front"),  # type: ignore[arg-type]
    )

    await ctx.emit("meshprocess", it, "processing mesh and building solid")
    exec_result = await _process_mesh(ctx, it, mesh_rel, spec, post, make_solid=True)

    if not exec_result.get("ok"):
        return False, exec_result.get("error") or "mesh processing failed"

    ctx.mesh_info["post"] = post.model_dump()
    ctx.mesh_info["orientation"] = {"up": post.up, "front": post.front}
    ctx.mesh_info["bbox_size"] = (exec_result.get("analysis") or {}).get("bbox_size")
    ctx.mesh_info["triangles"] = (exec_result.get("mesh") or {}).get("triangles")
    ctx.hybrid_mesh_info = {
        "bbox_size": (exec_result.get("analysis") or {}).get("bbox_size"),
        "volume": (exec_result.get("analysis") or {}).get("volume"),
        "triangles": (exec_result.get("mesh") or {}).get("triangles"),
    }

    if (exec_result.get("mesh") or {}).get("solid_ok") and exec_result["mesh"].get("solid_relpath"):
        ctx.hybrid_solid = storage.resolve(exec_result["mesh"]["solid_relpath"])
        return True, "mesh solid ready"

    # retry once with stronger decimation for OCCT stability
    s = get_settings()
    post2 = merge_post_params(post, {"decimate_faces": min(15_000, s.mesh_decimate_faces)})
    await ctx.emit("meshprocess", it, "solid conversion failed — retrying with fewer faces")
    exec_result = await _process_mesh(ctx, it, mesh_rel, spec, post2, make_solid=True)
    if exec_result.get("ok") and (exec_result.get("mesh") or {}).get("solid_ok"):
        ctx.hybrid_solid = storage.resolve(exec_result["mesh"]["solid_relpath"])
        ctx.mesh_info["post"] = post2.model_dump()
        ctx.mesh_info["bbox_size"] = (exec_result.get("analysis") or {}).get("bbox_size")
        ctx.hybrid_mesh_info = {
            "bbox_size": (exec_result.get("analysis") or {}).get("bbox_size"),
            "volume": (exec_result.get("analysis") or {}).get("volume"),
            "triangles": (exec_result.get("mesh") or {}).get("triangles"),
        }
        return True, "mesh solid ready (decimated)"

    # keep the processed mesh as a candidate for the mesh-mode fallback
    if exec_result.get("ok"):
        ctx.mesh_info["_pre_result"] = True
        ctx.mesh_info["_pre_result_exec"] = exec_result
        ctx.mesh_info["_pre_result_mesh_rel"] = mesh_rel
    return False, "mesh-to-solid conversion failed"


# --------------------------------------------------------------------------- #
# main state machine                                                           #
# --------------------------------------------------------------------------- #


async def run_generation(ctx: AgentCtx) -> RunOutcome:
    s = get_settings()
    outcome = RunOutcome()
    t0 = time.monotonic()

    max_iterations = ctx.settings.max_iterations or s.agent_max_iterations
    max_repairs = ctx.settings.max_repairs or s.agent_max_repairs
    pass_score = ctx.settings.pass_score or s.agent_pass_score
    tol_pct = ctx.settings.dim_tolerance_pct or s.agent_dim_tolerance_pct
    tol_mm = ctx.settings.dim_tolerance_mm or s.agent_dim_tolerance_mm
    clarify = (ctx.settings.clarification_mode or s.agent_clarification_mode) == "ask"

    async def cancelled() -> bool:
        if await ctx.check_cancel():
            outcome.status = "cancelled"
            return True
        return False

    # ---------- INTAKE ----------
    spec = ctx.spec
    if spec is None:
        await ctx.emit("intake", 0, "analysing images and description")
        try:
            spec = await intake_step.run(ctx)
        except Exception as e:
            outcome.error = f"intake failed: {e}"
            return outcome
    outcome.spec = spec.model_dump()

    # ---------- CLARIFY ----------
    blocking = [q for q in spec.open_questions if q.blocking]
    if clarify and blocking:
        outcome.status = "waiting_for_input"
        outcome.questions = [q.model_dump() for q in spec.open_questions]
        outcome.resume_state = {"spec": spec.model_dump()}
        await ctx.emit("intake", 0, f"waiting for answers ({len(blocking)} blocking questions)")
        return outcome

    # ---------- PLAN ----------
    plan = ctx.plan
    if plan is None:
        await ctx.emit("plan", 0, "drafting modeling plan")
        try:
            plan = await plan_step.run(ctx, spec)
        except Exception as e:
            log.warning("agent.plan_failed", error=str(e))
            plan = None
    ctx.plan = plan
    outcome.plan = plan.model_dump() if plan else None

    # ---------- MODE RESOLUTION ----------
    mode, mode_note = resolve_mode(ctx)
    ctx.resolved_mode = mode
    outcome.resolved_mode = mode
    outcome.mode_note = mode_note
    await ctx.emit("plan", 0, f"approach: {mode} — {mode_note}")

    kwargs: dict[str, Any] = dict(
        max_iterations=max_iterations,
        pass_score=pass_score,
        tol_pct=tol_pct,
        tol_mm=tol_mm,
    )

    # ---------- MESH ----------
    if mode == "mesh":
        return await _run_mesh(ctx, outcome, spec, plan, t0, cancelled, **kwargs)

    # ---------- HYBRID ----------
    if mode == "hybrid":
        ready, note = await _prepare_hybrid_mesh(ctx, spec, plan, cancelled)
        await ctx.emit("meshprocess", 1, note)
        if ready:
            outcome = await _run_cad(
                ctx, outcome, spec, plan, t0, cancelled, max_repairs=max_repairs, **kwargs
            )
            if outcome.status != "failed" and ctx.mesh_info:
                outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
            return outcome
        # fallbacks
        if ctx.mesh_info.get("_pre_result"):
            outcome.resolved_mode = "mesh"
            ctx.resolved_mode = "mesh"
            outcome.mode_note += " | hybrid conversion failed — mesh-only result"
            await ctx.emit("plan", 0, outcome.mode_note)
            return await _run_mesh(ctx, outcome, spec, plan, t0, cancelled, **kwargs)
        outcome.resolved_mode = "cad"
        ctx.resolved_mode = "cad"
        outcome.mode_note += f" | hybrid failed ({note}) — CAD fallback"
        await ctx.emit("plan", 0, outcome.mode_note)
        return await _run_cad(
            ctx, outcome, spec, plan, t0, cancelled, max_repairs=max_repairs, **kwargs
        )

    # ---------- CAD ----------
    return await _run_cad(
        ctx, outcome, spec, plan, t0, cancelled, max_repairs=max_repairs, **kwargs
    )
