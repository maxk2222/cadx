"""TopOpt phase — `_run_topopt_phase` and its payload/summary helpers.

Split out of the former `orchestrator.py` monolith; `kiara.agent.orchestrator`
re-exports every name so import paths stay stable."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from kiara import storage
from kiara.agent.executor_client import exec_error
from kiara.agent.orchestrator.shared import (
    RunOutcome,
    _copy_best_artifacts,
    _copy_fem_artifacts,
    _verify_spec_issue,
)
from kiara.agent.steps import optimize as optimize_step
from kiara.agent.types import AgentCtx
from kiara.llm.schemas import (
    PartSpec,
)
from kiara.logging import get_logger

log = get_logger(__name__)


def _normalize_topopt_base(
    stl: Path, spec_dims: list[float | None], it_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Clip the topopt base STL to the stated spec envelope; writes the
    normalized mesh to ``<it_dir>/_run/topopt_input.stl``.

    The voxel FEM domain IS the input mesh's bbox — a base mesh overshooting
    the declared spec dims shifts the whole TopOpt problem out of envelope.
    Normalizing here (worker side) covers both downstream paths: the agent
    session gets the clipped file as `base_stl`, the deterministic fallback
    as `mesh_rel`. Fails soft — any load/clip/export problem keeps the
    original STL; a spec without stated dims skips silently.

    Returns ``(stl_path, report)`` — report carries input_bbox/spec_bbox/
    input_normalized/base_mesh_relpath for the topopt report merge."""
    if not any(d is not None for d in spec_dims):
        return stl, {}
    report: dict[str, Any] = {"base_mesh_relpath": storage.relpath(stl)}
    try:
        from kiara.cad.engine import normalize_mesh_to_spec_envelope
        from kiara.cad.mesh_ops import load_mesh as _load_tri

        mesh = _load_tri(stl)
        out, rep = normalize_mesh_to_spec_envelope(mesh, spec_dims)
        report.update(rep)
    except Exception as e:
        log.warning("topopt.normalize_failed", error=str(e)[:200])
        return stl, {}
    if not report.get("input_normalized"):
        return stl, report
    dst = storage.ensure(it_dir / "_run") / "topopt_input.stl"
    try:
        out.export(str(dst), file_type="stl")
    except Exception as e:
        log.warning("topopt.normalize_export_failed", error=str(e)[:200])
        report["input_normalized"] = False
        report.pop("method", None)
        return stl, report
    report["normalized_mesh_relpath"] = storage.relpath(dst)
    log.info(
        "topopt.input_normalized",
        method=report.get("method"),
        input_bbox=report.get("input_bbox"),
        spec_bbox=report.get("spec_bbox"),
    )
    return dst, report
def _promote_topopt_exports(ctx: AgentCtx, iteration: int, res: dict[str, Any]) -> dict[str, str]:
    """Copy the winning optimize iteration's ``exports/model.*`` up to the
    version-level ``exports/`` dir so API/UI serve the optimized part, not the
    pre-topopt base. Returns artifact records for the promoted paths."""
    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration)
    out: dict[str, str] = {}
    exp_dir = storage.ensure(storage.version_dir(ctx.project_id, ctx.version_id) / "exports")
    for fmt, rel in (res.get("exports") or {}).items():
        if not rel:
            continue
        src = it_dir / "exports" / Path(str(rel)).name
        if not src.exists():
            continue
        dst = exp_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        out[f"export:{fmt}"] = storage.relpath(dst)
    return out
async def _run_topopt_phase(
    ctx: AgentCtx, outcome: RunOutcome, spec: PartSpec, t0: float
) -> RunOutcome:
    """OPTIMIZE phase after a successful base generation (gen_mode=topopt or
    plan.optimize): run SIMP on the best candidate + FEM-verify the result.

    `t0` is the job's start (time.monotonic) — the optimize session is
    bounded by the arq job timeout minus a finalization margin so a string
    of 47-minute SIMP runs can't outlive the job and leave zombie tasks."""
    if outcome.status not in ("succeeded", "needs_review"):
        return outcome
    if not spec.load_cases:
        # the caller already ran the focused extraction fallback; arriving
        # here still empty means the prompt had no usable load intent
        outcome.mode_note += " | topopt skipped — no load cases in the spec"
        await ctx.emit("optimize", 0, "skipped: intake declared no load cases")
        return outcome

    base_it = outcome.best_iteration or outcome.iterations or 1
    base_mode = outcome.resolved_mode or "cad"
    it_dir = (
        storage.resolve(outcome.art_dir)
        if outcome.art_dir
        else storage.iteration_dir(ctx.project_id, ctx.version_id, base_it)
    )
    stl = it_dir / ("_run/model.stl" if base_mode == "mesh" else "renders/mesh.stl")
    if not stl.exists():
        stl = it_dir / "renders/mesh.stl"
    if not stl.exists():
        outcome.mode_note += " | topopt skipped — no base mesh artifact"
        return outcome

    # clip an out-of-envelope base mesh to the stated spec dims once here so
    # the agent session AND the deterministic fallback both optimize the
    # normalized input — spec_dims also travels in the payload below so the
    # executor re-checks (and records domain_bbox) on its side
    spec_dims = _stated_spec_dims(spec)
    stl, norm_rep = _normalize_topopt_base(stl, spec_dims, it_dir)

    it = (outcome.iterations or base_it) + 1

    # agent-driven session: FEM probes + parameterized SIMP runs, agent picks
    # the best run — falls back to the deterministic single shot below
    res: dict[str, Any] | None = None
    base_sheet = it_dir / "contact_sheet.png"
    from kiara.config import get_settings

    agent = await optimize_step.run(
        ctx,
        spec,
        base_stl=stl,
        base_it=outcome.iterations or base_it,
        analysis=outcome.analysis,
        contact_sheet=base_sheet if base_sheet.exists() else None,
        deadline=t0 + get_settings().agent_job_timeout_s - 300,
    )
    if agent is not None and agent.chosen_run == -1:
        outcome.mode_note += " | topopt skipped — agent kept the base part"
        await ctx.emit("optimize", it, f"skipped: {agent.reasoning or 'agent kept the base part'}")
        return outcome
    if agent is not None and agent.res is not None and agent.iteration is not None:
        it = agent.iteration
        res = agent.res
        await ctx.emit(
            "optimize",
            it,
            f"agent selected run {agent.chosen_run} — {agent.reasoning or 'no note'}",
        )

    if res is None:
        await ctx.emit(
            "optimize",
            it,
            f"topology optimization on {base_mode} base ({len(spec.load_cases)} case(s))",
        )
        res = await ctx.executor.optimize(
            {
                "job_id": ctx.job_id,
                "project_id": str(ctx.project_id),
                "version_id": str(ctx.version_id),
                "iteration": it,
                "mesh_rel": storage.relpath(stl),
                "load_cases": [c.model_dump() for c in spec.load_cases],
                "optimize": _topopt_optimize_payload(ctx, spec, spec_dims),
                "material": ctx.settings.fem_material or spec.material,
                "voxel_mm": ctx.settings.fem_voxel_mm,
                "runtime_settings": ctx.runtime_settings.executor_values(),
                "name": "model",
                "anchors": (ctx.anchors or {}).get("holes"),
                "spec": spec.model_dump(mode="json"),
            }
        )
    if not res.get("ok"):
        outcome.mode_note += f" | topopt failed: {exec_error(res)}"
        await ctx.emit("optimize", it, f"failed: {exec_error(res)}")
        return outcome

    arts = _copy_best_artifacts(ctx, it, res)
    arts.update(_promote_topopt_exports(ctx, it, res))
    arts.update(_copy_fem_artifacts(ctx, it))
    outcome.artifacts.update(arts)
    outcome.fem = res.get("fem")
    outcome.analysis = res.get("analysis")
    top = res.get("topopt") or {}
    if norm_rep:
        # worker-side normalization metrics are authoritative — the executor
        # only ever saw the (already clipped) input file
        top = {**top, **norm_rep}
    if agent is not None and agent.runs:
        top = {**top, "agent_runs": agent.runs}
    if res.get("verify_spec"):
        top = {**top, "verify_spec": res["verify_spec"]}
    outcome.mesh = {**(outcome.mesh or {}), "pipeline": "topopt", "topopt": top}
    outcome.resolved_mode = "topopt"
    worst = _topopt_headline_sf(top, outcome.fem)
    if worst is not None and worst < 1.0:
        outcome.status = "needs_review"
        outcome.mode_note += f" | optimized part fails: SF {worst:.2f} < 1"
    if (top.get("verification") or {}).get("valid") is False:
        outcome.status = "needs_review"
        outcome.mode_note += " | export FEM solve degenerate — optimization unverified"
    # canonical verify_spec verdict heads the persisted issue list — the
    # topopt result dict plays the exec_result role here (bound-semantics
    # envelope checks, feature + structural tiers)
    vs_entry = _verify_spec_issue(res, stage="optimize")
    if vs_entry:
        outcome.issues = [vs_entry, *list(outcome.issues or [])]
    await ctx.emit(
        "optimize",
        it,
        f"done — volfrac {top.get('volfrac_achieved')}, "
        f"worst SF {worst}, {top.get('iterations')} iterations",
    )
    return outcome
def _stated_spec_dims(spec: PartSpec) -> list[float | None]:
    """[length_x, width_y, height_z] in mm for the axes the user actually
    stated — assumed/inferred dims are declared intent, not a hard envelope
    worth clipping to (same convention as verifyspec._stated_envelope)."""
    return [
        float(d.value) if d is not None and d.source == "stated" and (d.value or 0) > 0 else None
        for d in (spec.length_x, spec.width_y, spec.height_z)
    ]
def _topopt_headline_sf(top: dict[str, Any], fem: dict[str, Any] | None) -> float | None:
    """Headline SF for a topopt outcome: the verified realization worst-case,
    then the printed-geometry export SF, then the voxel-mask report — the
    mask number alone must never acquit a part whose printed export erodes
    to failure (docs/fem-topopt.md: gate on sf_verified). A solved 0.0 is a
    verdict, not a missing value — None-checks, never or-chains."""
    worst = top.get("sf_verified")
    if worst is None:
        worst = top.get("sf_export")
    if worst is None:
        mask_sfs = [
            c.get("safety_factor")
            for c in (fem or {}).get("cases", [])
            if c.get("safety_factor") is not None
        ]
        worst = min(mask_sfs) if mask_sfs else None
    return worst
def _topopt_optimize_payload(
    ctx: AgentCtx, spec: PartSpec, spec_dims: list[float | None]
) -> dict[str, Any]:
    """`optimize` dict for the deterministic executor call: spec OptimizeSpec
    + project overrides + the stated spec envelope (drives executor-side
    envelope normalization — see normalize_mesh_to_spec_envelope)."""
    opt: dict[str, Any] = spec.optimize.model_dump() if spec.optimize else {}
    # OptimizeSpec.multires is schema-False even when the field was never
    # set — a serialized dump would read as an opt-out to the executor, which
    # treats an ABSENT key as default-on. Drop it only when it wasn't
    # explicitly declared; an explicit multires=false is a real opt-out.
    if spec.optimize and "multires" not in spec.optimize.model_fields_set:
        opt.pop("multires", None)
    if ctx.settings.topopt_volfrac:
        opt["target_volume_fraction"] = ctx.settings.topopt_volfrac
    opt["max_iterations"] = ctx.settings.topopt_iterations
    if any(d is not None for d in spec_dims):
        opt["spec_dims_mm"] = spec_dims
    return opt
