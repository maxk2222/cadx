"""OPTIMIZE step — agent-driven FEM/topology-optimization tool session.

Runs after a successful base generation when the job asks for structural
optimization (gen_mode=topopt, plan.optimize or spec.optimize). Instead of a
single fixed-parameter SIMP call the agent gets `analyze_fem`/`topopt_run`
tools, can vary parameters across runs, inspect renders, and commits the best
run via `done(run_index)`.

Every executor call occupies its own iteration dir (base_it + 1, +2, …) so
each run's topopt/, fem/, renders/ and exports/ stay intact — the orchestrator
copies the chosen run's artifacts exactly like a deterministic phase would.

Returns None when unavailable (tools off / session failed) — the caller falls
back to the deterministic single-shot path.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cadx import storage
from cadx.agent.context import (
    anchors_block,
    renders_block,
    spec_block,
    system_prompt,
)
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import AgentCtx
from cadx.llm.schemas import LoadCase, PartSpec
from cadx.logging import get_logger

log = get_logger(__name__)

STEP = "optimize"


@dataclass
class AgentOptResult:
    chosen_run: int  # -1 = agent declined optimization
    reasoning: str = ""
    iteration: int | None = None  # iteration dir of the chosen run
    res: dict[str, Any] | None = None  # chosen run's executor result
    runs: list[dict[str, Any]] = field(default_factory=list)  # run metrics summary


def _run_summary(r: dict[str, Any], chosen: bool, reasoning: Any) -> dict[str, Any]:
    """Compact per-run metrics for the version's topopt.agent_runs table."""
    res = r.get("res") or {}
    top = res.get("topopt") or {}
    fem = res.get("fem") or {}
    # sf_export is the safety factor on the printed geometry — prefer it over
    # the voxel-mask fem report when the optimize pipeline produced both
    sfs = [
        c.get("safety_factor") for c in fem.get("cases") or [] if c.get("safety_factor") is not None
    ]
    worst_sf = top.get("sf_export") or (min(sfs) if sfs else None)
    # refine runs carry their finishing report — surface it so the version's
    # agent_runs table (and the GUI) sees applied/skipped features, probes
    # and analytic-STEP provenance instead of an empty record
    refine = (res.get("features") or {}).get("refine") or {}
    refine_block = {"refine": refine} if refine else {}
    return {
        "index": r["index"],
        "it": r.get("iteration"),
        "params": {k: v for k, v in (r.get("params") or {}).items() if k != "load_cases"},
        "volfrac": top.get("volfrac_target"),
        "achieved": top.get("volfrac_achieved"),
        "iterations": top.get("iterations"),
        "worst_sf": worst_sf,
        "sf_mask": top.get("sf_mask"),
        "sf_export": top.get("sf_export"),
        "extract": top.get("extract"),
        "verification": top.get("verification"),
        "min_member_mm": top.get("min_member_mm"),
        "thin_fraction": top.get("thin_fraction"),
        "n_components": top.get("n_components"),
        "warnings": (top.get("warnings") or [])[:5],
        **refine_block,
        "chosen": chosen,
        **({"reasoning": str(reasoning)} if chosen and reasoning else {}),
    }


def _best_run_index(session: ToolSession) -> int | None:
    """Index of the best completed run — for auto-commit when the session
    exhausts its budget without done(). Prefers sf_export (the safety factor
    on the printed export geometry), else the run's worst case SF; None when
    no run produced a metric."""
    best_idx: int | None = None
    best_sf = -1.0
    for r in session.struct_runs:
        res = r.get("res") or {}
        top = res.get("topopt") or {}
        fem = res.get("fem") or {}
        sfs = [c.get("safety_factor") for c in fem.get("cases") or [] if c.get("safety_factor")]
        sf = top.get("sf_export") or (min(sfs) if sfs else None)
        if sf is not None and float(sf) > best_sf:
            best_idx, best_sf = r["index"], float(sf)
    return best_idx


def _valid_regions(raw: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate keep/region dicts — same contract as load-case regions."""
    from cadx.llm.schemas import RegionSpec

    if raw is None:
        return None, None
    if not isinstance(raw, list):
        return None, "keep_regions must be a list"
    out: list[dict[str, Any]] = []
    for i, r in enumerate(raw):
        try:
            out.append(RegionSpec(**r).model_dump())
        except (ValidationError, TypeError) as e:
            return None, f"keep_regions[{i}] invalid: {e}"
    return out, None


def _valid_cases(raw: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate load-case dicts against the LLM contract — a bad case yields a
    tool error the model can fix instead of a wasted executor call."""
    if raw is None:
        return None, None
    if not isinstance(raw, list):
        return None, "load_cases must be a list"
    out: list[dict[str, Any]] = []
    for i, c in enumerate(raw):
        try:
            out.append(LoadCase(**c).model_dump())
        except (ValidationError, TypeError) as e:
            return None, f"load_cases[{i}] invalid: {e}"
    return out, None


@dataclass
class _SessionOps:
    """Shared state for the optimize-session tool callbacks."""

    ctx: AgentCtx
    base: dict[str, Any]
    spec: PartSpec
    spec_cases: list[dict[str, Any]]
    default_material: str
    alloc: Any
    contact_sheet: Path | None
    deadline: float | None = None  # monotonic timestamp — job-timeout bound


def _budget_exceeded(ops: _SessionOps, needed_s: float) -> dict[str, Any] | None:
    """Refuse a new executor run that cannot plausibly finish inside the
    job's wall-clock bound — the tool error pushes the model to `done()`."""
    if ops.deadline is None or time.monotonic() + needed_s <= ops.deadline:
        return None
    return {
        "ok": False,
        "error": (
            "job time budget exhausted — no new run can finish in time; "
            "call done(run_index) with the best completed run (or -1 to keep the base)"
        ),
    }


def _fem_case_images(res: dict[str, Any], it_dir: Path) -> list[str]:
    """Per-case FEM renders (vonmises/deformed PNGs) under <it_dir>/fem/ —
    the fem / fem_export reports carry bare filenames."""
    fem_dir = it_dir / "fem"
    out: list[str] = []
    for rep_key in ("fem", "fem_export"):
        for c in (res.get(rep_key) or {}).get("cases") or []:
            for p in (c.get("vonmises_png"), c.get("deformed_png")):
                if p and (fem_dir / p).exists():
                    out.append(str(fem_dir / p))
    return out


def _render_view_images(res: dict[str, Any]) -> list[str]:
    """Shape-view renders — the task layer rewrites `renders` to
    {view: relpath} via _relpath_exports, so resolve like contact_sheet."""
    out: list[str] = []
    for rel in (res.get("renders") or {}).values():
        if not rel:
            continue
        with contextlib.suppress(ValueError):
            p = storage.resolve(rel)
            if p.exists():
                out.append(str(p))
    return out


def _make_fem_fn(ops: _SessionOps) -> Any:
    ctx = ops.ctx

    async def fem_fn(a: dict[str, Any]) -> dict[str, Any]:
        cases, err = _valid_cases(a.get("load_cases"))
        if err:
            return {"ok": False, "error": err}
        if (guard := _budget_exceeded(ops, ctx.runtime_settings.fem_timeout_s + 60)) is not None:
            return guard
        it = next(ops.alloc)
        await ctx.emit("optimize", it, f"FEM probe — {len(cases or ops.spec_cases)} load case(s)")
        res = await ctx.executor.fem_analyze(
            {
                **ops.base,
                "iteration": it,
                "load_cases": cases if cases is not None else ops.spec_cases,
                "material": str(a.get("material") or "") or ops.default_material,
                "voxel_mm": a.get("voxel_mm") or ctx.settings.fem_voxel_mm or None,
            }
        )
        probe_sfs = [
            c.get("safety_factor")
            for c in (res.get("fem") or {}).get("cases") or []
            if c.get("safety_factor") is not None
        ]
        if probe_sfs:
            # live metric for the job SSE stream — emit failure must never
            # break the tool result
            with contextlib.suppress(Exception):
                await ctx.emit(
                    "optimize",
                    it,
                    f"FEM probe done — worst SF {min(probe_sfs)}",
                    metrics={"probe_sf": float(min(probe_sfs))},
                )
        fem_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, it) / "fem"
        images = [
            fem_dir / p
            for c in (res.get("fem") or {}).get("cases") or []
            for p in (c.get("vonmises_png"), c.get("deformed_png"))
            if p and (fem_dir / p).exists()
        ]
        res["_images"] = [str(p) for p in images]
        return res

    return fem_fn


def _opt_overrides(a: dict[str, Any]) -> dict[str, Any]:
    """Map `topopt_run` tool args onto `OptimizeSpec` fields."""
    opt: dict[str, Any] = {}
    num = {
        "target_volume_fraction": float,
        "max_iterations": int,
        "density_threshold": float,
        "min_member_mm": float,
        "min_gap_mm": float,
        "max_displacement_mm": float,
        "max_stress_mpa": float,
        "penal_init": float,
        "penal": float,
        "robust_delta_eta": float,
    }
    for k, conv in num.items():
        if a.get(k) is not None:
            opt[k] = conv(a[k])
    for k in (
        "keep_mode",
        "print_direction",
        "filter",
        "symmetry",
        "solver",
        "case_objective",
        "robust_designs",
    ):
        if a.get(k):
            opt[k] = str(a[k])
    for k in ("self_supporting", "multires", "robust"):
        if a.get(k) is not None:
            opt[k] = bool(a[k])
    return opt


def _make_topopt_fn(ops: _SessionOps) -> Any:
    ctx = ops.ctx

    async def topopt_fn(a: dict[str, Any]) -> dict[str, Any]:
        cases, err = _valid_cases(a.get("load_cases"))
        if err:
            return {"ok": False, "error": err}
        if (
            guard := _budget_exceeded(ops, ctx.runtime_settings.topopt_timeout_s + 180)
        ) is not None:
            return guard
        it = next(ops.alloc)
        opt = _opt_overrides(a)
        keep, kerr = _valid_regions(a.get("keep_regions"))
        if kerr:
            return {"ok": False, "error": kerr}
        if keep:
            opt["keep_regions"] = keep
        void, verr = _valid_regions(a.get("void_regions"))
        if verr:
            return {"ok": False, "error": verr.replace("keep_regions", "void_regions")}
        if void:
            opt["void_regions"] = void
        await ctx.emit(
            "optimize",
            it,
            f"SIMP run — volfrac {opt.get('target_volume_fraction', 'default')}, "
            f"{len(cases or ops.spec_cases)} case(s)",
        )
        res = await ctx.executor.optimize(
            {
                **ops.base,
                "iteration": it,
                "load_cases": cases if cases is not None else ops.spec_cases,
                "optimize": opt,
                "material": str(a.get("material") or "") or ops.default_material,
                "voxel_mm": a.get("voxel_mm") or ctx.settings.fem_voxel_mm or None,
            }
        )
        res["_iteration"] = it
        images: list[str] = []
        for key in ("contact_sheet", "thumbnail"):
            if rel := res.get(key):
                with contextlib.suppress(ValueError):
                    images.append(str(storage.resolve(rel)))
        # region-selection previews drawn by the executor
        for rel in (res.get("topopt") or {}).get("regions_png") or []:
            with contextlib.suppress(ValueError):
                p = storage.resolve(rel)
                if p.exists():
                    images.append(str(p))
        # FEM verification renders (mask + export-mesh reports) written under
        # <iteration_dir>/fem/, plus the optimized-shape view renders
        it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, it)
        images += _render_view_images(res)
        images += _fem_case_images(res, it_dir)
        # base render first so the model can compare base vs optimized
        res["_images"] = ([str(ops.contact_sheet)] if ops.contact_sheet else []) + images
        return res

    return topopt_fn


def _make_preview_fn(ops: _SessionOps) -> Any:
    ctx = ops.ctx

    async def preview_fn(a: dict[str, Any]) -> dict[str, Any]:
        cases, err = _valid_cases(a.get("load_cases"))
        if err:
            return {"ok": False, "error": err}
        if (guard := _budget_exceeded(ops, ctx.runtime_settings.fem_timeout_s + 30)) is not None:
            return guard
        it = next(ops.alloc)
        keep, kerr = _valid_regions(a.get("keep_regions"))
        if kerr:
            return {"ok": False, "error": kerr}
        await ctx.emit("optimize", it, "region preview — checking load placement")
        res = await ctx.executor.fem_analyze(
            {
                **ops.base,
                "iteration": it,
                "load_cases": cases if cases is not None else ops.spec_cases,
                "keep_regions": keep or [],
                "material": ops.default_material,
                "voxel_mm": a.get("voxel_mm") or ctx.settings.fem_voxel_mm or None,
                "preview_only": True,
                "render_pngs": bool(a.get("render_pngs", True)),
            }
        )
        images = []
        for rel in (res.get("preview") or {}).get("regions_png") or []:
            with contextlib.suppress(ValueError):
                p = storage.resolve(rel)
                if p.exists():
                    images.append(str(p))
        res["_images"] = images
        return res

    return preview_fn


def _make_refine_fn(ops: _SessionOps) -> Any:
    ctx = ops.ctx

    async def refine_fn(a: dict[str, Any]) -> dict[str, Any]:
        """Feature finishing on one completed topopt run — the tool handler
        resolves `_run` (struct_runs entry); we locate that run's STL, refine
        it, and let the handler register the result as a new run."""
        run = a.get("_run") or {}
        run_it = run.get("iteration")
        src = (
            storage.iteration_dir(ctx.project_id, ctx.version_id, run_it) / "_run" / "model.stl"
            if run_it
            else None
        )
        if src is None or not src.exists():
            return {"ok": False, "error": f"run {run.get('index')} has no mesh artifact"}
        cases, err = _valid_cases(a.get("load_cases"))
        if err:
            return {"ok": False, "error": err}
        rs = ctx.runtime_settings
        if (
            guard := _budget_exceeded(ops, rs.fem_timeout_s + rs.analytic_step_timeout_s + 120)
        ) is not None:
            return guard
        run_params = run.get("params") or {}
        spec_opt = ops.spec.optimize
        it = next(ops.alloc)
        await ctx.emit("optimize", it, f"feature finishing — run {run.get('index')}")
        res = await ctx.executor.refine(
            {
                **ops.base,  # mesh_rel (base part), anchors, runtime
                "iteration": it,
                "mesh_rel": storage.relpath(src),
                "base_mesh_rel": ops.base["mesh_rel"],
                "features": a.get("features"),
                "spec_features": [f.model_dump() for f in ops.spec.features],
                "keep_regions": (
                    run_params.get("keep_regions")
                    or [r.model_dump() for r in (spec_opt.keep_regions if spec_opt else [])]
                ),
                "load_cases": cases if cases is not None else ops.spec_cases,
                "material": ops.default_material,
                "restore_planes": (
                    a.get("restore_planes")
                    if a.get("restore_planes") is not None
                    else (spec_opt.restore_planes if spec_opt else True)
                ),
                "voxel_mm": ctx.settings.fem_voxel_mm or None,
            }
        )
        res["_iteration"] = it
        images: list[str] = []
        for key in ("contact_sheet", "thumbnail"):
            if rel := res.get(key):
                with contextlib.suppress(ValueError):
                    images.append(str(storage.resolve(rel)))
        # refine has no contact sheet — attach the shape views plus the FEM
        # re-verification renders written under <iteration_dir>/fem/
        it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, it)
        images += _render_view_images(res)
        images += _fem_case_images(res, it_dir)
        res["_images"] = images
        return res

    return refine_fn


def _make_anchors_fn(ops: _SessionOps) -> Any:
    ctx = ops.ctx

    async def anchors_fn(a: dict[str, Any]) -> dict[str, Any]:
        """Scan the base mesh for feature anchors (all axes) and return the
        merged table — also refreshed `anchors` in subsequent payload calls
        via ops.base so hole_index stays consistent."""
        res = await ctx.executor.fem_analyze({**ops.base, "scan_only": True})
        if res.get("ok") and res.get("anchors"):
            ops.base["anchors"] = res["anchors"]
        return res

    return anchors_fn


def _setup_session(
    ctx: AgentCtx,
    *,
    base_stl: Path,
    base_it: int,
    spec: PartSpec,
    spec_cases: list[dict[str, Any]],
    default_material: str,
    anchors: list[dict[str, Any]] | None,
    contact_sheet: Path | None,
    deadline: float | None = None,
) -> ToolSession:
    """ToolSession with fem/topopt/preview callbacks; each executor call
    allocates its own iteration dir and attaches renders as `_images`."""
    ops = _SessionOps(
        ctx=ctx,
        spec=spec,
        base={
            "job_id": ctx.job_id,
            "project_id": str(ctx.project_id),
            "version_id": str(ctx.version_id),
            "mesh_rel": storage.relpath(base_stl),
            "runtime_settings": ctx.runtime_settings.executor_values(),
            "name": "model",
            "anchors": anchors,
        },
        spec_cases=spec_cases,
        default_material=default_material,
        alloc=itertools.count(base_it + 1),
        contact_sheet=contact_sheet,
        deadline=deadline,
    )
    return ToolSession(
        ctx,
        kind="optimize",
        fem_fn=_make_fem_fn(ops),
        topopt_fn=_make_topopt_fn(ops),
        preview_fn=_make_preview_fn(ops),
        anchors_fn=_make_anchors_fn(ops),
        refine_fn=_make_refine_fn(ops),
    )


def _opt_messages(
    ctx: AgentCtx,
    spec: PartSpec,
    request: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    contact_sheet: Path | None,
) -> list[dict[str, Any]]:
    """System + user message for the optimize session (spec, anchors, renders)."""
    opt_note = (
        f"spec target: volfrac {spec.optimize.target_volume_fraction}, "
        f"min safety factor {spec.optimize.safety_factor_min}"
        if spec.optimize
        else "no explicit optimize spec — use sensible defaults"
    )
    req_opt = (request or {}).get("optimize") or {}
    if req_opt:
        opt_note = (
            f"user request target: volfrac "
            f"{req_opt.get('target_volume_fraction', 'default')}, "
            f"min safety factor {req_opt.get('safety_factor_min', 'default')}"
        )
    request_note = ""
    if request:
        request_note = (
            "\n\nUSER REQUEST for this optimization:\n"
            + json.dumps({k: v for k, v in request.items() if v}, default=str)[:1500]
        )
    return [
        {
            "role": "system",
            "content": system_prompt("optimize_system.md")
            + "\n\n"
            + system_prompt("tools_protocol.md"),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Optimize this part's topology. The base part is verified "
                        f"and stored — {opt_note}. Declared load cases are in the "
                        "spec; keep regions around mounting features." + request_note
                    ),
                },
                {"type": "text", "text": spec_block(spec)},
                *([{"type": "text", "text": anchors_block(ctx.anchors)}] if ctx.anchors else []),
                *renders_block(contact_sheet, analysis),
            ],
        },
    ]


async def run(
    ctx: AgentCtx,
    spec: PartSpec,
    *,
    base_stl: Path,
    base_it: int,
    analysis: dict[str, Any] | None = None,
    contact_sheet: Path | None = None,
    request: dict[str, Any] | None = None,
    deadline: float | None = None,
) -> AgentOptResult | None:
    """Run the optimize session. None → caller uses the deterministic path.

    `request` carries a standalone-optimize API request: its load_cases /
    optimize / material values override the spec-derived defaults and are
    shown to the model as an explicit user request. `deadline` (monotonic)
    bounds the phase — tool calls that cannot finish in time are refused so
    the session commits a completed run instead of dying mid-SIMP.
    """
    s = ctx.runtime_settings
    if not s.agent_struct_tools or not tools_enabled(ctx, "vision"):
        return None

    req_cases, req_err = _valid_cases((request or {}).get("load_cases"))
    if request and req_err:
        log.warning("optimize.request_cases_invalid", error=req_err)
    spec_cases = req_cases or [c.model_dump() for c in spec.load_cases]
    default_material = (request or {}).get("material") or ctx.settings.fem_material or spec.material
    session = _setup_session(
        ctx,
        base_stl=base_stl,
        base_it=base_it,
        spec=spec,
        spec_cases=spec_cases,
        default_material=default_material,
        anchors=(ctx.anchors or {}).get("holes") or None,
        contact_sheet=contact_sheet,
        deadline=deadline,
    )
    messages = _opt_messages(ctx, spec, request, analysis, contact_sheet)

    try:
        res = await run_session(
            ctx,
            "vision",
            messages,
            session=session,
            json_schema=None,
            max_calls=s.agent_tool_max_calls,
            step=STEP,
            iteration=base_it,
            temperature=0.0,
        )
    except Exception as e:
        log.warning("optimize.session_failed", error=str(e))
        return None

    sub = res.submitted
    if sub is None:
        # budget/turns exhausted without done() — salvage the best completed
        # run instead of discarding all executor work and re-running the
        # deterministic single-shot path from scratch
        salvage = _best_run_index(session) if res.exhausted else None
        if salvage is None:
            return None
        log.info("optimize.salvaged_run", run=salvage, reason=res.end_reason)
        chosen = salvage
        reasoning = (
            "auto-selected best completed run — the session exhausted its "
            "budget without calling done()"
        )
    else:
        chosen = int(sub.get("chosen_run", -1))
        reasoning = str(sub.get("reasoning", ""))
    out = AgentOptResult(
        chosen_run=chosen,
        reasoning=reasoning,
        runs=[_run_summary(r, chosen == r["index"], reasoning) for r in session.struct_runs],
    )
    if chosen >= 0:
        entry = session.struct_runs[chosen]
        out.iteration = entry["iteration"]
        out.res = entry["res"]
    return out
