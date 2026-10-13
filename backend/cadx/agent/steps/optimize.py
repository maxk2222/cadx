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
    sfs = [
        c.get("safety_factor") for c in fem.get("cases") or [] if c.get("safety_factor") is not None
    ]
    return {
        "index": r["index"],
        "it": r.get("iteration"),
        "params": {k: v for k, v in (r.get("params") or {}).items() if k != "load_cases"},
        "volfrac": top.get("volfrac_target"),
        "achieved": top.get("volfrac_achieved"),
        "iterations": top.get("iterations"),
        "worst_sf": min(sfs) if sfs else None,
        "min_member_mm": top.get("min_member_mm"),
        "thin_fraction": top.get("thin_fraction"),
        "n_components": top.get("n_components"),
        "warnings": (top.get("warnings") or [])[:5],
        "chosen": chosen,
        **({"reasoning": str(reasoning)} if chosen and reasoning else {}),
    }


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
    spec_cases: list[dict[str, Any]]
    default_material: str
    alloc: Any
    contact_sheet: Path | None


def _make_fem_fn(ops: _SessionOps) -> Any:
    ctx = ops.ctx

    async def fem_fn(a: dict[str, Any]) -> dict[str, Any]:
        cases, err = _valid_cases(a.get("load_cases"))
        if err:
            return {"ok": False, "error": err}
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


def _make_topopt_fn(ops: _SessionOps) -> Any:
    ctx = ops.ctx

    async def topopt_fn(a: dict[str, Any]) -> dict[str, Any]:
        cases, err = _valid_cases(a.get("load_cases"))
        if err:
            return {"ok": False, "error": err}
        it = next(ops.alloc)
        opt: dict[str, Any] = {}
        if a.get("target_volume_fraction") is not None:
            opt["target_volume_fraction"] = float(a["target_volume_fraction"])
        if a.get("max_iterations") is not None:
            opt["max_iterations"] = int(a["max_iterations"])
        keep, kerr = _valid_regions(a.get("keep_regions"))
        if kerr:
            return {"ok": False, "error": kerr}
        if keep:
            opt["keep_regions"] = keep
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


def _setup_session(
    ctx: AgentCtx,
    *,
    base_stl: Path,
    base_it: int,
    spec_cases: list[dict[str, Any]],
    default_material: str,
    anchors: list[dict[str, Any]] | None,
    contact_sheet: Path | None,
) -> ToolSession:
    """ToolSession with fem/topopt/preview callbacks; each executor call
    allocates its own iteration dir and attaches renders as `_images`."""
    ops = _SessionOps(
        ctx=ctx,
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
    )
    return ToolSession(
        ctx,
        kind="optimize",
        fem_fn=_make_fem_fn(ops),
        topopt_fn=_make_topopt_fn(ops),
        preview_fn=_make_preview_fn(ops),
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
) -> AgentOptResult | None:
    """Run the optimize session. None → caller uses the deterministic path.

    `request` carries a standalone-optimize API request: its load_cases /
    optimize / material values override the spec-derived defaults and are
    shown to the model as an explicit user request.
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
        spec_cases=spec_cases,
        default_material=default_material,
        anchors=(ctx.anchors or {}).get("holes") or None,
        contact_sheet=contact_sheet,
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
        return None
    chosen = int(sub.get("chosen_run", -1))
    out = AgentOptResult(
        chosen_run=chosen,
        reasoning=str(sub.get("reasoning", "")),
        runs=[
            _run_summary(r, chosen == r["index"], sub.get("reasoning", ""))
            for r in session.struct_runs
        ],
    )
    if chosen >= 0:
        entry = session.struct_runs[chosen]
        out.iteration = entry["iteration"]
        out.res = entry["res"]
    return out
