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
    # headline SF: the verified realization worst-case, then the printed-
    # geometry export SF, then the voxel-mask report — a solved 0.0 is a
    # verdict, so None-check rather than or-chain
    sfs = [
        c.get("safety_factor") for c in fem.get("cases") or [] if c.get("safety_factor") is not None
    ]
    worst_sf = top.get("sf_verified")
    if worst_sf is None:
        worst_sf = top.get("sf_export")
    if worst_sf is None:
        worst_sf = min(sfs) if sfs else None
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
        "volfrac_requested": top.get("volfrac_requested"),
        "achieved": top.get("volfrac_achieved"),
        "iterations": top.get("iterations"),
        # B18 outcome vocabulary + O6 pitch label on the run digest
        "status": top.get("status"),
        "pitch_mm": (res.get("mesh") or {}).get("pitch_mm"),
        "meets_safety_factor": (top.get("verification") or {}).get("meets_safety_factor"),
        "worst_sf": worst_sf,
        "sf_mask": top.get("sf_mask"),
        "sf_export": top.get("sf_export"),
        "extract": top.get("extract"),
        "verification": top.get("verification"),
        "min_member_mm": top.get("min_member_mm"),
        "thin_fraction": top.get("thin_fraction"),
        "n_components": top.get("n_components"),
        "overhang_pct": (top.get("verification") or {}).get("overhang_pct"),
        "n_bodies_export": top.get("n_bodies_export"),
        "load_path_ok": top.get("load_path_ok"),
        "verify_spec": (res.get("verify_spec") or {}).get("status"),
        "warnings": (top.get("warnings") or [])[:5],
        **refine_block,
        "chosen": chosen,
        **({"reasoning": str(reasoning)} if chosen and reasoning else {}),
    }


def _best_run_index(session: ToolSession) -> int | None:
    """Index of the best completed run — for auto-commit when the session
    exhausts its budget without done(). Prefers sf_export (the safety factor
    on the printed export geometry), else the run's worst case SF; None when
    no run produced a metric. Multi-body or unverified exports are skipped —
    a disconnected print or a degenerate solve must never win by SF."""
    best_idx: int | None = None
    best_sf = -1.0
    for r in session.struct_runs:
        res = r.get("res") or {}
        top = res.get("topopt") or {}
        fem = res.get("fem") or {}
        bodies = max(
            int(top.get("n_components") or 1),
            int(top.get("n_bodies_export") or 1),
        )
        if bodies > 1 or (top.get("verification") or {}).get("valid") is False:
            continue
        # a canonical FAIL (e.g. the part overshoots the declared envelope)
        # must not win by safety factor either
        if (res.get("verify_spec") or {}).get("status") == "FAIL":
            continue
        # a broken load path means at least one case cannot reach its
        # fixtures through the printed part — structurally disconnected
        # even when n_bodies reads 1 (smoothing bridges voxel gaps). A
        # high sf_export on a disconnected part is meaningless; None is
        # "certificate skipped" and stays eligible.
        if top.get("load_path_ok") is False:
            continue
        # a refine run whose applied features left no matching tunnel/dimple
        # (bore drilled into a void, wrong diameter) failed the task too —
        # salvaging it ships broken features
        if (res.get("features") or {}).get("refine", {}).get("verify_misses"):
            continue
        sfs = [c.get("safety_factor") for c in fem.get("cases") or [] if c.get("safety_factor")]
        # prefer the verified realization worst-case, then the export SF,
        # then the mask SF — a solved 0.0 is a verdict, never or-chain
        sf = top.get("sf_verified")
        if sf is None:
            sf = top.get("sf_export")
        if sf is None:
            sf = min(sfs) if sfs else None
        # when the voxel-mask and export solves diverge by >=2x, the part's
        # integrity depends on discretization luck — rank it by the
        # pessimistic bound, not the optimistic export number
        sf_mask, sf_exp = top.get("sf_mask"), top.get("sf_export")
        if sf_mask is not None and sf_exp is not None:
            lo, hi = min(float(sf_mask), float(sf_exp)), max(float(sf_mask), float(sf_exp))
            if lo > 0 and hi / lo >= 2.0:
                sf = min(sf, lo) if sf is not None else lo
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


def _fem_case_images(res: dict[str, Any], it_dir: Path) -> tuple[list[str], list[str]]:
    """Per-case FEM renders (vonmises/deformed PNGs) under <it_dir>/fem/ —
    the fem / fem_export reports carry bare filenames. Returns
    ``(vonmises, deformed)`` — vonmises carries the decision-relevant stress
    picture; deformed views are nice-to-have and the first to be dropped
    under the image budget."""
    fem_dir = it_dir / "fem"
    von: list[str] = []
    deformed: list[str] = []
    for rep_key in ("fem", "fem_export"):
        for c in (res.get(rep_key) or {}).get("cases") or []:
            for p, bucket in ((c.get("vonmises_png"), von), (c.get("deformed_png"), deformed)):
                if p and (fem_dir / p).exists():
                    bucket.append(str(fem_dir / p))
    return von, deformed


#: image budget for one topopt_run result — every render stays on disk, and
#: a single result exceeding ~8 images costs more context than the marginal
#: view is worth (a 17-image result ≈ 40k tokens on a local model; optimize
#: sessions accumulate renders across runs, so the cap is also the main
#: context lever — N4)
_TOPOPT_IMAGE_BUDGET = 8


def _topopt_images(
    ops: _SessionOps,
    res: dict[str, Any],
    it_dir: Path,
) -> list[str]:
    """Priority-ordered image set for a topopt_run result, capped at the
    budget: base sheet → run sheet/thumb → region previews → shape views →
    vonmises overlays → deformed views (first dropped)."""
    head: list[str] = []
    if ops.contact_sheet:
        head.append(str(ops.contact_sheet))
    for key in ("contact_sheet", "thumbnail"):
        if rel := res.get(key):
            with contextlib.suppress(ValueError):
                p = storage.resolve(rel)
                if p.exists():
                    head.append(str(p))
    for rel in (res.get("topopt") or {}).get("regions_png") or []:
        with contextlib.suppress(ValueError):
            p = storage.resolve(rel)
            if p.exists():
                head.append(str(p))
    views = _render_view_images(res)
    von, deformed = _fem_case_images(res, it_dir)
    images = head + views + von
    if len(images) < _TOPOPT_IMAGE_BUDGET:
        images += deformed[: _TOPOPT_IMAGE_BUDGET - len(images)]
    else:
        images = images[:_TOPOPT_IMAGE_BUDGET]
    if omitted := (len(head) + len(views) + len(von) + len(deformed)) - len(images):
        res.setdefault("warnings", []).append(
            f"{omitted} render image(s) omitted from this result (context budget) — "
            "they are written under the iteration dir"
        )
    return images


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
        # O1: probes default to 2x the verification pitch — a coarse SF
        # estimate, ~8x fewer elements. An explicit voxel_mm arg wins;
        # preview/check_load keep fem_voxel_mm (their region resolution must
        # match the grid a topopt_run would use) and the export verification
        # in the optimize pipeline always runs at the fine pitch.
        export_pitch = float(ctx.settings.fem_voxel_mm or ctx.runtime_settings.fem_voxel_mm or 0.0)
        probe_voxel = a.get("voxel_mm") or (2.0 * export_pitch if export_pitch else None)
        res = await ctx.executor.fem_analyze(
            {
                **ops.base,
                "iteration": it,
                "load_cases": cases if cases is not None else ops.spec_cases,
                "material": str(a.get("material") or "") or ops.default_material,
                "voxel_mm": probe_voxel,
            }
        )
        # O6: label a coarse probe pitch so the SF isn't read as the export
        # verification number
        fem_d = res.get("fem") or {}
        probe_pitch = (fem_d.get("grid") or {}).get("pitch")
        if (
            res.get("ok")
            and probe_pitch is not None
            and export_pitch
            and float(probe_pitch) > export_pitch + 1e-9
        ):
            fem_d["pitch_note"] = (
                f"probe at {float(probe_pitch):g}mm — coarse estimate; "
                "export verification runs finer"
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
        # impossible-target diagnostic — removing material cannot raise the
        # safety factor above the full-density baseline, so a probe already
        # below the spec minimum means no SIMP run will pass it. Surface that
        # once instead of letting the agent burn its run budget discovering it.
        opt = ops.spec.optimize
        if opt and probe_sfs and min(probe_sfs) < opt.safety_factor_min:
            res.setdefault("fem", {}).setdefault("warnings", []).append(
                f"baseline worst SF {min(probe_sfs)} is below the spec minimum "
                f"{opt.safety_factor_min} — topology optimization removes "
                "material and cannot raise the safety factor above this "
                "baseline; keep volfrac near the keep-region floor, or commit "
                "the best run rather than searching for a passing one"
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
    """Map `topopt_run`/`infill_run` tool args onto `OptimizeSpec` fields."""
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
        "volfrac_init": float,
        "volfrac_ramp_iters": int,
        "loadpath_corridor_mm": float,
        "mma_final_beta": float,
        "grow_mm": float,
        "aniso_factor": float,
        "beso_er": float,
        "infill_shell_mm": float,
        "infill_period_mm": float,
        "infill_min_density": float,
        "infill_max_density": float,
        "infill_alpha": float,
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
        "warmstart",
        "infill_scope",
        "infill_guide_solver",
        "infill_style",
    ):
        if a.get(k):
            opt[k] = str(a[k])
    if a.get("infill_seed") is not None:
        opt["infill_seed"] = int(a["infill_seed"])
    for k in (
        "self_supporting",
        "multires",
        "robust",
        "aniso_material",
        "infill_octet",
        "verify_robust",
        "verify_homogenized",
        "recut_bores",
    ):
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
        it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, it)
        res["_images"] = _topopt_images(ops, res, it_dir)
        return res

    return topopt_fn


def _make_infill_fn(ops: _SessionOps) -> Any:
    ctx = ops.ctx

    async def infill_fn(a: dict[str, Any]) -> dict[str, Any]:
        cases, err = _valid_cases(a.get("load_cases"))
        if err:
            return {"ok": False, "error": err}
        if (
            guard := _budget_exceeded(ops, ctx.runtime_settings.topopt_timeout_s + 180)
        ) is not None:
            return guard
        it = next(ops.alloc)
        opt = _opt_overrides(a)
        opt["structure_mode"] = "infill"
        await ctx.emit(
            "optimize",
            it,
            f"shell+infill run — {opt.get('infill_scope', 'guided')} scope",
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
        it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, it)
        res["_images"] = _topopt_images(ops, res, it_dir)
        return res

    return infill_fn


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
        # re-verification renders written under <iteration_dir>/fem/ —
        # capped like a topopt run: images accumulate across the session
        it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, it)
        images += _render_view_images(res)
        von, deformed = _fem_case_images(res, it_dir)
        images += von + deformed
        if len(images) > _TOPOPT_IMAGE_BUDGET:
            omitted = len(images) - _TOPOPT_IMAGE_BUDGET
            images = images[:_TOPOPT_IMAGE_BUDGET]
            res.setdefault("warnings", []).append(
                f"{omitted} render image(s) omitted from this result (context budget) — "
                "they are written under the iteration dir"
            )
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
    target_volfrac: float | None = None,
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
            "spec": spec.model_dump(mode="json"),
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
        infill_fn=_make_infill_fn(ops),
        target_volfrac=target_volfrac,
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
            # tools_protocol.md describes the code-buffer edit session —
            # this session's tools are analyze_fem/topopt_run/refine_features;
            # concatenating it taught the model nonexistent verbs and could
            # invite hallucinated verify/render_model calls
            "content": system_prompt("optimize_system.md"),
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
    # the session's volume target — request overrides the spec, the spec
    # overrides the runtime default; preview/check_load warn and topopt_run
    # refuses when the keep/load/fixed floor sits above it
    req_opt = (request or {}).get("optimize") or {}
    target_volfrac = (
        req_opt.get("target_volume_fraction")
        or (spec.optimize.target_volume_fraction if spec.optimize else None)
        or s.topopt_default_volfrac
    )
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
        target_volfrac=target_volfrac,
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
