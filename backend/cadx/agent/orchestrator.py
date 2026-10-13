"""Agent orchestrator: the bounded generate-verify-revise state machine.

INTAKE -> (CLARIFY?) -> PLAN -> mode resolution -> one of:
  cad:    [CODEGEN -> EXECUTE -> (REPAIR x N) -> ANALYZE -> CRITIQUE] x K -> FINALIZE
  mesh:   [MESH_GENERATE -> ORIENT(once) -> PROCESS_MESH -> ANALYZE -> CRITIQUE] x K -> FINALIZE
  hybrid: MESH_GENERATE -> ORIENT -> PROCESS_MESH(+solid) -> cad loop with load_mesh_solid()
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from cadx import storage
from cadx.agent.anchors import build_anchors
from cadx.agent.autofix import propose_fixes
from cadx.agent.mesh_client import MeshGenError, MeshGenRequest
from cadx.agent.steps import codegen as codegen_step
from cadx.agent.steps import critique as critique_step
from cadx.agent.steps import improve as improve_step
from cadx.agent.steps import intake as intake_step
from cadx.agent.steps import measure as measure_step
from cadx.agent.steps import mesh_orient as mesh_orient_step
from cadx.agent.steps import mesh_refine as mesh_refine_step
from cadx.agent.steps import optimize as optimize_step
from cadx.agent.steps import plan as plan_step
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx, InputAsset
from cadx.cad.allowlist import check_code
from cadx.cad.features import compare_features
from cadx.cad.fem.analyze import FemReport, fem_feedback_lines
from cadx.cad.mesh_ops import (
    MeshOp,
    MeshPostParams,
    merge_post_params,
    mesh_params_schema,
)
from cadx.cad.partspec import check_standard_part_calls
from cadx.cad.patch import apply_patch
from cadx.llm.schemas import CodeResponse, Critique, PartSpec, PatchResponse
from cadx.logging import get_logger

log = get_logger(__name__)


@dataclass
class Candidate:
    iteration: int
    code_response: CodeResponse | None
    exec_result: dict[str, Any]
    critique: Critique | None
    dim_issues: list[str]
    mesh_post: dict[str, Any] | None = None  # MeshPostParams used (mesh/hybrid modes)
    feature_findings: list[dict[str, Any]] = field(default_factory=list)

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
    bom: list[dict[str, Any]] | None = None
    mesh: dict[str, Any] | None = None
    fem: dict[str, Any] | None = None
    resolved_mode: str | None = None
    mode_note: str = ""
    score: int | None = None
    iterations: int = 0
    best_iteration: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # kind|view -> relpath
    questions: list[dict[str, Any]] | None = None
    resume_state: dict[str, Any] | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    fallback_to_cad: bool = False  # mesh loop hard-failed on a dimensioned spec
    switch_requested: bool = False  # plateaued — try the next strategy approach


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def stated_dim_count(spec: PartSpec) -> int:
    return sum(1 for d in (spec.length_x, spec.width_y, spec.height_z) if d.source == "stated")


def adaptive_max_iterations(
    spec: PartSpec, requested: int, *, n_images: int = 0, organic: bool = False
) -> int:
    """Simple dimensioned parts must not burn the global iteration budget —
    a stated wedge with one hole converges in a handful of attempts, whereas
    100 iterations of LLM+executor rounds take hours. Image-driven and organic
    (mesh/hybrid) parts are never "simple": visual convergence needs many more
    attempts, and weak intake may under-report the true feature count."""
    if organic or n_images >= 2:
        return requested
    stated = stated_dim_count(spec)
    n_features = len(spec.features)
    if stated >= 2 and n_features <= 2:
        return min(requested, 5)
    if stated >= 1 and n_features <= 6:
        return min(requested, 15)
    return requested


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


def build_issues(
    critique: Critique | None,
    dim_issues: list[str],
    feature_findings: list[dict[str, Any]],
    dfm: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Structured issue list for the IMPROVE step + trace/UI persistence."""
    issues: list[dict[str, Any]] = []
    for i in dim_issues:
        issues.append({"source": "dims", "severity": "major", "what": f"dimension mismatch — {i}"})
    for f in feature_findings:
        issues.append(
            {
                "source": "features",
                "severity": f.get("severity", "minor"),
                "what": f"{f.get('feature')}: expected {f.get('expected')}, actual {f.get('actual')}",
                "fix_hint": f.get("fix_hint", ""),
            }
        )
    if critique:
        for d in critique.discrepancies:
            issues.append(
                {
                    "source": "critique",
                    "severity": d.severity,
                    "what": d.what,
                    "where": d.where,
                    "fix_hint": d.fix_hint,
                }
            )
        for v in critique.feature_verdicts:
            if v.status in ("missing", "wrong"):
                issues.append(
                    {
                        "source": "critique",
                        "severity": "major" if v.status == "missing" else "minor",
                        "what": f"feature '{v.name}' is {v.status}"
                        + (f" — {v.note}" if v.note else ""),
                    }
                )
    if dfm:
        for w in dfm.get("warnings", []):
            if w.get("severity") == "high":
                issues.append({"source": "dfm", "severity": "major", "what": w["message"]})
    return issues[:12]


def _resolve_patch(
    base_code: str, resp: PatchResponse, patch_mode: str
) -> tuple[str | None, list[str]]:
    """Turn a PatchResponse into code: prefer edits; full_code fallback/explicit."""
    if resp.full_code and (not resp.edits or patch_mode == "never"):
        return resp.full_code, []
    if not resp.edits:
        return None, ["model returned neither edits nor full_code"]
    patched, errors = apply_patch(base_code, resp.edits)
    if patched is not None:
        return patched, []
    if resp.full_code:
        check = check_code(resp.full_code)
        if check.ok:
            return resp.full_code, errors
        errors = [*errors, "full_code also fails contract: " + "; ".join(check.errors[:4])]
    return None, errors


def _session_verify_fn(ctx: AgentCtx, iteration: int):
    """verify-tool callback: execute session buffer code in the sandbox."""

    def run(code: str):
        return ctx.executor.execute(_exec_payload(ctx, code, iteration, full_exports=False))

    return run


def _params_vs_spec(spec: PartSpec, literal_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Cheap consistency check: stated spec dims should appear in PARAMS values."""
    issues: list[dict[str, Any]] = []
    if not literal_params:
        return issues
    vals = [float(v) for v in literal_params.values() if isinstance(v, int | float)]
    for dim, name in (
        (spec.length_x, "length/X"),
        (spec.width_y, "width/Y"),
        (spec.height_z, "height/Z"),
    ):
        if dim.source != "stated":
            continue
        if vals and not any(abs(v - dim.value) <= 0.2 * dim.value for v in vals):
            issues.append(
                {
                    "source": "contract",
                    "severity": "minor",
                    "what": (
                        f"spec {name}={dim.value:g}mm does not appear in PARAMS — "
                        "possibly hardcoded wrong or derived incorrectly"
                    ),
                    "fix_hint": f"ensure a PARAMS value controls {name}={dim.value:g}mm",
                }
            )
    return issues


def resolve_mode(ctx: AgentCtx) -> tuple[str, str]:
    """Decide the effective generation mode (cad | mesh | hybrid) + reason.

    Requested mode comes from ctx.gen_mode ("auto" defers to the planner's
    plan.approach). Guards: mesh/hybrid need image input AND a reachable mesh
    server — otherwise we fall back to CAD.
    """
    requested = ctx.gen_mode or "auto"
    if requested == "topopt":
        # topology optimization is a post-phase: first build the base part
        # (planner's approach), then run SIMP + FEM verification on it
        ctx.topopt_requested = True
        requested = "auto"
    has_images = any(a.kind in IMAGE_INPUT_KINDS for a in ctx.assets)
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
        note = f"auto: planner chose '{approach}'" + (f" — {reason}" if reason else "")
        return approach, note
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
        "runtime_settings": ctx.runtime_settings.executor_values(),
        "full_exports": full_exports,
        "name": "model",
    }
    if ctx.hybrid_solid is not None:
        payload["mesh_solid"] = str(ctx.hybrid_solid)
        payload["mesh_bbox"] = ctx.hybrid_mesh_info.get("bbox_size")
    if ctx.anchors:
        payload["anchors"] = ctx.anchors
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
        "runtime_settings": ctx.runtime_settings.executor_values(),
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
    if exec_result.get("bom"):
        bom_path = v_dir / "bom.json"
        bom_path.write_text(json.dumps(exec_result["bom"], indent=2), encoding="utf-8")
        out["bom_json"] = storage.relpath(bom_path)
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


def _copy_fem_artifacts(ctx: AgentCtx, iteration: int) -> dict[str, str]:
    """Copy fem/ + topopt/ iteration outputs to the version root."""
    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration)
    v_dir = storage.version_dir(ctx.project_id, ctx.version_id)
    out: dict[str, str] = {}
    for sub in ("fem", "topopt"):
        src_dir = it_dir / sub
        if not src_dir.exists():
            continue
        dst_dir = storage.ensure(v_dir / sub)
        for f in src_dir.iterdir():
            if f.is_file():
                dst = dst_dir / f.name
                shutil.copy2(f, dst)
                out[f"fem:{f.stem}"] = storage.relpath(dst)
    return out


async def _run_topopt_phase(ctx: AgentCtx, outcome: RunOutcome, spec: PartSpec) -> RunOutcome:
    """OPTIMIZE phase after a successful base generation (gen_mode=topopt or
    plan.optimize): run SIMP on the best candidate + FEM-verify the result."""
    if outcome.status not in ("succeeded", "needs_review"):
        return outcome
    if not spec.load_cases:
        outcome.mode_note += " | topopt skipped — no load cases in the spec"
        await ctx.emit("optimize", 0, "skipped: intake declared no load cases")
        return outcome

    base_it = outcome.best_iteration or outcome.iterations or 1
    base_mode = outcome.resolved_mode or "cad"
    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, base_it)
    stl = it_dir / ("_run/model.stl" if base_mode == "mesh" else "renders/mesh.stl")
    if not stl.exists():
        stl = it_dir / "renders/mesh.stl"
    if not stl.exists():
        outcome.mode_note += " | topopt skipped — no base mesh artifact"
        return outcome

    it = (outcome.iterations or base_it) + 1

    # agent-driven session: FEM probes + parameterized SIMP runs, agent picks
    # the best run — falls back to the deterministic single shot below
    res: dict[str, Any] | None = None
    base_sheet = it_dir / "contact_sheet.png"
    agent = await optimize_step.run(
        ctx,
        spec,
        base_stl=stl,
        base_it=outcome.iterations or base_it,
        analysis=outcome.analysis,
        contact_sheet=base_sheet if base_sheet.exists() else None,
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
                "optimize": {
                    **(spec.optimize.model_dump() if spec.optimize else {}),
                    **(
                        {"target_volume_fraction": ctx.settings.topopt_volfrac}
                        if ctx.settings.topopt_volfrac
                        else {}
                    ),
                    "max_iterations": ctx.settings.topopt_iterations,
                },
                "material": ctx.settings.fem_material or spec.material,
                "voxel_mm": ctx.settings.fem_voxel_mm,
                "runtime_settings": ctx.runtime_settings.executor_values(),
                "name": "model",
                "anchors": (ctx.anchors or {}).get("holes"),
            }
        )
    if not res.get("ok"):
        outcome.mode_note += f" | topopt failed: {res.get('error')}"
        await ctx.emit("optimize", it, f"failed: {res.get('error')}")
        return outcome

    arts = _copy_best_artifacts(ctx, it, res)
    for fmt, rel in (res.get("exports") or {}).items():
        if rel:
            arts[f"export:{fmt}"] = rel
    arts.update(_copy_fem_artifacts(ctx, it))
    outcome.artifacts.update(arts)
    outcome.fem = res.get("fem")
    outcome.analysis = res.get("analysis")
    top = res.get("topopt") or {}
    if agent is not None and agent.runs:
        top = {**top, "agent_runs": agent.runs}
    outcome.mesh = {**(outcome.mesh or {}), "pipeline": "topopt", "topopt": top}
    outcome.resolved_mode = "topopt"
    sfs = [
        c.get("safety_factor")
        for c in (outcome.fem or {}).get("cases", [])
        if c.get("safety_factor") is not None
    ]
    worst = min(sfs) if sfs else None
    if worst is not None and worst < 1.0:
        outcome.status = "needs_review"
        outcome.mode_note += f" | optimized part fails: SF {worst:.2f} < 1"
    await ctx.emit(
        "optimize",
        it,
        f"done — volfrac {top.get('volfrac_achieved')}, "
        f"worst SF {worst}, {top.get('iterations')} iterations",
    )
    return outcome


def _write_mesh_json(project_id: Any, version_id: Any, info: dict[str, Any]) -> None:
    mesh_dir = storage.ensure(storage.version_dir(project_id, version_id) / "mesh")
    clean_info = {k: v for k, v in info.items() if not k.startswith("_")}
    (mesh_dir / "mesh.json").write_text(json.dumps(clean_info, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# mesh generation helpers                                                      #
# --------------------------------------------------------------------------- #


def _mesh_source_images(
    ctx: AgentCtx, iteration: int, spec: PartSpec | None = None
) -> tuple[list[Any], list[str]]:
    """Pick input images for the generator: planner's mesh_source_images order,
    hero first; later iterations rotate the hero among image assets.

    Generator roles come from the intake's image_roles when available — a top
    view of a flat part feeds the generator's "front" slot (its hero view),
    side/back views map to their namesake roles, everything else is "other".
    """
    images = [a for a in ctx.assets if a.kind in IMAGE_INPUT_KINDS]
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

    intake_roles: dict[str, str] = {}
    flat = False
    if spec is not None:
        intake_roles = {r.asset_id: r.role for r in spec.image_roles if r.asset_id}
        dims = (spec.length_x.value, spec.width_y.value, spec.height_z.value)
        flat = dims[2] > 0 and dims[2] < 0.3 * max(dims[0], dims[1], 1e-9)
    if flat:
        # The measure step classifies plan views deterministically (silhouette
        # aspect vs. stated dims). It beats intake labels — models keep calling
        # a face-up photo of a flat part "front", which feeds the generator a
        # plan view as its hero slot and produces a standing object.
        for aid, m in (ctx.measurements or {}).items():
            if m.get("role") in ("top", "bottom"):
                intake_roles[aid] = str(m["role"])
    if not intake_roles:
        return ordered, ["front"] + ["other"] * (len(ordered) - 1)
    return _map_gen_roles(ordered, intake_roles, flat)


def _map_gen_roles(
    ordered: list[Any], intake_roles: dict[str, str], flat: bool
) -> tuple[list[Any], list[str]]:
    """Translate intake image_roles into generator view roles."""

    def _gen_role(a: Any) -> str:
        role = intake_roles.get(a.asset_id, "other")
        if role == "top":
            return "front" if flat else "other"
        if role == "bottom":
            return "back" if flat else "other"
        if role in ("front", "back", "left", "right"):
            return role
        return "other"

    seen: set[str] = set()
    roles: list[str] = []
    for a in ordered:
        r = _gen_role(a)
        if r in seen:  # a second image for one slot degrades to "other"
            r = "other"
        seen.add(r)
        roles.append(r)
    # hero must be the generator's front view
    front_i = next((i for i, r in enumerate(roles) if r == "front"), None)
    if front_i is not None and front_i != 0:
        ordered.insert(0, ordered.pop(front_i))
        roles.insert(0, roles.pop(front_i))
    if "front" not in roles:
        roles[0] = "front"
    return ordered, roles


async def _resolve_mesh_adapter(ctx: AgentCtx, n_images: int) -> tuple[str, int]:
    """Pick (model_id, max_images) so the sidecar never sees an impossible request.

    'auto' stays 'auto' for single-image runs (server-side device preference);
    multi-image runs pick an explicit multiview adapter when one is available,
    else degrade to the best single-image adapter and cap the image list.
    """
    configured = ctx.settings.mesh_model or ctx.runtime_settings.mesh_model
    try:
        health = await ctx.mesh.health() if ctx.mesh else None
    except Exception:
        health = None
    models = [m for m in (health.models if health else []) if m.available and m.id != "fake"]
    device = health.device if health else ""
    if configured and configured != "auto":
        m = next((m for m in models if m.id == configured), None)
        return configured, (m.max_images if m else n_images)
    if not models:
        return "auto", n_images
    if n_images > 1:
        multiview = [m for m in models if m.supports_multiview]
        if multiview:
            pick = sorted(
                multiview, key=lambda m: (m.device == device, m.max_images), reverse=True
            )[0]
            return pick.id, pick.max_images
        pick = sorted(models, key=lambda m: m.device == device, reverse=True)[0]
        return pick.id, 1
    return "auto", n_images


def _base_seed(ctx: AgentCtx) -> int:
    """Job's deterministic base seed — stable across restarts.

    Derived from project+prompt+image ids (not the version uuid: a restart
    spawns a new version and the mesh cache only pays off when the request
    — and thus the seed — stays identical)."""
    if ctx.settings.mesh_seed is not None:
        return ctx.settings.mesh_seed
    import hashlib

    h = hashlib.sha256()
    h.update(str(ctx.project_id).encode())
    h.update((ctx.prompt or "").encode())
    for aid in sorted(a.asset_id for a in ctx.assets if a.kind in IMAGE_INPUT_KINDS):
        h.update(str(aid).encode())
    return int.from_bytes(h.digest()[:4]) % 100_000


async def _autogen_hero_image(ctx: AgentCtx, spec: PartSpec | None, plan: Any) -> None:
    """Synthesise a hero reference image for mesh/hybrid runs without inputs.

    Runs between PLAN and mode resolution: appending the generated asset to
    ctx.assets makes `has_images` true, so the existing mesh/hybrid guards
    and seed/mode logic work unchanged. Every failure degrades silently to
    the previous CAD fallback.
    """
    if ctx.image is None or not ctx.runtime_settings.image_autogen_mesh:
        return
    if any(a.kind in IMAGE_INPUT_KINDS for a in ctx.assets):
        return
    requested = ctx.gen_mode or "auto"
    if requested == "topopt":
        requested = "auto"  # topopt resolves like auto for the base part
    approach = plan.approach if plan and plan.approach else "cad"
    if not (
        requested in ("mesh", "hybrid") or (requested == "auto" and approach in ("mesh", "hybrid"))
    ):
        return
    try:
        models = await ctx.image.models()
    except Exception:
        models = []
    if not any(m.available and not m.last_error for m in models):
        return

    # same fallback chain as the mesh subject (:866)
    subject = (
        ctx.mesh_info.get("_regen_hint")
        or (plan.mesh_subject if plan else "")
        or (spec.name if spec else "")
        or ctx.prompt[:80]
    )
    from cadx.agent.context import read_prompt

    s = ctx.runtime_settings
    prompt = read_prompt("image_hero.md").format(subject=subject, prompt=ctx.prompt.strip())
    await ctx.emit("imagegen", 0, "no reference image — generating a hero image")

    async def _prog(msg: str, _p: float) -> None:
        await ctx.emit("imagegen", 0, msg)

    tmp = Path(tempfile.mkdtemp(prefix="cadx-hero-"))
    try:
        from cadx.agent.image_client import ImageGenRequest

        side = min(1024, s.image_max_side)
        req = ImageGenRequest(
            prompt=prompt,
            model=s.image_model,
            seed=_base_seed(ctx),
            count=1,
            width=side,
            height=side,
            steps=s.image_steps,
            guidance=s.image_guidance,
            out_dir=tmp,
        )
        result = await ctx.image.generate(req, _prog, ctx.check_cancel)
        from cadx.agent.image_assets import persist_generated_images
        from cadx.db.session import session_scope

        async with session_scope() as db:
            assets = persist_generated_images(
                db,
                ctx.project_id,
                result.image_paths,
                prompt=subject,
                caption=f"generated hero — {subject}",
                gen_params={
                    "prompt": prompt,
                    "subject": subject,
                    "model": result.model,
                    "backend": result.backend,
                    "seed": result.seed,
                    "steps": result.steps,
                    "guidance": result.guidance,
                    "width": side,
                    "height": side,
                    "duration_s": result.duration_s,
                    "source": "hero_autogen",
                },
            )
            await db.commit()
        for a in assets:
            ctx.assets.append(
                InputAsset(
                    asset_id=str(a.id),
                    kind="generated_image",
                    path=storage.resolve(a.path),
                    caption=a.caption,
                    filename=a.filename,
                )
            )
            ctx.generated_asset_ids.append(str(a.id))
        await ctx.emit("imagegen", 0, f"hero image generated ({result.model})")
    except Exception as e:
        log.warning("agent.hero_image_failed", error=str(e))
        await ctx.emit("imagegen", 0, f"hero image generation failed: {e} — continuing without")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def _candidate_seed(
    ctx: AgentCtx, iteration: int, spec: PartSpec, plan: Any, t0: float
) -> int | None:
    """Best-of-N cheap generation: N low-step candidates with different seeds,
    each scored deterministically (aspect + footprint IoU + hole count); the
    winning seed feeds the full-resolution generation. Runs once per job and
    only with >30 min of job budget left — it doubles the generation cost of
    the first roll but saves critique cycles on bad seeds."""
    s = ctx.runtime_settings
    n = s.mesh_candidates_per_iter
    if n <= 1 or ctx.mesh_info.get("_candidates_done") or ctx.mesh is None:
        return None
    if iteration == 1 and ctx.mesh_raw_reuse is not None:
        return None  # parent mesh reuse — no fresh candidates needed
    from cadx.config import get_settings

    if get_settings().agent_job_timeout_s - (time.monotonic() - t0) < 1800:
        return None
    ctx.mesh_info["_candidates_done"] = True
    outline = _measured_plan_outline(ctx)
    base = _base_seed(ctx)
    scores: dict[int, float] = {}
    for i in range(n):
        seed_i = base + i * 1000
        rel = await _generate_mesh(
            ctx, iteration, spec, plan, seed=seed_i, steps=s.mesh_candidate_steps, tag=f"_cand{i}"
        )
        if rel is None:
            continue
        try:
            import trimesh

            from cadx.agent.mesh_score import score_raw_mesh
            from cadx.cad.mesh_ops import (
                orient,
                orient_by_plan_view,
                rest_on_xy_center,
                scale_to_spec,
            )

            m = trimesh.load(storage.resolve(rel), force="mesh")
            if not isinstance(m, trimesh.Trimesh):
                continue
            if outline:
                hit = orient_by_plan_view(m, outline)
                if hit is not None:
                    m = orient(m, hit[0], hit[1], mirror_x=hit[2])
            m, _, _ = scale_to_spec(m, spec)
            m = rest_on_xy_center(m)
            scores[seed_i] = score_raw_mesh(m, spec, outline)
        except Exception as e:
            log.warning("mesh.candidate_score_failed", seed=seed_i, error=str(e)[:160])
    if not scores:
        return None
    best_seed = max(scores, key=lambda k: scores[k])
    await ctx.emit(
        "meshgen",
        iteration,
        f"candidate seeds scored: {scores} → using seed {best_seed}",
    )
    return best_seed


def _prepped_gen_images(ctx: AgentCtx, it_dir: Path, images: list[Any]) -> list[Any]:
    """Crop/background-isolate generation inputs when mesh_prep_images is on."""
    if not ctx.runtime_settings.mesh_prep_images:
        return images
    prep_dir = it_dir / "meshgen_input"
    prepped: list[Any] = []
    for a in images:
        try:
            from cadx.vision.prep import prepare_for_meshgen

            pp = prepare_for_meshgen(a.path, prep_dir / f"{a.asset_id}.png")
            prepped.append(a if pp == a.path else replace(a, path=pp))
        except Exception as e:
            log.warning("mesh.prep_failed", asset=a.asset_id, error=str(e)[:160])
            prepped.append(a)
    return prepped


async def _mesh_cache_lookup(
    ctx: AgentCtx,
    iteration: int,
    images: list[Any],
    roles: list[str],
    model: str,
    seed: int,
    steps: int | None,
    subject: str,
    out_path: Path,
    rel: str,
) -> tuple[str, bool]:
    """Return (cache_key, hit); on a hit copy the cached GLB to out_path
    and fill ctx.mesh_info['generation'] so the caller can return early."""
    from cadx.agent import mesh_cache

    if not ctx.runtime_settings.mesh_cache_enabled:
        return "", False
    key = mesh_cache.cache_key(
        [a.path for a in images],
        roles,
        model=model,
        seed=seed,
        steps=steps if steps is not None else ctx.settings.mesh_steps,
        guidance=ctx.settings.mesh_guidance,
        resolution=ctx.settings.mesh_resolution,
        subject=subject,
    )
    cached = mesh_cache.lookup(key)
    if cached is None:
        return key, False
    await ctx.emit("meshgen", iteration, "mesh from cache")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, out_path)
    ctx.mesh_info["generation"] = {
        "model": model,
        "seed": seed,
        "cached": True,
        "mesh_rel": rel,
        "images": [a.asset_id for a in images],
    }
    return key, True


async def _generate_mesh(
    ctx: AgentCtx,
    iteration: int,
    spec: PartSpec,
    plan: Any,
    *,
    seed: int | None = None,
    steps: int | None = None,
    tag: str = "",
    t0: float | None = None,
) -> str | None:
    """MESH_GENERATE step. Returns the relpath of the raw GLB or None."""
    if ctx.mesh is None:
        ctx.mesh_info["gen_error"] = "mesh server not configured (set mesh_base_url in Settings)"
        return None

    # best-of-N: pick the strongest seed once per job before the full run
    if seed is None and t0 is not None:
        seed = await _candidate_seed(ctx, iteration, spec, plan, t0)

    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration)
    out_path = it_dir / f"mesh_raw{tag}.glb"
    rel = storage.relpath(out_path)

    # reuse an already-generated raw mesh (hybrid->mesh fallback, edit re-run)
    if iteration == 1 and ctx.mesh_raw_reuse is not None and ctx.mesh_raw_reuse.exists():
        it_dir.mkdir(parents=True, exist_ok=True)
        if ctx.mesh_raw_reuse.resolve() != out_path.resolve():
            shutil.copy2(ctx.mesh_raw_reuse, out_path)
        return rel

    images, roles = _mesh_source_images(ctx, iteration, spec)
    if not images:
        ctx.mesh_info["gen_error"] = "no image input for mesh generation"
        return None
    await ctx.emit("meshgen", iteration, f"views: {', '.join(roles[: len(images)])}")

    base_seed = _base_seed(ctx)
    if seed is None:
        seed = base_seed + iteration - 1
    model, img_cap = await _resolve_mesh_adapter(ctx, len(images))
    if len(images) > img_cap:
        await ctx.emit(
            "meshgen",
            iteration,
            f"adapter '{model}' accepts max {img_cap} image(s) — using first {img_cap}",
        )
        images = images[:img_cap]
    subject = (
        ctx.mesh_info.get("_regen_hint")
        or (plan.mesh_subject if plan else "")
        or spec.name
        or ctx.prompt[:80]
    )

    # image preprocessing: isolated-object square crops generate cleaner meshes
    images = _prepped_gen_images(ctx, it_dir, images)

    # disk cache: identical inputs+params never hit the generator twice.
    # `subject` is planner-generated prose that varies run to run — keying
    # on the user prompt keeps restarts hitting the cache.
    key, hit = await _mesh_cache_lookup(
        ctx,
        iteration,
        images,
        roles[: len(images)],
        model,
        seed,
        steps,
        ctx.prompt[:80],
        out_path,
        rel,
    )
    if hit:
        return rel

    req = MeshGenRequest(
        images=[a.path for a in images],
        image_roles=roles[: len(images)],
        subject=subject,
        model=model,
        seed=seed,
        steps=steps if steps is not None else ctx.settings.mesh_steps,
        guidance=ctx.settings.mesh_guidance,
        resolution=ctx.settings.mesh_resolution,
        out_path=out_path,
        job_id=ctx.job_id,
    )

    async def on_progress(msg: str, p: float) -> None:
        await ctx.emit("meshgen", iteration, f"{msg} ({int(p * 100)}%)")

    try:
        result = await ctx.mesh.generate(req, on_progress, should_abort=ctx.check_cancel)
    except MeshGenError as e:
        await ctx.emit("meshgen", iteration, f"generation failed: {e}")
        log.warning("mesh.generate_failed", iteration=iteration, error=str(e))
        ctx.mesh_info["gen_error"] = str(e)
        return None
    except Exception as e:
        log.warning("mesh.generate_crashed", iteration=iteration, error=str(e))
        ctx.mesh_info["gen_error"] = str(e)
        return None

    if key:
        try:
            from cadx.agent import mesh_cache

            mesh_cache.store(key, result.mesh_path)
        except Exception as e:
            log.warning("mesh.cache_store_failed", error=str(e)[:160])

    gen = result.model_dump(mode="json")
    gen["mesh_rel"] = storage.relpath(result.mesh_path)
    gen.pop("mesh_path", None)
    gen["images"] = [a.asset_id for a in images]
    ctx.mesh_info["generation"] = gen
    return rel


async def _orient_mesh(ctx: AgentCtx, iteration: int, mesh_rel: str) -> dict[str, str]:
    """ORIENT step (once per job). Returns {up, front, mirror_x}.

    Deterministic first: when a plan-view measurement exists, the orientation
    whose XY footprint best matches the measured outline (IoU >= 0.7) wins
    directly — the LLM step only runs when no measured outline is available
    or no candidate reaches the threshold."""
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

    outline = _measured_plan_outline(ctx)
    if outline:
        try:
            import trimesh

            from cadx.cad.mesh_ops import orient_by_plan_view

            mesh_obj = trimesh.load(storage.resolve(mesh_rel), force="mesh")
            hit = (
                orient_by_plan_view(mesh_obj, outline)
                if isinstance(mesh_obj, trimesh.Trimesh)
                else None
            )
        except Exception as e:
            log.warning("orient.plan_view_failed", error=str(e)[:200])
            hit = None
        if hit is not None:
            up, front, mirror_x, iou = hit
            ctx.mesh_orientation = {"up": up, "front": front, "mirror_x": mirror_x}
            ctx.mesh_info["orientation_source"] = "plan_view_iou"
            await ctx.emit("orient", iteration, f"orientation from top view (IoU {iou:.2f})")
            return ctx.mesh_orientation

    choice = await mesh_orient_step.run(ctx, iteration, mesh_rel)
    ctx.mesh_orientation = choice
    ctx.mesh_info["orientation_source"] = ctx.mesh_info.get("orientation_source") or "llm"
    return ctx.mesh_orientation


def _measured_plan_outline(ctx: AgentCtx) -> list[list[float]] | None:
    """Best measured plan-view outline (confidence >= 0.5) for orientation,
    footprint scoring and clipping."""
    best: tuple[float, list[list[float]]] | None = None
    for m in (ctx.measurements or {}).values():
        conf = float(m.get("confidence") or 0)
        outline = m.get("outline_mm") or []
        if conf >= 0.5 and len(outline) >= 3 and (best is None or conf > best[0]):
            best = (conf, outline)
    return best[1] if best else None


def _outline_iou_mm(mesh_outline: list[list[float]], image_outline: list[list[float]]) -> float:
    """IoU of a processed-mesh footprint and the measured image outline — both
    in centred Frame mm, so directly comparable."""
    if len(mesh_outline) < 3 or len(image_outline) < 3:
        return 0.0
    try:
        from shapely.geometry import Polygon

        a = Polygon(mesh_outline).buffer(0)
        b = Polygon(image_outline).buffer(0)
        union = a.union(b).area
        return float(a.intersection(b).area / union) if union > 0 else 0.0
    except Exception:
        return 0.0


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
    finish_early: Any,
    *,
    max_iterations: int,
    max_repairs: int,
    pass_score: int,
    tol_pct: float,
    tol_mm: float,
    switch_check: Any = None,
    start_iteration: int = 1,
) -> RunOutcome:
    s = ctx.runtime_settings
    patch_mode = s.agent_patch_mode
    best: Candidate | None = None
    best_issues: list[dict[str, Any]] = []  # issue list belonging to `best`
    # score plateau tracker — a later candidate can outrank `best` on fewer
    # dim issues/majors while the critique score stays flat; that must not
    # reset the stall window forever
    best_score = 0
    best_score_it = start_iteration - 1
    issues: list[dict[str, Any]] = []
    iterations_done = 0
    improve_failures = 0

    def issues_text(items: list[dict[str, Any]]) -> str:
        from cadx.agent.context import issues_block

        # issues_block prefixes its own header — strip it for the codegen path
        return "\n".join(issues_block(items).splitlines()[1:])

    for it in range(start_iteration, start_iteration + max_iterations):
        iterations_done = it
        if await cancelled():
            return outcome
        if best is not None and await finish_early():
            await ctx.emit("finalize", it, "finish requested — keeping best candidate")
            break
        if ctx.tokens_used() > ctx.budget_tokens or time.monotonic() - t0 > s.agent_max_wall_s:
            await ctx.emit("budget", it, "budget exceeded — finalising best candidate")
            break

        # ---------- obtain code: fresh codegen vs targeted improve ----------
        base_code = best.code_response.code if best and best.code_response else ""
        use_improve = (
            bool(base_code) and it > start_iteration and issues != [] and improve_failures < 2
        )
        code_resp: CodeResponse | None = None
        if use_improve:
            await ctx.emit("improve", it, f"fixing {len(issues)} issue(s) on the best candidate")
            features = (best.exec_result.get("features") or None) if best else None
            patch_errors: list[str] | None = None
            for attempt in range(1, s.agent_patch_retries + 1):
                try:
                    presp = await improve_step.run(
                        ctx,
                        spec,
                        plan,
                        base_code=base_code,
                        issues=issues,
                        features=features,
                        patch_errors=patch_errors,
                        iteration=it,
                        attempt=attempt,
                        verify_fn=_session_verify_fn(ctx, it),
                    )
                except Exception as e:
                    log.warning("agent.improve_failed", iteration=it, error=str(e))
                    break
                new_code, patch_errors = _resolve_patch(base_code, presp, patch_mode)
                if new_code is not None:
                    code_resp = CodeResponse(
                        reasoning=presp.reasoning,
                        params=presp.params,
                        expected_bbox=presp.expected_bbox,
                        code=new_code,
                    )
                    break
            if code_resp is None:
                improve_failures += 1
                await ctx.emit("improve", it, "patch failed — falling back to fresh codegen next")
                continue
        else:
            await ctx.emit("codegen", it, f"writing CAD code (iteration {it}/{max_iterations})")
            try:
                code_resp = await codegen_step.run(
                    ctx,
                    spec,
                    plan,
                    iteration=it,
                    feedback=issues_text(issues) if issues else None,
                )
            except Exception as e:
                issues = [
                    {
                        "source": "codegen",
                        "severity": "major",
                        "what": f"code generation failed: {e}",
                    }
                ]
                log.warning("agent.codegen_failed", iteration=it, error=str(e))
                continue

        code = code_resp.code

        # ---------- execute with patch-based repairs ----------
        # The static contract check runs inside the repair loop: violations are
        # treated like exec failures (they'd fail in the executor anyway) and go
        # through the same patch-repair path without burning an executor call.
        exec_result: dict[str, Any] = {}
        chk = check_code(code)
        prev_err_sig = ""
        for repair_n in range(max_repairs + 1):
            if await cancelled():
                return outcome
            if best is not None and await finish_early():
                break
            chk = check_code(code)
            part_errors = check_standard_part_calls(code) if chk.ok else []
            if chk.ok and not part_errors:
                await ctx.emit(
                    "execute",
                    it,
                    "executing in sandbox"
                    if repair_n == 0
                    else f"repair attempt {repair_n}/{max_repairs}",
                )
                exec_result = await ctx.executor.execute(
                    _exec_payload(ctx, code, it, full_exports=False)
                )
            else:
                exec_result = {
                    "ok": False,
                    "error": "contract check failed: " + "; ".join((chk.errors + part_errors)[:8]),
                    "traceback": None,
                    "stdout": "",
                }
            if exec_result.get("ok"):
                break
            if repair_n < max_repairs:
                # identical failure twice in a row — patching isn't converging,
                # a fresh codegen pass has better odds than more repairs
                err_sig = (exec_result.get("error") or "")[:200]
                if err_sig and err_sig == prev_err_sig:
                    await ctx.emit("repair", it, "same error twice — repair not converging")
                    break
                prev_err_sig = err_sig
                await ctx.emit("repair", it, "fixing execution error")
                try:
                    presp = await codegen_step.repair(
                        ctx,
                        code,
                        spec,
                        plan,
                        exec_result.get("error"),
                        exec_result.get("traceback"),
                        exec_result.get("stdout"),
                        iteration=it,
                        repair_n=repair_n + 1,
                        verify_fn=_session_verify_fn(ctx, it),
                    )
                except Exception as e:
                    log.warning("agent.repair_failed", iteration=it, error=str(e))
                    break
                new_code, perrs = _resolve_patch(code, presp, patch_mode)
                if new_code is None:
                    await ctx.emit("repair", it, f"patch did not apply ({len(perrs)} hunk errors)")
                    log.warning("agent.repair_patch_failed", iteration=it, errors=perrs[:3])
                    break
                code = new_code

        if not exec_result.get("ok"):
            issues = [
                *best_issues,
                {
                    "source": "exec",
                    "severity": "major",
                    "what": (
                        "the last patch produced code that fails to execute — "
                        "avoid that change and write simpler geometry"
                    ),
                    "fix_hint": (exec_result.get("error") or "")[:400],
                },
            ]
            continue

        # repairs may have patched the code — keep the response in sync so the
        # persisted version holds the EXECUTED code, not the broken original
        code_resp = code_resp.model_copy(update={"code": code})

        pre_issues = _params_vs_spec(spec, chk.params)
        if (
            code_resp.expected_bbox
            and len(code_resp.expected_bbox) >= 3
            and spec.length_x.source == spec.width_y.source == spec.height_z.source == "stated"
        ):
            eb = code_resp.expected_bbox[:3]
            for dim, idx, name in (
                (spec.length_x, 0, "X"),
                (spec.width_y, 1, "Y"),
                (spec.height_z, 2, "Z"),
            ):
                tol = max(dim.value * tol_pct / 100.0, tol_mm)
                if abs(float(eb[idx]) - dim.value) > tol:
                    pre_issues.append(
                        {
                            "source": "contract",
                            "severity": "minor",
                            "what": (
                                f"declared expected_bbox {name}={eb[idx]:g}mm already "
                                f"violates spec {name}={dim.value:g}mm"
                            ),
                            "fix_hint": f"make the {name} dimension {dim.value:g}mm",
                        }
                    )

        # ---------- numeric checks + deterministic param autofix ----------
        analysis = exec_result.get("analysis") or {}
        dim_issues = compare_dims(spec, analysis, tol_pct, tol_mm)
        if dim_issues and s.agent_param_autofix:
            fixes = propose_fixes(
                spec, exec_result.get("params_used") or {}, analysis, tol_pct, tol_mm
            )
            if fixes:
                await ctx.emit(
                    "autofix",
                    it,
                    "trying param override: " + ", ".join(f"{k}={v:g}" for k, v in fixes.items()),
                )
                payload = _exec_payload(ctx, code, it, full_exports=False)
                payload["params_override"] = fixes
                auto_res = await ctx.executor.execute(payload)
                if auto_res.get("ok"):
                    new_dims = compare_dims(spec, auto_res.get("analysis") or {}, tol_pct, tol_mm)
                    if not new_dims:
                        await ctx.emit("autofix", it, "dimension fix applied without LLM")
                        exec_result = auto_res
                        analysis = exec_result.get("analysis") or {}
                        dim_issues = []
                    else:
                        await ctx.emit("autofix", it, "param override did not resolve dims")

        # ---------- feature verification (feedback only) ----------
        feature_findings: list[dict[str, Any]] = []
        features = exec_result.get("features") or {}
        if s.agent_feature_feedback and features and spec.features:
            try:
                from cadx.cad.features import MeasuredFeatures

                feature_findings = [
                    f.model_dump()
                    for f in compare_features(spec.features, MeasuredFeatures(**features))
                ]
            except Exception as e:
                log.warning("features.compare_failed", error=str(e)[:200])
        if s.agent_feature_feedback and features and ctx.anchors:
            try:
                from cadx.cad.features import (
                    MeasuredFeatures,
                    compare_anchor_positions,
                )

                feature_findings += [
                    f.model_dump()
                    for f in compare_anchor_positions(ctx.anchors, MeasuredFeatures(**features))
                ]
            except Exception as e:
                log.warning("features.anchor_compare_failed", error=str(e)[:200])
        f_iou = (analysis or {}).get("footprint_iou")
        if f_iou is not None and f_iou < 0.85:
            feature_findings.append(
                {
                    "feature": "footprint",
                    "expected": "matches the measured plan-view outline",
                    "actual": f"footprint IoU {f_iou:.2f}",
                    "severity": "major",
                    "fix_hint": (
                        "rebuild the body from the measured outline — "
                        "use load_image_profile() / profile_body()"
                    ),
                }
            )

        # ---------- FEM check (only when the spec declares loads) ----------
        fem_issues: list[dict[str, Any]] = []
        if spec.load_cases:
            it_rel = storage.relpath(storage.iteration_dir(ctx.project_id, ctx.version_id, it))
            await ctx.emit("fem", it, f"verifying {len(spec.load_cases)} load case(s)")
            try:
                fem_res = await ctx.executor.fem_analyze(
                    {
                        "job_id": ctx.job_id,
                        "project_id": str(ctx.project_id),
                        "version_id": str(ctx.version_id),
                        "iteration": it,
                        "mesh_rel": f"{it_rel}/renders/mesh.stl",
                        "load_cases": [c.model_dump() for c in spec.load_cases],
                        "material": ctx.settings.fem_material or spec.material,
                        "voxel_mm": ctx.settings.fem_voxel_mm,
                        "runtime_settings": ctx.runtime_settings.executor_values(),
                    }
                )
            except Exception as e:
                fem_res = {"ok": False, "error": str(e)}
            if fem_res.get("ok") and fem_res.get("fem"):
                exec_result["fem"] = fem_res["fem"]
                sf_min = spec.optimize.safety_factor_min if spec.optimize else 1.5
                rep = FemReport(**fem_res["fem"])
                worst = rep.worst_safety_factor
                for line in fem_feedback_lines(rep, sf_min):
                    fem_issues.append(
                        {
                            "source": "fem",
                            "severity": "major",
                            "what": line,
                            "fix_hint": "increase section sizes / add ribs on the load path",
                        }
                    )
                await ctx.emit(
                    "fem",
                    it,
                    f"done: worst safety factor {worst}"
                    if worst is not None
                    else "done: no solvable case",
                )
            else:
                await ctx.emit("fem", it, f"skipped: {fem_res.get('error') or 'no result'}")

        # ---------- vision critique ----------
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
                feature_findings=feature_findings,
                features=features,
                code=code,
            )
        except Exception as e:
            log.warning("agent.critique_failed", iteration=it, error=str(e))
            critique = Critique(score=0, passed=False, confidence=0.0)

        cand = Candidate(it, code_resp, exec_result, critique, dim_issues)
        cand.feature_findings = feature_findings
        cur_issues = (
            pre_issues
            + fem_issues
            + build_issues(critique, dim_issues, feature_findings, exec_result.get("dfm"))
        )
        if best is None or cand.rank > best.rank:
            best = cand
            # improve always patches best.code — its issue list must describe
            # that same code, not a regressed later candidate
            best_issues = cur_issues
        cur_score = cand.critique.score if cand.critique else 0
        if cur_score > best_score:
            best_score = cur_score
            best_score_it = it

        major = [d for d in critique.discrepancies if d.severity == "major"]
        await ctx.emit(
            "critique",
            it,
            f"score {critique.score}/100 — {len(major)} major, "
            f"{len(critique.discrepancies) - len(major)} minor issues",
        )

        if (
            critique.passed
            and critique.score >= pass_score
            and not dim_issues
            and not major
            and not fem_issues
        ):
            await ctx.emit("done", it, "all checks passed")
            break
        issues = best_issues

        # strategy plateau hook — a second approach may do better than more
        # iterations on a candidate stuck below the switch score
        if switch_check is not None and switch_check(
            it - start_iteration + 1,
            best.critique.score if best is not None and best.critique else None,
        ):
            await ctx.emit("plan", it, "score plateaued — switching approach")
            outcome.switch_requested = True
            break

        # stall early-exit: no new best for `agent_stall_patience` iterations —
        # more improve loops are unlikely to converge, finalise what we have.
        # Score plateau counts too: a reshuffling `best` on dim-issues while
        # the critique score stays flat is still a stall.
        if best is not None and s.agent_stall_patience and (
            it - best.iteration >= s.agent_stall_patience
            or it - best_score_it >= s.agent_stall_patience
        ):
            await ctx.emit(
                "finalize",
                it,
                f"no improvement for {s.agent_stall_patience} iterations — keeping best",
            )
            break

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
    artifacts.update(_copy_fem_artifacts(ctx, best.iteration))

    outcome.status = (
        "succeeded"
        if (best.critique and best.critique.passed and not best.dim_issues)
        else "needs_review"
    )
    outcome.fem = best.exec_result.get("fem")
    outcome.code = best.code_response.code if best.code_response else None
    outcome.params = dict(best.exec_result.get("params_used") or {})
    outcome.params_schema = (
        [p.model_dump() for p in best.code_response.params] if best.code_response else None
    )
    outcome.analysis = best.exec_result.get("analysis")
    outcome.critique = best.critique.model_dump() if best.critique else None
    if outcome.critique is not None:
        outcome.critique["feature_findings"] = best.feature_findings
        outcome.critique["issues"] = issues
    outcome.dfm = best.exec_result.get("dfm")
    outcome.bom = best.exec_result.get("bom")
    outcome.score = best.critique.score if best.critique else None
    outcome.best_iteration = best.iteration
    outcome.issues = issues
    if ctx.resolved_mode == "hybrid" and ctx.mesh_info:
        prep_it = int(ctx.mesh_info.get("_prep_it", 1))
        artifacts.update(_copy_mesh_artifacts(ctx.project_id, ctx.version_id, prep_it))
        outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
        _write_mesh_json(ctx.project_id, ctx.version_id, ctx.mesh_info)
    outcome.artifacts = artifacts
    outcome.usage = dict(getattr(ctx.llm, "total_usage", {}))
    return outcome


async def _run_mesh(  # noqa: C901
    ctx: AgentCtx,
    outcome: RunOutcome,
    spec: PartSpec,
    plan: Any,
    t0: float,
    cancelled: Any,
    finish_early: Any,
    *,
    max_iterations: int,
    pass_score: int,
    tol_pct: float,
    tol_mm: float,
    start_iteration: int = 1,
) -> RunOutcome:
    """Mesh-mode loop: generate -> baseline process -> refine session ->
    analyse -> critique -> refine-repair rounds -> best candidate.

    The refine session gives the agent mesh-transformation tools instead of
    blind re-generation; it can also decide the raw mesh is unfixable and ask
    for a new seed (`regenerate`). Sessions unavailable (tools off, vision
    model stripped) → previous seed-rotation behaviour.
    """
    s = ctx.runtime_settings
    best: Candidate | None = None
    iterations_done = 0
    refine_rounds = max(0, getattr(s, "agent_mesh_refine_rounds", 1))
    hard_fails = 0
    hard_fail_score = getattr(s, "agent_mesh_hard_fail_score", 30)
    # a multi-approach strategy keeps a fallback — burn fewer seeds here
    hard_fail_limit = (
        2
        if ctx.plan and len(ctx.plan.strategy) > 1
        else getattr(s, "agent_mesh_hard_fail_limit", 3)
    )

    def _sheet(exec_result: dict[str, Any]) -> Path | None:
        rel = exec_result.get("contact_sheet")
        return storage.resolve(rel) if rel else None

    async def _evaluate(it: int, exec_result: dict[str, Any]) -> tuple[Critique, list[str]]:
        """Dim check + vision critique + emit. Returns (critique, dim_issues)."""
        analysis = exec_result.get("analysis") or {}
        dim_issues = compare_dims(spec, analysis, tol_pct, tol_mm)
        dim_issues += list((exec_result.get("mesh") or {}).get("dim_issues") or [])
        try:
            c = await critique_step.run(
                ctx,
                spec,
                _sheet(exec_result),
                analysis,
                dim_issues,
                exec_result.get("dfm"),
                iteration=it,
            )
        except Exception as e:
            log.warning("agent.critique_failed", iteration=it, error=str(e))
            c = Critique(score=0, passed=False, confidence=0.0)
        major = [d for d in c.discrepancies if d.severity == "major"]
        await ctx.emit(
            "critique",
            it,
            f"score {c.score}/100 — {len(major)} major, "
            f"{len(c.discrepancies) - len(major)} minor issues",
        )
        return c, dim_issues

    async def _refine(
        it: int,
        mesh_rel: str,
        post: MeshPostParams,
        exec_result: dict[str, Any],
        *,
        issues: list[dict[str, Any]] | None = None,
    ) -> mesh_refine_step.RefineResult | None:
        await ctx.emit(
            "meshrefine", it, "mesh transform session" if not issues else "mesh repair session"
        )
        return await mesh_refine_step.run(
            ctx,
            spec,
            mesh_rel,
            post,
            iteration=it,
            initial_ops=list(post.ops) if issues else None,
            issues=issues,
            contact_sheet=_sheet(exec_result),
            analysis=exec_result.get("analysis"),
            dfm=exec_result.get("dfm"),
        )

    async def _apply_ops(
        it: int, mesh_rel: str, post: MeshPostParams, rr: mesh_refine_step.RefineResult
    ) -> dict[str, Any]:
        post.ops = rr.ops
        await ctx.emit("meshprocess", it, f"re-processing mesh with {len(rr.ops)} op(s)")
        res = await _process_mesh(ctx, it, mesh_rel, spec, post)
        ctx.mesh_info["post"] = post.model_dump()
        ctx.mesh_info["ops"] = [o.model_dump() for o in rr.ops]
        return res

    for it in range(start_iteration, start_iteration + max_iterations):
        iterations_done = it
        if await cancelled():
            return outcome
        if best is not None and await finish_early():
            await ctx.emit("finalize", it, "finish requested — keeping best candidate")
            break
        if ctx.tokens_used() > ctx.budget_tokens or time.monotonic() - t0 > s.agent_max_wall_s:
            await ctx.emit("budget", it, "budget exceeded — finalising best candidate")
            break

        # reuse a precomputed result (hybrid->mesh fallback)
        exec_result: dict[str, Any] | None = None
        mesh_rel: str | None = None
        post = MeshPostParams()
        if it == start_iteration and ctx.mesh_info.pop("_pre_result", None):
            exec_result = ctx.mesh_info["_pre_result_exec"]
            mesh_rel = ctx.mesh_info["_pre_result_mesh_rel"]
            post = MeshPostParams(**((exec_result.get("mesh") or {}).get("post") or {}))
        else:
            # MESH_GENERATE
            await ctx.emit("meshgen", it, f"generating mesh (iteration {it}/{max_iterations})")
            mesh_rel = await _generate_mesh(ctx, it, spec, plan, t0=t0)
            if mesh_rel is None:
                continue

            # ORIENT (once per job)
            orientation = await _orient_mesh(ctx, it, mesh_rel)
            post = MeshPostParams(
                up=orientation.get("up"),  # type: ignore[arg-type]
                front=orientation.get("front"),  # type: ignore[arg-type]
                mirror_x=bool(orientation.get("mirror_x")),
            )
            post = merge_post_params(post, {})

            # PROCESS_MESH baseline
            await ctx.emit("meshprocess", it, "cleaning, orienting and scaling mesh")
            exec_result = await _process_mesh(ctx, it, mesh_rel, spec, post)
            ctx.mesh_info["post"] = post.model_dump()
            ctx.mesh_info["orientation"] = {"up": post.up, "front": post.front}

            # MESH_REFINE shape session — transform before judging
            if exec_result.get("ok"):
                rr = await _refine(it, mesh_rel, post, exec_result)
                if rr is not None and rr.regenerate:
                    if rr.subject_hint:
                        ctx.mesh_info["_regen_hint"] = rr.subject_hint
                    await ctx.emit("meshrefine", it, f"regenerating — {rr.reasoning[:100]}")
                    # a rejected mesh that badly misses stated dims counts as a
                    # hard fail too — otherwise the loop re-rolls seeds on an
                    # input that cannot produce the spec'd part
                    if compare_dims(spec, exec_result.get("analysis") or {}, tol_pct, tol_mm):
                        hard_fails += 1
                        if hard_fails >= hard_fail_limit:
                            outcome.fallback_to_cad = stated_dim_count(spec) >= 2
                            await ctx.emit(
                                "meshgen",
                                it,
                                f"{hard_fails} rejected generations missing stated dims — "
                                + ("CAD fallback" if outcome.fallback_to_cad else "stopping"),
                            )
                            break
                    else:
                        hard_fails = 0
                    continue
                if rr is not None and rr.changed:
                    exec_result = await _apply_ops(it, mesh_rel, post, rr)

        if not exec_result or not exec_result.get("ok"):
            log.warning("mesh.process_failed", iteration=it, error=(exec_result or {}).get("error"))
            continue

        # pre-gate: deterministic score before spending a critique call —
        # aspect/footprint/holes that cannot be this part count as hard fails
        from cadx.agent.mesh_score import score_mesh_candidate

        sc = score_mesh_candidate(exec_result, spec, _measured_plan_outline(ctx))
        if sc < 0.35:
            hard_fails += 1
            await ctx.emit(
                "meshprocess",
                it,
                f"mesh rejected by pre-gate (score {sc:.2f}) — new seed",
            )
            if hard_fails >= hard_fail_limit:
                outcome.fallback_to_cad = stated_dim_count(spec) >= 2
                await ctx.emit(
                    "meshgen",
                    it,
                    f"{hard_fails} pre-gate rejections — "
                    + ("CAD fallback" if outcome.fallback_to_cad else "stopping"),
                )
                break
            continue

        # ANALYZE + CRITIQUE
        await ctx.emit("critique", it, "comparing renders with your images")
        critique, dim_issues = await _evaluate(it, exec_result)

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

        def _passed(c: Critique, di: list[str]) -> bool:
            major = [d for d in c.discrepancies if d.severity == "major"]
            return c.passed and c.score >= pass_score and not di and not major

        if _passed(critique, dim_issues):
            await ctx.emit("done", it, "all checks passed")
            break

        # MESH_REFINE repair session(s) — fix this mesh instead of a new seed
        for _round in range(refine_rounds):
            if await cancelled():
                return outcome
            if await finish_early():
                break
            issues = build_issues(critique, dim_issues, [], exec_result.get("dfm"))
            rr = await _refine(it, mesh_rel, post, exec_result, issues=issues)
            if rr is None:
                break  # sessions unavailable
            if rr.regenerate:
                if rr.subject_hint:
                    ctx.mesh_info["_regen_hint"] = rr.subject_hint
                await ctx.emit("meshrefine", it, f"regenerating — {rr.reasoning[:100]}")
                break
            if not rr.changed:
                break
            new_exec = await _apply_ops(it, mesh_rel, post, rr)
            if not new_exec.get("ok"):
                log.warning("mesh.refine_reprocess_failed", error=new_exec.get("error"))
                break
            exec_result = new_exec
            critique2, dim_issues2 = await _evaluate(it, exec_result)
            cand = Candidate(
                it,
                None,
                exec_result,
                critique2,
                dim_issues2,
                mesh_post=(exec_result.get("mesh") or {}).get("post"),
            )
            if best is None or cand.rank > best.rank:
                best = cand
            critique, dim_issues = critique2, dim_issues2
            if _passed(critique, dim_issues):
                break

        if _passed(critique, dim_issues):
            await ctx.emit("done", it, "all checks passed after refine")
            break

        # hard-fail accounting: a fundamentally wrong object (dim issues on a
        # stated spec, or a very low score with major discrepancies) won't be
        # fixed by another seed — burn no more minutes on it
        major = [d for d in critique.discrepancies if d.severity == "major"]
        if dim_issues or (critique.score <= hard_fail_score and major):
            hard_fails += 1
            if hard_fails >= hard_fail_limit:
                if stated_dim_count(spec) >= 2:
                    await ctx.emit(
                        "meshgen",
                        it,
                        f"{hard_fails} hard fails on a dimensioned part — CAD fallback",
                    )
                    outcome.fallback_to_cad = True
                else:
                    await ctx.emit(
                        "meshgen",
                        it,
                        f"{hard_fails} consecutive bad generations — stopping early",
                    )
                break
        else:
            hard_fails = 0
        # stall early-exit: best mesh unchanged for `agent_stall_patience` seeds
        if (
            best is not None
            and s.agent_stall_patience
            and it - best.iteration >= s.agent_stall_patience
        ):
            await ctx.emit(
                "finalize",
                it,
                f"no improvement for {s.agent_stall_patience} iterations — keeping best",
            )
            break
        # otherwise next iteration: new seed (+ optional regen subject hint)

    outcome.iterations = iterations_done

    # ---------- FINALIZE ----------
    if best is None:
        outcome.error = ctx.mesh_info.get("gen_error") or "all mesh iterations failed"
        # a dimensioned part that mesh can't produce is still buildable via CAD
        if not outcome.fallback_to_cad and stated_dim_count(spec) >= 2:
            outcome.fallback_to_cad = True
            outcome.mode_note += " | mesh exhausted its budget — CAD fallback"
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
    outcome.bom = best.exec_result.get("bom")
    outcome.score = best.critique.score if best.critique else None
    outcome.best_iteration = best.iteration
    outcome.artifacts = artifacts
    outcome.usage = dict(getattr(ctx.llm, "total_usage", {}))
    outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
    outcome.mesh["best_iteration"] = best.iteration
    _write_mesh_json(ctx.project_id, ctx.version_id, outcome.mesh)
    return outcome


def _fit_ops_to_stated(spec: PartSpec, bbox: list[float] | None) -> tuple[list[MeshOp], str]:
    """Deterministic rescue for a dimensionally-off organic base mesh.

    Generative meshes often capture the right shape with wrong proportions
    (too thick, too wide). Instead of discarding an expensive generation we
    fit the envelope to stated dims with bounded per-axis scale ops —
    recorded in `post.ops` so params jobs replay them. Returns ([], "") when
    the mismatch is too extreme to be a proportion error (wrong object).
    """
    if not bbox or len(bbox) != 3:
        return [], ""
    actual = [float(v) for v in bbox]
    factors: dict[int, float] = {}
    for idx, dim in ((0, spec.length_x), (1, spec.width_y), (2, spec.height_z)):
        if dim is not None and dim.source == "stated" and dim.value and actual[idx] > 0:
            factors[idx] = float(dim.value) / actual[idx]
    if not factors:
        return [], ""

    notes: list[str] = []
    ops: list[MeshOp] = []
    scale_args: dict[str, float] = {}
    for idx, axis in ((0, "x"), (1, "y")):
        f = factors.get(idx)
        if f is not None and abs(1 - f) > 0.02:
            if not 0.4 <= f <= 2.5:
                return [], ""
            scale_args[axis] = round(f, 4)

    fz = factors.get(2)
    if fz is not None and abs(1 - fz) > 0.02:
        # over-thick organic bases are scaled, not sliced: the top stated_h
        # slab of a fat mesh is a curved crust (volume collapses), while a
        # bounded z-scale preserves the shape and just flattens relief depth.
        # Image-to-3D models routinely inflate thickness ~3-6x on flat parts
        # (observed: 52mm blob for an 8mm handle scale) — allow down to 0.12;
        # the degenerate-fit volume guard in _fit_mesh_to_stated catches
        # meshes that collapse into a crust.
        if not 0.12 <= fz <= 2.0:
            return [], ""
        scale_args["z"] = round(fz, 4)

    if scale_args:
        ops.append(MeshOp(op="scale", args=scale_args))
        notes.append("scaled " + ", ".join(f"{a}x{f:g}" for a, f in scale_args.items()))
    if not ops:
        return [], ""
    return ops, "; ".join(notes)


_FLAT_BOTTOM_CUE = re.compile(
    r"(flat|planar|flush|ebene?|flach|plan\b)[^.;]{0,40}"
    r"(back|bottom|-z|underside|unter|base|r[üu]e?ck|rear)"
    r"|(back|bottom|-z|underside|unter|base|r[üu]e?ck|rear)[^.;]{0,40}"
    r"(flat|planar|flush|ebene?|flach|plan\b)",
    re.IGNORECASE,
)


def _spec_wants_flat_bottom(spec: PartSpec) -> bool:
    """Text cue: spec demands a planar -z face (mounting/flat back)."""
    texts = [spec.name or "", spec.summary or ""]
    texts += [f"{f.name} {f.description}" for f in spec.features]
    return any(_FLAT_BOTTOM_CUE.search(t) for t in texts)


def _bottom_bow_mm(stl_path: Path) -> float:
    """Depth of the bottom-surface bow: the mesh's lowest surface per XY cell
    forms a height map — its P95 minus the absolute min is the sagging depth
    to cut for a flat back. 0 for an already-planar bottom."""
    try:
        import numpy as np
        import trimesh

        m = trimesh.load(str(stl_path), force="mesh", process=False)
        if not isinstance(m, trimesh.Trimesh):
            return 0.0
        mn, mx = m.bounds
        step = max(0.8, float(mx[0] - mn[0]) / 60.0)
        lowest: dict[tuple[int, int], float] = {}
        for x, y, z in m.vertices:
            key = (int((x - mn[0]) // step), int((y - mn[1]) // step))
            if z < lowest.get(key, float("inf")):
                lowest[key] = float(z)
        if len(lowest) < 12:
            return 0.0
        vals = np.array(sorted(lowest.values()))
        bow = float(np.percentile(vals, 80) - mn[2])
        # cut no deeper than 45% of the z-range — beyond that the "flat back"
        # was probably never meant to be planar (genuinely curved underside)
        return min(bow, 0.45 * float(mx[2] - mn[2]))
    except Exception:
        return 0.0


async def _apply_flat_back(
    ctx: AgentCtx,
    it: int,
    mesh_rel: str,
    spec: PartSpec,
    post: MeshPostParams,
    exec_result: dict[str, Any],
) -> tuple[dict[str, Any], MeshPostParams]:
    """Deterministic flat-back pass: when the spec demands a planar -z face
    and the mesh bows its bottom, flatten (voxel fill-down) + floor-pinned
    smoothing BEFORE the dim gate."""
    stl_rel = (exec_result.get("mesh") or {}).get("stl_relpath")
    if not (stl_rel and _spec_wants_flat_bottom(spec)):
        return exec_result, post
    bow = _bottom_bow_mm(storage.resolve(stl_rel))
    if bow <= 0.35:
        return exec_result, post
    # a sag far beyond the stated thickness means the mesh rests on the wrong
    # axis — voxel-flattening a tall domain is a memory bomb; let the dim
    # gate / pre-gate reject the mesh instead
    hz = spec.height_z.value
    if hz > 0 and bow > 4 * hz:
        await ctx.emit(
            "meshprocess",
            it,
            f"skipping flat-back: {bow:.0f} mm sag exceeds 4x stated thickness ({hz:.0f} mm)",
        )
        return exec_result, post
    smooth_iters = ctx.runtime_settings.mesh_smooth_after_flatten_iters
    ops = _flat_back_ops(post, bow, smooth_iters)
    flat_post = merge_post_params(post, {"ops": ops})
    note = f"flattening bowed -z back ({bow:.1f} mm sag)"
    if smooth_iters > 0:
        note += f" + smoothing ({smooth_iters} it)"
    await ctx.emit("meshprocess", it, note)
    flat_result = await _process_mesh(ctx, it, mesh_rel, spec, flat_post, make_solid=True)
    if flat_result.get("ok"):
        rough = (flat_result.get("analysis") or {}).get("roughness_deg")
        rough_before = (exec_result.get("analysis") or {}).get("roughness_deg")
        if rough is not None and rough_before is not None:
            await ctx.emit(
                "meshprocess", it, f"surface roughness {rough_before:.0f}° → {rough:.0f}°"
            )
        return flat_result, flat_post
    return exec_result, post


def _flat_back_ops(post: MeshPostParams, bow_mm: float, smooth_iters: int) -> list[dict[str, Any]]:
    """Ops appended to flatten a bowed -z back, optionally followed by a
    floor-pinned smoothing pass (voxel rebuild leaves stair-step striations)."""
    ops = [
        *[o.model_dump() for o in post.ops],
        MeshOp(op="flatten_bottom", args={"height_mm": round(min(bow_mm, 49.0), 2)}).model_dump(),
    ]
    if smooth_iters > 0:
        ops.append(MeshOp(op="smooth_preserve", args={"iterations": smooth_iters}).model_dump())
    return ops


async def _fit_mesh_to_stated(
    ctx: AgentCtx,
    it: int,
    mesh_rel: str,
    spec: PartSpec,
    post: MeshPostParams,
    exec_result: dict[str, Any],
    tol_pct: float,
    tol_mm: float,
) -> tuple[dict[str, Any], MeshPostParams] | None:
    """Iteratively fit a processed mesh to stated dims via appended ops.

    Ops are computed on the CURRENT bbox, appended, then re-measured — a
    slice changes the XY envelope itself, so a second round corrects the
    residual (converges in ~2 rounds). Returns (exec_result, post) on success.
    """
    total_ops = [o.model_dump() for o in (post.ops or [])]
    cur_result, cur_post = exec_result, post
    for _ in range(3):
        fit_ops, fit_note = _fit_ops_to_stated(
            spec, (cur_result.get("analysis") or {}).get("bbox_size")
        )
        if not fit_ops:
            return None
        total_ops += [o.model_dump() for o in fit_ops]
        post_fit = merge_post_params(cur_post, {"ops": total_ops})
        await ctx.emit("meshprocess", it, f"fitting base mesh to stated dims ({fit_note})")
        fit_result = await _process_mesh(ctx, it, mesh_rel, spec, post_fit, make_solid=True)
        if not fit_result.get("ok"):
            return None
        cur_result, cur_post = fit_result, post_fit
        if not compare_dims(spec, fit_result.get("analysis") or {}, tol_pct, tol_mm):
            # degenerate-fit guard: ops that collapse the body to a sliver
            # (volume < ~8% of its own bbox) produce a crust, not a part
            ana = fit_result.get("analysis") or {}
            bb = ana.get("bbox_size") or []
            vol = ana.get("volume")
            if bb and vol and vol < 0.08 * (bb[0] * bb[1] * bb[2]):
                return None
            return cur_result, cur_post
    return None


async def _clip_mesh_to_measured(
    ctx: AgentCtx,
    it: int,
    mesh_rel: str,
    spec: PartSpec,
    post: MeshPostParams,
    exec_result: dict,
) -> tuple[dict, MeshPostParams]:
    """Clip the processed mesh to the measured plan-view outline prism when the
    footprint IoU is below 0.9; keep the result only if IoU does not regress."""
    outline = _measured_plan_outline(ctx)
    if not outline:
        return exec_result, post
    feats = (exec_result.get("mesh") or {}).get("features") or {}
    iou = _outline_iou_mm(feats.get("outline_mm") or [], outline)
    if iou >= 0.9:
        return exec_result, post
    post_c = merge_post_params(
        post,
        {
            "ops": [
                *post.ops,
                {"op": "clip_to_outline", "args": {"outline": outline, "margin_mm": 0.3}},
            ]
        },
    )
    res_c = await _process_mesh(ctx, it, mesh_rel, spec, post_c, make_solid=True)
    if not res_c.get("ok"):
        return exec_result, post
    feats2 = (res_c.get("mesh") or {}).get("features") or {}
    iou2 = _outline_iou_mm(feats2.get("outline_mm") or [], outline)
    if iou2 >= iou:
        await ctx.emit(
            "meshprocess",
            it,
            f"clipping footprint to measured outline (IoU {iou:.2f} → {iou2:.2f})",
        )
        return res_c, post_c
    await ctx.emit("meshprocess", it, f"footprint clip rejected (IoU {iou:.2f} → {iou2:.2f})")
    return exec_result, post


async def _reuse_parent_mesh_assets(ctx: AgentCtx, it: int) -> bool:
    """Copy the parent version's mesh intermediates into this version so the
    child is self-contained; wire ctx.hybrid_solid/mesh_info. Returns True."""
    if ctx.parent.mesh_solid is None or ctx.parent.mesh is None:
        return False
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
    return True


async def _prepare_hybrid_mesh(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: Any,
    cancelled: Any,
    t0: float,
    *,
    tol_pct: float,
    tol_mm: float,
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
    if (
        ctx.parent.mesh_solid is not None
        and ctx.parent.mesh
        and await _reuse_parent_mesh_assets(ctx, it)
    ):
        return True, "reused parent mesh solid"

    await ctx.emit("meshgen", it, "generating organic base mesh")
    mesh_rel = await _generate_mesh(ctx, it, spec, plan, t0=t0)
    if mesh_rel is None:
        return False, "mesh generation failed"
    ctx.mesh_raw_reuse = storage.resolve(mesh_rel)

    orientation = await _orient_mesh(ctx, it, mesh_rel)
    post = MeshPostParams(
        up=orientation.get("up"),  # type: ignore[arg-type]
        front=orientation.get("front"),  # type: ignore[arg-type]
        mirror_x=bool(orientation.get("mirror_x")),
    )

    await ctx.emit("meshprocess", it, "processing mesh and building solid")
    exec_result = await _process_mesh(ctx, it, mesh_rel, spec, post, make_solid=True)

    if not exec_result.get("ok"):
        return False, exec_result.get("error") or "mesh processing failed"

    # deterministic flat-back: when the spec demands a planar -z face and the
    # generative mesh bows its bottom, shave it flat BEFORE the dim gate — the
    # fit loop then re-scales z back to the stated thickness.
    exec_result, post = await _apply_flat_back(ctx, it, mesh_rel, spec, post, exec_result)

    # dimensional gate: a base mesh far outside stated dims is the wrong
    # object — CAD features can't repair a 40%-off organic body. Before
    # rejecting, try fitting the envelope deterministically — generative
    # meshes routinely get proportions wrong while capturing the shape.
    if stated_dim_count(spec) >= 2:
        dim_issues = compare_dims(spec, exec_result.get("analysis") or {}, tol_pct, tol_mm)
        dim_issues += list((exec_result.get("mesh") or {}).get("dim_issues") or [])
        if dim_issues:
            fitted = await _fit_mesh_to_stated(
                ctx, it, mesh_rel, spec, post, exec_result, tol_pct, tol_mm
            )
            if fitted is not None:
                exec_result, post = fitted
            else:
                ctx.mesh_info["_pre_result"] = False
                return False, f"base mesh violates stated dimensions ({dim_issues[0]})"

    # pre-gate: deterministic candidate score — a base mesh that cannot be
    # the part (wrong aspect/footprint) never reaches the CAD phase
    from cadx.agent.mesh_score import score_mesh_candidate

    sc = score_mesh_candidate(exec_result, spec, _measured_plan_outline(ctx))
    if sc < 0.35:
        return False, f"mesh rejected by pre-gate (score {sc:.2f})"

    # anchor reconciliation: measured top-view holes + detected mesh features
    # give authoritative hole positions for CAD/T-FEM; an X-mirrored mesh is
    # flipped once so image and mesh agree
    anchors = build_anchors(spec, ctx.measurements, (exec_result.get("mesh") or {}).get("features"))
    if anchors.mirror_x:
        post_m = merge_post_params(post, {"mirror_x": True})
        await ctx.emit("meshprocess", it, "mirroring mesh to match the top view (IoU)")
        res_m = await _process_mesh(ctx, it, mesh_rel, spec, post_m, make_solid=True)
        if res_m.get("ok"):
            exec_result, post = res_m, post_m
            anchors = build_anchors(
                spec, ctx.measurements, (exec_result.get("mesh") or {}).get("features")
            )
            anchors.mirror_x = True

    # footprint clip: when the XY silhouette still drifts from the measured
    # plan view, clip the mesh to the measured outline prism
    exec_result, post = await _clip_mesh_to_measured(ctx, it, mesh_rel, spec, post, exec_result)
    anchors = build_anchors(spec, ctx.measurements, (exec_result.get("mesh") or {}).get("features"))
    anchors.mirror_x = post.mirror_x

    if anchors.holes or anchors.image_outline:
        ctx.anchors = anchors.model_dump()
        await ctx.emit("anchors", it, anchors.note)

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
    s = ctx.runtime_settings
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


async def _run_approach(
    ctx: AgentCtx,
    outcome: RunOutcome,
    spec: PartSpec,
    plan: Any,
    t0: float,
    cancelled: Any,
    finish_early: Any,
    *,
    approach: str,
    max_repairs: int,
    switch_check: Any = None,
    start_iteration: int = 1,
    **kwargs: Any,
) -> RunOutcome:
    """Run one strategy approach (mesh | hybrid | cad) incl. its internal
    fallbacks. Extracted dispatch so `_run_strategy` can iterate a plan's
    ordered approach list."""
    if approach == "mesh":
        outcome = await _run_mesh(
            ctx,
            outcome,
            spec,
            plan,
            t0,
            cancelled,
            finish_early,
            start_iteration=start_iteration,
            **kwargs,
        )
        if outcome.fallback_to_cad:
            usage = dict(outcome.usage)
            note = outcome.mode_note + " | mesh could not meet stated dimensions — CAD fallback"
            outcome = RunOutcome(
                spec=spec.model_dump(),
                plan=outcome.plan,
                usage=usage,
            )
            outcome.mode_note = note
            outcome.resolved_mode = "cad"
            ctx.resolved_mode = "cad"
            await ctx.emit("plan", 0, note)
            outcome = await _run_cad(
                ctx,
                outcome,
                spec,
                plan,
                t0,
                cancelled,
                finish_early,
                max_repairs=max_repairs,
                **kwargs,
            )
    elif approach == "hybrid":
        ready, note = await _prepare_hybrid_mesh(
            ctx, spec, plan, cancelled, t0, tol_pct=kwargs["tol_pct"], tol_mm=kwargs["tol_mm"]
        )
        await ctx.emit("meshprocess", start_iteration, note)
        if ready:
            outcome = await _run_cad(
                ctx,
                outcome,
                spec,
                plan,
                t0,
                cancelled,
                finish_early,
                max_repairs=max_repairs,
                switch_check=switch_check,
                start_iteration=start_iteration,
                **kwargs,
            )
            if outcome.status != "failed" and ctx.mesh_info:
                outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
        elif ctx.mesh_info.get("_pre_result"):
            outcome.resolved_mode = "mesh"
            ctx.resolved_mode = "mesh"
            outcome.mode_note += " | hybrid conversion failed — mesh-only result"
            await ctx.emit("plan", 0, outcome.mode_note)
            outcome = await _run_mesh(
                ctx,
                outcome,
                spec,
                plan,
                t0,
                cancelled,
                finish_early,
                start_iteration=start_iteration,
                **kwargs,
            )
        else:
            outcome.resolved_mode = "cad"
            ctx.resolved_mode = "cad"
            outcome.mode_note += f" | hybrid failed ({note}) — CAD fallback"
            await ctx.emit("plan", 0, outcome.mode_note)
            outcome = await _run_cad(
                ctx,
                outcome,
                spec,
                plan,
                t0,
                cancelled,
                finish_early,
                max_repairs=max_repairs,
                switch_check=switch_check,
                start_iteration=start_iteration,
                **kwargs,
            )
    else:
        outcome = await _run_cad(
            ctx,
            outcome,
            spec,
            plan,
            t0,
            cancelled,
            finish_early,
            max_repairs=max_repairs,
            switch_check=switch_check,
            start_iteration=start_iteration,
            **kwargs,
        )
    return outcome


async def _run_strategy(
    ctx: AgentCtx,
    outcome: RunOutcome,
    spec: PartSpec,
    plan: Any,
    t0: float,
    cancelled: Any,
    finish_early: Any,
    *,
    mode: str,
    max_repairs: int,
    **kwargs: Any,
) -> RunOutcome:
    """Execute the plan's ordered approach list.

    Forced gen_modes run exactly `mode`. In auto the planner's `strategy`
    (1-2 entries) drives the loop: the next approach runs when the current
    one failed outright or plateaued below `agent_switch_score` after
    `agent_switch_after_iters` (switch_check hook in the CAD loop). The best
    outcome across approaches wins; iteration counters continue so artifacts
    are never overwritten."""
    s = ctx.runtime_settings
    forced = ctx.gen_mode not in (None, "auto", "topopt")
    approaches = [mode] if forced else (list(plan.strategy) if plan and plan.strategy else [mode])
    has_images = any(a.kind in IMAGE_INPUT_KINDS for a in ctx.assets)
    viable: list[str] = []
    for a in approaches[: s.agent_max_approaches]:
        if a in ("mesh", "hybrid") and (not has_images or not ctx.mesh_available):
            continue  # resolve_mode's guards, applied per strategy entry
        viable.append(a)
    if not viable:
        viable = [mode]
    n = len(viable)

    def _rank(o: RunOutcome) -> tuple[bool, int]:
        return (o.status in ("succeeded", "needs_review"), o.score or 0)

    def _time_left() -> float:
        from cadx.config import get_settings

        return get_settings().agent_job_timeout_s - (time.monotonic() - t0)

    best: RunOutcome | None = None
    used: list[str] = []
    start_it = 1
    for i, approach in enumerate(viable):
        if i > 0 and best is not None and best.status == "succeeded":
            break
        if await cancelled():
            break
        ctx.resolved_mode = approach
        used.append(approach)
        await ctx.emit("plan", 0, f"strategy {i + 1}/{n}: {approach}")

        switch_check = None
        if i + 1 < n:

            def switch_check(it_i: int, best_score: int | None) -> bool:
                return bool(
                    it_i >= s.agent_switch_after_iters
                    and (best_score or 0) < s.agent_switch_score
                    and _time_left() >= 900
                )

        sub = RunOutcome(
            spec=outcome.spec or (spec.model_dump() if spec else None),
            plan=outcome.plan,
            resolved_mode=approach,
            mode_note=outcome.mode_note,
        )
        sub = await _run_approach(
            ctx,
            sub,
            spec,
            plan,
            t0,
            cancelled,
            finish_early,
            approach=approach,
            max_repairs=max_repairs,
            switch_check=switch_check,
            start_iteration=start_it,
            **kwargs,
        )
        if best is None or _rank(sub) > _rank(best):
            best = sub
        start_it = max(start_it, sub.iterations + 1)
        if best.status == "succeeded":
            break
        if not sub.switch_requested and sub.status != "failed":
            break  # completed (needs_review) — switching just repeats the loop
    assert best is not None
    if len(approaches) > 1:
        best.mode_note += f" | strategy: {' → '.join(used)}"
    ctx.resolved_mode = best.resolved_mode or used[-1]
    return best


# --------------------------------------------------------------------------- #
# main state machine                                                           #
# --------------------------------------------------------------------------- #


async def run_generation(ctx: AgentCtx) -> RunOutcome:  # noqa: C901
    s = ctx.runtime_settings
    outcome = RunOutcome()
    t0 = time.monotonic()

    max_iterations = (
        ctx.settings.max_iterations
        if ctx.settings.max_iterations is not None
        else s.agent_max_iterations
    )
    max_repairs = (
        ctx.settings.max_repairs if ctx.settings.max_repairs is not None else s.agent_max_repairs
    )
    pass_score = (
        ctx.settings.pass_score if ctx.settings.pass_score is not None else s.agent_pass_score
    )
    tol_pct = (
        ctx.settings.dim_tolerance_pct
        if ctx.settings.dim_tolerance_pct is not None
        else s.agent_dim_tolerance_pct
    )
    tol_mm = (
        ctx.settings.dim_tolerance_mm
        if ctx.settings.dim_tolerance_mm is not None
        else s.agent_dim_tolerance_mm
    )
    clarify = (ctx.settings.clarification_mode or s.agent_clarification_mode) == "ask"

    async def cancelled() -> bool:
        if await ctx.check_cancel():
            outcome.status = "cancelled"
            return True
        return False

    async def finish_early() -> bool:
        """User clicked 'finish & keep best' — loops break at the next
        candidate checkpoint and finalise with the best result so far."""
        return bool(ctx.check_finish and await ctx.check_finish())

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

    # ---------- MEASURE ----------
    try:
        ctx.measurements = await measure_step.run(ctx, spec)
    except Exception as e:
        log.warning("agent.measure_failed", error=str(e))
        ctx.measurements = {}
    if ctx.measurements:
        # image-only anchors for the CAD path; hybrid prep reconciles with
        # detected mesh features and may mirror the mesh
        a = build_anchors(spec, ctx.measurements, None)
        if a.holes or a.image_outline:
            ctx.anchors = a.model_dump()
            await ctx.emit("anchors", 0, a.note)

    # adaptive budget: simple dimensioned parts don't need the global max
    # (an explicit per-version/per-project max_iterations always wins)
    if not ctx.explicit_max_iterations:
        capped = adaptive_max_iterations(
            spec,
            max_iterations,
            n_images=sum(1 for a in ctx.assets if a.kind in IMAGE_INPUT_KINDS),
            organic=ctx.gen_mode in ("mesh", "hybrid"),
        )
        if capped < max_iterations:
            max_iterations = capped
            await ctx.emit(
                "intake",
                0,
                f"simple stated spec — iteration budget capped at {max_iterations}",
            )

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

    # ---------- HERO IMAGE AUTOGEN ----------
    # mesh/hybrid without input images: synthesise a reference first so the
    # mode guards below see has_images=True (failures degrade to CAD as before)
    await _autogen_hero_image(ctx, spec, plan)

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

    # ---------- STRATEGY ----------
    outcome = await _run_strategy(
        ctx,
        outcome,
        spec,
        plan,
        t0,
        cancelled,
        finish_early,
        mode=mode,
        max_repairs=max_repairs,
        **kwargs,
    )

    # ---------- OPTIMIZE (topopt mode: SIMP + FEM verify on the base part) --
    wants_opt = (
        ctx.topopt_requested
        or (plan is not None and plan.optimize)
        or (spec.optimize is not None and spec.optimize.enabled)
    )
    if wants_opt:
        if ctx.topopt_requested:
            why = "requested by user"
        elif plan is not None and plan.optimize:
            why = plan.optimize_reason or "planner decision"
        else:
            why = "enabled in spec"
        await ctx.emit("optimize", 0, f"topology optimization: {why}")
        outcome = await _run_topopt_phase(ctx, outcome, spec)
    elif plan is not None and not plan.optimize:
        import re

        if re.search(r"optimi|leicht|light|gewicht|weight|\d+\s*N\b", ctx.prompt or "", re.I):
            await ctx.emit(
                "optimize",
                0,
                "note: prompt mentions loads/lightweight but planner set optimize=false",
            )

    if ctx.measurements:
        outcome.analysis = {**(outcome.analysis or {}), "measurements": ctx.measurements}
    if ctx.anchors:
        outcome.analysis = {**(outcome.analysis or {}), "anchors": ctx.anchors}
    return outcome
