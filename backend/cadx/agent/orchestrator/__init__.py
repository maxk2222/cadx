"""Agent orchestrator: the bounded generate-verify-revise state machine.

INTAKE -> (CLARIFY?) -> PLAN -> mode resolution -> one of:
  cad:    [CODEGEN -> EXECUTE -> (REPAIR x N) -> ANALYZE -> CRITIQUE] x K -> FINALIZE
  mesh:   [MESH_GENERATE -> ORIENT(once) -> PROCESS_MESH -> ANALYZE -> CRITIQUE] x K -> FINALIZE
  hybrid: MESH_GENERATE -> ORIENT -> PROCESS_MESH(+solid) -> cad loop with load_mesh_solid()


The pipelines live in `orchestrator/{cad,mesh,fem}_pipeline.py`, shared
helpers in `orchestrator/shared.py` — all re-exported here so every
`from cadx.agent.orchestrator import ...` call site keeps working.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from cadx import storage
from cadx.agent.anchors import build_anchors
from cadx.agent.executor_client import exec_error
from cadx.agent.steps import intake as intake_step
from cadx.agent.steps import measure as measure_step
from cadx.agent.steps import plan as plan_step
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx, InputAsset
from cadx.cad.mesh_ops import (
    MeshOp,
    MeshPostParams,
    merge_post_params,
)
from cadx.cad.spec_features import (
    active_features,
)
from cadx.llm.schemas import (
    Critique,
    Feature,
    PartSpec,
)
from cadx.logging import get_logger

log = get_logger(__name__)



# --- re-exports: keep every `from cadx.agent.orchestrator import ...` path stable ---
from cadx.agent.orchestrator.cad_pipeline import (  # noqa: E402,F401
    _attempt_history,
    _exec_error_signature,
    _exec_payload,
    _export_lost_holes,
    _has_degenerate_case,
    _has_unsolvable_case,
    _issue_source_rollup,
    _log_attempt,
    _measurement_authoritative,
    _note_exec_error,
    _params_vs_spec,
    _regression_critique,
    _repair_error_history,
    _resolve_patch,
    _run_cad,
    _select_best_candidate,
    _session_render_existing_fn,
    _session_render_fn,
    _session_verify_fn,
)
from cadx.agent.orchestrator.fem_pipeline import (  # noqa: E402,F401
    _normalize_topopt_base,
    _promote_topopt_exports,
    _run_topopt_phase,
    _stated_spec_dims,
    _topopt_headline_sf,
    _topopt_optimize_payload,
)
from cadx.agent.orchestrator.mesh_pipeline import (  # noqa: E402,F401
    _adapter_default_steps,
    _ahash,
    _arrange_gen_images,
    _assign_synth_roles,
    _base_seed,
    _bootstrap_concept_images,
    _cand_key,
    _candidate_adapters,
    _candidate_seed,
    _canonical_up_axis,
    _dedupe_near_identical,
    _ensure_synth_views,
    _fit_mesh_to_stated,
    _fit_ops_to_stated,
    _gate_generated_images,
    _gen_view_roles,
    _generate_mesh,
    _generate_mesh_batch,
    _is_flat_spec_dims,
    _map_gen_roles,
    _measured_plan_outline,
    _mesh_cache_lookup,
    _mesh_payload,
    _mesh_progress,
    _mesh_source_images,
    _none,
    _omni_control,
    _only_dim_issues,
    _orient_mesh,
    _orientation_decisive,
    _orientation_ranking,
    _parametrize_best,
    _prepped_gen_images,
    _process_mesh,
    _ref_side_mask,
    _resolve_mesh_adapter,
    _revolve_axis,
    _rotate_hero,
    _run_mesh,
    _score_candidate_mesh,
    _select_candidate,
    _spec_drill_ops,
    _spec_silhouette_ops,
    _synth_back_view,
    _try_heightfield_candidate,
    stated_dim_count,
)
from cadx.agent.orchestrator.shared import (  # noqa: E402,F401
    _ADAPTER_CANONICAL_UP,
    _CANDIDATE_VIABLE_SCORE,
    _CRITIQUE_PREV_ISSUES,
    _FEM_LIGHTWEIGHT_KW,
    _FEM_SMOKE_TIMEOUT_S,
    _FLAT_BOTTOM_CUE,
    _ISSUE_FEATURE_RE,
    _ISSUE_NEGATION_RE,
    _LIGHTWEIGHT_RE,
    _LOAD_HINT_RE,
    Candidate,
    RunOutcome,
    _annotate_anchor_sources,
    _clean_losing_attempts,
    _copy_best_artifacts,
    _copy_fem_artifacts,
    _copy_mesh_artifacts,
    _multi_solid,
    _resolve_issue_conflicts,
    _verify_spec_issue,
    _write_mesh_json,
    build_issues,
    compare_dims,
)


async def _apply_flat_back(
    ctx: AgentCtx,
    it: int,
    mesh_rel: str,
    spec: PartSpec,
    post: MeshPostParams,
    exec_result: dict[str, Any],
    art_attempt: Any,
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
    flat_result = await _process_mesh(
        ctx, it, mesh_rel, spec, flat_post, make_solid=True, subdir=f"a{next(art_attempt)}"
    )
    if flat_result.get("ok"):
        rough = (flat_result.get("analysis") or {}).get("roughness_deg")
        rough_before = (exec_result.get("analysis") or {}).get("roughness_deg")
        if rough is not None and rough_before is not None:
            await ctx.emit(
                "meshprocess", it, f"surface roughness {rough_before:.0f}° → {rough:.0f}°"
            )
        return flat_result, flat_post
    return exec_result, post
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
    from cadx.agent.image_check import generate_with_verification
    from cadx.agent.image_prompt import hero_prompt_parts

    s = ctx.runtime_settings
    parts = hero_prompt_parts(spec, ctx.prompt.strip(), subject)
    prompt = read_prompt("image_hero.md").format(**parts)
    await ctx.emit("imagegen", 0, "no reference image — generating a hero image")

    async def _prog(msg: str, _p: float) -> None:
        await ctx.emit("imagegen", 0, msg, sub_progress=_p, sub_label=msg)

    # generate → vision-verify → persist ONLY the winning attempt: rejected
    # images must never reach ctx.assets (a bad hero poisons the whole run)
    side = min(1024, s.image_max_side)
    tmp_dirs: list[Path] = []
    last: dict[str, Any] = {}

    async def _gen(p: str) -> list[InputAsset]:
        from cadx.agent.image_client import ImageGenRequest

        tmp = Path(tempfile.mkdtemp(prefix="cadx-hero-"))
        tmp_dirs.append(tmp)
        req = ImageGenRequest(
            prompt=p,
            model=s.image_model,
            seed=_base_seed(ctx) + len(tmp_dirs) * 977,
            count=1,
            width=side,
            height=side,
            steps=s.image_steps,
            guidance=s.image_guidance,
            negative=parts["negative"],
            out_dir=tmp,
        )
        result = await ctx.image.generate(req, _prog, ctx.check_cancel)
        last["result"] = result
        return [
            InputAsset(
                asset_id="",
                kind="generated_image",
                path=pp,
                caption="",
                filename=pp.name,
            )
            for pp in result.image_paths
        ]

    try:
        retries = int(getattr(s, "image_verify_retries", 0) or 0)
        gen_assets, verdict = await generate_with_verification(
            ctx, prompt=prompt, spec=spec, gen_fn=_gen, retries=retries
        )
        result = last["result"]
        from cadx.agent.image_assets import persist_generated_images
        from cadx.db.session import session_scope

        async with session_scope() as db:
            assets = persist_generated_images(
                db,
                ctx.project_id,
                [a.path for a in gen_assets],
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
                    "verify": verdict.model_dump(),
                    "attempts": len(tmp_dirs),
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
                    verify_severity=verdict.severity,
                )
            )
            ctx.generated_asset_ids.append(str(a.id))
        if verdict.severity == "major":
            await ctx.emit(
                "imagegen",
                0,
                "hero image does not match the spec after retries — continuing anyway",
            )
        await ctx.emit("imagegen", 0, f"hero image generated ({result.model})")
    except Exception as e:
        log.warning("agent.hero_image_failed", error=str(e))
        await ctx.emit("imagegen", 0, f"hero image generation failed: {e} — continuing without")
    finally:
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)
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
async def _clip_mesh_to_measured(
    ctx: AgentCtx,
    it: int,
    mesh_rel: str,
    spec: PartSpec,
    post: MeshPostParams,
    exec_result: dict,
    art_attempt: Any,
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
    res_c = await _process_mesh(
        ctx, it, mesh_rel, spec, post_c, make_solid=True, subdir=f"a{next(art_attempt)}"
    )
    if not res_c.get("ok"):
        return exec_result, post
    feats2 = (res_c.get("mesh") or {}).get("features") or {}
    iou2 = _outline_iou_mm(feats2.get("outline_mm") or [], outline)
    if iou2 >= iou:
        # dimension guard: an intersection can only shrink the part. When the
        # generated footprint is locally undersized vs the measured outline
        # (tips not reaching it), clipping pulls the bbox below stated dims —
        # observed: a fitted 119x34x8 scale clipped to 115x28.5. Prefer the
        # loose-footprint mesh over an undersized part.
        rt = ctx.runtime_settings
        tol_pct = (
            ctx.settings.dim_tolerance_pct
            if ctx.settings.dim_tolerance_pct is not None
            else rt.agent_dim_tolerance_pct
        )
        tol_mm = (
            ctx.settings.dim_tolerance_mm
            if ctx.settings.dim_tolerance_mm is not None
            else rt.agent_dim_tolerance_mm
        )
        dims_before = compare_dims(spec, exec_result.get("analysis") or {}, tol_pct, tol_mm)
        dims_after = compare_dims(spec, res_c.get("analysis") or {}, tol_pct, tol_mm)
        if len(dims_after) > len(dims_before):
            await ctx.emit(
                "meshprocess",
                it,
                f"footprint clip rejected — shrank dims ({len(dims_before)} → "
                f"{len(dims_after)} issues: {'; '.join(dims_after[:2])})",
            )
            return exec_result, post
        await ctx.emit(
            "meshprocess",
            it,
            f"clipping footprint to measured outline (IoU {iou:.2f} → {iou2:.2f})",
        )
        return res_c, post_c
    await ctx.emit("meshprocess", it, f"footprint clip rejected (IoU {iou:.2f} → {iou2:.2f})")
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
def _hybrid_prefers_cad(spec: PartSpec) -> bool:
    """A mesh-only rescue after a failed mesh->solid conversion is only a
    real fallback for organic parts — a dimensioned or optimization spec
    needs the parametric B-rep the rest of the pipeline verifies/optimizes
    (observed: a lattice scaffold that couldn't sew into a solid was still
    committed as the 'mesh' answer for a dimensioned L-bracket spec)."""
    return stated_dim_count(spec) >= 2 or (spec.optimize is not None and spec.optimize.enabled)
def _merge_features(old: list[Feature], new: list[Feature]) -> list[Feature]:
    """Merge second-chance extraction results into the existing spec list.

    ``spec.features = fixed.features`` used to REPLACE the list wholesale —
    the retry answers "every feature the prompt names" from scratch, so a
    partial intake lost the entries it had actually gotten right. Keep all
    existing features and append retry entries whose (name, type) pair is
    not already declared (P1)."""
    seen = {((f.name or "").strip().lower(), (f.type or "").strip().lower()) for f in old}
    merged = list(old)
    for f in new:
        key = ((f.name or "").strip().lower(), (f.type or "").strip().lower())
        if key not in seen:
            seen.add(key)
            merged.append(f)
    return merged
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
def _persist_hybrid_mesh(ctx: AgentCtx, it: int) -> None:
    """Best-effort early persistence of the mesh intermediates into
    versions/<vid>/mesh/ — a codegen crash in the following CAD loop must not
    lose the generated solid. The finalize path re-registers the same files
    as attempt artifacts, so persistence here must never break prep."""
    with contextlib.suppress(Exception):
        _copy_mesh_artifacts(ctx.project_id, ctx.version_id, it, ctx.mesh_info.get("_prep_art_dir"))
        _write_mesh_json(ctx.project_id, ctx.version_id, ctx.mesh_info)
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
    # artifact slots for the prep sub-passes — several process_mesh calls write
    # into iteration 0 and must not clobber each other's _run/ outputs
    art_attempt = itertools.count(1)
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
        if ctx.mesh_info.get("_mesh_hopeless"):
            return False, "no viable mesh candidate — inputs cannot produce the spec'd shape"
        return False, "mesh generation failed"
    ctx.mesh_raw_reuse = storage.resolve(mesh_rel)

    orientation = await _orient_mesh(ctx, it, mesh_rel)
    post = MeshPostParams(
        up=orientation.get("up"),  # type: ignore[arg-type]
        front=orientation.get("front"),  # type: ignore[arg-type]
        mirror_x=bool(orientation.get("mirror_x")),
        auto_yaw=True,
    )

    await ctx.emit("meshprocess", it, "processing mesh and building solid")
    exec_result = await _process_mesh(
        ctx, it, mesh_rel, spec, post, make_solid=True, subdir=f"a{next(art_attempt)}"
    )

    if not exec_result.get("ok"):
        return False, exec_error(exec_result, "mesh processing failed")

    # deterministic flat-back: when the spec demands a planar -z face and the
    # generative mesh bows its bottom, shave it flat BEFORE the dim gate — the
    # fit loop then re-scales z back to the stated thickness.
    exec_result, post = await _apply_flat_back(
        ctx, it, mesh_rel, spec, post, exec_result, art_attempt
    )

    # dimensional gate: a base mesh far outside stated dims is the wrong
    # object — CAD features can't repair a 40%-off organic body. Before
    # rejecting, try fitting the envelope deterministically — generative
    # meshes routinely get proportions wrong while capturing the shape.
    if stated_dim_count(spec) >= 2:
        dim_issues = compare_dims(spec, exec_result.get("analysis") or {}, tol_pct, tol_mm)
        dim_issues += list((exec_result.get("mesh") or {}).get("dim_issues") or [])
        if dim_issues:
            fitted = await _fit_mesh_to_stated(
                ctx, it, mesh_rel, spec, post, exec_result, tol_pct, tol_mm, art_attempt
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
        res_m = await _process_mesh(
            ctx, it, mesh_rel, spec, post_m, make_solid=True, subdir=f"a{next(art_attempt)}"
        )
        if res_m.get("ok"):
            exec_result, post = res_m, post_m
            anchors = build_anchors(
                spec, ctx.measurements, (exec_result.get("mesh") or {}).get("features")
            )
            anchors.mirror_x = True

    # footprint clip: when the XY silhouette still drifts from the measured
    # plan view, clip the mesh to the measured outline prism
    exec_result, post = await _clip_mesh_to_measured(
        ctx, it, mesh_rel, spec, post, exec_result, art_attempt
    )
    anchors = build_anchors(spec, ctx.measurements, (exec_result.get("mesh") or {}).get("features"))
    anchors.mirror_x = post.mirror_x

    if anchors.holes or anchors.image_outline:
        ctx.anchors = anchors.model_dump()
        await ctx.emit("anchors", it, anchors.note)

    ctx.mesh_info["post"] = post.model_dump()
    ctx.mesh_info["orientation"] = {"up": post.up, "front": post.front}
    ctx.mesh_info["bbox_size"] = (exec_result.get("analysis") or {}).get("bbox_size")
    ctx.mesh_info["triangles"] = (exec_result.get("mesh") or {}).get("triangles")
    ctx.mesh_info["_prep_art_dir"] = exec_result.get("art_dir")
    ctx.hybrid_mesh_info = {
        "bbox_size": (exec_result.get("analysis") or {}).get("bbox_size"),
        "volume": (exec_result.get("analysis") or {}).get("volume"),
        "triangles": (exec_result.get("mesh") or {}).get("triangles"),
    }

    if (exec_result.get("mesh") or {}).get("solid_ok") and exec_result["mesh"].get("solid_relpath"):
        ctx.hybrid_solid = storage.resolve(exec_result["mesh"]["solid_relpath"])
        _persist_hybrid_mesh(ctx, it)
        return True, "mesh solid ready"

    # retry once with a tighter cap on the *solid copy* only — the pipeline
    # mesh and its exports stay at full resolution
    s = ctx.runtime_settings
    cap = min(15_000, s.mesh_decimate_faces)
    await ctx.emit("meshprocess", it, "solid conversion failed — retrying with fewer faces")
    exec_result = await _process_mesh(
        ctx,
        it,
        mesh_rel,
        spec,
        post,
        make_solid=True,
        solid_max_faces=cap,
        subdir=f"a{next(art_attempt)}",
    )
    if exec_result.get("ok") and (exec_result.get("mesh") or {}).get("solid_ok"):
        ctx.hybrid_solid = storage.resolve(exec_result["mesh"]["solid_relpath"])
        ctx.mesh_info["post"] = post.model_dump()
        ctx.mesh_info["bbox_size"] = (exec_result.get("analysis") or {}).get("bbox_size")
        ctx.mesh_info["_prep_art_dir"] = exec_result.get("art_dir")
        ctx.hybrid_mesh_info = {
            "bbox_size": (exec_result.get("analysis") or {}).get("bbox_size"),
            "volume": (exec_result.get("analysis") or {}).get("volume"),
            "triangles": (exec_result.get("mesh") or {}).get("triangles"),
        }
        _persist_hybrid_mesh(ctx, it)
        return True, "mesh solid ready (decimated)"

    # keep the processed mesh as a candidate for the mesh-mode fallback
    if exec_result.get("ok"):
        ctx.mesh_info["_pre_result"] = True
        ctx.mesh_info["_pre_result_exec"] = exec_result
        ctx.mesh_info["_pre_result_mesh_rel"] = mesh_rel
    return False, "mesh-to-solid conversion failed"
def _prompt_hints_optimize(prompt: str) -> bool:
    """The prompt itself asking for optimization is a valid trigger — weak
    models regularly leave plan.optimize / spec.optimize unset."""
    return bool(re.search(r"optimi|topolog|leicht|light|gewicht|weight", prompt or "", re.I))
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
            # the planner's mesh/hybrid steps must not leak into CAD codegen —
            # they mention load_mesh_solid() which does not exist here
            plan = plan_step.cad_fallback_plan(plan)
            ctx.plan = plan
            outcome.plan = plan.model_dump() if plan else outcome.plan
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
            # a declared-cad_fallback best resolves "cad" inside _run_cad —
            # such an outcome must not report mesh provenance for a part
            # that is pure parametric CAD
            if outcome.status != "failed" and ctx.mesh_info and ctx.resolved_mode == "hybrid":
                outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
        elif ctx.mesh_info.get("_pre_result") and not _hybrid_prefers_cad(spec):
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
            plan = plan_step.cad_fallback_plan(plan)
            ctx.plan = plan
            outcome.plan = plan.model_dump() if plan else outcome.plan
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
        # cad can also be reached via mode-resolution fallbacks (mesh mode
        # without a server, strategy approach #2 after a failed mesh leg) —
        # the plan may still carry mesh-only steps; strip them
        if plan is not None and getattr(plan, "approach", "cad") != "cad":
            plan = plan_step.cad_fallback_plan(plan)
            ctx.plan = plan
            outcome.plan = plan.model_dump() if plan else outcome.plan
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
    viable, n_planned = _viable_approaches(ctx, mode, plan)
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
        if await ctx.check_cancel():
            sub.status = "cancelled"
            best = sub
            break
        if best.status == "succeeded":
            break
        if not sub.switch_requested and sub.status != "failed":
            break  # completed (needs_review) — switching just repeats the loop
    if best is None:
        outcome.status = "cancelled"
        return outcome
    if best.status == "cancelled":
        ctx.resolved_mode = best.resolved_mode or used[-1]
        return best
    if n_planned > 1:
        best.mode_note += f" | strategy: {' → '.join(used)}"
    ctx.resolved_mode = best.resolved_mode or used[-1]
    return best
def _spec_wants_flat_bottom(spec: PartSpec) -> bool:
    """Text cue: spec demands a planar -z face (mounting/flat back)."""
    texts = [spec.name or "", spec.summary or ""]
    texts += [f"{f.name} {f.description}" for f in active_features(spec)]
    return any(_FLAT_BOTTOM_CUE.search(t) for t in texts)
def _viable_approaches(ctx: AgentCtx, mode: str, plan: Any) -> tuple[list[str], int]:
    """Ordered approach list with resolve_mode's guards applied per entry.

    Returns (viable, planned_count) — planned_count keeps the unfiltered
    strategy length so the mode note can say 'strategy: a → b' even when
    guards dropped an approach."""
    s = ctx.runtime_settings
    forced = ctx.gen_mode not in (None, "auto", "topopt")
    approaches = [mode] if forced else (list(plan.strategy) if plan and plan.strategy else [mode])
    has_images = any(a.kind in IMAGE_INPUT_KINDS for a in ctx.assets)
    viable: list[str] = []
    for a in approaches[: s.agent_max_approaches]:
        if a in ("mesh", "hybrid"):
            # resolve_mode's reuse exemption, per entry: a stored parent solid
            # (hybrid) / raw GLB (mesh) stays viable without images or a live
            # mesh server — nothing is generated, only re-processed
            reuse = (a == "hybrid" and ctx.parent.mesh_solid is not None and ctx.parent.mesh) or (
                a == "mesh" and ctx.mesh_raw_reuse is not None
            )
            if not reuse and (not has_images or not ctx.mesh_available):
                continue  # resolve_mode's guards, applied per strategy entry
        viable.append(a)
    return viable or [mode], len(approaches)
def _wants_optimize(ctx: AgentCtx, plan: Any, spec: Any, prompt_hints_opt: bool) -> bool:
    """Decide whether the OPTIMIZE phase runs after base generation.

    Trigger channels: an explicit topopt request (gen_mode=topopt →
    ctx.topopt_requested), the planner's plan.optimize, spec.optimize.enabled,
    or the prompt mentioning optimization/weight. An explicitly forced
    gen_mode (cad/mesh/hybrid) wins over every channel: the caller chose the
    deliverable pipeline — "input for topology optimization" in the prompt
    describes intent, not a command — so the phase stays off and the caller
    can still run a dedicated optimize job."""
    if ctx.gen_mode not in (None, "auto", "topopt"):
        return False
    return bool(
        ctx.topopt_requested
        or (plan is not None and plan.optimize)
        or (spec is not None and spec.optimize is not None and spec.optimize.enabled)
        or prompt_hints_opt
    )
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
    n_features = len(active_features(spec))
    if stated >= 2 and n_features <= 2:
        return min(requested, 5)
    if stated >= 1 and n_features <= 6:
        return min(requested, 15)
    return requested
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
        # reuse exemption: a job that only re-processes stored intermediates
        # (parent solid.brep for a hybrid edit, a kept raw GLB for mesh mode)
        # needs neither image input nor a live mesh server
        if requested == "hybrid" and ctx.parent.mesh_solid is not None and ctx.parent.mesh:
            return "hybrid", "hybrid mode — reusing stored mesh solid"
        if requested == "mesh" and ctx.mesh_raw_reuse is not None:
            return "mesh", "mesh mode — reusing stored raw mesh"
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

    # ---------- RESEARCH (optional, gated) ----------
    if ctx.spec is None and ctx.research is None:
        try:
            from cadx.agent.steps import research as research_step

            ctx.research = await research_step.run(ctx)
        except Exception as e:
            log.info("research.skipped", error=str(e)[:200])

    # ---------- INTAKE ----------
    spec = ctx.spec
    if spec is None:
        await ctx.emit("intake", 0, "analysing images and description")
        try:
            spec = await intake_step.run(ctx)
        except Exception as e:
            outcome.error = f"intake failed: {e}"
            return outcome
    ctx.spec = spec
    outcome.spec = spec.model_dump()

    # ---- structural-load second chance ----
    # weak models routinely drop the forces/optimization half of the prompt:
    # `load_cases` empty or degenerate (`{"fixed":[],"loads":[]}`). Recover it
    # right after intake so the planner, per-iteration FEM checks and the
    # topopt phase all see real cases instead of unsolvable placeholders.
    prompt = ctx.prompt or ""
    prompt_hints_loads = bool(_LOAD_HINT_RE.search(prompt))
    tried_load_extraction = False
    intake_step.sanitize_load_cases(spec)
    dim_notes = intake_step.sanitize_dims(spec, prompt)
    dim_notes.extend(intake_step.resolve_thread_intents(spec))
    dim_notes.extend(intake_step.sanitize_standard_part_names(spec))
    dim_notes.extend(intake_step.sanitize_standard_part_dims(spec))
    pos_notes = intake_step.normalize_feature_positions(spec)
    dim_notes.extend(intake_step.sanitize_stated_dims(spec, prompt))
    if dim_notes or pos_notes:
        outcome.spec = spec.model_dump()
        ctx.spec = spec
        for note in (*dim_notes, *pos_notes):
            await ctx.emit("intake", 0, note)

    # ---- degenerate-spec second chance ----
    # a spec that echoes a tool name or drops every feature the prompt names
    # would produce a confident-but-wrong part downstream (observed: bracket
    # prompt -> `name:"search_catalog"` -> plain box scored 95)
    gaps = intake_step.spec_gaps(spec, prompt)
    want_loads = bool(prompt_hints_loads and not spec.load_cases)
    if gaps or want_loads:
        if gaps:
            await ctx.emit("intake", 0, "spec incomplete: " + "; ".join(gaps) + " — re-asking")
        # feature + load-case recovery are independent vision calls — gather
        fixed, loads_res = await asyncio.gather(
            intake_step.extract_features(ctx, spec) if gaps else _none(),
            intake_step.extract_load_cases(ctx, spec) if want_loads else _none(),
        )
        if gaps:
            tool_echo = any("tool echo" in g for g in gaps)
            if fixed.features:
                # degenerate spec (tool echo) — wholesale replace as before;
                # otherwise merge so entries intake got right survive (P1)
                spec.features = (
                    fixed.features if tool_echo else _merge_features(spec.features, fixed.features)
                )
                for note in intake_step.dedup_spec_features(spec):
                    await ctx.emit("intake", 0, note)
            if fixed.standard_parts:
                spec.standard_parts = fixed.standard_parts
            if fixed.name.strip() and tool_echo:
                spec.name, spec.summary = fixed.name, fixed.summary or spec.summary
            outcome.spec = spec.model_dump()
            ctx.spec = spec
            await ctx.emit("intake", 0, f"recovered {len(spec.features)} feature(s)")
        if want_loads:
            tried_load_extraction = True
            cases, opt = loads_res if loads_res else (None, None)
            if cases:
                spec.load_cases = cases
                if (
                    opt is not None
                    and opt.enabled
                    and (spec.optimize is None or not spec.optimize.enabled)
                ):
                    spec.optimize = opt
                intake_step.sanitize_load_cases(spec)
                outcome.spec = spec.model_dump()
                await ctx.emit("intake", 0, f"recovered {len(cases)} load case(s) missed by intake")
        # recovery may have replaced standard_parts — re-resolve thread
        # intents (idempotent: skips parts whose knobs are already set)
        for note in intake_step.resolve_thread_intents(spec):
            await ctx.emit("intake", 0, note)

    # ---------- MEASURE ----------
    try:
        ctx.measurements = await measure_step.run(ctx, spec)
    except Exception as e:
        log.warning("agent.measure_failed", error=str(e))
        ctx.measurements = {}
    # spec-declared + image anchors for the CAD path; spec anchors resolve
    # without any image measurement, hybrid prep later reconciles with
    # detected mesh features and may mirror the mesh
    a = build_anchors(spec, ctx.measurements or {}, None)
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
                max_iterations=max_iterations,
            )

    # ---------- CLARIFY ----------
    blocking = [q for q in spec.open_questions if q.blocking]
    norm_part_blocking = any("registered standard component" in q.why_it_matters for q in blocking)
    if blocking and not ctx.settings.eval_mode and (clarify or norm_part_blocking):
        outcome.status = "waiting_for_input"
        outcome.questions = [q.model_dump() for q in spec.open_questions]
        outcome.resume_state = {"spec": spec.model_dump()}
        await ctx.emit(
            "intake",
            0,
            f"waiting for answers ({len(blocking)} blocking questions)",
            max_iterations=max_iterations,
        )
        return outcome
    # not asking (assume/proceed): unresolved blocking questions must not
    # shadow-gate downstream paths (is_standard_only, recipes) — consume them
    # with their declared default assumption, same as a resume would
    resolved_any = False
    for question in spec.open_questions:
        if question.blocking:
            question.blocking = False
            resolved_any = True
            default = question.default_assumption
            if default and default not in spec.assumptions:
                spec.assumptions.append(default)
    if resolved_any:
        # apply the consequences of the defaults — a resolved "catalog dims
        # govern" question must actually update the spec dims, or verify keeps
        # flagging the stated-vs-catalog mismatch on every iteration
        resolved_notes = intake_step.resolve_thread_intents(spec)
        resolved_notes += intake_step.sanitize_standard_part_names(spec)
        resolved_notes += intake_step.sanitize_standard_part_dims(spec)
        outcome.spec = spec.model_dump()
        for note in resolved_notes:
            await ctx.emit("intake", 0, note)

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
    await ctx.emit("plan", 0, f"approach: {mode} — {mode_note}", max_iterations=max_iterations)

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
    forced_mode = ctx.gen_mode not in (None, "auto", "topopt")
    prompt_hints_opt = _prompt_hints_optimize(prompt)
    wants_opt = _wants_optimize(ctx, plan, spec, prompt_hints_opt)
    if (
        wants_opt
        and not spec.load_cases
        and not tried_load_extraction
        and outcome.status in ("succeeded", "needs_review")
    ):
        # last-ditch extraction when intake-time recovery was skipped (e.g.
        # planner decided to optimize a prompt without load keywords)
        cases, opt = await intake_step.extract_load_cases(ctx, spec)
        if cases:
            spec.load_cases = cases
            if (
                opt is not None
                and opt.enabled
                and (spec.optimize is None or not spec.optimize.enabled)
            ):
                spec.optimize = opt
            await ctx.emit(
                "optimize", 0, f"intake missed loads — retry extracted {len(cases)} case(s)"
            )
    if wants_opt and outcome.status == "cancelled":
        wants_opt = False
    if wants_opt:
        if ctx.topopt_requested:
            why = "requested by user"
        elif plan is not None and plan.optimize:
            why = plan.optimize_reason or "planner decision"
        elif spec.optimize is not None and spec.optimize.enabled:
            why = "enabled in spec"
        else:
            why = "prompt asks for loads/optimization"
        await ctx.emit("optimize", 0, f"topology optimization: {why}")
        outcome = await _run_topopt_phase(ctx, outcome, spec, t0)
    elif forced_mode and (
        (plan is not None and plan.optimize)
        or (spec.optimize is not None and spec.optimize.enabled)
        or prompt_hints_opt
    ):
        await ctx.emit(
            "optimize",
            0,
            f"note: optimization requested but gen_mode={ctx.gen_mode} "
            "— skipped (select topopt mode or run an optimize job)",
        )
    elif plan is not None and not plan.optimize and prompt_hints_loads:
        await ctx.emit(
            "optimize",
            0,
            "note: prompt mentions loads/lightweight but planner set optimize=false",
        )

    if ctx.measurements:
        outcome.analysis = {**(outcome.analysis or {}), "measurements": ctx.measurements}
    if ctx.anchors:
        outcome.analysis = {**(outcome.analysis or {}), "anchors": ctx.anchors}
    outcome.spec = spec.model_dump()
    return outcome
