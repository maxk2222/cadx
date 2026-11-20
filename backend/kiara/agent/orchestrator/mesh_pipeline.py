"""Mesh generation pipeline — `_run_mesh` plus image/candidate/orientation helpers.

Split out of the former `orchestrator.py` monolith; `kiara.agent.orchestrator`
re-exports every name so import paths stay stable."""

from __future__ import annotations

import asyncio
import itertools
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from kiara import storage
from kiara.agent.executor_client import exec_error
from kiara.agent.mesh_client import MeshGenError, MeshGenRequest, MeshGenVariant
from kiara.agent.orchestrator.shared import (
    _ADAPTER_CANONICAL_UP,
    _CANDIDATE_VIABLE_SCORE,
    Candidate,
    RunOutcome,
    _annotate_anchor_sources,
    _clean_losing_attempts,
    _copy_best_artifacts,
    _copy_mesh_artifacts,
    _multi_solid,
    _verify_spec_issue,
    _write_mesh_json,
    build_issues,
    compare_dims,
)
from kiara.agent.quality import det_quality
from kiara.agent.steps import critique as critique_step
from kiara.agent.steps import mesh_orient as mesh_orient_step
from kiara.agent.steps import mesh_refine as mesh_refine_step
from kiara.agent.types import IMAGE_INPUT_KINDS, AgentCtx
from kiara.cad.mesh_ops import (
    MESH_OPS_VERSION,
    MeshOp,
    MeshPostParams,
    merge_post_params,
    mesh_params_schema,
)
from kiara.cad.spec_features import (
    active_features,
    dropped_feature_indices,
)
from kiara.llm.schemas import (
    Critique,
    PartSpec,
)
from kiara.logging import get_logger

log = get_logger(__name__)


async def _adapter_default_steps(ctx: AgentCtx, model: str | None) -> int | None:
    """Adapter-native inference step count for `model` (None when unknown or
    not reported — distilled turbo models report ~5). Cached per job."""
    if ctx.mesh is None or not model or model == "auto":
        return None
    defaults = ctx.mesh_info.get("_model_steps")
    if defaults is None:
        defaults = {}
        try:
            for m in (await ctx.mesh.health()).models:
                if m.default_steps:
                    defaults[m.id] = int(m.default_steps)
        except Exception:
            pass
        ctx.mesh_info["_model_steps"] = defaults
    return defaults.get(model)
def _ahash(path: Path) -> int | None:
    """8x8 grayscale average-hash — cheap perceptual fingerprint."""
    try:
        from PIL import Image

        img = Image.open(path).convert("L").resize((8, 8), Image.LANCZOS)
        px = list(img.getdata())
        avg = sum(px) / len(px)
        return sum(1 << i for i, v in enumerate(px) if v > avg)
    except Exception:
        return None
async def _arrange_gen_images(
    ctx: AgentCtx,
    iteration: int,
    images: list[Any],
    roles: list[str],
    spec: PartSpec,
    prefer: str | None,
) -> tuple[str, list[Any], list[str]]:
    """Resolve the adapter, reorder for sketch-conditioned models, cap to the
    adapter's image budget (roles stay aligned). Returns (model, images, roles)."""
    model, img_cap = await _resolve_mesh_adapter(ctx, len(images), prefer=prefer, spec=spec)
    if img_cap > 1 and len(images) > 1:
        # multiview slots are semantically labelled (front/back/left/right) —
        # feeding a perspective shot into a view slot makes the generator
        # treat it as an orthographic view. Drop unlabelled extras.
        keep = {"front", "back", "left", "right", "scribble"}
        labelled = [(a, r) for a, r in zip(images, roles, strict=True) if r in keep]
        if len(labelled) < len(images):
            dropped = len(images) - len(labelled)
            if not labelled:
                labelled = [(images[0], roles[0])]
                dropped -= 1
            await ctx.emit(
                "meshgen",
                iteration,
                f"multiview: sending {len(labelled)} labelled view(s), "
                f"dropped {dropped} 'other' image(s)",
            )
            images = [a for a, _ in labelled]
            roles = [r for _, r in labelled]
    if model.endswith("-scribble"):
        # sketch-conditioned adapters read image[0] as the scribble — prefer a
        # user/agent-marked sketch (role_hint or intake 'scribble' role), fall
        # back to the measured-outline silhouette; photos stay as context
        si = next(
            (
                i
                for i, a in enumerate(images)
                if getattr(a, "role_hint", "") == "sketch" or roles[i] == "scribble"
            ),
            None,
        )
        if si is None:
            si = next((i for i, a in enumerate(images) if a.asset_id == "_synth_back"), None)
        if si:
            images.insert(0, images.pop(si))
            roles.insert(0, roles.pop(si))
    if len(images) > img_cap:
        await ctx.emit(
            "meshgen",
            iteration,
            f"adapter '{model}' accepts max {img_cap} image(s) — using first {img_cap}",
        )
        images = images[:img_cap]
        roles = roles[:img_cap]
    return model, images, roles
def _assign_synth_roles(
    ordered: list[Any], roles: list[str], synth_roles: dict[str, str]
) -> list[str]:
    """Synthesised side views occupy free left/right multiview slots; real
    labelled views keep theirs (first-free assignment)."""
    free = [r for r in ("left", "right") if r not in roles]
    for i, a in enumerate(ordered):
        if a.asset_id in synth_roles and roles[i] == "other" and free:
            roles[i] = free.pop(0)
    return roles
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
async def _bootstrap_concept_images(
    ctx: AgentCtx, iteration: int, spec: PartSpec, plan: Any
) -> list[Any]:
    """Synthesise a concept image when a mesh/hybrid run has zero input images.

    The image->3D adapters need at least one reference; with text-only prompts
    the planner can still choose mesh/hybrid and we generate one
    product-photo-style image from the subject description. Runs once per job —
    a failed attempt is flagged so later candidate/iteration calls don't retry.
    """
    if ctx.mesh_info.get("_concept_bootstrap"):
        return []
    ctx.mesh_info["_concept_bootstrap"] = True
    if ctx.image is None or not ctx.image_available:
        return []
    from kiara.agent.context import read_prompt

    subject = (plan.mesh_subject if plan else "") or (spec.name if spec else "") or ctx.prompt[:200]
    regen_hint = str(ctx.mesh_info.get("_regen_hint") or "").strip()
    if regen_hint:
        subject = f"{subject} — MUST include: {regen_hint}"
    if spec is not None:
        # same rich prompt as the hero path — proportions + visual features +
        # intent; the bare concept template produced an L-bracket for a bail
        # handle spec and failed the silhouette gate anyway
        from kiara.agent.image_prompt import hero_prompt_parts

        prompt = read_prompt("image_hero.md").format(
            **hero_prompt_parts(spec, ctx.prompt.strip(), subject)
        )
    else:
        prompt = read_prompt("image_concept.md").format(subject=subject)
    try:
        from kiara.agent.concept_image import generate_concept_images

        out = await generate_concept_images(
            ctx, prompt, count=1, source="concept_bootstrap", spec=spec
        )
    except Exception as e:
        log.warning("meshgen.concept_bootstrap_failed", error=str(e))
        await ctx.emit("meshgen", iteration, f"concept image generation failed: {e}")
        return []
    await ctx.emit(
        "meshgen",
        iteration,
        "no input images — generated concept reference for the mesh generator",
    )
    return list(out)
def _cand_key(seed: int, model: str | None) -> str:
    return f"{seed}|{model or ''}"
async def _candidate_adapters(ctx: AgentCtx, n: int) -> list[str]:
    """Adapter ids competing in best-of-N screening.

    `mesh_candidate_adapters` (csv) pins the set explicitly; "" spreads the
    candidates over the available adapters (device-native first) so different
    models compete instead of only seeds."""
    configured = [
        a.strip()
        for a in (ctx.runtime_settings.mesh_candidate_adapters or "").split(",")
        if a.strip()
    ]
    if configured:
        return configured[:n]
    # an explicitly pinned mesh_model wins over adapter spreading — the user
    # chose that model (e.g. to avoid a host-unstable adapter); candidates
    # then differ only by seed
    pinned = (ctx.settings.mesh_model or ctx.runtime_settings.mesh_model).strip()
    if pinned and pinned != "auto":
        return [pinned]
    try:
        health = await ctx.mesh.health() if ctx.mesh else None
    except Exception:
        health = None
    models = [
        m
        for m in (health.models if health else [])
        if m.available and m.id != "fake" and getattr(m, "kind", "mesh") == "mesh"
    ]
    if len(models) <= 1:
        return []
    device = health.device if health else ""
    models.sort(key=lambda m: (m.device == device, m.supports_multiview), reverse=True)
    return [m.id for m in models[:n]]
async def _candidate_seed(  # noqa: C901
    ctx: AgentCtx, iteration: int, spec: PartSpec, plan: Any, t0: float
) -> int | None:
    """Best-of-N cheap generation: N low-step candidates with different seeds,
    each scored deterministically (aspect + footprint IoU + hole count); the
    winning seed feeds the full-resolution generation. Runs once per job and
    only with >30 min of job budget left — it doubles the generation cost of
    the first roll but saves critique cycles on bad seeds."""
    s = ctx.runtime_settings
    n = s.mesh_candidates_per_iter
    if ctx.mesh_info.get("_candidates_done"):
        return None
    if iteration == 1 and ctx.mesh_raw_reuse is not None:
        return None  # parent mesh reuse — no fresh candidates needed
    ctx.mesh_info["_candidates_done"] = True

    # deterministic heightfield candidate for flat parts — exact footprint by
    # construction. Without candidate screening it stands alone; with
    # screening it competes against the generated candidates below.
    hf = _try_heightfield_candidate(ctx, iteration, spec)
    if hf is not None and (n <= 1 or ctx.mesh is None):
        hf_rel, hf_score = hf
        ctx.mesh_info["_heightfield_rel"] = hf_rel
        await ctx.emit(
            "meshgen",
            iteration,
            f"heightfield candidate scored {hf_score} — using deterministic flat-part mesh",
        )
        return None
    if n <= 1 or ctx.mesh is None:
        return None
    from kiara.config import get_settings

    if get_settings().agent_job_timeout_s - (time.monotonic() - t0) < 1800:
        return None
    outline = _measured_plan_outline(ctx)
    ref_mask, _ = _ref_side_mask(ctx, iteration, spec)
    base = _base_seed(ctx)
    adapters = await _candidate_adapters(ctx, n)
    scores: dict[tuple[int, str | None], float] = {}
    model_list = [adapters[i % len(adapters)] if adapters else None for i in range(n)]
    if len(set(model_list)) == 1:
        # same adapter for all candidates — submit one batch job so the model
        # loads once instead of per seed (~30-60s saved per extra candidate)
        model0 = model_list[0]
        cand_steps = min(
            s.mesh_candidate_steps,
            await _adapter_default_steps(ctx, model0) or s.mesh_candidate_steps,
        )
        seeds = [base + i * 1000 for i in range(n)]
        rels = await _generate_mesh_batch(
            ctx,
            iteration,
            spec,
            plan,
            seeds=seeds,
            steps=cand_steps,
            resolution=256,  # screening only — full gen runs at settings res
            model=model0,
            tag_prefix="_cand",
        )
        for i, rel in enumerate(rels):
            if await ctx.check_cancel():
                return None
            if rel is None:
                continue
            sc = _score_candidate_mesh(rel, spec, outline, ref_mask)
            if sc is not None:
                scores[(seeds[i], model0)] = sc
    else:
        for i in range(n):
            if await ctx.check_cancel():
                return None
            seed_i = base + i * 1000
            model_i = model_list[i]
            cand_steps = min(
                s.mesh_candidate_steps,
                await _adapter_default_steps(ctx, model_i) or s.mesh_candidate_steps,
            )
            rel = await _generate_mesh(
                ctx,
                iteration,
                spec,
                plan,
                seed=seed_i,
                steps=cand_steps,
                resolution=256,  # screening only — full gen runs at settings res
                tag=f"_cand{i}",
                model=model_i,
            )
            if rel is None:
                continue
            gen = ctx.mesh_info.get("generation") or {}
            ctx.mesh_info.setdefault("_cand_meta", {})[_cand_key(seed_i, model_i)] = {
                "rel": rel,
                "seed": seed_i,
                "model": gen.get("model") or model_i,
                "steps": gen.get("steps"),
                "resolution": gen.get("resolution"),
            }
            sc = _score_candidate_mesh(rel, spec, outline, ref_mask)
            if sc is not None:
                scores[(seed_i, model_i)] = sc
    # weak screening round: one extra roll on a fresh seed before giving up —
    # a single bad candidate must not doom the whole generation. Only when the
    # best roll is within reach of viable: at ~0 all candidates are garbage and
    # another roll on the same inputs just burns budget (hopeless gate below).
    if scores and _CANDIDATE_VIABLE_SCORE * 0.6 <= max(scores.values()) < 0.5 and len(scores) < 4:
        if await ctx.check_cancel():
            return None
        seed_i = base + 2000
        model_i = adapters[(len(scores)) % len(adapters)] if adapters else None
        await ctx.emit("meshgen", iteration, "weak candidates — drawing one extra roll")
        rel = await _generate_mesh(
            ctx,
            iteration,
            spec,
            plan,
            seed=seed_i,
            steps=s.mesh_candidate_steps,
            resolution=256,
            tag="_cand_extra",
            model=model_i,
        )
        if rel is not None:
            gen = ctx.mesh_info.get("generation") or {}
            ctx.mesh_info.setdefault("_cand_meta", {})[_cand_key(seed_i, model_i)] = {
                "rel": rel,
                "seed": seed_i,
                "model": gen.get("model") or model_i,
                "steps": gen.get("steps"),
                "resolution": gen.get("resolution"),
            }
            sc = _score_candidate_mesh(rel, spec, outline, ref_mask)
            if sc is not None:
                scores[(seed_i, model_i)] = sc
    return await _select_candidate(ctx, iteration, scores, hf)
def _canonical_up_axis(ctx: AgentCtx) -> str | None:
    model = str(
        (ctx.mesh_info.get("generation") or {}).get("model")
        or ctx.settings.mesh_model
        or ctx.runtime_settings.mesh_model
        or ""
    ).lower()
    for needle, axis in _ADAPTER_CANONICAL_UP.items():
        if needle in model:
            return axis
    return None
def _dedupe_near_identical(ctx: AgentCtx, ordered: list[Any]) -> list[Any]:
    """Drop near-duplicate conditioning images (same pose uploaded twice).
    A duplicate view adds nothing but wastes a conditioning slot — the
    generator treats it as a different view. aHash Hamming < 6 = dup."""
    if len(ordered) <= 1:
        return ordered
    seen: list[int] = []
    out: list[Any] = []
    dropped: list[str] = []
    for a in ordered:
        if getattr(a, "role_hint", "") == "synth_view":
            # synthesised side views fill labelled slots — a similar-looking
            # hero must never dedupe them away
            out.append(a)
            continue
        h = _ahash(a.path)
        if h is not None and any(bin(h ^ h2).count("1") < 6 for h2 in seen):
            dropped.append(a.asset_id)
            continue
        if h is not None:
            seen.append(h)
        out.append(a)
    if dropped:
        log.info("mesh.dedupe_dropped", assets=dropped)
    return out
async def _ensure_synth_views(
    ctx: AgentCtx, iteration: int, spec: PartSpec | None, front_asset: Any
) -> list[Any]:
    """Single-image non-flat part: synthesise up to two orthographic side
    views from the hero image so multiview adapters get left/right
    conditioning instead of guessing depth from one perspective shot.

    `front_asset` is the image the generator will actually use as its front
    view — an upload or a bootstrap concept image (the bootstrap hero had no
    conditioning at all: hunyuan reconstructed a solid channel instead of the
    hollow bail). Runs once per run (`_synth_views_done`); the generated
    assets are marked (`role_hint="synth_view"`, `synth:` caption,
    `_synth_view_roles` map) so reference-mask/carving/scoring paths never
    treat them as ground truth — they are generator conditioning only. Any
    failure degrades to the unchanged single-image path. Returns the new
    synth assets (left/right order)."""
    if ctx.mesh_info.get("_synth_views_done"):
        return []
    ctx.mesh_info["_synth_views_done"] = True
    if ctx.image is None or not ctx.image_available or spec is None or front_asset is None:
        return []
    if getattr(front_asset, "role_hint", "") == "synth_view":
        return []
    dims = (spec.length_x.value, spec.width_y.value, spec.height_z.value)
    if _is_flat_spec_dims(dims):
        return []  # flat parts get the measured-outline 'back' synth instead
    try:
        from kiara.agent.concept_image import generate_concept_images

        out = await generate_concept_images(
            ctx,
            "same object, orthographic side view, neutral grey background, no text",
            ref_asset=front_asset,
            count=2,
            source="synth_side_view",
            # conditioning views, not heroes — a real side view can't show the
            # whole feature list, so full-spec verification rejected valid
            # output and burned retries (observed: both synth attempts
            # verdicted major on "missing tabs/holes" for a bail handle)
            spec=None,
            # mflux edit defaults to 28 steps/image — ~19min per view on MLX;
            # 14 steps keeps both views under the image timeout budget
            steps=14,
            # persisted into gen_params so later versions reload the assets
            # with role_hint intact — runtime-only marking meant each fresh
            # run re-gated them as hero candidates and regenerated both views
            role_hint="synth_view",
        )
    except Exception as e:
        log.warning("meshgen.synth_views_failed", error=str(e)[:200])
        await ctx.emit(
            "meshgen",
            iteration,
            f"side-view synthesis failed: {e} — continuing with the single view",
        )
        return []
    roles: dict[str, str] = {}
    for a, slot in zip(out, ("left", "right"), strict=False):
        a.role_hint = "synth_view"
        a.caption = f"synth:{slot} — {a.caption}"
        roles[a.asset_id] = slot
    if roles:
        # persist the slot mapping — the runtime caption/role_hint die with
        # this run; a later version reloads assets from the DB and must be
        # able to rebuild _synth_view_roles instead of regenerating the views
        try:
            import uuid

            from kiara.db.models import Asset
            from kiara.db.session import session_scope

            async with session_scope() as db:
                for aid, slot in roles.items():
                    row = await db.get(Asset, uuid.UUID(aid))
                    if row is None:
                        continue
                    row.caption = f"synth:{slot} — {row.caption}"
                    row.gen_params = {**(row.gen_params or {}), "synth_slot": slot}
                await db.commit()
        except Exception as e:
            log.warning("meshgen.synth_slot_persist_failed", error=str(e)[:200])
        ctx.mesh_info["_synth_view_roles"] = roles
        await ctx.emit(
            "meshgen",
            iteration,
            f"generated {len(roles)} synthetic side view(s) for multiview conditioning",
        )
    return list(out)
async def _fit_mesh_to_stated(
    ctx: AgentCtx,
    it: int,
    mesh_rel: str,
    spec: PartSpec,
    post: MeshPostParams,
    exec_result: dict[str, Any],
    tol_pct: float,
    tol_mm: float,
    art_attempt: Any,
    *,
    make_solid: bool = True,
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
        fit_result = await _process_mesh(
            ctx,
            it,
            mesh_rel,
            spec,
            post_fit,
            make_solid=make_solid,
            subdir=f"a{next(art_attempt)}",
        )
        if not fit_result.get("ok"):
            return None
        # degenerate-fit guard: ops that collapse the body to a sliver produce
        # a crust, not a part. Compare post-op volume to the PRE-OP volume —
        # an absolute bbox fraction rejects open-frame parts whose envelope
        # is mostly void (observed: a correct bail-handle mesh at ~7% fill,
        # dims fitted, rejected by the old 8%-of-bbox rule).
        ana = fit_result.get("analysis") or {}
        vol = ana.get("volume")
        prev_vol = (cur_result.get("analysis") or {}).get("volume")
        if vol and prev_vol and vol < 0.5 * prev_vol:
            return None
        cur_result, cur_post = fit_result, post_fit
        if not compare_dims(spec, ana, tol_pct, tol_mm):
            return cur_result, cur_post
    return None
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
def _gate_generated_images(
    ordered: list[Any], spec: PartSpec | None, logged: set[str] | None = None
) -> list[Any]:
    """Drop generated reference images that fail the deterministic proportion
    gate at selection time.

    Generated heroes are verified at creation — but assets PERSIST across
    versions, so a hero generated before a spec/gate change (or under a
    different spec) reaches the generator unchecked (observed: tall-arch hero
    for a 4:1-flat handle spec feeding hunyuan in every iteration). A dropped
    hero just falls through to the bootstrap generator. User-supplied images
    are never gated — the user's photo is ground truth.

    ``logged`` is a caller-held dedupe set of already-reported asset ids —
    the gate runs per selection pass (hero list, combined list), so the same
    drop is only logged once per job.
    """
    if spec is None:
        return ordered
    from kiara.agent.image_check import aspect_mismatch

    kept, dropped = [], []
    for a in ordered:
        if getattr(a, "role_hint", "") == "synth_view":
            # conditioning side views, not hero candidates — their silhouette
            # legitimately differs from the front-view aspect the gate checks
            kept.append(a)
        elif a.kind == "generated_image" and getattr(a, "verify_severity", "") == "major":
            # the vision check certified this image as the wrong part — feeding
            # it to the generator builds a certified-wrong mesh (observed:
            # U-channel hero → 26 mm mesh for a 60 mm spec). The drop falls
            # through to the concept bootstrap, same as the aspect gate.
            dropped.append(a.asset_id)
        elif a.kind == "generated_image" and aspect_mismatch(a.path, spec):
            dropped.append(a.asset_id)
        else:
            kept.append(a)
    if dropped:
        new_dropped = [d for d in dropped if logged is None or d not in logged]
        if logged is not None:
            logged.update(dropped)
        if new_dropped:
            log.info("mesh.aspect_gate_dropped", assets=new_dropped)
    return kept
def _gen_view_roles(
    ctx: AgentCtx,
    ordered: list[Any],
    synth_roles: dict[str, str],
    iteration: int,
    spec: PartSpec | None,
) -> tuple[list[Any], list[str]]:
    """Map ordered source images to generator view roles (see
    `_mesh_source_images` docstring for the mapping rules)."""
    intake_roles: dict[str, str] = {}
    flat = False
    if spec is not None:
        intake_roles = {r.asset_id: r.role for r in spec.image_roles if r.asset_id}
        dims = (spec.length_x.value, spec.width_y.value, spec.height_z.value)
        flat = _is_flat_spec_dims(dims)
    if flat:
        # The measure step classifies plan views deterministically (silhouette
        # aspect vs. stated dims). It beats intake labels — models keep calling
        # a face-up photo of a flat part "front", which feeds the generator a
        # plan view as its hero slot and produces a standing object.
        for aid, m in (ctx.measurements or {}).items():
            if m.get("role") in ("top", "bottom"):
                intake_roles[aid] = str(m["role"])
    if not intake_roles:
        roles = ["front"] + ["other"] * (len(ordered) - 1)
    else:
        ordered, roles = _map_gen_roles(ordered, intake_roles, flat)
    if flat and "back" not in roles:
        outline = _measured_plan_outline(ctx)
        if outline:
            synth = _synth_back_view(ctx, iteration, outline)
            if synth is not None:
                ordered = [*ordered, synth]
                roles = [*roles, "back"]
    if synth_roles:
        roles = _assign_synth_roles(ordered, roles, synth_roles)
    return ordered, roles
async def _generate_mesh(  # noqa: C901
    ctx: AgentCtx,
    iteration: int,
    spec: PartSpec,
    plan: Any,
    *,
    seed: int | None = None,
    steps: int | None = None,
    tag: str = "",
    t0: float | None = None,
    model: str | None = None,
    resolution: int | None = None,
) -> str | None:
    """MESH_GENERATE step. Returns the relpath of the raw GLB or None."""
    if ctx.mesh is None:
        ctx.mesh_info["gen_error"] = "mesh server not configured (set mesh_base_url in Settings)"
        return None

    # best-of-N: pick the strongest seed once per job before the full run
    if seed is None and t0 is not None:
        seed = await _candidate_seed(ctx, iteration, spec, plan, t0)
        hf_rel = ctx.mesh_info.get("_heightfield_rel")
        if hf_rel:
            await ctx.emit("meshgen", iteration, "using heightfield candidate mesh")
            return str(hf_rel)
        if ctx.mesh_info.get("_mesh_hopeless") and seed is None:
            return None

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
        # text-only job: image->3D still needs a reference — synthesise a
        # product-photo concept image once per job when an image adapter is up
        boot = await _bootstrap_concept_images(ctx, iteration, spec, plan)
        if boot:
            images, roles = boot, ["front"] + ["other"] * (len(boot) - 1)
    if not images:
        ctx.mesh_info["gen_error"] = "no image input for mesh generation"
        return None
    if len(images) == 1:
        synth = await _ensure_synth_views(ctx, iteration, spec, images[0])
        images += synth
        roles += ["left", "right"][: len(synth)]
    await ctx.emit("meshgen", iteration, f"views: {', '.join(roles[: len(images)])}")

    base_seed = _base_seed(ctx)
    if seed is None:
        seed = base_seed + iteration - 1
    model, images, roles = await _arrange_gen_images(
        ctx, iteration, images, roles, spec, prefer=model or ctx.mesh_info.get("_candidate_model")
    )
    control_type, control_input = _omni_control(model, spec)
    subject = (
        ctx.mesh_info.get("_regen_hint")
        or (plan.mesh_subject if plan else "")
        or spec.name
        or ctx.prompt[:80]
    )

    # image preprocessing: isolated-object square crops generate cleaner meshes
    images = _prepped_gen_images(ctx, it_dir, images)

    # best-of-N winner reuse: when the winning candidate was already generated
    # at >= the winner's params (same adapter, seed, images), its GLB IS the
    # winner output — regenerating it costs a full model load + diffusion
    # (~4min on MPS) for an identical result. Consumed once: a later regen
    # with the same seed must actually generate, not replay the reject.
    cand = ctx.mesh_info.get("_candidate_winner")
    if cand and not cand.get("_used") and cand.get("rel"):
        eff_steps = steps if steps is not None else ctx.settings.mesh_steps
        eff_res = resolution if resolution is not None else ctx.settings.mesh_resolution
        c_steps, c_res = cand.get("steps"), cand.get("resolution")
        steps_ok = eff_steps is None or (c_steps is not None and eff_steps <= c_steps)
        res_ok = eff_res is None or (c_res is not None and eff_res <= c_res)
        if (
            cand.get("seed") == seed
            and cand.get("model") == model
            and steps_ok
            and res_ok
            and storage.resolve(cand["rel"]).exists()
        ):
            src = storage.resolve(cand["rel"])
            if src.resolve() != out_path.resolve():
                it_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_path)
            cand["_used"] = True
            ctx.mesh_info["generation"] = {
                # the adapter the candidate was ACTUALLY generated with —
                # `model` may still be the unresolved "auto" preference
                "model": cand.get("model") or model,
                "seed": seed,
                "cached": True,
                "candidate_reuse": True,
                "mesh_rel": rel,
                "images": [a.asset_id for a in images],
            }
            await ctx.emit(
                "meshgen",
                iteration,
                "reusing winning candidate mesh (same seed+adapter+params)",
            )
            return rel

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
        resolution=resolution,
        key_extra=f"ctl:{control_type}:{control_input}" if control_type else "",
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
        resolution=resolution if resolution is not None else ctx.settings.mesh_resolution,
        out_path=out_path,
        job_id=ctx.job_id,
        control_type=control_type,
        control_input=control_input,
    )

    on_progress = _mesh_progress(ctx, iteration)

    result = None
    # transport-level failures (server restarting, port not yet up) are
    # transient — retry briefly instead of burning the iteration slot;
    # MeshGenError (server-side job failure) stays single-shot
    for gen_attempt in range(3):
        try:
            result = await ctx.mesh.generate(req, on_progress, should_abort=ctx.check_cancel)
            break
        except MeshGenError as e:
            await ctx.emit("meshgen", iteration, f"generation failed: {e}")
            log.warning("mesh.generate_failed", iteration=iteration, error=str(e))
            ctx.mesh_info["gen_error"] = str(e)
            return None
        except Exception as e:
            import httpx

            if isinstance(e, httpx.TransportError) and gen_attempt < 2:
                await ctx.emit(
                    "meshgen",
                    iteration,
                    f"mesh server unreachable — retrying ({gen_attempt + 1}/2)",
                )
                log.warning(
                    "mesh.generate_transport_retry",
                    iteration=iteration,
                    attempt=gen_attempt + 1,
                    error=str(e)[:160],
                )
                await asyncio.sleep(5 * (gen_attempt + 1))
                continue
            log.warning("mesh.generate_crashed", iteration=iteration, error=str(e))
            ctx.mesh_info["gen_error"] = str(e)
            return None

    if key:
        try:
            from kiara.agent import mesh_cache

            mesh_cache.store(key, result.mesh_path, meta={"model": result.model})
        except Exception as e:
            log.warning("mesh.cache_store_failed", error=str(e)[:160])

    gen = result.model_dump(mode="json")
    gen["mesh_rel"] = storage.relpath(result.mesh_path)
    gen.pop("mesh_path", None)
    gen["images"] = [a.asset_id for a in images]
    ctx.mesh_info["generation"] = gen
    return rel
async def _generate_mesh_batch(  # noqa: C901
    ctx: AgentCtx,
    iteration: int,
    spec: PartSpec,
    plan: Any,
    *,
    seeds: list[int],
    steps: int | None,
    resolution: int | None,
    model: str | None,
    tag_prefix: str,
) -> list[str | None]:
    """Screening variants in ONE server job — the model loads a single time.

    Mirrors _generate_mesh's setup (image arrange/prep, subject, mesh cache)
    but submits all seeds as batch variants. Returns relpaths aligned with
    `seeds` (None for failed variants)."""
    n = len(seeds)
    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration)
    it_dir.mkdir(parents=True, exist_ok=True)
    images, roles = _mesh_source_images(ctx, iteration, spec)
    if not images:
        boot = await _bootstrap_concept_images(ctx, iteration, spec, plan)
        if boot:
            images, roles = boot, ["front"] + ["other"] * (len(boot) - 1)
    if not images:
        ctx.mesh_info["gen_error"] = "no image input for mesh generation"
        return [None] * n
    if len(images) == 1:
        synth = await _ensure_synth_views(ctx, iteration, spec, images[0])
        images += synth
        roles += ["left", "right"][: len(synth)]
    model_r, images, roles = await _arrange_gen_images(
        ctx, iteration, images, roles, spec, prefer=model
    )
    control_type, control_input = _omni_control(model_r, spec)
    # screening at the adapter's native step count — distilled turbo models
    # are built for ~5 steps; running them at 15 just burns wall-clock
    if steps is not None:
        dflt = await _adapter_default_steps(ctx, model_r)
        if dflt:
            steps = min(steps, dflt)
    images = _prepped_gen_images(ctx, it_dir, images)
    subject = (
        ctx.mesh_info.get("_regen_hint")
        or (plan.mesh_subject if plan else "")
        or spec.name
        or ctx.prompt[:80]
    )
    await ctx.emit("meshgen", iteration, f"views: {', '.join(roles[: len(images)])}")

    from kiara.agent import mesh_cache

    rels: list[str | None] = [None] * n
    pend: list[int] = []
    variants: list[MeshGenVariant] = []
    keys: dict[int, str] = {}
    meta = ctx.mesh_info.setdefault("_cand_meta", {})
    produced_model: str | None = None
    for i, seed in enumerate(seeds):
        out_path = it_dir / f"mesh_raw{tag_prefix}{i}.glb"
        rel = storage.relpath(out_path)
        key = ""
        if ctx.runtime_settings.mesh_cache_enabled:
            key = mesh_cache.cache_key(
                [a.path for a in images],
                roles,
                model=model_r,
                seed=seed,
                steps=steps,
                guidance=ctx.settings.mesh_guidance,
                resolution=resolution,
                subject=(
                    f"{subject}|ctl:{control_type}:{control_input}" if control_type else subject
                ),
            )
            hit = mesh_cache.lookup(key)
            if hit:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(hit, out_path)
                rels[i] = rel
                cached_model = mesh_cache.lookup_meta(key).get("model")
                produced_model = produced_model or cached_model or None
                meta[_cand_key(seed, model)] = {
                    "rel": rel,
                    "seed": seed,
                    # the adapter that produced the cached GLB when known —
                    # model_r may still be "auto"
                    "model": cached_model or model_r,
                    "steps": None,
                    "resolution": None,
                    "cached": True,
                }
                continue
        keys[i] = key
        pend.append(i)
        variants.append(
            MeshGenVariant(
                seed=seed,
                steps=steps,
                resolution=resolution,
                guidance=ctx.settings.mesh_guidance,
                out_path=out_path,
            )
        )

    if pend:
        req = MeshGenRequest(
            images=[a.path for a in images],
            image_roles=roles[: len(images)],
            subject=subject,
            model=model_r,
            seed=seeds[pend[0]],
            guidance=ctx.settings.mesh_guidance,
            out_path=variants[0].out_path,
            job_id=ctx.job_id,
            control_type=control_type,
            control_input=control_input,
        )

        on_progress = _mesh_progress(ctx, iteration)

        try:
            results = await ctx.mesh.generate_batch(
                req, variants, on_progress, should_abort=ctx.check_cancel
            )
        except MeshGenError as e:
            await ctx.emit("meshgen", iteration, f"batch generation failed: {e}")
            log.warning("mesh.batch_failed", iteration=iteration, error=str(e))
            ctx.mesh_info["gen_error"] = str(e)
            results = []
        except Exception as e:
            log.warning("mesh.batch_crashed", iteration=iteration, error=str(e))
            ctx.mesh_info["gen_error"] = str(e)
            results = []
        for j, res in zip(pend, results, strict=False):
            if res is None:
                continue
            rels[j] = storage.relpath(res.mesh_path)
            produced_model = res.model
            meta[_cand_key(seeds[j], model)] = {
                "rel": rels[j],
                "seed": seeds[j],
                "model": res.model,
                "steps": res.steps,
                "resolution": res.resolution,
            }
            if keys.get(j):
                try:
                    mesh_cache.store(keys[j], res.mesh_path, meta={"model": res.model})
                except Exception as e:
                    log.warning("mesh.cache_store_failed", error=str(e)[:160])
    ctx.mesh_info["generation"] = {
        # adapter that actually produced the batch — `model_r` may be the
        # unresolved "auto" preference while res.model is server-reported
        "model": produced_model or model_r,
        "seeds": seeds,
        "batch": True,
        "images": [a.asset_id for a in images],
    }
    return rels
def _is_flat_spec_dims(dims: tuple[float, float, float]) -> bool:
    """Flat = Z dramatically thinner than BOTH horizontal dims (plate-like).
    Comparing against max() misclassifies any elongated part — a 130x32x32
    grip is not flat just because it is long."""
    return dims[2] > 0 and dims[2] < 0.3 * min(dims[0], dims[1])
def _map_gen_roles(
    ordered: list[Any], intake_roles: dict[str, str], flat: bool
) -> tuple[list[Any], list[str]]:
    """Translate intake image_roles into generator view roles."""

    def _gen_role(a: Any) -> str:
        # explicit user/agent marking wins over intake labels
        if getattr(a, "role_hint", "") == "sketch":
            return "scribble"
        role = intake_roles.get(a.asset_id, "other")
        if role == "top":
            return "front" if flat else "other"
        if role == "bottom":
            return "back" if flat else "other"
        if role in ("front", "back", "left", "right"):
            return role
        if role == "sketch":
            return "scribble"
        return "other"

    seen: set[str] = set()
    roles: list[str] = []
    for a in ordered:
        r = _gen_role(a)
        if r in seen:  # a second image for one slot degrades to "other"
            r = "other"
        seen.add(r)
        roles.append(r)
    # hero must be the generator's front view — unless the lead image is a
    # marked scribble, which stays the conditioning slot for *-scribble
    # adapters (ordering is enforced again in _arrange_gen_images)
    front_i = next((i for i, r in enumerate(roles) if r == "front"), None)
    if front_i is not None and front_i != 0:
        ordered.insert(0, ordered.pop(front_i))
        roles.insert(0, roles.pop(front_i))
    if roles and "front" not in roles and roles[0] != "scribble":
        roles[0] = "front"
    return ordered, roles
def _measured_plan_outline(ctx: AgentCtx) -> list[list[float]] | None:
    """Best measured plan-view outline for orientation, footprint scoring
    and clipping.

    Only AUTHORITATIVE measurements feed silhouette reconstruction — an
    aspect-ratio fallback hit on a perspective photo must not become ground
    truth. When the planner explicitly asked for `use_image_profile` the best
    outline is still used (that is what the flag is for)."""
    want_profile = bool(getattr(ctx.plan, "use_image_profile", False))
    best: tuple[float, list[list[float]]] | None = None
    for m in (ctx.measurements or {}).values():
        conf = float(m.get("confidence") or 0)
        outline = m.get("outline_mm") or []
        if not want_profile and not m.get("authoritative"):
            continue
        if conf >= 0.5 and len(outline) >= 3 and (best is None or conf > best[0]):
            best = (conf, outline)
    return best[1] if best else None
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
    resolution: int | None = None,
    key_extra: str = "",
) -> tuple[str, bool]:
    """Return (cache_key, hit); on a hit copy the cached GLB to out_path
    and fill ctx.mesh_info['generation'] so the caller can return early."""
    from kiara.agent import mesh_cache

    if not ctx.runtime_settings.mesh_cache_enabled:
        return "", False
    key = mesh_cache.cache_key(
        [a.path for a in images],
        roles,
        model=model,
        seed=seed,
        steps=steps if steps is not None else ctx.settings.mesh_steps,
        guidance=ctx.settings.mesh_guidance,
        resolution=resolution if resolution is not None else ctx.settings.mesh_resolution,
        # key_extra folds conditioning (Omni bbox) into the cache identity —
        # same images+params under a different control must not hit
        subject=f"{subject}|{key_extra}" if key_extra else subject,
    )
    cached = mesh_cache.lookup(key)
    if cached is None:
        return key, False
    await ctx.emit("meshgen", iteration, "mesh from cache")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, out_path)
    ctx.mesh_info["generation"] = {
        # the adapter that produced the cached GLB when the sidecar knows —
        # `model` may still be the unresolved "auto" preference
        "model": mesh_cache.lookup_meta(key).get("model") or model,
        "seed": seed,
        "cached": True,
        "mesh_rel": rel,
        "images": [a.asset_id for a in images],
    }
    return key, True
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
    subdir: str = "",
    ref_mask_rel: str | None = None,
) -> dict[str, Any]:
    return {
        "job_id": ctx.job_id,
        "project_id": str(ctx.project_id),
        "version_id": str(ctx.version_id),
        "iteration": iteration,
        "subdir": subdir,
        "mode": mode,
        "mesh_rel": mesh_rel,
        "ref_mask_rel": ref_mask_rel,
        "post": (post or MeshPostParams()).model_dump(),
        "anchors": ctx.anchors,
        "spec": spec.model_dump() if spec else None,
        "settings": ctx.settings.model_dump(),
        "runtime_settings": ctx.runtime_settings.executor_values(),
        "make_solid": make_solid,
        "full_exports": full_exports,
        "name": "model",
    }
def _mesh_progress(ctx: AgentCtx, iteration: int):
    """Shared emit-closure for meshgen progress — both the batch path and
    the single-generate path reported identical nested closures."""

    async def on_progress(msg: str, p: float) -> None:
        await ctx.emit(
            "meshgen",
            iteration,
            f"{msg} ({int(p * 100)}%)",
            sub_progress=p,
            sub_label=msg,
        )

    return on_progress
def _mesh_source_images(
    ctx: AgentCtx, iteration: int, spec: PartSpec | None = None
) -> tuple[list[Any], list[str]]:
    """Pick input images for the generator: planner's mesh_source_images order,
    hero first; later iterations rotate the hero among image assets.

    Generator roles come from the intake's image_roles when available — a top
    view of a flat part feeds the generator's "front" slot (its hero view),
    side/back views map to their namesake roles, everything else is "other".
    A flat part with no real bottom view additionally gets a filled-silhouette
    'back' rendered from the measured outline.
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
    ordered = _dedupe_near_identical(ctx, ordered)
    # the gate runs for every selection pass — a ctx-level set keeps each
    # dropped asset logged once per job instead of once per pass
    gate_logged = getattr(ctx, "_aspect_gate_logged", None)
    if gate_logged is None:
        gate_logged = set()
        ctx._aspect_gate_logged = gate_logged
    ordered = _gate_generated_images(ordered, spec, gate_logged)
    if not ordered:
        # every reference was gated/deduped away — callers bootstrap a fresh
        # concept image on the empty list; mapping roles over nothing crashes
        return [], []
    synth_roles: dict[str, str] = ctx.mesh_info.get("_synth_view_roles") or {}
    if not synth_roles:
        # reload path: synth views persisted with role_hint + "synth:<slot>"
        # captions survive into later versions — recover their view slots so
        # they land on left/right again instead of being re-generated
        for a in ordered:
            if getattr(a, "role_hint", "") == "synth_view":
                m = re.match(r"synth:(left|right)", a.caption or "")
                if m:
                    synth_roles[a.asset_id] = m.group(1)
        if synth_roles:
            ctx.mesh_info["_synth_view_roles"] = synth_roles
    if iteration > 1 and len(ordered) > 1:
        ordered = _rotate_hero(ordered, synth_roles)
    return _gen_view_roles(ctx, ordered, synth_roles, iteration, spec)
async def _none() -> None:
    return None
def _omni_control(model: str | None, spec: PartSpec | None) -> tuple[str | None, Any]:
    """Hunyuan3D-Omni bbox conditioning: stated dims as `control_input` —
    the adapter normalises proportions internally so plain mm values work.
    Returns (None, None) for other adapters or underspecified dims."""
    if not model or "omni" not in model.lower() or spec is None:
        return None, None
    dims = [spec.length_x.value, spec.width_y.value, spec.height_z.value]
    if not all(d and float(d) > 0 for d in dims):
        return None, None
    return "bbox", [round(float(d), 3) for d in dims]
def _only_dim_issues(dim_issues: list[str], issues: list[dict[str, Any]]) -> bool:
    """True when dimension mismatches are the ONLY major problem — the
    deterministic fit already ran, so an LLM refine session on pure dims just
    wrecks the shape (observed: knob squashed to a capsule for a 4% dim fix)."""
    return bool(dim_issues) and not any(
        i.get("severity") == "major" and i.get("source") != "dims" for i in issues
    )
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

            from kiara.cad.mesh_ops import orient_by_plan_view

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

    table = None
    rank_fn = None
    if ctx.spec:
        s = ctx.runtime_settings
        ref_mask, _ = _ref_side_mask(ctx, iteration, ctx.spec)
        table, rank_fn = _orientation_ranking(
            mesh_rel,
            ctx.spec,
            s.agent_dim_tolerance_pct,
            s.agent_dim_tolerance_mm,
            canonical_up=_canonical_up_axis(ctx),
            ref_mask=ref_mask,
        )
        if table and len(table) >= 2:
            # decisive deterministic ranking: no stated-dim violation and a
            # clear margin to the runner-up — the LLM orient call adds cost
            # but cannot improve on this pick, so skip it entirely
            (best_key, best_choice), (second_key, _) = table[0], table[1]
            margin = second_key[1] - best_key[1]
            if best_key[0] == 0 and margin >= 0.15:
                ctx.mesh_orientation = dict(best_choice)
                ctx.mesh_info["orientation_source"] = "deterministic"
                await ctx.emit(
                    "orient",
                    iteration,
                    f"orientation from deterministic ranking "
                    f"{best_choice['up']}/{best_choice['front']} (margin {margin:.2f})",
                )
                return ctx.mesh_orientation

    choice = await mesh_orient_step.run(ctx, iteration, mesh_rel)
    # deterministic sanity: the LLM can orient a flat part standing on its
    # edge (up=+y on a handle scale → z-extent 40x the stated thickness) —
    # that fails the dimension gate downstream. Rank all up/front pairs by
    # stated-dim violations (then candidate score) and override the pick only
    # on a DECISIVE advantage: fewer dim violations, or equal violations with
    # a clear score margin (the same 0.15 bar as the skip-decision above).
    # Raw-mesh scores differ by <0.02 between plausible orientations of
    # near-cubic parts — flipping the LLM pick on that noise level put a knob
    # on its side (observed: +x/+y over a correct +y/+x at Δscore 0.01).
    if table and rank_fn is not None:
        llm_bad = rank_fn(choice)
        det_bad, det = table[0]
        if _orientation_decisive(llm_bad, det_bad):
            if det_bad[0] < llm_bad[0]:
                reason = (
                    f"violates {llm_bad[0]} stated dim(s) "
                    f"({det['up']}/{det['front']} has {det_bad[0]})"
                )
            else:
                reason = f"scores low ({-llm_bad[1]:.2f} vs {-det_bad[1]:.2f})"
            await ctx.emit(
                "orient",
                iteration,
                f"orientation {choice.get('up')}/{choice.get('front')} {reason} "
                f"— using {det['up']}/{det['front']}",
            )
            choice = {**det, **({"mirror_x": True} if choice.get("mirror_x") else {})}
            # the deterministic ranking chose this orientation — lock it in
            # refine sessions the same way the skip-LLM path does
            ctx.mesh_info["orientation_source"] = "deterministic"
    ctx.mesh_orientation = choice
    ctx.mesh_info["orientation_source"] = ctx.mesh_info.get("orientation_source") or "llm"
    return ctx.mesh_orientation
def _orientation_decisive(llm_bad: tuple[int, float], det_bad: tuple[int, float]) -> bool:
    """Whether the deterministic top pick may override the LLM orientation.

    Keys are (stated-dim violations, -score). Decisive = strictly fewer
    violations, or equal violations with a score margin >= 0.15 — the same
    bar the early-skip uses. Anything below that is scoring noise and the
    LLM's semantic judgement (or the canonical-up prior) stands.
    """
    return det_bad[0] < llm_bad[0] or (det_bad[0] == llm_bad[0] and llm_bad[1] - det_bad[1] >= 0.15)
def _orientation_ranking(
    mesh_rel: str,
    spec: PartSpec,
    tol_pct: float,
    tol_mm: float,
    *,
    canonical_up: str | None = None,
    ref_mask: Any = None,
) -> tuple[
    list[tuple[tuple[int, float], dict[str, str]]] | None,
    Any,
]:
    """Score every (up, front) orientation: (#stated-dim violations, -score).

    The score folds in a support term (how stably the part rests — a knob on
    its skirt scores higher than the same knob lying on its side) and a small
    canonical-up prior (adapters like Hunyuan emit Y-up parts, so a Y-up pick
    wins ties). Returns (table, key_fn): `table` is the (up, front) candidates
    sorted best first; `key_fn(choice)` scores an arbitrary LLM pick with the
    same ruler so the caller can override a physically impossible choice (e.g.
    a flat scale standing on its edge). Both are None when the mesh cannot be
    loaded."""
    try:
        import math

        import trimesh

        from kiara.agent.mesh_score import score_raw_mesh, support_score
        from kiara.cad.mesh_ops import (
            decimate,
            footprint_yaw_deg,
            orient,
            rest_on_xy_center,
            scale_to_spec,
        )

        m = trimesh.load(storage.resolve(mesh_rel), force="mesh")
        if not isinstance(m, trimesh.Trimesh):
            return None, None
        # raw generator meshes can be >500k faces — the loop orients+scores
        # ~30 candidates below, each copying the mesh; decimate once upfront
        m = decimate(m, 30_000)

        def _key(up: str, front: str, mirror: bool = False) -> tuple[int, float]:
            c = rest_on_xy_center(orient(m, up, front, mirror_x=mirror))
            # rank dims the way baseline_mesh will process them: a generative
            # mesh keeps the image's iso yaw, and auto_yaw straightens the
            # footprint before the dim gate — scoring the un-rotated box
            # would reject a perfectly good candidate for the wrong reason
            yaw = footprint_yaw_deg(c)
            if yaw:
                c.apply_transform(
                    trimesh.transformations.rotation_matrix(math.radians(yaw), [0, 0, 1])
                )
                c = rest_on_xy_center(c)
            c, _, _ = scale_to_spec(c, spec)
            n = len(
                compare_dims(spec, {"bbox_size": [float(v) for v in c.extents]}, tol_pct, tol_mm)
            )
            score = score_raw_mesh(c, spec, None, ref_mask=ref_mask) or 0.0
            score += 0.15 * support_score(c)
            if canonical_up and up.lstrip("+-") == canonical_up:
                score += 0.03
            return n, -score

        def _key_choice(choice: dict[str, Any]) -> tuple[int, float]:
            return _key(
                str(choice.get("up") or "+z"),
                str(choice.get("front") or "+y"),
                bool(choice.get("mirror_x")),
            )

        axes = ("+x", "-x", "+y", "-y", "+z", "-z")
        table: list[tuple[tuple[int, float], dict[str, str]]] = []
        for up in axes:
            for front in axes:
                if up[-1] == front[-1]:
                    continue
                table.append((_key(up, front), {"up": up, "front": front}))
        table.sort(key=lambda t: t[0])
        return table, _key_choice
    except Exception as e:
        log.warning("orient.sanity_failed", error=str(e)[:200])
        return None, None
async def _parametrize_best(
    ctx: AgentCtx,
    best: Candidate,
    spec: PartSpec,
    *,
    tol_pct: float,
    tol_mm: float,
) -> dict[str, Any] | None:
    """Best-effort parametric reconstruction of the winning mesh candidate.

    Dispatches the ``parametrize_mesh`` executor task (replay final mesh ->
    reconstruct -> sandbox execute -> chamfer check) and returns the accepted
    result dict. Returns None on any failure — the reconstruction only
    enriches the outcome with editable code + schema, it never changes it."""
    mesh_meta = best.exec_result.get("mesh") or {}
    payload = _mesh_payload(
        ctx,
        best.iteration,
        mode="final",
        post=MeshPostParams(**(mesh_meta.get("post") or best.mesh_post or {})),
        spec=spec,
        subdir="parametrize",
    )
    # the executor prefers the stored final STL — the exact artifact the
    # candidate was scored on; raw GLB + post stay as the replay fallback
    payload["stl_rel"] = mesh_meta.get("stl_relpath")
    payload["analysis"] = best.exec_result.get("analysis")
    if ctx.project_id is not None:
        raw = storage.iteration_dir(ctx.project_id, ctx.version_id, best.iteration) / "mesh_raw.glb"
        if raw.exists():
            payload["mesh_rel"] = storage.relpath(raw)
    try:
        res = await ctx.executor.parametrize_mesh(payload)
    except Exception as e:
        log.warning("mesh.parametrize_failed", error=str(e)[:200])
        await ctx.emit("parametrize", best.iteration, "parametric reconstruction skipped")
        return None

    reason = ""
    if not res.get("ok"):
        reason = (
            exec_error(res)
            if res.get("timed_out")
            else str(res.get("reason") or res.get("error") or "executor failed")
        )
    else:
        chamfer_raw = res.get("chamfer_mm")
        chamfer = float(chamfer_raw) if chamfer_raw is not None else float("inf")
        bbox = (best.exec_result.get("analysis") or {}).get("bbox_size") or []
        max_dim = max((float(v) for v in bbox), default=0.0)
        if max_dim <= 0:
            max_dim = max(spec.length_x.value, spec.width_y.value, spec.height_z.value)
        # organic sections (scallops/asymmetry, mean circularity < 0.9) can't
        # be chamfer-tight as a revolve — the reconstructed parametric body is
        # a form-faithful handle, not a surface copy. Still rejects garbage
        # (a bore-less solid revolve measured 3.0mm on the grip).
        circ = ((res.get("meta") or {}).get("section_circularity_mean")) or 1.0
        limit = max(0.5, (0.005 if float(circ) >= 0.9 else 0.012) * max_dim)
        recon_analysis = (res.get("exec_result") or {}).get("analysis") or {}
        if recon_analysis.get("bbox_size"):
            dim_issues = compare_dims(spec, recon_analysis, tol_pct, tol_mm)
        else:
            dim_issues = ["no reconstruction analysis"] if stated_dim_count(spec) else []
        if chamfer > limit:
            reason = f"chamfer {chamfer:.2f}mm > {limit:.2f}mm"
        elif dim_issues:
            reason = f"dim mismatch ({dim_issues[0]})"
        else:
            await ctx.emit(
                "parametrize",
                best.iteration,
                f"parametric reconstruction ({res.get('kind')}, chamfer {chamfer:.2f}mm)",
            )
            return {
                "kind": res.get("kind"),
                "code": res.get("code"),
                "params": res.get("params") or {},
                "params_schema": res.get("params_schema") or [],
                "chamfer_mm": chamfer,
                "confidence": res.get("confidence"),
                "exec_result": res.get("exec_result") or {},
            }
    await ctx.emit("parametrize", best.iteration, f"parametric reconstruction skipped ({reason})")
    return None
def _prepped_gen_images(ctx: AgentCtx, it_dir: Path, images: list[Any]) -> list[Any]:
    """Crop/background-isolate generation inputs when mesh_prep_images is on."""
    if not ctx.runtime_settings.mesh_prep_images:
        return images
    prep_dir = it_dir / "meshgen_input"
    prepped: list[Any] = []
    for a in images:
        try:
            from kiara.vision.prep import prepare_for_meshgen

            pp = prepare_for_meshgen(a.path, prep_dir / f"{a.asset_id}.png")
            prepped.append(a if pp == a.path else replace(a, path=pp))
        except Exception as e:
            log.warning("mesh.prep_failed", asset=a.asset_id, error=str(e)[:160])
            prepped.append(a)
    return prepped
async def _process_mesh(
    ctx: AgentCtx,
    iteration: int,
    mesh_rel: str,
    spec: PartSpec,
    post: MeshPostParams,
    *,
    make_solid: bool = False,
    solid_max_faces: int | None = None,
    subdir: str = "",
    ref_mask_rel: str | None = None,
) -> dict[str, Any]:
    payload = _mesh_payload(
        ctx,
        iteration,
        mode="final",
        mesh_rel=mesh_rel,
        post=post,
        spec=spec,
        make_solid=make_solid,
        subdir=subdir,
        # the worker extracts the subject mask once per job — every
        # process_mesh call reuses it for executor-side silhouette IoU
        ref_mask_rel=ref_mask_rel or ctx.mesh_info.get("_ref_mask_rel"),
    )
    if solid_max_faces is not None:
        payload["solid_max_faces"] = solid_max_faces
    res = await ctx.executor.process_mesh(payload)
    # worker/executor image drift gate: the executor echoes the ops-semantics
    # version it replays with — a mismatch means ops written by this worker
    # may replay with different semantics (observed: worker v5 vs executor v4)
    remote_ver = (res.get("mesh") or {}).get("mesh_ops_version")
    if remote_ver is not None and int(remote_ver) != MESH_OPS_VERSION:
        log.warning(
            "mesh.ops_version_drift",
            worker=MESH_OPS_VERSION,
            executor=remote_ver,
        )
        await ctx.emit(
            "meshprocess",
            iteration,
            f"WARNING: executor mesh_ops v{remote_ver} != worker v{MESH_OPS_VERSION}"
            " — rebuild both images",
        )
    return res
def _ref_side_mask(ctx: AgentCtx, iteration: int, spec: PartSpec | None) -> tuple[Any, str | None]:
    """Subject mask of the best *side* reference image (+ its stored PNG rel).

    Only side-ish views qualify — a plan/top silhouette lives in a different
    plane (footprint IoU covers that). The mask PNG is stored once so the
    executor can reuse it for silhouette IoU on the processed mesh instead of
    re-running rembg per process_mesh call."""
    cached = ctx.mesh_info.get("_ref_mask_done")
    if cached:
        return ctx.mesh_info.get("_ref_mask"), ctx.mesh_info.get("_ref_mask_rel")
    ctx.mesh_info["_ref_mask_done"] = True
    mask, rel = None, None
    try:
        from kiara.cad.silhouette import reference_mask, save_mask

        images, roles = _mesh_source_images(ctx, max(1, iteration), spec)
        for a, role in zip(images, roles, strict=False):
            if role in ("top", "bottom", "scribble"):
                continue
            # synthesised conditioning views are not ground truth — neither
            # the generated side views nor the measured-outline back render
            if getattr(a, "role_hint", "") == "synth_view" or a.asset_id == "_synth_back":
                continue
            p = getattr(a, "path", None)
            if not p or not Path(p).exists():
                continue
            cand = reference_mask(p)
            if cand is None:
                continue
            # sanity gate: near-empty extractions (rembg speckle on noise can
            # produce <1% coverage) and full-frame failures carry no shape
            # evidence — they'd drag silhouette IoU to ~0 for any mesh
            frac = float(cand.sum()) / max(1, cand.size)
            if not 0.03 <= frac <= 0.97:
                log.debug("orient.ref_mask_rejected", path=str(p), coverage=round(frac, 3))
                continue
            mask = cand
            break
        if mask is not None:
            out = storage.version_dir(ctx.project_id, ctx.version_id) / "_ref_mask.png"
            save_mask(mask, out)
            rel = storage.relpath(out)
            ctx.mesh_info["_ref_mask_rel"] = rel
        ctx.mesh_info["_ref_mask"] = mask
    except Exception as e:
        log.debug("orient.ref_mask_failed", error=str(e)[:160])
    return mask, rel
async def _resolve_mesh_adapter(
    ctx: AgentCtx, n_images: int, prefer: str | None = None, spec: Any | None = None
) -> tuple[str, int]:
    """Pick (model_id, max_images) so the sidecar never sees an impossible request.

    'auto' stays 'auto' for single-image runs unless bench data says otherwise;
    multi-image runs pick an explicit multiview adapter when one is available,
    else degrade to the best single-image adapter and cap the image list.
    `prefer` is an explicit adapter override (candidate-screening winner).
    Model choice is data-driven via DATA_DIR/mesh_bench.json when present.
    """
    configured = prefer or ctx.settings.mesh_model or ctx.runtime_settings.mesh_model
    try:
        health = await ctx.mesh.health() if ctx.mesh else None
    except Exception:
        health = None
    models = [
        m
        for m in (health.models if health else [])
        if m.available and m.id != "fake" and getattr(m, "kind", "mesh") == "mesh"
    ]
    device = health.device if health else ""
    if configured and configured != "auto":
        m = next((m for m in models if m.id == configured), None)
        return configured, (m.max_images if m else n_images)
    if not models:
        return "auto", n_images
    # a marked sketch (role_hint="sketch") only works on scribble adapters —
    # route there explicitly when auto-selecting
    if any(getattr(a, "role_hint", "") == "sketch" for a in ctx.assets):
        m = next((m for m in models if m.id.endswith("-scribble")), None)
        if m is not None:
            return m.id, m.max_images
    from kiara.agent.model_pick import load_bench, part_class, pick_adapter

    bench = load_bench()
    # hunyuan3d-omni takes the spec's stated dims as bbox conditioning
    # (_omni_control) — prefer it over the generic auto ladder for fully
    # dimensioned single-image parts when the weights are already local.
    # Bench data, when present, stays authoritative.
    if bench is None:
        omni = next((m for m in models if "omni" in m.id.lower() and m.downloaded), None)
        dims_full = spec is not None and all(
            float(d.value or 0) > 0 for d in (spec.length_x, spec.width_y, spec.height_z)
        )
        if omni is not None and dims_full and n_images <= max(1, omni.max_images or 1):
            return omni.id, omni.max_images
    pick = pick_adapter(
        models,
        part_class(spec),
        need_multiview=n_images > 1,
        bench=bench,
        device=device,
    )
    if pick is None:
        return "auto", n_images
    if bench is None and n_images <= 1:
        return "auto", n_images  # no data — keep server-side device preference
    return pick.id, (pick.max_images if n_images > 1 else n_images)
def _revolve_axis(spec: PartSpec, analysis: dict[str, Any]) -> str | None:
    """Rotation axis of a revolve-like part, or None when not provable.

    Evidence gate: the executor's circularity (when it reports one) or spec
    wording (round/cylindrical/rotational — shared regex in mesh_score) AND
    a measured bbox whose two radial extents agree within 20%. All-equal
    extents (cubic) resolve to "z" — profile stations then read off any
    side view."""
    from kiara.agent.mesh_score import spec_suggests_revolve

    circ = analysis.get("circularity")
    measured_round = circ is not None and float(circ) >= 0.85
    if not (spec_suggests_revolve(spec) or measured_round):
        return None
    bb = analysis.get("bbox_size") or []
    if len(bb) != 3:
        return None
    L, W, H = (float(v) for v in bb)
    pairs = {"z": (L, W), "x": (W, H), "y": (L, H)}
    best_axis, best_dev = None, 1e9
    for axis, (a, b) in pairs.items():
        hi = max(a, b)
        if hi <= 0:
            continue
        dev = abs(a - b) / hi
        if dev < best_dev:
            best_axis, best_dev = axis, dev
    return best_axis if best_dev <= 0.2 else None
def _rotate_hero(ordered: list[Any], synth_roles: dict[str, str]) -> list[Any]:
    """Iteration>1 hero rotation among REAL images only — a synthesised side
    view must never slide into the generator's front slot."""
    real = [a for a in ordered if a.asset_id not in synth_roles]
    if len(real) <= 1:
        return ordered
    synth = [a for a in ordered if a.asset_id in synth_roles]
    return real[1:] + real[:1] + synth
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
    """Mesh-mode loop: generate -> baseline process -> critique ->
    refine-repair rounds -> best candidate.

    The baseline is always critiqued and registered as a candidate BEFORE any
    LLM session touches it — a wrecked refine round can no longer hide the
    pristine result. Repair sessions get real critique issues; when the
    baseline already passes, no session runs at all. Sessions unavailable
    (tools off, vision model stripped) → previous seed-rotation behaviour.
    """
    s = ctx.runtime_settings
    best: Candidate | None = None
    iterations_done = 0
    refine_rounds = max(0, getattr(s, "agent_mesh_refine_rounds", 1))
    hard_fails = 0
    # consecutive repair-round regressions — when ops-level repair keeps
    # wrecking the shape (e.g. a generator V-notch no transform can undo),
    # further sessions only burn tokens; new seeds are the better spend
    repair_regressions = 0
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

    async def _evaluate(
        it: int, exec_result: dict[str, Any]
    ) -> tuple[Critique, list[str], list[dict[str, Any]]]:
        """Dim check + vision critique + emit. Returns (critique, dim_issues,
        feature_findings)."""
        analysis = exec_result.get("analysis") or {}
        dim_issues = compare_dims(spec, analysis, tol_pct, tol_mm)
        # baseline scale_to_spec reports aspect mismatches vs the uniform
        # pre-ops scale — after ops ran (dim-fit scales, clips) those numbers
        # are stale and the measured bbox above is the truth (observed:
        # critique repeated 'measured 45.3x45.7' on a fitted 39.9 part)
        if not ((exec_result.get("mesh") or {}).get("post") or {}).get("ops"):
            dim_issues += list((exec_result.get("mesh") or {}).get("dim_issues") or [])
        # deterministic feature evidence on the processed mesh — the same
        # authority the CAD path feeds the critic, converted from the
        # ray-grid tunnel/dimple detection every mesh result carries
        feature_findings: list[dict[str, Any]] = []
        mesh_feats = (exec_result.get("mesh") or {}).get("features") or {}
        if s.agent_feature_feedback and mesh_feats:
            try:
                from kiara.cad.features import (
                    compare_anchor_positions,
                    compare_features,
                    mesh_features_to_measured,
                )

                measured = mesh_features_to_measured(mesh_feats)
                if active_features(spec):
                    from kiara.cad.standard_parts import covered_feature_indices

                    covered = covered_feature_indices(spec, keep_host_voids=True)
                    dropped = dropped_feature_indices(spec)
                    feature_findings += [
                        f.model_dump()
                        for f in compare_features(
                            [
                                f
                                for i, f in enumerate(spec.features)
                                if i not in covered and i not in dropped
                            ],
                            measured,
                        )
                    ]
                if ctx.anchors:
                    anchor_findings = [
                        f.model_dump() for f in compare_anchor_positions(ctx.anchors, measured)
                    ]
                    _annotate_anchor_sources(anchor_findings, ctx.anchors)
                    feature_findings += anchor_findings
            except Exception as e:
                log.warning("mesh.features_compare_failed", error=str(e)[:200])
        sil = (exec_result.get("mesh") or {}).get("silhouette_iou")
        if sil is not None:
            sil = float(sil)
            ctx.mesh_info["_silhouette_iou"] = sil
            if sil < 0.8:
                # deterministic shape verdict vs the reference image — the
                # only objective form check when no plan-view measurement exists
                feature_findings.append(
                    {
                        "feature": "silhouette",
                        "expected": "matches the reference image outline",
                        "actual": f"side-view silhouette IoU {sil:.2f}",
                        "severity": "major" if sil < 0.6 else "minor",
                        "fix_hint": (
                            "the side silhouette clearly misses the reference — "
                            "regenerate or reshape the gross outline"
                            if sil < 0.6
                            else "silhouette follows the reference with visible "
                            "deviations — refine locally (radial profile, scale), "
                            "do NOT rebuild"
                        ),
                    }
                )
        try:
            c = await critique_step.run(
                ctx,
                spec,
                _sheet(exec_result),
                analysis,
                dim_issues,
                exec_result.get("dfm"),
                iteration=it,
                feature_findings=feature_findings,
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
            score=c.score,
        )
        return c, dim_issues, feature_findings

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
        it: int,
        mesh_rel: str,
        post: MeshPostParams,
        rr: mesh_refine_step.RefineResult,
        art_attempt: Any,
    ) -> dict[str, Any]:
        post.ops = rr.ops
        if rr.ops:
            post.ops_version = MESH_OPS_VERSION
        await ctx.emit("meshprocess", it, f"re-processing mesh with {len(rr.ops)} op(s)")
        res = await _process_mesh(ctx, it, mesh_rel, spec, post, subdir=f"a{next(art_attempt)}")
        ops_warn = (res.get("mesh") or {}).get("ops_version_warning")
        if ops_warn:
            # worker recorded ops with a different mesh_ops version than the
            # executor replays — surfaces a worker/executor image skew that
            # otherwise only shows up in executor logs
            await ctx.emit("meshprocess", it, f"WARNING: {ops_warn}")
            log.warning("mesh.ops_version_drift", warning=ops_warn)
        ctx.mesh_info["post"] = post.model_dump()
        ctx.mesh_info["ops"] = [o.model_dump() for o in rr.ops]
        return res

    for it in range(start_iteration, start_iteration + max_iterations):
        iterations_done = it
        if await cancelled():
            outcome.status = "cancelled"
            return outcome
        if best is not None and await finish_early():
            await ctx.emit("finalize", it, "finish requested — keeping best candidate")
            break
        if ctx.tokens_used() > ctx.budget_tokens or time.monotonic() - t0 > s.agent_max_wall_s:
            await ctx.emit("budget", it, "budget exceeded — finalising best candidate")
            break

        # per-iteration artifact slots: every process call writes renders/stl
        # — without a subdir a later (worse) refine round would overwrite the
        # files the best candidate's score refers to
        art_attempt = itertools.count(1)

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
                if ctx.mesh_info.get("_mesh_hopeless"):
                    outcome.fallback_to_cad = stated_dim_count(spec) >= 2
                    await ctx.emit(
                        "meshgen",
                        it,
                        "no viable mesh candidate — "
                        + ("CAD fallback" if outcome.fallback_to_cad else "stopping"),
                    )
                    break
                continue

            # ORIENT (once per job)
            orientation = await _orient_mesh(ctx, it, mesh_rel)
            post = MeshPostParams(
                up=orientation.get("up"),  # type: ignore[arg-type]
                front=orientation.get("front"),  # type: ignore[arg-type]
                mirror_x=bool(orientation.get("mirror_x")),
                auto_yaw=True,
            )
            post = merge_post_params(post, {})

            # extract the side-reference subject mask once per job — feeds
            # silhouette IoU in the executor (critique calibration) and the
            # raw-candidate scorer
            _ref_side_mask(ctx, it, spec)

            # measured plan-view outline is the authoritative footprint for
            # flat parts — clip first so a drifted generator shape (banana)
            # cannot poison scoring and downstream CAD conversion
            outline = _measured_plan_outline(ctx)
            dims = (spec.length_x.value, spec.width_y.value, spec.height_z.value)
            flat = _is_flat_spec_dims(dims)
            if flat and outline:
                post.ops = [
                    MeshOp(
                        op="clip_to_outline",
                        args={"outline": outline, "margin_mm": 0.5},
                    )
                ]

            # PROCESS_MESH baseline
            await ctx.emit("meshprocess", it, "cleaning, orienting and scaling mesh")
            exec_result = await _process_mesh(
                ctx, it, mesh_rel, spec, post, subdir=f"a{next(art_attempt)}"
            )
            ctx.mesh_info["post"] = post.model_dump()
            ctx.mesh_info["orientation"] = {"up": post.up, "front": post.front}

            # deterministic fit to stated dims BEFORE the refine session —
            # generative meshes often get proportions wrong while capturing
            # the shape; fitting here means the session doesn't spend its
            # whole budget (and ctx tokens) on manual per-axis scaling
            if stated_dim_count(spec) >= 2 and exec_result.get("ok"):
                dim_issues0 = compare_dims(spec, exec_result.get("analysis") or {}, tol_pct, tol_mm)
                dim_issues0 += list((exec_result.get("mesh") or {}).get("dim_issues") or [])
                if dim_issues0:
                    fitted = await _fit_mesh_to_stated(
                        ctx,
                        it,
                        mesh_rel,
                        spec,
                        post,
                        exec_result,
                        tol_pct,
                        tol_mm,
                        art_attempt,
                        make_solid=False,
                    )
                    if fitted is not None:
                        exec_result, post = fitted
                        ctx.mesh_info["post"] = post.model_dump()

            # deterministic silhouette refit BEFORE the baseline critique —
            # the reference mask's outline is objective shape evidence: a
            # spec'd revolve gets radial profile stations, other parts a
            # visual-hull carve. Runs before drill ops so bore coverage is
            # judged on the reshaped mesh.
            sil_ops = _spec_silhouette_ops(spec, ctx, exec_result or {})
            if sil_ops:
                post_s = merge_post_params(
                    post, {"ops": [*post.ops, *[o.model_dump() for o in sil_ops]]}
                )
                res_s = await _process_mesh(
                    ctx,
                    it,
                    mesh_rel,
                    spec,
                    post_s,
                    make_solid=False,
                    subdir=f"a{next(art_attempt)}",
                )
                if res_s.get("ok"):
                    exec_result, post = res_s, post_s
                    ctx.mesh_info["post"] = post.model_dump()
                    await ctx.emit(
                        "meshprocess",
                        it,
                        "applied deterministic silhouette ops "
                        f"({', '.join(o.op for o in sil_ops)})",
                    )

            # deterministic spec features BEFORE the baseline critique —
            # a declared centred through-bore is cut right away so the
            # critique never carries an unfixable "missing bore" major and
            # the repair session doesn't spend a round on it (grip_mesh:
            # every iteration regressed exactly this way)
            drill_ops = _spec_drill_ops(spec, exec_result or {})
            if drill_ops:
                post_d = merge_post_params(
                    post, {"ops": [*post.ops, *[o.model_dump() for o in drill_ops]]}
                )
                res_d = await _process_mesh(
                    ctx,
                    it,
                    mesh_rel,
                    spec,
                    post_d,
                    make_solid=False,
                    subdir=f"a{next(art_attempt)}",
                )
                if res_d.get("ok"):
                    exec_result, post = res_d, post_d
                    ctx.mesh_info["post"] = post.model_dump()
                    await ctx.emit(
                        "meshprocess",
                        it,
                        f"drilled {len(drill_ops[0].args['holes'])} declared bore(s)",
                    )

        if not exec_result or not exec_result.get("ok"):
            log.warning("mesh.process_failed", iteration=it, error=(exec_result or {}).get("error"))
            continue

        # pre-gate: deterministic score before spending a critique call —
        # aspect/footprint/holes that cannot be this part count as hard fails
        from kiara.agent.mesh_score import score_mesh_candidate

        sc = score_mesh_candidate(exec_result, spec, _measured_plan_outline(ctx))
        mesh_meta = exec_result.get("mesh") or {}
        reject_reason: str | None = None
        if sc < 0.35:
            reject_reason = f"score {sc:.2f}"
        else:
            # clearly-bad gate: the silhouette measurably misses the reference
            # (a matched view means the 0.xx is real, not an unusable mask)
            # AND dims are off — either alone is fixable, both is a wrong part
            sil = mesh_meta.get("silhouette_iou")
            if sil is not None and mesh_meta.get("silhouette_view") and float(sil) < 0.3:
                dim_gate = compare_dims(spec, exec_result.get("analysis") or {}, tol_pct, tol_mm)
                if dim_gate:
                    reject_reason = (
                        f"silhouette IoU {float(sil):.2f} + dim mismatch ({dim_gate[0]})"
                    )
        if reject_reason is not None:
            hard_fails += 1
            await ctx.emit(
                "meshprocess",
                it,
                f"mesh rejected by pre-gate ({reject_reason}) — new seed",
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

        # MESH-VS-SPEC GATE — check the processed baseline against the SPEC,
        # not the reference image: a generative mesh can pass the silhouette
        # pre-gate while being topologically wrong when the conditioning image
        # itself was wrong (T-bar photo of an arched handle). On a major
        # mismatch the producing generated assets are dropped and the run
        # re-bootstraps a corrected concept image → new mesh; bounded to one
        # regen per job, re-armed once so the regenerated baseline is checked
        # again. Warn-and-continue when the budget is spent.
        if spec is not None and not ctx.mesh_info.get("_mesh_gate_done"):
            ctx.mesh_info["_mesh_gate_done"] = True
            gate_reasons: list[str] = []
            # deterministic half: sorted side-length signature of the spec vs
            # the mesh bbox — the scorer's axis-wise aspect+height terms let a
            # wrong Z/W signature through when L/W happens to match
            spec_sig = sorted(
                (float(d.value or 0) for d in (spec.length_x, spec.width_y, spec.height_z)),
                reverse=True,
            )
            ana_bbox = list((exec_result.get("analysis") or {}).get("bbox_size") or [])
            if len(ana_bbox) == 3 and all(v > 0 for v in spec_sig):
                mesh_sig = sorted((float(v) for v in ana_bbox), reverse=True)
                worst = max(
                    abs(a - b) / max(a, 1e-6) for a, b in zip(spec_sig, mesh_sig, strict=True)
                )
                if worst > 0.35:
                    gate_reasons.append(f"bbox shape signature off by {worst:.0%}")
            # vision half: only when the mesh was conditioned by SYNTHETIC
            # images — user-supplied references are trusted ground truth,
            # generated ones are the untrusted conditioning this gate exists
            # for (also keeps a vision call out of runs that never made one)
            used = set((ctx.mesh_info.get("generation") or {}).get("images") or [])
            has_gen = any(a.kind == "generated_image" and a.asset_id in used for a in ctx.assets)
            sheet = _sheet(exec_result)
            verdict = None
            if sheet is not None and has_gen:
                from kiara.agent.image_check import verify_concept_image

                verdict = await verify_concept_image(ctx, sheet, spec)
            if verdict is not None and verdict.severity == "major":
                gate_reasons.append(
                    "render does not match spec: "
                    + ", ".join(verdict.missing[:4] or ["shape mismatch"])
                )
            regens = int(ctx.mesh_info.get("_mesh_gate_regens") or 0)
            if gate_reasons and regens < 1 and it < start_iteration + max_iterations - 1:
                ctx.mesh_info["_mesh_gate_regens"] = regens + 1
                ctx.mesh_info["_mesh_gate_done"] = False  # re-arm for the regenerated baseline
                if used:
                    ctx.assets[:] = [
                        a
                        for a in ctx.assets
                        if not (a.kind == "generated_image" and a.asset_id in used)
                    ]
                ctx.mesh_info["_concept_bootstrap"] = False
                ctx.mesh_info["_regen_hint"] = "; ".join(gate_reasons)
                await ctx.emit(
                    "meshgen",
                    it,
                    "mesh does not match the spec ("
                    + "; ".join(gate_reasons)
                    + ") — regenerating reference image",
                )
                continue
            if gate_reasons:
                await ctx.emit(
                    "meshgen",
                    it,
                    "mesh/spec mismatch persists (" + "; ".join(gate_reasons) + ") — continuing",
                )

        # ANALYZE + CRITIQUE the baseline BEFORE any refine session — the
        # deterministic baseline is always a scored candidate, so an
        # unproductive refine round can no longer replace it with a wreck
        # (observed: 0.95-screened knob re-oriented + flattened into a sliver
        # before the first critique ever saw it)
        await ctx.emit("critique", it, "comparing renders with your images")
        critique, dim_issues, feature_findings = await _evaluate(it, exec_result)

        cand = Candidate(
            it,
            None,
            exec_result,
            critique,
            dim_issues,
            mesh_post=(exec_result.get("mesh") or {}).get("post"),
        )
        cand.feature_findings = feature_findings
        cand.dq = det_quality(exec_result, dim_issues)
        cand.wrecked = critique.score <= getattr(s, "agent_mesh_hard_fail_score", 30) and any(
            d.severity == "major" for d in critique.discrepancies
        )
        if best is None or cand.rank > best.rank:
            best = cand

        def _passed(c: Critique, di: list[str], res: dict[str, Any]) -> bool:
            major = [d for d in c.discrepancies if d.severity == "major"]
            return (
                c.passed
                and c.score >= pass_score
                and not di
                and not major
                and not _multi_solid(res)
            )

        if _passed(critique, dim_issues, exec_result):
            await ctx.emit("done", it, "all checks passed")
            break

        # MESH_REFINE repair session(s) — fix this mesh instead of a new seed.
        # Repair gate: when stated dims are the ONLY major problem the
        # deterministic fit above already had its shot — an LLM session on
        # pure dim issues just wrecks the shape (knob squashed to a capsule)
        issues0 = build_issues(
            critique,
            dim_issues,
            feature_findings,
            exec_result.get("dfm"),
            exec_result.get("analysis"),
            spec=spec,
        )
        only_dims = _only_dim_issues(dim_issues, issues0)
        if only_dims:
            await ctx.emit(
                "meshrefine",
                it,
                "repair skipped — only dimension issues remain (deterministic fit already applied)",
            )
        skip_repair = only_dims or repair_regressions >= 2
        if repair_regressions >= 2:
            await ctx.emit(
                "meshrefine",
                it,
                "repair skipped — last repair sessions all regressed; trying a fresh seed instead",
            )
        for _round in range(0 if skip_repair else refine_rounds):
            if await cancelled():
                outcome.status = "cancelled"
                return outcome
            if await finish_early():
                break
            rr = await _refine(it, mesh_rel, post, exec_result, issues=issues0)
            if rr is None:
                break  # sessions unavailable
            if rr.regenerate:
                if rr.subject_hint:
                    ctx.mesh_info["_regen_hint"] = rr.subject_hint
                await ctx.emit("meshrefine", it, f"regenerating — {rr.reasoning[:100]}")
                break
            if not (rr.changed or rr.new_mesh_rel):
                break
            prev_ops = list(post.ops)
            prev_rank = cand.rank
            if rr.new_mesh_rel:
                mesh_rel = rr.new_mesh_rel
            new_exec = await _apply_ops(it, mesh_rel, post, rr, art_attempt)
            if not new_exec.get("ok"):
                log.warning("mesh.refine_reprocess_failed", error=new_exec.get("error"))
                post.ops = prev_ops
                ctx.mesh_info["post"] = post.model_dump()
                break
            critique2, dim_issues2, findings2 = await _evaluate(it, new_exec)
            cand = Candidate(
                it,
                None,
                new_exec,
                critique2,
                dim_issues2,
                mesh_post=(new_exec.get("mesh") or {}).get("post"),
            )
            cand.feature_findings = findings2
            cand.dq = det_quality(new_exec, dim_issues2)
            cand.wrecked = critique2.score <= getattr(s, "agent_mesh_hard_fail_score", 30) and any(
                d.severity == "major" for d in critique2.discrepancies
            )
            if best is None or cand.rank > best.rank:
                best = cand
            if rr.new_mesh_rel or cand.rank >= prev_rank:
                # accept: progress, or a swapped-in regenerated baseline that
                # cannot be rank-compared against the old mesh
                repair_regressions = 0
                exec_result = new_exec
                critique, dim_issues = critique2, dim_issues2
                # refresh the issue list — the next repair round must chase
                # what is still broken, not the baseline's stale complaints
                issues0 = build_issues(
                    critique,
                    dim_issues,
                    findings2 or [],
                    exec_result.get("dfm"),
                    exec_result.get("analysis"),
                    spec=spec,
                )
            else:
                # deterministic regression — restore the better op list so the
                # wreck never becomes the session state a later round builds on
                repair_regressions += 1
                post.ops = prev_ops
                ctx.mesh_info["post"] = post.model_dump()
                ctx.mesh_info["ops"] = [o.model_dump() for o in prev_ops]
                await ctx.emit(
                    "meshrefine",
                    it,
                    f"refine round regressed (score {cand.critique.score if cand.critique else 0}"
                    f" vs {prev_rank[6]}) — keeping previous ops",
                )
                break
            if _passed(critique, dim_issues, exec_result):
                break

        if _passed(critique, dim_issues, exec_result):
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
    # export + parametric reconstruction are independent executor tasks —
    # run them concurrently instead of serializing two round trips
    param_coro = (
        _parametrize_best(ctx, best, spec, tol_pct=tol_pct, tol_mm=tol_mm)
        if ctx.runtime_settings.agent_parametrize
        else None
    )
    exports, param_res = await asyncio.gather(
        ctx.executor.export_all(
            {
                "job_id": ctx.job_id,
                "project_id": str(ctx.project_id),
                "version_id": str(ctx.version_id),
                "iteration": best.iteration,
                "art_dir": best.exec_result.get("art_dir"),
                "name": "model",
            }
        ),
        param_coro if param_coro is not None else _none(),
    )
    artifacts = _copy_best_artifacts(ctx, best.iteration, best.exec_result)
    for fmt, rel in (exports.get("exports") or {}).items():
        if rel:
            artifacts[f"export:{fmt}"] = rel
    artifacts.update(
        _copy_mesh_artifacts(
            ctx.project_id, ctx.version_id, best.iteration, best.exec_result.get("art_dir")
        )
    )
    if param_res is not None:
        ctx.mesh_info["parametric"] = param_res

    outcome.status = (
        "succeeded"
        if (
            best.critique
            and best.critique.passed
            and not best.dim_issues
            and not _multi_solid(best.exec_result)
        )
        else "needs_review"
    )
    outcome.art_dir = best.exec_result.get("art_dir")
    _clean_losing_attempts(ctx, outcome.art_dir)
    rec = ctx.mesh_info.get("parametric")
    if rec:
        # accepted parametric reconstruction replaces the mesh-op params —
        # downstream editing now works on real code + schema
        outcome.code = rec.get("code")
        outcome.params = rec.get("params")
        outcome.params_schema = rec.get("params_schema")
    else:
        outcome.code = None
        post_used = best.mesh_post or {}
        outcome.params = post_used or dict(best.exec_result.get("params_used") or {})
        outcome.params_schema = [p.model_dump() for p in mesh_params_schema()]
    outcome.analysis = best.exec_result.get("analysis")
    outcome.features = best.exec_result.get("features") or (best.exec_result.get("mesh") or {}).get(
        "features"
    )
    outcome.critique = best.critique.model_dump() if best.critique else None
    outcome.issues = build_issues(
        best.critique,
        best.dim_issues,
        [],
        best.exec_result.get("dfm"),
        best.exec_result.get("analysis"),
        spec=spec,
    )
    vs_entry = _verify_spec_issue(best.exec_result, stage="cad")
    if vs_entry:
        outcome.issues = [vs_entry, *list(outcome.issues or [])]
    if outcome.critique is not None:
        outcome.critique["issues"] = outcome.issues
    outcome.dfm = best.exec_result.get("dfm")
    outcome.bom = best.exec_result.get("bom")
    outcome.score = best.critique.score if best.critique else None
    outcome.best_iteration = best.iteration
    outcome.artifacts = artifacts
    outcome.usage = dict(getattr(ctx.llm, "total_usage", {}))
    outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
    if rec:
        # compact summary only — the full code/schema already lives on the
        # outcome's own code/params/params_schema fields
        outcome.mesh["parametric"] = {
            "kind": rec.get("kind"),
            "chamfer_mm": rec.get("chamfer_mm"),
            "confidence": rec.get("confidence"),
        }
    outcome.mesh["best_iteration"] = best.iteration
    _write_mesh_json(ctx.project_id, ctx.version_id, outcome.mesh)
    return outcome
def _score_candidate_mesh(
    rel: str,
    spec: PartSpec,
    outline: list[list[float]] | None,
    ref_mask: Any = None,
) -> float | None:
    """Load + orient + spec-scale a raw candidate GLB and score it."""
    try:
        import trimesh

        from kiara.agent.mesh_score import score_raw_mesh
        from kiara.cad.mesh_ops import (
            candidate_meshes,
            decimate,
            orient,
            orient_by_plan_view,
            rest_on_xy_center,
            scale_to_spec,
        )

        m = trimesh.load(storage.resolve(rel), force="mesh")
        if not isinstance(m, trimesh.Trimesh):
            return None
        # candidates are scored on 6 orientations each — decimate the raw
        # generator mesh once so ray-casting stays cheap
        m = decimate(m, 30_000)
        if outline:
            hit = orient_by_plan_view(m, outline)
            if hit is not None:
                m = orient(m, hit[0], hit[1], mirror_x=hit[2])
            m, _, _ = scale_to_spec(m, spec)
            m = rest_on_xy_center(m)
            return score_raw_mesh(m, spec, outline, ref_mask=ref_mask)
        # no plan view: the raw generator orientation is arbitrary — a part
        # lying on its side would fail the aspect check spuriously. Score all
        # 6 stable orientations and keep the best.
        best = 0.0
        for cand in candidate_meshes(m).values():
            c, _, _ = scale_to_spec(cand, spec)
            sc = score_raw_mesh(c, spec, None, ref_mask=ref_mask)
            if sc is not None and sc > best:
                best = sc
        return best
    except Exception as e:
        log.warning("mesh.candidate_score_failed", rel=rel, error=str(e)[:160])
        return None
async def _select_candidate(
    ctx: AgentCtx,
    iteration: int,
    scores: dict[tuple[int, str | None], float],
    hf: tuple[str, float] | None,
) -> int | None:
    """Pick the winning (seed, adapter) from cheap candidates, or let the
    deterministic heightfield take over when it outscores the generator."""

    def _fmt(scores: dict) -> str:
        return (
            "{"
            + ", ".join(
                (f"{k[0]}@{k[1]}" if k[1] else str(k[0])) + f": {v}" for k, v in scores.items()
            )
            + "}"
        )

    best_gen = 0.0
    best_key: tuple[int, str | None] | None = None
    if scores:
        best_key = max(scores, key=lambda k: scores[k])
        if scores[best_key] < _CANDIDATE_VIABLE_SCORE:
            # every cheap roll missed aspect/footprint badly — the full-res
            # generation would produce the same failure at 10x the cost
            ctx.mesh_info["_mesh_hopeless"] = True
            await ctx.emit(
                "meshgen",
                iteration,
                f"candidate seeds scored: {_fmt(scores)} — none viable, skipping generation",
            )
        else:
            best_gen = scores[best_key]
            seed_i, model_i = best_key
            ctx.mesh_info["_candidate_model"] = model_i
            meta = (ctx.mesh_info.get("_cand_meta") or {}).get(_cand_key(seed_i, model_i))
            if meta:
                ctx.mesh_info["_candidate_winner"] = dict(meta)
            await ctx.emit(
                "meshgen",
                iteration,
                f"candidate seeds scored: {_fmt(scores)} → using seed {seed_i}"
                + (f" on adapter '{model_i}'" if model_i else ""),
            )

    # heightfield candidate (built above) competes against the best seed —
    # but only as a rescue: a viable generated mesh keeps its organic detail
    # instead of being replaced by a flat slab that merely matches the outline
    if hf is not None:
        hf_rel, hf_score = hf
        gen_weak = best_gen < _CANDIDATE_VIABLE_SCORE or (
            best_gen < 0.5 and hf_score > best_gen + 0.25
        )
        if gen_weak and hf_score > best_gen:
            ctx.mesh_info["_heightfield_rel"] = hf_rel
            ctx.mesh_info["_mesh_hopeless"] = False
            await ctx.emit(
                "meshgen",
                iteration,
                f"heightfield candidate scored {hf_score} > best seed {best_gen} — "
                "using deterministic flat-part mesh",
            )
            return None
        if hf_score > best_gen:
            await ctx.emit(
                "meshgen",
                iteration,
                f"heightfield {hf_score} available but generator {best_gen} kept "
                "(organic detail wins)",
            )
    if best_key is None or ctx.mesh_info.get("_mesh_hopeless"):
        return None
    return best_key[0]
def _spec_drill_ops(spec: PartSpec, exec_result: dict[str, Any]) -> list[MeshOp]:
    """Deterministic `drill_holes` ops for spec'd through-bores the generator
    structurally cannot produce (image→3D never makes a through-tunnel).

    Only *centred* bores are drilled here — positioned hole patterns go to
    the repair session / anchors. A spec hole counts as covered when the
    measured tunnels already contain one on the same axis within 15% dia.
    """
    from kiara.cad.features import _feature_axis_hint

    feats = (exec_result.get("mesh") or {}).get("features") or {}
    bb = (exec_result.get("analysis") or {}).get("bbox_size") or []
    if len(bb) != 3:
        return []
    holes: list[dict[str, Any]] = []
    for f in active_features(spec):
        if str(getattr(f, "type", "") or "").lower() not in ("hole", "bore", "through-hole"):
            continue
        dia = (f.dims or {}).get("diameter") or (f.dims or {}).get("bore_diameter")
        ax = _feature_axis_hint(f)
        if not dia or not ax:
            continue
        pos = f"{f.position or ''} {f.description or ''}".lower()
        # only clearly-centred bores — "through" alone also describes
        # positioned holes, which belong to the repair session / anchors.
        # "cente"/"centr" cover centered (US) and centred (UK) spellings.
        if not any(
            k in pos
            for k in ("cente", "centr", "mittig", "zentriert", "koaxial", "coaxial", "middle")
        ):
            continue
        letter = ax.lstrip("+-")
        covered = any(
            h.get("kind") == "tunnel"
            and str(h.get("axis")) == letter
            and abs(float(h.get("dia") or 0) - float(dia)) <= max(0.5, 0.15 * float(dia))
            for h in feats.get("holes") or []
        )
        if covered:
            continue
        # bore centreline = cross-section centre on the plane ⟂ to the axis;
        # parts are XY-centred and sit on z=0 (x,y args = perp-plane coords)
        c2 = bb[2] / 2.0
        xy = {"x": (0.0, c2), "y": (0.0, c2), "z": (0.0, 0.0)}[letter]
        for _ in range(max(1, int(f.count or 1))):
            holes.append({"x": xy[0], "y": xy[1], "dia": float(dia), "axis": letter})
    return [MeshOp(op="drill_holes", args={"holes": holes})] if holes else []
def _spec_silhouette_ops(
    spec: PartSpec, ctx: AgentCtx, exec_result: dict[str, Any]
) -> list[MeshOp]:
    """Deterministic shape ops from the reference-image side silhouette.

    The worker extracted the subject mask once (`_ref_side_mask` — real
    front/side images only, synthetic conditioning is excluded there) and
    the executor reported which view it matches (`silhouette_view`). A
    declared/measured revolve gets `radial_profile_auto` stations read off
    the outline; other parts get a `silhouette_carve` visual-hull trim —
    but only when the measured IoU leaves meaningful room. Conservative:
    emits [] without a mask, a matched view, or a usable outline.
    """
    mesh_meta = exec_result.get("mesh") or {}
    analysis = exec_result.get("analysis") or {}
    bb = analysis.get("bbox_size") or []
    view = mesh_meta.get("silhouette_view")
    mask = ctx.mesh_info.get("_ref_mask")
    if (
        len(bb) != 3
        or view not in ("+x", "-x", "+y", "-y")
        or mask is None
        or not getattr(mask, "any", lambda: bool(mask))()
    ):
        return []
    try:
        from kiara.cad.silhouette import mask_to_outline_mm, profile_stations_from_outline
    except Exception:
        return []
    # view-plane extents (mask_to_outline_mm basis): v is always world +Z,
    # u is the horizontal in-plane axis — y for ±x views, x for ±y views
    extent_u = float(bb[1] if view.lstrip("+-") == "x" else bb[0])
    extent_v = float(bb[2])
    # aspect-consistency gate: the mask's subject-crop aspect must roughly
    # match the mesh's aspect in the matched view. These ops REFINE a
    # roughly-right shape — a wildly different aspect means the extraction is
    # junk or the part is wrong entirely (regeneration territory, not a trim)
    try:
        import numpy as np

        from kiara.cad.silhouette import _deskew_mask, _mask_crop

        m = np.asarray(mask)
        crop = _mask_crop(m)
        # deskew before the aspect check — a diagonal photo subject reads
        # squarer than the part (grip: 2.8:1 photo, 4:1 part) and would
        # wrongly reject or wrongly pass depending on the tilt
        if crop is not None:
            crop = _deskew_mask(crop)
        # subject-crop aspect: occupied columns / occupied rows
        mc = m if crop is None else crop
        mask_aspect = float(mc.any(axis=0).sum()) / max(float(mc.any(axis=1).sum()), 1.0)
        mesh_aspect = extent_u / max(extent_v, 1e-9)
        if abs(mask_aspect - mesh_aspect) / max(mask_aspect, mesh_aspect) > 0.4:
            log.debug(
                "mesh.silhouette_aspect_mismatch",
                mask_aspect=round(mask_aspect, 2),
                mesh_aspect=round(mesh_aspect, 2),
            )
            return []
    except Exception:
        return []
    try:
        outline = mask_to_outline_mm(mask, extent_u, extent_v)
    except Exception as e:
        log.debug("mesh.silhouette_outline_failed", error=str(e)[:160])
        return []
    if len(outline) < 3:
        return []
    sil_iou = mesh_meta.get("silhouette_iou")
    axis = _revolve_axis(spec, analysis)
    if axis is not None:
        # looking down the rotation axis the silhouette is a disc — a flat
        # radius profile carries no shape; only side-on views qualify
        if axis == view.lstrip("+-"):
            return []
        axial = float(bb["xyz".index(axis)])
        stations = profile_stations_from_outline(outline, axis=axis, n_stations=16)
        if not 2 <= len(stations) <= 32:
            return []
        # dims are authoritative: anchor the measured profile to the fitted
        # outer radius (max station -> extent_v/2) so the photo's proportions
        # taper the waist without shrinking the spec diameter. A rescale
        # beyond 2x means the mask and the part disagree grossly — skip.
        peak = max(s["radius_mm"] for s in stations)
        fitted_r = extent_v / 2.0
        if peak <= 1e-6 or not 0.5 <= fitted_r / peak <= 2.0:
            return []
        for s in stations:
            s["radius_mm"] = round(s["radius_mm"] * (fitted_r / peak), 4)
        return [
            MeshOp(
                op="radial_profile_auto",
                args={
                    "axis": axis,
                    "stations": stations,
                    # small gaussian blend keeps the profile a soft waist
                    "blend_mm": round(max(axial * 0.04, 0.2), 3),
                },
            )
        ]
    # carve only a plausible-but-imperfect match: below ~0.25 the
    # mask/mesh pair is more likely mismatched than trimmable (the
    # pre-gate rejects those candidates anyway); above 0.85 carving adds
    # nothing. `None` means no measurement — don't trim blind.
    if sil_iou is None or not 0.25 <= float(sil_iou) < 0.85:
        return []
    return [
        MeshOp(
            op="silhouette_carve",
            args={"view": view, "outline": outline, "tolerance_mm": 0.5},
        )
    ]
def _synth_back_view(ctx: AgentCtx, iteration: int, outline_mm: list[list[float]]) -> Any | None:
    """Filled-silhouette render of the measured plan-view outline as a
    synthetic 'back' view — anchors a flat part's underside for the generator
    without implying wrong bottom features (a mirrored photo would)."""
    if len(outline_mm) < 3:
        return None
    try:
        from PIL import Image, ImageDraw

        xs = [p[0] for p in outline_mm]
        ys = [p[1] for p in outline_mm]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        span = max(x1 - x0, y1 - y0, 1e-9)
        size = 512
        pad = size * 0.08
        scale = (size - 2 * pad) / span

        def px(p: list[float]) -> tuple[float, float]:
            return (
                (p[0] - (x0 + x1) / 2) * scale + size / 2,
                size / 2 - (p[1] - (y0 + y1) / 2) * scale,
            )

        img = Image.new("RGB", (size, size), "white")
        ImageDraw.Draw(img).polygon([px(p) for p in outline_mm], fill="black")
        out = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration) / ("_synth_back.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        from kiara.agent.types import InputAsset

        return InputAsset(
            asset_id="_synth_back",
            kind="generated_image",
            path=out,
            caption="synthetic bottom silhouette (measured outline)",
            filename="_synth_back.png",
        )
    except Exception:
        log.debug("mesh.synth_back_failed", exc_info=True)
        return None
def _try_heightfield_candidate(
    ctx: AgentCtx, iteration: int, spec: PartSpec | None
) -> tuple[str, float] | None:
    """Build+score the deterministic flat-part candidate. Returns (glb relpath,
    score) or None when not applicable/unbuildable."""
    if spec is None or not getattr(ctx.runtime_settings, "mesh_heightfield_flat", True):
        return None
    outline = _measured_plan_outline(ctx)
    # the measurement dict that produced the outline carries px_per_mm and
    # points at the source image for the relief map
    meas = None
    for m in (ctx.measurements or {}).values():
        if m.get("outline_mm") == outline:
            meas = m
            break
    img_path = None
    px_per_mm = None
    if meas:
        px_per_mm = meas.get("px_per_mm")
        by_id = {a.asset_id: a for a in ctx.assets}
        src = by_id.get(str(meas.get("asset_id")))
        img_path = src.path if src else None
    from kiara.agent.mesh_score import score_raw_mesh
    from kiara.cad.heightfield import heightfield_candidate

    mesh = heightfield_candidate(spec, outline, image_path=img_path, px_per_mm=px_per_mm)
    if mesh is None:
        return None
    score = score_raw_mesh(mesh, spec, outline)
    if score < _CANDIDATE_VIABLE_SCORE:
        return None
    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration)
    it_dir.mkdir(parents=True, exist_ok=True)
    out = it_dir / "mesh_raw_hf.glb"
    try:
        mesh.export(str(out), file_type="glb")
    except Exception:
        return None
    ctx.mesh_info["generation"] = {
        "model": "heightfield",
        "seed": None,
        "cached": False,
        "heightfield": True,
        "mesh_rel": storage.relpath(out),
        "images": [],
    }
    return storage.relpath(out), score
def stated_dim_count(spec: PartSpec) -> int:
    return sum(1 for d in (spec.length_x, spec.width_y, spec.height_z) if d.source == "stated")
