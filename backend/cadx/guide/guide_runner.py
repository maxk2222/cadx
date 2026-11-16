"""Guide runner — the mission orchestrator behind ``run_guide_job``.

Runs the missions sequentially, each as a bounded ``run_session`` with a
mission-scoped ``GuideToolSession``:

    doc → steps → parts → assembly → review → (repair loop) → annotate → emit

- ``ask_user`` inside a mission parks the job: the runner returns
  ``waiting_for_input`` with the questions; ``enqueue_resume`` (generic,
  payload-func dispatch) restarts the same mission with the answers in
  ``state.pending_answers``.
- The review verdict drives a bounded repair loop: blocker findings reopen
  the flagged mission (``missions_done`` pruned, plan.md annotated) up to
  ``guide_review_max_iter`` times.
- ``emit`` is deterministic code, not an LLM call: unity_guide.json +
  artifacts are written server-side — the mission list in the prompt doc
  names it for readability only.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

from cadx import storage
from cadx.agent.context import system_prompt
from cadx.agent.orchestrator import RunOutcome
from cadx.agent.toolloop import SessionResult, run_session
from cadx.agent.types import AgentCtx
from cadx.guide import scene, unity
from cadx.guide.session import GuideToolSession
from cadx.guide.state import GuideState
from cadx.llm.client import LLMError
from cadx.logging import get_logger

log = get_logger(__name__)

MISSION_ORDER = ["doc", "steps", "parts", "assembly", "review", "annotate"]
# blender-mode order: the free Blender build mission replaces
# parts+assembly entirely — no ports, no solver; the scene IS the
# placement state and gets gated by blender_check after the mission
BLENDER_MISSION_ORDER = ["doc", "steps", "build", "review", "annotate"]


def _mission_order(state: GuideState) -> list[str]:
    return BLENDER_MISSION_ORDER if state.guide_mode == "blender" else MISSION_ORDER


def _resolve_auto_mode(
    state: GuideState, assets: list[dict[str, Any]], state_dir: Path
) -> None:
    """Decide guide_mode=="auto" once, durably: blender only when the run
    ships zero CAD assets AND a Blender MCP endpoint answers — 'auto'
    falls back honestly to cad when Blender is not running."""
    if state.guide_mode != "auto":
        return
    from cadx.guide import blender_client, cadmeta

    has_cad = any(
        Path(str(x.get("path") or x.get("filename") or "")).suffix.lower()
        in cadmeta._BREP_EXTS | cadmeta._MESH_EXTS
        for x in assets
    )
    state.guide_mode = (
        "cad" if has_cad or blender_client.probe_client() is None else "blender"
    )
    state.save(state_dir)
    log.info("guide.mode_resolved", mode=state.guide_mode, has_cad=has_cad)
_MAX_CALLS = {
    "doc": 24,
    "steps": 60,
    "parts": 40,
    "assembly": 80,
    "build": 80,
    "review": 24,
    "annotate": 24,
}
# missions whose failure degrades to a warning instead of killing the
# whole job — annotations are cosmetic overlays; the review verdict is
# the real quality gate and already ran by the time annotate executes
_OPTIONAL = {"annotate"}


def _persist_tool_log(state_dir: Any, mission: str, attempt: int, result: Any) -> None:
    """Durable failure forensics: the per-call tool log dies with the
    session — persist it per mission+attempt so a repeat_loop or budget
    death can be audited from the actual call sequence. The index is
    MONOTONIC across mission re-opens: a reopened mission restarts its
    attempt counter at 0, so naming by attempt alone silently overwrote
    the earlier mission's log (observed: the 100-call first steps run
    vanished under the 94-call repair churn)."""
    try:
        seq = len(list(Path(state_dir).glob(f"tool_log_{mission}_*.jsonl")))
        p = Path(state_dir) / f"tool_log_{mission}_{seq}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for e in result.tool_log or []:
                fh.write(json.dumps(e, ensure_ascii=False, default=str)[:8000] + "\n")
    except Exception:
        log.warning("guide.tool_log_persist_failed", mission=mission)


def _tool_failures(result: Any, limit: int = 6) -> list[str]:
    """Readable ok:false tool results from a finished session — fed into
    the retry nudge AND the failure log so diagnostics survive."""
    fails = []
    for e in result.tool_log or []:
        if not isinstance(e, dict):
            continue
        res = e.get("result")
        if isinstance(res, dict):
            if not res.get("ok", True):
                detail = (
                    res.get("error")
                    or res.get("issues")
                    or res.get("bad_joints")
                    or res.get("unfulfilled_mates")
                    or res.get("unexpected_mates")
                    or res.get("disconnected_instances")
                    or res.get("collisions")
                    or "ok=false"
                )
                fails.append(f"{e.get('name')}: {str(detail)[:160]}")
        elif isinstance(res, str) and '"ok": false' in res:
            # _truncate can stringify long results — salvage a useful cause.
            # audit results carry per-joint "ok": false rows but the real
            # story is bad_joints/buried_instances, not the "error" field.
            # Slice the raw tail after the key — a truncated array has no
            # closing bracket for a regex to match, which is exactly where
            # "(truncated failure)" used to hide the cause.
            detail = None
            for key in (
                "error",
                "bad_joints",
                "unfulfilled_mates",
                "unexpected_mates",
                "buried_instances",
                "disconnected_instances",
                "issues",
                "collisions",
            ):
                m = re.search(rf'"{key}"\s*:\s*', res)
                if m:
                    tail = res[m.end() : m.end() + 160].strip()
                    if tail and tail not in ("[]", '""', "null"):
                        detail = f"{key}: {tail}"
                        break
            fails.append(f"{e.get('name')}: {detail or '(truncated failure)'}")
    return fails[-limit:]


def _calls_budget(mission: str, state: GuideState) -> int:
    """Tool-call budget per mission. Assembly scales with the number of
    sub-steps — each placement needs ~5 calls (ports/preview/place/render/
    collide), a fixed cap would break longer manuals."""
    base = _MAX_CALLS.get(mission, 40)
    n = sum(max(1, len(s.sub_steps)) for s in state.steps) or len(state.steps)
    if mission == "doc":
        # page-by-page text+figure reads cost ~3 calls per page
        pages = len((state.document.pages if state.document else []) or [])
        base = max(base, 10 + 3 * pages)
    elif mission == "assembly":
        base = max(base, 10 + 6 * n)
    elif mission == "review":
        # the reviewer legitimately renders a preview per step — a fixed cap
        # starves longer manuals before findings are written
        base = max(base, 8 + 2 * len(state.steps) + n)
    elif mission == "annotate":
        # ~2 calls per step (arrow per placement + verify render/list) —
        # a fixed cap dies mid-annotation on longer manuals
        base = max(base, 10 + 2 * len(state.steps) + n)
    elif mission == "steps":
        # every manual step costs ~4-6 calls (add_step + sub_steps +
        # assign_media); scale by page count — the steps do not exist yet
        pages = len((state.document.pages if state.document else []) or [])
        base = max(base, 30 + 10 * pages)
    elif mission == "parts":
        # one register per uploaded asset + aliases/matches per required
        # manual name — a fixed cap starves manuals with many BOM entries
        names = len((state.document.detected_part_names if state.document else []) or [])
        names += sum(len(st.required_parts) for st in state.steps)
        base = max(base, 15 + 5 * names)
    elif mission == "build":
        # free Blender modeling: ~12-15 calls per distinct part (exec
        # iterations + screenshot checks + register) plus one render per
        # step — a fixed cap starves manuals with many part types
        names = len((state.document.detected_part_names if state.document else []) or [])
        names += sum(len(st.required_parts) for st in state.steps)
        base = max(base, 30 + 12 * names + 2 * len(state.steps))
    return base


def _mission_prompt(
    mission: str,
    state: GuideState,
    repair_notes: str = "",
    sess: Any = None,
    ctx: Any = None,
) -> str:
    sys_p = system_prompt(f"guide_{mission}.md")
    parts = [{"role": "system", "content": sys_p}]
    ctx_lines = [
        f"Guide language: {state.language}",
        f"User notes ({len(state.user_notes)}): {'; '.join(state.user_notes[:6]) or 'none'}",
    ]
    if mission == "assembly" and sess is not None:
        # After a truncation/retry or context condensation the agent must
        # see WHERE the assembly stands — otherwise it re-explores steps
        # and burns the call budget before finishing.
        from cadx.guide.assembly_tools import _completeness_issues

        plan_fail = (state.assembly.planner or {}).get("failed") or []
        if plan_fail:
            ctx_lines.append(
                "Deterministic pre-assembly could NOT place these — solve "
                "them yourself: "
                + "; ".join(
                    f"step {f['step']} '{f['part']}' ({f.get('reason', '')})" for f in plan_fail[:6]
                )
            )
        if state.assembly.instances:
            issues = _completeness_issues(sess)
            ctx_lines.append(
                f"Assembly progress: {len(state.assembly.instances)} instance(s) placed."
                + (
                    f" Still missing: {'; '.join(issues[:8])}"
                    if issues
                    else " All required instances are present — verify with "
                    "validate_assembly_completeness, then call done_mission. "
                    "Do NOT re-read steps/pages."
                )
            )
    if mission == "parts" and state.parts:
        ctx_lines.append(
            "Already registered (persisted — do NOT register again; "
            "register_uploaded_part on a covered asset errors): "
            + ", ".join(f"{p.id} ({p.name})" for p in state.parts[:30])
        )
    if mission == "steps" and not state.allow_part_generation:
        kit = ", ".join(
            a["filename"] for a in (_guide_assets(ctx) if ctx else []) if a["kind"] != "pdf"
        ) or "(no CAD uploads)"
        ctx_lines.append(
            "PART GENERATION IS DISABLED for this run — the kit is fixed "
            f"to the uploaded CAD assets: {kit}. The parts mission "
            "registers them later, so list_parts stays empty now — these "
            "filenames are the whole kit. A manual step that installs a "
            "part the kit does not contain keeps that part in `text`/"
            "`target` as context — NEVER in required_parts or a sub_step "
            "`part`."
        )
    if mission == "steps":
        detected = getattr(state.document, "detected_part_qty", None) or {}
        if detected:
            ctx_lines.append(
                "Stückliste quantities the manual itself states — the plan "
                "must NEVER consume more copies of a part than this count: "
                + ", ".join(f"{k}={v}x" for k, v in detected.items())
            )
    if mission == "parts" and not state.allow_part_generation:
        ctx_lines.append(
            "PART GENERATION IS DISABLED for this run: every required name "
            "must resolve to an UPLOADED part via register_uploaded_part + "
            "alias_part. Manuals often name the same fitting differently "
            "in the parts list vs. step text — alias those variants onto "
            "the same part instead of treating them as missing."
        )
    if state.pending_answers:
        ctx_lines.append(
            "ANSWERS from the user (read them now via list_user_notes, they "
            "are also pre-pended here): " + " | ".join(state.pending_answers)
        )
    if repair_notes:
        ctx_lines.append(f"REPAIR ITERATION — reviewer flagged your mission:\n{repair_notes}")
    parts.append({"role": "user", "content": "\n".join(ctx_lines)})
    return parts


def _guide_assets(ctx: AgentCtx) -> list[dict[str, Any]]:
    return [
        {
            "id": a.asset_id,
            "filename": Path(a.path).name,
            "kind": a.kind,
            "path": a.path,
        }
        for a in ctx.assets
    ]


async def _emit_artifacts(
    state: GuideState, state_dir: Path, executor: Any = None
) -> dict[str, str]:
    """Deterministic emit mission — write unity_guide.json + media."""
    out_dir = state_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    guide = unity.build_unity_guide(state)
    guide_path = out_dir / "unity_guide.json"
    guide_path.write_text(json.dumps(guide, indent=1), encoding="utf-8")
    state.unity_guide_path = str(guide_path)
    artifacts = {"guide:unity_guide": storage.relpath(guide_path)}
    # ship each step's media_image alongside the JSON — registered as
    # render artifacts so the app can download them by artifact endpoint
    media_dir = out_dir / "media"
    for step in state.steps:
        src = Path(step.media_image) if step.media_image else None
        if src is None or not src.exists():
            continue
        media_dir.mkdir(exist_ok=True)
        dst = media_dir / src.name
        dst.write_bytes(src.read_bytes())
        artifacts[f"guide:media:{src.name}"] = storage.relpath(dst)
    # blender-mode durable evidence: the scene file, the full-scene GLB
    # and the measured manifest ship alongside the standard artifacts
    if state.guide_mode == "blender":
        for extra, key in (
            (state_dir / "blender" / "guide.blend", "guide:blend"),
            (state_dir / "blender" / "assembly_blender.glb", "guide:assembly_blender_glb"),
        ):
            if extra.exists():
                artifacts[key] = storage.relpath(extra)
        if state.blender_manifest:
            manifest_path = out_dir / "blender_manifest.json"
            manifest_path.write_text(
                json.dumps(state.blender_manifest, indent=1), encoding="utf-8"
            )
            artifacts["guide:blender_manifest"] = storage.relpath(manifest_path)
    artifacts.update(await _emit_assembly_artifacts(state, out_dir, executor))
    state.save(state_dir)
    return artifacts


_FINAL_VIEWS = ("iso", "front", "top")


async def _emit_assembly_artifacts(
    state: GuideState, out_dir: Path, executor: Any
) -> dict[str, str]:
    """Finished-assembly deliverables: one merged GLB (drives the version
    page's interactive 3D viewer) plus a fixed set of render views so the
    gallery always shows the final build — independent of which renders the
    agent happened to request mid-mission."""
    artifacts: dict[str, str] = {}
    placements: list[tuple[str, str, list]] = []
    entries: list[dict[str, Any]] = []
    for i, inst in enumerate(state.assembly.instances):
        part = state.part_by_id(inst.part_id)
        mesh_path = getattr(part, "mesh_path", None)
        if not part or not mesh_path:
            continue
        placements.append((inst.instance_id, mesh_path, inst.transform))
        entries.append(
            {
                "mesh_path": mesh_path,
                "transform": inst.transform,
                "color": scene.PALETTE[i % len(scene.PALETTE)],
                "opacity": 1.0,
            }
        )
    if not placements:
        return artifacts
    glb = scene.merged_glb(placements, out_dir / "assembly.glb")
    if glb is not None:
        artifacts["export:glb"] = storage.relpath(glb)
    if executor is not None:
        render_dir = out_dir / "renders"
        for view in _FINAL_VIEWS:
            out = render_dir / f"assembly_{view}.png"
            try:
                res = await executor.render_assembly(
                    {"entries": entries, "view": view, "out_path": str(out)}
                )
            except Exception as exc:
                log.warning("guide.emit_render_failed", view=view, error=str(exc)[:200])
                continue
            if res.get("ok") and res.get("image_path"):
                artifacts[f"render:assembly_{view}"] = storage.relpath(Path(str(res["image_path"])))
    return artifacts


def _load_or_init_state(
    ctx: AgentCtx,
    state_dir: Path,
    resume_state: dict[str, Any],
    allow_generation: bool,
    language: str = "",
    audience: str = "",
    mode: str = "cad",
) -> GuideState:
    state = GuideState.load(state_dir)
    if state is None:
        lang = language.strip().lower()
        state = GuideState(
            project_id=str(ctx.project_id or ""),
            guide_version_id=str(ctx.version_id),
            output_dir=str(state_dir),
            allow_part_generation=allow_generation,
            language=lang or "de",
            language_locked=bool(lang),
            audience=audience.strip(),
            guide_mode=mode if mode in ("cad", "blender", "auto") else "cad",
        )
        # the user's request is durable context for EVERY mission —
        # without this the prompt only lives on the version row and the
        # extraction/assembly agents never see requirements like
        # "watch the gasket washers" or "crimp all joints at the end"
        if ctx.prompt and ctx.prompt.strip():
            state.user_notes.append(ctx.prompt.strip())
        state.save(state_dir)
    # resume: fold the answered clarification into pending_answers
    if resume_state.get("answers_text"):
        state.pending_answers.append(str(resume_state["answers_text"]))
        state.save(state_dir)
    return state


def _review_verdict(
    state: GuideState, state_dir: Path, score_target: float, max_iter: int
) -> tuple[str, str]:
    """(verdict, repair_notes) — verdict ∈ ship|repair|fail.

    Only a clean verdict ships: score ≥ target AND zero blockers. Anything
    else is a bounded repair round (write_review_report's consistency gate
    guarantees a sub-target score always carries blocker findings). At the
    iteration cap with unresolved issues the verdict is FAIL — a guide that
    never earned its score must not exit 'succeeded'."""
    rep = state.review
    blockers = [f for f in rep.findings if f.severity == "blocker"]
    if rep.iteration == 0:
        # review mission never produced a report — cannot judge, cannot
        # ship; one repair round re-runs it (outer belt bounds this)
        if "review" in state.missions_done:
            state.missions_done.remove("review")
        state.save(state_dir)
        log.warning("guide.review_missing", missions_done=state.missions_done)
        return "repair", "the review produced no report — re-run it and call write_review_report"
    if rep.score >= score_target and not blockers:
        return "ship", ""
    # every blocker is a kit-limitation the pipeline cannot repair — the
    # parts mission cannot register hardware that is not in the kit and
    # the steps mission cannot conjure it either. Fail fast into
    # needs_review instead of burning 3 futile repair iterations
    # (two_pipes: 800s of steps-mission thrash, still failed).
    if blockers and all(
        "kit" in (f.evidence or "").lower() and "limit" in (f.evidence or "").lower()
        for f in blockers
    ):
        log.info(
            "guide.review_kit_limited",
            blockers=len(blockers),
            score=rep.score,
            iter=rep.iteration,
        )
        return "fail", ""
    if rep.iteration >= max_iter:
        log.warning(
            "guide.review_iter_cap",
            iter=rep.iteration,
            score=rep.score,
            blockers=len(blockers),
        )
        return "fail", ""
    order = _mission_order(state)
    flagged = {f.mission for f in blockers if f.mission in order} or {
        "build" if state.guide_mode == "blender" else "assembly"
    }
    notes = "\n".join(f"- [{f.severity}] {f.text}" for f in blockers)[:2000]
    # "review" must reopen too — the repaired assembly needs a FRESH verdict
    # (rep.iteration only advances inside write_review_report; keeping review
    # in missions_done re-reads the stale rep forever = infinite repair loop)
    for m in flagged | {"review", "annotate"}:
        if m in state.missions_done:
            state.missions_done.remove(m)
    state.save(state_dir)
    log.info(
        "guide.review_repair",
        iter=rep.iteration,
        score=rep.score,
        blockers=len(blockers),
        reopen=sorted(flagged),
    )
    return "repair", notes


def _plan_lint_reopen(
    lint_sess: Any, state: GuideState, state_dir: Path, *, allow_reopen: bool
) -> tuple[str | None, list[str]]:
    """Deterministic plan lint before assembly spends its budget —
    unresolvable refs route back to the owning mission (steps|parts) with
    the lint text as repair notes. Returns (notes, fatal_blockers).

    When allow_reopen is False (cycle cap hit) the blockers are FATAL:
    the plan is known-invalid and only the capped owning missions could
    fix it — assembly tools (place_instance/solve_attachment) can never
    register a missing host part or rewrite a step ref. Reopening without
    a break would re-run steps/parts on every pass (observed: 25→18→18
    blocker thrash without progress), and letting assembly run anyway
    burns its whole budget placing a plan that cannot succeed."""
    from cadx.guide.planlint import plan_lint

    findings = plan_lint(lint_sess)
    blockers = [f for f in findings if f["severity"] == "blocker"]
    # durable lint evidence — every cycle is recorded, including the clean
    # one that converged the plan; the blocker texts are the forensic
    # trail of which reference failed and which mission owned the fix
    n_warnings = sum(1 for f in findings if f["severity"] != "blocker")
    state.lint_cycles = [
        *state.lint_cycles[-11:],
        {
            "cycle": (state.lint_cycles[-1].get("cycle", 0) + 1)
            if state.lint_cycles
            else 1,
            "blockers": [f["text"] for f in blockers],
            "missions": sorted({f["mission"] for f in blockers}),
            "warnings": n_warnings,
        },
    ]
    if not blockers:
        state.save(state_dir)
        return None, []
    texts = [f["text"] for f in blockers]
    flagged = {f["mission"] for f in blockers}
    return _gate_reopen(state, state_dir, texts, flagged, allow_reopen=allow_reopen)


def _gate_reopen(
    state: GuideState,
    state_dir: Path,
    blocker_texts: list[str],
    flagged: set[str],
    *,
    allow_reopen: bool,
) -> tuple[str | None, list[str]]:
    """Reopen the missions a gate flagged with the blocker texts as notes,
    or report the blockers as fatal once the reopen cap is spent.

    In blender mode lint findings name cad-side missions — "parts" and
    "assembly" both collapse onto the "build" mission that owns them."""
    if state.guide_mode == "blender":
        flagged = {"build" if m in ("parts", "assembly") else m for m in flagged}
    if not allow_reopen:
        log.info(
            "guide.gate_uncapped",
            mode=state.guide_mode,
            blockers=len(blocker_texts),
            texts=blocker_texts,
        )
        state.save(state_dir)
        return None, blocker_texts
    for m in flagged:
        if m in state.missions_done:
            state.missions_done.remove(m)
    state.save(state_dir)
    log.info(
        "guide.gate_reopen",
        mode=state.guide_mode,
        reopen=sorted(flagged),
        blockers=len(blocker_texts),
    )
    return "\n".join(f"- {t}" for t in blocker_texts), []


async def _plan_gate(
    mission: str, lint_sess: Any, state: GuideState, state_dir: Path, *, lint_cycles: int
) -> RunOutcome | str | None:
    """Plan gate before the state-consuming mission — cad mode gates
    before ``assembly``, blender mode before ``review`` (parts only exist
    after the build mission registers them). Returns a failed RunOutcome
    when blockers outlive the reopen cap, repair notes when the owning
    missions were reopened, or None on a clean plan."""
    gate_at = "review" if state.guide_mode == "blender" else "assembly"
    if mission != gate_at or mission in state.missions_done:
        return None
    # blender mode first re-reads the scene: the placement truth lives in
    # Blender, and only measured scene facts may gate the run
    if state.guide_mode == "blender":
        from cadx.guide import blender_check

        scene_blockers, manifest = await blender_check.run_scene_gate(lint_sess)
        state.blender_manifest = manifest
        state.save(state_dir)
        if scene_blockers:
            notes, fatal = _gate_reopen(
                state,
                state_dir,
                scene_blockers,
                {"build"},
                allow_reopen=lint_cycles < 2,
            )
            if fatal:
                return RunOutcome(
                    status="failed",
                    error="blender scene gate blockers persist after "
                    f"{lint_cycles} reopen cycles — unrepairable by the "
                    "build mission: " + " | ".join(fatal),
                )
            return notes
    notes, fatal = _plan_lint_reopen(
        lint_sess, state, state_dir, allow_reopen=lint_cycles < 2
    )
    if fatal:
        return RunOutcome(
            status="failed",
            error="plan lint blockers persist after "
            f"{lint_cycles} reopen cycles — unrepairable by "
            "the owning missions: " + " | ".join(fatal),
        )
    return notes


async def _mission_early(
    sess: Any, ctx: AgentCtx, state: GuideState, state_dir: Path
) -> RunOutcome | None:
    """parked/cancelled -> outcome; mission still running -> None."""
    if sess.parked:
        state.save(state_dir)
        return RunOutcome(
            status="waiting_for_input",
            questions=sess.parked_questions,
            resume_state={"guide_dir": str(state_dir)},
        )
    if ctx.check_cancel is not None and await ctx.check_cancel():
        return RunOutcome(status="cancelled")
    return None


async def _run_mission(
    mission: str,
    notes: str,
    *,
    ctx: AgentCtx,
    state: GuideState,
    state_dir: Path,
    mk_session: Any,
) -> RunOutcome | None:
    """One mission turn. None = mission completed; else early-exit
    outcome (parked/cancelled/failed).

    A bare final-answer response (the model writing prose instead of
    calling tools) ends the session WITHOUT done_mission — that is a
    wobble, not a mission failure, and the incrementally-written state
    survives it. Retry once with a continuation nudge before failing."""
    if mission in state.missions_done:
        return None
    last_failures = ""
    last_reason = ""
    # the durable assembly is written per tool call — a mission that dies
    # mid-repair must not leave a half-stripped scene behind, so keep one
    # pre-session copy and restore it on ANY failure. Taken once, right
    # after preassembly: a retry must restore the deterministic floor, not
    # the half-stripped state the crashed first attempt left behind.
    assembly_snapshot = None
    for attempt in (0, 1):
        sess = mk_session(mission)
        if mission == "assembly" and not state.assembly.planner.get("ran"):
            # deterministic pre-assembly: place everything the step plan
            # resolves unambiguously — the LLM session below only refines
            from cadx.guide.preassembly import run_preassembly

            try:
                await run_preassembly(sess)
            except Exception:
                log.exception("guide.preassembly.failed")
                state.assembly.planner = {"ran": True, "error": "planner crashed — LLM assembles"}
        if mission == "assembly" and assembly_snapshot is None:
            assembly_snapshot = state.assembly.model_copy(deep=True)
        nudge = notes
        if attempt:
            cause = (
                f"it exhausted its tool-call budget ({last_reason}) — work "
                "efficiently this time: batch reads, no redundant checks, "
                "finish the mission promptly"
                if last_reason in ("budget", "turns")
                else "it ended without done_mission "
                f"({last_reason or 'no reason'})"
            )
            nudge = (f"{notes}\n\n" if notes else "") + (
                f"CONTINUATION: the previous session is gone — {cause}. "
                "The durable state keeps everything already applied — do "
                "NOT redo completed work. Resume the remaining work now "
                "and finish with done_mission."
            )
            if last_failures:
                nudge += (
                    "\nThe previous attempt also hit these tool errors — "
                    "do NOT repeat the same calls verbatim:\n" + last_failures
                )
            if mission == "assembly":
                # scoped repair list — an unscoped retry makes the model
                # re-audit the WHOLE scene and "fix" correct placements,
                # spawning duplicate instances (adapter_rn22#4 observed).
                # Name exactly what remains and what is stray.
                try:
                    from cadx.guide.assembly_tools import (
                        _completeness_issues,
                        joint_geometry_audit,
                    )

                    open_items = _completeness_issues(sess)
                    stray = [
                        d["instance"]
                        for d in joint_geometry_audit(sess)["disconnected_instances"]
                    ]
                    scope: list[str] = []
                    if open_items:
                        scope.append(
                            "MISSING placements still to finish:\n"
                            + "\n".join(f"  - {m}" for m in open_items)
                        )
                    if stray:
                        scope.append(
                            f"STRAY instances not connected to the anchor "
                            f"— unplace_instance or re-mate THESE, never add "
                            f"extra copies of their part: {stray}"
                        )
                    if scope:
                        nudge += (
                            "\nScoped repair list — ONLY fix these; every "
                            "other placed instance is already correct and "
                            "must stay untouched (no unplace/duplicates):\n"
                            + "\n".join(scope)
                        )
                except Exception:
                    log.exception("guide.assembly_scope_failed")
        calls_budget = _calls_budget(mission, state)
        try:
            result = await run_session(
                ctx,
                role="vision",
                messages=_mission_prompt(mission, state, nudge, sess, ctx),
                session=sess,
                max_calls=calls_budget,
                turn_budget=calls_budget,  # ~1 tool call per turn — the call cap is the real bound
                step=f"guide_{mission}",
            )
        except LLMError as exc:
            # infrastructure failure, not a model failure: the durable state
            # survives (saved per tool call) — record the outage as this
            # attempt's reason so the retry nudge + failure log name it
            log.warning(
                "guide.mission_llm_error",
                mission=mission,
                attempt=attempt,
                error=str(exc)[:200],
            )
            result = SessionResult(
                end_reason=f"llm_error: {str(exc)[:120]}",
                tool_log=[],
            )
        _persist_tool_log(state_dir, mission, attempt, result)
        early = await _mission_early(sess, ctx, state, state_dir)
        if early is not None:
            return early
        if mission in state.missions_done:
            return None
        fails = _tool_failures(result)
        last_reason = result.end_reason or ""
        if attempt == 0:
            last_failures = "\n".join(f"- {f}" for f in fails)
            log.info(
                "guide.mission_retry",
                mission=mission,
                reason=last_reason,
                failures=fails,
            )
            continue
        if assembly_snapshot is not None:
            state.assembly = assembly_snapshot
            log.info("guide.assembly_restored", mission=mission)
        state.mark_failed(mission)
        state.save(state_dir)
        if mission in _OPTIONAL:
            # an optional mission that never finished degrades to a durable
            # warning — the review verdict already gated quality
            log.warning(
                "guide.mission_skipped",
                mission=mission,
                reason=last_reason,
                failures=fails,
            )
            return None
        log.warning(
            "guide.mission_failed",
            mission=mission,
            reason=last_reason,
            failures=fails,
        )
        return RunOutcome(
            status="failed",
            error=f"mission '{mission}' ended without done_mission "
            f"({result.end_reason or 'no reason'})",
        )
    return None


async def _maybe_apply_audience_rewrite(
    state: GuideState, state_dir: Path, ctx: AgentCtx
) -> None:
    """Spawn-time audience choice: rewrite the step text BEFORE emit so
    the shipped unity_guide.json already carries the variant. One extra
    LLM call — a failure must never kill a finished guide, ship the
    original."""
    if not state.audience:
        return
    if ctx.emit is not None:
        with contextlib.suppress(Exception):
            await ctx.emit("guide_rewrite", 0, f"rewrite: {state.audience}")
    try:
        from cadx.guide.rewrite import apply_rewrite

        rw = await apply_rewrite(
            state,
            audience=state.audience,
            language=state.language,
            llm=ctx.llm,
        )
        log.info(
            "guide.audience_rewritten",
            audience=state.audience,
            variant=rw["key"],
            applied=rw["applied"],
        )
    except Exception as exc:
        log.warning("guide.audience_rewrite_failed", error=str(exc)[:300])
    state.save(state_dir)


async def run_guide(
    ctx: AgentCtx,
    *,
    state_dir: Path,
    resume_state: dict[str, Any],
    allow_generation: bool = True,
    attach_images: bool = True,
    language: str = "",
    audience: str = "",
    mode: str = "cad",
    park_job: Any = None,
) -> RunOutcome:
    """Mission loop. ``park_job`` (optional) is invoked with the question
    payload when a mission calls ask_user — it persists job.questions etc;
    either way the runner returns waiting_for_input."""
    state = _load_or_init_state(
        ctx, state_dir, resume_state, allow_generation, language, audience, mode
    )
    assets = _guide_assets(ctx)
    _resolve_auto_mode(state, assets, state_dir)
    mission_order = _mission_order(state)
    rt = ctx.runtime_settings
    score_target = float(getattr(rt, "guide_review_score_target", 8.0))
    max_iter = int(getattr(rt, "guide_review_max_iter", 3))

    async def _park(questions: list[dict[str, Any]]) -> None:
        if park_job is not None:
            await park_job(questions)

    def _session(mission: str) -> GuideToolSession:
        return GuideToolSession(
            ctx,
            mission=mission,
            state=state,
            state_dir=state_dir,
            project_id=ctx.project_id or ctx.version_id,
            guide_assets=assets,
            executor=ctx.executor,
            attach_images=attach_images,
            emit=ctx.emit,
            park_fn=_park,
            check_cancel=ctx.check_cancel,
            enqueue_fn=getattr(ctx, "enqueue_fn", None),
        )

    iterations = 0
    repair_notes = ""
    lint_cycles = 0
    best_score = -1.0
    best_assembly = None
    failed_review = False
    while True:
        lint_reopened = False
        for mission in mission_order:
            # deterministic plan lint before the state-consuming mission
            # spends its budget (cad: assembly, blender: review — the
            # blender gate also re-measures the scene first):
            # unresolvable refs / ordering bugs are routed back to the
            # owning mission (steps|parts|build) with the lint text as
            # notes — a degenerate extraction gets fixed, not faithfully
            # placed
            gate = await _plan_gate(
                mission, _session("assembly"), state, state_dir, lint_cycles=lint_cycles
            )
            if isinstance(gate, RunOutcome):
                return gate
            if gate:  # lint reopened the owning missions — restart order
                lint_cycles += 1
                repair_notes = gate
                lint_reopened = True
                log.info("guide.plan_lint_cycle", cycle=lint_cycles)
                break
            # findings go to the missions that actually re-run — a skipped
            # (already-done) mission must not consume them
            notes = repair_notes if mission not in state.missions_done else ""
            early = await _run_mission(
                mission, notes, ctx=ctx, state=state, state_dir=state_dir, mk_session=_session
            )
            if early is not None:
                return early

        # lint reopened owning missions — restart the mission order WITHOUT
        # consulting the review verdict (there may be no report yet)
        if lint_reopened:
            continue
        verdict, repair_notes = _review_verdict(state, state_dir, score_target, max_iter)
        # keep the best-SCORED assembly across repair iterations — an LLM
        # repair that "succeeds" can still degrade the build (7.0 -> 4.0 in
        # eval round 5), and the failure-restore only guards mission death
        if state.assembly.instances and state.review.score > best_score:
            best_score = state.review.score
            best_assembly = state.assembly.model_copy(deep=True)
        if verdict == "ship":
            break
        iterations += 1
        if verdict == "fail" or iterations >= max_iter:
            # iter-cap with open blockers, or the belt catching a review
            # that never produced a verdict — either way the guide never
            # earned its score and must not exit 'succeeded'
            if verdict != "fail":
                log.warning("guide.repair_loop_cap", max_iter=max_iter)
            failed_review = True
            break

    # ship the best-scored geometry, not the last repair attempt — a
    # regressed final iteration must not downgrade the build
    if best_assembly is not None and state.review.score < best_score:
        log.info(
            "guide.assembly_best_restored",
            restored_score=best_score,
            final_score=state.review.score,
        )
        state.assembly = best_assembly
        state.review.score = best_score
        state.save(state_dir)

    await _maybe_apply_audience_rewrite(state, state_dir, ctx)

    artifacts = await _emit_artifacts(state, state_dir, ctx.executor)
    unresolved = [
        f.text for f in state.review.findings if f.severity == "blocker"
    ][:8]
    outcome = RunOutcome(
        # a review that never reached target with blockers open ships as
        # needs_review — evidence stays, the verdict stays honest
        status="needs_review" if failed_review else "succeeded",
        # review.score is 0..10; version.score (and every other outcome
        # score) is 0..100 — scale or the header renders "9/100"
        score=round(state.review.score * 10),
        iterations=iterations,
        plan={
            "missions_done": state.missions_done,
            "steps": len(state.steps),
            "parts": len(state.parts),
            "instances": len(state.assembly.instances),
            "review_iterations": state.review.iteration,
            "unresolved_blockers": unresolved,
        },
    )
    outcome.artifacts.update(artifacts)
    return outcome
