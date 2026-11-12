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

import json
from pathlib import Path
from typing import Any

from cadx import storage
from cadx.agent.context import system_prompt
from cadx.agent.orchestrator import RunOutcome
from cadx.agent.toolloop import run_session
from cadx.agent.types import AgentCtx
from cadx.guide import unity
from cadx.guide.session import GuideToolSession
from cadx.guide.state import GuideState
from cadx.logging import get_logger

log = get_logger(__name__)

MISSION_ORDER = ["doc", "steps", "parts", "assembly", "review", "annotate"]
_MAX_CALLS = {"doc": 24, "steps": 60, "parts": 40, "assembly": 80, "review": 24, "annotate": 24}


def _mission_prompt(mission: str, state: GuideState, repair_notes: str = "") -> str:
    sys_p = system_prompt(f"guide_{mission}.md")
    parts = [{"role": "system", "content": sys_p}]
    ctx_lines = [
        f"Guide language: {state.language}",
        f"User notes ({len(state.user_notes)}): {'; '.join(state.user_notes[:6]) or 'none'}",
    ]
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


async def _emit_artifacts(state: GuideState, state_dir: Path) -> dict[str, str]:
    """Deterministic emit mission — write unity_guide.json + diagnostics."""
    out_dir = state_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    guide = unity.build_unity_guide(state)
    guide_path = out_dir / "unity_guide.json"
    guide_path.write_text(json.dumps(guide, indent=1), encoding="utf-8")
    state.unity_guide_path = str(guide_path)
    state.save(state_dir)
    return {"guide:unity_guide": storage.relpath(guide_path)}


def _load_or_init_state(ctx: AgentCtx, state_dir: Path, resume_state: dict[str, Any]) -> GuideState:
    state = GuideState.load(state_dir)
    if state is None:
        state = GuideState(
            project_id=str(ctx.project_id or ""),
            guide_version_id=str(ctx.version_id),
            output_dir=str(state_dir),
        )
        state.save(state_dir)
    # resume: fold the answered clarification into pending_answers
    if resume_state.get("answers_text"):
        state.pending_answers.append(str(resume_state["answers_text"]))
        state.save(state_dir)
    return state


def _review_verdict(
    state: GuideState, state_dir: Path, score_target: float, max_iter: int
) -> tuple[bool, str]:
    """(should_ship, repair_notes). Reopens flagged missions as a side
    effect when a bounded repair iteration is warranted."""
    rep = state.review
    blockers = [f for f in rep.findings if f.severity == "blocker"]
    if (
        rep.score >= score_target
        or (not blockers and rep.iteration > 0)
        or rep.iteration >= max_iter
    ):
        return True, ""
    flagged = {f.mission for f in blockers if f.mission in MISSION_ORDER} or {"assembly"}
    notes = "\n".join(f"- [{f.severity}] {f.text}" for f in blockers)[:2000]
    # "review" must reopen too — the repaired assembly needs a FRESH verdict
    # (rep.iteration only advances inside write_review_report; keeping review
    # in missions_done re-reads the stale rep forever = infinite repair loop)
    for m in flagged | {"review", "annotate"}:
        if m in state.missions_done:
            state.missions_done.remove(m)
    state.save(state_dir)
    log.info("guide.review_repair", iter=rep.iteration, score=rep.score, reopen=sorted(flagged))
    return False, notes


async def run_guide(
    ctx: AgentCtx,
    *,
    state_dir: Path,
    resume_state: dict[str, Any],
    park_job: Any = None,
) -> RunOutcome:
    """Mission loop. ``park_job`` (optional) is invoked with the question
    payload when a mission calls ask_user — it persists job.questions etc;
    either way the runner returns waiting_for_input."""
    state = _load_or_init_state(ctx, state_dir, resume_state)
    assets = _guide_assets(ctx)
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
            attach_images=True,
            emit=ctx.emit,
            park_fn=_park,
            check_cancel=ctx.check_cancel,
            enqueue_fn=getattr(ctx, "enqueue_fn", None),
        )

    async def _run_mission(mission: str, notes: str) -> RunOutcome | None:
        """One mission turn. None = mission completed; else early-exit
        outcome (parked/cancelled/failed)."""
        if mission in state.missions_done:
            return None
        sess = _session(mission)
        result = await run_session(
            ctx,
            role="vision",
            messages=_mission_prompt(mission, state, notes),
            session=sess,
            max_calls=_MAX_CALLS.get(mission, 40),
            step=f"guide_{mission}",
        )
        if sess.parked:
            state.save(state_dir)
            return RunOutcome(
                status="waiting_for_input",
                questions=sess.parked_questions,
                resume_state={"guide_dir": str(state_dir)},
            )
        if ctx.check_cancel is not None and await ctx.check_cancel():
            return RunOutcome(status="cancelled")
        if mission not in state.missions_done:
            state.mark_failed(mission)
            state.save(state_dir)
            return RunOutcome(
                status="failed",
                error=f"mission '{mission}' ended without done_mission "
                f"({result.end_reason or 'no reason'})",
            )
        return None

    iterations = 0
    repair_notes = ""
    while True:
        for mission in MISSION_ORDER:
            # findings go to the missions that actually re-run — a skipped
            # (already-done) mission must not consume them
            notes = repair_notes if mission not in state.missions_done else ""
            early = await _run_mission(mission, notes)
            if early is not None:
                return early

        ship, repair_notes = _review_verdict(state, state_dir, score_target, max_iter)
        if ship:
            break
        iterations += 1
        if iterations >= max_iter:
            # belt: rep.iteration bounds this normally; if review can't
            # complete a verdict we must not loop forever
            log.warning("guide.repair_loop_cap", max_iter=max_iter)
            break

    artifacts = await _emit_artifacts(state, state_dir)
    outcome = RunOutcome(
        status="succeeded",
        score=round(state.review.score),
        iterations=iterations,
        plan={
            "missions_done": state.missions_done,
            "steps": len(state.steps),
            "parts": len(state.parts),
            "instances": len(state.assembly.instances),
            "review_iterations": state.review.iteration,
        },
    )
    outcome.artifacts.update(artifacts)
    return outcome
