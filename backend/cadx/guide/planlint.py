"""Deterministic plan lint — runs AFTER the parts mission registered every
uploaded/generated part and BEFORE the assembly mission spends its budget.

LLM extraction is the pipeline's biggest variance source: the same manual
yields different step plans per run — wrong part names, split bridge legs,
out-of-order targets, phantom instance counts. Deterministic placement
faithfully executes even a degenerate plan, so the review finds the wreck
and the repair mission then fights plan contradictions with geometry
tools it cannot fix.

The lint checks what a plan can contradict BEFORE any placement:
- every placement sub-step's part/target/target2 must resolve to a
  registered part or alias — unresolvable names route to the ``parts``
  mission (alias/register coverage) or ``steps`` (fix the reference text)
- a target must be anchored or placed no later than the step using it —
  a target whose earliest placement is a LATER step is an ordering bug
  the ``steps`` mission must fix (move the step or the reference)
- every assembly step should place something — an empty assembly step is
  a warning (check/explain steps are legitimately empty)

Findings carry the mission that owns the fix; the runner reopens exactly
those missions with the lint text as repair notes.
"""

from __future__ import annotations

from typing import Any

from structlog import get_logger

from cadx.guide.assembly_tools import (
    _PLACE_ACTIONS,
    _name_index,
    _norm2,
    _resolve_name,
)
from cadx.guide.session import GuideToolSession

log = get_logger(__name__)


def _resolve_part(sess: GuideToolSession, name: str | None, step_no: int) -> str | None:
    """Part resolution with step context — mirrors preassembly._rp."""
    if not name:
        return None
    st = sess.state.step_by_number(step_no)
    ctx = f"{st.text} {st.title}" if st else ""
    return _resolve_name(sess, name, context=ctx)


def _suggest(sess: GuideToolSession, name: str) -> str:
    """Closest registered name/alias for an unresolvable reference — gives
    the repair mission a concrete target instead of a dead end."""
    import difflib

    keys = list(sess._aliases_store) + list(_name_index(sess))
    close = difflib.get_close_matches(_norm2(name), keys, n=1, cutoff=0.6)
    if close:
        pid = sess._aliases_store.get(close[0]) or _name_index(sess).get(close[0])
        return f" — did you mean '{pid or close[0]}'?"
    ids = [p.id for p in sess.state.parts][:8]
    return f" — registered parts: {ids}"


def _lint_refs(
    sess: GuideToolSession, steps: list[Any]
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Name resolution: unresolvable placement parts route to ``parts``
    (register/alias the asset), unresolvable targets to ``steps`` (fix the
    reference text). Also builds the earliest-placement map for ordering."""
    findings: list[dict[str, Any]] = []
    first_place: dict[str, int] = {}
    for st in steps:
        for i, ss in enumerate(st.sub_steps):
            # only PLACEMENT targets must resolve — check/orient targets are
            # context anchors (the pre-existing host, walls, "Wärmepumpe")
            # that legitimately match no kit part
            for name in (ss.target, ss.target2):
                if name and ss.action in _PLACE_ACTIONS and not _resolve_name(sess, name):
                    findings.append(
                        {
                            "mission": "steps",
                            "severity": "blocker",
                            "text": f"step {st.number} sub {i}: target '{name}' "
                            "matches no registered part"
                            + _suggest(sess, name)
                            + " — fix the reference text or alias the part",
                        }
                    )
            if ss.action in _PLACE_ACTIONS:
                pid = _resolve_part(sess, ss.part, st.number)
                if pid:
                    first_place.setdefault(pid, st.number)
                else:
                    findings.append(
                        {
                            "mission": "parts",
                            "severity": "blocker",
                            "text": f"step {st.number} sub {i}: part '{ss.part}' "
                            "matches no registered part"
                            + _suggest(sess, ss.part)
                            + " — register/alias the asset or fix the text",
                        }
                    )
    # anchors: referenced but never placed — preassembly places them at the
    # earliest referencing step, so they satisfy targets from there on
    anchor_step: dict[str, int] = {}
    for st in steps:
        for ss in st.sub_steps:
            for name in (ss.target, ss.target2):
                pid = _resolve_name(sess, name or "") if name else None
                if pid and pid not in first_place:
                    anchor_step.setdefault(pid, st.number)
    return findings, first_place, anchor_step


def _lint_order(
    sess: GuideToolSession,
    steps: list[Any],
    first_place: dict[str, int],
    anchor_step: dict[str, int],
) -> list[dict[str, Any]]:
    """A PLACED (non-anchor) target must exist no later than the step
    referencing it — targeting a part that only appears later is a
    plan-order bug the steps mission must resolve."""
    findings: list[dict[str, Any]] = []
    for st in steps:
        for i, ss in enumerate(st.sub_steps):
            if ss.action not in _PLACE_ACTIONS:
                continue
            for name in (ss.target, ss.target2):
                if not name:
                    continue
                pid = _resolve_name(sess, name)
                if pid is None or pid in anchor_step:
                    continue
                first = first_place.get(pid)
                if first is not None and first > st.number:
                    findings.append(
                        {
                            "mission": "steps",
                            "severity": "blocker",
                            "text": f"step {st.number} sub {i}: targets '{name}' "
                            f"({pid}) but '{pid}' is first placed in step "
                            f"{first} — reorder steps or fix the reference",
                        }
                    )
    for st in steps:
        if (
            st.action_type == "assembly"
            and not any(ss.action in _PLACE_ACTIONS for ss in st.sub_steps)
            and not st.required_parts
        ):
            findings.append(
                {
                    "mission": "steps",
                    "severity": "warning",
                    "text": f"step {st.number} is an assembly step placing "
                    "nothing — convert to check/explain or add sub-steps",
                }
            )
    return findings


def plan_lint(sess: GuideToolSession) -> list[dict[str, Any]]:
    """Plan-vs-registry contradictions, deterministically detected.

    Returns findings as {mission, severity, text} — ``severity="blocker"``
    means placement CANNOT succeed; ``"warning"`` means suspicious but
    legal. Only blockers reopen missions.
    """
    steps = sorted(sess.state.steps, key=lambda s: s.number)
    findings, first_place, anchor_step = _lint_refs(sess, steps)
    findings += _lint_order(sess, steps, first_place, anchor_step)
    blockers = sum(1 for f in findings if f["severity"] == "blocker")
    log.info(
        "guide.plan_lint",
        steps=len(steps),
        parts=len(sess.state.parts),
        blockers=blockers,
        warnings=len(findings) - blockers,
    )
    return findings
