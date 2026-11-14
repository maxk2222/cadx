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
  mission (alias/register coverage — steps runs BEFORE parts, so name
  fixes can only happen there)
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
    """Name resolution: both unresolvable placement parts and targets route
    to ``parts`` — the steps mission has no alias_part (parts haven't been
    registered when steps runs), so name-landscape fixes (alias the manual
    name onto the registered part, or register a missed asset) can only
    happen there. Also builds the earliest-placement map for ordering."""
    findings: list[dict[str, Any]] = []
    first_place: dict[str, int] = {}
    from cadx.guide.part_tools import _is_context

    for st in steps:
        for i, ss in enumerate(st.sub_steps):
            # only PLACEMENT targets must resolve — check/orient targets are
            # context anchors (the pre-existing host, walls, "Wärmepumpe")
            # that legitimately match no kit part
            for name in (ss.target, ss.target2):
                if not name or ss.action not in _PLACE_ACTIONS:
                    continue
                if _is_context(sess, name):
                    findings.append(
                        {
                            "mission": "parts",
                            "severity": "blocker",
                            "text": f"step {st.number} sub {i}: target '{name}' "
                            "is marked context but a placement mates to it — "
                            "the name denotes a FEATURE of an existing part "
                            "(nozzle/port on the host): alias_part "
                            f"'{name}' onto that registered part (e.g. the "
                            "enclosure/base), or mark_context_part it only "
                            "if the substep gets retargeted",
                        }
                    )
                elif not _resolve_name(sess, name):
                    findings.append(
                        {
                            "mission": "parts",
                            "severity": "blocker",
                            "text": f"step {st.number} sub {i}: target '{name}' "
                            "matches no registered part"
                            + _suggest(sess, name)
                            + " — alias_part the name onto the host part "
                            "or register a missed asset",
                        }
                    )
            if ss.action in _PLACE_ACTIONS:
                pid = _resolve_part(sess, ss.part, st.number)
                if pid:
                    first_place.setdefault(pid, st.number)
                elif _is_context(sess, ss.part):
                    findings.append(
                        {
                            "mission": "steps",
                            "severity": "blocker",
                            "text": f"step {st.number} sub {i}: '{ss.part}' is "
                            "marked context — a context name cannot be "
                            "placed; drop the substep or retarget it",
                        }
                    )
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


def _lint_quantities(sess: GuideToolSession) -> list[dict[str, Any]]:
    """Stückliste 'Menge' bounds the total copies a plan may consume — a
    plan needing 5 of a 2-piece part maps the wrong part onto steps (the
    kiara bare run: socket-socket elbows on steps 1/2/5/6/10) and no
    geometry repair can fix that; it is a steps-mission plan bug."""
    from cadx.guide.assembly_tools import _step_requirements

    totals: dict[str, int] = {}
    for counts in _step_requirements(sess).values():
        for pid, n in counts.items():
            totals[pid] = totals.get(pid, 0) + n
    # Stückliste quantities the doc mission captured bound the plan even
    # when the register call omitted `quantity` (heatpump run: qty=None
    # everywhere → 5x l500 placed for a 2-piece kit)
    detected: dict[str, int] = {}
    doc = getattr(sess.state, "document", None)
    for nrm, qty in (getattr(doc, "detected_part_qty", None) or {}).items():
        pid = sess._aliases_store.get(nrm) or _name_index(sess).get(nrm)
        if pid:
            detected[pid] = qty
    findings: list[dict[str, Any]] = []
    for p in sess.state.parts:
        bound = p.quantity if p.quantity is not None else detected.get(p.id)
        if bound is None:
            continue
        need = totals.get(p.id, 0)
        if need > bound:
            findings.append(
                {
                    "mission": "steps",
                    "severity": "blocker",
                    "text": f"plan requires {need}x '{p.id}' but the parts "
                    f"list ships only {bound} — wrong part mapping or "
                    "phantom occurrences; fix the step sub_steps",
                }
            )
    return findings


def _lint_anchor(
    sess: GuideToolSession, steps: list[Any]
) -> list[dict[str, Any]]:
    """The biggest registered part is the assembly's host/base — a plan that
    neither places it nor mates anything to it builds a satellite assembly
    disconnected from the host (observed: adapters stacked on each other,
    zero casing joints). Legit leaf-big-parts (a cover mounted last) are
    the `part` of a placement and skip the flag; guides without any host
    part have nothing to flag either."""
    if len(sess.state.parts) < 2:
        return []  # a lone-part guide has nothing to mate to the base
    from cadx.guide.assembly_tools import _host_part_id

    hid = _host_part_id(sess)
    if hid is None:
        return []  # no clear size outlier — no identifiable host to anchor to
    biggest = next(p for p in sess.state.parts if p.id == hid)
    referenced = False
    for st in steps:
        for ss in st.sub_steps:
            if ss.action not in _PLACE_ACTIONS:
                continue
            if _resolve_part(sess, ss.part, st.number) == biggest.id:
                return []  # the big part is itself mounted — a leaf
            for name in (ss.target, ss.target2):
                if name and _resolve_name(sess, name) == biggest.id:
                    referenced = True
    if referenced:
        return []
    return [
        {
            "mission": "parts",
            "severity": "blocker",
            "text": f"no placement substep mates anything to '{biggest.id}' "
            "— it is the largest part (the host/base the assembly builds "
            "on). Targets naming its features (nozzles, ports, 'Wärmepumpe', "
            "the device itself) must alias_part onto it, or the assembly "
            "ends up floating disconnected",
        }
    ]


def plan_lint(sess: GuideToolSession) -> list[dict[str, Any]]:
    """Plan-vs-registry contradictions, deterministically detected.

    Returns findings as {mission, severity, text} — ``severity="blocker"``
    means placement CANNOT succeed; ``"warning"`` means suspicious but
    legal. Only blockers reopen missions.
    """
    steps = sorted(sess.state.steps, key=lambda s: s.number)
    findings, first_place, anchor_step = _lint_refs(sess, steps)
    findings += _lint_order(sess, steps, first_place, anchor_step)
    findings += _lint_quantities(sess)
    findings += _lint_anchor(sess, steps)
    blockers = sum(1 for f in findings if f["severity"] == "blocker")
    log.info(
        "guide.plan_lint",
        steps=len(steps),
        parts=len(sess.state.parts),
        blockers=blockers,
        warnings=len(findings) - blockers,
    )
    return findings
