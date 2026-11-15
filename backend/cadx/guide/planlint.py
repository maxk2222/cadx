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
    _tokset,
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


# positional/ordinal qualifiers that turn a part noun into an INSTANCE
# reference — "linkes vertikales Rohr" = the pipe already standing on the
# left, not a new part name
_INSTANCE_QUALIFIERS = {
    "links",
    "linkes",
    "linker",
    "linke",
    "rechts",
    "rechtes",
    "rechter",
    "rechte",
    "oben",
    "oberes",
    "oberer",
    "obere",
    "unten",
    "unteres",
    "unterer",
    "untere",
    "vorne",
    "vorderes",
    "vorderer",
    "hinten",
    "hinteres",
    "hinterer",
    "hintere",
    "vertikal",
    "vertikales",
    "vertikaler",
    "vertikale",
    "horizontal",
    "horizontales",
    "mittlerer",
    "mittleres",
    "seitlich",
    "seitliches",
    "linken",
    "rechten",
    "oberen",
    "unteren",
    "vorderen",
    "hinteren",
    "seitlicher",
    "left",
    "right",
    "upper",
    "lower",
    "top",
    "bottom",
    "front",
    "rear",
    "back",
    "vertical",
    "middle",
    "side",
    "inner",
    "outer",
    "first",
    "second",
    "third",
    "erste",
    "zweite",
    "dritte",
    "erster",
    "zweiter",
    "dritter",
}


def _is_instance_ref(sess: GuideToolSession, name: str) -> bool:
    """A qualifier + part-noun phrase that names a placed instance by
    position rather than a registered part — placement resolves it via
    the direction words against placed instances."""
    toks = _tokset(name)
    if not (toks & _INSTANCE_QUALIFIERS):
        return False
    part_toks: set[str] = set()
    for k in _name_index(sess):
        part_toks |= _tokset(k)
    return bool(toks & part_toks)


def _lint_required_context(sess: GuideToolSession, st: Any) -> list[dict[str, Any]]:
    """A context-marked required_part pressures the parts mission into
    generating host geometry (covers, stubs) the kit never ships — it
    belongs in the step text, not the parts list."""
    from cadx.guide.part_tools import _is_context

    findings: list[dict[str, Any]] = []
    for name in st.required_parts or []:
        if name and _is_context(sess, name) and not _resolve_name(sess, name):
            findings.append(
                {
                    "mission": "steps",
                    "severity": "blocker",
                    "text": f"step {st.number}: required_parts '{name}' is "
                    "marked context — premounted host components and "
                    "consumables are scene context (text/target), not "
                    "kit parts the step installs; remove it from "
                    "required_parts",
                }
            )
    return findings


def _lint_required_resolvable(sess: GuideToolSession, st: Any) -> list[dict[str, Any]]:
    """A required_part matching no registered part is unsatisfiable at
    assembly: completeness counts it as a needed instance forever. The
    plan must resolve it — alias onto a registered part, or drop it from
    required_parts via update_step (kit lacks the part and generation is
    off). Context-marked names already carry their own blocker."""
    from cadx.guide.part_tools import _is_context

    findings: list[dict[str, Any]] = []
    for name in st.required_parts or []:
        if not name or _is_context(sess, name) or _resolve_name(sess, name):
            continue
        remedy = (
            "alias_part it onto a registered part or update_step to drop "
            "it from required_parts"
        )
        if not getattr(sess.state, "allow_part_generation", True):
            remedy += " (generation is disabled — the kit part must come "
            "from an uploaded asset or the step text)"
        findings.append(
            {
                "mission": "parts",
                "severity": "blocker",
                "text": f"step {st.number}: required_parts '{name}' "
                "matches no registered part"
                + _suggest(sess, name)
                + " — "
                + remedy,
            }
        )
    return findings


def _lint_target(sess: GuideToolSession, st: Any, i: int, name: str) -> list[dict[str, Any]]:
    """One placement target reference: context-marked, ambiguous,
    instance-style, or simply unresolvable — each routes to the mission
    that can actually fix it."""
    from cadx.guide.part_tools import _is_context

    # alias wins over a context mark — the last explicit agent action is
    # the intended resolution; a stale context mark repaired via
    # alias_part must not keep blocking placement targets
    if _is_context(sess, name) and not _resolve_name(sess, name):
        return [
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
        ]
    if _resolve_name(sess, name):
        return []
    amb = getattr(sess, "_ambiguous", {}).get(name)
    if amb:
        # the name tied between different parts — never silently
        # guessed; the steps mission must qualify it with the
        # parts-list name that carries the discriminator
        # ('Rohrbogen 1x außen 1x innen')
        return [
            {
                "mission": "steps",
                "severity": "blocker",
                "text": f"step {st.number} sub {i}: target "
                f"'{name}' is ambiguous — matches {amb}; rename "
                "it to the detected part name that carries the "
                "discriminator",
            }
        ]
    if _is_instance_ref(sess, name):
        # "linkes vertikales Rohr" references a PLACED instance by
        # position, not a part name — the direction words pick the
        # instance at placement time (preassembly groups fall through
        # to every placed instance). Not repairable by alias: the
        # qualifier is the whole point. Warn, don't block.
        return [
            {
                "mission": "steps",
                "severity": "warning",
                "text": f"step {st.number} sub {i}: target "
                f"'{name}' is an instance-style reference "
                "(positional qualifier + part noun) — it "
                "resolves per position at placement",
            }
        ]
    return [
        {
            "mission": "parts",
            "severity": "blocker",
            "text": f"step {st.number} sub {i}: target '{name}' "
            "matches no registered part"
            + _suggest(sess, name)
            + " — alias_part the name onto the host part "
            "or register a missed asset",
        }
    ]


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
        findings.extend(_lint_required_context(sess, st))
        findings.extend(_lint_required_resolvable(sess, st))
        for i, ss in enumerate(st.sub_steps):
            for name in (ss.target, ss.target2):
                # only PLACEMENT targets must resolve — check/orient targets
                # are context anchors (the pre-existing host, walls,
                # "Wärmepumpe") that legitimately match no kit part
                if not name or ss.action not in _PLACE_ACTIONS:
                    continue
                findings.extend(_lint_target(sess, st, i, name))
            if ss.action not in _PLACE_ACTIONS:
                continue
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
                amb = getattr(sess, "_ambiguous", {}).get(ss.part)
                if amb:
                    findings.append(
                        {
                            "mission": "steps",
                            "severity": "blocker",
                            "text": f"step {st.number} sub {i}: part "
                            f"'{ss.part}' is ambiguous — matches {amb}; "
                            "use the detected part name that carries "
                            "the discriminator",
                        }
                    )
                else:
                    findings.append(
                        {
                            "mission": "parts",
                            "severity": "blocker",
                            "text": f"step {st.number} sub {i}: part "
                            f"'{ss.part}' matches no registered part"
                            + _suggest(sess, ss.part)
                            + " — register/alias the asset or fix the "
                            "text",
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


def _lint_phantom_sizes(sess: GuideToolSession, steps: list[Any]) -> list[dict[str, Any]]:
    """A size-marked name whose length exists in NO detected part name and
    NO registered part is a dimension/cut-length callout, not a kit part
    (observed: figure lengths l=100/200mm became phantom steps + generated
    pipe_l100/pipe_l200 — the Stückliste only ships l=70/500). The repair
    lives in the plan, so it routes to ``steps``."""
    from cadx.guide.part_tools import _sig

    kit_sizes: set[str] = set()
    doc = getattr(sess.state, "document", None)
    for n in (getattr(doc, "detected_part_names", None) or []):
        _, t = _sig(n)
        kit_sizes |= {x for x in t if x.startswith("l")}
    # registered parts count as kit evidence only when they carry a real
    # uploaded asset — a generated part can itself BE the phantom this
    # check hunts (l=200 pipe generated from a figure dimension)
    for p in sess.state.parts:
        if p.asset_id:
            _, t = _sig(f"{p.name} {p.id}")
            kit_sizes |= {x for x in t if x.startswith("l")}
    if not kit_sizes:
        return []
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for st in steps:
        names = list(st.required_parts or [])
        for ss in st.sub_steps or []:
            names += [ss.part, ss.target, ss.target2]
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            _, toks = _sig(name)
            phantom = {t for t in toks if t.startswith("l")} - kit_sizes
            if phantom:
                findings.append(
                    {
                        "mission": "steps",
                        "severity": "blocker",
                        "text": f"step {st.number}: '{name}' carries size "
                        f"{sorted(phantom)} that no Stückliste part has "
                        f"(kit sizes: {sorted(kit_sizes)}) — it is a "
                        "dimension/cut-length callout, not a part. Remove "
                        "the substep/required_parts entry; do not alias "
                        "or generate it",
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


def _lint_anchor(sess: GuideToolSession, steps: list[Any]) -> list[dict[str, Any]]:
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
            "on). Targets naming its features (nozzles, ports, the host "
            "device itself) must alias_part onto it, or the assembly "
            "ends up floating disconnected",
        }
    ]


def _lint_dropped_placements(sess: GuideToolSession, steps: list[Any]) -> list[dict[str, Any]]:
    """A place-substep under a non-assembly step is silently dropped:
    _plan and _step_requirements only read assembly steps, so the
    declared placement never runs, the instance never materializes and
    downstream audits see a mate that vanished. Either the step's
    action_type is mislabeled (fix it to 'assembly') or the sub-step
    belongs on an assembly step."""
    findings: list[dict[str, Any]] = []
    for st in steps:
        if st.action_type == "assembly":
            continue
        for i, ss in enumerate(st.sub_steps):
            if ss.action in _PLACE_ACTIONS:
                findings.append(
                    {
                        "mission": "steps",
                        "severity": "blocker",
                        "text": f"step {st.number} sub {i}: placement of "
                        f"'{ss.part}' sits under action_type='{st.action_type}' "
                        "— the planner skips non-assembly steps so this mate "
                        "is silently dropped. Either set the step's "
                        "action_type to 'assembly' (update_step), or if the "
                        "sub-step operates on an already-placed part "
                        "(tighten/crimp/re-check) retag its action to "
                        "'check'/'orient'",
                    }
                )
    return findings


_PREMOUNT_RE = None  # lazily built — see _premount_words()


def _premount_words():
    import re

    global _PREMOUNT_RE
    if _PREMOUNT_RE is None:
        _PREMOUNT_RE = re.compile(
            r"vormontiert|premount|pre-mount|preinstall|pre-install|"
            r"bereits montiert|already mounted|factory.?fit",
            re.I,
        )
    return _PREMOUNT_RE


def _lint_premount_refs(sess: GuideToolSession, steps: list[Any]) -> list[dict[str, Any]]:
    """BOM "N + M vormontiert" declares M device-mounted copies — the plan
    must reference them as premount TARGETS somewhere (the preassembly
    planner then spawns the actual mount tasks). A plan that only mounts
    the kit copies and never names the pre-installed ones silently drops
    part of the documented state — same class as a dropped placement."""
    doc = getattr(sess.state, "document", None)
    pm_map = getattr(doc, "detected_premount_qty", None) or {}
    if not pm_map:
        return []
    findings: list[dict[str, Any]] = []
    for nrm, pm_qty in pm_map.items():
        pid = sess._aliases_store.get(nrm) or _name_index(sess).get(nrm)
        if not pid:
            continue
        referenced = False
        for st in steps:
            for ss in st.sub_steps:
                for tgt in (ss.target, ss.target2):
                    if (
                        tgt
                        and _premount_words().search(tgt)
                        and _resolve_name(sess, tgt) == pid
                    ):
                        referenced = True
        if not referenced:
            findings.append(
                {
                    "mission": "steps",
                    "severity": "blocker",
                    "text": f"parts list marks {pm_qty}x '{pid}' as already "
                    "mounted on the device (vormontiert) — no step references "
                    "them as a premounted target. Add 'vormontierter <part>' "
                    "targets where the manual's later steps connect to them "
                    "(they are initial-state context, not loose kit parts)",
                }
            )
    return findings


def _lint_side_coverage(sess: GuideToolSession, steps: list[Any]) -> list[dict[str, Any]]:
    """A side-qualified placement target matching none of the part's
    planned instance signatures is an unresolved position reference —
    either a premounted/context instance the plan never declared, or
    wrong side words. Without this check placement silently binds the
    nearest instance at the WRONG side (observed: 'Adapter an der
    unteren Anschlussstelle' binding the upper kit adapter — chains
    built top-side, honest success verdict on the wrong topology).
    Marker-carrying targets spawn premounted copies in preassembly and
    are exempt here."""
    from cadx.guide.preassembly import _PREMOUNT_MARK, _task_side_signature

    tasks = [
        {
            "step": st.number,
            "substep_idx": i,
            "part": ss.part,
            "target": ss.target or "",
            "text": ss.text or "",
        }
        for st in steps
        for i, ss in enumerate(st.sub_steps or [])
        if ss.action in _PLACE_ACTIONS
    ]
    planned: dict[str, list[frozenset[str]]] = {}
    for t in tasks:
        pid = _resolve_part(sess, t["part"], t["step"])
        if pid:
            planned.setdefault(pid, []).append(frozenset(_task_side_signature(t)))
    findings: list[dict[str, Any]] = []
    for t in tasks:
        tpid = _resolve_name(sess, t["target"])
        if not tpid or tpid == _resolve_part(sess, t["part"], t["step"]):
            continue
        ref = frozenset(_task_side_signature(t))
        if not ref:
            continue
        sigs = [s for s in planned.get(tpid, []) if s]
        if len(sigs) < 2 or any(ref <= s for s in sigs):
            continue
        if _PREMOUNT_MARK.search(f"{t['target']} {t['text']}"):
            continue  # preassembly spawns the premounted copy
        findings.append(
            {
                "mission": "steps",
                "severity": "blocker",
                "text": f"step {t['step']}: target '{t['target'][:60]}' "
                f"locates a '{tpid}' at side {{{','.join(sorted(ref))}}} "
                "but the plan only places it at "
                f"{{{'; '.join(','.join(sorted(s)) for s in sigs)}}} — "
                "either mark the target 'vormontierter <part>' (a "
                "device-side piece the kit never installs) or fix the "
                "side words to name a placed instance",
            }
        )
    return findings


def _lint_step_refs(sess: GuideToolSession, steps: list[Any]) -> list[dict[str, Any]]:
    """'(<part> aus) Schritt N' references must pin a step that actually
    PROVIDES the ref's kind — the pin is the strongest instance
    discriminator and a wrong one either pins an empty pool (preassembly
    then reported a miss) or the wrong side's instance (observed:
    'rechter Adapter (Schritt 2)' bound the 'linker' adapter step 2
    placed). Manuals number by figure, plans by guide step — a mismatch
    is a plan defect the steps mission must fix, not a runtime guess."""
    from cadx.guide.preassembly import _STEP_REF, _dir_axes, _kind_pool

    findings: list[dict[str, Any]] = []
    by_no = {st.number: st for st in steps}
    for st in steps:
        for ss in st.sub_steps or []:
            for tgt in (ss.target, ss.target2):
                if not tgt:
                    continue
                m = _STEP_REF.search(tgt)
                if not m:
                    continue
                n = int(m.group(1))
                ref_pool = _kind_pool(sess, tgt)
                pinned = by_no.get(n)
                if pinned is None:
                    findings.append(
                        {
                            "mission": "steps",
                            "severity": "blocker",
                            "text": f"step {st.number}: target "
                            f"'{tgt[:60]}' pins '(Schritt {n})' but the "
                            "plan has no such step — reference the PLAN "
                            "step number that placed the part",
                        }
                    )
                    continue
                # kinds step N provides: placement substep parts +
                # step-level required_parts (premount/context sources)
                provided: set[str] = set()
                ref_axes = dict(_dir_axes(tgt))
                kind_sides: list[dict[int, float]] = []
                for pss in pinned.sub_steps or []:
                    pid = _resolve_part(sess, pss.part, n)
                    if not pid:
                        continue
                    provided.add(pid)
                    if pss.action in _PLACE_ACTIONS and (
                        not ref_pool or pid in ref_pool
                    ):
                        kind_sides.append(
                            dict(_dir_axes(f"{pss.target or ''} {pss.part or ''}"))
                        )
                for nm in pinned.required_parts or []:
                    pid = _resolve_part(sess, nm, n)
                    if pid:
                        provided.add(pid)
                if ref_pool and not (ref_pool & provided):
                    findings.append(
                        {
                            "mission": "steps",
                            "severity": "blocker",
                            "text": f"step {st.number}: target "
                            f"'{tgt[:60]}' pins '(Schritt {n})' but that "
                            f"step provides {sorted(provided) or 'nothing'} — "
                            f"the ref's kind {sorted(ref_pool)} is placed "
                            "elsewhere. Fix the reference (plan step "
                            "number or drop the pin)",
                        }
                    )
                    continue
                # side check: a ref naming 'rechter … (Schritt N)' is
                # consistent iff SOME kind-compatible substep of step N
                # carries no opposite sign on any ref axis
                if ref_axes and kind_sides and not any(
                    not any(ks.get(a) == -sg for a, sg in ref_axes.items())
                    for ks in kind_sides
                ):
                    findings.append(
                        {
                            "mission": "steps",
                            "severity": "blocker",
                            "text": f"step {st.number}: target "
                            f"'{tgt[:60]}' pins '(Schritt {n})' but "
                            f"step {n} places that kind at a different "
                            "side — the pin or the side word is wrong; "
                            "fix the reference",
                        }
                    )
    return findings


def _lint_bridge_reach(sess: GuideToolSession, steps: list[Any]) -> list[dict[str, Any]]:
    """A bridge leg (target+target2) can only close TOWARD its partner —
    a fitting turns at most ~90° per joint, it can never reach back past
    the chain's own anchor. When the chain feeding `target` declares
    'nach unten' growth while `target2` attaches ABOVE the chain's host
    anchor, the loop is declared in the geometrically impossible
    direction: deterministic placement would faithfully build a chain
    that dives 500mm away and then leave the elbow reaching for a mate
    hundreds of mm behind it (observed: l500 chains descending from the
    lower bores declared to bridge back to upper-bore stubs).

    Chain growth comes from open-direction phrases ('nach unten',
    'vertikale Verbindung nach unten') along the placement chain; the
    relative anchor positions come from the side words of each leg's
    host-attachment target ('untere Bohrung' vs 'obere Bohrung'). Only
    flags when both legs hang on the SAME host part — different anchors
    carry incomparable frames and yield no verdict."""

    findings: list[dict[str, Any]] = []
    by_no = {st.number: st for st in steps}
    placements: dict[str, list[tuple[int, Any]]] = {}
    for st in steps:
        for ss in st.sub_steps or []:
            if ss.action in _PLACE_ACTIONS:
                pid = ss.part_pid or _resolve_part(sess, ss.part, st.number)
                if pid:
                    placements.setdefault(pid, []).append((st.number, ss))
    earliest = min((n for lst in placements.values() for n, _ in lst), default=None)
    for st in steps:
        for ss in st.sub_steps or []:
            if ss.action not in _PLACE_ACTIONS or not ss.target2:
                continue
            f = _bridge_leg_finding(
                sess, ss, st.number, placements, by_no, earliest
            )
            if f:
                findings.append(f)
    return findings


def _bridge_producer(
    sess: GuideToolSession,
    placements: dict[str, list[tuple[int, Any]]],
    pid: str,
    ref: str | None,
) -> tuple[int, Any] | None:
    """(step, substep) that places `pid`, honoring the ref's '(Schritt N)'
    pin; symmetric siblings fall to the side words or the first match —
    mirrored siblings share the growth axis the caller checks, so the
    verdict is safe on either."""
    from cadx.guide.preassembly import _STEP_REF, _dir_axes

    m = _STEP_REF.search(ref or "")
    cands = placements.get(pid, [])
    if m:
        n = int(m.group(1))
        return next(((sn, ssx) for sn, ssx in cands if sn == n), None)
    if len(cands) == 1:
        return cands[0]
    ref_axes = dict(_dir_axes(ref or ""))
    if ref_axes:
        best, best_score = None, -1.0
        for sn, ssx in cands:
            t_axes = dict(_dir_axes(ssx.target or ""))
            score = sum(
                1.0 if t_axes.get(a) == sg else -1.0 if a in t_axes else 0.0
                for a, sg in ref_axes.items()
            )
            if score > best_score:
                best, best_score = (sn, ssx), score
        if best and best_score > 0:
            return best
    return cands[0] if cands else None


def _bridge_chain(
    sess: GuideToolSession,
    placements: dict[str, list[tuple[int, Any]]],
    by_no: dict[int, Any],
    earliest: int | None,
    pid: str,
    ref: str | None,
) -> tuple[dict[int, float], Any, int] | None:
    """Accumulated growth axes + the chain's host-attachment substep —
    the link whose target is the anchor part or an unplaced element."""
    from cadx.guide.preassembly import _open_axes

    growth: dict[int, float] = {}
    cur = _bridge_producer(sess, placements, pid, ref)
    seen: set[int] = set()
    while cur:
        csn, css = cur
        if id(css) in seen:
            return None
        seen.add(id(css))
        cst = by_no.get(csn)
        hint = f"{css.target or ''} {css.text or ''} {cst.title if cst else ''}"
        for ax, sg in _open_axes(hint):
            growth[ax] = growth.get(ax, 0.0) + sg
        n_pid = css.target_pid or _resolve_part(sess, css.target or "", csn)
        if not n_pid or n_pid not in placements:
            return growth, css, csn
        if min(sn for sn, _ in placements[n_pid]) == earliest:
            # attaches onto the anchor/host part — THIS substep's
            # target locates the chain's root on the host
            return growth, css, csn
        cur = _bridge_producer(sess, placements, n_pid, css.target)
    return None


def _bridge_leg_finding(
    sess: GuideToolSession,
    ss: Any,
    step_no: int,
    placements: dict[str, list[tuple[int, Any]]],
    by_no: dict[int, Any],
    earliest: int | None,
) -> dict[str, Any] | None:
    """One bridge substep's direction contradiction — the chain feeding
    `target` grows away from the `target2` partner's host position."""
    from cadx.guide.preassembly import _dir_axes

    t_pid = ss.target_pid or _resolve_part(sess, ss.target or "", step_no)
    p_pid = ss.target2_pid or _resolve_part(sess, ss.target2 or "", step_no)
    if not t_pid or not p_pid:
        return None
    chain = _bridge_chain(sess, placements, by_no, earliest, t_pid, ss.target)
    if not chain:
        return None
    growth, root_ss, root_step = chain
    growth = {a: g for a, g in growth.items() if abs(g) >= 0.5}
    if not growth:
        return None
    # the partner's attachment point: its producing substep's target —
    # comparable only when it hangs on the same host
    prod = _bridge_producer(sess, placements, p_pid, ss.target2)
    if prod:
        p_step, p_ss = prod
        p_target = p_ss.target or ""
        if p_step >= step_no:
            return None
        p_pid_attach = p_ss.target_pid or _resolve_part(sess, p_target, p_step)
    else:
        # partner not placed by the plan — compare its own ref axes
        # only when it names the same host part
        p_target = ss.target2 or ""
        p_pid_attach = _resolve_name(sess, p_target)
    r_pid = root_ss.target_pid or _resolve_part(sess, root_ss.target or "", root_step)
    if r_pid is None or p_pid_attach is None or r_pid != p_pid_attach:
        return None
    r_axes = dict(_dir_axes(root_ss.target or ""))
    p_axes = dict(_dir_axes(p_target))
    for ax, g in growth.items():
        rel = p_axes.get(ax, 0.0) - r_axes.get(ax, 0.0)
        if g * rel < -0.5:
            side = {2: "vertically", 1: "front/back", 0: "left/right"}[ax]
            return {
                "mission": "steps",
                "severity": "blocker",
                "text": f"step {step_no}: the chain feeding "
                f"'{(ss.target or '')[:40]}' grows away from "
                f"the declared mate '{(ss.target2 or '')[:40]}' "
                f"({side}) — a closing fitting at the chain's "
                "far end can never reach back past its own "
                "anchor. The loop direction or the partner "
                "reference contradicts the figure — re-read "
                "the manual figure and re-order the legs "
                "(which fitting sits at the chain's top end, "
                "which at its bottom?)",
            }
    return None


def _lint_asset_coverage(sess: GuideToolSession) -> list[dict[str, Any]]:
    """Every uploaded CAD file is real geometry the plan must account
    for — an unregistered upload means the assembly silently dropped a
    part (observed: Casing.glb classified as context → no host to anchor
    to, satellite assembly). Doc/page assets carry no geometry and are
    skipped; only STEP/mesh bodies must map to a registered part."""
    assets = getattr(sess, "guide_assets", None) or []
    if not assets:
        return []
    from cadx.guide import cadmeta

    used = {p.asset_id for p in sess.state.parts if getattr(p, "asset_id", None)}
    findings: list[dict[str, Any]] = []
    for a in assets:
        fn = str(a.get("filename") or a.get("path") or "")
        from pathlib import Path as _P

        if _P(fn).suffix.lower() not in (cadmeta._BREP_EXTS | cadmeta._MESH_EXTS):
            continue  # pdf/page assets carry no placeable geometry
        keys = {str(a.get("id") or ""), fn, _P(fn).name}
        if keys & used:
            continue
        findings.append(
            {
                "mission": "parts",
                "severity": "blocker",
                "text": f"uploaded CAD file '{_P(fn).name}' maps to no "
                "registered part — the assembly silently drops its "
                "geometry. register_part it (a host/context part still "
                "needs registering so the planner can anchor to it)",
            }
        )
    return findings


def canonicalize_refs(sess: GuideToolSession) -> None:
    """Bind every sub-step reference ONCE to its registered part id.

    The free-text name lives in ``part``/``target``/``target2`` for prompts
    and display; ``*_pid`` is the auditable identity downstream consumers
    (preassembly, placement, lints) use. Unresolvable or ambiguous names
    stay None — the lints below report them as findings instead of
    consumers each re-guessing through the fuzzy cascade."""
    for st in sess.state.steps:
        for ss in st.sub_steps:
            ss.part_pid = _resolve_part(sess, ss.part, st.number) if ss.part else None
            ss.target_pid = (
                _resolve_part(sess, ss.target, st.number) if ss.target else None
            )
            ss.target2_pid = (
                _resolve_part(sess, ss.target2, st.number) if ss.target2 else None
            )


def plan_lint(sess: GuideToolSession) -> list[dict[str, Any]]:
    """Plan-vs-registry contradictions, deterministically detected.

    Returns findings as {mission, severity, text} — ``severity="blocker"``
    means placement CANNOT succeed; ``"warning"`` means suspicious but
    legal. Only blockers reopen missions.
    """
    steps = sorted(sess.state.steps, key=lambda s: s.number)
    canonicalize_refs(sess)
    findings, first_place, anchor_step = _lint_refs(sess, steps)
    findings += _lint_order(sess, steps, first_place, anchor_step)
    findings += _lint_quantities(sess)
    findings += _lint_dropped_placements(sess, steps)
    findings += _lint_phantom_sizes(sess, steps)
    findings += _lint_anchor(sess, steps)
    findings += _lint_premount_refs(sess, steps)
    findings += _lint_side_coverage(sess, steps)
    findings += _lint_step_refs(sess, steps)
    findings += _lint_bridge_reach(sess, steps)
    findings += _lint_asset_coverage(sess)
    blockers = sum(1 for f in findings if f["severity"] == "blocker")
    log.info(
        "guide.plan_lint",
        steps=len(steps),
        parts=len(sess.state.parts),
        blockers=blockers,
        warnings=len(findings) - blockers,
    )
    return findings
