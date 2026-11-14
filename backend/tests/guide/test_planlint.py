"""plan_lint: deterministic plan-vs-registry contradiction checks."""

from __future__ import annotations

from types import SimpleNamespace

from cadx.guide.planlint import plan_lint
from cadx.guide.state import AssemblyState, GuidePart, GuideState, GuideStep, GuideSubStep


def _part(pid: str) -> GuidePart:
    return GuidePart(id=pid, name=pid, bbox_min=[0, 0, 0], bbox_max=[1, 1, 1])


def _sess(parts, steps) -> SimpleNamespace:
    state = GuideState(output_dir="/tmp/x")
    state.parts = parts
    state.steps = steps
    state.assembly = AssemblyState()
    return SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")


def _step(no, subs, atype="assembly") -> GuideStep:
    return GuideStep(
        number=no,
        title=f"s{no}",
        text="",
        action_type=atype,
        sub_steps=[GuideSubStep(**s) for s in subs],
    )


def test_clean_plan_no_blockers() -> None:
    sess = _sess(
        [_part("casing"), _part("adapter"), _part("pipe")],
        [
            _step(1, [{"action": "place", "part": "adapter", "target": "casing"}]),
            _step(2, [{"action": "place", "part": "pipe", "target": "adapter"}]),
        ],
    )
    assert not [f for f in plan_lint(sess) if f["severity"] == "blocker"]


def test_unresolvable_target_routes_to_parts() -> None:
    """Unresolvable placement targets route to ``parts`` — the steps
    mission runs BEFORE parts exist and cannot alias; name-landscape
    fixes (alias_part / register) live in the parts mission."""
    sess = _sess(
        [_part("adapter"), _part("pipe")],
        [_step(1, [{"action": "place", "part": "pipe", "target": "mystery mount"}])],
    )
    fs = plan_lint(sess)
    b = [f for f in fs if f["severity"] == "blocker"]
    assert b and all(f["mission"] == "parts" for f in b)
    assert "mystery mount" in b[0]["text"]


def test_unresolvable_part_routes_to_parts() -> None:
    sess = _sess(
        [_part("pipe")],
        [_step(1, [{"action": "place", "part": "gizmoid 3000", "target": "pipe"}])],
    )
    fs = plan_lint(sess)
    b = [f for f in fs if f["severity"] == "blocker"]
    assert any(f["mission"] == "parts" for f in b)


def test_target_placed_later_is_ordering_blocker() -> None:
    sess = _sess(
        [_part("adapter"), _part("pipe")],
        [
            _step(1, [{"action": "place", "part": "pipe", "target": "adapter"}]),
            _step(3, [{"action": "place", "part": "adapter", "target": ""}]),
        ],
    )
    b = [f for f in plan_lint(sess) if f["severity"] == "blocker"]
    assert any("first placed in step 3" in f["text"] for f in b)


def test_anchor_target_is_no_ordering_blocker() -> None:
    """The casing is never placed — preassembly anchors it; referencing it
    from step 1 is the normal pattern, not an ordering bug."""
    sess = _sess(
        [_part("casing"), _part("adapter")],
        [_step(1, [{"action": "place", "part": "adapter", "target": "casing"}])],
    )
    assert not [f for f in plan_lint(sess) if f["severity"] == "blocker"]


def test_same_step_chain_is_fine() -> None:
    """Pipe onto adapter placed in the SAME step — the first placement
    exists by the time the second attaches."""
    sess = _sess(
        [_part("adapter"), _part("pipe")],
        [
            _step(
                2,
                [
                    {"action": "place", "part": "adapter", "target": ""},
                    {"action": "place", "part": "pipe", "target": "adapter"},
                ],
            )
        ],
    )
    assert not [f for f in plan_lint(sess) if f["severity"] == "blocker"]


def test_empty_assembly_step_warns_only() -> None:
    sess = _sess(
        [_part("pipe")],
        [_step(1, [{"action": "check", "part": "", "target": ""}])],
    )
    fs = plan_lint(sess)
    assert not [f for f in fs if f["severity"] == "blocker"]
    assert any(f["severity"] == "warning" for f in fs)


def test_quantity_overflow_routes_to_steps() -> None:
    """Kiara bare failure shape: the plan mapped the 2-piece elbow onto
    five steps — no geometry repair can satisfy it, the steps mission owns
    the fix (wrong part mapping / phantom occurrences)."""
    elbow = _part("elbow_ss")
    elbow.quantity = 2
    sess = _sess(
        [elbow],
        [
            _step(n, [{"action": "attach", "part": "elbow_ss", "target": ""}])
            for n in (1, 2, 3, 4, 5)
        ],
    )
    b = [f for f in plan_lint(sess) if f["severity"] == "blocker"]
    assert any(
        f["mission"] == "steps" and "5x 'elbow_ss'" in f["text"] and "only 2" in f["text"]
        for f in b
    ), b


def test_quantity_none_unbounded() -> None:
    """Unknown quantity must not lint — many manuals list no counts."""
    sess = _sess(
        [_part("pipe")],
        [
            _step(n, [{"action": "attach", "part": "pipe", "target": ""}])
            for n in (1, 2, 3, 4, 5, 6)
        ],
    )
    assert not [f for f in plan_lint(sess) if "requires" in f["text"]]


def test_host_without_incoming_mates_is_blocker() -> None:
    """The largest registered part is the assembly's base — a plan mating
    nothing to it produces a satellite disconnected from the host (kiara
    heatpump: adapters stacked onto each other, zero casing joints)."""
    from cadx.guide.planlint import plan_lint
    from cadx.guide.state import GuidePart

    big = GuidePart(id="casing", name="Gehäuse", bbox_min=[0, 0, 0], bbox_max=[10, 10, 10])
    small = GuidePart(id="adapter", name="Adapter", bbox_min=[0, 0, 0], bbox_max=[1, 1, 1])
    sess = _sess(
        [big, small],
        [_step(1, [{"action": "attach", "part": "adapter", "target": "adapter"}])],
    )
    fs = plan_lint(sess)
    anchor = [f for f in fs if "host/base" in f["text"]]
    assert anchor and anchor[0]["mission"] == "parts", fs
    # once a placement targets the big part the flag clears
    sess2 = _sess(
        [big, small],
        [_step(1, [{"action": "attach", "part": "adapter", "target": "casing"}])],
    )
    fs2 = plan_lint(sess2)
    assert not [f for f in fs2 if "host/base" in f["text"]], fs2


def test_context_target_routes_to_parts_for_aliasing() -> None:
    """A placement target that IS context (a nozzle/port feature name) is
    fixed by alias_part onto the enclosing part — only the parts mission
    has that tool."""
    from cadx.guide.planlint import plan_lint

    sess = _sess(
        [_part("casing"), _part("adapter")],
        [_step(1, [{"action": "attach", "part": "adapter", "target": "Anschlussrohr"}])],
    )
    sess.state.context_names = ["Anschlussrohr"]
    fs = plan_lint(sess)
    ctx = [f for f in fs if "Anschlussrohr" in f["text"]]
    assert ctx and ctx[0]["mission"] == "parts", fs


def test_detected_qty_bounds_plan_when_register_omits() -> None:
    """Register calls that omit `quantity` still get bounded by the
    Stückliste numbers the doc mission captured (heatpump: qty=None →
    5x l500 placed for a 2-piece kit)."""
    from cadx.guide.planlint import plan_lint
    from cadx.guide.state import DocumentIndex

    p = _part("pipe_l500")
    p.name = "Rohr DN22, l = 500 mm"
    sess = _sess(
        [p],
        [
            _step(i, [{"action": "insert", "part": "pipe_l500", "target": "elbow"}])
            for i in (1, 2, 3, 4)
        ],
    )
    sess.state.document = DocumentIndex(
        detected_part_names=["Rohr DN22, l = 500 mm"],
        detected_part_qty={"rohr dn22 l 500 mm": 2},
    )
    fs = plan_lint(sess)
    qf = [f for f in fs if "ships only" in f["text"]]
    assert qf and "4x" in qf[0]["text"] and qf[0]["mission"] == "steps", fs


def test_split_name_qty_variants() -> None:
    from cadx.guide.doc_tools import _split_name_qty

    assert _split_name_qty("2x Rohr DN22") == ("Rohr DN22", 2)
    assert _split_name_qty("Rohr DN22 x2") == ("Rohr DN22", 2)
    assert _split_name_qty("Rohr DN22 (2x)") == ("Rohr DN22", 2)
    assert _split_name_qty("Adapter DN22 — 2 Stück") == ("Adapter DN22", 2)
    assert _split_name_qty({"name": "Rohr", "qty": 3}) == ("Rohr", 3)
    assert _split_name_qty("Rohr DN22, l = 500 mm") == ("Rohr DN22, l = 500 mm", None)
