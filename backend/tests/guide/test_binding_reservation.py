"""Regression tests for canonical binding precedence, bridge-port
reservation and declared-mate repair routing.

Each test pins a generalizable rule, not a fixture:
- a '(Schritt N)' provenance pin outranks a conflicting lexical bind
- a stored binding outside the pinned step's produced set is a blocker
- an exact-declaration duplicate substep is a phantom occurrence blocker
- quantity blockers name the offending requirement sites
- a reserved bridge port is used for every task except its owner
- resolve_open_mates fails honestly when nothing is open
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cadx.guide import preassembly as pa
from cadx.guide import solver
from cadx.guide.planlint import _resolve_part, plan_lint
from cadx.guide.state import (
    AssemblyInstance,
    AssemblyState,
    GuidePart,
    GuideState,
    GuideStep,
    GuideSubStep,
    PortInfo,
)


def _part(pid: str, ports: list[PortInfo] | None = None) -> GuidePart:
    return GuidePart(
        id=pid, name=pid, ports=ports or [], bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]
    )


def _sess(parts, steps=None, assembly=None) -> SimpleNamespace:
    state = GuideState(output_dir="/tmp/x")
    state.parts = parts
    state.steps = steps or []
    state.assembly = assembly or AssemblyState()
    return SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")


def _step(no, subs, atype="assembly") -> GuideStep:
    return GuideStep(
        number=no,
        title=f"s{no}",
        text="",
        action_type=atype,
        sub_steps=[GuideSubStep(**s) for s in subs],
    )


def test_pin_overrides_conflicting_lexical_bind() -> None:
    """'rechtes vertikales Rohr (Schritt 9)' pins the part step 9
    produced — a coarse 'Rohr' kind noun spanning the sibling family must
    not bind the ref to the wrong pipe kind."""
    parts = [_part("pipe_l70"), _part("pipe_l500"), _part("elbow")]
    steps = [
        _step(9, [{"action": "place", "part": "pipe_l500", "part_pid": "pipe_l500"}]),
        _step(10, [{"action": "place", "part": "elbow"}]),
    ]
    sess = _sess(parts, steps)
    assert (
        _resolve_part(sess, "rechtes vertikales Rohr (Schritt 9)", 10) == "pipe_l500"
    )


def test_binding_conflict_blocker_on_stale_binding() -> None:
    """A stored binding outside the pinned step's produced set is the
    defect that once filtered the pinned pool empty — durable blocker.
    plan_lint canonicalizes first (which now binds pin-first), so the
    lint is exercised directly as the defense-in-depth for bindings
    written outside the canonicalizer."""
    from cadx.guide.planlint import _lint_binding_conflicts

    steps = [
        _step(3, [{"action": "place", "part": "pipe_l70", "part_pid": "pipe_l70"}]),
        _step(9, [{"action": "place", "part": "pipe_l500", "part_pid": "pipe_l500"}]),
        _step(
            10,
            [
                {
                    "action": "place",
                    "part": "elbow",
                    "part_pid": "elbow",
                    "target": "Rohr (Schritt 3)",
                    "target2": "rechtes vertikales Rohr (Schritt 9)",
                    "target2_pid": "pipe_l70",
                }
            ],
        ),
    ]
    sess = _sess([_part("pipe_l70"), _part("pipe_l500"), _part("elbow")], steps)
    blockers = [f for f in _lint_binding_conflicts(sess, steps) if f["severity"] == "blocker"]
    hit = [f for f in blockers if "provides" in f["text"] and "pipe_l70" in f["text"]]
    assert hit, blockers
    assert "pipe_l500" in hit[0]["text"]


def test_canonicalize_rewrites_conflicting_binding() -> None:
    """End-to-end: canonicalize_refs rebinds the pinned ref to what the
    pinned step produced — the stale pipe_l70 binding never survives
    into downstream consumers."""
    from cadx.guide.planlint import canonicalize_refs

    steps = [
        _step(3, [{"action": "place", "part": "pipe_l70", "part_pid": "pipe_l70"}]),
        _step(9, [{"action": "place", "part": "pipe_l500", "part_pid": "pipe_l500"}]),
        _step(
            10,
            [
                {
                    "action": "place",
                    "part": "elbow",
                    "target2": "rechtes vertikales Rohr (Schritt 9)",
                    "target2_pid": "pipe_l70",
                }
            ],
        ),
    ]
    sess = _sess([_part("pipe_l70"), _part("pipe_l500"), _part("elbow")], steps)
    canonicalize_refs(sess)
    assert steps[2].sub_steps[0].target2_pid == "pipe_l500"


def test_duplicate_occurrence_past_bound_is_blocker() -> None:
    """A third identical declaration when the kit ships two is the
    phantom-occurrence class — the blocker names both substeps."""
    adapter = _part("adapter")
    adapter.quantity = 2
    sub = {
        "action": "place",
        "part": "adapter",
        "part_pid": "adapter",
        "target": "rechter oberer Anschluss des Gehaeuses",
        "target_pid": "casing",
    }
    steps = [
        _step(1, [dict(sub)]),
        _step(2, [dict(sub)]),
        _step(3, [dict(sub)]),
    ]
    sess = _sess([adapter, _part("casing")], steps)
    fs = [f for f in plan_lint(sess) if "duplicates step" in f["text"]]
    assert any(f["severity"] == "blocker" for f in fs), fs


def test_duplicate_occurrence_within_bound_is_warning() -> None:
    """Identical repeats inside the shipped quantity — four feet on four
    symmetric corners — are evidence, not a halt."""
    foot = _part("foot")
    foot.quantity = 3
    sub = {
        "action": "place",
        "part": "foot",
        "part_pid": "foot",
        "target": "Ecke der Basis",
        "target_pid": "base",
    }
    steps = [_step(i + 1, [dict(sub)]) for i in range(3)]
    sess = _sess([foot, _part("base")], steps)
    fs = [f for f in plan_lint(sess) if "duplicates step" in f["text"]]
    assert fs and all(f["severity"] == "warning" for f in fs), fs


def test_distinct_target_occurrence_is_no_blocker() -> None:
    """Same part kind on DIFFERENT declared mates is a legitimate repeat —
    left and right mounts of one adapter must both survive the lint."""
    steps = [
        _step(
            1,
            [
                {
                    "action": "place",
                    "part": "adapter",
                    "part_pid": "adapter",
                    "target": "linker oberer Anschluss",
                    "target_pid": "casing",
                }
            ],
        ),
        _step(
            2,
            [
                {
                    "action": "place",
                    "part": "adapter",
                    "part_pid": "adapter",
                    "target": "rechter oberer Anschluss",
                    "target_pid": "casing",
                }
            ],
        ),
    ]
    sess = _sess([_part("adapter"), _part("casing")], steps)
    assert not [
        f
        for f in plan_lint(sess)
        if f["severity"] == "blocker" and "duplicates step" in f["text"]
    ]


def test_quantity_blocker_names_requirement_sites() -> None:
    """'need 3, ship 2' alone sent the repair mission re-deriving the same
    surplus — the blocker must name every substep that consumes the part."""
    adapter = _part("adapter")
    adapter.quantity = 2
    subs = [
        {
            "action": "place",
            "part": "adapter",
            "part_pid": "adapter",
            "target": f"Anschluss {i}",
            "target_pid": "casing",
        }
        for i in range(3)
    ]
    steps = [_step(i + 1, [s]) for i, s in enumerate(subs)]
    sess = _sess([adapter, _part("casing")], steps)
    blockers = [f for f in plan_lint(sess) if f["severity"] == "blocker"]
    hit = [f for f in blockers if "ships only 2" in f["text"]]
    assert hit, blockers
    for site in ("step 1 sub 0", "step 2 sub 0", "step 3 sub 0"):
        assert site in hit[0]["text"]


def _rim(name: str) -> PortInfo:
    return PortInfo(
        name=name, kind="cylinder", pos=[0, 0, 0], axis=[0, 0, 1], radius=5, gender="female"
    )


def test_reserved_port_is_used_for_foreign_tasks_only() -> None:
    """A pending bridge leg's port reads consumed to every task but the
    owner that declared the mate — the greedy fallback that once stole a
    retry's only candidate can no longer see it free."""
    pipe = _part("pipe", [_rim("rim_1"), _rim("rim_2")])
    inst = AssemblyInstance(
        part_id="pipe",
        instance_id="pipe#1",
        transform=solver.identity_transform(),
        placed_in_step=1,
    )
    asm = AssemblyState(instances=[inst])
    sess = _sess([pipe], assembly=asm)
    sess._reserved_ports = {"pipe#1": {"rim_1": "8:0"}}
    assert "rim_1" not in pa._used_port_names(sess, inst, owner="8:0")
    assert "rim_1" in pa._used_port_names(sess, inst, owner="10:0")
    assert "rim_1" in pa._used_port_names(sess, inst)


def test_reservations_do_not_leak_across_sessions() -> None:
    """No reservation attribute → plain joint occupancy, unchanged."""
    pipe = _part("pipe", [_rim("rim_1")])
    inst = AssemblyInstance(
        part_id="pipe",
        instance_id="pipe#1",
        transform=solver.identity_transform(),
        placed_in_step=1,
    )
    sess = _sess([pipe], assembly=AssemblyState(instances=[inst]))
    assert pa._used_port_names(sess, inst) == set()


def test_bridge_leg_signature_ignores_shared_clause() -> None:
    """A bridge substep's text spans BOTH ends — its side words must not
    merge into one leg's signature: 'brückt das Rohr (unten) mit dem
    Kurzrohr (oben)' locates neither leg; the leg's own ref does."""
    task = {
        "step": 8,
        "target": "freie obere Seite des linken vertikalen Rohrs (Schritt 7)",
        "target2": "freie untere Seite des linken oberen Rohrs (Schritt 3)",
        "text": "Rohrbogen brueckt das linke vertikale Rohr (unten) mit "
        "dem linken oberen Kurzrohr (oben)",
    }
    assert pa._task_side_signature(task) == {"l"}
    # a plain placement keeps clause words — they often carry the only
    # location phrase the target name lacks
    plain = {
        "step": 7,
        "target": "Rohrbogen links unten an der Waermepumpe",
        "target2": "",
        "text": "Langrohr einfuehren",
    }
    assert pa._task_side_signature(plain) == {"l", "d"}


@pytest.mark.asyncio
async def test_resolve_open_mates_fails_honestly_when_nothing_open() -> None:
    """No declared second mate → the tool says so instead of pretending
    to have repaired something."""
    from cadx.guide.assembly_tools import resolve_open_mates

    inst = AssemblyInstance(
        part_id="adapter",
        instance_id="adapter#1",
        transform=solver.identity_transform(),
        placed_in_step=1,
    )
    steps = [_step(1, [{"action": "place", "part": "adapter", "target": "casing"}])]
    sess = _sess(
        [_part("adapter"), _part("casing")],
        steps,
        AssemblyState(instances=[inst]),
    )
    out = await resolve_open_mates(sess, {})
    assert out["ok"] is False
    assert "no open bridge legs" in out["error"]
