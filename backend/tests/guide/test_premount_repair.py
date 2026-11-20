"""Regression tests for the premount/bridge repair plan.

Each test pins a generalizable rule, not the Kiara demo:
- back-solve failures carry a durable reason tag, never a silent None
- a side-signed chain task wins over an earlier unsigned sibling
- the closure estimate discriminates wrong-side premount poses
- an uncloseable mate walks back to its chain ROOT, not the leaf
- request_repair cascades the subtree hanging on a torn-down root
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np

from cadx.guide import preassembly as pa
from cadx.guide.assembly_tools import _mate_blocking_root, request_repair
from cadx.guide.state import (
    AssemblyInstance,
    AssemblyState,
    GuidePart,
    GuideState,
    PortInfo,
)


def _part(
    pid: str,
    ports: list[PortInfo] | None = None,
    bbox: tuple[tuple[float, ...], tuple[float, ...]] = ((-10, -10, -10), (10, 10, 10)),
) -> GuidePart:
    return GuidePart(
        id=pid,
        name=pid,
        ports=ports or [],
        bbox_min=list(bbox[0]),
        bbox_max=list(bbox[1]),
    )


def _cyl(name: str, pos: tuple[float, ...], axis: tuple[float, ...], gender: str = "female") -> PortInfo:
    return PortInfo(name=name, kind="cylinder", pos=list(pos), axis=list(axis), radius=11.0, gender=gender)


def _sess(parts: list[GuidePart], assembly: AssemblyState | None = None) -> SimpleNamespace:
    state = GuideState(output_dir="/tmp/x")
    state.parts = parts
    state.assembly = assembly or AssemblyState()
    return SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")


def _inst(pid: str, iid: str, tr: np.ndarray | None = None, **kw: Any) -> AssemblyInstance:
    return AssemblyInstance(
        part_id=pid,
        instance_id=iid,
        transform=(tr if tr is not None else np.eye(4)).tolist(),
        placed_in_step=kw.get("step", 1),
        substep_idx=kw.get("substep_idx", 0),
        premounted=kw.get("premounted", False),
    )


def _txyz(x: float, y: float, z: float) -> np.ndarray:
    t = np.eye(4)
    t[0, 3], t[1, 3], t[2, 3] = x, y, z
    return t


# --------------------------------------------------------------------- #
# P0/P1.3 — durable back-solve evidence
# --------------------------------------------------------------------- #


def test_premount_chain_transform_reason_tags() -> None:
    """Every back-solve miss names WHY — the silent None used to leave a
    wrong heuristic pose unexplainable."""
    s = _sess([_part("adapt"), _part("elbow"), _part("pipe")])
    # no chain at all after the premount task
    tasks = [{"step": 2, "part": "adapt", "part_pid": "adapt", "premount": True}]
    tr, reason = pa._premount_chain_transform(s, tasks, 0, tasks[0], "adapt")
    assert tr is None and reason == "no_bridge_in_chain"
    # a bridge task directly after the premount = no mid links to unwind
    tasks = [
        {"step": 2, "part": "adapt", "part_pid": "adapt", "premount": True},
        {
            "step": 3,
            "part": "elbow",
            "part_pid": "elbow",
            "target": "adapt",
            "target_pid": "adapt",
            "target2": "pipe",
            "target2_pid": "pipe",
        },
    ]
    tr, reason = pa._premount_chain_transform(s, tasks, 0, tasks[0], "adapt")
    assert tr is None and reason == "no_mid_links"


# --------------------------------------------------------------------- #
# P1.1 — side-aware chain walk
# --------------------------------------------------------------------- #


def test_side_filtered_chain_prefers_signed_sibling() -> None:
    """Sibling branches share part ids: an unsigned task earlier in the
    plan must NOT absorb the left chain — the side-signed task wins and
    the walk reaches the LEFT chain's bridge."""
    s = _sess([_part("host"), _part("adapt"), _part("elbow"), _part("pipe"), _part("bridge")])
    tasks = [
        {"step": 1, "part": "host", "part_pid": "host", "anchor": True},
        {
            "step": 2,
            "part": "adapt",
            "part_pid": "adapt",
            "target": "host",
            "target_pid": "host",
            "premount": True,
            "_ref_sides": ["l"],
        },
        # sibling branch root — same part id, no side words: fallback only
        {"step": 3, "part": "elbow", "part_pid": "elbow", "target": "adapt", "target_pid": "adapt"},
        # sibling branch's own next link
        {"step": 4, "part": "pipe", "part_pid": "pipe", "target": "elbow", "target_pid": "elbow"},
        # OUR side's chain root — signed 'l' via its text
        {
            "step": 5,
            "part": "elbow",
            "part_pid": "elbow",
            "target": "adapt",
            "target_pid": "adapt",
            "text": "linker Bogen montieren",
        },
        # OUR chain's closing bridge
        {
            "step": 6,
            "part": "bridge",
            "part_pid": "bridge",
            "target": "elbow",
            "target_pid": "elbow",
            "target2": "host",
            "target2_pid": "host",
        },
    ]
    mids, bridge = pa._side_filtered_chain(s, tasks, 1, "adapt", {"l"})
    # the unsigned sibling root would have walked into ITS pipe and found
    # no bridge — the signed task skips it and reaches our closure
    assert mids == ["elbow"]
    assert bridge is tasks[5]


# --------------------------------------------------------------------- #
# P1.2 — closure-aware candidate ranking
# --------------------------------------------------------------------- #


def test_premount_closure_res_discriminates_pose() -> None:
    """The estimate is the gap between where the chain ENDS and the
    partner tip it must reach — a wrong-side host port predicts a large
    residual, the closable one a small one."""
    host = _part("host", bbox=((-50, -50, 0), (50, 50, 100)))
    adapt = _part("adapt", [_cyl("out", (0, 0, 0), (0, 0, 1))], bbox=((-10, -10, -10), (10, 10, 10)))
    pipe = _part(
        "pipe",
        [_cyl("a", (0, 0, 0), (0, 0, -1)), _cyl("b", (0, 0, 200), (0, 0, 1))],
        bbox=((-5, -5, 0), (5, 5, 200)),
    )
    stub = _part("stub", [_cyl("free", (0, 0, 0), (0, 0, 1))], bbox=((-5, -5, -20), (5, 5, 0)))
    bridge = _part("bridge")
    asm = AssemblyState()
    asm.instances = [
        _inst("host", "host#1"),
        # partner tip: its free port sits at world z=210
        _inst("stub", "stub#1", _txyz(0.0, 0.0, 210.0), step=4),
    ]
    s = _sess([host, adapt, pipe, stub, bridge], asm)
    tasks = [
        {
            "step": 2,
            "part": "adapt",
            "part_pid": "adapt",
            "target": "host",
            "target_pid": "host",
            "premount": True,
            "_ref_sides": ["l"],
        },
        {
            "step": 3,
            "part": "pipe",
            "part_pid": "pipe",
            "target": "adapt",
            "target_pid": "adapt",
        },
        {
            "step": 5,
            "part": "bridge",
            "part_pid": "bridge",
            "target": "pipe",
            "target_pid": "pipe",
            "target2": "stub",
            "target2_pid": "stub",
        },
    ]
    out_port = adapt.ports[0]
    # chain reach: adapt half-span 10 + pipe link (200 - 15) = 195 above
    # the premount port — a pose at z~15 lands the tip on the partner
    near = pa._premount_closure_res(
        s, tasks, 0, tasks[0], "adapt", _txyz(0.0, 0.0, 15.0).tolist(), out_port, "host"
    )
    far = pa._premount_closure_res(
        s, tasks, 0, tasks[0], "adapt", _txyz(400.0, 0.0, 15.0).tolist(), out_port, "host"
    )
    assert near is not None and far is not None
    assert near < far
    assert near <= pa.CLOSURE_SLACK_MM
    assert far > pa.CLOSURE_SLACK_MM


# --------------------------------------------------------------------- #
# P2 — causal attribution
# --------------------------------------------------------------------- #


def test_mate_blocking_root_walks_to_premount() -> None:
    """The residual is measured on the leaf; the REPAIR belongs to the
    root the chain hangs on — a premount has no upstream joint, so the
    walk ends there."""
    s = _sess([_part("host"), _part("adapt"), _part("elbow"), _part("pipe")])
    asm = s.state.assembly
    asm.instances = [
        _inst("host", "host#1", step=1),
        _inst("adapt", "adapt#4", _txyz(0.0, 0.0, 11.0), step=7, premounted=True),
        _inst("elbow", "elbow#1", _txyz(0.0, 0.0, 40.0), step=8),
        _inst("pipe", "pipe#1", _txyz(0.0, 0.0, 240.0), step=9),
    ]
    asm.joints = [
        {"step": 8, "moving_instance": "elbow#1", "fixed_instance": "adapt#4"},
        {"step": 9, "moving_instance": "pipe#1", "fixed_instance": "elbow#1"},
    ]
    cause = _mate_blocking_root(s, "pipe#1")
    assert cause is not None
    assert cause["root"] == "adapt#4"
    assert cause["step"] == 7
    assert cause["kind"] == "premount"
    # a dead-ended (non-premount) leaf subtree reports honestly too
    asm.instances.append(_inst("pipe", "pipe#9", _txyz(500.0, 0.0, 0.0), step=11))
    dead = _mate_blocking_root(s, "pipe#9")
    assert dead is not None and dead["root"] == "pipe#9" and dead["kind"] == "unjoined"


# --------------------------------------------------------------------- #
# P3 — repair cascade
# --------------------------------------------------------------------- #


def test_request_repair_cascades_subtree() -> None:
    """Removing a flagged chain root orphans the subtree hanging on it —
    the cascade removes those floating strays immediately; a bridge
    instance still jointed to a healthy second host survives."""
    s = _sess([_part("host"), _part("adapt"), _part("elbow"), _part("pipe"), _part("bridge")])
    asm = s.state.assembly
    asm.instances = [
        _inst("host", "host#1", step=1),
        _inst("adapt", "adapt#4", _txyz(0.0, 0.0, 11.0), step=7, premounted=True),
        _inst("elbow", "elbow#1", _txyz(0.0, 0.0, 40.0), step=8),
        _inst("pipe", "pipe#1", _txyz(0.0, 0.0, 240.0), step=9),
        _inst("bridge", "bridge#1", _txyz(0.0, 0.0, 300.0), step=10),
    ]
    asm.joints = [
        {"step": 8, "moving_instance": "elbow#1", "fixed_instance": "adapt#4"},
        {"step": 9, "moving_instance": "pipe#1", "fixed_instance": "elbow#1"},
        {"step": 10, "moving_instance": "bridge#1", "fixed_instance": "pipe#1"},
        {"step": 10, "moving_instance": "bridge#1", "fixed_instance": "host#1"},
    ]
    # the measured mate residual implicated this root upstream
    s._mate_blocking = {"adapt#4": "adapt#4"}
    s._premount_tasks = {}
    s._planner_tasks = []
    res = asyncio.run(request_repair(s, {"step": 7}))
    assert res["ok"]
    alive = {i.instance_id for i in asm.instances}
    # root + its single-upstream subtree are gone…
    assert "adapt#4" not in alive and "elbow#1" not in alive and "pipe#1" not in alive
    # …but the doubly-attached bridge and the healthy host survive
    assert "bridge#1" in alive and "host#1" in alive
    assert set(res["cascade_removed"]) == {"elbow#1", "pipe#1"}
    # joints are cleaned to the surviving set only
    ends = {
        e
        for j in asm.joints
        for e in (j.get("moving_instance"), j.get("fixed_instance"))
    }
    assert ends <= alive
