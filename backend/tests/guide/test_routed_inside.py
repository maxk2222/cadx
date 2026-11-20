"""Regression tests for the cavity-routing audit.

The rule, not the fixture: a part passing THROUGH a foreign part's
hollow interior — never jointed to it, jointed to something outside —
is physically wrong even though boolean collision sees nothing and
AABB burial exempts enclosures. Legit enclosure content must stay
clean: content jointed to its host, and content whose partners are
all inside with it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from cadx.guide import solver
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


def _boxes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Hollow host + a pipe passing through it + its outside partner."""
    return {
        "host#1": (np.array([0.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0])),
        "pipe#1": (np.array([40.0, 40.0, -50.0]), np.array([60.0, 60.0, 80.0])),
        "elbow#1": (np.array([40.0, 40.0, -100.0]), np.array([60.0, 60.0, -60.0])),
        "pcb#1": (np.array([40.0, 40.0, 40.0]), np.array([60.0, 60.0, 60.0])),
    }


def test_routed_inside_flags_pass_through() -> None:
    """pipe#1 sits inside host#1's AABB, is NOT jointed to it, and its
    partner elbow#1 sits outside — the pass-through signature the mesh
    probe then confirms."""
    boxes = _boxes()
    joints = [("pipe#1", "elbow#1")]
    frac = lambda h, p: 0.6 if (h, p) == ("host#1", "pipe#1") else 0.0  # noqa: E731
    flags = solver.routed_inside_instances(boxes, joints, frac)
    assert len(flags) == 1
    assert flags[0]["instance"] == "pipe#1"
    assert flags[0]["host"] == "host#1"
    assert flags[0]["outside_partner"] == "elbow#1"


def test_routed_inside_engagement_and_content_stay_clean() -> None:
    """A part jointed to the host is engagement, not routing; content
    whose partners are all inside with it is enclosure content, not a
    pass-through."""
    boxes = _boxes()
    frac = lambda h, p: 0.6 if p == "pipe#1" else 0.5  # noqa: E731
    # jointed to the host → exempt even with an outside partner
    flags = solver.routed_inside_instances(
        boxes, [("pipe#1", "host#1"), ("pipe#1", "elbow#1")], frac
    )
    assert flags == []
    # all partners inside the same host → enclosure content, not routing
    flags = solver.routed_inside_instances(boxes, [("pcb#1", "pipe#1")], frac)
    assert flags == []
    # below the mesh threshold → no evidence, no flag
    flags = solver.routed_inside_instances(
        boxes, [("pipe#1", "elbow#1")], lambda h, p: 0.2
    )
    assert flags == []
    # mesh probe cannot evaluate (None) → no evidence, never a violation
    flags = solver.routed_inside_instances(
        boxes, [("pipe#1", "elbow#1")], lambda h, p: None
    )
    assert flags == []


def test_flagged_instances_includes_boundary_root(monkeypatch: Any) -> None:
    """Repairing the inside link itself re-derives the same route under
    the same chain root — the flagged set must carry the premount root
    upstream of the inside link, not just the leaf."""
    import cadx.guide.assembly_tools as at

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
    audit = {
        "bad_joints": [],
        "buried_instances": [],
        "disconnected_instances": [],
        "unfulfilled_mates": [],
        "unexpected_mates": [],
        "bridge_incomplete": [],
        "routed_inside": [
            {
                "instance": "pipe#1",
                "host": "host#1",
                "inside_frac": 0.5,
                "outside_partner": "elbow#1",
                "issue": "pipe#1 routes through host#1",
                "boundary_root": "adapt#4",
                "root_step": 7,
            }
        ],
    }
    monkeypatch.setattr(at, "joint_geometry_audit", lambda sess: audit)
    flagged = at._flagged_instances(s)
    assert {"pipe#1", "adapt#4"} <= flagged
