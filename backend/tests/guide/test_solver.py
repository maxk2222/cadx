"""Placement solver unit tests — deterministic math, no LLM."""

import numpy as np

from cadx.guide import solver
from cadx.guide.state import GuidePart, PortInfo


def _part(pid: str, ports: list[PortInfo], bbox=((-10, -10, -10), (10, 10, 10))) -> GuidePart:
    return GuidePart(
        id=pid,
        name=pid,
        ports=ports,
        bbox_min=list(bbox[0]),
        bbox_max=list(bbox[1]),
    )


def test_port_to_target_alignment() -> None:
    """Boss (+z male) onto plate bore (-z female) → mating transform puts
    the moving port exactly on the target port, normals antiparallel."""
    moving = _part(
        "peg",
        [
            PortInfo(
                name="tip", kind="cylinder", pos=[0, 0, 10], axis=[0, 0, 1], radius=5, gender="male"
            )
        ],
    )
    fixed = _part(
        "plate",
        [
            PortInfo(
                name="hole",
                kind="cylinder",
                pos=[30, 0, 0],
                axis=[0, 0, -1],
                radius=5,
                gender="female",
            )
        ],
        bbox=((0, -20, -5), (60, 20, 5)),
    )
    ident = np.eye(4)
    T = solver.transform_port_to_target(
        moving, moving.ports[0], ident, fixed.ports[0], fixed, rotation_degrees=0
    )
    gap, ndot = solver.evaluate_mate(moving.ports[0], T, fixed.ports[0], ident)
    assert gap < 1e-6
    assert ndot < -0.999
    # proper rotation, no reflection (KIARA Bug 3 regression)
    assert np.linalg.det(T[0:3, 0:3]) > 0.99


def test_roll_snap_pivots_at_target() -> None:
    """90° roll must keep the mating port on its target (KIARA RC-1 bug:
    rotating about origin displaced the port by mm-scale distances)."""
    moving = _part(
        "arm",
        [PortInfo(name="base", pos=[0, 0, -50], axis=[0, 0, -1], radius=8, gender="male")],
        bbox=((-10, -10, -50), (10, 10, 50)),
    )
    fixed = _part(
        "base",
        [PortInfo(name="socket", pos=[0, 0, 100], axis=[0, 0, 1], radius=8, gender="female")],
        bbox=((-30, -30, 90), (30, 30, 110)),
    )
    ident = np.eye(4)
    for deg in (0, 90, 180, 270):
        T = solver.transform_port_to_target(
            moving, moving.ports[0], ident, fixed.ports[0], fixed, rotation_degrees=deg
        )
        gap, ndot = solver.evaluate_mate(moving.ports[0], T, fixed.ports[0], ident)
        assert gap < 1e-6, f"roll {deg}° moved the mate by {gap} mm"
        assert ndot < -0.999


def test_solve_attachment_candidates() -> None:
    moving = _part(
        "peg",
        [PortInfo(name="tip", pos=[0, 0, 5], axis=[0, 0, 1], radius=5, gender="male")],
    )
    fixed = _part(
        "plate",
        [PortInfo(name="hole", pos=[0, 0, 0], axis=[0, 0, -1], radius=5, gender="female")],
    )
    cands = solver.solve_attachment(
        moving, "tip", fixed, solver.identity_transform(), "hole", rotation_snaps=[0, 90]
    )
    assert len(cands) == 4  # 2 rolls x 2 axis conventions (rim axes are sign-ambiguous)
    assert all(c["geometry_ok"] for c in cands)
    assert cands[0]["joint_gap_mm"] < 0.05
    assert {c["axis_flipped"] for c in cands} == {False, True}


def test_gender_and_radius_gates() -> None:
    m = _part("m", [PortInfo(name="a", pos=[0, 0, 0], axis=[0, 0, 1], radius=5, gender="male")])
    f_same = _part(
        "f1", [PortInfo(name="b", pos=[0, 0, 0], axis=[0, 0, -1], radius=5, gender="female")]
    )
    f_tiny = _part(
        "f2", [PortInfo(name="b", pos=[0, 0, 0], axis=[0, 0, -1], radius=1, gender="female")]
    )
    male2 = _part(
        "m2", [PortInfo(name="b", pos=[0, 0, 0], axis=[0, 0, -1], radius=5, gender="male")]
    )
    assert solver.find_matching_ports(m, f_same)[0]["score"] > 0
    assert solver.find_matching_ports(m, f_tiny) == []  # radius ratio < 0.5
    assert solver.find_matching_ports(m, male2) == []  # male→male rejected


def test_multi_attachment_picks_residual_min() -> None:
    moving = _part(
        "flange",
        [
            PortInfo(name="p1", pos=[0, 0, -10], axis=[0, 0, -1], radius=4, gender="male"),
            PortInfo(name="p2", pos=[20, 0, -10], axis=[0, 0, -1], radius=4, gender="male"),
        ],
    )
    fixed = _part(
        "deck",
        [
            PortInfo(name="h1", pos=[0, 0, 0], axis=[0, 0, 1], radius=4, gender="female"),
            PortInfo(name="h2", pos=[20, 0, 0], axis=[0, 0, 1], radius=4, gender="female"),
        ],
        bbox=((-20, -20, -5), (40, 20, 5)),
    )
    T = solver.solve_multi_attachment(
        moving, fixed, solver.identity_transform(), [("p1", "h1"), ("p2", "h2")]
    )
    assert T is not None
    m = np.asarray(T)
    for mp, tp in zip(moving.ports, fixed.ports, strict=True):
        gap, _ = solver.evaluate_mate(mp, m, tp, np.eye(4))
        assert gap < 2.0  # both holes within reach (roll resolved)


def test_free_placement_on_table_and_top() -> None:
    part = _part("box", [], bbox=((-10, -10, 0), (10, 10, 20)))
    t = solver.solve_free_placement(part, [], "on table")
    mn, _mx = solver.transform_bbox(part, t)
    assert abs(mn[2]) < 1e-6  # rests on z=0
    stacked = solver.solve_free_placement(
        part,
        [{"bbox_min_world": [-5, -5, 0], "bbox_max_world": [5, 5, 40]}],
        "top",
    )
    mn, _ = solver.transform_bbox(part, stacked)
    assert abs(mn[2] - 42.0) < 1e-6  # on top (40) + 2mm gap


def test_direction_hint_parses_vertical_down() -> None:
    vec, spec = solver.direction_hint("Rohr vertikal nach unten führen")
    assert spec == "dir"
    assert vec is not None and list(vec) == [0.0, 0.0, -1.0]


def test_direction_hint_axis_words() -> None:
    vec, spec = solver.direction_hint("vertikal montieren")
    assert vec is None and spec == "z"
    vec, spec = solver.direction_hint("horizontal verlegen")
    assert vec is None and spec == "xy"


def test_direction_hint_none_on_plain_text() -> None:
    vec, spec = solver.direction_hint("Schraube M8 anziehen")
    assert vec is None and spec == ""


def test_grow_dir_score_alignment() -> None:
    down = np.array([0.0, 0.0, -1.0])
    assert solver.grow_dir_score(down, "dir", down) == 1.0
    assert solver.grow_dir_score(-down, "dir", down) == 0.0
    assert solver.grow_dir_score(np.array([1.0, 0, 0]), "z", None) == 0.0
    assert solver.grow_dir_score(down, "z", None) == 1.0
    assert solver.grow_dir_score(down, "xy", None) == 0.0
    # unknown spec / degenerate dir → neutral 0.5
    assert solver.grow_dir_score(None, "dir", down) == 0.5


def test_candidate_grow_dir_from_port_to_body() -> None:
    moving = _part("pipe", [], bbox=((0, 0, -500), (10, 10, 0)))
    fixed = _part(
        "elbow",
        [PortInfo(name="out", kind="socket", pos=[5, 5, 0], axis=[0, 0, -1], radius=8)],
    )
    gdir = solver.candidate_grow_dir(
        moving,
        np.eye(4),
        fixed,
        np.eye(4),
        "out",
    )
    assert gdir is not None
    assert gdir[2] < -0.9  # body centre far below the port → −z growth
