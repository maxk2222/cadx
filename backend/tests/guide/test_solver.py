"""Placement solver unit tests — deterministic math, no LLM."""

import numpy as np
import pytest

from kiara.guide import solver
from kiara.guide.state import GuidePart, PortInfo


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
    assert gdir[2] < -0.9  # body centre far below the port → -z growth


def test_direction_hint_end_side_phrases() -> None:
    # "untere Seite in die Muffe" → bottom end mates, body grows up
    vec, spec = solver.direction_hint("Rohr mit der unteren Seite in die Muffe stecken")
    assert spec == "dir" and vec is not None and vec[2] == 1.0
    vec, spec = solver.direction_hint("Obere Seite des Rohrs in die Muffe des Adapters")
    assert spec == "dir" and vec is not None and vec[2] == -1.0
    # bare positional adjective without insertion → no hint
    vec, spec = solver.direction_hint("auf der unteren Seite des Gehäuses markieren")
    assert vec is None and spec == ""


def test_direction_hint_vertical_ruehre_cross_ref() -> None:
    vec, spec = solver.direction_hint("Rohrbogen auf linke vertikale Röhre stecken")
    assert vec is None and spec == "z"


def test_world_bbox_implementations_agree() -> None:
    """The 8-corner AABB-under-transform exists once (solver.world_bbox) —
    transform_bbox, _rotated_bbox, _bbox_corners_world and the router's
    _world_bbox all delegate to it."""
    from kiara.api.routers import guide as guide_router
    from kiara.guide import preassembly

    mn_l = [-3.0, -7.0, 1.0]
    mx_l = [11.0, -1.0, 6.0]
    rot = solver.axis_angle_rotation_matrix(np.array([0.3, 0.8, 0.5]), 137.0)
    t = np.eye(4)
    t[0:3, 0:3] = rot
    t[0:3, 3] = [40.0, -20.0, 7.0]

    ref = solver.world_bbox(mn_l, mx_l, t)
    mn2, mx2 = guide_router._world_bbox(mn_l, mx_l, t.tolist())
    assert np.allclose(ref[0], mn2) and np.allclose(ref[1], mx2)
    part = _part("p", [], bbox=(mn_l, mx_l))
    mn3, mx3 = solver.transform_bbox(part, t.tolist())
    assert np.allclose(ref[0], mn3) and np.allclose(ref[1], mx3)
    corners = np.asarray(preassembly._bbox_corners_world(mn_l, mx_l, t.tolist()))
    assert corners.shape == (8, 3)
    assert np.allclose(ref[0], corners.min(axis=0))
    assert np.allclose(ref[1], corners.max(axis=0))
    # pure rotation: the 3x3 variant equals the 4x4 with zero translation
    mn4, mx4 = preassembly._rotated_bbox(mn_l, mx_l, rot)
    t_rot = np.eye(4)
    t_rot[0:3, 0:3] = rot
    ref_rot = solver.world_bbox(mn_l, mx_l, t_rot)
    assert np.allclose(ref_rot[0], mn4) and np.allclose(ref_rot[1], mx4)


def _union_containment_ref(
    mn: np.ndarray, mx: np.ndarray, others: list[tuple[np.ndarray, np.ndarray]], grid: int = 8
) -> float:
    """The pre-vectorization triple loop — parity oracle."""
    if not others:
        return 0.0
    span = mx - mn
    if float(np.prod(span)) <= 1e-9:
        return 0.0
    qs = [
        np.linspace(mn[i] + span[i] / (2 * grid), mx[i] - span[i] / (2 * grid), grid)
        for i in range(3)
    ]
    inside = 0
    for x in qs[0]:
        for y in qs[1]:
            for z in qs[2]:
                p = np.array([x, y, z])
                for omn, omx in others:
                    if bool(np.all(p >= omn) and np.all(p <= omx)):
                        inside += 1
                        break
    return inside / (grid**3)


def test_union_containment_matches_reference() -> None:
    """The numpy-broadcast implementation must reproduce the loop exactly —
    same grid points, same inside-any-box-once semantics."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        mn = rng.uniform(-50, 0, 3)
        mx = mn + rng.uniform(1, 40, 3)
        others = []
        for _ in range(rng.integers(1, 5)):
            omn = rng.uniform(-60, 40, 3)
            others.append((omn, omn + rng.uniform(0.5, 30, 3)))
        got = solver.union_containment(mn, mx, others)
        assert got == pytest.approx(_union_containment_ref(mn, mx, others))
    # edge cases
    assert solver.union_containment(np.zeros(3), np.ones(3), []) == 0.0
    assert (
        solver.union_containment(np.zeros(3), np.zeros(3), [(np.zeros(3), np.ones(3))]) == 0.0
    )
    assert solver.union_containment(np.zeros(3), np.ones(3), [(np.zeros(3), np.ones(3))]) == 1.0
    # overlap between two 'others' counts a shared region once
    half = solver.union_containment(
        np.zeros(3), np.ones(3), [(np.full(3, 0.5), np.ones(3)), (np.full(3, 0.75), np.ones(3))]
    )
    assert half == pytest.approx(_union_containment_ref(np.zeros(3), np.ones(3), [(np.full(3, 0.5), np.ones(3))]))


def test_disconnected_premounted_instances_are_roots() -> None:
    """A fitting chain hanging off a device-mounted (premounted) part is
    connected to the MACHINE, not the casing — premounted instances are
    second rigid anchors, not debris."""
    import numpy as np

    boxes = {
        "casing#1": (np.array([-50, -50, 0.0]), np.array([50, 50, 300.0])),
        "adapter#1": (np.array([-60, -20, -30.0]), np.array([-30, 20, -10.0])),
        "elbow#1": (np.array([-60, -20, -60.0]), np.array([-30, 20, -30.0])),
        "pipe#1": (np.array([-55, -10, -560.0]), np.array([-35, 10, -60.0])),
        "debris#1": (np.array([300.0, 300.0, 0.0]), np.array([320.0, 320.0, 20.0])),
    }
    # casing subtree joined to premounted adapter#1 — physically the pump
    # body chain: adapter (premount root) → elbow → pipe
    pairs = [
        ("adapter#1", "elbow#1"),
        ("elbow#1", "pipe#1"),
    ]
    ids = sorted(boxes)
    # without premount roots the whole lower chain reads as debris
    lost = solver.disconnected_instances(boxes, pairs, ids)
    assert {r["instance"] for r in lost} == {"adapter#1", "elbow#1", "pipe#1", "debris#1"}
    # with adapter#1 as premounted root only the true debris remains
    lost = solver.disconnected_instances(
        boxes, pairs, ids, extra_roots={"adapter#1"}
    )
    assert {r["instance"] for r in lost} == {"debris#1"}
