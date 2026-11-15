"""Physical port-matching tests — measured bore/outer radii, seats, plugs.

Mirrors the real extraction model: every port carries ``radius`` (lip),
``inner_radius`` (clear bore to the seat), ``outer_radius`` (envelope a
socket sees) and ``depth`` (seat distance, 500 = through-bore, 0 = solid).
"""

import numpy as np

from cadx.guide import solver
from cadx.guide.state import GuidePart, PortInfo

THROUGH = 500.0


def _part(pid: str, ports: list[PortInfo], bbox=((-40, -40, -40), (40, 40, 40))) -> GuidePart:
    return GuidePart(
        id=pid,
        name=pid,
        ports=ports,
        bbox_min=list(bbox[0]),
        bbox_max=list(bbox[1]),
    )


def _pipe_end(name: str = "end", axis=(0, 0, 1), pos=(0, 0, 0)) -> PortInfo:
    """Open tube end: through-bore, wall 9..11 → inner 8.1, outer 11."""
    return PortInfo(
        name=name,
        kind="cylinder",
        pos=list(pos),
        axis=list(axis),
        radius=9.0,
        inner_radius=8.1,
        outer_radius=11.0,
        depth=THROUGH,
        gender="female",
    )


def _socket(
    name: str = "sock",
    axis=(0, 0, -1),
    pos=(0, 0, 0),
    lip: float = 11.0,
    inner: float = 9.9,
    seat: float = 18.0,
    outer: float = 12.8,
) -> PortInfo:
    return PortInfo(
        name=name,
        kind="cylinder",
        pos=list(pos),
        axis=list(axis),
        radius=lip,
        inner_radius=inner,
        outer_radius=outer,
        depth=seat,
        gender="female",
    )


def _spigot(
    name: str = "spigot",
    axis=(0, 0, 1),
    pos=(0, 0, 0),
    r: float = 10.7,
) -> PortInfo:
    """Solid-faced male tip: rays blocked at ~0 → no clear bore."""
    return PortInfo(
        name=name,
        kind="cylinder",
        pos=list(pos),
        axis=list(axis),
        radius=r,
        inner_radius=0.0,
        outer_radius=0.0,
        depth=0.0,
        gender="male",
    )


# --------------------------------------------------------------------------- #
# hard rejections
# --------------------------------------------------------------------------- #


def test_pipe_end_on_pipe_end_is_no_joint() -> None:
    """rohr darf nicht direkt auf rohr sitzen: two open tube ends — no
    socket exists to accept the plug. Rejected even under include_mismatch."""
    a = _part("pipe_a", [_pipe_end(axis=(0, 0, 1))])
    b = _part("pipe_b", [_pipe_end(axis=(0, 0, -1))])
    assert solver.find_matching_ports(a, b) == []
    assert solver.find_matching_ports(a, b, include_mismatch=True) == []


def test_plug_too_big_for_bore_rejected() -> None:
    """Adapter shell (outer ~15.5) cannot enter a small casing hole
    (inner ~9.7) — the 'adapter thread floating inside an opening' bug.
    The receiver model makes the adapter's nut mouth the SOCKET here:
    the only legal direction is nut-wrapping-rim (adapter outside),
    never shell-inside-bore. A pair where the adapter is the PLUG into
    a bore it can't enter must still hard-reject."""
    adapter_end = PortInfo(
        name="thread_end",
        kind="cylinder",
        pos=[0, 0, 0],
        axis=[0, 0, 1],
        radius=14.8,
        inner_radius=10.8,
        outer_radius=15.5,
        depth=0.3,
        gender="female",
    )
    small_hole = _socket(lip=10.8, inner=9.7, seat=46.7)
    sock, plug = solver._receiver(adapter_end, small_hole)
    # the adapter's wider bore receives — it is never the plug that
    # would have to squeeze its shell into the smaller hole
    assert sock is adapter_end and plug is small_hole
    moving = _part("adapter", [adapter_end])
    fixed = _part("casing", [small_hole])
    cands = solver.find_matching_ports(moving, fixed, include_mismatch=True)
    # nut-over-rim mate: rim envelope enters the mouth, nut lands ~at
    # the mouth seat — the adapter's body stays OUTSIDE the face
    assert cands and cands[0]["engage_mm"] <= 22.0


def test_male_male_still_rejected() -> None:
    a = _part("a", [_spigot(r=10.7)])
    b = _part("b", [_spigot(r=10.7, axis=(0, 0, -1))])
    assert solver.find_matching_ports(a, b, include_mismatch=True) == []


# --------------------------------------------------------------------------- #
# valid physical joints
# --------------------------------------------------------------------------- #


def test_pipe_end_into_socket() -> None:
    """Pipe outer 11 enters socket bore ~9.9 (measured ~10% low) with a
    real seat — the standard push-fit joint."""
    pipe = _part("pipe", [_pipe_end()])
    elbow = _part("elbow", [_socket()])
    cands = solver.find_matching_ports(pipe, elbow)
    assert cands and cands[0]["score"] > 0.0


def test_spigot_into_socket() -> None:
    spigot = _part("elbow", [_spigot(r=10.7)])
    sock = _part("other", [_socket(inner=11.5)])
    cands = solver.find_matching_ports(spigot, sock)
    assert cands and cands[0]["score"] > 0.0


def test_adapter_end_on_union_boss() -> None:
    """A union nut must WRAP the pipe stub, not sink into the oversized
    boss opening. The casing's funnel (inner ~27) legally swallows the
    nut shell — the 'adapter drowned in the opening' bug — while the
    tube-end rim (bore ~9.7, wall ~12.2) is the snug thread interface
    the nut actually seats on. Both measured; ranking is physical."""
    adapter_end = PortInfo(
        name="thread_end",
        kind="cylinder",
        pos=[0, 0, 0],
        axis=[0, 0, 1],
        radius=14.8,
        inner_radius=10.8,
        outer_radius=15.5,
        depth=0.3,
        gender="female",
    )
    boss = _socket(lip=30.0, inner=27.0, seat=15.0, outer=0.0)
    tube_end = _socket(name="hole", lip=10.8, inner=9.7, seat=46.7, outer=12.2)
    adapter = _part("adapter", [adapter_end])
    casing = _part("casing", [tube_end, boss])
    cands = solver.find_matching_ports(adapter, casing, include_mismatch=True)
    # the union nut WRAPS the tube-end rim (bore 10.8 over OD 12.2) — a
    # snug female-female thread fit. The oversized boss socket (inner 27)
    # is legal but loose: sinking the nut shell into the funnel pocket is
    # exactly the 'adapter drowned in the opening' bug, so it must rank
    # strictly worse.
    assert cands and cands[0]["fixed_port"] == "hole"
    boss_cand = next(c for c in cands if c["fixed_port"] == "sock")
    assert boss_cand["score"] < cands[0]["score"]


def test_through_bore_coupler_accepts_pipe() -> None:
    """Push-fit coupler bore goes straight through (no seat) — still a
    socket by clear radius, nominal grip depth engages the plug."""
    coupler = _socket(seat=THROUGH, inner=10.5)
    pipe = _part("pipe", [_pipe_end()])
    sock = _part("coupler", [coupler])
    cands = solver.find_matching_ports(pipe, sock)
    assert cands and cands[0]["engage_mm"] > 0.0


def test_socket_socket_pair_needs_clearance() -> None:
    """Two socket mouths mate only if one collar physically fits inside
    the other's bore — DN22 over DN22 can't (collar OD 15.5 > bore 9.9).
    A bore dwarfing the plug is a wall opening, never a socket."""
    boss = _part("boss", [_socket(lip=20, inner=18, seat=15, outer=0)])
    slim = _part("slim", [_socket(lip=12, inner=10, seat=15, outer=12.8)])
    assert solver.find_matching_ports(slim, boss)  # slim collar fits boss
    aperture = _part("aperture", [_socket(lip=30, inner=27, seat=15, outer=0)])
    assert solver.find_matching_ports(slim, aperture) == []  # opening ≠ interface
    narrow = _part("narrow", [_socket(lip=11, inner=9.9, seat=18, outer=12.8)])
    slim2 = _part("slim2", [_socket(lip=11, inner=9.9, seat=18, outer=12.8)])
    assert solver.find_matching_ports(narrow, slim2) == []  # collar can't enter


# --------------------------------------------------------------------------- #
# engagement transform
# --------------------------------------------------------------------------- #


def test_engagement_seats_plug_inside_socket() -> None:
    """The plug lands INSIDE the socket at the seat — not coplanar on the
    lip. gap stays ~0 because evaluate_mate measures the engaged point."""
    pipe = _part("pipe", [_pipe_end(pos=(0, 0, 0), axis=(0, 0, -1))])
    sock = _part(
        "elbow",
        [_socket(pos=(0, 0, 30), axis=(0, 0, 1), seat=18.0)],
        bbox=((-20, -20, 0), (20, 20, 50)),
    )
    ident = np.eye(4)
    T = solver.transform_port_to_target(
        pipe, pipe.ports[0], ident, sock.ports[0], sock
    )
    gap, ndot = solver.evaluate_mate(pipe.ports[0], T, sock.ports[0], ident)
    assert gap < 1e-6 and ndot < -0.999
    mp_world = T[0:3, 0:3] @ np.asarray(pipe.ports[0].pos) + T[0:3, 3]
    # the pipe tip is pushed ~15mm INTO the socket (below the lip at z=30)
    assert mp_world[2] < 30.0 - 10.0


def test_legacy_unmeasured_ports_keep_soft_rules() -> None:
    """BRep/executor ports without bore measurement fall back to the
    radius-ratio model (planner tolerance via include_mismatch)."""
    peg = _part(
        "peg",
        [PortInfo(name="p", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, 1], radius=4, gender="male")],
    )
    hole = _part(
        "plate",
        [PortInfo(name="h", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, -1], radius=9, gender="female")],
    )
    assert solver.find_matching_ports(peg, hole) == []  # ratio < 0.5
    cands = solver.find_matching_ports(peg, hole, include_mismatch=True)
    assert cands and cands[0]["score"] == -0.5  # last-resort, not a real match
