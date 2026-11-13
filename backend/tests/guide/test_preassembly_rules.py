"""Regression tests for the reworked preassembly pipeline rules.

Each test pins a generalizable physical rule, not the Kiara demo:
- port KIND incompatibility (plane never mates a rim)
- radius mismatch = explicit opt-in, last-resort ranking
- collar demotion (coincident male rim behind a female mouth)
- coincident ports share occupancy (spigot rim + bore lip = one feature)
- "Öffnung nach oben" is an ORIENTATION phrase, not a location axis
- bridge-feed host ports must be perpendicular to the incoming leg
- +Y front default when the text says neither front nor rear
- premount spawning only for genuinely per-side instance semantics
- anchor-lift chain traversal follows ONE chain, stops at bridges
- symmetric targets pair 1:1 (oldest free instance wins)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cadx.guide import preassembly as pa
from cadx.guide import solver
from cadx.guide.state import (
    AssemblyInstance,
    AssemblyState,
    GuidePart,
    GuideState,
    PortInfo,
)


def _part(pid: str, ports: list[PortInfo] | None = None, bbox=((-10, -10, -10), (10, 10, 10))) -> GuidePart:
    return GuidePart(
        id=pid,
        name=pid,
        ports=ports or [],
        bbox_min=list(bbox[0]),
        bbox_max=list(bbox[1]),
    )


def _sess(parts: list[GuidePart], assembly: AssemblyState | None = None) -> SimpleNamespace:
    state = GuideState(output_dir="/tmp/x")
    state.parts = parts
    state.assembly = assembly or AssemblyState()
    return SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")


# --------------------------------------------------------------------- #
# solver rules
# --------------------------------------------------------------------- #


def test_kind_mismatch_never_mates() -> None:
    """A plane/flat face cannot accept a cylindrical rim — the gender
    gate alone used to let adapter face_* 'mate' casing rims."""
    moving = _part(
        "m", [PortInfo(name="p", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, 1], radius=5, gender="male")]
    )
    fixed = _part(
        "f",
        [
            PortInfo(name="face", kind="plane", pos=[0, 0, 0], axis=[0, 0, -1], radius=5, gender="flat"),
            PortInfo(name="rim", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, -1], radius=5, gender="female"),
        ],
    )
    cands = solver.find_matching_ports(moving, fixed, include_mismatch=True)
    assert [c["fixed_port"] for c in cands] == ["rim"]


def test_radius_mismatch_is_opt_in_last_resort() -> None:
    """Strict default rejects radius-incompatible pairs; the planner may
    opt in (representative kits measure collars), but mismatches rank
    below every true match."""
    m = _part("m", [PortInfo(name="a", pos=[0, 0, 0], axis=[0, 0, 1], radius=5, gender="male", kind="cylinder")])
    f = _part(
        "f",
        [
            PortInfo(name="big", pos=[0, 0, 0], axis=[0, 0, -1], radius=30, gender="female", kind="cylinder"),
            PortInfo(name="ok", pos=[0, 0, 0], axis=[0, 0, -1], radius=5, gender="female", kind="cylinder"),
        ],
    )
    assert solver.find_matching_ports(m, f) == [] or all(
        c["fixed_port"] == "ok" for c in solver.find_matching_ports(m, f)
    )
    cands = solver.find_matching_ports(m, f, include_mismatch=True)
    by_name = {c["fixed_port"]: c["score"] for c in cands}
    assert "big" in by_name  # opt-in exposes the representative match
    assert by_name["big"] < by_name["ok"]  # … but always below true fits


def test_collar_edge_demoted_fixed_side() -> None:
    """A male rim coincident with a female mouth on the same fixed part
    is the collar edge — the mouth is the real interface. (A male moving
    port would be gender-rejected anyway; sockets are the real case.)"""
    moving = _part(
        "socket_pipe",
        [PortInfo(name="end", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, 1], radius=5.5, gender="female")],
    )
    fixed = _part(
        "elbow",
        [
            # socket mouth + its collar edge at the same spot, same axis
            PortInfo(name="mouth", kind="cylinder", pos=[0, 0, 10], axis=[0, 0, -1], radius=6, gender="female"),
            PortInfo(name="collar", kind="cylinder", pos=[0, 0, 10.5], axis=[0, 0, -1], radius=6.5, gender="male"),
            PortInfo(name="other", kind="cylinder", pos=[10, 0, 0], axis=[0, 0, -1], radius=6, gender="female"),
        ],
    )
    cands = solver.find_matching_ports(moving, fixed)
    by_port = {c["fixed_port"]: c["score"] for c in cands}
    assert by_port["collar"] < by_port["mouth"]


def test_collar_moving_side_requires_recess() -> None:
    """On the moving part only a RECESS-shadowed male counts as collar —
    a genuine spigot protrudes past its bore lip and stays a real port."""
    moving = _part(
        "spigot_elbow",
        [
            # protruding spigot (extends past its coincident bore lip)
            PortInfo(name="spigot", kind="cylinder", pos=[0, 0, -10], axis=[0, 0, -1], radius=5, gender="male"),
            PortInfo(name="bore", kind="cylinder", pos=[0, 0, -9], axis=[0, 0, -1], radius=4.5, gender="female"),
            PortInfo(name="socket", kind="cylinder", pos=[10, 0, 0], axis=[0, 0, 1], radius=6, gender="female"),
        ],
    )
    fixed = _part(
        "adapter", [PortInfo(name="mouth", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, 1], radius=5.2, gender="female")]
    )
    cands = solver.find_matching_ports(moving, fixed)
    spigot = [c for c in cands if c["moving_port"] == "spigot"]
    assert spigot and spigot[0]["score"] > 0


# --------------------------------------------------------------------- #
# direction-phrase split
# --------------------------------------------------------------------- #


def test_nach_phrases_are_orientation_not_location() -> None:
    """'Öffnung nach oben' describes a FREE port's direction — mixing it
    into the location axes pushed mounts to the wrong side."""
    assert (2, 1.0) in pa._open_axes("freie Öffnung nach oben zeigend")
    assert not any(ax == 2 for ax, _ in pa._dir_axes("freie Öffnung nach oben zeigend"))
    # real location words still parse
    assert (2, -1.0) in pa._dir_axes("unterer linker Adapter")


# --------------------------------------------------------------------- #
# host port choice (feed_perp + front default)
# --------------------------------------------------------------------- #


def test_host_port_feed_perp_loses_parallel() -> None:
    """A chain feeding a 90° bridge cannot start on a stub parallel to
    the incoming leg — the bridge could never close. Perpendicular ports
    must win even against side words."""
    host = _part(
        "host",
        [
            PortInfo(name="top", kind="cylinder", pos=[0, 0, 30], axis=[0, 0, 1], radius=5, gender="male"),
            PortInfo(name="front", kind="cylinder", pos=[0, 20, 0], axis=[0, 1, 0], radius=5, gender="male"),
        ],
        bbox=((-30, -30, -30), (30, 30, 30)),
    )
    # "oberer Anschluss" prefers the top port by position …
    pick = pa._pick_host_port(host, [(2, 1.0)], feed_perp=None)
    assert pick.name == "top"
    # … but when the chain approaches along Z, top is geometrically dead
    pick = pa._pick_host_port(host, [(2, 1.0)], feed_perp=2)
    assert pick.name == "front"


def test_host_port_front_default() -> None:
    """No front/rear word → +Y ports win (frame contract: front works)."""
    host = _part(
        "host",
        [
            PortInfo(name="rear", kind="cylinder", pos=[0, -20, 0], axis=[0, -1, 0], radius=5, gender="male"),
            PortInfo(name="front", kind="cylinder", pos=[0, 20, 0], axis=[0, 1, 0], radius=5, gender="male"),
        ],
        bbox=((-30, -30, -30), (30, 30, 30)),
    )
    pick = pa._pick_host_port(host, [(0, -1.0)], feed_perp=None)
    assert pick.name == "front"
    # an explicit rear word overrides the default
    pick = pa._pick_host_port(host, [(0, -1.0), (1, -1.0)], feed_perp=None)
    assert pick.name == "rear"


# --------------------------------------------------------------------- #
# premount spawning
# --------------------------------------------------------------------- #


def test_no_premount_for_single_instance_side_words() -> None:
    """'linke obere Seite des Gehäuses' names a FEATURE on the one base
    instance — not a second base. Single-instance parts never premount."""
    host = _part("base")
    sess = _sess([host])
    tasks = [
        {"step": 1, "substep_idx": -1, "part": "base", "target": "", "anchor": True},
        {
            "step": 2,
            "substep_idx": 0,
            "part": "plug",
            "target": "base an linker oberer Seite",
            "text": "Plug in linke obere Seite montieren",
        },
    ]
    out = pa._spawn_premounts(sess, tasks)
    assert not any(t.get("premount") for t in out)


def test_premount_for_uncovered_side_signature() -> None:
    """A part placed at 'oben links' and 'oben rechts' whose target says
    'unterer linker' means a pre-mounted third instance — spawn it."""
    parts = [_part("base"), _part("adapter"), _part("elbow")]
    sess = _sess(parts)
    tasks = [
        {"step": 1, "substep_idx": -1, "part": "base", "target": "", "anchor": True},
        {
            "step": 2, "substep_idx": 0, "part": "adapter",
            "target": "base oberer linker", "text": "Adapter am oberen linken Anschluss",
        },
        {
            "step": 3, "substep_idx": 0, "part": "adapter",
            "target": "base oberer rechter", "text": "Adapter am oberen rechten Anschluss",
        },
        {
            "step": 5, "substep_idx": 0, "part": "elbow",
            "target": "unterer linker adapter", "text": "Bogen auf unteren linken Adapter",
        },
    ]
    out = pa._spawn_premounts(sess, tasks)
    prems = [t for t in out if t.get("premount")]
    assert len(prems) == 1 and prems[0]["part"] == "adapter"
    assert prems[0]["target"] == "base"
    # sorted before the referencing step
    assert out.index(prems[0]) < out.index(tasks[3])


# --------------------------------------------------------------------- #
# chain height + coincident occupancy
# --------------------------------------------------------------------- #


def test_chain_height_single_chain_stops_at_bridge() -> None:
    """Anchor lift follows the ONE chain growing from the premount —
    sibling sides and bridge tasks never inflate it. Reach is physical:
    the premount root contributes its mid-body port height, each later
    part its port-to-port span minus one engagement."""
    parts = [
        _part("base"),
        _part("adapter", bbox=((-5, -5, 0), (5, 5, 20))),
        _part(
            "elbow",
            bbox=((-5, -5, 0), (5, 5, 30)),
            ports=[
                PortInfo(name="a", kind="cylinder", pos=[0, 0, 0], axis=[0, -1, 0], radius=9),
                PortInfo(name="b", kind="cylinder", pos=[0, 0, 40], axis=[0, 0, 1], radius=9),
            ],
        ),
        _part(
            "pipe",
            bbox=((-5, -5, 0), (5, 5, 500)),
            ports=[
                PortInfo(name="e0", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, -1], radius=9),
                PortInfo(name="e1", kind="cylinder", pos=[0, 0, 500], axis=[0, 0, 1], radius=9),
            ],
        ),
        _part("bridge_elbow", bbox=((-5, -5, 0), (5, 5, 30))),
    ]
    sess = _sess(parts)
    tasks = [
        {"step": 2, "substep_idx": -1, "part": "adapter", "target": "base", "premount": True,
         "text": "unterer linker adapter"},
        {"step": 2, "substep_idx": -1, "part": "adapter", "target": "base", "premount": True,
         "text": "unterer rechter adapter"},
        {"step": 3, "substep_idx": 0, "part": "elbow", "target": "adapter"},
        {"step": 4, "substep_idx": 0, "part": "pipe", "target": "elbow"},
        {"step": 5, "substep_idx": 0, "part": "bridge_elbow", "target": "pipe", "target2": "base"},
    ]
    h = pa._chain_height(sess, tasks, 0, "base")
    # root adapter: 20/2=10; elbow port-span 40-15=25; pipe span 500-15=485
    assert h == pytest.approx(10 + 25 + 485)


def test_coincident_ports_share_occupancy() -> None:
    """rim_2 (female bore lip) and rim_4 (male spigot rim) sit at the same
    spot — physically ONE connection. Mating one must occupy the other."""
    elbow = _part(
        "elbow",
        [
            PortInfo(name="rim_2", kind="cylinder", pos=[0, 0, 10], axis=[0, 0, -1], radius=9.2, gender="female"),
            PortInfo(name="rim_4", kind="cylinder", pos=[0, 0, 10.5], axis=[0, 0, -1], radius=10.7, gender="male"),
            PortInfo(name="rim_0", kind="cylinder", pos=[10, 0, 0], axis=[1, 0, 0], radius=9.2, gender="female"),
        ],
    )
    inst = AssemblyInstance(part_id="elbow", transform=solver.identity_transform(), placed_in_step=1)
    inst.instance_id = "elbow#1"
    sess = _sess([elbow], AssemblyState(
        instances=[inst],
        joints=[{
            "step": 1, "moving_part": "pipe", "fixed_part": "elbow",
            "moving_instance": "pipe#1", "fixed_instance": "elbow#1",
            "moving_port": "p0", "fixed_port": "rim_2",
        }],
    ))
    used = pa._used_port_names(sess, inst)
    assert {"rim_2", "rim_4"} <= used
    assert "rim_0" not in used


# --------------------------------------------------------------------- #
# symmetric target pairing
# --------------------------------------------------------------------- #


def test_pick_target_pairs_symmetric_1to1() -> None:
    """Left/right adapters each take their own pipe — the oldest instance
    with a free port wins, so pipes never pile onto the newest adapter."""
    adapter = _part(
        "adapter",
        [
            PortInfo(name="in", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, -1], radius=5, gender="female"),
            PortInfo(name="out", kind="cylinder", pos=[10, 0, 0], axis=[1, 0, 0], radius=5, gender="female"),
        ],
    )
    pipe = _part("pipe", [PortInfo(name="e", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, 1], radius=5, gender="male")])
    i1 = AssemblyInstance(part_id="adapter", transform=solver.identity_transform(), placed_in_step=1)
    i1.instance_id = "adapter#1"
    i2 = AssemblyInstance(part_id="adapter", transform=solver.identity_transform(), placed_in_step=1)
    i2.instance_id = "adapter#2"
    sess = _sess([adapter, pipe], AssemblyState(instances=[i1, i2], joints=[]))
    first = pa._pick_target_instance(sess, "adapter", moving_pid="pipe")
    assert first == "adapter#1"
    sess.state.assembly.joints.append({
        "step": 2, "moving_part": "pipe", "fixed_part": "adapter",
        "moving_instance": "pipe#1", "fixed_instance": "adapter#1",
        "moving_port": "e", "fixed_port": "out",
    })
    second = pa._pick_target_instance(sess, "adapter", moving_pid="pipe")
    assert second == "adapter#2"


def test_port_qualifier_words_do_not_poison_side_signature() -> None:
    """'obere Öffnung des linken unteren Rohrbogens' — 'obere' modifies the
    PORT, not the elbow's location. A {links,unten,oben} ref that isn't
    covered by the elbow's {links,unten} sig spawned a phantom premount
    instance in the live run (duplicate elbows, horizontal L500s)."""
    parts = [
        GuidePart(id="casing", name="Gehäuse"),
        GuidePart(id="adapter", name="Adapter DN22"),
        GuidePart(id="l500", name="Rohr DN22, l = 500 mm"),
        GuidePart(id="elbow_in", name="Rohrbogen DN22, 1x außen 1x innen"),
        GuidePart(id="elbow_out", name="Rohrbogen DN22, 2x außen"),
    ]
    sess = _sess(parts)
    tasks = [
        {"step": 1, "substep_idx": -1, "part": "casing", "target": "", "anchor": True},
        {"step": 2, "substep_idx": 0, "part": "adapter", "target": "Gehäuse",
         "text": "Adapter auf das Gewinde des linken oberen Anschlusses"},
        {"step": 3, "substep_idx": 0, "part": "adapter", "target": "Gehäuse",
         "text": "Adapter auf das Gewinde des rechten oberen Anschlusses"},
        {"step": 6, "substep_idx": 0, "part": "Rohrbogen DN22, 1x außen 1x innen",
         "target": "Adapter",
         "text": "Rohrbogen in den vormontierten Adapter am linken unteren "
                "Anschluss einstecken, Öffnung zeigt nach oben"},
        {"step": 7, "substep_idx": 0, "part": "Rohrbogen DN22, 1x außen 1x innen",
         "target": "Adapter",
         "text": "Rohrbogen in den vormontierten Adapter am rechten unteren "
                "Anschluss einstecken, Öffnung zeigt nach oben"},
        {"step": 8, "substep_idx": 0, "part": "Rohr DN22, l = 500 mm",
         "target": "Rohrbogen DN22, 1x außen 1x innen",
         "text": "Rohr DN22 l=500mm in die obere Öffnung des linken unteren "
                "Rohrbogens einstecken"},
        {"step": 10, "substep_idx": 0, "part": "Rohr DN22, l = 500 mm",
         "target": "Rohrbogen DN22, 1x außen 1x innen",
         "text": "Rohr DN22 l=500mm in die obere Öffnung des rechten unteren "
                "Rohrbogens einstecken"},
    ]
    out = pa._spawn_premounts(sess, tasks)
    prems = [t for t in out if t.get("premount")]
    # only the two lower adapter premounts — NO elbow premounts
    assert len(prems) == 2 and all(t["part"] == "adapter" for t in prems)
