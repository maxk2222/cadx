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

import numpy as np
import pytest

from cadx.guide import preassembly as pa
from cadx.guide import solver
from cadx.guide.state import (
    AssemblyInstance,
    AssemblyState,
    GuidePart,
    GuideState,
    GuideStep,
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
    'vormontierter unterer linker' means a pre-mounted third instance —
    the explicit marker licenses the spawn."""
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
            "target": "unterer linker vormontierter adapter",
            "text": "Bogen auf unteren linken Adapter",
        },
    ]
    out = pa._spawn_premounts(sess, tasks)
    prems = [t for t in out if t.get("premount")]
    assert len(prems) == 1 and prems[0]["part"] == "adapter"
    assert prems[0]["target"] == "base"
    # sorted before the referencing step
    assert out.index(prems[0]) < out.index(tasks[3])


def test_no_premount_without_marker() -> None:
    """The same side-signature mismatch WITHOUT a premount marker is a
    reference to a placed instance, not a new copy — '(unteres Ende)'
    picks the end feature, not another pipe (observed: phantom l70/l500
    copies that consumed the kit quota and derailed the chain)."""
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
    assert not [t for t in out if t.get("premount")]


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


# --------------------------------------------------------------------- #
# interface semantics: union-nut socket, physical reject, orientation
# --------------------------------------------------------------------- #


def test_shallow_bore_female_port_is_still_a_socket() -> None:
    """A union nut's extracted depth probe only catches the mouth lip
    (~0.3mm) — depth that shallow used to demote the port to a PLUG, so
    the solver rejected the real pipe tip and fell back to a recessed
    opening. A female port with a measured bore always receives."""
    nut = PortInfo(
        name="rim_2", kind="cylinder", pos=[0, 0, 0], axis=[1, 0, 0],
        radius=15.5, inner_radius=11.06, outer_radius=15.5, depth=0.29,
        gender="female",
    )
    assert solver._socket_depth(nut) > 0.0
    solid = PortInfo(
        name="face", kind="cylinder", pos=[0, 0, 0], axis=[1, 0, 0],
        radius=15.5, inner_radius=0.0, depth=0.29, gender="female",
    )
    assert solver._socket_depth(solid) == 0.0


def test_unmeasured_oversize_plug_rejected_against_measured_socket() -> None:
    """The soft (unmeasured) path let a Ø35 bottom stub 'mate' a Ø22 union
    nut — its drawn radius is still its outer envelope and can never enter
    the measured bore. One measured side is enough for the physical reject."""
    nut = PortInfo(
        name="nut", kind="cylinder", pos=[0, 0, 0], axis=[0, 1, 0],
        radius=15.5, inner_radius=11.06, outer_radius=15.5, depth=0.29,
        gender="female",
    )
    fat_stub = PortInfo(
        name="stub", kind="cylinder", pos=[0, 0, 0], axis=[0, -1, 0],
        radius=17.4, inner_radius=0.0, depth=0.0, gender="male",
    )
    assert solver._score_pair(nut, fat_stub, include_mismatch=True) < 0.0
    # ...while a genuinely fitting stub stays legal
    slim = fat_stub.model_copy(update={"radius": 12.0})
    assert solver._score_pair(nut, slim, include_mismatch=False) > 0.0


def test_anchor_orientation_turns_connection_hardware_to_front() -> None:
    """The asset frame is arbitrary — a host whose pipe stubs point UP in
    local space must be rotated so they face the working side (+Y). The
    old identity placement left the real interfaces on top and let the
    adapters sink into whatever opening happened to face forward."""
    host = _part(
        "gehaeuse",
        [
            # exposed pipe tip pointing UP in the asset frame
            PortInfo(
                name="stub", kind="cylinder", pos=[46.0, 50.0, 70.0],
                axis=[0, 0, 1], radius=13.0, inner_radius=10.7,
                outer_radius=13.0, depth=0.0, gender="male",
            ),
            # a loose funnel mouth facing front — legal but the wrong interface
            PortInfo(
                name="funnel", kind="cylinder", pos=[0.0, 50.0, 0.0],
                axis=[0, 1, 0], radius=30.0, inner_radius=27.0,
                depth=15.0, gender="female",
            ),
        ],
        bbox=((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0)),
    )
    host.name = "Gehäuse"
    adapter = _part(
        "adapter",
        [
            PortInfo(
                name="nut", kind="cylinder", pos=[10.0, 0, 0],
                axis=[1, 0, 0], radius=15.5, inner_radius=14.0,
                outer_radius=15.5, depth=0.3, gender="female",
            )
        ],
        bbox=((0.0, -16.0, -16.0), (16.0, 16.0, 16.0)),
    )
    sess = _sess([host, adapter])
    sess.state.steps = [GuideStep(number=1, title="Adapter montieren")]
    tasks = [
        {"step": 1, "part": "adapter", "target": "Anschluss am Gehäuse",
         "text": "Adapter auf den Anschluss schrauben"},
    ]
    rot = pa._anchor_orientation(sess, host, tasks)
    stub_w = rot @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(stub_w, [0.0, 1.0, 0.0], atol=1e-6)


def test_anchor_orientation_never_picks_recessed_port() -> None:
    """A mouth 100mm inside the body is an interior junction, not a mount
    interface — no fitting reaches it without tunneling through the part.
    If it is the ONLY fit, orientation must not rotate for it."""
    host = _part(
        "gehaeuse",
        [
            PortInfo(
                name="inner", kind="cylinder", pos=[0.0, 0.0, 0.0],
                axis=[0, 0, 1], radius=12.0, inner_radius=11.0,
                depth=100.0, gender="female",
            ),
        ],
        bbox=((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0)),
    )
    host.name = "Gehäuse"
    plug_part = _part(
        "rohr",
        [
            PortInfo(
                name="spigot", kind="cylinder", pos=[0, 0, 0],
                axis=[0, 0, 1], radius=10.0, gender="male",
            )
        ]
    )
    sess = _sess([host, plug_part])
    sess.state.steps = [GuideStep(number=1)]
    tasks = [
        {"step": 1, "part": "rohr", "target": "Gehäuse",
         "text": "Rohr in den Anschluss"},
    ]
    rot = pa._anchor_orientation(sess, host, tasks)
    assert np.allclose(rot, np.eye(3))


def test_attach_hint_cannot_outrank_physical_fit() -> None:
    """'oberer Anschluss' added the port's raw world height (hundreds of
    mm) to the candidate hint — a top funnel mouth beat the real front
    stub purely by being higher. Location words must disambiguate
    near-tied interfaces, never overturn the physical fit; and a mouth
    reachable only by tunneling through the body is not an interface."""
    host = _part(
        "gehaeuse",
        [
            # the real connection: exposed pipe tip on the front face, low
            PortInfo(
                name="stub", kind="cylinder", pos=[-46.0, 71.0, 50.0],
                axis=[0, 1, 0], radius=11.9, inner_radius=10.7,
                outer_radius=13.1, depth=0.0, gender="female",
            ),
            # a coarse male stub nested in a top funnel — the 'opening on
            # top' the adapter kept being fitted onto
            PortInfo(
                name="top_stub", kind="cylinder", pos=[-60.0, -60.0, 355.0],
                axis=[0, 0, 1], radius=21.4, inner_radius=21.4,
                outer_radius=21.4, depth=0.0, gender="male", nested_in="funnel",
            ),
            PortInfo(
                name="funnel", kind="cylinder", pos=[-60.0, -60.0, 355.0],
                axis=[0, 0, 1], radius=30.0, inner_radius=27.0,
                depth=15.0, gender="female",
            ),
            # interior junction 130mm inside the body — good fit, unreachable
            PortInfo(
                name="inner", kind="cylinder", pos=[-46.0, -60.0, 50.0],
                axis=[0, 1, 0], radius=12.1, inner_radius=3.2,
                outer_radius=12.1, depth=0.0, gender="male", nested_in="inner_sock",
            ),
        ],
        bbox=((-125.0, -195.0, 0.0), (125.0, 71.5, 355.0)),
    )
    host.name = "Gehäuse"
    adapter = _part(
        "adapter",
        [
            PortInfo(
                name="nut", kind="cylinder", pos=[10.0, 0, 0],
                axis=[1, 0, 0], radius=15.5, inner_radius=11.06,
                outer_radius=15.5, depth=0.3, gender="female",
            )
        ],
        bbox=((0.0, -16.0, -16.0), (16.0, 16.0, 16.0)),
    )
    sess = _sess([host, adapter])
    inst = AssemblyInstance(
        part_id="gehaeuse", instance_id="gehaeuse#1",
        transform=np.eye(4).tolist(), placed_in_step=1,
    )
    rows = pa._score_pairs_for_instance(
        sess, adapter, inst, "gehaeuse#1",
        axes=[(0, -1.0), (2, 1.0)],  # "linker oberer Anschluss"
        open_axes=[], orient=None, feed_perp=None, picked=False,
    )
    rows.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    assert rows and rows[0][4]["fixed_port"] == "stub"
    picked = {r[4]["fixed_port"] for r in rows}
    assert "inner" not in picked  # recessed past reach — never an interface
    assert "top_stub" in picked   # legal thread seat — but below the snug stub
    assert rows[0][2] > next(r[2] for r in rows if r[4]["fixed_port"] == "top_stub")


# --------------------------------------------------------------------- #
# bridge residual → measured host correction → retry
# --------------------------------------------------------------------- #


def test_bridge_miss_residuals_drive_per_axis_consensus() -> None:
    """Bound-rejected bridge candidates keep their measured residual —
    the host subtree shift is the per-axis MEAN of them, signed by which
    side of the mate sits in the anchor subgraph."""
    sess = _sess([])
    tasks = [
        {"bridge_miss": {"resid": [0.0, 8.0, 16.0], "i1": "el#1", "i2": "p#1", "res": 17.9}},
        {"bridge_miss": {"resid": [0.0, 9.0, 17.0], "i1": "el#2", "i2": "p#2", "res": 19.2}},
    ]
    shift = pa._bridge_closure_shift(sess, {"p#1", "p#2"}, tasks)
    assert shift is not None
    assert abs(float(shift[2]) - 16.5) < 0.6
    assert abs(float(shift[1]) - 8.5) < 0.6
    assert float(shift[0]) == 0.0


def test_bridge_miss_conflicting_axis_cancels() -> None:
    """Mirrored bridges (left needs +x, right needs -x) must NOT average
    into a sideways guess — only the axis every bridge agrees on is
    corrected."""
    sess = _sess([])
    tasks = [
        {"bridge_miss": {"resid": [30.0, 0.0, 16.0], "i1": "e1", "i2": "h1", "res": 34.0}},
        {"bridge_miss": {"resid": [-30.0, 0.0, 16.0], "i1": "e2", "i2": "h2", "res": 34.0}},
    ]
    shift = pa._bridge_closure_shift(sess, {"h1", "h2"}, tasks)
    assert shift is not None
    assert float(shift[0]) == 0.0
    assert abs(float(shift[2]) - 16.0) < 1e-6


def test_bridge_miss_sign_follows_host_side() -> None:
    """resid = moving-socket minus fixed-point: when the FIRST mate's
    instance is the one in the anchor subtree, the moving part rides
    with it — the shift opposes the residual."""
    sess = _sess([])
    tasks = [{"bridge_miss": {"resid": [0.0, 0.0, 12.0], "i1": "h", "i2": "f", "res": 12.0}}]
    shift = pa._bridge_closure_shift(sess, {"h"}, tasks)
    assert shift is not None and float(shift[2]) == -12.0


def test_bridge_miss_absurd_residual_is_no_pose_error() -> None:
    """A 300mm miss is a wrong pairing/orientation — never a placement
    offset the host subtree should chase."""
    sess = _sess([])
    tasks = [{"bridge_miss": {"resid": [0.0, 0.0, 300.0], "i1": "e", "i2": "h", "res": 300.0}}]
    assert pa._bridge_closure_shift(sess, {"h"}, tasks) is None


# --------------------------------------------------------------------- #
# cross-substep bridge legs
# --------------------------------------------------------------------- #


def _two_cyl_part(pid: str) -> GuidePart:
    return _part(
        pid,
        [
            PortInfo(name="a", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, 1], radius=5),
            PortInfo(name="b", kind="cylinder", pos=[0, 0, 0], axis=[1, 0, 0], radius=5),
        ],
    )


def test_cross_substep_leg_merges_into_bridge() -> None:
    """Text-only extractions split one physical bridge into two substeps:
    "elbow onto the vertical pipe" + "free socket onto the upper pipe".
    Same part + DIFFERENT target part = the second is the bridge leg."""
    sess = _sess([_two_cyl_part("elbow"), _part("pipe500"), _part("pipe70")])
    tasks = [
        {"step": 9, "substep_idx": 0, "part": "elbow", "target": "pipe500", "text": ""},
        {"step": 9, "substep_idx": 1, "part": "elbow", "target": "pipe70", "text": ""},
    ]
    pa._infer_bridges(sess, tasks)
    assert tasks[0]["target2"] == "pipe70"
    assert tasks[0]["target2_inferred"] is True
    assert tasks[1].get("_leg_of") == 0


def test_cross_substep_two_instances_same_host_is_no_leg() -> None:
    """"Dowel into the panel" twice — same part, SAME target part = two
    occurrences, never a bridge leg (the merge would swallow one)."""
    sess = _sess([_two_cyl_part("dowel"), _part("panel")])
    tasks = [
        {"step": 4, "substep_idx": 0, "part": "dowel", "target": "panel", "text": ""},
        {"step": 4, "substep_idx": 1, "part": "dowel", "target": "panel", "text": ""},
    ]
    pa._infer_bridges(sess, tasks)
    assert not tasks[0].get("target2")
    assert tasks[1].get("_leg_of") is None


def test_cross_substep_single_port_part_never_bridges() -> None:
    """A part with <2 cylindrical ports physically cannot close a second
    mate — no merge even with a different target."""
    one_port = _part(
        "plug", [PortInfo(name="a", kind="cylinder", pos=[0, 0, 0], axis=[0, 0, 1], radius=5)]
    )
    sess = _sess([one_port, _part("host"), _part("other")])
    tasks = [
        {"step": 3, "substep_idx": 0, "part": "plug", "target": "host", "text": ""},
        {"step": 3, "substep_idx": 1, "part": "plug", "target": "other", "text": ""},
    ]
    pa._infer_bridges(sess, tasks)
    assert not tasks[0].get("target2")
    assert tasks[1].get("_leg_of") is None


def test_lift_correct_shifts_only_host_subtree_and_clamps() -> None:
    """The correction moves the anchor subtree as ONE rigid group —
    floor-chain instances stay put — and without an estimated lift the
    host may not float in z."""
    host = _part("host")
    floor = _part("floor")
    host_i = AssemblyInstance(
        part_id="host", instance_id="host#1",
        transform=np.eye(4).tolist(), placed_in_step=1,
    )
    floor_i = AssemblyInstance(
        part_id="floor", instance_id="floor#1",
        transform=np.eye(4).tolist(), placed_in_step=2,
    )
    child_i = AssemblyInstance(
        part_id="floor", instance_id="floor#2",
        transform=np.eye(4).tolist(), placed_in_step=3,
    )
    sess = _sess(
        [host, floor],
        AssemblyState(
            instances=[host_i, floor_i, child_i],
            joints=[{
                "step": 3, "moving_part": "floor", "fixed_part": "host",
                "moving_instance": "floor#2", "fixed_instance": "host#1",
                "moving_port": "a", "fixed_port": "b",
            }],
        ),
    )
    tasks = [{"bridge_miss": {"resid": [0.0, 5.0, 20.0], "i1": "e", "i2": "host#1", "res": 20.6}}]
    rep: dict = {}
    pa._lift_correct_for_bridges(sess, "host", lift=0.0, report=rep, tasks=tasks)
    # lift=0 → z component suppressed; y still corrected
    assert np.asarray(child_i.transform)[1, 3] == pytest.approx(5.0)
    assert np.asarray(child_i.transform)[2, 3] == pytest.approx(0.0)
    assert np.asarray(host_i.transform)[1, 3] == pytest.approx(5.0)
    # floor#1 is not jointed to the host — untouched
    assert np.asarray(floor_i.transform)[1, 3] == pytest.approx(0.0)


def test_retry_bridges_restores_joints_when_still_unclosed() -> None:
    """A retry that still misses the bound must hand back the fallback
    joints it borrowed — never leave the part floating joint-less."""
    moving = _part("elbow")  # no ports → _bridge_search declines
    target = _part("pipe")
    mi = AssemblyInstance(
        part_id="elbow", instance_id="elbow#1",
        transform=np.eye(4).tolist(), placed_in_step=9,
    )
    ti = AssemblyInstance(
        part_id="pipe", instance_id="pipe#1",
        transform=np.eye(4).tolist(), placed_in_step=8,
    )
    joint = {
        "step": 9, "moving_part": "elbow", "fixed_part": "pipe",
        "moving_instance": "elbow#1", "fixed_instance": "pipe#1",
        "moving_port": "a", "fixed_port": "b",
    }
    sess = _sess(
        [moving, target], AssemblyState(instances=[mi, ti], joints=[joint])
    )
    tasks = [
        {
            "step": 9, "substep_idx": 0, "part": "elbow",
            "target": "pipe", "target2": "pipe",
            "bridge_miss": {"resid": [0.0, 0.0, 12.0], "i1": "pipe#1", "i2": "pipe#1", "res": 12.0},
        }
    ]
    import asyncio

    asyncio.run(pa._retry_bridges(sess, tasks, {}))
    assert sess.state.assembly.joints == [joint]
    assert np.asarray(mi.transform).reshape(4, 4)[0:3, 3].tolist() == [0.0, 0.0, 0.0]


def test_bound_pids_preferred_over_fuzzy_names() -> None:
    """canonicalize_refs binding wins: a task carrying part_pid/target_pid
    resolves those ids even when the free-text name is ambiguous."""
    from types import SimpleNamespace

    from cadx.guide.preassembly import _tpart, _ttgt
    from cadx.guide.state import AssemblyState, GuidePart, GuideState

    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        GuidePart(
            id="elbow_io", name="Rohrbogen DN22 1x aussen 1x innen",
            bbox_min=[0, 0, 0], bbox_max=[1, 1, 1],
        ),
        GuidePart(
            id="elbow_oo", name="Rohrbogen DN22 2x aussen",
            bbox_min=[0, 0, 0], bbox_max=[1, 1, 1],
        ),
    ]
    state.assembly = AssemblyState()
    sess = SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")

    bound = {"part": "Rohrbogen", "part_pid": "elbow_io",
             "target": "Rohrbogen", "target_pid": "elbow_oo", "step": 3}
    assert _tpart(sess, bound) == "elbow_io"
    assert _ttgt(sess, bound) == "elbow_oo"
    # unbound task falls back to live resolution (ambiguous → None)
    unbound = {"part": "Rohrbogen", "target": "Rohrbogen", "step": 3}
    assert _tpart(sess, unbound) is None
    assert _ttgt(sess, unbound) is None


def _mk_state_with_instances() -> tuple[SimpleNamespace, list]:
    """Parts + placed instances mimicking the heatpump topology."""
    from cadx.guide.state import (
        AssemblyInstance,
        AssemblyState,
        GuidePart,
        GuideState,
    )

    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        GuidePart(id="elbow_io", name="Rohrbogen DN22 1x aussen 1x innen",
                  bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]),
        GuidePart(id="elbow_oo", name="Rohrbogen DN22 2x aussen",
                  bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]),
        GuidePart(id="pipe_l70", name="Rohr DN22 l=70mm",
                  bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]),
        GuidePart(id="pipe_l500", name="Rohr DN22 l=500mm",
                  bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]),
    ]
    state.assembly = AssemblyState()
    insts = [
        # step 3/4 placed the short pipes, 5/6 the spigot elbows, 7/9 long pipes
        AssemblyInstance(instance_id="pipe_l70#1", part_id="pipe_l70",
                         placed_in_step=3),
        AssemblyInstance(instance_id="pipe_l70#2", part_id="pipe_l70",
                         placed_in_step=4),
        AssemblyInstance(instance_id="elbow_io#1", part_id="elbow_io",
                         placed_in_step=5),
        AssemblyInstance(instance_id="elbow_io#2", part_id="elbow_io",
                         placed_in_step=6),
        AssemblyInstance(instance_id="pipe_l500#1", part_id="pipe_l500",
                         placed_in_step=7),
        AssemblyInstance(instance_id="pipe_l500#2", part_id="pipe_l500",
                         placed_in_step=9),
    ]
    state.assembly.instances = insts
    sess = SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")
    return sess, insts


def test_ref_step_parsing() -> None:
    from cadx.guide.preassembly import _ref_step

    assert _ref_step("oberes Rohr aus Schritt 3") == 3
    assert _ref_step("upper pipe from step 7") == 7
    assert _ref_step("linkes Rohr") is None
    assert _ref_step("") is None


def test_ref_instances_provenance_pin() -> None:
    """'Rohr aus Schritt 3' — the bare noun is ambiguous between the two
    pipes, but provenance pins the instance step 3 placed."""
    from cadx.guide.preassembly import _ref_instances

    sess, _insts = _mk_state_with_instances()
    pool = _ref_instances(sess, "linkes oberes Rohr aus Schritt 3")
    assert [i.instance_id for i in pool] == ["pipe_l70#1"]
    # generic elbow noun + step pin → the spigot elbow of step 5
    pool = _ref_instances(sess, "Rohrbogen an der Anschlussstelle (Schritt 5)")
    assert [i.instance_id for i in pool] == ["elbow_io#1"]


def test_ref_instances_kind_pool() -> None:
    """Ambiguous kind noun without a step pin restricts to the tied
    parts' instances — never every placed instance."""
    from cadx.guide.preassembly import _ref_instances

    sess, _insts = _mk_state_with_instances()
    pool = _ref_instances(sess, "Rohrbogen")
    assert sorted(i.instance_id for i in pool) == ["elbow_io#1", "elbow_io#2"]
    # a name carrying no part signal at all → no pool (free fallthrough
    # for true feature names like 'oberer Anschlusspunkt')
    assert _ref_instances(sess, "oberer Anschlusspunkt") is None


def test_kind_pool_step_provenance_pin() -> None:
    """'(aus) Schritt N' narrows the kind pool to what the step produced —
    'oberes Rohr aus Schritt 3' is {l70}, not the whole pipe family. The
    live run bound a bridge leg against l500 because the loose pipe pool
    admitted it; a {l70}-pool would have flagged the joint as unexpected."""
    from cadx.guide.preassembly import _kind_pool
    from cadx.guide.state import GuideStep, GuideSubStep

    sess, _insts = _mk_state_with_instances()
    sess.state.steps = [
        GuideStep(
            number=3, action_type="assembly", text="linkes Rohr oben",
            sub_steps=[GuideSubStep(
                action="insert", part="Rohr DN22 l=70mm", part_pid="pipe_l70",
            )],
        ),
        GuideStep(
            number=7, action_type="assembly", text="linkes langes Rohr",
            sub_steps=[GuideSubStep(
                action="insert", part="Rohr DN22 l=500mm", part_pid="pipe_l500",
            )],
        ),
    ]
    assert _kind_pool(sess, "linkes oberes Rohr aus Schritt 3") == {"pipe_l70"}
    assert _kind_pool(sess, "vertikales Rohr aus Schritt 7") == {"pipe_l500"}
    # no pin → the full family pool stays (honest ambiguity, not a guess)
    assert _kind_pool(sess, "oberes Rohr") == {"pipe_l70", "pipe_l500"}


def test_ref_instances_empty_pin_pool_is_a_miss() -> None:
    """'(Schritt 1)' when step 1 placed NOTHING is a structured miss —
    returning None meant 'no signal' and callers free-fell onto every
    placed instance (pipe onto casing, skipping its adapter). An empty
    list is terminal evidence, not a widening license."""
    from cadx.guide.preassembly import _ref_instances

    sess, _insts = _mk_state_with_instances()
    assert _ref_instances(sess, "Bohrung des Adapters (Schritt 1)") == []
    # same without pin: a resolved kind with zero placed instances
    assert _ref_instances(sess, "Rohrbogen DN22 2x aussen") == []


def test_ref_instances_compound_noun_kind_pool() -> None:
    """German compound kind nouns classify by their suffix head —
    'Anschlussrohr' is a pipe ref even though '\brohr\b' misses the
    compound."""
    from cadx.guide.preassembly import _ref_instances

    sess, _insts = _mk_state_with_instances()
    pool = _ref_instances(sess, "freies Ende des linken oberen Anschlussrohrs")
    assert pool is not None
    assert sorted(i.instance_id for i in pool) == [
        "pipe_l500#1",
        "pipe_l500#2",
        "pipe_l70#1",
        "pipe_l70#2",
    ]


def test_infer_legs_consumes_orient_substep_declaration() -> None:
    """Extraction split one bridge into 'insert elbow onto l500' +
    'orient the free socket onto l70' — the orient sub-step is not a
    placement task, but its part+target declaration IS the second leg.
    _infer_legs must merge it into target2 and persist it on the
    placing sub-step so the audit sees the contract."""
    from types import SimpleNamespace

    from cadx.guide.preassembly import _infer_legs
    from cadx.guide.state import (
        AssemblyState,
        GuidePart,
        GuideState,
        GuideStep,
        GuideSubStep,
        PortInfo,
    )

    cyl = [
        PortInfo(name=f"r{i}", kind="cylinder", pos=[0, 0, 0],
                 axis=[0, 0, 1], radius=9, gender="female")
        for i in range(2)
    ]
    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        GuidePart(id="elbow_oo", name="Rohrbogen DN22 2x aussen", ports=cyl,
                  bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]),
        GuidePart(id="pipe_l500", name="Rohr DN22 l=500mm",
                  bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]),
        GuidePart(id="pipe_l70", name="Rohr DN22 l=70mm",
                  bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]),
    ]
    state.assembly = AssemblyState()
    state.steps = [
        GuideStep(
            number=8, text="Bogen stecken", title="t",
            sub_steps=[
                GuideSubStep(
                    action="insert",
                    part="Rohrbogen DN22 2x aussen",
                    part_pid="elbow_oo",
                    target="oberes Ende des linken vertikalen Rohrs",
                    target_pid="pipe_l500",
                ),
                GuideSubStep(
                    action="orient",
                    part="Rohrbogen DN22 2x aussen",
                    part_pid="elbow_oo",
                    target="linkes oberes Rohr DN22 l=70",
                    target_pid="pipe_l70",
                ),
            ],
        )
    ]
    sess = SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")
    tasks = [
        {
            "step": 8, "substep_idx": 0,
            "part": "Rohrbogen DN22 2x aussen", "part_pid": "elbow_oo",
            "target": "oberes Ende des linken vertikalen Rohrs",
            "target_pid": "pipe_l500", "target2": "",
        }
    ]
    _infer_legs(sess, tasks)
    assert tasks[0]["target2"] == "pipe_l70"
    assert tasks[0]["target2_inferred"] is True
    # persisted onto the placing sub-step — durable evidence for the audit
    src = state.steps[0].sub_steps[0]
    assert src.target2_pid == "pipe_l70" and "l=70" in src.target2


def test_feeds_bridge_dir_points_at_placed_partner() -> None:
    """A chain feeding a loop aims its free ports at the OTHER leg's
    placed instances: adapter placed at the lower bore (z-), the l70
    partner already sits at the upper bore (z+) — the elbow feeding the
    descending-closing bridge gets +Z hints so the solver doesn't run
    the chain away from its declared mate."""
    from cadx.guide.state import (
        AssemblyInstance,
        AssemblyState,
        GuideState,
    )

    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        _part("casing"),
        _part("adapter"),
        _part("elbow"),
        _part("pipe"),
        _part("stub"),
    ]
    state.assembly = AssemblyState(
        instances=[
            AssemblyInstance(
                instance_id="casing#1", part_id="casing", placed_in_step=1,
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ),
            AssemblyInstance(
                instance_id="adapter#1", part_id="adapter", placed_in_step=2,
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -50], [0, 0, 0, 1]],
            ),
            AssemblyInstance(
                instance_id="stub#1", part_id="stub", placed_in_step=3,
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 80], [0, 0, 0, 1]],
            ),
        ]
    )
    sess = SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")
    tasks = [
        {"step": 4, "part": "elbow", "part_pid": "elbow",
         "target": "adapter links", "target_pid": "adapter",
         "text": "Bogen auf den Adapter"},
        {"step": 5, "part": "pipe", "part_pid": "pipe",
         "target": "elbow links", "target_pid": "elbow", "text": "Rohr"},
        {"step": 6, "part": "elbow2", "part_pid": "elbow",
         "target": "pipe links", "target_pid": "pipe",
         "target2": "stub links (Schritt 3)", "target2_pid": "stub",
         "text": "Bogen schließt"},
    ]
    fd = pa._feeds_bridge_dir(sess, tasks, 0, tasks[0], "elbow")
    assert fd, "bridge-feeding elbow must get a direction hint"
    ax = dict(fd)
    assert ax.get(2) == 1.0, f"partner is ABOVE the chain root — {fd}"


def test_feeds_bridge_dir_silent_without_partner() -> None:
    """Partner not placed yet → no hint (the mechanism never invents a
    direction from thin air)."""
    from cadx.guide.state import (
        AssemblyInstance,
        AssemblyState,
        GuideState,
    )

    state = GuideState(output_dir="/tmp/x")
    state.parts = [_part("casing"), _part("adapter"), _part("elbow"), _part("pipe"), _part("stub")]
    state.assembly = AssemblyState(
        instances=[
            AssemblyInstance(
                instance_id="casing#1", part_id="casing", placed_in_step=1,
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ),
        ]
    )
    sess = SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")
    tasks = [
        {"step": 4, "part": "elbow", "part_pid": "elbow",
         "target": "adapter links", "target_pid": "adapter", "text": "x"},
        {"step": 6, "part": "elbow2", "part_pid": "elbow",
         "target": "pipe", "target_pid": "pipe",
         "target2": "stub (Schritt 9)", "target2_pid": "stub", "text": "x"},
    ]
    assert pa._feeds_bridge_dir(sess, tasks, 0, tasks[0], "elbow") == []
