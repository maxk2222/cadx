"""Regression tests for bugs found in the full-implementation audit.

Each test names the failure it would have caught:
- GuidePart.status/.error missing fields (pydantic setattr crash)
- part.part_id vs part.id (AttributeError in warning paths)
- group_vectors (unique, inverse) misused as flat list
- manifold3d has no ``*`` operator — every collision check failed
- review verdict never reopened "review" → infinite repair loop
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadx.guide import cadmeta, solver
from cadx.guide import scene as gscene
from cadx.guide.guide_runner import _review_verdict
from cadx.guide.state import GuidePart, GuideState, PortInfo, ReviewFinding


def _box_mesh(path: Path, size=(20.0, 10.0, 5.0)) -> Path:
    import trimesh

    m = trimesh.creation.box(extents=list(size))
    m.export(str(path))
    return path


def test_guidepart_status_fields_exist() -> None:
    """cadmeta sets part.status/.error — they must be model fields."""
    p = GuidePart(id="x", name="y")
    p.status = "analyzed"  # raised ValueError before the fields existed
    p.error = "e"
    assert p.status == "analyzed"


def test_mesh_port_extraction_finds_planar_ports(tmp_path: Path) -> None:
    """group_vectors returns (unique, inverse) — the flat-list misuse left
    every mesh part with zero ports."""
    mesh = _box_mesh(tmp_path / "plate.stl")
    part = GuidePart(id="plate", name="Plate", mesh_path=str(mesh))
    part = cadmeta.extract_ports_worker(part)
    assert part.status == "analyzed", part.error
    assert part.ports, "no planar ports detected on a box mesh"
    assert all(p.kind == "plane" for p in part.ports)
    assert part.bbox_max[0] == pytest.approx(10.0, abs=0.5)


def test_collision_report_real_meshes(tmp_path: Path) -> None:
    """manifold3d batch_boolean — ``a * b`` raised TypeError, flagging every
    candidate as colliding."""
    a_path = _box_mesh(tmp_path / "a.stl")
    sc = gscene.Scene()
    sc.add("a", a_path, solver.identity_transform())
    # far away — no collision
    far = np.eye(4)
    far[0, 3] = 100.0
    rep = sc.collision_report("b", a_path, far.tolist())
    assert rep["ok"], rep
    assert not rep["collisions"]
    # overlapping — real collision (same location)
    rep = sc.collision_report("b", a_path, solver.identity_transform())
    assert not rep["ok"], rep
    assert rep["collisions"][0]["volume_mm3"] > 500.0


def test_review_verdict_reopens_review_mission(tmp_path: Path) -> None:
    """Without reopening 'review' the verdict re-reads the stale report
    forever — the repair loop never terminates."""
    state = GuideState(output_dir=str(tmp_path))
    state.missions_done = ["doc", "steps", "parts", "assembly", "review", "annotate"]
    state.review.iteration = 1
    state.review.score = 5.0
    state.review.findings = [
        ReviewFinding(severity="blocker", mission="assembly", text="part floats")
    ]
    ship, notes = _review_verdict(state, tmp_path, score_target=8.0, max_iter=3)
    assert not ship and notes
    assert "review" not in state.missions_done, "review must re-run for a fresh verdict"
    assert "assembly" not in state.missions_done
    assert "annotate" not in state.missions_done
    assert "doc" in state.missions_done  # untouched


def test_free_placement_front_is_positive_y() -> None:
    """CADX convention (AGENTS.md): +Y is front."""
    part = GuidePart(id="p", name="P", bbox_min=[-5, -5, 0], bbox_max=[5, 5, 10])
    placed = [{"bbox_min_world": [-5, -5, 0], "bbox_max_world": [5, 5, 10]}]
    t = np.asarray(solver.solve_free_placement(part, placed, "front")).reshape(4, 4)
    assert t[1, 3] > 0, "front placement must move toward +Y"
    t = np.asarray(solver.solve_free_placement(part, placed, "back")).reshape(4, 4)
    assert t[1, 3] < 0, "back placement must move toward -Y"


def test_env_hint_word_boundaries() -> None:
    """'laufen'/'verkauf' must not trigger the 'auf' hint."""
    part = GuidePart(id="p", name="P", bbox_min=[0, 0, 0], bbox_max=[10, 10, 10])
    placed = [{"bbox_min_world": [0, 0, 0], "bbox_max_world": [10, 10, 10]}]
    t_top = np.asarray(solver.solve_env_placement(part, placed, "auf die Platte"))
    t_none = np.asarray(solver.solve_env_placement(part, placed, "laufen Sie es ein"))
    assert t_top[2, 3] > t_none[2, 3]


def test_evaluate_mate_antiparallel() -> None:
    mp = PortInfo(name="m", pos=[0, 0, 0], axis=[0, 0, 1])
    fp = PortInfo(name="f", pos=[0, 0, 0], axis=[0, 0, 1])
    t_move = np.eye(4)
    t_move[0:3, 0:3] = solver.axis_angle_rotation_matrix(np.array([1.0, 0, 0]), 180.0)
    gap, ndot = solver.evaluate_mate(mp, t_move, fp, np.eye(4))
    assert gap == pytest.approx(0.0, abs=1e-9)
    assert ndot == pytest.approx(-1.0, abs=1e-6)


# --------------------------------------------------------------------- #
# audit round 2
# --------------------------------------------------------------------- #


def test_ask_user_question_shape() -> None:
    """job.questions render q.question in the UI — the OpenQuestion
    contract, not {text, context}."""
    import asyncio

    from cadx.guide.session import GuideToolSession

    class _Ctx:
        runtime_settings = type("R", (), {"guide_image_budget": 24})()
        emit = None
        check_cancel = None
        executor = None

    class _S(GuideToolSession):
        def __init__(self) -> None:
            pass

    s = _S()
    s._park_fn = lambda qs: asyncio.sleep(0)
    s.submitted = None
    res = asyncio.run(
        s._tool_ask_user({"question": "welche Schraube?", "context": "für Schritt 3"})
    )
    q = s.parked_questions[0]
    assert res["ok"] and s.parked
    assert q["question"] == "welche Schraube?"
    assert q["blocking"] is True


def test_done_mission_preloads_guards(tmp_path) -> None:
    """done_mission as the FIRST tool call must still hit the parts guard —
    module import is lazy, DONE_GUARDS could be empty."""
    import importlib
    import sys

    from cadx.guide import tools as gtools

    # guards register at *_tools module import — earlier test files may
    # already have imported them, and then a bare DONE_GUARDS.clear() can
    # never repopulate: reload forces real re-registration (the lazy-import
    # path this test protects)
    gtools.DONE_GUARDS.clear()
    for name in (
        "doc_tools", "step_tools", "part_tools",
        "assembly_tools", "review_tools", "annotate_tools",
    ):
        mod = sys.modules.get(f"cadx.guide.{name}")
        if mod is not None:
            importlib.reload(mod)
    gtools.merged_handlers()
    assert "parts" in gtools.DONE_GUARDS
    assert "assembly" in gtools.DONE_GUARDS


def test_unity_model_ref_prefers_glb_basename(tmp_path: Path) -> None:
    from cadx.guide import unity

    p = GuidePart(id="a", name="A", step_path="/data/x/base.step")
    p.mesh_path = "/data/x/base.glb"
    assert unity._model_ref(p) == "base.glb"
    p2 = GuidePart(id="b", name="B", step_path="/data/x/only.step")
    assert unity._model_ref(p2) == "only.step"


def test_unity_joint_arrow_anchors_at_fixed_port(tmp_path: Path) -> None:
    """Arrow origin must sit at the mate surface (fixed port world pos),
    not the part origin — a corner-origin part would offset arrows."""
    from cadx.guide import unity
    from cadx.guide.state import AssemblyInstance, AssemblyState

    st = GuideState(output_dir=str(tmp_path))
    fixed = GuidePart(
        id="f",
        name="F",
        ports=[PortInfo(name="hole", pos=[10, 0, 5], axis=[0, 0, 1], gender="female")],
        mesh_path="/x/f.glb",
    )
    st.parts = [fixed, GuidePart(id="m", name="M", mesh_path="/x/m.glb")]
    st.assembly = AssemblyState(
        instances=[
            AssemblyInstance(part_id="f", transform=solver.identity_transform(), placed_in_step=1),
            AssemblyInstance(part_id="m", transform=solver.identity_transform(), placed_in_step=1),
        ],
        joints=[
            {
                "step": 1,
                "moving_part": "m",
                "fixed_part": "f",
                "moving_port": "p",
                "fixed_port": "hole",
                "approach_axis": [0, 0, 1],
            }
        ],
        step_instances={"1": ["m", "f"]},
    )
    st.steps = [
        # minimal step — number/display required
        __import__("cadx.guide.state", fromlist=["GuideStep"]).GuideStep(
            number=1, text="t", display_text="d"
        )
    ]
    g = unity.build_unity_guide(st)
    arrow = g["steps"][0]["arrows"][0]
    # anchor = port world pos (10,0,5) - axis*30mm -> z = 5-30 = -25mm ->
    # unity (x,z,y)*0.001 -> (0.010, -0.025, 0.0)
    assert arrow["origin"] == pytest.approx([0.010, -0.025, 0.0], abs=1e-4)


def test_emit_artifacts_registers_media(tmp_path: Path) -> None:
    """step.media_image ships next to the JSON as a render artifact."""
    import asyncio

    from cadx import storage
    from cadx.guide.guide_runner import _emit_artifacts
    from cadx.guide.state import GuideStep

    # state dir must live under data root or relpath() raises
    sd = storage.data_root() / "guide-test" / "guide"
    sd.mkdir(parents=True, exist_ok=True)
    fig = tmp_path / "fig.png"
    fig.write_bytes(b"PNG")
    st = GuideState(output_dir=str(sd))
    st.steps = [GuideStep(number=1, text="t", display_text="d", media_image=str(fig))]
    arts = asyncio.run(_emit_artifacts(st, sd))
    assert "guide:unity_guide" in arts
    media = [k for k in arts if k.startswith("guide:media:")]
    assert media and (sd / "output" / "media" / "fig.png").exists()


def test_env_hint_no_false_positive_inside_words(tmp_path: Path) -> None:
    """already covered by word-boundary test — 'verkauf' must not hit 'auf'."""
    part = GuidePart(id="p", name="P", bbox_min=[0, 0, 0], bbox_max=[10, 10, 10])
    placed = [{"bbox_min_world": [0, 0, 0], "bbox_max_world": [10, 10, 10]}]
    t = np.asarray(solver.solve_env_placement(part, placed, "beim Verkauf beachten"))
    base = np.asarray(solver.solve_env_placement(part, placed, "xyz"))
    assert np.allclose(t, base)


def test_meter_glb_normalized_to_mm(tmp_path):
    """glTF is meter-spec — an uploaded .glb under ~5 units gets a x1000
    normalized copy; all downstream consumers then see mm."""
    import trimesh

    from cadx.guide.cadmeta import normalize_mesh_units

    tube = trimesh.creation.cylinder(radius=0.011, height=0.07, sections=16)
    src = tmp_path / "pipe.glb"
    tube.export(str(src))
    out = normalize_mesh_units(src, tmp_path / "parts")
    assert out != src and out.exists()
    m = trimesh.load(out)
    dims = m.bounds[1] - m.bounds[0]
    assert max(dims) > 5.0  # 70mm now, not 0.07m


def test_rim_ports_on_elbow_mesh():
    """Closed sharp-edge loops -> real mating ports: curved fittings the
    planar-normal heuristic used to miss entirely."""
    import trimesh

    from cadx.guide.cadmeta import _rim_ports
    from cadx.guide.state import GuidePart

    # L-bend: two cylinders at right angles, sharp rim at each open end
    a = trimesh.creation.cylinder(radius=5, height=20, sections=24)
    b = trimesh.creation.cylinder(radius=5, height=20, sections=24)
    b.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0], [0, 0, 10]))
    b.apply_translation([10, 0, 10])
    mesh = trimesh.util.concatenate([a, b])
    part = GuidePart(id="p", name="elbow", source="uploaded")
    ports = _rim_ports(mesh, part)
    assert ports, "rim loops must yield ports on an open-ended fitting"
    assert all(p.radius > 0 for p in ports)


def test_mission_retries_once_on_bare_final():
    """A mission session ending without done_mission gets one nudge retry —
    plain-text endings are a wobble, not a failure."""
    import inspect

    from cadx.guide import guide_runner

    src = inspect.getsource(guide_runner._run_mission)
    assert "for attempt in (0, 1)" in src and "CONTINUATION" in src
    assert "mission_retry" in src


def test_optional_mission_failure_does_not_fail_job(monkeypatch) -> None:
    """annotate is cosmetic — its error_streak must degrade to a durable
    warning (missions_failed), never kill the 400s guide job."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide import guide_runner
    from cadx.guide.state import GuideState

    state = GuideState(output_dir="/tmp/x")
    calls = {"n": 0}

    async def fake_session(*a, **kw):
        calls["n"] += 1
        return SimpleNamespace(
            end_reason="error_streak",
            tool_log=[{"name": "add_arrow", "result": {"ok": False, "error": "boom"}}],
        )

    monkeypatch.setattr(guide_runner, "run_session", fake_session)
    sess = SimpleNamespace(parked=False, parked_questions=[], state=state)
    ctx = SimpleNamespace(check_cancel=None)
    out = asyncio.run(
        guide_runner._run_mission(
            "annotate", "", ctx=ctx, state=state, state_dir=Path("/tmp/x"),
            mk_session=lambda m: sess,
        )
    )
    assert out is None  # mission loop continues to emit
    assert "annotate" in state.missions_failed
    assert calls["n"] == 2  # retry still bounded


def test_required_mission_failure_still_fails(monkeypatch) -> None:
    """Non-optional missions keep the hard-fail contract."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide import guide_runner
    from cadx.guide.state import GuideState

    state = GuideState(output_dir="/tmp/x")

    async def fake_session(*a, **kw):
        return SimpleNamespace(end_reason="error_streak", tool_log=[])

    monkeypatch.setattr(guide_runner, "run_session", fake_session)
    sess = SimpleNamespace(parked=False, parked_questions=[], state=state)
    ctx = SimpleNamespace(check_cancel=None)
    out = asyncio.run(
        guide_runner._run_mission(
            "steps", "", ctx=ctx, state=state, state_dir=Path("/tmp/x"),
            mk_session=lambda m: sess,
        )
    )
    assert out is not None and out.status == "failed"
    assert "steps" in state.missions_failed


def test_llm_outage_is_clean_mission_failure_not_crash(monkeypatch) -> None:
    """HTTP 502 from the LLM backend must not traceback-crash run_guide —
    it becomes a mission failure with the outage named in the error."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide import guide_runner
    from cadx.guide.state import GuideState
    from cadx.llm.client import LLMError

    state = GuideState(output_dir="/tmp/x")

    async def fake_session(*a, **kw):
        raise LLMError("HTTP 502")

    monkeypatch.setattr(guide_runner, "run_session", fake_session)
    sess = SimpleNamespace(parked=False, parked_questions=[], state=state)
    ctx = SimpleNamespace(check_cancel=None)
    out = asyncio.run(
        guide_runner._run_mission(
            "steps", "", ctx=ctx, state=state, state_dir=Path("/tmp/x"),
            mk_session=lambda m: sess,
        )
    )
    assert out is not None and out.status == "failed"
    assert "llm_error: HTTP 502" in out.error
    assert "steps" in state.missions_failed


# --------------------------------------------------------------------- #
# physical port-model regressions (adapter-in-nut / buried pipe bug)
# --------------------------------------------------------------------- #


def _phys_part(pid: str, ports: list[PortInfo], bbox=((-20, -20, -20), (20, 20, 20))) -> GuidePart:
    return GuidePart(
        id=pid, name=pid, ports=ports,
        bbox_min=list(bbox[0]), bbox_max=list(bbox[1]),
    )


def _sock(name: str, *, axis=(0, 0, -1), pos=(0, 0, 0), lip=11.0, inner=9.9,
          seat=18.0, outer=12.8) -> PortInfo:
    return PortInfo(
        name=name, kind="cylinder", pos=list(pos), axis=list(axis),
        radius=lip, inner_radius=inner, outer_radius=outer,
        depth=seat, gender="female",
    )


def _plug(name: str, *, axis=(0, 0, 1), pos=(0, 0, 0), outer=11.0) -> PortInfo:
    return PortInfo(
        name=name, kind="cylinder", pos=list(pos), axis=list(axis),
        radius=outer, inner_radius=0.0, outer_radius=outer,
        depth=0.0, gender="male",
    )


def test_snug_socket_beats_loose_bore() -> None:
    """A plug mates the TIGHTEST legal receiver — the union nut's wide
    clear bore (inner 27) used to win over the actual seat (inner ~21),
    sinking the adapter into the nut cavity."""
    plug = _phys_part("adapter", [_plug("end", outer=15.5)])
    host = _phys_part(
        "casing",
        [
            _sock("nut", inner=27.0, seat=15.0, outer=0.0, lip=30.0),
            _sock("stub_socket", inner=21.4, seat=15.0, outer=0.0, lip=22.0),
        ],
    )
    cands = solver.find_matching_ports(plug, host, include_mismatch=True)
    assert cands and cands[0]["fixed_port"] == "stub_socket"


def test_exposed_mouth_beats_recessed_shoulder() -> None:
    """A shoulder inside a through-bore (same radius as the mouth) must
    lose to the exposed opening — otherwise the pipe tunnels through
    the fitting's own bore into the casing wall."""
    pipe = _phys_part("pipe", [_plug("end", outer=11.0)])
    adapter = _phys_part(
        "adapter",
        [
            _sock("mouth", pos=(0, 0, 20), inner=9.9),
            _sock("shoulder", pos=(0, 0, 5), inner=9.9),  # 15mm recessed
        ],
        bbox=((-20, -20, -20), (20, 20, 20)),
    )
    cands = solver.find_matching_ports(pipe, adapter, include_mismatch=True)
    assert cands and cands[0]["fixed_port"] == "mouth"


def test_engage_is_receiver_side_not_max() -> None:
    """Loose collar mate (nut bore 27 ≫ plug 15.5) stays at the mouth —
    the old max(both-depths) rule sank the plug 20mm into the cavity."""
    plug = _phys_part("adapter", [_plug("end", outer=15.5)])
    nut = _phys_part("casing", [_sock("nut", inner=27.0, seat=15.0, lip=30.0)])
    cands = solver.find_matching_ports(plug, nut, include_mismatch=True)
    assert cands and cands[0]["engage_mm"] <= 7.0
    # snug fit (inner 11 ≥ plug 11, margin ~0) travels to the seat
    snug = _phys_part("elbow", [_sock("muffe", inner=11.0, seat=24.0)])
    cands = solver.find_matching_ports(plug, snug, include_mismatch=True)
    assert cands and cands[0]["engage_mm"] > 15.0


def test_joint_audit_part_alias_is_not_a_foreign_host(tmp_path) -> None:
    """Instances keyed by BOTH instance_id and part_id counted themselves
    as foreign hosts → every part flagged 100% buried. Partner exclusion:
    a plug inside its own mate is legitimate; only a non-jointed host
    counts as burial."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import joint_geometry_audit
    from cadx.guide.state import AssemblyInstance, AssemblyState

    state = GuideState(output_dir=str(tmp_path))
    host = _phys_part("host", [_sock("m", inner=11.0)], bbox=((-50, -50, 0), (50, 50, 100)))
    pipe = _phys_part("pipe", [_plug("e", outer=11.0)], bbox=((-5, -5, 0), (5, 5, 70)))
    # shell encloses ONLY the pipe's region — the host stays outside it
    shell = _phys_part("shell", [], bbox=((190, -10, -5), (210, 10, 75)))
    state.parts = [host, pipe, shell]
    tp = np.eye(4)
    tp[0, 3] = 200.0  # pipe placed inside the shell, away from the host
    asm = AssemblyState()
    asm.instances = [
        AssemblyInstance(
            part_id="host", instance_id="host#1", transform=np.eye(4).tolist()
        ),
        AssemblyInstance(
            part_id="pipe", instance_id="pipe#1", transform=tp.tolist(),
        ),
        AssemblyInstance(
            part_id="shell", instance_id="shell#1", transform=np.eye(4).tolist()
        ),
    ]
    asm.joints = [
        {
            "step": 1, "moving_instance": "pipe#1", "fixed_instance": "host#1",
            "moving_port": "e", "fixed_port": "m",
        }
    ]
    state.assembly = asm
    s = SimpleNamespace(state=state)
    audit = joint_geometry_audit(s)
    buried_ids = {b["instance"] for b in audit["buried_instances"]}
    # the non-jointed shell fully contains the pipe → buried
    assert "pipe#1" in buried_ids
    # hosts never flag themselves via part-keyed alias duplicates —
    # the containment check only counts OTHER instances' AABBs
    assert "host#1" not in buried_ids and "shell#1" not in buried_ids


def test_joint_audit_socket_depth_is_engagement_allowance(tmp_path) -> None:
    """A plug 8mm from flush in a 24mm socket is still engaged (union
    nuts never bottom out); the same 8mm on a face joint is detached."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import joint_geometry_audit
    from cadx.guide.state import AssemblyInstance, AssemblyState

    def _state(gap: float, sock_seat: float) -> GuideState:
        state = GuideState(output_dir=str(tmp_path))
        fixed = _phys_part(
            "elbow",
            [_sock("m", pos=(0, 0, 50), axis=(0, 0, -1), seat=sock_seat)],
            bbox=((-20, -20, 0), (20, 20, 60)),
        )
        moving = _phys_part(
            "pipe", [_plug("e", axis=(0, 0, 1), outer=11.0)],
            bbox=((-5, -5, -60), (5, 5, 0)),
        )
        state.parts = [fixed, moving]
        # the engaged mate point sits engage_mm INSIDE the bore (behind
        # the mouth); 'gap' = how far the plug tip is short of it
        engage = solver._engage_mm(moving.ports[0], fixed.ports[0])
        tm = np.eye(4)
        tm[2, 3] = 50.0 + engage - gap
        asm = AssemblyState()
        asm.instances = [
            AssemblyInstance(part_id="elbow", instance_id="elbow#1", transform=np.eye(4).tolist()),
            AssemblyInstance(part_id="pipe", instance_id="pipe#1", transform=tm.tolist()),
        ]
        asm.joints = [
            {"step": 1, "moving_instance": "pipe#1", "fixed_instance": "elbow#1",
             "moving_port": "e", "fixed_port": "m"}
        ]
        state.assembly = asm
        return state

    # deep socket: 8mm short of flush = partially engaged → ok
    s = SimpleNamespace(state=_state(gap=8.0, sock_seat=24.0))
    assert not joint_geometry_audit(s)["bad_joints"]
    # face joint (seat≈0): same 8mm = detached
    s = SimpleNamespace(state=_state(gap=8.0, sock_seat=0.5))
    assert joint_geometry_audit(s)["bad_joints"]


def test_resolve_name_context_breaks_part_ties() -> None:
    """Two elbow types share rohrbogen/dn/22 — bare name keeps the
    longest-key convention; the step's '(Muffe auf Muffe)' qualifier
    picks the muffe-muffe variant."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import _resolve_name

    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        _phys_part("spigot_elbow", [], bbox=((0, 0, 0), (1, 1, 1))),
        _phys_part("muffe_elbow", [], bbox=((0, 0, 0), (1, 1, 1))),
    ]
    state.parts[0].name = "Rohrbogen DN 22, Muffe auf Spitzende"
    state.parts[1].name = "Rohrbogen DN 22, Muffe auf Muffe"
    s = SimpleNamespace(state=state, _aliases_store={})
    # bare: longest key wins (deterministic, no context bleed)
    assert _resolve_name(s, "Rohrbogen DN 22, Anschluss 1x außen 1x innen") == "spigot_elbow"
    # context names the muffe-muffe variant
    got = _resolve_name(
        s,
        "Rohrbogen DN 22, Anschluss 2x außen",
        context="Rohrbogen (Muffe auf Muffe) auf das Rohr stecken",
    )
    assert got == "muffe_elbow"


def test_step_requirements_resolve_with_step_context(tmp_path) -> None:
    """_step_requirements counted the bare-name winner while the planner
    resolved with context → 'not required by any step' quota rejection."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import _step_requirements
    from cadx.guide.state import GuideStep, GuideSubStep

    state = GuideState(output_dir=str(tmp_path))
    a = _phys_part("spigot_elbow", [])
    b = _phys_part("muffe_elbow", [])
    a.name = "Rohrbogen DN 22, Muffe auf Spitzende"
    b.name = "Rohrbogen DN 22, Muffe auf Muffe"
    state.parts = [a, b]
    state.steps = [
        GuideStep(
            number=11, action_type="assembly",
            text="Rohrbogen (Muffe auf Muffe) auf Rohr 500 und Rohr 70 stecken",
            sub_steps=[
                GuideSubStep(
                    action="attach",
                    part="Rohrbogen DN 22, Anschluss 2x außen",
                    target="Rohr 500", target2="Rohr 70",
                )
            ],
        )
    ]
    s = SimpleNamespace(state=state, _aliases_store={})
    reqs = _step_requirements(s)
    assert reqs[11] == {"muffe_elbow": 1}


def test_review_report_rejects_high_score_on_bad_audit(tmp_path) -> None:
    """Score 9 over a detached joint must fail — the reviewer's eyeball
    verdict cannot overrule measured geometry."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide.review_tools import write_review_report
    from cadx.guide.state import AssemblyInstance, AssemblyState

    state = GuideState(output_dir=str(tmp_path))
    fixed = _phys_part(
        "elbow", [_sock("m", pos=(0, 0, 50), axis=(0, 0, -1), seat=2.0)],
        bbox=((-20, -20, 0), (20, 20, 60)),
    )
    moving = _phys_part(
        "pipe", [_plug("e", axis=(0, 0, 1), outer=11.0)],
        bbox=((-5, -5, -60), (5, 5, 0)),
    )
    state.parts = [fixed, moving]
    tm = np.eye(4)
    tm[2, 3] = 42.0  # 8mm gap on a face joint → detached
    state.assembly = AssemblyState(
        instances=[
            AssemblyInstance(part_id="elbow", instance_id="elbow#1", transform=np.eye(4).tolist()),
            AssemblyInstance(part_id="pipe", instance_id="pipe#1", transform=tm.tolist()),
        ],
        joints=[
            {"step": 1, "moving_instance": "pipe#1", "fixed_instance": "elbow#1",
             "moving_port": "e", "fixed_port": "m"}
        ],
    )
    s = SimpleNamespace(state=state)
    r = asyncio.run(write_review_report(s, {"score": 9.0, "findings": []}))
    assert not r["ok"], "score ≥8 must be rejected while the audit reports findings"
    # an honest low score goes through
    r = asyncio.run(write_review_report(s, {"score": 5.0, "findings": []}))
    assert r["ok"] and r.get("audit_ok") is False
