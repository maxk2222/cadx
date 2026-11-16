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
    verdict, notes = _review_verdict(state, tmp_path, score_target=8.0, max_iter=3)
    assert verdict == "repair" and notes
    assert "review" not in state.missions_done, "review must re-run for a fresh verdict"
    assert "assembly" not in state.missions_done
    assert "annotate" not in state.missions_done
    assert "doc" in state.missions_done  # untouched


def test_review_verdict_cap_is_fail_not_ship(tmp_path: Path) -> None:
    """Iteration cap with open blockers must FAIL the run — the old
    `iter >= max_iter → ship` shipped score 4.0 + blockers as succeeded."""
    state = GuideState(output_dir=str(tmp_path))
    state.missions_done = ["doc", "steps", "parts", "assembly", "review", "annotate"]
    state.review.iteration = 3
    state.review.score = 4.0
    state.review.findings = [
        ReviewFinding(severity="blocker", mission="parts", text="no host in kit")
    ]
    verdict, _ = _review_verdict(state, tmp_path, score_target=8.0, max_iter=3)
    assert verdict == "fail"
    # clean verdict still ships
    state.review.score = 9.0
    state.review.findings = []
    verdict, _ = _review_verdict(state, tmp_path, score_target=8.0, max_iter=3)
    assert verdict == "ship"


def test_review_report_consistency_gates(tmp_path: Path) -> None:
    """A sub-target score without blockers (or ≥8 with them, or without
    visual evidence) is rejected — score 6.0 + only warnings used to
    ship instantly via (no blockers, iter>0)."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide.review_tools import write_review_report

    state = GuideState(output_dir=str(tmp_path))
    s = SimpleNamespace(state=state)
    vis = "pipe: expected vertical below casing — observed vertical — match. " * 2
    res = asyncio.run(
        write_review_report(s, {"score": 6.0, "findings": [], "visual_comparison": vis})
    )
    assert not res["ok"] and "blocker" in res["error"]
    res = asyncio.run(
        write_review_report(
            s,
            {
                "score": 9.0,
                "findings": [{"severity": "blocker", "mission": "assembly", "text": "x"}],
                "visual_comparison": vis,
            },
        )
    )
    assert not res["ok"] and "contradicts" in res["error"]
    res = asyncio.run(
        write_review_report(s, {"score": 9.0, "findings": [], "visual_comparison": ""})
    )
    assert not res["ok"] and "visual_comparison" in res["error"]


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
        "doc_tools",
        "step_tools",
        "part_tools",
        "assembly_tools",
        "review_tools",
        "annotate_tools",
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
    # tip lands on the mate: tail = port world pos (10,0,5) + axis*standoff;
    # moving part has no bbox -> standoff = max(30, 15) = 30mm -> z = 35mm ->
    # unity (x,z,y)*0.001 -> (0.010, 0.035, 0.0)
    assert arrow["origin"] == pytest.approx([0.010, 0.035, 0.0], abs=1e-4)
    # direction = negated fixed-port outward axis -> arrow points AT the port
    assert arrow["direction"] == pytest.approx([-0.0, -1.0, -0.0], abs=1e-4)


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
            "annotate",
            "",
            ctx=ctx,
            state=state,
            state_dir=Path("/tmp/x"),
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
            "steps",
            "",
            ctx=ctx,
            state=state,
            state_dir=Path("/tmp/x"),
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
            "steps",
            "",
            ctx=ctx,
            state=state,
            state_dir=Path("/tmp/x"),
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
        id=pid,
        name=pid,
        ports=ports,
        bbox_min=list(bbox[0]),
        bbox_max=list(bbox[1]),
    )


def _sock(
    name: str, *, axis=(0, 0, -1), pos=(0, 0, 0), lip=11.0, inner=9.9, seat=18.0, outer=12.8
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


def _plug(name: str, *, axis=(0, 0, 1), pos=(0, 0, 0), outer=11.0) -> PortInfo:
    return PortInfo(
        name=name,
        kind="cylinder",
        pos=list(pos),
        axis=list(axis),
        radius=outer,
        inner_radius=0.0,
        outer_radius=outer,
        depth=0.0,
        gender="male",
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
    slim_plug = _phys_part("adapter", [_plug("end", outer=11.0)])
    snug = _phys_part("elbow", [_sock("muffe", inner=11.0, seat=24.0)])
    cands = solver.find_matching_ports(slim_plug, snug, include_mismatch=True)
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
        AssemblyInstance(part_id="host", instance_id="host#1", transform=np.eye(4).tolist()),
        AssemblyInstance(
            part_id="pipe",
            instance_id="pipe#1",
            transform=tp.tolist(),
        ),
        AssemblyInstance(part_id="shell", instance_id="shell#1", transform=np.eye(4).tolist()),
    ]
    asm.joints = [
        {
            "step": 1,
            "moving_instance": "pipe#1",
            "fixed_instance": "host#1",
            "moving_port": "e",
            "fixed_port": "m",
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
            "pipe",
            [_plug("e", axis=(0, 0, 1), outer=11.0)],
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
            {
                "step": 1,
                "moving_instance": "pipe#1",
                "fixed_instance": "elbow#1",
                "moving_port": "e",
                "fixed_port": "m",
            }
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
            number=11,
            action_type="assembly",
            text="Rohrbogen (Muffe auf Muffe) auf Rohr 500 und Rohr 70 stecken",
            sub_steps=[
                GuideSubStep(
                    action="attach",
                    part="Rohrbogen DN 22, Anschluss 2x außen",
                    target="Rohr 500",
                    target2="Rohr 70",
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
        "elbow",
        [_sock("m", pos=(0, 0, 50), axis=(0, 0, -1), seat=2.0)],
        bbox=((-20, -20, 0), (20, 20, 60)),
    )
    moving = _phys_part(
        "pipe",
        [_plug("e", axis=(0, 0, 1), outer=11.0)],
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
            {
                "step": 1,
                "moving_instance": "pipe#1",
                "fixed_instance": "elbow#1",
                "moving_port": "e",
                "fixed_port": "m",
            }
        ],
    )
    s = SimpleNamespace(state=state)
    vis = "pipe: expected seated on elbow — observed floating 8mm above — deviation"
    r = asyncio.run(
        write_review_report(s, {"score": 9.0, "findings": [], "visual_comparison": vis})
    )
    assert not r["ok"], "score ≥8 must be rejected while the audit reports findings"
    # an honest low score goes through (blocker finding names the fault)
    r = asyncio.run(
        write_review_report(
            s,
            {
                "score": 5.0,
                "findings": [
                    {"severity": "blocker", "mission": "assembly", "text": "detached joint"}
                ],
                "visual_comparison": vis,
            },
        )
    )
    assert r["ok"] and r.get("audit_ok") is False


def _stub(name: str, *, pos=(0, 0, 0), outer: float, nested_in: str | None) -> PortInfo:
    return PortInfo(
        name=name,
        kind="cylinder",
        pos=list(pos),
        axis=[0.0, 0.0, -1.0],
        radius=outer,
        inner_radius=outer,
        outer_radius=outer,
        depth=0.0,
        gender="male",
        nested_in=nested_in,
    )


def test_thread_seat_beats_oversized_socket() -> None:
    """A nut/female mouth whose bore is too small for a boss-nested male
    stub still seats ON the thread site — the stub inside a funnel/boss
    opening is the real interface, not the enclosing socket pocket that
    would swallow the fitting (adapter landing inside the union nut)."""
    nut = _phys_part(
        "adapter", [_sock("mouth", pos=(0, 0, -20), inner=11.0, outer=15.5, seat=0.3, lip=14.75)]
    )
    host = _phys_part(
        "casing",
        [
            _sock("funnel", pos=(0, 0, -20), inner=27.0, seat=15.0, outer=0.0, lip=30.0),
            _stub("stub", pos=(0, 0, -20), outer=21.4, nested_in="funnel"),
        ],
    )
    cands = solver.find_matching_ports(nut, host, include_mismatch=True)
    assert cands and cands[0]["fixed_port"] == "stub"
    assert cands[0]["score"] > 0.4
    # the funnel socket must lose — plugging a nut INTO the nut pocket
    # is physically wrong
    funnel = next(c for c in cands if c["fixed_port"] == "funnel")
    assert funnel["score"] < cands[0]["score"]
    # seat engagement: a few mm tuck at the mouth plane, not a dive
    assert 0.0 < solver._engage_mm(nut.ports[0], host.ports[1]) <= 4.0


def test_no_seat_on_bare_stub() -> None:
    """A bare male stub (not nested in a socket) keeps the hard reject —
    thread-seat is only for boss-nested thread sites."""
    nut = _phys_part(
        "adapter", [_sock("mouth", pos=(0, 0, -20), inner=11.0, outer=15.5, seat=0.3, lip=14.75)]
    )
    host = _phys_part("pump", [_stub("stub", outer=21.4, nested_in=None)])
    cands = solver.find_matching_ports(nut, host, include_mismatch=True)
    assert not cands or cands[0]["score"] <= 0.0


def test_deep_socket_seats_below_thread_mouth() -> None:
    """A deep receiving socket (the muffe end) seats worse than a shallow
    thread mouth — the nut face, not the push-fit bore, is the thread
    interface."""
    mover = _phys_part(
        "adapter",
        [
            _sock("mouth", pos=(0, 0, -20), inner=11.0, outer=15.5, seat=0.3, lip=14.75),
            _sock(
                "muffe", pos=(0, 0, 20), axis=(0, 0, 1), inner=9.9, outer=12.8, seat=500.0, lip=11.0
            ),
        ],
    )
    host = _phys_part(
        "casing",
        [
            _sock("funnel", pos=(0, 0, -20), inner=27.0, seat=15.0),
            _stub("stub", pos=(0, 0, -20), outer=21.4, nested_in="funnel"),
        ],
    )
    cands = solver.find_matching_ports(mover, host, include_mismatch=True)
    assert cands and cands[0]["moving_port"] == "mouth"
    muffe = next(c for c in cands if c["moving_port"] == "muffe" and c["fixed_port"] == "stub")
    assert muffe["score"] < cands[0]["score"]


def test_resolve_name_distinctive_single_token() -> None:
    """A feature phrase sharing ONE token with a part resolves when that
    token names exactly one part — 'linker oberer Anschluss am Gehäuse'
    must reach the casing so the planner anchors it. Generic tokens
    ('rohr') shared by several parts still can't resolve alone."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import _resolve_name

    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        _phys_part("casing", [], bbox=((0, 0, 0), (1, 1, 1))),
        _phys_part("adapter", [], bbox=((0, 0, 0), (1, 1, 1))),
        _phys_part("pipe_l70", [], bbox=((0, 0, 0), (1, 1, 1))),
        _phys_part("pipe_l500", [], bbox=((0, 0, 0), (1, 1, 1))),
    ]
    state.parts[0].name = "Gehäuse / Verkleidung (Casing)"
    state.parts[1].name = "Adapter DN22, Innengewinde/Muffe"
    state.parts[2].name = "Rohr DN22, l = 70 mm"
    state.parts[3].name = "Rohr DN22, l = 500 mm"
    s = SimpleNamespace(state=state, _aliases_store={})
    assert _resolve_name(s, "linker oberer Anschluss am Gehäuse") == "casing"
    assert _resolve_name(s, "rechter oberer Anschluss am Gehäuse") == "casing"
    # 'rohr' lives on two parts — no distinctive token → stays unresolved
    assert _resolve_name(s, "Anschluss am Rohr") is None


def test_resolve_name_context_feature_never_token_scores() -> None:
    """Live bug: 'linkes oberes vormontiertes Anschlussrohr in der
    Verkleidung' names a DEVICE STUB (a declared context feature), but
    the resolver let the context token itself fall back to token scoring
    — 'anschlussrohr'/'oberes' bound it to pipe_l70/pipe_l500, phantom
    premounts spawned and every downstream task cascade-failed.

    Rules now: a part noun next to the context word still binds the kit
    part ('Adapter am Anschlussrohr' → adapter); the pure feature form
    resolves only through an exact feature binding, else demotes to the
    host the stub belongs to. A '(Schritt N)' pin is provenance and wins
    over host demotion."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import _resolve_name
    from cadx.guide.planlint import _resolve_part
    from cadx.guide.state import GuideStep, GuideSubStep

    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        _phys_part("casing", [], bbox=((0, 0, 0), (300, 200, 600))),
        _phys_part("adapter", [], bbox=((0, 0, 0), (1, 1, 1))),
        _phys_part("pipe_l70", [], bbox=((0, 0, 0), (1, 1, 1))),
        _phys_part("pipe_l500", [], bbox=((0, 0, 0), (1, 1, 1))),
    ]
    state.parts[0].name = "Wärmepumpe (Gehäuse)"
    state.parts[1].name = "Adapter DN22, Innengewinde/Muffe"
    state.parts[2].name = "Rohr DN22, l = 70 mm"
    state.parts[3].name = "Rohr DN22, l = 500 mm"
    state.context_names = ["Verkleidung", "Anschlussrohr"]
    state.steps = [
        GuideStep(
            number=2,
            title="Adapter vormontiert",
            text="Adapter vormontiert",
            sub_steps=[
                GuideSubStep(
                    action="attach",
                    part="adapter",
                    part_pid="adapter",
                )
            ],
        )
    ]
    s = SimpleNamespace(state=state, _aliases_store={})
    # the pure feature form: device stub → host, NEVER a kit pipe
    assert (
        _resolve_name(s, "linkes oberes vormontiertes Anschlussrohr in der Verkleidung") == "casing"
    )
    assert (
        _resolve_name(s, "rechtes oberes vormontiertes Anschlussrohr in der Verkleidung")
        == "casing"
    )
    # a part noun next to the context word still binds the kit part
    assert _resolve_name(s, "vormontierter Adapter am Anschlussrohr") == "adapter"
    # an exact feature binding beats host demotion
    s._aliases_store["anschlussrohr"] = "pipe_l70"
    assert _resolve_name(s, "oberes Anschlussrohr") == "pipe_l70"
    del s._aliases_store["anschlussrohr"]
    # step pin is provenance — it resolves what step 2 placed (the
    # adapter), not the context feature it sat on
    assert _resolve_name(s, "oberes Anschlussrohr (Schritt 2)") is None
    assert _resolve_part(s, "oberes Anschlussrohr (Schritt 2)", 4) == "adapter"


def test_resolve_name_feature_of_host_when_context_undeclared() -> None:
    """Live bug (run 4f7e23f8): the parts mission never declared
    'Anschlussrohr' context — it appeared only in step targets, so
    match_required_parts showed nothing missing and the model moved on.
    'linkes oberes Anschlussrohr am Gehäuse' then token-resolved to
    None (the pipe-category head killed the casing candidate),
    instance-ref-linted as a mere warning, and every preassembly task
    cascade-failed (placed=0/12).

    The attachment phrase itself carries the semantics: 'X am Gehäuse'
    names a feature living ON the host. A head naming a unique kit kind
    keeps it; an ambiguous/kindless head demotes to the resolved host —
    identical to what mark_context_part yields, derived when the model
    skipped the declaration."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import _resolve_name

    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        _phys_part("heat_pump_casing", [], bbox=((0, 0, 0), (300, 200, 600))),
        _phys_part("adapter_rn22", [], bbox=((0, 0, 0), (1, 1, 1))),
        _phys_part("pipe_l70", [], bbox=((0, 0, 0), (1, 1, 1))),
        _phys_part("pipe_l500", [], bbox=((0, 0, 0), (1, 1, 1))),
    ]
    state.parts[0].name = "KIARA-Wärmepumpe (Gehäuse)"
    state.parts[1].name = "Adapter DN22, Innengewinde/Muffe"
    state.parts[2].name = "Rohr DN22, l = 70 mm"
    state.parts[3].name = "Rohr DN 22, l = 500 mm"
    s = SimpleNamespace(state=state, _aliases_store={})
    # the observed run's actual targets — ambiguous pipe head → host
    assert (
        _resolve_name(s, "linkes oberes Anschlussrohr am Gehäuse")
        == "heat_pump_casing"
    )
    assert (
        _resolve_name(s, "rechtes oberes Anschlussrohr am Gehäuse")
        == "heat_pump_casing"
    )
    # unique kind head keeps the mountable — host demotion precedent
    assert (
        _resolve_name(
            s, "vormontierter Adapter am linken unteren Anschluss der Wärmepumpe"
        )
        == "adapter_rn22"
    )
    # kindless head ('Seite') demotes to the resolved host
    assert (
        _resolve_name(s, "freie obere Seite des linken Rohrs DN 22, l = 70 mm")
        == "pipe_l70"
    )
    # no locative → no rescue: a bare ambiguous kind still blocks
    assert _resolve_name(s, "Anschlussrohr") is None
    # positional instance refs keep their own path (no host phrase)
    assert _resolve_name(s, "linkes vertikales Rohr") is None


def test_context_name_resolves_to_host_and_refuses_alias() -> None:
    """Live bug (run aee696f6): the model declared 'linkes oberes
    Anschlussrohr in der Verkleidung' a context feature AND aliased it
    onto pipe_l70 — the alias won resolution, the pipes placed in step 3
    became forward-referenced targets of step 1, and lint reopened the
    parts mission until the job died.

    Two guards: a reference that IS a declared context name resolves
    through the context path to the host (no filter may exempt it); and
    alias_part rejects the contradictory declaration outright instead of
    poisoning the alias store."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import _resolve_name
    from cadx.guide.part_tools import alias_part

    state = GuideState(output_dir="/tmp/x")
    state.parts = [
        _phys_part("casing", [], bbox=((0, 0, 0), (300, 200, 600))),
        _phys_part("pipe_l70", [], bbox=((0, 0, 0), (1, 1, 1))),
    ]
    state.parts[0].name = "Wärmepumpe (Gehäuse)"
    state.parts[1].name = "Rohr DN22, l = 70 mm"
    state.context_names = [
        "linkes oberes Anschlussrohr in der Verkleidung",
        "rechtes oberes Anschlussrohr in der Verkleidung",
    ]
    s = SimpleNamespace(state=state, _aliases_store={})
    # exact context-name reference → host demotion, never a kit pipe
    assert (
        _resolve_name(s, "linkes oberes Anschlussrohr in der Verkleidung") == "casing"
    )
    # the contradictory alias is refused, not stored
    r = asyncio.run(
        alias_part(
            s,
            {"part_id": "pipe_l70", "alias": "linkes oberes Anschlussrohr in der Verkleidung"},
        )
    )
    assert r["ok"] is False and "context" in r["error"], r
    assert "linkes oberes anschlussrohr in der verkleidung" not in s._aliases_store


def test_resolve_name_side_position_beats_incidental_token() -> None:
    """Live bug from run 214939: the parts mission named the spigot elbow
    'Muffe auf Spitzende' — the ref 'Rohrbogen am rechten unteren
    Anschluss' shared 'anschluss' with ONLY the 2x-outside name, so token
    overlap bound step 9's lower-right target to the upper bridge elbow
    and cross-wired both long pipes into elbow_2x#1. Position words are
    the real discriminator: the part placed in a 'rechts unten' step is
    the lower-right elbow, regardless of which sibling's name carries
    the incidental token."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import _resolve_name
    from cadx.guide.state import GuideStep, GuideSubStep

    state = GuideState(output_dir="/tmp/x")
    spigot = _phys_part("elbow_socket_spigot", [])
    outside = _phys_part("elbow_2x_outside", [])
    spigot.name = "Rohrbogen DN22, Muffe auf Spitzende"
    outside.name = "Rohrbogen DN22, Anschluss 2x außen"
    state.parts = [spigot, outside]
    state.steps = [
        GuideStep(
            number=5,
            title="Rohrbogen links unten",
            text="Rohrbogen links unten",
            sub_steps=[
                GuideSubStep(
                    action="attach",
                    part="spigot",
                    part_pid="elbow_socket_spigot",
                )
            ],
        ),
        GuideStep(
            number=6,
            title="Rohrbogen rechts unten",
            text="Rohrbogen rechts unten",
            sub_steps=[
                GuideSubStep(
                    action="attach",
                    part="spigot",
                    part_pid="elbow_socket_spigot",
                )
            ],
        ),
        GuideStep(
            number=8,
            title="Rohrbogen links",
            text="Bogen auf linkes Rohr",
            sub_steps=[
                GuideSubStep(
                    action="attach",
                    part="2x",
                    part_pid="elbow_2x_outside",
                )
            ],
        ),
        GuideStep(
            number=10,
            title="Rohrbogen rechts",
            text="Bogen auf rechtes Rohr",
            sub_steps=[
                GuideSubStep(
                    action="attach",
                    part="2x",
                    part_pid="elbow_2x_outside",
                )
            ],
        ),
    ]
    s = SimpleNamespace(state=state, _aliases_store={})
    # 'unten' pins the lower elbow family — the incidental 'anschluss'
    # token must not pull the upper bridge elbow
    assert (
        _resolve_name(s, "Rohrbogen am rechten unteren Anschluss der Wärmepumpe")
        == "elbow_socket_spigot"
    )
    assert (
        _resolve_name(s, "Rohrbogen am linken unteren Anschluss der Wärmepumpe")
        == "elbow_socket_spigot"
    )
    # explicit registered name still wins outright
    assert _resolve_name(s, "Rohrbogen DN22, Anschluss 2x außen") == "elbow_2x_outside"
    # bare noun stays honestly ambiguous
    assert _resolve_name(s, "Rohrbogen") is None


def test_side_words_canonical_inflections() -> None:
    """'(links oben)' and 'linken oberen' are the same side signature —
    mixed inflection vocab produced disjoint sets and phantom premounts."""
    from cadx.guide.preassembly import _side_words

    assert _side_words("(links oben)") == _side_words("linken oberen")
    assert _side_words("rechts unten") == _side_words("rechter unterer")
    assert _side_words("obere Öffnung des linken unteren Rohrbogens") == {"u", "l", "d"}


def test_assembly_snapshot_survives_preassembly(monkeypatch, tmp_path) -> None:
    """Snapshot ordering: taken BEFORE run_preassembly, a failed LLM refine
    restored the EMPTY pre-preassembly state and destroyed 15 deterministic
    placements (observed in the 19:19 eval run). The snapshot must be taken
    after preassembly so the deterministic floor is what gets restored."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide import guide_runner
    from cadx.guide.state import AssemblyInstance, GuideState

    state = GuideState(output_dir=str(tmp_path))
    state.missions_done = ["doc", "steps", "parts"]

    async def fake_preassembly(sess) -> None:
        state.assembly.instances.append(AssemblyInstance(instance_id="pipe#1", part_id="pipe"))
        state.assembly.planner = {"ran": True}

    async def fake_session(*a, **kw):
        # the LLM refine strips the instance, then dies — restore must
        # bring back the preassembly floor, not the empty pre-plan state
        state.assembly.instances.clear()
        return SimpleNamespace(end_reason="repeat_loop", tool_log=[])

    monkeypatch.setattr("cadx.guide.preassembly.run_preassembly", fake_preassembly)
    monkeypatch.setattr(guide_runner, "run_session", fake_session)
    sess = SimpleNamespace(parked=False, parked_questions=[], state=state)
    ctx = SimpleNamespace(check_cancel=None)
    out = asyncio.run(
        guide_runner._run_mission(
            "assembly",
            "",
            ctx=ctx,
            state=state,
            state_dir=tmp_path,
            mk_session=lambda m: sess,
        )
    )
    assert out is not None and out.status == "failed"
    assert "assembly" in state.missions_failed
    assert [i.instance_id for i in state.assembly.instances] == ["pipe#1"]


def _mate_state(tmp_path, *, sub_action="place") -> object:
    """Two parts: moving declared a mate onto fixed but jointed elsewhere."""
    from types import SimpleNamespace

    from cadx.guide.state import (
        AssemblyInstance,
        GuideState,
        GuideStep,
        GuideSubStep,
    )

    state = GuideState(output_dir=str(tmp_path))
    state.parts = [
        _phys_part("host", []),
        _phys_part("pipe_a", []),
        _phys_part("pipe_b", []),
    ]
    state.steps = [
        GuideStep(
            number=1,
            action_type="assembly",
            text="pipe_a auf pipe_b stecken",
            sub_steps=[GuideSubStep(action=sub_action, part="pipe_a", target="pipe_b")],
        )
    ]
    state.assembly.instances = [
        AssemblyInstance(part_id="host", instance_id="host#1", placed_in_step=1, substep_idx=-1),
        AssemblyInstance(part_id="pipe_a", instance_id="pipe_a#1", placed_in_step=1, substep_idx=0),
        AssemblyInstance(
            part_id="pipe_b", instance_id="pipe_b#1", placed_in_step=1, substep_idx=-1
        ),
    ]
    state.assembly.joints = [{"moving_instance": "pipe_a#1", "fixed_instance": "host#1"}]
    return SimpleNamespace(state=state, _aliases_store={})


def test_declared_mates_flags_wrong_and_missing_mates(tmp_path) -> None:
    """A clean mate on the WRONG part passes gap/alignment but violates
    the plan contract — both directions must surface."""
    from cadx.guide.assembly_tools import _declared_mates

    s = _mate_state(tmp_path)
    out = _declared_mates(s)
    assert any("pipe_b" in m["missing"] for m in out["unfulfilled_mates"])
    assert any("host" in m["jointed"] for m in out["unexpected_mates"])


def test_declared_mates_flags_unplaced_substep(tmp_path) -> None:
    """A place sub-step whose instance never materialized is silent plan
    loss (dropped task, quota block, prep-mislabeled step)."""
    from cadx.guide.assembly_tools import _declared_mates

    s = _mate_state(tmp_path)
    s.state.assembly.instances = [i for i in s.state.assembly.instances if i.part_id != "pipe_a"]
    s.state.assembly.joints = []
    out = _declared_mates(s)
    assert any("no instance materialized" in m["issue"] for m in out["unfulfilled_mates"])


def test_declared_mates_clean_when_plan_matches(tmp_path) -> None:
    from cadx.guide.assembly_tools import _declared_mates

    s = _mate_state(tmp_path)
    s.state.assembly.joints = [{"moving_instance": "pipe_a#1", "fixed_instance": "pipe_b#1"}]
    out = _declared_mates(s)
    assert not out["unfulfilled_mates"]
    assert not out["unexpected_mates"]


def test_unfulfilled_mate_reports_port_gap(tmp_path) -> None:
    """The issue must carry the measured port distance — '392mm' reads as
    a broken plan, '2mm' as a retryable miss; without it the mission
    cannot tell them apart (baseline r1 churned 94 calls on a 390mm gap)."""
    from cadx.guide.assembly_tools import _declared_mates

    s = _mate_state(tmp_path)
    s.state.parts[1].ports = [PortInfo(name="p", pos=[0, 0, 0])]
    s.state.parts[2].ports = [PortInfo(name="q", pos=[0, 0, 0])]
    t_a = np.eye(4).tolist()
    t_b = np.eye(4)
    t_b[0, 3] = 390.0
    s.state.assembly.instances[1].transform = t_a
    s.state.assembly.instances[2].transform = t_b.tolist()
    s.state.assembly.joints = [{"moving_instance": "pipe_a#1", "fixed_instance": "host#1"}]
    out = _declared_mates(s)
    issue = next(m["issue"] for m in out["unfulfilled_mates"] if "pipe_b" in m["missing"])
    assert "pipe_b~390mm" in issue


def test_mark_context_part_rejects_kit_name_collision(tmp_path) -> None:
    """'Rohrbogen DN 22' as context while both elbows are kit parts
    poisoned the resolver in baseline r1 (targets → host → 390mm
    bridges) — the contradictory declaration must fail honestly."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide.part_tools import mark_context_part

    st = GuideState(output_dir=str(tmp_path))
    st.parts = [
        GuidePart(id="elbow_socket_socket", name="Rohrbogen DN22, Anschluss 2x außen"),
        GuidePart(id="elbow_socket_spigot", name="Rohrbogen DN 22, 1x außen 1x innen"),
        GuidePart(id="pipe_l500", name="Pipe l500"),
    ]
    sess = SimpleNamespace(state=st, _aliases_store={})
    res = asyncio.run(mark_context_part(sess, {"name": "Rohrbogen DN 22"}))
    assert res["ok"] is False and "kit part" in res["error"]
    res = asyncio.run(mark_context_part(sess, {"name": "Anschlussrohr"}))
    assert res["ok"] is True and "Anschlussrohr" in st.context_names
    # qualified premount name does not collide — device-side copies of a
    # kit kind stay declarable
    res = asyncio.run(mark_context_part(sess, {"name": "vormontierte Bögen am Gerät"}))
    assert res["ok"] is True


def test_context_tokens_ignore_kit_collisions(tmp_path) -> None:
    """A stale colliding declaration in a saved state must not demote real
    part references to the host — the token is dropped, not trusted."""
    from types import SimpleNamespace

    from cadx.guide.part_tools import _context_tokens

    st = GuideState(output_dir=str(tmp_path))
    st.parts = [GuidePart(id="elbow_a", name="Rohrbogen DN22, Anschluss 2x außen")]
    st.context_names = ["Rohrbogen DN 22", "Anschlussrohr"]
    sess = SimpleNamespace(state=st, _aliases_store={})
    assert _context_tokens(sess, "oberes Ende des Rohrbogens DN 22") == []
    assert _context_tokens(sess, "oberes Anschlussrohr") == ["Anschlussrohr"]


def test_lint_dropped_placements_blocks_prep_step(tmp_path) -> None:
    """A place sub-step under a non-assembly step is silently dropped by
    the planner — plan_lint must block it before preassembly."""
    from types import SimpleNamespace

    from cadx.guide.planlint import _lint_dropped_placements
    from cadx.guide.state import GuideState, GuideStep, GuideSubStep

    state = GuideState(output_dir=str(tmp_path))
    state.parts = [_phys_part("host", []), _phys_part("elbow", [])]
    state.steps = [
        GuideStep(
            number=1,
            action_type="prep",
            text="vorbereitung",
            sub_steps=[GuideSubStep(action="place", part="elbow", target="host", target2="pipe")],
        )
    ]
    s = SimpleNamespace(state=state, _aliases_store={})
    findings = _lint_dropped_placements(s, state.steps)
    assert findings and findings[0]["severity"] == "blocker"
    # assembly steps are untouched
    state.steps[0].action_type = "assembly"
    assert not _lint_dropped_placements(s, state.steps)


def _mate_geom_state(tmp_path, mv_z: float = 0.0):
    """Declared pipe_a→pipe_b mate with real ports; mv_z positions the
    moving plug. Returns (session-like, engage_mm, move_transform)."""
    from types import SimpleNamespace

    from cadx.guide.state import AssemblyInstance, GuideStep, GuideSubStep

    pa = _plug("pa", axis=(0, 0, 1), pos=(0, 0, 0))
    sb = _sock("sb", axis=(0, 0, -1), pos=(0, 0, 10))
    state = GuideState(output_dir=str(tmp_path))
    state.parts = [_phys_part("pipe_a", [pa]), _phys_part("pipe_b", [sb])]
    state.steps = [
        GuideStep(
            number=1,
            action_type="assembly",
            text="pipe_a auf pipe_b",
            sub_steps=[GuideSubStep(action="place", part="pipe_a", target="pipe_b")],
        )
    ]
    eng = solver._engage_mm(pa, sb, moving_part=state.parts[0], fixed_part=state.parts[1])
    ident = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    moved = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 10 + eng + mv_z],
        [0, 0, 0, 1],
    ]
    state.assembly.instances = [
        AssemblyInstance(
            part_id="pipe_a",
            instance_id="pipe_a#1",
            placed_in_step=1,
            substep_idx=0,
            transform=moved,
        ),
        AssemblyInstance(
            part_id="pipe_b",
            instance_id="pipe_b#1",
            placed_in_step=1,
            substep_idx=-1,
            transform=ident,
        ),
    ]
    return SimpleNamespace(state=state, _aliases_store={})


def test_materialize_geometrically_satisfied_mate(tmp_path) -> None:
    """A declared mate whose persisted transforms already seat the port
    pair is a physical fact — the audit records the joint instead of
    flagging a correct placement for an LLM repair loop."""
    from cadx.guide.assembly_tools import _declared_mates

    s = _mate_geom_state(tmp_path)
    out = _declared_mates(s)
    assert any(m["part"] == "pipe_b" for m in out["materialized_mates"]), out
    assert not out["unfulfilled_mates"]
    j = s.state.assembly.joints[0]
    assert j["materialized"] and j["moving_instance"] == "pipe_a#1"


def test_materialize_rejects_unsatisfied_mate(tmp_path) -> None:
    """A declared mate NOT geometrically seated stays an honest
    unfulfilled finding — materialization never fakes the joint."""
    from cadx.guide.assembly_tools import _declared_mates

    s = _mate_geom_state(tmp_path, mv_z=50.0)
    out = _declared_mates(s)
    assert not out["materialized_mates"]
    assert any("pipe_b" in m["missing"] for m in out["unfulfilled_mates"])
    assert not s.state.assembly.joints


def test_unplace_demolish_gate_and_request_repair(tmp_path, monkeypatch) -> None:
    """unplace_instance refuses a healthy instance while the audit flags
    others; request_repair removes ONLY the flagged ones."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide import assembly_tools as at
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import AssemblyInstance, GuideStep

    state = GuideState(output_dir=str(tmp_path))
    state.parts = [_phys_part("elbow", []), _phys_part("pipe", [])]
    state.steps = [GuideStep(number=8, action_type="assembly", text="x")]
    eye = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    insts = [
        AssemblyInstance(
            part_id="elbow",
            instance_id="elbow#1",
            placed_in_step=8,
            substep_idx=0,
            transform=eye,
        ),
        AssemblyInstance(
            part_id="pipe",
            instance_id="pipe#1",
            placed_in_step=8,
            substep_idx=-1,
            transform=eye,
        ),
    ]
    state.assembly.instances = insts
    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir=str(tmp_path),
        project_id=None,
        guide_assets=[],
        executor=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(at, "_flagged_instances", lambda s: {"pipe#1"})
    r = asyncio.run(at.unplace_instance(sess, {"part_id": "elbow#1"}))
    assert r["ok"] is False and "pipe#1" in str(r.get("flagged")), r
    r = asyncio.run(at.unplace_instance(sess, {"part_id": "elbow#1", "confirm_demolish": True}))
    assert r["ok"] is True
    state.assembly.instances.append(insts[0])
    r = asyncio.run(at.request_repair(sess, {"step": 8}))
    assert r["ok"] and r["unplaced"] == ["pipe#1"], r
    assert {i.instance_id for i in state.assembly.instances} == {"elbow#1"}
    # reset_step obeys the same gate — a step holding unflagged instances
    # refuses full demolition without the explicit opt-out
    state.assembly.instances.append(insts[1])
    state.assembly.step_instances["8"] = ["elbow#1", "pipe#1"]
    r = asyncio.run(at.reset_step(sess, {"step": 8}))
    assert r["ok"] is False and "unflagged" in r, r
    r = asyncio.run(at.reset_step(sess, {"step": 8, "confirm_demolish": True}))
    assert r["ok"] is True and not state.assembly.instances


def test_dedup_tasks_drops_exact_duplicates(tmp_path) -> None:
    """Identical place tasks (recreated sub-steps) collapse; tasks that
    differ in mate or qualifier text are genuine operations and stay."""
    from types import SimpleNamespace

    from cadx.guide.preassembly import _dedup_tasks

    s = SimpleNamespace(state=GuideState(output_dir=str(tmp_path)), _aliases_store={})
    base = {
        "step": 3,
        "substep_idx": 0,
        "part": "adapter",
        "part_pid": "adapter",
        "target": "rohr",
        "target_pid": "rohr",
        "target2": "",
        "text": "x",
    }
    tasks = [
        dict(base),
        dict(base, substep_idx=1),  # exact duplicate → drop
        dict(base, substep_idx=2, text="y"),  # different qualifier → keep
        dict(base, substep_idx=3, target_pid="rohr2"),  # other mate → keep
    ]
    out = _dedup_tasks(s, tasks)
    assert len(out) == 3
    assert s._plan_dedup == [{"step": 3, "part": "adapter"}]


def test_reorder_reports_missing_and_duplicated(tmp_path) -> None:
    """reorder_steps must name WHAT is wrong — 'current' alone left the
    model guessing which numbers to fix."""
    import asyncio
    from types import SimpleNamespace

    from cadx.guide.state import GuideStep
    from cadx.guide.step_tools import reorder_steps

    state = GuideState(output_dir=str(tmp_path))
    state.steps = [GuideStep(number=i, text=f"s{i}", display_text=f"d{i}") for i in (1, 2, 3)]
    s = SimpleNamespace(state=state)
    r = asyncio.run(reorder_steps(s, {"order": [1, 1, 2]}))
    assert r["ok"] is False
    assert r["missing"] == [3] and r["duplicated"] == [1]
    assert r["current"] == [1, 2, 3]
