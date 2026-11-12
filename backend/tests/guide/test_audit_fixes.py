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
    from cadx.guide import tools as gtools

    # simulates a fresh interpreter where no *_tools module was imported
    gtools.DONE_GUARDS.clear()
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
