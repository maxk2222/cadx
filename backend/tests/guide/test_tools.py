"""Tool registry + session dispatch tests — allowlists and schemas."""

from unittest.mock import MagicMock

import pytest

from cadx.guide import tools as gtools
from cadx.guide.session import SESSION_SCHEMAS, GuideToolSession
from cadx.guide.state import GuideState


def test_every_mission_tool_has_schema_and_dispatch() -> None:
    schemas = gtools.merged_schemas()
    handlers = gtools.merged_handlers()
    for mission, names in gtools.MISSION_TOOLS.items():
        for name in names:
            in_session = name in SESSION_SCHEMAS
            in_module = name in schemas
            assert in_session or in_module, f"{mission}: '{name}' has no schema"
            if in_module:
                assert name in handlers, f"{mission}: '{name}' has no handler"


def test_mission_allowlists_filter_tools() -> None:
    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    sess = GuideToolSession(
        ctx,
        mission="doc",
        state=GuideState(),
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    doc_tools = {t["function"]["name"] for t in sess.tools()}
    assert "prepare_pdf" in doc_tools
    assert "create_step" not in doc_tools
    sess.mission = "assembly"
    asm_tools = {t["function"]["name"] for t in sess.tools()}
    assert "place_instance" in asm_tools and "solve_attachment_preview" in asm_tools
    assert "prepare_pdf" not in asm_tools


def test_place_instance_rejects_unknown_candidate() -> None:
    """The LLM cannot inject transforms — only server-issued candidate_ids."""
    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    from cadx.guide.state import GuidePart, GuideStep

    state.parts.append(GuidePart(id="a", name="A", mesh_path="/tmp/a.glb"))
    state.steps.append(GuideStep(number=1, text="t", display_text="d"))
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    from cadx.guide.assembly_tools import place_instance

    res = pytest.importorskip("asyncio").run(
        place_instance(sess, {"part_id": "a", "candidate_id": "invented", "step": 1})
    )
    assert res["ok"] is False
    assert "candidate" in res["error"]


def test_done_guard_blocks_incomplete_steps() -> None:
    from cadx.guide.step_tools import HANDLERS  # noqa: F401 — registers guard

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    sess = GuideToolSession(
        ctx,
        mission="steps",
        state=GuideState(),
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    res = sess._tool_done_mission({"reasoning": "x"})
    assert res["ok"] is False


def test_match_required_suggests_renamed_fittings() -> None:
    """Manuals rename the same fitting between parts list and step text —
    match_required_parts must suggest registered candidates (category+size),
    never silently mark them missing."""
    from cadx.guide import part_tools
    from cadx.guide.state import DocumentIndex, GuidePart

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.document = DocumentIndex(
        detected_part_names=[
            "Elbow DN20, end 2x external",
            "Elbow DN20, end 1x external 1x internal",
            "Pipe DN20, l = 300 mm",
        ]
    )
    state.parts.extend(
        [
            GuidePart(
                id="elbow-ss",
                name="Elbow DN20 socket to spigot",
                mesh_path="/x/elbow_rn20_socket_to_spigot.glb",
            ),
            GuidePart(
                id="elbow-ff",
                name="Elbow DN20 socket to socket",
                mesh_path="/x/elbow_rn20_soket_to_socket.glb",
            ),
            GuidePart(id="pipe-300", name="Pipe DN20", mesh_path="/x/pipe_l300.glb"),
        ]
    )
    sess = GuideToolSession(
        ctx,
        mission="parts",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    import asyncio

    res = asyncio.run(part_tools.match_required_parts(sess, {}))
    assert res["required_count"] == 3
    elbow_suggs = {
        r["name"]: set(r.get("suggested_part_ids", []))
        for r in res["matches"]
        if "Elbow" in r["name"]
    }
    # both elbow naming variants suggest both registered elbows, never the pipe
    for s in elbow_suggs.values():
        assert s == {"elbow-ss", "elbow-ff"}
    pipe_row = next(r for r in res["matches"] if "Pipe" in r["name"])
    assert pipe_row["matched_part_id"] == "pipe-300"  # substring match, no suggestion needed

    # generation refused -> refusal also carries the same candidates
    state.allow_part_generation = False
    res2 = asyncio.run(
        part_tools.generate_missing_part(
            sess, {"part_id": "x", "name": "Elbow DN20, end 2x external", "description": "y" * 30}
        )
    )
    assert res2["ok"] is False
    assert set(res2["suggested_part_ids"]) == {"elbow-ss", "elbow-ff"}


def test_same_part_placed_twice_creates_instances() -> None:
    """Left/right sides and repeated fittings: one registered part must be
    placeable multiple times — each placement is its own instance."""
    from cadx.guide.assembly_tools import _completeness_issues, place_instance
    from cadx.guide.state import DocumentIndex, GuidePart, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.document = DocumentIndex(detected_part_names=["Adapter"])
    state.parts.append(GuidePart(id="adapter", name="Adapter DN20"))
    # two assembly steps each needing one adapter instance
    for n in (1, 2):
        state.steps.append(
            GuideStep(
                number=n,
                text="t",
                display_text="d",
                required_parts=["Adapter"],
                sub_steps=[{"action": "attach", "part": "Adapter", "target": "Gehäuse"}],
            )
        )
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    import asyncio

    cand = {"transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    sess._preview_cache["c1"] = dict(cand)
    sess._preview_cache["c2"] = dict(cand)
    r1 = asyncio.run(place_instance(sess, {"part_id": "adapter", "candidate_id": "c1", "step": 1}))
    assert r1["ok"], r1
    r2 = asyncio.run(place_instance(sess, {"part_id": "adapter", "candidate_id": "c2", "step": 2}))
    assert r2["ok"], r2  # second placement of the SAME part must succeed
    ids = [i.instance_id for i in state.assembly.instances]
    assert ids == ["adapter#1", "adapter#2"]
    assert _completeness_issues(sess) == []
    # and removing only step 2 leaves step 1's instance untouched
    from cadx.guide.assembly_tools import reset_step

    asyncio.run(reset_step(sess, {"step": 2}))
    assert [i.instance_id for i in state.assembly.instances] == ["adapter#1"]


def test_preassembly_places_deterministically() -> None:
    """The planner resolves the step plan without any LLM: base part via
    env fallback, then the mating part attached onto its port."""
    import asyncio

    from cadx.guide.preassembly import run_preassembly
    from cadx.guide.state import GuidePart, GuideStep, GuideSubStep, PortInfo

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts += [
        GuidePart(
            id="base",
            name="Base",
            ports=[PortInfo(name="p0", pos=[0, 0, 10], axis=[0, 0, 1], radius=10, gender="female")],
        ),
        GuidePart(
            id="plug",
            name="Plug",
            ports=[PortInfo(name="m0", pos=[0, 0, 0], axis=[0, 0, -1], radius=10, gender="male")],
        ),
    ]
    state.steps += [
        GuideStep(
            number=1,
            text="Base aufstellen",
            display_text="d",
            required_parts=["Base"],
            sub_steps=[GuideSubStep(action="place", part="Base")],
        ),
        GuideStep(
            number=2,
            text="Plug einstecken",
            display_text="d",
            required_parts=["Plug"],
            sub_steps=[GuideSubStep(action="attach", part="Plug", target="Base, linke obere Seite")],
        ),
    ]
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir="/tmp/x",
        project_id=None,
        guide_assets=[],
        executor=None,
    )
    report = asyncio.run(run_preassembly(sess))
    insts = {i.instance_id for i in state.assembly.instances}
    assert insts == {"base#1", "plug#1"}, (insts, report)
    plug = next(i for i in state.assembly.instances if i.part_id == "plug")
    assert plug.transform[2][3] != 0.0  # attached ABOVE the base, not at origin
    assert state.assembly.joints[0]["moving_instance"] == "plug#1"
    assert state.assembly.planner["placed"] == 2

    # resume-safe: a second run skips the satisfied requirements
    report2 = asyncio.run(run_preassembly(sess))
    assert len(state.assembly.instances) == 2
    assert all(r["reason"] == "already satisfied" for r in report2["skipped"])


def test_emit_assembly_artifacts_merges_glb(tmp_path) -> None:
    """The finished assembly ships as one glb artifact — the version page
    viewer keys off kind=glb, so a guide with placed instances must produce
    exactly that."""
    import asyncio

    import numpy as np
    import trimesh

    from cadx.guide.guide_runner import _emit_assembly_artifacts
    from cadx.guide.state import AssemblyInstance, GuidePart

    mesh = tmp_path / "part.glb"
    trimesh.creation.box(extents=(10, 10, 10)).export(str(mesh))
    state = GuideState()
    state.parts.append(GuidePart(id="cube", name="Cube", mesh_path=str(mesh)))
    t = np.eye(4)
    t[0, 3] = 50.0
    state.assembly.instances.append(
        AssemblyInstance(
            part_id="cube", instance_id="cube#1", placed_in_step=1, transform=t.tolist()
        )
    )
    from cadx import storage

    arts = asyncio.run(
        _emit_assembly_artifacts(state, storage.data_root() / "tmp_test_emit", executor=None)
    )
    glb_rel = arts.get("export:glb")
    assert glb_rel, arts

    sc = trimesh.load(str(storage.resolve(glb_rel)))
    assert "cube#1" in sc.geometry
    assert sc.bounds[0][0] > 40.0  # transform applied


def test_emit_assembly_artifacts_empty(tmp_path) -> None:
    import asyncio

    from cadx import storage
    from cadx.guide.guide_runner import _emit_assembly_artifacts

    assert asyncio.run(
        _emit_assembly_artifacts(GuideState(), storage.data_root() / "tmp_test_emit2", executor=None)
    ) == {}


def test_preassembly_pairs_symmetric_targets_1to1() -> None:
    """Two placed adapters + two pipe attach-tasks must pair adapter#1↔pipe#1
    and adapter#2↔pipe#2 — never both pipes onto the newest adapter."""
    from types import SimpleNamespace

    import numpy as np

    from cadx.guide.preassembly import _pick_target_instance
    from cadx.guide.state import AssemblyInstance, GuidePart, GuideState

    state = GuideState()
    state.parts.append(GuidePart(id="adapter", name="Adapter"))
    state.parts.append(GuidePart(id="pipe", name="Rohr"))
    for i in (1, 2):
        state.assembly.instances.append(
            AssemblyInstance(
                part_id="adapter",
                instance_id=f"adapter#{i}",
                placed_in_step=i,
                transform=np.eye(4).tolist(),
            )
        )
    # pipe#1 already joined adapter#1 — the next pipe must take adapter#2
    state.assembly.joints.append(
        {
            "moving_instance": "pipe#1",
            "moving_part": "pipe",
            "fixed_instance": "adapter#1",
            "fixed_port": "rim_0",
        }
    )
    sess = SimpleNamespace(state=state, _aliases_store={})
    assert _pick_target_instance(sess, "Adapter", moving_pid="pipe") == "adapter#2"


def test_free_placement_spreads_beside() -> None:
    """No-hint env placement lays parts out in a row — never a tower."""
    import numpy as np

    from cadx.guide.solver import solve_free_placement
    from cadx.guide.state import GuidePart

    p = GuidePart(id="p", name="P", bbox_min=[0, 0, 0], bbox_max=[10, 10, 10])
    placed = [{"bbox_min_world": [0, 0, 0], "bbox_max_world": [10, 10, 10]}]
    t = np.asarray(solve_free_placement(p, placed, ""), dtype=float).reshape(4, 4)
    assert t[0, 3] > 5.0  # pushed +X beside the placed part
    assert t[2, 3] == 0.0  # resting on the same level — not on top


def test_add_arrow_resolves_forgiving_ids() -> None:
    """Arrows must accept instance ids, names and manual labels — a 4x
    error streak on resolvable ids killed live annotate runs."""
    import asyncio

    from cadx.guide.annotate_tools import add_arrow
    from cadx.guide.state import AssemblyInstance, GuidePart, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(
        GuidePart(id="pipe_l70", name="Rohr DN22, l = 70 mm", mesh_path="/tmp/a.glb")
    )
    state.steps.append(GuideStep(number=2, text="t", display_text="d"))
    state.assembly.instances.append(
        AssemblyInstance(instance_id="pipe_l70#1", part_id="pipe_l70", placed_in_step=2)
    )
    sess = GuideToolSession(
        ctx, mission="annotate", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,  # type: ignore[arg-type]
    )
    for pid in ("pipe_l70#1", "Rohr DN22", "pipe_l70"):
        res = asyncio.run(add_arrow(sess, {"step": "step 2", "part_id": pid, "direction": "down"}))
        assert res["ok"] is True, pid
        assert state.annotations[-1]["part_id"] == "pipe_l70"
    res = asyncio.run(add_arrow(sess, {"step": 99, "part_id": "pipe_l70", "direction": "down"}))
    assert res["ok"] is False and "valid steps" in res["error"]
