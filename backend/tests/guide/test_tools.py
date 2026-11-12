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
