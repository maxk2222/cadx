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
