"""Tool session loop: read/edit/done flows, budgets, error handling."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import PatchResponse

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOX_CODE = (FIXTURES / "simple_box.py").read_text()


def _ctx(llm: FakeLLM) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    return AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="test",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
    )


def _call(tool: str, **args) -> dict:
    return {"tool_calls": [{"name": tool, "arguments": args}]}


MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]


async def test_edit_session_read_edit_done():
    """read -> set_param -> done produces a changed buffer."""
    llm = FakeLLM(
        [
            _call("search_code", pattern="fillet_r"),
            _call("set_param", name="fillet_r", value=2.0),
            _call("done", reasoning="bigger fillet"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    assert tools_enabled(ctx, "code")
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    assert res.submitted is not None
    assert res.code is not None and '"fillet_r": 2.0' in res.code
    assert res.tool_calls_made == 3


async def test_session_stops_when_job_cancelled():
    """check_cancel is polled once per turn — a session must not keep
    burning LLM calls after the job was cancelled/finalised externally."""
    llm = FakeLLM([_call("search_code", pattern="x"), {"text": "final answer"}])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)

    calls = 0

    async def cancelled_from_second_turn() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    ctx.check_cancel = cancelled_from_second_turn
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    assert res.cancelled is True
    assert res.exhausted is True
    assert res.tool_calls_made == 1  # first turn ran, second turn aborted before the LLM call
    # protocol: assistant echo + tool results fed back
    assert any(m.get("role") == "tool" for c in llm.calls for m in c["messages"])


async def test_session_content_fallback_calls():
    """A provider smuggling the call into content JSON still drives the loop."""
    llm = FakeLLM(
        [
            {
                "content_tool_call": {
                    "name": "set_param",
                    "arguments": {"name": "height", "value": 12.0},
                }
            },
            _call("done"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    assert res.code and '"height": 12.0' in res.code


async def test_done_rejects_invalid_then_accepts():
    """done runs the contract check: a broken buffer is fed back as an error."""
    llm = FakeLLM(
        [
            _call("replace_lines", start=1, end=1, code="import os"),
            _call("done"),  # rejected: contract check fails (bad import + no PARAMS?)
            _call("replace_lines", start=1, end=1, code="from build123d import *"),
            _call("done"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    assert res.code is not None
    assert "import os" not in res.code


async def test_budget_exhaustion_keeps_buffer():
    llm = FakeLLM(
        [
            _call("set_param", name="width", value=30.0),
            _call("set_param", name="height", value=15.0),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=2, step="improve")
    assert res.exhausted
    assert res.code and '"width": 30.0' in res.code


async def test_unknown_tool_is_result_error():
    llm = FakeLLM([_call("bogus_tool"), _call("done")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    assert res.tool_log[0]["result"]["ok"] is False
    assert "not available" in res.tool_log[0]["result"]["error"]


async def test_info_session_final_answer():
    """info session: tool calls explore, final content answer is the schema JSON."""
    llm = FakeLLM(
        [
            _call("search_catalog", query="countersunk m4"),
            {"score": 90, "passed": True, "discrepancies": [], "matches": [], "confidence": 0.9},
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    from cadx.llm.schemas import Critique

    session = ToolSession(ctx, kind="info")
    res = await run_session(
        ctx,
        "vision",
        MESSAGES,
        session=session,
        json_schema=Critique,
        max_calls=6,
        step="critique",
    )
    assert res.response is not None
    assert isinstance(res.response.parsed, Critique)
    assert res.response.parsed.score == 90


async def test_malformed_arguments_become_tool_error():
    llm = FakeLLM(
        [
            {"tool_calls": [{"name": "read_lines", "arguments": "{broken"}]},
            _call("done"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    assert res.tool_log[0]["name"] == "read_lines"
    assert res.tool_log[0]["result"]["ok"] is False


async def test_verify_tool_runs_sandbox():
    """verify executes the buffer via verify_fn and reports the outcome."""
    llm = FakeLLM([_call("verify"), _call("done")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    calls: list[str] = []

    async def verify_fn(code: str):
        calls.append(code)
        return {"ok": True, "analysis": {"bbox_size": [40, 20, 10], "is_valid": True}}

    session = ToolSession(ctx, kind="edit", code=BOX_CODE, verify_fn=verify_fn)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    assert calls and calls[0] == BOX_CODE
    assert res.tool_log[0]["result"]["ok"]


async def test_tools_disabled_for_legacy():
    llm = FakeLLM([PatchResponse(reasoning="x").model_dump()])
    assert not tools_enabled(_ctx(llm), "code")


async def test_tools_enabled_is_per_role():
    """A vision-model 400 must not disable tools for the code model."""
    llm = FakeLLM([])
    llm.tools_mode = "auto"
    llm.tools_ok = {"vision": False}
    ctx = _ctx(llm)
    assert not tools_enabled(ctx, "vision")
    assert tools_enabled(ctx, "code")


async def test_unlisted_tool_call_rejected():
    """submit_code in an inspect session must not end the session — the call
    is rejected as unavailable and the model can still answer."""
    llm = FakeLLM(
        [
            _call("submit_code", code="from build123d import *\nresult = Box(1,1,1)"),
            {"score": 88, "passed": True, "discrepancies": [], "confidence": 0.9},
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    from cadx.llm.schemas import Critique

    session = ToolSession(ctx, kind="inspect", code=BOX_CODE)
    res = await run_session(
        ctx,
        "vision",
        MESSAGES,
        session=session,
        json_schema=Critique,
        max_calls=6,
        step="critique",
    )
    assert res.tool_log[0]["result"]["ok"] is False
    assert "not available" in res.tool_log[0]["result"]["error"]
    assert session.submitted is None  # the rogue call did NOT submit
    assert isinstance(res.response.parsed, Critique)
    assert res.response.parsed.score == 88


async def test_verify_drops_from_schema_after_budget():
    """session.tools() is recomputed each turn — a spent verify budget must
    remove the tool from the advertised schema."""
    llm = FakeLLM([_call("verify"), _call("verify"), _call("done")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)

    async def verify_fn(code: str):
        return {"ok": True, "analysis": {"bbox_size": [40, 20, 10]}}

    session = ToolSession(ctx, kind="edit", code=BOX_CODE, verify_fn=verify_fn)
    session.verify_left = 1  # force a one-call budget
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    assert "verify" in llm.calls[0]["tool_names"]
    assert "verify" not in llm.calls[1]["tool_names"]
    assert "not available" in res.tool_log[1]["result"]["error"]


async def test_grace_turn_rescues_schema_answer():
    """Exhausted info sessions get one tools-off final turn for the JSON."""
    llm = FakeLLM(
        [
            _call("search_catalog", query="m4"),
            {"score": 77, "passed": False, "discrepancies": [], "confidence": 0.5},
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    from cadx.llm.schemas import Critique

    session = ToolSession(ctx, kind="info")
    res = await run_session(
        ctx,
        "vision",
        MESSAGES,
        session=session,
        json_schema=Critique,
        max_calls=1,
        step="critique",
    )
    assert res.exhausted
    assert isinstance(res.response.parsed, Critique)
    assert res.response.parsed.score == 77
    # grace turn ran without tools
    assert llm.calls[-1]["tools"] is False


async def test_library_tools_search_and_attach():
    """search_library / library_part / attach_library_part on the user library."""
    from cadx.db.models import LibraryPart
    from cadx.db.session import create_all, session_scope

    await create_all()
    async with session_scope() as db:
        from sqlmodel import delete

        await db.execute(delete(LibraryPart))
        await db.commit()
    step = LibraryPart(
        name="M4 insert boss",
        kind="step",
        filename="boss.step",
        path=f"library/{uuid.uuid4()}/boss.step",
        caption="threaded boss",
    )
    tmpl = LibraryPart(
        name="plate template",
        kind="template",
        code="PARAMS = {'l': 50.0}",
        params={"l": {"value": 50.0}},
    )
    async with session_scope() as db:
        db.add(step)
        db.add(tmpl)
        await db.commit()

    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="codegen")
    assert "attach_library_part" in session.allowed_names()

    res = await session.execute("search_library", {"query": "boss"})
    assert res["ok"] and res["count"] == 1
    assert res["results"][0]["id"] == str(step.id)

    # kind filter excludes non-matching kinds
    res = await session.execute("search_library", {"query": "", "kind": "template"})
    assert [r["name"] for r in res["results"]] == ["plate template"]

    # details: file part shows load_reference usage, template shows code
    res = await session.execute("library_part", {"part_id": str(step.id)})
    assert f'load_reference("{step.id}")' in res["usage"]
    res = await session.execute("library_part", {"part_id": str(tmpl.id)})
    assert "PARAMS" in res["code"]

    # attach: file part becomes a reference asset on ctx
    res = await session.execute("attach_library_part", {"part_id": str(step.id)})
    assert res["ok"] and f'"{step.id}"' in res["usage"]
    assert ctx.assets[-1].asset_id == str(step.id)
    assert ctx.assets[-1].kind == "reference_step"
    assert str(step.id) in ctx.library_attached

    # attach: template lands in library_templates
    res = await session.execute("attach_library_part", {"part_id": str(tmpl.id)})
    assert res["ok"]
    assert ctx.library_templates[-1]["name"] == "plate template"

    # double attach is a no-op
    res = await session.execute("attach_library_part", {"part_id": str(step.id)})
    assert res["note"] == "already attached"
    assert ctx.library_attached.count(str(step.id)) == 1

    # unknown id is an error result, not an exception
    res = await session.execute("attach_library_part", {"part_id": str(uuid.uuid4())})
    assert not res["ok"]


async def test_library_attach_via_session():
    """An info session can pick a library part mid-conversation."""
    from cadx.db.models import LibraryPart
    from cadx.db.session import create_all, session_scope

    await create_all()
    part = LibraryPart(
        name="gear body",
        kind="step",
        filename="gear.step",
        path=f"library/{uuid.uuid4()}/gear.step",
    )
    async with session_scope() as db:
        db.add(part)
        await db.commit()

    llm = FakeLLM(
        [
            _call("search_library", query="gear"),
            _call("attach_library_part", part_id=str(part.id)),
            {"ok": True, "note": "using the saved gear"},
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="info")
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=6, step="plan")
    assert res.response is not None
    assert str(part.id) in ctx.library_attached
    assert ctx.assets[-1].asset_id == str(part.id)


async def test_inspect_session_cannot_attach():
    """Critique sessions get library search but not attach."""
    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="inspect")
    names = session.allowed_names()
    assert "search_library" in names
    assert "attach_library_part" not in names


# --------------------------------------------------------------------------- #
# mesh sessions + image-bearing tool results


def test_result_messages_with_images():
    """`_images` in a tool result becomes a user message with image parts —
    OpenAI tool messages can't carry images."""
    from cadx.agent.toolloop import _result_messages
    from cadx.llm.tools import _mk_call

    img = FIXTURES / "images" / "knob.png"
    tc = _mk_call("call_1", "mesh_preview", "{}")
    res = {"ok": True, "stats": {"faces": 5}, "_images": [str(img)]}

    native = _result_messages(tc, res, native=True)
    assert native[0]["role"] == "tool" and native[0]["tool_call_id"] == "call_1"
    assert native[1]["role"] == "user"
    assert any(p.get("type") == "image_url" for p in native[1]["content"])

    fallback = _result_messages(tc, res, native=False)
    assert fallback[0]["role"] == "user"
    assert "TOOL RESULT" in fallback[0]["content"]
    assert fallback[1]["role"] == "user"
    assert any(p.get("type") == "image_url" for p in fallback[1]["content"])


def test_result_messages_missing_image_skipped(tmp_path):
    """A missing image file must not break the message chain."""
    from cadx.agent.toolloop import _result_messages
    from cadx.llm.tools import _mk_call

    tc = _mk_call("call_2", "mesh_preview", "{}")
    res = {"ok": True, "_images": [str(tmp_path / "gone.png")]}
    msgs = _result_messages(tc, res, native=True)
    # only the tool message — no image user-message appended
    assert len(msgs) == 1 and msgs[0]["role"] == "tool"


def _mesh_wb(ctx: AgentCtx, tmp_path: Path, **kw):
    import trimesh

    from cadx.agent.mesh_tools import MeshWorkbench
    from cadx.cad.mesh_ops import MeshPostParams

    raw = tmp_path / "raw.glb"
    trimesh.creation.box([10, 10, 10]).export(str(raw), file_type="glb")
    kw.setdefault("spec", None)
    return MeshWorkbench(ctx, raw_path=raw, post=MeshPostParams(), iteration=1, **kw)


async def test_mesh_session_transform_and_done(tmp_path):
    """mesh kind session: transforms go to the workbench, done submits ops."""
    llm = FakeLLM(
        [
            _call("mesh_inspect"),
            _call("mesh_rotate", axis="z", deg=90),
            _call("mesh_flatten_bottom", height_mm=2.0),
            _call("done", reasoning="oriented + flat base"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    wb = _mesh_wb(ctx, tmp_path)
    session = ToolSession(ctx, kind="mesh", workbench=wb)
    res = await run_session(
        ctx, "vision", MESSAGES, session=session, max_calls=10, step="meshrefine"
    )
    assert res.submitted is not None
    ops = res.submitted["ops"]
    assert [o["op"] for o in ops] == ["rotate", "flatten_bottom"]
    assert wb.work is not None
    assert wb.work.bounds[0][2] == pytest.approx(0, abs=1e-6)


async def test_mesh_session_failed_tool_does_not_mutate(tmp_path):
    """Invalid args / out-of-bounds ops return ok:False and keep the work mesh."""
    llm = FakeLLM(
        [
            _call("mesh_slice", axis="z", at_mm=999.0, keep="high"),  # outside bounds
            _call("mesh_rotate", axis="z", deg=999),  # out of pydantic bounds
            _call("mesh_rotate", axis="z", deg=45),  # valid recovery
            _call("done"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    wb = _mesh_wb(ctx, tmp_path)
    faces0 = len(wb.work.faces)
    session = ToolSession(ctx, kind="mesh", workbench=wb)
    res = await run_session(
        ctx, "vision", MESSAGES, session=session, max_calls=10, step="meshrefine"
    )
    assert res.tool_log[0]["result"]["ok"] is False
    assert res.tool_log[1]["result"]["ok"] is False
    assert res.submitted is not None
    assert [o["op"] for o in res.submitted["ops"]] == ["rotate"]
    assert len(wb.work.faces) == faces0  # rotate didn't change face count


async def test_mesh_session_undo_reset(tmp_path):
    ctx = _ctx(FakeLLM([]))
    wb = _mesh_wb(ctx, tmp_path)
    assert (await wb.handle("mesh_undo", {}))["ok"] is False  # empty ops
    r = await wb.handle("mesh_rotate", {"axis": "z", "deg": 30})
    assert r["ok"] and len(wb.ops) == 1
    r = await wb.handle("mesh_undo", {})
    assert r["ok"] and not wb.ops
    r = await wb.handle("mesh_rotate", {"axis": "x", "deg": 15})
    r = await wb.handle("mesh_reset", {})
    assert r["ok"] and r["cleared"] == 1 and not wb.ops


async def test_mesh_session_op_budget(tmp_path):
    ctx = _ctx(FakeLLM([]))
    wb = _mesh_wb(ctx, tmp_path, max_ops=1)
    assert (await wb.handle("mesh_rotate", {"axis": "z", "deg": 10}))["ok"]
    res = await wb.handle("mesh_rotate", {"axis": "z", "deg": 20})
    assert res["ok"] is False and "budget" in res["error"]


async def test_mesh_session_preview(tmp_path, monkeypatch):
    """preview exports the work mesh, calls the executor, returns _images."""
    ctx = _ctx(FakeLLM([]))
    wb = _mesh_wb(ctx, tmp_path, preview_budget=1)
    captured: list[dict] = []

    async def fake_process(payload):
        captured.append(payload)
        from cadx import storage

        png = storage.data_root() / "prev_iso.png"
        png.write_bytes((FIXTURES / "images" / "knob.png").read_bytes())
        return {
            "ok": True,
            "images": {"iso": storage.relpath(png)},
            "stats": {"faces": 12, "watertight": True},
        }

    monkeypatch.setattr(ctx.executor, "process_mesh", fake_process)
    res = await wb.handle("mesh_preview", {})
    assert res["ok"] and res["_images"] and res["preview_left"] == 0
    assert captured[0]["mode"] == "preview"
    # budget exhausted → schema drops the tool
    session = ToolSession(ctx, kind="mesh", workbench=wb)
    names = session.allowed_names()
    assert "mesh_preview" not in names
    res = await wb.handle("mesh_preview", {})
    assert res["ok"] is False and "budget" in res["error"]


async def test_mesh_session_regenerate(tmp_path):
    llm = FakeLLM(
        [
            _call("mesh_rotate", axis="z", deg=10),
            _call("regenerate", subject_hint="rounder knob", note="wrong topology"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    wb = _mesh_wb(ctx, tmp_path)
    session = ToolSession(ctx, kind="mesh", workbench=wb)
    res = await run_session(
        ctx, "vision", MESSAGES, session=session, max_calls=10, step="meshrefine"
    )
    assert res.submitted["regenerate"] is True
    assert res.submitted["subject_hint"] == "rounder knob"
    assert [o["op"] for o in res.submitted["ops"]] == ["rotate"]


async def test_mesh_session_regenerate_rejected_in_other_kinds():
    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await session.execute("regenerate", {"subject_hint": "x"})
    assert res["ok"] is False  # regenerate isn't in the edit schema at all


async def test_view_image_tool_returns_image():
    """view_image resolves an image asset and returns it via _images."""
    from cadx.agent.types import InputAsset

    ctx = _ctx(FakeLLM([]))
    asset = InputAsset(
        asset_id="abcd1234-0000-0000-0000-000000000000",
        kind="image",
        path=FIXTURES / "images" / "knob.png",
        filename="knob.png",
        caption="front view",
    )
    ctx.assets.append(asset)
    session = ToolSession(ctx, kind="inspect")
    assert "view_image" in session.allowed_names()

    res = await session.execute("view_image", {"asset_id": asset.asset_id})
    assert res["ok"] and res["_images"] == [str(asset.path)]
    assert res["caption"] == "front view"

    # unique prefix also resolves
    res = await session.execute("view_image", {"asset_id": "abcd1234"})
    assert res["ok"]


async def test_view_image_unknown_and_missing(tmp_path):
    from cadx.agent.types import InputAsset

    ctx = _ctx(FakeLLM([]))
    ctx.assets.append(
        InputAsset(
            asset_id="a1b2",
            kind="image",
            path=tmp_path / "gone.png",
            filename="gone.png",
        )
    )
    session = ToolSession(ctx, kind="info")
    res = await session.execute("view_image", {"asset_id": "nope"})
    assert res["ok"] is False and res["available"] == ["a1b2"]
    res = await session.execute("view_image", {"asset_id": "a1b2"})
    assert res["ok"] is False and "missing" in res["error"]


async def test_render_model_returns_contact_sheet():
    """render_model executes the buffer and attaches the render via _images."""
    from cadx import storage

    llm = FakeLLM([_call("render_model"), _call("done")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)

    sheet = storage.data_root() / "render_sheet_test.png"
    sheet.write_bytes((FIXTURES / "images" / "knob.png").read_bytes())

    async def verify_fn(code: str):
        return {
            "ok": True,
            "analysis": {"bbox_size": [40, 20, 10], "is_valid": True, "watertight": True},
            "contact_sheet": storage.relpath(sheet),
        }

    session = ToolSession(ctx, kind="edit", code=BOX_CODE, verify_fn=verify_fn)
    assert "render_model" in session.allowed_names()
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    result = res.tool_log[0]["result"]
    assert result["ok"] and result["bbox_size"] == [40, 20, 10]
    assert result["_images"] == [str(sheet)]


async def test_render_model_exec_failure_diag():
    """A failed run returns the same diagnosis shape as verify."""

    async def verify_fn(code: str):
        return {"ok": False, "error": "NameError: nope"}

    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="edit", code=BOX_CODE, verify_fn=verify_fn)
    res = await session.execute("render_model", {})
    assert res["ok"] is False and res["render"] == "execution failed"
    assert res["error"]


async def test_render_model_shares_verify_budget():
    """render and verify draw on the same sandbox-execution budget."""
    ctx = _ctx(FakeLLM([]))

    async def verify_fn(code: str):
        return {"ok": True, "analysis": {}}

    session = ToolSession(ctx, kind="edit", code=BOX_CODE, verify_fn=verify_fn)
    session.verify_left = 1
    res = await session.execute("render_model", {})
    assert res["ok"]
    res = await session.execute("verify", {})
    assert res["ok"] is False
    # spent budget drops both tools from the recomputed schema
    assert "render_model" not in session.allowed_names()
    assert "verify" not in session.allowed_names()


# --------------------------------------------------------------------------- #
# optimize session (structural tools)                                          #

_FEM_RES = {
    "ok": True,
    "fem": {
        "material": {"name": "PLA"},
        "grid": {"pitch": 1.5, "elements": 100, "dofs": 300},
        "cases": [
            {
                "name": "c1",
                "safety_factor": 3.0,
                "max_vm_mpa": 10.0,
                "max_disp_mm": 0.2,
                "converged": True,
                "warnings": [],
            }
        ],
        "warnings": [],
    },
}
_TOP_RES = {
    "ok": True,
    "topopt": {
        "volfrac_target": 0.4,
        "volfrac_achieved": 0.41,
        "iterations": 20,
        "converged": True,
        "warnings": [],
    },
    "fem": {"cases": [{"name": "c1", "safety_factor": 1.8}], "warnings": []},
    "_iteration": 7,
}
_LC = {
    "name": "c1",
    "fixed": [{"kind": "face", "axis": "-x"}],
    "loads": [{"region": {"kind": "face", "axis": "+x"}, "force_n": [0, 0, -10]}],
}


async def test_optimize_session_fem_topopt_done():
    """analyze_fem + topopt_run + done(run_index) — compact metrics, run stored."""

    async def fem_fn(a):
        return _FEM_RES

    async def top_fn(a):
        return _TOP_RES

    llm = FakeLLM(
        [
            _call("analyze_fem", load_cases=[_LC]),
            _call("topopt_run", target_volume_fraction=0.4),
            _call("done", run_index=0, reasoning="meets SF"),
        ]
    )
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="optimize", fem_fn=fem_fn, topopt_fn=top_fn)
    res = await run_session(ctx, "vision", MESSAGES, session=session, max_calls=10, step="optimize")
    assert res.submitted == {"chosen_run": 0, "reasoning": "meets SF"}
    assert len(session.struct_runs) == 1
    assert session.struct_runs[0]["iteration"] == 7
    fem_log = next(t for t in session.tool_log if t["name"] == "analyze_fem")
    assert fem_log["result"]["cases"][0]["safety_factor"] == 3.0
    top_log = next(t for t in session.tool_log if t["name"] == "topopt_run")
    assert top_log["result"]["run_index"] == 0
    assert top_log["result"]["worst_safety_factor"] == 1.8


async def test_optimize_done_invalid_index_rejected():
    """done(run_index) out of range errors; -1 declines cleanly."""
    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="optimize")
    res = await session.execute("done", {"run_index": 5})
    assert res["ok"] is False
    res = await session.execute("done", {"run_index": -1, "reasoning": "base is fine"})
    assert res["ok"] and session.submitted["chosen_run"] == -1


async def test_optimize_budget_drops_tools():
    """Exhausted budgets remove the tools from the schema."""
    from cadx.runtime_config import RuntimeSettings

    async def top_fn(a):
        return _TOP_RES

    ctx = _ctx(FakeLLM([]))
    ctx.runtime_settings = RuntimeSettings(agent_topopt_max_runs=1, agent_fem_max_runs=0)
    session = ToolSession(ctx, kind="optimize", topopt_fn=top_fn)
    names = session.allowed_names()
    assert "topopt_run" in names and "analyze_fem" not in names
    res = await session.execute("topopt_run", {})
    assert res["ok"] and res["topopt_calls_left"] == 0
    assert "topopt_run" not in session.allowed_names()
    res = await session.execute("topopt_run", {})
    assert res["ok"] is False


async def test_optimize_fem_requires_cases():
    """analyze_fem without load_cases fails before touching the executor."""
    called = False

    async def fem_fn(a):
        nonlocal called
        called = True
        return _FEM_RES

    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="optimize", fem_fn=fem_fn)
    res = await session.execute("analyze_fem", {})
    assert res["ok"] is False and not called


async def test_optimize_tools_coerce_single_objects():
    """Models that send {obj} or a JSON string instead of [obj] get coerced."""
    seen: dict = {}

    async def fem_fn(a):
        seen["cases"] = a.get("load_cases")
        return _FEM_RES

    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="optimize", fem_fn=fem_fn)
    res = await session.execute("analyze_fem", {"load_cases": _LC})
    assert res["ok"] and seen["cases"] == [_LC]
    res = await session.execute("analyze_fem", {"load_cases": json.dumps([_LC])})
    assert res["ok"] and seen["cases"] == [_LC]


async def test_optimize_topopt_coerces_keep_regions():
    """topopt_run with a single keep_region dict reaches the executor as a list."""
    seen: dict = {}

    async def top_fn(a):
        seen["keep"] = a.get("keep_regions")
        return _TOP_RES

    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="optimize", topopt_fn=top_fn)
    region = {"kind": "face", "axis": "-z", "margin_mm": 1.0}
    res = await session.execute("topopt_run", {"keep_regions": region})
    assert res["ok"] and seen["keep"] == [region]


async def test_mesh_session_measure_section_overlay(tmp_path):
    """New inspect tools: measure vs spec/anchors, section + overlay images."""
    from cadx.llm.schemas import Dim, PartSpec

    spec = PartSpec(
        name="t",
        summary="t",
        units="mm",
        length_x=Dim(value=10.0, unit="mm", source="stated"),
        width_y=Dim(value=10.0, unit="mm", source="stated"),
        height_z=Dim(value=10.0, unit="mm", source="stated"),
        features=[],
    )
    anchors = {"holes": [{"index": 1, "x": 0.0, "y": 0.0, "dia": 4.0, "source": "image"}]}
    outline = [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]]
    ctx = _ctx(FakeLLM([]))
    wb = _mesh_wb(ctx, tmp_path, spec=spec, anchors=anchors, measured_outline=outline)
    r = await wb.handle("mesh_inspect", {})
    assert r["ok"] and r["features"]["outline_points"] >= 3 and r["anchors"]
    r = await wb.handle("mesh_measure", {})
    assert r["ok"] and r["spec_delta_mm"]["length_x"] == pytest.approx(0.0, abs=0.1)
    assert r["missing_holes"] == 1 and "drill_holes" in r["hint"]
    r = await wb.handle("mesh_overlay", {})
    assert r["ok"] and r["_images"] and r["preview_left"] == 3
    r = await wb.handle("mesh_section", {"axis": "z"})
    assert r["ok"] and r["_images"]
    r = await wb.handle("mesh_section", {"axis": "z", "at_mm": 99.0})
    assert r["ok"] is False and "outside" in r["error"]


async def test_mesh_session_drill_holes_via_tool(tmp_path):
    """mesh_drill_holes with from_anchors drills a real tunnel."""
    ctx = _ctx(FakeLLM([]))
    wb = _mesh_wb(ctx, tmp_path, anchors={"holes": [{"index": 1, "x": 0.0, "y": 0.0, "dia": 4.0}]})
    r = await wb.handle("mesh_drill_holes", {"from_anchors": True})
    assert r["ok"], r
    assert wb.ops[-1].op == "drill_holes"
    m = await wb.handle("mesh_measure", {})
    assert m["missing_holes"] == 0  # tunnel now exists at the anchor position


async def test_mesh_session_new_ops_via_tools(tmp_path):
    ctx = _ctx(FakeLLM([]))
    wb = _mesh_wb(ctx, tmp_path)
    assert (await wb.handle("mesh_clip_height", {"z_max": 5.0}))["ok"]
    assert wb.work.bounds[1][2] == pytest.approx(5.0, abs=0.5)
    assert (await wb.handle("mesh_symmetrize", {"axis": "y"}))["ok"]
    assert wb.work.is_watertight
    assert (await wb.handle("mesh_remesh", {"voxel_mm": 1.0}))["ok"]
    # part is only 5 mm thick after clip_height — use a thin shell
    assert (await wb.handle("mesh_shell", {"thickness_mm": 1.0, "opening": "bottom"}))["ok"]
    # unknown tool still rejected
    res = await wb.handle("mesh_teleport", {})
    assert res["ok"] is False and "unknown mesh tool" in res["error"]
