"""M3 repair convergence guards: elision rejection, thinking budget, stall.

Covers the lazy-elision detector (tokenize-based, '#' inside strings is not a
comment), its wiring into submit_code/done, the per-job thinking-call budget
latch in thinking_on(), the job-wide exec-error signature tally that flips
repairs to full_code, and the deterministic error-history digest.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.orchestrator import (
    _exec_error_signature,
    _note_exec_error,
    _repair_error_history,
)
from cadx.agent.thinking import thinking_body, thinking_on
from cadx.agent.toolspecs import ToolSession, elision_marker
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOX_CODE = (FIXTURES / "simple_box.py").read_text()

ON_BODY = '{"chat_template_kwargs": {"enable_thinking": true}}'
OFF_BODY = '{"chat_template_kwargs": {"enable_thinking": false}}'


def _ctx(llm: FakeLLM | None = None, **settings: Any) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f: Any) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    return AgentCtx(
        llm=llm or FakeLLM([{"text": "ok"}]),
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
        runtime_settings=RuntimeSettings(**settings),
    )


# --------------------------------------------------------------------------- #
# lazy elision                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "comment",
    [
        "# ... rest unchanged",
        "# ... rest",
        "# ...",
        "# unchanged",
        "# previous code",
        "# existing code above",
        "# TODO rest",
        "# rest of the code stays as before",
        "# code is unchanged",
        "# omitted for brevity",
        "# same as before",
        "# keep the rest",
        "    # implementation same as previous",
    ],
)
def test_elision_marker_detects_lazy_comments(comment: str) -> None:
    code = BOX_CODE + "\n" + comment + "\n"
    assert elision_marker(code) is not None


@pytest.mark.parametrize(
    "code",
    [
        "x = '# ... rest unchanged'\n",  # inside a string literal — not a comment
        "s = '''docstring\n# ... rest\n'''\n",
        "# keep wall thickness unchanged at 2mm\n",
        "# the fillet radius changed to 1.5\n",
        "result = Box(1, 1, 1)  # done\n",
        "# TODO parametrize the fillet radius later\n",
    ],
)
def test_elision_marker_ignores_strings_and_normal_comments(code: str) -> None:
    assert elision_marker(code) is None


def test_elision_marker_unparseable_falls_back() -> None:
    # tokenize fails on broken code — comment-only lines still get checked
    assert elision_marker("def broken(:\n# ... rest of code\n") is not None


async def test_submit_code_rejects_elision() -> None:
    ctx = _ctx()
    session = ToolSession(ctx, kind="codegen")
    res = await session.execute("submit_code", {"code": BOX_CODE + "\n# ... rest unchanged\n"})
    assert res["ok"] is False
    assert "COMPLETE" in res["error"]
    assert session.submitted is None  # not accepted as the answer


async def test_submit_code_accepts_clean_code() -> None:
    ctx = _ctx()
    session = ToolSession(ctx, kind="codegen")
    res = await session.execute("submit_code", {"code": BOX_CODE})
    assert res["ok"] is True
    assert session.submitted is not None


async def test_done_rejects_elided_buffer() -> None:
    ctx = _ctx()
    session = ToolSession(ctx, kind="edit", code=BOX_CODE + "\n# ... rest\n")
    res = await session.execute("done", {})
    assert res["ok"] is False
    assert "COMPLETE" in res["error"]
    assert session.submitted is None


# --------------------------------------------------------------------------- #
# thinking budget latch                                                       #
# --------------------------------------------------------------------------- #


def test_thinking_budget_latches_off() -> None:
    """After llm_thinking_max_calls thinking-ON calls the policy resolves OFF
    for every step — even a hard step and even under the escalation latch."""
    ctx = _ctx(
        llm_thinking="auto",
        llm_thinking_max_calls=2,
        llm_thinking_body_on=ON_BODY,
        llm_thinking_body_off=OFF_BODY,
    )
    ctx.thinking_escalated = True
    assert thinking_on(ctx, "repair_1") is True
    assert thinking_body(ctx, "repair_1") == {"chat_template_kwargs": {"enable_thinking": True}}
    assert ctx.thinking_calls_used == 1
    thinking_body(ctx, "repair_2")
    assert ctx.thinking_calls_used == 2
    # budget spent: hard step + escalation latch both lose
    assert thinking_on(ctx, "repair_3") is False
    assert ctx.thinking_budget_exhausted is True
    assert thinking_body(ctx, "repair_3") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert ctx.thinking_calls_used == 2  # off calls don't consume budget


def test_thinking_budget_counts_only_thinking_calls() -> None:
    ctx = _ctx(llm_thinking="auto", llm_thinking_max_calls=2, llm_thinking_hard_steps="repair")
    # soft steps resolve OFF under "auto" — they must not drain the budget
    assert thinking_on(ctx, "critique") is False
    thinking_body(ctx, "critique")
    assert ctx.thinking_calls_used == 0
    thinking_body(ctx, "repair")
    assert ctx.thinking_calls_used == 1


def test_thinking_budget_also_caps_always_mode() -> None:
    ctx = _ctx(llm_thinking="always", llm_thinking_max_calls=1)
    assert thinking_on(ctx, "critique") is True
    thinking_body(ctx, "critique")
    assert thinking_on(ctx, "critique") is False
    assert ctx.thinking_budget_exhausted is True


def test_thinking_off_never_latches_budget() -> None:
    ctx = _ctx(llm_thinking="off", llm_thinking_max_calls=0)
    for _ in range(3):
        assert thinking_on(ctx, "repair") is False
        thinking_body(ctx, "repair")
    assert ctx.thinking_budget_exhausted is False  # off != exhausted


# --------------------------------------------------------------------------- #
# exec-error signature tally + error history                                 #
# --------------------------------------------------------------------------- #


def test_exec_error_signature_normalizes_numbers() -> None:
    r1 = {"error": "ValueError: Failed creating a fillet with radius of 3.25 at line 10"}
    r2 = {"error": "ValueError: Failed creating a fillet with radius of 9.99 at line 42"}
    assert _exec_error_signature(r1) == _exec_error_signature(r2)
    other = {"error": "NameError: name 'helper' is not defined"}
    assert _exec_error_signature(other) != _exec_error_signature(r1)
    assert _exec_error_signature({"ok": True, "error": ""}) == ""
    assert _exec_error_signature(r1).startswith("occ-fillet")


def test_note_exec_error_flips_at_second_sighting() -> None:
    counts: dict[str, int] = {}
    sig = "api-misuse:AttributeError: 'Edge' object has no attribute 'radius'"
    assert _note_exec_error(counts, sig) is False  # first sighting
    assert _note_exec_error(counts, sig) is True  # repeated → strategy switch
    assert _note_exec_error(counts, sig) is True  # stays stalled
    assert _note_exec_error(counts, "other:sig") is False  # distinct signature
    assert _note_exec_error(counts, "") is False  # no error text


def test_repair_error_history_is_deterministic() -> None:
    counts = {
        "occ-fillet-empty:ValueError: Failed creating a fillet": 3,
        "api-misuse:AttributeError: 'Edge' object": 2,
        "name:NameError: name 'x' is not defined": 1,
    }
    h1 = _repair_error_history(counts, attempts=5)
    h2 = _repair_error_history(counts, attempts=5)
    assert h1 == h2
    assert "ERROR HISTORY" in h1
    assert "3x" in h1 and "2x" in h1 and "1x" in h1
    assert "occ-fillet" in h1 and "api-misuse" in h1
    assert "repair attempts so far: 5" in h1
    assert _repair_error_history({}, attempts=0) == ""


# --------------------------------------------------------------------------- #
# repair() integration: docs + history in context, elided answer dropped     #
# --------------------------------------------------------------------------- #


async def test_repair_injects_docs_and_error_history() -> None:
    """One-shot repair path (no tool session): the classified AttributeError
    pulls the Edge member digest, the passed history lands verbatim."""
    from cadx.agent.steps import codegen as codegen_step
    from cadx.llm.schemas import PartSpec

    dim = {"value": 60.0, "unit": "mm", "source": "stated"}
    spec = PartSpec.model_validate(
        {
            "name": "part",
            "summary": "s",
            "units": "mm",
            "length_x": dim,
            "width_y": dim,
            "height_z": dim,
            "image_roles": [],
            "features": [],
        }
    )
    llm = FakeLLM([{"reasoning": "fixed", "edits": [{"find": "radius()", "replace": "radius"}]}])
    ctx = _ctx(llm=llm)
    await codegen_step.repair(
        ctx,
        "r = e.radius()\nresult = e\n",
        spec,
        None,
        "AttributeError: 'Edge' object has no attribute 'radius'",
        'File "model.py", line 1, in <module>',
        "",
        iteration=1,
        repair_n=1,
        error_history="ERROR HISTORY this job\n  2x api-misuse:x",
    )
    user = llm.calls[0]["messages"][1]["content"]
    text = user if isinstance(user, str) else "\n".join(p.get("text", "") for p in user)
    assert "CORRECT API" in text
    assert "radius" in text
    assert "ERROR HISTORY" in text


async def test_repair_drops_elided_full_code() -> None:
    """An elided full_code is dropped BEFORE the caller can apply it — the
    remaining clean edits still go through."""
    from cadx.agent.steps import codegen as codegen_step
    from cadx.llm.schemas import PartSpec

    dim = {"value": 60.0, "unit": "mm", "source": "stated"}
    spec = PartSpec.model_validate(
        {
            "name": "part",
            "summary": "s",
            "units": "mm",
            "length_x": dim,
            "width_y": dim,
            "height_z": dim,
            "image_roles": [],
            "features": [],
        }
    )
    llm = FakeLLM(
        [
            {
                "reasoning": "fixed",
                "edits": [{"find": "a", "replace": "b"}],
                "full_code": "PARAMS = {}\n# ... rest unchanged\nresult = None\n",
            }
        ]
    )
    ctx = _ctx(llm=llm)
    presp = await codegen_step.repair(
        ctx,
        "result = None\n",
        spec,
        None,
        "RuntimeError: boom",
        None,
        None,
        iteration=1,
        repair_n=1,
    )
    assert presp.full_code is None  # elision dropped
    assert len(presp.edits) == 1  # clean edits survive
    assert "elision" in presp.reasoning
