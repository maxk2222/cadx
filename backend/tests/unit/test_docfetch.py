"""docfetch — error-triggered quickref injection (M2).

The failing symbol is extracted from classified exec errors and resolved to
a compact runtime-introspected digest; verify/render error results then carry
a `docs` field so the repair context holds the CORRECT signature. Everything
is best-effort: nothing resolvable → no `docs` key at all.
"""

from __future__ import annotations

import uuid

import pytest

from cadx.agent.docfetch import docs_for_error
from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.toolspecs import ToolSession
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings

pytest.importorskip("build123d")  # digests introspect the installed package


def _ctx() -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f: object) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    return AgentCtx(
        llm=FakeLLM([{"text": "ok"}]),
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
        runtime_settings=RuntimeSettings(),
    )


def test_attribute_error_property_marks_kind() -> None:
    """The observed failure: Edge.radius is a PROPERTY — calling it produced
    the AttributeError; the digest names the kind so the repair doesn't
    guess again."""
    docs = docs_for_error("AttributeError: 'Edge' object has no attribute 'radius'")
    assert docs
    assert "Edge" in docs
    assert "radius" in docs
    assert "property" in docs
    assert len(docs) <= 400


def test_attribute_error_unknown_member_lists_real_members() -> None:
    docs = docs_for_error("AttributeError: 'Edge' object has no attribute 'bogus_thing'")
    assert docs
    assert "has no member 'bogus_thing'" in docs
    assert "Edge members:" in docs
    assert len(docs) <= 400


def test_attribute_error_class_owner() -> None:
    docs = docs_for_error("AttributeError: type object 'Edge' has no attribute 'filletX'")
    assert docs
    assert "Edge" in docs


def test_module_attribute_error_lists_module_members() -> None:
    docs = docs_for_error("AttributeError: module 'build123d' has no attribute 'Extrude'")
    assert docs
    assert "no member 'Extrude'" in docs
    assert "extrude" in docs  # the real name is in the member listing


def test_name_error_resolves_build123d() -> None:
    docs = docs_for_error("NameError: name 'Pos' is not defined")
    assert docs
    assert "Pos" in docs
    assert len(docs) <= 400


def test_name_error_resolves_cadlib() -> None:
    docs = docs_for_error("NameError: name 'hole_grid' is not defined")
    assert docs
    assert "hole_grid(" in docs


def test_type_error_missing_arg_shows_signature() -> None:
    docs = docs_for_error("TypeError: extrude() missing 1 required positional argument: 'amount'")
    assert docs
    assert "extrude(" in docs
    assert "amount" in docs


def test_type_error_unexpected_kwarg_shows_signature() -> None:
    docs = docs_for_error("TypeError: Box() got an unexpected keyword argument 'size'")
    assert docs
    assert "Box(" in docs


def test_type_error_too_many_args_shows_signature() -> None:
    docs = docs_for_error(
        "TypeError: Cylinder() takes from 1 to 3 positional arguments but 5 were given"
    )
    assert docs
    assert "Cylinder(" in docs


def test_dotted_callee_uses_owner_context() -> None:
    docs = docs_for_error("TypeError: Edge.fillet() takes 2 positional arguments but 4 were given")
    assert docs
    assert "Edge" in docs
    assert "fillet" in docs


def test_unresolvable_error_returns_empty() -> None:
    assert docs_for_error("RuntimeError: something unrelated exploded") == ""
    assert docs_for_error("") == ""
    assert docs_for_error(None, "Traceback...\nRuntimeError: boom") == ""


def test_failing_line_fallback() -> None:
    """No symbol in the message → the call on the failing source line."""
    code = "from build123d import *\nsolid = extrude(to_extrude=sk, amount=None)\n"
    docs = docs_for_error("TypeError: cannot unpack non-iterable", None, code, 2)
    assert docs
    assert "extrude(" in docs


async def test_verify_error_result_carries_docs() -> None:
    async def failing_verify(code: str) -> dict:
        return {
            "ok": False,
            "error": "AttributeError: 'Edge' object has no attribute 'radius'",
            "traceback": 'File "model.py", line 4, in <module>\n  r = e.radius()',
        }

    ctx = _ctx()
    session = ToolSession(
        ctx,
        kind="edit",
        code="from build123d import *\nr = e.radius()\nresult = e\n",
        verify_fn=failing_verify,
    )
    res = await session.execute("verify", {})
    assert res["ok"] is False
    assert res["category"] == "api-misuse"
    assert "docs" in res
    assert "radius" in res["docs"]
    assert len(res["docs"]) <= 400


async def test_verify_error_without_symbol_has_no_docs() -> None:
    async def failing_verify(code: str) -> dict:
        return {"ok": False, "error": "RuntimeError: something unrelated"}

    ctx = _ctx()
    session = ToolSession(ctx, kind="edit", code="result = None\n", verify_fn=failing_verify)
    res = await session.execute("verify", {})
    assert res["ok"] is False
    assert "docs" not in res
