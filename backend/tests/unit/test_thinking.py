"""M11 adaptive thinking: policy resolution, client merge order, session wiring."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest

import cadx.llm.client as client_mod
from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.thinking import step_is_hard, thinking_body, thinking_on
from cadx.agent.toolloop import run_session
from cadx.agent.toolspecs import ToolSession
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.client import LLMClient
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import Critique
from cadx.runtime_config import RuntimeSettings

ON = '{"chat_template_kwargs": {"enable_thinking": true}}'
OFF = '{"chat_template_kwargs": {"enable_thinking": false}}'
ON_BODY = {"chat_template_kwargs": {"enable_thinking": True}}
OFF_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]


def _ctx(llm: FakeLLM | None = None, escalated: bool = False, **settings: Any) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f: Any) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    ctx = AgentCtx(
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
    ctx.thinking_escalated = escalated
    return ctx


def _call(tool: str, **args: Any) -> dict:
    return {"tool_calls": [{"name": tool, "arguments": args}]}


# --------------------------------------------------------------------------- #
# policy                                                                       #
# --------------------------------------------------------------------------- #


def test_step_is_hard_prefix_matching():
    csv = "repair,improve"
    assert step_is_hard("repair", csv)
    assert step_is_hard("repair_2", csv)  # attempt suffix counts
    assert step_is_hard("improve_3", " repair , improve ")
    assert not step_is_hard("critique", csv)
    assert not step_is_hard("intake", csv)
    assert not step_is_hard("repairman", csv)  # boundary-aware, not a bare prefix
    # only numeric attempt suffixes prefix-match — minor improve rounds keep
    # their own name out of the hard-steps set (thinking budget stays for
    # real hard steps)
    assert not step_is_hard("improve_minor", csv)
    assert not step_is_hard("improve_minor_2", csv)
    assert step_is_hard("improve_minor", "improve_minor")  # exact names still honoured
    assert not step_is_hard("rep", csv)
    assert not step_is_hard("repair", "")


@pytest.mark.parametrize(
    ("mode", "step", "escalated", "expected"),
    [
        ("off", "repair", False, "off"),
        ("off", "critique", True, "off"),  # the latch can't override "off"
        ("always", "critique", False, "on"),
        ("always", "intake", True, "on"),
        ("auto", "repair", False, "on"),
        ("auto", "repair_2", False, "on"),
        ("auto", "improve_1", False, "on"),
        ("auto", "critique", False, "off"),
        ("auto", "intake", False, "off"),
        ("auto", "mesh_orient", False, "off"),
        ("auto", "critique", True, "on"),  # stall latch escalates any step
        ("auto", "plan", True, "on"),
    ],
)
def test_policy_matrix(mode: str, step: str, escalated: bool, expected: str):
    ctx = _ctx(
        escalated=escalated,
        llm_thinking=mode,
        llm_thinking_body_on=ON,
        llm_thinking_body_off=OFF,
        llm_thinking_hard_steps="repair,improve",
    )
    assert thinking_on(ctx, step) is (expected == "on")
    assert thinking_body(ctx, step) == (ON_BODY if expected == "on" else OFF_BODY)


def test_empty_body_settings_return_none():
    ctx = _ctx(llm_thinking="always")
    assert thinking_body(ctx, "critique") is None
    ctx = _ctx(llm_thinking="auto")
    assert thinking_body(ctx, "repair") is None
    assert thinking_body(ctx, "critique") is None


def test_malformed_body_json_degrades_to_none():
    """The RuntimeSettings validator guards the settings API; the policy still
    must not raise when a malformed value lands on the model directly."""
    ctx = _ctx(llm_thinking="always")
    ctx.runtime_settings.llm_thinking_body_on = "{not json"
    assert thinking_body(ctx, "critique") is None
    ctx.runtime_settings.llm_thinking_body_on = "[1, 2]"  # valid JSON, not an object
    assert thinking_body(ctx, "critique") is None
    ctx.runtime_settings.llm_thinking_body_on = '"plain"'
    assert thinking_body(ctx, "critique") is None


def test_body_json_parsed_per_call():
    ctx = _ctx(
        llm_thinking="auto",
        llm_thinking_body_on='{"reasoning_effort": "high"}',
        llm_thinking_body_off='{"reasoning_effort": "low"}',
    )
    assert thinking_body(ctx, "repair") == {"reasoning_effort": "high"}
    assert thinking_body(ctx, "intake") == {"reasoning_effort": "low"}


# --------------------------------------------------------------------------- #
# client + fake                                                                #
# --------------------------------------------------------------------------- #


async def test_fake_llm_records_extra_body():
    llm = FakeLLM([{"text": "ok"}])
    await llm.chat("code", [{"role": "user", "content": "hi"}], extra_body={"k": 1})
    assert llm.calls[0]["extra_body"] == {"k": 1}
    await llm.chat("code", [{"role": "user", "content": "hi"}])
    assert llm.calls[1]["extra_body"] is None


async def test_client_per_call_extra_body_wins():
    """chat() merges payload <- self.extra_body <- extra_body (last wins)."""
    c = LLMClient(base_url="http://x", api_key="k")
    c.stream = False
    c.extra_body = {"reasoning_effort": "low", "shared": "client"}
    posted: list[dict] = []

    async def fake_post(payload: dict, role: str):
        posted.append(payload)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }

    c._post = fake_post  # type: ignore[method-assign]
    await c.chat(
        "code",
        [{"role": "user", "content": "hi"}],
        extra_body={"reasoning_effort": "high", "per_call": True},
    )
    p = posted[0]
    assert p["reasoning_effort"] == "high"  # per-call wins on conflict
    assert p["shared"] == "client"  # client-wide keys still merged
    assert p["per_call"] is True


async def test_400_fallback_strips_extra_body_keys(monkeypatch):
    """A dialect key the provider rejects is dropped on retry — the retried
    payload is the merged one minus the offending keys."""
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        if len(posted) == 1:
            return httpx.Response(400, json={"error": "unsupported key"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    orig = httpx.AsyncClient

    def patched(*a: Any, **kw: Any) -> httpx.AsyncClient:
        kw["transport"] = httpx.MockTransport(handler)
        return orig(*a, **kw)

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", patched)
    c = LLMClient(base_url="http://x", api_key="k")
    c.stream = False
    r = await c.chat(
        "code",
        [{"role": "user", "content": "hi"}],
        extra_body={"enable_thinking": True},
    )
    assert r.text == "ok"
    assert posted[0]["enable_thinking"] is True
    assert "enable_thinking" not in posted[1]
    assert posted[1]["model"] and posted[1]["messages"]


# --------------------------------------------------------------------------- #
# run_session wiring                                                           #
# --------------------------------------------------------------------------- #


async def test_session_soft_step_gets_off_body():
    llm = FakeLLM([{"text": "final"}])
    ctx = _ctx(
        llm=llm,
        llm_thinking="auto",
        llm_thinking_body_on=ON,
        llm_thinking_body_off=OFF,
    )
    session = ToolSession(ctx, kind="info")
    res = await run_session(ctx, "vision", MESSAGES, session=session, max_calls=3, step="critique")
    assert res.response is not None
    assert llm.calls[0]["extra_body"] == OFF_BODY
    assert ctx.trace.records[0]["parsed"]["thinking"] is False


async def test_session_hard_step_gets_on_body():
    llm = FakeLLM([{"text": "final"}])
    ctx = _ctx(llm=llm, llm_thinking="auto", llm_thinking_body_on=ON, llm_thinking_body_off=OFF)
    session = ToolSession(ctx, kind="info")
    await run_session(ctx, "vision", MESSAGES, session=session, max_calls=3, step="improve_2")
    assert llm.calls[0]["extra_body"] == ON_BODY
    assert ctx.trace.records[0]["parsed"]["thinking"] is True


async def test_session_escalated_latch_gets_on_body():
    llm = FakeLLM([{"text": "final"}])
    ctx = _ctx(
        llm=llm,
        escalated=True,
        llm_thinking="auto",
        llm_thinking_body_on=ON,
    )
    session = ToolSession(ctx, kind="info")
    await run_session(ctx, "vision", MESSAGES, session=session, max_calls=3, step="critique")
    assert llm.calls[0]["extra_body"] == ON_BODY
    assert ctx.trace.records[0]["parsed"]["thinking"] is True


async def test_grace_turn_carries_thinking_body():
    """The tools-off rescue turn resolves the same body as the session turns."""
    llm = FakeLLM(
        [
            _call("search_catalog", query="m4"),
            {"score": 77, "passed": False, "discrepancies": [], "confidence": 0.5},
        ]
    )
    ctx = _ctx(llm=llm, llm_thinking="always", llm_thinking_body_on=ON)
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
    assert len(llm.calls) == 2  # session turn + grace turn
    assert all(c["extra_body"] == ON_BODY for c in llm.calls)
    assert ctx.trace.records[-1]["parsed"]["thinking"] is True
