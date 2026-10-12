"""Tool-call parsing: response shape, stream accumulation, content fallback."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from cadx.llm.client import LLMClient
from cadx.llm.tools import (
    _StreamToolAcc,
    parse_content_tool_call,
    parse_tool_calls,
)


def test_parse_message_tool_calls():
    msg = {
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "read_lines", "arguments": '{"start": 1, "end": 5}'},
            }
        ]
    }
    calls = parse_tool_calls(msg)
    assert len(calls) == 1
    assert calls[0].name == "read_lines"
    assert calls[0].arguments == {"start": 1, "end": 5}
    assert not calls[0].arguments_error


def test_parse_message_tool_calls_bad_json():
    msg = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "done", "arguments": "{not json"}},
            {"id": "c2", "function": {"name": "verify", "arguments": "{}"}},
        ]
    }
    calls = parse_tool_calls(msg)
    assert calls[0].arguments_error
    assert calls[0].arguments == {}
    assert calls[1].arguments == {}
    assert not calls[1].arguments_error


def test_parse_message_tool_calls_dict_args():
    msg = {
        "tool_calls": [
            {"id": "c", "function": {"name": "set_param", "arguments": {"name": "x", "value": 1}}},
        ]
    }
    calls = parse_tool_calls(msg)
    assert calls[0].arguments == {"name": "x", "value": 1}


def test_stream_accumulation():
    acc = _StreamToolAcc()
    acc.feed([{"index": 0, "id": "call_1", "function": {"name": "read_"}}])
    acc.feed([{"index": 0, "function": {"name": "lines", "arguments": '{"sta'}}])
    acc.feed([{"index": 0, "function": {"arguments": 'rt": 1}'}}])
    acc.feed([{"index": 1, "id": "call_2", "function": {"name": "done", "arguments": "{}"}}])
    calls = acc.result()
    assert [c.name for c in calls] == ["read_lines", "done"]
    assert calls[0].arguments == {"start": 1}
    assert calls[0].id == "call_1"


def test_content_fallback_shapes():
    tc = parse_content_tool_call('{"name": "read_lines", "arguments": {"start": 2, "end": 9}}')
    assert tc is not None and tc.name == "read_lines" and tc.via_content
    assert tc.arguments == {"start": 2, "end": 9}

    tc2 = parse_content_tool_call('{"action": "verify", "args": {}}')
    assert tc2 is not None and tc2.name == "verify"

    tc3 = parse_content_tool_call(
        '{"tool_calls": [{"function": {"name": "done", "arguments": "{}"}}]}'
    )
    assert tc3 is not None and tc3.name == "done" and tc3.via_content


def test_content_fallback_rejects_normal_json():
    # a spec/critique JSON is NOT a tool call — "name" exists but there are no args
    assert parse_content_tool_call('{"name": "my part", "summary": "a plate"}') is not None
    # ^ returns a call named "my part" — the CALLER filters by offered tool names
    assert parse_content_tool_call("plain text answer") is None
    assert parse_content_tool_call('["a", "b"]') is None
    assert parse_content_tool_call("{broken json") is None


# --------------------------------------------------------------------------- #
# client-level: per-role tools_ok, strict-on, re-ask strip
# --------------------------------------------------------------------------- #

_TOOLS = [{"type": "function", "function": {"name": "done", "parameters": {}}}]


class _StubClient:
    """Minimal stand-in for httpx.AsyncClient in _handle_400."""

    def __init__(self, resp: httpx.Response) -> None:
        self.resp = resp
        self.posted: list[dict] = []

    async def post(self, url: str, json: dict | None = None, headers: dict | None = None):
        self.posted.append(json or {})
        return self.resp


def test_send_tools_per_role():
    c = LLMClient(base_url="http://x", api_key="k")
    c.tools_mode = "auto"
    c.tools_ok = {"vision": False}
    assert c._send_tools(_TOOLS, "code") is True
    assert c._send_tools(_TOOLS, "vision") is False
    c.tools_mode = "off"
    assert c._send_tools(_TOOLS, "code") is False


async def test_handle_400_marks_only_that_role():
    c = LLMClient(base_url="http://x", api_key="k")
    c.tools_mode = "auto"
    client = _StubClient(httpx.Response(200, json={"ok": 1}))
    payload = {"tools": _TOOLS, "tool_choice": "auto", "messages": []}
    resp, _out = await c._handle_400(client, httpx.Response(400), payload, {}, "vision")
    assert resp.status_code == 200
    assert "tools" not in client.posted[-1]
    assert c.tools_ok == {"vision": False}
    assert c._send_tools(_TOOLS, "code") is True  # other role unaffected


async def test_handle_400_strict_on_never_strips():
    c = LLMClient(base_url="http://x", api_key="k")
    c.tools_mode = "on"
    client = _StubClient(httpx.Response(200, json={"ok": 1}))
    payload = {"tools": _TOOLS, "tool_choice": "auto", "messages": []}
    resp, out = await c._handle_400(client, httpx.Response(400), payload, {}, "code")
    assert resp.status_code == 400  # propagates, no retry post
    assert client.posted == []
    assert out.get("tools") is not None
    assert c.tools_ok == {}  # nothing marked


async def test_reask_strips_tools():
    """A schema re-ask must not re-offer tools — the model could answer with
    another tool call instead of the required JSON."""

    class _Ans(BaseModel):
        ok: bool = True

    c = LLMClient(base_url="http://x", api_key="k")
    c.stream = False
    posted: list[dict] = []

    async def fake_post(payload: dict, role: str):
        posted.append(payload)
        bad = len(posted) == 1
        return {
            "choices": [
                {
                    "message": {"content": "not json at all" if bad else '{"ok": true}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

    c._post = fake_post  # type: ignore[method-assign]
    resp = await c.chat(
        "code",
        [{"role": "user", "content": "go"}],
        json_schema=_Ans,
        tools=_TOOLS,
    )
    assert resp.parsed is not None and resp.parsed.ok
    assert "tools" in posted[0]
    assert "tools" not in posted[1] and "tool_choice" not in posted[1]
