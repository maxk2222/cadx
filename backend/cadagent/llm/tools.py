"""Tool-call plumbing for OpenAI-compatible function calling.

Two layers live here:

- ``ToolCall`` + parsing of the provider response shape (``message.tool_calls``
  and streamed ``delta.tool_calls`` fragments).
- ``parse_content_tool_call``: some models/providers cannot emit real
  tool_calls and instead answer with a JSON object in ``content`` that looks
  like a tool call (``{"name": ..., "arguments": {...}}`` or
  ``{"action": ..., "args": {...}}``). In auto mode we accept that as a
  synthetic tool call so the session protocol still works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""  # original JSON text, for error reporting
    arguments_error: str = ""  # set when arguments failed to parse as an object
    via_content: bool = False  # synthesised from content JSON, not real tool_calls


def parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Extract tool calls from a completed (non-streamed) response message."""
    out: list[ToolCall] = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        out.append(_mk_call(tc.get("id") or f"call_{i}", fn.get("name") or "", fn.get("arguments")))
    return out


def _mk_call(call_id: str, name: str, raw: Any) -> ToolCall:
    args: dict[str, Any] = {}
    err = ""
    if isinstance(raw, dict):
        args = raw
        raw = json.dumps(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                args = parsed
            else:
                err = "arguments is not a JSON object"
        except json.JSONDecodeError as e:
            err = f"invalid JSON arguments: {e}"
    elif raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = raw or ""
    else:
        err = f"unsupported arguments type: {type(raw).__name__}"
        raw = str(raw)
    return ToolCall(
        id=str(call_id), name=name, arguments=args, raw_arguments=raw or "", arguments_error=err
    )


class _StreamToolAcc:
    """Accumulate streamed ``delta.tool_calls`` fragments by index."""

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, Any]] = {}

    def feed(self, fragments: list[dict[str, Any]]) -> None:
        for frag in fragments:
            idx = int(frag.get("index", 0))
            slot = self._by_index.setdefault(idx, {"id": "", "name": "", "args": ""})
            if frag.get("id"):
                slot["id"] += str(frag["id"])
            fn = frag.get("function") or {}
            if fn.get("name"):
                slot["name"] += str(fn["name"])
            if fn.get("arguments"):
                slot["args"] += str(fn["arguments"])

    def result(self) -> list[ToolCall]:
        return [
            _mk_call(v["id"] or f"call_{i}", v["name"], v["args"])
            for i, (_, v) in enumerate(sorted(self._by_index.items()))
        ]


_ARG_KEYS = ("arguments", "args", "parameters", "input")


def parse_content_tool_call(text: str) -> ToolCall | None:
    """Detect a tool call smuggled into plain content.

    Shapes accepted:
      {"name": "tool", "arguments": {...}}          — OpenAI-ish
      {"action": "tool", "args": {...}}             — common JSON-loop idiom
      {"tool_calls": [{"function": {...}}]}         — wrapped
      {"function": {"name": ..., "arguments": ...}} — bare function object

    Returns None when the content is not a single JSON object or lacks a
    recognisable tool name — callers then treat the text as a final answer.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    if isinstance(data.get("tool_calls"), list):
        calls = parse_tool_calls({"tool_calls": data["tool_calls"]})
        if calls:
            calls[0].via_content = True
        return calls[0] if calls else None
    fn = data.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        tc = _mk_call("call_0", fn["name"], fn.get("arguments"))
        tc.via_content = True
        return tc
    name = data.get("name") or data.get("action") or data.get("tool")
    if not isinstance(name, str) or not name:
        return None
    args: Any = None
    for key in _ARG_KEYS:
        if key in data:
            args = data[key]
            break
    if args is None:
        args = {}
    tc = _mk_call("call_0", name, args)
    tc.via_content = True
    return tc
