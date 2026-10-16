"""Scripted fake LLM for tests and `cadx run --fake`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cadx.llm.client import LLMResponse, Role
from cadx.llm.structured import parse_model
from cadx.llm.tools import _mk_call, parse_content_tool_call


class FakeLLM:
    """Returns queued responses. Each call pops the next item; the LAST item is
    repeated forever so tests don't need to pad their script.

    Script entries:
      "text"                                    — plain content answer
      {"content": "..."}                        — same, explicit
      {"tool_calls": [{"name": ..., "arguments": {...}}]} — a tool call
      {"content_tool_call": {"name": ..., "arguments": {...}}} — tool call
        smuggled into content JSON (legacy provider simulation)
    """

    def __init__(self, responses: list[Any]) -> None:
        self._script = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.models: dict[Role, str] = {"vision": "fake-vision", "code": "fake-code"}
        # scripts containing tool_calls entries drive sessions automatically;
        # plain-content scripts keep the legacy JSON-schema path.
        self.tools_mode = (
            "auto"
            if any(
                isinstance(r, dict) and (r.get("tool_calls") or r.get("content_tool_call"))
                for r in responses
            )
            else "off"
        )
        self.tools_ok: dict[str, bool] = {}  # per-role probe state, like LLMClient

    @classmethod
    def from_file(cls, path: Path) -> FakeLLM:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    async def chat(
        self,
        role: Role,
        messages: list[dict[str, Any]],
        *,
        json_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        on_delta: Callable[[str], None] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "tools": bool(tools),
                "tool_names": [t.get("function", {}).get("name") for t in (tools or [])],
                "extra_body": extra_body,
            }
        )
        self.total_usage["calls"] += 1
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        resp = self._respond(role, item, json_schema, tools)
        if on_delta is not None and resp.text:
            # deterministic "stream": emit cumulative snapshots in ~16-char
            # slices so tests can observe partial persistence
            for i in range(0, len(resp.text), 16):
                on_delta(resp.text[: i + 16])
        return resp

    def _respond(
        self,
        role: Role,
        item: Any,
        json_schema: type[BaseModel] | None,
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        if tools and isinstance(item, dict) and item.get("tool_calls"):
            calls = [
                _mk_call(tc.get("id") or f"call_{i}", tc.get("name", ""), tc.get("arguments"))
                for i, tc in enumerate(item["tool_calls"])
            ]
            return LLMResponse(
                text=item.get("content") or "",
                model=f"fake-{role}",
                tool_calls=calls,
                finish_reason="tool_calls",
            )
        if isinstance(item, dict):
            if item.get("content_tool_call"):
                text = json.dumps(item["content_tool_call"])
            elif "content" in item:
                text = str(item["content"])
            else:
                text = json.dumps(item)  # raw dict = JSON content (legacy scripts)
        else:
            text = str(item)
        # mimic the real client: a tool-call JSON in content becomes a
        # synthetic call when the caller offered matching tools
        if tools and text.strip().startswith("{"):
            syn = parse_content_tool_call(text)
            if syn and any(t.get("function", {}).get("name") == syn.name for t in tools):
                return LLMResponse(
                    text=text,
                    model=f"fake-{role}",
                    tool_calls=[syn],
                    finish_reason="tool_calls",
                )
        parsed = parse_model(text, json_schema) if json_schema else None
        return LLMResponse(text=text, parsed=parsed, model=f"fake-{role}")

    async def test(self) -> dict[str, Any]:
        return {"ok": True, "vision_ok": True, "latency_ms": 0}
