"""Scripted fake LLM for tests and `cadagent run --fake`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cadagent.llm.client import LLMResponse, Role
from cadagent.llm.structured import parse_model


class FakeLLM:
    """Returns queued responses. Each call pops the next item; the LAST item is
    repeated forever so tests don't need to pad their script."""

    def __init__(self, responses: list[str | dict[str, Any]]) -> None:
        self._script = [json.dumps(r) if isinstance(r, dict) else str(r) for r in responses]
        self.calls: list[dict[str, Any]] = []
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.models: dict[Role, str] = {"vision": "fake-vision", "code": "fake-code"}

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
    ) -> LLMResponse:
        self.calls.append({"role": role, "messages": messages})
        self.total_usage["calls"] += 1
        text = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        parsed = parse_model(text, json_schema) if json_schema else None
        return LLMResponse(text=text, parsed=parsed, model=f"fake-{role}")

    async def test(self) -> dict[str, Any]:
        return {"ok": True, "vision_ok": True, "latency_ms": 0}
