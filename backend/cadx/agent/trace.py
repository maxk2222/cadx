"""Agent trace sinks (in-memory for CLI/tests, DB for the worker)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace image data with placeholders so traces stay small and leak-free."""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if part.get("type") == "image_url":
                    new_content.append({"type": "image_url", "image_url": "<image>"})
                else:
                    new_content.append(part)
            out.append({**m, "content": new_content})
        else:
            out.append(m)
    return out


class MemoryTrace:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def record(
        self,
        *,
        step: str,
        iteration: int,
        model: str,
        messages: list[dict[str, Any]],
        response_text: str,
        parsed: dict[str, Any] | None,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> None:
        self.records.append(
            {
                "step": step,
                "iteration": iteration,
                "model": model,
                "messages": sanitize_messages(messages),
                "response": response_text[:20_000],
                "parsed": parsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
            }
        )


class JsonlTrace(MemoryTrace):
    """MemoryTrace that also appends every record to a JSONL file right away.

    The consolidated trace.json is written at run end — an OOM kill (SIGKILL)
    skips that dump entirely. Appending per record keeps a durable postmortem
    even when the process is killed mid-step.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    async def record(self, **kwargs: Any) -> None:
        await super().record(**kwargs)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self.records[-1], default=str) + "\n")
        except OSError:
            pass
