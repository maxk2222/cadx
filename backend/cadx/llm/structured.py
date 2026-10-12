"""Tolerant JSON extraction from LLM responses."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMFormatError(Exception):
    pass


def extract_json(text: str) -> dict:
    """Extract the first JSON object from arbitrary LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # drop first fence line and trailing fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMFormatError(f"no JSON object found in response ({len(text)} chars)")
    try:
        data: dict = json.loads(cleaned[start : end + 1])
        return data
    except json.JSONDecodeError as e:
        raise LLMFormatError(f"invalid JSON: {e}") from e


def parse_model(text: str, model: type[T]) -> T:
    data = extract_json(text)
    try:
        return model(**data)
    except ValidationError as e:
        raise LLMFormatError(f"schema validation failed: {e}") from e
