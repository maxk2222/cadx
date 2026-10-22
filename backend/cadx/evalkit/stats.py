"""Stats + variant helpers — ported from the old eval runner/compare.

Keep the record-level semantics identical so HTTP-mode AgentTrace rows and
local JsonlTrace records aggregate the same way.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cadx.evalkit.paths import DEFAULT_VARIANT

_VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_variant(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse a `--variant NAME=<json>` CLI value into (name, overrides).

    Overrides are validated against RuntimeSettings fields so a typo fails
    before a whole eval run starts.
    """
    from cadx.runtime_config import RuntimeSettings

    name, sep, raw = spec.partition("=")
    name = name.strip()
    if not sep or not raw.strip():
        raise ValueError(f"--variant expects NAME=<json-object>, got {spec!r}")
    if not _VARIANT_NAME.fullmatch(name):
        raise ValueError(f"--variant name {name!r} must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if name == DEFAULT_VARIANT:
        raise ValueError(f"--variant name {name!r} is reserved for un-named runs")
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--variant {name}: malformed JSON: {e}") from e
    if not isinstance(overrides, dict):
        raise ValueError(f"--variant {name}: overrides must be a JSON object")
    unknown = sorted(set(overrides) - set(RuntimeSettings.model_fields))
    if unknown:
        raise ValueError(f"--variant {name}: unknown RuntimeSettings keys: {', '.join(unknown)}")
    try:
        RuntimeSettings.model_validate(overrides)
    except ValueError as e:
        raise ValueError(f"--variant {name}: invalid values: {e}") from e
    return name, overrides


def step_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-step LLM stats out of trace records.

    `thinking_*` is the reasoning split: tokens spent on calls where the
    adaptive-thinking policy resolved to ON (`parsed.thinking` flag).
    """
    agg: dict[str, dict[str, int]] = {}
    for r in records:
        step = r.get("step")
        if not step or "latency_ms" not in r:
            continue  # pure event records (tool_call emits) carry no latency
        a = agg.setdefault(
            step,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "thinking_calls": 0,
                "thinking_prompt_tokens": 0,
                "thinking_completion_tokens": 0,
            },
        )
        a["calls"] += 1
        a["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
        a["completion_tokens"] += int(r.get("completion_tokens") or 0)
        a["latency_ms"] += int(r.get("latency_ms") or 0)
        if (r.get("parsed") or {}).get("thinking"):
            a["thinking_calls"] += 1
            a["thinking_prompt_tokens"] += int(r.get("prompt_tokens") or 0)
            a["thinking_completion_tokens"] += int(r.get("completion_tokens") or 0)
    return agg


def iterations(records: list[dict[str, Any]]) -> int:
    """Deepest iteration index seen in the trace (0 = nothing recorded)."""
    it = 0
    for r in records:
        i = r.get("iteration")
        if isinstance(i, int):
            it = max(it, i)
    return it


def step_totals(result: dict[str, Any]) -> dict[str, int]:
    """Sum per-step token stats — fallback when top-level token fields are
    missing (older summaries)."""
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thinking_calls": 0,
        "thinking_prompt_tokens": 0,
        "thinking_completion_tokens": 0,
    }
    for step in (result.get("step_stats") or {}).values():
        if not isinstance(step, dict):
            continue
        for k in totals:
            totals[k] += int(step.get(k) or 0)
    return totals


def safe_label(label: str) -> str:
    """Filesystem-safe run-dir label component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._") or "run"


def run_dir(root: Path, case: str, label: str, rep: int = 0) -> Path:
    """Per-run output dir: `<root>/<case>/<ts>_<label>[_r<rep>]/`."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{safe_label(label)}"
    if rep > 0:
        name += f"_r{rep + 1}"
    return root / case / name
