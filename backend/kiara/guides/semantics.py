from __future__ import annotations

from typing import Any

from kiara.guides.schemas import StepSemanticSchema

ALLOWED_ACTION_TYPES = {"prepare", "position", "assemble", "tighten", "check", "test", "other"}


def normalize_step_semantic(raw_sem: dict[str, Any] | None) -> StepSemanticSchema:
    """
    Normalize LLM-provided semantic metadata into StepSemanticSchema.
    """
    if not raw_sem or not isinstance(raw_sem, dict):
        return StepSemanticSchema()

    action_type = str(raw_sem.get("actionType", "")).strip() or "other"
    if action_type not in ALLOWED_ACTION_TYPES:
        action_type = "other"

    focus_parts_raw = raw_sem.get("focusParts") or []
    if isinstance(focus_parts_raw, str):
        focus_parts = [focus_parts_raw.strip()] if focus_parts_raw.strip() else []
    elif isinstance(focus_parts_raw, list):
        focus_parts = [str(item).strip() for item in focus_parts_raw if str(item).strip()]
    else:
        focus_parts = []

    tools_raw = raw_sem.get("tools") or []
    if isinstance(tools_raw, str):
        tools = [tools_raw.strip()] if tools_raw.strip() else []
    elif isinstance(tools_raw, list):
        tools = [str(item).strip() for item in tools_raw if str(item).strip()]
    else:
        tools = []

    safety_hint = str(raw_sem.get("safetyHint", "")).strip()

    return StepSemanticSchema(
        actionType=action_type,
        focusParts=focus_parts,
        tools=tools,
        safetyHint=safety_hint,
    )
