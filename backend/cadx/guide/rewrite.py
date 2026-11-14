"""Shared guide-step text rewrite — the audience/language variant pass.

Extracted from ``api/routers/guide.py::rewrite_guide`` so the REST
endpoint (post-hoc variants) and the guide runner (spawn-time
``state.audience`` rewrite before emit) share ONE implementation — the
variant contract (``_original`` snapshot, ``<audience>:<language>`` key,
apply-to-canonical-steps) must not diverge.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from cadx.agent.context import read_prompt
from cadx.guide.state import GuideState
from cadx.logging import get_logger

log = get_logger(__name__)


class _RewriteStep(BaseModel):
    number: int
    title: str = ""
    text: str
    display_text: str


class _RewriteOut(BaseModel):
    steps: list[_RewriteStep]


async def apply_rewrite(
    state: GuideState,
    *,
    audience: str,
    language: str,
    llm: Any,
) -> dict[str, Any]:
    """Rewrite ``state.steps`` for an audience/language, store the variant
    and apply it to the canonical steps.

    Returns ``{"applied": int, "key": "<audience>:<language>"}``. Raises
    on LLM/parse failure — callers decide whether that is a 502 (REST) or
    a shippable warning (runner pre-emit). Does NOT save the state or
    regenerate unity_guide.json — both callers persist right after.
    """
    payload = [
        {
            "number": s.number,
            "title": s.title,
            "text": s.text,
            "display_text": s.display_text,
        }
        for s in state.steps
    ]
    prompt = (
        read_prompt("guide_rewrite.md")
        .replace("{audience}", audience)
        .replace("{language}", language.strip() or "original")
    )
    res = await llm.chat(
        "code",
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        json_schema=_RewriteOut,
    )
    out = (
        _RewriteOut.model_validate(res.parsed)
        if res.parsed is not None
        else _RewriteOut.model_validate_json(res.text)
    )

    # snapshot the pre-rewrite text once — the variant picker must be
    # able to return to the untranslated original
    if "_original" not in state.variants:
        state.variants["_original"] = [
            {
                "number": s.number,
                "title": s.title,
                "text": s.text,
                "display_text": s.display_text,
            }
            for s in state.steps
        ]
    by_no = {s.number: s for s in out.steps}
    applied = 0
    variant_rows: list[dict] = []
    for s in state.steps:
        r = by_no.get(s.number)
        if r is None:
            continue
        if r.title.strip():
            s.title = r.title.strip()
        if r.text.strip():
            s.text = r.text.strip()
        if r.display_text.strip():
            s.display_text = r.display_text.strip()
        variant_rows.append(
            {
                "number": s.number,
                "title": s.title,
                "text": s.text,
                "display_text": s.display_text,
            }
        )
        applied += 1
    key = f"{audience}:{(language.strip() or 'original').lower()}"
    state.variants[key] = variant_rows
    return {"applied": applied, "key": key}
