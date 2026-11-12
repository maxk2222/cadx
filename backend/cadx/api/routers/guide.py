"""Guide-version endpoints — read/edit the finished guide's steps, media
and LLM text rewrites. Operates on the durable GuideState under
``version_dir/guide``; every mutation re-writes unity_guide.json so the
shipped artifact always mirrors the edited state.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from cadx import storage
from cadx.agent.context import read_prompt
from cadx.api.deps import get_session
from cadx.db.models import Version
from cadx.guide import unity
from cadx.guide.state import GuideState
from cadx.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["guide"])


def _guide_ctx(v: Version) -> tuple[GuideState, Path]:
    """Load the durable state for a guide version or 404."""
    if v.kind != "guide":
        raise HTTPException(404, "not a guide version")
    guide_dir = storage.version_dir(v.project_id, v.id) / "guide"
    state = GuideState.load(guide_dir)
    if state is None:
        raise HTTPException(404, "guide state not found (job still running?)")
    return state, guide_dir


def _step_by_no(state: GuideState, no: int):
    step = next((s for s in state.steps if s.number == no), None)
    if step is None:
        raise HTTPException(404, f"step {no} not found")
    return step


def _regen_unity(state: GuideState, guide_dir: Path) -> None:
    """Re-emit unity_guide.json at the path the registered artifact already
    points to — edits must flow into the shipped deliverable."""
    try:
        guide = unity.build_unity_guide(state)
        out = (
            Path(state.unity_guide_path)
            if state.unity_guide_path
            else (guide_dir / "output" / "unity_guide.json")
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(guide, indent=1), encoding="utf-8")
    except Exception as exc:
        # state.json stays authoritative — a unity rebuild hiccup must not
        # lose the edit
        log.warning("guide.unity_regen_failed", error=str(exc)[:200])


def _media_files(guide_dir: Path) -> list[str]:
    media = guide_dir / "output" / "media"
    if not media.is_dir():
        return []
    return sorted(p.name for p in media.iterdir() if p.is_file())


def _step_out(s: Any) -> dict[str, Any]:
    return {
        "number": s.number,
        "title": s.title,
        "text": s.text,
        "display_text": s.display_text,
        "action_type": s.action_type,
        "required_parts": s.required_parts,
        "safety_hint": s.safety_hint,
        "media_image": Path(s.media_image).name if s.media_image else None,
        "source_pages": s.source_pages,
        "sub_steps": [ss.model_dump() for ss in s.sub_steps],
    }


@router.get("/versions/{version_id}/guide")
async def get_guide(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    return {
        "steps": [_step_out(s) for s in state.steps],
        "media": _media_files(guide_dir),
        "master_notes": state.master_notes,
        "missions_done": state.missions_done,
        "review": {
            "score": state.review.score,
            "iteration": state.review.iteration,
            "findings": [f.model_dump() for f in state.review.findings],
        },
        "parts": [
            {"id": p.id, "name": p.name, "source": p.source, "status": p.status}
            for p in state.parts
        ],
        "instances": len(state.assembly.instances),
        "joints": len(state.assembly.joints),
    }


@router.get("/versions/{version_id}/guide/media/{name}")
async def guide_media(version_id: uuid.UUID, name: str, db: AsyncSession = Depends(get_session)):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    _, guide_dir = _guide_ctx(v)
    p = (guide_dir / "output" / "media" / name).resolve()
    media_root = (guide_dir / "output" / "media").resolve()
    if media_root not in p.parents or not p.is_file():
        raise HTTPException(404, "media not found")
    return FileResponse(str(p))


class StepPatch(BaseModel):
    title: str | None = None
    text: str | None = None
    display_text: str | None = None
    safety_hint: str | None = None
    media_image: str | None = None  # media filename, "" clears


@router.patch("/versions/{version_id}/guide/steps/{number}")
async def patch_step(
    version_id: uuid.UUID,
    number: int,
    body: StepPatch,
    db: AsyncSession = Depends(get_session),
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    step = _step_by_no(state, number)
    if body.title is not None:
        step.title = body.title
    if body.text is not None:
        step.text = body.text
    if body.display_text is not None:
        step.display_text = body.display_text
    if body.safety_hint is not None:
        step.safety_hint = body.safety_hint
    if body.media_image is not None:
        if body.media_image == "":
            step.media_image = None
        else:
            name = Path(body.media_image).name  # strip any path trickery
            if name not in _media_files(guide_dir):
                raise HTTPException(400, f"unknown media file '{name}'")
            step.media_image = str((guide_dir / "output" / "media" / name).resolve())
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return _step_out(step)


class RewriteIn(BaseModel):
    audience: str = Field(pattern="^(beginner|pro)$")


class _RewriteStep(BaseModel):
    number: int
    title: str = ""
    text: str
    display_text: str


class _RewriteOut(BaseModel):
    steps: list[_RewriteStep]


@router.post("/versions/{version_id}/guide/rewrite")
async def rewrite_guide(
    version_id: uuid.UUID,
    body: RewriteIn,
    db: AsyncSession = Depends(get_session),
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    if not state.steps:
        raise HTTPException(400, "guide has no steps to rewrite")

    from cadx.llm.client import make_llm
    from cadx.runtime_config import runtime_from_db_or_default

    runtime = await runtime_from_db_or_default()
    llm = make_llm(runtime)
    payload = [
        {
            "number": s.number,
            "title": s.title,
            "text": s.text,
            "display_text": s.display_text,
        }
        for s in state.steps
    ]
    prompt = read_prompt("guide_rewrite.md").replace("{audience}", body.audience)
    try:
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
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("guide.rewrite_failed", error=str(exc)[:300])
        raise HTTPException(502, f"rewrite failed: {str(exc)[:200]}") from exc

    by_no = {s.number: s for s in out.steps}
    applied = 0
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
        applied += 1
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {
        "ok": True,
        "audience": body.audience,
        "rewritten": applied,
        "steps": [_step_out(s) for s in state.steps],
    }
