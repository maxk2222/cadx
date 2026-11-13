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

import anyio
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
        "variants": state.variants,
        "annotations": state.annotations,
        "review": {
            "score": state.review.score,
            "iteration": state.review.iteration,
            "findings": [f.model_dump() for f in state.review.findings],
        },
        "parts": [
            {
                "id": p.id,
                "name": p.name,
                "source": p.source,
                "status": p.status,
                "has_mesh": bool(p.mesh_path),
            }
            for p in state.parts
        ],
        "instances": len(state.assembly.instances),
        "joints": len(state.assembly.joints),
        "instance_list": [
            {
                "id": i.instance_id or i.part_id,
                "part_id": i.part_id,
                "placed_in_step": i.placed_in_step,
                "transform": i.transform,
                "color": i.color,
                "opacity": i.opacity,
            }
            for i in state.assembly.instances
        ],
    }


@router.get("/versions/{version_id}/guide/parts/{part_id}/mesh")
async def guide_part_mesh(
    version_id: uuid.UUID, part_id: str, db: AsyncSession = Depends(get_session)
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    part = state.part_by_id(part_id)

    def _resolve() -> Path | None:
        p = Path(part.mesh_path).resolve() if part and part.mesh_path else None
        if p is None or guide_dir.resolve() not in p.parents or not p.is_file():
            return None
        return p

    p = await anyio.to_thread.run_sync(_resolve)
    if p is None:
        raise HTTPException(404, "part mesh not found")
    return FileResponse(str(p), media_type="model/gltf-binary")


class InstancePatch(BaseModel):
    transform: list[list[float]] | None = None
    placed_in_step: int | None = None
    color: str | None = None  # "" clears the override
    opacity: float | None = None


@router.patch("/versions/{version_id}/guide/instances/{iid}")
async def patch_instance(
    version_id: uuid.UUID,
    iid: str,
    body: InstancePatch,
    db: AsyncSession = Depends(get_session),
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    inst = next(
        (i for i in state.assembly.instances if (i.instance_id or i.part_id) == iid), None
    )
    if inst is None:
        raise HTTPException(404, "instance not found")
    if body.transform is not None:
        t = body.transform
        if (
            len(t) != 4
            or any(len(r) != 4 for r in t)
            or not all(isinstance(x, (int, float)) for r in t for x in r)
        ):
            raise HTTPException(400, "transform must be a 4x4 matrix")
        inst.transform = [[float(x) for x in r] for r in t]
    if body.placed_in_step is not None:
        if body.placed_in_step < 0 or (
            state.steps and body.placed_in_step > state.steps[-1].number
        ):
            raise HTTPException(400, "placed_in_step out of range")
        inst.placed_in_step = body.placed_in_step
    if body.color is not None:
        inst.color = body.color or None
    if body.opacity is not None:
        inst.opacity = min(1.0, max(0.05, body.opacity))
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    _regen_glb(state, guide_dir)
    return {
        "ok": True,
        "id": iid,
        "placed_in_step": inst.placed_in_step,
        "transform": inst.transform,
        "color": inst.color,
        "opacity": inst.opacity,
    }


def _world_bbox(local_min: list[float], local_max: list[float], t: list[list[float]]):
    """8-corner world bbox of a local-axis-aligned box under a 4x4."""
    import numpy as np

    m = np.asarray(t, dtype=float).reshape(4, 4)
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (local_min[0], local_max[0])
            for y in (local_min[1], local_max[1])
            for z in (local_min[2], local_max[2])
        ]
    )
    w = (m @ corners.T).T[:, :3]
    return w.min(axis=0), w.max(axis=0)


class InstanceCreate(BaseModel):
    part_id: str
    step_no: int = 0


@router.post("/versions/{version_id}/guide/instances")
async def add_instance(
    version_id: uuid.UUID,
    body: InstanceCreate,
    db: AsyncSession = Depends(get_session),
):
    """Insert a new occurrence of a registered part — the asset-library
    "drop" action. Placed beside the current union bbox (same free-slot
    rule the solver uses) so it never lands inside existing geometry."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    part = state.part_by_id(body.part_id)
    if part is None:
        raise HTTPException(404, f"part '{body.part_id}' not found")

    from cadx.guide.solver import solve_free_placement
    from cadx.guide.state import AssemblyInstance

    by_id = {p.id: p for p in state.parts}
    placed = []
    for i in state.assembly.instances:
        src = by_id.get(i.part_id)
        if src is None or not i.transform:
            continue
        mn, mx = _world_bbox(src.bbox_min, src.bbox_max, i.transform)
        placed.append({"bbox_min_world": mn.tolist(), "bbox_max_world": mx.tolist()})
    transform = solve_free_placement(part, placed, "")

    seq = [
        int((i.instance_id or "").rsplit("#", 1)[-1])
        for i in state.assembly.instances
        if i.part_id == part.id and "#" in (i.instance_id or "")
    ]
    iid = f"{part.id}#{(max(seq) + 1) if seq else 1}"
    inst = AssemblyInstance(
        part_id=part.id,
        instance_id=iid,
        transform=transform,
        placed_in_step=max(0, body.step_no),
    )
    state.assembly.instances.append(inst)
    if state.assembly.status == "empty":
        state.assembly.status = "partial"
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    _regen_glb(state, guide_dir)
    return {
        "ok": True,
        "id": iid,
        "placed_in_step": inst.placed_in_step,
        "transform": inst.transform,
    }


@router.delete("/versions/{version_id}/guide/instances/{iid}")
async def delete_instance(
    version_id: uuid.UUID,
    iid: str,
    db: AsyncSession = Depends(get_session),
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    asm = state.assembly
    before = len(asm.instances)
    asm.instances = [i for i in asm.instances if (i.instance_id or i.part_id) != iid]
    if len(asm.instances) == before:
        raise HTTPException(404, "instance not found")
    # drop joints bound to this exact instance; legacy part-level joints
    # stay — they remain valid while any instance of that part exists
    asm.joints = [
        j
        for j in asm.joints
        if iid not in (j.get("moving_instance"), j.get("fixed_instance"))
        and j.get("moving_part") != iid
    ]
    for ids in asm.step_instances.values():
        while iid in ids:
            ids.remove(iid)
    if not asm.instances:
        asm.status = "empty"
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    _regen_glb(state, guide_dir)
    return {"ok": True, "id": iid, "instances": len(asm.instances)}


def _regen_glb(state: Any, guide_dir: Path) -> None:
    """Rewrite the merged assembly GLB in place so the artifact file stays
    valid after instance edits."""
    from cadx.guide.scene import merged_glb

    by_id = {p.id: p for p in state.parts}
    placements = [
        (i.instance_id or i.part_id, by_id[i.part_id].mesh_path, i.transform)
        for i in state.assembly.instances
        if i.part_id in by_id and by_id[i.part_id].mesh_path
    ]
    try:
        merged_glb(placements, guide_dir / "output" / "assembly.glb")
    except Exception as exc:
        log.warning("guide.glb_regen_failed", error=str(exc)[:200])


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
    audience: str = Field(pattern="^(beginner|intermediate|pro)$")
    language: str = ""  # free-form target language; "" = keep original


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
    prompt = (
        read_prompt("guide_rewrite.md")
        .replace("{audience}", body.audience)
        .replace("{language}", body.language.strip() or "original")
    )
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
    key = f"{body.audience}:{(body.language.strip() or 'original').lower()}"
    state.variants[key] = variant_rows
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {
        "ok": True,
        "audience": body.audience,
        "variant": key,
        "rewritten": applied,
        "steps": [_step_out(s) for s in state.steps],
    }


class VariantIn(BaseModel):
    key: str  # e.g. "beginner:en" or "_original"


@router.post("/versions/{version_id}/guide/variant")
async def apply_variant(
    version_id: uuid.UUID,
    body: VariantIn,
    db: AsyncSession = Depends(get_session),
):
    """Re-apply a stored rewrite variant to the canonical steps — the
    shipped unity_guide.json mirrors whichever variant is active."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    rows = state.variants.get(body.key)
    if rows is None:
        raise HTTPException(404, f"unknown variant '{body.key}'")
    by_no = {int(r["number"]): r for r in rows}
    applied = 0
    for s in state.steps:
        r = by_no.get(s.number)
        if r is None:
            continue
        s.title = str(r.get("title") or s.title)
        s.text = str(r.get("text") or s.text)
        s.display_text = str(r.get("display_text") or s.display_text)
        applied += 1
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {"ok": True, "variant": body.key, "applied": applied,
            "steps": [_step_out(s) for s in state.steps]}
