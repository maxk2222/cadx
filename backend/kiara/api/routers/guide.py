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
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from kiara import storage
from kiara.api.deps import get_session
from kiara.db.models import Asset, AssetKind, Version
from kiara.guide import unity
from kiara.guide.annotate_tools import _DIR_VECS
from kiara.guide.state import GuideState
from kiara.guide.umlauts import restore_umlauts, state_umlaut_map
from kiara.logging import get_logger

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


def _step_out(s: Any, umap: dict[str, str] | None = None) -> dict[str, Any]:
    umap = umap or {}
    title = str(restore_umlauts(s.title or "", umap) or "")
    disp = str(restore_umlauts(s.display_text or "", umap) or "").strip()
    if disp and len(title.split()) < 3:
        # a bare verb ("Schrauben") is no step text — the full imperative
        # display_text is the better headline (mirrors step_tools fallback)
        title = disp
    return {
        "number": s.number,
        "title": title,
        "text": restore_umlauts(s.text, umap),
        "display_text": restore_umlauts(s.display_text, umap),
        "action_type": s.action_type,
        "required_parts": [restore_umlauts(str(r), umap) for r in (s.required_parts or [])],
        "safety_hint": restore_umlauts(s.safety_hint, umap),
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
    umap = state_umlaut_map(state)
    return {
        "steps": [_step_out(s, umap) for s in state.steps],
        "media": _media_files(guide_dir),
        "master_notes": state.master_notes,
        "missions_done": state.missions_done,
        # frame the instance transforms are expressed in — CAD route is
        # Z-up mm, blender mode is glTF Y-up. Three.js viewers rotate the
        # Z-up world into the display frame; see GuideScene's wrapper.
        "world_frame": state.world_frame,
        "variants": state.variants,
        "annotations": [
            {
                **a,
                "text": restore_umlauts(a.get("text"), umap),
                "label": restore_umlauts(a.get("label"), umap),
            }
            for a in state.annotations
        ],
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
        if p is None or not p.is_file():
            return None
        # durable location is guide_dir/parts — but runs emitted before the
        # copy-in fix recorded generated meshes under their source version's
        # exports dir; both live inside the data root, which is the real
        # trust boundary (state-recorded paths, files we wrote)
        if p.suffix.lower() not in (".glb", ".gltf", ".stl", ".step", ".stp"):
            return None
        root = storage.data_root().resolve()
        if guide_dir.resolve() not in p.parents and root not in p.parents:
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
    from kiara.guide.solver import world_bbox

    return world_bbox(local_min, local_max, t)


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

    from kiara.guide.solver import solve_free_placement
    from kiara.guide.state import AssemblyInstance

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
    from kiara.guide.scene import merged_glb

    by_id = {p.id: p for p in state.parts}
    placements = [
        (i.instance_id or i.part_id, by_id[i.part_id].mesh_path, i.transform)
        for i in state.assembly.instances
        if i.part_id in by_id and by_id[i.part_id].mesh_path
    ]
    try:
        # same frame contract as the initial emit — blender-mode
        # placements are already glTF Y-up and must pass through
        merged_glb(placements, guide_dir / "output" / "assembly.glb", world_frame=state.world_frame)
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
    return _step_out(step, state_umlaut_map(state))


class StepCreate(BaseModel):
    insert_at: int | None = None  # step number to insert BEFORE; None = append
    title: str = ""
    text: str = ""
    display_text: str = ""


def _renumber(state: GuideState, delta: int, from_no: int) -> None:
    """Shift step numbers >= from_no by delta — plus every reference that
    addresses a step (instances, step_instances map, annotations)."""
    for s in state.steps:
        if s.number >= from_no:
            s.number += delta
    for i in state.assembly.instances:
        if i.placed_in_step >= from_no:
            i.placed_in_step += delta
    state.assembly.step_instances = {
        str(int(k) + delta if int(k) >= from_no else int(k)): v
        for k, v in state.assembly.step_instances.items()
    }
    for an in state.annotations:
        if int(an.get("step", 0)) >= from_no:
            an["step"] = int(an["step"]) + delta


@router.post("/versions/{version_id}/guide/steps")
async def add_step(
    version_id: uuid.UUID,
    body: StepCreate,
    db: AsyncSession = Depends(get_session),
):
    from kiara.guide.state import GuideStep

    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    if body.insert_at is None:
        no = max((s.number for s in state.steps), default=0) + 1
    else:
        no = body.insert_at
        if no < 1 or no > max((s.number for s in state.steps), default=0) + 1:
            raise HTTPException(400, f"insert_at must be 1..{len(state.steps) + 1}")
        _renumber(state, +1, no)
    state.steps.append(
        GuideStep(
            number=no,
            title=body.title or f"Schritt {no}",
            text=body.text,
            display_text=body.display_text,
        )
    )
    state.steps.sort(key=lambda s: s.number)
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {"ok": True, "step": _step_out(state.step_by_number(no), state_umlaut_map(state))}


@router.delete("/versions/{version_id}/guide/steps/{number}")
async def delete_step(
    version_id: uuid.UUID,
    number: int,
    db: AsyncSession = Depends(get_session),
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    step = _step_by_no(state, number)
    state.steps.remove(step)
    # parts first placed by the deleted step stay visible from the step
    # before — deleting must not orphan placed instances
    for i in state.assembly.instances:
        if i.placed_in_step == number:
            i.placed_in_step = max(1, number - 1)
    state.assembly.step_instances.pop(str(number), None)
    for an in state.annotations:
        if int(an.get("step", 0)) == number:
            an["step"] = max(1, number - 1)
    _renumber(state, -1, number + 1)
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {"ok": True, "deleted": number, "steps": len(state.steps)}


class RewriteIn(BaseModel):
    audience: str = Field(pattern="^(beginner|intermediate|pro)$")
    language: str = ""  # free-form target language; "" = keep original


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

    from kiara.guide.rewrite import apply_rewrite
    from kiara.llm.client import make_llm
    from kiara.runtime_config import runtime_from_db_or_default

    runtime = await runtime_from_db_or_default()
    llm = make_llm(runtime)
    try:
        out = await apply_rewrite(
            state, audience=body.audience, language=body.language, llm=llm
        )
    except Exception as exc:
        log.warning("guide.rewrite_failed", error=str(exc)[:300])
        raise HTTPException(502, f"rewrite failed: {str(exc)[:200]}") from exc

    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {
        "ok": True,
        "audience": body.audience,
        "variant": out["key"],
        "rewritten": out["applied"],
        "steps": [_step_out(s, state_umlaut_map(state)) for s in state.steps],
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
            "steps": [_step_out(s, state_umlaut_map(state)) for s in state.steps]}


class GuideNoteIn(BaseModel):
    note: str = Field(min_length=1, max_length=2000)


@router.post("/versions/{version_id}/guide/notes")
async def add_guide_note(
    version_id: uuid.UUID,
    body: GuideNoteIn,
    db: AsyncSession = Depends(get_session),
):
    """Append a free-form user note to the guide's ``user_notes.jsonl`` —
    the exact line format ``attach_guide_note`` writes from chat, so a
    running guide picks it up via ``list_user_notes`` mid-mission. Unlike
    the read endpoints this works before state.json exists (early-run
    notes still land in the same durable directory)."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    if v.kind != "guide":
        raise HTTPException(404, "not a guide version")
    note = body.note.strip()
    if not note:
        raise HTTPException(422, "note required")
    guide_dir = storage.version_dir(v.project_id, v.id) / "guide"
    guide_dir.mkdir(parents=True, exist_ok=True)
    p = guide_dir / "user_notes.jsonl"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"note": note[:2000]}) + "\n")
    return {"ok": True}


# ---------------------------------------------------------------------------
# annotations (arrows + free text fields)
# ---------------------------------------------------------------------------

_IMAGE_KINDS = {AssetKind.image, AssetKind.generated_image}
_ANN_KINDS = {"arrow", "text"}


class AnnotationIn(BaseModel):
    kind: str = Field(pattern="^(arrow|text)$")
    step: int = Field(ge=1)
    part_id: str = ""  # arrow anchor — instance id or part id
    approach_axis: list[float] | None = None  # arrow direction (vec3)
    direction: str = ""  # named axis instead ("up" … _DIR_VECS)
    label: str = Field(default="", max_length=500)
    position: list[float] | None = None  # text field world pos (vec3)
    text: str = Field(default="", max_length=500)
    color: str = Field(default="", max_length=32)
    size: float = Field(default=1.0, gt=0, le=10)


class AnnotationPatch(BaseModel):
    step: int | None = Field(default=None, ge=1)
    part_id: str | None = None
    approach_axis: list[float] | None = None
    direction: str | None = None
    label: str | None = Field(default=None, max_length=500)
    position: list[float] | None = None
    text: str | None = Field(default=None, max_length=500)
    color: str | None = Field(default=None, max_length=32)
    size: float | None = Field(default=None, gt=0, le=10)


def _norm_axis(v: list[float] | None) -> list[float] | None:
    if not v or len(v) != 3:
        return None
    n = sum(c * c for c in v) ** 0.5
    return [round(c / n, 6) for c in v] if n > 1e-9 else None


def _resolve_axis(axis: list[float] | None, direction: str | None) -> list[float] | None:
    """explicit vec3 wins; else look the named direction up in _DIR_VECS."""
    v = _norm_axis(axis)
    if v is not None:
        return v
    if direction:
        v = _DIR_VECS.get(direction.strip().lower())
        return list(v) if v else None
    return None


def _ann_checks(body_step: int | None, state: GuideState) -> None:
    if body_step is not None and not any(s.number == body_step for s in state.steps):
        raise HTTPException(400, f"step {body_step} does not exist")


def _ann_at(state: GuideState, idx: int) -> dict:
    if idx < 0 or idx >= len(state.annotations):
        raise HTTPException(404, f"annotation {idx} not found")
    return state.annotations[idx]


@router.post("/versions/{version_id}/guide/annotations")
async def add_annotation(
    version_id: uuid.UUID, body: AnnotationIn, db: AsyncSession = Depends(get_session)
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    _ann_checks(body.step, state)
    ann: dict[str, Any] = {"kind": body.kind, "step": body.step}
    if body.kind == "arrow":
        axis = _resolve_axis(body.approach_axis, body.direction) or [0.0, 1.0, 0.0]
        ann.update(
            part_id=body.part_id,
            approach_axis=axis,
            label=body.label,
        )
    else:
        ann.update(
            position=body.position if body.position and len(body.position) == 3 else [0, 0, 0],
            text=body.text,
        )
    if body.color:
        ann["color"] = body.color
    if body.size != 1.0:
        ann["size"] = body.size
    state.annotations.append(ann)
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {"ok": True, "index": len(state.annotations) - 1}


@router.patch("/versions/{version_id}/guide/annotations/{idx}")
async def patch_annotation(
    version_id: uuid.UUID, idx: int, body: AnnotationPatch, db: AsyncSession = Depends(get_session)
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    ann = _ann_at(state, idx)
    if body.step is not None:
        _ann_checks(body.step, state)
        ann["step"] = body.step
    if body.part_id is not None and ann.get("kind") == "arrow":
        ann["part_id"] = body.part_id
    axis = _resolve_axis(body.approach_axis, body.direction)
    if axis is not None and ann.get("kind") == "arrow":
        ann["approach_axis"] = axis
    if body.label is not None and ann.get("kind") == "arrow":
        ann["label"] = body.label
    if body.position is not None and len(body.position) == 3 and ann.get("kind") == "text":
        ann["position"] = list(body.position)
    if body.text is not None and ann.get("kind") == "text":
        ann["text"] = body.text
    if body.color is not None:
        if body.color:
            ann["color"] = body.color
        else:
            ann.pop("color", None)
    if body.size is not None:
        if body.size != 1.0:
            ann["size"] = body.size
        else:
            ann.pop("size", None)
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {"ok": True, "index": idx, "annotation": ann}


@router.delete("/versions/{version_id}/guide/annotations/{idx}")
async def delete_annotation(
    version_id: uuid.UUID, idx: int, db: AsyncSession = Depends(get_session)
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    state, guide_dir = _guide_ctx(v)
    _ann_at(state, idx)
    state.annotations.pop(idx)
    state.save(guide_dir)
    _regen_unity(state, guide_dir)
    return {"ok": True, "annotations": len(state.annotations)}


# ---------------------------------------------------------------------------
# guide media — step photos must live in guide/output/media (patch_step
# validates the filename against that directory)
# ---------------------------------------------------------------------------

_MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_MEDIA_MAX = 10 * 1024 * 1024


def _media_dir(guide_dir: Path) -> Path:
    d = guide_dir / "output" / "media"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unique_media_name(media_dir: Path, raw: str) -> str:
    name = storage.safe_filename(raw) or "image.png"
    stem, dot, ext = name.rpartition(".")
    stem, ext = (stem or "image"), ("." + ext if dot else ".png")
    cand = f"{stem}{ext}"
    i = 2
    while (media_dir / cand).exists():
        cand = f"{stem}-{i}{ext}"
        i += 1
    return cand


@router.post("/versions/{version_id}/guide/media")
async def upload_guide_media(
    version_id: uuid.UUID, file: UploadFile, db: AsyncSession = Depends(get_session)
):
    """Direct media upload (viewport snapshot, user photo). The file lands
    in guide/output/media so patch_step can reference it as media_image."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    _state, guide_dir = _guide_ctx(v)
    name_in = Path(file.filename or "image.png").name
    ext = Path(name_in).suffix.lower()
    if ext not in _MEDIA_EXTS:
        raise HTTPException(400, f"unsupported media type '{ext or name_in}'")
    if file.size is not None and file.size > _MEDIA_MAX:
        raise HTTPException(413, "media file too large (max 10 MB)")
    data = await file.read(_MEDIA_MAX + 1)
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > _MEDIA_MAX:
        raise HTTPException(413, "media file too large (max 10 MB)")
    media_dir = _media_dir(guide_dir)
    name = _unique_media_name(media_dir, name_in)
    (media_dir / name).write_bytes(data)
    return {"ok": True, "name": name}


class MediaFromAsset(BaseModel):
    asset_id: uuid.UUID


@router.post("/versions/{version_id}/guide/media/from-asset")
async def guide_media_from_asset(
    version_id: uuid.UUID, body: MediaFromAsset, db: AsyncSession = Depends(get_session)
):
    """Copy a project image asset into the guide media directory."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    _state, guide_dir = _guide_ctx(v)
    asset = await db.get(Asset, body.asset_id)
    if not asset or asset.project_id != v.project_id:
        raise HTTPException(404, "asset not found")
    if asset.kind not in _IMAGE_KINDS:
        raise HTTPException(400, "asset is not an image")
    src = storage.asset_path(asset.project_id, asset.id, asset.filename)
    if not src.is_file():
        raise HTTPException(404, "asset file missing")
    if src.stat().st_size > _MEDIA_MAX:
        raise HTTPException(413, "media file too large (max 10 MB)")
    media_dir = _media_dir(guide_dir)
    name = _unique_media_name(media_dir, asset.filename)
    media_dir.joinpath(name).write_bytes(src.read_bytes())
    return {"ok": True, "name": name}
