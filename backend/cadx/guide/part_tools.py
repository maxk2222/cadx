"""Parts-mission tools — register uploaded CAD assets, match them against
the manual's required parts, and bridge missing ones into the normal
sandboxed ``spawn_generate`` path (never a custom exec path).

KIARA reference: PartAnalyst + CAD-analysis sub-agent (v4 cad_tools.py /
part_tools.py) — adapted to CADX assets/jobs.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cadx.guide import cadmeta
from cadx.guide.state import GuidePart
from cadx.guide.tools import _b, _fn, _s
from cadx.logging import get_logger

if TYPE_CHECKING:
    from cadx.guide.session import GuideToolSession

log = get_logger(__name__)

_CAD_KINDS = {"reference_step", "reference_stl", "reference_dxf", "cad"}
_MAX_GENERATED = 8

SCHEMAS: dict[str, dict[str, Any]] = {
    "register_uploaded_part": _fn(
        "register_uploaded_part",
        "Register ONE uploaded CAD file as a guide part. Give it a short "
        "unique id (slug) and the display name. Ports/bbox are extracted "
        "deterministically (STEP via the executor, meshes locally).",
        {
            "part_id": _s("short unique slug, e.g. 'base_plate'"),
            "name": _s("display name — prefer the manual's term"),
            "asset_id": _s("asset_id from list_assets"),
        },
        ["part_id", "name", "asset_id"],
    ),
    "match_required_parts": _fn(
        "match_required_parts",
        "Compare the manual's required part names (detected_part_names + "
        "step required_parts) with registered parts. Returns per-name "
        "match status and a list of names with NO CAD part — the missing "
        "set you must resolve via generate_missing_part or ask_user.",
        {},
        [],
    ),
    "alias_part": _fn(
        "alias_part",
        "Map a manual part NAME (e.g. 'Gehäusedeckel') onto a registered "
        "part id when the names differ — resolves it in match_required_parts.",
        {"part_id": _s("registered part id"), "alias": _s("manual part name")},
        ["part_id", "alias"],
    ),
    "generate_missing_part": _fn(
        "generate_missing_part",
        "Generate a missing CAD part through the normal CADX pipeline "
        "(sandboxed build123d codegen — same path as 'generate this part' "
        "in chat). Include ALL geometric detail from the manual in the "
        "description. Blocks until the generation job finishes. "
        f"Hard cap: {_MAX_GENERATED} generated parts per guide.",
        {
            "part_id": _s("short unique slug for the new part"),
            "name": _s("display name"),
            "description": _s(
                "detailed geometric spec: shape, dimensions mm, holes, threads — "
                "everything the manual states plus user answers"
            ),
            "wait": _b("wait for the job to finish (default true)"),
        },
        ["part_id", "name", "description"],
    ),
}


def _slug(pid: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", pid.strip().lower()).strip("_")
    return s[:60]


async def register_uploaded_part(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pid = _slug(str(a.get("part_id", "")))
    if not pid:
        return {"ok": False, "error": "part_id must be a non-empty slug"}
    if s.state.part_by_id(pid) is not None:
        return {"ok": False, "error": f"part_id '{pid}' already registered"}
    aid = str(a.get("asset_id", ""))
    asset = next((x for x in s.guide_assets if str(x.get("id")) == aid), None)
    if asset is None:
        return {"ok": False, "error": f"asset '{aid}' not found — see list_assets"}
    path = Path(str(asset.get("path") or ""))
    if not path.exists():  # noqa: ASYNC240
        return {"ok": False, "error": f"asset file missing: {path}"}
    ext = path.suffix.lower()
    part = GuidePart(
        id=pid,
        name=str(a.get("name", "")).strip() or pid,
        source="uploaded",
        asset_id=aid,
        step_path=str(path) if ext in {".step", ".stp", ".brep"} else None,
        mesh_path=str(path) if ext not in {".step", ".stp", ".brep"} else None,
    )
    part = await cadmeta.ensure_metadata(part, s.executor)
    s.state.parts.append(part)
    return {
        "ok": part.status == "analyzed",
        "part_id": pid,
        "status": part.status,
        "ports": len(part.ports),
        "bbox": [part.bbox_min, part.bbox_max],
        "error": part.error,
        "note": "no ports found — mating will use bbox/free placement"
        if part.status == "analyzed" and not part.ports
        else "",
    }


def _required_names(s: GuideToolSession) -> list[str]:
    names: list[str] = []
    doc = s.state.document
    if doc is not None:
        names.extend(doc.detected_part_names)
    for st in s.state.steps:
        names.extend(st.required_parts)
    seen: set[str] = set()
    out = []
    for n in names:
        key = n.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(n.strip())
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", s.lower()).strip()


async def match_required_parts(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    required = _required_names(s)
    aliases = s._aliases_store
    registered = {p.id: p for p in s.state.parts}
    name_index = {_norm(p.name): p.id for p in s.state.parts}
    id_index = {_norm(p.id): p.id for p in s.state.parts}
    rows = []
    missing = []
    for name in required:
        norm = _norm(name)
        pid = aliases.get(norm) or name_index.get(norm) or id_index.get(norm)
        if pid is None:
            # substring fallback both ways
            for rp in registered.values():
                if norm and (norm in _norm(rp.name) or _norm(rp.name) in norm):
                    pid = rp.id
                    break
        rows.append(
            {"name": name, "matched_part_id": pid, "status": "matched" if pid else "missing"}
        )
        if pid is None:
            missing.append(name)
    return {
        "ok": True,
        "required_count": len(required),
        "matched": sum(1 for r in rows if r["matched_part_id"]),
        "missing": missing,
        "matches": rows,
        "note": "resolve every missing name via generate_missing_part or alias_part "
        "before done_mission — assembly needs a CAD source for each",
    }


async def alias_part(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    part = s.state.part_by_id(_slug(str(a.get("part_id", ""))))
    if part is None:
        return {"ok": False, "error": "part not registered"}
    alias = _norm(str(a.get("alias", "")))
    if not alias:
        return {"ok": False, "error": "alias required"}
    s._aliases_store[alias] = part.id
    return {"ok": True, "alias": alias, "part_id": part.id}


async def generate_missing_part(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pid = _slug(str(a.get("part_id", "")))
    if not pid:
        return {"ok": False, "error": "part_id must be a non-empty slug"}
    if s.state.part_by_id(pid) is not None:
        return {"ok": False, "error": f"part_id '{pid}' already registered"}
    generated = [p for p in s.state.parts if p.source == "generated"]
    if len(generated) >= _MAX_GENERATED:
        return {
            "ok": False,
            "error": f"generated-part cap reached ({_MAX_GENERATED}) — ask_user for "
            "alternatives or reuse an existing part",
        }
    desc = str(a.get("description", "")).strip()
    if len(desc) < 20:
        return {"ok": False, "error": "description too short — the generator needs real geometry"}

    from cadx.db import session_scope
    from cadx.models import Project
    from cadx.spawn import SpawnError, spawn_generate, wait_job_terminal

    name = str(a.get("name", "")).strip() or pid
    prompt = (
        f"{name}: {desc}\n\n"
        "Context: this part is generated for a step-by-step assembly guide. "
        "Model it exactly per the description, in millimetres, Z-up, resting "
        "on the XY plane. Keep interfaces (bores, threads, tabs) faithful — "
        "other parts must mate with them."
    )
    try:
        async with session_scope() as db:
            project = await db.get(Project, s.project_id)
            if project is None:
                return {"ok": False, "error": "guide project missing"}
            version, job = await spawn_generate(
                db,
                project,
                prompt,
                part_name=name,
                enqueue_fn=s._enqueue_fn,
            )
    except SpawnError as e:
        return {"ok": False, "error": str(e)}

    if not bool(a.get("wait", True)):
        return {
            "ok": True,
            "status": "queued",
            "job_id": str(job.id),
            "version_id": str(version.id),
            "note": "poll again or re-call with wait=true",
        }

    res = await wait_job_terminal(
        job.id,
        cancel_check=s._cancel_fn,
        timeout_s=float(getattr(s.ctx.runtime_settings, "chat_spawn_timeout_s", 1200)),
    )
    status = str(res.get("status", ""))
    if status == "waiting_for_input":
        # forward the child job's questions so the LLM can relay via ask_user
        return {
            "ok": False,
            "status": "waiting_for_input",
            "job_id": str(job.id),
            "version_id": str(version.id),
            "questions": res.get("questions") or [],
            "note": "the generator needs clarification — relay these via ask_user, "
            "then re-call generate_missing_part with the answers added to "
            "the description",
        }
    if status != "succeeded":
        return {
            "ok": False,
            "status": status,
            "error": res.get("error") or "generation failed",
            "job_id": str(job.id),
        }

    # locate the finished version's CAD source (STEP preferred, else STL/GLB)
    from cadx import storage
    from cadx.models import Version

    async with session_scope() as db:
        v = await db.get(Version, uuid.UUID(str(version.id)))
    if v is None:
        return {"ok": False, "error": "generated version vanished"}
    vdir = storage.version_dir(v.project_id, v.id)
    ed = storage.export_dir(v.project_id, v.id)
    step = next((p for p in (ed / "model.step", ed / "model.stp") if p.exists()), None)
    mesh = None
    for cand in (ed / "model.glb", ed / "model.stl", vdir / "model.glb"):
        if cand.exists():
            mesh = cand
            break
    if step is None and mesh is None:
        return {
            "ok": False,
            "error": "generation finished but no STEP/GLB export found — "
            "retry or ask_user for a different approach",
            "version_id": str(v.id),
        }
    part = GuidePart(
        id=pid,
        name=name,
        source="generated",
        version_id=str(v.id),
        step_path=str(step) if step else None,
        mesh_path=str(mesh) if mesh else None,
    )
    part = await cadmeta.ensure_metadata(part, s.executor)
    s.state.parts.append(part)
    return {
        "ok": part.status == "analyzed",
        "part_id": pid,
        "version_id": str(v.id),
        "status": part.status,
        "ports": len(part.ports),
        "bbox": [part.bbox_min, part.bbox_max],
    }


def _done_guard(s: GuideToolSession) -> dict[str, Any] | None:
    required = _required_names(s)
    if not required:
        return None
    aliases = s._aliases_store
    names = {_norm(p.name): p.id for p in s.state.parts}
    ids = {_norm(p.id): p.id for p in s.state.parts}
    missing = [
        n
        for n in required
        if aliases.get(_norm(n)) is None
        and names.get(_norm(n)) is None
        and ids.get(_norm(n)) is None
    ]
    if missing:
        return {
            "ok": False,
            "error": f"unmatched required parts: {', '.join(missing[:6])} — "
            "register/alias/generate them first",
        }
    not_analyzed = [p.id for p in s.state.parts if p.status == "error"]
    if not_analyzed:
        return {"ok": False, "error": f"parts failed analysis: {', '.join(not_analyzed)}"}
    return None


from cadx.guide.tools import DONE_GUARDS  # noqa: E402

DONE_GUARDS["parts"] = _done_guard

HANDLERS = {
    "register_uploaded_part": register_uploaded_part,
    "match_required_parts": match_required_parts,
    "alias_part": alias_part,
    "generate_missing_part": generate_missing_part,
}
