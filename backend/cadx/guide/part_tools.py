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
from cadx.guide.assembly_tools import _norm2, _tokset
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
        "unique id (slug), the display name, and the parts-list quantity "
        "(Stückliste 'Menge' — total copies the kit ships, premounted "
        "included) so the plan can be checked against it. Ports/bbox are "
        "extracted deterministically (STEP via the executor, meshes "
        "locally). Re-registering the same part_id+asset updates the "
        "fields instead of failing.",
        {
            "part_id": _s("short unique slug, e.g. 'base_plate'"),
            "name": _s("display name — prefer the manual's term"),
            "asset_id": _s("asset UUID or uploaded filename from list_assets"),
            "quantity": {
                "type": "integer",
                "description": "copies in the parts list (Menge), e.g. 2 — omit if the manual gives none",
            },
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
        "in chat). Describe the SIMPLE functional form: primary shape, "
        "mm dimensions, mating interfaces (bores/threads/sockets) — never "
        "decorative detail (no crimp rings, knurls, logos, cosmetic "
        "fillets). Blocks until the generation job finishes. "
        f"Hard cap: {_MAX_GENERATED} generated parts per guide.",
        {
            "part_id": _s("short unique slug for the new part"),
            "name": _s("display name"),
            "description": _s(
                "detailed geometric spec: shape, dimensions mm, holes, threads — "
                "everything the manual states plus user answers"
            ),
            "quantity": {
                "type": "integer",
                "description": "copies in the parts list (Menge) — omit if the manual gives none",
            },
            "force": _b(
                "set true ONLY when match_required_parts/alias cannot map any "
                "registered part to this manual name — generation is slow"
            ),
            "wait": _b("wait for the job to finish (default true)"),
        },
        ["part_id", "name", "description"],
    ),
    "mark_context_part": _fn(
        "mark_context_part",
        "Mark a required manual name as NOT a kit part — host equipment "
        "(Wärmepumpe, wall, floor), tools, consumables, or a figure-only "
        "size reference (e.g. 'Rohr DN22 l=200mm' that is only a dimension "
        "callout in the drawing). The name stops counting as unmatched. "
        "Placement substeps still can't target it — those need a steps fix.",
        {"name": _s("manual name to mark as context"),
         "reason": _s("why it is not a kit part")},
        ["name"],
    ),
}


def _slug(pid: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", pid.strip().lower()).strip("_")
    return s[:60]


def _resolve_asset(s: GuideToolSession, ref: str, *, part_id: str, name: str) -> Any | None:
    """Forgiving asset lookup: exact asset_id → basename/filename →
    unambiguous filename token overlap. Hallucinated/stale UUIDs are a
    common LLM failure — the part's own name+slug shares tokens with the
    real filename ("adapter_rn22_ig_plain" → "Adapter_rn22_internal_…
    .glb" shares adapter/rn22/plain); ties stay an error."""
    asset = next((x for x in s.guide_assets if str(x.get("id")) == ref), None)
    if asset is None and ref:
        # models know upload FILE names, not UUIDs — a hallucinated id is
        # often a real filename; accept basename/filename as the reference
        norm = ref.lower().rsplit("/", 1)[-1]
        asset = next(
            (
                x
                for x in s.guide_assets
                if norm
                and norm
                in {
                    str(x.get("filename") or "").lower(),
                    Path(str(x.get("path") or "")).name.lower(),
                }
            ),
            None,
        )
    if asset is None:
        qt = _tokset(f"{ref} {part_id} {name}")
        scored: list[tuple[int, Any]] = []
        for x in s.guide_assets:
            if Path(str(x.get("path") or "")).suffix.lower() not in (
                cadmeta._BREP_EXTS | cadmeta._MESH_EXTS
            ):
                continue
            ft = _tokset(Path(str(x.get("filename") or x.get("path") or "")).stem)
            ov = len(qt & ft)
            if ov:
                scored.append((ov, x))
        scored.sort(key=lambda t: -t[0])
        if scored and scored[0][0] >= 2 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            asset = scored[0][1]
    return asset


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _guide_mesh(s: GuideToolSession, src: Path) -> str:
    """Mesh path that stays durable + servable: glTF gets the mm-normalized
    copy first (uploads are meter-scale per spec); whatever normalize returns
    must live under the guide parts dir — a src outside it (generated-part
    exports, already-mm uploads) would 404 the /guide/parts/{id}/mesh guard
    and detach the scene evidence from the guide dir."""
    parts_dir = s.state_dir / "parts"
    norm = Path(cadmeta.normalize_mesh_units(src, parts_dir)).resolve()
    try:
        norm.relative_to(parts_dir.resolve())
        return str(norm)
    except ValueError:
        import shutil

        parts_dir.mkdir(parents=True, exist_ok=True)
        dst = parts_dir / src.name
        shutil.copyfile(src, dst)
        return str(dst)


async def register_uploaded_part(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pid = _slug(str(a.get("part_id", "")))
    if not pid:
        return {"ok": False, "error": "part_id must be a non-empty slug"}
    quantity = _int_or_none(a.get("quantity"))
    aid = str(a.get("asset_id", ""))
    asset = _resolve_asset(
        s, aid, part_id=str(a.get("part_id") or ""), name=str(a.get("name") or "")
    )
    if asset is not None:
        aid = str(asset.get("id"))
    if asset is None:
        avail = [
            str(x.get("filename") or x.get("id"))
            for x in s.guide_assets
            if Path(str(x.get("path") or "")).suffix.lower()
            in cadmeta._BREP_EXTS | cadmeta._MESH_EXTS
        ]
        return {
            "ok": False,
            "error": f"asset '{aid or '<missing>'}' not found — pass the "
            f"filename or asset_id from list_assets. available={avail[:20]}",
        }
    path = Path(str(asset.get("path") or ""))
    existing = next(
        (p for p in s.state.parts if p.asset_id and p.asset_id == aid), None
    )
    if existing is not None:
        if existing.id == pid:
            # idempotent upsert — re-registering the same part_id+asset
            # refreshes name/quantity instead of erroring
            existing.name = str(a.get("name", "")).strip() or existing.name
            if quantity is not None:
                existing.quantity = quantity
            return {
                "ok": True,
                "part_id": existing.id,
                "updated": True,
                "quantity": existing.quantity,
            }
        # the agent re-registered the same FILE under a new id — the
        # semantic intent is "make this name resolve", which is exactly an
        # alias; fulfilling it beats a dead-end error the model retries
        # verbatim until error_streak (observed: parts mission death). The
        # auto-alias goes through the same size/category sanity as
        # alias_part — 'Rohr l=500' must never resolve to the l=70 part.
        new_name = str(a.get("name", "")).strip()
        made: list[str] = []
        blocked: list[str] = []
        for alias in {new_name, pid} - {existing.name, existing.id, ""}:
            na = _norm(alias)
            if not na:
                continue
            if _name_part_conflict(alias, existing.name, existing.id):
                blocked.append(alias)
                continue
            s._aliases_store[na] = existing.id
            made.append(alias)
        if quantity is not None:
            existing.quantity = quantity
        return {
            "ok": True,
            "part_id": existing.id,
            "aliased_to": existing.id,
            "aliases_added": made,
            "aliases_blocked": blocked,
            "note": f"asset was already registered as '{existing.id}' — "
            f"mapped {made or ['<none>']} onto it"
            + (
                f"; refused {blocked} (size/category contradicts "
                f"'{existing.name}') — if the manual really means that "
                "name, it belongs to a different part"
                if blocked
                else ""
            )
            + f"; use part_id '{existing.id}' from here",
        }
    # part_id uniqueness is a hard invariant — a second asset under an
    # existing id orphans it (part_by_id finds only the first). Observed:
    # 'rohr_70mm' registered to an ELBOW file first, then the real
    # Pipe_l70.glb under the same id → the true mesh became unreachable.
    taken = s.state.part_by_id(pid)
    if taken is not None:
        return {
            "ok": False,
            "error": f"part_id '{pid}' is already registered to a different "
            f"asset ('{taken.name}') — if the manual uses two names for the "
            f"same physical part, call alias_part('{pid}', '<other name>'); "
            "if this is a genuinely different part, pick a distinct part_id",
        }
    if not path.exists():  # noqa: ASYNC240
        return {"ok": False, "error": f"asset file missing: {path}"}
    ext = path.suffix.lower()
    if ext not in cadmeta._BREP_EXTS and ext not in cadmeta._MESH_EXTS:
        return {
            "ok": False,
            "error": f"asset '{asset.get('filename')}' is not a CAD file "
            f"(.{ext.lstrip('.')}) — pick a STEP/GLB/STL asset",
        }
    mesh_path: str | None = None
    if ext in cadmeta._MESH_EXTS:
        mesh_path = _guide_mesh(s, path)
    part = GuidePart(
        id=pid,
        name=str(a.get("name", "")).strip() or pid,
        source="uploaded",
        asset_id=aid,
        step_path=str(path) if ext in cadmeta._BREP_EXTS else None,
        mesh_path=mesh_path,
        quantity=quantity,
    )
    part = await cadmeta.ensure_metadata(part, s.executor)
    s.state.parts.append(part)
    # name↔file category sanity: 'Rohr l=70' bound to 'Elbow_…_socket.glb'
    # is almost always a wrong-asset pick (observed: an elbow file
    # registered under the pipe's name, orphaning the real pipe). Advisory
    # only — arbitrary part names are legitimate.
    ncat, _ = _sig(str(part.name))
    fcat, _ = _sig(path.stem)
    warn = (
        f"WARNING: name '{part.name}' looks like a {ncat} but file "
        f"'{path.name}' looks like a {fcat} — double-check this is the "
        "right asset; if not, register the correct one under a distinct id"
        if ncat != "other" and fcat != "other" and ncat != fcat
        else ""
    )
    return {
        "ok": part.status == "analyzed",
        "part_id": pid,
        "status": part.status,
        "ports": len(part.ports),
        "bbox": [part.bbox_min, part.bbox_max],
        "error": part.error,
        "note": (warn + " | " if warn else "")
        + (
            "no ports found — mating will use bbox/free placement"
            if part.status == "analyzed" and not part.ports
            else ""
        ),
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
    return _norm2(s)


_CAT_TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("elbow", re.compile(r"rohrbogen|elbow|\bbogen\b")),
    ("pipe", re.compile(r"\brohr\b|\bpipe\b")),
    ("adapter", re.compile(r"adapter")),
    ("casing", re.compile(r"geh[aä]use|casing|housing")),
    ("coupling", re.compile(r"\bmuffe\b|coupling|kupplung")),
]


def _sig(s: str) -> tuple[str, frozenset[str]]:
    n = _norm(s)
    cat = next((c for c, rx in _CAT_TERMS if rx.search(n)), "other")
    toks = {f"dn{m}" for m in re.findall(r"(?:dn|rn)\s*(\d+)", n)}
    toks |= {f"l{m}" for m in re.findall(r"\bl\s*(\d+)", n)}
    return cat, frozenset(toks)


def _suggest_parts(s: GuideToolSession, name: str) -> list[str]:
    """Deterministic candidate ids for a required name: same category and no
    conflicting size tokens (manuals rename fittings between the parts list
    and the step text — end-type words like aussen/innen vs spigot/socket are
    left to the LLM's `alias_part` decision)."""
    cat, toks = _sig(name)
    out = []
    for p in s.state.parts:
        pcat, ptoks = _sig(p.name + " " + (p.mesh_path or "") + " " + (p.step_path or ""))
        if pcat != cat:
            continue
        if toks and ptoks and not (toks & ptoks):
            continue
        out.append(p.id)
    return out


def _is_context(s: GuideToolSession, name: str) -> bool:
    n = _norm(name)
    return bool(n) and n in {_norm(x) for x in s.state.context_names}


async def mark_context_part(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    name = str(a.get("name", "")).strip()
    if not name:
        return {"ok": False, "error": "name required"}
    n = _norm(name)
    if n not in {_norm(x) for x in s.state.context_names}:
        s.state.context_names.append(name)
    return {
        "ok": True,
        "context_names": list(s.state.context_names),
        "note": f"'{name}' marked as context — no part needs registering. "
        "Any placement substep still referencing it must be corrected by "
        "the steps mission (plan lint will route it).",
    }


def _resolve_required(s: GuideToolSession, name: str) -> str | None:
    """Single resolution chain for a required manual name → part id:
    explicit alias → part name → part id → substring containment both ways.
    match_required_parts and the done-guard MUST agree on what "matched"
    means — divergent rules trap the agent in an alias/done reject loop."""
    norm = _norm(name)
    aliases = s._aliases_store
    pid = (
        aliases.get(norm)
        or {_norm(p.name): p.id for p in s.state.parts}.get(norm)
        or {_norm(p.id): p.id for p in s.state.parts}.get(norm)
    )
    if pid is None and norm:
        for rp in s.state.parts:
            if norm in _norm(rp.name) or _norm(rp.name) in norm:
                return rp.id
    return pid


async def match_required_parts(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    required = _required_names(s)
    rows = []
    missing = []
    for name in required:
        pid = _resolve_required(s, name)
        sugg = [] if pid else _suggest_parts(s, name)
        rows.append(
            {
                "name": name,
                "matched_part_id": pid,
                "status": "matched"
                if pid
                else ("context" if _is_context(s, name) else "missing"),
                **({"suggested_part_ids": sugg} if sugg and not pid else {}),
            }
        )
        if pid is None and not _is_context(s, name):
            missing.append(name)
    return {
        "ok": True,
        "required_count": len(required),
        "matched": sum(1 for r in rows if r["matched_part_id"]),
        "missing": missing,
        "matches": rows,
        "note": "resolve every missing name via alias_part onto a suggested "
        "candidate (or generate_missing_part when generation is allowed) "
        "before done_mission — assembly needs a CAD source for each",
    }


def _name_part_conflict(alias: str, part_name: str, part_id: str) -> str | None:
    """An alias must rename the SAME physical part: 'Rohr DN22 l=500mm' on
    the 70mm part corrupts every downstream resolution (targets,
    requirements, lint). Blocks on contradictory size markers present on
    both sides, and on specific-vs-specific category clashes (an elbow
    name onto a pipe part is nearly always a wrong mapping)."""
    acat, atoks = _sig(alias)
    pcat, ptoks = _sig(f"{part_name} {part_id}")
    for fam in ("dn", "l"):
        a_vals = {t for t in atoks if t.startswith(fam)}
        p_vals = {t for t in ptoks if t.startswith(fam)}
        if a_vals and p_vals and a_vals != p_vals:
            return (
                f"carries different size markers {sorted(a_vals | p_vals)} "
                f"than part '{part_id}' ('{part_name}')"
            )
    if acat != "other" and pcat != "other" and acat != pcat:
        return (
            f"looks like a {acat} but part '{part_id}' ('{part_name}') is "
            f"a {pcat} — aliasing across categories means a different "
            "physical part"
        )
    return None


async def alias_part(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    part = s.state.part_by_id(_slug(str(a.get("part_id", ""))))
    if part is None:
        return {"ok": False, "error": "part not registered"}
    raw_alias = str(a.get("alias", ""))
    alias = _norm(raw_alias)
    if not alias:
        return {"ok": False, "error": "alias required"}
    conflict = _name_part_conflict(raw_alias, part.name, part.id)
    if conflict:
        return {
            "ok": False,
            "error": f"alias '{raw_alias}' {conflict}. Check whether the "
            "manual name belongs to another registered part instead.",
        }
    s._aliases_store[alias] = part.id
    return {"ok": True, "alias": alias, "part_id": part.id}


def _generation_guard(
    s: GuideToolSession, name: str, desc: str, force: bool
) -> dict[str, Any] | None:
    """Pre-flight for generation: disabled runs refuse outright; enabled runs
    must not spawn a codegen job when a registered part plausibly IS this
    manual name — generation is the most expensive path in the pipeline and
    manuals rename fittings aggressively ("Muffe auf Muffe" == "socket to
    socket"), so a same-category/size candidate is almost certainly the
    same part: the right move is alias_part, not a redundant generation."""
    # a part labelled "vormontiert/premounted" is initial-state host
    # geometry — never something to GENERATE (generation produces loose
    # parts the steps assemble). Alias it onto the host/casing or
    # mark_context_part.
    label = f"{name} {desc}".lower()
    if "vormontiert" in label or "premounted" in label or "pre-mounted" in label:
        return {
            "ok": False,
            "error": f"'{name}' is marked premounted — premounted components "
            "are part of the initial state / host, never generated geometry. "
            "alias_part it onto the host/casing part or mark_context_part.",
        }
    if not s.state.allow_part_generation:
        sugg = _suggest_parts(s, name)
        return {
            "ok": False,
            "error": "part generation is DISABLED for this guide run — every "
            "required part must come from the uploads. Alias the missing "
            "name onto the closest registered part (manuals rename the same "
            "fitting between parts list and step text) or ask_user.",
            **({"suggested_part_ids": sugg} if sugg else {}),
        }
    if force:
        return None
    sugg = _suggest_parts(s, name)
    if not sugg:
        return None
    return {
        "ok": False,
        "suggested_part_ids": sugg,
        "error": f"registered part(s) {sugg} look like the same category/"
        "size — try alias_part(part_id, '<manual name>') first. If "
        "you verified they are genuinely different parts (e.g. "
        "different end types the alias can't capture), re-call "
        "generate_missing_part with force=true",
    }


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
    err = _generation_guard(
        s, str(a.get("name", "")) or pid, desc, bool(a.get("force"))
    )
    if err:
        return err

    from cadx.agent.context import read_prompt
    from cadx.db.models import Project
    from cadx.db.session import session_scope
    from cadx.spawn import SpawnError, spawn_generate, wait_job_terminal

    name = str(a.get("name", "")).strip() or pid
    prompt = read_prompt("guide_generate.md").replace(
        "{prompt}", f"{name}: {desc}"
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
                # guide parts need functional geometry, not polish: cap the
                # improve loop (build + one repair cycle) — a valid first
                # candidate still exits early on 'all checks passed'
                options={"max_iterations": 3},
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
    from cadx.db.models import Version

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
        mesh_path=_guide_mesh(s, mesh) if mesh else None,
        quantity=_int_or_none(a.get("quantity")),
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
    missing = [
        n
        for n in required
        if _resolve_required(s, n) is None and not _is_context(s, n)
    ]
    if missing:
        return {
            "ok": False,
            "error": f"unmatched required parts: {', '.join(missing[:6])} — "
            "register/alias/generate them, or mark_context_part if they "
            "are host/tools/consumables",
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
    "mark_context_part": mark_context_part,
}
