"""Assembly-mission tools — deterministic port matching, candidate previews
(server-cached, LLM picks by id), placement, collision checks, renders.

The LLM NEVER submits a transform matrix (plan §4): ``solve_*_preview``
returns candidates carrying opaque ``candidate_id``s; ``place_instance``
consumes one from the session's server-side cache. Ported from KIARA's
p5 loop tools, hardened per the master plan.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cadx.guide import scene as gscene
from cadx.guide import solver
from cadx.guide.state import AssemblyInstance
from cadx.guide.tools import _b, _fn, _i, _num, _s
from cadx.logging import get_logger

if TYPE_CHECKING:
    from cadx.guide.session import GuideToolSession

log = get_logger(__name__)

_PREVIEW_TTL = 64  # candidates kept per mission session

SCHEMAS: dict[str, dict[str, Any]] = {
    "find_matching_ports": _fn(
        "find_matching_ports",
        "Deterministically rank compatible port pairs between a part to "
        "place and an already-placed part — call BEFORE solve_attachment_preview "
        "to pick the right ports.",
        {
            "part_id": _s("part to place"),
            "target_part_id": _s("placed target — part_id or instance_id (adapter#1)"),
        },
        ["part_id", "target_part_id"],
    ),
    "solve_attachment_preview": _fn(
        "solve_attachment_preview",
        "Compute candidate placements for a port pair (roll snaps, gap/"
        "normal gates, collision-checked). Returns candidate_id list — "
        "pass one to place_instance.",
        {
            "part_id": _s("part to place"),
            "moving_port": _s("port name on the moving part"),
            "target_part_id": _s("placed target — part_id or instance_id (adapter#1)"),
            "target_port": _s("port name on the target part"),
            "step": _i("step number — its text drives the growth-direction hint"),
            "substep_idx": _i("index within the step's sub_steps (default 0)"),
        },
        ["part_id", "moving_port", "target_part_id", "target_port"],
    ),
    "solve_free_preview": _fn(
        "solve_free_preview",
        "BBox-based placement honouring a spatial hint (on table / on top / "
        "left / right / front / back of the placed set). For parts without "
        "ports or loosely specified mounts. Returns ONE candidate_id.",
        {
            "part_id": _s("part to place"),
            "hint": _s("spatial hint: 'on table'|'top'|'left'|'right'|'front'|'back'"),
        },
        ["part_id"],
    ),
    "solve_env_preview": _fn(
        "solve_env_preview",
        "Free placement derived by keyword-scanning the step/sub-step text "
        "(German + English hints). Returns ONE candidate_id.",
        {
            "part_id": _s("part to place"),
            "step": _i("step number whose text drives the hint"),
        },
        ["part_id", "step"],
    ),
    "place_instance": _fn(
        "place_instance",
        "Commit a placement using a candidate_id from a solve_*_preview "
        "call — you cannot invent transforms. Records the instance + joint.",
        {
            "part_id": _s("part being placed"),
            "candidate_id": _s("candidate id from a preview call"),
            "step": _i("step number this placement belongs to"),
            "substep_idx": _i("index within the step's sub_steps (default 0)"),
        },
        ["part_id", "candidate_id", "step"],
    ),
    "rotate_instance": _fn(
        "rotate_instance",
        "Apply an additional roll rotation (deg) about the placed part's "
        "attachment axis — only for placed instances.",
        {
            "part_id": _s("placed part_id or instance_id (adapter#1)"),
            "degrees": _num("roll degrees"),
        },
        ["part_id", "degrees"],
    ),
    "unplace_instance": _fn(
        "unplace_instance",
        "Remove a part's placement (and its joints) so it can be re-placed.",
        {"part_id": _s("part_id or instance_id to unplace")},
        ["part_id"],
    ),
    "reset_step": _fn(
        "reset_step",
        "Remove ALL placements of one step — use when the step's assembly "
        "went wrong and must be redone.",
        {"step": _i("step number")},
        ["step"],
    ),
    "render_assembly": _fn(
        "render_assembly",
        "Render the current assembly (all placed parts) to a PNG via the "
        "executor and attach it for visual verification.",
        {
            "view": _s("camera: iso|front|top|right (default iso)"),
            "step": _i("limit to parts placed up to this step (0 = all)"),
            "each_step": _b(
                "render EVERY step state in one call (per-step build "
                "progression) — preferred for reviews over looping step renders"
            ),
        },
        [],
    ),
    "check_collisions": _fn(
        "check_collisions",
        "manifold3d boolean intersection check — either for one part's "
        "current transform or all placed pairs.",
        {"part_id": _s("check only this part (default: all pairs)")},
        [],
    ),
    "validate_assembly_geometry": _fn(
        "validate_assembly_geometry",
        "Geometry audit: joint gaps + normal alignment for every recorded "
        "joint — the numbers review scores on.",
        {},
        [],
    ),
    "validate_assembly_completeness": _fn(
        "validate_assembly_completeness",
        "Check every assembly step's required_parts are placed and every "
        "placement happened inside a step — the coverage gate.",
        {},
        [],
    ),
}


def _next_inst_no(existing: list[AssemblyInstance]) -> int:
    """1 + the max live ``part#N`` suffix — never reuse a live instance id."""
    top = 0
    for i in existing:
        tail = str(i.instance_id or "").rsplit("#", 1)
        if len(tail) == 2 and tail[1].isdigit():
            top = max(top, int(tail[1]))
    return top + 1


def _inst_by_id(s: GuideToolSession, inst_id: str) -> AssemblyInstance | None:
    return next((i for i in s.state.assembly.instances if i.instance_id == inst_id), None)


def _insts_of(s: GuideToolSession, pid: str) -> list[AssemblyInstance]:
    return [i for i in s.state.assembly.instances if i.part_id == pid]


def _resolve_inst(s: GuideToolSession, x: str) -> tuple[AssemblyInstance | None, str | None]:
    """Resolve a placed instance by instance_id ("adapter#1") or part_id
    (shortcut when exactly one instance of that part is placed — ambiguous
    ids fail loudly and name the valid instance ids)."""
    inst = _inst_by_id(s, x)
    if inst is not None:
        return inst, None
    # a "part#N" reference to a non-existent instance must NOT silently
    # resolve to a different instance — name the real candidates instead
    if "#" in x:
        siblings = _insts_of(s, x.split("#")[0])
        ids = [i.instance_id for i in siblings]
        return None, (
            f"instance '{x}' does not exist — "
            + (
                f"placed instances of that part: {ids}"
                if ids
                else f"part '{x.split('#')[0]}' has NO placed instances "
                f"(placed now: {[i.instance_id for i in s.state.assembly.instances]})"
            )
        )
    insts = _insts_of(s, x)
    if len(insts) == 1:
        return insts[0], None
    if insts:
        ids = [i.instance_id for i in insts]
        return None, f"'{x}' has {len(insts)} placed instances — pass instance_id: {ids}"
    placed = [i.instance_id for i in s.state.assembly.instances][:15]
    return None, f"no placed instance '{x}' — placed: {placed}"


def _name_index(s: GuideToolSession) -> dict[str, str]:
    index = {_norm2(p.name): p.id for p in s.state.parts}
    index.update({_norm2(p.id): p.id for p in s.state.parts})
    # upload filenames are identifiers too — steps name parts after the
    # CAD asset ("Elbow_90deg_RN22_socket_to_spigot"), which shares no
    # tokens with the manual's display name ("Rohrbogen Muffe/Spitzende")
    for p in s.state.parts:
        stem = Path(str(p.mesh_path or "")).stem
        stem = re.sub(r"^[0-9a-f-]{8,}__", "", stem)  # asset-uuid prefix
        stem = re.sub(r"_mm$", "", stem)
        if stem:
            index.setdefault(_norm2(stem), p.id)
    return index


def _part_err(s: GuideToolSession) -> str:
    ids = [p.id for p in s.state.parts][:15]
    return f"part not found — valid part_ids: {ids}"


def _part_ref(s: GuideToolSession, ref: Any) -> Any:
    """Resolve a part argument that may be a part id OR an instance id —
    agents iterating list_parts' `instances` pass 'pipe_l500#1' where a
    part id is expected (observed: 4x get_part_features error-streak). The
    intent is unambiguous — the instance's part — so resolve it instead of
    erroring."""
    ref = str(ref or "")
    part = s.state.part_by_id(ref)
    if part is not None:
        return part
    inst = _inst_by_id(s, ref)
    if inst is not None:
        return s.state.part_by_id(inst.part_id)
    return None


def _host_part_id(s: GuideToolSession) -> str | None:
    """The largest-bbox registered part — the assembly's host/base. In a
    multi-match target phrase the host name is almost always a locative
    qualifier ("Adapter AM GEHÄUSE" mates to the adapter), so it loses
    ties to any mountable part."""
    vols: list[tuple[float, str]] = []
    for p in s.state.parts:
        if p.bbox_min is None or p.bbox_max is None:
            continue
        vol = float(
            (p.bbox_max[0] - p.bbox_min[0])
            * (p.bbox_max[1] - p.bbox_min[1])
            * (p.bbox_max[2] - p.bbox_min[2])
        )
        vols.append((vol, p.id))
    vols.sort(reverse=True)
    # the host must be a real size outlier — among equal-sized parts the
    # biggest is arbitrary (two elbows tie) and demoting it picks randomly
    if len(vols) >= 2 and vols[0][0] < 3.0 * vols[1][0]:
        return None
    return vols[0][1] if vols else None


def _fuzzy_pid(n: str, cands: list[tuple[str, str]]) -> str | None:
    """Best full-name fuzzy match ≥0.92, or None. The cutoff sits above
    the one-token-swap band (socket→spigot ≈0.91) but below real typos
    (soket→socket ≈0.98); best-of rather than first-hit keeps sibling
    names deterministic."""
    import difflib

    best_pid, best_ratio = None, 0.0
    seen_pid: set[str] = set()
    for k, pid in cands:
        if pid in seen_pid:
            continue
        seen_pid.add(pid)
        r = difflib.SequenceMatcher(None, n, k).ratio()
        if r > best_ratio:
            best_ratio, best_pid = r, pid
    return best_pid if best_pid is not None and best_ratio >= 0.92 else None


def _resolve_name(s: GuideToolSession, name: str, context: str = "") -> str | None:
    """Resolve a manual label to a part id: alias → exact name/id →
    containment. Manuals attach qualifiers to part names ("Gehäuse, linker
    oberer Anschluss", "Adapter links oben") — containment on the LONGEST
    matching alias/name keeps that usable instead of collapsing to free
    placement. ``context`` (e.g. the parent step text) breaks token ties:
    "Anschluss 2x außen" matches both elbow names equally, but the step's
    "(Muffe auf Muffe)" confirms which one's tokens are actually meant."""
    n = _norm2(name)
    if not n:
        return None
    hit = s._aliases_store.get(n) or _name_index(s).get(n)
    if hit:
        return hit
    cands = list(s._aliases_store.items()) + list(_name_index(s).items())
    # typo-robust match for long names — registered part names come from
    # upload filenames and carry the SOURCE'S typos ("soket_to_socket");
    # the plan spells the word correctly and then ties against the wrong
    # sibling name on token overlap. A high-ratio fuzzy hit on the whole
    # normalized name is stronger evidence than token ties.
    if len(n) >= 12:
        hit = _fuzzy_pid(n, cands)
        if hit:
            return hit
    # keys shorter than 3 chars would containment-match almost anything
    hits = [(k, pid) for k, pid in cands if len(k) >= 3 and (k in n or n in k)]
    # token-overlap fallback: feature phrases ("Rohrbogen am linken
    # Anschluss", "vertikales Rohr 500mm") share a part's distinctive
    # tokens even with no alias/containment — ≥2 shared tokens keeps it
    # specific, longest key breaks ties toward the richer name. A SINGLE
    # shared token also resolves when it's distinctive — used by exactly
    # one part ("Gehäuse" in "linker oberer Anschluss am Gehäuse");
    # generic tokens shared by several parts ("rohr") still need 2 hits.
    nt = _tokset(n)
    ctx = _tokset(context)
    tok_owners: dict[str, set[str]] = {}
    for k, pid in cands:
        for t in _tokset(k):
            tok_owners.setdefault(t, set()).add(pid)
    scored = []
    from cadx.guide.part_tools import _sig

    name_cat, _ = _sig(name)
    for k, pid in cands:
        shared = _tokset(k) & nt
        overlap = len(shared)
        if overlap < 2 and not any(len(tok_owners.get(t, set())) == 1 for t in shared):
            continue
        # category guard on token matches: shared size tokens ("dn22")
        # must not map a fitting name onto a pipe part — 'Adapter DN22'
        # resolved to pipe_l70 in the two-pipes run and sent the agent
        # hunting nonexistent adapter targets
        if name_cat != "other":
            part = s.state.part_by_id(pid)
            pcat, _ = _sig(f"{part.name} {part.id}" if part else k)
            if pcat != "other" and pcat != name_cat:
                continue
        # context confirms distinctive tokens: "Anschluss 2x außen" ties
        # on rohrbogen/dn/22 for both elbows — the step's "(Muffe auf
        # Muffe)" leaves spitzende's token UNCONFIRMED → fewer unconfirmed
        # tokens = the meant part. Bare names keep longest-key.
        tie = -len(_tokset(k) - nt - ctx) if ctx else 0
        scored.append((overlap, tie, len(k), pid))
    # host demotion: a target phrase naming BOTH the biggest part and a
    # mountable one mates to the mountable — "Adapter am Gehäuse" means
    # the adapter, the casing is just where it sits. Only applies with a
    # real alternative: a phrase naming ONLY the host still resolves to it
    host = _host_part_id(s)
    matched = {pid for _, pid in hits} | {t[3] for t in scored}
    if host and host in matched and len(matched) > 1:
        hits = [h for h in hits if h[1] != host]
        scored = [t for t in scored if t[3] != host]
    if hits:
        # longest containing/contained key wins — "rohrbogen dn22 muffe" beats "rohr"
        return max(hits, key=lambda kv: len(kv[0]))[1]
    if scored:
        return max(scored)[3]
    return None


def _tokset(text: str) -> set[str]:
    """Content tokens (letters / digit-runs) of a normalized name."""
    return {
        t
        for t in re.findall(r"[a-zäöüß]+|\d+", _norm2(text))
        if len(t) >= 2 or t.isdigit()
    }


# sub_step actions that consume a NEW part occurrence — orient/check inspect
# or adjust an already-placed instance and must not inflate the count
_PLACE_ACTIONS = {"place", "attach", "insert", "fasten", "mount"}


def _step_requirements(s: GuideToolSession) -> dict[int, dict[str, int]]:
    """Per-step required instance counts: {step_no: {part_id: n}}. Each
    placement sub-step names one occurrence of a part; steps without
    sub-steps fall back to required_parts occurrences (1 per name)."""
    reqs: dict[int, dict[str, int]] = {}
    for st in s.state.steps:
        if st.action_type != "assembly":
            continue
        counts: dict[str, int] = {}
        for ss in st.sub_steps:
            if ss.action not in _PLACE_ACTIONS:
                continue
            # the step prose carries the discriminating qualifier — two
            # elbow types tie on rohrbogen/dn/22; "(Muffe auf Muffe)" in
            # the same sentence decides, so requirements must resolve
            # with context like the planner's _rp does
            pid = _resolve_name(s, ss.part, context=f"{ss.text or ''} {st.text} {st.title}")
            if pid:
                counts[pid] = counts.get(pid, 0) + 1
        if not counts:
            for n in st.required_parts:
                pid = _resolve_name(s, n, context=f"{st.text} {st.title}")
                counts[pid or _norm2(n) + " (unresolved)"] = (
                    counts.get(pid or _norm2(n) + " (unresolved)", 0) + 1
                )
        reqs[st.number] = counts
    return reqs


def _completeness_issues(s: GuideToolSession) -> list[str]:
    """Every instance a step requires must exist — per-step counting, a part
    placed once does NOT satisfy later steps that need their own copy."""
    issues: list[str] = []
    # a committed bridge joint's moving instance also covers the absorbed
    # leg sub-step ("freie Muffe aufs Rohr") — one physical instance can
    # satisfy two placement substeps when the second names a mate, not a
    # new occurrence
    legs: dict[tuple[int, str], int] = {}
    for j in s.state.assembly.joints:
        if j.get("bridge"):
            key = (int(j.get("step", 0)), str(j.get("moving_part") or ""))
            legs[key] = legs.get(key, 0) + 1
    for st_no, counts in _step_requirements(s).items():
        for pid, need in counts.items():
            have = sum(
                1
                for i in s.state.assembly.instances
                if i.placed_in_step == st_no and i.part_id == pid
            ) + legs.get((st_no, pid), 0)
            if have < need:
                msg = f"step {st_no}: '{pid}' needs {need} instance(s), has {have}"
                if pid.endswith(" (unresolved)"):
                    msg += (
                        " — the name matches no registered part: "
                        "alias_part it onto a registered part or "
                        "update_step to drop it from required_parts"
                    )
                issues.append(msg)
    # every step TARGET resolving to a registered part needs a placed
    # instance — a fitting cannot joint onto a host the plan referenced
    # but nobody placed (heatpump run: steps targeted the casing, it
    # stayed unplaced, the adapter→casing joint became impossible and the
    # fittings grew as a detached pipe tree). A host is reference
    # geometry, not an optional part.
    placed_parts = {i.part_id for i in s.state.assembly.instances}
    seen_target_issues: set[str] = set()
    for st in s.state.steps:
        if st.action_type != "assembly":
            continue
        for ss in st.sub_steps:
            for nm in (ss.target, ss.target2):
                pid = _resolve_name(
                    s, nm, context=f"{ss.text or ''} {st.text} {st.title}"
                )
                if pid and pid not in placed_parts and pid not in seen_target_issues:
                    seen_target_issues.add(pid)
                    issues.append(
                        f"step {st.number}: target '{nm}' resolves to "
                        f"'{pid}' but no instance of it is placed — place "
                        "it first (the host is reference geometry, "
                        "not optional)"
                    )
    if not s.state.assembly.instances:
        issues.append("no parts placed at all")
    return issues


def _named_port(s: GuideToolSession, part_id: str, name: Any) -> Any:
    part = s.state.part_by_id(part_id)
    if part is None:
        return None
    return next((p for p in part.ports if p.name == name), None)


def _placed_bboxes(s: GuideToolSession) -> list[dict[str, Any]]:
    out = []
    for inst in s.state.assembly.instances:
        part = s.state.part_by_id(inst.part_id)
        if part is None:
            continue
        mn, mx = solver.transform_bbox(part, inst.transform)
        out.append(
            {
                "instance_id": inst.instance_id,
                "part_id": inst.part_id,
                "bbox_min_world": mn,
                "bbox_max_world": mx,
            }
        )
    return out


def _scene(s: GuideToolSession) -> gscene.Scene:
    sc = getattr(s, "_scene_cache", None)
    if sc is None:
        sc = gscene.Scene()
        for inst in s.state.assembly.instances:
            part = s.state.part_by_id(inst.part_id)
            if part is not None and part.mesh_path:
                sc.add(inst.instance_id or inst.part_id, part.mesh_path, inst.transform)
        s._scene_cache = sc
    return sc


def _cache_candidates(s: GuideToolSession, cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in cands:
        cid = uuid.uuid4().hex[:12]
        s._preview_cache[cid] = c
        row = {k: v for k, v in c.items() if k != "transform"}
        row["candidate_id"] = cid
        out.append(row)
    # bound the cache
    while len(s._preview_cache) > _PREVIEW_TTL:
        s._preview_cache.pop(next(iter(s._preview_cache)))
    return out


async def find_matching_ports(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    moving = _part_ref(s, a.get("part_id", ""))
    fixed_inst, err = _resolve_inst(s, str(a.get("target_part_id", "")))
    if moving is None or fixed_inst is None:
        return {"ok": False, "error": err or _part_err(s)}
    fixed = s.state.part_by_id(fixed_inst.part_id)
    if fixed is None:
        return {"ok": False, "error": _part_err(s)}
    used = {
        str(j.get("fixed_port"))
        for j in s.state.assembly.joints
        if j.get("fixed_instance") == fixed_inst.instance_id
        or (not j.get("fixed_instance") and j.get("fixed_part") == fixed_inst.part_id)
    } | {
        str(j.get("moving_port"))
        for j in s.state.assembly.joints
        if j.get("moving_instance") == fixed_inst.instance_id
    }
    cands = solver.find_matching_ports(
        moving, fixed, used_fixed_ports=used, include_mismatch=True
    )
    if not cands:
        return {
            "ok": True,
            "candidates": [],
            "note": "no compatible port pairs — use solve_free_preview or "
            "solve_env_preview for bbox placement",
        }
    return {
        "ok": True,
        "candidates": cands[:8],
        "moving_ports": [p.model_dump() for p in moving.ports][:10],
    }


async def solve_attachment_preview(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    moving = _part_ref(s, a.get("part_id", ""))
    fixed_inst, err = _resolve_inst(s, str(a.get("target_part_id", "")))
    if moving is None or fixed_inst is None:
        return {"ok": False, "error": err or _part_err(s)}
    fixed = s.state.part_by_id(fixed_inst.part_id)
    if fixed is None:
        return {"ok": False, "error": _part_err(s)}
    cands = solver.solve_attachment(
        moving,
        str(a.get("moving_port", "")),
        fixed,
        fixed_inst.transform or solver.identity_transform(),
        str(a.get("target_port", "")),
    )
    if not cands:
        return _no_cand_err(moving, fixed)
    # collision-check each candidate against the placed set; also expose the
    # burial fraction so the LLM can prefer a candidate whose body stays
    # OUTSIDE foreign parts instead of discovering the sink only at commit
    sc = _scene(s)
    for c in cands:
        c["fixed_part"] = fixed.id
        c["fixed_instance"] = fixed_inst.instance_id
        if moving.mesh_path:
            rep = sc.collision_report(moving.id, moving.mesh_path, c["transform"])
            c["collision_free"] = rep["ok"]
            c["collisions"] = rep["collisions"][:3]
        bf, bh, sf, si = _candidate_burial(s, moving, c)
        if bf > 0.05:
            c["buried_frac"] = round(bf, 2)
            c["buried_in"] = bh
        if sf > 0.05 and moving.id != _host_part_id(s):
            c["swallow_frac"] = round(sf, 2)
            c["swallows"] = si
        # the commit gates (buried >0.45, swallow >0.6) reject these at
        # place_instance — fold them into geometry_ok NOW so the agent
        # only ever sees viable candidates ranked on top; committing a
        # buried candidate is deterministic rejection, not a choice
        if bf > 0.45:
            c["geometry_ok"] = False
            c["rejected_reason"] = (
                f"mates the body {round(bf * 100)}% inside foreign '{bh}'"
            )
        elif sf > 0.6 and moving.id != _host_part_id(s):
            c["geometry_ok"] = False
            c["rejected_reason"] = (
                f"lands on top of '{si}' ({round(sf * 100)}% enveloped)"
            )
    gvec, gspec = _grow_dir_hint(s, a)
    ovec, ospec = _out_dir_hint(s, a, moving)
    _rank_candidates(s, a, cands, moving, fixed, fixed_inst, gvec, gspec, ovec, ospec)
    out: dict[str, Any] = {
        "ok": True,
        "candidates": _cache_candidates(s, cands),
        "note": "pass one candidate_id to place_instance — candidates with "
        "geometry_ok=false are pre-rejected (buried/swallowing); pick a "
        "geometry_ok=true candidate or a different port pair",
    }
    if gspec:
        hint_txt = f"grow_dir '{gspec}'"
        if gvec is not None:
            hint_txt += f" {[round(v, 2) for v in gvec]}"
        out["grow_dir"] = hint_txt + " — prefer high dir_match"
    if ospec:
        out["out_dir"] = (
            f"next part grows '{ospec}'"
            + (f" {[round(v, 2) for v in ovec]}" if ovec is not None else "")
            + " — prefer high out_dir_match (free port faces the follower)"
        )
    if not any(c["geometry_ok"] for c in cands):
        _add_alternative_pairs(out, s, moving, fixed, a)
    return out


def _rank_candidates(
    s: GuideToolSession,
    a: dict[str, Any],
    cands: list[dict[str, Any]],
    moving: Any,
    fixed: Any,
    fixed_inst: Any,
    gvec: Any,
    gspec: str,
    ovec: Any,
    ospec: str,
) -> None:
    """Direction-aware candidate ordering (in place).

    ``dir_match`` — the moving body's growth direction vs the step text
    ("vertikal nach unten"). ``out_dir_match`` — the direction of this
    part's best FREE port vs where the NEXT part grows: a 90° elbow's
    free socket must face the "vertikale Röhre" coming after it — the
    roll chosen HERE is where verticality is actually decided (a pipe
    seated in a socket has no roll freedom left). Geometry verdicts
    always outrank both; the hints only order legal candidates."""
    ft = fixed_inst.transform or solver.identity_transform()
    for c in cands:
        if gspec:
            gdir = solver.candidate_grow_dir(
                moving, c["transform"], fixed, ft, str(a.get("target_port", ""))
            )
            c["dir_match"] = round(solver.grow_dir_score(gdir, gspec, gvec), 3)
        if ospec:
            mt = np.asarray(c["transform"], dtype=float).reshape(4, 4)
            best = 0.0
            for p in moving.ports:
                if p.name == str(a.get("moving_port", "")):
                    continue
                wa = mt[0:3, 0:3] @ np.asarray(p.axis, dtype=float)
                n = float(np.linalg.norm(wa))
                if n > 1e-6:
                    best = max(best, solver.grow_dir_score(wa / n, ospec, ovec))
            c["out_dir_match"] = round(best, 3)
    cands.sort(
        key=lambda c: (
            not c["geometry_ok"],
            -(c.get("out_dir_match") or 0.5),
            -(c.get("dir_match") or 0.5),
            c.get("joint_gap_mm") or 0.0,
        )
    )


def _out_dir_hint(s: GuideToolSession, a: dict[str, Any], moving: Any) -> tuple[Any, str]:
    """Direction the NEXT part grows — found on the first sub_step AFTER
    the current one (plan order) that targets the moving part. Its text
    describes the follow-on part's insertion ("unteren Seite in die
    Muffe" → the follower grows +z), which is exactly where THIS part's
    free port must face. Only sub_steps after (step, substep_idx) count —
    earlier mates don't constrain this fitting's remaining sockets."""
    step_no = int(a.get("step") or 0)
    ss_idx = int(a.get("substep_idx") or 0)
    if not step_no:
        return None, ""
    for st in sorted(s.state.steps, key=lambda x: x.number):
        for i, ss in enumerate(st.sub_steps):
            if (st.number, i) <= (step_no, ss_idx):
                continue
            refs = (
                (ss.target or "").split("(")[0].strip(),
                (ss.target2 or "").split("(")[0].strip(),
            )
            if moving.id not in refs:
                continue
            vec, spec = solver.direction_hint(ss.text or "")
            if spec:
                return vec, spec
            return None, ""  # targeted mate without direction words — no hint
    return _sibling_out_hint(s, moving)


def _sibling_out_hint(s: GuideToolSession, moving: Any) -> tuple[Any, str]:
    """Free-port direction of an already-placed instance of the SAME part.
    A parallel pair of elbows has parallel free sockets — the sibling's
    proven roll is the strongest roll hint available."""
    for inst in s.state.assembly.instances:
        if inst.part_id != moving.id or inst.transform is None:
            continue
        j = next(
            (
                x
                for x in s.state.assembly.joints
                if str(x.get("fixed_instance")) == inst.instance_id
            ),
            None,
        )
        pname = (j or {}).get("fixed_port")
        p = next((p for p in moving.ports if p.name == pname), None)
        if p is None:
            continue
        mt = np.asarray(inst.transform, dtype=float).reshape(4, 4)
        wa = mt[0:3, 0:3] @ np.asarray(p.axis, dtype=float)
        n = float(np.linalg.norm(wa))
        if n > 1e-6:
            return wa / n, "dir"
    return None, ""


def _sibling_dir_hint(s: GuideToolSession, pid: str) -> tuple[Any, str]:
    """Growth vector of an already-placed instance of the SAME part —
    repeated fittings in a manual are almost always a parallel pair (two
    vertical pipes, two symmetric rails). The sibling's measured growth
    (fixed-port → body-centre direction) beats any text inference: the
    first copy's proven build becomes the second copy's hint."""
    insts = {i.instance_id: i for i in s.state.assembly.instances}
    for inst in s.state.assembly.instances:
        if inst.part_id != pid or inst.transform is None:
            continue
        j = next(
            (
                x
                for x in s.state.assembly.joints
                if str(x.get("moving_instance")) == inst.instance_id
            ),
            None,
        )
        fi = insts.get(str(j.get("fixed_instance"))) if j else None
        fp_part = s.state.part_by_id(fi.part_id) if fi else None
        fp = next(
            (p for p in (fp_part.ports if fp_part else []) if p.name == (j or {}).get("fixed_port")),
            None,
        )
        part = s.state.part_by_id(pid)
        if not (fi and fi.transform and fp and part):
            continue
        ft = np.asarray(fi.transform, dtype=float).reshape(4, 4)
        port_w = ft[0:3, 0:3] @ np.asarray(fp.pos, dtype=float) + ft[0:3, 3]
        mn, mx = solver.world_bbox(
            part.bbox_min, part.bbox_max, np.asarray(inst.transform, dtype=float)
        )
        d = (mn + mx) / 2.0 - port_w
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            return d / n, "dir"
    return None, ""


def _no_cand_err(moving: Any, fixed: Any) -> dict[str, Any]:
    # self-healing error: name the valid ports AND the compatible pairs so
    # the next call succeeds instead of guessing port names blindly
    err_out: dict[str, Any] = {
        "ok": False,
        "error": "no candidates for those port names",
        "moving_ports": [p.name for p in moving.ports],
        "fixed_ports": [p.name for p in fixed.ports],
    }
    empty = []
    if not moving.ports:
        empty.append(moving.id)
    if not fixed.ports:
        empty.append(fixed.id)
    if empty:
        err_out["error"] = (
            f"{empty} have no extracted ports — run get_part_features "
            "on them first, then retry"
        )
        err_out["next_action"] = f"get_part_features({empty[0]!r})"
        return err_out
    auto = solver.find_matching_ports(moving, fixed, include_mismatch=True)
    if auto:
        err_out["compatible_pairs"] = auto[:6]
        err_out["hint"] = "retry with moving_port/target_port from compatible_pairs"
    return err_out


def _grow_dir_hint(s: GuideToolSession, a: dict[str, Any]) -> tuple[Any, str]:
    """(vector|None, spec) — direction intent parsed from the referenced
    sub_step (+step) text. No hint when step/substep are absent or the
    prose carries no directional word."""
    st = s.state.step_by_number(int(a.get("step") or 0))
    if st is None:
        return None, ""
    idx = int(a.get("substep_idx") or 0)
    ss_txt = ""
    if 0 <= idx < len(st.sub_steps):
        ss_txt = st.sub_steps[idx].text or ""
    vec, spec = solver.direction_hint(f"{ss_txt} {st.text or ''} {st.title or ''}")
    if spec:
        return vec, spec
    # cross-step evidence: a LATER sub_step on the same part often carries
    # the direction word the placing step lacks ("Rohrbogen auf die linke
    # vertikale Röhre" — the pipe itself was placed two steps earlier).
    # sub_step.part carries the part id, optionally with a "(links)" suffix
    pid = str(a.get("part_id") or "")
    if pid:
        # first hit in plan order wins — the placing sub_step's own
        # end-side phrasing outranks later descriptive mentions
        for st2 in s.state.steps:
            for ss in st2.sub_steps:
                refs = (
                    (ss.part or "").split("(")[0].strip(),
                    (ss.target or "").split("(")[0].strip(),
                    (ss.target2 or "").split("(")[0].strip(),
                )
                if pid not in refs:
                    continue
                vec, spec = solver.direction_hint(ss.text or "")
                if spec:
                    return vec, spec
    return _sibling_dir_hint(s, pid)


def _add_alternative_pairs(
    out: dict[str, Any],
    s: GuideToolSession,
    moving: Any,
    fixed: Any,
    a: dict[str, Any],
) -> None:
    """All candidates for the probed pair are pre-rejected — name the pairs
    that still have a chance so the next call succeeds instead of
    re-probing this sink."""
    auto = solver.find_matching_ports(moving, fixed, include_mismatch=True)
    probed = (str(a.get("moving_port", "")), str(a.get("target_port", "")))
    alt = [
        p for p in auto
        if (p.get("moving_port"), p.get("fixed_port")) != probed
    ]
    if alt:
        out["alternative_pairs"] = alt[:6]
        out["hint"] = (
            "every candidate for this pair is pre-rejected — retry "
            "solve_attachment_preview with a pair from alternative_pairs; "
            "do NOT re-commit this pair"
        )


async def solve_free_preview(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    part = _part_ref(s, a.get("part_id", ""))
    if part is None:
        return {"ok": False, "error": _part_err(s)}
    t = solver.solve_free_placement(part, _placed_bboxes(s), str(a.get("hint", "")))
    sc = _scene(s)
    rep = (
        sc.collision_report(part.id, part.mesh_path, t)
        if part.mesh_path
        else {"ok": True, "collisions": []}
    )
    cands = [
        {
            "transform": t,
            "hint": str(a.get("hint", "")),
            "collision_free": rep["ok"],
            "collisions": rep["collisions"][:3],
            "joint_gap_mm": None,
            "normal_dot": None,
            "geometry_ok": True,
        }
    ]
    return {"ok": True, "candidates": _cache_candidates(s, cands)}


async def solve_env_preview(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    part = _part_ref(s, a.get("part_id", ""))
    if part is None:
        return {"ok": False, "error": _part_err(s)}
    step = s.state.step_by_number(int(a.get("step", 0)))
    text = step.text if step else ""
    t = solver.solve_env_placement(part, _placed_bboxes(s), text)
    sc = _scene(s)
    rep = (
        sc.collision_report(part.id, part.mesh_path, t)
        if part.mesh_path
        else {"ok": True, "collisions": []}
    )
    cands = [
        {
            "transform": t,
            "hint": "env",
            "collision_free": rep["ok"],
            "collisions": rep["collisions"][:3],
            "geometry_ok": True,
        }
    ]
    return {"ok": True, "candidates": _cache_candidates(s, cands)}


def _target_mismatch(
    s: GuideToolSession, step: Any, pid: str, req_idx: int, cand: dict[str, Any]
) -> str | None:
    """Reject a candidate whose fixed part contradicts the substep's declared
    target — the solver happily returns collision-free candidates mating to
    the casing when the plan says 'Adapter', burying the part in the host.
    Unresolvable targets are host/env features (a casing nozzle is not a
    registered part) and skip the check; bridge substeps accept either
    target/target2."""
    ctx = f"{step.text} {step.title}"
    subs = [
        (i, ss)
        for i, ss in enumerate(step.sub_steps)
        if ss.action in _PLACE_ACTIONS
        and _resolve_name(s, ss.part, context=f"{ss.text or ''} {ctx}") == pid
    ]
    targets: list[tuple[str, str | None]] = []
    for idx, ss in subs:
        if req_idx and idx != req_idx and any(i == req_idx for i, _ in subs):
            continue
        for name in (ss.target, ss.target2):
            if name:
                targets.append((name, _resolve_name(s, name, context=f"{ss.text or ''} {ctx}")))
    resolved = {t for _, t in targets if t}
    fixed_pid = str(cand.get("fixed_part") or "")
    if not resolved or fixed_pid in resolved:
        return None
    wanted = next(n for n, t in targets if t)
    want_pid = sorted(resolved)[0]
    placed_want = [i.instance_id for i in _insts_of(s, want_pid)]
    ref = placed_want[0] if len(placed_want) == 1 else want_pid
    return (
        f"candidate mates to '{fixed_pid or 'nothing'}' but step {step.number} "
        f"requires target '{wanted}' (resolves to '{want_pid}') — re-run "
        "solve_attachment_preview/find_matching_ports with "
        f"target_part_id='{ref}'"
    )


def _candidate_burial(
    s: GuideToolSession, part: Any, cand: dict[str, Any]
) -> tuple[float, str | None, float, str | None]:
    """Burial in BOTH directions, both against instances the candidate is
    NOT mating to (its fixed instance legitimately swallows the plug):

    - ``buried``: fraction of the moving part's world AABB inside foreign
      instances — the transform pushes the body into a foreign host
      (pipe sliding through a fitting into the casing behind it).
    - ``swallow``: the worst fraction of a foreign instance's AABB inside
      the candidate's box — the candidate lands ON TOP of already-placed
      parts (an elbow dropped over an existing adapter+pipe stack keeps
      valid port mates yet duplicates a branch in the same space).

    Returns (buried_frac, buried_host, swallow_frac, swallowed_id)."""
    if part.bbox_min is None or part.bbox_max is None or cand.get("transform") is None:
        return 0.0, None, 0.0, None
    mn, mx = solver.world_bbox(
        part.bbox_min, part.bbox_max, np.asarray(cand["transform"], dtype=float)
    )
    partner = str(cand.get("fixed_instance") or "")
    boxes = _placed_boxes(s, exclude=partner)
    enclosures = _enclosure_hosts(boxes)
    chain2 = _joint_neighbours(s, partner)
    foreign: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    chain: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for iid, bb in boxes.items():
        if iid in enclosures:
            continue  # cavity placement — the collision check governs
        (chain if iid in chain2 else foreign)[iid] = bb
    frac = solver.union_containment(mn, mx, list(foreign.values()))
    host = None
    if frac > 0.0:
        best = 0.0
        for iid, bb in foreign.items():
            f = solver.union_containment(mn, mx, [bb])
            if f > best:
                best, host = f, iid
    # FULL containment inside a chain-2 host is still a tunnel — partial
    # overlap is engagement adjacency, >0.85 means it went THROUGH
    for iid, bb in chain.items():
        f = solver.union_containment(mn, mx, [bb])
        if f > 0.85 and f > frac:
            frac, host = f, iid
    swallow = 0.0
    swallowed = None
    for iid, bb in boxes.items():
        if iid in enclosures:
            continue
        f = solver.union_containment(bb[0], bb[1], [(mn, mx)])
        if f > swallow:
            swallow, swallowed = f, iid
    return frac, host, swallow, swallowed


def _placed_boxes(
    s: GuideToolSession, *, exclude: str = ""
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """instance_id → world AABB for every placed instance (minus ``exclude``)."""
    boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for inst in s.state.assembly.instances:
        if inst.instance_id == exclude or inst.transform is None:
            continue
        p2 = s.state.part_by_id(inst.part_id)
        if p2 is None or p2.bbox_min is None or p2.bbox_max is None:
            continue
        boxes[inst.instance_id or inst.part_id] = solver.world_bbox(
            p2.bbox_min, p2.bbox_max, np.asarray(inst.transform, dtype=float)
        )
    return boxes


def _joint_neighbours(s: GuideToolSession, instance_id: str) -> set[str]:
    """Instances the given one is already jointed to — the candidate's
    joint neighbourhood. An elbow mating an adapter that sits inside the
    casing legitimately reaches into the casing's volume: plugged chains
    share engagement space (candidate→partner→partner's mates)."""
    out: set[str] = set()
    for j in s.state.assembly.joints:
        for a, b in (
            (j.get("moving_instance"), j.get("fixed_instance")),
            (j.get("fixed_instance"), j.get("moving_instance")),
        ):
            if a == instance_id and b:
                out.add(str(b))
    return out


def _enclosure_hosts(
    boxes: dict[str, tuple[np.ndarray, np.ndarray]],
) -> set[str]:
    """Instances acting as ENCLOSURES — delegates to solver.enclosure_hosts
    so the preview gate, post-commit audit and unity emit share ONE rule."""
    return solver.enclosure_hosts(boxes)


def _burial_gate_error(
    s: GuideToolSession, part: Any, pid: str, cand: dict[str, Any]
) -> dict[str, Any] | None:
    """Rejection payload when a candidate's transform puts the part inside
    foreign geometry — either direction (part swallowed by a host, or
    landing on top of placed hardware)."""
    buried_frac, buried_host, swallow_frac, swallowed_id = _candidate_burial(
        s, part, cand
    )
    # enclosure exemption: a cover/host-scale part legitimately contains
    # already-built hardware (last-step "Verkleidung über alles"); the
    # swallow gate targets parts that duplicate a branch at the same spot
    if swallow_frac > 0.6 and pid != _host_part_id(s):
        return {
            "ok": False,
            "error": f"candidate lands on top of '{swallowed_id}' — "
            f"{round(swallow_frac * 100)}% of that placed part ends up "
            f"inside '{pid}'s box. This duplicates an already-built "
            "branch at the same spot: mate to a port of that instance "
            "instead (it is the physical continuation), or pick a "
            "different target port",
        }
    if buried_frac <= 0.45:
        return None
    # name the actionable alternatives concretely — retrying the same
    # candidate_id is the observed dead-loop; the fix is a DIFFERENT
    # moving port (the part's other end) or a different target port
    other_ports = [p.name for p in part.ports if p.name != cand.get("moving_port")]
    hint = (
        f"run solve_attachment_preview with a different moving_port "
        f"(other ports of '{pid}': {other_ports[:6]}) or a different "
        "target (find_matching_ports first)."
    )
    if str(cand.get("fixed_part") or "") == _host_part_id(s):
        # the host shell itself is not a fitting — manuals name the
        # device when they mean the hardware mounted ON it. The real
        # mate is a placed non-host instance (adapter/socket).
        fittings = [
            i.instance_id
            for i in s.state.assembly.instances
            if i.part_id != _host_part_id(s)
        ]
        hint = (
            f"'{cand.get('fixed_part')}' is the host SHELL, not a "
            "fitting — parts never mate INTO it; they mate the "
            f"hardware mounted on it. Placed fittings: {fittings[:8]}. "
            "Preview against one of those instance ids instead."
        )
    return {
        "ok": False,
        "error": f"candidate buries {round(buried_frac * 100)}% of "
        f"'{pid}' inside '{buried_host}' — it is not jointed to that "
        "part. The mate direction points the body into a foreign "
        f"host. Do NOT retry this candidate — {hint}",
        "other_moving_ports": other_ports,
    }


def _kit_bound(s: GuideToolSession, part: Any) -> int | None:
    """Hard physical bound: the kit's shipped count for this part. The
    registered quantity wins; otherwise fall back to the Stückliste counts
    the doc mission captured (register calls often omit quantity — the
    heatpump run placed 4x a 2-piece pipe because nothing bounded it)."""
    if getattr(part, "quantity", None) is not None:
        return int(part.quantity)
    doc = getattr(s.state, "document", None)
    for nrm, qty in (getattr(doc, "detected_part_qty", None) or {}).items():
        pid = s._aliases_store.get(nrm) or _name_index(s).get(nrm)
        if pid == part.id:
            return int(qty)
    return None


async def place_instance(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pid = str(a.get("part_id", ""))
    part = s.state.part_by_id(pid)
    if part is None:
        return {"ok": False, "error": _part_err(s)}
    cid = str(a.get("candidate_id", ""))
    cand = s._preview_cache.get(cid)
    if cand is None:
        return {"ok": False, "error": "unknown/expired candidate_id — run a solve_*_preview first"}
    step_no = int(a.get("step", 0))
    step = s.state.step_by_number(step_no)
    if step is None:
        return {"ok": False, "error": f"step {step_no} not found"}
    existing = _insts_of(s, pid)
    reqs = _step_requirements(s)
    need = sum(req.get(pid, 0) for req in reqs.values())
    # per-step quota: a step that already holds its required count of this
    # part must not absorb another — repairs go through unplace_instance,
    # not through stacking extras into an already-satisfied step
    # planner-driven pre-mounts (initial-state hardware the manual names but
    # never mounts itself — e.g. fittings already on the device) legitimately
    # add instances beyond the step requirements; they still respect the cap
    premount = bool(a.get("premount"))
    need_here = reqs.get(step_no, {}).get(pid, 0)
    have_here = sum(1 for i in existing if i.placed_in_step == step_no)
    if not premount and need_here and have_here >= need_here:
        still_open = [
            st_no
            for st_no, req in reqs.items()
            if req.get(pid, 0)
            > sum(1 for i in existing if i.placed_in_step == st_no and i.part_id == pid)
        ]
        hint = (
            f" — step(s) {still_open} still need it: call place_instance "
            f"with step={still_open[0]} (or unplace_instance first to move "
            "an existing one)" if still_open
            else f" — all {need} required instance(s) already placed"
        )
        return {
            "ok": False,
            "error": f"step {step_no} already has its {have_here}/{need_here} "
            f"required '{pid}' instance(s){hint}",
        }
    if not premount and need and need_here == 0 and len(existing) >= need:
        return {
            "ok": False,
            "error": f"all {need} required '{pid}' instance(s) already placed — "
            "use unplace_instance to reposition instead of adding more",
        }
    if not premount and need == 0 and existing:
        # parts the manual never mounts (host/anchor referenced only in
        # check steps) exist exactly once — never let a repair loop add
        # another casing/base. Name the still-open requirements too: the
        # caller usually wants the OTHER part id that is actually missing
        open_req = [
            f"{p}@step{st_no}"
            for st_no, req in reqs.items()
            for p, n in req.items()
            if n
            > sum(1 for i in s.state.assembly.instances if i.placed_in_step == st_no and i.part_id == p)
        ]
        hint = f" — still needed elsewhere: {open_req[:6]}" if open_req else ""
        return {
            "ok": False,
            "error": f"'{pid}' is already placed and not required by any "
            f"assembly step — it is the fixed host, not a mountable part{hint}",
        }
    # Stückliste quantity is a hard bound — the kit ships exactly N copies
    # (premounts count inside it); a plan overflowing it is a steps bug,
    # not a geometry problem, and the error must say so
    kit = _kit_bound(s, part)
    if kit is not None and len(existing) >= kit:
        return {
            "ok": False,
            "error": f"parts list ships only {kit}x '{pid}' — all "
            "placed. The plan requires more copies than the kit has: fix "
            "the step plan (wrong part mapping), or unplace_instance one first",
        }
    cap = (kit if kit is not None else max(need, 2)) + 2  # bound runaway
    if len(existing) >= cap:
        return {
            "ok": False,
            "error": f"'{pid}' already has {len(existing)} instances (required {need}) — "
            "unplace_instance one first or ask_user",
        }
    if not cand.get("collision_free", True):
        return {
            "ok": False,
            "error": "candidate collides with placed parts — pick another or rotate",
            "collisions": cand.get("collisions", []),
        }
    # burial gate: a mate whose transform sinks the body into an instance it
    # is NOT connected to is physically wrong even when port gap/normals and
    # mesh collision pass (long pipe through its fitting into the host
    # behind it — invisible to boolean collision inside a hollow AABB)
    err = _burial_gate_error(s, part, pid, cand)
    if err:
        return err
    # the candidate must mate to the part the plan declares — a solver
    # preview run against the wrong fixed part (e.g. casing instead of the
    # adapter already seated on it) produces perfectly collision-free
    # candidates that bury the moving part in the host
    if not premount:
        err = _target_mismatch(s, step, pid, int(a.get("substep_idx", 0)), cand)
        if err:
            return {"ok": False, "error": err}
    transform = cand["transform"]
    inst = AssemblyInstance(
        part_id=pid,
        # next index from the max live suffix — len()+1 recycles a live id
        # after unplace (observed: two distinct 'elbow#2' instances)
        instance_id=f"{pid}#{_next_inst_no(existing)}",
        transform=[[float(v) for v in row] for row in transform],
        placed_in_step=step_no,
        substep_idx=int(a.get("substep_idx", 0)),
    )
    s.state.assembly.instances.append(inst)
    s.state.assembly.step_instances.setdefault(str(step_no), []).append(inst.instance_id)
    if cand.get("moving_port") and cand.get("fixed_port"):
        fixed_part = s.state.part_by_id(str(cand.get("fixed_part") or ""))
        fixed_port = next(
            (p for p in (fixed_part.ports if fixed_part else []) if p.name == cand["fixed_port"]),
            None,
        )
        s.state.assembly.joints.append(
            {
                "step": step_no,
                "moving_part": pid,
                "fixed_part": str(cand.get("fixed_part") or ""),
                "moving_instance": inst.instance_id,
                "fixed_instance": str(cand.get("fixed_instance") or ""),
                "moving_port": cand["moving_port"],
                "fixed_port": cand["fixed_port"],
                # solver emitted this candidate under the flipped moving-axis
                # convention — post-hoc joint audits must evaluate with the
                # same convention or normal_dot reads +1 on a valid mate
                "axis_flipped": bool(cand.get("axis_flipped")),
                # FIXED port's axis in the FIXED part's local frame — unity.py
                # rotates it by the fixed instance's transform for arrows, and
                # rotate_instance rolls about it at the port's world position
                "approach_axis": list(fixed_port.axis) if fixed_port else [0, 0, -1],
            }
        )
    s._scene_cache = None
    s.state.assembly.status = "partial"
    return {
        "ok": True,
        "placed": pid,
        "step": step_no,
        "joint_gap_mm": cand.get("joint_gap_mm"),
        "normal_dot": cand.get("normal_dot"),
    }


async def rotate_instance(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    inst, err = _resolve_inst(s, str(a.get("part_id", "")))
    if inst is None:
        return {"ok": False, "error": err}
    part = s.state.part_by_id(inst.part_id)
    deg = float(a.get("degrees", 0.0))
    if abs(deg) < 1e-6 or part is None:
        return {"ok": False, "error": "degrees must be non-zero"}
    joint = next(
        (
            j
            for j in s.state.assembly.joints
            if (j.get("moving_instance") or j.get("moving_part")) == inst.instance_id
            or (not j.get("moving_instance") and j.get("moving_part") == inst.part_id)
        ),
        None,
    )
    m = np.asarray(inst.transform, dtype=float).reshape(4, 4)
    if joint is not None:
        # roll about the mate axis at the FIXED port's world position —
        # approach_axis is stored in the fixed part's local frame
        fixed = _inst_by_id(s, str(joint.get("fixed_instance", ""))) or next(
            iter(_insts_of(s, str(joint.get("fixed_part", "")))), None
        )
        fixed_part = s.state.part_by_id(str(joint.get("fixed_part", "")))
        fixed_port = next(
            (
                p
                for p in (fixed_part.ports if fixed_part else [])
                if p.name == joint.get("fixed_port")
            ),
            None,
        )
        axis_local = np.asarray(joint.get("approach_axis") or [0, 0, 1], dtype=float)
        if fixed is not None:
            fp = np.asarray(fixed.transform, dtype=float).reshape(4, 4)
            axis_w = fp[0:3, 0:3] @ axis_local
            if fixed_port is not None:
                pivot = fp[0:3, 0:3] @ np.asarray(fixed_port.pos, dtype=float) + fp[0:3, 3]
            else:
                pivot = fp[0:3, 3]
            inst.transform = (solver.delta_about_pivot(axis_w, deg, pivot) @ m).tolist()
        else:
            inst.transform = (solver.delta_about_pivot(axis_local, deg, m[0:3, 3]) @ m).tolist()
    else:
        # free parts: rotate about own bbox center
        c = (np.asarray(part.bbox_min) + np.asarray(part.bbox_max)) / 2.0
        pivot = m[0:3, 0:3] @ c + m[0:3, 3]
        inst.transform = (solver.delta_about_pivot(np.array([0, 0, 1.0]), deg, pivot) @ m).tolist()
    s._scene_cache = None
    return {"ok": True, "instance_id": inst.instance_id, "rolled_deg": deg}


async def unplace_instance(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    inst, err = _resolve_inst(s, str(a.get("part_id", "")))
    if inst is None:
        return {"ok": False, "error": err}
    iid = inst.instance_id
    s.state.assembly.instances = [i for i in s.state.assembly.instances if i.instance_id != iid]
    for k, ids in list(s.state.assembly.step_instances.items()):
        s.state.assembly.step_instances[k] = [x for x in ids if x != iid]
    s.state.assembly.joints = [
        j
        for j in s.state.assembly.joints
        if (j.get("moving_instance") or j.get("moving_part")) != iid
        and (j.get("fixed_instance") or j.get("fixed_part")) != iid
        and j.get("moving_instance") is not None  # legacy part_id joints stay part-scoped
    ]
    # legacy joints (no instance ids) belong to the removed instance too
    s.state.assembly.joints = [
        j
        for j in s.state.assembly.joints
        if j.get("moving_instance") is not None
        or (j.get("moving_part") != inst.part_id and j.get("fixed_part") != inst.part_id)
    ]
    s._scene_cache = None
    return {"ok": True, "unplaced": iid}


async def reset_step(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step_no = str(int(a.get("step", 0)))
    ids = list(s.state.assembly.step_instances.get(step_no, []))
    for iid in ids:
        inst = _inst_by_id(s, iid)
        if inst is None:  # legacy rows stored part_id, not instance_id
            inst = next(iter(_insts_of(s, iid)), None)
        if inst is not None:
            s.state.assembly.instances.remove(inst)
    s.state.assembly.step_instances[step_no] = []
    s.state.assembly.joints = [
        j for j in s.state.assembly.joints if int(j.get("step", 0)) != int(step_no)
    ]
    s._scene_cache = None
    return {"ok": True, "unplaced": ids}


async def render_assembly(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    if s.executor is None:
        return {"ok": False, "error": "executor unavailable"}
    upto = int(a.get("step", 0) or 0)
    each = bool(a.get("each_step"))
    step_nos = (
        sorted({i.placed_in_step for i in s.state.assembly.instances} | {0})
        if each
        else [upto]
    )
    images: list[str] = []
    rendered_total = 0
    for sn in step_nos:
        entries = []
        for i, inst in enumerate(s.state.assembly.instances):
            if sn and inst.placed_in_step > sn:
                continue
            part = s.state.part_by_id(inst.part_id)
            if part is None or not part.mesh_path or not Path(part.mesh_path).exists():  # noqa: ASYNC240
                continue
            entries.append(
                {
                    "mesh_path": part.mesh_path,
                    "transform": inst.transform,
                    "color": gscene.PALETTE[i % len(gscene.PALETTE)],
                    "opacity": 1.0,
                }
            )
        if not entries:
            continue
        out = (
            s.state_dir / "renders" / f"assembly_s{sn}_{uuid.uuid4().hex[:8]}.png"
        )
        res = await s.executor.render_assembly(
            {"entries": entries, "view": str(a.get("view") or "iso"), "out_path": str(out)}
        )
        if res.get("ok"):
            images.append(str(res["image_path"]))
            rendered_total += len(entries)
    if not images:
        return {"ok": False, "error": "nothing placed (or no meshes) yet"}
    return s._images_result(images, rendered=rendered_total, steps=len(images))


async def check_collisions(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pid = str(a.get("part_id", "") or "")
    sc = _scene(s)
    if pid:
        inst, err = _resolve_inst(s, pid)
        part = s.state.part_by_id(inst.part_id) if inst else None
        if inst is None or part is None or not part.mesh_path:
            return {"ok": False, "error": err or "instance not placed or no mesh"}
        key = inst.instance_id or inst.part_id
        # check against all OTHERS — temporarily drop self
        sc.remove(key)
        try:
            rep = sc.collision_report(key, part.mesh_path, inst.transform)
        finally:
            sc.add(key, part.mesh_path, inst.transform)
        return {"ok": True, "instance_id": key, **rep}
    return {"ok": True, **sc.all_pairs_collision_report()}


def _joint_pairs(s: GuideToolSession, _ikey: Any) -> list[tuple[str, str]]:
    """Joint graph as (instance, instance) pairs, resolved through _ikey so
    legacy part-keyed joints land on their first instance."""
    pairs: list[tuple[str, str]] = []
    for j in s.state.assembly.joints:
        a = _ikey(j.get("moving_instance") or j.get("moving_part"))
        b = _ikey(j.get("fixed_instance") or j.get("fixed_part"))
        pairs.append((a, b))
    return pairs


def _buried_instances_audit(
    s: GuideToolSession,
    bboxes: dict[str, tuple[np.ndarray, np.ndarray]],
    _ikey: Any,
) -> list[dict[str, Any]]:
    return solver.buried_instances(bboxes, _joint_pairs(s, _ikey))


def _connectivity_audit(
    s: GuideToolSession,
    bboxes: dict[str, tuple[np.ndarray, np.ndarray]],
    _ikey: Any,
) -> list[dict[str, Any]]:
    """Every placed instance must be reachable from the anchor host through
    the joint graph — a fitting cluster assembled NEXT TO the casing is
    topologically complete yet physically wrong (observed: heatpump run
    joined 10 joints adapter→pipe→elbow, zero joints to the casing)."""
    insts = {i.instance_id: i for i in s.state.assembly.instances}
    return solver.disconnected_instances(bboxes, _joint_pairs(s, _ikey), sorted(insts))


def joint_geometry_audit(s: GuideToolSession) -> dict[str, Any]:
    """Deterministic physical audit — shared by the validator tool, the
    review-write guard and the emit path so every gate sees THE SAME
    numbers.

    Beyond gap/alignment this catches what boolean collision cannot:
    a part swallowed inside another part's cavity (pipe inside a union
    nut inside the casing wall intersects NO solid — but its AABB sits
    inside the union of the others)."""
    insts = {i.instance_id: i for i in s.state.assembly.instances}
    # legacy joints may reference part_id instead of instance_id — resolve
    # those keys to the (first) instance of that part so every key below
    # is an INSTANCE id; duplicating instances under their part key would
    # count each part as a foreign host containing itself
    part_to_iid: dict[str, str] = {}
    for i in s.state.assembly.instances:
        part_to_iid.setdefault(i.part_id, i.instance_id)

    def _ikey(ref: Any) -> str:
        k = str(ref or "")
        return k if k in insts else part_to_iid.get(k, k)

    bboxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for iid, inst in insts.items():
        part = s.state.part_by_id(inst.part_id)
        if part is None or inst.transform is None:
            continue
        bboxes[iid] = solver.world_bbox(
            part.bbox_min, part.bbox_max, np.asarray(inst.transform, dtype=float)
        )
    buried = _buried_instances_audit(s, bboxes, _ikey)
    disconnected = _connectivity_audit(s, bboxes, _ikey)
    # world-space dominant axis per instance — layout evidence for the
    # reviewer ("pipe vertical" is checkable against the manual, not
    # vibes): a 500mm pipe must not report horizontal when the figure
    # shows it hanging down
    axis_profiles = []
    for iid, (mn, mx) in bboxes.items():
        span = np.asarray(mx, dtype=float) - np.asarray(mn, dtype=float)
        tot = float(span.sum())
        if tot <= 0:
            continue
        i = int(np.argmax(span))
        axis_profiles.append(
            {
                "instance": iid,
                "dominant_axis": ("x", "y", "z")[i],
                "extent_mm": round(float(span[i]), 1),
                "axis_share": round(float(span[i]) / tot, 2),
            }
        )
    sib_pairs = solver.sibling_pairs(
        bboxes,
        {iid: i.part_id for iid, i in insts.items()},
        {iid: i.transform for iid, i in insts.items() if i.transform is not None},
        {p.id: p for p in s.state.parts},
    )
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    approx: list[dict[str, Any]] = []
    for j in s.state.assembly.joints:
        mv = insts.get(_ikey(j.get("moving_instance") or j.get("moving_part")))
        fx = insts.get(_ikey(j.get("fixed_instance") or j.get("fixed_part")))
        if mv is None or fx is None:
            continue
        mp = _named_port(s, mv.part_id, j.get("moving_port"))
        tp = _named_port(s, fx.part_id, j.get("fixed_port"))
        if mp is None or tp is None:
            continue
        if j.get("axis_flipped"):
            mp = mp.model_copy(update={"axis": [-a for a in mp.axis]})
        gap, ndot = solver.evaluate_mate(
            mp,
            np.asarray(mv.transform, dtype=float).reshape(4, 4),
            tp,
            np.asarray(fx.transform, dtype=float).reshape(4, 4),
            moving_part=s.state.part_by_id(mv.part_id),
            fixed_part=s.state.part_by_id(fx.part_id),
        )
        # plug-in-socket joints stay engaged as long as the plug tip is
        # inside the bore — a union nut never bottoms out. Tolerance =
        # the receiving socket's depth (whichever side is female); face
        # joints (both flush) keep the tight post-hoc limit.
        socket_depth = 0.0
        if str(getattr(tp, "gender", "")) == "female":
            socket_depth = float(getattr(tp, "depth", 0.0) or 0.0)
        elif str(getattr(mp, "gender", "")) == "female":
            socket_depth = float(getattr(mp, "depth", 0.0) or 0.0)
        allow = max(solver.GAP_GATE_MM * 20, min(socket_depth, 30.0))
        ok = gap < allow and ndot <= -0.95  # post-hoc tolerance wider than gate
        row = {
            "joint": f"{mv.instance_id or mv.part_id}→{fx.instance_id or fx.part_id}",
            "gap_mm": round(gap, 4),
            "normal_dot": round(ndot, 4),
            "allow_mm": round(allow, 2),
            "ok": ok,
        }
        # a joint seated on a SYNTHESIZED thread site is an approximation —
        # but only when the seat tier actually fired: a female mouth onto a
        # male stub nested in a socket whose drawn OD the bore can't take
        # (the stub stands in for a thread the mesh doesn't model). A slim
        # nested stub that really fits the bore is a plain spigot joint —
        # extraction just emitted the tube end as nested, nothing inferred.
        if (
            mp is not None
            and solver._thread_seat(mp, tp) is not None
            and solver._inner_r(mp) * 1.15 + 0.5 < solver._outer_r(tp)
        ):
            row["approximated_interface"] = True
            approx.append(
                {
                    **row,
                    "note": "joint seats on an approximated thread site inside "
                    "an enclosing socket — verify against the manual figure "
                    "that this is the intended connection, not a fitting "
                    "sunk into an oversized opening",
                }
            )
        rows.append(row)
        if not ok:
            bad.append(row)
    return {
        "joints": rows,
        "bad_joints": bad,
        "buried_instances": buried,
        "disconnected_instances": disconnected,
        "axis_profiles": axis_profiles,
        "sibling_pairs": sib_pairs,
        "approximated_joints": approx,
    }


async def validate_assembly_geometry(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    audit = joint_geometry_audit(s)
    col = _scene(s).all_pairs_collision_report()
    return {
        "ok": not audit["bad_joints"]
        and col["ok"]
        and not audit["buried_instances"]
        and not audit["disconnected_instances"],
        **audit,
        "collisions": col["collisions"],
        # a part buried inside another's AABB is WRONG even when boolean
        # collision sees no intersection (cavity-hidden = invisible to
        # manifold intersection but physically impossible)
    }


async def validate_assembly_completeness(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    issues = _completeness_issues(s)
    placed = sorted(i.instance_id or i.part_id for i in s.state.assembly.instances)
    return {"ok": not issues, "issues": issues, "placed": placed}


def _done_guard(s: GuideToolSession) -> dict[str, Any] | None:
    if not s.state.assembly.instances:
        return {"ok": False, "error": "nothing placed — run solve_*_preview + place_instance"}
    # completeness gate: every required INSTANCE per step must be placed —
    # a part placed once does not satisfy later steps needing their own copy
    issues = _completeness_issues(s)
    if issues:
        return {"ok": False, "error": "incomplete assembly: " + "; ".join(issues[:6])}
    # physical gate: done must not ship a scene the audit calls wrong —
    # bad joints, buried parts, floating clusters, real solid
    # intersections. The reviewer reads the same numbers; the mission
    # must fix them HERE, not explain them away.
    audit = joint_geometry_audit(s)
    col = _scene(s).all_pairs_collision_report()
    probs: list[str] = []
    probs += [f"bad joint {r['joint']}" for r in audit["bad_joints"][:4]]
    probs += [f"{b['instance']} buried" for b in audit["buried_instances"][:4]]
    probs += [d["issue"] for d in audit["disconnected_instances"][:4]]
    probs += [f"collision {c}" for c in col["collisions"][:4]]
    if probs:
        return {"ok": False, "error": "unhealthy assembly: " + "; ".join(probs)}
    s.state.assembly.status = "complete"
    return None


def _norm2(x: str | None) -> str:
    """Canonical label form: case/punctuation-insensitive PLUS umlaut-folded
    both ways (ä→a, ae→a) — manuals and extractors mix "Wärmepumpe" with
    "Waermepumpe" freely. Applied to every alias/key AND lookup so a
    digraph label still resolves the umlaut part name."""
    import re

    n = re.sub(r"[^a-z0-9äöüß]+", " ", (x or "").lower()).strip()
    n = n.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    return re.sub(r"ae|oe|ue", lambda m: m.group(0)[0], n)


from cadx.guide.tools import DONE_GUARDS  # noqa: E402

DONE_GUARDS["assembly"] = _done_guard

HANDLERS = {
    "find_matching_ports": find_matching_ports,
    "solve_attachment_preview": solve_attachment_preview,
    "solve_free_preview": solve_free_preview,
    "solve_env_preview": solve_env_preview,
    "place_instance": place_instance,
    "rotate_instance": rotate_instance,
    "unplace_instance": unplace_instance,
    "reset_step": reset_step,
    "render_assembly": render_assembly,
    "check_collisions": check_collisions,
    "validate_assembly_geometry": validate_assembly_geometry,
    "validate_assembly_completeness": validate_assembly_completeness,
}
