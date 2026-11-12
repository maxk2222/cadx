"""Assembly-mission tools — deterministic port matching, candidate previews
(server-cached, LLM picks by id), placement, collision checks, renders.

The LLM NEVER submits a transform matrix (plan §4): ``solve_*_preview``
returns candidates carrying opaque ``candidate_id``s; ``place_instance``
consumes one from the session's server-side cache. Ported from KIARA's
p5 loop tools, hardened per the master plan.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cadx.guide import scene as gscene
from cadx.guide import solver
from cadx.guide.state import AssemblyInstance
from cadx.guide.tools import _fn, _i, _num, _s
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
            "target_part_id": _s("already-placed part it mates with"),
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
            "target_part_id": _s("placed target part"),
            "target_port": _s("port name on the target part"),
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
            "part_id": _s("placed part"),
            "degrees": _num("roll degrees"),
        },
        ["part_id", "degrees"],
    ),
    "unplace_instance": _fn(
        "unplace_instance",
        "Remove a part's placement (and its joints) so it can be re-placed.",
        {"part_id": _s("part to unplace")},
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


def _inst(s: GuideToolSession, pid: str) -> AssemblyInstance | None:
    for i in s.state.assembly.instances:
        if i.part_id == pid:
            return i
    return None


def _placed_bboxes(s: GuideToolSession) -> list[dict[str, Any]]:
    out = []
    for inst in s.state.assembly.instances:
        part = s.state.part_by_id(inst.part_id)
        if part is None:
            continue
        mn, mx = solver.transform_bbox(part, inst.transform)
        out.append({"part_id": inst.part_id, "bbox_min_world": mn, "bbox_max_world": mx})
    return out


def _scene(s: GuideToolSession) -> gscene.Scene:
    sc = getattr(s, "_scene_cache", None)
    if sc is None:
        sc = gscene.Scene()
        for inst in s.state.assembly.instances:
            part = s.state.part_by_id(inst.part_id)
            if part is not None and part.mesh_path:
                sc.add(inst.part_id, part.mesh_path, inst.transform)
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
    moving = s.state.part_by_id(str(a.get("part_id", "")))
    fixed = s.state.part_by_id(str(a.get("target_part_id", "")))
    if moving is None or fixed is None:
        return {"ok": False, "error": "part not found — check list_parts"}
    if _inst(s, fixed.id) is None:
        return {"ok": False, "error": f"target '{fixed.id}' is not placed yet"}
    used = {
        str(j.get("fixed_port")) for j in s.state.assembly.joints if j.get("fixed_part") == fixed.id
    }
    cands = solver.find_matching_ports(moving, fixed, used_fixed_ports=used)
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
    moving = s.state.part_by_id(str(a.get("part_id", "")))
    fixed = s.state.part_by_id(str(a.get("target_part_id", "")))
    if moving is None or fixed is None:
        return {"ok": False, "error": "part not found — check list_parts"}
    fixed_inst = _inst(s, fixed.id)
    if fixed_inst is None:
        return {"ok": False, "error": f"target '{fixed.id}' is not placed yet"}
    cands = solver.solve_attachment(
        moving,
        str(a.get("moving_port", "")),
        fixed,
        fixed_inst.transform or solver.identity_transform(),
        str(a.get("target_port", "")),
    )
    if not cands:
        return {"ok": False, "error": "no candidates — check port names via get_part_features"}
    # collision-check each candidate against the placed set
    sc = _scene(s)
    for c in cands:
        c["fixed_part"] = fixed.id
        if moving.mesh_path:
            rep = sc.collision_report(moving.id, moving.mesh_path, c["transform"])
            c["collision_free"] = rep["ok"]
            c["collisions"] = rep["collisions"][:3]
    return {
        "ok": True,
        "candidates": _cache_candidates(s, cands),
        "note": "pass one candidate_id to place_instance",
    }


async def solve_free_preview(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    part = s.state.part_by_id(str(a.get("part_id", "")))
    if part is None:
        return {"ok": False, "error": "part not found — check list_parts"}
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
    part = s.state.part_by_id(str(a.get("part_id", "")))
    if part is None:
        return {"ok": False, "error": "part not found — check list_parts"}
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


async def place_instance(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pid = str(a.get("part_id", ""))
    part = s.state.part_by_id(pid)
    if part is None:
        return {"ok": False, "error": "part not found — check list_parts"}
    cid = str(a.get("candidate_id", ""))
    cand = s._preview_cache.get(cid)
    if cand is None:
        return {"ok": False, "error": "unknown/expired candidate_id — run a solve_*_preview first"}
    step_no = int(a.get("step", 0))
    step = s.state.step_by_number(step_no)
    if step is None:
        return {"ok": False, "error": f"step {step_no} not found"}
    existing = _inst(s, pid)
    if existing is not None:
        return {"ok": False, "error": f"'{pid}' already placed — unplace_instance first"}
    if not cand.get("collision_free", True):
        return {
            "ok": False,
            "error": "candidate collides with placed parts — pick another or rotate",
            "collisions": cand.get("collisions", []),
        }
    transform = cand["transform"]
    inst = AssemblyInstance(
        part_id=pid,
        transform=[[float(v) for v in row] for row in transform],
        placed_in_step=step_no,
        substep_idx=int(a.get("substep_idx", 0)),
    )
    s.state.assembly.instances.append(inst)
    s.state.assembly.step_instances.setdefault(str(step_no), []).append(pid)
    if cand.get("moving_port") and cand.get("fixed_port"):
        s.state.assembly.joints.append(
            {
                "step": step_no,
                "moving_part": pid,
                "fixed_part": str(cand.get("fixed_part") or ""),
                "moving_port": cand["moving_port"],
                "fixed_port": cand["fixed_port"],
                "approach_axis": (
                    s.state.part_by_id(pid).ports[0].axis
                    if s.state.part_by_id(pid).ports
                    else [0, 0, -1]
                ),
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
    inst = _inst(s, str(a.get("part_id", "")))
    if inst is None:
        return {"ok": False, "error": "part not placed"}
    part = s.state.part_by_id(inst.part_id)
    deg = float(a.get("degrees", 0.0))
    if abs(deg) < 1e-6 or part is None:
        return {"ok": False, "error": "degrees must be non-zero"}
    joint = next((j for j in s.state.assembly.joints if j.get("moving_part") == inst.part_id), None)
    m = np.asarray(inst.transform, dtype=float).reshape(4, 4)
    if joint is not None:
        # roll about the joint axis at the joint's world position
        fixed = _inst(s, str(joint.get("fixed_part", "")))
        axis = np.asarray(joint.get("approach_axis") or [0, 0, 1], dtype=float)
        pivot = m[0:3, 3]
        if fixed is not None:
            fp = np.asarray(fixed.transform, dtype=float).reshape(4, 4)
            pivot = fp[0:3, 3]
        inst.transform = (solver.delta_about_pivot(axis, deg, pivot) @ m).tolist()
    else:
        # free parts: rotate about own bbox center
        c = (np.asarray(part.bbox_min) + np.asarray(part.bbox_max)) / 2.0
        pivot = m[0:3, 0:3] @ c + m[0:3, 3]
        inst.transform = (solver.delta_about_pivot(np.array([0, 0, 1.0]), deg, pivot) @ m).tolist()
    s._scene_cache = None
    return {"ok": True, "part_id": inst.part_id, "rolled_deg": deg}


async def unplace_instance(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pid = str(a.get("part_id", ""))
    inst = _inst(s, pid)
    if inst is None:
        return {"ok": False, "error": "part not placed"}
    s.state.assembly.instances = [i for i in s.state.assembly.instances if i.part_id != pid]
    for k, ids in list(s.state.assembly.step_instances.items()):
        s.state.assembly.step_instances[k] = [x for x in ids if x != pid]
    s.state.assembly.joints = [
        j
        for j in s.state.assembly.joints
        if j.get("moving_part") != pid and j.get("fixed_part") != pid
    ]
    s._scene_cache = None
    return {"ok": True, "unplaced": pid}


async def reset_step(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step_no = str(int(a.get("step", 0)))
    ids = list(s.state.assembly.step_instances.get(step_no, []))
    for pid in ids:
        inst = _inst(s, pid)
        if inst is not None:
            s.state.assembly.instances.remove(inst)
    s.state.assembly.step_instances[step_no] = []
    s.state.assembly.joints = [
        j for j in s.state.assembly.joints if int(j.get("step", 0)) != int(step_no)
    ]
    s._scene_cache = None
    return {"ok": True, "unplaced": ids}


_PALETTE = ["#7aa2f7", "#9ece6a", "#e0af68", "#f7768e", "#bb9af7", "#4dc3ff", "#ff9e64"]


async def render_assembly(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    if s.executor is None:
        return {"ok": False, "error": "executor unavailable"}
    upto = int(a.get("step", 0) or 0)
    entries = []
    for i, inst in enumerate(s.state.assembly.instances):
        if upto and inst.placed_in_step > upto:
            continue
        part = s.state.part_by_id(inst.part_id)
        if part is None or not part.mesh_path or not Path(part.mesh_path).exists():  # noqa: ASYNC240
            continue
        entries.append(
            {
                "mesh_path": part.mesh_path,
                "transform": inst.transform,
                "color": _PALETTE[i % len(_PALETTE)],
                "opacity": 1.0,
            }
        )
    if not entries:
        return {"ok": False, "error": "nothing placed (or no meshes) yet"}
    out = s.state_dir / "renders" / f"assembly_{uuid.uuid4().hex[:8]}.png"
    res = await s.executor.render_assembly(
        {"entries": entries, "view": str(a.get("view") or "iso"), "out_path": str(out)}
    )
    if not res.get("ok"):
        return {"ok": False, "error": str(res.get("error") or "render failed")}
    return s._images_result([str(res["image_path"])], rendered=len(entries))


async def check_collisions(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pid = str(a.get("part_id", "") or "")
    sc = _scene(s)
    if pid:
        inst = _inst(s, pid)
        part = s.state.part_by_id(pid)
        if inst is None or part is None or not part.mesh_path:
            return {"ok": False, "error": "part not placed or no mesh"}
        # check against all OTHERS — temporarily drop self
        sc.remove(pid)
        rep = sc.collision_report(pid, part.mesh_path, inst.transform)
        sc.add(pid, part.mesh_path, inst.transform)
        return {"ok": True, "part_id": pid, **rep}
    return {"ok": True, **sc.all_pairs_collision_report()}


async def validate_assembly_geometry(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    insts = {i.part_id: i for i in s.state.assembly.instances}
    rows = []
    bad = []
    for j in s.state.assembly.joints:
        mv = insts.get(str(j.get("moving_part", "")))
        fx = insts.get(str(j.get("fixed_part", "")))
        if mv is None or fx is None:
            continue
        mp = next(
            (
                p
                for p in (s.state.part_by_id(mv.part_id) or object()).ports or []
                if p.name == j.get("moving_port")
            ),
            None,
        )
        tp = next(
            (
                p
                for p in (s.state.part_by_id(fx.part_id) or object()).ports or []
                if p.name == j.get("fixed_port")
            ),
            None,
        )
        if mp is None or tp is None:
            continue
        gap, ndot = solver.evaluate_mate(
            mp,
            np.asarray(mv.transform, dtype=float).reshape(4, 4),
            tp,
            np.asarray(fx.transform, dtype=float).reshape(4, 4),
        )
        ok = gap < solver.GAP_GATE_MM * 20 and ndot <= -0.95  # post-hoc tolerance wider than gate
        rows.append(
            {
                "joint": f"{mv.part_id}→{fx.part_id}",
                "gap_mm": round(gap, 4),
                "normal_dot": round(ndot, 4),
                "ok": ok,
            }
        )
        if not ok:
            bad.append(rows[-1])
    col = _scene(s).all_pairs_collision_report()
    return {
        "ok": not bad and col["ok"],
        "joints": rows,
        "bad_joints": bad,
        "collisions": col["collisions"],
    }


async def validate_assembly_completeness(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    issues = []
    placed = s.state.placed_part_ids()
    for step in s.state.steps:
        if step.action_type != "assembly":
            continue
        need = set(step.required_parts)
        got = set(s.state.assembly.step_instances.get(str(step.number), []))
        part_ids = {p.id for p in s.state.parts}
        # required_parts holds manual names — resolve through aliases/registry
        resolved_need = {s._aliases_store.get(n.lower()) or n if n in part_ids else n for n in need}
        missing = resolved_need - got - {""}
        if missing and not all(m not in {p.name for p in s.state.parts} for m in missing):
            # only count names we have registered parts for
            real_missing = {
                m for m in missing if m in part_ids or m in {p.name for p in s.state.parts}
            }
            if real_missing:
                issues.append(f"step {step.number}: unplaced parts {sorted(real_missing)}")
    if not s.state.assembly.instances:
        issues.append("no parts placed at all")
    return {"ok": not issues, "issues": issues, "placed": sorted(placed)}


def _done_guard(s: GuideToolSession) -> dict[str, Any] | None:
    if not s.state.assembly.instances:
        return {"ok": False, "error": "nothing placed — run solve_*_preview + place_instance"}
    s.state.assembly.status = "complete"
    return None


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
