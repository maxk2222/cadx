"""Deterministic pre-assembly planner (KIARA GuidePlanner).

Runs between the ``parts`` and ``assembly`` missions — NO LLM involved.
It reads the step plan (sub-step attach actions + required_parts) and
drives the exact same solver tools the assembly agent would use:
port matching → attachment preview → place_instance, with env/free
placement as fallback. The LLM assembly mission then only refines:
fix collisions, re-target ambiguous joints, place whatever the planner
could not resolve.

Deterministic choices, in order:
- one placement task per ``place|attach|insert|fasten|mount`` sub-step;
  steps without placement sub-steps get one task per ``required_parts``
  entry (counted — a part required twice yields two tasks)
- the sub-step's ``target`` resolves to the NEWEST placed instance of
  that part which still offers a compatible free port (assembly chains
  almost always continue from the last-mounted piece)
- port pair = first ranked pair from ``solver.find_matching_ports``;
  placement = first collision-free candidate from ``solve_attachment``
- no ports / no candidates → ``solve_env_preview`` (bbox + step-text
  hint) as the deterministic free-placement fallback
- already-satisfied requirements are skipped → resume-safe
"""

from __future__ import annotations

from typing import Any

import structlog

from cadx.guide.assembly_tools import (
    _PLACE_ACTIONS,
    _resolve_name,
    _step_requirements,
    find_matching_ports,
    place_instance,
    solve_attachment_preview,
    solve_env_preview,
)
from cadx.guide.session import GuideToolSession

log = structlog.get_logger("guide.preassembly")


def _plan(sess: GuideToolSession) -> list[dict[str, Any]]:
    """Ordered placement tasks: {step, substep_idx, part, target}."""
    tasks: list[dict[str, Any]] = []
    for st in sorted(sess.state.steps, key=lambda s: s.number):
        if st.action_type != "assembly":
            continue
        subs = [(i, ss) for i, ss in enumerate(st.sub_steps) if ss.action in _PLACE_ACTIONS]
        if subs:
            tasks.extend(
                {
                    "step": st.number,
                    "substep_idx": i,
                    "part": ss.part,
                    "target": ss.target,
                }
                for i, ss in subs
            )
        else:
            for i, name in enumerate(st.required_parts):
                tasks.append({"step": st.number, "substep_idx": i, "part": name, "target": ""})
    return tasks


def _already_satisfied(sess: GuideToolSession, step_no: int, pid: str) -> bool:
    """Does this step already hold all required instances of ``pid``?
    (Instances from EARLIER steps never satisfy a later step's quota —
    each step needs its own physical occurrences.)"""
    reqs = _step_requirements(sess).get(step_no, {})
    need = reqs.get(pid, 1)
    have = sum(
        1 for i in sess.state.assembly.instances if i.placed_in_step == step_no and i.part_id == pid
    )
    return have >= need


def _pick_target_instance(
    sess: GuideToolSession, target_name: str, moving_pid: str = ""
) -> str | None:
    """Resolve a sub-step target label to an instance_id.

    Oldest-first pairing: prefer the EARLIEST placed instance of the
    target part that has not yet received a joint from this moving part
    type. Symmetric manuals (left/right adapter each taking a pipe) then
    pair 1:1 by order instead of every connection piling onto the newest
    instance — which produced star topologies and coincident part heaps.
    Chains still continue correctly because a different moving part type
    is a different pair."""
    tid = _resolve_name(sess, target_name) if target_name else None
    if not tid:
        return None
    insts = [i for i in sess.state.assembly.instances if i.part_id == tid]
    if not insts:
        return None
    used_ports: dict[str, set[str]] = {}
    used_types: dict[str, set[str]] = {}
    for j in sess.state.assembly.joints:
        fi = str(j.get("fixed_instance") or "")
        if fi:
            used_ports.setdefault(fi, set()).add(str(j.get("fixed_port")))
            used_types.setdefault(fi, set()).add(str(j.get("moving_part") or ""))

    def _free(inst: Any) -> bool:
        part = sess.state.part_by_id(inst.part_id)
        if not (part and part.ports):
            return True
        free = {p.name for p in part.ports} - used_ports.get(
            inst.instance_id or inst.part_id, set()
        )
        return bool(free)

    candidates = [i for i in insts if _free(i)]
    for inst in candidates:
        iid = inst.instance_id or inst.part_id
        if moving_pid and moving_pid not in used_types.get(iid, set()):
            return iid
    return (candidates or insts)[-1].instance_id or (candidates or insts)[-1].part_id


async def run_preassembly(sess: GuideToolSession) -> dict[str, Any]:
    """Place what the step plan determines — returns (and persists) a report."""
    report: dict[str, Any] = {"placed": [], "skipped": [], "failed": []}
    for task in _plan(sess):
        pid = _resolve_name(sess, task["part"])
        if not pid:
            report["failed"].append({**task, "reason": f"unresolved part '{task['part']}'"})
            continue
        if _already_satisfied(sess, task["step"], pid):
            report["skipped"].append({**task, "reason": "already satisfied"})
            continue
        placed = await _try_attach(sess, task, pid)
        if not placed:
            placed = await _try_env(sess, task, pid)
        if placed:
            report["placed"].append({**task, "part_id": pid, "instance_id": placed})
        else:
            report["failed"].append({**task, "part_id": pid, "reason": "no candidate"})
    sess.state.assembly.planner = {
        "ran": True,
        "placed": len(report["placed"]),
        "skipped": len(report["skipped"]),
        "failed": report["failed"][:10],
    }
    sess.state.save(sess.state_dir)
    log.info(
        "guide.preassembly.done",
        placed=len(report["placed"]),
        skipped=len(report["skipped"]),
        failed=len(report["failed"]),
    )
    return report


async def _try_attach(sess: GuideToolSession, task: dict[str, Any], pid: str) -> str | None:
    """Port-driven attach onto a resolved target instance; None if it
    cannot work (no target / no pairs / all candidates collide)."""
    target_iid = _pick_target_instance(sess, task["target"], moving_pid=pid)
    if not target_iid:
        return None
    ports = await find_matching_ports(sess, {"part_id": pid, "target_part_id": target_iid})
    if not ports.get("ok") or not ports.get("candidates"):
        return None
    for pair in ports["candidates"][:3]:
        prev = await solve_attachment_preview(
            sess,
            {
                "part_id": pid,
                "moving_port": pair.get("moving_port", ""),
                "target_part_id": target_iid,
                "target_port": pair.get("fixed_port", ""),
            },
        )
        if not prev.get("ok"):
            continue
        for cand in prev.get("candidates", []):
            if not cand.get("collision_free", True):
                continue
            res = await place_instance(
                sess,
                {
                    "part_id": pid,
                    "candidate_id": cand["candidate_id"],
                    "step": task["step"],
                    "substep_idx": task["substep_idx"],
                },
            )
            if res.get("ok"):
                insts = [i for i in sess.state.assembly.instances if i.part_id == pid]
                return insts[-1].instance_id if insts else None
    return None


async def _try_env(sess: GuideToolSession, task: dict[str, Any], pid: str) -> str | None:
    prev = await solve_env_preview(sess, {"part_id": pid, "step": task["step"]})
    if not prev.get("ok"):
        return None
    for cand in prev.get("candidates", []):
        if not cand.get("collision_free", True):
            continue
        res = await place_instance(
            sess,
            {
                "part_id": pid,
                "candidate_id": cand["candidate_id"],
                "step": task["step"],
                "substep_idx": task["substep_idx"],
            },
        )
        if res.get("ok"):
            insts = [i for i in sess.state.assembly.instances if i.part_id == pid]
            return insts[-1].instance_id if insts else None
    return None
