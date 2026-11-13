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

import numpy as np
import structlog

from cadx.guide import solver
from cadx.guide.assembly_tools import (
    _PLACE_ACTIONS,
    _resolve_name,
    _step_requirements,
    place_instance,
    solve_attachment_preview,
    solve_env_preview,
)
from cadx.guide.session import GuideToolSession

log = structlog.get_logger("guide.preassembly")


def _plan(sess: GuideToolSession) -> list[dict[str, Any]]:
    """Ordered placement tasks: {step, substep_idx, part, target}.

    Additionally emits ANCHOR tasks: any registered part that is referenced
    anywhere (sub-step part/target, required_parts — including check steps
    like "Gehäuse sitzt auf Gerät") but never receives a placement task is
    the assembly's base. Without an instance, every target that means one
    of its features (connection stubs, bosses, ports) resolves to nothing
    and the build floats in env-fallback space. Anchors are inserted at the
    earliest referencing step and ordered first there so later tasks can
    attach onto them."""
    tasks: list[dict[str, Any]] = []
    referenced: dict[str, int] = {}  # resolved pid -> earliest step referencing it

    def _ref(name: str | None, step_no: int) -> None:
        if not name:
            return
        pid = _resolve_name(sess, name)
        if pid and pid not in referenced:
            referenced[pid] = step_no

    for st in sorted(sess.state.steps, key=lambda s: s.number):
        # every mention counts toward anchor detection — even on non-assembly steps
        for ss in st.sub_steps:
            _ref(ss.part, st.number)
            _ref(ss.target, st.number)
        for name in st.required_parts:
            _ref(name, st.number)
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
                    "target2": getattr(ss, "target2", ""),
                    "text": ss.text,
                }
                for i, ss in subs
            )
        else:
            for i, name in enumerate(st.required_parts):
                tasks.append({"step": st.number, "substep_idx": i, "part": name, "target": ""})
    planned = {_resolve_name(sess, t["part"]) for t in tasks}
    anchors = [
        {"step": step_no, "substep_idx": -1, "part": pid, "target": "", "anchor": True}
        for pid, step_no in referenced.items()
        if pid not in planned
    ]
    # anchors first within their step — other tasks may attach onto them
    tasks.extend(anchors)
    tasks.sort(key=lambda t: (t["step"], not t.get("anchor")))
    # pre-mounted duplicates: a place-task whose TARGET text carries a
    # side qualifier ("unterer linker Adapter") refers to an instance at
    # THAT side — when the part's own placements all happen later or at
    # other sides, the manual means a pre-mounted piece the steps never
    # place. Spawn a mount task for it onto the earliest referenceable
    # host before the referencing step.
    return _spawn_premounts(sess, tasks)


_SIDE_WORDS = {
    "links", "link", "left", "rechts", "recht", "right",
    "unten", "unter", "lower", "bottom", "oben", "ober", "upper", "top",
}


def _side_words(text: str) -> set[str]:
    toks = (text or "").lower().split()
    return {w for w in _SIDE_WORDS for tok in toks if tok.startswith(w)}


def _spawn_premounts(
    sess: GuideToolSession, tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Side-qualified targets ("unterer linker Adapter") mean an instance
    at THAT side — when the part's planned placements cover other sides
    only, the manual means a pre-mounted piece the steps never place:
    spawn a mount task for it onto the earliest host part, before the
    referencing step."""
    host_pid = next((t["part"] for t in tasks if t.get("anchor")), None)
    if host_pid is None:
        host_pid = tasks[0]["part"] if tasks else ""
    planned_sides: dict[str, set[str]] = {}
    for t in tasks:
        pid = _resolve_name(sess, t["part"])
        if pid:
            planned_sides.setdefault(pid, set()).update(
                _side_words(f"{t.get('target') or ''} {t.get('text') or ''}")
            )
    extra: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("anchor") or task.get("premount"):
            continue
        tpid = _resolve_name(sess, task.get("target"))
        if not tpid or tpid in (_resolve_name(sess, task["part"]), host_pid):
            continue
        missing = _side_words(task.get("target") or "") - planned_sides.get(tpid, set())
        missing -= {s for _, s in ((e["part"], e.get("text")) for e in extra) if _ == tpid for s in _side_words(s or "")}
        if not missing:
            continue
        extra.append(
            {
                "step": task["step"],
                "substep_idx": -1,
                "part": tpid,
                "target": host_pid,
                "target2": "",
                "text": task.get("target") or "",
                "premount": True,
            }
        )
        planned_sides.setdefault(tpid, set()).update(missing)
    return sorted(
        tasks + extra,
        key=lambda t: (t["step"], not t.get("anchor"), not t.get("premount")),
    )


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
        mi = str(j.get("moving_instance") or "")
        if fi:
            used_ports.setdefault(fi, set()).add(str(j.get("fixed_port")))
            used_types.setdefault(fi, set()).add(str(j.get("moving_part") or ""))
        if mi:
            used_ports.setdefault(mi, set()).add(str(j.get("moving_port")))

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
    tasks = _plan(sess)
    for ti, task in enumerate(tasks):
        pid = _resolve_name(sess, task["part"])
        if not pid:
            report["failed"].append({**task, "reason": f"unresolved part '{task['part']}'"})
            continue
        if _already_satisfied(sess, task["step"], pid):
            report["skipped"].append({**task, "reason": "already satisfied"})
            continue
        placed = await _try_bridge(sess, task, pid)
        if not placed:
            placed = await _try_attach(sess, task, pid, _lookahead(sess, tasks, ti, pid))
        if not placed and task.get("anchor"):
            placed = await _place_anchor(sess, task, pid)
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


# direction words manuals use to disambiguate symmetric mounts —
# axis: X = across the device (left/right), Y = depth (+front), Z = up
_DIR_TOKENS: dict[str, tuple[int, float]] = {
    "links": (0, -1.0), "link": (0, -1.0), "left": (0, -1.0),
    "rechts": (0, 1.0), "recht": (0, 1.0), "right": (0, 1.0),
    "vorne": (1, 1.0), "vorder": (1, 1.0), "front": (1, 1.0),
    "hinten": (1, -1.0), "hinter": (1, -1.0), "rear": (1, -1.0), "back": (1, -1.0),
    "oben": (2, 1.0), "ober": (2, 1.0), "upper": (2, 1.0), "top": (2, 1.0),
    "unten": (2, -1.0), "unter": (2, -1.0), "lower": (2, -1.0), "bottom": (2, -1.0),
}

# moving-part orientation words — "vertikales Rohr" wants the part's long
# axis along Z, "horizontal" wants it ⊥Z
_VERTICAL = {"vertikal", "senkrecht", "vertical"}
_HORIZONTAL = {"horizontal", "waagerecht", "waagrecht"}


def _dir_axes(text: str) -> list[tuple[int, float]]:
    """One (axis, sign) hint per axis found in a target/location phrase."""
    t = " " + (text or "").lower() + " "
    out: list[tuple[int, float]] = []
    seen: set[int] = set()
    for tok, (ax, sg) in _DIR_TOKENS.items():
        if ax in seen:
            continue
        if tok in t:
            out.append((ax, sg))
            seen.add(ax)
    return out


def _orient_word(text: str) -> str | None:
    t = " " + (text or "").lower() + " "
    if any(w in t for w in _VERTICAL):
        return "V"
    if any(w in t for w in _HORIZONTAL):
        return "H"
    return None


def _world_pos(transform: list[list[float]], pos: list[float]) -> list[float]:
    return [
        sum(transform[r][c] * pos[c] for c in range(3)) + transform[r][3]
        for r in range(3)
    ]


def _world_dir(transform: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(transform[r][c] * vec[c] for c in range(3)) for r in range(3)]


def _lookahead(
    sess: GuideToolSession,
    tasks: list[dict[str, Any]],
    ti: int,
    pid: str,
) -> tuple[list[tuple[int, float]], str | None]:
    """Direction/orient requirements of the NEXT task mating onto this
    part — the elbow being placed now must leave a free port facing where
    the 'vertikales Rohr' will need it. Applies only when the next task's
    target resolves to this part or to nothing (chain fallback); a target
    resolving elsewhere constrains that part, not this one."""
    for nxt in tasks[ti + 1 :]:
        npid = _resolve_name(sess, nxt["part"])
        if not npid or npid == pid:
            continue
        tpid = _resolve_name(sess, nxt.get("target") or "")
        if tpid is not None and tpid != pid:
            return [], None
        text = f"{nxt.get('target') or ''} {nxt.get('text') or ''}"
        return _dir_axes(text), _orient_word(text)
    return [], None


def _orient_score(
    moving: Any,
    mated_port: str,
    transform: list[list[float]],
    axes: list[tuple[int, float]],
    orient: str | None,
    next_axes: tuple[list[tuple[int, float]], str | None] = ([], None),
) -> float:
    """How well the candidate leaves the part ready for the NEXT mate:
    remaining ports should face the hinted direction (an elbow mounted
    'unten' keeps its free socket pointing down for the vertical pipe),
    and 'vertikal'/'horizontal' pin the part's dominant bbox axis."""
    score = 0.0
    n_axes, n_orient = next_axes
    free = [p for p in moving.ports if p.name != mated_port]
    if axes:
        best = 0.0
        for p in free:
            w = _world_dir(transform, p.axis)
            best = max(best, max(sg * w[ax] for ax, sg in axes))
        score += best
    if n_axes or n_orient:
        # lookahead: a free port must aim where the NEXT part approaches —
        # ports mate anti-parallel, so the incoming part extends along the
        # free port's own axis direction
        best = 0.0
        for p in free:
            w = _world_dir(transform, p.axis)
            v = max(sg * w[ax] for ax, sg in n_axes) if n_axes else 0.0
            if n_orient == "V":
                v = max(v, w[2])
            elif n_orient == "H":
                v = max(v, 1.0 - abs(w[2]))
            best = max(best, v)
        score += 0.8 * best
    if orient:
        dims = [
            moving.bbox_max[i] - moving.bbox_min[i] for i in range(3)
        ]
        dom = dims.index(max(dims))
        w = _world_dir(transform, [1.0 if i == dom else 0.0 for i in range(3)])
        z = abs(w[2])
        score += z if orient == "V" else -z
    else:
        # no orientation word: assemblies overwhelmingly grow upward — a
        # weak vertical bias keeps long parts from sprawling sideways off
        # a compact cluster, but explicit text always wins over it
        dims = [
            moving.bbox_max[i] - moving.bbox_min[i] for i in range(3)
        ]
        dom = dims.index(max(dims))
        w = _world_dir(transform, [1.0 if i == dom else 0.0 for i in range(3)])
        score += 0.4 * abs(w[2])
    return score


def _bbox_corners_world(
    bbox_min: list[float], bbox_max: list[float], transform: list[list[float]]
) -> list[list[float]]:
    return [
        _world_pos(transform, [x, y, z])
        for x in (bbox_min[0], bbox_max[0])
        for y in (bbox_min[1], bbox_max[1])
        for z in (bbox_min[2], bbox_max[2])
    ]


def _cluster_bbox_growth(
    sess: GuideToolSession, moving: Any, transform: list[list[float]]
) -> float:
    """How much the candidate inflates the placed set's world bbox — a
    compactness prior: attaching a long part sideways into empty space is
    a worse read of any manual than keeping the build dense."""
    if not transform:
        return 0.0
    corners = _bbox_corners_world(moving.bbox_min, moving.bbox_max, transform)
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    for inst in sess.state.assembly.instances:
        fp = sess.state.part_by_id(inst.part_id)
        if fp is None or not inst.transform:
            continue
        for c in _bbox_corners_world(fp.bbox_min, fp.bbox_max, inst.transform):
            for i in range(3):
                lo[i] = min(lo[i], c[i])
                hi[i] = max(hi[i], c[i])
    # growth = final bbox volume minus the placed-only volume — cheaper to
    # approximate with the union extent product
    return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])


def _same_transform_spot(
    sess: GuideToolSession, moving_pid: str, transform: list[list[float]]
) -> bool:
    """Two instances of one part at ~identical transforms is a stack, not
    an assembly — reject candidates duplicating an existing placement."""
    if not transform:
        return False
    t = [transform[r][3] for r in range(3)]
    for inst in sess.state.assembly.instances:
        if inst.part_id != moving_pid or not inst.transform:
            continue
        e = [inst.transform[r][3] for r in range(3)]
        if max(abs(t[i] - e[i]) for i in range(3)) < 2.0:
            return True
    return False


def _score_pairs_for_instance(
    sess: GuideToolSession,
    moving: Any,
    inst: Any,
    iid: str,
    axes: list[tuple[int, float]],
    orient: str | None,
    picked: bool,
) -> list[tuple[float, float, float, str, dict[str, Any]]]:
    """(hint, picked-bonus, solver-score, iid, pair) rows for one placed
    instance — the fixed port's world position AND its world axis both feed
    the hint: a pipe mated to a downward-facing port runs downward."""
    if inst is None or not inst.transform:
        return []
    fixed = sess.state.part_by_id(inst.part_id)
    if fixed is None:
        return []
    # a port is occupied by a joint on EITHER side — a rim that already
    # received a pipe (as moving port) cannot host another
    used = {
        str(j.get("fixed_port"))
        for j in sess.state.assembly.joints
        if j.get("fixed_instance") == iid
        or (not j.get("fixed_instance") and j.get("fixed_part") == inst.part_id)
    } | {
        str(j.get("moving_port"))
        for j in sess.state.assembly.joints
        if j.get("moving_instance") == iid
    }
    out: list[tuple[float, float, float, str, dict[str, Any]]] = []
    # uncapped solver call — the tool truncates to its top-8 by solver
    # score, which silently drops ports a hint needs
    for pair in solver.find_matching_ports(moving, fixed, used_fixed_ports=used):
        fp = next((p for p in fixed.ports if p.name == pair.get("fixed_port")), None)
        wp = _world_pos(inst.transform, fp.pos) if fp else [0.0, 0.0, 0.0]
        axis_hint = 0.0
        if fp is not None:
            fa = _world_dir(inst.transform, fp.axis)
            axis_hint = sum(sg * fa[ax] for ax, sg in axes)
            if orient == "V":
                axis_hint += abs(fa[2])
            elif orient == "H":
                axis_hint += 1.0 - abs(fa[2])
        hint = sum(sg * wp[ax] for ax, sg in axes) + 25.0 * axis_hint
        out.append((hint, 1.0 if picked else 0.0, float(pair.get("score") or 0.0), iid, pair))
    return out


def _free_port_names(sess: GuideToolSession, inst: Any) -> set[str]:
    used = {
        str(j.get("fixed_port"))
        for j in sess.state.assembly.joints
        if j.get("fixed_instance") == inst.instance_id
        or (not j.get("fixed_instance") and j.get("fixed_part") == inst.part_id)
    } | {
        str(j.get("moving_port"))
        for j in sess.state.assembly.joints
        if j.get("moving_instance") == inst.instance_id
    }
    fp = sess.state.part_by_id(inst.part_id)
    return {p.name for p in (fp.ports if fp else [])} - used


async def _try_bridge(sess: GuideToolSession, task: dict[str, Any], pid: str) -> str | None:
    """Two-sided mate for bridging parts (elbow joining pipe A to pipe B):
    when the sub-step names a second target and the part has >=2 ports,
    solve both mates simultaneously — the roll around the primary axis is
    chosen so the SECOND mate also closes."""
    t1, t2 = _resolve_name(sess, task["target"]), _resolve_name(sess, task.get("target2"))
    moving = sess.state.part_by_id(pid)
    if not (t1 and t2) or moving is None or len(moving.ports) < 2:
        return None
    insts = {
        iid: i
        for i in sess.state.assembly.instances
        for iid in [i.instance_id or i.part_id]
    }
    targets = {
        t: [i for i in insts.values() if i.part_id == t] for t in (t1, t2)
    }
    fixed1 = sess.state.part_by_id(t1)
    fixed2 = sess.state.part_by_id(t2)
    best: tuple[float, list, dict, dict] | None = None
    for i1 in targets[t1]:
        for i2 in targets[t2]:
            p1s = solver.find_matching_ports(
                moving, fixed1, used_fixed_ports=_used(sess, i1)
            )[:4]
            p2s = solver.find_matching_ports(
                moving, fixed2, used_fixed_ports=_used(sess, i2)
            )[:4]
            for p1 in p1s:
                for p2 in p2s:
                    if p1["moving_port"] == p2["moving_port"]:
                        continue
                    m = solver.solve_bridge_attachment(
                        moving,
                        [
                            (p1["moving_port"], fixed1, i1.transform, p1["fixed_port"]),
                            (p2["moving_port"], fixed2, i2.transform, p2["fixed_port"]),
                        ],
                    )
                    if m is None:
                        continue
                    mp2 = next(
                        p for p in moving.ports if p.name == p2["moving_port"]
                    )
                    fp2 = next(
                        p for p in fixed2.ports if p.name == p2["fixed_port"]
                    )
                    res = solver.evaluate_mate(
                        mp2,
                        np.asarray(m),
                        fp2,
                        np.asarray(i2.transform),
                    )[0]
                    if best is None or res < best[0]:
                        best = (res, m, (i1, p1), (i2, p2))
    if best is None:
        return None
    _res, m, (i1, p1), (i2, p2) = best
    cid = f"bridge-{pid}-{task['step']}-{task['substep_idx']}"
    sess._preview_cache[cid] = {
        "transform": m,
        "collision_free": True,
        "moving_port": p1["moving_port"],
        "fixed_port": p1["fixed_port"],
        "fixed_part": i1.part_id,
        "fixed_instance": i1.instance_id,
        "candidate_id": cid,
    }
    res = await place_instance(
        sess,
        {
            "part_id": pid,
            "candidate_id": cid,
            "step": task["step"],
            "substep_idx": task["substep_idx"],
        },
    )
    if not res.get("ok"):
        return None
    sess.state.assembly.joints.append(
        {
            "step": task["step"],
            "moving_part": pid,
            "fixed_part": i2.part_id,
            "moving_instance": res.get("instance_id"),
            "fixed_instance": i2.instance_id,
            "moving_port": p2["moving_port"],
            "fixed_port": p2["fixed_port"],
            "approach_axis": p2.get("axis") or [0.0, 0.0, 1.0],
            "bridge": True,
        }
    )
    return res.get("instance_id")


def _used(sess: GuideToolSession, inst: Any) -> set[str]:
    fp = sess.state.part_by_id(inst.part_id)
    return {p.name for p in (fp.ports if fp else [])} - _free_port_names(sess, inst)


async def _try_attach(
    sess: GuideToolSession,
    task: dict[str, Any],
    pid: str,
    next_axes: tuple[list[tuple[int, float]], str | None] = ([], None),
) -> str | None:
    """Port-driven attach onto the best target instance; None if nothing
    works (no pairs / all candidates collide).

    Candidates are gathered from the resolved target instance AND every
    placed instance: manuals often name a feature ("linkes oberes
    Anschlussrohr") that isn't itself a part, or name a pre-mounted piece
    ("Adapter am unteren Anschluss") while meaning the connection beneath
    it. Direction words in the target then rank each candidate pair by the
    fixed port's world position along the hinted axes — without them the
    resolved instance keeps precedence."""
    resolved_pid = _resolve_name(sess, task["target"]) if task["target"] else None
    picked_iid = (
        _pick_target_instance(sess, task["target"], moving_pid=pid)
        if resolved_pid
        else None
    )
    st = sess.state.step_by_number(task["step"])
    # direction words live in the sub-step prose ("vertikales Rohr", "nach
    # oben öffnen"), the target name alone rarely carries them
    hint_text = f"{task['target'] or ''} {task.get('text') or ''} {st.title if st else ''}"
    axes = _dir_axes(hint_text)
    orient = _orient_word(hint_text)
    inst_by_iid = {
        (i.instance_id or i.part_id): i for i in sess.state.assembly.instances
    }
    moving = sess.state.part_by_id(pid)
    if moving is None:
        return None
    # a resolved part name restricts candidates to THAT part's instances —
    # direction words only disambiguate which of its ports/instances, never
    # switch to an unrelated part (a high free rim would otherwise invite
    # pipe-on-pipe stacks). Feature names that resolve to nothing ("linkes
    # oberes Anschlussrohr") fall through to every placed instance.
    resolved_iids = {iid for iid, i in inst_by_iid.items() if i.part_id == resolved_pid}
    # chain locality: an attachment almost always mates onto something the
    # manual JUST placed (pipe → adapter → casing in the last steps).
    # Recent instances form the middle tier so an unresolvable target
    # ("Rohrbogen") reaches for the pipe it follows, not the casing's
    # highest free port.
    recent_iids = {
        iid
        for iid, i in inst_by_iid.items()
        if (i.placed_in_step or 0) >= task["step"] - 2
    }
    rest_iids = [iid for iid in inst_by_iid if iid not in resolved_iids]
    groups = (
        [
            [iid for iid in inst_by_iid if iid in resolved_iids],
            [iid for iid in rest_iids if iid in recent_iids],
            [iid for iid in rest_iids if iid not in recent_iids],
        ]
        if resolved_pid
        else [
            [iid for iid in inst_by_iid if iid in recent_iids],
            [iid for iid in inst_by_iid if iid not in recent_iids],
        ]
    )
    scored: list[tuple[float, float, float, str, dict[str, Any]]] = []
    for group in groups:
        if not group:
            continue
        for iid in group:
            scored.extend(
                _score_pairs_for_instance(
                    sess, moving, inst_by_iid.get(iid), iid, axes, orient, iid == picked_iid
                )
            )
        if scored:
            break  # resolved part yielded pairs — other part types stay out
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    for _h, _r, _s, target_iid, pair in scored[:6]:
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
        # orientation-aware pick among the roll/variant candidates: the
        # first collision-free one is NOT automatically the readable build.
        # Rows strip "transform" (kept in _preview_cache by candidate_id)
        cands = [c for c in prev.get("candidates", []) if c.get("collision_free", True)]
        cands.sort(
            key=lambda c: (
                _orient_score(
                    moving,
                    pair.get("moving_port", ""),
                    (sess._preview_cache.get(c.get("candidate_id") or "") or {}).get(
                        "transform"
                    )
                    or [],
                    axes,
                    orient,
                    next_axes,
                ),
                -_cluster_bbox_growth(
                    sess,
                    moving,
                    (sess._preview_cache.get(c.get("candidate_id") or "") or {}).get(
                        "transform"
                    )
                    or [],
                ),
                -float(c.get("joint_gap_mm") or 0.0),
            ),
            reverse=True,
        )
        for cand in cands:
            tr = (sess._preview_cache.get(cand.get("candidate_id") or "") or {}).get(
                "transform"
            )
            if tr and _same_transform_spot(sess, pid, tr):
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


async def _place_anchor(sess: GuideToolSession, task: dict[str, Any], pid: str) -> str | None:
    """The base/host part rests upright at the origin — env-hint previews
    drifted it into open space when nothing else was placed yet."""
    part = sess.state.part_by_id(pid)
    if part is None:
        return None
    cid = f"anchor-{pid}-{task['step']}"
    sess._preview_cache[cid] = {
        "transform": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -float(part.bbox_min[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "collision_free": True,
        "candidate_id": cid,
    }
    res = await place_instance(
        sess,
        {
            "part_id": pid,
            "candidate_id": cid,
            "step": task["step"],
            "substep_idx": task["substep_idx"],
        },
    )
    return res.get("instance_id") if res.get("ok") else None


async def _try_env(sess: GuideToolSession, task: dict[str, Any], pid: str) -> str | None:
    prev = await solve_env_preview(sess, {"part_id": pid, "step": task["step"]})
    if not prev.get("ok"):
        return None
    for cand in prev.get("candidates", []):
        if not cand.get("collision_free", True):
            continue
        tr = (sess._preview_cache.get(cand.get("candidate_id") or "") or {}).get(
            "transform"
        )
        if tr and _same_transform_spot(sess, pid, tr):
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
