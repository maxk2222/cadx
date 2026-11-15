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

import itertools
import re
from types import SimpleNamespace
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


def _rp(sess: GuideToolSession, name: str | None, step_no: int) -> str | None:
    """Resolve a manual part name with the parent step's text as context —
    token ties (the two elbow types share rohrbogen/dn/22) break toward
    the part the step text actually names."""
    if not name:
        return None
    st = sess.state.step_by_number(step_no)
    ctx = f"{st.text} {st.title}" if st else ""
    return _resolve_name(sess, name, context=ctx)


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

    def _ref(name: str | None, step_no: int, *, ctx: bool = False) -> None:
        pid = _rp(sess, name, step_no) if ctx else _resolve_name(sess, name or "")
        if pid and pid not in referenced:
            referenced[pid] = step_no

    for st in sorted(sess.state.steps, key=lambda s: s.number):
        # every mention counts toward anchor detection — even on non-assembly steps
        for ss in st.sub_steps:
            _ref(ss.part, st.number, ctx=True)
            _ref(ss.target, st.number)
        for name in st.required_parts:
            _ref(name, st.number, ctx=True)
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
    planned = {_rp(sess, t["part"], t["step"]) for t in tasks}
    anchors = [
        {"step": step_no, "substep_idx": -1, "part": pid, "target": "", "anchor": True}
        for pid, step_no in referenced.items()
        if pid not in planned
    ]
    # anchors first within their step — other tasks may attach onto them
    tasks.extend(anchors)
    tasks.sort(key=lambda t: (t["step"], not t.get("anchor")))
    _infer_bridges(sess, tasks)
    # pre-mounted duplicates: a place-task whose TARGET text carries a
    # side qualifier ("unterer linker Adapter") refers to an instance at
    # THAT side — when the part's own placements all happen later or at
    # other sides, the manual means a pre-mounted piece the steps never
    # place. Spawn a mount task for it onto the earliest referenceable
    # host before the referencing step.
    return _spawn_premounts(sess, tasks)


def _infer_bridges(sess: GuideToolSession, tasks: list[dict[str, Any]]) -> None:
    """Recover omitted ``target2`` from the sub-step text — manuals write
    both mates into one instruction ("Muffe auf das vertikale Rohr, dann
    die freie Muffe auf das obere Rohr") and extraction leaves target2
    empty half the time, silently downgrading bridge steps to single
    mates. Only parts with ≥2 cylinder ports can bridge; a later clause
    must resolve to a DIFFERENT part the plan places no later than this
    step — the physical pairing in ``_try_bridge`` still decides."""
    plan_step: dict[str, int] = {}
    for t in tasks:
        pid = _rp(sess, t["part"], t["step"])
        if pid:
            plan_step.setdefault(pid, t["step"])
    for t in tasks:
        if t.get("premount") or t.get("anchor") or str(t.get("target2") or "").strip():
            continue
        pid = _rp(sess, t["part"], t["step"])
        part = sess.state.part_by_id(pid) if pid else None
        if (
            part is None
            or sum(1 for p in part.ports or [] if p.kind == "cylinder") < 2
        ):
            continue
        target_pid = _resolve_name(sess, t.get("target") or "")
        clauses = re.split(
            r"[,;.]| dann | danach | anschlie\w* | then | and ",
            (t.get("text") or "").lower(),
        )
        for clause in clauses[1:]:
            pid2 = _resolve_name(sess, clause.strip())
            if (
                pid2
                and pid2 not in (pid, target_pid)
                and plan_step.get(pid2, 10**9) <= t["step"]
            ):
                t["target2"] = pid2
                t["target2_inferred"] = True
                break

    _infer_legs(sess, tasks)


def _infer_legs(sess: GuideToolSession, tasks: list[dict[str, Any]]) -> None:
    """Cross-substep legs: text-only extractions split one bridge into two
    placement sub-steps ("Bogen aufs vertikale Rohr" + "freie Muffe aufs
    obere Rohr") — the second sub-step names the SAME part against a
    DIFFERENT target part. Merge it as target2 on the first task and tag
    the second as a leg: if the bridge physically fails the leg still
    places as a genuine second instance (IKEA "dowel into A, dowel into
    B" resolves targets to the SAME host and never reaches this merge)."""
    by_step: dict[int, list[int]] = {}
    for ti, t in enumerate(tasks):
        by_step.setdefault(t["step"], []).append(ti)
    for tis in by_step.values():
        for a_pos, ai in enumerate(tis):
            a = tasks[ai]
            if a.get("premount") or a.get("anchor") or str(a.get("target2") or "").strip():
                continue
            pid_a = _rp(sess, a["part"], a["step"])
            part = sess.state.part_by_id(pid_a) if pid_a else None
            if part is None or sum(1 for p in part.ports or [] if p.kind == "cylinder") < 2:
                continue
            tgt_a = _resolve_name(sess, a.get("target") or "")
            for bi in tis[a_pos + 1 :]:
                b = tasks[bi]
                if b.get("premount") or b.get("anchor") or b.get("_leg_of") is not None:
                    continue
                if _rp(sess, b["part"], b["step"]) != pid_a:
                    continue
                tgt_b = _resolve_name(sess, b.get("target") or "")
                if not tgt_b or tgt_b in (pid_a, tgt_a):
                    continue
                a["target2"] = tgt_b
                a["target2_inferred"] = True
                b["_leg_of"] = ai
                break


# canonical side keys — inflected surface forms of the same direction
# must collapse ("(links oben)" vs "linken oberen" produced disjoint
# token sets {link,links,oben} vs {link,ober} and phantom premounts)
_SIDE_WORDS = {
    "links": "l", "link": "l", "left": "l",
    "rechts": "r", "recht": "r", "right": "r",
    "unten": "d", "unter": "d", "lower": "d", "bottom": "d",
    "oben": "u", "ober": "u", "upper": "u", "top": "u",
}


def _side_words(text: str) -> set[str]:
    # word tokens only — "(links" must match like "links"
    toks = re.findall(r"[a-zäöüß]+", (text or "").lower())
    return {side for w, side in _SIDE_WORDS.items() for tok in toks if tok.startswith(w)}


def _task_side_signature(t: dict[str, Any]) -> set[str]:
    """Side words that describe a task's TARGET location: the bare target
    name plus only the first text clause (the one naming it) — later
    clauses describe OTHER parts ("freie Seite auf das obere Rohr")."""
    first_clause = (t.get("text") or "").split(",")[0]
    for cut in (" dann ", " anschließ", " und "):
        first_clause = first_clause.split(cut)[0]
    # orientation phrases ("Öffnung nach oben") are not LOCATION
    # qualifiers — strip them or they poison the signature
    first_clause = re.sub(
        r"nach\s+(oben|unten|links|rechts|vorne|hinten|oben zeigend)\w*",
        " ",
        first_clause.lower(),
    )
    # port qualifiers modify a FEATURE, not the instance location:
    # "obere Öffnung des linken unteren Rohrbogens" locates the elbow
    # left+lower — "obere" describes its port and must stay out
    first_clause = re.sub(
        r"\b(ober|oberer|obere|oberen|unter|unterer|untere|unteren|"
        r"linker|linke|linken|rechter|rechte|rechten|"
        r"upper|lower|top|bottom|left|right)\s+"
        r"(öffnung|muffe|bohrung|lippe|kante|ende|"
        r"opening|socket|bore|mouth|lip|edge|end)\w*",
        r" \2",
        first_clause,
    )
    return _side_words(f"{t.get('target') or ''} {first_clause}")


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
    def _ref_sides(t: dict[str, Any]) -> set[str]:
        return _task_side_signature(t)

    # every task contributes its own side signature — an instance only
    # covers a reference when ONE signature contains ALL the reference's
    # side words (an upper-left adapter is not a lower-left one)
    planned_sigs: dict[str, list[frozenset[str]]] = {}
    for t in tasks:
        pid = _rp(sess, t["part"], t["step"])
        if pid:
            planned_sigs.setdefault(pid, []).append(frozenset(_ref_sides(t)))
    extra: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("anchor") or task.get("premount"):
            continue
        tpid = _resolve_name(sess, task.get("target"))
        if not tpid or tpid in (_rp(sess, task["part"], task["step"]), host_pid):
            continue
        ref = _ref_sides(task)
        if not ref:
            continue
        # only when the part's own placements split into per-side
        # instances does a side-qualified target mean ANOTHER instance —
        # for a single-instance part the side words locate a feature ON it
        if len({s for s in planned_sigs.get(tpid, []) if s}) < 2:
            continue
        if any(ref <= sig for sig in planned_sigs.get(tpid, [])):
            continue
        quota = getattr(sess, "_premount_quota", None)
        if quota is None:
            quota = sess._premount_quota = {}
        quota[tpid] = quota.get(tpid, 0) + 1
        extra.append(
            {
                "step": task["step"],
                "substep_idx": -1,
                "part": tpid,
                "target": host_pid,
                "target2": "",
                "text": f"{task.get('target') or ''} {task.get('text') or ''}",
                "premount": True,
                # which side-qualified reference spawned this copy — the
                # chain back-solve needs it to pick the matching partner
                "_ref_sides": sorted(ref),
            }
        )
        planned_sigs.setdefault(tpid, []).append(frozenset(ref))
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
    # "vormontierter X" binds to the premounted copy, never a kit instance —
    # without this the elbow step mated straight onto the upper adapter
    tl = (target_name or "").lower()
    if any(
        w in tl
        for w in ("vormontiert", "premount", "pre-mount", "preinstall", "pre-install")
    ):
        pm = [i for i in insts if getattr(i, "premounted", False)]
        if pm:
            insts = pm
    # side-qualified target ("linker unterer Adapter") — the instance AT
    # that side wins over mere placement order. Positions normalize per
    # axis over the candidates so one large span can't drown the other.
    axes = _dir_axes(target_name)
    if axes and len(insts) > 1:
        origins = {
            iid: np.asarray(_world_pos(i.transform, [0.0, 0.0, 0.0]), dtype=float)
            for i in insts
            for iid in [i.instance_id or i.part_id]
            if i.transform
        }
        if len(origins) > 1:
            lo = np.min(list(origins.values()), axis=0)
            hi = np.max(list(origins.values()), axis=0)
            rng = np.maximum(hi - lo, 1.0)
            mid = (lo + hi) / 2.0
            insts = sorted(
                insts,
                key=lambda i: sum(
                    sg * float(origins.get(i.instance_id or i.part_id, mid)[a] - mid[a]) / float(rng[a])
                    for a, sg in axes
                ),
                reverse=True,
            )
    used_types: dict[str, set[str]] = {}
    for j in sess.state.assembly.joints:
        fi = str(j.get("fixed_instance") or "")
        if fi:
            used_types.setdefault(fi, set()).add(str(j.get("moving_part") or ""))

    def _free(inst: Any) -> bool:
        part = sess.state.part_by_id(inst.part_id)
        if not (part and part.ports):
            return True
        return bool(_free_port_names(sess, inst))

    candidates = [i for i in insts if _free(i)]
    for inst in candidates:
        iid = inst.instance_id or inst.part_id
        if moving_pid and moving_pid not in used_types.get(iid, set()):
            return iid
    return (candidates or insts)[-1].instance_id or (candidates or insts)[-1].part_id


def _pick_host_port(host: Any, axes: list[tuple[int, float]], feed_perp: int | None) -> Any | None:
    """Best host port for a task: side-word position match; when the
    chain feeds a bridge, ports parallel to the incoming leg are
    geometrically impossible and lose hard."""
    center = [(host.bbox_min[i] + host.bbox_max[i]) / 2 for i in range(3)]
    best, best_s = None, -1e18
    front_bias = 0.0 if any(a == 1 for a, _ in axes) else 0.3
    for fp in host.ports:
        if fp.kind != "cylinder":
            continue
        rel = [fp.pos[i] - center[i] for i in range(3)]
        s = sum(sg * rel[a] for a, sg in axes) + front_bias * rel[1]
        if feed_perp is not None:
            s += -400.0 if abs(fp.axis[feed_perp]) > 0.8 else 150.0 * (1.0 - abs(fp.axis[feed_perp]))
        if s > best_s:
            best_s, best = s, fp
    return best


def _chain_height(sess: GuideToolSession, tasks: list[dict[str, Any]], prem_idx: int, host_pid: str) -> float:
    """Height of the ONE chain growing from the premount: follow the task
    graph tip→tip (sibling sides share part ids and stay out); stop at a
    bridge task (it closes the loop, adds no height) or when a task
    re-targets the host."""
    tip = _rp(sess, tasks[prem_idx]["part"], tasks[prem_idx]["step"])
    chain_pids = [tip] if tip else []
    for t in tasks[prem_idx + 1 :]:
        if t.get("premount"):
            continue
        tp = _resolve_name(sess, t.get("target") or "")
        if tp == host_pid:
            break  # chain reached back to the host
        if tp != tip:
            continue
        if str(t.get("target2") or "").strip():
            break  # bridge task — the chain terminates here
        pp = _rp(sess, t["part"], t["step"])
        if pp:
            chain_pids.append(pp)
            tip = pp
    # physical reach, not bbox sums: the premount root contributes its
    # free port height (fittings carry it mid-body); every later part adds
    # the distance between its chain-in and chain-out ports minus one
    # engagement (a plug seats ~15mm inside the socket, not on its lip)
    h = 0.0
    for i, cpid in enumerate(chain_pids):
        p = sess.state.part_by_id(cpid)
        if p is None:
            continue
        span = float(p.bbox_max[2] - p.bbox_min[2])
        if i == 0:
            h += span / 2.0
            continue
        cyl = [pp for pp in p.ports if pp.kind == "cylinder"]
        if len(cyl) >= 2:
            best = max(
                float(
                    np.linalg.norm(np.asarray(a.pos, dtype=float) - np.asarray(b.pos, dtype=float))
                )
                for x, a in enumerate(cyl)
                for b in cyl[x + 1 :]
            )
            h += max(best - 15.0, span * 0.5)
        else:
            h += span
    return h


_ROT_CUBE: list[np.ndarray] | None = None


def _cube_rotations() -> list[np.ndarray]:
    """The 24 orientation-preserving axis permutations. Identity first so
    argmax keeps the asset frame whenever orientations tie."""
    global _ROT_CUBE
    if _ROT_CUBE is None:
        rots = [np.eye(3)]
        for perm in itertools.permutations(range(3)):
            p = np.zeros((3, 3))
            for i, j in enumerate(perm):
                p[i, j] = 1.0
            for signs in itertools.product((1.0, -1.0), repeat=3):
                r = np.diag(signs) @ p
                if round(float(np.linalg.det(r))) == 1 and not np.allclose(r, np.eye(3)):
                    rots.append(r)
        _ROT_CUBE = rots
    return _ROT_CUBE


def _rotated_bbox(mn: Any, mx: Any, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    )
    w = corners @ rot.T
    return w.min(axis=0), w.max(axis=0)


def _world_part_view(inst: Any, part: Any) -> SimpleNamespace:
    """Ports/bbox of a placed instance in WORLD frame — for picking a host
    port by the task's world-space direction words regardless of the
    instance's local frame."""
    tr = inst.transform
    ports = [
        p.model_copy(
            update={
                "pos": _world_pos(tr, p.pos),
                "axis": _world_dir(tr, p.axis),
            }
        )
        for p in part.ports
    ]
    mn, mx = solver.world_bbox(
        part.bbox_min, part.bbox_max, np.asarray(tr, dtype=float).reshape(4, 4)
    )
    return SimpleNamespace(ports=ports, bbox_min=mn, bbox_max=mx)


def _rotated_part_view(part: Any, rot: np.ndarray) -> SimpleNamespace:
    """Part-like view (ports/bbox only) under a rotation — lets
    _pick_host_port & lift math run on the candidate orientation."""
    ports = [
        p.model_copy(
            update={
                "pos": [float(x) for x in rot @ np.asarray(p.pos, dtype=float)],
                "axis": [float(x) for x in rot @ np.asarray(p.axis, dtype=float)],
            }
        )
        for p in part.ports
    ]
    mn, mx = _rotated_bbox(part.bbox_min, part.bbox_max, rot)
    return SimpleNamespace(ports=ports, bbox_min=mn, bbox_max=mx)


def _anchor_orientation(sess: GuideToolSession, part: Any, tasks: list[dict[str, Any]]) -> np.ndarray:
    """Best-scoring host rotation — see ``_anchor_orientations``."""
    scored = _anchor_orientations(sess, part, tasks)
    return scored[0][1] if scored else np.eye(3)


def _anchor_orientations(
    sess: GuideToolSession, part: Any, tasks: list[dict[str, Any]]
) -> list[tuple[float, np.ndarray]]:
    """Every cube rotation scored by how usable the host's ports are for
    the manual's attach tasks — solver pair legality + lateral reach +
    side words. The asset frame is arbitrary — a host port pointing
    straight up/down forces an impossible over-the-top or under-floor
    mate; a host's connection hardware faces the assembly's working side
    (+Y front by frame contract). Returns (score, rot) sorted desc — the
    caller trial-places the top candidates because scores can tie within
    noise and only the measured result is authoritative."""
    host_ports = [p for p in part.ports if p.kind == "cylinder"]
    if not host_ports:
        return [(0.0, np.eye(3))]
    attach: list[tuple[dict[str, Any], Any]] = []
    for t in tasks:
        if t.get("anchor") or t.get("premount"):
            continue
        if _resolve_name(sess, t.get("target") or "") != part.id:
            continue
        mpid = _rp(sess, t["part"], t["step"])
        moving = sess.state.part_by_id(mpid) if mpid else None
        if moving is not None and moving.id != part.id:
            attach.append((t, moving))
        if len(attach) >= 4:
            break
    if not attach:
        return [(0.0, np.eye(3))]
    low_bias = _below_chain(sess, tasks)
    scored: list[tuple[float, np.ndarray]] = []
    for rot in _cube_rotations():
        ports_r = [
            (
                rot @ np.asarray(p.pos, dtype=float),
                rot @ np.asarray(p.axis, dtype=float),
                p,
            )
            for p in host_ports
        ]
        mn_r, mx_r = _rotated_bbox(part.bbox_min, part.bbox_max, rot)
        center = (mn_r + mx_r) / 2.0
        span = float((mx_r - mn_r).max()) or 1.0
        span_z = float(mx_r[2] - mn_r[2]) or 1.0
        exposure = _port_exposures(ports_r, mn_r, mx_r)
        # an opening aimed at the floor is unreachable — no mate can
        # approach through the ground plane the host rests on
        s = -1.0 * sum(1 for _, ax_w, _ in ports_r if float(ax_w[2]) < -0.8)
        win: dict[int, np.ndarray] = {}
        for ti, (t, moving) in enumerate(attach):
            st = sess.state.step_by_number(t["step"])
            hint = f"{t.get('target') or ''} {t.get('text') or ''} {st.title if st else ''}"
            task_best = -1e18
            task_pos: np.ndarray | None = None
            for (pos_w, ax_w, fp), exp in zip(ports_r, exposure, strict=True):
                if exp > 60.0:
                    continue  # recessed past any fitting's reach — an
                    # interior junction, not a mount interface
                cand = _interface_score(
                    moving, fp, pos_w, ax_w, exp,
                    _dir_axes(hint), span, center, pos_w[2], mn_r[2], span_z,
                    low_bias,
                )
                if cand > task_best:
                    task_best, task_pos = cand, pos_w
            if task_best > -1e18:
                s += task_best
                if task_pos is not None:
                    win[ti] = task_pos
        # sibling consistency: two tasks on the SAME part differing only
        # in side words ("linker"/"rechter Anschluss") must land on ports
        # separated along the DIFFERING side axis — a rotation that stacks
        # a left/right pair vertically is physically wrong even when every
        # per-task score looked fine (casing lying on its side)
        s += _sibling_sep_score(sess, attach, win)
        if np.allclose(rot, np.eye(3)):
            s += 0.01  # keep the asset frame when no orientation wins outright
        scored.append((s, rot))
    scored.sort(key=lambda sr: -sr[0])
    return scored


def _port_exposures(
    ports_r: list[tuple[np.ndarray, np.ndarray, Any]],
    mn_r: np.ndarray,
    mx_r: np.ndarray,
) -> list[float]:
    """Recession depth of each port below the rotated silhouette along its
    own axis — a mouth 100mm inside the body can't be reached without
    tunneling through the part's other openings."""
    corners_r = np.array(
        [
            [x, y, z]
            for x in (mn_r[0], mx_r[0])
            for y in (mn_r[1], mx_r[1])
            for z in (mn_r[2], mx_r[2])
        ]
    )
    return [
        max(
            float((corners_r @ solver._unit(ax_w)).max() - float(np.dot(pos_w, solver._unit(ax_w)))),
            0.0,
        )
        for pos_w, ax_w, _ in ports_r
    ]


def _sibling_sep_score(
    sess: GuideToolSession,
    attach: list[tuple[dict[str, Any], Any]],
    win: dict[int, np.ndarray],
) -> float:
    """Reward rotations where same-part sibling tasks separate along the
    axis their side words differ in — 'linker/rechter' siblings on X, not
    stacked on Z."""
    bonus = 0.0
    for i in range(len(attach)):
        for j in range(i + 1, len(attach)):
            ta, tb = attach[i][0], attach[j][0]
            pa = _rp(sess, ta["part"], ta["step"])
            if pa is None or pa != _rp(sess, tb["part"], tb["step"]):
                continue
            sa = _task_side_signature(ta)
            sb = _task_side_signature(tb)
            diff = sa ^ sb
            if not diff or i not in win or j not in win:
                continue
            sep = np.asarray(win[i] - win[j], dtype=float)
            n = float(np.linalg.norm(sep))
            if n < 1e-6:
                continue
            su = sep / n
            for w in diff:
                axes = {
                    "l": (0, -1.0), "r": (0, 1.0),
                    "f": (1, 1.0), "b": (1, -1.0),
                    "u": (2, 1.0), "d": (2, -1.0),
                }.get(w)
                if axes is None:
                    continue
                a, _sg = axes
                # dominant separation along the hinted axis = good;
                # vertical stacking of a horizontal pair = bad
                proj = abs(float(su[a]))
                vert = abs(float(su[2])) if a != 2 else 0.0
                bonus += 0.8 * proj - 0.8 * vert
    return bonus


def _interface_score(
    moving: Any,
    fp: Any,
    pos_w: np.ndarray,
    ax_w: np.ndarray,
    exposure: float,
    axes: list[tuple[int, float]],
    span: float,
    center: np.ndarray,
    pos_z: float,
    mn_z: float,
    span_z: float,
    low_bias: bool,
) -> float:
    """How usable one host port is for an attach task: solver pair fit,
    minus recession, plus lateral reach (front > side > back >> vertical),
    plus the task's side words, plus the low-on-body prior when the chain
    grows from below."""
    fit = max(
        (
            solver._score_pair(mp, fp, include_mismatch=False)
            for mp in moving.ports
            if mp.kind == "cylinder"
        ),
        default=-1.0,
    )
    if fit <= 0.0:
        return -1e18
    fit -= min(exposure / 25.0, 0.55)
    az = abs(float(ax_w[2]))
    reach = (
        1.0 + 0.5 * float(ax_w[1])
        if az < 0.35
        else (-1.0 if az > 0.8 else 0.0)
    )
    pos_hint = (
        0.15 * sum(sg * float(pos_w[a] - center[a]) for a, sg in axes) / span
    )
    low = 0.5 * (1.0 - float(pos_z - mn_z) / span_z) if low_bias else 0.0
    return fit + reach + pos_hint + low


def _below_chain(sess: GuideToolSession, tasks: list[dict[str, Any]]) -> bool:
    """True when the manual mounts hardware BELOW the host (a premount
    chain growing from the floor) — then the host's interfaces belong to
    its lower body region."""
    for t in tasks:
        if not t.get("premount"):
            continue
        st = sess.state.step_by_number(t["step"])
        hint = (
            f"{t.get('target') or ''} {t.get('text') or ''} "
            f"{st.title if st else ''}"
        )
        if any(sg < 0 for ax, sg in _dir_axes(hint) if ax == 2):
            return True
    return False


def _anchor_lift(
    sess: GuideToolSession, tasks: list[dict[str, Any]], rot: np.ndarray | None = None
) -> float:
    """When the manual mounts hardware BELOW the host (e.g. fittings
    pre-mounted on a device the casing covers), the anchor can't sit on
    the floor — the chain from the floor-level premounts (fitting → elbow
    → pipe) must reach the host's ports. Lift = estimated chain height
    above the premount - the host port's unlifted height."""
    prem_idx = next((i for i, t in enumerate(tasks) if t.get("premount")), None)
    if prem_idx is None:
        return 0.0
    prem = tasks[prem_idx]
    st0 = sess.state.step_by_number(prem["step"])
    hint0 = f"{prem.get('target') or ''} {prem.get('text') or ''} {st0.title if st0 else ''}"
    if not any(sg < 0 for ax, sg in _dir_axes(hint0) if ax == 2):
        return 0.0  # premounts beside/above the host — host stays grounded
    host_pid = _resolve_name(sess, prem.get("target") or "")
    host = sess.state.part_by_id(host_pid) if host_pid else None
    if host is None:
        return 0.0
    host_view = _rotated_part_view(host, rot) if rot is not None else host
    chain_h = _chain_height(sess, tasks, prem_idx, host_pid)
    if chain_h <= 0:
        return 0.0
    # host port the first host-bound task will actually mate to — the
    # fit/reach interface the anchor orientation was chosen for (side
    # words alone can pick a positionally-matching port the joint can
    # never use)
    port_local_z = 0.0
    low_bias = _below_chain(sess, tasks)
    mn_v = np.asarray(host_view.bbox_min, dtype=float)
    mx_v = np.asarray(host_view.bbox_max, dtype=float)
    center_v = (mn_v + mx_v) / 2.0
    span_v = float((mx_v - mn_v).max()) or 1.0
    span_z = float(mx_v[2] - mn_v[2]) or 1.0
    corners_v = np.array(
        [
            [x, y, z]
            for x in (mn_v[0], mx_v[0])
            for y in (mn_v[1], mx_v[1])
            for z in (mn_v[2], mx_v[2])
        ]
    )
    for t in tasks:
        if _resolve_name(sess, t.get("target") or "") != host_pid:
            continue
        tpid = _rp(sess, t["part"], t["step"])
        moving = sess.state.part_by_id(tpid) if tpid else None
        if moving is None:
            continue
        st = sess.state.step_by_number(t["step"])
        hint = f"{t.get('target') or ''} {t.get('text') or ''} {st.title if st else ''}"
        axes = _dir_axes(hint)
        best_s, best_p = -1e18, None
        for fp in host_view.ports:
            if fp.kind != "cylinder":
                continue
            ax_w = np.asarray(fp.axis, dtype=float)
            pos_w = np.asarray(fp.pos, dtype=float)
            exp = max(
                float((corners_v @ solver._unit(ax_w)).max() - float(np.dot(pos_w, solver._unit(ax_w)))),
                0.0,
            )
            if exp > 60.0:
                continue
            sc = _interface_score(
                moving, fp, pos_w, ax_w, exp, axes, span_v, center_v,
                float(pos_w[2]), float(mn_v[2]), span_z, low_bias,
            )
            if sc > best_s:
                best_s, best_p = sc, fp
        if best_p is not None:
            port_local_z = float(best_p.pos[2])
        break
    unlifted_port_z = port_local_z - float(host_view.bbox_min[2])
    return max(0.0, chain_h - unlifted_port_z)


def _host_subgraph(sess: GuideToolSession, anchor_iid: str) -> set[str]:
    """Instances in the anchor's attachment tree — child links follow
    PRIMARY (non-bridge) joints only. A bridge part's primary mate roots
    it in the other chain (the floor run), so it stays out — exactly the
    boundary a lift correction must not cross."""
    children: dict[str, list[str]] = {}
    for j in sess.state.assembly.joints:
        if j.get("bridge"):
            continue
        children.setdefault(str(j.get("fixed_instance") or ""), []).append(
            str(j.get("moving_instance") or "")
        )
    out: set[str] = set()
    frontier = [anchor_iid]
    while frontier:
        iid = frontier.pop()
        if iid in out:
            continue
        out.add(iid)
        frontier.extend(children.get(iid, []))
    return out


def _bridge_joint_resid(
    sess: GuideToolSession,
    j: dict[str, Any],
    parts: dict[str, Any],
    inst: dict[str, Any],
) -> np.ndarray | None:
    """World residual vector (moving socket minus engaged fixed point)
    of one recorded bridge joint; None when it can't be measured."""
    mv = inst.get(str(j.get("moving_instance") or ""))
    fx = inst.get(str(j.get("fixed_instance") or ""))
    if mv is None or fx is None:
        return None
    mp = next((p for p in parts[mv.part_id].ports if p.name == j.get("moving_port")), None)
    tp = next((p for p in parts[fx.part_id].ports if p.name == j.get("fixed_port")), None)
    if mp is None or tp is None:
        return None
    t_m = np.asarray(mv.transform, dtype=float).reshape(4, 4)
    t_f = np.asarray(fx.transform, dtype=float).reshape(4, 4)
    mpw = t_m[0:3, 0:3] @ solver._v(mp.pos) + t_m[0:3, 3]
    fpw = (
        t_f[0:3, 0:3]
        @ (
            solver._v(tp.pos)
            - solver._v(tp.axis)
            * solver._engage_mm(
                mp, tp, moving_part=parts[mv.part_id], fixed_part=parts[fx.part_id]
            )
        )
        + t_f[0:3, 3]
    )
    return mpw - fpw


def _bridge_closure_shift(
    sess: GuideToolSession, host_set: set[str], tasks: list[dict[str, Any]] | None = None
) -> np.ndarray | None:
    """Offset the anchor subtree needs so the bridge mates close.

    The anchor pose is an estimate — ``_chain_height`` cannot know the
    bridge part's socket-to-socket rise and the host's X/Y came from
    centering — so once placement is done the measured residual is
    authoritative: for every bridge mate exactly one side belongs to the
    anchor subtree, and moving that side by the residual closes the gap
    while every subtree-internal joint stays exact. Bound-REJECTED
    candidates contribute their stored residual vector too — the bound
    gates the joint record, not the measurement.

    Consensus is per-axis: an offset is a systematic placement error only
    where every bridge agrees on it — opposing residuals (mirrored
    parts) cancel that axis instead of producing a guess."""
    parts = {p.id: p for p in sess.state.parts}
    inst = {i.instance_id: i for i in sess.state.assembly.instances}
    shifts: list[np.ndarray] = []
    for j in sess.state.assembly.joints:
        if not j.get("bridge"):
            continue
        resid = _bridge_joint_resid(sess, j, parts, inst)
        if resid is None:
            continue
        in_m = str(j.get("moving_instance") or "") in host_set
        in_f = str(j.get("fixed_instance") or "") in host_set
        if in_f and not in_m:
            shifts.append(resid)
        elif in_m and not in_f:
            shifts.append(-resid)
    for t in tasks or []:
        miss = t.get("bridge_miss")
        if not miss:
            continue
        resid = np.asarray(miss["resid"], dtype=float)
        res_n = float(np.linalg.norm(resid))
        if res_n < 1e-6 or res_n > 80.0:
            continue  # absurdly large miss is a pairing error, not a pose error
        i1h = miss.get("i1") in host_set
        i2h = miss.get("i2") in host_set
        if i2h and not i1h:
            shifts.append(resid)
        elif i1h and not i2h:
            shifts.append(-resid)
    if not shifts:
        return None
    out = np.zeros(3)
    any_axis = False
    for ax in range(3):
        vals = [float(s[ax]) for s in shifts]
        if max(vals) - min(vals) > 10.0:
            continue
        mean = float(np.mean(vals))
        if abs(mean) >= 0.5:
            out[ax] = mean
            any_axis = True
    return out if any_axis else None


def _lift_correct_for_bridges(
    sess: GuideToolSession,
    anchor_pid: str | None,
    lift: float,
    report: dict[str, Any],
    tasks: list[dict[str, Any]] | None = None,
) -> None:
    """Refine the estimated anchor lift with the measured bridge residual.

    The floor chain (premounts → risers → bridge primaries) is grounded —
    it can't move — so the correction shifts only the anchor subtree
    (host + everything jointed onto it) along the lift axis. Gated on an
    existing lift: it refines an estimate, it never invents one."""
    if not anchor_pid:
        return
    for inst in sess.state.assembly.instances:
        if inst.part_id != anchor_pid:
            continue
        host_set = _host_subgraph(sess, inst.instance_id)
        shift = _bridge_closure_shift(sess, host_set, tasks)
        if shift is None:
            continue
        if lift <= 0.0:
            shift = shift.copy()
            shift[2] = 0.0  # no estimated lift — the host may not float
        shift[2] = float(max(-lift, min(60.0, shift[2])))
        shift[0] = float(max(-60.0, min(60.0, shift[0])))
        shift[1] = float(max(-60.0, min(60.0, shift[1])))
        if float(np.linalg.norm(shift)) < 0.5:
            continue
        for i in sess.state.assembly.instances:
            if i.instance_id in host_set and i.transform:
                t = np.asarray(i.transform, dtype=float).reshape(4, 4)
                t[0:3, 3] += shift
                i.transform = [[float(v) for v in row] for row in t.tolist()]
        report["lift_correction_mm"] = [round(float(v), 2) for v in shift]
        log.info(
            "guide.preassembly.lift_correction",
            shift_mm=[round(float(v), 2) for v in shift],
            anchor=inst.instance_id,
        )
        return


async def _retry_bridges(
    sess: GuideToolSession, tasks: list[dict[str, Any]], report: dict[str, Any]
) -> None:
    """Re-solve bridge mates after the lift correction moved the host
    subtree. Tasks whose best candidate failed the residual bound keep
    their measured miss in ``bridge_miss``; once the host side sits at
    the corrected height the same candidate closes inside the bound.

    The fallback attach already placed the part on its primary mate —
    the retry swaps the instance transform surgically (the bridge solve
    keeps the primary mate exact) and rewrites that step's joints from
    the solved pair set. Only leaf instances are touched — a part other
    joints hang on stays as placed."""
    for task in tasks:
        miss = task.pop("bridge_miss", None)
        if not miss:
            continue
        pid = _rp(sess, task["part"], task["step"])
        moving = sess.state.part_by_id(pid) if pid else None
        if moving is None:
            continue
        placed = [
            i
            for i in sess.state.assembly.instances
            if i.part_id == pid
            and i.placed_in_step == task["step"]
            and i.substep_idx == task["substep_idx"]
        ]
        if not placed:
            continue
        inst = placed[-1]
        # leaf only — re-transforming a part children hang on orphans them
        if any(
            str(j.get("fixed_instance") or "") == inst.instance_id
            for j in sess.state.assembly.joints
        ):
            continue
        # free the ports this instance already occupies — its primary-mate
        # joint would otherwise mask the very candidate being re-solved
        old_joints = [
            j
            for j in sess.state.assembly.joints
            if str(j.get("moving_instance") or "") == inst.instance_id
            and j.get("step") == task["step"]
        ]
        sess.state.assembly.joints[:] = [
            j
            for j in sess.state.assembly.joints
            if not (
                str(j.get("moving_instance") or "") == inst.instance_id
                and j.get("step") == task["step"]
            )
        ]
        best = _bridge_search(sess, task, pid)
        if best is None or best[1] > _bridge_bound(sess, task, pid, best):
            sess.state.assembly.joints.extend(old_joints)  # keep the honest miss
            continue
        _key, res, m, (i1, p1), (i2, p2), _rv = best
        inst.transform = [[float(v) for v in row] for row in np.asarray(m).tolist()]
        sess.state.assembly.joints.extend(
            [
                {
                    "step": task["step"],
                    "moving_part": pid,
                    "fixed_part": i1.part_id,
                    "moving_instance": inst.instance_id,
                    "fixed_instance": i1.instance_id,
                    "moving_port": p1["moving_port"],
                    "fixed_port": p1["fixed_port"],
                    "approach_axis": p1.get("axis") or [0.0, 0.0, 1.0],
                },
                {
                    "step": task["step"],
                    "moving_part": pid,
                    "fixed_part": i2.part_id,
                    "moving_instance": inst.instance_id,
                    "fixed_instance": i2.instance_id,
                    "moving_port": p2["moving_port"],
                    "fixed_port": p2["fixed_port"],
                    "approach_axis": p2.get("axis") or [0.0, 0.0, 1.0],
                    "bridge": True,
                },
            ]
        )
        sess._scene_cache = None
        report.setdefault("bridge_retries", []).append(
            {"step": task["step"], "residual_mm": round(res, 2)}
        )
        log.info(
            "guide.preassembly.bridge_retry",
            step=task["step"],
            residual_mm=round(res, 2),
        )


async def _place_tasks(
    sess: GuideToolSession,
    tasks: list[dict[str, Any]],
    lift: float,
    anchor_rot: np.ndarray | None,
    report: dict[str, Any],
) -> None:
    """The deterministic placement pass for one anchor rotation."""
    for ti, task in enumerate(tasks):
        task.pop("bridge_miss", None)  # stale residual of a prior trial
        leg_of = task.get("_leg_of")
        if leg_of is not None and tasks[leg_of].get("_bridged"):
            # partner bridge closed both mates — this sub-step is covered
            report["skipped"].append({**task, "reason": "bridge leg absorbed"})
            continue
        pid = _rp(sess, task["part"], task["step"])
        if not pid:
            report["failed"].append({**task, "reason": f"unresolved part '{task['part']}'"})
            continue
        if _already_satisfied(sess, task["step"], pid):
            report["skipped"].append({**task, "reason": "already satisfied"})
            continue
        lookahead = _lookahead(sess, tasks, ti, pid)
        feed_perp = _feeds_bridge_axis(sess, tasks, ti, pid)
        if task.get("premount"):
            # initial-state hardware is not mated onto a kit port — give it
            # a documented reference pose beside/below the host instead of
            # a fake joint to an incompatible rim
            st0 = sess.state.step_by_number(task["step"])
            hint0 = f"{task.get('target') or ''} {task.get('text') or ''} {st0.title if st0 else ''}"
            placed = await _place_premount_ref(
                sess, task, pid, _dir_axes(hint0), feed_perp, tasks=tasks, ti=ti
            )
            if not placed:
                placed = await _try_attach(
                    sess, task, pid, lookahead, feed_perp, tasks=tasks
                )
        else:
            placed = await _try_bridge(sess, task, pid)
        if not placed:
            placed = await _try_attach(
                sess, task, pid, lookahead, feed_perp, tasks=tasks
            )
        if not placed and task.get("anchor"):
            placed = await _place_anchor(sess, task, pid, lift, anchor_rot)
        if not placed:
            placed = await _try_env(sess, task, pid)
        if placed:
            report["placed"].append({**task, "part_id": pid, "instance_id": placed})
        else:
            report["failed"].append({**task, "part_id": pid, "reason": "no candidate"})


def _attempt_quality(sess: GuideToolSession, report: dict[str, Any]) -> tuple[float, ...]:
    """How consistent one trial placement came out: more parts placed and
    smaller measured bridge residuals win. The scorecard orientation
    only ranks candidates — this is the measured verdict."""
    resid = 0.0
    parts = {p.id: p for p in sess.state.parts}
    inst = {i.instance_id: i for i in sess.state.assembly.instances}
    for j in sess.state.assembly.joints:
        if not j.get("bridge"):
            continue
        mv = inst.get(str(j.get("moving_instance") or ""))
        fx = inst.get(str(j.get("fixed_instance") or ""))
        if mv is None or fx is None:
            continue
        mp = next((p for p in parts[mv.part_id].ports if p.name == j.get("moving_port")), None)
        tp = next((p for p in parts[fx.part_id].ports if p.name == j.get("fixed_port")), None)
        if mp is None or tp is None:
            continue
        resid += solver.evaluate_mate(
            mp,
            np.asarray(mv.transform, dtype=float).reshape(4, 4),
            tp,
            np.asarray(fx.transform, dtype=float).reshape(4, 4),
            moving_part=parts[mv.part_id],
            fixed_part=parts[fx.part_id],
        )[0]
    # orphan penalty — an instance that placed but never reached the anchor
    # through joints is debris the assembly mission must fix. A rotation
    # that lands every part connected beats one leaving the same count
    # stranded beside the host (observed: elbow committed onto a floating
    # stub chain, flagged disconnected at audit time).
    bboxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for iid, i in inst.items():
        part = parts.get(i.part_id)
        if part is None or i.transform is None:
            continue
        bboxes[iid] = solver.world_bbox(
            part.bbox_min, part.bbox_max, np.asarray(i.transform, dtype=float)
        )
    pairs = [
        (str(j.get("moving_instance") or ""), str(j.get("fixed_instance") or ""))
        for j in sess.state.assembly.joints
    ]
    orphans = solver.disconnected_instances(bboxes, pairs, sorted(bboxes))
    return (
        float(len(report["placed"])),
        -float(len(report["failed"])),
        -float(len(orphans)),
        -round(resid, 1),
    )


async def run_preassembly(sess: GuideToolSession) -> dict[str, Any]:
    """Place what the step plan determines — returns (and persists) a report.

    Orientation is a verified choice, not a one-shot guess: the top
    scoring rotations are trial-placed and the attempt with the most
    placed parts and smallest measured bridge residual is kept. Ties in
    the port-scorecard happen (same tube's front/rear ends score within
    noise) — only the measured result is authoritative."""
    tasks = _plan(sess)
    anchor_pid = next(
        (_rp(sess, t["part"], t["step"]) for t in tasks if t.get("anchor")), None
    )
    anchor_part = sess.state.part_by_id(anchor_pid) if anchor_pid else None
    rot_cands = (
        [rot for _, rot in _anchor_orientations(sess, anchor_part, tasks)[:3]]
        if anchor_part is not None
        else [None]
    )
    asm = sess.state.assembly
    snap_i = len(asm.instances)
    snap_j = len(asm.joints)
    snap_si = {k: len(v) for k, v in asm.step_instances.items()}
    best: tuple[tuple[float, ...], dict[str, Any], list, list, dict] | None = None
    for rot in rot_cands:
        report: dict[str, Any] = {"placed": [], "skipped": [], "failed": []}
        lift = _anchor_lift(sess, tasks, rot)
        try:
            await _place_tasks(sess, tasks, lift, rot, report)
            _lift_correct_for_bridges(sess, anchor_pid, lift, report, tasks)
            await _retry_bridges(sess, tasks, report)
            quality = _attempt_quality(sess, report)
        except Exception:
            # a crashed trial must not poison the next rotation's attempt —
            # the rollback below still runs, then worse rotations get tried
            quality = (-1e9, -1e9, -1e9, -1e9)
            log.exception("guide.preassembly.trial_failed")
        tail_i = list(asm.instances[snap_i:])
        tail_j = list(asm.joints[snap_j:])
        tail_si = {
            k: list(v[snap_si.get(k, 0) :]) for k, v in asm.step_instances.items()
        }
        del asm.instances[snap_i:]
        del asm.joints[snap_j:]
        for k, v in asm.step_instances.items():
            del v[snap_si.get(k, 0) :]
        sess._scene_cache = None
        if best is None or quality > best[0]:
            best = (quality, report, tail_i, tail_j, tail_si)
        if quality[1] == 0.0 and quality[2] == 0.0 and quality[3] > -2.0:
            break  # fully consistent plan — no need to try worse rotations
    assert best is not None
    _q, report, tail_i, tail_j, tail_si = best
    asm.instances.extend(tail_i)
    asm.joints.extend(tail_j)
    for k, v in tail_si.items():
        asm.step_instances.setdefault(k, []).extend(v)
    sess._scene_cache = None
    sess.state.assembly.planner = {
        "ran": True,
        "placed": len(report["placed"]),
        "skipped": len(report["skipped"]),
        "failed": report["failed"][:10],
        "quality": list(_q),
        "lift_correction_mm": report.get("lift_correction_mm"),
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


# "Öffnung nach oben" / "nach unten zeigend" describe the direction a FREE
# port must face — not where the target sits. Kept separate: mixing them
# into location axes pushed mounts to the wrong side
_OPEN_PHRASE = re.compile(r"nach\s+(oben|unten|links|rechts|vorne|hinten)\w*")


def _open_axes(text: str) -> list[tuple[int, float]]:
    """(axis, sign) hints from 'nach <dir>' opening/orientation phrases."""
    out: list[tuple[int, float]] = []
    seen: set[int] = set()
    for word in _OPEN_PHRASE.findall((text or "").lower()):
        hit = _DIR_TOKENS.get(word)
        if hit and hit[0] not in seen:
            out.append(hit)
            seen.add(hit[0])
    return out


def _dir_axes(text: str) -> list[tuple[int, float]]:
    """One (axis, sign) hint per axis found in a target/location phrase."""
    t = _OPEN_PHRASE.sub(" ", " " + (text or "").lower() + " ")
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


def _feeds_bridge_axis(
    sess: GuideToolSession,
    tasks: list[dict[str, Any]],
    ti: int,
    pid: str,
) -> int | None:
    """Axis a later bridge's OTHER leg approaches along, when this task's
    part (or the chain built on it) is one bridge endpoint. A straight
    fitting continues its host port's axis — so that port must face
    PERPENDICULAR to the incoming leg, or a 90° bridge can never close.
    Leg direction comes from the bridge task's own words ("vertikales
    Rohr"); absent a word, chains build upward → assume Z."""
    downstream = {pid}
    for t in tasks[ti + 1 :]:
        pp = _rp(sess, t["part"], t["step"])
        tp = _resolve_name(sess, t.get("target") or "")
        t2 = _resolve_name(sess, t.get("target2"))
        if t2 and (tp in downstream or t2 in downstream):
            other = t.get("target") if t2 in downstream else t.get("target2")
            text = f"{other or ''} {t.get('text') or ''}".lower()
            if any(w in text for w in _HORIZONTAL):
                return None  # horizontal legs can run along X or Y — no axis
            return 2
        if pp and tp in downstream:
            downstream.add(pp)
    return None


def _chain_side_dir(
    sess: GuideToolSession,
    tasks: list[dict[str, Any]],
    task: dict[str, Any],
    feed_perp: int,
) -> np.ndarray | None:
    """Horizontal direction the floor chain feeding this bridge stands
    at — a host port feeding the bridge must FACE the chain or the elbow
    can never close. Derived from the premount task sharing this task's
    left/right signature, evaluated with the same host-port pick the
    premount placement itself uses (world frame, rotation-aware)."""
    insts = sess.state.assembly.instances
    host_inst = insts[0] if insts else None
    if host_inst is None or not host_inst.transform:
        return None
    host = sess.state.part_by_id(host_inst.part_id)
    if host is None:
        return None
    host_view = _world_part_view(host_inst, host)
    c_w = np.asarray(
        _world_pos(
            host_inst.transform,
            [(host.bbox_min[i] + host.bbox_max[i]) / 2 for i in range(3)],
        )
    )
    my_sides = _side_words(f"{task.get('target') or ''} {task.get('text') or ''}")
    for t in tasks:
        if not t.get("premount"):
            continue
        st0 = sess.state.step_by_number(t["step"])
        ph = (
            f"{t.get('target') or ''} {t.get('text') or ''} "
            f"{st0.title if st0 else ''}"
        )
        p_lr = _side_words(ph) & {"l", "r"}
        if p_lr and not (p_lr & my_sides):
            continue
        xy_axes = [(a, s) for a, s in _dir_axes(ph) if a < 2]
        fp = _pick_host_port(host_view, xy_axes, feed_perp)
        if fp is None:
            continue
        d = np.asarray(fp.pos[:2], dtype=float) - c_w[:2]
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            return d / n
    return None


def _mounted_on_host(sess: GuideToolSession, fixed_inst: Any) -> bool:
    """Is the fixed instance the assembly's base/host — the anchor every
    fitting mounts onto? Hardware on a host presents its free port
    outward; chain mates (pipe→elbow) must not get the radial prior."""
    insts = sess.state.assembly.instances
    return bool(insts and fixed_inst is insts[0])


def _outward_free_score(
    moving: Any,
    mated_port: str,
    transform: list[list[float]],
    fixed_port_world: list[float],
    host_center: list[float],
) -> float:
    """Hardware mounted on a host presents its open port pointing AWAY
    from the host body — radial outward — so downstream parts can mate
    to it (a rear-facing socket lets a 90° elbow's free opening point
    up)."""
    d = [fixed_port_world[i] - host_center[i] for i in range(3)]
    n = float(np.sqrt(sum(x * x for x in d))) or 1.0
    d = [x / n for x in d]
    best = 0.0
    for p in moving.ports:
        if p.name == mated_port:
            continue
        w = _world_dir(transform, p.axis)
        best = max(best, sum(w[i] * d[i] for i in range(3)))
    return best


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
        npid = _rp(sess, nxt["part"], nxt["step"])
        if not npid or npid == pid:
            continue
        tpid = _resolve_name(sess, nxt.get("target") or "")
        if tpid is not None and tpid != pid:
            return [], None
        text = f"{nxt.get('target') or ''} {nxt.get('text') or ''}"
        return _open_axes(text) or _dir_axes(text), _orient_word(text)
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
    open_axes: list[tuple[int, float]],
    orient: str | None,
    feed_perp: int | None,
    picked: bool,
    chain_dir: np.ndarray | None = None,
    sibling_pos: list[list[float]] | None = None,
) -> list[tuple[float, float, float, str, dict[str, Any]]]:
    """(hint, picked-bonus, solver-score, iid, pair) rows for one placed
    instance — the fixed port's world POSITION follows the location axes
    ("unterer linker Anschluss"), while its world AXIS follows opening
    phrases ("Öffnung nach oben") — a pipe mated to a downward-facing port
    runs downward. ``feed_perp`` is the axis a later bridge's other leg
    approaches along: the mated port must be PERPENDICULAR to it, or a
    90° bridge can never close (a vertical pipe can't elbow onto a
    vertical stub)."""
    if inst is None or not inst.transform:
        return []
    fixed = sess.state.part_by_id(inst.part_id)
    if fixed is None:
        return []
    # a port is occupied by a joint on EITHER side — a rim that already
    # received a pipe (as moving port) cannot host another; coincident
    # extraction duplicates count as the same physical connection
    used = _used_port_names(sess, inst)
    # location words must DISAMBIGUATE symmetric interfaces, not outrank
    # the physical fit: raw mm ("oberer" → +376 for a top funnel vs +71
    # for the real stub) let a merely-higher port win over the right
    # interface by hundreds of points. Normalize by the host's world
    # span per axis (bounded ~±0.5 each) and fold the solver fit into the
    # same sum at 4x — a snug thread interface beats a merely-higher
    # coarse seat, equal fits still split cleanly left/right/up/down.
    mn_w, mx_w = solver.world_bbox(
        fixed.bbox_min, fixed.bbox_max, np.asarray(inst.transform, dtype=float).reshape(4, 4)
    )
    center_w = (mn_w + mx_w) / 2.0
    span_w = np.maximum(mx_w - mn_w, 1.0)
    corners_w = np.array(
        [
            [x, y, z]
            for x in (mn_w[0], mx_w[0])
            for y in (mn_w[1], mx_w[1])
            for z in (mn_w[2], mx_w[2])
        ]
    )
    out: list[tuple[float, float, float, str, dict[str, Any]]] = []
    # uncapped solver call — the tool truncates to its top-8 by solver
    # score, which silently drops ports a hint needs
    for pair in solver.find_matching_ports(
        moving, fixed, used_fixed_ports=used, include_mismatch=True
    ):
        fp = next((p for p in fixed.ports if p.name == pair.get("fixed_port")), None)
        wp = _world_pos(inst.transform, fp.pos) if fp else [0.0, 0.0, 0.0]
        axis_hint = 0.0
        if fp is not None:
            fa = _world_dir(inst.transform, fp.axis)
            # a mouth buried deep inside the host silhouette along its own
            # axis is an interior junction — unreachable without tunneling
            # through the body, same rule the anchor orientation applies
            if (
                float((corners_w @ solver._unit(np.asarray(fa))).max() - np.dot(np.asarray(wp), solver._unit(np.asarray(fa))))
                > 60.0
            ):
                continue
            if fp.kind == "plane":
                # a rim/socket mate is what assembly manuals mean — flat
                # faces only win when no cylindrical port is available
                axis_hint -= 2.0
            axis_hint += sum(sg * fa[ax] for ax, sg in open_axes)
            if orient == "V":
                axis_hint += abs(fa[2])
            elif orient == "H":
                axis_hint += 1.0 - abs(fa[2])
            if feed_perp is not None:
                # bridge feasibility dominates location words: a stub
                # PARALLEL to the incoming leg can never be bridged to —
                # a 90° elbow physically cannot close that loop
                if abs(fa[feed_perp]) > 0.8:
                    axis_hint -= 10.0
                else:
                    axis_hint += 6.0 * (1.0 - abs(fa[feed_perp]))
                if chain_dir is not None:
                    # the mated port must also FACE the chain's side — a
                    # rear pipe end is a valid female mouth too, but a
                    # bridge from the front chain can never reach it
                    fa_xy = np.asarray([fa[0], fa[1]], dtype=float)
                    n_xy = float(np.linalg.norm(fa_xy))
                    if n_xy > 0.3:
                        axis_hint += 4.0 * float(
                            np.dot(fa_xy / n_xy, chain_dir)
                        )
            if float(pair.get("score") or 0.0) < 0.0:
                axis_hint += 2.0 * float(pair["score"])
        pos_hint = 1.5 * sum(
            sg * (wp[ax] - float(center_w[ax])) / float(span_w[ax]) for ax, sg in axes
        )
        if sibling_pos:
            # paired stubs sit tens of mm apart on one face; opposite
            # faces are hundreds of mm away — a ~150mm-decaying pull
            # binds same-part siblings to the same stub cluster without
            # ever beating the physical fit
            d = min(
                float(np.linalg.norm(np.asarray(wp) - np.asarray(sp)))
                for sp in sibling_pos
            )
            pos_hint += 3.0 * max(0.0, 1.0 - d / 150.0)
        hint = pos_hint + 25.0 * axis_hint + 4.0 * float(pair.get("score") or 0.0)
        out.append((hint, 1.0 if picked else 0.0, float(pair.get("score") or 0.0), iid, pair))
    return out


def _used_port_names(sess: GuideToolSession, inst: Any) -> set[str]:
    """Ports occupied on an instance — expanded over COINCIDENT ports:
    extraction emits both the spigot's outer rim and its bore lip at the
    same spot (male+female, same axis) — physically one connection, so
    mating one occupies all."""
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
    if fp is None:
        return used
    for name in list(used):
        p = next((q for q in fp.ports if q.name == name), None)
        if p is None:
            continue
        for q in fp.ports:
            if q.name in used:
                continue
            same_spot = max(abs(p.pos[i] - q.pos[i]) for i in range(3)) < 2.5
            same_axis = abs(sum(p.axis[i] * q.axis[i] for i in range(3))) > 0.9
            if same_spot and same_axis:
                used.add(q.name)
    return used


def _free_port_names(sess: GuideToolSession, inst: Any) -> set[str]:
    fp = sess.state.part_by_id(inst.part_id)
    return {p.name for p in (fp.ports if fp else [])} - _used_port_names(sess, inst)


def _bridge_search(
    sess: GuideToolSession, task: dict[str, Any], pid: str
) -> tuple[float, float, Any, tuple, tuple, np.ndarray] | None:
    """Two-sided mate for bridging parts (elbow joining pipe A to pipe B):
    when the sub-step names a second target and the part has >=2 ports,
    solve both mates simultaneously — the roll around the primary axis is
    chosen so the SECOND mate also closes.

    Returns ``(key, residual, transform, (i1, p1), (i2, p2), resid_vec)``
    for the best candidate or None when nothing solves. The residual is
    NOT bound-checked here — the caller decides whether it closes,
    records a ``bridge_miss`` for the lift-correction pass, or commits."""
    if not str(task.get("target2") or "").strip():
        return None
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
    if fixed1 is None or fixed2 is None:
        return None
    st_b = sess.state.step_by_number(task["step"])
    hint_b = f"{task.get('target') or ''} {task.get('target2') or ''} {task.get('text') or ''} {st_b.title if st_b else ''}"
    axes_b = _dir_axes(hint_b)

    def _side(i: Any, p: Any, port_name: str) -> float:
        if not (i.transform and p and p.ports):
            return 0.0
        fp = next((q for q in p.ports if q.name == port_name), None)
        wp = _world_pos(i.transform, fp.pos) if fp else _world_pos(
            i.transform, [(p.bbox_min[k] + p.bbox_max[k]) / 2 for k in range(3)]
        )
        return sum(sg * wp[ax] for ax, sg in axes_b)

    best: tuple[float, float, Any, tuple, tuple, np.ndarray] | None = None
    for i1 in targets[t1]:
        for i2 in targets[t2]:
            p1s = solver.find_matching_ports(
                moving, fixed1, used_fixed_ports=_used(sess, i1), include_mismatch=True
            )[:4]
            p2s = solver.find_matching_ports(
                moving, fixed2, used_fixed_ports=_used(sess, i2), include_mismatch=True
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
                    t_m = np.asarray(m, dtype=float).reshape(4, 4)
                    t_f = np.asarray(i2.transform, dtype=float).reshape(4, 4)
                    eng = solver._engage_mm(
                        mp2, fp2, moving_part=moving, fixed_part=fixed2
                    )
                    mpw = t_m[0:3, 0:3] @ solver._v(mp2.pos) + t_m[0:3, 3]
                    fpw = (
                        t_f[0:3, 0:3] @ (solver._v(fp2.pos) - solver._v(fp2.axis) * eng)
                        + t_f[0:3, 3]
                    )
                    resid_v = mpw - fpw
                    res = float(np.linalg.norm(resid_v))
                    # side words ("linkes Rohr … linkes oberes Rohr")
                    # disambiguate which INSTANCE pair the bridge closes —
                    # bounded: a direction hint may break near-ties but can
                    # never buy an unclosed gap (raw mm would swamp res)
                    side = _side(i1, fixed1, p1["fixed_port"]) + _side(
                        i2, fixed2, p2["fixed_port"]
                    )
                    key = res - 15.0 * float(np.tanh(side / 40.0))
                    if best is None or key < best[0]:
                        best = (key, res, m, (i1, p1), (i2, p2), resid_v)
    return best


def _bridge_bound(sess: GuideToolSession, task: dict[str, Any], pid: str, best: tuple) -> float:
    """Acceptance limit for a bridge residual: at most half the second
    socket's grip (plug must reach well into the bore)."""
    _key, _res, _m, _p1, (_i2, p2), _rv = best
    moving = sess.state.part_by_id(pid)
    fixed2 = sess.state.part_by_id(_resolve_name(sess, task.get("target2")) or "")
    mp2b = next((q for q in (moving.ports if moving else []) if q.name == p2["moving_port"]), None)
    fp2b = next(
        (q for q in (fixed2.ports if fixed2 else []) if q.name == p2["fixed_port"]),
        None,
    )
    sd2 = max(
        solver._socket_depth(mp2b) if mp2b is not None else 0.0,
        solver._socket_depth(fp2b) if fp2b is not None else 0.0,
    )
    return max(1.0, min(sd2 * 0.5, 15.0))


async def _try_bridge(sess: GuideToolSession, task: dict[str, Any], pid: str) -> str | None:
    best = _bridge_search(sess, task, pid)
    if best is None:
        return None
    _key, res, m, (i1, p1), (i2, p2), resid_v = best
    # a bridge that can't physically close isn't a bridge — when the best
    # residual exceeds half the second socket's grip the plug barely
    # reaches the bore; fall back to the primary mate instead of
    # recording an open gap as a joint. The measured residual is KEPT on
    # the task — a mostly-vertical miss is a lift error the correction
    # pass can still close, after which the bridge is retried.
    if res > _bridge_bound(sess, task, pid, best):
        task["bridge_miss"] = {
            "resid": [float(v) for v in resid_v],
            "i1": str(i1.instance_id),
            "i2": str(i2.instance_id),
            "res": float(res),
        }
        return None
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
            "premount": bool(task.get("premount")),
        },
    )
    if not res.get("ok"):
        return None
    insts = sess.state.assembly.instances
    iid = insts[-1].instance_id if insts else None
    sess.state.assembly.joints.append(
        {
            "step": task["step"],
            "moving_part": pid,
            "fixed_part": i2.part_id,
            "moving_instance": iid,
            "fixed_instance": i2.instance_id,
            "moving_port": p2["moving_port"],
            "fixed_port": p2["fixed_port"],
            "approach_axis": p2.get("axis") or [0.0, 0.0, 1.0],
            "bridge": True,
        }
    )
    # a cross-substep merge (if any) is confirmed only now — the leg
    # task checks this flag before deciding between skip and new instance
    task["_bridged"] = True
    return iid


def _used(sess: GuideToolSession, inst: Any) -> set[str]:
    return _used_port_names(sess, inst)


def _sibling_port_spots(sess: GuideToolSession, pid: str) -> list[list[float]]:
    """World positions of the fixed ports that already-placed instances of
    part `pid` mated onto — "rechtes Anschlussrohr" targets the OTHER stub
    of the pair: adjacent to the first sibling's mate, not a distant port."""
    insts = {i.instance_id: i for i in sess.state.assembly.instances}
    out: list[list[float]] = []
    for j in sess.state.assembly.joints:
        mv = insts.get(str(j.get("moving_instance") or ""))
        if mv is None or mv.part_id != pid:
            continue
        fi = insts.get(str(j.get("fixed_instance") or ""))
        fp_part = sess.state.part_by_id(fi.part_id) if fi else None
        fp = next(
            (p for p in (fp_part.ports if fp_part else []) if p.name == j.get("fixed_port")),
            None,
        )
        if fi is not None and fi.transform and fp is not None:
            out.append(_world_pos(fi.transform, fp.pos))
    return out


async def _try_attach(
    sess: GuideToolSession,
    task: dict[str, Any],
    pid: str,
    next_axes: tuple[list[tuple[int, float]], str | None] = ([], None),
    feed_perp: int | None = None,
    tasks: list[dict[str, Any]] | None = None,
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
    # location axes rank the FIXED port's world position; "nach <dir>"
    # opening phrases instead constrain THIS part's free ports
    axes = _dir_axes(hint_text)
    open_axes = _open_axes(hint_text)
    # the accessible side is +Y (AGENTS frame contract: +Y = front) —
    # manuals that don't say front/back mean the working side; a weak
    # bias breaks front/rear ties without overriding explicit words
    if not any(a == 1 for a, _ in axes):
        axes = [*axes, (1, 0.3)]
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
    # sibling-instance ports: a second "Pipe_l500" targeting "rechtes
    # Anschlussrohr" means the OTHER stub of the same pair — adjacent to
    # the first sibling's stub, not on the casing's opposite face
    sib_spots = _sibling_port_spots(sess, pid)
    for group in groups:
        if not group:
            continue
        anchor_iid = (
            sess.state.assembly.instances[0].instance_id
            if sess.state.assembly.instances
            else None
        )
        chain_dir = (
            _chain_side_dir(sess, tasks or [], task, feed_perp)
            if feed_perp is not None
            else None
        )
        for iid in group:
            # the perpendicular-port rule applies where a bridge-feeding
            # chain STARTS — the host port — not mid-chain, where a
            # vertical pipe legitimately mates an upward socket
            scored.extend(
                _score_pairs_for_instance(
                    sess, moving, inst_by_iid.get(iid), iid, axes, open_axes, orient,
                    feed_perp if iid == anchor_iid else None,
                    iid == picked_iid,
                    chain_dir if iid == anchor_iid else None,
                    sibling_pos=sib_spots,
                )
            )
        if scored:
            break  # resolved part yielded pairs — other part types stay out
    if not scored:
        return None
    # the picked INSTANCE outranks port-level hints: "rechter oberer
    # Adapter" means adapter#2 — a free port on the already-paired #1 must
    # never win on position alone (observed: both L70s stacked onto #1)
    scored.sort(key=lambda t: (t[1], t[0], t[2]), reverse=True)
    for _h, _r, _s, target_iid, pair in scored[:6]:
        prev = await solve_attachment_preview(
            sess,
            {
                "part_id": pid,
                "moving_port": pair.get("moving_port", ""),
                "target_part_id": target_iid,
                "target_port": pair.get("fixed_port", ""),
                "step": task["step"],
                "substep_idx": task.get("substep_idx", 0),
            },
        )
        if not prev.get("ok"):
            continue
        # orientation-aware pick among the roll/variant candidates: the
        # first collision-free one is NOT automatically the readable build.
        # Rows strip "transform" (kept in _preview_cache by candidate_id)
        fixed_inst = inst_by_iid.get(target_iid)
        fixed_part = sess.state.part_by_id(fixed_inst.part_id) if fixed_inst else None
        fp = next(
            (p for p in (fixed_part.ports if fixed_part else []) if p.name == pair.get("fixed_port")),
            None,
        )
        fp_world = _world_pos(fixed_inst.transform, fp.pos) if (fixed_inst and fp) else [0.0] * 3
        host_center = (
            _world_pos(
                fixed_inst.transform,
                [
                    (fixed_part.bbox_min[i] + fixed_part.bbox_max[i]) / 2
                    for i in range(3)
                ],
            )
            if fixed_inst and fixed_part
            else [0.0] * 3
        )
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
                    open_axes or axes,
                    orient,
                    next_axes,
                )
                + (
                    2.0
                    * _outward_free_score(
                        moving,
                        pair.get("moving_port", ""),
                        (sess._preview_cache.get(c.get("candidate_id") or "") or {}).get(
                            "transform"
                        )
                        or [],
                        fp_world,
                        host_center,
                    )
                    if _mounted_on_host(sess, fixed_inst)
                    else 0.0
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
                    "premount": bool(task.get("premount")),
                },
            )
            if res.get("ok"):
                insts = [i for i in sess.state.assembly.instances if i.part_id == pid]
                return insts[-1].instance_id if insts else None
    return None


def _rot_from_to(a: list[float], b: list[float]) -> list[list[float]]:
    """Rotation matrix (3x3) mapping unit-ish vector a onto unit-ish b."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    va /= np.linalg.norm(va) or 1.0
    vb /= np.linalg.norm(vb) or 1.0
    v = np.cross(va, vb)
    c = float(np.dot(va, vb))
    if c < -0.999999:
        # antiparallel: rotate 180° about any perpendicular axis
        perp = np.cross(va, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(va, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return (2.0 * np.outer(perp, perp) - np.eye(3)).tolist()
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return (np.eye(3) + k + k @ k * (1.0 / (1.0 + c))).tolist()


def _horiz_port_span(part: Any) -> float:
    """Largest distance between two cylinder ports projected onto the
    horizontal plane — an elbow's sideways leg length."""
    cyl = [p for p in (part.ports or []) if p.kind == "cylinder"]
    best = 0.0
    for i, a in enumerate(cyl):
        for b in cyl[i + 1 :]:
            d = float(np.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]))
            best = max(best, d)
    return best


def _walk_premount_chain(
    sess: GuideToolSession, tasks: list[dict[str, Any]], prem_idx: int
) -> tuple[list[str], dict[str, Any] | None]:
    """(mid part ids, bridge task) of the chain a premount task feeds —
    same walk as _chain_height, stopping at the bridge that closes it."""
    tip = _rp(sess, tasks[prem_idx]["part"], tasks[prem_idx]["step"])
    mids: list[str] = []
    for t in tasks[prem_idx + 1 :]:
        if t.get("premount"):
            continue
        tp = _resolve_name(sess, t.get("target") or "")
        if tp != tip:
            continue
        if str(t.get("target2") or "").strip():
            return mids, t
        pp = _rp(sess, t["part"], t["step"])
        if not pp:
            break
        mids.append(pp)
        tip = pp
    return mids, None


def _partner_tip_out(
    sess: GuideToolSession, t2: str, out: list[float]
) -> float | None:
    """Outward coordinate of the bridge partner's free end — a placed
    instance port whose axis points ~along +out."""
    part2 = sess.state.part_by_id(t2)
    if part2 is None:
        return None
    best: float | None = None
    for inst in sess.state.assembly.instances:
        if inst.part_id != t2 or not inst.transform:
            continue
        used = _used_port_names(sess, inst)
        for p in part2.ports:
            if p.kind != "cylinder" or p.name in used:
                continue
            wa = _world_dir(inst.transform, p.axis)
            if wa[0] * out[0] + wa[1] * out[1] < 0.6:
                continue
            wp = _world_pos(inst.transform, p.pos)
            s = wp[0] * out[0] + wp[1] * out[1]
            if best is None or s > best:
                best = s
    return best


def _chain_out_correction(
    sess: GuideToolSession,
    tasks: list[dict[str, Any]],
    prem_idx: int,
    out: list[float],
    ref_out: float,
    part: Any,
) -> float:
    """Back-substitute the premount's outward offset from the bridge it
    must close: the chain's column has to stand so the top elbow's
    secondary socket can land just beyond the partner's free tip. Without
    the correction the column sits next to the host and the bridge ends
    ~half a fitting short — the residual the solver can't roll away."""
    mids, bridge = _walk_premount_chain(sess, tasks, prem_idx)
    if bridge is None:
        return 0.0
    tip_out = _partner_tip_out(sess, _resolve_name(sess, bridge.get("target2") or ""), out)
    bpart = sess.state.part_by_id(_rp(sess, bridge["part"], bridge["step"]) or "")
    if tip_out is None or bpart is None:
        return 0.0
    # the column must stand so the secondary socket's mouth lands ~one
    # engagement depth BEHIND the tip plane (the pipe end inserts INTO
    # the socket, it doesn't stop at its face)
    column_needed = tip_out + max(_horiz_port_span(bpart) * 0.7, 25.0) - 14.0
    # where the column lands with the default offset: reference + the
    # premount's half-extent + each mid part's horizontal leg
    half = max(
        float(part.bbox_max[i] - part.bbox_min[i]) * abs(out[i]) for i in range(3)
    ) / 2.0
    column_default = ref_out + half + 3.0 + sum(
        _horiz_port_span(mp) * 0.7
        for mpid in mids
        if (mp := sess.state.part_by_id(mpid)) is not None
    )
    return max(0.0, column_needed - column_default)


def _side_matched_instance(
    sess: GuideToolSession,
    tasks: list[dict[str, Any]],
    pid: str,
    want_sides: set[str],
) -> Any | None:
    """The placed instance of ``pid`` sitting on the reference's side —
    branch chains are side-bound ("linker unterer Adapter" feeds the LEFT
    column). Matches the placing task's own side signature first, then the
    instance's world position along the wanted axes."""
    cands = [i for i in sess.state.assembly.instances if i.part_id == pid and i.transform]
    if len(cands) <= 1:
        return cands[0] if cands else None
    best, best_s = None, -1.0
    for i in cands:
        placer = next(
            (
                t
                for t in tasks
                if t["step"] == i.placed_in_step
                and _rp(sess, t["part"], t["step"]) == pid
                and not t.get("premount")
            ),
            None,
        )
        s = (
            float(len(_task_side_signature(placer) & want_sides))
            if placer is not None
            else 0.0
        )
        if s > best_s:
            best_s, best = s, i
    if best_s > 0:
        return best
    # no side text — fall back to position: L/R spreads along ±X
    axes = [(0, -1.0)] if "l" in want_sides else ([(0, 1.0)] if "r" in want_sides else [])
    if axes:
        return max(
            cands,
            key=lambda i: sum(
                sg
                * float(
                    np.asarray(_world_pos(i.transform, [0.0, 0.0, 0.0]))[a]
                )
                for a, sg in axes
            ),
        )
    return cands[0]


def _free_cyl_port_names(sess: GuideToolSession, inst: Any) -> set[str]:
    return {p.name for p in _free_cyl_ports(sess, inst)}


def _free_cyl_ports(sess: GuideToolSession, inst: Any) -> list[Any]:
    part = sess.state.part_by_id(inst.part_id)
    if part is None:
        return []
    used = _used_port_names(sess, inst)
    return [p for p in (part.ports or []) if p.kind == "cylinder" and p.name not in used]


def _side_filtered_chain(
    sess: GuideToolSession,
    tasks: list[dict[str, Any]],
    ti: int,
    pid: str,
    sides: set[str],
) -> tuple[list[str], dict[str, Any] | None]:
    """Forward walk from the premount, restricted to OUR side's tasks —
    sibling branches share part ids, so a right-side elbow must not
    contaminate the left chain's link order."""
    tip = pid
    mids: list[str] = []
    for t in tasks[ti + 1 :]:
        if t.get("premount") or t.get("anchor"):
            continue
        if _resolve_name(sess, t.get("target") or "") != tip:
            continue
        t_sides = _task_side_signature(t)
        if sides and t_sides and not (t_sides & sides):
            continue
        if str(t.get("target2") or "").strip():
            return mids, t
        pp = _rp(sess, t["part"], t["step"])
        if not pp:
            break
        mids.append(pp)
        tip = pp
    return mids, None


def _premount_chain_transform(
    sess: GuideToolSession,
    tasks: list[dict[str, Any]],
    ti: int,
    task: dict[str, Any],
    pid: str,
) -> list[list[float]] | None:
    """Back-solve a premounted fitting's pose through the chain it feeds.

    The fitting's real seat is wherever the chain CLOSES: the bridge at
    the top mates the chain's last pipe AND an already-placed partner
    tip — unwind the tip's world pose link by link (bridge → mids →
    premount) through the same solver the forward pass uses, and the
    only consistent premount pose falls out. Returns a 4x4 transform —
    None when the plan has no closable chain (heuristic handles those)."""
    part = sess.state.part_by_id(pid)
    if part is None:
        return None
    sides = set(task.get("_ref_sides") or [])
    mids, bridge = _side_filtered_chain(sess, tasks, ti, pid, sides)
    if bridge is None or not mids:
        return None
    bpid = _rp(sess, bridge["part"], bridge["step"])
    t2 = _resolve_name(sess, bridge.get("target2") or "")
    bpart = sess.state.part_by_id(bpid) if bpid else None
    t2part = sess.state.part_by_id(t2) if t2 else None
    if bpart is None or t2part is None:
        return None
    tip_inst = _side_matched_instance(sess, tasks, t2, sides)
    if tip_inst is None or not tip_inst.transform:
        return None
    free_tip = _free_cyl_ports(sess, tip_inst)
    if not free_tip:
        return None
    tf_tip = np.asarray(tip_inst.transform, dtype=float).reshape(4, 4)
    # the chain's second leg points toward the premount's side: a "lower"
    # premount means the pipe rises +Z toward the bridge — the bridge's
    # secondary socket therefore faces DOWN (-Z) to receive it
    leg_axis = np.array([0.0, 0.0, -1.0 if "d" in sides else 1.0])
    snaps = list(np.linspace(0.0, 360.0, 72, endpoint=False))
    b_cyl = [p for p in (bpart.ports or []) if p.kind == "cylinder"]
    if len(b_cyl) < 2:
        return None
    best_b = _solve_bridge_onto_tip(bpart, b_cyl, t2part, tf_tip, free_tip, leg_axis, snaps)
    if best_b is None:
        return None
    # unwind mid-chain top-down: each part's remaining cylinder port is
    # the link downward (pipes/elbows have exactly two ends)
    _s, prev_m, prev_port = best_b
    prev_part = bpart
    out_dir = np.asarray(_world_dir(tip_inst.transform, free_tip[0].axis))
    for mpid in reversed(mids):
        nxt_link = _solve_mid_link(sess, mpid, prev_part, prev_m, prev_port, leg_axis, out_dir, snaps)
        if nxt_link is None:
            return None
        _s, prev_m, prev_part, prev_port = nxt_link
    # final link: the premount part mates the lowest mid's free end
    best_a: tuple[float, np.ndarray] | None = None
    for ap in part.ports or []:
        if ap.kind != "cylinder":
            continue
        for c in solver.solve_attachment(
            part, ap.name, prev_part, prev_m.tolist(), prev_port,
        ):
            if not c.get("geometry_ok"):
                continue
            s = -float(c["joint_gap_mm"])
            if best_a is None or s > best_a[0]:
                best_a = (s, np.asarray(c["transform"], dtype=float).reshape(4, 4))
    return best_a[1].tolist() if best_a else None


def _solve_bridge_onto_tip(
    bpart: Any,
    b_cyl: list[Any],
    t2part: Any,
    tf_tip: np.ndarray,
    free_tip: list[Any],
    leg_axis: np.ndarray,
    snaps: list[float],
) -> tuple[float, np.ndarray, str] | None:
    """Bridge pose on the tip: whichever socket mates, its OTHER socket
    must face down the chain leg — otherwise the bridge can never close."""
    best: tuple[float, np.ndarray, str] | None = None
    for bs in b_cyl:
        other = next((p for p in b_cyl if p.name != bs.name), None)
        if other is None:
            continue
        for tp_p in free_tip:
            for c in solver.solve_attachment(
                bpart, bs.name, t2part, tf_tip.tolist(), tp_p.name,
                rotation_snaps=snaps,
            ):
                if not c.get("geometry_ok"):
                    continue
                m = np.asarray(c["transform"], dtype=float).reshape(4, 4)
                wa = solver._unit(m[0:3, 0:3] @ solver._v(other.axis))
                s = float(np.dot(wa, leg_axis)) - c["joint_gap_mm"] / 50.0
                if best is None or s > best[0]:
                    best = (s, m, other.name)
    return best


def _solve_mid_link(
    sess: GuideToolSession,
    mpid: str,
    prev_part: Any,
    prev_m: np.ndarray,
    prev_port: str,
    leg_axis: np.ndarray,
    out_dir: np.ndarray,
    snaps: list[float],
) -> tuple[float, np.ndarray, Any, str] | None:
    """One mid-chain link: the part mates its best cylinder port to the
    link above; its REMAINING port becomes the next link downward. Angled
    fittings prefer free ends pointing back toward the device body."""
    mpart = sess.state.part_by_id(mpid)
    if mpart is None:
        return None
    m_cyl = [p for p in (mpart.ports or []) if p.kind == "cylinder"]
    flat = out_dir[:2]
    flat = flat / np.linalg.norm(flat) if np.linalg.norm(flat) > 1e-6 else np.zeros(2)
    best: tuple[float, np.ndarray, Any] | None = None
    for mp_p in m_cyl:
        rest = [p for p in m_cyl if p.name != mp_p.name]
        if not rest:
            continue
        for c in solver.solve_attachment(
            mpart, mp_p.name, prev_part, prev_m.tolist(), prev_port,
            rotation_snaps=snaps,
        ):
            if not c.get("geometry_ok"):
                continue
            m = np.asarray(c["transform"], dtype=float).reshape(4, 4)
            nxt = rest[0]
            wa = solver._unit(m[0:3, 0:3] @ solver._v(nxt.axis))
            s = -float(c["joint_gap_mm"])
            if len(m_cyl) == 2 and abs(float(nxt.axis[2])) < 0.5:
                s -= 3.0 * float(np.dot(wa[:2], flat))
            else:
                s -= 3.0 * float(np.dot(wa, -leg_axis))
            if best is None or s > best[0]:
                best = (s, m, nxt)
    if best is None:
        return None
    s, m, nxt_p = best
    return s, m, mpart, nxt_p.name


async def _place_premount_ref(
    sess: GuideToolSession,
    task: dict[str, Any],
    pid: str,
    axes: list[tuple[int, float]],
    feed_perp: int | None = None,
    *,
    tasks: list[dict[str, Any]] | None = None,
    ti: int = 0,
) -> str | None:
    """Reference placement for initial-state hardware that does NOT sit on
    a host port (e.g. fittings pre-mounted on the device body, beside or
    below the casing): place the part under the host port where its chain
    closes back to the host, at floor level, with its leading free port
    pointing outward."""
    part = sess.state.part_by_id(pid)
    host_inst = sess.state.assembly.instances[0] if sess.state.assembly.instances else None
    # host = the instance the premount target resolved to, else the anchor
    if task.get("target"):
        tp = _resolve_name(sess, task["target"])
        for i in sess.state.assembly.instances:
            if i.part_id == tp:
                host_inst = i
                break
    # chain back-solve first: when the premount feeds a bridge chain, the
    # partner tip fixes its pose exactly — the heuristic is the fallback
    if tasks is not None:
        tr = _premount_chain_transform(sess, tasks, ti, task, pid)
        if tr is not None:
            cid = f"premount-chain-{pid}-{task['step']}"
            sess._preview_cache[cid] = {
                "transform": tr,
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
                    "premount": True,
                    "_premount_task": True,
                },
            )
            if res.get("ok"):
                insts = sess.state.assembly.instances
                log.info(
                    "guide.preassembly.premount_chain",
                    part=pid,
                    instance=insts[-1].instance_id if insts else None,
                )
                return insts[-1].instance_id if insts else None
    if part is None or host_inst is None or not host_inst.transform:
        return None
    host = sess.state.part_by_id(host_inst.part_id)
    if host is None:
        return None
    # reference point: the host port the chain closes at — side words give
    # left/right (the vertical word only means "on the floor"); when the
    # chain feeds a bridge, the closable port wins over the side score.
    # The pick must run on WORLD positions — under an anchor rotation a
    # local "left" port can land on the world right.
    xy_axes = [(a, s) for a, s in axes if a < 2]
    center_local = [(host.bbox_min[i] + host.bbox_max[i]) / 2 for i in range(3)]
    center_w = _world_pos(host_inst.transform, center_local)
    host_view = _world_part_view(host_inst, host)
    best_fp = _pick_host_port(host_view, xy_axes, feed_perp)
    best_p = list(best_fp.pos) if best_fp is not None else None
    if best_p is not None:
        ref_xy = best_p[:2]
    else:
        d = [0.0, 0.0, 0.0]
        for ax, sg in xy_axes:
            d[ax] += sg
        n = float(np.sqrt(d[0] ** 2 + d[1] ** 2)) or 1.0
        d = [d[0] / n, d[1] / n, 0.0]
        ext = [(host.bbox_max[i] - host.bbox_min[i]) / 2 for i in range(3)]
        t_min = min((ext[i] / abs(d[i])) for i in range(3) if abs(d[i]) > 1e-6)
        ref_xy = [center_w[i] + d[i] * t_min for i in range(3)][:2]
    # outward = the host port's own horizontal facing (the direction the
    # connection opens toward); fall back to radial from the host centre
    if best_fp is not None and abs(best_fp.axis[2]) < 0.7:
        out = [best_fp.axis[0], best_fp.axis[1], 0.0]
    else:
        out = [ref_xy[0] - center_w[0], ref_xy[1] - center_w[1], 0.0]
    n = float(np.sqrt(out[0] ** 2 + out[1] ** 2)) or 1.0
    out = [out[0] / n, out[1] / n, 0.0]
    # height: a "lower" hint drops the fitting to the ground plane — the
    # missing host body (e.g. the heat pump under its casing) can't be
    # invented, so the floor is the stable documented reference
    below = any(sg < 0 for ax, sg in axes if ax == 2)
    # outward-facing port = the one sticking out most along its own axis
    ports = part.ports or []
    out_port = (
        max(ports, key=lambda p: sum(p.pos[i] * p.axis[i] for i in range(3)))
        if ports
        else None
    )
    out_axis = out_port.axis if out_port else [1.0, 0.0, 0.0]
    r = _rot_from_to(list(out_axis), out)
    # rotate the part bbox to measure the footprint; the outward port's own
    # position defines the "free end" that must stay outside the host
    corners_l = _bbox_corners_world(part.bbox_min, part.bbox_max, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    corners_w = [
        [sum(r[i][c] * p[c] for c in range(3)) for i in range(3)]
        for p in corners_l
    ]
    min_z = min(p[2] for p in corners_w)
    z_target = (-min_z + 1.0) if below else (best_p[2] if best_p else 0.0)
    # keep the part just outside the host silhouette: its INNER face sits
    # gap mm past the reference point — the centre offset is the rotated
    # extent along `out`, not the bare gap
    pl = [(part.bbox_min[i] + part.bbox_max[i]) / 2 for i in range(3)]
    pc = [sum(r[i][c] * pl[c] for c in range(3)) for i in range(3)]
    face_in = min(sum(out[k] * p[k] for k in range(3)) for p in corners_w)
    off = 3.0 - face_in
    if tasks is not None:
        ref_out = ref_xy[0] * out[0] + ref_xy[1] * out[1]
        off += _chain_out_correction(sess, tasks, ti, out, ref_out, part)
    tr = [
        r[0] + [ref_xy[0] + out[0] * off - pc[0]],
        r[1] + [ref_xy[1] + out[1] * off - pc[1]],
        r[2] + [z_target - pc[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    cid = f"premount-ref-{pid}-{task['step']}"
    sess._preview_cache[cid] = {
        "transform": tr,
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
            "premount": True,
            "_premount_task": True,
        },
    )
    if not res.get("ok"):
        return None
    insts = sess.state.assembly.instances
    return insts[-1].instance_id if insts else None


async def _place_anchor(
    sess: GuideToolSession,
    task: dict[str, Any],
    pid: str,
    lift: float = 0.0,
    rot: np.ndarray | None = None,
) -> str | None:
    """The base/host part rests upright at the origin — env-hint previews
    drifted it into open space when nothing else was placed yet. `lift`
    raises it above the floor when lower premounted hardware needs the
    vertical room (host on a device, fittings below it). `rot` re-orients
    the host first when the manual's attach tasks need its interfaces on
    a different side — asset frames are arbitrary."""
    part = sess.state.part_by_id(pid)
    if part is None:
        return None
    r = rot if rot is not None else np.eye(3)
    mn_r, _ = _rotated_bbox(part.bbox_min, part.bbox_max, r)
    cid = f"anchor-{pid}-{task['step']}"
    sess._preview_cache[cid] = {
        "transform": [
            [float(r[0][0]), float(r[0][1]), float(r[0][2]), 0.0],
            [float(r[1][0]), float(r[1][1]), float(r[1][2]), 0.0],
            [float(r[2][0]), float(r[2][1]), float(r[2][2]), -float(mn_r[2]) + lift],
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
            "premount": bool(task.get("premount")),
        },
    )
    if not res.get("ok"):
        return None
    insts = sess.state.assembly.instances
    return insts[-1].instance_id if insts else None


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
                "premount": bool(task.get("premount")),
            },
        )
        if res.get("ok"):
            insts = [i for i in sess.state.assembly.instances if i.part_id == pid]
            return insts[-1].instance_id if insts else None
    return None
