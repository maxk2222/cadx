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
    # pre-mounted duplicates: a place-task whose TARGET text carries a
    # side qualifier ("unterer linker Adapter") refers to an instance at
    # THAT side — when the part's own placements all happen later or at
    # other sides, the manual means a pre-mounted piece the steps never
    # place. Spawn a mount task for it onto the earliest referenceable
    # host before the referencing step.
    return _spawn_premounts(sess, tasks)


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
        # side words that describe the TARGET location: the bare target
        # name plus only the first text clause (the one naming it) —
        # later clauses describe OTHER parts ("freie Seite auf das obere
        # Rohr") and would spawn bogus pre-mounts
        first_clause = (t.get("text") or "").split(",")[0]
        for cut in (" dann ", " anschließ", " und "):
            first_clause = first_clause.split(cut)[0]
        # orientation phrases ("Öffnung nach oben", "nach unten zeigend")
        # are not LOCATION qualifiers — strip them or they poison the sig
        first_clause = re.sub(
            r"nach\s+(oben|unten|links|rechts|vorne|hinten|oben zeigend)\w*",
            " ",
            first_clause.lower(),
        )
        # port qualifiers modify a FEATURE, not the instance location:
        # "obere Öffnung des linken unteren Rohrbogens" locates the elbow
        # left+lower — "obere" describes its port and must stay out of the
        # side signature (a phantom {links,unten,oben} ref spawned a bogus
        # premount elbow). "Anschluss"/"Seite" stay — they name the
        # instance's mount location in these manuals.
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
        extra.append(
            {
                "step": task["step"],
                "substep_idx": -1,
                "part": tpid,
                "target": host_pid,
                "target2": "",
                "text": f"{task.get('target') or ''} {task.get('text') or ''}",
                "premount": True,
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
    """Rotate the host so the manual's attach tasks find compatible,
    laterally reachable interfaces. The asset frame is arbitrary — a
    host port pointing straight up/down forces an impossible
    over-the-top or under-floor mate; a host's connection hardware faces
    the assembly's working side (+Y front by frame contract). Scores the
    first few host-bound tasks under every cube rotation by pair
    legality (the solver's fit) + lateral reach + the task's side words,
    then picks the best orientation."""
    host_ports = [p for p in part.ports if p.kind == "cylinder"]
    if not host_ports:
        return np.eye(3)
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
        return np.eye(3)
    low_bias = _below_chain(sess, tasks)
    best_rot, best_s = np.eye(3), -1e18
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
        corners_r = np.array(
            [
                [x, y, z]
                for x in (mn_r[0], mx_r[0])
                for y in (mn_r[1], mx_r[1])
                for z in (mn_r[2], mx_r[2])
            ]
        )
        # recession depth of each port below the rotated silhouette along
        # its own axis — a mouth 100mm inside the body can't be reached
        # without tunneling through the part's other openings
        exposure = [
            max(float((corners_r @ solver._unit(ax_w)).max() - float(np.dot(pos_w, solver._unit(ax_w)))), 0.0)
            for pos_w, ax_w, _ in ports_r
        ]
        # an opening aimed at the floor is unreachable — no mate can
        # approach through the ground plane the host rests on
        s = -1.0 * sum(1 for _, ax_w, _ in ports_r if float(ax_w[2]) < -0.8)
        for t, moving in attach:
            st = sess.state.step_by_number(t["step"])
            hint = f"{t.get('target') or ''} {t.get('text') or ''} {st.title if st else ''}"
            task_best = -1e18
            for (pos_w, ax_w, fp), exp in zip(ports_r, exposure, strict=True):
                if exp > 60.0:
                    continue  # recessed past any fitting's reach — an
                    # interior junction, not a mount interface
                cand = _interface_score(
                    moving, fp, pos_w, ax_w, exp,
                    _dir_axes(hint), span, center, pos_w[2], mn_r[2], span_z,
                    low_bias,
                )
                task_best = max(task_best, cand)
            if task_best > -1e18:
                s += task_best
        if np.allclose(rot, np.eye(3)):
            s += 0.01  # keep the asset frame when no orientation wins outright
        if s > best_s:
            best_rot, best_s = rot, s
    return best_rot


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


async def run_preassembly(sess: GuideToolSession) -> dict[str, Any]:
    """Place what the step plan determines — returns (and persists) a report."""
    report: dict[str, Any] = {"placed": [], "skipped": [], "failed": []}
    tasks = _plan(sess)
    anchor_pid = next(
        (_rp(sess, t["part"], t["step"]) for t in tasks if t.get("anchor")), None
    )
    anchor_part = sess.state.part_by_id(anchor_pid) if anchor_pid else None
    anchor_rot = (
        _anchor_orientation(sess, anchor_part, tasks) if anchor_part is not None else None
    )
    lift = _anchor_lift(sess, tasks, anchor_rot)
    for ti, task in enumerate(tasks):
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
                placed = await _try_attach(sess, task, pid, lookahead, feed_perp)
        else:
            placed = await _try_bridge(sess, task, pid)
        if not placed:
            placed = await _try_attach(sess, task, pid, lookahead, feed_perp)
        if not placed and task.get("anchor"):
            placed = await _place_anchor(sess, task, pid, lift, anchor_rot)
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
            if float(pair.get("score") or 0.0) < 0.0:
                axis_hint += 2.0 * float(pair["score"])
        pos_hint = 1.5 * sum(
            sg * (wp[ax] - float(center_w[ax])) / float(span_w[ax]) for ax, sg in axes
        )
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


async def _try_bridge(sess: GuideToolSession, task: dict[str, Any], pid: str) -> str | None:
    """Two-sided mate for bridging parts (elbow joining pipe A to pipe B):
    when the sub-step names a second target and the part has >=2 ports,
    solve both mates simultaneously — the roll around the primary axis is
    chosen so the SECOND mate also closes."""
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

    best: tuple[float, list, dict, dict] | None = None
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
                    res = solver.evaluate_mate(
                        mp2,
                        np.asarray(m),
                        fp2,
                        np.asarray(i2.transform),
                    )[0]
                    # side words ("linkes Rohr … linkes oberes Rohr")
                    # disambiguate which INSTANCE pair the bridge closes —
                    # without this a left-side bridge grabbed the right pipe
                    side = _side(i1, fixed1, p1["fixed_port"]) + _side(
                        i2, fixed2, p2["fixed_port"]
                    )
                    key = res - 25.0 * side
                    if best is None or key < best[0]:
                        best = (key, m, (i1, p1), (i2, p2))
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
    return iid


def _used(sess: GuideToolSession, inst: Any) -> set[str]:
    return _used_port_names(sess, inst)


async def _try_attach(
    sess: GuideToolSession,
    task: dict[str, Any],
    pid: str,
    next_axes: tuple[list[tuple[int, float]], str | None] = ([], None),
    feed_perp: int | None = None,
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
    for group in groups:
        if not group:
            continue
        anchor_iid = (
            sess.state.assembly.instances[0].instance_id
            if sess.state.assembly.instances
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
