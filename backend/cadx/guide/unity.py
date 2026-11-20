"""Unity export — unity_guide.json + coordinate conversion.

CADX frame: mm, Z-up, right-handed (x=len, y=width, z=height).
Unity frame: m, Y-up, left-handed (x=right, y=up, z=forward).

The mapping p_u = (x, z, y)·0.001 applies the spatial reflection
``M = [[1,0,0],[0,0,1],[0,1,0]]`` (det -1 — the handedness flip) plus the
mm→m scale. Rotations conjugate through it: ``R_u = M·R·Mᵀ`` — a proper
rotation stays proper. DO NOT copy KIARA's constants here — KIARA is
cm/Y-up internally; that asymmetry was a v1 plan bug.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadx.guide.scene import Scene
from cadx.guide.solver import (
    _unit,
    buried_instances,
    disconnected_instances,
    evaluate_mate,
    routed_inside_instances,
    sibling_pairs,
    touch_pairs,
    world_bbox,
)

# spatial change-of-basis (unit vectors); det = -1 → RH→LH
_M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
_MM_TO_M = 0.001


def to_unity_position(pos_cad_mm: list[float]) -> list[float]:
    """CAD (x,y,z) mm → Unity (x, z, y) m."""
    x, y, z = (float(v) for v in pos_cad_mm[:3])
    return [x * _MM_TO_M, z * _MM_TO_M, y * _MM_TO_M]


def to_unity_scale(scale_cad: list[float]) -> list[float]:
    x, y, z = (float(v) for v in scale_cad[:3])
    return [x, z, y]


def to_unity_quaternion(rot_cad_3x3: list[list[float]] | np.ndarray) -> list[float]:
    """CAD 3x3 rotation → Unity quaternion [x,y,z,w] (R_u = M·R·Mᵀ)."""
    r_u = _M @ np.asarray(rot_cad_3x3, dtype=float).reshape(3, 3) @ _M.T
    return _mat_to_quat(r_u)


def transform_to_unity(t4: list[list[float]]) -> dict[str, list[float]]:
    """Full 4x4 CAD transform → Unity {position, quaternion, scale=[1,1,1]}."""
    m = np.asarray(t4, dtype=float).reshape(4, 4)
    return {
        "position": to_unity_position(m[0:3, 3].tolist()),
        "quaternion": to_unity_quaternion(m[0:3, 0:3]),
        "scale": [1.0, 1.0, 1.0],
    }


def _mat_to_quat(r: np.ndarray) -> list[float]:
    """3x3 rotation → quaternion [x,y,z,w] (Shepperd's method)."""
    tr = float(np.trace(r))
    if tr > 0:
        s = 0.5 / float(np.sqrt(tr + 1.0))
        w, x, y, z = (
            0.25 / s,
            (r[2, 1] - r[1, 2]) * s,
            (r[0, 2] - r[2, 0]) * s,
            (r[1, 0] - r[0, 1]) * s,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * float(np.sqrt(max(1e-12, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])))
        w, x, y, z = (
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * float(np.sqrt(max(1e-12, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])))
        w, x, y, z = (
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        )
    else:
        s = 2.0 * float(np.sqrt(max(1e-12, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])))
        w, x, y, z = (
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([x, y, z, w], dtype=float)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return [round(float(v), 8) for v in q]


# --------------------------------------------------------------------------- #
# guide document assembly
# --------------------------------------------------------------------------- #

_GUIDE_VERSION = "1.0"


def _emit_scene(insts: dict[str, Any], parts: dict[str, Any]) -> Scene:
    """Placed meshes rebuilt for the emit-time interior probe — the
    mission gate already verdicted this state; an emit-side rebuild keeps
    evidence parity for downstream consumers. Fail-open: a mesh that
    cannot load here simply yields no flags, same as the gate's
    check_failed semantics."""
    sc = Scene()
    for iid, inst in insts.items():
        part = parts.get(inst.part_id)
        mp = getattr(part, "mesh_path", None) if part is not None else None
        if inst.transform is None or not mp:
            continue
        try:
            sc.add(iid, mp, inst.transform)
        except Exception:
            continue
    return sc


def _joint_metrics(state: Any) -> tuple[dict[int, dict], dict[str, float], dict[str, Any]]:
    """Physical audit numbers per joint index + buried-fraction per
    instance — emitted into the doc as durable mate evidence so gates
    and downstream tooling verify geometry, not just topology."""
    parts = {p.id: p for p in state.parts}
    insts = {i.instance_id: i for i in state.assembly.instances}
    # legacy joints key by part_id — resolve to the first instance of the
    # part; never duplicate instances under the part key (each would count
    # itself as a foreign host → false buried flags everywhere)
    part_to_iid: dict[str, str] = {}
    for i in state.assembly.instances:
        part_to_iid.setdefault(i.part_id, i.instance_id)

    def _ikey(ref: Any) -> str:
        k = str(ref or "")
        return k if k in insts else part_to_iid.get(k, k)

    bboxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for iid, inst in insts.items():
        part = parts.get(inst.part_id)
        if part is None or inst.transform is None:
            continue
        bboxes[iid] = world_bbox(
            part.bbox_min, part.bbox_max, np.asarray(inst.transform, dtype=float)
        )
    joint_pairs: list[tuple[str, str]] = []
    for j in state.assembly.joints:
        joint_pairs.append((
            _ikey(j.get("moving_instance") or j.get("moving_part")),
            _ikey(j.get("fixed_instance") or j.get("fixed_part")),
        ))
    if state.guide_mode == "blender":
        # no declared joints exist in a free-scene build — physical
        # contact IS the joint graph; without it correct insertions emit
        # as buried and reachable parts as disconnected
        have = {frozenset(p) for p in joint_pairs}
        joint_pairs += [p for p in touch_pairs(bboxes) if frozenset(p) not in have]
    buried_rows = buried_instances(bboxes, joint_pairs)
    # premounted hardware roots its own subtree — it mounts on the
    # machine body, not the casing host
    roots = {iid for iid, i in insts.items() if getattr(i, "premounted", False)}
    disconnected_rows = disconnected_instances(
        bboxes, joint_pairs, sorted(insts), extra_roots=roots
    )
    buried = {r["instance"]: r["buried_frac"] for r in buried_rows}
    metrics: dict[int, dict] = {}
    for idx, j in enumerate(state.assembly.joints):
        mv = insts.get(_ikey(j.get("moving_instance") or j.get("moving_part")))
        fx = insts.get(_ikey(j.get("fixed_instance") or j.get("fixed_part")))
        mp = next(
            (p for p in (parts.get(mv.part_id).ports if mv else []) if p.name == j.get("moving_port")),
            None,
        )
        tp = next(
            (p for p in (parts.get(fx.part_id).ports if fx else []) if p.name == j.get("fixed_port")),
            None,
        )
        m: dict[str, Any] = {}
        if mv is not None and fx is not None and mp is not None and tp is not None:
            if j.get("axis_flipped"):
                # joint was committed under the flipped moving-axis
                # convention — evaluate with the same convention or a
                # valid mate reads normal_dot=+1
                mp = mp.model_copy(update={"axis": [-a for a in mp.axis]})
            gap, ndot = evaluate_mate(
                mp,
                np.asarray(mv.transform, dtype=float).reshape(4, 4),
                tp,
                np.asarray(fx.transform, dtype=float).reshape(4, 4),
                moving_part=parts.get(mv.part_id),
                fixed_part=parts.get(fx.part_id),
            )
            m = {"gap_mm": round(gap, 3), "normal_dot": round(ndot, 3)}
            key = mv.instance_id or mv.part_id
            if key in buried:
                m["buried_frac"] = buried[key]
                m["buried"] = buried[key] > 0.45
        metrics[idx] = m
    # world dominant axis per instance — layout evidence durable in the
    # doc so gates/reviewers check "pipe hangs vertical" against numbers,
    # not renders alone
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
    # sibling pairs — two copies of the same part are almost always a
    # parallel/adjacent pair in the manual (two vertical pipes, two
    # symmetric rails). Distance + axis alignment are durable numbers the
    # review must quote instead of hand-waving "looks right".
    sib_pairs = sibling_pairs(
        bboxes,
        {iid: i.part_id for iid, i in insts.items()},
        {iid: i.transform for iid, i in insts.items() if i.transform is not None},
        parts,
    )
    audit = {
        "buried_instances": buried_rows,
        "disconnected_instances": disconnected_rows,
        "routed_inside": routed_inside_instances(
            bboxes, joint_pairs, _emit_scene(insts, parts).mesh_interior_frac
        ),
        "axis_profiles": axis_profiles,
        "sibling_pairs": sib_pairs,
    }
    return metrics, buried, audit


def _extent_along(mn: list[float], mx: list[float], axis: np.ndarray) -> float:
    """AABB projection onto a unit axis: Σ|axis_i|·dims_i — the part's
    world extent along that direction (exact for axis-aligned boxes, a
    safe upper bound otherwise)."""
    dims = np.asarray(mx, dtype=float) - np.asarray(mn, dtype=float)
    return float(np.abs(axis) @ dims)


def _joint_arrows(joints: list[dict], inst_by_id: dict, parts: dict) -> list[dict]:
    """Approach-direction arrows from this step's recorded joints — tip on
    the fixed port (the mate surface, not the part origin which can sit at
    a bbox corner), tail outside along the approach direction.

    Convention: ``direction`` is the arrow's travel direction TOWARD the
    target (same as annotation arrows) — ``approach_axis`` is the fixed
    port's outward normal (where the moving part comes FROM), so the
    emitted direction is its negation."""
    arrows = []
    for j in joints:
        to_inst = inst_by_id.get(j.get("fixed_instance") or j.get("fixed_part", ""))
        if to_inst is None or to_inst.transform is None:
            continue
        m = np.asarray(to_inst.transform, dtype=float).reshape(4, 4)
        axis_local = np.asarray(j.get("approach_axis") or [0.0, 0.0, -1.0], dtype=float)
        axis_world = _unit(m[0:3, 0:3] @ axis_local)
        anchor = m[0:3, 3].copy()
        fixed_part = parts.get(str(j.get("fixed_part", "")))
        if fixed_part is not None:
            port = next((p for p in fixed_part.ports if p.name == j.get("fixed_port")), None)
            if port is not None:
                anchor = m[0:3, 0:3] @ np.asarray(port.pos, dtype=float) + m[0:3, 3]
        # tail where the moving part comes from — its own extent along the
        # approach axis makes the arrow read as the real insertion slide
        moving = parts.get(str(j.get("moving_part", "")))
        mov_inst = inst_by_id.get(j.get("moving_instance") or j.get("moving_part", ""))
        standoff = 30.0
        if moving is not None and mov_inst is not None and mov_inst.transform is not None:
            mn, mx = world_bbox(
                moving.bbox_min,
                moving.bbox_max,
                np.asarray(mov_inst.transform, dtype=float).reshape(4, 4),
            )
            standoff = max(30.0, 0.5 * _extent_along(mn, mx, axis_world) + 15.0)
        origin = anchor + axis_world * standoff
        arrows.append(
            {
                "instance_id": j.get("moving_instance") or j.get("moving_part", ""),
                "origin": to_unity_position(origin.tolist()),
                "direction": to_unity_position((-axis_world * 1000.0).tolist()),
                "label_position": to_unity_position(
                    (origin - axis_world * min(20.0, standoff * 0.5)).tolist()
                ),
            }
        )
    return arrows


def _annotation_arrows(
    annotations: list[dict], step_no: int, inst_by_id: dict, parts: dict
) -> list[dict]:
    """User-authored annotation arrows: WORLD-frame direction hints, tip on
    the annotated part's world bbox center, tail pushed past the bbox face
    along the axis — a fixed standoff would bury the shaft inside any part
    larger than 60mm along the direction."""
    from cadx.guide import solver as _solver

    arrows = []
    for an in annotations:
        if an.get("kind") != "arrow" or int(an.get("step", -1)) != step_no:
            continue
        tgt = inst_by_id.get(str(an.get("part_id", "")))
        part = parts.get(str(an.get("part_id", "")))
        if tgt is None or part is None or tgt.transform is None:
            continue
        mn, mx = _solver.transform_bbox(part, tgt.transform)
        anchor = (np.asarray(mn) + np.asarray(mx)) / 2.0
        axis = _unit(np.asarray(an.get("approach_axis") or [0.0, 0.0, -1.0], dtype=float))
        standoff = max(25.0, 0.5 * _extent_along(mn, mx, axis) + 15.0)
        origin = anchor - axis * standoff
        arrows.append(
            {
                "instance_id": str(an.get("part_id", "")),
                "origin": to_unity_position(origin.tolist()),
                "direction": to_unity_position((axis * 1000.0).tolist()),
                "label": str(an.get("label", "")),
                "label_position": to_unity_position(
                    (origin - axis * min(20.0, standoff * 0.5)).tolist()
                ),
            }
        )
    return arrows


def _annotation_texts(
    annotations: list[dict],
    step_no: int,
    inst_by_id: dict,
    parts: dict,
    placed: list[str],
) -> list[dict]:
    """Free text fields (kind:"text"): every field emits the SAME constant
    orientation — identity quaternion, the prefab renders upright — while
    ``position`` is freely settable (REST PATCH / editor translate gizmo /
    add_text tool). Missing/zero position falls back deterministically to
    above this step's placed parts, else the visible assembly."""
    from cadx.guide import solver as _solver

    texts = []
    for an in annotations:
        if an.get("kind") != "text" or int(an.get("step", -1)) != step_no:
            continue
        pos = an.get("position")
        if not pos or len(pos) != 3 or all(abs(float(c)) < 1e-9 for c in pos):
            pos = _default_text_pos(inst_by_id, parts, placed, _solver)
        row: dict[str, Any] = {
            "text": str(an.get("text", "")),
            "position": to_unity_position([float(c) for c in pos]),
            # every text field identical orientation — consumers must NOT
            # derive rotation from the anchor part's transform
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
        if an.get("part_id"):
            row["part_id"] = str(an["part_id"])
        if an.get("color"):
            row["color"] = str(an["color"])
        if an.get("size") not in (None, 1.0):
            row["size"] = an["size"]
        texts.append(row)
    return texts


def _default_text_pos(inst_by_id: dict, parts: dict, placed: list[str], _solver: Any) -> list[float]:
    """Deterministic text-field anchor: centered above the bbox of this
    step's placed instances (fallback: all placed instances) + a fixed
    45mm standoff — an unset position must land somewhere readable, not
    at the world origin inside the assembly."""
    mn_o: list[float] | None = None
    mx_o: list[float] | None = None
    for iids in (placed, list(inst_by_id)):
        for iid in iids:
            inst = inst_by_id.get(iid)
            if inst is None or inst.transform is None:
                continue
            part = parts.get(inst.part_id)
            if part is None:
                continue
            mn, mx = _solver.transform_bbox(part, inst.transform)
            mn_o = mn if mn_o is None else [min(a, b) for a, b in zip(mn_o, mn, strict=True)]
            mx_o = mx if mx_o is None else [max(a, b) for a, b in zip(mx_o, mx, strict=True)]
        if mn_o is not None:
            break
    if mn_o is None or mx_o is None:
        return [0.0, 0.0, 45.0]
    return [(mn_o[0] + mx_o[0]) / 2.0, (mn_o[1] + mx_o[1]) / 2.0, mx_o[2] + 45.0]


def build_unity_guide(state) -> dict[str, Any]:
    """GuideState → unity_guide.json payload.

    Per step: display_text for the AR user, ordered instance placements
    (Unity transform per part instance — instance identity = part_id since
    each part is placed once), highlights (newly placed parts), arrows
    (approach directions from the joints of that step), cumulative
    visible set.
    """
    parts = {p.id: p for p in state.parts}
    inst_by_id = {i.instance_id or i.part_id: i for i in state.assembly.instances}
    for i in state.assembly.instances:  # annotations address parts by id — map to first instance
        inst_by_id.setdefault(i.part_id, i)
    joint_by_step: dict[int, list[dict]] = {}
    for j in state.assembly.joints:
        joint_by_step.setdefault(int(j.get("step", 0)), []).append(j)
    steps_out: list[dict[str, Any]] = []
    visible: list[str] = []
    for step in state.steps:
        placed = state.assembly.step_instances.get(str(step.number), [])
        placements = []
        for iid in placed:
            inst = inst_by_id.get(iid)
            if inst is None:
                continue
            part = parts.get(inst.part_id)
            placements.append(
                {
                    "instance_id": inst.instance_id or iid,
                    "part_id": inst.part_id,
                    "model_ref": _model_ref(part),
                    "transform": transform_to_unity(inst.transform),
                }
            )
            if (inst.instance_id or iid) not in visible:
                visible.append(inst.instance_id or iid)
        arrows = _joint_arrows(joint_by_step.get(step.number, []), inst_by_id, parts)
        arrows += _annotation_arrows(state.annotations, step.number, inst_by_id, parts)
        texts = _annotation_texts(state.annotations, step.number, inst_by_id, parts, placed)
        row: dict[str, Any] = {
            "index": step.number,
            "display_text": step.display_text or step.text,
            "internal_text": step.text,
            "placements": placements,
            "highlights": list(placed),
            "arrows": arrows,
            "texts": texts,
            "visible_instance_ids": list(visible),
        }
        # media_image is copied into output/media/ by _emit_artifacts —
        # reference it by file name so the JSON stays portable
        if step.media_image:
            row["media_image"] = step.media_image.rsplit("/", 1)[-1]
        if step.safety_hint:
            row["safety_hint"] = step.safety_hint
        if step.tools:
            row["tools"] = list(step.tools)
        steps_out.append(row)
    doc_name = ""
    if state.document is not None and state.document.pdf_path:
        doc_name = state.document.pdf_path.rsplit("/", 1)[-1]
    joint_metrics, _buried, audit = _joint_metrics(state)
    return {
        "guide_version": _GUIDE_VERSION,
        "title": doc_name.replace(".pdf", "") or "AR Assembly Guide",
        "source_pdf": doc_name,
        "units": "meters",
        "coordinate_system": "unity-left-handed-y-up",
        "parts": [
            {
                "part_id": p.id,
                "name": p.name,
                "source": p.source,
                "model_ref": _model_ref(p),
                "missing_generated": p.source == "generated",
            }
            for p in state.parts
            # blender-mode: orphan registrations (no instance, no step
            # requires them — the scene gate records the verdict) must
            # not ship as phantom parts
            if state.guide_mode != "blender"
            or p.id not in set(state.blender_manifest.get("orphan_parts", []))
        ],
        "steps": steps_out,
        # durable joint graph — the authoritative mate evidence the planner
        # and agent produced; gates + downstream tooling verify against THIS,
        # not derived arrows or free-text claims
        "joints": [
            {
                **{
                    k: j.get(k)
                    for k in (
                        "step",
                        "moving_part",
                        "fixed_part",
                        "moving_instance",
                        "fixed_instance",
                        "moving_port",
                        "fixed_port",
                        "bridge",
                        "materialized",
                    )
                    if j.get(k) is not None
                },
                **joint_metrics.get(idx, {}),
            }
            for idx, j in enumerate(state.assembly.joints)
        ],
        # authoritative end-state audit snapshot — gates read THESE rows,
        # never a re-derived private computation (emit↔audit consistency)
        "audit": audit,
        "review": {
            "score": state.review.score,
            "iterations": state.review.iteration,
            "visual_comparison": state.review.visual_comparison,
            "blockers": [
                f.text for f in state.review.findings if f.severity == "blocker"
            ],
        },
    }


def _model_ref(part: Any) -> str:
    """Basename of the part's best Unity-loadable model: prefer the GLB
    tessellation (STEP/BRep can't be rendered by Unity)."""
    if part is None:
        return ""
    src = part.mesh_path or part.step_path or ""
    return src.rsplit("/", 1)[-1] if src else ""
