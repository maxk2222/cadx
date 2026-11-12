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


def build_unity_guide(state) -> dict[str, Any]:
    """GuideState → unity_guide.json payload.

    Per step: display_text for the AR user, ordered instance placements
    (Unity transform per part instance — instance identity = part_id since
    each part is placed once), highlights (newly placed parts), arrows
    (approach directions from the joints of that step), cumulative
    visible set.
    """
    parts = {p.id: p for p in state.parts}
    inst_by_part = {i.part_id: i for i in state.assembly.instances}
    joint_by_step: dict[int, list[dict]] = {}
    for j in state.assembly.joints:
        joint_by_step.setdefault(int(j.get("step", 0)), []).append(j)
    steps_out: list[dict[str, Any]] = []
    visible: list[str] = []
    for step in state.steps:
        placed = state.assembly.step_instances.get(str(step.number), [])
        placements = []
        arrows = []
        for pid in placed:
            inst = inst_by_part.get(pid)
            part = parts.get(pid)
            if inst is None:
                continue
            placements.append(
                {
                    "instance_id": pid,
                    "part_id": pid,
                    "model_ref": (part.step_path or part.mesh_path or "") if part else "",
                    "transform": transform_to_unity(inst.transform),
                }
            )
            if pid not in visible:
                visible.append(pid)
        for j in joint_by_step.get(step.number, []):
            to_inst = inst_by_part.get(j.get("fixed_part", ""))
            if to_inst is None:
                continue
            m = np.asarray(to_inst.transform, dtype=float).reshape(4, 4)
            axis_local = np.asarray(j.get("approach_axis") or [0.0, 0.0, -1.0], dtype=float)
            axis_world = _unit(m[0:3, 0:3] @ axis_local)
            arrows.append(
                {
                    "instance_id": j.get("moving_part", ""),
                    "origin": to_unity_position((m[0:3, 3] - axis_world * 30.0).tolist()),
                    "direction": to_unity_position((axis_world * 1000.0).tolist()),
                }
            )
        # user-authored annotations: WORLD-frame direction hints (down/up/…)
        # anchored at the annotated part's world position — never rotated
        for an in state.annotations:
            if an.get("kind") != "arrow" or int(an.get("step", -1)) != step.number:
                continue
            tgt = inst_by_part.get(str(an.get("part_id", "")))
            part = parts.get(str(an.get("part_id", "")))
            if tgt is None or part is None:
                continue
            from cadx.guide import solver as _solver

            mn, mx = _solver.transform_bbox(part, tgt.transform)
            anchor = (np.asarray(mn) + np.asarray(mx)) / 2.0
            axis = _unit(np.asarray(an.get("approach_axis") or [0.0, 0.0, -1.0], dtype=float))
            arrows.append(
                {
                    "instance_id": str(an.get("part_id", "")),
                    "origin": to_unity_position((anchor - axis * 30.0).tolist()),
                    "direction": to_unity_position((axis * 1000.0).tolist()),
                    "label": str(an.get("label", "")),
                }
            )
        steps_out.append(
            {
                "index": step.number,
                "display_text": step.display_text or step.text,
                "internal_text": step.text,
                "placements": placements,
                "highlights": list(placed),
                "arrows": arrows,
                "visible_instance_ids": list(visible),
            }
        )
    doc_name = ""
    if state.document is not None and state.document.pdf_path:
        doc_name = state.document.pdf_path.rsplit("/", 1)[-1]
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
                "model_ref": p.step_path or p.mesh_path or "",
                "missing_generated": p.source == "generated",
            }
            for p in state.parts
        ],
        "steps": steps_out,
        "review": {"score": state.review.score, "iterations": state.review.iteration},
    }


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v
