"""Deterministic GuideSchema emitter — internal ``GuideState`` → the AR
client's published document (KIARA web-suite parity).

This is a PORT of the suite's V4 emit
(``ar_guide_agent_v4/tools/output_tools.py::write_unity_guide`` +
``common/guide_object_builders.py``) onto our internal state and frames:

- steps[].objects is a CUMULATIVE scene — every instance is re-emitted
  from its ``introducedAtStep`` onward; ``isNew`` + color/opacity mark the
  parts introduced by the current step.
- Object builders produce the exact field surface the client consumes
  (prefabName ``{stem}(Clone)``, pos/rot/scale, ``path``/``handPath``
  keyframe shells, ``originalObjectId = uuid5(DNS, instance_id)``,
  ``poseOrigin``/``localOverride`` conventions).
- Frame math comes from ``kiara.guide.unity``: our CAD route is Z-up mm
  (``p_u = (x, z, y)·0.001``, ``R_u = M·R·Mᵀ``); the blender route stores
  transforms already conjugated to the glTF Y-up frame, whose conversion
  is the suite's own (``(x, y, -z)`` + flipZ quaternion). Positions emit
  in meters (``sourceUnits: "m"``).
- The client rotates every loaded model 180° about Y at mesh level (suite
  C2 §2k R3) — like the suite we ALWAYS pre-compensate by post-multiplying
  the emitted rotation with R_yaw180 (the suite's ``compensate_fbx_yaw``
  semantic); ``emit_yaw_compensation=False`` exists for headset
  verification, not for fixtures.

Everything here is deterministic — no LLM, no rendering, no guessing.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kiara.guide import solver as _solver
from kiara.guide.unity import _mat_to_quat, _unit

# ---------------------------------------------------------------------------
# emit configuration — V4Config defaults (suite output_tools.Config)
# ---------------------------------------------------------------------------

NEW_PART_COLOR = "#FFD700"
NEW_PART_OPACITY = 0.85
CONTEXT_COLOR = "#808080"
CONTEXT_OPACITY = 0.70
DIM_CONTEXT_PARTS = True
HERO_COLOR = "#ff6600"
HERO_OPACITY = 1.0
HERO_GAP_M = 0.05
HERO_PHOTO_GAP_M = 0.35
HERO_PREVIEW_ENABLED = True
ARROW_COLOR = "#0000FF"
TEXT_X_OFFSET_M = 0.2
PHOTO_X_OFFSET_M = 0.2
UI_ABOVE_M = 0.5
PHOTO_TEXT_SCALE_X = 3.0
ARROW_SCALE = 0.7
ARROW_TARGET_STANDOFF_M = 0.05

GUIDE_FORMAT_VERSION = 2

# action_type (internal) → StepSemantic actionType (suite enum)
_ACTION_TYPE_MAP = {
    "assembly": "assemble",
    "check": "check",
    "safety": "other",
    "prep": "prepare",
}


def _uuid5(seed: str) -> str:
    """Deterministic object identity — uuid5(DNS, seed).

    The suite's builders feed the same seeds into uuid4; we keep the seed
    shape but make it deterministic so a republished guide keeps stable
    objectIds (required for client-side identity tracking across
    re-runs of the same project guide)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def _empty_path() -> dict[str, Any]:
    return {
        "runTime": 0,
        "loop": False,
        "timeStamps": [],
        "posPath": [],
        "rotPath": [],
        "scalePath": [],
    }


def _empty_hand_path() -> dict[str, Any]:
    hp = _empty_path()
    hp["HandData"] = {"frameList": []}
    hp["handNode"] = 4
    hp["handControllerArmature"] = {"instanceID": 0}
    return hp


def _v3(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z)}


def _q(x: float, y: float, z: float, w: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}


# ---------------------------------------------------------------------------
# frame conversion — world frame branches on state.world_frame
# ---------------------------------------------------------------------------

_YAW180 = np.diag([-1.0, 1.0, -1.0])
_M_ZUP_TO_UNITY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def _world_to_unity_pos(pos_mm: list[float] | np.ndarray, world_frame: str) -> dict[str, float]:
    """World-space position → Unity meters. z_up input is the canonical CAD
    frame (x,y,z mm → (x,z,y)·0.001 m); y_up is the blender/glTF frame
    (x,y,z mm → (x,y,-z)·0.001 m)."""
    x, y, z = (float(v) for v in list(pos_mm)[:3])
    if world_frame == "y_up":
        return _v3(x * 0.001, y * 0.001, -z * 0.001)
    return _v3(x * 0.001, z * 0.001, y * 0.001)


def _world_to_unity_dir(dir_world: list[float] | np.ndarray, world_frame: str) -> np.ndarray:
    """Unit direction (no translation) → Unity frame, still a unit vector."""
    x, y, z = (float(v) for v in list(dir_world)[:3])
    if world_frame == "y_up":
        return _unit(np.array([x, y, -z]))
    return _unit(np.array([x, z, y]))


def _world_to_unity_rot(
    rot3x3: np.ndarray, world_frame: str, compensate_yaw: bool
) -> dict[str, float]:
    """World 3x3 rotation → Unity quaternion dict {x,y,z,w}.

    z_up: R_u = M·R·Mᵀ (handedness flip). y_up: RH→LH = flip Z,
    i.e. R_u = F·R·F with F = diag(1,1,-1) — conjugating by the same
    reflection the position uses, then the suite's flipZ quaternion
    (-qx,-qy,qz,qw) — equivalent to conjugation, kept explicit for parity
    with the suite's code path.
    """
    r = np.asarray(rot3x3, dtype=float).reshape(3, 3)
    if world_frame == "y_up":
        f = np.diag([1.0, 1.0, -1.0])
        r_u = f @ r @ f
        if compensate_yaw:
            r_u = r_u @ _YAW180
        qx, qy, qz, qw = _mat_to_quat(r_u)
        # suite flipZ convention for y_up inputs
        return _q(-qx, -qy, qz, qw)
    r_u = _M_ZUP_TO_UNITY @ r @ _M_ZUP_TO_UNITY.T
    if compensate_yaw:
        r_u = r_u @ _YAW180
    x, y, z, w = _mat_to_quat(r_u)
    return _q(x, y, z, w)


def _as_t4(t4: Any) -> np.ndarray:
    """Coerce any 4x4-ish transform (list, ndarray, empty/None) to a matrix."""
    if t4 is None:
        return np.eye(4)
    try:
        m = np.asarray(t4, dtype=float)
    except (TypeError, ValueError):
        return np.eye(4)
    if m.size != 16:
        return np.eye(4)
    return m.reshape(4, 4)


def _transform_to_unity(
    t4: Any, world_frame: str, compensate_yaw: bool
) -> tuple[dict[str, float], dict[str, float]]:
    m = _as_t4(t4)
    pos = _world_to_unity_pos(m[0:3, 3], world_frame)
    rot = _world_to_unity_rot(m[0:3, 0:3], world_frame, compensate_yaw)
    return pos, rot


def _direction_to_arrow_quaternion(direction_unity: np.ndarray) -> dict[str, float]:
    """Port of the suite's ``direction_to_arrow_quaternion``: rotate the
    arrow prefab's local -Y onto the Unity-space direction, with the
    client's mesh-level R_yaw180 already compensated (R_yaw180⁻¹·d)."""
    d = np.asarray(direction_unity, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return _q(0.0, 0.0, 0.0, 1.0)
    d = d / n
    target = np.array([-d[0], d[1], -d[2]])
    tn = float(np.linalg.norm(target))
    if tn < 1e-9:
        return _q(0.0, 0.0, 0.0, 1.0)
    target = target / tn
    src = np.array([0.0, -1.0, 0.0])
    cross = np.cross(src, target)
    dot = float(np.dot(src, target))
    if float(np.linalg.norm(cross)) < 1e-9:
        if dot > 0:
            return _q(0.0, 0.0, 0.0, 1.0)
        return _q(1.0, 0.0, 0.0, 0.0)
    axis = cross / np.linalg.norm(cross)
    half = float(np.arccos(np.clip(dot, -1.0, 1.0))) / 2.0
    return _q(
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
        math.cos(half),
    )


# ---------------------------------------------------------------------------
# object builders — suite field surface verbatim
# ---------------------------------------------------------------------------


def _estimate_text_field_scale(text: str) -> dict[str, float]:
    lines = (text or "").split("\n")
    num_lines = max(1, len(lines))
    max_line_len = max((len(line.strip()) for line in lines), default=0)
    scale_x = min(1.5 + max(0, max_line_len - 40) * 0.025, 3.5)
    scale_x = max(1.5, scale_x)
    scale_y = min(1.5 + (num_lines - 1) * 0.5, 5.0)
    scale_y = max(1.5, scale_y)
    return _v3(scale_x, scale_y, 1.5)


def _base_object(obj_id: str, step: int, obj_num: int) -> dict[str, Any]:
    return {
        "objectId": obj_id,
        "step": step,
        "objectNumber": obj_num,
        "filename": "",
        "prefabName": "",
        "isTextField": False,
        "text": "",
        "isPhoto": False,
        "photoLocation": "",
        "aspectRatio": 0,
        "resolution": [0, 0],
        "isVideo": False,
        "videoLocation": "",
        "noResolutionScale": _v3(0, 0, 0),
        "isAudio": False,
        "voiceLocation": "",
        "is3DModel": False,
        "modelLocation": "",
        "pos": _v3(0, 0, 0),
        "rot": _q(0, 0, 0, 1),
        "scale": _v3(1, 1, 1),
        "hasAnimation": False,
        "path": _empty_path(),
        "isHand": False,
        "handPath": _empty_hand_path(),
        "isVisible": True,
        "originalObjectId": obj_id,
        "localOverride": False,
        "introducedAtStep": step,
        "poseOrigin": "web",
    }


def _build_text_field(
    step: int, obj_num: int, text: str, introduced_at: int, pos: dict[str, float]
) -> dict[str, Any]:
    obj_id = _uuid5(f"textfield_step{step}_obj{obj_num}")
    obj = _base_object(obj_id, step, obj_num)
    obj.update(
        {
            "prefabName": "P_MovableTextFieldWithButton(Clone)",
            "isTextField": True,
            "text": text,
            "scale": _estimate_text_field_scale(text),
            "pos": pos,
            "localOverride": True,
            "introducedAtStep": introduced_at,
            "poseOrigin": "unity",
        }
    )
    return obj


def _build_cad_object(
    step: int,
    obj_num: int,
    filename: str,
    guide_id: str,
    pos: dict[str, float],
    rot: dict[str, float],
    introduced_at: int,
    instance_id: str,
    *,
    is_new: bool,
    color: str,
    opacity: float,
    scale: dict[str, float] | None = None,
    is_hero_preview: bool = False,
) -> dict[str, Any]:
    obj_id = _uuid5(f"cad_{filename}_step{step}_obj{obj_num}")
    lineage = _uuid5(instance_id) if instance_id else obj_id
    obj = _base_object(obj_id, step, obj_num)
    obj.update(
        {
            "filename": filename,
            "prefabName": f"{Path(filename).stem}(Clone)",
            "is3DModel": True,
            "modelLocation": f"/api/assets/{guide_id}/cad_models/{filename}",
            "assetId": "",
            "authoredHash": "",
            "pos": pos,
            "rot": rot,
            "scale": scale if scale is not None else _v3(1, 1, 1),
            "color": color,
            "opacity": opacity,
            "originalObjectId": lineage,
            "instanceId": instance_id,
            "isNew": is_new,
            "introducedAtStep": introduced_at,
            "sourceUnits": "m",
        }
    )
    if is_hero_preview:
        obj["isHeroPreview"] = True
    return obj


def _build_arrow_object(
    step: int,
    obj_num: int,
    target_unity: dict[str, float],
    dir_unity: np.ndarray,
    introduced_at: int,
    *,
    target_lineage_id: str = "",
    label: str = "",
    scale: float = ARROW_SCALE,
    color: str = ARROW_COLOR,
) -> dict[str, Any]:
    obj_id = _uuid5(f"arrow_step{step}_obj{obj_num}")
    arrow_pos = _v3(
        target_unity["x"] - float(dir_unity[0]) * ARROW_TARGET_STANDOFF_M,
        target_unity["y"] - float(dir_unity[1]) * ARROW_TARGET_STANDOFF_M,
        target_unity["z"] - float(dir_unity[2]) * ARROW_TARGET_STANDOFF_M,
    )
    obj = _base_object(obj_id, step, obj_num)
    obj.update(
        {
            "filename": "ArrowStraightLP.fbx",
            "prefabName": "ArrowStraightLP(Clone)",
            "is3DModel": True,
            "text": label,
            "pos": arrow_pos,
            "rot": _direction_to_arrow_quaternion(dir_unity),
            "scale": _v3(scale, scale, scale),
            "color": color,
            "opacity": 1,
            "introducedAtStep": introduced_at,
            "sourceUnits": "m",
            "targetPosition": target_unity,
            "targetLineageId": target_lineage_id,
        }
    )
    return obj


def _build_step_photo(
    step: int,
    obj_num: int,
    photo_location: str,
    introduced_at: int,
    pos: dict[str, float],
    photo_obj_id: str,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    obj_id = photo_obj_id or _uuid5(f"photo_step{step}_obj{obj_num}")
    w, h = resolution
    aspect = w / h if h > 0 else 1.0
    img_scale_x = PHOTO_TEXT_SCALE_X
    img_scale_y = PHOTO_TEXT_SCALE_X / aspect if aspect > 0 else PHOTO_TEXT_SCALE_X
    obj = _base_object(obj_id, step, obj_num)
    obj.update(
        {
            "prefabName": "P_Photograph",
            "isPhoto": True,
            "photoLocation": photo_location,
            "aspectRatio": aspect,
            "resolution": [w, h],
            "pos": pos,
            # pre-compensated +90°X so the client's mesh-level fix lands the
            # photo upright facing +Z (suite build_step_photo verbatim)
            "rot": _q(0.7071067811865476, 0, 0, 0.7071067811865475),
            "scale": _v3(img_scale_x, img_scale_y, img_scale_x),
            "localOverride": True,
            "introducedAtStep": introduced_at,
            "poseOrigin": "unity",
            "sourceUnits": "m",
        }
    )
    return obj


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------


def _compute_ui_pos(
    cumulative_iids: list[str],
    inst_by_id: dict[str, Any],
    world_frame: str,
) -> dict[str, float]:
    """Mean x/z of instance origins + (max origin y + UI_ABOVE_M) — the
    suite's compute_ui_pos_above_assembly_meters, world-frame aware."""
    centers: list[dict[str, float]] = []
    for iid in cumulative_iids:
        inst = inst_by_id.get(iid)
        if inst is None or not inst.transform:
            continue
        t = np.asarray(inst.transform, dtype=float).reshape(4, 4)
        centers.append(_world_to_unity_pos(t[0:3, 3], world_frame))
    if not centers:
        return _v3(0.0, UI_ABOVE_M, 0.0)
    mean_x = sum(c["x"] for c in centers) / len(centers)
    mean_z = sum(c["z"] for c in centers) / len(centers)
    max_y = max(c["y"] for c in centers)
    return _v3(mean_x, max_y + UI_ABOVE_M, mean_z)


def _hero_placement(
    t4: Any,
    part: Any,
    target_pos: dict[str, float],
    world_frame: str,
    compensate_yaw: bool,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Port of compute_hero_placement: hero sits at original scale so its
    bbox center lands on target_pos; returns (pos, rot, bbox_width_m)."""
    t = _as_t4(t4)
    pos, rot = _transform_to_unity(t, world_frame, compensate_yaw)
    bbox_width_m = 0.1
    mn = getattr(part, "bbox_min", None)
    mx = getattr(part, "bbox_max", None)
    if mn and mx and len(mn) >= 3 and len(mx) >= 3:
        bmin = np.asarray(mn[:3], dtype=float)
        bmax = np.asarray(mx[:3], dtype=float)
        max_dim = float(np.max(bmax - bmin))
        if max_dim > 1e-6:
            bbox_width_m = max_dim * 0.001  # mm → m
            bc_local = (bmin + bmax) * 0.5
            bc_world = (t[0:3, 0:3] @ bc_local) + t[0:3, 3]
            bc_unity = _world_to_unity_pos(bc_world, world_frame)
            pos = _v3(
                target_pos["x"] - (bc_unity["x"] - pos["x"]),
                target_pos["y"] - (bc_unity["y"] - pos["y"]),
                target_pos["z"] - (bc_unity["z"] - pos["z"]),
            )
    return pos, rot, bbox_width_m


# ---------------------------------------------------------------------------
# arrows from our joint + annotation evidence
# ---------------------------------------------------------------------------


def _joint_arrow_rows(
    joints: list[dict],
    inst_by_id: dict[str, Any],
    parts: dict[str, Any],
    world_frame: str,
) -> list[dict[str, Any]]:
    """Approach-direction evidence → (target_unity, dir_unity, lineage, label).

    Same geometry as ``unity._joint_arrows``: the arrow's travel direction
    points TOWARD the fixed port anchor; the target sits on the port."""
    rows = []
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
        target_unity = _world_to_unity_pos(anchor, world_frame)
        dir_unity = _world_to_unity_dir(-axis_world, world_frame)
        lineage_id = str(to_inst.instance_id or j.get("fixed_part", ""))
        rows.append(
            {
                "target": target_unity,
                "direction": dir_unity,
                "lineage": _uuid5(lineage_id) if lineage_id else "",
                "label": "",
            }
        )
    return rows


def _annotation_arrow_rows(
    annotations: list[dict],
    step_no: int,
    inst_by_id: dict[str, Any],
    parts: dict[str, Any],
    world_frame: str,
) -> list[dict[str, Any]]:
    rows = []
    for an in annotations:
        if an.get("kind") != "arrow" or int(an.get("step", -1)) != step_no:
            continue
        tgt = inst_by_id.get(str(an.get("part_id", "")))
        part = parts.get(str(an.get("part_id", "")))
        if tgt is None or part is None or tgt.transform is None:
            continue
        mn, mx = _solver.transform_bbox(part, tgt.transform)
        anchor = (np.asarray(mn, dtype=float) + np.asarray(mx, dtype=float)) / 2.0
        axis = _unit(np.asarray(an.get("approach_axis") or [0.0, 0.0, -1.0], dtype=float))
        target_unity = _world_to_unity_pos(anchor, world_frame)
        dir_unity = _world_to_unity_dir(axis, world_frame)
        lineage_id = str(tgt.instance_id or an.get("part_id", ""))
        rows.append(
            {
                "target": target_unity,
                "direction": dir_unity,
                "lineage": _uuid5(lineage_id) if lineage_id else "",
                "label": str(an.get("label", "")),
                "color": str(an.get("color") or ARROW_COLOR),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


def _model_filename(part: Any, overrides: dict[str, str] | None = None) -> str:
    """Filename the client downloads — the part's Unity-loadable mesh
    (GLB/FBX basename); mirrors unity._model_ref with a deterministic
    fallback so every CAD object carries a non-empty filename/prefab.
    ``overrides`` carries the publish materializer's collision-safe names."""
    if part is None:
        return ""
    pid = str(getattr(part, "id", "") or "")
    if overrides and pid and pid in overrides:
        return overrides[pid]
    src = getattr(part, "mesh_path", None) or getattr(part, "step_path", None) or ""
    if src:
        return src.rsplit("/", 1)[-1]
    return f"{pid or 'part'}.glb"


def _resolve_photo(
    guide_id: str,
    media_image: str | None,
) -> tuple[str, tuple[int, int]]:
    """(photoLocation, (w,h)) — media ships under output/media/ in the
    published guide tree; resolution read from the image when reachable."""
    if not media_image:
        return "", (768, 768)
    name = media_image.rsplit("/", 1)[-1]
    resolution = (768, 768)
    try:
        from PIL import Image

        with Image.open(media_image) as img:
            resolution = (img.width, img.height)
    except Exception:
        pass
    return f"/api/assets/{guide_id}/output/media/{name}", resolution


@dataclass
class _EmitCtx:
    """Shared lookups for per-step object emission."""

    world_frame: str
    yaw: bool
    guide_id: str
    photo_obj_id: str
    parts: dict[str, Any]
    inst_by_id: dict[str, Any]
    step_to_iids: dict[str, list[str]]
    joint_by_step: dict[int, list[dict]]
    annotations: list[dict]
    filename_overrides: dict[str, str] | None = None


def _first_step_of(ctx: _EmitCtx, iid: str, default: int) -> int:
    """Earliest step index that introduces instance `iid`."""
    introduced = default
    for s_key, s_iids in ctx.step_to_iids.items():
        if iid in s_iids:
            cand = int(s_key)
            if cand < introduced:
                introduced = cand
    return introduced


def _step_model_objects(
    ctx: _EmitCtx, step_idx: int, cum_iids: list[str], obj_num: int
) -> tuple[list[dict], int]:
    """Cumulative CAD scene: every visible instance re-emitted; new ones lit."""
    step_new_iids = set(ctx.step_to_iids.get(str(step_idx), []))
    objects: list[dict] = []
    for iid in cum_iids:
        inst = ctx.inst_by_id.get(iid)
        if inst is None or inst.transform is None:
            continue
        part = ctx.parts.get(inst.part_id)
        filename = _model_filename(part, ctx.filename_overrides)
        pos, rot = _transform_to_unity(inst.transform, ctx.world_frame, ctx.yaw)
        is_step_new = iid in step_new_iids
        if is_step_new:
            color, opacity = NEW_PART_COLOR, NEW_PART_OPACITY
        elif DIM_CONTEXT_PARTS:
            color, opacity = CONTEXT_COLOR, CONTEXT_OPACITY
        else:
            color, opacity = "#FFFFFF", 1.0
        # editor overrides are rendering-only — keep pipeline palette
        # semantics but honor explicit user choices on the instance
        if getattr(inst, "color", None):
            color = inst.color
        if getattr(inst, "opacity", None) is not None:
            opacity = float(inst.opacity)
        obj_num += 1
        objects.append(
            _build_cad_object(
                step_idx,
                obj_num,
                filename,
                ctx.guide_id,
                pos,
                rot,
                _first_step_of(ctx, iid, step_idx),
                iid,
                is_new=is_step_new,
                color=color,
                opacity=opacity,
            )
        )
    return objects, obj_num


def _step_hero_objects(
    ctx: _EmitCtx,
    step_idx: int,
    start_x: float,
    ui_pos: dict[str, float],
    obj_num: int,
) -> tuple[list[dict], int]:
    """Hero previews — one extra object per NEW part, original scale, in a
    row right of the step photo."""
    objects: list[dict] = []
    current_x = start_x
    for h_iid in ctx.step_to_iids.get(str(step_idx), []):
        h_inst = ctx.inst_by_id.get(h_iid)
        if h_inst is None or h_inst.transform is None:
            continue
        h_part = ctx.parts.get(h_inst.part_id)
        h_filename = _model_filename(h_part, ctx.filename_overrides)
        h_t4 = np.asarray(h_inst.transform, dtype=float).reshape(4, 4)
        bbox_width_m = 0.1
        bmin = getattr(h_part, "bbox_min", None) if h_part is not None else None
        bmax = getattr(h_part, "bbox_max", None) if h_part is not None else None
        if bmin and bmax:
            try:
                d = np.asarray(bmax[:3], dtype=float) - np.asarray(bmin[:3], dtype=float)
                md = float(np.max(d))
                if md > 1e-6:
                    bbox_width_m = md * 0.001
            except Exception:
                pass
        target = _v3(current_x + bbox_width_m / 2, ui_pos["y"], ui_pos["z"])
        h_pos, h_rot, _ = _hero_placement(h_t4, h_part, target, ctx.world_frame, ctx.yaw)
        obj_num += 1
        objects.append(
            _build_cad_object(
                step_idx,
                obj_num,
                h_filename,
                ctx.guide_id,
                h_pos,
                h_rot,
                _first_step_of(ctx, h_iid, step_idx),
                h_iid,
                is_new=True,
                color=HERO_COLOR,
                opacity=HERO_OPACITY,
                scale=_v3(1.0, 1.0, 1.0),
                is_hero_preview=True,
            )
        )
        current_x += bbox_width_m + HERO_GAP_M
    return objects, obj_num


def _step_annotation_texts(
    ctx: _EmitCtx, step_idx: int, ui_pos: dict[str, float], obj_num: int
) -> tuple[list[dict], int]:
    """User-authored annotation text fields (positions stored in world mm)."""
    objects: list[dict] = []
    for an in ctx.annotations:
        if an.get("kind") != "text" or int(an.get("step", -1)) != step_idx:
            continue
        pos_mm = an.get("position")
        if not pos_mm or len(pos_mm) != 3 or all(abs(float(c)) < 1e-9 for c in pos_mm):
            apos = ui_pos
        else:
            apos = _world_to_unity_pos([float(c) for c in pos_mm], ctx.world_frame)
        obj_num += 1
        a_obj = _build_text_field(step_idx, obj_num, str(an.get("text", "")), step_idx, apos)
        a_obj["sourceUnits"] = "m"
        if an.get("color"):
            a_obj["color"] = str(an["color"])
        size = an.get("size")
        if size not in (None, 1.0):
            try:
                f = float(size)
                a_obj["scale"] = _v3(
                    a_obj["scale"]["x"] * f,
                    a_obj["scale"]["y"] * f,
                    a_obj["scale"]["z"],
                )
            except (TypeError, ValueError):
                pass
        objects.append(a_obj)
    return objects, obj_num


def _step_arrow_objects(
    ctx: _EmitCtx, step_idx: int, obj_num: int
) -> tuple[list[dict], int]:
    """Joint approach-evidence arrows + user annotation arrows."""
    objects: list[dict] = []
    for row in _joint_arrow_rows(
        ctx.joint_by_step.get(step_idx, []), ctx.inst_by_id, ctx.parts, ctx.world_frame
    ):
        obj_num += 1
        objects.append(
            _build_arrow_object(
                step_idx,
                obj_num,
                row["target"],
                row["direction"],
                step_idx,
                target_lineage_id=row["lineage"],
            )
        )
    for row in _annotation_arrow_rows(
        ctx.annotations, step_idx, ctx.inst_by_id, ctx.parts, ctx.world_frame
    ):
        obj_num += 1
        objects.append(
            _build_arrow_object(
                step_idx,
                obj_num,
                row["target"],
                row["direction"],
                step_idx,
                target_lineage_id=row["lineage"],
                label=row["label"],
                color=row.get("color") or ARROW_COLOR,
            )
        )
    return objects, obj_num


def build_guide_doc(
    state: Any,
    guide_id: str,
    *,
    title: str = "",
    source_file_id: str = "",
    emit_yaw_compensation: bool = True,
    filename_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """GuideState → published GuideSchema document dict.

    Deterministic: same state in → byte-stable doc out (uuid5-seeded
    objectIds, no timestamps of our own — createdAt/updatedAt/version are
    stamped by the mutation primitive).
    """
    world_frame = getattr(state, "world_frame", "z_up")
    yaw = bool(emit_yaw_compensation)
    parts = {p.id: p for p in state.parts}
    inst_by_id = {i.instance_id or i.part_id: i for i in state.assembly.instances}
    for i in state.assembly.instances:
        inst_by_id.setdefault(i.part_id, i)
    joint_by_step: dict[int, list[dict]] = {}
    for j in state.assembly.joints:
        joint_by_step.setdefault(int(j.get("step", 0)), []).append(j)
    step_to_iids = state.assembly.step_instances or {}

    # cumulative instance set per step (suite §output: ordered first-seen)
    ordered_steps = sorted(state.steps, key=lambda s: s.number)
    cumulative_iids: dict[int, list[str]] = {}
    cum: list[str] = []
    for s in ordered_steps:
        for iid in step_to_iids.get(str(s.number), []):
            if iid and iid not in cum:
                cum.append(iid)
        cumulative_iids[s.number] = list(cum)

    photo_obj_id = _uuid5("step_photo_shared")
    ctx = _EmitCtx(
        world_frame=world_frame,
        yaw=yaw,
        guide_id=guide_id,
        photo_obj_id=photo_obj_id,
        parts=parts,
        inst_by_id=inst_by_id,
        step_to_iids=step_to_iids,
        joint_by_step=joint_by_step,
        annotations=list(state.annotations),
        filename_overrides=filename_overrides,
    )
    guide_steps: list[dict[str, Any]] = []

    for s in ordered_steps:
        step_idx = s.number
        step_text = s.display_text or s.text
        objects: list[dict[str, Any]] = []
        obj_num = 0

        cum_iids = cumulative_iids.get(step_idx, [])
        ui_pos = _compute_ui_pos(cum_iids, inst_by_id, world_frame)

        # 1. step text field — left of the photo
        obj_num += 1
        text_pos = _v3(ui_pos["x"] - TEXT_X_OFFSET_M, ui_pos["y"], ui_pos["z"])
        text_obj = _build_text_field(step_idx, obj_num, step_text, step_idx, text_pos)
        text_obj["sourceUnits"] = "m"
        objects.append(text_obj)

        # 2. arrows — joint approach evidence + user annotation arrows
        objs, obj_num = _step_arrow_objects(ctx, step_idx, obj_num)
        objects.extend(objs)

        # 3. CAD models — cumulative scene; new parts highlighted
        objs, obj_num = _step_model_objects(ctx, step_idx, cum_iids, obj_num)
        objects.extend(objs)

        # 4. step photo — shared objectId across steps (P_Photograph)
        photo_x = ui_pos["x"] + PHOTO_X_OFFSET_M
        photo_pos = _v3(photo_x, ui_pos["y"], ui_pos["z"])
        photo_location, resolution = _resolve_photo(guide_id, s.media_image)
        obj_num += 1
        objects.append(
            _build_step_photo(
                step_idx, obj_num, photo_location, step_idx, photo_pos,
                photo_obj_id, resolution,
            )
        )

        # 5. hero previews — one extra CAD object per NEW part this step
        if HERO_PREVIEW_ENABLED:
            objs, obj_num = _step_hero_objects(
                ctx, step_idx, photo_x + HERO_PHOTO_GAP_M, ui_pos, obj_num
            )
            objects.extend(objs)

        # 6. annotation text fields — user-authored, positions in world mm
        objs, obj_num = _step_annotation_texts(ctx, step_idx, ui_pos, obj_num)
        objects.extend(objs)

        guide_steps.append(
            {
                "objects": objects,
                "semantic": {
                    "actionType": _ACTION_TYPE_MAP.get(s.action_type, "other"),
                    "focusParts": list(s.required_parts or []),
                    "tools": list(s.tools or []),
                    "safetyHint": s.safety_hint or "",
                },
                "step": step_idx,
            }
        )

    doc: dict[str, Any] = {
        "guideId": guide_id,
        "title": title or "AR Assembly Guide",
        "formatVersion": GUIDE_FORMAT_VERSION,
        "steps": guide_steps,
        "unityGuidePath": f"/api/assets/{guide_id}/output/ar_guide_generation/unity_guide.json",
    }
    if source_file_id:
        doc["source_file_id"] = str(source_file_id)
    return doc
