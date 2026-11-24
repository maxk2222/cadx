"""Annotate-mission tools — arrows/highlights layered on the finished
assembly (KIARA AnnotationAgent port). Stored on ``state.annotations`` and
folded into the unity guide at emit time.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from kiara.guide.tools import _fn, _i, _s
from kiara.logging import get_logger

if TYPE_CHECKING:
    from kiara.guide.session import GuideToolSession

log = get_logger(__name__)

SCHEMAS: dict[str, dict[str, Any]] = {
    "add_arrow": _fn(
        "add_arrow",
        "Add an annotation arrow to a step: direction + anchor part. "
        "Direction is a named hint (down/in/-x/…) — the server resolves "
        "it against the placed part, you never supply coordinates.",
        {
            "step": _i("step number"),
            "part_id": _s("part the arrow points at"),
            "direction": _s(
                "approach direction: down|up|left|right|front|back|into|-x|+x|-y|+y|-z|+z"
            ),
            "label": _s("optional short label e.g. 'insert here'"),
        },
        ["step", "part_id", "direction"],
    ),
    "add_text": _fn(
        "add_text",
        "Add a free text field to a step (note/warning that is not an arrow). "
        "Anchor it to a part/instance — deterministic position above its "
        "top face — or pass an explicit world position [x,y,z] in mm. "
        "Text fields always render in one fixed upright orientation; only "
        "their position varies.",
        {
            "step": _i("step number"),
            "text": _s("short note shown in AR"),
            "part_id": _s("optional part/instance the note belongs to"),
            "position": {
                "type": "array",
                "items": {"type": "number"},
                "description": "optional world position [x,y,z] mm (CAD frame)",
            },
        },
        ["step", "text"],
    ),
    "list_arrows": _fn(
        "list_arrows",
        "List all annotations recorded so far.",
        {},
        [],
    ),
    "remove_arrow": _fn(
        "remove_arrow",
        "Remove an annotation by its index from list_arrows.",
        {"index": _i("annotation index")},
        ["index"],
    ),
}

# solver-frame (Z-up) arrow approach axes. Side words follow the operator
# convention (solver.GUIDE_*): the assembly's working face is +Y, so "front"
# approaches from +Y and operator-left is +X — the previous table had
# front/back swapped AND left/right part-relative.
_DIR_VECS = {
    "down": [0.0, 0.0, -1.0],
    "up": [0.0, 0.0, 1.0],
    "left": [1.0, 0.0, 0.0],
    "right": [-1.0, 0.0, 0.0],
    "front": [0.0, 1.0, 0.0],
    "back": [0.0, -1.0, 0.0],
    "into": [0.0, 0.0, -1.0],
    "in": [0.0, 0.0, -1.0],
    "-x": [-1.0, 0.0, 0.0],
    "+x": [1.0, 0.0, 0.0],
    "-y": [0.0, -1.0, 0.0],
    "+y": [0.0, 1.0, 0.0],
    "-z": [0.0, 0.0, -1.0],
    "+z": [0.0, 0.0, 1.0],
}


def _step_no(raw: Any) -> int:
    """Forgiving step-number parse — models emit "step 4"/"4"/4.0 alike."""
    if isinstance(raw, (int, float)):
        return int(raw)
    m = re.search(r"\d+", str(raw or ""))
    return int(m.group(0)) if m else 0


def _resolve_arrow_part(s: GuideToolSession, raw: str) -> str | None:
    """part_id | instance_id | slug | manual name → GuidePart.id."""
    from kiara.guide.assembly_tools import _resolve_inst, _resolve_name

    pid = str(raw or "").strip()
    part = s.state.part_by_id(pid) or s.state.part_by_id(pid.split("#", 1)[0])
    if part is not None:
        return part.id
    inst, _ = _resolve_inst(s, pid)
    if inst is not None:
        return inst.part_id
    return _resolve_name(s, pid)


async def add_arrow(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step_no = _step_no(a.get("step", 0))
    if s.state.step_by_number(step_no) is None:
        nums = [st.number for st in s.state.steps]
        return {"ok": False, "error": f"step {step_no} not found — valid steps: {nums}"}
    pid = _resolve_arrow_part(s, str(a.get("part_id", "")))
    if pid is None:
        ids = [p.id for p in s.state.parts][:15]
        insts = [i.instance_id for i in s.state.assembly.instances][:15]
        return {
            "ok": False,
            "error": f"part '{a.get('part_id')}' not found — valid part_ids: {ids}; "
            f"instance ids: {insts} (names/aliases resolve too)",
        }
    d = str(a.get("direction", "")).strip().lower()
    vec = _DIR_VECS.get(d)
    if vec is None:
        return {"ok": False, "error": f"direction must be one of {sorted(_DIR_VECS)}"}
    s.state.annotations.append(
        {
            "kind": "arrow",
            "step": step_no,
            "part_id": pid,
            "approach_axis": vec,
            "label": str(a.get("label", ""))[:80],
        }
    )
    return {"ok": True, "index": len(s.state.annotations) - 1}


def _insts_of_part(s: GuideToolSession, pid: str) -> list[Any]:
    return [i for i in s.state.assembly.instances if i.part_id == pid]


def _top_anchor(s: GuideToolSession, raw: str) -> tuple[list[float] | None, str | None]:
    """Default text-field position: centered above the anchored instance's
    world bbox top + a fixed standoff. Returns (position, resolved part_id)."""
    from kiara.guide.assembly_tools import _resolve_inst
    from kiara.guide.solver import transform_bbox

    inst, _ = _resolve_inst(s, raw)
    part = None
    if inst is not None:
        part = s.state.part_by_id(inst.part_id)
    else:
        pid = _resolve_arrow_part(s, raw)
        part = s.state.part_by_id(pid) if pid else None
        if part is not None:
            inst = next(
                (i for i in _insts_of_part(s, part.id) if i.transform is not None), None
            )
    if inst is None or part is None or inst.transform is None:
        return None, part.id if part else None
    mn, mx = transform_bbox(part, inst.transform)
    pos = [
        (float(mn[0]) + float(mx[0])) / 2.0,
        (float(mn[1]) + float(mx[1])) / 2.0,
        float(mx[2]) + 45.0,
    ]
    return pos, part.id


def _assembly_top(s: GuideToolSession) -> list[float]:
    """Fallback anchor above the whole placed assembly."""
    from kiara.guide.solver import transform_bbox

    mn_o: list[float] | None = None
    mx_o: list[float] | None = None
    for inst in s.state.assembly.instances:
        part = s.state.part_by_id(inst.part_id)
        if part is None or inst.transform is None:
            continue
        mn, mx = transform_bbox(part, inst.transform)
        mn_o = list(mn) if mn_o is None else [min(a, b) for a, b in zip(mn_o, mn, strict=True)]
        mx_o = list(mx) if mx_o is None else [max(a, b) for a, b in zip(mx_o, mx, strict=True)]
    if mn_o is None or mx_o is None:
        return [0.0, 0.0, 45.0]
    return [(mn_o[0] + mx_o[0]) / 2.0, (mn_o[1] + mx_o[1]) / 2.0, mx_o[2] + 45.0]


async def add_text(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step_no = _step_no(a.get("step", 0))
    if s.state.step_by_number(step_no) is None:
        nums = [st.number for st in s.state.steps]
        return {"ok": False, "error": f"step {step_no} not found — valid steps: {nums}"}
    text = str(a.get("text", "")).strip()[:240]
    if not text:
        return {"ok": False, "error": "text must be non-empty"}
    pid: str | None = None
    anchor_pos: list[float] | None = None
    raw_part = str(a.get("part_id", "") or "").strip()
    if raw_part:
        anchor_pos, pid = _top_anchor(s, raw_part)
        if anchor_pos is None:
            return {
                "ok": False,
                "error": f"part '{raw_part}' has no placed instance — "
                "anchor a placed part or pass an explicit position",
            }
    pos = a.get("position")
    if pos is not None:
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            return {"ok": False, "error": "position must be [x,y,z] mm"}
        try:
            pos = [float(c) for c in pos]
        except (TypeError, ValueError):
            return {"ok": False, "error": "position must be numeric [x,y,z] mm"}
    if pos is None:
        pos = anchor_pos if anchor_pos is not None else _assembly_top(s)
    ann: dict[str, Any] = {
        "kind": "text",
        "step": step_no,
        "text": text,
        "position": pos,
    }
    if pid:
        ann["part_id"] = pid
    s.state.annotations.append(ann)
    return {"ok": True, "index": len(s.state.annotations) - 1, "position": pos}


async def list_arrows(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "annotations": s.state.annotations}


async def remove_arrow(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    idx = int(a.get("index", -1))
    if idx < 0 or idx >= len(s.state.annotations):
        return {"ok": False, "error": "index out of range"}
    s.state.annotations.pop(idx)
    return {"ok": True, "annotations": len(s.state.annotations)}


def _done_guard(s: GuideToolSession) -> dict[str, Any] | None:
    return None  # annotations are optional — empty set is valid


from kiara.guide.tools import DONE_GUARDS  # noqa: E402

DONE_GUARDS["annotate"] = _done_guard

HANDLERS = {
    "add_arrow": add_arrow,
    "add_text": add_text,
    "list_arrows": list_arrows,
    "remove_arrow": remove_arrow,
}
