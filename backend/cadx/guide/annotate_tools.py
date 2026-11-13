"""Annotate-mission tools — arrows/highlights layered on the finished
assembly (KIARA AnnotationAgent port). Stored on ``state.annotations`` and
folded into the unity guide at emit time.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from cadx.guide.tools import _fn, _i, _s
from cadx.logging import get_logger

if TYPE_CHECKING:
    from cadx.guide.session import GuideToolSession

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

_DIR_VECS = {
    "down": [0.0, 0.0, -1.0],
    "up": [0.0, 0.0, 1.0],
    "left": [-1.0, 0.0, 0.0],
    "right": [1.0, 0.0, 0.0],
    "front": [0.0, -1.0, 0.0],
    "back": [0.0, 1.0, 0.0],
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
    from cadx.guide.assembly_tools import _resolve_inst, _resolve_name

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


from cadx.guide.tools import DONE_GUARDS  # noqa: E402

DONE_GUARDS["annotate"] = _done_guard

HANDLERS = {
    "add_arrow": add_arrow,
    "list_arrows": list_arrows,
    "remove_arrow": remove_arrow,
}
