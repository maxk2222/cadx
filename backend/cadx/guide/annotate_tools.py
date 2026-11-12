"""Annotate-mission tools — arrows/highlights layered on the finished
assembly (KIARA AnnotationAgent port). Stored on ``state.annotations`` and
folded into the unity guide at emit time.
"""

from __future__ import annotations

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


async def add_arrow(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step_no = int(a.get("step", 0))
    if s.state.step_by_number(step_no) is None:
        return {"ok": False, "error": f"step {step_no} not found"}
    pid = str(a.get("part_id", ""))
    if s.state.part_by_id(pid) is None:
        return {"ok": False, "error": f"part '{pid}' not found"}
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
    # arrows also enrich the step's joints so unity.py can place them
    s.state.assembly.joints.append(
        {
            "step": step_no,
            "moving_part": pid,
            "fixed_part": pid,
            "approach_axis": vec,
            "annotation": True,
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
