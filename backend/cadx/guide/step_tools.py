"""Steps-mission tools — ordered instruction steps with the dual-text
contract (``text`` internal detail / ``display_text`` AR label).

Ported tool-for-tool from KIARA's StepAuthor surface (v4
tools/step_tools.py).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from cadx.guide.state import GuideStep, GuideSubStep
from cadx.guide.tools import _arr, _fn, _i, _s
from cadx.logging import get_logger

if TYPE_CHECKING:
    from cadx.guide.session import GuideToolSession

log = get_logger(__name__)

_ACTION_TYPES = ["assembly", "check", "safety", "prep"]
_SUB_ACTIONS = ["place", "attach", "orient", "insert", "fasten", "check"]

SCHEMAS: dict[str, dict[str, Any]] = {
    "create_step": _fn(
        "create_step",
        "Create one instruction step. REQUIRED dual text: `text` is the "
        "detailed internal description for downstream processing (parts, "
        "tools, spatial hints, tolerances — several sentences OK); "
        "`display_text` is the short AR-facing instruction shown to the "
        "user (≤140 chars, imperative, manual language).",
        {
            "number": _i("step number (manual order, 1-based)"),
            "title": _s("short step title"),
            "text": _s("detailed INTERNAL step description"),
            "display_text": _s("short AR-facing instruction (≤140 chars)"),
            "action_type": {
                "type": "string",
                "enum": _ACTION_TYPES,
                "description": "step kind, default assembly",
            },
            "required_parts": _arr(
                _s("canonical part name from detected_part_names or a new label"),
                "parts this step places/uses",
            ),
            "tools": _arr(_s("physical tool name"), "tools needed (e.g. wrench)"),
            "safety_hint": _s("safety warning shown with the step, '' if none"),
            "source_pages": _arr(_i("page"), "PDF pages this step comes from"),
        },
        ["number", "text", "display_text"],
    ),
    "create_steps": _fn(
        "create_steps",
        "Create MANY steps in ONE call — preferred over looping create_step: "
        "every manual step with its full dual text plus its sub_steps and "
        "optional media assignment in a single batch. Each item is "
        "validated independently; per-item errors come back indexed while "
        "valid items still apply. If emitting the whole guide at once is "
        "too large, send 2-3 steps per call — NEVER call with empty args.",
        {
            "steps": _arr(
                {
                    "type": "object",
                    "properties": {
                        "number": _i("step number (1-based)"),
                        "title": _s("short step title"),
                        "text": _s("detailed INTERNAL step description"),
                        "display_text": _s("short AR-facing instruction (≤140 chars)"),
                        "action_type": {"type": "string", "enum": _ACTION_TYPES},
                        "required_parts": _arr(_s("canonical part name"), "parts placed/used"),
                        "tools": _arr(_s("tool"), "tools needed"),
                        "safety_hint": _s("safety warning, '' if none"),
                        "source_pages": _arr(_i("page"), "PDF pages"),
                        "sub_steps": _arr(
                            {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": _SUB_ACTIONS},
                                    "part": _s("part name/id"),
                                    "target": _s("already-placed part/feature this part mates onto — include the manual's position qualifier ('linker unterer Stutzen', 'vormontierter Adapter unten'), '' only for the base part"),
                                    "target2": _s("second mate when ONE part bridges two already-placed parts in ONE action (a coupler joining section A AND section B) — replaces a second place substep; '' otherwise"),
                                    "text": _s("internal detail"),
                                    "side": _s("left/right/top/bottom/front/back, '' if n/a"),
                                },
                                "required": ["action", "part"],
                            },
                            "atomic actions — one place/attach/insert/fasten per NEW part occurrence; re-seat=orient, tighten/crimp/verify on already-placed=check",
                        ),
                        "media": {
                            "type": "object",
                            "properties": {
                                "page": _i("source page"),
                                "figure_index": _i("figure index from list_figures"),
                            },
                        },
                    },
                    "required": ["number", "text", "display_text"],
                },
                "one entry per manual step, in manual order",
            ),
        },
        ["steps"],
    ),
    "update_step": _fn(
        "update_step",
        "Update fields of an existing step (same fields as create_step; "
        "only provided fields change).",
        {
            "number": _i("step number to update"),
            "title": _s("new title"),
            "text": _s("new internal text"),
            "display_text": _s("new AR text (≤140 chars)"),
            "action_type": {"type": "string", "enum": _ACTION_TYPES},
            "required_parts": _arr(_s("part name"), "replace required parts"),
            "tools": _arr(_s("tool"), "replace tools"),
            "safety_hint": _s("new safety hint"),
            "source_pages": _arr(_i("page"), "replace source pages"),
        },
        ["number"],
    ),
    "add_sub_step": _fn(
        "add_sub_step",
        "Add one atomic sub-action to a step — MANDATORY for every part "
        "placement (one sub_step per part placed). The assembly agent "
        "executes sub_steps in order.",
        {
            "step": _i("step number"),
            "action": {"type": "string", "enum": _SUB_ACTIONS},
            "part": _s("part name/id this sub-step acts on"),
            "target": _s("already-placed part/feature it mates onto — include the manual's position qualifier where given; '' only for the base part"),
            "target2": _s("second mate when this ONE part bridges two already-placed parts in ONE action; '' otherwise"),
            "text": _s("internal detail for the placement solver"),
            "side": _s("spatial hint: left/right/top/bottom/front/back, '' if n/a"),
        },
        ["step", "action", "part"],
    ),
    "update_sub_step": _fn(
        "update_sub_step",
        "Update fields of an existing sub-step (index within the step's "
        "sub_steps list — use get_step to see indices). Only provided "
        "fields change. Use this to retag a wrong action (e.g. crimp on "
        "an already-placed part must be 'check', not 'place') or fix a "
        "wrong target/target2.",
        {
            "step": _i("step number"),
            "index": _i("sub-step index (0-based)"),
            "action": {"type": "string", "enum": _SUB_ACTIONS},
            "part": _s("part name/id"),
            "target": _s("already-placed part/feature it mates onto"),
            "target2": _s("second mate when this ONE part bridges two already-placed parts in ONE action"),
            "text": _s("internal detail"),
            "side": _s("spatial hint"),
        },
        ["step", "index"],
    ),
    "delete_sub_step": _fn(
        "delete_sub_step",
        "Remove a sub-step by index — for duplicate/phantom extractions "
        "(the same physical part emitted twice). Use get_step to see "
        "indices.",
        {
            "step": _i("step number"),
            "index": _i("sub-step index (0-based)"),
        },
        ["step", "index"],
    ),
    "assign_media": _fn(
        "assign_media",
        "Assign a PDF figure crop (from list_figures) as the step's media_image shown in AR.",
        {
            "step": _i("step number"),
            "page": _i("source page"),
            "figure_index": _i("figure index on that page (list_figures)"),
        },
        ["step", "page", "figure_index"],
    ),
    "reorder_steps": _fn(
        "reorder_steps",
        "Renumber all steps by the given order of current numbers.",
        {"order": _arr(_i("current step number"), "step numbers in new order")},
        ["order"],
    ),
    "finalize_steps": _fn(
        "finalize_steps",
        "Sort, renumber 1..N and validate the step list. REQUIRED before "
        "done_mission. Fails when: no steps, missing dual text, assembly "
        "steps without required_parts or sub_steps.",
        {},
        [],
    ),
}


def _validate_dual_text(text: str, display: str) -> str | None:
    if not text.strip():
        return "text (internal description) is required"
    if not display.strip():
        return "display_text (AR-facing instruction) is required"
    if len(display) > 240:
        return "display_text too long (≤240 chars hard cap; aim ≤140)"
    return None


def _step_fields(a: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("title", "text", "display_text", "action_type", "safety_hint"):
        if k in a:
            out[k] = str(a[k])
    for k in ("required_parts", "tools"):
        if k in a:
            v = a[k]
            # models sometimes send the list as a string literal
            # ("['Rohr DN22']") — iterating that string would store
            # single characters, so parse list-shaped strings first
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("["):
                    try:
                        import ast as _ast
                        import json as _json

                        try:
                            v = _json.loads(s)
                        except _json.JSONDecodeError:
                            v = _ast.literal_eval(s)
                    except (ValueError, SyntaxError):
                        v = [s]
                else:
                    v = [s] if s else []
            out[k] = [str(x) for x in (v or []) if str(x).strip()][:20]
    if "source_pages" in a:
        pages: set[int] = set()
        for x in a["source_pages"] or []:
            try:
                if int(x) > 0:
                    pages.add(int(x))
            except (TypeError, ValueError):
                continue
        out["source_pages"] = sorted(pages)[:20]
    return out


def _title_key(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


async def create_step(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    num = int(a.get("number", 0))
    if num < 1:
        return {"ok": False, "error": "number must be ≥ 1"}
    if s.state.step_by_number(num) is not None:
        return {"ok": False, "error": f"step {num} already exists — use update_step"}
    # runaway-guard: a step with the SAME title already exists — the manual's
    # sequence is being re-created instead of refined (observed degeneration:
    # duplicate batches re-numbering the same actions 14..37). Distinct
    # left/right variants keep distinct titles and pass.
    key = _title_key(str(a.get("title", "")))
    if key:
        for st in s.state.steps:
            if _title_key(st.title) == key:
                return {
                    "ok": False,
                    "error": f"step {st.number} already covers "
                    f"'{st.title}' — use update_step {st.number} or "
                    "finalize_steps instead of re-creating it",
                }
    if err := _validate_dual_text(str(a.get("text", "")), str(a.get("display_text", ""))):
        return {"ok": False, "error": err}
    fields = _step_fields(a)
    if fields.get("action_type") and fields["action_type"] not in _ACTION_TYPES:
        return {"ok": False, "error": f"action_type must be one of {_ACTION_TYPES}"}
    step = GuideStep(number=num, **fields)
    s.state.steps.append(step)
    s.state.steps.sort(key=lambda x: x.number)
    return {"ok": True, "step": num, "total_steps": len(s.state.steps)}


async def create_steps(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    items = a.get("steps") or []
    if not isinstance(items, list) or not items:
        return {"ok": False, "error": "steps must be a non-empty list"}
    created: list[int] = []
    errors: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append({"index": idx, "error": "item must be an object"})
            continue
        media = item.pop("media", None)
        subs = item.pop("sub_steps", None) or []
        if isinstance(subs, str):
            # string-encoded list from the model — iterating it would
            # explode one sub-step per character
            try:
                import ast as _ast
                import json as _json

                try:
                    subs = _json.loads(subs)
                except _json.JSONDecodeError:
                    subs = _ast.literal_eval(subs)
            except (ValueError, SyntaxError):
                subs = []
        num0 = int(item.get("number") or 0)
        if s.state.step_by_number(num0) is not None:
            # upsert: a reopened steps mission re-sends the whole plan —
            # rejecting every existing number leaves it half-created and
            # stacked (observed: 13→19 duplicate steps then order fights).
            # Update fields and REPLACE the sub-step list with the item's.
            res = await update_step(s, item)
            if not res.get("ok"):
                errors.append(
                    {"index": idx, "number": num0, "error": res.get("error")}
                )
                continue
            step = s.state.step_by_number(num0)
            step.sub_steps.clear()
            step_num = num0
        else:
            res = await create_step(s, item)
            if not res.get("ok"):
                errors.append(
                    {"index": idx, "number": item.get("number"), "error": res.get("error")}
                )
                continue
            step_num = int(res["step"])
        ok = True
        for j, sub in enumerate(subs):
            r = await add_sub_step(s, {**sub, "step": step_num})
            if not r.get("ok"):
                errors.append(
                    {"index": idx, "number": step_num, "sub_step": j, "error": r.get("error")}
                )
                ok = False
        if isinstance(media, dict):
            r = await assign_media(s, {**media, "step": step_num})
            if not r.get("ok"):
                errors.append({"index": idx, "number": step_num, "media": r.get("error")})
        if ok:
            created.append(step_num)
    return {
        "ok": not errors,
        "created": created,
        "total_steps": len(s.state.steps),
        "errors": errors[:20],
    }


async def update_step(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    num = int(a.get("number", 0))
    step = s.state.step_by_number(num)
    if step is None:
        return {"ok": False, "error": f"step {num} not found"}
    fields = _step_fields(a)
    new_text = fields.get("text", step.text)
    new_disp = fields.get("display_text", step.display_text)
    if err := _validate_dual_text(new_text, new_disp):
        return {"ok": False, "error": err}
    if fields.get("action_type") and fields["action_type"] not in _ACTION_TYPES:
        return {"ok": False, "error": f"action_type must be one of {_ACTION_TYPES}"}
    for k, v in fields.items():
        setattr(step, k, v)
    return {"ok": True, "step": num}


async def add_sub_step(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step = s.state.step_by_number(int(a.get("step", 0)))
    if step is None:
        return {"ok": False, "error": f"step {a.get('step')} not found"}
    action = str(a.get("action", ""))
    if action not in _SUB_ACTIONS:
        return {"ok": False, "error": f"action must be one of {_SUB_ACTIONS}"}
    part = str(a.get("part", "")).strip()
    if not part:
        return {"ok": False, "error": "part is required"}
    step.sub_steps.append(
        GuideSubStep(
            action=action,
            part=part,
            target=str(a.get("target", "")),
            target2=str(a.get("target2", "")),
            text=str(a.get("text", "")),
            side=str(a.get("side", "")),
        )
    )
    return {"ok": True, "step": step.number, "sub_steps": len(step.sub_steps)}


async def update_sub_step(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step = s.state.step_by_number(int(a.get("step", 0)))
    if step is None:
        return {"ok": False, "error": f"step {a.get('step')} not found"}
    idx = int(a.get("index", -1))
    if not (0 <= idx < len(step.sub_steps)):
        return {
            "ok": False,
            "error": f"sub-step index {idx} out of range — step {step.number} "
            f"has {len(step.sub_steps)} sub_steps",
        }
    ss = step.sub_steps[idx]
    if "action" in a:
        action = str(a["action"])
        if action not in _SUB_ACTIONS:
            return {"ok": False, "error": f"action must be one of {_SUB_ACTIONS}"}
        ss.action = action
    for k in ("part", "target", "target2", "text", "side"):
        if k in a:
            setattr(ss, k, str(a[k]))
    return {"ok": True, "step": step.number, "index": idx}


async def delete_sub_step(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step = s.state.step_by_number(int(a.get("step", 0)))
    if step is None:
        return {"ok": False, "error": f"step {a.get('step')} not found"}
    idx = int(a.get("index", -1))
    if not (0 <= idx < len(step.sub_steps)):
        return {
            "ok": False,
            "error": f"sub-step index {idx} out of range — step {step.number} "
            f"has {len(step.sub_steps)} sub_steps",
        }
    removed = step.sub_steps.pop(idx)
    return {
        "ok": True,
        "step": step.number,
        "removed": {"action": removed.action, "part": removed.part},
    }


async def assign_media(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    step = s.state.step_by_number(int(a.get("step", 0)))
    if step is None:
        return {"ok": False, "error": f"step {a.get('step')} not found"}
    page_no = int(a.get("page", 0))
    idx = int(a.get("figure_index", 0))
    p = s._page(page_no)
    if p is None:
        return {"ok": False, "error": f"page {page_no} not found"}
    for f in p.figures:
        if int(f.get("index", 0)) == idx and f.get("image_path"):
            step.media_image = str(f["image_path"])
            if page_no not in step.source_pages:
                step.source_pages = sorted([*step.source_pages, page_no])
            return {"ok": True, "step": step.number, "media_image": step.media_image}
    avail = [f.get("index") for f in p.figures if f.get("image_path")]
    return {
        "ok": False,
        "error": f"figure {idx} not found on page {page_no}",
        "available_figure_indices": avail,
    }


async def reorder_steps(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    order = [int(x) for x in (a.get("order") or [])]
    cur = sorted(s.state.steps, key=lambda x: x.number)
    if sorted(order) != [x.number for x in cur]:
        return {
            "ok": False,
            "error": "order must contain every existing step number exactly once",
            "current": [x.number for x in cur],
        }
    rank = {old: i + 1 for i, old in enumerate(order)}
    for step in s.state.steps:
        step.number = rank[step.number]
    s.state.steps.sort(key=lambda x: x.number)
    return {"ok": True, "order": order}


async def finalize_steps(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    if not s.state.steps:
        return {"ok": False, "error": "no steps created"}
    s.state.steps.sort(key=lambda x: x.number)
    errors: list[str] = []
    for i, step in enumerate(s.state.steps, start=1):
        step.number = i
        if err := _validate_dual_text(step.text, step.display_text):
            errors.append(f"step {i}: {err}")
        if not step.source_pages:
            errors.append(f"step {i}: source_pages missing")
        if step.action_type == "assembly":
            if not step.required_parts:
                errors.append(f"step {i}: assembly step without required_parts")
            if not step.sub_steps:
                errors.append(f"step {i}: assembly step without sub_steps")
        step.parts_delta = list(step.required_parts)
    if errors:
        return {"ok": False, "error": "validation failed", "errors": errors[:20]}
    return {"ok": True, "steps": len(s.state.steps)}


def _done_guard(s: GuideToolSession) -> dict[str, Any] | None:
    # must match finalize_steps' validation — otherwise done_mission closes
    # the mission on a step list finalize would have rejected
    res_errors: list[str] = []
    if not s.state.steps:
        return {"ok": False, "error": "no steps — create steps and call finalize_steps first"}
    for st in s.state.steps:
        if err := _validate_dual_text(st.text, st.display_text):
            res_errors.append(f"step {st.number}: {err}")
        if not st.source_pages:
            res_errors.append(f"step {st.number}: source_pages missing")
        if st.action_type == "assembly" and not st.sub_steps:
            res_errors.append(f"step {st.number}: assembly step without sub_steps")
    if res_errors:
        return {
            "ok": False,
            "error": "; ".join(res_errors[:6]) + " — fix then finalize_steps",
        }
    return None


from cadx.guide.tools import DONE_GUARDS  # noqa: E402

DONE_GUARDS["steps"] = _done_guard

HANDLERS = {
    "create_step": create_step,
    "create_steps": create_steps,
    "update_step": update_step,
    "add_sub_step": add_sub_step,
    "update_sub_step": update_sub_step,
    "delete_sub_step": delete_sub_step,
    "assign_media": assign_media,
    "reorder_steps": reorder_steps,
    "finalize_steps": finalize_steps,
}
