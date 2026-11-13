"""GuideToolSession — mission-scoped tool surface for the guide pipeline.

Subclasses ``ToolSession`` with the ``kind="info"`` placeholder trick
(``ProjectToolSession`` precedent: ``self.kind`` is reassigned after
``super().__init__`` because the Literal is closed). The session owns:

- state: ``GuideState`` loaded from/saved to ``state_dir/state.json``
- shared lifecycle tools (``done_mission``, ``ask_user``, plan scratchpad)
- shared read tools (pages/steps/parts/assets listing)

Mission domain tools live in the guide/*_tools.py modules and are
dispatched through ``tools.merged_handlers()``.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import AgentCtx
from cadx.guide import tools as gtools
from cadx.guide.state import GuideState
from cadx.logging import get_logger

log = get_logger(__name__)

PLAN_MAX_BYTES = 200 * 1024

# image-lookup failure budget per session before the tools short-circuit —
# enough rope for one typo, not enough for a retry spiral
_IMAGE_TOOLS = {"get_figure_image", "get_page_image"}
_IMAGE_FAIL_CAP = 3

# shared session-owned tool schemas (dispatched via _tool_* methods below)
SESSION_SCHEMAS: dict[str, dict[str, Any]] = {
    "done_mission": gtools._fn(
        "done_mission",
        "Finish this mission and persist its results. Call LAST, only after "
        "all required work for this mission is complete and validated.",
        {"reasoning": gtools._s("short summary of what the mission accomplished")},
        ["reasoning"],
    ),
    "get_page_image": gtools._fn(
        "get_page_image",
        "Attach the rendered PNG of one PDF page to the next message so you "
        "can read figures/labels visually.",
        {"page": gtools._i("1-based page number")},
        ["page"],
    ),
    "get_page_text": gtools._fn(
        "get_page_text",
        "Get the extracted text of one PDF page (≤2000 chars).",
        {"page": gtools._i("1-based page number")},
        ["page"],
    ),
    "get_all_page_text": gtools._fn(
        "get_all_page_text",
        "Get a compact dump of extracted text for all pages (truncated per "
        "page) — use to find which page describes which step.",
        {},
        [],
    ),
    "list_figures": gtools._fn(
        "list_figures",
        "List cropped figure images detected in the PDF: page, path, bbox.",
        {"page": gtools._i("optional page filter")},
        [],
    ),
    "get_figure_image": gtools._fn(
        "get_figure_image",
        "Attach one cropped figure image (from list_figures) to the next "
        "message for visual inspection.",
        {"page": gtools._i("page number"), "index": gtools._i("figure index on that page")},
        ["page", "index"],
    ),
    "list_steps": gtools._fn(
        "list_steps",
        "List current guide steps: number, title, action_type, required_parts, "
        "source_pages. full=true adds text/display_text/sub_steps/media per step.",
        {"full": gtools._b("include full step text + sub_steps + media")},
        [],
    ),
    "get_step": gtools._fn(
        "get_step",
        "Full detail of one step: internal text, display_text, sub_steps, media, pages.",
        {"number": gtools._i("step number")},
        ["number"],
    ),
    "list_parts": gtools._fn(
        "list_parts",
        "List registered CAD parts: id, name, source, port count, bbox, placed status.",
        {},
        [],
    ),
    "get_part_features": gtools._fn(
        "get_part_features",
        "Full port/bbox detail of one registered part (from list_parts).",
        {"part_id": gtools._s("part id")},
        ["part_id"],
    ),
    "list_assets": gtools._fn(
        "list_assets",
        "List the files attached to this guide run: asset_id, filename, kind.",
        {},
        [],
    ),
    "add_master_note": gtools._fn(
        "add_master_note",
        "Write a masternode — a durable tip/assumption/extra hint for the "
        "person who will follow this guide (e.g. a documented placement "
        "assumption, a torque/crimping note, a warning). Masternotes are "
        "emitted as a project asset at job end; they are NOT step text.",
        {"text": gtools._s("the note text in the guide's language")},
        ["text"],
    ),
    "list_user_notes": gtools._fn(
        "list_user_notes",
        "Read supplemental instructions/answers the user provided via chat.",
        {},
        [],
    ),
    "ask_user": gtools._fn(
        "ask_user",
        "Ask the user a blocking question — the guide job parks and resumes "
        "when they answer. Batch related questions in ONE call. Use only "
        "when the manual genuinely lacks required info and you cannot "
        "proceed with a documented assumption.",
        {
            "question": gtools._s("the question in the manual's language"),
            "context": gtools._s("what you need it for"),
        },
        ["question"],
    ),
    "read_plan": gtools._fn(
        "read_plan",
        "Read the shared plan/review scratchpad (plan.md) — findings from "
        "previous missions and the review agent.",
        {},
        [],
    ),
    "append_plan": gtools._fn(
        "append_plan",
        "Append a note to the shared plan scratchpad for later missions.",
        {"text": gtools._s("note to append")},
        ["text"],
    ),
    "write_plan": gtools._fn(
        "write_plan",
        "Overwrite the shared plan scratchpad (use for structured rewrites).",
        {"text": gtools._s("new scratchpad content")},
        ["text"],
    ),
    "clear_plan": gtools._fn(
        "clear_plan",
        "Clear the shared plan scratchpad.",
        {},
        [],
    ),
}


class GuideToolSession(ToolSession):
    """One mission's tool session. ``run_session`` drives it; the runner
    constructs it with the mission-scoped allow list."""

    def __init__(
        self,
        ctx: AgentCtx,
        *,
        mission: str,
        state: GuideState,
        state_dir: Path,
        project_id: uuid.UUID,
        guide_assets: list[dict[str, Any]],
        executor: Any,
        attach_images: bool = True,
        emit: Callable[..., Awaitable[None]] | None = None,
        park_fn: Callable[[list[dict[str, Any]]], Awaitable[None]] | None = None,
        check_cancel: Callable[[], Awaitable[bool]] | None = None,
        enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
    ) -> None:
        super().__init__(ctx, kind="info")
        self.kind = "guide"
        self.mission = mission
        self.state = state
        self.state_dir = Path(state_dir)
        self.project_id = project_id
        self.guide_assets = guide_assets  # [{id, filename, kind, path}]
        self.executor = executor
        self.attach_images = attach_images
        self._emit_fn = emit
        self._park_fn = park_fn
        self._cancel_fn = check_cancel
        self._enqueue_fn = enqueue_fn
        self._preview_cache: dict[str, dict[str, Any]] = {}
        self._aliases_store = self._load_aliases()
        self._bad_image_calls = 0
        self.parked = False
        self.parked_questions: list[dict[str, Any]] = []
        # image budget for _images results (page images, assembly renders)
        self.image_left = int(getattr(ctx.runtime_settings, "guide_image_budget", 24))

    # -- session interface --------------------------------------------------- #

    def tools(self) -> list[dict[str, Any]]:
        names = gtools.MISSION_TOOLS.get(self.mission, [])
        schemas = gtools.merged_schemas()
        out = []
        for n in names:
            spec = SESSION_SCHEMAS.get(n) or schemas.get(n)
            if spec is not None:
                out.append(spec)
        return out

    async def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        # image-lookup circuit breaker: repeated invalid figure/page indices
        # burned whole mission budgets — after _IMAGE_FAIL_CAP failures the
        # tools short-circuit and redirect the agent to renders/audits
        if name in _IMAGE_TOOLS and self._bad_image_calls >= _IMAGE_FAIL_CAP:
            # _streak_exempt: a guardrail message, not a model error — it
            # must not feed error_streak or the breaker itself kills the
            # mission it exists to protect
            return {
                "ok": False,
                "_streak_exempt": True,
                "error": "PDF image lookups disabled for this session "
                "after repeated invalid indices — verify via "
                "render_assembly and the numeric audits instead",
            }
        handler = getattr(self, f"_tool_{name}", None)
        if handler is not None:
            res = handler(args)
            res = await res if inspect.iscoroutinefunction(handler) else res
        else:
            module_handler = gtools.merged_handlers().get(name)
            if module_handler is None:
                return {"ok": False, "error": f"unknown tool '{name}'"}
            res = await module_handler(self, args)
            # module handlers mutate self.state — persist unless opted out
            if isinstance(res, dict) and not res.get("_no_save"):
                self.state.save(self.state_dir)
        if name in _IMAGE_TOOLS and isinstance(res, dict) and not res.get("ok"):
            self._bad_image_calls += 1
        return res

    async def _emit(self, progress: float, label: str, **fields: Any) -> None:
        if self._emit_fn is not None:
            with contextlib.suppress(Exception):
                await self._emit_fn(
                    f"guide_{self.mission}",
                    0,
                    label,
                    sub_progress=progress,
                    sub_label=label,
                    **fields,
                )

    def _save(self) -> None:
        self.state.save(self.state_dir)
        self._save_aliases()

    def _load_aliases(self) -> dict[str, str]:
        p = self.state_dir / "aliases.json"
        if not p.exists():
            return {}
        try:
            from cadx.guide.assembly_tools import _norm2

            # keys may predate the umlaut-folding normalizer — re-fold on
            # load so old aliases still resolve ("wärmepumpe" → "warmepumpe")
            return {
                _norm2(str(k)): str(v)
                for k, v in json.loads(p.read_text()).items()
            }
        except Exception:
            return {}

    def _save_aliases(self) -> None:
        with contextlib.suppress(Exception):
            (self.state_dir / "aliases.json").write_text(
                json.dumps(self._aliases_store), encoding="utf-8"
            )

    def _doc(self) -> Any:
        return self.state.document

    def _page(self, n: int) -> Any:
        doc = self._doc()
        if doc is None or n < 1 or n > len(doc.pages):
            return None
        return doc.pages[n - 1]

    def _page_err(self) -> dict[str, Any]:
        doc = self._doc()
        n = len(doc.pages) if doc else 0
        return {"ok": False, "error": f"page not found — document has pages 1..{n}"}

    def _images_result(self, paths: list[str], **extra: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": True, **extra}
        if self.attach_images and self.image_left > 0 and paths:
            out["_images"] = paths[: self.image_left]
            self.image_left -= len(out["_images"])
        elif paths:
            out["images_suppressed"] = "image budget/role unavailable"
        return out

    # -- lifecycle tools ------------------------------------------------------ #

    def _tool_done_mission(self, a: dict[str, Any]) -> dict[str, Any]:
        # ensure tool modules (and their DONE_GUARDS registrations) are
        # loaded — otherwise done_mission as the FIRST call skips the guard
        gtools.merged_handlers()
        guard = gtools.DONE_GUARDS.get(self.mission)
        if guard is not None and (err := guard(self)):
            return err
        self.state.mark_done(self.mission)
        self._save()
        self.submitted = {"mission": self.mission, "reasoning": str(a.get("reasoning", ""))}
        return {"ok": True, "mission_done": self.mission}

    async def _tool_ask_user(self, a: dict[str, Any]) -> dict[str, Any]:
        if self._park_fn is None:
            return {"ok": False, "error": "asking the user is not available here"}
        # OpenQuestion contract (llm/schemas.py): the frontend renders
        # q.question — "text"/"context" keys would show as empty bullets
        question = {
            "question": str(a.get("question", "")).strip() or "Question from guide builder",
            "why_it_matters": str(a.get("context", "")).strip(),
            "blocking": True,
        }
        self.parked = True
        self.parked_questions = [question]
        await self._park_fn([question])
        # end the tool loop like done_mission does — the job parks here
        self.submitted = {"parked": True, "questions": [question]}
        return {
            "ok": True,
            "parked": True,
            "note": "question sent to the user — the job parks until answered",
        }

    # -- page / document tools -------------------------------------------------- #

    def _tool_get_page_image(self, a: dict[str, Any]) -> dict[str, Any]:
        p = self._page(int(a.get("page", 0)))
        if p is None or not p.image_path:
            return {"ok": False, "error": "page not found or not rendered"}
        return self._images_result([p.image_path], page=p.page)

    def _tool_get_page_text(self, a: dict[str, Any]) -> dict[str, Any]:
        p = self._page(int(a.get("page", 0)))
        if p is None:
            return self._page_err()
        text = p.text or ""
        note = ""
        if len(text) > 2000:
            text, note = text[:2000], f" (truncated {len(text) - 2000} chars)"
        return {
            "ok": True,
            "page": p.page,
            "text": text,
            "note": note,
            "classification": p.classification,
        }

    def _tool_get_all_page_text(self, a: dict[str, Any]) -> dict[str, Any]:
        doc = self._doc()
        if doc is None:
            return {"ok": False, "error": "document not prepared"}
        pages = []
        for p in doc.pages:
            t = (p.text or "")[:800]
            pages.append({"page": p.page, "classification": p.classification, "text": t})
        return {"ok": True, "page_count": doc.page_count, "pages": pages}

    def _tool_list_figures(self, a: dict[str, Any]) -> dict[str, Any]:
        doc = self._doc()
        if doc is None:
            return {"ok": False, "error": "document not prepared"}
        page_filter = int(a.get("page", 0) or 0)
        figs = []
        for p in doc.pages:
            if page_filter and p.page != page_filter:
                continue
            for f in p.figures:
                figs.append({"page": p.page, **{k: v for k, v in f.items() if k != "data"}})
        return {"ok": True, "figures": figs[:60], "total": len(figs)}

    def _tool_get_figure_image(self, a: dict[str, Any]) -> dict[str, Any]:
        p = self._page(int(a.get("page", 0)))
        if p is None:
            return self._page_err()
        idx = int(a.get("index", 0))
        for f in p.figures:
            if int(f.get("index", 0)) == idx and f.get("image_path"):
                return self._images_result([str(f["image_path"])], page=p.page, index=idx)
        return {
            "ok": False,
            "error": (
                f"figure not found — page {p.page} has figure indices "
                f"{[int(f.get('index', 0)) for f in p.figures if f.get('image_path')]} "
                "(call list_figures to inspect them)"
            ),
        }

    # -- state read tools -------------------------------------------------------- #

    def _tool_list_steps(self, a: dict[str, Any]) -> dict[str, Any]:
        full = bool(a.get("full"))
        rows = []
        for s in sorted(self.state.steps, key=lambda s: s.number):
            row: dict[str, Any] = {
                "number": s.number,
                "title": s.title,
                "action_type": s.action_type,
                "required_parts": s.required_parts,
                "tools": s.tools,
                "source_pages": s.source_pages,
                "sub_steps": len(s.sub_steps),
            }
            if full:
                row.update(
                    {
                        "text": s.text,
                        "display_text": s.display_text,
                        "safety_hint": s.safety_hint,
                        "media_image": s.media_image,
                        "sub_steps_detail": [x.model_dump() for x in s.sub_steps],
                    }
                )
            rows.append(row)
        return {"ok": True, "steps": rows}

    def _tool_get_step(self, a: dict[str, Any]) -> dict[str, Any]:
        s = self.state.step_by_number(int(a.get("number", 0)))
        if s is None:
            nums = sorted(st.number for st in self.state.steps)
            return {"ok": False, "error": f"step {a.get('number')} not found — valid steps: {nums}"}
        return {"ok": True, "step": s.model_dump()}

    def _tool_list_parts(self, a: dict[str, Any]) -> dict[str, Any]:
        insts_by_part: dict[str, list[str]] = {}
        for i in self.state.assembly.instances:
            insts_by_part.setdefault(i.part_id, []).append(i.instance_id or i.part_id)
        return {
            "ok": True,
            "parts": [
                {
                    "id": p.id,
                    "name": p.name,
                    "source": p.source,
                    "ports": len(p.ports),
                    "quantity": p.quantity,
                    "bbox": [p.bbox_min, p.bbox_max],
                    "placed": p.id in insts_by_part,
                    "instances": insts_by_part.get(p.id, []),
                }
                for p in self.state.parts
            ],
        }

    def _tool_get_part_features(self, a: dict[str, Any]) -> dict[str, Any]:
        from cadx.guide.assembly_tools import _part_ref

        p = _part_ref(self, a.get("part_id", ""))
        if p is None:
            ids = [x.id for x in self.state.parts][:15]
            return {"ok": False, "error": f"part '{a.get('part_id')}' not found — valid part_ids: {ids}"}
        return {"ok": True, "part": p.model_dump()}

    def _tool_list_assets(self, a: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "assets": self.guide_assets}

    def _tool_add_master_note(self, a: dict[str, Any]) -> dict[str, Any]:
        text = str(a.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "text required"}
        if text not in self.state.master_notes:
            self.state.master_notes.append(text[:2000])
        self._save()
        return {"ok": True, "count": len(self.state.master_notes)}

    def _tool_list_user_notes(self, a: dict[str, Any]) -> dict[str, Any]:
        # drain answers that arrived via resume into the notes list once
        if self.state.pending_answers:
            self.state.user_notes.extend(self.state.pending_answers)
            self.state.pending_answers.clear()
        # live notes from chat (attach_guide_note writes user_notes.jsonl)
        notes_file = self.state_dir / "user_notes.jsonl"
        if notes_file.exists():
            try:
                for line in notes_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    n = str(json.loads(line).get("note", "")).strip()
                    if n and n not in self.state.user_notes:
                        self.state.user_notes.append(n)
                notes_file.unlink()
            except Exception:
                pass
        self._save()
        return {
            "ok": True,
            "user_notes": self.state.user_notes,
            "pending_answers": [],
        }

    # -- plan scratchpad tools ---------------------------------------------------- #

    def _plan_path(self) -> Path:
        return self.state_dir / "plan.md"

    def _tool_read_plan(self, a: dict[str, Any]) -> dict[str, Any]:
        p = self._plan_path()
        if not p.exists():
            return {"ok": True, "text": "", "note": "plan scratchpad is empty"}
        return {"ok": True, "text": p.read_text(encoding="utf-8")[:8000]}

    def _tool_append_plan(self, a: dict[str, Any]) -> dict[str, Any]:
        p = self._plan_path()
        cur = p.read_text(encoding="utf-8") if p.exists() else ""
        text = str(a.get("text", ""))
        new = (cur + ("\n" if cur and not cur.endswith("\n") else "") + text)[-PLAN_MAX_BYTES:]
        p.write_text(new, encoding="utf-8")
        return {"ok": True}

    def _tool_write_plan(self, a: dict[str, Any]) -> dict[str, Any]:
        self._plan_path().write_text(str(a.get("text", ""))[:PLAN_MAX_BYTES], encoding="utf-8")
        return {"ok": True}

    def _tool_clear_plan(self, a: dict[str, Any]) -> dict[str, Any]:
        p = self._plan_path()
        if p.exists():
            p.unlink()
        return {"ok": True}
