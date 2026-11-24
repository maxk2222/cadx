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
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from kiara.agent.toolspecs import ToolSession
from kiara.agent.types import AgentCtx
from kiara.guide import tools as gtools
from kiara.guide.state import GuideState
from kiara.logging import get_logger

log = get_logger(__name__)

PLAN_MAX_BYTES = 200 * 1024

# image-lookup failure budget per session before the tools short-circuit —
# enough rope for one typo, not enough for a retry spiral
_IMAGE_TOOLS = {"get_figure_image", "get_page_image"}
_IMAGE_FAIL_CAP = 3
# placement/solve tools whose rejections the agent must not retry verbatim —
# an identical call with identical args returns the identical result, so a
# 3rd attempt is pure budget burn. The breaker returns a streak-exempt
# guardrail (no solver run) telling the agent what to change instead.
_REPEAT_REJECT_TOOLS = {
    "place_instance",
    "solve_attachment_preview",
    "solve_env_preview",
    "solve_free_preview",
}
_REPEAT_REJECT_CAP = 3
# attachment-preview dead-pair cap: all-rejected candidates on the same
# target_port across ANY moving ports — see _dead_pair_counts below
_DEAD_PAIR_CAP = 3
# error-class loop breaker: the same tool failing the SAME WAY N times in
# one mission is a stuck model, not bad luck — 'create_step exists' and
# 'update_step not found' loops burned whole mission budgets because the
# failures varied in args but not in class. Once a class trips the tool
# short-circuits for the rest of the mission. Meta tools stay exempt:
# done_mission/ask_user failures are legitimate gates the model must
# retry after fixing.
_ERR_CLASS_CAP = 3
_ERR_CLASS_EXEMPT = {"done_mission", "ask_user"}
# entity-keyed write-streak breaker: SUCCESSFUL rewrites of the SAME
# entity are invisible to every other breaker — a repair mission rewrote
# update_sub_step(12, index 2) 47 times while context condensation kept
# erasing the model's memory of having already fixed it (repeat_streak
# needs identical args/prose, churn only covers edit/codegen kinds, the
# error-class breaker only sees failures). Writes past the warn cap
# still execute — the model's latest intent is stored honestly — but
# carry a warning; past the block cap the call short-circuits with a
# streak-exempt guardrail showing the stored content; past 2x the block
# cap the honest error returns so a truly stuck mission aborts.
_WRITE_ENTITY_TOOLS = {
    "update_step": lambda a: str(a.get("number", "")),
    "update_sub_step": lambda a: f"{a.get('step', '')}:{a.get('index', '')}",
}
# writes 4+ carry an in-result warning; once _WRITE_BLOCK_AT writes are
# stored the next call is blocked (streak-exempt), and past 2x that the
# honest error returns so a truly stuck mission aborts
_WRITE_WARN_AT = 4
_WRITE_BLOCK_AT = 6
# page-image streak hint: sequential get_page_image reads burn the doc
# mission's small call budget — a model that paged 22 images one by one
# died on budget without ever calling classify_pages. The counter only
# hints (image reads are legitimate work), it never blocks.
_PAGE_IMAGE_HINT_AT = 6


def _err_class(res: dict[str, Any]) -> str | None:
    """Error-class key for the loop breaker: the failure text with digits
    folded to 'N' (step 3 vs step 12 is the same class) plus a digest of
    the structured errors list — finalize_steps legitimately fails while
    the model works through DIFFERENT items, which is progress, not a loop.
    Stagnation only trips when the same class repeats unchanged."""
    if res.get("_streak_exempt"):
        return None
    err = res.get("error")
    if not err:
        return None
    cls = re.sub(r"\d+", "N", str(err))[:120]
    extra = res.get("errors")
    if extra:
        cls += "|" + json.dumps(
            [re.sub(r"\d+", "N", str(e))[:80] for e in extra],
            default=str,
        )
    return cls


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
        {"page": gtools._i("page number"), "index": gtools._i("1-based figure index on that page")},
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
        # preview candidates live on the durable state (see GuideState.
        # preview_cache) — a fresh session object must still resolve the
        # candidate_ids a previous attempt handed out
        self._preview_cache = self.state.preview_cache
        self._aliases_store = self._load_aliases()
        # how many premounted copies the planner spawned per part —
        # bounds place_instance's premount allowance so "Adapter 2 + 2
        # vormontiert" can place 4 total but never 5
        self._premount_quota: dict[str, int] = {}
        self._bad_image_calls = 0
        self._reject_counts: dict[str, int] = {}
        # dead-pair breaker: attachment previews return ok=true even when
        # EVERY candidate is pre-rejected — an args-keyed reject counter
        # never sees the dead end and a stuck model re-probes the same
        # target_port with different moving ports (73x in one run). Key
        # on (target instance, target_port): the port itself is the sink.
        self._dead_pair_counts: dict[tuple[str, str], int] = {}
        # (tool, error-class) -> repeat count for the loop breaker
        self._err_class_counts: dict[tuple[str, str], int] = {}
        # (tool, entity-key) -> successful write count for the write-streak
        # breaker; successful page-image reads for the triage hint
        self._write_counts: dict[tuple[str, str], int] = {}
        self._page_image_calls = 0
        # PDF images already attached this session: the transport dedupes
        # by bytes so a refetch returns literally nothing — a model that
        # re-polls pages 2,3,4,5 -> 2,3,4,5 instead of writing steps must
        # get an honest refusal, not an empty ok (observed: steps mission
        # burned its whole budget in a get_page_image cycle)
        self._delivered_images: set[tuple[Any, ...]] = set()
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
        # identical-arg retry breaker: a rejected call repeated verbatim is
        # deterministic budget burn — short-circuit at the cap with a
        # streak-exempt guardrail naming what must change
        sig: str | None = None
        if name in _REPEAT_REJECT_TOOLS:
            sig = json.dumps({"t": name, "a": args}, sort_keys=True, default=str)
            if self._reject_counts.get(sig, 0) >= _REPEAT_REJECT_CAP:
                return {
                    "ok": False,
                    "_streak_exempt": True,
                    "error": f"'{name}' with these exact arguments was already "
                    f"rejected {_REPEAT_REJECT_CAP}x — change the moving "
                    "part's port, the target instance, or skip this substep "
                    "with a documented plan note (list_parts/list_steps show "
                    "the live state). Retrying verbatim cannot succeed.",
                }
        # dead-pair breaker: every candidate on this target_port was already
        # pre-rejected _DEAD_PAIR_CAP times — the port is the sink, varying
        # the moving port cannot revive it
        dead_gate = self._dead_pair_gate(name, args)
        if dead_gate is not None:
            return dead_gate
        # entity-keyed write-streak breaker: successful rewrites of the
        # same entity count up — warns inside the result, then blocks
        # with a streak-exempt guardrail, then re-escalates to an honest
        # error so a model that ignores the guardrail aborts the mission
        wkey_fn = _WRITE_ENTITY_TOOLS.get(name)
        wkey = (name, wkey_fn(args)) if wkey_fn else None
        wn = self._write_counts.get(wkey, 0) if wkey else 0
        gate = self._write_streak_gate(name, args, wn) if wkey else None
        if gate is not None:
            return gate
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
        if sig is not None and isinstance(res, dict) and not res.get("ok"):
            self._reject_counts[sig] = self._reject_counts.get(sig, 0) + 1
        self._dead_pair_count(name, args, res)
        if wkey and isinstance(res, dict) and res.get("ok"):
            self._write_counts[wkey] = wn + 1
            if wn + 1 >= _WRITE_WARN_AT:
                res["warning"] = (
                    f"this entity was already updated {wn + 1}x in this "
                    "mission — verify it is now correct and move on; "
                    "rewriting without new information burns the call "
                    "budget"
                )
        if name == "get_page_image" and isinstance(res, dict) and res.get("ok"):
            self._page_image_calls += 1
            if self._page_image_calls >= _PAGE_IMAGE_HINT_AT:
                # name a tool that actually exists in THIS mission —
                # classify_pages is doc-only, the hint misfired when a
                # steps-mission looped page fetches
                next_tool = {"doc": "classify_pages", "steps": "create_steps"}.get(
                    self.mission, "the mission's write tools"
                )
                res["hint"] = (
                    f"{self._page_image_calls} page images fetched — "
                    f"proceed with {next_tool} on the pages already in "
                    "context (re-fetching a delivered page is refused); "
                    "the mission call budget is finite"
                )
        return self._err_class_gate(name, res)

    def _dead_pair_gate(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Attachment previews return ok=true even when every candidate is
        pre-rejected, so the identical-arg reject counter never sees the
        dead end and a stuck model re-probes the same target_port with
        different moving ports (73x in one run). The port is the sink —
        key on (target instance, target_port)."""
        if name != "solve_attachment_preview":
            return None
        key = (str(args.get("target_part_id", "")), str(args.get("target_port", "")))
        if self._dead_pair_counts.get(key, 0) < _DEAD_PAIR_CAP:
            return None
        return {
            "ok": False,
            "_streak_exempt": True,
            "error": f"target port '{key[1]}' on '{key[0]}' rejected every "
            f"candidate {_DEAD_PAIR_CAP}x across different moving ports — "
            "the port pair cannot mate. Pick a different target_port from "
            "alternative_pairs / find_matching_ports (other spigots/sockets "
            "on the same part score equally), or place via "
            "solve_env_preview only if the manual shows no port mate",
        }

    def _dead_pair_count(self, name: str, args: dict[str, Any], res: Any) -> None:
        """Count all-rejected attachment previews on (target instance,
        target_port) for _dead_pair_gate."""
        if name != "solve_attachment_preview":
            return
        if not isinstance(res, dict) or not res.get("ok"):
            return
        if any(c.get("geometry_ok") for c in res.get("candidates") or [{}]):
            return
        key = (str(args.get("target_part_id", "")), str(args.get("target_port", "")))
        self._dead_pair_counts[key] = self._dead_pair_counts.get(key, 0) + 1

    def _err_class_gate(self, name: str, res: Any) -> Any:
        """Error-class loop breaker: the same tool failing the SAME WAY N
        times in one mission is a stuck model, not bad luck. Past the cap
        the call short-circuits with a streak-exempt guardrail; past 2x
        the cap the honest error returns so the mission can abort."""
        if (
            name in _ERR_CLASS_EXEMPT
            or not isinstance(res, dict)
            or res.get("ok")
            or res.get("_streak_exempt")
        ):
            return res
        cls = _err_class(res)
        if cls is None:
            return res
        key = (name, cls)
        self._err_class_counts[key] = self._err_class_counts.get(key, 0) + 1
        n_fail = self._err_class_counts[key]
        if n_fail > _ERR_CLASS_CAP:
            log.info(
                "guide.tool_circuit",
                mission=self.mission,
                tool=name,
                err_class=cls[:120],
                count=n_fail,
            )
        if n_fail > _ERR_CLASS_CAP * 2:
            # stubborn loop: the streak-exempt phase did not break it —
            # let the honest error feed error_streak again so the mission
            # aborts instead of burning its whole budget
            return {
                "ok": False,
                "error": f"'{name}' keeps failing with the same "
                f"error class ({n_fail}x): '{cls[:100]}' — the "
                "mission cannot make progress this way.",
            }
        if n_fail > _ERR_CLASS_CAP:
            # the first CAP calls returned honest errors; past the cap
            # the same class is a deterministic loop — answer with a
            # streak-exempt guardrail that redirects the model instead of
            # feeding mission-abort
            return {
                "ok": False,
                "_streak_exempt": True,
                "error": f"'{name}' already failed {n_fail}x with "
                f"the same error class — '{cls[:100]}'. The state "
                "will not change on its own; inspect live state "
                "(list_steps/list_parts/validate), use a DIFFERENT "
                "tool, or record a plan note and move on.",
            }
        return res

    def _write_streak_gate(
        self, name: str, args: dict[str, Any], wn: int
    ) -> dict[str, Any] | None:
        """Block an entity write once _WRITE_BLOCK_AT writes are already
        stored: streak-exempt guardrail (with the stored content) up to
        2x the cap, then the honest error so a truly stuck mission can
        abort — an exempt-forever response is its own loop."""
        if wn < _WRITE_BLOCK_AT:
            return None
        wkey = (name, _WRITE_ENTITY_TOOLS[name](args))
        self._write_counts[wkey] = wn + 1
        if wn >= _WRITE_BLOCK_AT * 2:
            return {
                "ok": False,
                "error": f"'{name}' already wrote this entity {wn}x "
                "in this mission — the mission cannot make progress "
                "this way.",
            }
        stored = json.dumps(self._entity_snapshot(name, args), ensure_ascii=False)
        return {
            "ok": False,
            "_streak_exempt": True,
            "error": f"'{name}' already wrote this entity {wn}x in "
            f"this mission — stored content: {stored}. If it is still "
            "wrong, state WHICH field and why in a plan note, then "
            "done_mission; blind rewrites cannot converge.",
        }

    def _entity_snapshot(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Current stored content of the entity a write-streak guardrail is
        blocking — the model must see what already exists instead of
        rewriting blind."""
        try:
            if name == "update_step":
                st = self.state.step_by_number(int(args.get("number", 0)))
                return (
                    {"title": st.title, "text": st.text[:200]} if st else {"missing": True}
                )
            if name == "update_sub_step":
                st = self.state.step_by_number(int(args.get("step", 0)))
                idx = int(args.get("index", -1))
                if st is None or not (0 <= idx < len(st.sub_steps)):
                    return {"missing": True}
                ss = st.sub_steps[idx]
                return {
                    "action": ss.action,
                    "part": ss.part,
                    "target": ss.target,
                    "target2": ss.target2,
                    "text": ss.text[:160],
                }
        except (TypeError, ValueError):
            return {"missing": True}
        return {}

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
            from kiara.guide.assembly_tools import _norm2

            # keys may predate the umlaut-folding normalizer — re-fold on
            # load so old aliases still resolve ("wärmepumpe" → "warmepumpe")
            return {_norm2(str(k)): str(v) for k, v in json.loads(p.read_text()).items()}
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
        return {
            "ok": False,
            "_streak_exempt": True,
            "error": f"page not found — document has pages 1..{n}",
        }

    def _images_result(self, paths: list[str], **extra: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": True, **extra}
        if self.attach_images and self.image_left > 0 and paths:
            # dedup-aware budget: the transport stubs byte-identical
            # re-sends anyway, so only FRESH bytes cost a slot — otherwise
            # two each_step render batches starve the reviewer mid-
            # comparison (observed: 'image budget/role unavailable' while
            # the visual_comparison was still open)
            import hashlib

            seen = self._sent_image_hashes
            fresh: list[str] = []
            for p in paths:
                try:
                    d = hashlib.sha1(Path(p).read_bytes()).hexdigest()
                except Exception:
                    d = ""
                if not d or d not in seen:
                    fresh.append(p)
            out["_images"] = fresh[: self.image_left]
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
        qtext = str(a.get("question", "")).strip() or "Question from guide builder"
        # a parked job resumes into a FRESH mission session — an ask that
        # changes nothing loops ask->park->resume->ask until the repeat
        # detector kills it. Two durable guards: dedupe on identical
        # questions, and a per-mission park cap so paraphrased re-asks
        # (observed: 10 asks about one unresolvable part name) hit an
        # honest guardrail instead of burning sessions.
        mission = getattr(self, "mission", "")
        qnorm = f"{mission}|{' '.join(qtext.lower().split())[:120]}"
        asked = getattr(getattr(self, "state", None), "asked_questions", None)
        mission_asks = (
            sum(1 for q in asked if q.startswith(f"{mission}|"))
            if asked is not None
            else 0
        )
        if asked is not None and (qnorm in asked or mission_asks >= 3):
            return {
                "ok": False,
                "error": "questions like this were already asked and came "
                "back without a deciding answer — re-asking cannot "
                "resolve it. Decide autonomously: document the assumption "
                "via add_master_note, mark_context_part for host "
                "features, or rename_required_name/update_step for plan "
                "names that cannot resolve",
            }
        if asked is not None:
            asked.append(qnorm)
            self.state.save(self.state_dir)
        # OpenQuestion contract (llm/schemas.py): the frontend renders
        # q.question — "text"/"context" keys would show as empty bullets
        question = {
            "question": qtext,
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
        key = ("page", p.page)
        if key in self._delivered_images:
            return self._redeliver_refusal(f"page {p.page} image")
        res = self._images_result([p.image_path], page=p.page)
        if "_images" in res:
            self._delivered_images.add(key)
        return res

    @staticmethod
    def _redeliver_refusal(what: str) -> dict[str, Any]:
        """Refusing a refetch is honest, not restrictive: the transport
        dedupes images by bytes so the call can only ever return the
        identical image again — a guaranteed no-op."""
        return {
            "ok": False,
            "_streak_exempt": True,
            "error": (
                f"{what} was already attached in this mission — a re-fetch "
                "returns byte-identical content and only burns the call "
                "budget. Work from the images already in context; "
                "get_page_text/list_figures answer what the image did not"
            ),
        }

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
                key = ("fig", p.page, idx)
                if key in self._delivered_images:
                    return self._redeliver_refusal(f"figure {idx} on page {p.page}")
                res = self._images_result([str(f["image_path"])], page=p.page, index=idx)
                if "_images" in res:
                    self._delivered_images.add(key)
                return res
        return {
            "ok": False,
            "_streak_exempt": True,
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
            # a read-only miss that returns the valid domain is guidance,
            # not a failure — it must not burn error_streak budget
            return {
                "ok": False,
                "_streak_exempt": True,
                "error": f"step {a.get('number')} not found — valid steps: {nums}",
            }
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
        from kiara.guide.assembly_tools import _part_ref

        p = _part_ref(self, a.get("part_id", ""))
        if p is None:
            ids = [x.id for x in self.state.parts][:15]
            return {
                "ok": False,
                "_streak_exempt": True,
                "error": f"part '{a.get('part_id')}' not found — valid part_ids: {ids}",
            }
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
