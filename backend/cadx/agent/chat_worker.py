"""Chat turn worker — one arq task per user burst on the 'agent' queue.

Lifecycle of `run_chat_turn(ctx, {"chat_id": ...})`:

1. Atomic claim: ``UPDATE chat SET turn_running=true WHERE NOT turn_running``.
   The API enqueues unconditionally; the second task for the same chat loses
   the claim and returns — the running turn picks up pending messages via its
   ``drain_fn`` between LLM calls.
2. Drain pending user messages → persist them into ``chat.llm_messages``
   immediately (crash-safe), hydrate for the LLM, run the tool session.
3. Stream assistant text: ``on_delta`` snapshots are published to the
   ``chat:<id>`` channel and debounced-persisted into the streaming
   ChatMessage row (~1.5 s) so a page reload mid-turn shows partial output.
4. Finalize: authoritative text, tool log and spawned jobs land on the
   assistant message; the session's new messages are dehydrated into
   ``llm_messages``; loop while pending user messages remain.
5. Release the claim in every path (success / error / cancel / crash guard).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import update as sa_update
from sqlmodel import col, select

from cadx import events, storage
from cadx.agent.chat_tools import ChatToolSession
from cadx.agent.context import read_prompt
from cadx.agent.toolloop import run_session
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx, InputAsset
from cadx.db.models import (
    Asset,
    Chat,
    ChatMessage,
    ChatRole,
    Project,
)
from cadx.db.session import session_scope
from cadx.llm.capabilities import (
    estimate_prompt_tokens,
    load_caps_session,
    record_caps_session,
)
from cadx.llm.images import image_part
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings, load_runtime_settings

log = get_logger(__name__)

_PARTIAL_FLUSH_S = 1.5  # debounce for streaming-row persistence


def _u(s: str | uuid.UUID) -> uuid.UUID:
    return s if isinstance(s, uuid.UUID) else uuid.UUID(str(s))


async def _publish(chat_id: uuid.UUID, event: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        await events.publish_chat(str(chat_id), event)


# --------------------------------------------------------------------------- #
# llm_messages <-> wire messages                                              #
# --------------------------------------------------------------------------- #


def _dehydrate(m: dict[str, Any]) -> dict[str, Any]:
    """LLM message → persisted form: image parts become asset_ref/text
    markers — no base64 blobs in the JSON column."""
    c = m.get("content")
    if not isinstance(c, list):
        return m
    parts = []
    for p in c:
        t = p.get("type")
        if t == "image_url":
            parts.append({"type": "text", "text": "[image]"})
        elif t == "asset_ref":
            parts.append(p)
        else:
            parts.append(p)
    return {**m, "content": parts}


def _hydrate(
    m: dict[str, Any],
    vision: bool,
    assets: dict[str, InputAsset],
    max_images: int = 8,
) -> dict[str, Any]:
    """Persisted message → LLM message: asset_ref markers become real image
    parts (vision role, capped at ``max_images`` per message) or textual
    labels (code role)."""
    c = m.get("content")
    if not isinstance(c, list):
        return m
    parts: list[dict[str, Any]] = []
    images_used = 0
    for p in c:
        if p.get("type") != "asset_ref":
            parts.append(p)
            continue
        aid = str(p.get("asset_id") or "")
        asset = assets.get(aid)
        if vision and images_used < max_images and asset is not None and asset.path.exists():
            parts.append({"type": "text", "text": f"Image (asset {aid}): {asset.caption}"})
            try:
                parts.append(image_part(asset.path))
                images_used += 1
                continue
            except Exception:
                pass
        parts.append(
            {
                "type": "text",
                "text": f"[image asset {aid}"
                + (f" — {asset.caption}]" if asset else "]")
                + ("" if vision else " (text-only model — use view_image)"),
            }
        )
    return {**m, "content": parts}


def _user_llm_msg(m: ChatMessage) -> dict[str, Any]:
    """Persisted user ChatMessage → dehydrated llm_messages entry."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": m.content or ""}]
    for aid in m.asset_ids or []:
        parts.append({"type": "asset_ref", "asset_id": aid})
    return {"role": "user", "content": parts}


def _trim_history(msgs: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Cap the conversation at `keep` messages — cut at a user boundary so
    assistant tool_call/result pairs are never split."""
    if len(msgs) <= keep:
        return msgs
    msgs = msgs[-keep:]
    while msgs and msgs[0].get("role") != "user":
        msgs = msgs[1:]
    return msgs


_SUMMARY_MARKER = "[EARLIER CONVERSATION SUMMARY — older messages condensed]"


def _flatten_content(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict):
                t = p.get("type")
                if t == "asset_ref":
                    parts.append(f"[image asset {p.get('asset_id')}]")
                elif t == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(str(p.get("text") or ""))
            else:
                parts.append(str(p))
        return " ".join(parts)
    return ""


def _digest_messages(dropped: list[dict[str, Any]], keep_digests: bool = True) -> str:
    """One line per dropped message — a summary, not a silent cut. An earlier
    digest is carried forward verbatim so re-condenses stay lossless —
    unless `keep_digests` is false (aggressive token-pressure path), where
    it collapses to a normal line."""
    lines = [_SUMMARY_MARKER]
    for m in dropped:
        text = re.sub(r"\s+", " ", _flatten_content(m.get("content"))).strip()
        if keep_digests and _SUMMARY_MARKER in text[:120]:
            lines.append(text)
            continue
        role = str(m.get("role") or "?")
        calls = m.get("tool_calls") or []
        names = ",".join(
            str((tc.get("function") or {}).get("name") or "")
            for tc in calls
            if isinstance(tc, dict)
        )
        if names:
            text += f" [tools: {names}]"
        if len(text) > 160:
            text = text[:159] + "…"
        lines.append(f"{role}: {text}" if text else role)
    return "\n".join(lines)


def _summarize_history(msgs: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Cap the conversation at `keep` messages — the dropped prefix becomes a
    digest user message (summarized, not truncated). Cut stays at a user
    boundary so tool_call/result pairs are never split."""
    if len(msgs) <= keep:
        return msgs
    cut = len(msgs) - keep
    while cut < len(msgs) and msgs[cut].get("role") != "user":
        cut += 1
    if cut >= len(msgs):
        return msgs[-keep:]
    return [{"role": "user", "content": _digest_messages(msgs[:cut])}, *msgs[cut:]]


def _summarize_for_ctx(
    msgs: list[dict[str, Any]], est_fn: Any, budget: int
) -> list[dict[str, Any]]:
    """Token-pressure variant: msgs = [system, *history] — collapse the oldest
    half of history into a digest until the estimate fits `budget`
    (bounded rounds)."""
    sys, rest = msgs[0], list(msgs[1:])
    for _ in range(6):
        if len(rest) <= 2 or est_fn([sys, *rest]) <= budget:
            break
        cut = max(1, (len(rest) + 1) // 2)
        while cut < len(rest) and rest[cut].get("role") != "user":
            cut += 1
        if cut >= len(rest):
            break
        rest = [
            {"role": "user", "content": _digest_messages(rest[:cut], keep_digests=False)},
            *rest[cut:],
        ]
    return [sys, *rest]


# --------------------------------------------------------------------------- #
# turn runner                                                                 #
# --------------------------------------------------------------------------- #


async def run_chat_turn(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    chat_id = _u(payload["chat_id"])
    # atomic turn claim — second concurrent task for the same chat exits here
    async with session_scope() as db:
        res = await db.execute(
            sa_update(Chat)
            .where(col(Chat.id) == chat_id, col(Chat.turn_running).is_(False))
            .values(turn_running=True)
        )
        await db.commit()
    if (res.rowcount or 0) == 0:
        return {"ok": True, "skipped": "turn already running"}

    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        chat.cancel_requested = False
        await db.commit()

    try:
        return await _run_turns(ctx, chat_id)
    except asyncio.CancelledError:
        await _finish_error(chat_id, "chat turn cancelled (worker shutdown/timeout)")
        raise
    except Exception as e:
        log.exception("chat.turn_failed", chat=str(chat_id))
        await _finish_error(chat_id, f"{type(e).__name__}: {e}")
        return {"ok": False, "error": str(e)[:300]}
    finally:
        async with session_scope() as db:
            chat = await db.get(Chat, chat_id)
            if chat is not None:
                chat.turn_running = False
                await db.commit()


async def _finish_error(chat_id: uuid.UUID, error: str) -> None:
    """Mark the streaming assistant row (if any) failed + publish error."""
    try:
        async with session_scope() as db:
            rows = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.chat_id == chat_id)
                .order_by(col(ChatMessage.created_at).desc())
            )
            latest = rows.scalars().first()
            if latest is not None and latest.meta.get("streaming"):
                latest.meta = {**latest.meta, "streaming": False, "error": error}
                await db.commit()
        await _publish(chat_id, {"type": "error", "message": error})
    except Exception:
        pass


async def _drain_pending(chat_id: uuid.UUID) -> list[dict[str, Any]]:
    """Move pending user ChatMessages into chat.llm_messages NOW (crash-safe)
    and return the persisted (dehydrated) LLM messages — callers hydrate them
    for the model context."""
    persisted: list[dict[str, Any]] = []
    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None:
            return []
        rows = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id, col(ChatMessage.pending).is_(True))
            .order_by(col(ChatMessage.created_at))
        )
        for m in rows.scalars().all():
            if m.role != ChatRole.user:
                m.pending = False
                continue
            persisted.append(_user_llm_msg(m))
            m.pending = False
        if persisted:
            chat.llm_messages = [*(chat.llm_messages or []), *persisted]
            await db.commit()
    return persisted


async def _pending_count(chat_id: uuid.UUID) -> int:
    async with session_scope() as db:
        rows = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id, col(ChatMessage.pending).is_(True))
            .limit(1)
        )
        return 1 if rows.scalars().first() is not None else 0


async def _persist_partial(message_id: uuid.UUID, text: str) -> None:
    """Debounced write of the streaming snapshot into the assistant row."""
    try:
        async with session_scope() as db:
            m = await db.get(ChatMessage, message_id)
            if m is not None and m.meta.get("streaming"):
                m.content = text
                await db.commit()
    except Exception:
        pass


async def _chat_emit(chat_id: uuid.UUID, event: str, data: dict[str, Any]) -> None:
    await _publish(chat_id, {"type": event, **data})


def _make_on_delta(
    chat_id: uuid.UUID, amsg_id: uuid.UUID
) -> tuple[Any, dict[str, str], set[asyncio.Task]]:
    """on_delta closure: publish cumulative snapshots + debounced persist.

    Returns the spawned-task set — the caller must gather it before the
    final write so a queued partial flush can't clobber the finished text
    (lost-update race)."""
    buf: dict[str, str] = {"text": ""}
    last_flush: dict[str, float] = {"t": 0.0}
    tasks: set[asyncio.Task] = set()

    def spawn(coro: Any) -> None:
        t = asyncio.get_running_loop().create_task(coro)
        tasks.add(t)
        t.add_done_callback(tasks.discard)

    def on_delta(text: str) -> None:
        buf["text"] = text
        with contextlib.suppress(RuntimeError):
            spawn(
                _publish(
                    chat_id, {"type": "message_delta", "message_id": str(amsg_id), "text": text}
                )
            )
        now = time.monotonic()
        if now - last_flush["t"] >= _PARTIAL_FLUSH_S:
            last_flush["t"] = now
            with contextlib.suppress(RuntimeError):
                spawn(_persist_partial(amsg_id, text))

    return on_delta, buf, tasks


async def _run_turns(  # noqa: C901
    ctx: dict[str, Any], chat_id: uuid.UUID
) -> dict[str, Any]:
    async def _load_project_ctx(
        chat: Chat,
    ) -> tuple[Project | None, list[InputAsset], dict[str, InputAsset]]:
        """Project + hydrated assets for the chat's CURRENT binding — the
        binding can change between iterations via create_project/attach."""
        async with session_scope() as db:
            project = await db.get(Project, chat.project_id) if chat.project_id else None
            if project is None:
                return None, [], {}
            rows = await db.execute(select(Asset).where(Asset.project_id == project.id))
            project_assets = list(rows.scalars().all())
        assets = [
            InputAsset(
                asset_id=str(a.id),
                kind=a.kind.value,
                path=storage.resolve(a.path),
                caption=a.caption,
                filename=a.filename,
                role_hint=str((a.gen_params or {}).get("role_hint") or ""),
            )
            for a in project_assets
        ]
        return project, assets, {a.asset_id: a for a in assets}

    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None:
            return {"ok": False, "error": "chat not found"}
        runtime = await load_runtime_settings(db)

    project, assets, assets_by_id = await _load_project_ctx(chat)
    bound_pid = chat.project_id

    role = runtime.chat_model_role
    vision = role == "vision"
    # the tool loop enforces runtime_settings.agent_tool_max_turns — chat
    # gets its own (larger) budget, so the ctx carries an adjusted copy
    turn_runtime = runtime.model_copy(update={"agent_tool_max_turns": runtime.chat_max_turns})
    llm = _make_llm(runtime, caps=await load_caps_session(), caps_cb=record_caps_session)
    image = _make_image(runtime)
    image_ok = await _image_available(image)

    async def check_cancel() -> bool:
        async with session_scope() as db:
            c = await db.get(Chat, chat_id)
            return c is None or bool(c.cancel_requested)

    async def has_pending() -> bool:
        return bool(await _pending_count(chat_id))

    async def _noop_emit(step: str, iteration: int, message: str, **fields: Any) -> None:
        return None

    sys_prompt = read_prompt("chat_system.md")
    drained_this_turn: deque[dict[str, Any]] = deque()

    while True:
        # drain pending user messages BEFORE the LLM call — persisted to
        # llm_messages inside _drain_pending, so a crash never loses them
        fresh = await _drain_pending(chat_id)
        async with session_scope() as db:
            chat = await db.get(Chat, chat_id)
            if chat is None:
                return {"ok": False, "error": "chat deleted"}
            history = list(chat.llm_messages or [])
        # the binding may have changed between iterations (agent called
        # create_project/attach_project, or the user bound via the API) —
        # reload project context so hydration/tools see the right assets
        if chat.project_id != bound_pid:
            project, assets, assets_by_id = await _load_project_ctx(chat)
            bound_pid = chat.project_id
        if not fresh and not history:
            break  # nothing to answer — defensive; the API always enqueues on a msg
        history = _summarize_history(history, runtime.chat_history_messages)
        msgs = [
            {"role": "system", "content": sys_prompt},
            *[_hydrate(m, vision, assets_by_id, runtime.chat_image_max) for m in history],
        ]
        # token-pressure pass — when the detected context window is known and
        # the hydrated prompt is over budget, collapse more history into a
        # digest before the provider rejects with a 400
        ctx_len = (getattr(llm, "context_len", None) or {}).get(role)
        if ctx_len:

            def _est(m: list[dict[str, Any]]) -> int:
                return estimate_prompt_tokens(
                    m,
                    image_detail=runtime.llm_image_detail,
                    image_max_edge=runtime.llm_image_max_edge,
                )

            if _est(msgs) > int(ctx_len * runtime.agent_context_budget_frac):
                msgs = _summarize_for_ctx(
                    msgs, _est, int(ctx_len * runtime.agent_context_budget_frac)
                )
        n0 = len(msgs)
        drained_this_turn.clear()

        async def drain_fn() -> list[dict[str, Any]]:
            got = await _drain_pending(chat_id)
            drained_this_turn.extend(got)
            # reads the function-level name at call time — deliberately sees
            # the assets of the current binding (see the reload above)
            return [
                _hydrate(m, vision, assets_by_id, runtime.chat_image_max)  # noqa: B023
                for m in got
            ]

        async def publish_fn(event: str, data: dict[str, Any]) -> None:
            await _chat_emit(chat_id, event, data)
            if event == "tool_result":
                # tool_log/spawned live only in session memory until turn_end —
                # flush them onto the assistant row now so a reload mid-turn
                # restores the tool chips and spawned job cards from the
                # snapshot, and emit a meta event so connected clients mount
                # the card the moment the job is spawned. session/amsg_id are
                # read at call time — the current turn's binding (as above).
                meta = {
                    "streaming": True,
                    "tool_log": session.tool_log,  # noqa: B023
                    "spawned": session.spawned,  # noqa: B023
                }
                async with session_scope() as db:
                    m = await db.get(ChatMessage, amsg_id)  # noqa: B023
                    if m is not None and m.meta.get("streaming"):
                        m.meta = meta
                        await db.commit()
                await _chat_emit(
                    chat_id,
                    "meta",
                    {"message_id": str(amsg_id), "meta": meta},  # noqa: B023
                )

        actx = AgentCtx(
            llm=llm,
            executor=None,  # chat never calls the executor directly — jobs do
            trace=MemoryTrace(),
            emit=_noop_emit,
            check_cancel=check_cancel,
            job_id=f"chat:{chat_id}",
            project_id=chat.project_id,  # may be None — unbound chat
            version_id=chat_id,  # unused by chat tools; keeps the dataclass happy
            mode="chat",
            prompt="",
            settings=project_settings(project.settings if project else {}, runtime),
            check_finish=None,
            runtime_settings=turn_runtime,
            assets=assets,
            image=image,
            image_available=image_ok,
        )
        session = ChatToolSession(
            actx,
            chat_id=chat_id,
            attach_images=vision,
            cancel_fn=check_cancel,
            interrupt_fn=has_pending,
            publish_fn=publish_fn,
        )

        # streaming assistant row — deltas land here (debounced) + on the wire
        async with session_scope() as db:
            amsg = ChatMessage(
                chat_id=chat_id, role=ChatRole.assistant, content="", meta={"streaming": True}
            )
            db.add(amsg)
            await db.commit()
            amsg_id = amsg.id
        await _chat_emit(chat_id, "turn_start", {"message_id": str(amsg_id)})

        on_delta, buf, delta_tasks = _make_on_delta(chat_id, amsg_id)

        try:
            res = await run_session(
                actx,
                role,
                msgs,
                session=session,
                max_calls=runtime.chat_tool_max_calls,
                step="chat",
                drain_fn=drain_fn,
                on_delta=on_delta,
            )
        except Exception as e:
            # tool loop blew up — persist whatever streamed so far
            if delta_tasks:
                await asyncio.gather(*delta_tasks, return_exceptions=True)
            async with session_scope() as db:
                amsg = await db.get(ChatMessage, amsg_id)
                if amsg is not None:
                    amsg.content = buf["text"]
                    # keep the tool chips + spawned cards visible after a crash
                    amsg.meta = {
                        "streaming": False,
                        "error": str(e)[:400],
                        "tool_log": session.tool_log,
                        "spawned": session.spawned,
                    }
                    await db.commit()
            await _chat_emit(chat_id, "error", {"message": str(e)[:400]})
            raise

        # drain queued partial-persist/delta-publish tasks BEFORE the final
        # write — a late partial flush must not overwrite the finished text
        if delta_tasks:
            await asyncio.gather(*delta_tasks, return_exceptions=True)

        final_text = (res.response.text if res.response else "") or buf["text"]

        # persist the session tail: everything after n0, minus entries
        # drain_fn already stored (matched positionally, in order)
        new_llm: list[dict[str, Any]] = []
        for m in res.messages_out[n0:]:
            dm = _dehydrate(m)
            if drained_this_turn and dm == drained_this_turn[0]:
                drained_this_turn.popleft()
                continue
            new_llm.append(dm)
        # run_session's messages_out ends at the last tool echo — the final
        # assistant answer lives on res.response and must be appended for
        # the conversation to survive into the next turn
        if final_text:
            new_llm.append({"role": "assistant", "content": final_text})

        meta = {
            "streaming": False,
            "tool_log": session.tool_log,
            "spawned": session.spawned,
            "cancelled": res.cancelled,
            "exhausted": res.exhausted,
        }
        async with session_scope() as db:
            chat = await db.get(Chat, chat_id)
            if chat is not None:
                chat.llm_messages = _summarize_history(
                    [*(chat.llm_messages or []), *new_llm], runtime.chat_history_messages
                )
                chat.updated_at = datetime.now(UTC)
            amsg = await db.get(ChatMessage, amsg_id)
            if amsg is not None:
                amsg.content = final_text
                amsg.meta = meta
            await db.commit()
        await _chat_emit(
            chat_id,
            "turn_end",
            {
                "message_id": str(amsg_id),
                "cancelled": res.cancelled,
                "exhausted": res.exhausted,
                "meta": meta,
            },
        )
        await _chat_emit(
            chat_id,
            "message_update",
            {"message_id": str(amsg_id), "content": final_text},
        )

        if not await has_pending():
            break
    return {"ok": True}


# --------------------------------------------------------------------------- #
# reaper — runs at worker startup                                             #
# --------------------------------------------------------------------------- #


async def reap_orphaned_chats(ctx: dict[str, Any]) -> None:
    """Clear turn_running left behind by a worker restart and re-enqueue
    turns for chats that still have unanswered pending user messages."""
    try:
        async with session_scope() as db:
            stale = (
                (await db.execute(select(Chat).where(col(Chat.turn_running).is_(True))))
                .scalars()
                .all()
            )
            pending = (
                (
                    await db.execute(
                        select(ChatMessage.chat_id)
                        .where(col(ChatMessage.pending).is_(True))
                        .distinct()
                    )
                )
                .scalars()
                .all()
            )
            for c in stale:
                c.turn_running = False
                log.info("chat.reaper.clear_orphan", chat=str(c.id))
            await db.commit()
        redis = ctx.get("arq_redis")
        for cid in pending:
            if redis is not None:
                await redis.enqueue_job("run_chat_turn", {"chat_id": str(cid)}, _queue_name="agent")
                log.info("chat.reaper.reenqueued", chat=str(cid))
    except Exception:
        log.exception("chat.reaper_failed")


# --------------------------------------------------------------------------- #
# small factories (mirrors of worker_jobs helpers, kept separate so a broken  #
# executor/mesh config cannot break the chat loop)                            #
# --------------------------------------------------------------------------- #


def _make_llm(
    runtime: RuntimeSettings,
    *,
    caps: dict[str, Any] | None = None,
    caps_cb: Any = None,
):
    from cadx.llm.client import make_llm

    return make_llm(runtime, caps=caps, caps_cb=caps_cb)


def _make_image(runtime: RuntimeSettings):
    from cadx.agent.image_client import make_image_client

    return make_image_client(runtime_settings=runtime)


async def _image_available(client: Any) -> bool:
    if client is None:
        return False
    try:
        return any(m.available for m in await client.models())
    except Exception:
        return False


def project_settings(raw: dict[str, Any], runtime: RuntimeSettings):
    """project.settings → ProjectSettings with runtime fallbacks filled in."""
    from cadx.db.models import ProjectSettings

    s = ProjectSettings(**{k: v for k, v in raw.items() if not k.startswith("ui_")})
    if s.fem and not s.fem_material and s.material:
        s.fem_material = s.material
    for key, default in (
        ("clarification_mode", runtime.agent_clarification_mode),
        ("max_iterations", runtime.agent_max_iterations),
        ("max_repairs", runtime.agent_max_repairs),
        ("pass_score", runtime.agent_pass_score),
        ("dim_tolerance_pct", runtime.agent_dim_tolerance_pct),
        ("dim_tolerance_mm", runtime.agent_dim_tolerance_mm),
        ("fem_voxel_mm", runtime.fem_voxel_mm),
        ("topopt_volfrac", runtime.topopt_default_volfrac),
        ("topopt_iterations", runtime.topopt_max_iterations),
    ):
        if getattr(s, key) is None:
            setattr(s, key, default)
    if not s.mesh_model:
        s.mesh_model = runtime.mesh_model
    return s
