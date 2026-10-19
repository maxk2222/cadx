"""Chat endpoints — persistent conversations with the CADX agent.

Flow: the client POSTs a user message (persisted immediately, `pending`),
then the agent worker is enqueued (idempotent — an atomic claim in
run_chat_turn deduplicates concurrent turns). Clients stream progress on
`GET /chats/{id}/events` (SSE on the `chat:<id>` channel — non-terminating,
snapshot-first like the job stream).
"""

from __future__ import annotations

import contextlib
import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadx import events
from cadx.api.deps import enqueue, get_session
from cadx.api.schemas import (
    ChatAttachIn,
    ChatCreate,
    ChatListItem,
    ChatMessageIn,
    ChatMessageOut,
    ChatOut,
    ChatStateOut,
    ChatUpdate,
)
from cadx.db.models import Asset, Chat, ChatMessage, ChatRole, Project

router = APIRouter(tags=["chat"])


def _chat_out(c: Chat) -> ChatOut:
    return ChatOut(
        id=c.id,
        project_id=c.project_id,
        title=c.title,
        turn_running=c.turn_running,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


def _msg_out(m: ChatMessage) -> ChatMessageOut:
    return ChatMessageOut(
        id=m.id,
        chat_id=m.chat_id,
        role=m.role.value,
        content=m.content,
        asset_ids=list(m.asset_ids or []),
        meta=m.meta or {},
        pending=m.pending,
        created_at=m.created_at,
    )


async def _chat_or_404(db: AsyncSession, chat_id: uuid.UUID) -> Chat:
    c = await db.get(Chat, chat_id)
    if c is None:
        raise HTTPException(404, "chat not found")
    return c


async def _ensure_turn(chat_id: uuid.UUID) -> None:
    """Enqueue a turn — the worker's atomic claim deduplicates."""
    await enqueue("run_chat_turn", {"chat_id": str(chat_id)})


@router.post("/chats", response_model=ChatOut, status_code=201)
async def create_chat(body: ChatCreate, db: AsyncSession = Depends(get_session)):
    if body.project_id is not None:
        project = await db.get(Project, body.project_id)
        if project is None:
            raise HTTPException(404, "project not found")
    chat = Chat(project_id=body.project_id, title=body.title or "Chat")
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return _chat_out(chat)


@router.get("/chats", response_model=list[ChatListItem])
async def list_all_chats(db: AsyncSession = Depends(get_session)):
    rows = await db.execute(
        select(Chat, Project.name)
        .join(Project, col(Chat.project_id) == Project.id, isouter=True)
        .order_by(col(Chat.updated_at).desc())
        .limit(200)
    )
    return [
        ChatListItem(**_chat_out(c).model_dump(), project_name=pname) for c, pname in rows.all()
    ]


@router.post("/chats/{chat_id}/project", response_model=ChatOut)
async def attach_chat_project(
    chat_id: uuid.UUID, body: ChatAttachIn, db: AsyncSession = Depends(get_session)
):
    """Bind a chat to a project: an existing one (`project_id`) or a new one
    created on the fly (`name`; empty body → 'New project')."""
    chat = await _chat_or_404(db, chat_id)
    if body.project_id is not None:
        project = await db.get(Project, body.project_id)
        if project is None:
            raise HTTPException(404, "project not found")
    else:
        project = Project(
            name=body.name.strip() or "New project",
            description=body.description.strip(),
            settings={},
        )
        db.add(project)
        await db.flush()
    chat.project_id = project.id
    chat.updated_at = datetime.now(UTC)
    await db.commit()
    with contextlib.suppress(Exception):
        await events.publish_chat(
            str(chat.id), {"type": "chat_update", "chat": _chat_out(chat).model_dump()}
        )
    return _chat_out(chat)


@router.get("/projects/{project_id}/chats", response_model=list[ChatOut])
async def list_chats(project_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    rows = await db.execute(
        select(Chat).where(Chat.project_id == project_id).order_by(col(Chat.updated_at).desc())
    )
    return [_chat_out(c) for c in rows.scalars().all()]


@router.get("/chats/{chat_id}", response_model=ChatStateOut)
async def get_chat(chat_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    chat = await _chat_or_404(db, chat_id)
    rows = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(col(ChatMessage.created_at))
    )
    return ChatStateOut(chat=_chat_out(chat), messages=[_msg_out(m) for m in rows.scalars().all()])


@router.patch("/chats/{chat_id}", response_model=ChatOut)
async def rename_chat(
    chat_id: uuid.UUID, body: ChatUpdate, db: AsyncSession = Depends(get_session)
):
    chat = await _chat_or_404(db, chat_id)
    chat.title = body.title
    await db.commit()
    return _chat_out(chat)


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    chat = await _chat_or_404(db, chat_id)
    if chat.turn_running:
        chat.cancel_requested = True
    rows = await db.execute(select(ChatMessage).where(ChatMessage.chat_id == chat_id))
    for m in rows.scalars().all():
        await db.delete(m)
    await db.delete(chat)
    await db.commit()
    return {"ok": True}


@router.post("/chats/{chat_id}/messages", response_model=ChatMessageOut, status_code=201)
async def post_message(
    chat_id: uuid.UUID, body: ChatMessageIn, db: AsyncSession = Depends(get_session)
):
    """Persist the user message immediately (pending=True → crash-safe), then
    ensure exactly one worker turn is running for the chat."""
    chat = await _chat_or_404(db, chat_id)
    if not body.content.strip() and not body.asset_ids:
        raise HTTPException(422, "message needs text or attachments")
    # attachments must belong to this project — and require one to exist
    if body.asset_ids and chat.project_id is None:
        raise HTTPException(409, "attach a project before sending images")
    if body.asset_ids:
        rows = await db.execute(
            select(Asset).where(
                col(Asset.id).in_(body.asset_ids), Asset.project_id == chat.project_id
            )
        )
        found = {a.id for a in rows.scalars().all()}
        missing = [str(a) for a in body.asset_ids if a not in found]
        if missing:
            raise HTTPException(404, f"asset(s) not in this project: {', '.join(missing)}")
    msg = ChatMessage(
        chat_id=chat.id,
        role=ChatRole.user,
        content=body.content,
        asset_ids=[str(a) for a in body.asset_ids],
        pending=True,
    )
    chat.updated_at = datetime.now(UTC)
    if not chat.title or chat.title == "Chat":
        chat.title = (body.content or "image upload")[:80]
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    await _ensure_turn(chat.id)
    with contextlib.suppress(Exception):
        # best-effort — the row is durable; SSE subscribers get it via
        # the snapshot on reconnect anyway
        await events.publish_chat(
            str(chat.id), {"type": "message", "message": _msg_out(msg).model_dump()}
        )
    return _msg_out(msg)


@router.post("/chats/{chat_id}/run")
async def run_turn(chat_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Ensure a turn is running — idempotent; used for retry/continue."""
    await _chat_or_404(db, chat_id)
    await _ensure_turn(chat_id)
    return {"ok": True}


@router.post("/chats/{chat_id}/cancel")
async def cancel_chat(chat_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    chat = await _chat_or_404(db, chat_id)
    if not chat.turn_running:
        raise HTTPException(409, "no turn running")
    chat.cancel_requested = True
    await db.commit()
    return {"ok": True}


@router.get("/chats/{chat_id}/events")
async def chat_events(chat_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """SSE: snapshot of the full conversation, then live events forever
    (non-terminating — the client reconnects on drop)."""
    chat = await _chat_or_404(db, chat_id)

    async def gen():
        # subscribe BEFORE reading the snapshot — pub/sub has no backlog, so
        # a message committed+published between the read and subscribe would
        # be lost. The post-subscribe read catches anything committed in the
        # gap; `seen` dedupes a message delivered both ways (published after
        # subscribe but committed before the snapshot read).
        client, ps = events.subscribe_chat(str(chat_id))
        try:
            await ps.subscribe(events.chat_channel(str(chat_id)))
            rows = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.chat_id == chat_id)
                .order_by(col(ChatMessage.created_at))
            )
            messages = [_msg_out(m).model_dump() for m in rows.scalars().all()]
            seen = {str(m["id"]) for m in messages}
            yield _fmt("snapshot", {"chat": _chat_out(chat).model_dump(), "messages": messages})
            while True:
                m = await ps.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if m and m.get("data"):
                    try:
                        data = json.loads(m["data"])
                    except Exception:
                        continue
                    if data.get("type") == "message":
                        mobj = data.get("message")
                        mid = mobj.get("id") if isinstance(mobj, dict) else None
                        if mid is not None:
                            if str(mid) in seen:
                                continue
                            seen.add(str(mid))
                    yield _fmt(data.get("type", "event"), data)
                else:
                    yield ": keepalive\n\n"
        finally:
            await ps.unsubscribe(events.chat_channel(str(chat_id)))
            await client.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _fmt(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
