"""Chat worker: turn claim, pending drain, assistant persistence (FakeLLM)."""

from __future__ import annotations

import uuid

import pytest
from sqlmodel import col, select

from cadx.agent import chat_worker
from cadx.db.models import Chat, ChatMessage, ChatRole, Project
from cadx.db.session import create_all, session_scope
from cadx.llm.fake import FakeLLM


@pytest.fixture(scope="module", autouse=True)
async def _tables():
    await create_all()


async def _mk_chat(content: str = "make me a cube") -> tuple[uuid.UUID, uuid.UUID]:
    async with session_scope() as db:
        p = Project(name=f"chat-test-{uuid.uuid4().hex[:6]}")
        db.add(p)
        await db.commit()
        await db.refresh(p)
        c = Chat(project_id=p.id, title="t")
        db.add(c)
        await db.commit()
        await db.refresh(c)
        db.add(ChatMessage(chat_id=c.id, role=ChatRole.user, content=content, pending=True))
        await db.commit()
    return p.id, c.id


def _patch_llm(monkeypatch, fake: FakeLLM) -> None:
    monkeypatch.setattr(chat_worker, "_make_llm", lambda _rt: fake)

    def _no_image(_rt):
        return None

    monkeypatch.setattr(chat_worker, "_make_image", _no_image)


async def _messages(chat_id: uuid.UUID) -> list[ChatMessage]:
    async with session_scope() as db:
        rows = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(col(ChatMessage.created_at))
        )
        return list(rows.scalars().all())


async def test_turn_persists_answer_and_history(monkeypatch):
    """A text turn: user msg drained into llm_messages, assistant row filled,
    pending cleared, claim released."""
    _pid, cid = await _mk_chat()
    fake = FakeLLM(["I'll build a 20mm cube for you."])
    _patch_llm(monkeypatch, fake)

    res = await chat_worker.run_chat_turn({}, {"chat_id": str(cid)})
    assert res["ok"]

    msgs = await _messages(cid)
    user = next(m for m in msgs if m.role == ChatRole.user)
    asst = next(m for m in msgs if m.role == ChatRole.assistant)
    assert user.pending is False  # drained, not deleted
    assert asst.content == "I'll build a 20mm cube for you."
    assert asst.meta["streaming"] is False
    assert asst.meta["cancelled"] is False

    async with session_scope() as db:
        chat = await db.get(Chat, cid)
        assert chat.turn_running is False  # claim released
        history = chat.llm_messages or []
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


async def test_turn_claim_is_atomic(monkeypatch):
    """A second run while turn_running=True must exit immediately — the API
    enqueues unconditionally, the claim deduplicates."""
    _pid, cid = await _mk_chat()
    async with session_scope() as db:
        c = await db.get(Chat, cid)
        c.turn_running = True
        await db.commit()

    fake = FakeLLM(["should never be called"])
    _patch_llm(monkeypatch, fake)
    res = await chat_worker.run_chat_turn({}, {"chat_id": str(cid)})
    assert res == {"ok": True, "skipped": "turn already running"}
    assert fake.calls == []  # no LLM traffic on a lost claim

    # message is still pending for the real turn
    msgs = await _messages(cid)
    assert next(m for m in msgs if m.role == ChatRole.user).pending is True


async def test_cancel_stops_turn(monkeypatch):
    """cancel_requested raised mid-turn aborts the session — the flag is
    cleared on claim, so it must be set while the turn is running."""
    _pid, cid = await _mk_chat()

    class CancellingLLM(FakeLLM):
        async def chat(self, *a, **kw):
            # first turn: tool call + raise the cancel flag; the loop's
            # next-turn check_cancel() then aborts the session
            async with session_scope() as db:
                c = await db.get(Chat, cid)
                c.cancel_requested = True
                await db.commit()
            return await super().chat(*a, **kw)

    fake = CancellingLLM([{"tool_calls": [{"name": "list_versions", "arguments": {}}]}, "done"])
    _patch_llm(monkeypatch, fake)

    res = await chat_worker.run_chat_turn({}, {"chat_id": str(cid)})
    assert res["ok"]

    msgs = await _messages(cid)
    asst = next(m for m in msgs if m.role == ChatRole.assistant)
    assert asst.meta["cancelled"] is True
    async with session_scope() as db:
        chat = await db.get(Chat, cid)
        assert chat.turn_running is False


async def test_drain_pending_persists_immediately(monkeypatch):
    """Crash safety: after _drain_pending the text is in llm_messages even
    before any LLM call happened."""
    _pid, cid = await _mk_chat("first question")
    drained = await chat_worker._drain_pending(cid)
    assert len(drained) == 1
    assert drained[0]["role"] == "user"

    msgs = await _messages(cid)
    assert msgs[0].pending is False

    async with session_scope() as db:
        chat = await db.get(Chat, cid)
        assert len(chat.llm_messages) == 1
        parts = chat.llm_messages[0]["content"]
        assert parts[0]["text"] == "first question"

    # second drain returns nothing — no duplicates
    assert await chat_worker._drain_pending(cid) == []


async def test_asset_refs_dehydrate_and_hydrate():
    """asset_ref markers survive the llm_messages round trip; hydrate emits
    image parts only for the vision role."""
    from cadx.agent.types import InputAsset

    m = {
        "role": "user",
        "content": [{"type": "text", "text": "see"}, {"type": "asset_ref", "asset_id": "a1"}],
    }
    d = chat_worker._dehydrate(m)
    assert d["content"][1]["type"] == "asset_ref"

    from pathlib import Path

    assets = {
        "a1": InputAsset(
            asset_id="a1",
            kind="image",
            path=Path("/nonexistent.png"),
            caption="cap",
            filename="x.png",
        )
    }
    # vision: missing file falls back to a caption-only label
    h = chat_worker._hydrate(m, vision=True, assets=assets)
    assert h["content"][1]["type"] == "text"
    assert "cap" in h["content"][1]["text"]
    # code role: explicitly text-only
    h2 = chat_worker._hydrate(m, vision=False, assets=assets)
    assert all(p["type"] == "text" for p in h2["content"])
    assert "text-only" in h2["content"][-1]["text"]


def test_trim_history_cuts_at_user_boundary():
    msgs = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a3"},
    ]
    out = chat_worker._trim_history(msgs, 4)
    assert len(out) <= 4
    assert out[0]["role"] == "user"
    # tool messages must never be orphaned at the cut boundary
    assert out[0].get("role") != "tool"
