"""Chat-agent tool session — the full CADX capability set as LLM tools.

`ChatToolSession` subclasses the generic `ToolSession` so the catalog /
library / image-inspection handlers are reused verbatim; chat adds
project/version inspection plus ACTION tools that spawn real versions and
jobs through `cadx.spawn` (the identical code path the REST routers take)
and block until the job reaches a terminal state. The agent therefore sees
each result — including contact-sheet renders via `_images` on the vision
role — and can relay outcomes, clarify, or chain follow-up work.

The project/tool machinery itself lives in `project_tools.ProjectToolSession`
— this subclass adds only the Chat-row specifics: the persisted binding
(chat.project_id), chat_update publishing and the three LLM-touching tools
(generate_part, edit_version, answer_questions) that LLM-free surfaces such
as the MCP server must not offer.

Safety boundaries: no arbitrary code execution (every CAD/mesh run goes
through the sandboxed executor pipeline as a normal job), no filesystem
access, no deletion tools. Everything the agent does is visible in the
normal version tree and jobs dashboard.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from cadx.agent.project_tools import (
    CHAT_ACTION_TOOLS,
    CHAT_INFO_TOOLS,
    LLM_ACTION_TOOLS,
    PROJECT_TOOLS,
    SAFE_ACTION_TOOLS,
    ProjectToolSession,
)
from cadx.agent.types import AgentCtx
from cadx.db.models import Chat, Job, Project, Version
from cadx.db.session import session_scope
from cadx.logging import get_logger
from cadx.spawn import SpawnError, enqueue_resume, spawn_edit, spawn_generate

log = get_logger(__name__)

__all__ = [
    "CHAT_ACTION_TOOLS",
    "CHAT_INFO_TOOLS",
    "LLM_ACTION_TOOLS",
    "PROJECT_TOOLS",
    "SAFE_ACTION_TOOLS",
    "ChatToolSession",
]


class ChatToolSession(ProjectToolSession):
    """kind="chat" — project inspection + job-spawning actions."""

    def __init__(
        self,
        ctx: AgentCtx,
        *,
        chat_id: uuid.UUID,
        attach_images: bool = True,
        cancel_fn: Callable[[], Awaitable[bool]] | None = None,
        interrupt_fn: Callable[[], Awaitable[bool]] | None = None,
        enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        publish_fn: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(
            ctx,
            attach_images=attach_images,
            cancel_fn=cancel_fn,
            interrupt_fn=interrupt_fn,
            enqueue_fn=enqueue_fn,
            publish_fn=publish_fn,
        )
        self.kind = "chat"
        self.chat_id = chat_id

    # -- chat specifics ------------------------------------------------------ #

    def _action_tools(self) -> list[dict[str, Any]]:
        return CHAT_ACTION_TOOLS

    async def _refresh_binding(self) -> None:
        # a project may have been bound via the API while this turn runs —
        # pick it up so the next tool call sees the full toolset
        if self._pid is None:
            async with session_scope() as db:
                chat = await db.get(Chat, self.chat_id)
                if chat is not None and chat.project_id is not None:
                    await self._apply_bind(db, chat.project_id)

    async def _apply_bind(self, db: Any, project_id: uuid.UUID) -> None:
        """Re-point the session at the project the CHAT row holds — the chat
        binding lives on the row, not just in memory."""
        await self._bind(db, project_id)

    async def _project(self, db: Any) -> Project:
        # read the binding live from the chat row — it may have changed via
        # the API or a bind tool since the session was constructed
        chat = await db.get(Chat, self.chat_id)
        pid = chat.project_id if chat is not None else None
        if pid is None:
            raise SpawnError(
                "no project bound — call create_project or attach_project first",
                status=409,
            )
        if pid != self._pid:
            await self._apply_bind(db, pid)
        p = await db.get(Project, pid)
        if p is None:
            raise SpawnError("project not found", status=404)
        return p

    def _chat_out_dict(self, chat: Chat) -> dict[str, Any]:
        return {
            "id": str(chat.id),
            "project_id": str(chat.project_id) if chat.project_id else None,
            "title": chat.title,
            "turn_running": chat.turn_running,
            "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat(),
        }

    async def _persist_bind(self, db: Any, project: Project) -> dict[str, Any]:
        """Persist the binding on the Chat row + notify the UI."""
        chat = await db.get(Chat, self.chat_id)
        if chat is None:
            return {"ok": False, "error": "chat not found"}
        chat.project_id = project.id
        chat.updated_at = datetime.now(UTC)
        await self._apply_bind(db, project.id)
        await db.commit()
        await self._publish("chat_update", {"chat": self._chat_out_dict(chat)})
        return {
            "ok": True,
            "project_id": str(project.id),
            "project_name": project.name,
            "bound": True,
        }

    # -- chat-only action tools (spawn LLM-driven jobs) ----------------------- #

    async def _tool_generate_part(self, a: dict[str, Any]) -> dict[str, Any]:
        prompt = str(a.get("prompt", "")).strip()

        async def spawn(db: Any) -> tuple[Version, Job]:
            project = await self._project(db)
            return await spawn_generate(
                db,
                project,
                prompt,
                gen_mode=str(a.get("mode") or ""),
                asset_ids=[str(x) for x in a.get("asset_ids") or []],
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("generate_part", spawn, bool(a.get("wait", True)))

    async def _tool_edit_version(self, a: dict[str, Any]) -> dict[str, Any]:
        prompt = str(a.get("prompt", "")).strip()

        async def spawn(db: Any) -> tuple[Version, Job]:
            parent = await self._version(db, str(a.get("version_id", "")))
            project = await self._project(db)
            return await spawn_edit(
                db,
                project,
                parent,
                prompt,
                gen_mode=str(a.get("mode") or ""),
                enqueue_fn=self._enqueue_fn,
            )

        return await self._spawn_and_wait("edit_version", spawn, bool(a.get("wait", True)))

    async def _tool_answer_questions(self, a: dict[str, Any]) -> dict[str, Any]:
        answers = a.get("answers") or []
        if not isinstance(answers, list) or not answers:
            return {"ok": False, "error": "answers list required"}
        answers_text = "\n".join(
            f"A{i + 1}: {x.get('answer') if isinstance(x, dict) else x}"
            for i, x in enumerate(answers)
        )

        async def spawn(db: Any) -> tuple[Version, Job]:
            try:
                jid = uuid.UUID(str(a.get("job_id", "")))
            except ValueError:
                raise SpawnError("invalid job id", status=422) from None
            j = await db.get(Job, jid)
            if j is None:
                raise SpawnError("job not found", status=404)
            return await enqueue_resume(db, j, answers_text, enqueue_fn=self._enqueue_fn)

        return await self._spawn_and_wait("answer_questions", spawn, bool(a.get("wait", True)))
