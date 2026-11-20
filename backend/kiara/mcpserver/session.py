"""MCP-side tool session + per-transport session pool.

`McpToolSession` is a `ProjectToolSession` with `ctx.llm = None` — the HARD
contract of the MCP surface: every exposed tool is deterministic (spawned
jobs run through the same sandboxed executor pipeline as the REST API).

Spawn tools default to `wait=false` here — external agents poll via
`wait_job` / `job_status` rather than holding an HTTP request open for the
duration of a FEM/topopt run.

Sessions are pooled per transport session (Streamable-HTTP `Mcp-Session-Id`,
a constant key for stdio) so project bindings and `spawned` tracking survive
across requests within one MCP session.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import Any

from kiara.agent.project_tools import MCP_HANDLERS, SAFE_ACTION_TOOLS, ProjectToolSession
from kiara.agent.trace import MemoryTrace
from kiara.agent.types import AgentCtx
from kiara.db.models import ProjectSettings
from kiara.logging import get_logger
from kiara.runtime_config import RuntimeSettings

log = get_logger(__name__)

# spawn-ish tools whose schema offers a `wait` flag — MCP defaults it to
# False (the agent polls), unlike the chat surface which blocks by default
_WAIT_DEFAULT_FALSE = {
    t["function"]["name"]
    for t in SAFE_ACTION_TOOLS
    if "wait" in (t["function"].get("parameters", {}).get("properties") or {})
}

# pool key for transports without a session id (stdio)
STDIO_SESSION_KEY = "stdio"


async def _noop_emit(step: str, iteration: int, message: str, **fields: Any) -> None:
    return None


async def _never_cancel() -> bool:
    return False


class McpToolSession(ProjectToolSession):
    """LLM-free project tool session for the MCP server."""

    def __init__(self, runtime_settings: RuntimeSettings | None = None) -> None:
        from kiara.agent.executor_client import QueueExecutor

        rt = runtime_settings or RuntimeSettings()
        ctx = AgentCtx(
            llm=None,  # HARD contract — the MCP surface never triggers an LLM
            executor=QueueExecutor(None, runtime_settings=rt),
            trace=MemoryTrace(),
            emit=_noop_emit,
            check_cancel=_never_cancel,
            job_id="mcp",
            project_id=None,  # bound per-session via create_project/attach_project
            version_id=uuid.uuid4(),  # unused by MCP tools; dataclass field
            mode="mcp",
            prompt="",
            settings=ProjectSettings(),
            runtime_settings=rt,
        )
        super().__init__(ctx, attach_images=True)
        from kiara.api.deps import enqueue

        # real arq enqueue — spawned jobs run on the normal agent queue
        self._enqueue_fn = enqueue
        self._mcp_handlers_enabled = True
        self._mcp_tools_loaded = False
        self._warmed = False

    async def aclose(self) -> None:
        """Release resources the session lazily built — the QueueExecutor's
        arq pool and the mesh/image sidecar clients' httpx pools. All closes
        are best-effort: a half-torn-down session must never raise."""
        for obj in (
            self.ctx.executor,
            getattr(self.ctx, "mesh", None),
            getattr(self.ctx, "image", None),
        ):
            close = getattr(obj, "aclose", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()

    # -- lazy sidecar wiring -------------------------------------------------- #

    async def warm(self) -> None:
        """Construct the mesh/image ports on first use and probe availability
        (bounded — a dead sidecar must not stall session creation)."""
        if self._warmed:
            return
        self._warmed = True
        rt = self.ctx.runtime_settings
        try:
            from kiara.agent.mesh_client import make_mesh_client

            self.ctx.mesh = make_mesh_client(runtime_settings=rt)
            if self.ctx.mesh is not None:
                health = await asyncio.wait_for(self.ctx.mesh.health(), timeout=5.0)
                self.ctx.mesh_available = bool(health and health.ok)
        except Exception as e:
            log.warning("mcp.mesh_probe_failed", error=str(e)[:200])
        try:
            from kiara.agent.image_client import make_image_client

            self.ctx.image = make_image_client(runtime_settings=rt)
            if self.ctx.image is not None:
                models = await asyncio.wait_for(self.ctx.image.models(), timeout=5.0)
                self.ctx.image_available = any(m.available for m in models)
        except Exception as e:
            log.warning("mcp.image_probe_failed", error=str(e)[:200])
        if self.ctx.image is not None and self.ctx.image_available:
            self.image_left = rt.image_tool_budget

    # -- MCP tool surface ----------------------------------------------------- #

    def _load_mcp_tools(self) -> None:
        """Import `kiara.mcpserver.tools` once — registers MCP_HANDLERS and
        provides MCP_TOOL_SCHEMAS. Tolerant while SA-3's file is absent."""
        if self._mcp_tools_loaded:
            return
        self._mcp_tools_loaded = True
        with contextlib.suppress(ImportError):
            from kiara.mcpserver import tools as mcp_tools

            # merge SA-3's handler table into the shared registry — the
            # `_dispatch` fallback chain reads project_tools.MCP_HANDLERS
            MCP_HANDLERS.update(mcp_tools.MCP_HANDLERS)

    def _mcp_schemas(self) -> list[dict[str, Any]]:
        try:
            from kiara.mcpserver.tools import MCP_TOOL_SCHEMAS
        except ImportError:
            return []
        return MCP_TOOL_SCHEMAS

    def tools(self) -> list[dict[str, Any]]:
        self._load_mcp_tools()
        return [*super().tools(), *self._mcp_schemas()]

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self._load_mcp_tools()
        args = dict(args)
        if name in _WAIT_DEFAULT_FALSE and "wait" not in args:
            args["wait"] = False
        return await super().execute(name, args)


# --------------------------------------------------------------------------- #
# session pool                                                                #
# --------------------------------------------------------------------------- #

# keyed by Streamable-HTTP `Mcp-Session-Id` (STDIO_SESSION_KEY for stdio) —
# keeps the project binding and spawned-job tracking alive across requests.
# Bounded: abandoned transport ids would otherwise grow the pool forever;
# the cap evicts the least-recently-used session (dicts keep insert order).
SESSIONS: dict[str, McpToolSession] = {}
_SESSION_TOUCHED: dict[str, float] = {}
_MAX_SESSIONS = 32
# idle TTL: an abandoned transport that never fires drop() must not hold its
# arq pool + sidecar clients forever; eviction loses the session binding, not
# any queued work (spawned jobs run server-side independent of the session)
_SESSION_TTL_S = 3600.0


async def get_or_create(key: str = STDIO_SESSION_KEY) -> McpToolSession:
    """Return the pooled session for this transport key, creating + warming
    it on first use (sidecar probes + a fresh runtime-settings snapshot)."""
    now = time.monotonic()
    for k in [k for k, ts in _SESSION_TOUCHED.items() if now - ts > _SESSION_TTL_S]:
        _SESSION_TOUCHED.pop(k, None)
        stale = SESSIONS.pop(k, None)
        if stale is not None:
            await stale.aclose()
    session = SESSIONS.pop(key, None)
    _SESSION_TOUCHED.pop(key, None)
    if session is None:
        from kiara.runtime_config import runtime_from_db_or_default

        session = McpToolSession(await runtime_from_db_or_default())
        await session.warm()
    SESSIONS[key] = session  # reinsert at the end — LRU touch
    _SESSION_TOUCHED[key] = now
    while len(SESSIONS) > _MAX_SESSIONS:
        oldest = next(iter(SESSIONS))
        evicted = SESSIONS.pop(oldest)
        _SESSION_TOUCHED.pop(oldest, None)
        await evicted.aclose()
    return session


def drop(key: str) -> None:
    """Forget a pooled session — called when its transport connection closes.

    Runs as a sync exit-stack callback: it schedules the async close on the
    running loop. With no loop (interpreter teardown) the pool's fds die with
    the process anyway — nothing to leak."""
    session = SESSIONS.pop(key, None)
    _SESSION_TOUCHED.pop(key, None)
    if session is None:
        return
    with contextlib.suppress(RuntimeError):
        asyncio.get_running_loop().create_task(session.aclose())
