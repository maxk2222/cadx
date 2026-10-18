"""Streamable-HTTP transport for the CADX MCP server.

`session_manager().handle_request` implements the full Streamable-HTTP spec
(POST for JSON-RPC, GET for the SSE stream, DELETE for session teardown).
`mcp_lifespan` enters the manager's task group from the host app's lifespan;
`MCPAuthMiddleware` gates the endpoint behind `Authorization: Bearer <token>`
when `settings.mcp_token` (env `CADX_MCP_TOKEN`) is configured — empty token
means open access (local/dev default).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from cadx.logging import get_logger

log = get_logger(__name__)

Scope = dict[str, Any]
Receive = Any
Send = Any
ASGIApp = Any


class MCPDispatchMiddleware:
    """Pure-ASGI prefix dispatcher: forwards `{prefix}` and `{prefix}/…`
    requests to the MCP sub-application. Used instead of Starlette's
    `Mount`, whose regex only matches `{prefix}/` and 307-redirects bare
    `{prefix}` — a hop MCP clients are not required to follow."""

    def __init__(self, app: ASGIApp, prefix: str = "/mcp", subapp: ASGIApp | None = None) -> None:
        self._app = app
        self._prefix = prefix
        self._subapp = subapp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http":
            path: str = scope.get("path") or ""
            if path == self._prefix or path.startswith(self._prefix + "/"):
                if self._subapp is None:
                    self._subapp = mcp_asgi_app()
                await self._subapp(scope, receive, send)
                return
        await self._app(scope, receive, send)


class MCPAuthMiddleware:
    """Pure-ASGI bearer-auth wrapper — 401 JSON when the token is configured
    and the Authorization header is missing or wrong."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}" if token else ""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not self._expected:
            await self._app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1").strip()
        if auth != self._expected:
            await _reject(send)
            return
        await self._app(scope, receive, send)


async def _reject(send: Send) -> None:
    body = (
        b'{"ok":false,"error":"unauthorized - '
        b'expected Authorization: Bearer <CADX_MCP_TOKEN>"}'
    )
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def mcp_asgi_app() -> ASGIApp:
    """The mountable ASGI app: bearer auth around the Streamable-HTTP
    session manager's request handler."""
    from cadx.config import get_settings
    from cadx.mcpserver.server import session_manager

    manager = session_manager()

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await manager.handle_request(scope, receive, send)

    return MCPAuthMiddleware(app, get_settings().mcp_token)


@asynccontextmanager
async def mcp_lifespan(app: Any) -> AsyncIterator[None]:
    """Run the session manager's task group for the host app's lifetime —
    enter this from the FastAPI lifespan before serving any /mcp request."""
    from cadx.mcpserver.server import session_manager

    async with session_manager().run():
        yield
