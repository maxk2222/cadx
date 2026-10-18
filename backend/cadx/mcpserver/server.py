"""MCP server core — exposes the cadx tool surface over the Model Context Protocol.

Low-level `mcp.server.Server` (no FastMCP): the tool list is mapped 1:1 from
the session's `_fn()` schemas and `tools/call` runs `session.execute` —
a failing tool is a result, never an exception, so the wire always gets a
CallToolResult (`isError` on `ok is False`, ImageContent for `_images`).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mcp.types as types
from mcp.server import Server

from cadx.logging import get_logger
from cadx.mcpserver.session import STDIO_SESSION_KEY, McpToolSession, drop, get_or_create

if TYPE_CHECKING:
    from mcp.server.lowlevel.server import ServerRequestContext
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# session resolution                                                          #
# --------------------------------------------------------------------------- #


async def _session_for(ctx: ServerRequestContext[Any]) -> McpToolSession:
    """Resolve the pooled McpToolSession for this request — keyed by the
    transport session id (Streamable-HTTP `Mcp-Session-Id`, set on the
    connection by serve_loop), STDIO_SESSION_KEY for stdio."""
    conn = getattr(getattr(ctx, "session", None), "_connection", None)
    key = getattr(conn, "session_id", None) or STDIO_SESSION_KEY
    session = await get_or_create(key)
    # drop the pooled session when the transport connection unwinds
    if conn is not None and not conn.state.get("cadx_drop_registered"):
        conn.state["cadx_drop_registered"] = True
        conn.exit_stack.callback(drop, key)
    return session


# --------------------------------------------------------------------------- #
# schema + result mapping                                                     #
# --------------------------------------------------------------------------- #


def _tool_schema(spec: dict[str, Any]) -> types.Tool:
    """`_fn()` format -> mcp.types.Tool (1:1)."""
    fn = spec.get("function") or {}
    return types.Tool(
        name=str(fn.get("name") or ""),
        description=str(fn.get("description") or ""),
        inputSchema=fn.get("parameters") or {"type": "object", "properties": {}},
    )


def _image_content(path: str) -> types.ImageContent | None:
    try:
        p = Path(path)
        data = base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception as e:
        log.warning("mcp.image_read_failed", path=path, error=str(e)[:200])
        return None
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return types.ImageContent(type="image", data=data, mimeType=mime)


def _result_content(result: dict[str, Any]) -> types.CallToolResult:
    result = dict(result)
    images = result.pop("_images", None) or []
    content: list[types.TextContent | types.ImageContent] = [
        types.TextContent(type="text", text=json.dumps(result, default=str))
    ]
    for path in images:
        img = _image_content(str(path))
        if img is not None:
            content.append(img)
    return types.CallToolResult(
        content=content,
        isError=result.get("ok") is False,
        structuredContent=result if isinstance(result, dict) else None,
    )


# --------------------------------------------------------------------------- #
# request handlers                                                            #
# --------------------------------------------------------------------------- #


async def _on_list_tools(
    ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    session = await _session_for(ctx)
    return types.ListToolsResult(tools=[_tool_schema(t) for t in session.tools()])


async def _on_call_tool(
    ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
) -> types.CallToolResult:
    try:
        session = await _session_for(ctx)
        result = await session.execute(params.name, dict(params.arguments or {}))
    except Exception as e:
        log.warning("mcp.call_failed", tool=params.name, error=str(e)[:300])
        payload = {"ok": False, "error": f"tool '{params.name}' failed: {e}"}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))],
            isError=True,
        )
    try:
        return _result_content(result)
    except Exception as e:
        log.warning("mcp.result_encode_failed", tool=params.name, error=str(e)[:300])
        payload = {"ok": False, "error": f"tool '{params.name}' result encoding failed: {e}"}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))],
            isError=True,
        )


# --------------------------------------------------------------------------- #
# server + transports                                                         #
# --------------------------------------------------------------------------- #

server: Server[Any] = Server("cadx", on_list_tools=_on_list_tools, on_call_tool=_on_call_tool)

_session_manager: StreamableHTTPSessionManager | None = None


def session_manager() -> StreamableHTTPSessionManager:
    """The shared Streamable-HTTP session manager — enter `.run()` from the
    host app's lifespan, serve requests via `.handle_request`."""
    global _session_manager
    if _session_manager is None:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        _session_manager = StreamableHTTPSessionManager(server)
    return _session_manager


async def run_stdio() -> None:
    """Serve MCP over stdin/stdout (`cadx mcp` default transport)."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
