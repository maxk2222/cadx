"""Bearer-token auth middleware (pure ASGI).

Pure ASGI rather than BaseHTTPMiddleware on purpose: the latter buffers
response bodies and breaks SSE streaming (`/v1/chat/completions` relays
llama.cpp/mlx-lm chunks verbatim). This middleware only inspects request
headers and either passes the scope through or answers 401 itself.

Exempt paths: `/v1/health` (load balancers/monitoring shouldn't need a
key) and every OPTIONS preflight (CORS). When no api keys are configured
the middleware is a no-op — loopback keeps its open-by-default behaviour;
the create_app boot guard refuses non-loopback binds without keys.
"""

from __future__ import annotations

import json
from typing import Any

_UNAUTHORIZED = json.dumps(
    {
        "error": {
            "message": "unauthorized — send 'Authorization: Bearer <key>'",
            "type": "authentication_error",
            "code": "invalid_api_key",
        }
    }
).encode()

_EXEMPT_PATHS = frozenset({"/v1/health"})


class BearerAuthMiddleware:
    def __init__(self, app: Any, api_keys: tuple[str, ...]) -> None:
        self.app = app
        self._tokens = {k.encode() for k in api_keys if k}

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not self._tokens:
            await self.app(scope, receive, send)
            return
        if scope.get("method") == "OPTIONS" or scope.get("path") in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        auth = b""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                auth = value.strip()
                break
        # the scheme is case-insensitive ("Bearer"/"bearer"); the token
        # itself must match exactly — case-folding it would weaken the key
        if auth[:7].lower() == b"bearer " and auth[7:] in self._tokens:
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_UNAUTHORIZED)).encode()),
                    (b"www-authenticate", b'Bearer realm="meshserver"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED})
