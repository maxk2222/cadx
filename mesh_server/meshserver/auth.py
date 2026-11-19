"""Bearer-token auth middleware (pure ASGI).

Pure ASGI rather than BaseHTTPMiddleware on purpose: the latter buffers
response bodies and breaks SSE streaming (`/v1/chat/completions` relays
llama.cpp/mlx-lm chunks verbatim). This middleware only inspects request
headers and either passes the scope through or answers 401 itself.

Exempt paths: `/v1/health` (load balancers/monitoring shouldn't need a
key) and every OPTIONS preflight (CORS). When no api keys are configured
the middleware is a no-op — loopback keeps its open-by-default behaviour;
the create_app boot guard refuses non-loopback binds without keys.

Also lives here: `request_priority` — the internal/external caller
classification every scheduling lane shares (job queue + llm queue).
"""

from __future__ import annotations

import json
from typing import Any

_INTERNAL_HEADER = "x-mesh-internal"
_PRIORITY_HEADER = "x-mesh-priority"


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testclient"})


def _is_loopback(request: Any) -> bool:
    host = getattr(getattr(request, "client", None), "host", None)
    if host is None:
        return True
    return host.strip().lower() in _LOOPBACK_HOSTS


def is_internal_caller(request: Any) -> bool:
    """True when the request comes from the trusted local side — either the
    CADX backend (it stamps `X-Mesh-Internal: 1` on every sidecar call) or a
    loopback peer (local tooling/tests). External = authenticated LAN
    clients. The header only selects the configured *class* — it cannot
    escalate, so external callers gain nothing by forging it."""
    if request is None:
        return True
    if (request.headers.get(_INTERNAL_HEADER) or "").strip() == "1":
        return True
    return _is_loopback(request)


def request_priority(request: Any, cfg: Any, *, internal_key: str, external_key: str) -> int:
    """Scheduling priority for a request — classified globally, not
    self-declared: internal callers read `internal_key`, external read
    `external_key` (both runtime-mutable via /v1/config). A raw
    `X-Mesh-Priority` override is honored ONLY from a loopback peer —
    the header value is operator-defined code tooling, and an external
    client must never raise its own priority (it could forge
    X-Mesh-Internal to reach this branch otherwise)."""
    effective = getattr(cfg, "effective", None)
    internal = is_internal_caller(request)
    if internal and request is not None and _is_loopback(request):
        raw = (request.headers.get(_PRIORITY_HEADER) or "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
    key = internal_key if internal else external_key
    if callable(effective):
        try:
            return int(effective(key) or 0)
        except Exception:
            return 0
    return int(getattr(cfg, key, 0) or 0)

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
