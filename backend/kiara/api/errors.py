"""
C3 Slice 8-B (F-049): an HTTP error that also carries a code the web translates. `detail` keeps the text
the route returned before, so every reader of `detail`, the headset included, sees no change. Only the
routes that raise this class get the two extra fields; every other error keeps its `{"detail": ...}` body.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class CodedHTTPException(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        detail: Any,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code
        self.params = params


async def coded_http_exception_handler(request: Request, exc: CodedHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code, "params": exc.params or {}},
        headers=exc.headers,
    )
