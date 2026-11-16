"""Blender MCP socket client (guide blender-mode bridge).

Talks plain JSON over TCP to the BlenderMCP addon listening inside a
running Blender instance: request ``{"type": ..., "params": {...}}`` ->
single ``{"status": "success"|"error", "result"|"message": ...}`` reply.

Config (env):
- ``CADX_BLENDER_HOST`` — explicit override. Empty = auto-probe
  ``host.docker.internal`` (container) then ``localhost``.
- ``CADX_BLENDER_PORT`` — default ``9876``.
- ``CADX_DATA_HOST_ROOT`` — absolute host-side path of the data dir the
  worker mounts at ``/data``. Blender writes files only on the host, so
  every export/screenshot path the tools hand to Blender must be
  translated worker->host and back through this root. Empty = the file
  bridge is unavailable (tools report it honestly).

The client itself never fakes success: unreachable host, transport
errors and ``status:"error"`` replies all surface as ``BlenderError``
with the server's own message attached.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

from cadx.logging import get_logger

log = get_logger(__name__)

DEFAULT_PORT = 9876
_CONNECT_TIMEOUT_S = 4.0
_RECV_CHUNK = 65536


class BlenderError(Exception):
    """Transport failure or server-side ``status:"error"`` reply."""


def _in_container() -> bool:
    return Path("/.dockerenv").exists()


def _candidate_hosts() -> list[str]:
    env_host = (os.environ.get("CADX_BLENDER_HOST") or "").strip()
    if env_host:
        return [env_host]
    # the worker usually runs in docker; a native dev-worker reaches the
    # same Blender via localhost. Probe order: container bridge first
    # when we are in a container, otherwise localhost first.
    return (
        ["host.docker.internal", "localhost", "127.0.0.1"]
        if _in_container()
        else ["localhost", "127.0.0.1", "host.docker.internal"]
    )


def blender_port() -> int:
    try:
        return int(os.environ.get("CADX_BLENDER_PORT") or DEFAULT_PORT)
    except ValueError:
        return DEFAULT_PORT


def data_host_root() -> Path | None:
    """Host-side absolute path of the worker's data root, if configured."""
    raw = (os.environ.get("CADX_DATA_HOST_ROOT") or "").strip()
    return Path(raw).resolve() if raw else None


def to_host_path(worker_path: Path) -> str:
    """Map a path inside the worker data root onto the host path Blender sees.

    Raises BlenderError when no host root is configured or the path lies
    outside the data dir — never returns a guessed path Blender could not
    actually reach.
    """
    root = data_host_root()
    if root is None:
        raise BlenderError(
            "CADX_DATA_HOST_ROOT is not set — the worker cannot translate "
            "its /data paths into the host paths Blender writes to. Set it "
            "to the host-side absolute path of the data dir."
        )
    from cadx import storage

    resolved = Path(worker_path).resolve()
    try:
        rel = resolved.relative_to(storage.data_root().resolve())
    except ValueError as exc:
        raise BlenderError(
            f"path {resolved} is outside the data root — Blender can only "
            "exchange files inside the mounted data dir"
        ) from exc
    return str(root / rel)


def to_worker_path(host_path: str | Path) -> Path:
    """Inverse of to_host_path: host path -> path this worker can read."""
    root = data_host_root()
    if root is None:
        raise BlenderError("CADX_DATA_HOST_ROOT is not set")
    resolved = Path(host_path).resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise BlenderError(
            f"host path {resolved} is outside CADX_DATA_HOST_ROOT ({root})"
        ) from exc
    from cadx import storage

    return storage.data_root() / rel


class BlenderClient:
    """One BlenderMCP socket endpoint. Commands open a fresh connection —
    the addon replies with one JSON blob per command and a stale stream
    after a timeout would poison the next command otherwise."""

    def __init__(self, host: str, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port

    def command(
        self, cmd_type: str, params: dict[str, Any] | None = None, *, timeout_s: float = 30.0
    ) -> Any:
        payload = json.dumps({"type": cmd_type, "params": params or {}}).encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=_CONNECT_TIMEOUT_S) as sock:
                sock.settimeout(timeout_s)
                sock.sendall(payload)
                return self._recv_result(sock)
        except BlenderError:
            raise
        except (TimeoutError, OSError) as exc:
            raise BlenderError(
                f"Blender MCP unreachable at {self.host}:{self.port} "
                f"({type(exc).__name__}: {exc}) — open Blender with the "
                "BlenderMCP addon enabled"
            ) from exc

    @staticmethod
    def _recv_result(sock: socket.socket) -> Any:
        buffer = b""
        while True:
            chunk = sock.recv(_RECV_CHUNK)
            if not chunk:
                raise BlenderError("Blender MCP closed the connection without a reply")
            buffer += chunk
            try:
                reply = json.loads(buffer.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # partial blob — keep receiving
            if reply.get("status") == "success":
                return reply.get("result")
            raise BlenderError(
                str(reply.get("message") or reply.get("error") or f"Blender error: {reply}")[:2000]
            )

    def ping(self) -> bool:
        return bool(self.command("ping", timeout_s=5.0).get("pong"))

    def execute_code(self, code: str, *, timeout_s: float = 120.0) -> str:
        result = self.command("execute_code", {"code": code}, timeout_s=timeout_s)
        if isinstance(result, dict):
            return str(result.get("result") or "")
        return str(result or "")

    def get_scene_info(self) -> dict[str, Any]:
        result = self.command("get_scene_info", timeout_s=15.0)
        return result if isinstance(result, dict) else {}

    def export_scene(
        self,
        host_filepath: str,
        *,
        fmt: str = "glb",
        object_names: list[str] | None = None,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        result = self.command(
            "export_scene",
            {
                "filepath": host_filepath,
                "format": fmt,
                "object_names": object_names or [],
            },
            timeout_s=timeout_s,
        )
        return result if isinstance(result, dict) else {"result": result}

    def viewport_screenshot(
        self, host_filepath: str, *, max_size: int = 800, timeout_s: float = 60.0
    ) -> dict[str, Any]:
        result = self.command(
            "get_viewport_screenshot",
            {"filepath": host_filepath, "max_size": max_size},
            timeout_s=timeout_s,
        )
        return result if isinstance(result, dict) else {"result": result}


_lock = threading.Lock()
_client: BlenderClient | None = None


def probe_client() -> BlenderClient | None:
    """First reachable endpoint wins; the choice is cached for the worker's
    lifetime (Blender does not migrate hosts mid-run)."""
    global _client
    with _lock:
        if _client is not None:
            return _client
        for host in _candidate_hosts():
            candidate = BlenderClient(host, blender_port())
            try:
                if candidate.ping():
                    log.info("guide.blender_connected", host=host, port=candidate.port)
                    _client = candidate
                    return candidate
            except BlenderError:
                continue
        return None


def get_client() -> BlenderClient:
    """Connected client or an honest error describing the remediation."""
    client = probe_client()
    if client is None:
        raise BlenderError(
            f"Blender MCP not reachable on port {blender_port()} — start "
            "Blender with the BlenderMCP addon enabled (the socket server "
            "auto-starts when a .blend file loads)"
        )
    return client


def reset_client() -> None:
    """Drop the cached endpoint — tests and post-suspend re-probing."""
    global _client
    with _lock:
        _client = None
