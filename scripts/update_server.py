#!/usr/bin/env python3
# CADX updater daemon — tiny stdlib-only HTTP wrapper around scripts/update.sh.
# Runs on the HOST (not in docker) so it survives the stack restart it drives.
#
# Endpoints (Bearer token on everything except /v1/health):
#   GET  /v1/health          -> {"ok": true}
#   GET  /v1/version         -> {"sha","branch","dirty","ts"}
#   GET  /v1/update/check    -> cached run/update-check.json; spawns a
#                               background check when missing or >6h stale
#   POST /v1/update/check    -> spawn `update.sh --check [--ref]` (202)
#   POST /v1/update          -> spawn `update.sh --yes [--ref] [--force]` (202/409)
#   GET  /v1/update/status   -> run/update-state.json + log tail
#   POST /v1/update/rollback -> spawn `update.sh --rollback --yes` (202/409)
#
# Env: UPDATE_HOST (default 127.0.0.1) · UPDATE_PORT (default 8766)
#      CADX_UPDATE_TOKEN or run/updater.token · CADX_UPDATE_SCRIPT (test hook)
import contextlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run"
LOGS = ROOT / "logs"
STATE_FILE = RUN / "update-state.json"
CHECK_FILE = RUN / "update-check.json"
TOKEN_FILE = RUN / "updater.token"
UPDATE_SCRIPT = os.environ.get("CADX_UPDATE_SCRIPT") or str(ROOT / "scripts" / "update.sh")
CHECK_MAX_AGE_S = 6 * 3600

HOST = os.environ.get("UPDATE_HOST") or "127.0.0.1"
PORT = int(os.environ.get("UPDATE_PORT") or 8766)

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_self_mtime = Path(__file__).stat().st_mtime

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,79}$")
_RUNNING_STATES = {"checking", "running", "rolling-back"}


def _load_token() -> str:
    tok = os.environ.get("CADX_UPDATE_TOKEN", "").strip()
    if tok:
        return tok
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    # first boot without start.sh: create one so we never run unauthenticated
    RUN.mkdir(exist_ok=True)
    tok = secrets.token_hex(24)
    TOKEN_FILE.write_text(tok)
    TOKEN_FILE.chmod(0o600)
    return tok


TOKEN = _load_token()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _version_info() -> dict:
    return {
        "sha": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": len(_git("status", "--porcelain").splitlines()),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"state": "idle", "phase": None}


def _log_tail(path: str | None, lines: int = 40) -> list[str]:
    if not path:
        return []
    f = Path(path) if Path(path).is_absolute() else ROOT / path
    try:
        data = f.read_bytes()[-64 * 1024 :].decode("utf-8", "replace")
    except OSError:
        return []
    rows = [r.strip() for r in data.replace("\r", "\n").splitlines()]
    return [r for r in rows if r][-lines:]


def _seed_state(argv: list[str]) -> None:
    """Mark the run as started before update.sh writes its own state — closes
    the idle-looking gap right after a 202. Checks never seed: they write
    update-check.json only and must not fake a "checking" update state."""
    if "--check" in argv:
        return
    state = "rolling-back" if "--rollback" in argv else "running"
    with contextlib.suppress(OSError):
        STATE_FILE.write_text(
            json.dumps(
                {
                    "state": state,
                    "phase": "starting",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
        )


def _spawn(argv: list[str]) -> bool:
    """Spawn the update script (at most one at a time). Returns False when busy."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return False
        if _state().get("state") in _RUNNING_STATES:
            return False
        _seed_state(argv)
        LOGS.mkdir(exist_ok=True)
        with (LOGS / "updater-launch.log").open("ab") as launch:
            launch.write(("\n$ " + " ".join(argv) + "\n").encode())
            launch.flush()
            _proc = subprocess.Popen(
                ["bash", UPDATE_SCRIPT, *argv],
                cwd=ROOT,
                stdout=launch,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CADX_UPDATE_NONINTERACTIVE": "1"},
            )
        return True


def _check_stale() -> bool:
    try:
        return time.time() - CHECK_FILE.stat().st_mtime > CHECK_MAX_AGE_S
    except OSError:
        return True


def _maybe_self_reexec() -> None:
    """After a completed update: re-exec if our own source changed on disk."""
    try:
        if _state().get("state") != "done":
            return
        if Path(__file__).stat().st_mtime <= _self_mtime:
            return
    except OSError:
        return
    sys.stderr.write("[updater] self source changed — re-executing\n")
    os.execv(sys.executable, [sys.executable, str(Path(__file__))])


class Handler(BaseHTTPRequestHandler):
    server_version = "cadx-updater/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quiet — keep stderr clean
        pass

    # ---------- helpers ----------
    def _auth(self) -> bool:
        return self.headers.get("Authorization", "") == f"Bearer {TOKEN}"

    def _json(self, code: int, doc) -> None:
        body = json.dumps(doc).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if not (0 < n < 1 << 20):
            return {}
        try:
            return json.loads(self.rfile.read(n)) or {}
        except Exception:
            return {}

    # ---------- routes ----------
    def do_GET(self) -> None:
        if self.path == "/v1/health":
            return self._json(200, {"ok": True})
        if not self._auth():
            return self._json(401, {"error": "unauthorized"})
        if self.path == "/v1/version":
            return self._json(200, _version_info())
        if self.path == "/v1/update/check":
            doc = {}
            try:
                doc = json.loads(CHECK_FILE.read_text())
            except Exception:
                doc = {"behind": None, "stale": True}
            if _check_stale():
                _spawn(["--check"])  # refresh in background; may 409 quietly
                doc.setdefault("refreshing", True)
            return self._json(200, doc)
        if self.path == "/v1/update/status":
            doc = _state()
            doc["log_tail"] = _log_tail(doc.get("log"))
            doc["version"] = _version_info()
            return self._json(200, doc)
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._auth():
            return self._json(401, {"error": "unauthorized"})
        body = self._body()
        ref = str(body.get("ref") or "")
        if ref and not _REF_RE.match(ref):
            return self._json(400, {"error": "invalid ref"})
        if self.path == "/v1/update/check":
            argv = ["--check"] + (["--ref", ref] if ref else [])
            started = _spawn(argv)
            return self._json(202 if started else 409, {"started": started})
        if self.path == "/v1/update":
            argv = ["--yes"]
            if ref:
                argv += ["--ref", ref]
            if body.get("force"):
                argv.append("--force")
            if not _spawn(argv):
                return self._json(409, {"error": "update already running"})
            return self._json(202, {"started": True})
        if self.path == "/v1/update/rollback":
            if not body.get("confirm"):
                return self._json(400, {"error": "confirm:true required"})
            if not _spawn(["--rollback", "--yes", "--force"]):
                return self._json(409, {"error": "update already running"})
            return self._json(202, {"started": True})
        return self._json(404, {"error": "not found"})


def _reset_stale_state() -> None:
    """A fresh daemon can never have spawned a run — a leftover running state
    means the previous daemon died mid-update. Mark it failed so updates stay
    possible; the orphaned update.sh (if still alive) keeps writing anyway."""
    try:
        doc = json.loads(STATE_FILE.read_text())
    except Exception:
        return
    if doc.get("state") in _RUNNING_STATES:
        doc["state"] = "failed"
        doc["error"] = "updater daemon restarted mid-run"
        doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with contextlib.suppress(OSError):
            STATE_FILE.write_text(json.dumps(doc))


def main() -> None:
    RUN.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    _reset_stale_state()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    sys.stderr.write(f"[updater] listening on {HOST}:{PORT} (root={ROOT})\n")
    # poll occasionally: re-exec ourselves once a completed update replaced us
    threading.Thread(target=_watchdog, daemon=True).start()
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()


def _watchdog() -> None:
    while True:
        time.sleep(15)
        if _proc is None or _proc.poll() is not None:
            _maybe_self_reexec()


if __name__ == "__main__":
    main()
