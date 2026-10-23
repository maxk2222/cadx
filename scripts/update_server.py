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
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run"
LOGS = ROOT / "logs"
STATE_FILE = RUN / "update-state.json"
CHECK_FILE = RUN / "update-check.json"
TOKEN_FILE = RUN / "updater.token"
MODE_FILE = RUN / "mode"
MESH_ENV_FILE = RUN / "mesh-server.env"
WATCHDOG_FILE = RUN / "watchdog.json"
UPDATE_SCRIPT = os.environ.get("CADX_UPDATE_SCRIPT") or str(ROOT / "scripts" / "update.sh")
CHECK_MAX_AGE_S = 6 * 3600

HOST = os.environ.get("UPDATE_HOST") or "127.0.0.1"
PORT = int(os.environ.get("UPDATE_PORT") or 8766)

# supervisor knobs — generous: GPU builds and docker pulls legitimately take
# a long time; progress is measured by LOG mtime, not phase boundaries
WATCHDOG_INTERVAL_S = int(os.environ.get("CADX_WATCHDOG_INTERVAL") or 15)
UPDATE_STALL_S = float(os.environ.get("CADX_UPDATE_STALL_S") or 1800)  # no log progress
UPDATE_MAX_S = float(os.environ.get("CADX_UPDATE_MAX_S") or 3 * 3600)  # absolute cap
CHECK_STALL_S = float(os.environ.get("CADX_CHECK_STALL_S") or 900)  # --check run cap
MESH_HEALTH_FAILS = 3  # consecutive /v1/health misses before respawn
COMPOSE_BAD_TICKS = 3  # consecutive unhealthy/exited!=0 ticks before healing
HB_STALE_S = 120.0  # worker heartbeat age that counts as "wedged"
FLAP_WINDOW_S = 600.0
FLAP_MAX = 5  # respawns per service per window, then suppress

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_proc_started = 0.0
_proc_check = False
_self_mtime = Path(__file__).stat().st_mtime

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,79}$")
_RUNNING_STATES = {"checking", "running", "rolling-back"}

# supervisor state
_wd_actions: list[dict] = []
_wd_fail_counts: dict[str, int] = {}
_wd_flap: dict[str, list[float]] = {}
_wd_suppressed = False


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
    global _proc, _proc_started, _proc_check
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
                # own process group — the watchdog's killpg must not take the
                # daemon down with the stalled update
                start_new_session=True,
            )
            _proc_started = time.time()
            _proc_check = "--check" in argv
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
        if self.path == "/v1/stack/status":
            return self._json(200, _stack_status())
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
        if self.path.startswith("/v1/stack/restart/"):
            svc = self.path.rsplit("/", 1)[-1]
            ok, why = _stack_restart(svc)
            return self._json(202 if ok else 400, {"ok": ok, "detail": why})
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


# ---------- supervisor ----------

def _wd_note(action: str, detail: str = "") -> None:
    _wd_actions.append({"ts": int(time.time()), "action": action, "detail": detail[:400]})
    del _wd_actions[:-50]
    sys.stderr.write(f"[updater] watchdog {action}: {detail[:300]}\n")


def _wd_write() -> None:
    doc = {
        "ts": int(time.time()),
        "suppressed": _wd_suppressed,
        "fails": dict(_wd_fail_counts),
        "actions": _wd_actions[-20:],
    }
    try:
        tmp = WATCHDOG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc))
        tmp.replace(WATCHDOG_FILE)
    except OSError:
        pass


def _wd_flap_ok(name: str) -> bool:
    """False once a service hit the respawn limit inside the flap window."""
    now = time.time()
    hits = [t for t in _wd_flap.get(name, []) if now - t < FLAP_WINDOW_S]
    _wd_flap[name] = hits
    if len(hits) >= FLAP_MAX:
        _wd_note("flap-suppressed", f"{name}: {len(hits)} respawns in {FLAP_WINDOW_S:.0f}s")
        return False
    hits.append(now)
    return True


def _run_capture(argv: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a run/*.env file (KEY=value lines written by start.sh)."""
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def _patch_state(**kw) -> None:
    """Update fields of update-state.json in place (same convention as
    _reset_stale_state)."""
    try:
        doc = json.loads(STATE_FILE.read_text())
    except Exception:
        return
    doc.update(kw)
    with contextlib.suppress(OSError):
        STATE_FILE.write_text(json.dumps(doc))


def _stall_check_update() -> None:
    """Kill a hung update process. Stall = no output in the update log for
    UPDATE_STALL_S — state-mtime alone false-positives on long builds since
    run_step keeps writing the log between phase transitions."""
    if _proc is None or _proc.poll() is not None:
        return
    state = _state()
    logf = state.get("log") or str(LOGS / "update.log")
    try:
        log_mtime = Path(logf).stat().st_mtime
    except OSError:
        log_mtime = 0.0
    try:
        state_mtime = STATE_FILE.stat().st_mtime
    except OSError:
        state_mtime = 0.0
    # _proc_started must be in the max: a stale update.log (days old) belongs
    # to the LAST run — a freshly spawned --check/update has no fresh files
    # yet and would otherwise be killed instantly as "idle for days"
    idle_for = time.time() - max(state_mtime, log_mtime, _proc_started)
    ran_for = time.time() - _proc_started
    budget = CHECK_STALL_S if _proc_check else UPDATE_STALL_S
    if idle_for <= budget and ran_for <= UPDATE_MAX_S:
        return
    reason = (
        f"no output for {idle_for:.0f}s"
        if idle_for > budget
        else f"absolute cap {UPDATE_MAX_S:.0f}s hit"
    )
    _wd_note("kill-stalled", f"{state.get('state')}:{reason} pgid={_proc.pid}")
    with contextlib.suppress(Exception):
        os.killpg(_proc.pid, signal.SIGTERM)
    try:
        _proc.wait(timeout=15)
    except Exception:
        with contextlib.suppress(Exception):
            os.killpg(_proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            _proc.wait(timeout=10)
    if state.get("state") in _RUNNING_STATES:
        lf = Path(logf)
        with contextlib.suppress(OSError):
            with lf.open("a") as f:
                f.write(f"\n[updater] update stalled ({reason}) — killed by watchdog\n")
        _patch_state(
            state="failed",
            step="stalled",
            error=f"stalled: {reason}",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


def _api_health_url() -> str:
    mode = MODE_FILE.read_text().strip() if MODE_FILE.exists() else "docker"
    port = 8080 if mode == "docker" else 5173
    return f"http://127.0.0.1:{port}/api/health"


def _mesh_state() -> dict:
    """Supervised mesh-server facts: supervised flag, pid liveness, health."""
    env = _load_env_file(MESH_ENV_FILE)
    pid = 0
    try:
        pid = int((RUN / "mesh-server.pid").read_text().strip())
    except (OSError, ValueError):
        pass
    port = env.get("MESH_PORT") or "8765"
    code, _ = _http_get(f"http://127.0.0.1:{port}/v1/health", timeout=4)
    return {
        "supervised": MESH_ENV_FILE.exists(),
        "pid": pid,
        "alive": pid > 0 and _pid_alive(pid),
        "healthy": code == 200,
        "port": port,
        "env": env,
    }


def _mesh_respawn(env: dict[str, str]) -> None:
    """Start meshserver from the recorded supervision env."""
    venv_bin = ROOT / "mesh_server" / ".venv" / "bin" / "meshserver"
    if not venv_bin.exists():
        _wd_note("mesh-respawn-skip", "mesh_server/.venv missing")
        return
    LOGS.mkdir(exist_ok=True)
    try:
        with (LOGS / "mesh-server.log").open("ab") as out:
            proc = subprocess.Popen(
                [str(venv_bin), "serve"],
                cwd=ROOT,
                stdout=out,
                stderr=subprocess.STDOUT,
                env={**os.environ, **env},
                start_new_session=True,
            )
        (RUN / "mesh-server.pid").write_text(f"{proc.pid}\n")
        _wd_note("mesh-respawn", f"pid={proc.pid}")
    except OSError as e:
        _wd_note("mesh-respawn-fail", str(e))


def _supervise_mesh() -> None:
    """Respawn the native mesh server — only when it was started under CADX
    supervision (start.sh wrote run/mesh-server.env). A foreign process on
    the port without the env file is never adopted."""
    st = _mesh_state()
    if not st["supervised"]:
        _wd_fail_counts.pop("mesh-server", None)
        return
    if st["alive"] and st["healthy"]:
        _wd_fail_counts["mesh-server"] = 0
        return
    _wd_fail_counts["mesh-server"] = _wd_fail_counts.get("mesh-server", 0) + 1
    if _wd_fail_counts["mesh-server"] < MESH_HEALTH_FAILS:
        return
    if not _wd_flap_ok("mesh-server"):
        return
    if st["alive"]:
        with contextlib.suppress(Exception):
            os.kill(st["pid"], signal.SIGKILL)
        time.sleep(0.5)
    _mesh_respawn(st["env"])
    # startup grace: a respawned server takes seconds to bind+answer — a
    # negative counter delays the next respawn attempt by that many ticks
    _wd_fail_counts["mesh-server"] = -4


def _compose_services() -> list[dict] | None:
    """`docker compose ps -a --format json` — list of service dicts, [] when
    the stack is fully down, None on error. `-a` is required: without it
    compose omits exited containers entirely and a crashed service is
    invisible to the watchdog."""
    res = _run_capture(["docker", "compose", "ps", "-a", "--format", "json"], timeout=15)
    if res is None or res.returncode != 0:
        _wd_note("compose-ps-fail", (res.stderr if res else "timeout")[:200])
        return None
    raw = res.stdout.strip()
    if not raw:
        return []
    try:
        svcs = json.loads(raw)
        return [svcs] if isinstance(svcs, dict) else svcs
    except json.JSONDecodeError:
        with contextlib.suppress(json.JSONDecodeError):
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return []


def _supervise_stack() -> None:
    """Compose-mode healing: crashed/unhealthy containers via `compose ps`,
    wedged workers via heartbeat ages in /api/health. Intentional exits
    (ExitCode 0) and a fully-down stack are never touched."""
    if MODE_FILE.exists() and MODE_FILE.read_text().strip() != "docker":
        return
    services = _compose_services()
    if services is None or not services:
        return  # error → next tick; empty → probably intentional, never resurrect
    for svc in services:
        name = svc.get("Service") or svc.get("Name") or ""
        if not name:
            continue
        state = str(svc.get("State", ""))
        health = str(svc.get("Health", ""))
        exit_code = svc.get("ExitCode", 0)
        if state == "exited" and exit_code in (0, 143, None):
            # 0 = clean exit; 143 = SIGTERM (docker/compose stop) — both mean
            # "someone asked for it", never resurrect. Crashes look like 1
            # (exception), 137 (SIGKILL/OOM), 139 (SIGSEGV) → healed below.
            continue
        bad = (state == "exited") or health == "unhealthy"
        key = f"compose:{name}"
        if not bad:
            _wd_fail_counts[key] = 0
            continue
        _wd_fail_counts[key] = _wd_fail_counts.get(key, 0) + 1
        if _wd_fail_counts[key] < COMPOSE_BAD_TICKS or not _wd_flap_ok(key):
            continue
        verb = ["up", "-d", name] if state == "exited" else ["restart", name]
        res = _run_capture(["docker", "compose", *verb], timeout=180)
        _wd_note(
            f"compose-heal-{name}",
            f"{' '.join(verb)} exit={res.returncode if res else 't/o'} "
            f"{(res.stderr or '')[:200] if res else ''}",
        )
        _wd_fail_counts[key] = 0
    # deep health: restart wedged workers (stale heartbeat) reported by the api
    code, body = _http_get(_api_health_url(), timeout=6)
    if code != 200:
        return  # api down — compose-level healthcheck/compose ps handles it
    try:
        workers = (json.loads(body) or {}).get("workers") or {}
    except json.JSONDecodeError:
        workers = {}
    for role, svc_name in (("agent", "worker"), ("cad", "cad-executor")):
        age = (workers.get(role) or {}).get("age_s")
        key = f"hb:{svc_name}"
        if age is not None and float(age) <= HB_STALE_S:
            _wd_fail_counts[key] = 0
            continue
        _wd_fail_counts[key] = _wd_fail_counts.get(key, 0) + 1
        if _wd_fail_counts[key] < COMPOSE_BAD_TICKS or not _wd_flap_ok(key):
            continue
        row = next((s for s in services if s.get("Service") == svc_name), None)
        if not row or row.get("State") != "running":
            _wd_fail_counts[key] = 0
            continue
        r = _run_capture(["docker", "compose", "restart", svc_name], timeout=120)
        _wd_note(
            f"compose-heal-{svc_name}",
            f"heartbeat age={age} restart={r.returncode if r else 't/o'}",
        )
        _wd_fail_counts[key] = 0


def _stack_status() -> dict:
    """Full supervised-stack snapshot for /v1/stack/status + status.sh."""
    mode = MODE_FILE.read_text().strip() if MODE_FILE.exists() else "docker"
    doc: dict = {
        "mode": mode,
        "suppressed": _wd_suppressed,
        "update_state": _state().get("state"),
        "mesh": None,
        "compose": None,
        "watchdog": {"fails": dict(_wd_fail_counts), "actions": _wd_actions[-10:]},
    }
    m = _mesh_state()
    doc["mesh"] = {k: m[k] for k in ("supervised", "pid", "alive", "healthy", "port")}
    if mode == "docker":
        doc["compose"] = _compose_services()
    code, body = _http_get(_api_health_url(), timeout=6)
    try:
        doc["api_health"] = json.loads(body) if code else None
    except json.JSONDecodeError:
        doc["api_health"] = None
    doc["api_health_status"] = code
    return doc


_HEALABLE = {"api", "worker", "cad-executor", "frontend"}


def _stack_restart(svc: str) -> tuple[bool, str]:
    """Manual service restart via /v1/stack/restart/<svc>."""
    if svc == "mesh-server":
        st = _mesh_state()
        if not st["supervised"]:
            return False, "mesh-server is not supervised (no run/mesh-server.env)"
        if st["alive"]:
            with contextlib.suppress(Exception):
                os.kill(st["pid"], signal.SIGTERM)
            for _ in range(40):
                if not _pid_alive(st["pid"]):
                    break
                time.sleep(0.25)
            if _pid_alive(st["pid"]):
                with contextlib.suppress(Exception):
                    os.kill(st["pid"], signal.SIGKILL)
        _mesh_respawn(st["env"])
        return True, "respawned"
    if svc not in _HEALABLE:
        return False, f"unknown or unhealable service {svc!r}"
    if MODE_FILE.exists() and MODE_FILE.read_text().strip() != "docker":
        return False, "stack restarts only in docker mode"
    if _state().get("state") in _RUNNING_STATES:
        return False, "update in progress — restart suppressed"
    res = _run_capture(["docker", "compose", "restart", svc], timeout=180)
    if res is None:
        return False, "docker compose restart timed out"
    if res.returncode != 0:
        return False, (res.stderr or res.stdout or "compose restart failed")[:300]
    return True, "restarted"


def _watchdog() -> None:
    """Supervisor tick: refresh the update check, kill stalled update runs,
    then heal supervised services — fully suppressed while an update or
    restart is in flight so `docker compose down` never races the watchdog."""
    global _wd_suppressed
    while True:
        time.sleep(WATCHDOG_INTERVAL_S)
        try:
            _reset_stale_state()
            if _check_stale():
                _spawn(["--check"])
            _maybe_self_reexec()
            _stall_check_update()
            _wd_suppressed = _state().get("state") in _RUNNING_STATES
            if not _wd_suppressed:
                _supervise_mesh()
                _supervise_stack()
            _wd_write()
        except Exception:
            pass


if __name__ == "__main__":
    main()
