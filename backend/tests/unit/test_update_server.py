"""update_server.py daemon: auth, spawn serialization, state/log proxying.

The daemon resolves its repo root from its own path (ROOT = parent of the
scripts/ dir containing it), so tests copy the script into a tmp layout with a
stub update.sh — no git or docker needed.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_SRC = REPO_ROOT / "scripts" / "update_server.py"
TOKEN = "test-token-abc"

STUB = """#!/usr/bin/env bash
mkdir -p run logs
echo '{"state":"running","phase":"deps"}' > run/update-state.json
echo "stub: $*" > run/stub-args.txt
sleep 0.4
echo '{"state":"done","phase":"done","new_sha":"abc123"}' > run/update-state.json
"""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def daemon(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(SERVER_SRC, scripts / "update_server.py")
    stub = scripts / "update.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(scripts / "update_server.py")],
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "UPDATE_HOST": "127.0.0.1",
            "UPDATE_PORT": str(port),
            "CADX_UPDATE_TOKEN": TOKEN,
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/v1/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("daemon did not start")
    yield base
    proc.terminate()
    proc.wait(timeout=5)


def _req(base: str, method: str, path: str, body: dict | None = None, auth=True):
    req = urllib.request.Request(
        f"{base}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    if auth:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_auth_and_health(daemon):
    code, _ = _req(daemon, "GET", "/v1/health", auth=False)
    assert code == 200
    code, _ = _req(daemon, "GET", "/v1/version", auth=False)
    assert code == 401
    code, d = _req(daemon, "GET", "/v1/version")
    assert code == 200 and "sha" in d


def test_update_lifecycle_and_409(daemon):
    code, d = _req(daemon, "POST", "/v1/update", {})
    assert code == 202 and d["started"] is True
    # second run while the stub is still sleeping -> conflict
    code, d = _req(daemon, "POST", "/v1/update", {})
    assert code == 409
    code, d = _req(daemon, "GET", "/v1/update/status")
    assert d["state"] in ("running", "done")
    # wait for the stub to finish and report done
    for _ in range(50):
        code, d = _req(daemon, "GET", "/v1/update/status")
        if d["state"] == "done":
            break
        time.sleep(0.1)
    assert d["state"] == "done" and d["new_sha"] == "abc123"


def test_ref_validation_and_rollback_confirm(daemon):
    code, _ = _req(daemon, "POST", "/v1/update", {"ref": "bad ref; rm -rf"})
    assert code == 400
    code, _ = _req(daemon, "POST", "/v1/update/rollback", {})
    assert code == 400
    code, _ = _req(daemon, "POST", "/v1/update/rollback", {"confirm": True})
    assert code == 202


def test_check_endpoint(daemon):
    code, _ = _req(daemon, "GET", "/v1/update/check")
    assert code == 200  # no cache file yet -> stale placeholder + refresh spawn
    code, _ = _req(daemon, "POST", "/v1/update/check", {"ref": "main"})
    assert code in (202, 409)  # 409 is fine when the GET already spawned a check
