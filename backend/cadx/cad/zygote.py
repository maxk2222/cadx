"""Warm fork-server for the CAD sandbox.

Started once per executor process (`python -I zygote.py`). Imports the heavy
build123d/OCP stack ONCE, then serves run requests over stdin/stdout by
forking a child per request — the child inherits the imported modules, so
each model run skips the ~2-4 s interpreter+import startup.

Protocol (one JSON line per message):
    request : {"workdir": "...", "mem_mb": int, "cpu_s": int}
    reply 1 : {"pid": <child pid>}      — sent right after fork
    reply 2 : {"rc": <exit>, "stderr": "<tail>"}  — sent after child exits

The child applies the same rlimits + setsid as the cold path, redirects its
fd1/fd2 to a pipe (protocol stays clean, diagnostics captured) and runs
runner_entry.main(workdir) — identical semantics to the cold subprocess.
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import sys
import traceback
from pathlib import Path


def _limits(mem_mb: int, cpu_s: int) -> None:
    """Same rlimits as the cold path's preexec_fn."""
    attempts = [
        (resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024)),
        (resource.RLIMIT_CPU, (cpu_s, cpu_s + 5)),
        (resource.RLIMIT_NOFILE, (64, 64)),
        (resource.RLIMIT_FSIZE, (512 * 1024 * 1024, 512 * 1024 * 1024)),
    ]
    for res, lim in attempts:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(res, lim)


def _child(workdir: str, mem_mb: int, cpu_s: int, pipe_w: int) -> None:
    """Runs inside the forked child. Never returns — always os._exit."""
    rc = 0
    try:
        os.dup2(pipe_w, 1)
        os.dup2(pipe_w, 2)
        os.setsid()  # own process group -> parent can killpg on timeout
        _limits(mem_mb, cpu_s)
        os.environ["HOME"] = workdir
        os.chdir(workdir)
        from cadx.cad import runner_entry

        rc = runner_entry.main(Path(workdir))
    except BaseException:
        with contextlib.suppress(Exception):
            traceback.print_exc()  # goes to the stderr pipe
        rc = 1
    finally:
        os._exit(rc)


def _read_all(fd: int) -> str:
    chunks: list[bytes] = []
    while True:
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    os.close(fd)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _reply(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main() -> int:
    # --- warmup: pull in the expensive modules so children get them via fork
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # cadlib import
    try:
        import build123d  # noqa: F401
    except Exception:
        _reply({"error": "warmup failed: build123d not importable"})
        return 1
    with contextlib.suppress(Exception):
        import cadlib  # noqa: F401

    _reply({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            workdir = str(req["workdir"])
            mem_mb = int(req["mem_mb"])
            cpu_s = int(req["cpu_s"])
        except Exception as e:
            _reply({"error": f"bad request: {e}"})
            continue

        pipe_r, pipe_w = os.pipe()
        try:
            pid = os.fork()
        except OSError as e:
            os.close(pipe_r)
            os.close(pipe_w)
            _reply({"error": f"fork failed: {e}"})
            continue
        if pid == 0:
            os.close(pipe_r)
            _child(workdir, mem_mb, cpu_s, pipe_w)
            os._exit(1)  # unreachable, _child never returns
        os.close(pipe_w)
        _reply({"pid": pid})
        # drain the child's fd1/fd2 until EOF (= child exit) so a chatty
        # child can't block on a full pipe; then reap it.
        stderr = _read_all(pipe_r)
        _, status = os.waitpid(pid, 0)
        rc = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status
        _reply({"rc": rc, "stderr": stderr[-4000:]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
