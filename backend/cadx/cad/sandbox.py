"""Sandboxed execution of LLM-generated model code.

Process isolation: child runs `python -I runner_entry.py <workdir>` with CPU/memory
rlimits, a stripped environment and a watchdog that kills the whole process group.
Real isolation comes from the container (internal network, read-only FS, non-root).
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from cadx.cad import resources
from cadx.cad.allowlist import apply_params_override, check_code, validate_overrides
from cadx.runtime_config import RuntimeSettings

RUNNER = Path(__file__).parent / "runner_entry.py"
ZYGOTE = Path(__file__).parent / "zygote.py"


@dataclass
class Reference:
    asset_id: str
    path: str  # absolute path inside the executor


@dataclass
class RunnerOutput:
    ok: bool
    error: str | None = None
    traceback: str | None = None
    stdout: str = ""
    params_used: dict[str, object] = field(default_factory=dict)
    thread_count: int = 0
    preview_thread_count: int = 0
    detail_mode: str = "preview"
    design_features: list[dict[str, object]] = field(default_factory=list)
    bom: list[dict[str, object]] = field(default_factory=list)
    brep_path: Path | None = None
    flat_brep_path: Path | None = None
    duration_s: float = 0.0
    static_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # geometry inventory of the failed namespace — where solids actually are
    geometry_at_failure: list[dict[str, object]] = field(default_factory=list)


def _limits(mem_mb: int, cpu_s: int) -> None:
    """preexec_fn: apply rlimits in the child (best-effort per limit).

    Linux enforces all of them; macOS ignores/rejects RLIMIT_AS — that is fine
    because real isolation lives in the executor container.
    """
    attempts = [
        (resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024)),
        (resource.RLIMIT_CPU, (cpu_s, cpu_s + 5)),
        (resource.RLIMIT_NOFILE, (64, 64)),
        (resource.RLIMIT_FSIZE, (512 * 1024 * 1024, 512 * 1024 * 1024)),
    ]
    for res, lim in attempts:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(res, lim)
    os.setsid()


class _Zygote:
    """One warm fork-server (`python -I zygote.py`) per executor process.

    The zygote imports build123d once at spawn; each request forks a child
    that applies the same rlimits/setsid as the cold path, then runs
    runner_entry.main(workdir). Requests are serialized — one child at a
    time keeps the stdin/stdout protocol trivially in sync.
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.buf = b""

    def spawn(self) -> None:
        self.kill()
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "LANG": "C.UTF-8",
        }
        resources.configure_thread_env(env)
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(ZYGOTE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        line = self._read_line(30.0)
        if not line or not line.startswith("{"):
            raise RuntimeError("zygote failed to signal ready")
        if json.loads(line).get("ready") is not True:
            raise RuntimeError(f"zygote warmup failed: {line[:200]}")

    def kill(self) -> None:
        if self.proc is not None:
            with contextlib.suppress(Exception):
                self.proc.kill()
            self.proc = None
        self.buf = b""

    def _read_line(self, timeout_s: float) -> str:
        """Read one \n-terminated line from zygote stdout; "" on timeout/EOF."""
        assert self.proc is not None and self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        while True:
            if b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                return line.decode("utf-8", errors="replace")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ""
            r, _, _ = select.select([fd], [], [], min(remaining, 0.5))
            if not r:
                if self.proc.poll() is not None:
                    return ""
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return ""
            self.buf += chunk

    def run(self, workdir: Path, timeout_s: int, mem_mb: int, cpu_s: int) -> RunnerOutput:
        """Serve one run. Raises RuntimeError when the zygote is unusable so
        the caller falls back to the cold subprocess path. ``timeout_s`` is
        the wall-clock watchdog budget; ``cpu_s`` the RLIMIT_CPU budget
        (thread-scaled by the caller)."""
        if self.proc is None or self.proc.poll() is not None:
            self.spawn()
        assert self.proc is not None and self.proc.stdin is not None
        started = time.monotonic()
        self.proc.stdin.write(
            (
                json.dumps({"workdir": str(workdir), "mem_mb": mem_mb, "cpu_s": cpu_s}) + "\n"
            ).encode()
        )
        self.proc.stdin.flush()
        first = self._read_line(15.0)
        if not first:
            self.kill()
            raise RuntimeError("zygote did not answer")
        msg = json.loads(first)
        if msg.get("error"):
            raise RuntimeError(f"zygote request rejected: {msg['error']}")
        child_pid = int(msg["pid"])
        budget = timeout_s + 15
        second = self._read_line(budget)
        if not second:
            # child (or zygote) wedged — kill the run; killpg works because
            # the child setsid()s first thing, kill(pid) is the race fallback
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(child_pid, signal.SIGKILL)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(child_pid, signal.SIGKILL)
            self._read_line(5.0)  # resync: drain the done line the zygote will send
            return RunnerOutput(
                ok=False,
                error=f"execution timed out after {budget}s (killed)",
                duration_s=time.monotonic() - started,
            )
        done = json.loads(second)
        stderr = str(done.get("stderr") or "")
        duration = time.monotonic() - started
        rc = int(done.get("rc", 0))
        if not (workdir / "result.json").exists():
            detail = stderr[-2000:]
            if rc == -signal.SIGXCPU:
                msg = f"CPU limit exceeded ({timeout_s}s)"
            elif rc == -signal.SIGKILL:
                msg = "process killed (memory or runtime limit)"
            else:
                msg = f"runner crashed without result (exit {rc})"
            return RunnerOutput(ok=False, error=f"{msg}.\n{detail}".strip(), duration_s=duration)
        return _result_from_workdir(workdir, duration, stderr)


_ZYGOTE = _Zygote()
_ZYGOTE_LOCK = threading.Lock()


def run_user_code(
    code: str,
    workdir: Path,
    *,
    params_override: dict[str, object] | None = None,
    references: list[Reference] | None = None,
    mesh_solid: Path | None = None,
    mesh_bbox: list[float] | None = None,
    anchors: dict[str, object] | None = None,
    detail_mode: str = "preview",
    timeout_s: int | None = None,
    mem_mb: int | None = None,
    warm: bool | None = None,
    threads: int | None = None,
) -> RunnerOutput:
    if detail_mode not in {"preview", "export"}:
        return RunnerOutput(ok=False, error=f"invalid detail_mode '{detail_mode}'")
    settings = RuntimeSettings()
    timeout_s = timeout_s or settings.exec_timeout_s
    mem_mb = mem_mb or resources.effective_exec_mem_mb(settings)
    warm = settings.exec_warm_runner if warm is None else warm
    nthreads = resources.thread_budget(threads or settings.exec_threads)
    # RLIMIT_CPU counts CPU-seconds across threads — scale it so the budget
    # stays "wall-clock seconds of work"; the proc timeout is the real deadline
    cpu_s = resources.rlimit_cpu_s(timeout_s, nthreads)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. static checks (no execution on failure)
    check = check_code(code)
    if not check.ok:
        return RunnerOutput(
            ok=False,
            error="static check failed:\n" + "\n".join(check.errors),
            static_errors=check.errors,
        )

    overrides = params_override or {}
    errors = validate_overrides(overrides, check.params)
    if errors:
        return RunnerOutput(ok=False, error="param override failed:\n" + "\n".join(errors))

    # 2. write model.py (with injected overrides) + input.json
    if overrides:
        try:
            code = apply_params_override(code, overrides)
        except ValueError as e:
            return RunnerOutput(ok=False, error=f"param override failed: {e}")

    (workdir / "model.py").write_text(code, encoding="utf-8")
    (workdir / "input.json").write_text(
        json.dumps(
            {
                "params_override": overrides,
                "references": [{"asset_id": r.asset_id, "path": r.path} for r in references or []],
                "mesh_solid": str(mesh_solid) if mesh_solid else None,
                "mesh_bbox": mesh_bbox,
                "anchors": anchors or {},
                "detail_mode": detail_mode,
            }
        ),
        encoding="utf-8",
    )

    # 3. spawn sandboxed runner — warm fork-server when enabled, with the
    # cold subprocess as automatic fallback on any zygote failure
    if warm:
        try:
            with _ZYGOTE_LOCK:
                return _ZYGOTE.run(workdir, timeout_s, mem_mb, cpu_s)
        except Exception:
            _ZYGOTE.kill()  # a half-broken zygote must not poison later runs

    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workdir),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "LANG": "C.UTF-8",
    }
    resources.configure_thread_env(env, nthreads)
    started = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-I", str(RUNNER), str(workdir)],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: _limits(mem_mb, cpu_s),
        text=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout_s + 15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return RunnerOutput(
            ok=False,
            error=f"execution timed out after {timeout_s + 15}s (killed)",
            duration_s=time.monotonic() - started,
        )

    duration = time.monotonic() - started
    result_path = workdir / "result.json"
    if not result_path.exists():
        detail = (stderr or "")[-2000:] if isinstance(stderr, str) else ""
        rc = proc.returncode
        if rc == -signal.SIGXCPU:
            msg = f"CPU limit exceeded ({timeout_s}s)"
        elif rc == -signal.SIGKILL:
            msg = "process killed (memory or runtime limit)"
        else:
            msg = f"runner crashed without result (exit {rc})"
        return RunnerOutput(ok=False, error=f"{msg}.\n{detail}".strip(), duration_s=duration)

    return _result_from_workdir(workdir, duration)


def _result_from_workdir(workdir: Path, duration: float, stderr: str = "") -> RunnerOutput:
    """Parse <workdir>/result.json into a RunnerOutput — shared by the cold
    subprocess path and the warm zygote path (the zygote child writes the
    identical file)."""
    result_path = workdir / "result.json"
    if not result_path.exists():
        detail = stderr[-2000:]
        return RunnerOutput(
            ok=False,
            error=f"runner crashed without result.\n{detail}".strip(),
            duration_s=duration,
        )
    data = json.loads(result_path.read_text(encoding="utf-8"))
    out = RunnerOutput(
        ok=bool(data.get("ok")),
        error=data.get("error"),
        traceback=data.get("traceback"),
        stdout=data.get("stdout", ""),
        params_used=data.get("params_used", {}) or {},
        thread_count=int(data.get("thread_count", 0)),
        preview_thread_count=int(data.get("preview_thread_count", 0)),
        detail_mode=str(data.get("detail_mode") or "preview"),
        design_features=list(data.get("design_features") or []),
        bom=list(data.get("bom") or []),
        brep_path=(workdir / "model.brep") if (workdir / "model.brep").exists() else None,
        flat_brep_path=(
            (workdir / "flat_profile.brep") if (workdir / "flat_profile.brep").exists() else None
        ),
        warnings=list(data.get("warnings") or []),
        geometry_at_failure=list(data.get("geometry_at_failure") or []),
        duration_s=duration,
    )
    if not out.ok and not out.error:
        out.error = "runner failed without error message"
    return out
