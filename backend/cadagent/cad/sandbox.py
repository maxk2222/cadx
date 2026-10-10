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
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from cadagent.cad.allowlist import apply_params_override, check_code, validate_overrides
from cadagent.config import get_settings

RUNNER = Path(__file__).parent / "runner_entry.py"


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
    brep_path: Path | None = None
    flat_brep_path: Path | None = None
    duration_s: float = 0.0
    static_errors: list[str] = field(default_factory=list)


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


def run_user_code(
    code: str,
    workdir: Path,
    *,
    params_override: dict[str, object] | None = None,
    references: list[Reference] | None = None,
    mesh_solid: Path | None = None,
    mesh_bbox: list[float] | None = None,
    timeout_s: int | None = None,
    mem_mb: int | None = None,
) -> RunnerOutput:
    settings = get_settings()
    timeout_s = timeout_s or settings.exec_timeout_s
    mem_mb = mem_mb or settings.exec_mem_limit_mb
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
            }
        ),
        encoding="utf-8",
    )

    # 3. spawn sandboxed subprocess
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workdir),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "2",
        "LANG": "C.UTF-8",
    }
    started = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-I", str(RUNNER), str(workdir)],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: _limits(mem_mb, timeout_s),
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

    data = json.loads(result_path.read_text(encoding="utf-8"))
    out = RunnerOutput(
        ok=bool(data.get("ok")),
        error=data.get("error"),
        traceback=data.get("traceback"),
        stdout=data.get("stdout", ""),
        params_used=data.get("params_used", {}) or {},
        thread_count=int(data.get("thread_count", 0)),
        brep_path=(workdir / "model.brep") if (workdir / "model.brep").exists() else None,
        flat_brep_path=(
            (workdir / "flat_profile.brep") if (workdir / "flat_profile.brep").exists() else None
        ),
        duration_s=duration,
    )
    if not out.ok and not out.error:
        out.error = "runner failed without error message"
    return out
