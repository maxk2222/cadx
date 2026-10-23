"""Host/container resource detection for the CAD executor.

Everything CPU/memory-bound derives from these helpers instead of
hard-coded constants. Detection honours the actual cgroup limits, not
just the hardware: ``cpus:`` in compose sets the CFS quota
(``/sys/fs/cgroup/cpu.max``), which ``os.sched_getaffinity`` does NOT
reflect — a 4-cpu container on a 16-core host reports 16 via affinity.

Rules of thumb:
- ``detect_cpus``: min(affinity, floor(cfs_quota/period)) — cgroup v2 and
  v1, falling back to affinity/cpu_count when no quota is set.
- ``detect_mem_mb``: cgroup memory limit (v2 ``memory.max`` / v1
  ``memory.limit_in_bytes``), else physical RAM via sysconf. An
  "unlimited"/absent cgroup value falls back to physical.
- ``thread_budget``: threads a single CPU-bound process may spawn —
  OpenBLAS/OMP/llvmpipe/OCCT. Capped so 2 concurrent cad jobs don't
  oversubscribe.
- ``pool_size``: fork/spawn worker count for parallel case solves and
  renders.
- ``rlimit_cpu_s``: RLIMIT_CPU counts CPU-seconds across ALL threads —
  N threads burn the budget Nx faster, so the sandbox CPU limit scales
  with the thread budget (wall-clock stays the real deadline).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)

_MAX_THREADS = 8
_MAX_POOL = 4
_DETECTED: dict[str, Any] | None = None


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota() -> int | None:
    """CPU count implied by the cgroup CFS quota, or None when unlimited."""
    # v2: "MAX 100000" | "<quota_us> <period_us>"
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except OSError:
        pass
    # v1
    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota is not None and period and quota > 0:
        return max(1, quota // period)
    return None


def _cgroup_mem_bytes() -> int | None:
    """cgroup memory limit in bytes, or None when unlimited/absent."""
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            raw = Path(path).read_text().strip()
            if raw == "max":
                continue
            value = int(raw)
            # kernels report "unlimited" as a huge sentinel — ignore values
            # beyond a plausible physical size (>1 PB)
            if 0 < value < 1 << 50:
                return value
        except (OSError, ValueError):
            continue
    return None


def _phys_mem_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None


def detect_cpus() -> int:
    """Usable CPUs: sched_getaffinity (cpuset/hotplug) intersected with the
    CFS quota (docker ``cpus:``). Never below 1."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None:
        return max(1, min(affinity, quota))
    return max(1, affinity)


def detect_mem_mb() -> int:
    """Usable memory in MB — cgroup limit when set, else physical RAM."""
    value = _cgroup_mem_bytes() or _phys_mem_bytes()
    if value is None:
        return 4096
    return max(512, int(value / (1024 * 1024)))


def _cgroup_mem_current_bytes() -> int | None:
    """Live cgroup usage in bytes — memory.current (v2) / usage_in_bytes
    (v1); None when unreadable."""
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def mem_headroom_mb() -> float | None:
    """Free bytes under the cgroup limit in MB (limit - current), or None
    when either side is unmeasurable (bare metal, no cgroup) — callers must
    treat None as "unknown", never as "plenty"."""
    limit = _cgroup_mem_bytes()
    current = _cgroup_mem_current_bytes()
    if limit is None or current is None:
        return None
    return (limit - current) / (1024 * 1024)


def thread_budget(override: int = 0) -> int:
    """Threads one CPU-bound stage may use; ``override <= 0`` = auto."""
    if override > 0:
        return override
    return max(1, min(detect_cpus(), _MAX_THREADS))


def pool_size(override: int = 0) -> int:
    """Process-pool workers for parallel solves/renders; ``1`` disables."""
    if override > 0:
        return max(1, override)
    return max(1, min(_MAX_POOL, detect_cpus() - 1))


def rlimit_cpu_s(timeout_s: int, threads: int | None = None) -> int:
    """CPU-seconds budget for a sandbox child. RLIMIT_CPU accumulates
    across threads, so it scales with the thread budget — the wall-clock
    watchdog (proc timeout) remains the true deadline."""
    threads = threads if threads is not None else thread_budget()
    return timeout_s * max(1, threads)


def effective_exec_mem_mb(rt: RuntimeSettings) -> int:
    """RLIMIT_AS / FEM budget: explicit ``exec_mem_limit_mb`` wins,
    ``0`` = auto (60 % of detected memory, bounded)."""
    if rt.exec_mem_limit_mb > 0:
        return rt.exec_mem_limit_mb
    return max(2048, min(int(detect_mem_mb() * 0.6), 32768))


def configure_thread_env(env: dict[str, str], threads: int | None = None) -> None:
    """setdefault the BLAS/OMP/llvmpipe thread vars on ``env`` (in place).
    ``env=os.environ`` configures the current process; a stripped dict
    configures a child env. Never overrides an explicit value."""
    n = str(threads if threads is not None else thread_budget())
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "LP_NUM_THREADS",  # llvmpipe software rasterizer (offscreen renders)
    ):
        env.setdefault(var, n)


def detected() -> dict[str, Any]:
    """Memoized detection snapshot for logs and the settings API."""
    global _DETECTED
    if _DETECTED is None:
        _DETECTED = {
            "cpus": detect_cpus(),
            "mem_mb": detect_mem_mb(),
            "threads": thread_budget(),
            "pool": pool_size(),
        }
    return dict(_DETECTED)


def log_detected() -> None:
    d = detected()
    log.info("resources.detected", **d)
