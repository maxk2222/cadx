"""Resource auto-detection: cgroup-aware cpu/mem, thread/pool budgets."""

from __future__ import annotations

import cadx.cad.resources as res
from cadx.runtime_config import RuntimeSettings


def _reset_detected(monkeypatch) -> None:
    monkeypatch.setattr(res, "_DETECTED", None)


def test_detect_cpus_affinity_only(monkeypatch) -> None:
    monkeypatch.setattr(res.os, "sched_getaffinity", raising=False, value=lambda _p: set(range(12)))
    monkeypatch.setattr(res, "_cgroup_cpu_quota", lambda: None)
    assert res.detect_cpus() == 12


def test_detect_cpus_quota_wins(monkeypatch) -> None:
    # docker `cpus: 4` is a CFS quota, NOT a cpuset — affinity still
    # reports every host core, so the quota must clamp the result
    monkeypatch.setattr(res.os, "sched_getaffinity", raising=False, value=lambda _p: set(range(16)))
    monkeypatch.setattr(res, "_cgroup_cpu_quota", lambda: 4)
    assert res.detect_cpus() == 4


def test_detect_cpus_affinity_below_quota(monkeypatch) -> None:
    monkeypatch.setattr(res.os, "sched_getaffinity", raising=False, value=lambda _p: {0, 1})
    monkeypatch.setattr(res, "_cgroup_cpu_quota", lambda: 8)
    assert res.detect_cpus() == 2


def test_detect_cpus_never_zero(monkeypatch) -> None:
    monkeypatch.setattr(res.os, "sched_getaffinity", raising=False, value=lambda _p: set())
    monkeypatch.setattr(res, "_cgroup_cpu_quota", lambda: None)
    assert res.detect_cpus() >= 1


def _fake_path(monkeypatch, files: dict[str, str]) -> None:
    """Patch res.Path so the cgroup paths read from ``files``; every other
    path still resolves via the real Path."""

    class _FakePath:
        def __init__(self, p: str) -> None:
            self._p = p

        def read_text(self, *a: object, **kw: object) -> str:
            if self._p in files:
                return files[self._p]
            raise OSError("absent")

    monkeypatch.setattr(res, "Path", _FakePath)


def test_cgroup_cpu_quota_v2(monkeypatch) -> None:
    _fake_path(monkeypatch, {"/sys/fs/cgroup/cpu.max": "400000 100000"})
    assert res._cgroup_cpu_quota() == 4


def test_cgroup_cpu_quota_v2_max_is_none(monkeypatch) -> None:
    _fake_path(monkeypatch, {"/sys/fs/cgroup/cpu.max": "max 100000"})
    assert res._cgroup_cpu_quota() is None


def test_thread_budget_auto_and_override(monkeypatch) -> None:
    monkeypatch.setattr(res, "detect_cpus", lambda: 16)
    assert res.thread_budget(0) == 8  # capped
    assert res.thread_budget(3) == 3
    monkeypatch.setattr(res, "detect_cpus", lambda: 2)
    assert res.thread_budget(0) == 2


def test_pool_size(monkeypatch) -> None:
    monkeypatch.setattr(res, "detect_cpus", lambda: 16)
    assert res.pool_size(0) == 4  # capped
    assert res.pool_size(1) == 1
    assert res.pool_size(6) == 6
    monkeypatch.setattr(res, "detect_cpus", lambda: 1)
    assert res.pool_size(0) == 1


def test_rlimit_cpu_scales_with_threads(monkeypatch) -> None:
    monkeypatch.setattr(res, "detect_cpus", lambda: 8)
    assert res.rlimit_cpu_s(100, 4) == 400
    assert res.rlimit_cpu_s(100, 1) == 100


def test_effective_exec_mem_explicit_wins() -> None:
    s = RuntimeSettings(exec_mem_limit_mb=4096)
    assert res.effective_exec_mem_mb(s) == 4096


def test_effective_exec_mem_auto(monkeypatch) -> None:
    monkeypatch.setattr(res, "detect_mem_mb", lambda: 16 * 1024)
    assert res.effective_exec_mem_mb(RuntimeSettings(exec_mem_limit_mb=0)) == int(16 * 1024 * 0.6)
    monkeypatch.setattr(res, "detect_mem_mb", lambda: 1024)
    assert res.effective_exec_mem_mb(RuntimeSettings(exec_mem_limit_mb=0)) == 2048
    monkeypatch.setattr(res, "detect_mem_mb", lambda: 512 * 1024)
    assert res.effective_exec_mem_mb(RuntimeSettings(exec_mem_limit_mb=0)) == 32768


def test_configure_thread_env_setdefault(monkeypatch) -> None:
    env: dict[str, str] = {"OMP_NUM_THREADS": "1"}  # explicit value wins
    res.configure_thread_env(env, threads=5)
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "5"
    assert env["MKL_NUM_THREADS"] == "5"
    assert env["LP_NUM_THREADS"] == "5"


def test_detected_snapshot(monkeypatch) -> None:
    _reset_detected(monkeypatch)
    monkeypatch.setattr(res, "detect_cpus", lambda: 8)
    monkeypatch.setattr(res, "detect_mem_mb", lambda: 8192)
    d = res.detected()
    assert d == {"cpus": 8, "mem_mb": 8192, "threads": 8, "pool": 4}


def test_executor_fields_include_resource_knobs() -> None:
    vals = RuntimeSettings().executor_values()
    for key in ("exec_threads", "exec_pool_workers", "audit_parallel_runs", "exec_mem_limit_mb"):
        assert key in vals
