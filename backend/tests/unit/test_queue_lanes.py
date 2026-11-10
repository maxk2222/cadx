"""Two-lane cad queue: light codegen-loop tasks must never queue behind a
heavy solve on the serial 'cad' lane."""

from __future__ import annotations

import asyncio

from cadx.agent.executor_client import _LIGHT_FUNCS, CAD_LIGHT_QUEUE, CAD_QUEUE, QueueExecutor
from cadx.queue import CadLightWorkerSettings, CadWorkerSettings


def _names(settings) -> set[str]:
    return {f.__name__ for f in settings.functions}


def test_lanes_are_disjoint_and_cover_all_cad_funcs() -> None:
    light = _names(CadLightWorkerSettings)
    heavy = _names(CadWorkerSettings)
    assert not light & heavy, "a func must live on exactly one lane"
    assert light == set(_LIGHT_FUNCS)
    # the heavy lane keeps every CPU/mem-bound task
    assert {"optimize", "topopt_resample", "fem_analyze", "organic", "refine"} <= heavy
    assert light | heavy == {
        "execute_version",
        "audit_params",
        "edit_feature",
        "export_all",
        "process_mesh",
        "parametrize_mesh",
        "fem_analyze",
        "optimize",
        "topopt_resample",
        "refine",
        "organic",
    }
    assert CadWorkerSettings.queue_name == CAD_QUEUE
    assert CadLightWorkerSettings.queue_name == CAD_LIGHT_QUEUE
    assert CadLightWorkerSettings.max_jobs >= 1


def test_run_task_routes_light_funcs_to_light_lane() -> None:
    enqueued: list[tuple[str, str]] = []

    class _Pool:
        async def enqueue_job(self, func, payload, _queue_name, _job_id):
            enqueued.append((func, _queue_name))
            raise RuntimeError("stop after routing")

    ex = QueueExecutor(pool=_Pool())  # type: ignore[arg-type]

    async def run() -> None:
        await ex._run_task("execute_version", {}, 60)
        await ex._run_task("optimize", {}, 60)

    asyncio.run(run())
    assert enqueued == [
        ("execute_version", "cad-light"),
        ("optimize", "cad"),
    ]


def test_run_task_polls_status_on_the_enqueue_lane(monkeypatch) -> None:
    """Regression: a task still queued behind a heavy job must report
    queued — never 'lost'. Job defaults to the 'arq:queue' keyspace, so a
    missing _queue_name polled zscore('arq:queue') -> not_found -> the
    60s lost-patience fired while the real cad-lane entry still waited."""
    from cadx.agent import executor_client
    from cadx.agent.executor_client import JobStatus

    seen: dict[str, str] = {}

    class _Pool:
        async def enqueue_job(self, func, payload, _queue_name, _job_id):
            seen["queue"] = _queue_name
            seen["job_id"] = _job_id

        async def zcard(self, name):
            return 1

    class _Job:
        def __init__(self, job_id, pool, _queue_name):
            seen["status_queue"] = _queue_name

        async def status(self):
            return JobStatus.complete

        async def result(self, *a, **kw):
            return {"ok": True}

    monkeypatch.setattr(executor_client, "Job", _Job)
    ex = QueueExecutor(pool=_Pool())  # type: ignore[arg-type]

    async def run() -> dict:
        return await ex._run_task("fem_analyze", {}, 60)

    out = asyncio.run(run())
    assert out["ok"] is True
    assert seen["queue"] == "cad" == seen["status_queue"]


def test_run_task_resubmits_once_on_inflight_crash(monkeypatch) -> None:
    """A worker killed mid-flight leaves 'max N retries exceeded' as the
    arq result — the task never executed, so one resubmit is safe. A
    genuine task failure must NOT resubmit."""
    from cadx.agent import executor_client
    from cadx.agent.executor_client import JobStatus

    attempts: list[str] = []

    class _Pool:
        async def enqueue_job(self, func, payload, _queue_name, _job_id):
            attempts.append(_job_id)

        async def zcard(self, name):
            return 0

    class _CrashJob:
        def __init__(self, job_id, pool, _queue_name):
            pass

        async def status(self):
            return JobStatus.complete

        async def result(self, *a, **kw):
            raise RuntimeError("Job execution failed: max 1 retries exceeded")

    class _OkJob(_CrashJob):
        async def result(self, *a, **kw):
            return {"ok": True}

    seq = [_CrashJob, _OkJob]
    monkeypatch.setattr(executor_client, "Job", lambda *a, **kw: seq.pop(0)(*a, **kw))
    ex = QueueExecutor(pool=_Pool())  # type: ignore[arg-type]

    async def run() -> dict:
        return await ex._run_task("fem_analyze", {}, 60)

    out = asyncio.run(run())
    assert out["ok"] is True and len(attempts) == 2

    # a real task failure (its own exception) must not resubmit
    seq[:] = [_CrashJob]
    monkeypatch.setattr(executor_client, "Job", lambda *a, **kw: _FailJob(*a, **kw))

    class _FailJob(_CrashJob):
        async def result(self, *a, **kw):
            raise ValueError("region 'foot' resolved to 0 nodes")

    out = asyncio.run(run())
    assert out["ok"] is False and len(attempts) == 3  # no resubmit on task-level failure
