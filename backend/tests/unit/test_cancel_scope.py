"""Job-cancel vs task-timeout distinction (timeout-partial salvage).

`cancel:{job_id}` = user pressed cancel -> result discarded.
`cancel:{_task_id}` = the worker's wait deadline fired -> a task that
still completes keeps its partial result (exports/topopt/fem), marked
`status:partial` + `timed_out`, instead of being overwritten to a bare
'cancelled' — the lost-salvage bug that made timed-out optimize calls
look like hard failures and provoked blind retries.
"""

from __future__ import annotations

import pytest

from cadx.cad.tasks import _cancelled_scope, _salvage_task_result


class _FakeRedis:
    def __init__(self, keys: set[str]):
        self._keys = keys

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self._keys)


def _ctx(*flags: str) -> dict:
    return {"redis": _FakeRedis(set(flags))}


@pytest.mark.asyncio
async def test_scope_none_without_flags() -> None:
    ctx = _ctx()
    payload = {"job_id": "j1", "_task_id": "t1"}
    assert await _cancelled_scope(ctx, payload) == "none"


@pytest.mark.asyncio
async def test_scope_job_cancel_wins() -> None:
    ctx = _ctx("cancel:j1", "cancel:t1")  # both set — job must dominate
    payload = {"job_id": "j1", "_task_id": "t1"}
    assert await _cancelled_scope(ctx, payload) == "job"


@pytest.mark.asyncio
async def test_scope_task_only() -> None:
    ctx = _ctx("cancel:t1")
    payload = {"job_id": "j1", "_task_id": "t1"}
    assert await _cancelled_scope(ctx, payload) == "task"


@pytest.mark.asyncio
async def test_scope_missing_task_id_field() -> None:
    ctx = _ctx("cancel:None")
    payload = {"job_id": "j1"}  # no _task_id at all
    assert await _cancelled_scope(ctx, payload) == "none"


def test_salvage_keeps_partial_artifacts() -> None:
    d = _salvage_task_result({"exports": {"stl": "x.stl"}, "ok": True})
    assert d["ok"] is True
    assert d["status"] == "partial"
    assert d["timed_out"] is True
    assert d["exports"]["stl"] == "x.stl"
    assert "error" not in d


def test_salvage_marks_topopt_partial() -> None:
    d = _salvage_task_result({"topopt": {"iterations": 11, "density_rel": "d.npz"}})
    assert d["ok"] is True and d["status"] == "partial"
    assert any("partial" in w for w in d["warnings"])


def test_salvage_nukes_empty_result() -> None:
    d = _salvage_task_result({"ok": False})
    assert d["ok"] is False
    assert d["error"] == "cancelled"
    assert "timed_out" not in d


def test_salvage_fem_result_counts_as_usable() -> None:
    d = _salvage_task_result({"fem": {"worst_sf": 7.4}})
    assert d["ok"] is True and d["status"] == "partial"
