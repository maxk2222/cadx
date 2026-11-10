"""Regression: same-version stage spawns must wait for the parent version to
settle — a job row reads terminal before the version's own status flips, and
a fem/optimize POST inside that window is 422'd by the editability gate."""

from __future__ import annotations

import pytest

from cadx.evalkit.http import DriverCtx, EvalApiError, HttpDriver
from cadx.evalkit.suite import EvalStage


class _FakeApi:
    """Minimal request router over the driver's `_req` — records call order
    so the test can assert the spawn POST only fired once the parent read
    terminal."""

    def __init__(self, statuses: list[str]) -> None:
        self.statuses = list(statuses)
        self.post_seen_while_busy = False
        self._last_status = ""

    async def req(self, method: str, path: str, **kw):
        if method == "GET" and path.startswith("/versions/"):
            self._last_status = self.statuses.pop(0) if self.statuses else "succeeded"
            return {"status": self._last_status, "id": path.rsplit("/", 1)[-1]}
        if method == "POST" and "/fem" in path:
            if self._last_status in ("queued", "running", "waiting_for_input"):
                self.post_seen_while_busy = True
            return {"version_id": "v1", "job_id": "j1"}
        if method == "GET" and path.startswith("/jobs/") and path.endswith("/trace"):
            return []
        if method == "GET" and path.startswith("/jobs/"):
            return {"status": "succeeded"}
        raise AssertionError(f"unexpected {method} {path}")


@pytest.mark.asyncio
async def test_parent_spawn_waits_for_version_settle(monkeypatch):
    driver = HttpDriver()
    fake = _FakeApi(["running", "running", "succeeded"])
    monkeypatch.setattr(driver, "_req", fake.req)
    await driver.close()

    ctx = DriverCtx()
    ctx.project_id = "p1"
    ctx.parent_version_id = "v1"
    stage = EvalStage(id="fem", type="fem", load_cases=[{"name": "lc", "loads": []}])
    out = await driver.dispatch(stage, ctx)

    assert out.status == "succeeded"
    assert fake.post_seen_while_busy is False
    # two busy polls consumed, then terminal — spawn happened on the third read
    assert fake.statuses == []


@pytest.mark.asyncio
async def test_parent_spawn_surfaces_gate_after_settle_budget(monkeypatch):
    """A version that never settles within the budget still gets POSTed — the
    spawn endpoint's gate error is the authoritative failure to report."""
    driver = HttpDriver()
    fake = _FakeApi(["running"] * 1000)
    monkeypatch.setattr(driver, "_req", fake.req)
    monkeypatch.setattr(driver, "_wait_version_settled", _short_settle(driver))
    await driver.close()

    ctx = DriverCtx()
    ctx.project_id = "p1"
    ctx.parent_version_id = "v1"
    stage = EvalStage(id="fem", type="fem", load_cases=[{"name": "lc", "loads": []}])
    out = await driver.dispatch(stage, ctx)

    assert fake.post_seen_while_busy is True
    assert out.status == "succeeded"  # fake API still spawned it


def _short_settle(driver: HttpDriver):
    async def _wait(vid: str, timeout_s: float = 0.05):
        return await HttpDriver._wait_version_settled(driver, vid, timeout_s)

    return _wait


@pytest.mark.asyncio
async def test_wait_version_settled_timeout_proceeds(monkeypatch):
    driver = HttpDriver()
    fake = _FakeApi(["running"] * 100)
    monkeypatch.setattr(driver, "_req", fake.req)
    await driver.close()

    await driver._wait_version_settled("v1", timeout_s=0.2)
    assert len(fake.statuses) < 100  # polled repeatedly, then gave up


@pytest.mark.asyncio
async def test_wait_version_settled_tolerates_errors(monkeypatch):
    driver = HttpDriver()
    calls = {"n": 0}

    async def flaky(method: str, path: str, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EvalApiError("boom", status=500)
        return {"status": "succeeded"}

    monkeypatch.setattr(driver, "_req", flaky)
    await driver.close()

    await driver._wait_version_settled("v1", timeout_s=5.0)
    assert calls["n"] == 2
