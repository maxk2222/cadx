"""Unit-test defaults: stub the arq enqueue so spawn endpoints don't need a
live redis.

`spawn._enqueue` resolves `cadx.api.deps.enqueue` lazily per call, so this
patch is seen by every spawn path (REST routers can't take an enqueue_fn).
Tests that exercise the failure path monkeypatch `deps.enqueue` themselves —
their patch lands after this autouse fixture and wins.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _stub_enqueue(monkeypatch: pytest.MonkeyPatch):
    import cadx.api.deps as deps

    async def fake_enqueue(func: str, payload: dict[str, Any], dedup_id: str | None = None) -> str:
        return f"test-arq-{func}"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)
