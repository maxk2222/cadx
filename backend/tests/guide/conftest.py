"""Guide-test defaults: stub the arq enqueue like tests/unit does."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _stub_enqueue(monkeypatch: pytest.MonkeyPatch):
    import cadx.api.deps as deps

    async def fake_enqueue(func: str, payload: dict[str, Any], dedup_id: str | None = None) -> str:
        return f"test-arq-{func}"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)
