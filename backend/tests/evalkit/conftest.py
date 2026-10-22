"""evalkit tests: stub the arq enqueue so spawn paths don't need live redis
(same pattern as tests/unit/conftest.py)."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _stub_enqueue(monkeypatch: pytest.MonkeyPatch):
    import cadx.api.deps as deps

    async def fake_enqueue(func: str, payload: dict[str, Any]) -> str:
        return f"test-arq-{func}"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)
