"""find_or_create_project must PATCH case settings onto a re-run project."""

from __future__ import annotations

import json

import httpx
import pytest

from cadx.evalkit.http import HttpDriver


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> HttpDriver:
    d = HttpDriver("http://testserver")
    return d


def _handler(captured: dict):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/projects":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "name": "eval:helical-pinion-shaft",
                        "settings": {
                            "gen_mode": "cad",
                            "clarification_mode": "assume",
                            "min_wall_mm": 0.8,
                        },
                    }
                ],
            )
        if request.method == "PATCH":
            captured["patch_body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "11111111-1111-1111-1111-111111111111",
                    "name": "eval:helical-pinion-shaft",
                    "settings": captured["patch_body"]["settings"],
                },
            )
        return httpx.Response(404)

    return handle


@pytest.mark.asyncio
async def test_existing_project_gets_case_settings_patch(driver, monkeypatch):
    captured: dict = {}
    driver._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(captured)), base_url="http://testserver"
    )
    p = await driver.find_or_create_project(
        "eval:helical-pinion-shaft",
        {"_eval": True, "clarification_mode": "assume", "min_wall_mm": 0.1},
    )
    assert captured.get("patch_body"), "case settings must be PATCHed onto an existing project"
    patched = captured["patch_body"]["settings"]
    assert patched["min_wall_mm"] == 0.1
    # untouched case keys merge over, existing project keys survive
    assert patched["gen_mode"] == "cad"
    assert p["settings"]["min_wall_mm"] == 0.1


@pytest.mark.asyncio
async def test_existing_project_same_settings_no_patch(driver, monkeypatch):
    captured: dict = {}
    driver._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(captured)), base_url="http://testserver"
    )
    await driver.find_or_create_project(
        "eval:helical-pinion-shaft",
        {"gen_mode": "cad", "clarification_mode": "assume", "min_wall_mm": 0.8},
    )
    assert "patch_body" not in captured
