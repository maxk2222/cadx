"""Per-model sampling profiles: llm_sampling_profiles payload overrides."""

from __future__ import annotations

import json

import httpx
import pytest

from cadx.llm.client import LLMClient
from cadx.runtime_config import RuntimeSettings


def _patch_transport(monkeypatch, handler):
    orig = httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return orig(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", patched)


def _recorder(seen: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    return handler


PROFILES = json.dumps(
    {
        "qwen3.8-27b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "qwen3.8-flash-next": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "qwen3.8": {"temperature": 9.9},  # decoy — longer patterns must win
    }
)


def _client(profiles: str = PROFILES, model: str = "qwen3.8-27b-instruct") -> LLMClient:
    c = LLMClient(
        api_key="k",
        base_url="http://llm.test/v1",
        model_code=model,
        runtime_settings=RuntimeSettings(
            llm_temperature_code=0.2,
            llm_sampling_profiles=profiles,
        ),
    )
    c.stream = False
    return c


async def test_profile_temperature_and_extras_applied(monkeypatch):
    seen: list[dict] = []
    _patch_transport(monkeypatch, _recorder(seen))
    await _client().chat("code", [{"role": "user", "content": "hi"}])
    body = next(b for b in seen if 'messages' in b)
    assert body["temperature"] == 0.6
    assert body["top_p"] == 0.95 and body["top_k"] == 20
    assert "min_p" not in body and "presence_penalty" not in body


async def test_longest_pattern_wins(monkeypatch):
    seen: list[dict] = []
    _patch_transport(monkeypatch, _recorder(seen))
    await _client(model="Qwen3.8-Flash-Next-A3B").chat(
        "code", [{"role": "user", "content": "hi"}]
    )
    assert next(b for b in seen if 'messages' in b)["temperature"] == 1.0


async def test_explicit_call_temperature_wins(monkeypatch):
    seen: list[dict] = []
    _patch_transport(monkeypatch, _recorder(seen))
    await _client().chat(
        "code", [{"role": "user", "content": "hi"}], temperature=0.0
    )
    assert next(b for b in seen if 'messages' in b)["temperature"] == 0.0
    assert next(b for b in seen if 'messages' in b)["top_p"] == 0.95  # non-temperature params still apply


async def test_no_match_falls_back_to_global(monkeypatch):
    seen: list[dict] = []
    _patch_transport(monkeypatch, _recorder(seen))
    await _client(model="gpt-5").chat("code", [{"role": "user", "content": "hi"}])
    assert next(b for b in seen if 'messages' in b)["temperature"] == 0.2
    assert "top_p" not in next(b for b in seen if 'messages' in b)


async def test_invalid_json_ignored(monkeypatch):
    seen: list[dict] = []
    _patch_transport(monkeypatch, _recorder(seen))
    await _client(profiles="{not json").chat(
        "code", [{"role": "user", "content": "hi"}]
    )
    assert next(b for b in seen if 'messages' in b)["temperature"] == 0.2


async def test_extra_body_still_overrides(monkeypatch):
    seen: list[dict] = []
    _patch_transport(monkeypatch, _recorder(seen))
    c = _client()
    c.extra_body = {"top_p": 0.5}
    await c.chat("code", [{"role": "user", "content": "hi"}])
    assert next(b for b in seen if 'messages' in b)["top_p"] == 0.5
