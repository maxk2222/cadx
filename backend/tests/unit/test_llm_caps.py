"""Context-window detection: provider probing, error learning, budgets."""

from __future__ import annotations

import json

import httpx
import pytest

from cadx.llm.capabilities import (
    ModelCaps,
    caps_from_card,
    estimate_prompt_tokens,
    parse_ctx_from_error,
    probe_context,
    prompt_over_budget,
    resolve_max_tokens,
)
from cadx.llm.client import LLMClient, LLMError
from cadx.runtime_config import RuntimeSettings


def _patch_transport(monkeypatch, handler):
    orig = httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return orig(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", patched)


def _route(routes: dict) -> object:
    """MockTransport handler keyed by (method, path) → (status, body)."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key in routes:
            status, body = routes[key]
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": "not found"})

    return handler


# --------------------------------------------------------------------------- #
# model-card parsing                                                          #


def test_caps_from_card_vllm():
    caps = caps_from_card({"id": "m", "max_model_len": 262144})
    assert caps and caps.context_tokens == 262144 and caps.source == "models_api"


def test_caps_from_card_openrouter():
    caps = caps_from_card(
        {
            "id": "m",
            "context_length": 200000,
            "top_provider": {"max_completion_tokens": 64000},
        }
    )
    assert caps
    assert caps.context_tokens == 200000
    assert caps.max_output_tokens == 64000


def test_caps_from_card_sidecar_and_lmstudio_legacy():
    assert caps_from_card({"id": "m", "context_len": 131072}).context_tokens == 131072
    assert caps_from_card({"id": "m", "max_context_length": 8192}).context_tokens == 8192
    assert caps_from_card({"id": "m", "n_ctx": 4096}).context_tokens == 4096


def test_caps_from_card_none():
    assert caps_from_card({"id": "m"}) is None
    assert caps_from_card({"id": "m", "max_model_len": None}) is None
    assert caps_from_card({"id": "m", "max_model_len": 3}) is None  # below sanity floor


# --------------------------------------------------------------------------- #
# probe_context — provider shapes                                             #


async def test_probe_vllm_models(monkeypatch):
    _patch_transport(
        monkeypatch,
        _route(
            {
                ("GET", "/v1/models"): (
                    200,
                    {"data": [{"id": "qwen", "max_model_len": 131072}]},
                )
            }
        ),
    )
    caps = await probe_context("http://x/v1", "", ["qwen"])
    assert caps["qwen"].context_tokens == 131072
    assert caps["qwen"].source == "models_api"


async def test_probe_sidecar_listing(monkeypatch):
    _patch_transport(
        monkeypatch,
        _route(
            {
                ("GET", "/v1/models"): (
                    200,
                    {"models": [{"id": "qwen3.5-9b-mlx", "kind": "llm", "context_len": 262144}]},
                )
            }
        ),
    )
    caps = await probe_context("http://x/v1", "", ["qwen3.5-9b-mlx"])
    assert caps["qwen3.5-9b-mlx"].context_tokens == 262144
    assert caps["qwen3.5-9b-mlx"].source == "sidecar"


async def test_probe_lmstudio_loaded_instance_wins(monkeypatch):
    """Loaded-instance config context_length beats the model's max."""
    _patch_transport(
        monkeypatch,
        _route(
            {
                ("GET", "/v1/models"): (200, {"data": [{"id": "qwen"}]}),
                ("GET", "/api/v1/models"): (
                    200,
                    {
                        "models": [
                            {
                                "id": "qwen",
                                "max_context_length": 131072,
                                "loaded_instances": [{"config": {"context_length": 32768}}],
                            }
                        ]
                    },
                ),
            }
        ),
    )
    caps = await probe_context("http://x/v1", "", ["qwen"])
    assert caps["qwen"].context_tokens == 32768
    assert caps["qwen"].source == "lmstudio"


async def test_probe_lmstudio_max_fallback(monkeypatch):
    _patch_transport(
        monkeypatch,
        _route(
            {
                ("GET", "/v1/models"): (200, {"data": [{"id": "qwen"}]}),
                ("GET", "/api/v0/models"): (
                    200,
                    {"data": [{"id": "qwen", "max_context_length": 65536}]},
                ),
            }
        ),
    )
    caps = await probe_context("http://x/v1", "", ["qwen"])
    assert caps["qwen"].context_tokens == 65536
    assert caps["qwen"].source == "lmstudio"


async def test_probe_llamacpp_props(monkeypatch):
    _patch_transport(
        monkeypatch,
        _route(
            {
                ("GET", "/props"): (
                    200,
                    {"default_generation_settings": {"n_ctx": 16384}},
                )
            }
        ),
    )
    caps = await probe_context("http://x:8080", "", ["served"])
    assert caps["served"].context_tokens == 16384
    assert caps["served"].source == "llamacpp"


async def test_probe_ollama_show(monkeypatch):
    _patch_transport(
        monkeypatch,
        _route(
            {
                ("POST", "/api/show"): (
                    200,
                    {"model_info": {"llama.context_length": 131072, "llama.rope": {}}},
                )
            }
        ),
    )
    caps = await probe_context("http://x:11434", "", ["llama3"])
    # /api/show resolves llama3 directly; static table would also match —
    # the probed (ollama) source wins
    assert caps["llama3"].source == "ollama"
    assert caps["llama3"].context_tokens == 131072


async def test_probe_static_table_fallback(monkeypatch):
    """Opaque provider (401 on /models) falls back to the name table."""
    _patch_transport(monkeypatch, _route({}))
    caps = await probe_context("http://x/v1", "", ["gpt-4o"])
    assert caps["gpt-4o"].context_tokens == 128000
    assert caps["gpt-4o"].source == "static"


async def test_probe_unknown_model_absent(monkeypatch):
    _patch_transport(monkeypatch, _route({}))
    caps = await probe_context("http://x/v1", "", ["totally-custom-model"])
    assert "totally-custom-model" not in caps


# --------------------------------------------------------------------------- #
# error-body learning                                                         #


def test_parse_ctx_openai_style():
    assert parse_ctx_from_error(400, "This model's maximum context length is 8192 tokens.") == 8192


def test_parse_ctx_llamacpp_json():
    body = json.dumps({"error": {"message": "n_ctx exceeded", "n_ctx": 16384}})
    assert parse_ctx_from_error(400, body) == 16384


def test_parse_ctx_vllm_request_too_long():
    assert parse_ctx_from_error(400, "maximum context length is 32768, got 40000") == 32768


def test_parse_ctx_unrelated():
    assert parse_ctx_from_error(400, "invalid request body") is None
    # a non-context error status is only mined when the body actually talks
    # about context — but then the number is fair game (best-effort learn)
    assert parse_ctx_from_error(500, "internal error") is None
    assert parse_ctx_from_error(500, "maximum context length is 8192") == 8192


# --------------------------------------------------------------------------- #
# estimation + budget resolution                                              #


def test_estimate_prompt_tokens_text():
    msgs = [{"role": "user", "content": "x" * 400}]
    assert estimate_prompt_tokens(msgs) == 400 // 4 + 4


def test_estimate_prompt_tokens_images_and_tools():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        },
        {"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]},
    ]
    est = estimate_prompt_tokens(msgs, image_detail="high", image_max_edge=1568)
    # 1568 → 4 tiles → 85 + 170*16 = 2805 image tokens
    assert est > 2800


def test_resolve_max_tokens_unknown_ctx():
    s = RuntimeSettings(llm_max_tokens=0)
    assert resolve_max_tokens(s, None, 1000, ctx=0) == 8000
    s2 = RuntimeSettings(llm_max_tokens=4096)
    assert resolve_max_tokens(s2, None, 1000, ctx=0) == 4096


def test_resolve_max_tokens_auto():
    s = RuntimeSettings(llm_max_tokens=0)
    # ctx 128k → auto = 32k; ample room → 32k
    assert resolve_max_tokens(s, None, 1000, ctx=128_000) == 32_000
    # ctx 8k → auto = 2048 floor; headroom 1024, est small → fits
    assert resolve_max_tokens(s, None, 100, ctx=8192) == 2048


def test_resolve_max_tokens_explicit_cap_and_fit():
    s = RuntimeSettings(llm_max_tokens=50_000)
    # explicit cap honoured when it fits
    assert resolve_max_tokens(s, None, 10_000, ctx=128_000) == 50_000
    # but clamped so prompt+completion fits: 32768 - 30000 - 2048 = 720
    assert resolve_max_tokens(s, None, 30_000, ctx=32_768) == 720
    # prompt already over → floor 512
    assert resolve_max_tokens(s, None, 32_000, ctx=32_768) == 512


def test_prompt_over_budget():
    s = RuntimeSettings(agent_context_budget_frac=0.8)
    assert prompt_over_budget(s, 100_000, 81_000) is True
    assert prompt_over_budget(s, 100_000, 79_000) is False
    assert prompt_over_budget(s, 0, 10) is False


# --------------------------------------------------------------------------- #
# LLMClient integration                                                       #


def _chat_ok(request: httpx.Request) -> httpx.Response:
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


async def test_client_manual_ctx_no_probe(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return _chat_ok(request)

    _patch_transport(monkeypatch, handler)
    c = LLMClient(
        api_key="k",
        base_url="http://llm.test/v1",
        runtime_settings=RuntimeSettings(llm_context_tokens=99_000),
    )
    c.stream = False
    await c.chat("code", [{"role": "user", "content": "hi"}])
    assert c.context_len["code"] == 99_000
    assert "/models" not in seen  # manual override short-circuits the probe


async def test_client_cached_caps_no_probe(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return _chat_ok(request)

    _patch_transport(monkeypatch, handler)
    c = LLMClient(
        api_key="k",
        base_url="http://llm.test/v1",
        model_code="qwen",
        caps={"qwen": ModelCaps(context_tokens=64_000, source="models_api")},
        runtime_settings=RuntimeSettings(),
    )
    c.stream = False
    await c.chat("code", [{"role": "user", "content": "hi"}])
    assert c.context_len["code"] == 64_000
    assert "/models" not in seen


async def test_client_lazy_probe_clamps_max_tokens(monkeypatch):
    recorded: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen", "max_model_len": 32_768}]})
        recorded.update(json.loads(request.content))
        return _chat_ok(request)

    _patch_transport(monkeypatch, handler)
    c = LLMClient(
        api_key="k",
        base_url="http://llm.test/v1",
        model_code="qwen",
        runtime_settings=RuntimeSettings(llm_max_tokens=0),
    )
    c.stream = False
    await c.chat("code", [{"role": "user", "content": "hi"}])
    assert c.context_len["code"] == 32_768
    # auto: clamp(32768//4, 2048, 128k) = 8192
    assert recorded["max_tokens"] == 8192


async def test_client_probe_once_per_role(monkeypatch):
    hits = {"models": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            hits["models"] += 1
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "qwen", "n_ctx": 8192},
                        {"id": "qwen-v", "n_ctx": 65536},
                    ]
                },
            )
        return _chat_ok(request)

    _patch_transport(monkeypatch, handler)
    c = LLMClient(
        api_key="k",
        base_url="http://llm.test/v1",
        model_code="qwen",
        model_vision="qwen-v",
    )
    c.stream = False
    await c.chat("code", [{"role": "user", "content": "a"}])
    await c.chat("code", [{"role": "user", "content": "b"}])
    await c.chat("vision", [{"role": "user", "content": "c"}])
    # distinct models → one probe each; repeat code-role calls don't re-probe
    assert hits["models"] == 2
    assert c.context_len["code"] == 8192 and c.context_len["vision"] == 65536

    # shared model id → the second role reuses the caps cache (no probe)
    hits["models"] = 0
    c2 = LLMClient(
        api_key="k", base_url="http://llm.test/v1", model_code="qwen", model_vision="qwen"
    )
    c2.stream = False
    await c2.chat("code", [{"role": "user", "content": "a"}])
    await c2.chat("vision", [{"role": "user", "content": "b"}])
    assert hits["models"] == 1


async def test_client_learns_ctx_from_400(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                400, json={"error": {"message": "maximum context length is 8192 tokens"}}
            )
        return _chat_ok(request)

    _patch_transport(monkeypatch, handler)
    c = LLMClient(
        api_key="k", base_url="http://llm.test/v1", model_code="qwen", model_vision="qwen"
    )
    c.stream = False
    c.tools_mode = "off"
    with pytest.raises(LLMError):
        await c.chat("code", [{"role": "user", "content": "hi"}])
    assert c.caps["qwen"].context_tokens == 8192
    assert c.context_len["code"] == 8192


async def test_client_test_reports_context(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen", "max_model_len": 65_536}]})
        return _chat_ok(request)

    _patch_transport(monkeypatch, handler)
    c = LLMClient(
        api_key="k", base_url="http://llm.test/v1", model_code="qwen", model_vision="qwen"
    )
    c.stream = False
    out = await c.test()
    assert out["ok"] is True
    assert out["context"]["code"]["context_tokens"] == 65_536
    assert out["context"]["code"]["source"] == "models_api"


# --------------------------------------------------------------------------- #
# chat_worker history summarization                                           #


def test_summarize_history_digests_dropped():
    from cadx.agent.chat_worker import _summarize_history

    msgs = [
        {"role": "user", "content": "first question about the bracket"},
        {"role": "assistant", "content": "answer with some detail"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "latest"},
    ]
    out = _summarize_history(msgs, keep=2)
    # digest user msg + last 2 → total 3; digest carries the dropped content
    assert out[0]["role"] == "user"
    assert "[EARLIER CONVERSATION SUMMARY" in out[0]["content"]
    assert "first question" in out[0]["content"]
    assert out[-1]["content"] == "latest"


def test_summarize_history_boundary_and_short():
    from cadx.agent.chat_worker import _summarize_history

    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "tool_calls": [{"function": {"name": "t"}}]},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "next"},
    ]
    # keep=2 cuts before "next" boundary — digest carries q+a+result
    out = _summarize_history(msgs, keep=2)
    assert out[1]["content"] == "next"
    # tool name lands in the digest
    assert "[tools: t]" in out[0]["content"]
    # under the cap → untouched
    assert _summarize_history(msgs, keep=10) == msgs


def test_summarize_for_ctx_collapses_until_fit():
    from cadx.agent.chat_worker import _summarize_for_ctx

    msgs = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i} " + "x" * 200}
        for i in range(12)
    ]
    est = lambda m: sum(len(str(x.get("content", ""))) // 4 for x in m)  # noqa: E731
    budget = 350
    out = _summarize_for_ctx(msgs, est, budget)
    assert est(out) <= budget
    assert out[0]["role"] == "system"
    assert "[EARLIER CONVERSATION SUMMARY" in out[1]["content"]
