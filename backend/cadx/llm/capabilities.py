"""Model capability detection: context window (and max output) probing.

Trust order for the context length of a model:
1. explicit user setting ``llm_context_tokens`` (handled by the caller)
2. provider APIs — ``GET {base}/models`` (vLLM ``max_model_len``, OpenRouter
   ``context_length``, sidecar ``context_len``), then native admin endpoints
   derived from the base URL (LM Studio ``/api/v1/models`` +
   ``/api/v0/models``, llama.cpp ``/props``, Ollama ``/api/show``)
3. provider error bodies naming the limit (``parse_ctx_from_error`` —
   llama.cpp ``n_ctx``, vLLM/OpenAI "maximum context length is N")
4. a compact static name table for providers that report nothing (OpenAI
   cloud and friends)

Everything here is best-effort and non-fatal: a probe that fails leaves the
model "unknown" and the caller keeps its heuristic defaults. Detected caps
persist in the ``AppSetting`` row ``llm_caps`` so restarts skip the probe.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from cadx.logging import get_logger

log = get_logger(__name__)

_PROBE_TIMEOUT = 5.0
_CTX_MIN = 1024
_CTX_MAX = 8_000_000
# completion fallback when no context info is known at all (the historical
# llm_max_tokens default)
LEGACY_MAX_TOKENS = 8000
# auto llm_max_tokens cap ("auto default max 128k")
_AUTO_MAX_TOKENS = 128_000


@dataclass
class ModelCaps:
    """Detected capabilities of one model id."""

    context_tokens: int | None = None
    max_output_tokens: int | None = None
    source: str = ""  # models_api|lmstudio|llamacpp|ollama|sidecar|error|static|manual
    seen_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return {
            "ctx": self.context_tokens,
            "max_out": self.max_output_tokens,
            "source": self.source,
            "ts": int(self.seen_at),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ModelCaps:
        return cls(
            context_tokens=_as_int(d.get("ctx")),
            max_output_tokens=_as_int(d.get("max_out")),
            source=str(d.get("source") or ""),
            seen_at=float(d.get("ts") or 0),
        )


def _as_int(v: Any) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if _CTX_MIN <= n <= _CTX_MAX else None


def _clamp_ctx(v: Any) -> int | None:
    return _as_int(v)


# --------------------------------------------------------------------------- #
# /models parsing (OpenAI `data[]` and sidecar management `models[]` shapes)   #

_CTX_FIELDS = (
    "max_model_len",  # vLLM (null on LoRA cards — _as_int drops them)
    "context_length",  # OpenRouter
    "max_context_length",  # LM Studio (legacy listing)
    "context_window",
    "max_input_tokens",
    "context_len",  # cadx mesh-server sidecar (ModelInfo)
    "n_ctx_train",
    "n_ctx",
)


def caps_from_card(m: dict[str, Any]) -> ModelCaps | None:
    """Public alias of the model-card parser — for callers that list models
    themselves (settings router) instead of probing."""
    return _caps_from_model_card(m)


def _caps_from_model_card(m: dict[str, Any]) -> ModelCaps | None:
    for k in _CTX_FIELDS:
        if ctx := _clamp_ctx(m.get(k)):
            caps = ModelCaps(context_tokens=ctx, source="models_api")
            tp = m.get("top_provider")
            if isinstance(tp, dict):
                caps.max_output_tokens = _as_int(tp.get("max_completion_tokens"))
            return caps
    return None


def _native_base(base_url: str) -> str:
    """Strip the trailing /v1 (and /api) off an OpenAI base to reach the
    provider's native endpoints (LM Studio /api/v1, Ollama /api, llama.cpp)."""
    b = base_url.rstrip("/")
    for suffix in ("/v1", "/api"):
        if b.endswith(suffix):
            b = b[: -len(suffix)]
    return b


def _match_model(entry: dict[str, Any], model_ids: set[str]) -> str | None:
    """Return the requested id a listing entry corresponds to."""
    for key in ("id", "key", "model", "name"):
        v = str(entry.get(key) or "")
        if v in model_ids:
            return v
    return None


# --------------------------------------------------------------------------- #
# static name table — providers that report nothing (OpenAI cloud, Anthropic-  #
# style ids surfaced through proxies, common HF repo names)                    #

_STATIC_CTX: tuple[tuple[str, int], ...] = (
    (r"gpt-5", 400_000),
    (r"gpt-4\.1", 1_000_000),
    (r"gpt-4o|gpt-4-turbo|gpt-4-1106|gpt-4-0125", 128_000),
    (r"gpt-4-32k|gpt-4-0613", 32_768),
    (r"gpt-4\b", 8_192),
    (r"gpt-3\.5", 16_385),
    (r"o1|o3|o4", 200_000),
    (r"claude", 200_000),
    (r"gemini-.*pro|gemini-2", 1_000_000),
    (r"gemini", 1_000_000),
    (r"qwen3|qwen-3|Qwen3", 262_144),
    (r"qwen2\.5|qwen-2\.5", 131_072),
    (r"qwen2|qwen-2", 131_072),
    (r"deepseek.*v3|deepseek.*r1", 65_536),
    (r"deepseek", 64_000),
    (r"llama-3|llama3", 131_072),
    (r"llama-2|codellama", 16_384),
    (r"mistral-large|mistral-medium", 131_072),
    (r"mistral|mixtral", 32_768),
    (r"phi-4", 16_384),
    (r"glm-4", 131_072),
    (r"gpt-image|dall-e", 0),  # not chat models — marks known-not-llm
)
_STATIC_RES = tuple((re.compile(p, re.I), n) for p, n in _STATIC_CTX)


def _static_ctx(model_id: str) -> int | None:
    for rx, n in _STATIC_RES:
        if rx.search(model_id):
            return n or None
    return None


# --------------------------------------------------------------------------- #
# probing                                                                     #


async def probe_context(  # noqa: C901 — provider probing is inherently a chain
    base_url: str,
    api_key: str,
    model_ids: list[str],
    *,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, ModelCaps]:
    """Detect context windows for `model_ids`. Non-fatal; unknown models are
    simply absent from the result (plus static-table entries which always
    resolve)."""
    wanted = set(model_ids)
    out: dict[str, ModelCaps] = {}
    headers = {"Content-Type": "application/json", **(extra_headers or {})}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    base = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            # 1) OpenAI /models (also covers the sidecar management listing,
            #    which uses `models` + `context_len` instead of `data`)
            try:
                r = await client.get(f"{base}/models", headers=headers)
                if r.status_code == 200:
                    body = r.json()
                    entries = body.get("data") or body.get("models") or []
                    for m in entries:
                        if not isinstance(m, dict):
                            continue
                        mid = _match_model(m, wanted)
                        if mid and mid not in out and (caps := _caps_from_model_card(m)):
                            if m.get("context_len") is not None:
                                caps.source = "sidecar"
                            out[mid] = caps
            except Exception:
                pass

            missing = wanted - set(out)
            if not missing:
                return out

            native = _native_base(base)
            # 2) LM Studio native (v1 first, then v0): configured ctx of the
            #    loaded instance beats the model's architectural maximum
            for path in ("/api/v1/models", "/api/v0/models"):
                if not missing:
                    break
                try:
                    r = await client.get(f"{native}{path}", headers=headers)
                    if r.status_code != 200:
                        continue
                    body = r.json()
                    entries = body.get("models") or body.get("data") or []
                    for m in entries:
                        if not isinstance(m, dict):
                            continue
                        mid = _match_model(m, missing)
                        if not mid:
                            continue
                        ctx: int | None = None
                        for inst in m.get("loaded_instances") or []:
                            cfg = inst.get("config") if isinstance(inst, dict) else None
                            if isinstance(cfg, dict):
                                ctx = _clamp_ctx(cfg.get("context_length")) or ctx
                        ctx = ctx or _clamp_ctx(m.get("max_context_length"))
                        if ctx:
                            out[mid] = ModelCaps(context_tokens=ctx, source="lmstudio")
                            missing.discard(mid)
                except Exception:
                    continue

            # 3) llama.cpp server props — applies to whichever model is loaded
            if missing:
                try:
                    r = await client.get(f"{native}/props", headers=headers)
                    if r.status_code == 200:
                        dgs = r.json().get("default_generation_settings") or {}
                        ctx = _clamp_ctx(dgs.get("n_ctx"))
                        if ctx:
                            for mid in list(missing):
                                out[mid] = ModelCaps(context_tokens=ctx, source="llamacpp")
                            missing.clear()
                except Exception:
                    pass

            # 4) Ollama /api/show — per model, model_info.<arch>.context_length
            for mid in list(missing):
                try:
                    r = await client.post(f"{native}/api/show", json={"name": mid}, headers=headers)
                    if r.status_code != 200:
                        continue
                    ctx = _deep_ctx(r.json().get("model_info"))
                    if ctx:
                        out[mid] = ModelCaps(context_tokens=ctx, source="ollama")
                        missing.discard(mid)
                except Exception:
                    continue
    except Exception:
        pass

    # 5) static name table — last resort for opaque providers
    for mid in wanted - set(out):
        if ctx := _static_ctx(mid):
            out[mid] = ModelCaps(context_tokens=ctx, source="static")
    return out


def _deep_ctx(node: Any) -> int | None:
    """Ollama model_info nests per-arch keys: llama.context_length etc."""
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k).endswith("context_length") and (ctx := _clamp_ctx(v)):
                return ctx
        for v in node.values():
            if ctx := _deep_ctx(v):
                return ctx
    return None


# --------------------------------------------------------------------------- #
# error-body learning                                                         #

_ERR_RES = (
    re.compile(r"maximum context length is (\d+)", re.I),
    re.compile(r"context length is (\d+)", re.I),
    re.compile(r"context size of (\d+)", re.I),
    re.compile(r"n_ctx[^0-9]{0,20}(\d+)", re.I),
)


def parse_ctx_from_error(status: int, body: str) -> int | None:
    """Extract the declared context limit from a provider error response.

    Covers vLLM/OpenAI ("maximum context length is N tokens") and llama.cpp
    ("exceeds the available context size", JSON `n_ctx`)."""
    if status not in (400, 413, 422) and "context" not in body.lower():
        return None
    try:
        data = json.loads(body)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            if ctx := _clamp_ctx(err.get("n_ctx")):
                return ctx
            body = str(err.get("message") or body)
    except Exception:
        pass
    for rx in _ERR_RES:
        if (m := rx.search(body)) and (ctx := _clamp_ctx(m.group(1))):
            return ctx
    return None


# --------------------------------------------------------------------------- #
# prompt-size estimation + budget resolution                                  #

_CHARS_PER_TOKEN = 4  # conservative for code/JSON-heavy prompts


def estimate_prompt_tokens(
    messages: list[dict[str, Any]], *, image_detail: str = "high", image_max_edge: int = 1568
) -> int:
    """Heuristic prompt-token estimate from the wire payload (no tokenizer).

    Text counts len//4; images use the OpenAI tiling formula on the capped
    edge: low detail = 85, high detail = 85 + 170·tiles where tiles =
    ceil(edge/512)² for a square-ish worst case."""
    if image_detail == "low":
        img_cost = 85
    else:
        tiles = max(1, -(-min(image_max_edge, 2048) // 512))
        img_cost = 85 + 170 * tiles * tiles
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // _CHARS_PER_TOKEN
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    total += len(str(part)) // _CHARS_PER_TOKEN
                elif part.get("type") == "image_url":
                    total += img_cost
                else:
                    total += len(str(part.get("text") or "")) // _CHARS_PER_TOKEN
        if m.get("tool_calls"):
            total += len(json.dumps(m["tool_calls"])) // _CHARS_PER_TOKEN
    # structural overhead per message (role, wrappers)
    return total + 4 * len(messages)


def resolve_context(settings: Any, caps: ModelCaps | None) -> int:
    """Effective context window: manual override > detected caps > 0."""
    manual = int(getattr(settings, "llm_context_tokens", 0) or 0)
    if manual > 0:
        return manual
    return int(caps.context_tokens) if caps and caps.context_tokens else 0


def resolve_max_tokens(
    settings: Any, caps: ModelCaps | None, est_prompt: int, *, ctx: int = 0
) -> int:
    """Effective completion budget for one call.

    ``llm_max_tokens == 0`` means auto: ``clamp(ctx//4, 2048, 128k)``. An
    explicit value is a hard cap. Either way the result is clamped so
    prompt + completion fits the context window (headroom = max(1024, ctx/16)).
    With unknown ctx the configured (or legacy 8000) value stands."""
    ctx = ctx or resolve_context(settings, caps)
    cfg = int(getattr(settings, "llm_max_tokens", 0) or 0)
    if ctx <= 0:
        return cfg or LEGACY_MAX_TOKENS
    auto = cfg if cfg > 0 else max(2048, min(_AUTO_MAX_TOKENS, ctx // 4))
    headroom = max(1024, ctx // 16)
    return max(512, min(auto, ctx - est_prompt - headroom))


def prompt_over_budget(settings: Any, ctx: int, est_prompt: int) -> bool:
    """Whether the estimated prompt exceeds the usable fraction of ctx."""
    if ctx <= 0:
        return False
    frac = float(getattr(settings, "agent_context_budget_frac", 0.8) or 0.8)
    return est_prompt > ctx * frac


# --------------------------------------------------------------------------- #
# DB-backed capability cache (AppSetting row "llm_caps")                        #

_CAPS_KEY = "llm_caps"


async def load_llm_caps(db: Any) -> dict[str, ModelCaps]:
    """Read the cached capability map; empty on any error (cache, not config)."""
    try:
        from cadx.db.models import AppSetting

        row = await db.get(AppSetting, _CAPS_KEY)
        models = (row.value or {}).get("models") if row else None
        if not isinstance(models, dict):
            return {}
        return {k: ModelCaps.from_json(v) for k, v in models.items() if isinstance(v, dict)}
    except Exception:
        return {}


async def record_llm_caps(db: Any, model_id: str, caps: ModelCaps) -> None:
    """Merge one model's caps into the cache row. Never raises — the cache is
    nice-to-have, the in-memory copy already covers this process."""
    try:
        from cadx.db.models import AppSetting

        row = await db.get(AppSetting, _CAPS_KEY)
        value = dict(row.value) if row and isinstance(row.value, dict) else {}
        models = dict(value.get("models") or {})
        existing = models.get(model_id)
        if (
            isinstance(existing, dict)
            and _as_int(existing.get("ctx")) == caps.context_tokens
            and _as_int(existing.get("max_out")) == caps.max_output_tokens
        ):
            return  # unchanged — skip the write
        models[model_id] = caps.to_json()
        value["models"] = models
        if row is None:
            db.add(AppSetting(key=_CAPS_KEY, value=value))
        else:
            row.value = value
        await db.flush()
    except Exception:
        return


async def load_caps_session() -> dict[str, ModelCaps]:
    """load_llm_caps with its own DB session — for callers that only need
    the cache once (LLM construction)."""
    try:
        from cadx.db.session import session_scope

        async with session_scope() as db:
            return await load_llm_caps(db)
    except Exception:
        return {}


async def record_caps_session(model_id: str, caps: ModelCaps) -> None:
    """LLMClient caps_cb: persist a new detection in its own committed
    session — the worker's ambient session must not be touched mid-loop."""
    try:
        from cadx.db.session import session_scope

        async with session_scope() as db:
            await record_llm_caps(db, model_id, caps)
            await db.commit()
    except Exception:
        return
