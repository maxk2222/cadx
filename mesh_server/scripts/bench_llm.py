#!/usr/bin/env python3
"""LLM serving benchmark — concurrency/KV-cache matrix for the sidecar.

Fires chat-completion bursts at a running mesh-server and measures per-
level: wall time, per-request latency percentiles and aggregate decode
throughput. Run the same matrix twice — e.g. `llm_kv_unified` on vs off,
or `llm_kv_cache_type` "" vs "q8_0" — by patching `/v1/config` between
runs (settings apply at the next model load; the resident model evicts
`unload_idle_s` after going idle, or restart the sidecar).

Usage:

    cd mesh_server && uv run python scripts/bench_llm.py \
        --base http://localhost:8765 --key $MESH_API_KEY \
        --model qwen3.8-27b-gguf --levels 1,4,8 --requests 8 \
        --prompt-tokens 512 --completion-tokens 128

    # repeat a fixed system prompt to exercise prompt caching:
    ... --system-repeat 2000

Output: JSON summary on stdout (append to a file for the matrix log).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

try:
    import httpx
except ImportError:
    sys.exit("needs httpx — run via `uv run` inside mesh_server/")


async def _one(client: httpx.AsyncClient, body: dict) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post("/v1/chat/completions", json=body)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "dt": dt}
        u = r.json().get("usage") or {}
        return {
            "ok": True,
            "dt": dt,
            "completion": int(u.get("completion_tokens") or 0),
            "prompt": int(u.get("prompt_tokens") or 0),
        }
    except (TimeoutError, httpx.HTTPError, OSError) as e:
        return {"ok": False, "status": str(e)[:120], "dt": time.perf_counter() - t0}


async def _level(client, body, level, n_req):
    results = await asyncio.gather(*[_one(client, body) for _ in range(n_req)])
    ok = [r for r in results if r["ok"]]
    dts = sorted(r["dt"] for r in ok)
    comp = sum(r["completion"] for r in ok)
    return {
        "level": level,
        "requests": len(results),
        "ok": len(ok),
        "errors": [r.get("status") for r in results if not r["ok"]][:5],
        "mean_latency_s": round(sum(dts) / len(dts), 2) if dts else None,
        "p50_s": round(dts[len(dts) // 2], 2) if dts else None,
        "p95_s": round(dts[int(len(dts) * 0.95)], 2) if dts else None,
        "completion_tps": round(comp / max(dts), 1) if dts and comp else None,
        "completion_tokens": comp,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8765")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--levels", default="1,4,8")
    ap.add_argument("--requests", type=int, default=8, help="requests per level")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--completion-tokens", type=int, default=128)
    ap.add_argument("--system-repeat", type=int, default=0,
                    help="repeat this many system-prompt tokens (prefix-cache probe)")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    body = {
        "model": args.model,
        "messages": [
            *([{"role": "system",
                "content": "You are a CAD assistant. " * (args.system_repeat // 5)}]
              if args.system_repeat else []),
            {"role": "user",
             "content": "Describe a flange plate. " * max(1, args.prompt_tokens // 5)},
        ],
        "max_tokens": args.completion_tokens,
        "temperature": 0.2,
    }
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(
        base_url=args.base.rstrip("/"), headers=headers, timeout=timeout
    ) as client:
        # warmup: one request (also triggers the load)
        t0 = time.perf_counter()
        w = await _one(client, body)
        out = {
            "model": args.model,
            "warmup_s": round(time.perf_counter() - t0, 1),
            "warmup_ok": w["ok"],
            "levels": [],
        }
        if not w["ok"]:
            out["error"] = w.get("status")
            print(json.dumps(out, indent=2))
            return
        for lvl in (int(x) for x in args.levels.split(",")):
            n = max(args.requests, lvl)
            t0 = time.perf_counter()
            res = await _level(client, body, lvl, n)
            res["elapsed_s"] = round(time.perf_counter() - t0, 1)
            out["levels"].append(res)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
