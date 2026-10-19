#!/usr/bin/env python3
"""Collect a durable E2E report for one or more generation jobs.

Polls the API until each job is terminal, then dumps into a per-job folder:
  - job.json (JobOut incl. live.step_history)
  - version.json (VersionDetail: spec/plan/code/issues/critique/dfm/fem/mesh)
  - artifacts/* (all artifact files: renders, exports, meshes)
  - trace.tsv (per-step token/latency/iteration from Postgres agenttrace)
  - summary.md (one-page human-readable digest)

Usage:
  backend/.venv/bin/python scripts/e2e_collect.py JOB_ID [JOB_ID ...] \
      [--out evals/e2e_opt/iter01] [--timeout 5400] [--poll 15]

Postgres is read via `docker exec cadx-postgres-1 psql` (read-only queries only).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

API = "http://localhost:8080/api"
PG = ["docker", "exec", "cadx-postgres-1", "psql", "-U", "cadx", "-d", "cadx", "-t", "-A"]
TERMINAL = {"succeeded", "failed", "cancelled", "needs_review"}


def psql(sql: str) -> str:
    r = subprocess.run(
        PG + ["-c", sql], capture_output=True, text=True, timeout=60
    )
    return r.stdout.strip()


def trace_rows(job_id: str) -> list[list[str]]:
    out = psql(
        "SELECT iteration, step, prompt_tokens, completion_tokens, latency_ms, "
        "coalesce(parsed->>'thinking',''), coalesce(parsed->>'thinking_effort',''), "
        "coalesce(parsed->>'thinking_max_tokens','') "
        f"FROM agenttrace WHERE job_id='{job_id}' ORDER BY id"
    )
    return [r.split("|") for r in out.splitlines() if r.strip()]


def summarize(version: dict, job: dict, rows: list[list[str]]) -> str:
    md: list[str] = []
    md.append(f"# {version.get('id','?')[:8]} — {version.get('status')}")
    md.append("")
    md.append(f"- resolved_mode: `{version.get('resolved_mode')}`  kind: `{version.get('kind')}`")
    md.append(f"- score: **{version.get('score')}**  iterations: {version.get('iterations')}")
    live = (job.get("live") or {})
    md.append(f"- wall: {live.get('total_seconds','?'):>8}  steps: {live.get('step_history')}")
    iss = version.get("issues") or []
    md.append(f"- issues ({len(iss)}):")
    for i in iss[:30]:
        md.append(
            f"  - [{i.get('severity')}] {i.get('source')}:{i.get('kind')} — {i.get('msg','')[:160]}"
        )
    crit = version.get("critique") or {}
    md.append(f"- critique score: {crit.get('score')}  passed: {crit.get('passed')}")
    for d in (crit.get("discrepancies") or [])[:20]:
        md.append(f"  - {str(d)[:180]}")
    ptok = sum(int(r[2] or 0) for r in rows)
    ctok = sum(int(r[3] or 0) for r in rows)
    think = sum(1 for r in rows if r[5] == "True")
    md.append(f"- tokens: prompt {ptok} + completion {ctok} = {ptok+ctok}  thinking-calls: {think}")
    fem = version.get("fem")
    if fem:
        md.append(f"- fem: {json.dumps(fem)[:400]}")
    return "\n".join(md) + "\n"


async def collect(job_id: str, out: Path, timeout: int, poll: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    async with httpx.AsyncClient(base_url=API, timeout=60) as c:
        while True:
            j = (await c.get(f"/jobs/{job_id}")).json()
            if j["status"] in TERMINAL or time.time() > deadline:
                break
            await asyncio.sleep(poll)
        (out / "job.json").write_text(json.dumps(j, indent=2, default=str))
        vid = j.get("version_id") or j.get("result_version_id")
        version: dict = {}
        if vid:
            version = (await c.get(f"/versions/{vid}")).json()
            (out / "version.json").write_text(json.dumps(version, indent=2, default=str))
            adir = out / "artifacts"
            adir.mkdir(exist_ok=True)
            for a in version.get("artifacts") or []:
                try:
                    data = (await c.get(f"/artifacts/{a['id']}/file")).content
                    (adir / f"{a['kind']}_{a.get('id','x')[:8]}{Path(a.get('name','bin')).suffix or '.bin'}").write_bytes(data)
                except Exception as e:  # noqa: BLE001 — artifact fetch best-effort
                    print(f"artifact {a.get('id')} failed: {e}", file=sys.stderr)
    rows = trace_rows(job_id)
    (out / "trace.tsv").write_text(
        "iter\tstep\tptok\tctok\tms\tthinking\teffort\tmax_tok\n"
        + "\n".join("\t".join(r) for r in rows)
        + "\n"
    )
    (out / "summary.md").write_text(summarize(version, j, rows))
    print(json.dumps({
        "job": job_id[:8], "status": j["status"],
        "score": version.get("score"), "iters": version.get("iterations"),
        "mode": version.get("resolved_mode"), "out": str(out),
    }))
    return j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--poll", type=int, default=15)
    args = ap.parse_args()
    base = Path(args.out) if args.out else Path("evals/e2e_opt") / time.strftime("iter_%H%M")
    for jid in args.jobs:
        asyncio.run(collect(jid, base / jid[:8], args.timeout, args.poll))


if __name__ == "__main__":
    main()
