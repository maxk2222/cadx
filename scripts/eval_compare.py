#!/usr/bin/env python3
"""Compare eval variants: markdown table of variant x benchmark metrics.

Reads the per-run `result.json` summaries `cadx eval` writes under the eval
output root (`$DATA_DIR/eval_out`, override with CADX_EVAL_OUT or --root):

    python scripts/eval_compare.py                    # table on stdout
    python scripts/eval_compare.py --root backend/data/eval_out
    python scripts/eval_compare.py --json cmp.json    # machine-readable rows

Columns: pass = runs with passed=true; exec-ok = runs that reached at least
one sandbox execution; reasoning ctok = completion tokens on thinking-on
calls (only where the trace recorded the flag); iters = deepest improvement
iteration; wall/exec s = mean wall / in-executor seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from cadx.eval.compare import (
        aggregate,
        load_run_summaries,
        render_markdown,
        rows_to_dicts,
    )
except (
    ModuleNotFoundError
):  # running outside the backend env — fall back to source tree
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from cadx.eval.compare import (
        aggregate,
        load_run_summaries,
        render_markdown,
        rows_to_dicts,
    )


def _default_root() -> Path:
    env = os.environ.get("CADX_EVAL_OUT")
    if env:
        return Path(env)
    # DATA_DIR is "./data" resolved at run cwd: backend/data for `make eval`,
    # repo-root data for docker-style layouts — accept either
    for cand in ("data/eval_out", "backend/data/eval_out", "eval_out"):
        p = REPO_ROOT / cand
        if p.is_dir():
            return p
    return REPO_ROOT / "data" / "eval_out"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="eval output root (default: $CADX_EVAL_OUT or <repo>/data/eval_out)",
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="also write the aggregated rows to this JSON file",
    )
    ap.add_argument(
        "--benchmark",
        "-b",
        action="append",
        default=None,
        help="only include these benchmarks (repeatable)",
    )
    args = ap.parse_args()

    root = args.root or _default_root()
    summaries = load_run_summaries(root)
    if args.benchmark:
        wanted = set(args.benchmark)
        summaries = [s for s in summaries if s.get("benchmark") in wanted]
    rows = aggregate(summaries)
    if not rows:
        print(f"no eval results found under {root}", file=sys.stderr)
        return 1
    print(render_markdown(rows))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(rows_to_dicts(rows), indent=1) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
