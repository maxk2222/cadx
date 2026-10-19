"""Deterministic quality signal — an LLM-free progress number (0..100).

The critique score saturates fast (normalize_score floors at 0 after four
majors, and verification-answer "no"s pile on duplicates of the same real
defect), so it is a poor stall detector: a clearly improving iteration can
still score 0. ``det_quality`` instead derives progress from the executor's
``verify_spec`` conformance report (measured PASS/FAIL checks), which keeps
a usable gradient exactly where the critique flatlines.
"""

from __future__ import annotations

from typing import Any


def det_quality(exec_result: dict[str, Any] | None, dim_issues: list[str]) -> int:
    """0..100 from the executor-side verify_spec report — LLM-free progress signal."""
    if not exec_result or not exec_result.get("ok"):
        return 0
    report = exec_result.get("verify_spec")
    checks: list[dict[str, Any]] = []
    if isinstance(report, dict):
        checks = [c for c in (report.get("checks") or []) if isinstance(c, dict)]
    n_pass = sum(1 for c in checks if c.get("status") == "PASS")
    n_fail = sum(1 for c in checks if c.get("status") == "FAIL")
    # UNVERIFIED checks carry no evidence either way — ignored on purpose;
    # no report at all (older executor images) lands on a neutral 50
    base = round(100 * n_pass / max(1, n_pass + n_fail)) if checks else 50
    # structural failure caps the score: a fractured part must never read as
    # near-done no matter how many feature checks passed
    structural_fail = any(
        c.get("name") == "single_valid_solid" and c.get("status") == "FAIL" for c in checks
    )
    analysis = exec_result.get("analysis") or {}
    if (
        structural_fail
        or int(exec_result.get("solid_count") or analysis.get("solid_count") or 1) > 1
    ):
        base = min(base, 30)
    base -= 10 * len(dim_issues)
    return max(0, min(100, base))
