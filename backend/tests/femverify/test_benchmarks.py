"""Fast FEM verification cases — exact analytical references, run in the
normal `make test` lane. Heavy sweeps live in test_benchmarks_verify.py
(`verify` marker, `make fem-verify`)."""

from __future__ import annotations

import pytest

from . import harness


def _fast_cases():
    return [harness.load_spec(p) for p in harness.case_files("fast")]


@pytest.mark.parametrize("spec", _fast_cases(), ids=lambda s: s["name"])
def test_fem_benchmark_fast(spec: dict) -> None:
    results = harness.evaluate(spec)
    failures = [r for r in results if not r.ok]
    assert not failures, (
        f"{spec['name']} failed {len(failures)}/{len(results)} probes\n"
        + harness.format_results(results)
    )
