"""Heavy FEM verification lane — published references, curved-geometry
benchmarks and pitch-swept convergence studies.

Runs under `make fem-verify` (`pytest tests/femverify -m verify`); the
`verify` marker keeps them out of the normal `make test` lane — each
case is seconds-to-tens-of-seconds, not milliseconds.
"""

from __future__ import annotations

import pytest

from . import harness


def _verify_cases():
    return [harness.load_spec(p) for p in harness.case_files("verify")]


@pytest.mark.verify
@pytest.mark.parametrize("spec", _verify_cases(), ids=lambda s: s["name"])
def test_fem_benchmark_verify(spec: dict) -> None:
    if spec.get("kind") in ("trend", "convergence"):
        results = harness.evaluate_trend(spec)
    else:
        results = harness.evaluate(spec)
    failures = [r for r in results if not r.ok]
    assert not failures, (
        f"{spec['name']} failed {len(failures)}/{len(results)} probes\n"
        + harness.format_results(results)
    )
