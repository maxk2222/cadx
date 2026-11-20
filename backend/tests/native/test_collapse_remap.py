"""Native-vs-Python parity for the boundary-stitch union-find remap."""

from __future__ import annotations

import numpy as np
import pytest

kiara_native = pytest.importorskip("kiara_native")

from kiara.cad import native  # noqa: E402


def _py_remap(pairs: set[tuple[int, int]], n: int) -> np.ndarray:
    parent = np.arange(n)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for a, b in pairs:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([_find(i) for i in range(n)], dtype=np.int64)


def test_collapse_remap_parity():
    rng = np.random.default_rng(11)
    n = 5000
    # chain + star patterns like real crack stitches produce
    pairs = {(int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(300)}
    pairs |= {(i, i + 1) for i in range(0, 100)}
    native_out = native.collapse_remap(pairs, n)
    assert native_out is not None
    np.testing.assert_array_equal(native_out, _py_remap(pairs, n))


def test_collapse_remap_empty():
    out = native.collapse_remap(set(), 10)
    np.testing.assert_array_equal(out, np.arange(10))
