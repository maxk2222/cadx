"""Native `feature_edges` kernel: mesh-side fillet-band evidence.

The kernel must recover a known constant-radius fillet from its tessellation
(radius from a circle fit on the strip-boundary midpoints) and must stay
silent on sharp- and flat-edged geometry — false-positive fillet evidence is
worse than none, since verifyspec consumes it as measured features. Skips
cleanly when cadx_native or build123d is absent."""

from __future__ import annotations

import numpy as np
import pytest

b123d = pytest.importorskip("build123d")
cadx_native = pytest.importorskip("cadx_native")

from cadx.cad import native  # noqa: E402


def _tess(shape, tol: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    verts, tris = shape.tessellate(tol)
    v = np.asarray([[p.X, p.Y, p.Z] for p in verts], dtype=np.float64)
    return v, np.asarray(tris, dtype=np.int64)


class _Mesh:
    """Duck-typed stand-in — the adapter only reads vertices/faces."""

    def __init__(self, v: np.ndarray, f: np.ndarray):
        self.vertices = v
        self.faces = f


def test_fillet_bands_recover_true_radius():
    b = b123d.Box(20, 20, 20)
    f = b123d.fillet(b.edges(), 2.0)
    rows = native.feature_edges(_Mesh(*_tess(f)))
    assert rows is not None
    convex = [r for r in rows if r["side"] == "convex" and r["edge_count"] >= 10]
    # one band per filleted edge — all twelve, at the true radius
    assert len(convex) == 12
    for r in convex:
        assert r["radius"] == pytest.approx(2.0, abs=0.15)
        assert r["source"] == "mesh_edge"
        assert r["angle_deg"] > 60.0
        assert r["axis"] in "xyz"


def test_sharp_edges_report_nothing():
    for shape in (
        b123d.Box(20, 20, 20),
        b123d.chamfer(b123d.Box(20, 20, 20).edges(), 1.0),
    ):
        rows = native.feature_edges(_Mesh(*_tess(shape)))
        assert rows is not None
        # chamfers and sharp edges are not rounded surfaces — no evidence
        assert [r for r in rows if r["radius"] > 1.0] == []
