"""Deterministic heightfield candidate for flat parts."""

from __future__ import annotations

import numpy as np
import pytest

from cadx.cad.heightfield import build_heightfield_mesh, heightfield_candidate


class _Dim:
    def __init__(self, v):
        self.value = v


class _Spec:
    def __init__(self, lx=119.0, w=34.0, h=8.0):
        self.length_x = _Dim(lx)
        self.width_y = _Dim(w)
        self.height_z = _Dim(h)


RECT = [[-60, -17], [60, -17], [60, 17], [-60, 17], [-60, -17]]


def test_build_heightfield_watertight_and_spec_shaped():
    m = build_heightfield_mesh(RECT, 8.0)
    assert m.is_watertight
    ext = m.extents
    assert ext[0] == pytest.approx(120.0, abs=2.0)
    assert ext[1] == pytest.approx(34.0, abs=2.0)
    assert ext[2] == pytest.approx(8.0, abs=2.0)
    # rests on z=0, centred in XY
    assert m.bounds[0][2] >= -0.01
    c = (m.bounds[0] + m.bounds[1]) / 2
    assert abs(c[0]) < 1.0 and abs(c[1]) < 1.0


def test_build_heightfield_rounded_outline():
    """Non-rectangular outlines are followed, not boxed."""
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    outline = [[float(30 * np.cos(a)), float(15 * np.sin(a))] for a in t]
    m = build_heightfield_mesh(outline, 6.0)
    assert m.is_watertight
    assert m.extents[0] == pytest.approx(60.0, abs=2.0)
    # ellipse area ~ pi*30*15 = 1414 — a bounding box would be ~1800
    assert m.volume / 6.0 == pytest.approx(1414, rel=0.15)


def test_heightfield_candidate_rejects_non_flat():
    spec = _Spec(lx=40.0, w=40.0, h=40.0)  # cube — not flat
    assert heightfield_candidate(spec, RECT) is None


def test_heightfield_candidate_needs_outline_and_height():
    assert heightfield_candidate(_Spec(), None) is None
    assert heightfield_candidate(_Spec(), []) is None
    spec = _Spec(h=0.0)  # no stated height
    assert heightfield_candidate(spec, RECT) is None


def test_heightfield_candidate_flat_builds():
    m = heightfield_candidate(_Spec(), RECT)
    assert m is not None
    assert m.is_watertight
    assert m.extents[2] == pytest.approx(8.0, abs=2.0)
