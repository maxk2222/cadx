"""compare.py — trusted geometry comparison (diffs, fit, alignment)."""

from __future__ import annotations

import pytest
from build123d import Box, Cylinder, Pos

from cadx.cad.compare import (
    align_check,
    diff_summary,
    fit_check,
    shape_compare,
)


def test_shape_compare_overlapping_boxes():
    """Two 10^3 boxes offset +5 on X: overlap 5x10x10, added=removed=500."""
    a = Box(10, 10, 10)  # x in [-5,5]
    b = Pos(5, 0, 0) * Box(10, 10, 10)  # x in [0,10]
    r = shape_compare(a, b)
    assert "error" not in r
    assert r["boolean"] == "ok"
    assert r["volume_a"] == pytest.approx(1000.0)
    assert r["volume_b"] == pytest.approx(1000.0)
    assert r["volume_delta"] == pytest.approx(0.0)
    # b-a is the x in [5,10] slab, a-b the x in [-5,0] slab
    assert r["added_volume"] == pytest.approx(500.0)
    assert r["removed_volume"] == pytest.approx(500.0)
    assert r["added_bbox"]["min"][0] == pytest.approx(5.0)
    assert r["removed_bbox"]["max"][0] == pytest.approx(0.0)
    assert r["solids_a"] == r["solids_b"] == 1
    assert r["bbox_delta"]["min"][0] == pytest.approx(5.0)
    assert r["bbox_delta"]["max"][0] == pytest.approx(5.0)


def test_shape_compare_grown_box():
    a = Box(10, 10, 10)  # z in [-5,5]
    b = Box(10, 10, 20)  # same footprint, z in [-10,10], contains a
    r = shape_compare(a, b)
    assert r["boolean"] == "ok"
    # added = the two 5 mm slabs above/below a: 2x(10x10x5); removed = 0 (a in b)
    assert r["added_volume"] == pytest.approx(1000.0)
    assert r["removed_volume"] == pytest.approx(0.0, abs=1e-4)
    assert r["volume_delta"] == pytest.approx(1000.0)


def test_shape_compare_identical():
    a = Box(10, 10, 10)
    b = Box(10, 10, 10)
    r = shape_compare(a, b)
    assert r["added_volume"] == pytest.approx(0.0, abs=1e-4)
    assert r["removed_volume"] == pytest.approx(0.0, abs=1e-4)


def test_fit_check_part_inside_housing():
    """A 16^3 part inside the 20^3 cavity of a 30^3 housing: 2 mm wall gap."""
    housing = Box(30, 30, 30) - Box(20, 20, 20)
    part = Box(16, 16, 16)
    r = fit_check(part, housing)
    assert "error" not in r
    assert r["fits"] is True
    assert r["interference_volume"] == pytest.approx(0.0)
    assert r["min_distance"] == pytest.approx(2.0)
    # tighter than the actual 2 mm gap -> no fit
    r2 = fit_check(part, housing, clearance_mm=3.0)
    assert r2["fits"] is False
    # looser -> still fits
    assert fit_check(part, housing, clearance_mm=1.5)["fits"] is True


def test_fit_check_intersecting_boxes():
    """Part sticking through the housing wall: real clash volume."""
    housing = Box(30, 30, 30) - Box(20, 20, 20)
    part = Pos(12, 0, 0) * Box(16, 16, 16)  # x in [4,20], wall starts at x=10
    r = fit_check(part, housing)
    assert r["fits"] is False
    # clash = part & housing material = x in [10,15]x16x16 = 1280 mm^3
    assert r["interference_volume"] == pytest.approx(1280.0)
    assert r["interference_bbox"]["min"][0] == pytest.approx(10.0)
    assert r["interference_bbox"]["max"][0] == pytest.approx(15.0)


def test_fit_check_embedded_is_not_a_fit():
    """Solid box fully inside solid box = maximal interference, not a fit."""
    r = fit_check(Box(4, 4, 4), Box(20, 20, 20))
    assert r["fits"] is False
    assert r["interference_volume"] == pytest.approx(64.0)
    # but containment diagnostics still answer "is a inside b"
    assert r["a_volume_outside_b"] == pytest.approx(0.0, abs=1e-4)


def test_fit_check_apart():
    r = fit_check(Box(5, 5, 5), Pos(20, 0, 0) * Box(5, 5, 5))
    assert r["fits"] is True
    assert r["interference_volume"] == pytest.approx(0.0)
    assert r["min_distance"] == pytest.approx(15.0)


def test_align_check_coaxial_cylinders():
    c1 = Cylinder(radius=5, height=20)  # Z axis, com z=0
    c2 = Pos(0, 0, 30) * Cylinder(radius=8, height=10)  # same axis, com z=30
    r = align_check(c1, c2)
    assert "error" not in r
    assert r["coaxial"] is True
    assert r["axes"]["Z"]["com_offset"] == pytest.approx(-30.0)
    assert r["axes"]["X"]["com_offset"] == pytest.approx(0.0, abs=1e-3)


def test_align_check_off_axis_cylinders():
    c1 = Cylinder(radius=5, height=20)
    c2 = Pos(3, 0, 0) * Cylinder(radius=5, height=20)  # parallel, offset 3 mm
    r = align_check(c1, c2)
    assert r["coaxial"] is False
    assert "perpendicular" in r["coaxial_reason"]
    assert r["axes"]["X"]["com_offset"] == pytest.approx(-3.0)


def test_align_check_boxes_not_axisymmetric():
    r = align_check(Box(10, 20, 30), Pos(2, 0, 0) * Box(10, 20, 30))
    assert r["coaxial"] is False
    assert "axisymmetric" in r["coaxial_reason"]
    assert r["axes"]["X"]["com_offset"] == pytest.approx(-2.0)
    assert r["axes"]["X"]["min_face_delta"] == pytest.approx(-2.0)


def test_align_check_bad_axis():
    r = align_check(Box(1, 1, 1), Box(1, 1, 1), axes="XQ")
    assert "error" in r


def test_diff_summary_smoke():
    a = Box(10, 10, 10)
    b = Pos(5, 0, 0) * Box(10, 10, 10)
    s = diff_summary(a, b)
    assert "+500" in s and "added" in s
    assert "-500" in s and "removed" in s


def test_defensive_on_bad_input():
    r = shape_compare(object(), Box(1, 1, 1))
    assert "error" in r
    r2 = fit_check(object(), Box(1, 1, 1))
    assert "error" in r2 or r2["fits"] is None
