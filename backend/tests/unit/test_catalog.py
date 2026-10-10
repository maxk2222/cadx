"""Catalog completeness + validation + lookup tests (needs bd_warehouse)."""

from __future__ import annotations

import pytest

from cadagent.cad import catalog


@pytest.fixture(scope="module", autouse=True)
def _catalog():
    return catalog.build_catalog()


def test_all_categories_present(_catalog):
    for category in (
        "screws",
        "nuts",
        "washers",
        "bearings",
        "retaining_rings",
        "o_rings",
        "shaft_keys",
        "threads",
        "gears",
        "flanges",
        "open_builds",
        "machining",
    ):
        assert _catalog.get(category), f"missing category {category}"


def test_new_classes_registered(_catalog):
    for cls in (
        "SingleRowTaperedRollerBearing",
        "ExternalSnapRing",
        "ORing",
        "ShaftKey",
        "MetricTrapezoidalThread",
        "PlasticBottleThread",
        "IsoThread",
        "SpurGear",
        "PressFitHole",
        "InsertHole",
    ):
        assert catalog.resolve_class(cls) is not None, cls


def test_normed_vs_parametric(_catalog):
    assert _catalog["screws"]["CounterSunkScrew"]["kind"] == "normed"
    assert "iso10642" in _catalog["screws"]["CounterSunkScrew"]["types"]
    assert _catalog["gears"]["SpurGear"]["kind"] == "parametric"
    assert "module" in _catalog["gears"]["SpurGear"]["params"]


def test_heat_set_nut_pairs_insert_hole(_catalog):
    assert _catalog["nuts"]["HeatSetNut"]["holes"] == ["InsertHole"]


def test_validate_part_ref():
    assert (
        catalog.validate_part_ref("CounterSunkScrew", fastener_type="iso10642", size="M4-0.7") == []
    )
    errs = catalog.validate_part_ref("CounterSunkScrew", fastener_type="iso9999", size="M4-0.9")
    assert len(errs) == 2
    assert catalog.validate_part_ref("NoSuchClass")
    # parametric classes can't be size-validated
    assert catalog.validate_part_ref("SpurGear") == []


def test_lookup_part():
    d = catalog.lookup_part("CounterSunkScrew", fastener_type="iso10642", size="M4-0.7")
    assert d["clearance_hole_diameters"]["Normal"] == pytest.approx(4.5)
    assert d["tap_hole_diameters"]["Soft"] == pytest.approx(3.3)
    with pytest.raises(ValueError):
        catalog.lookup_part("BogusClass")
    with pytest.raises(ValueError):
        catalog.lookup_part("HexNut", fastener_type="iso4032", size="M99-9")


def test_nominal_hole_diameter():
    assert catalog.nominal_hole_diameter(4) == pytest.approx(4.5)
    assert catalog.nominal_hole_diameter(4, "tap") == pytest.approx(3.3)
    assert catalog.nominal_hole_diameter(999) is None


def test_catalog_excerpt():
    out = catalog.catalog_excerpt(["CounterSunkScrew", "Bogus"])
    assert "CounterSunkScrew" in out and "iso10642" in out
    assert "NOT IN CATALOG" in out


def test_class_listing():
    out = catalog.class_listing()
    assert "screws" in out and "CounterSunkScrew" in out


def test_search():
    hits = catalog.search("iso10642")
    assert any(h["class"] == "CounterSunkScrew" for h in hits)
    assert catalog.search("") == []
