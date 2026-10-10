"""Static catalog of bd_warehouse standard parts (types + sizes) for the UI and prompts."""

from __future__ import annotations

from typing import Any

_CATALOG: dict[str, Any] | None = None


def build_catalog() -> dict[str, Any]:
    """Collect fastener classes -> {type: [sizes]}. Cached."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    from bd_warehouse.bearing import SingleRowDeepGrooveBallBearing
    from bd_warehouse.fastener import Nut, Screw, Washer

    catalog: dict[str, Any] = {"screws": {}, "nuts": {}, "washers": {}, "bearings": {}}
    for base, key in ((Screw, "screws"), (Nut, "nuts"), (Washer, "washers")):
        for cls in base.__subclasses__():
            try:
                types = sorted(cls.types())
            except Exception:
                continue
            entry: dict[str, list[str]] = {}
            for t in types:
                try:
                    entry[t] = list(cls.sizes(t))
                except Exception:
                    entry[t] = []
            if entry:
                catalog[key][cls.__name__] = entry
    try:
        entry = {}
        for t in sorted(SingleRowDeepGrooveBallBearing.types()):
            entry[t] = list(SingleRowDeepGrooveBallBearing.sizes(t))
        catalog["bearings"] = {"SingleRowDeepGrooveBallBearing": entry}
    except Exception:
        catalog["bearings"] = {}
    _CATALOG = catalog
    return catalog
