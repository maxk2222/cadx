## Canonical worked example (study this structure — it satisfies the contract)

Spec: "Plate 100x60x8 mm, four M4 countersunk holes inset 10 mm from the edges,
1 mm chamfer on the top perimeter, FDM printing."

```python
from build123d import *
from bd_warehouse.fastener import ClearanceHole, CounterSunkScrew

PARAMS = {
    "length": 100.0,
    "width": 60.0,
    "thickness": 8.0,
    "hole_inset": 10.0,
    "chamfer": 1.0,
    "screw_size": "M4-0.7",
}

screw = CounterSunkScrew(size=PARAMS["screw_size"], length=PARAMS["thickness"])

with BuildPart() as p:
    Box(
        PARAMS["length"],
        PARAMS["width"],
        PARAMS["thickness"],
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    top = p.faces().sort_by(Axis.Z)[-1]
    with Locations(top):
        with GridLocations(
            PARAMS["length"] - 2 * PARAMS["hole_inset"],
            PARAMS["width"] - 2 * PARAMS["hole_inset"],
            2,
            2,
        ):
            ClearanceHole(fastener=screw, depth=None)  # None = through
    chamfer(top.edges(), length=PARAMS["chamfer"])

result = p.part
```

Expected CodeResponse fields: `expected_bbox: [100, 60, 8]`, `params` mirrors
the PARAMS dict with units/descriptions, `reasoning` summarises the approach.

### Notes on this example

- `CounterSunkScrew` + `ClearanceHole` gives a norm countersunk hole — never
  hand-model countersinks with cones unless the library lacks the size.
- `Locations(top)` puts the hole plane ON the top face — the most common
  mistake is placing holes at z=0 (they then cut the bottom face or nothing).
- Every user-stated dimension lives in PARAMS and is referenced by name — never
  repeat literal numbers inside the geometry code.
- `chamfer(top.edges())` applies only to the top perimeter; for all vertical
  edges use `p.edges().filter_by(Axis.Z)`.
