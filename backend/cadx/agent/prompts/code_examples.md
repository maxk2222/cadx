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

## Second example: measured plan view (anchor_holes + load_image_profile)

Spec: "Ergonomic grip with the same footprint and hole positions as the top
view, 8 mm thick, crowned top."

```python
from build123d import *
import cadlib

PARAMS = {
    "thickness": 8.0,
    "crown": 2.5,
    "edge_fillet": 1.5,
    "hole_dia": 5.0,
}

# measured top-view outline -> body; holes at the anchored positions
result = cadlib.profile_body(
    load_image_profile(),
    PARAMS["thickness"],
    crown_mm=PARAMS["crown"],
    edge_fillet=PARAMS["edge_fillet"],
)
for h in anchor_holes():
    # cutter BASE at z=-1 via Align.MIN — Pos() alone would centre the
    # cylinder at z=-1 and leave a membrane sealing the top of the hole
    result -= Pos(h["x"], h["y"], -1) * Cylinder(
        radius=PARAMS["hole_dia"] / 2,
        height=PARAMS["thickness"] + 2,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
```

- `load_image_profile()` raises when no top view was measured — use it only
  when the context lists "measured image features".
- `anchor_holes()` returns `{"x", "y", "dia", ...}` dicts; take `x`/`y`
  verbatim, keep your own spec `dia`.

## Third example: multi-body bracket (strut + host-bounded holes)

Spec: "90° L-bracket 120x30x100 mm — 5 mm vertical leg at −X, 5 mm top arm,
diagonal 10x6 mm brace from the leg (z=35) to the arm underside (x=+15),
two Ø5.5 holes in the arm (x=−25, x=+45), two Ø5.5 holes in the leg (z=15,
z=70)."

```python
from build123d import *
import cadlib

PARAMS = {
    "leg_t": 5.0,
    "arm_t": 5.0,
    "width": 30.0,
    "hole_d": 5.5,
    "strut_w": 10.0,
    "strut_t": 6.0,
}

# leg plate x=-60..-55, arm plate z=95..100 — both centred on Y
part = Pos(-57.5, 0.0, 50.0) * Box(PARAMS["leg_t"], PARAMS["width"], 100.0)
part = part + Pos(0.0, 0.0, 97.5) * Box(120.0, PARAMS["width"], PARAMS["arm_t"])

# diagonal brace — a straight bar between its two contact points,
# NOT a filled triangular gusset
part = part + cadlib.strut((-55, 0, 35), (15, 0, 95), PARAMS["strut_w"], PARAMS["strut_t"])

# arm holes, axis Z — workplane on the arm's TOP face (z=100, the hi side)
# and the cut bounded to the plate: an unbounded bore would keep going
# down the Z line and drill the strut beneath x=-25
with BuildPart() as _hb:
    add(part)
    with Locations(Pos(-25, 0, 100), Pos(45, 0, 100)):
        cadlib.through_hole(PARAMS["hole_d"], depth=PARAMS["arm_t"] + 0.5)
    part = _hb.part

# leg holes, axis X — Rot(0, 90, 0) maps the local +Z bore axis onto world
# +X; the workplane sits on the leg's outer face x=-55 (hi side) so the cut
# runs lo-ward through the plate only, never into the brace at x≈-14
with BuildPart() as _hb:
    add(part)
    with Locations(Pos(-55, 0, 15) * Rot(0, 90, 0), Pos(-55, 0, 70) * Rot(0, 90, 0)):
        cadlib.through_hole(PARAMS["hole_d"], depth=PARAMS["leg_t"] + 0.5)
    part = _hb.part

result = part
```

- The strut is `strut(p0, p1, w, t)` between the spec's contact points —
  never a swept/filled triangle.
- Both hole groups sit on their host plate's HI face and cut `thickness +
  0.5` deep — the bore pierces its own plate fully but stops before any
  other body on the same axis line.
- `Rot(0, 90, 0)` on the Pos maps local +Z (the cutter direction) onto +X —
  the pattern for every `axis="x"` bore; `axis="y"` uses `Rot(-90, 0, 0)`.
