## build123d 0.13 reference (verified idioms)

`from build123d import *` gives everything. Two equivalent styles:

### Builder mode
```python
with BuildPart() as p:
    Box(100, 60, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with GridLocations(80, 40, 2, 2):  # x_spacing, y_spacing, nx, ny
        Hole(2.25)  # RADIUS, depth=None -> through part (workplane normal)
    fillet(p.edges().filter_by(Axis.Z), radius=3)
result = p.part
```

```python
# Sketch on a face -> cut a pocket
with BuildPart() as p:
    Box(80, 40, 15, align=(Align.CENTER, Align.CENTER, Align.MIN))
    top = p.faces().sort_by(Axis.Z)[-1]
    with BuildSketch(top) as s:
        RectangleRounded(60, 25, radius=4)
    extrude(amount=-5, mode=Mode.SUBTRACT)  # negative -> into the part

# Slots / text / patterns
with BuildPart() as p:
    Box(80, 40, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with BuildSketch(p.faces().sort_by(Axis.Z)[-1]):
        SlotOverall(40, 8)  # or SlotCenterPoint / SlotArc
        with GridLocations(0, 25, 1, 2):
            SlotOverall(30, 6)
    extrude(amount=-10, mode=Mode.SUBTRACT)

# Revolve a profile (knob, bushing, pipe adapter)
with BuildPart() as p:
    with BuildSketch(Plane.XZ) as s:
        with BuildLine() as l:
            Polyline((0, 0), (10, 0), (10, 20), (6, 20), (6, 30), (0, 30), close=True)
        make_face()
    revolve(axis=Axis.Z)

# Shell (open box with walls)
with BuildPart() as p:
    Box(60, 40, 30, align=(Align.CENTER, Align.CENTER, Align.MIN))
    offset(amount=-2, openings=p.faces().sort_by(Axis.Z)[-1])

# Chamfer / counterbore (generic, without a named fastener)
chamfer(p.edges().filter_by(Axis.Z), length=1)
with Locations(p.faces().sort_by(Axis.Z)[-1]):
    CounterBoreHole(radius=2.2, counter_bore_radius=4, counter_bore_depth=4)
```

### Algebra mode (position + boolean)
```python
body = Box(50, 30, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
body -= Pos(0, 0, 5) * Cylinder(4, 20)
body += Pos(20, 0, 0) * Cylinder(5, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
body = fillet(body.edges().filter_by(Axis.Z), 2)
result = body
```

### 2D flat part (also exposes the laser/CNC profile)
```python
with BuildSketch() as s:
    RectangleRounded(100, 50, 5)
    with GridLocations(80, 30, 2, 2):
        Circle(2.5, mode=Mode.SUBTRACT)
flat_profile = s.sketch
result = extrude(s.sketch, amount=3)
```

### Selection cheats
- faces: `p.faces().sort_by(Axis.Z)[-1]` (top) `[0]` (bottom); `filter_by(Plane.XY)`;
  `filter_by(Axis.X)` (normal along X); `sort_by(SortBy.AREA)[-1]` (largest).
- edges: `p.edges().filter_by(Axis.Z)` (parallel to Z); `group_by(Axis.Z)[-1]` (highest);
  `sort_by(SortBy.LENGTH)[-1]`; `filter_by(GeomType.LINE)` / `GeomType.CIRCLE`.
- locations: `Locations(face)`, `GridLocations(x, y, nx, ny)`, `PolarLocations(r, n, start_angle=0, rotate=True)`, `HexLocations`, `Pos(x,y,z)`, `Rot(angle=a, about=Axis.Z)`, `Plane.XZ`, `Plane.YZ`.

### PITFALLS (fix these first when a run fails)
- `Pos(x, y, z) * Box(...)` as a BARE STATEMENT inside `with BuildPart():` adds
  the primitive at the ORIGIN and discards the moved copy — the part silently
  keeps a body where you never wanted it. Scoped instead:
  `with Locations(Pos(x, y, z)): Box(...)` — or `add(Pos(...) * Box(...))`.
  Same trap outside builders: the expression's result is discarded unless you
  assign it or pass it to a boolean.
- `p.faces().sort_by(Axis.Z)[-1]` is the HIGHEST face — on an L-shape that's
  the leg's top, NOT the base's top. Pick by position:
  `p.faces().filter_by(Axis.Z).group_by(Axis.Z)` and choose the group at the
  height you need (or place cuts absolutely with `Pos` + a cutter).
- Inner corner edges (e.g. where two plates meet) run along the joint's
  direction — an L-bracket's inside edge runs along Y. Select by direction
  AND position: `p.edges().filter_by(Axis.Y)` then pick by `e.center()`.
  NOTE `e.center()` is a METHOD — `e.center.X` raises AttributeError.
- `ShapeList` has `filter_by()` / `sort_by()` / `group_by()` — NOT the Python
  `filter()`. `p.edges().filter(lambda e: ...)` raises AttributeError; use a
  list comprehension `[e for e in p.edges() if ...]` instead.
- Oblong slots: never fillet a box to round its ends (r == half-width always
  crashes "Failed creating a fillet"). Use `cadlib.oblong_slot(width, length)`
  or a `SlotOverall(w, l)` sketch + `extrude(amount=-depth, mode=SUBTRACT)`.
- Bolt circle / Lochkreis (PCD): `with PolarLocations(radius, count): Hole(r)`
  inside BuildPart, or place cutters at `(r*cos(2πi/n), r*sin(2πi/n), z)`.
- `p.part += x` inside `with BuildPart() as p:` works but bypasses the
  builder — prefer `add(x)` (or `add(x, mode=Mode.SUBTRACT)` for cuts); it
  keeps pending edges/sketches consistent.
- Primitives are centred in ALL axes by default -> ALWAYS pass `align=(Align.CENTER, Align.CENTER, Align.MIN)` for base bodies (contract requires bottom on XY).
- `Pos(x, y, z) * Cylinder(r, h)` puts the cylinder's CENTRE at (x, y, z), not its
  base — `Pos(x, y, -1) * Cylinder(r, h)` therefore stops short and leaves a thin
  membrane sealing the hole (renders "look like" holes but nothing goes through).
  For a through-hole either pass `align=(Align.CENTER, Align.CENTER, Align.MIN)`
  so z is the cutter base, centre the cutter at the part's mid-height with extra
  height, or use `cadlib.through_hole(d)` inside a `Locations` scope.
- `Hole(radius)` not diameter. `CounterSinkHole(radius, counter_sink_radius, counter_sink_angle)` also uses radii.
- Custom countersink Ø`d`×`depth` opening at the TOP face (z_top) — subtract
  `Pos(x, y, z_top - depth/2) * Cone(bottom_radius=r, top_radius=d/2, height=depth)`:
  the cone CENTER sits half-depth below the surface, the WIDE radius is at the TOP.
  `Pos(z=z_top-depth)*Cone(bottom_radius=d/2, top_radius=r)` (wide end at the
  bottom, centered on the surface) is wrong — it never reaches the surface.
- Cutting a pocket from a face sketch: `extrude(amount=-depth, mode=Mode.SUBTRACT)` — the face's workplane normal points outward, so a positive amount goes outward.
- `fillet(edges, radius)` raises `BRep_API: command not done` when radius is too large or the edge selection is empty/wrong -> reduce radius or select edges BEFORE the fillet you already applied (edge indices change after every operation).
- `BuildSketch(face)` draws on the face's plane with origin at the face CENTRE.
- Side profiles use `Plane.XZ` / `Plane.YZ`, not the default XY.
- In `BuildSketch`, `mode=Mode.SUBTRACT` creates a cutout inside the sketch.
- `result` must be `p.part` (or a `Part`), not the builder `p`.
- Do NOT call export functions — the runner handles exports.
- `Text("ABC", font_size=8)` only inside a BuildSketch.
- `import cadquery` / `cq_warehouse` does NOT exist here -> always build123d / bd_warehouse.
- Threads as real helixes are expensive — see bd_warehouse reference for the fast path.
