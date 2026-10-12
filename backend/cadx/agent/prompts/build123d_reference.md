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
- Primitives are centred in ALL axes by default -> ALWAYS pass `align=(Align.CENTER, Align.CENTER, Align.MIN)` for base bodies (contract requires bottom on XY).
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
