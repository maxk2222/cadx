## CODE CONTRACT — rules for the Python file you write

1. Only these imports are allowed: `build123d` (incl. `from build123d import *`),
   `bd_warehouse.*` (fastener, thread, gear, bearing, bushing, o_rings, retaining_ring,
   shaft_key, sprocket, pipe, flange, open_builds), `cadlib`, `math`, `numpy`,
   `typing`, `dataclasses`. Prefer `cadlib` helpers (see cadlib reference) for
   common features instead of hand-rolled geometry.
   No file I/O, no network, no exports, no `open()`/`eval()`/`exec()`. `print()` only for
   short debug values.
2. At the top of the file: a dict literal named `PARAMS` with every tunable value:
   ```python
   PARAMS = {
       "length": 120.0,  # mm, overall length along X
       "screw": "M4-0.7",  # fastener size string (str params allowed)
   }
   ```
   Use `PARAMS["name"]` everywhere below. Never hard-code a dimension twice.
3. Coordinate frame: X = length (left->right), Y = width (+Y is the "front" face),
   Z = height (up). The part must rest on the XY plane (min Z = 0) and be centred on
   the origin in X and Y. Use `align=(Align.CENTER, Align.CENTER, Align.MIN)` on
   primitives to achieve this. Units are millimetres.
4. Build the part with build123d (builder mode `with BuildPart() as p:` or algebra mode).
   At the end assign: `result = p.part` — a build123d Part/Solid/Compound with exactly
   ONE solid unless the task is an assembly.
   Optional for flat 2D-CNC/laser parts: `flat_profile = <Sketch or Face>`.
   Positioning rules (violating them silently breaks geometry):
   - `shape.translate(...)`, `shape.rotate(...)`, `shape.moved(...)` RETURN a NEW
     object — they do NOT move the original. A bare `Box(...).translate(...)`
     statement leaves the box at the origin. Position primitives with
     `Pos(x, y, z) * Box(...)` (algebra) or `Locations`/`locate()` (in-place).
   - Inside `with BuildPart()`, NEVER call `.locate()`/`.move()` on a builder
     primitive — the builder registered its own copy at the origin, so the
     fused part keeps the un-moved shape. Use `with Locations(Pos(...)):`
     or `Pos(...) * Box(...)` inside the builder instead.
   - Rotating a `Cylinder`/`Cone` about `Axis.Z` is a no-op (its own axis). To
     lay a cylinder over, rotate about X or Y — or build it directly on a
     rotated `Plane`.
   - Booleans only work on 3D solids: `part - Ellipse(...)` or
     `mesh.cut(Pos * Ellipse(...))` are invalid — `Ellipse`/`Circle`/`Rectangle`
     are 2D curves. Use `Cylinder`/`Box`/`extrude` tools instead.
   - `Rot` takes `(x, y, z)` Euler angles, `(axis=Axis.Y, angle=90)` or
     `(Axis.Y, 90)` — `Rot(a=90, about=Axis.Y)` raises "Unexpected keyword
     arguments". Same discipline for every constructor: check the signature,
     don't invent kwargs.
   - `removeSplitter`/`unify`/`makeSolid` do NOT exist (CadQuery/OpenSCAD
     idioms). To merge coplanar faces left over by boolean ops use
     `part = part.clean()` — or simply omit it.
   - Bounding-box extents: `bb = part.bounding_box()` returns a BoundBox;
     `bb.min`/`bb.max`/`bb.size` are Vectors with UPPERCASE `.X`/`.Y`/`.Z`
     (or `[0]`/`[1]`/`[2]` indexing). `boundingBox()`, `bb.xlen`, `bb.xmax`,
     `bb.size.x` do NOT exist — prefer `cadlib.bbox_dims(part) -> [dx, dy, dz]`.
   - Ribs/gussets must live INSIDE the part envelope: place them flush with or
     recessed from the outer faces, never proud of the bounding box in the spec.
     A rib that extends past the side wall silently inflates the overall
     width/length and fails the dimensional check. When the spec says "rib
     between wall A and wall B", the rib's outer face goes ON the inner edge of
     the wall, not past it. A rib must also OVERLAP the parent body so the fuse
     joins it — a rib merely touching a face (or separated by a gap) leaves a
     second disconnected solid and fails the one-solid check.
   - Cutters must pierce along the face NORMAL of the surface they cut. A hole
     or slot through a vertical wall (thin in X or Y): sketch the profile in a
     plane PARALLEL to that wall (e.g. `Plane.YZ`/`Plane.XZ` or via
     `BuildSketch(plane)`) and extrude THROUGH the wall thickness — never
     extrude the top-view profile along Z and reposition it; that produces a
     stubby pocket, not a through slot. `cadlib.oblong_slot` handles the
     vertical case directly: `oblong_slot(6.5, 15, direction="+z", normal="+x")`
     cuts a slot running along Z through a wall thin in X — prefer it over
     hand-rolled sketches whenever the axes fit.
   - Axes to faces: a hole/slot with `axis="x"` opens on the ±X faces — the
     bore line runs along X and the cutter frame must map local +Z onto X
     (`Pos(x, y, z) * Rot(0, 90, 0)` or an equivalent `Plane`); `axis="y"`
     maps onto Y (`Rot(-90, 0, 0)`); `axis="z"` is the un-rotated default.
     When a sketch on `Plane.XZ`/`Plane.YZ` must cut material on BOTH sides
     of the plane, pass `both=True` to `extrude` — the default one-sided
     extrude can miss the wall entirely.
   - HOST-BOUNDED holes: in multi-body parts (brackets, housings) a through-
     hole pierces ONLY the plate it is declared in — never keep drilling
     down the whole axis line: the bore would also cut a strut/rib/opposite
     wall on that line and fails verification. Put the hole workplane on the
     host plate's outer (hi) face and bound the cut to the host thickness
     (`through_hole(d, depth=host_thickness + 0.5)`), unless the spec says
     the bore goes through several bodies.
   - DIAGONAL BRACE / STRUT / "Strebe" is a straight rectangular bar between
     two points — `cadlib.strut(p0, p1, width, thickness)`. It is NOT a
     filled triangular gusset: only build `rib()`/triangles when the spec
     explicitly wants a gusset/rib. A filled gusset under a declared bore
     line gets drilled by the hole and then fails "brace width" checks.
   - Turned/revolved parts (grips, nozzles, bodies of revolution): build the
     profile as `(x, radius)` points in `BuildSketch(Plane.XZ)` — u maps to
     X, v is the radial distance — and close the profile DOWN TO THE AXIS
     (endpoints at z=0 plus a closing line, or `closed_polyline`). Then
     `revolve(axis=Axis.X)` inside `BuildPart()`. An open profile or tips
     that don't reach the axis leave the solid open/gapped.
   <!-- adapted from build123d-mcp b123d-modeling skill (Apache-2.0) -->
   - After EVERY boolean (`+`, `-`, `&`, `mode=Mode.SUBTRACT`) verify it
     actually changed the part: `measure(part)` and compare face/solid counts
     — unchanged counts mean the boolean silently no-op'd (misaligned or
     grazing cutter). Fix the cutter before adding the next feature.
   - NEVER fuse solids one-by-one inside a Python loop (`arch = arch + seg` or
     `arch += seg` per segment): each iteration runs a full OCC boolean on the
     GROWN solid, and curved chains of 10+ pieces stall for minutes. Collect
     the pieces in a list and fuse ONCE after the loop
     (`parts[0].fuse(*parts[1:])`). For a swept/curved profile prefer
     `sweep()`/`loft()` or `revolve` over stacked primitives.
5. A helper `load_reference(asset_id: str) -> Shape` is predefined for user-provided
   STEP/STL reference bodies (their asset ids are listed in the context).
   Injected inspect helpers (bare names, no import needed):
   - `measure(shape)` -> dict with keys `bbox_min`/`bbox_max`, `dims` ({x,y,z}
     extents), `volume_mm3`, `area_mm2`, `solids`, `faces`, `edges`,
     `face_types`, `is_valid`. There is no `volume` key — use `volume_mm3`.
   - `find_holes(shape)` -> list of dicts with keys `diameter`, `axis`,
     `direction`, `center`, `through`, `depth`, `id`.
   - `bbox_dims(shape)` -> `[dx, dy, dz]` overall extents in mm.
6. MEASURED PLAN VIEW (only when the context shows "measured image features"):
   - `anchor_holes()` returns the hole positions measured from the top-view image
     as a list of `{"index", "x", "y", "dia", "axis"}` dicts (XY in the part frame, the
     origin at the bbox centre — same frame as `MESH_BBOX`). `axis` is the
     measured drill direction (plan-view holes: `"+z"`) — drill along THAT axis;
     a hole at the right XY position but bored through a side face is a
     rejected part. When `anchor_holes()`
     returns entries, positioned through-holes MUST use those `x`/`y` coordinates
     verbatim (combine with `spine_y(x)` only when the anchor list is empty) and
     MUST keep `dia` from your spec, not from the image (image diameters are
     perspective estimates). A missing anchor hole or one off by >2 mm is a
     rejected part. This applies only when the context marks the measurement
     authoritative or the anchor source is `image`/`image+mesh` — anchors
     sourced `mesh` or `image-approx` are hints.
   - `load_image_profile() -> Face` returns the measured plan-view outline as a
     planar Face on the XY plane, centred on the origin, in real mm. For
     footprint-driven parts ("same footprint/outline as the top view") build the
     body from it — `extrude(face, amount=h)` or `cadlib.profile_body(face, ...)`
     — instead of drawing your own guessed outline.
   - `IMAGE_PROFILE` is the raw `[[x, y], ...]` outline list for custom
     constructions.
7. STANDARD PARTS: when `PART SPEC.standard_parts` is non-empty, use the exact
   catalog class, norm type, size and constructor parameters. Never replace a
   named screw, nut, insert, bearing, gear or other catalog object with guessed
   hand-built dimensions. Use `create_standard_part` to get a validated recipe;
   for a standalone catalog-only spec the codegen pipeline may provide the whole
   recipe directly. Purchased fasteners are included in `result` when they are
   declared as standard parts; hole/seat features remain separate cuts.
   THREAD DETAIL: `CADX_DETAIL_MODE` is injected by the executor. For catalog
   fasteners with a `simple` option, set `simple=(CADX_DETAIL_MODE == "preview")`
   only when the structured feature declares a thread. Add a JSON-safe
   `CADX_FEATURES` entry for each thread with `kind`, `thread_spec`,
   `major_diameter`, `pitch`, `length`, `start`, `external`, `hand`, `axis`
   and `position_mm` — `start`/`pitch`/`length`/`major_diameter` are scalar
   mm numbers, `position_mm` the [x, y, z] mouth centre. The overlay is
   drawn as `position_mm + axis * (start … start+length)`: `start` is the
   axial distance ALONG `axis` from `position_mm` to the first turn, NOT an
   absolute z — a top-face bore opening downward takes `axis: "-z"`,
   `start: 0.0`, `position_mm` at the mouth, and a thread ending flush
   reports `length` = hole depth. Preview mode renders the helix FROM THIS METADATA —
   never subtract or fuse any thread cue/ridge/groove in preview mode: a
   helical bore wall hides the underlying hole from feature measurement and
   fails verification. Preview threads live in metadata only. For INTERNAL
   threads on any part, subtract `cadlib.tapped_hole(major_d, pitch, length)`
   and assign it a name — it returns the right cutter per mode (smooth
   tap-drill bore in preview, real ISO groove in export); do not hand-roll
   Helix sweeps. `ThreadedHole(fastener=..., depth=..., simple=...)`
   inside BuildPart is the fastener-bound equivalent. Final export mode is
   rebuilt by the executor with `CADX_DETAIL_MODE == "export"` and must
   contain the real helix. Never silently export the `simple=True` core
   when a thread was requested. NEVER fork geometry on `CADX_DETAIL_MODE`
   (`if CADX_DETAIL_MODE == "export": …`) — the preview execs only ever run
   one branch, so a defect in the other ships unverified (an export-only
   probe runs your exact code when threads are declared, and the finalize
   rebuild hard-fails on a missing helix). Keep the mode switch inside
   `simple=(CADX_DETAIL_MODE == "preview")` arguments and
   `cadlib.tapped_hole`, which branch internally and stay verified.
   Keep nominal dimensions separate from FDM
   compensation.
8. Keep the code deterministic, under ~200 lines, with a short comment naming the feature
   each block creates.
9. BOUND FEATURE SNIPPETS: when the context contains "BOUND FEATURE SNIPPETS"
    those calls are resolved geometry — copy them VERBATIM (axes, direction,
    normal, dims, frame rotations). Do NOT re-derive or "fix" their arguments;
    only fill in the `<…>` placement coordinates and decide the build order.
    Features without a snippet are yours to construct normally.
