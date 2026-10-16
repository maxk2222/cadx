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
3. Build the part with build123d (builder mode `with BuildPart() as p:` or algebra mode).
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
   - Turned/revolved parts (grips, nozzles, bodies of revolution): build the
     profile as `(x, radius)` points in `BuildSketch(Plane.XZ)` — u maps to
     X, v is the radial distance — and close the profile DOWN TO THE AXIS
     (endpoints at z=0 plus a closing line, or `closed_polyline`). Then
     `revolve(axis=Axis.X)` inside `BuildPart()`. An open profile or tips
     that don't reach the axis leave the solid open/gapped.
4. Coordinate frame: X = length (left->right), Y = width (+Y is the "front" face),
   Z = height (up). The part must rest on the XY plane (min Z = 0) and be centred on
   the origin in X and Y. Use `align=(Align.CENTER, Align.CENTER, Align.MIN)` on
   primitives to achieve this.
5. Units are millimetres.
6. A helper `load_reference(asset_id: str) -> Shape` is predefined for user-provided
   STEP/STL reference bodies (their asset ids are listed in the context).
7. HYBRID MODE (only when the context mentions it): `load_mesh_solid() -> Shape` returns
   the generated organic mesh as a faceted B-Rep solid, and `MESH_BBOX` holds its
   `[L, W, H]` in mm. The solid is already oriented and scaled: it rests on XY and is
   centred at the origin. Rules for mesh solids:
   - NEVER rotate, mirror or translate the mesh solid — it is already in the part
     frame: `MESH_BBOX[0]` is the length along X (the part's main axis),
     `MESH_BBOX[1]` the width along Y, `MESH_BBOX[2]` the height along Z.
   - add exact features ONLY via boolean ops on primitives: `mesh - Cylinder(...)` for
     holes/bores, `mesh - Box(...)` or `split` for flat mounting faces, `mesh + Box(...)`/
     `mesh + extrude(...)` for flanges and bosses.
   - NEVER assign the result of an intersection back to the body:
     `body = body & tool`, `body = body.common(tool)` and `body = body.intersect(tool)`
     can return `None` on faceted mesh solids (OCC fails silently — the part then
     becomes None and the run reports "no result assigned"). To intersect, use
     `body = cadlib.mesh_intersect(body, tool)` (raises a real error). To flatten
     the underside use `body = cadlib.trim_below(body, z=0)` — a cut, not an
     intersection.
   - Booleans on mesh solids can shed sub-mm SLIVER solids — the result then
     holds several disconnected fragments and fails the single-part check.
     Wrap cuts and fuses in `cadlib.mesh_cut(body, tool)` /
     `cadlib.mesh_fuse(body, tool)` so debris is dropped at the op; plain
     `body -= tool` / `body += tool` are accepted but less safe. Finish with
     `result = cadlib.collapse_solids(body)` as a last-line guard.
   - NEVER call fillet/chamfer on mesh edges and never select mesh faces via
     `faces().filter_by(...)` — the solid has thousands of triangular facets; address
     positions numerically (bbox from `MESH_BBOX`, e.g. centred hole:
     `Cylinder(radius=..., height=MESH_BBOX[2]*3)` through Z=0).
   - Organic bodies are curved/asymmetric: the bbox centreline is NOT the body's
     spine. `spine_y(x)` returns the body's material-centre Y at position x
     (`spine_x(y)` vice versa). ALWAYS place positioned features (holes, pockets)
     on the spine, NEVER at y=0/x=0: a centreline hole at x=+40 goes to
     `y = spine_y(40)`. `MESH_PROFILE` maps `"x"`/`"y"` to rows
     `[pos, min_a, max_a, min_b, max_b]` — the actual material extent per axis
     slice — when you need margins rather than the centre.
   - `top_z(x, y)` returns the body's LOCAL top-surface height directly above that
     point (falls back to MESH_BBOX[2] outside the silhouette). Organic tops dip
     several mm below the global bbox max — countersinks, pockets and recesses
     anchored at `MESH_BBOX[2]` float in air and cut NOTHING. Anchor them to the
     local surface: `zt = top_z(x, y); cone_center_z = zt - depth/2`. Extend
     shallow cutters slightly above `zt` so they break through local peaks.
   - `result` must contain the mesh solid (possibly with additions/cuts). Do not replace
     it with a parametric approximation.
7b. MEASURED PLAN VIEW (only when the context shows "measured image features"):
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
8. STANDARD PARTS: whenever the user names a screw/nut/insert/bearing/thread size, do
   NOT type its dimensions by hand. Create the bd_warehouse object and cut the hole with
   `ClearanceHole` / `TapHole` / `ThreadedHole` / `InsertHole` / `PressFitHole` (see the
   bd_warehouse reference). Purchased parts (screws, nuts, bearings) are NOT part of
   `result` unless `include_fasteners` is enabled; only their holes/seats are cut.
   Threads: use `simple=True` (cosmetic) unless the user asks for printed/machined real
   threads; real threads must use `end_finishes=("fade","square")` or `("raw","raw")`
   and `manufacturing_compensation=PARAMS["thread_comp"]` for FDM.
9. Keep the code deterministic, under ~200 lines, with a short comment naming the feature
   each block creates.
10. BOUND FEATURE SNIPPETS: when the context contains "BOUND FEATURE SNIPPETS"
    those calls are resolved geometry — copy them VERBATIM (axes, direction,
    normal, dims, frame rotations). Do NOT re-derive or "fix" their arguments;
    only fill in the `<…>` placement coordinates and decide the build order.
    Features without a snippet are yours to construct normally.
