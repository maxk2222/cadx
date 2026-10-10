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
   - add exact features ONLY via boolean ops on primitives: `mesh - Cylinder(...)` for
     holes/bores, `mesh - Box(...)` or `split` for flat mounting faces, `mesh + Box(...)`/
     `mesh + extrude(...)` for flanges and bosses.
   - NEVER call fillet/chamfer on mesh edges and never select mesh faces via
     `faces().filter_by(...)` — the solid has thousands of triangular facets; address
     positions numerically (bbox from `MESH_BBOX`, e.g. centred hole:
     `Cylinder(radius=..., height=MESH_BBOX[2]*3)` through Z=0).
   - `result` must contain the mesh solid (possibly with additions/cuts). Do not replace
     it with a parametric approximation.
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
