HYBRID MODE: `load_mesh_solid() -> Shape` returns the generated organic mesh
as a faceted B-Rep solid, and `MESH_BBOX` holds its `[L, W, H]` in mm. The
solid is already oriented and scaled: it rests on XY and is centred at the
origin. Rules for mesh solids:

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
- Boolean errors on the mesh solid: use simple convex primitives oversized to
  fully intersect, positioned from `MESH_BBOX`; prefer `result = mesh_solid - tool`
  algebra over builder contexts.
- `result` must contain the mesh solid (possibly with additions/cuts). Do not replace
  it with a parametric approximation.
