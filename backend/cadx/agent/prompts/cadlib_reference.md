# cadlib — reviewed helpers (import cadlib)

`cadlib` ships with the executor: deterministic, tested macros for common
features. Prefer these over hand-rolled geometry — they encode known-good
parameters. Import either the package or the function:

    import cadlib
    from cadlib import hole_grid, boss, mounting_ear

## Cutouts — run INSIDE `with BuildPart() as p:`

Position via an enclosing `with Locations(...)`, or pass `on=<Face|Plane|Location>`
to apply at that workplane (usually a face, e.g. `p.faces().sort_by(Axis.Z)[-1]`).

### hole_grid

- `hole_grid(screw, nx=2, ny=2, pitch_x=20, pitch_y=20, kind="clearance",
  fit="Normal", depth=None, counter_sunk=None, material="Soft", on=None)` —
  nx×ny hole pattern for a bd_warehouse fastener/nut/insert.
  kind ∈ clearance | countersink | counterbore | tap | threaded | insert.
  For countersink use a `CounterSunkScrew`, for counterbore a `SocketHeadCapScrew`.

### insert_pocket

- `insert_pocket(insert, depth=None, compensation=0.1, fit="Normal", on=None)` —
  blind pocket for a `HeatSetNut`; depth defaults to the insert's own depth.

### bearing_seat

- `bearing_seat(bearing, interference=0, fit="Normal", depth=None,
  retain="pressfit", groove_offset=2.0, on=None)` — press-fit bore; `retain`
  ∈ pressfit | through | snapring (snapring adds an internal snap-ring groove
  `groove_offset` mm below the workplane).

### through_hole

- `through_hole(diameter, depth=None, csk_dia=None, csk_angle=90,
  cbore_dia=None, cbore_depth=None, on=None)` — Ø`diameter` bore along the
  workplane normal, guaranteed through (oversized cutter — never leaves a
  membrane). `depth` makes it blind; `csk_dia` adds a countersink cone and
  `cbore_dia`/`cbore_depth` a counterbore, both opening at the workplane.
  Safest way to place holes at measured anchor positions:
  `with Locations(Pos(x, y, 0)): through_hole(5.0)`.

### oblong_slot

- `oblong_slot(width, length, depth=None, direction="+x", normal="+z", on=None)`
  — stadium through-slot `length` tip-to-tip along `direction` (+x|-x|+y|-y|+z|-z),
  `width` across, cut through along `normal` (perpendicular to `direction`);
  `depth` makes a blind pocket. USE THIS for elongated mounting slots —
  hand-built slots from two boxes get skipped or leak membranes. A vertical
  slot through a wall thin in X:
  `oblong_slot(6.5, 15, direction="+z", normal="+x")`. The cut applies to the
  ACTIVE BuildPart — never into a throwaway box. For algebra-style bodies:
  `body = body - Pos(x, y, z) * slot_cutter(w, l, direction=..., normal=..., part=body)`.

### slot_cutter

- `slot_cutter(width, length, direction="+x", normal="+z", part=None, depth=None)`
  — stadium-slot cutter as a solid for algebra-style bodies: subtract it,
  `body = body - Pos(x, y, z) * slot_cutter(6.5, 15, direction="+z", normal="+x", part=body)`.
  `part` sizes the through-cut over the target's bounding box.

### keyhole_slot

- `keyhole_slot(big_d, small_d, spacing, thickness, direction="+x", on=None)` —
  keyhole cutout cutting `thickness` mm down from the workplane.

### t_slot

- `t_slot(bolt_d, head_w, head_h, slot_w, length, on=None)` — T-slot groove
  along ±X opening at the workplane.

### dovetail_slot

- `dovetail_slot(width_top, width_bottom, height, length)` — returns a cutter
  Part (not a context call): `p.part -= Pos(x, y, z) * dovetail_slot(...)`.

## Shapes — pure factories, return a Part you position yourself

No BuildPart context needed. Parts rest on XY (min Z = 0), centred — place with
`Pos(x, y, z) * helper(...)` and fuse: `result += ear` or `p.part += ear`.

### boss

- `boss(diameter, height, hole_d=None, hole_depth=None, draft_deg=0)` —
  vertical screw dome, optional bore and moulding draft.

### tube

- `tube(outer_d, inner_d, length)` — open hollow cylinder along Z resting on
  XY (bore through both ends). Lay flat with `Rot(0, 90, 0)` (axis → X) or
  `Rot(90, 0, 0)` (→ Y): raised tube holders, bushings, spacers. A closed
  cylinder is NOT a tube — always use this for open bores.

### rib

- `rib(length, height, thickness)` — triangular gusset, base along +X on XY,
  apex at X=0, `thickness` centred on Y=0. Returns a Part — fuse it with
  OVERLAP into the parent so the union joins (a rib merely touching a face
  stays a separate solid):
  `p.part += Pos(x, y, z) * Rot(0, -90, 0) * rib(10, 10, 3)` places it on a
  vertical wall with the base running up Z.

### strut

- `strut(p0, p1, width, thickness, embed=1.0)` — STRAIGHT rectangular bar
  between two world points `p0`/`p1` (mm tuples); `width` runs along the
  axis closest to world Y, `thickness` lies in the bar plane, both ends
  extend `embed` mm so the fuse overlaps. USE THIS for diagonal
  braces/struts/"Streben" between two bodies — e.g.
  `part = part + strut((-55, 0, 35), (15, 0, 95), 10, 6)`. It is NOT a
  filled triangular gusset — a spec that says "diagonal brace"/"Strebe"
  wants this bar; only use `rib()` when the spec explicitly wants the
  filled triangle. A filled gusset under a bore line gets drilled by
  through-holes and fails verification.

### arch_bridge

- `arch_bridge(span_x, apex_z, width_y, thickness, *, foot_z=0, embed=1.0,
  n_sections=15)` — ARCHED bar/handle between two feet at ±`span_x`/2 rising
  in the XZ plane so the top surface reaches `apex_z`; feet extend `embed`
  mm below `foot_z` so the fuse overlaps. Cross-section: `thickness` radial
  (in the arch plane) × `width_y` along Y. USE THIS for arched handles and
  bridges — a hand-rolled loft collapses to a flat bar because
  `Plane(normal=...)` is not valid build123d (use `z_dir`). `strut` stays
  the choice for STRAIGHT braces: `part += arch_bridge(90, 25, 8, 6)`.

### mounting_ear

- `mounting_ear(length, width, thickness, hole_d=None, hole_offset=0,
  tip_radius=None)` — flat ear, optional through-hole along Z, rounded tip.

### snap_fit_cantilever

- `snap_fit_cantilever(length, width, thickness, hook_height=None)` — snap-fit
  beam along +X with a hook wedge at the tip.

### living_hinge

- `living_hinge(length, width, thickness=0.3)` — membrane hinge segment.

### knurl_ring

- `knurl_ring(diameter, height, pitch=2.0, depth=0.8)` — axial grip ridges for
  a cylindrical surface of `diameter`; fuse onto a knob or shaft.

### text_label

- `text_label(text, size=6.0, depth=0.4, font="Arial")` — text raised `depth` mm;
  union for embossing, subtract for engraving.

### filleted_box

- `filleted_box(length, width, height, radius)` — box, all edges filleted.

### l_bracket

- `l_bracket(leg_x, leg_z, width, thickness, fillet_r=0)` — L-bracket: foot
  `leg_x` along +X, leg `leg_z` up Z at the X=0 end, `width` centred on Y,
  optional inner-corner fillet. Whole body in one reviewed call — orient it
  once (`Rot(0, 0, 180)` puts the leg at −X) instead of building foot+leg
  from two boxes.

### profile_body

- `profile_body(face, thickness, crown_mm=0, edge_fillet=0)` — prismatic body
  from a planar outline Face (e.g. `load_image_profile()`); optional crowned
  top (`crown_mm` raises an inset loft cap) and top-edge fillet. The result
  rests on XY centred exactly like the outline — the fastest correct way to
  turn a measured top view into a solid.

### safe_fillet

- `safe_fillet(edges, radius, min_radius=0.2)` — fillet with fallback: halves
  the radius until OCC accepts it, returns the applied radius or `0.0`.
  This is a builder operation and MUST run inside `with BuildPart() as p:`.
  It returns a float, not the modified Shape: call it as a statement, then
  read `p.part`. For algebra-mode code use
  `part, applied_r = safe_fillet_shape(part, edges, radius)`; that helper
  returns the new Shape and accepted radius. A nonzero radius by itself is
  not proof the input Shape changed.
  USE THIS instead of raw `fillet()` whenever the edge selector could be
  empty (stale `part.edges()` saved before a `locate`/`translate`, strict
  filters, edges consumed by a previous feature) — an empty selection crashes
  the whole run otherwise.

### safe_chamfer

- `safe_chamfer(edges, length, min_length=0.2)` — chamfer with the same
  shrink-to-fit fallback as `safe_fillet`: halves the leg length until OCC
  accepts it, returns the applied length or `0.0`. Builder-mode only —
  inside `with BuildPart() as p:`, then read `p.part`. For algebra mode use
  `part, applied_l = safe_chamfer_shape(part, edges, length)`, which returns
  the new Shape and the accepted length. `safe_chamfer(concave_edges(axis="y"), length=2)`.

## Edge selection — build fillet/chamfer selections without OCP code

`edges() >> Axis.Z` does NOT sort or filter the way it looks like it does —
use these selectors instead. All accept a Shape/Part or nothing inside an
active `BuildPart` (reads `p.part`) and return `list[Edge]` — `[]` on no
match, so pair them with `safe_fillet`/`safe_chamfer`.

### edges_at

- `edges_at(shape=None, *, axis=None, x=None, y=None, z=None, x_range=None,
  y_range=None, z_range=None, geom=None, tol=0.6)` — filter edges by
  mid-tangent axis (`"x"/"y"/"z"`, signs ignored), bbox-centre position
  (`x`/`y`/`z` ±tol or inside `*_range=[lo,hi]`) and geometry kind
  (`geom="line"|"circle"`).
  `inner = edges_at(p.part, axis="y", x=-36, z=6, tol=1)`.

### concave_edges / convex_edges

- `concave_edges(shape=None, *, axis=None, x_range=None, y_range=None,
  z_range=None)` — re-entrant edges only: inner fillet roots, groove/pocket
  corners, the L-bracket inner join — the edges a fillet is usually meant
  for. `convex_edges(...)` same signature, proud outer edges (silhouette,
  boss rims) for exterior rounding/chamfering.

### hole_rim_edges

- `hole_rim_edges(shape=None, *, diameter=None, axis=None, z=None, tol=0.5)`
  — circular rim edges of bores/counterbores/countersink mouths/boss
  shoulders; `diameter` matches the rim circle ±tol (a Ø5 bore gives two
  ~Ø5 rims), `z` the rim height, `axis` the rim-plane normal direction.
  `rims = hole_rim_edges(diameter=5, axis="z", z=plate_top)`.

### closed_polyline

- `closed_polyline(pts, tol=1e-3)` — planar Face from a dense sampled outline
  `[(x, y), ...]`; tolerates consecutive duplicates, a repeated closing point
  and stray spikes that would crash raw `Polyline`. USE THIS whenever you
  generate an outline from sampled math (S-curves, parametric sweeps) instead
  of calling `Polyline` yourself — duplicate-sample crashes are the most
  common exec error. Then `extrude(face, amount=h)` or feed it to
  `profile_body` — or `revolve(face-or-section, axis=Axis.X)` for turned
  parts (profile in a plane containing the axis, closed onto it).
### mesh_intersect (hybrid mode)

- `mesh_intersect(body, tool)` — `body & tool` that raises a real error when
  OCC returns `None` (faceted mesh solids fail silently, the part becomes
  `None` and the run reports "no result assigned"). USE THIS instead of
  `body = body & tool`, `.common()` or `.intersect()` on a mesh solid.

### mesh_cut / mesh_fuse (hybrid mode)

- `mesh_cut(body, tool)` / `mesh_fuse(body, tool)` — `body -/+ tool` that
  raises on `None` AND drops sliver fragments: booleans on faceted mesh
  solids can produce a multi-solid compound (debris, not features). USE
  THESE for every boolean on a mesh solid.

### collapse_solids / largest_solid (hybrid mode)

- `collapse_solids(shape)` — returns the dominant solid when a boolean
  result fragmented into slivers; passes single-solid shapes through.
  `largest_solid(shape)` always returns the biggest solid. Last-line guard:
  `result = cadlib.collapse_solids(body)`.

### trim_below (hybrid mode)

- `trim_below(body, z=0, margin=400)` — cuts everything below the plane `z`,
  sliver-safe and falls back to the input when the cut fails. USE THIS to
  flatten the underside of a mesh solid instead of
  `body & Box(..., Align.MIN)` — an intersection that silently returns
  `None` on faceted meshes.

### clip_to_box (hybrid mode)

- `clip_to_box(body, x=None, y=None, z_min=None, z_max=None, tol=0.01)` —
  trims a mesh solid to the declared envelope: `x`/`y` are half-extents
  (material beyond ±x/±y goes), `z_min`/`z_max` absolute Z bounds. Each side
  is checked against the real bbox first and sides already inside the bound
  get NO cutter — it skips non-overlapping cuts instead of raising. USE THIS
  for envelope trimming instead of hand-rolled `mesh_cut` box-cutter calls:
  `body = clip_to_box(body, x=L/2, y=W/2, z_min=0)`.

### bbox_dims

- `bbox_dims(shape)` — returns `[dx, dy, dz]` extents of a shape's bounding
  box in mm. USE THIS instead of touching `bounding_box()` yourself —
  `bb.size` is a Vector with uppercase `.X`/`.Y`/`.Z`, and OCP names like
  `xlen`, `xmax` or `.size.x` do NOT exist in build123d.

## Common pitfalls

- A triangular gusset/"Dreiecksrippe" is `cadlib.rib(...)` — a `strut()` is
  only the straight bar, never the filled triangle.
- `shape.translate(...)`/`shape.locate(...)` return a NEW object — keep the
  return value (`moved = obj.translate((x, y, z))`) or place copies with
  `Pos(x, y, z) * obj`; calling it bare leaves the original unmoved.
- A through-hole cutter belongs to its HOST plate: build it inside the
  plate's context/algebra (`through_hole(depth=host_span+0.5)` on the plate's
  hi face, or `plate = plate - Pos(...) * cutter`) so it drills only that
  plate and never struts/ribs sharing the bore line.
- `edges() >> Axis.Z` neither sorts nor filters the selection the way it
  reads — use `edges_at`/`concave_edges`/`hole_rim_edges` (above) to build
  the edge set explicitly.
