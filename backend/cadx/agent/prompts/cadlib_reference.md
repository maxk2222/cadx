# cadlib — reviewed helpers (import cadlib)

`cadlib` ships with the executor: deterministic, tested macros for common
features. Prefer these over hand-rolled geometry — they encode known-good
parameters. Import either the package or the function:

    import cadlib
    from cadlib import hole_grid, boss, mounting_ear

## Cutouts — run INSIDE `with BuildPart() as p:`

Position via an enclosing `with Locations(...)`, or pass `on=<Face|Plane|Location>`
to apply at that workplane (usually a face, e.g. `p.faces().sort_by(Axis.Z)[-1]`).

- `hole_grid(screw, nx=2, ny=2, pitch_x=20, pitch_y=20, kind="clearance",
  fit="Normal", depth=None, counter_sunk=None, material="Soft", on=None)` —
  nx×ny hole pattern for a bd_warehouse fastener/nut/insert.
  kind ∈ clearance | countersink | counterbore | tap | threaded | insert.
  For countersink use a `CounterSunkScrew`, for counterbore a `SocketHeadCapScrew`.
- `insert_pocket(insert, depth=None, compensation=0.1, fit="Normal", on=None)` —
  blind pocket for a `HeatSetNut`; depth defaults to the insert's own depth.
- `bearing_seat(bearing, interference=0, fit="Normal", depth=None,
  retain="pressfit", groove_offset=2.0, on=None)` — press-fit bore; `retain`
  ∈ pressfit | through | snapring (snapring adds an internal snap-ring groove
  `groove_offset` mm below the workplane).
- `through_hole(diameter, depth=None, csk_dia=None, csk_angle=90,
  cbore_dia=None, cbore_depth=None, on=None)` — Ø`diameter` bore along the
  workplane normal, guaranteed through (oversized cutter — never leaves a
  membrane). `depth` makes it blind; `csk_dia` adds a countersink cone and
  `cbore_dia`/`cbore_depth` a counterbore, both opening at the workplane.
  Safest way to place holes at measured anchor positions:
  `with Locations(Pos(x, y, 0)): through_hole(5.0)`.
- `keyhole_slot(big_d, small_d, spacing, thickness, direction="+x", on=None)` —
  keyhole cutout cutting `thickness` mm down from the workplane.
- `t_slot(bolt_d, head_w, head_h, slot_w, length, on=None)` — T-slot groove
  along ±X opening at the workplane.
- `dovetail_slot(width_top, width_bottom, height, length)` — returns a cutter
  Part (not a context call): `p.part -= Pos(x, y, z) * dovetail_slot(...)`.

## Shapes — pure factories, return a Part you position yourself

No BuildPart context needed. Parts rest on XY (min Z = 0), centred — place with
`Pos(x, y, z) * helper(...)` and fuse: `result += ear` or `p.part += ear`.

- `boss(diameter, height, hole_d=None, hole_depth=None, draft_deg=0)` —
  vertical screw dome, optional bore and moulding draft.
- `rib(length, height, thickness)` — triangular gusset, base along +X on XY,
  apex at X=0.
- `mounting_ear(length, width, thickness, hole_d=None, hole_offset=0,
  tip_radius=None)` — flat ear, optional through-hole along Z, rounded tip.
- `snap_fit_cantilever(length, width, thickness, hook_height=None)` — snap-fit
  beam along +X with a hook wedge at the tip.
- `living_hinge(length, width, thickness=0.3)` — membrane hinge segment.
- `knurl_ring(diameter, height, pitch=2.0, depth=0.8)` — axial grip ridges for
  a cylindrical surface of `diameter`; fuse onto a knob or shaft.
- `text_label(text, size=6.0, depth=0.4, font="Arial")` — text raised `depth` mm;
  union for embossing, subtract for engraving.
- `filleted_box(length, width, height, radius)` — box, all edges filleted.
- `l_bracket(leg_x, leg_z, width, thickness, fillet_r=0)` — L-bracket, legs on
  +X and +Z.
- `profile_body(face, thickness, crown_mm=0, edge_fillet=0)` — prismatic body
  from a planar outline Face (e.g. `load_image_profile()`); optional crowned
  top (`crown_mm` raises an inset loft cap) and top-edge fillet. The result
  rests on XY centred exactly like the outline — the fastest correct way to
  turn a measured top view into a solid.
- `closed_polyline(pts, tol=1e-3)` — planar Face from a dense sampled outline
  `[(x, y), ...]`; tolerates consecutive duplicates, a repeated closing point
  and stray spikes that would crash raw `Polyline`. USE THIS whenever you
  generate an outline from sampled math (S-curves, parametric sweeps) instead
  of calling `Polyline` yourself — duplicate-sample crashes are the most
  common exec error. Then `extrude(face, amount=h)` or feed it to
  `profile_body`.
