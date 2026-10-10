## DFM RULES — design for the chosen manufacturing process (checked after each run)

### FDM / SLA printing
- Min wall thickness: 0.8 mm for a 0.4 mm nozzle (2 perimeters); structural walls >= 1.2 mm.
- Avoid overhangs steeper than ~50 deg from vertical; add chamfers (self-supporting) instead.
- Holes: horizontal holes print elliptical on top -> use teardrop profile or +0.2 mm clearance.
- Fits: clearance holes via `ClearanceHole(fit="Normal"|"Loose")`; press fits need -0.1..-0.2 mm.
- Heat-set inserts: `InsertHole` + >= 2 mm wall around the insert.
- Bridging: horizontal spans > ~20 mm need support — prefer slopes/chamfers.
- Printed threads: use `manufacturing_compensation` (typ. 0.1-0.3 mm) and orient the axis
  vertical; coarse pitches print better than fine.
- First layer: give the part a flat bottom face (contract: it rests on the XY plane anyway).

### CNC (3-axis, Z-up setup)
- Internal vertical corners: tool cannot make them sharp -> fillet >= tool radius
  (default 1.5 mm, i.e. 3 mm end mill). Model fillets, do not rely on the CAM.
- No undercuts: all surfaces must be reachable from +Z (else two-sided machining).
- Pockets: depth/width <= 4:1; bottom corner radii = tool radius.
- Min wall thickness ~1 mm (aluminium) / ~2 mm (plastics) to survive cutting forces.
- Threads: machinists tap threads — prefer `TapHole` (tap drill) or cosmetic threads and
  note the tap size, do not model helixes.
- Holes deeper than ~5x diameter are slow; prefer through holes.

### Both
- Exactly ONE solid for single parts (assemblies only when requested).
- Overall size within the build volume (default 256^3 mm FDM, 300x200x100 CNC).
- Watertight, valid B-Rep — no zero-thickness features, no coincident faces.
