# DfM checks

`cad/dfm/` runs after every successful execution and produces `dfm.json`
plus `DfmWarning` entries (code, severity `info|medium|high`, message,
value, threshold). The project `process` setting selects the rule family:
`fdm`/`sla` run the additive checks, `cnc3`/`cnc2d` the machining checks.

## Common (`common.py`)

- **Wall thickness**: ray-cast thickness sampling via trimesh on the STL
  mesh. Flags `thin_wall` only when the 5th-percentile is below
  `min_wall_mm` *and* >15% of samples are thin — edge-grazing rays produce
  false small values, so the proportion check is essential.
- Shared helpers: surface stats, bbox utilities.

## Mesh-mode checks

For `resolved_mode="mesh"` versions there is no analytic B-Rep — the DfM
functions are split into `fdm_mesh_checks` / `fdm_brep_checks` (and
`cnc_mesh_checks` / `cnc_brep_checks`), with `fdm_checks`/`cnc_checks` as
wrappers that run both. Mesh-specific warnings:

| Check | Rule |
|-------|------|
| `non_manifold` | mesh not watertight / winding inconsistent (also an `Analysis.warning`). |
| `high_triangle_count` | > 500k triangles → suggest decimation. |
| `no_flat_base` | downward-facing area near min-Z < 2 % of surface → brim/raft or `flatten_bottom_mm` post-param. |

Mesh + B-Rep checks share the same machinery (thickness rays, overhangs)
because `common_checks` works on the tessellated mesh either way.

## FDM (`fdm.py`)

| Check | Rule |
|-------|------|
| `thin_wall`   | p5 thickness < `min_wall_mm` (default 0.8 mm ≈ 2 perimeters at 0.4 nozzle) and >15% of surface thin. Severity high if >50%. |
| `overhang`    | Downward faces > ~50° from vertical, above z>0.5 mm; warn if >5% of area, high if >20%. |
| `bed_fit`     | Bounding box vs configured build volume. |
| `thread_comp` | Informational: printed threads need compensation (project `thread_compensation_mm`) or post-tapping. |

## CNC (`cnc.py`)

| Check | Rule |
|-------|------|
| `inside_radius` | Concave vertical LINE edges → internal corners need tool radius (`tool_radius_mm`). Verified empirically: for adjacent-face normals n1,n2 and edge tangent t, `cross(n1,n2)·t > 0` means concave. |
| `deep_pocket`   | +Z planar faces enclosed below top by near-vertical walls; pocket depth vs tool reach heuristic. |
| `undercut`      | Faces with normal z < −0.1 not reachable from 3-axis top. |
| `thin_wall`     | Same ray-cast check, CNC min-wall default. |

## Consumption

- Stored to `<version>/dfm.json`, registered as a `dfm` artifact, exposed in
  the UI `DfmPanel`.
- The CRITIQUE step receives the warnings so the vision judge can weigh real
  manufacturability issues, and PLAN/CODEGEN prompts carry the rules to
  prevent them upstream.

## Limitations

These are heuristics, not simulation: pocket depth uses a face-enclosure
approximation, thin-wall sampling is statistical, and thread checking relies
on catalog metadata rather than geometry inspection. Tune thresholds via
project settings (`min_wall_mm`, `tool_radius_mm`, `build_volume`) — see
[configuration.md](configuration.md).

## See also

- [Web GUI](webgui-ux.md) — where DfM warnings appear
- [FEM & TopOpt](fem-topopt.md) — structural checks beyond geometry
