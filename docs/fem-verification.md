# FEM verification suite (`tests/femverify/`)

Declarative, in-process verification of the voxel FEM stack
(`cad/fem/voxelize.py` → `regions.py` → `solver.py`) and the TopOpt layer —
no API, no evalkit, no external services. Cases are JSON files with explicit
references and tolerances; the harness runs them and reports per-probe errors.

```bash
make test         # fast lane + regression (verify-marked cases are excluded)
make fem-verify   # everything in tests/femverify, incl. the heavy verify lane
```

## Lanes

| Lane | Marker | Contents | Runtime |
|---|---|---|---|
| fast | — | analytical cases (exact or tight references) | ~1 s total |
| verify | `@pytest.mark.verify` | published benchmarks, curved boundaries, pitch-swept convergence studies | ~10 s |
| mms | — | method of manufactured solutions, O(h²) order assert | <1 s |
| topopt | — | optimizer invariants (volume, beats-uniform, OC≈MMA, determinism, robust ordering) | ~4 s |
| ccx | opt-in | same-mesh cross-check vs CalculiX | needs `ccx` |

## Case format

One JSON per case in `tests/femverify/cases/`:

```jsonc
{
  "name": "kirsch_plate",
  "lane": "verify",                      // "fast" | "verify"
  "geometry": {"builder": "plate_hole", "args": {...}},
  "material": {"E": 70000.0, "nu": 0.33},
  "pitch_mm": 2.0,
  "cases": [{
    "name": "tension",
    "fixed": [{"kind": "face", "axis": "-x"}],
    "loads": [{"kind": "face", "axis": "+x", "force_n": [2400, 0, 0]}],
    "manual_pins":  [{"dof": "z", "sel": {"z": 0.0}}],      // node-exact BCs
    "manual_loads": [{"pressure_mpa": 1.0, "dirs": ["-z"],
                      "cell_sel": {"ell_a": 32.5, "ell_b": 27.5, "ell_r_min": 26.5}}]
  }],
  "expect": [
    {"probe": "stress_near", "at": [20, 0, 3], "r_mm": 2.0, "comp": "sy",
     "ref": 92.7, "tol": 0.25},
    {"probe": "resultant_n", "axis": "x", "ref": 2400, "tol": 0.01}
  ],
  "source": "NAFEMS LE1; geometry/BCs per seamplex fino case file",
  "notes":  "why the reference was scaled / what the voxel staircasing does"
}
```

`kind: "trend"` / `"convergence"` cases sweep `pitches_mm` instead and assert
the final error **and** a monotone approach to the reference (see
`cantilever_conv`, `nafems_le1`, `nafems_le10`).

## Probes

| Probe | Meaning |
|---|---|
| `resultant_n` | ΣF of the applied load vector on `axis` (equilibrium sanity) |
| `max_disp_mm` | max ‖u‖ |
| `compliance_j` | F·u |
| `work_disp` | compliance/‖F‖ — work-equivalent displacement for spread loads |
| `disp_at` / `disp_nodes` / `disp_diff` | u at a point / node-set mean / difference of two |
| `vm_at` / `stress_at` / `stress_near` | von Mises or component at a cell; `stress_near` = extremum within `r_mm` — the honest probe for boundary peaks whose owning cell moves with the pitch |
| `converged` | solver-converged flag |

## Selectors (`manual_pins`, `manual_loads`, `sel`)

Region kinds select cells — a cell always spans ≥ 2 node planes, which makes
region-based knife-edge supports and exact symmetry planes impossible. The
harness therefore supports node/cell coordinate predicates:

- `x`/`y`/`z`: exact value, or `"min"`/`"max"` (outermost used-node plane)
- `xmin`/`xmax`/…: bounds
- `rxy_min`/`rxy_max`: cylindrical radius about the z axis through the domain centre
- `ell_a`/`ell_b` + `ell_r_min`/`ell_r_max`: elliptical radius `sqrt((x/a)²+(y/b)²)`
- `boundary: true` (cells): has a void/exterior neighbour in-plane

Use them for: textbook pin supports, symmetry planes, strict end-plane loads
(Cook), elliptical rim BCs (NAFEMS LE10), pressure on staircased rims.

## Verified references (measured)

| Case | Reference | Result |
|---|---|---|
| tension/compression, bending, patch | analytic | exact to ~0.5 % |
| simply supported (EB + shear) | analytic | ~5 % (spread-load end effect) |
| cantilever tip, 3 pitches | Timoshenko | monotone convergence, final ~12 % under |
| Cook's membrane | u·E·t/P = 23.91 → 0.5978 mm | 0.58 mm (~3 %) |
| Kirsch plate hole | Kt = 3 → σr = 30 MPa | ~11 % (staircased hole) |
| Lamé tube | analytic σr/σθ | ~5 % |
| NAFEMS LE1 | σyy@D = 92.7 MPa | converges 60→79→101→**95.8** |
| NAFEMS LE10 | σyy@D = −5.38 MPa | monotone −3.1→−3.8→−4.3→… (under-reads, right sign/trend) |
| MMS sine field | O(h²) | measured rates 2.05, 1.99 |

Curved/boundary-peak cases converge toward but do not hit the reference —
that is the honest voxel-staircasing behaviour the trend asserts measure,
not a failure.

## Findings shipped by this suite

- `regions._surface_cells`: `face`/`surface` used to pick **every** exposed
  face in a direction — including the inner wall of through-holes (747 N of
  a 2400 N Kirsch load landed in the bore). Now the per-column outer
  envelope; `bore`/`hole`/`cylinder` remain the way to load inner walls.
- `voxelize._fill_grid` fallback: `binary_closing` erased the outermost cell
  rows on non-watertight meshes (`border_value=0`); boundary material is now
  preserved — LE1/LE10 symmetry pins land on the real edge.
- `test_topopt_inv.test_invariant_beso_contract` documents that soft-kill
  BESO is a greedy heuristic whose optima can sit above uniform-field
  compliance on coarse grids — the invariant asserts its contract (volume +
  one connected load path), not optimality.

## Cross-solver lane (CalculiX)

`tests/femverify/xverify/ccx.py` exports the voxel domain **as a C3D8 mesh**
(our node order matches Abaqus') and runs `ccx` on identical BCs/loads —
displacements must agree to ~1e-4 rel. Opt-in:

```bash
CCX_PATH=/path/to/ccx ../.venv/bin/uv run pytest tests/femverify/test_ccx.py
```

Without the binary the comparison test skips; the deck-writer structure test
always runs. To import published `.inp` cases (e.g. the CalculiX test suite)
add a loader that maps the Abaqus keywords onto a case spec.

## Adding a case

1. Write the JSON (pick `lane`, declare `source` and `notes` — scaled
   references must say so).
2. If a new geometry is needed, add a builder to `geometry.py`; make the mesh
   watertight (`mesh.process(validate=True)`) so voxelize takes the exact
   contains path.
3. Calibrate honestly: run once, look at the error structure before setting
   `tol` — don't tighten around a bug, and prefer `stress_near`/trends for
   boundary peaks.
