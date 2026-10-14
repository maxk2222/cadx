# FEM analysis and topology optimization

The structural workflow adds voxel-based linear-elastic FEM and SIMP topology
optimization to the CAD/mesh generation pipeline. Numerical work runs in
`cad-executor` on the `cad` queue; the LLM worker sends mesh paths and
structured load cases, not generated solver code.

## Workflows

### FEM feedback during generation

Describe what is fixed and how the part is loaded in the prompt, or enable
`FEM check` in run options and fill in its load/fixture note. Intake converts
that description into one or more `spec.load_cases`. For every candidate the
agent runs FEM after geometric checks and uses the resulting safety factor and
stress feedback during critique/repair. A missing or unsolvable load case is
reported as a warning; it must not be interpreted as a successful strength
check.

### Autonomous topology optimization

Choose `TopOpt` as the generation mode, or enable topology optimization in run
options. The agent first creates the base geometry, then runs SIMP on its
voxelized material domain and verifies the extracted result with FEM. A
planner may also request this phase. The input geometry bounds the design
space: SIMP removes material inside that domain and does not grow the part
outside it. Fixed, loaded, and explicit `keep_regions` are held solid.

The optimization minimizes compliance across the supplied load cases under a
target volume fraction; per-case `weight` scales each case's objective
contribution, and `LoadSpec.direction_cone_deg` expands a force into sampled
cone directions for direction-robust layouts (capped by
`fem_max_effective_cases`). The extracted boundary mesh is regularized for
watertight export. FEM verification uses the corresponding voxel mask rather
than the smoothed export mesh, so export smoothing does not change the
reported structural result.

`optimize.void_regions` pin cells to the void floor (forbidden material —
cable runs, clearance), `optimize.keep_regions` pin cells solid; a keep region
with `material` overrides the per-cell stiffness (insert zones, e.g. a brass
heat-set insert inside a polymer part). `optimize.keep_mode: "auto"` adds a
keep ring around every detected anchor (`topopt_auto_keep_margin_mm`),
`min_member_mm` post-thickens members below the printable floor, and
`density_threshold` sets the extraction iso. The stored `topopt/density.npz`
field can be re-extracted at a different threshold/min-member without a new
SIMP run via `POST /versions/{id}/topopt-resample` (`JobType.resample`).

Printability and filter options (`optimize.*`):

- `self_supporting` + `print_direction` — 45° AM filter: a cell is only solid
  when enough material supports it from below along the build direction
  (stencil approximates the 45° overhang rule). Use for support-free FDM.
- `filter: "heaviside"` — cone filter + projected physical density with a
  β-continuation schedule; produces crisp near-0/1 fields whose extraction is
  insensitive to `density_threshold`, at the cost of more iterations.
- `multires` — a coarse SIMP pass at 2× pitch warm-starts the fine grid
  (~3× fewer fine iterations on large parts); skipped below a domain-size
  floor and when an explicit `warm_density_relpath` is given.
- `penal` — SIMP penalization exponent (default 3); leave alone unless
  experimenting.

### Feature finishing (post-TopOpt)

The extracted+smoothed mesh is only approximately exact at functional
interfaces — `POST /versions/{id}/refine` (`JobType.refine`, `VersionKind
.refine` child) deterministically re-cuts them via `cad/fem/refine.py`:

- **Subtractive re-cut** — explicit `features` or auto-resolution from the
  spec's declared `features` merged with scanned anchors produces cutter
  solids (bores, `thread`/`tap` at tap-drill diameter `D − pitch`, inserts
  at `D + 0.1`, bearing seats, countersinks/counterbores) tessellated and
  subtracted with manifold3d booleans at the measured anchor axis/position.
- **Keep-surface restore** — `opt ∪ (base ∩ grown keep-box)` unions the
  ORIGINAL part surface back inside keep regions / anchor rings (Taubin
  smoothing shrinks pinned regions by ~one pitch). Gated by
  `optimize.restore_planes` / the request's `restore_planes` (default on);
  needs the pre-optimization base mesh (resolved from the grandparent for
  topopt children).
- **FEM re-verification** — the declared load cases re-run on the finished
  mesh; the version is flagged `needs_review` when the worst safety factor
  drops below 1 or every feature was skipped.

`optimize.auto_refine_features: true` (spec or request) auto-enqueues the
refine child right after a successful optimize. `Feature.thread_spec`
(`"M4"`, `"M6x0.75"`, `"M8-6H"`…) overrides the nominal diameter for thread
re-cuts; `optimize.restore_planes` controls the keep-surface union (default
on, needs the pre-optimize base mesh on the parent/grandparent version). In
the agent optimize session the `refine_features(run_index)` tool does the
same on any completed `topopt_run` and registers the finished mesh as a new
selectable run index.

### Feature anchors

Measured anchors drive `hole`/`anchor`/`bore` regions on ANY axis. Sources:
top-view image holes, mesh tunnels, spec features — merged with an on-demand
axis-agnostic mesh scan (`fem/anchors3d.py`, +x/+y/+z heightfields) that also
enriches measured anchors with real depth. Anchor fields: `index` (1-based),
`kind` (hole/thread/insert/bearing_seat/dimple), `axis`, `x`/`y`/`pos`
(3D in-plane position), `z` (axial centre), `dia`, `depth_mm`,
`bolt_size`. `GET /versions/{id}/anchors` returns the merged table.

### Manual optimization and FEM check (structural editor)

On a completed version, **Structural editor** opens a full-screen 3D editor
where support, load and keep regions are painted directly on the part surface
with an adjustable brush (per-region radius, with eraser + per-stroke undo),
or declared as geometric regions: `face`/`surface` (+axis), `area`, `cylinder`,
`box`, `anchor`/`bore`/`hole` (measured anchors — `bore` selects the bore WALL
for bolt-type loads), or `all`. Anchor markers are rendered as numbered
sprites on the model; clicking near one while placing an anchor region snaps
`hole_index` to it. Load gizmos (force arrows, moment rings, preload pairs)
render at the resolved region centres. Painted picks are stored as
`RegionSpec kind="points"` — up to 4000 `[x,y,z]` mm surface points, a brush
radius (`radius_mm`), and the averaged surface normal (drives pressure loads).
Resolution rasterizes each point in index space (deduplicating stroke points
closer than half a pitch) and selects solid cells within `radius_mm + pitch`,
i.e. a near-surface patch of material.

Load kinds (`LoadSpec` fields): `force_n` (distributed resultant),
`pressure_mpa` (against the region normal), `bearing_n` + `bearing_dir`
(cosine-distributed shaft load on a `bore`/`anchor` region — the physically
correct bolt load), `moment_nm` (pure couple via rigidity tensor),
`preload_n` (bolt pretension squeezing the two bore mouths),
`force_n` + `remote_point` (same resultant plus the induced lever moment),
`g_load` (body acceleration in g units over the region's mass). Support
regions take `fix_mode`: `"all"` (encastre) or `"normal"` (pins only the
axis DOF — roller/resting/symmetry; reported as `(nnodes,3)` DOF masks).

Three actions share the declared load cases:

- **Check regions** — server-side voxel resolution only (`preview_only`):
  returns grid pitch/dims, per-case fixed/loaded node counts, total force and
  applied moment, warnings, and the *resolved* selection as subsampled
  cell-centre point clouds (`cells` — fixed/load/keep per case) for a 3D
  overlay. Legacy 2D overlay renders (fixed blue, load red, keep green PNGs)
  are only produced when `render_pngs` is set; the agent-side optimize
  session still uses them.
- **FEM check** — a standalone FEM run on the *same* version (`JobType.fem`,
  no child): writes `version.fem` plus stress/deformation artifacts and flags
  the version `needs_review` when the worst safety factor drops below 1.
- **Optimize** — creates an LLM-free `optimize` child version (SIMP + FEM
  verification) as before; `keep_regions` hold their material solid.

Selections, material and target volume persist on the version under
`options.ui_editor` (`ui_*` keys are stripped before settings merge).

## Load-case contract

Coordinates are millimeters; forces are Newtons; pressure and stress are MPa
(N/mm²). Each case contains fully fixed regions and one or more loads. Regions
can be a bounding-box face, cylinder, box, hole anchor, painted surface patch
(`points`), or the whole domain. Axes use
`+x`, `-x`, `+y`, `-y`, `+z`, `-z`.

Example request fragment:

```json
{
  "load_cases": [
    {
      "name": "cantilever",
      "fixed": [{"kind": "face", "axis": "-x"}],
      "loads": [
        {"region": {"kind": "face", "axis": "+x"}, "force_n": [0, 0, -100]}
      ]
    }
  ],
  "optimize": {
    "enabled": true,
    "target_volume_fraction": 0.4,
    "safety_factor_min": 1.5,
    "keep_regions": []
  }
}
```

A force load distributes its vector over its selected region. A pressure load
acts against the region normal — for `points` regions the GUI records the
averaged surface normal for this. For generated parts, make the physical
attachment/load locations explicit (for example, clamp the mounting holes
rather than an unrelated outer face) — prefer `bore`/`anchor` regions with
`bearing_n` for bolts and pins. The FEM report adds per-case `reactions`
(resultant at the supports) — compare it against the applied resultant for
equilibrium sanity.

## Numerical method and outputs

- Regular voxel grid with linear 8-node hexahedral elements (Hex8), isotropic
  linear elasticity, and 2×2×2 Gauss integration.
- SIMP stiffness interpolation, compliance objective, optimality-criteria
  update, and cone sensitivity filter; multiple load cases contribute to the
  objective (weighted via `LoadCase.weight`). Per-cell Young's modulus
  supports material zones; `x0` warm-starts from a prior density field.
  Element-assembly index arrays (`iK`/`jK`) are cached across iterations
  since the sparsity pattern is constant for a fixed domain.
- Mesh extraction uses marching cubes (scikit-image) when available and falls
  back to the voxel-boundary surface, so `skimage` is optional at runtime.
- The solver accepts per-DOF `(nnodes,3)` fixed masks; systems beyond
  `fem_spsolve_dofs` use SSOR-preconditioned conjugate gradients.
- Grid pitch starts at `fem_voxel_mm` and is adaptively coarsened to stay
  under `fem_max_elements`.
- Material properties come from the built-in table (`pla`, `petg`, `abs`,
  `asa`, `pa`, `pc`, `resin`, `aluminum`, `steel`); unknown names fall back to
  PLA. The table uses typical bulk properties and is not a manufacturing
  process/material certificate.
- `topopt_default_volfrac`, `topopt_max_iterations`, `topopt_default_rmin_mm`,
  `topopt_min_member_mm`, `topopt_smooth_iterations`, `topopt_auto_keep`,
  `topopt_auto_keep_margin_mm`, `fem_spsolve_dofs` and
  `fem_max_effective_cases` are runtime settings edited in the Settings page;
  `fem_timeout_s` and `topopt_timeout_s` bound executor task time. Per-run
  overrides live in the options panel / version options. The full
  settings reference is in [configuration.md](configuration.md).

The agent optimize session offers `list_anchors` (the measured anchor table),
`check_load` (region resolution + resultant/moment without a solve),
`preview_regions`, `analyze_fem`, `topopt_run` and `done`.

The FEM report includes per-case max von Mises stress, max displacement,
compliance, safety factor, solver/convergence status, and warnings. Safety
factor is based on the selected material's yield value divided by the computed
maximum von Mises stress. Reports and generated artifacts include:

```text
versions/<version-id>/fem/report.json
versions/<version-id>/fem/vonmises_<case>.png
versions/<version-id>/fem/deformed_<case>.png
versions/<version-id>/topopt/density.npz
versions/<version-id>/topopt/history.json
```

Renders are optional when offscreen VTK rendering is unavailable. The optimized
mesh exports normally (STL, 3MF, GLB); analytic STEP, DXF, and SVG are not
available from the voxel TopOpt result.

## Limits and interpretation

This is design-feedback FEM, not a certification solver. The current model does
not account for print-direction anisotropy, plasticity, contact, buckling,
fatigue, geometric nonlinearity, or manufacturing residual stress. Boundary
conditions and material properties dominate result quality; voxel resolution
adds discretization error, and loads at sharp constraints may create local
stress concentrations. Validate important designs with an independent
engineering analysis and physical testing. DfM and support/orientation checks
remain necessary for additive manufacturing; a watertight mesh is not itself a
printability guarantee.

## Verification

The native regression suite includes a cantilever comparison against the
Euler–Bernoulli beam solution, region-resolution checks, SIMP volume-fraction
and convergence checks, watertight extraction checks, executor-pipeline tests,
API tests, and a FakeLLM agent-loop test that exercises automatic TopOpt mode.
Run from `backend/`:

```bash
.venv/bin/python -m pytest tests/cad/test_fem.py tests/unit/test_agent_modes.py tests/unit/test_api.py -q
.venv/bin/ruff check cadx tests
.venv/bin/python -m mypy cadx
```

## See also

- [Configuration](configuration.md) — all `fem_*` / `topopt_*` settings
- [Web GUI](webgui-ux.md) — the structural editor
- [HTTP API](api.md) — `/fem`, `/fem-preview`, `/optimize`, `/topopt-resample`, `/refine`, `/anchors`
