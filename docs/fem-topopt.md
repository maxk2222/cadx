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
target volume fraction. The extracted boundary mesh is regularized for
watertight export. FEM verification uses the corresponding voxel mask rather
than the smoothed export mesh, so export smoothing does not change the
reported structural result.

### Manual optimization

On a completed version, use **Topology-optimize this part** to create an
LLM-free child version. The dialog accepts one or more load cases, face
supports/loads, force magnitude and direction, material, and target volume
fraction. When the dialog receives no load cases from the parent spec, define
them there. The worker persists the optimized mesh, report, and optimizer
history on the child version.

## Load-case contract

Coordinates are millimeters; forces are Newtons; pressure and stress are MPa
(N/mm²). Each case contains fully fixed regions and one or more loads. Regions
can be a bounding-box face, cylinder, box, or the whole domain. Axes use
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
uses `pressure_mpa` instead of `force_n`. For generated parts, make the
physical attachment/load locations explicit (for example, clamp the mounting
holes rather than an unrelated outer face). The manual dialog currently
models simple axial face forces; detailed region types and multi-component
forces can be expressed through the API/schema or prompt-derived spec.

## Numerical method and outputs

- Regular voxel grid with linear 8-node hexahedral elements (Hex8), isotropic
  linear elasticity, and 2×2×2 Gauss integration.
- SIMP stiffness interpolation, compliance objective, optimality-criteria
  update, and cone sensitivity filter; multiple load cases contribute to the
  objective.
- Grid pitch starts at `fem_voxel_mm` and is adaptively coarsened to stay
  under `fem_max_elements`.
- Material properties come from the built-in table (`pla`, `petg`, `abs`,
  `asa`, `pa`, `pc`, `resin`, `aluminum`, `steel`); unknown names fall back to
  PLA. The table uses typical bulk properties and is not a manufacturing
  process/material certificate.
- `topopt_default_volfrac`, `topopt_max_iterations`, and
  `topopt_default_rmin_mm` set optimizer defaults. `fem_timeout_s` and
  `topopt_timeout_s` bound executor task time. All of these are runtime
  settings edited in the Settings page; per-run overrides live in the
  options panel / version options.

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
