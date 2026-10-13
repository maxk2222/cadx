# Evaluation

`cadx eval` (`make eval`) runs the benchmark suite through the real
agent loop — INTAKE→PLAN→CODEGEN→EXECUTE→ANALYZE→CRITIQUE→FINALIZE — with an
in-process executor, no queues required.

## Benchmarks

`backend/cadx/eval/benchmarks/*.json`:

```json
{
  "prompt": "40x20x10 block with 1.5mm edge fillets",
  "expected_bbox": [40, 20, 10],
  "tolerance_mm": 1.5,
  "mode": "cad",                  // optional: auto|cad|mesh|hybrid|topopt
  "images": ["fixtures/images/knob_ref.png"],  // optional input images
  "fake_mesh": {"size": [30, 22, 18]},         // optional: in-process fake mesh server
  "settings": {"max_iterations": 2},
  "runtime": {"mesh_candidates_per_iter": 1},  // optional: RuntimeSettings overrides
  "llm_tools": "auto",             // optional: enable scripted tool sessions
  "llm_script": [ ... ]          // optional: deterministic FakeLLM run
}
```

- With `llm_script`: fully deterministic — the FakeLLM replays the scripted
  INTAKE/PLAN/CODEGEN/CRITIQUE responses. Useful for regression-testing the
  pipeline itself.
- With `fake_mesh`: no mesh server is needed — a `FakeMeshClient` generates a
  deterministic parametric blob (knob-ish sphere + stem) at the given size.
- `cadx eval --mode mesh` runs only benchmarks with that `mode`.
- Without: the configured real provider/mesh server is used. Useful for
  measuring prompt/model quality.

Mesh benchmarks shipped:

| benchmark | mode | what it checks |
|---|---|---|
| `organic_knob_mesh.json` | mesh | MESH_GENERATE→ORIENT→PROCESS→CRITIQUE→export, fake mesh |
| `flat_scale_mesh.json` | mesh | hopeless-candidate pre-gate → CAD fallback → codegen, fake mesh |
| `knob_with_m4_insert_hybrid.json` | hybrid | mesh→solid→`load_mesh_solid()`→boolean cut→export |

## Results

Per benchmark the runner reports: final status, score, measured bbox vs
`expected_bbox` within `tolerance_mm` (`dim_ok`), wall time, error. Pass =
`succeeded` and `dim_ok`.

## Baseline

Target from the master plan: ≥ 8/10 pass rate with the default cloud model.
Record results here after each prompt/model change.

| Date | Model | Suite | Pass | Notes |
|------|-------|-------|------|-------|
| _initial_ | — | 2 scripted | 2/2 | FakeLLM pipeline check |

Add benchmarks for each new part class (brackets, enclosures, gears,
heat-set inserts, thin walls) — the suite should grow with user feedback.
