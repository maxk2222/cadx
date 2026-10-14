# Evaluation

`cadx eval` (`make eval`) runs the benchmark suite through the real agent
loop — INTAKE → PLAN → CODEGEN → EXECUTE → ANALYZE → CRITIQUE → FINALIZE —
with an in-process executor, no queues required.

```bash
cadx eval                          # all scripted (FakeLLM) benchmarks
cadx eval --benchmark l_bracket_cad_scripted
cadx eval --mode mesh              # only benchmarks with this gen_mode
cadx eval --real                   # real configured LLM (real-only benchmarks)
```

Without `--real`, benchmarks that are `real_only` or lack an `llm_script` are
skipped with a note. `--real` builds the client from the DB runtime settings —
the same endpoint/key/models a real job would use.

## Benchmark format

`backend/cadx/eval/benchmarks/*.json` — filename becomes the benchmark name:

```json
{
  "prompt": "40x20x10 block with 1.5mm edge fillets",
  "expected_bbox": [40, 20, 10],
  "tolerance_mm": 1.5,
  "gen_mode": "cad",
  "images": ["backend/tests/fixtures/images/knob.png"],
  "fake_mesh": true,
  "settings": {"max_iterations": 2},
  "runtime": {"mesh_candidates_per_iter": 1},
  "llm_tools": "auto",
  "llm_script": [ ... ],
  "real_only": false,
  "expected_features": [{"type": "hole", "count": 2, "dia": 5.5}],
  "expect_watertight": true,
  "max_wall_s": 600,
  "max_prompt_tokens": 1500000
}
```

| Field | Meaning |
|-------|---------|
| `prompt` | user request sent to the agent (required) |
| `expected_bbox` / `tolerance_mm` | dimension gate (tolerance default 1.0 mm) |
| `gen_mode` | `auto` / `cad` / `mesh` / `hybrid` (default `auto`) |
| `images` | repo-relative fixture image paths |
| `fake_mesh` | use `FakeMeshClient` — a deterministic parametric blob, no mesh server needed |
| `settings` | `ProjectSettings` overrides (`max_iterations`, `gen_mode`-independent knobs) |
| `runtime` | `RuntimeSettings` overrides for this run |
| `llm_script` | scripted FakeLLM responses — fully deterministic, no provider |
| `llm_tools` | `auto` enables scripted tool sessions; default `off` |
| `real_only` | skip in `cadx eval`; only runs under `--real` |
| `expected_features` | deterministic feature gate (hole counts/diameters, …) |
| `expect_watertight` | require a watertight result mesh |
| `max_wall_s` / `max_prompt_tokens` | per-benchmark cost gates |

## Shipped benchmarks

| Benchmark | gen_mode | Scripted | Notes |
|-----------|----------|----------|-------|
| `rounded_box` | auto | yes | baseline parametric part |
| `plate_holes` | auto | yes | hole features + positions |
| `l_bracket_cad` / `l_bracket_cad_scripted` | cad | real / yes | dims, slots, ribs, fillet |
| `organic_knob_mesh` | mesh | yes (fake mesh) | generate → orient → process → critique → export |
| `flat_scale_mesh` | mesh | yes (fake mesh) | hopeless-candidate pre-gate → CAD fallback |
| `grip_mesh` | mesh | real only | organic grip, real mesh server |
| `knife_scale_hybrid` | hybrid | real only | mesh → solid → CAD features |
| `knob_with_m4_insert_hybrid` | hybrid | yes (fake mesh) | `load_mesh_solid()` + boolean cut → export |
| `firesteel_auto` / `firesteel_auto_scripted` | auto / cad | real / yes | multi-feature part under auto mode |

## Results

Per benchmark the runner reports: final status, score, measured bbox vs
`expected_bbox` within `tolerance_mm` (`dim_ok`), feature check
(`features_ok`), watertightness, LLM calls and prompt tokens, wall time and
error. A benchmark passes when it finishes `succeeded`/`needs_review` and all
set gates (dims, features, watertight, budgets) hold.

## Baseline

Record results here after each prompt/model change:

| Date | Model | Suite | Pass | Notes |
|------|-------|-------|------|-------|
| _initial_ | — | scripted | pipeline check | FakeLLM regression runs |

Add benchmarks for each new part class (brackets, enclosures, gears,
heat-set inserts, thin walls) — the suite should grow with user feedback.

## See also

- [Agent loop](agent-loop.md) — what the harness exercises
- [Getting started](getting-started.md) — CLI quick reference
