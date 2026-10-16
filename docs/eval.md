# Evaluation

`cadx eval` (`make eval`) runs the benchmark suite through the real agent
loop — INTAKE → PLAN → CODEGEN → EXECUTE → ANALYZE → CRITIQUE → FINALIZE —
with an in-process executor, no queues required.

```bash
cadx eval                          # all scripted (FakeLLM) benchmarks
cadx eval --benchmark l_bracket_cad_scripted
cadx eval --mode mesh              # only benchmarks with this gen_mode
cadx eval --real                   # real configured LLM (real-only benchmarks)
cadx eval --real --real-all        # also run scripted benchmarks with the real LLM
cadx eval --real -b plate_holes --repeat 3 --queue-exec --json out.json
```

`--queue-exec` dispatches execution through the cad queue (production-true —
runs inside the executor container incl. the warm fork-server). `--repeat N`
prints a median aggregate per benchmark; `--json` writes every EvalResult
including `step_stats` (per-step calls/tokens/latency) for A/B reports.

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
| `expected_features` | deterministic feature gate — `{type: "hole", count, dia, axis, positions, pcd}` checks measured hole centers (±1.5 mm) and bolt-circle diameter; `slot`, `countersink`, `counterbore`, `fillet`, `chamfer` count measured features |
| `expect_watertight` | require a watertight result mesh |
| `max_wall_s` / `max_prompt_tokens` | per-benchmark cost gates |

## Shipped benchmarks

| Benchmark | gen_mode | Scripted | Notes |
|-----------|----------|----------|-------|
| `rounded_box` | auto | yes | baseline parametric part |
| `plate_holes` | auto | yes | hole features + positions |
| `l_bracket_cad` / `l_bracket_cad_scripted` | cad | real / yes | dims, slot, ribs, fillet, hole positions |
| `flange_pcd_cad` | cad | real only | hub flange: 6-hole bolt circle (`pcd`), central bore, chamfer, fillet |
| `plate_holes_lite` | cad | real only | same part as `plate_holes` under `agent_plan_mode=lite` (A/B) |
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
| M8 post-change (cad focus) | local endpoint | real, `--queue-exec` | see below | warm runner, convergent repair, plan-lite, feature gates |

M8 measured real runs (all `dim_ok=True`, `watertight=True`):

| Benchmark | n | Passed | Score (min/med/max) | Wall median | Prompt tokens median | Notes |
|-----------|---|--------|--------------------|-------------|---------------------|-------|
| `plate_holes` (full plan) | 3 | 3/3 | 100/100/100 | 75 s | 24,702 | 1 exec each, 4 calls |
| `plate_holes_lite` | 3 | 3/3 | 100/100/100 | 90 s | 21,928 | plan prompt 3,935 → 1,221 tok; one rep needed a repair exec (79,913 tok) |
| `flange_pcd_cad` | 3 | 0/3 → 1/1 after feature fix | 90/90/92 | 194 s | 177,918 | all three were geometrically correct; `features_ok=False` was a measurement gap (torus fillet + coaxial rim chamfer), fixed in `features.py` |
| `l_bracket_cad` | 3 | 0/3 | 15/50/55 | 1,147 s | 454,152 | watertight + dims OK; missing slot and detached/protruding ribs persist — repair now converges (bounded sessions, no budget-death) but the model still fails these features |
| `l_bracket_cad` (post code-contract + oblong_slot) | 3 | 0/3 | 10/25/50 | 1,163 s | 535,093 | model now calls `oblong_slot("+z","+x")` but misuses BuildPart semantics (cutter fused then re-added → ghost pillar) and misplaces ribs — new failure classes, now lintable |

Second l_bracket series surfaced four concrete gaps, all fixed the same
day: builder-fused solids re-`add()`ed (ghost pillar → `dim_ok`/`slots`
miss), double transforms on the same variable (`moved()` then
`translate()` → misplaced cutters), vertical-slot helper missing a
`normal` axis, and tunnel/slot detection silently disabled on
non-watertight STL (now repaired via `fill_holes`, marked `approximate`).
All three reps' failing code paths are caught by the new lint warnings —
a confirmation run on the final image validates the chain end-to-end.

Takeaways: plan-lite cuts the plan prompt ~3.2× but barely moves total cost
on a trivial part (LLM latency dominates; a single repair doubles tokens).
Complex parts remain expensive — L-bracket repair churn is bounded now but
slot/rib generation is the next leverage point (new code-contract rules
shipped; effect not yet re-measured).

Add benchmarks for each new part class (brackets, enclosures, gears,
heat-set inserts, thin walls) — the suite should grow with user feedback.

## See also

- [Agent loop](agent-loop.md) — what the harness exercises
- [Getting started](getting-started.md) — CLI quick reference
