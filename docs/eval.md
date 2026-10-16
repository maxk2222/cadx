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
| `l_bracket_cad` (full fix chain: lint + slot_cutter + hints) | 1 | 0/1 | 15 | 1,047 s | 327,908 | lint clean, helpers used, issue text carries the exact call — model still picks wrong axes (`direction="+y", normal="+z"` instead of `"+z"/"+x"`) and leaves ribs proud (width 60 vs 40) |

Second l_bracket series surfaced four concrete gaps, all fixed the same
day: builder-fused solids re-`add()`ed (ghost pillar → `dim_ok`/`slots`
miss), double transforms on the same variable (`moved()` then
`translate()` → misplaced cutters), vertical-slot helper missing a
`normal` axis, and tunnel/slot detection silently disabled on
non-watertight STL (now repaired via `fill_holes`, marked `approximate`).
All three reps' failing code paths are caught by the new lint warnings.

The confirmation run on the complete fix chain is the decisive data
point: infrastructure, hints and helpers all work as designed (lint
clean, `oblong_slot` actually called, issue text contains the exact
`direction="+z", normal="+x"` invocation) — yet the model still derives
the wrong axes and proud ribs. Honest read: prompt/lint fixes are
necessary but not sufficient on this benchmark. The remaining lever is
structured generation — let the plan emit a feature→cadlib mapping
(e.g. `{"type":"slot","direction":"+z","normal":"+x",...}`) that is
instantiated deterministically, so the model never re-derives axes from
prose; or a `cadlib.l_bracket()` template selected at plan level.

Takeaways: plan-lite cuts the plan prompt ~3.2× but barely moves total cost
on a trivial part (LLM latency dominates; a single repair doubles tokens).
Complex parts remain expensive — L-bracket repair churn is bounded now
(stall-patience stops after ~5 flat iterations instead of burning all 8)
but slot/rib generation does not improve from prompt hints alone;
the evidence points at structured feature→cadlib instantiation as the
next step.

## Binder era (deterministic feature→cadlib snippets)

The structured-generation step is implemented: `featurebind.py` resolves
each spec feature into an exact cadlib snippet (`BOUND FEATURE SNIPPETS`
block in the codegen/improve/repair context); the model fills only `<…>`
placement coordinates. Snippets are self-contained `BuildPart` blocks and
carry the required `from cadlib import …` line.

| Benchmark | n | Passed | Score | Wall | Prompt tokens | Notes |
|-----------|---|--------|-------|------|---------------|-------|
| `l_bracket_cad` (binder, pre slot-merge fix) | 1 | 0/1 | 55 | **449 s** | 647,061 | dim_ok=True exact 60×40×54, watertight — the first rep produced correct geometry in one shot; `features_ok=False` was a NEW measurement gap: the mesh tunnel scan only ran when the B-Rep found zero analytic holes, so a real slot next to real holes was invisible |
| `l_bracket_cad` (binder + slot-merge fix) | 2 | **2/2** | 30/92 | 578 s | 316,194 | first passing reps ever: `dim_ok`, `watertight`, `features_ok` all true on both — bound snippets produce the vertical slot deterministically. Residual: rep-1 score 30 from a mis-oriented slot in an early iteration (recovered to features_ok by finalize); rep-2 score 92 in **276 s** with only minor issues (4 mm rib overhang on X, foot pattern emitted 1/2 holes) |

Rep-1 and rep-2 of the binder era each surfaced a failure class that was
fixed the same day:

- bound snippets emitted bare helper calls but every cadlib cutout REQUIRES
  `BuildPart` context (and `safe_fillet` returns the applied radius, not a
  part — the model wrote `part = safe_fillet(...)`, float → TypeError).
  Snippets are now self-contained `with BuildPart() as _xb:` blocks.
- `Box(...).locate(Pos(...))` inside `BuildPart` silently keeps the fused
  copy at the origin (locate is in-place on the returned object only) —
  observed leg at z=0..50 in the fused part while the returned object sat
  at z=4..54. New static lint warning + contract rule.
- bare helper names in snippets hit `NameError` under `import cadlib` —
  the block now emits the `from cadlib import …` line deterministically.
- mirrored `-Y` rib improvised as `Rot(0,180,0)` flipped the rib under the
  part — rib snippets now forbid rotation and document the ~0.5 mm sink
  into both contact faces.

Infra finding (not a code bug): the cad-executor runs every task via
`asyncio.to_thread` — one hung CPU-bound job (`export_all` from a parallel
eval) starved the GIL so the arq loop never dequeued `execute_version`;
jobs timed out at 480 s while the queue looked empty. Mitigated:
`cad/tasks.py` now wraps the pipeline thread in an `asyncio.wait_for`
deadline (`exec_timeout_s`) so a wedged run returns a timeout failure
instead of occupying the worker forever. Full isolation would still need
a process pool — the watchdog cannot kill the underlying thread.

Add benchmarks for each new part class (brackets, enclosures, gears,
heat-set inserts, thin walls) — the suite should grow with user feedback.

## See also

- [Agent loop](agent-loop.md) — what the harness exercises
- [Getting started](getting-started.md) — CLI quick reference
