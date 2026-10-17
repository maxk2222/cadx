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

## Variants & comparison

`--variant NAME=<json>` (repeatable) overlays a partial `RuntimeSettings`
object onto every selected benchmark — the benchmark's own `runtime` keys
win on conflict (variants set defaults, benchmarks may pin). Each variant
runs the whole selection and writes per-run summaries to
`$DATA_DIR/eval_out/<bench>/<NAME>/<run_id>/result.json`; un-named runs keep
the flat `…/<bench>/<run_id>/` layout and record `"variant": "default"`.

```bash
# reasoning on/off/adaptive matrix
cadx eval --real -b plate_holes \
  --variant thinking_off='{"llm_thinking":"off"}' \
  --variant auto='{}' \
  --variant always='{"llm_thinking":"always"}'

# run under the backend env (cadx package + structlog)
cd backend && ../.venv/bin/uv run python ../scripts/eval_compare.py
../.venv/bin/uv run python ../scripts/eval_compare.py --json cmp.json
```

`scripts/eval_compare.py` groups the summaries by variant x benchmark and
reports pass rate, exec-ok rate (runs reaching >=1 sandbox execution), mean
score, mean iterations, prompt + completion tokens (with a reasoning split
for steps whose trace record carries the `thinking` flag), and wall time.
Set `CADX_EVAL_OUT` or pass `--root` to point it at a different tree.

Running `--real` from the host needs the DB for runtime settings — postgres
is not published to the host, so run it inside a container that reaches both
`cadx_internal` (postgres/redis) and `cadx_public` (LLM egress). The
`cad-cad-executor` sits on the internal net only (no egress — runs die at
intake with DNS errors); the worker has both but gets recreated on
`docker compose` deploys mid-run. Robust pattern: a dedicated container
(`docker run -d --network cadx_public ... cadx-worker`, then
`docker network connect cadx_internal <name>`) plus `nohup cadx eval ... >
/data/eval_matrix.log` so the run survives the exec session.

## External CAD-bench tasks

Eight tasks ported from the MIT-licensed CAD-bench prompt set live as
`cadbench_*.json` (cube, boxes, ring, L-bracket, extrusion, slotted plate,
flange, PCD ring). They are `real_only` CAD-mode benchmarks that run through
CADX's own `result` contract and bbox/feature/watertight gates — an
anti-overfit check against prompts the benchmark suite was not tuned on.
None of the source benchmark's verifier or gold code was vendored.

## SFT / repair-pair export

`scripts/export_sft.py` harvests verified training pairs from run traces:
codegen examples (prompt → code that executed successfully and passed the
quality gates) and repair pairs (failed attempt → later successful attempt
from the same version). Output is Axolotl/messages-style JSONL,
deterministically ordered and deduplicated; secret-shaped strings are
filtered and only verified code is ever emitted.

```bash
../.venv/bin/uv run python ../scripts/export_sft.py --out sft.jsonl
../.venv/bin/uv run python ../scripts/export_sft.py --kinds repair --out repair.jsonl
# also harvest real project runs (not just eval_out):
../.venv/bin/uv run python ../scripts/export_sft.py --include-project-versions --out all.jsonl
```

## Real-model matrix (2026-10, qwen3.8-27b @ remote vLLM)

3 variants x 3 CAD benchmarks x 1 rep, `--queue-exec`. `staged_auto` =
staged build + thinking auto, `whole_auto` = classic codegen + thinking
auto, `staged_off` = staged build + thinking off.

| benchmark | staged_auto | whole_auto | staged_off |
|---|---|---|---|
| plate_holes | PASS 95 (76 s, 1.6k ctok) | PASS 92 (78 s) | PASS 95 (72 s) |
| flange_pcd_cad | PASS 95/92, 1x FAIL | PASS 95 (629 s, 14.8k ctok) | PASS 92 (156 s, 3.3k ctok) |
| l_bracket_cad | needs_review 0 (10.5k s, 260k ctok) | needs_review 40 (9.1k s, 233k ctok) | **PASS 100** (709 s, 15.9k ctok) |

Findings:

- **Thinking off won everywhere on this model.** Qwen3-27B thinking mode
  produced 5k+ reasoning chunks per call (2-10 min/call); sessions burned
  230-260k completion tokens over 2.5-3 h *without converging*. The same
  l_bracket that scored 0-40 under thinking passed with 100 when thinking
  was off — in ~1/14 of the wall time and ~1/15 of the completion tokens.
  `llm_thinking=auto` escalated correctly (stall/patch-miss latch fired and
  was traced), but escalation amplified cost instead of rescuing runs.
- **High run-to-run variance**: staged_auto on flange passed twice (92, 95)
  and failed once (all 8 iterations exec-fail). n=1 is not enough to rank
  variants — treat single cells as indicative, not conclusive.
- **Fixed defect — "codegen session produced no code"**: under thinking,
  build sessions repeatedly ended without a `done` submission (huge
  reasoning calls eat the turn budget). The codegen step now submits a
  grown buffer when the session ends without `done` (f4da8ff); a
  seed-only buffer still fails correctly. Post-fix validation:
  `l_bracket_cad` `staged_bufrescue` **PASS 92** in 464 s (25 calls,
  3 iterations, all checks true) — best `staged` result on this
  benchmark, though the fix itself was not exercised (all sessions
  submitted `done` normally; run-to-run variance is high).
- **The recurring l_bracket trap**: spec'd ribs must sit *inside* the 40 mm
  envelope; all variants initially place them proud (42.5-43.5 mm) and the
  leg slot on the wrong axis. Only staged_off converged through the
  improve/repair loop.
- `verify_spec` (M6) reports appeared in production issues
  (`verify_spec PASS/FAIL — N pass, M fail`), confirming the canonical
  conformance path is live; quiddity pattern evidence and the STEP
  roundtrip gate ran without blocking runs.

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
| `cadbench_*` (8) | cad | real only | external CAD-bench ports — see below |

## External benchmarks

The `cadbench_*` benchmarks adopt tasks from the public
[CAD-bench](https://github.com/CAD-bench/cad-bench) suite (MIT license) —
a Harbor benchmark of 17 build123d tasks whose graders score topology and
dimensions deterministically (mesh ray-probes plus weighted `near_score`
composites, continuous 0-1 `overall_score`). They exist to counter
overfitting to our own hand-written benchmark prompts.

**What was ported:** each task's `instruction.md` requirements (verbatim)
and the expected geometry from its `contract.toml`. Their verifier code,
Harbor task harness, gold solutions, and fixtures were **not** vendored —
CADX runs the prompts through its own agent loop and judges with its own
checks. The CAD-bench submission contract ("leave the model in a
top-level variable named `part`") was rewritten to the CADX contract
(`result`) as a contract line appended to the prompt.

**Grading mapping** (their check → CADX gate):

| CAD-bench verifier | CADX gate |
|---|---|
| bbox size `near_score` (0.05-0.45 mm zero band) | `expected_bbox` + `tolerance_mm` (sorted axes, max(tol, 2%) per dim) |
| z-pose / XY centering probes | prompt requirements + CADX frame contract (not separately gated) |
| hole openness ray-probes | `expected_features` `hole` entries (`count`, `dia`, `axis`, `positions` ±1.5 mm, `pcd`) |
| slot openness probes | `expected_features` `slot` count |
| single-solid mesh build | `expect_watertight` |
| `volume_mm3` in contract.toml | not checked — approximated by dims + feature gates |
| level heights, ring/harmonic profiles | not checkable — part of tolerance/scope notes below |

`tolerance_mm` values (0.5-1.0 mm) are tighter than most native CADX
benchmarks because the CAD-bench graders use sub-millimetre zero bands;
ours remain looser pass/fail gates, so a CADX pass is necessary but not
sufficient for a CAD-bench-perfect score.

**Ported tasks** (source `dataset/<task_id>` → benchmark name → difficulty):

| CAD-bench task | Benchmark | Difficulty | Encoded gates |
|---|---|---|---|
| `cube_20mm_z_minus` | `cadbench_cube_20mm` | easy | 20³ bbox, z≤0 pose via prompt |
| `box_30x20x10_z_plus` | `cadbench_box_30x20x10` | easy | 30×20×10 bbox, z≥0 pose |
| `ring_30_20_h10_z_plus` | `cadbench_ring_30_20_h10` | easy | 30×30×10 bbox, Ø20 through-bore |
| `l_bracket_40x40x6_2_hole` | `cadbench_l_bracket_40x40x6` | medium | 40×40×6 bbox, 2×Ø6 holes at (10,10)/(30,10); L footprint only via prompt |
| `aluminum_extrusion_20x20x60_5_bore` | `cadbench_extrusion_20x20x60` | medium | 20×20×60 bbox, 1×Ø5 + 4×Ø3 bores at fixed XY |
| `plate_80x40x8_slot_4_hole` | `cadbench_plate_80x40x8_slot` | medium | 80×40×8 bbox, slot + 4×Ø6 holes at (±30,±15); slot size only via prompt |
| `flange_40_24_h20_bolt4` | `cadbench_flange_40x24_h20` | hard | 40×40×20 bbox, Ø12 centre bore, 4×Ø5 on PCD30; boss Ø24/step via prompt |
| `ring_50_30_h12_6_bolt4_pcd40` | `cadbench_ring_50x30_pcd40` | hard | 50×50×12 bbox, Ø30 bore, 6×Ø4 on PCD40 |

**Not ported:** the three `insane` gearbox tasks and
`custom_threaded_pair_nonstandard` (Blender rigid-body simulation
graders), `m3x6_socket_head_zminus` (needs a fixture STEP oracle),
`spur_gear_m1_z16_t6_bore5` and `hex_nut_af17_h8_bore10` (tooth/hexagon
harmonic profile checks have no CADX equivalent; hex bbox is also
rotation-dependent), `plate_100x60x12_slot_8_hole` and
`step_block_60_40_30_4_hole` (redundant with the ported plate/L-bracket
class; the step-block's per-level heights aren't expressible as
`expected_features`).

All eight are `real_only` (no `llm_script` — external anti-overfitting
benchmarks make no sense against a canned FakeLLM). Run them with
`cadx eval --real -b cadbench_<name>`.

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
