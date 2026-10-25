# Evaluation — evalkit

`cadx eval` runs a small, canonical suite of real benchmark objects through
the production pipeline. Two drivers:

- **http** (default, durable): every stage is spawned through the REST API —
  real `Version`/`Job` rows under fixed `eval:*` projects, visible in the
  GUI, executed by the normal arq worker + executor stack. Results are
  archived under `evals/results/` (gitignored).
- **local** (`--local`, scratch): the same suite in-process against a
  throwaway sqlite + data dir inside the run folder — the *production*
  spawn + `run_*_job` path with a no-op enqueue and an in-process executor.
  No API, Redis or Docker needed; results land in `evals/scratch/`
  (gitignored, ephemeral) and are excluded from reports and baselines
  unless explicitly included (`--include-scratch`).

```bash
make eval                            # durable suite run via the live API
make eval-quick                      # scratch run — fast dev loop
cd backend && ../.venv/bin/uv run cadx eval cases   # list suite/stages/gates
cadx eval run --case l_bracket_rib --stages codegen --label experiment
cadx eval run --local --case robot_arm_mount --stages fem --from <dir>
```

## The suite

`evals/suite/*.json` — one file per canonical object, sorted by
`difficulty`:

| Case | Difficulty | Object | Stage chain |
|------|-----------|--------|-------------|
| `l_bracket_rib` | 1 | aluminium L-bracket + triangular rib | `codegen` → `fem` (+ `params`, disabled) |
| `sprocket_520_45t` | 2 | 7075-T6 rear sprocket, 45 teeth, 520 chain | `codegen` (+ `imagegen` → `meshgen`, disabled) |
| `robot_arm_mount` | 3 | topopt-ready robot-arm mount | `codegen` → `fem` → `topopt` → `refine` |
| `bearing_bracket_topopt` | 3 | generative-design bearing bracket — interfaces-only prompt | `codegen` → `fem` → `topopt` → `refine` |
| `agent_fem_bc` | 2 | agent-declared boundary conditions (cantilever bar) | `codegen` → `fem` (spec fallback + `fem_loadcases`) |

`bearing_bracket_topopt` is the *interfaces-only* / generative-design
pattern: the prompt declares just the mounting interfaces (3×M8 bolt
triangle + a dia-30 bearing bore) and a maximum envelope — the agent is
free to pick the seed topology, the optimizer designs the actual
structure (`keep_mode:"auto"` preserves the interfaces), and the
boundary conditions are declared deterministically in the stage
`load_cases`: the radial case drives `bearing_n` on a `bore` region
bound by its `(axis, dia, center)` hint — no `hole_index`, so the
binding survives any scan-order or topology change.

`robot_arm_mount` (and the `robot_arm_infill_*` A/B cases) declare the
same boundary conditions in every stage: `axial` puts the 600 N arm
weight on the hint-bound M12 seat wall (`bore`), `overturn` applies a
100 N·m couple spread over the boss section (`cylinder` clipped to
`z_range [10,25]` — the dominant real load; the thin bore-wall ring
would concentrate the couple into a discretization artifact), and
`link` pulls on the two Ø6.5 flange pin holes (`hole` regions) so the
flange is structurally necessary. The `fem_loadcases` gate asserts the
resolved physics — resultant/centroid/mechanism per case plus the
`moment_mag_nm` band for the pure-couple case.

Each stage maps to exactly one pipeline phase and entity outcome:

| Stage type | Route / job fn | Entity |
|---|---|---|
| `generate`, `edit`, `params`, `edit_feature`, `optimize`, `resample`, `refine`, `organic`, `mesh_edit`, `meshgen` | `POST /projects/{id}/generate` etc. | child `Version` + `Job` |
| `fem`, `reexport` | `POST /versions/{id}/fem`, `/reexport` | `Job` on the *same* version (writes `version.fem`/exports) |
| `imagegen`, `imageedit` | `POST /projects/{id}/images` | `Asset` rows (synchronous, no job) |

Stage chaining is implicit: a stage's parent is the previous
child-version stage. `fem`/`reexport` don't advance the chain (same-version
jobs), so `codegen → fem → topopt` runs FEM on the codegen version and
optimizes from it. `--stages` selects a subset (ids or types);
`--from-version` (http) / `--from <dir>` (local, STL/BREP) injects a
parent. Disabled stages (`"enabled": false`) are skipped unless explicitly
selected by id.

### `use_parent_fallbacks` (default `true`)

`fem`/`optimize`/`resample`/`refine` stages may leave `load_cases` empty —
the drivers then resolve the parent's declared cases exactly like the job
layer does (`worker_jobs.py`): `fem`/`optimize` read `parent.spec`
→ `load_cases`, `resample`/`refine` read `parent.params` first (the
previous run's actual inputs) then `parent.spec`. The outcome's
`extras.load_cases_source` records the provenance (`parent_spec` /
`parent_params` / `stage`). Setting the flag `false` skips the driver
injection only — the product-side fallbacks still apply (`spawn_fem` and
`run_fem_job` resolve `spec.load_cases` themselves, so a fem stage without
cases behaves exactly like the GUI/API path). This is what makes
"the agent must pick the BCs itself" cases possible: the suite declares no
cases, the generated spec carries them, and `fem_loadcases` grades what
resolved.

## Gates

Per-stage declarative checks in the suite JSON, evaluated in
`cadx.evalkit.gates`:

- **outcome gates** — `status`, `score`, `bbox` (`match`/`within`,
  `tol_mm`), feature gates (`holes`/`tunnel`/`slot`/`countersink`/
  `counterbore`/`fillet`/`chamfer` with `count`/`dia`/`axis`/`positions`/
  `pcd`), `watertight`, `verify_spec`, `issue_contains`, `code`, `mode`,
  `fem` (`field`/`min`/`max`, resolves across `worst_safety_factor` →
  `sf_verified` → `sf_export` → per-case), `topopt` (`field`/`min`),
  `volfrac`, `envelope_within`, `asset_image`, `execs_min`, `max_wall_s`,
  `max_prompt_tokens`.
  `fem_loadcases` is the **semantic BC gate** — it re-checks the *resolved*
  load cases (not the request): `min_cases`, `source` (asserts
  `extras.load_cases_source`), and per-case `checks` on
  `resultant_n {min,max}`, `resultant_dir` (+`dir_tol_deg`),
  `load_centroid_mm` (force-weighted applied-load centroid),
  `fixed_centroid_mm` (where the supports actually landed — both
  `tol_mm`), `fixed_at_axis`/`load_at_axis` (centroid within `tol_mm` of
  the bbox extreme along `+x`/`-z`/… — "clamped at the -X end" without
  pinning absolute coordinates), `mechanisms` (subset of the applied
  load kinds),
  `warnings_absent` (substrings that must not appear),
  `sf_min`/`sf_max`, and `declared_resultant_rel_tol` (resolved resultant
  vs. the spec-declared `force_n` sum — catches "asked 200 N down, got
  40 N sideways"). Missing evidence (no fem report, no reactions, no
  declared case) makes a check inconclusive (`ok=null`), never a silent
  pass or a noise failure.
- **mesh gates** — `radial_peaks` (tooth/lobe count on a polar radius
  profile), `volume`, `volume_fraction` (`of: bbox|disk`), `contains_point`
  (rib/gusset probe), `bodies`. Computed with trimesh on the stage's mesh
  artifact; they fail with a detail note when no mesh exists.

A stage passes when all gates hold and the status isn't
failed/cancelled/timeout; with no gates declared, `succeeded` alone
decides.

## Run archives

Every run writes `<root>/<case>/<ts>_<label>/` (`evals/results` for http,
`evals/scratch` for local):

- `result.json` — schema_version 2: case, driver, variant, stage results
  (status, metrics, gate verdicts), run meta.
- `meta.json` — git sha/dirty flag, applied runtime overrides.
- `summary.md` — per-stage table with failed gates.
- `stages/<id>.json` — persisted after every stage (crash-safe).
- `trace.<vid>.jsonl` — AgentTrace records (http: `GET /jobs/{id}/trace`,
  local: scratch DB rows).
- `artifacts/<stage>/` — downloaded/collected mesh artifacts (http).

## Variants & comparison

`--variant NAME=<json>` (repeatable) overlays `RuntimeSettings` fields —
validated before any run starts, so typos fail fast. http runs swap the
live settings (`backup → PUT → run → restore`, restore filtered to
`SettingsUpdate` fields); local runs use `runtime.model_copy()` and never
touch the real DB. A case's own `runtime` keys are merged last for that
case only. `--settings <json>` applies extra runtime overrides on top of
every variant (http only).

```bash
cadx eval run --case sprocket_520_45t \
  --variant thinking_off='{"llm_thinking":"off"}' \
  --variant auto='{}' \
  --repeat 2
cadx eval report                       # variant x case table
cadx eval report --include-scratch     # fold scratch runs in (opt-in)
cadx eval baseline save before-change  # latest run per (case,variant) -> evals/baselines/
cadx eval baseline diff before-change after-change
cadx eval report --baseline before-change   # diff current results vs baseline
```

Baselines are committed under `evals/baselines/`; `diff` reports pass
flips, per-stage gate flips and metric regressions (score −3 pts,
wall/tokens +20/25%).

## Cleanup

```bash
make eval-clean                              # dry-run
cadx eval clean --wipe-db --purge-files --yes   # execute
```

`--wipe-db` deletes every project-scoped row in FK-safe order
(AgentTrace → Job → Artifact → Version → Asset → ChatMessage → Chat →
LibraryPart → Project), refuses while jobs are queued/running unless
`--force`, and preserves `AppSetting` (runtime settings + encrypted
secrets) unless `--with-settings`. `--purge-files` removes
`data/projects/`, `data/eval_*` and the `evals/results|reports|scratch`
trees — `evals/suite` and `evals/baselines` are never touched. Both are
dry-run by default; `--yes` executes.

## API additions

- `POST /api/projects/{id}/meshgen` — image→mesh over REST (same
  `spawn_meshgen` path the MCP tool uses); validates image assets.
- `GET /api/jobs/{id}/trace?limit=&full=` — AgentTrace records of a job.
  Slim by default (step/iteration/model/tokens/latency/parsed);
  `full=1` adds `messages`/`response`.

## SFT / repair-pair export

`scripts/export_sft.py` harvests verified training pairs from run traces
(codegen prompt → code that executed and passed gates; repair pairs:
failed attempt → later success). It reads the v2 result schema under
`evals/results` (one context per code-bearing stage, resolving scratch
run `data/` subtrees too) and remains backward-compatible with old
`eval_out` v1 summaries.

```bash
cd backend && ../.venv/bin/uv run python ../scripts/export_sft.py \
    --root ../evals/results --out sft.jsonl
```

## Adding a case

Add `evals/suite/<name>.json` — `name`, `project` (defaults to
`eval:<name>`), `difficulty` (1-3), optional `settings`/`runtime`/
`images`, and an ordered `stages` list. The first stage must be
`generate` or `meshgen`; `from_stage`/`assets_from` may only reference
earlier stages. Pydantic validates the file at load — `cadx eval cases`
shows the parse errors.
