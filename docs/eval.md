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
| `robot_arm_mount` | 3 | topopt-ready robot-arm mount | `codegen` → `fem` → `topopt` (+ `refine`, disabled) |

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
