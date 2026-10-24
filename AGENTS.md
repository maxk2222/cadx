# AGENTS.md — working rules for this repo

## What this is
LLM-driven CAD agent ("CADX"): images + text prompt → parametric build123d code →
verified 3D model (STEP/STL/3MF/GLB/DXF/SVG) for 3D printing & CNC.
Stack: FastAPI + SQLModel/Postgres + arq/Redis + build123d/bd_warehouse + React/Vite/Three.js.
The authoritative design document is `docs/architecture.md` (derived from the master plan).
Documentation index: `docs/README.md`; human-facing dev guide: `CONTRIBUTING.md`.

## Commands (run from repo root)
- One-command start:     `make start` / `./scripts/start.sh` — auto-detects GPU
  tier (mlx|cuda|rocm|none), installs the mesh adapter, starts docker stack +
  native mesh-server. `make stop`, `make status`, `make autosetup`,
  `scripts/setup.sh --doctor`, `make help` lists every annotated target.
- Launcher scripts share `scripts/lib/ui.sh` (spinner/step output, `run_step`
  live-log, `wait_for_url`, port/disk/pidfile helpers, failure traps —
  bash 3.2, TTY+NO_COLOR aware) and `scripts/lib/pkg.sh` (venv/installer/
  clone/HF-download helpers used by `mesh_server/scripts/install_*.sh`).
  Per-run logs land in `logs/<name>-<ts>.log`. Extend the libs rather than
  duplicating output/detect code in a new script.
- Self-update: `make update` / `scripts/update.sh` (`--check`, `--ref`,
  `--rollback`, `--force`) — pull + **pg_dump backup** + deps + rebuild +
  restart + verify, driven on the host by the token-authed updater daemon
  `scripts/update_server.py` (:8766, survives the restart it drives); the api
  only proxies it via `/api/system/*` and the GUI card lives in Settings →
  System. Never put update/git logic inside the api container — it can't see
  the repo. See `docs/updating.md`.
- Local-production hardening (`docs/local-production.md`): the updater daemon
  doubles as the **watchdog** — stalled-update kill, mesh-server respawn via
  the `run/mesh-server.env` contract (never adopts foreign processes), compose
  healing (never resurrects an intentional `stop.sh` down / ExitCode 0), all
  suppressed mid-update. Job recovery: `Job.payload` envelopes + Redis AOF +
  a 60s sweeper re-enqueueing lost `queued` rows (deterministic arq job ids).
  Ops: `make backup` (pg_dump → data/backups), `make prune` (aged logs/tmp +
  docker dangling), `make autostart` (launchd/systemd supervisor install),
  `GET /api/system/stack`, `POST /api/system/stack/restart/<svc>`,
  `run/watchdog.json`. Frontend: ErrorBoundary + webglcontextlost recovery —
  every user-facing string goes through `t()` in all 9 locales.
- Setup env once:        `python3 -m venv .venv && .venv/bin/python -m pip install uv`
- Install backend deps:  `make sync`            (creates backend/.venv via uv)
- Infra for local dev:   `make infra`           (postgres+redis via docker-compose.dev.yml)
- Run api (local):       `make dev-api`
- Run agent worker:      `make dev-worker`
- Run cad executor:      `make dev-executor`
- Run frontend dev:      `cd frontend && npm install && npm run dev`
- Full stack (docker):   `make up` → http://localhost:8080
- Tests:                 `make test`  (or `cd backend && ../.venv/bin/uv run pytest tests -x -q`)
- Lint/format check:     `make lint`, autofix `make fmt`
- Evalkit (durable):     `make eval` — runs `evals/suite/*.json` through the live
  API; every stage spawns real versions/jobs under fixed `eval:*` projects,
  archives under `evals/results/` (gitignored). Options via `ARGS`, e.g.
  `make eval ARGS="--case l_bracket_rib --stages codegen"`.
- Evalkit (scratch):     `make eval-quick` — same suite in-process on a
  throwaway sqlite+data dir (`evals/scratch/`, ephemeral, excluded from
  reports/baselines); no API/Redis/Docker needed.
- Eval reports:          `make eval-cases` (suite overview), `make eval-report`
  (variant x case table, `ARGS="--include-scratch"` folds in scratch runs),
  `make eval-baseline ARGS="save <label>"` / `ARGS="diff a b"`,
  `make eval-clean` (guarded dry-run; `ARGS="--wipe-db --purge-files --yes"`
  executes — preserves runtime settings/secrets unless `--with-settings`).
- Mesh sidecar (dev):    `make mesh-sync && make mesh-server`  (own uv env in
  `mesh_server/`; `MESH_FAKE=1` for GPU-free runs; see `docs/mesh-generation.md`)
- Mesh-server tests:     `cd mesh_server && uv run pytest`

## Conventions (must follow)
- Python: type hints everywhere; `ruff` line-length 100; async FastAPI handlers;
  pydantic v2 for all JSON; no business logic in routers; `structlog` not `print`.
- TypeScript strict; functional components; TanStack Query for server state; Tailwind only.
- Never put LLM/HF secrets in logs/traces or return them via the API — stored
  runtime secrets are Fernet-encrypted at rest (`cadx/secrets_crypto.py`,
  `enc:v1:` in DB, key `CADX_SETTINGS_KEY` or `data/.settings.key`).
  Trace messages replace image data with placeholders.
- Every prompt lives in `backend/cadx/agent/prompts/*.md` — never inline prompt text in Python.
- All file paths go through `cadx/storage.py`. All DB writes in short transactions.
- LLM-generated code must ONLY ever run through `cad/sandbox.py` (AST allow-list + subprocess
  rlimits + import hook) inside the executor environment. Never `exec()` user code in-process.
  With `exec_warm_runner` (default on) the sandbox forks a pre-warmed `cad/zygote.py`
  child instead of a cold subprocess — identical isolation; falls back to cold on any
  zygote failure. Keep both paths' limits/guards in sync when touching either.
- CPU/memory budgets come from `cad/resources.py` (cgroup-aware: `cpus:` in
  compose is a CFS quota, NOT a cpuset — always intersect affinity with
  cpu.max). `exec_threads`/`exec_pool_workers`/`audit_parallel_runs`/
  `exec_mem_limit_mb` are `0` = auto; explicit values win. Fork pools share
  state via the module-global `_POOL` (COW) — `CaseSolve.factor` is not
  picklable. Parallel audits must run the cold path (zygote serializes).
  RLIMIT_CPU counts CPU-seconds across threads — scale it via
  `resources.rlimit_cpu_s`; the proc timeout stays the wall-clock deadline.
  Container caps auto-size via `scripts/lib/host_detect.sh` → `.env`
  (`CADX_EXECUTOR_CPUS/_MEM`, `CADX_WORKER_*`); user keys win.
- Allowed imports for generated code: build123d, bd_warehouse.*, cadlib, math, numpy,
  typing, dataclasses. `cadlib` (`cadx/cad/cadlib/`) is the reviewed helper library —
  extend it deliberately, never import unlisted modules from it (the runtime guard
  treats its internals like generated code).
- Data hierarchy: `Project → Part → Version` — every version belongs to one
  part; each part has at most one **accepted master version**
  (`Version.accepted_at` non-null, user-verified). Set/clear only via
  `parts_svc.set_version_accepted` (clears siblings atomically; REST PATCH
  `{accepted}`, chat/MCP `accept_version`, `PartOut.accepted_version_id`).
  Newer versions never inherit the flag; deleting a master returns
  `deleted_master:true` and leaves the part masterless — no auto-reassign.
- Standard-part contract: `spec.standard_parts` (StandardPartSpec) is the structured
  catalog intent — `standard_parts.build_standard_part_code` turns it into a
  sandbox recipe (PARAMS + `CADX_FEATURES` + `CADX_BOM` declarations), never a
  free-form guess. Threads are level-of-detail split by the executor-injected
  `CADX_DETAIL_MODE`: "preview" = fast simple body + helix-line render overlay,
  "export" = real helix (`simple=False`) — the orchestrator re-runs the winning
  candidate in export mode and fails if `thread_count` falls short.
- Agent robustness pipeline (local-model friendly): `cad/errordiag.py` classifies
  exec failures, `cad/patch.py` applies `PatchResponse` find/replace hunks
  (contract-validated), `cad/features.py` measures geometry (holes, countersinks,
  rounds, chamfers — plus quiddity patterns/bosses/fillets via
  `cad/recognition.py`) for feedback, `agent/autofix.py` retries dim errors via
  `params_override` without an LLM call, `agent/steps/improve.py` patches the
  best-so-far code instead of regenerating. Structured iteration issues persist
  on `Version.issues`; `cad/verifyspec.py` adds the canonical
  PASS/FAIL/UNVERIFIED conformance report on top.
- build123d-mcp ports (the sandbox boundary still rules): `cad/inspect_tools.py`
  injects `measure`/`find_holes`/`cross_sections`/`clearance`/`bbox_dims` into
  generated code via runner_entry — never import them from generated code or
  cadlib (the ImportGuard blocks it). Staged `kind="build"` codegen sessions
  construct parts feature-by-feature (`add_block`/`checkpoint`/`rollback` +
  sandbox `verify`) when `agent_staged_build` != off. `cad/design_audit.py`
  perturbs PARAMS ±ε for robustness (audit_params task + run_params
  pre-flight), `cad/editfeature.py` resizes plain through-holes on existing
  B-Reps behind a hard gate (`POST /versions/{id}/edit-feature`),
  `cad/defects.py` locates BRepCheck faults with coordinates and the STEP
  roundtrip gate catches "valid in memory, broken on export",
  `cad/compare.py` diffs/interference-checks solids.
- Context-window detection (`llm/capabilities.py` + `LLMClient._ensure_caps`): the
  effective window resolves manual `llm_context_tokens` (>0) → DB cache
  (`AppSetting` row `llm_caps`, keyed by model id) → live probe
  (`/models` vLLM `max_model_len` / OpenRouter `context_length` / sidecar
  `context_len` → LM Studio `/api/v1|v0/models` → llama.cpp `/props` →
  Ollama `/api/show` → static name table), plus opportunistic learning from
  provider error bodies (`parse_ctx_from_error`). Probing is lazy, once per
  role per client; results persist via `caps_cb`. `llm_max_tokens=0` means
  auto → `clamp(ctx//4, 2048, 128k)`, additionally clamped so
  prompt (`estimate_prompt_tokens`) + completion fits the window; explicit
  values are hard caps still clamped to fit. When the estimated prompt
  exceeds `agent_context_budget_frac` of the window, the tool loop condenses
  early and `chat_worker` collapses history into a `[EARLIER CONVERSATION
  SUMMARY]` digest instead of silently truncating. The mesh sidecar reports
  its effective ctx as `context_len` in `/v1/models` (`MESH_LLM_CTX` beats
  the adapter catalog max).
- Adaptive reasoning: `agent/thinking.py` resolves per-call `extra_body`
  (provider-dialect JSON in `llm_thinking_body_on/_off/_medium/_high`) —
  `llm_thinking=auto` enables thinking for `llm_thinking_hard_steps` (empty by
  default) and once the orchestrator latches `ctx.thinking_escalated`. The
  escalation is staged on `ctx.thinking_level` (1 → medium, 2 → high) and
  fires ONLY on stall (`llm_thinking_escalate_after` iterations without
  progress) or a measured dq regression ≥ 10 — never on a single patch miss.
  Qwen3.8 dialect maps on=low → medium → xhigh; send `reasoning_effort` both
  top-level and inside `chat_template_kwargs` (never bare `"high"` — the
  template enum is low/medium/xhigh). Thinking calls get
  `max_tokens=llm_thinking_max_tokens` when set — reasoning shares the
  completion budget.
  `agent/quickref.py` serves runtime-introspected signatures to the `quickref`
  tool.
- Loop progress is measured by `agent/quality.py` `det_quality` (verify_spec
  PASS/FAIL → 0..100) — the critique score saturates at 0 and cannot detect
  stalls. A stall gets ONE fresh-codegen escape before finalising; a
  mesh/hybrid→CAD fallback rewrites the plan via `cad_fallback_plan` so
  `load_mesh_solid` never reaches the prompt.
- `cadlib.strut(p0, p1, width, thickness)` builds a straight diagonal bar
  between two points — a "Strebe/brace" is never a filled triangular gusset
  (`rib` is). Bound hole snippets cut only through their declared host plate
  (`through_hole(depth=host_span+0.5)` on the hi face) so struts/ribs on the
  same bore line stay intact.
- Coordinate frame contract: X=length, Y=width (+Y front), Z=height; part rests on XY plane,
  centered at origin; mm units. Result must be assigned to `result`; flat parts may also
  assign `flat_profile`. The same contract applies to post-processed meshes
  (`cad/mesh_ops.py`) — raw generator output lives in normalized space and is
  oriented/scaled by the executor pipeline, never by the mesh server.
- Mesh generation: only the worker talks to `mesh_server` (`MeshPort` in
  `agent/mesh_client.py`); the executor receives GLBs as `DATA_DIR` files via
  the `process_mesh` task. `load_mesh_solid()` exists in generated code only
  for `resolved_mode="hybrid"`.
- Mesh refine: agent mesh transforms are recorded as `MeshOp` lists
  (`cad/mesh_ops.py` `OPS_REGISTRY` → `MeshPostParams.ops` → `version.params`)
  and replayed deterministically in the executor (`baseline_mesh` +
  `apply_ops`). The worker-side workbench (`agent/mesh_tools.py`) previews
  the same code path — never treat the workbench mesh as the authoritative
  output.
- Hybrid mesh booleans: `cad/mesh_boolean.py` is the trusted manifold3d
  backend (no `cadx` imports — it loads inside the sandbox). `runner_entry`
  injects it via `cadlib.ops.set_boolean_backend()` so `mesh_cut`/`mesh_fuse`/
  `mesh_intersect`/`trim_below` keep OCC first and only rescue on
  None/exception/wreckage. The runner also attaches the source trimesh to the
  imported solid (`_cadx_mesh` attr) — re-tessellating a sewn solid costs
  minutes, never drop that attachment. `mesh_to_solid` returns a
  `UnifySameDomain`-unified solid; anchor bores measured on the image are
  deterministically re-cut by the runner when generated code missed them.
- Structural verification hardening (`docs/fem-topopt.md`): the headline
  safety factor is `sf_verified` — the worst case across the nominal /
  eroded / dilated export-mask realizations solved by
  `engine._verify_export_structure` (shared by the topopt, infill and
  resample pipelines). `sf_export` is the nominal printed-mesh value and
  `sf_mask` the idealized voxel mask — gate on `sf_verified`, never on the
  mask number alone. `voxelize(coverage=True)` records per-cell fill
  fractions (`fem_verify_coverage_e`) which scale `elem_E` at the boundary;
  `fem_loadpath_cert` attaches the FEM-independent min-cut certificate
  (`load_path_ok`, `bottleneck_mm2`, `sf_floor` — `load_path_ok=false`
  invalidates the run); `fem_verify_mesh_shift` flags grid sensitivity.
  Infill runs may add the homogenized lattice model (`sf_homog`,
  `sf_micro`; `elem_class`/`class_D`/`micro_amp` into `run_fem_domain`,
  `model="homogenized"`). Every auxiliary layer degrades to warnings —
  only a nominal-solve failure falls back to the mask report. When adding
  SF-bearing fields, thread them through `toolloop._METRIC_KEYS` so
  compaction doesn't drop them, and chain fallbacks on `is None` (a solved
  SF of 0.0 is a verdict, not a missing value).
- Commits: one per milestone step, descriptive message.

## MCP server (`cadx/mcpserver/`)
- Exposes the deterministic tool surface to external agents over MCP —
  **strictly LLM-free**. `McpToolSession` runs with `ctx.llm = None`;
  `generate_part`/`edit_version`/`answer_questions`/`enqueue_resume` and every
  `delete_*` tool must never appear in `MCP_TOOL_SCHEMAS`/`MCP_HANDLERS`.
- New MCP-only tools go in `mcpserver/tools.py` (schemas use the agent
  `_fn()` format; handlers are async `(session, args) -> dict` and must
  delegate to `session`/spawn/executor helpers — never duplicate business
  logic). `MCP_HANDLERS` shadows same-named `_tool_*` methods on MCP
  sessions only (`_mcp_handlers_enabled`).
- Transports: `cadx mcp` (stdio) and `/mcp` on the API (Streamable HTTP,
  `CADX_MCP_TOKEN` bearer when configured). Spawn tools default `wait=false`
  on this surface. See `docs/mcp-server.md`.

## Testing
- `tests/cad` must run both natively and inside the executor image
  (`docker compose run cad-executor pytest tests/cad` — requires tests mounted; use local run).
- Agent loop tests use `FakeLLM` — never call a real LLM in tests.
- Renders: pyvista offscreen; tests must skip gracefully if rendering unavailable.
