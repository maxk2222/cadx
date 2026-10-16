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
  `scripts/setup.sh --doctor`
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
- Eval harness:          `make eval`
- Mesh sidecar (dev):    `make mesh-sync && make mesh-server`  (own uv env in
  `mesh_server/`; `MESH_FAKE=1` for GPU-free runs; see `docs/mesh-generation.md`)
- Mesh-server tests:     `cd mesh_server && uv run pytest`

## Conventions (must follow)
- Python: type hints everywhere; `ruff` line-length 100; async FastAPI handlers;
  pydantic v2 for all JSON; no business logic in routers; `structlog` not `print`.
- TypeScript strict; functional components; TanStack Query for server state; Tailwind only.
- Never put the LLM API key in DB/logs/traces. Trace messages replace image data with placeholders.
- Every prompt lives in `backend/cadx/agent/prompts/*.md` — never inline prompt text in Python.
- All file paths go through `cadx/storage.py`. All DB writes in short transactions.
- LLM-generated code must ONLY ever run through `cad/sandbox.py` (AST allow-list + subprocess
  rlimits + import hook) inside the executor environment. Never `exec()` user code in-process.
  With `exec_warm_runner` (default on) the sandbox forks a pre-warmed `cad/zygote.py`
  child instead of a cold subprocess — identical isolation; falls back to cold on any
  zygote failure. Keep both paths' limits/guards in sync when touching either.
- Allowed imports for generated code: build123d, bd_warehouse.*, cadlib, math, numpy,
  typing, dataclasses. `cadlib` (`cadx/cad/cadlib/`) is the reviewed helper library —
  extend it deliberately, never import unlisted modules from it (the runtime guard
  treats its internals like generated code).
- Agent robustness pipeline (local-model friendly): `cad/errordiag.py` classifies
  exec failures, `cad/patch.py` applies `PatchResponse` find/replace hunks
  (contract-validated), `cad/features.py` measures geometry (holes, countersinks,
  rounds, chamfers) for feedback, `agent/autofix.py` retries dim errors via
  `params_override` without an LLM call, `agent/steps/improve.py` patches the
  best-so-far code instead of regenerating. Structured iteration issues persist
  on `Version.issues`.
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
- Commits: one per milestone step, descriptive message.

## Testing
- `tests/cad` must run both natively and inside the executor image
  (`docker compose run cad-executor pytest tests/cad` — requires tests mounted; use local run).
- Agent loop tests use `FakeLLM` — never call a real LLM in tests.
- Renders: pyvista offscreen; tests must skip gracefully if rendering unavailable.
