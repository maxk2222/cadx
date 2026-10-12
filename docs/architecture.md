# Architecture

CADX is a local, single-user system that turns multimodal input
(images + text + dimensions) into manufacturing-ready CAD via an
OpenAI-compatible LLM. Generated geometry is build123d *code* — parametric,
diffable, re-runnable.

## Services (docker-compose.yml)

| Service      | Role                                                          |
|--------------|---------------------------------------------------------------|
| `frontend`   | nginx + React/Vite SPA on :8080, proxies `/api/*` and `/events/*` to `api` |
| `api`        | FastAPI app: projects/assets/versions/jobs, SSE, file serving  |
| `worker`     | arq worker on queue `agent`: runs the LLM agent loop + bounded tool sessions |
| `cad-executor` | arq worker on queue `cad`: sandboxed CAD execution, render, DfM, voxel FEM and SIMP TopOpt |
| `postgres`   | SQLModel/SQLAlchemy database (async)                          |
| `redis`      | arq queues + pub/sub for SSE events                            |
| `mesh-server` | (optional, external) image→3D sidecar: local models (TRELLIS.2, Hunyuan3D, TripoSG). Runs **natively** on the host (macOS has no docker GPU) or on a CUDA box (`--profile gpu`). Contract: `docs/mesh-server-api.md`. |

The split between `worker` and `cad-executor` is deliberate: the agent loop
is cheap I/O, while CAD execution is CPU-bound and sandboxed. They scale and
fail independently. The mesh server is deliberately *not* in the default
compose stack — only the worker talks to it (`mesh_base_url` runtime
setting), and generated
GLBs flow to the executor purely through `DATA_DIR` files.

## Request flow

```
Browser ──> nginx (:8080) ──> api:8000
                                 │ enqueue arq job
                                 ▼
                        worker (queue "agent")
                                 │ run_generation(ctx)
                                 │   ├─ LLM calls (httpx, OpenAI-compatible)
                                 │   ├─ mesh-server (HTTP, GLB → DATA_DIR)   [mesh/hybrid modes]
                                 │   └─ CAD/mesh work ──> QueueExecutor
                                 ▼                    │
                        cad-executor (queue "cad") <───┘
                                 │  sandbox exec → analysis → renders → DfM → exports
                                 │  process_mesh → clean/orient/scale → solid (hybrid)
                                 │  voxel FEM / SIMP TopOpt → FEM verification
                                 ▼
                        data/projects/<pid>/versions/<vid>/...
                        postgres (versions, jobs, artifacts, traces)
                                 │
                        SSE /events/jobs/<id> ──> browser progress
```

## Code layout (backend/cadx)

- `agent/` — orchestrator state machine, per-step modules (`steps/`),
  context/trace/types, `worker_jobs.py` (arq job runners)
- `cad/` — sandbox + allowlist, runner_entry, engine pipeline, analysis,
  exporters, render, contact_sheet, `dfm/` (common/fdm/cnc/report), `fem/`
  (voxel Hex8 analysis, SIMP TopOpt, mesh extraction), `catalog.py` (full
  bd_warehouse registry + lookup/search), `partspec.py` (AST validation of
  standard-part calls), `cadlib/` (reviewed helpers allowed in the sandbox),
  tasks (queue payloads)
- `llm/` — `client.py` (httpx, OpenAI-compatible), `structured.py`
  (parse/validate/re-ask), `images.py` (EXIF→resize→JPEG→data-URL),
  `schemas.py` (Pydantic contracts), `fake.py` (deterministic test LLM)
- `api/` — FastAPI routers, schemas, deps
- `db/` — SQLModel tables, async session, Alembic migrations
- `eval/` — benchmark harness (`cadx eval`)
- `queue.py` — arq `WorkerSettings` for both queues
- `storage.py`, `config.py`, `events.py`, `logging.py`, `cli.py`, `main.py`

## Data layout

```
data/projects/<pid>/
  assets/<aid>/<file>            uploaded inputs (images, STEP/STL/DXF refs)
  versions/<vid>/
    model.py                     winning generated code (cad/hybrid only)
    params.json                  winning PARAMS values (or mesh post-params)
    spec.json / plan.json        agent intake + plan (via DB too)
    mesh/                        mesh-mode intermediates (see docs/mesh-generation.md)
      mesh_raw.glb               generator output (normalized space)
      mesh_clean.stl             post-processed mesh (mm, oriented, scaled)
      solid.brep                 hybrid: faceted solid for load_mesh_solid()
      mesh.json                  provenance (model, seed, orientation, post params)
    renders/{front,...,iso}.png  best-iteration views
    contact_sheet.png  thumbnail.png
    dfm.json
    bom.json                      standard parts actually instantiated in model.py
                                (namespace scan in runner_entry; also Version.bom)
    fem/report.json             load-case results + stress/deformation renders
    topopt/{density.npz,history.json}  density field + SIMP convergence history
    exports/model.{step,stl,3mf,glb,dxf,svg}
    iterations/<n>/              per-iteration run dir (_run/, renders/, …)
    iterations/0/                hybrid: mesh generation + orientation + solid prep
data/library/<part_id>/<file>    reusable user library (uploaded STEP/STL bodies;
                                parametric code templates are stored in the DB row)
```

## Design decisions

- **Code, not meshes**: LLMs produce build123d source. Parametric `PARAMS`
  dict at top; deterministic rebuilds; cheap edits. Exception: `mesh`/`hybrid`
  modes — see `docs/mesh-generation.md`. Hybrid is *CAD-hosted*: the generated
  mesh becomes a faceted B-Rep solid (`load_mesh_solid()`) that build123d
  code modifies via booleans.
- **Bounded LLM sessions**: every step starts with fresh system+user context;
  when enabled, native tool calls run in a budgeted turn loop for targeted
  search/read/edit/check/verify. Provider support is probed per role and falls
  back independently for vision and code models (`llm_tools=auto`).
- **Structured output via JSON mode** with automatic fallback to prompt-only
  JSON (`none`) for providers without `response_format`.
- **Sandboxed execution**: AST allowlist + restricted import guard + resource
  limits (best-effort on macOS, enforced in Linux containers).
- **Verification loop**: numeric bbox check vs declared spec, 7-view render +
  contact sheet, DfM rules, vision critique. Best candidate across iterations
  wins, not the last one.
- **Parts library, three layers**: (1) the full bd_warehouse catalog is
  introspected once and served to planner/codegen as exact type/size tables
  (`catalog.py`, `partspec.py` validates calls statically pre-execution);
  (2) `cadlib` provides deterministic helpers for recurring geometry so the
  model calls tested macros instead of hand-rolling threads/holes;
  (3) a user library (`LibraryPart` rows + `data/library/`) holds reusable
  STEP/STL bodies and parametric code templates — selected by the user per
  version or by the agent via `search_library`/`attach_library_part` tools,
  then available in generated code through `load_reference(part_id)`.
- **Structural verification and optimization**: optional voxel Hex8 FEM feeds
  stress/safety-factor feedback into the agent; SIMP TopOpt runs as a post-phase
  or an LLM-free child job. Assumptions, API contract, artifacts and limits:
  `docs/fem-topopt.md`.
- **No auth**: single-user local. Do not expose the stack on a network.

## Security notes

- `cad-executor` runs generated code: `no-new-privileges`, read-only FS where
  possible, capped CPU/memory/time, allowlisted imports only.
- The LLM API key lives in env/`AppSetting` and is never logged or returned
  by the API (settings responses mask it).
- Uploaded files go through `storage.safe_filename`; file serving resolves
  paths under `DATA_DIR` only.
