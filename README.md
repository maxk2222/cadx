# CADX

<p><img src="frontend/public/favicon.svg" width="56" alt="CADX logo"></p>

**Images + text in — manufacturing-ready CAD out.** CADX uses a multimodal
LLM (any OpenAI-compatible API) to turn reference photos, drawings and a text
description into parametric, verified 3D models you can 3D print or CNC mill.

- **Generate** parametric models from multiple images + a prompt with dimensions
- **Mesh, hybrid & TopOpt modes**: image→3D generation (TRELLIS.2 / Hunyuan3D /
  TripoSG), organic mesh + exact CAD features, or base CAD followed by
  voxel-FEM-verified SIMP topology optimization (`auto | cad | mesh | hybrid | topopt`)
- **Edit and reuse**: versioned prompt/image edits, deterministic parameter
  reruns, reusable STEP/STL bodies and parametric templates in a user library
- **Verify** automatically: numeric dimension and feature checks, sandboxed
  execution, multi-view renders, vision critique, iterative repair and optional
  structural FEM feedback
- **Manufacture**: STEP / STL / 3MF / GLB / DXF / SVG exports, heuristic FDM/CNC
  DfM checks, plus a BOM for instantiated `bd_warehouse` parts
- **Standard parts & helpers**: introspected `bd_warehouse` catalog and reviewed
  `cadlib` helpers, available through lookup tools in the agent

> ⚠️ DfM checks are heuristics, not a manufacturing guarantee. Always verify
> dimensions and process limits yourself before cutting or printing.

## Architecture

```
 browser ──► frontend (React/three.js, nginx :8080)
                │ /api (proxy)
                ▼
             api (FastAPI :8000) ──► Postgres (projects, versions, jobs, traces)
                │                    Redis (queues + SSE pub/sub)
                ▼  enqueue
             worker (arq, queue "agent") ──► LLM via /chat/completions
                │  agent loop: intake → plan → codegen → execute → critique
                │          tool sessions: search/read/edit/check/verify/repair
                │          └─► mesh-server (HTTP :8765) for mesh/hybrid modes
                ▼  enqueue CAD/FEM/TopOpt tasks
             cad-executor (arq, queue "cad", hardened container)
                  sandboxed build123d · FEM · SIMP TopOpt · renders · DfM · exports
```

Generated model code runs in a **sealed subprocess** (AST import allowlist,
no network/subprocess/filesystem builtins, rlimits on Linux) inside a read-only,
`no-new-privileges` container. Available modules are `build123d`,
`bd_warehouse`, reviewed `cadlib`, `math`, `numpy` and the injected
`load_reference()` helper.

## Quickstart (one command)

```bash
./scripts/start.sh    # or: make start
```

The first run auto-detects your GPU tier, installs the matching mesh adapter
and brings up the whole stack (docker services + native mesh-server):

| Tier | Detection | Mesh model installed |
|---|---|---|
| `mlx` | Apple Silicon (arm64 mac) | Hunyuan3D-2.1 via MLX port |
| `cuda` | local NVIDIA GPU | TRELLIS.2 (≥20 GB VRAM) else TripoSG |
| `none` | no GPU | none — CAD fallback (or `--fake` for tests) |

Open http://localhost:8080 → **Settings** and set the LLM provider URL,
models and API key there. All app configuration lives in the app database.
A `.env` file is only read by docker compose for infrastructure overrides
(`POSTGRES_*`, `MESH_BASE_URL` seed) — never by the app itself.

```bash
scripts/status.sh    # health of all components (make status)
scripts/stop.sh      # stop everything           (make stop)
scripts/setup.sh     # re-run the installer      (make autosetup)
scripts/setup.sh --doctor   # diagnostics: tier, adapters, .env overrides
```

Native dev mode instead of docker: `scripts/start.sh --dev` (api/worker/
executor/vite as background processes, logs under `logs/`).

Services: `postgres`, `redis`, `api`, `worker`, `cad-executor`, `frontend`,
plus the native `mesh-server`. Models, renders and exports persist under
`./data/`.

### Upgrading from `cadagent`

The system was renamed to **CADX** (package `cadx`, CLI `cadx`, compose project
`cadx`). Existing installs keep their data by running the one-shot migration —
it clones the old `cadagent_*` docker volumes and renames the Postgres
role/database inside:

```bash
scripts/migrate-cadx.sh
```

Or keep the old credentials without migrating: `POSTGRES_USER=cadagent
POSTGRES_PASSWORD=cadagent POSTGRES_DB=cadagent` in `.env`.

## Mesh generation (optional, auto-installed)

A local mesh server turns photos into 3D meshes — required for `mesh` and
`hybrid` modes. `scripts/setup.sh` picks the adapter automatically; to install
manually instead:

```bash
make mesh-sync                       # uv venv + deps for mesh_server/
mesh_server/scripts/install_hunyuan_mlx.sh   # macOS arm64 (or install_*.sh for CUDA)
make mesh-server                     # http://localhost:8765  (health: /v1/health)
```

`MESH_FAKE=1` gives a deterministic GPU-free server for tests/demos
(`start.sh --fake`). Optional extras: `setup.sh --extra mps` adds the
torch/MPS Hunyuan3D-2 mini/multi-view adapters on Apple Silicon.
Details: `docs/mesh-generation.md`, `mesh_server/README.md`. Licensing note:
Hunyuan models carry the Tencent Community License (EU/UK/KR excluded) —
TRELLIS.2 / TripoSG are MIT; licenses are shown in Settings.

### Local image generation (optional, auto-installed)

The same sidecar hosts a text-to-image adapter family (FLUX.1-schnell via
`mflux` on Apple Silicon, FLUX/SDXL via diffusers on CUDA) — used for concept
images in the asset manager, the agent's `generate_concept_image` tool, and
automatic hero images for mesh mode without reference photos. Setup installs
it alongside the mesh adapter (`--skip-image` opts out). Details:
`docs/image-generation.md`.

## Local development

```bash
make sync            # uv venv + locked deps (backend)
cd frontend && npm install

make infra           # postgres + redis in docker
make migrate         # alembic upgrade head
make dev-api         # FastAPI on :8000 (reload)
make dev-worker      # agent queue worker
make dev-executor    # cad queue worker
make dev-web         # Vite dev server on :5173 (proxies /api)
```

Without a full stack you can still exercise everything in-process:

```bash
cd backend
../.venv/bin/uv run cadx run "a 40x20x10 block with 1.5mm edge fillets" --fake   # no LLM needed
../.venv/bin/uv run cadx exec ../path/to/model.py                               # pipeline only
../.venv/bin/uv run cadx eval                                                   # benchmarks
```

## LLM provider configuration

All runtime configuration lives in the app database and is edited in the
**Settings** page (`/settings`): LLM provider + API key (write-only), agent
behaviour and budgets, tool sessions, mesh server, FEM/TopOpt and executor
limits. Changes apply to new jobs; a running job keeps its startup snapshot.

| Setting key | Default | Notes |
|---|---|---|
| `llm_base_url` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `llm_api_key` | — | write-only in the UI, required for real runs |
| `llm_model_vision` | `gpt-4o` | needs image input for intake/critique |
| `llm_model_code` | `gpt-4o` | code generation |
| `llm_json_mode` | `json_object` | `json_object` / `json_schema` / `none` |
| `llm_tools` | `auto` | `auto` probes native tool support per role; `on` is strict; `off` uses legacy JSON |
| `agent_tool_max_calls` | `10` | tool turns per edit/codegen session |
| `agent_tool_max_calls_info` | `6` | lookup turns per intake/plan/critique session |
| `agent_tool_verify` | `true` | allow bounded sandbox verification in edit sessions |

The Settings page also carries a **Test provider** button that probes the
saved code + vision models without exposing the key. Infrastructure
(database/redis DSNs, ports, container limits) is fixed by compose defaults;
override it via a `docker-compose.override.yml` when needed.

Provider examples:

| Provider | Base URL | Models |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o`, `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | e.g. `anthropic/claude-sonnet-4` |
| Ollama | `http://localhost:11434/v1` | e.g. `qwen2.5vl:32b`, `qwen2.5-coder:32b` |
| LM Studio | `http://localhost:1234/v1` | any loaded vision model |

Local models need to be strong at structured output and code generation —
weak models degrade gracefully into `needs_review` rather than silently
producing bad parts, but expect more repair iterations.

## How the agent works

1. **INTAKE** (vision): extracts a `PartSpec` — dims (stated vs inferred vs
   assumed), features, image roles, open questions. With
   `agent_clarification_mode=ask`, blocking questions pause the job
   (`waiting_for_input`) until you answer in the UI.
2. **PLAN**: modeling strategy + param definitions + standard parts.
3. **ITERATE** (≤ `agent_max_iterations`): codegen → sandboxed execute →
   numeric bbox check vs stated dims (incl. swapped-axis detection) → vision
   critique on a labeled contact sheet. Failures go through repair passes
   (≤ `agent_max_repairs`).
4. **REPAIR / IMPROVE**: when available, native OpenAI tool-calling sessions
   let the model search and read targeted code ranges, apply minimal line edits,
   check the contract, and optionally execute in the sandbox. Weak providers
   fall back to the legacy JSON path; `llm_tools=on` is strict.
5. **FEM / TopOpt (optional)**: describe supports and loads or enable FEM in
   run options; TopOpt runs SIMP on a voxelized design domain and FEM-checks the
   result. This is engineering feedback, not certification.
6. **FINALIZE**: the best-ranked candidate is exported and versioned; issues,
   measured geometry, BOM, FEM and optimization artifacts remain attached.

Every LLM call and tool session is logged to the `agenttrace` table for debugging.

## Generated-code contract

Model files must define a top-level literal `PARAMS` dict, create a build123d
shape and assign `result = p.part`; code runs in millimetres under the sandbox
allow-list. The Params panel re-executes saved code without another LLM call.
User-library file parts are available through `load_reference(asset_id)`;
parametric templates are guidance to adapt, not automatically executed modules.

## Feature guides

- Image editing before generation (crop/mask/background removal):
  `docs/image-editing.md`
- Agent tool sessions, provider fallbacks and per-step tools: `docs/agent-loop.md`
- User parts library and standard-part catalog: `docs/parts-library.md`
- FEM and topology optimization: `docs/fem-topopt.md`
- Mesh adapters and hardware setup: `docs/mesh-generation.md`
- Local text-to-image (concept/hero images): `docs/image-generation.md`
- Full manual smoke checklist: `docs/e2e-checklist.md`

## API sketch

```
POST /api/projects                       create project
POST /api/projects/{id}/assets           upload images/STEP/STL/DXF (multipart)
POST /api/assets/{id}/derivative         save an edited image (PNG, keeps alpha) as a new asset
POST /api/assets/{id}/remove-bg          rembg subject cutout -> PNG bytes (editor applies it)
POST /api/projects/{id}/images           local text-to-image -> generated_image assets
GET  /api/images/models                  installed image adapters
POST /api/projects/{id}/generate         {prompt, asset_ids, library_part_ids, mode, options}
POST /api/versions/{id}/edit             {prompt, asset_ids, library_part_ids, mode, options}
POST /api/versions/{id}/params           {params}  (new version, no LLM)
POST /api/versions/{id}/optimize         manual FEM/SIMP TopOpt child version
POST /api/versions/{id}/reexport
GET  /api/projects/{id}/versions         version tree
GET  /api/versions/{id}                  full detail incl. BOM/FEM/artifacts
GET  /api/jobs/{id}/events               SSE progress stream
POST /api/jobs/{id}/answer               clarification answers -> resume job
POST /api/jobs/{id}/cancel
GET  /api/artifacts/{id}/file            downloads (step/stl/3mf/glb/dxf/svg/png)
GET  /api/catalog                        bd_warehouse types+sizes
GET|POST /api/library                    list/upload reusable STEP/STL parts
PATCH|DELETE /api/library/{id}           edit metadata / remove a library part
POST /api/library/from-version/{id}      save STEP body or code template
GET  /api/library/{id}/file              download a library body
GET|PUT /api/settings                    LLM/provider config
GET  /api/mesh/health                    mesh server reachability + device
GET  /api/mesh/models                    installed adapter/model info + licenses
```

## Tests

```bash
cd backend
../.venv/bin/uv run pytest tests -x -q         # complete backend suite
../.venv/bin/uv run pytest tests/cad -q        # CAD executor only
../.venv/bin/uv run ruff check .               # lint
../.venv/bin/uv run ruff format --check .      # formatting
../.venv/bin/uv run mypy cadx/             # types
```

The suite covers the sandbox security model (blocked imports, builtin
tampering, fs/network/subprocess attempts), analysis, exports, DfM
heuristics, the full agent loop (FakeLLM + real pipeline), the worker
persistence path and the REST API on an isolated SQLite DB.

## Limitations

- Single-user, local operation (no auth) — don't expose it to a network.
- FEM/TopOpt uses simplified linear-elastic voxel analysis; it is not a certified
  structural solver. See `docs/fem-topopt.md` for assumptions and limitations.
- One CAD kernel: build123d (CadQuery is excluded; incompatible OCP ranges).
- `load_reference` supports STEP and STL; DXF assets are documentation only.
- DfM warnings are geometric heuristics, not process guarantees.
- Mesh/hybrid STEP exports are tessellated (faceted), not analytic.
- Hybrid mesh→solid conversion is limited to watertight meshes ≤
  `mesh_max_faces_for_cad` (default 60k); failure falls back to mesh-only.
- Mesh generation quality/hardware depends on the local model; wrappers for
  TRELLIS.2 (CUDA), Hunyuan3D-2.1 (MLX) and TripoSG ship in `mesh_server/` —
  see its README for manual validation notes.
