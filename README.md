# cadagent

**Images + text in — manufacturing-ready CAD out.** cadagent uses a multimodal
LLM (any OpenAI-compatible API) to turn reference photos, drawings and a text
description into parametric, verified 3D models you can 3D print or CNC mill.

- **Generate** parametric models from multiple images + a prompt with dimensions
- **Mesh & hybrid modes**: image→3D generation (local TRELLIS.2 / Hunyuan3D /
  TripoSG) — `cad | mesh | hybrid | auto` per job; hybrid = organic mesh +
  exact build123d features via boolean ops
- **Edit** existing models with new prompts and/or more images (version tree)
- **Verify** automatically: numeric dimension check, sandboxed execution,
  multi-view renders, vision-model critique, iterative repair
- **Manufacture**: exports as STEP / STL / 3MF / GLB / DXF / SVG plus heuristic
  DfM checks for FDM and CNC
- **Standard parts**: bd_warehouse fasteners (screws, nuts, washers, threads,
  heat-set inserts), bearings, gears and more via a curated catalog

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
                │          └─► mesh-server (HTTP :8765) for mesh/hybrid modes
                ▼  enqueue "execute_version"
             cad-executor (arq, queue "cad", hardened container)
                  sandboxed build123d run · analysis · renders · DfM · exports
```

The generated model code runs in a **sealed subprocess** (AST import allowlist,
no network/subprocess/fs builtins, rlimits on Linux) inside a read-only,
`no-new-privileges` container. Only `build123d`, `bd_warehouse`, `math`,
`numpy` and the injected `load_reference()` helper are available.

## Quickstart (Docker Compose)

```bash
cp .env.example .env       # set LLM_API_KEY + models
docker compose up --build  # first build takes a while (OCP/VTK are large)
# open http://localhost:8080
```

Services: `postgres`, `redis`, `api`, `worker`, `cad-executor`, `frontend`.
Models, renders and exports persist under `./data/`.

## Mesh generation (optional)

A local mesh server turns photos into 3D meshes — required for `mesh` and
`hybrid` modes. It runs **natively** (Docker Desktop has no GPU access) or on
a CUDA box:

```bash
make mesh-sync                       # uv venv + deps for mesh_server/
mesh_server/scripts/install_hunyuan_mlx.sh   # macOS arm64 (or install_*.sh for CUDA)
make mesh-server                     # http://localhost:8765  (health: /v1/health)
```

Set `MESH_BASE_URL` in `.env`; compose dev stacks get
`http://host.docker.internal:8765` automatically. `MESH_FAKE=1` gives a
deterministic GPU-free server for tests/demos. Details:
`docs/mesh-generation.md`, `mesh_server/README.md`. Licensing note: Hunyuan
models carry the Tencent Community License (EU/UK/KR excluded) — TRELLIS.2 /
TripoSG are MIT; licenses are shown in Settings.

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
../.venv/bin/uv run cadagent run "a 40x20x10 block with 1.5mm edge fillets" --fake   # no LLM needed
../.venv/bin/uv run cadagent exec ../path/to/model.py                               # pipeline only
../.venv/bin/uv run cadagent eval                                                   # benchmarks
```

## LLM provider configuration

Set in `.env` (or later in the UI at `/settings`, except the API key):

| Variable | Default | Notes |
|---|---|---|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `LLM_API_KEY` | — | required for real runs |
| `LLM_MODEL_VISION` | `gpt-4o` | needs image input for intake/critique |
| `LLM_MODEL_CODE` | `gpt-4o` | code generation |
| `LLM_JSON_MODE` | `json_object` | `json_object` / `json_schema` / `none` |

Provider examples:

| Provider | `LLM_BASE_URL` | Models |
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
   `AGENT_CLARIFICATION_MODE=ask`, blocking questions pause the job
   (`waiting_for_input`) until you answer in the UI.
2. **PLAN**: modeling strategy + param definitions + standard parts.
3. **ITERATE** (≤ `AGENT_MAX_ITERATIONS`): codegen → sandboxed execute →
   numeric bbox check vs stated dims (incl. swapped-axis detection) → vision
   critique on a labeled contact sheet. Failures go through repair passes
   (≤ `AGENT_MAX_REPARS`).
4. **FINALIZE**: best-ranked candidate (no dim errors > fewer major issues >
   higher score) is exported to all formats and versioned immutably.

Every LLM call is logged to the `agenttrace` table for debugging.

## Generated-code contract

Model files must end with a `result` (or `results`) shape and may declare
`PARAMS` (re-executable via the Params panel), `PARAMS_SCHEMA` and
`expected_bbox` (checked numerically). Available names: all of `build123d`,
`bd_warehouse`, `math`, `numpy`, `load_reference(asset_id)`.

## API sketch

```
POST /api/projects                       create project
POST /api/projects/{id}/assets           upload images/STEP/STL/DXF (multipart)
POST /api/projects/{id}/generate         {prompt, asset_ids, mode} -> {version_id, job_id}
POST /api/versions/{id}/edit             {prompt, asset_ids, mode, regenerate_mesh}
POST /api/versions/{id}/params           {params}  (new version, no LLM)
POST /api/versions/{id}/reexport
GET  /api/projects/{id}/versions         version tree
GET  /api/versions/{id}                  full detail incl. artifacts
GET  /api/jobs/{id}/events               SSE progress stream
POST /api/jobs/{id}/answer               clarification answers -> resume job
POST /api/jobs/{id}/cancel
GET  /api/artifacts/{id}/file            downloads (step/stl/3mf/glb/dxf/svg/png)
GET  /api/catalog                        bd_warehouse types+sizes
GET|PUT /api/settings                    LLM/provider config
GET  /api/mesh/health                    mesh server reachability + device
GET  /api/mesh/models                    installed adapter/model info + licenses
```

## Tests

```bash
cd backend
../.venv/bin/uv run pytest tests -x -q        # everything
../.venv/bin/uv run pytest tests/cad -q       # CAD executor only
../.venv/bin/uv run pytest tests -m "not slow"  # fast subset
../.venv/bin/uv run ruff check .              # lint
../.venv/bin/uv run mypy cadagent             # types
```

The suite covers the sandbox security model (blocked imports, builtin
tampering, fs/network/subprocess attempts), analysis, exports, DfM
heuristics, the full agent loop (FakeLLM + real pipeline), the worker
persistence path and the REST API on an isolated SQLite DB.

## Limitations

- Single-user, local operation (no auth) — don't expose it to a network.
- One CAD kernel: build123d (CadQuery is excluded; incompatible OCP ranges).
- `load_reference` supports STEP and STL; DXF assets are documentation only.
- DfM warnings are geometric heuristics, not process guarantees.
- Mesh/hybrid STEP exports are tessellated (faceted), not analytic.
- Hybrid mesh→solid conversion is limited to watertight meshes ≤
  `MESH_MAX_FACES_FOR_CAD` (default 60k); failure falls back to mesh-only.
- Mesh generation quality/hardware depends on the local model; wrappers for
  TRELLIS.2 (CUDA), Hunyuan3D-2.1 (MLX) and TripoSG ship in `mesh_server/` —
  see its README for manual validation notes.
