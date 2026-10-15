# CADX

<p><img src="frontend/public/favicon.svg" width="56" alt="CADX logo"></p>

**Images + text in — manufacturing-ready CAD out.** CADX uses a multimodal
LLM (any OpenAI-compatible API) to turn reference photos, drawings and a text
description into parametric, verified 3D models you can 3D print or CNC mill.

- **Generate** parametric models from multiple images + a prompt with
  dimensions
- **Five modes** — `auto | cad | mesh | hybrid | topopt`: image→3D generation
  (TRELLIS.2 / Hunyuan3D / TripoSG), organic mesh + exact CAD features, or
  voxel-FEM-verified SIMP topology optimization
- **Edit and reuse** — versioned prompt/image edits, deterministic parameter
  reruns, a visual mesh editor, and reusable STEP/STL bodies and parametric
  templates in a user library
- **Verify** automatically — numeric dimension and feature checks, sandboxed
  execution, multi-view renders, vision critique, iterative repair and
  optional structural FEM feedback
- **Manufacture** — STEP / STL / 3MF / GLB / DXF / SVG exports, heuristic
  FDM/CNC DfM checks, plus a BOM for instantiated `bd_warehouse` parts

> ⚠️ DfM checks are heuristics, not a manufacturing guarantee. Always verify
> dimensions and process limits yourself before cutting or printing.

## Web GUI

![CADX web GUI — 3D viewer on a generated hybrid part](docs/screenshots/viewer-orbit.gif)

|  |  |
|---|---|
| ![Projects overview](docs/screenshots/projects.png) | ![Version detail — CAD result with agent activity](docs/screenshots/version-cad.png) |
| ![Visual mesh editor](docs/screenshots/editor.png) | ![Jobs overview with live progress](docs/screenshots/jobs.png) |

Walkthrough: [docs/webgui-ux.md](docs/webgui-ux.md).

## Quickstart

```bash
./scripts/start.sh    # or: make start
```

The first run auto-detects your GPU tier (MLX / CUDA / none), installs the
matching mesh adapter and brings up the whole stack — docker services plus the
native mesh-server sidecar. Open **http://localhost:8080** → **Settings** and
configure your LLM provider; everything else works out of the box.

Full install, dev-mode and migration docs: **[docs/getting-started.md](docs/getting-started.md)**.

```bash
scripts/status.sh            # health of all components   (make status)
scripts/stop.sh              # stop everything            (make stop)
scripts/setup.sh --doctor    # diagnostics
```

## How it works

```
 browser ──► frontend (React/three.js, nginx :8080)
                │ /api (REST + SSE)
                ▼
             api (FastAPI :8000) ──► Postgres (projects, versions, jobs, traces)
                │                    Redis (queues + SSE pub/sub)
                ▼  enqueue
             worker (arq, queue "agent") ──► LLM via /chat/completions
                │  agent loop: intake → plan → codegen → execute → critique
                │          └─► mesh-server (HTTP :8765) for mesh/hybrid modes
                ▼  enqueue CAD/FEM/TopOpt tasks
             cad-executor (arq, queue "cad", hardened container)
                  sandboxed build123d · FEM · SIMP TopOpt · renders · DfM · exports
```

Generated model code runs in a **sealed subprocess** (AST import allowlist,
resource limits) inside a read-only, `no-new-privileges` container. Details:
[docs/architecture.md](docs/architecture.md) and
[docs/agent-loop.md](docs/agent-loop.md).

## Modes

| Mode | What runs | Output |
|------|-----------|--------|
| `cad` | build123d code → B-Rep | analytic STEP + all exports |
| `mesh` | image→3D model → trimesh pipeline | STL/3MF/GLB (+ tessellated STEP when convertible) |
| `hybrid` | mesh → faceted B-Rep solid → build123d booleans | organic shape + exact features (STEP tessellated) |
| `topopt` | base CAD → SIMP on a voxel domain → FEM verify | optimized mesh + density/history artifacts |
| `auto` | planner picks per spec | — |

`mesh`/`hybrid` require an installed adapter and fall back to CAD cleanly when
the mesh server is unreachable — see
[docs/mesh-generation.md](docs/mesh-generation.md).

## Documentation

**[docs/](docs/README.md)** — the full documentation index:

- **Run it**: [Getting started](docs/getting-started.md) ·
  [Configuration](docs/configuration.md) · [Troubleshooting](docs/troubleshooting.md)
- **Use it**: [Web GUI](docs/webgui-ux.md) · [Image editing](docs/image-editing.md) ·
  [Image generation](docs/image-generation.md) · [Parts library](docs/parts-library.md) ·
  [E2E checklist](docs/e2e-checklist.md)
- **Understand it**: [Architecture](docs/architecture.md) ·
  [Agent loop](docs/agent-loop.md) · [Mesh generation](docs/mesh-generation.md) ·
  [FEM & TopOpt](docs/fem-topopt.md) · [DfM](docs/dfm.md) ·
  [Prompts](docs/prompts.md) · [HTTP API](docs/api.md) ·
  [mesh-server API](docs/mesh-server-api.md) · [Evaluation](docs/eval.md) ·
  [Roadmap](docs/roadmap.md)

## Project layout

| Path | Contents |
|------|----------|
| `backend/` | FastAPI app, agent loop, CAD pipeline, CLI (`cadx`) |
| `frontend/` | React/Vite/Three.js web GUI |
| `mesh_server/` | image→3D + text-to-image sidecar (own uv project) |
| `scripts/` | `setup.sh` / `start.sh` / `stop.sh` / `status.sh` |
| `docs/` | documentation |
| `data/` | projects, library, models, caches (runtime data) |

## Development

```bash
make sync && make test && make lint    # backend deps, tests, lint
cd mesh_server && uv run pytest        # sidecar suite
cd frontend && npx tsc --noEmit        # frontend typecheck
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and
[AGENTS.md](AGENTS.md) for the coding rules.

## Limitations

- Single-user, local operation (no auth) — don't expose it to a network.
- FEM/TopOpt uses simplified linear-elastic voxel analysis; it is not a
  certified structural solver — see [docs/fem-topopt.md](docs/fem-topopt.md).
- Mesh/hybrid STEP exports are tessellated (faceted), not analytic.
- `load_reference` supports STEP and STL; DXF assets are documentation only.
- One CAD kernel: build123d.

## License

[MIT](LICENSE) — note that downloadable generation models carry their own
licenses (Hunyuan Community License excludes the EU/UK/South Korea);
TRELLIS.2, TripoSG and FLUX.1-schnell are MIT/Apache-2.0.
