<p align="center">
  <img src="frontend/public/brand/hero.png" width="100%" alt="CADX">
</p>

<h3 align="center">Text &amp; photos in — verified, parametric, manufacturing-ready CAD out.</h3>

<p align="center">
  CADX is a local, open-weights multimodal agent: feed it reference photos, drawings
  or sketches plus a prompt with dimensions, and it iterates — generate, execute,
  measure, critique, repair — until the part checks out. The result is parametric
  CAD with real manufacturing exports, not just a mesh.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm--Noncommercial-blue.svg" alt="License: PolyForm Noncommercial"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white" alt="Docker Compose">
  <img src="https://img.shields.io/badge/LLM-any%20OpenAI--compatible-412991.svg" alt="LLM: any OpenAI-compatible">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#showcase">Showcase</a> ·
  <a href="#modes">Modes</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#license">License</a>
</p>

> [!WARNING]
> **Early development** — CADX is under active development. Expect breaking
> changes, rough edges and an evolving API/data model between versions.
> Feedback and bug reports are very welcome.
>
> **Hardware coverage:** only the Apple Silicon path (MLX/MPS) has been
> exercised on real hardware so far. The NVIDIA **CUDA** and AMD **ROCm**
> tiers are implemented and mock-tested, but not yet validated on an actual
> GPU — expect a report/fix round on first contact with real hardware.

---

## Highlights

<table>
  <tr>
    <td width="50%">
      <b>Parametric, not just a mesh</b><br>
      Output is diffable build123d <i>code</i> with named parameters — versioned,
      re-runnable and editable, so "make it 5 mm taller" is a one-line change.
    </td>
    <td width="50%">
      <b>Open-weights &amp; local-first</b><br>
      Image→3D via TRELLIS.2, Hunyuan3D-2.1 or TripoSG; local text-to-image
      (FLUX, Qwen, SDXL); and any OpenAI-compatible LLM — Ollama, LM Studio,
      vLLM or hosted. Runs entirely on your own hardware.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <b>Five generation modes</b><br>
      <code>auto</code> · <code>cad</code> · <code>mesh</code> · <code>hybrid</code> ·
      <code>topopt</code> — analytic B-Rep CAD, organic mesh shapes or
      FEM-verified topology optimization — plus deterministic organic restyling
      (gyroid / trabecular / voronoi / branches).
    </td>
    <td width="50%">
      <b>Verified by construction</b><br>
      Generated code runs in a sealed sandbox; measured geometry checks
      (through-holes, countersinks, inserts), multi-view renders and a vision
      critique drive an automatic repair loop — dimension autofixes and
      patch-based edits included.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <b>Structural analysis built in</b><br>
      Voxel FEM with explicit load cases, safety factors and stress/deformation
      reports; SIMP topology optimization with self-support filtering,
      multi-resolution solving and feature re-cutting (<code>refine</code>).
    </td>
    <td width="50%">
      <b>Manufacturing output</b><br>
      STEP, STL, 3MF, GLB, DXF and SVG exports; FDM/CNC DfM heuristics; and a
      BOM for real standard parts — fasteners, heat-set inserts, bearings —
      from the <code>bd_warehouse</code> catalog.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <b>Edit everything</b><br>
      Prompt and image edits, parameter-only reruns, a visual mesh editor
      (slice, shell, drill, boolean, smooth…), a paint-on structural editor and
      a reusable library for STEP/STL bodies and parametric templates.
    </td>
    <td width="50%">
      <b>A real web app</b><br>
      Guided new-part wizard, live progress with in-loop renders, jobs
      dashboard, statistics, asset image editing and local concept-image
      generation — served from a single compose stack.
    </td>
  </tr>
</table>

> DfM results are engineering heuristics, not a manufacturing guarantee — always
> verify dimensions and process limits yourself before cutting or printing.

## Showcase

<p align="center">
  <img src="docs/screenshots/viewer-orbit.gif" width="720" alt="CADX web GUI — 3D viewer orbiting a generated hybrid part">
</p>

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/version-cad.png" alt="Version detail — CAD result with agent activity"></td>
    <td width="50%"><img src="docs/screenshots/version-topopt.png" alt="Version detail — FEM-verified TopOpt result"></td>
  </tr>
  <tr>
    <td align="center"><sub>Version detail — CAD result, agent activity, measured geometry</sub></td>
    <td align="center"><sub>Topology optimization — FEM-verified with safety factor</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/screenshots/editor.png" alt="Visual mesh editor"></td>
    <td width="50%"><img src="docs/screenshots/jobs.png" alt="Jobs dashboard with live progress"></td>
  </tr>
  <tr>
    <td align="center"><sub>Visual mesh editor — live-previewed operations</sub></td>
    <td align="center"><sub>Jobs — live progress, scores and control across projects</sub></td>
  </tr>
</table>

Full GUI walkthrough: **[docs/webgui-ux.md](docs/webgui-ux.md)**.

## Quickstart

```bash
./scripts/start.sh    # or: make start
```

The first run auto-detects your GPU tier (MLX / CUDA / ROCm / none), installs the
matching mesh adapter and brings up the whole stack — docker services plus the
native mesh-server sidecar. Open **http://localhost:8080** → **Settings** and
point it at your LLM provider (Ollama, LM Studio, vLLM, OpenAI, OpenRouter —
anything OpenAI-compatible); everything else works out of the box.

The landing page is a guided wizard — describe the part, drop in reference
images (or generate a concept image locally), pick a mode, review — and live
progress renders right there until the model is done. Existing work lives
under **Projects**.

```bash
scripts/status.sh            # health of all components   (make status)
scripts/stop.sh              # stop everything            (make stop)
scripts/setup.sh --doctor    # diagnostics
```

Full install, dev-mode and migration docs:
**[docs/getting-started.md](docs/getting-started.md)**.

## Modes

| Mode | Pipeline | Output |
|------|----------|--------|
| `cad` | build123d code → analytic B-Rep | exact STEP + all exports |
| `mesh` | image→3D model → trimesh pipeline | STL/3MF/GLB (+ tessellated STEP when convertible) |
| `hybrid` | mesh → faceted B-Rep solid → build123d booleans | organic shape with exact machined features |
| `topopt` | base CAD → SIMP on a voxel domain → FEM verify | optimized mesh + density/history artifacts |
| `auto` | planner picks per spec | — |

A finished version can also be **restyled** into a deterministic organic
structure (gyroid, trabecular, voronoi, branches) — an aesthetic
transformation, not a structural optimization
([docs/organic.md](docs/organic.md)).

`mesh`/`hybrid` require an installed adapter and fall back to CAD cleanly when
the mesh server is unreachable — see
[docs/mesh-generation.md](docs/mesh-generation.md).

## How it works

```mermaid
flowchart LR
    browser(["Browser · :8080"]) --> api["api · FastAPI"]
    api --> db[("Postgres")]
    api --> redis[("Redis · queues + SSE")]
    api --> worker["worker · agent loop<br/>intake → plan → codegen → critique"]
    worker --> llm["LLM · any OpenAI-compatible endpoint"]
    worker --> mesh["mesh-server · open-weights models<br/>image→3D + text→image"]
    worker --> exec["cad-executor · sealed sandbox"]
    exec --> out["build123d · FEM · TopOpt · DfM<br/>STEP · STL · 3MF · GLB · DXF · SVG"]
```

Each iteration of the agent loop writes build123d code, executes it in a
**sealed subprocess** (AST import allowlist, resource limits, read-only
`no-new-privileges` container), measures the geometry — through-holes,
countersinks, wall thicknesses — renders it for a vision critique, then repairs
or regenerates until the spec checks out.

Details: **[docs/architecture.md](docs/architecture.md)** ·
**[docs/agent-loop.md](docs/agent-loop.md)**.

## Documentation

**[docs/README.md](docs/README.md)** — the full documentation index, grouped by
audience:

- **Run it** — [Getting started](docs/getting-started.md) ·
  [Configuration](docs/configuration.md) ·
  [Troubleshooting](docs/troubleshooting.md)
- **Use it** — [Web GUI](docs/webgui-ux.md) ·
  [Image editing](docs/image-editing.md) ·
  [Image generation](docs/image-generation.md) ·
  [Parts library](docs/parts-library.md) ·
  [E2E checklist](docs/e2e-checklist.md)
- **Understand it** — [Architecture](docs/architecture.md) ·
  [Agent loop](docs/agent-loop.md) ·
  [Mesh generation](docs/mesh-generation.md) ·
  [FEM & TopOpt](docs/fem-topopt.md) ·
  [Organic structures](docs/organic.md) · [DfM](docs/dfm.md) ·
  [Prompts](docs/prompts.md) · [HTTP API](docs/api.md) ·
  [mesh-server API](docs/mesh-server-api.md) · [Evaluation](docs/eval.md) ·
  [Roadmap](docs/roadmap.md)

<details>
<summary><b>Project layout</b></summary>

| Path | Contents |
|------|----------|
| `backend/` | FastAPI app, agent loop, CAD pipeline, CLI (`cadx`) |
| `frontend/` | React/Vite/Three.js web GUI |
| `mesh_server/` | image→3D + text→image sidecar (own uv project) |
| `scripts/` | `setup.sh` / `start.sh` / `stop.sh` / `status.sh` |
| `docs/` | documentation |
| `data/` | projects, library, models, caches (runtime data) |

</details>

<details>
<summary><b>Development</b></summary>

```bash
make sync && make test && make lint    # backend deps, tests, lint
cd mesh_server && uv run pytest        # sidecar suite
cd frontend && npx tsc --noEmit        # frontend typecheck
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and
[AGENTS.md](AGENTS.md) for the coding rules.

</details>

## Limitations

- Single-user, local operation (no auth) — don't expose it to a network.
- FEM/TopOpt is a simplified linear-elastic voxel analysis, not a certified
  structural solver; organic restyling is aesthetic only —
  [docs/fem-topopt.md](docs/fem-topopt.md) ·
  [docs/organic.md](docs/organic.md).
- Mesh/hybrid STEP exports are tessellated (faceted), not analytic.
- `load_reference` supports STEP and STL; DXF assets are documentation only.
- One CAD kernel: build123d.

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal, research,
educational and other noncommercial use; commercial use requires a separate
license (contact the author). Note that downloadable generation models carry
their own licenses (Hunyuan Community License excludes the EU/UK/South Korea);
TRELLIS.2, TripoSG and FLUX.1-schnell are MIT/Apache-2.0.
