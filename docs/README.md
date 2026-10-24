<p align="center"><img src="../frontend/public/brand/hero.png" width="100%" alt="CADX"></p>

<h2 align="center">Documentation</h2>

<p align="center">
  <i>Text &amp; photos in — verified, parametric, manufacturing-ready CAD out.</i><br>
  CADX turns reference images and a text prompt into parametric, verified 3D
  models for 3D printing and CNC. This index groups the documentation by
  audience — start with <a href="getting-started.md">Getting started</a> if you
  are new.
</p>

## Getting started

| Doc | Contents |
|-----|----------|
| [Getting started](getting-started.md) | Prerequisites, one-command install, GPU tiers, docker vs. native dev mode, first run |
| [Configuration](configuration.md) | Authoritative reference for all runtime settings, project options and environment variables |
| [Updating](updating.md) | Self-update from GitHub: `make update` / the Settings button, channels, rollback, the updater daemon |
| [Troubleshooting](troubleshooting.md) | Diagnostics (`--doctor`, `status.sh`) and common failures |

## Using CADX

| Doc | Contents |
|-----|----------|
| [Web GUI](webgui-ux.md) | Pages and workflows: projects, versions, mesh editor, structural editor, settings |
| [Chat agent](chat.md) | Conversational agent: chat UI, tool set, SSE events, crash-safe turn lifecycle |
| [Image editing](image-editing.md) | Crop/mask/background-removal of reference images before generation |
| [Image generation](image-generation.md) | Local text-to-image adapters, concept/hero images, image edits, sketches |
| [Parts library](parts-library.md) | Standard-part catalog, `cadlib` helpers, user library, BOM |
| [MCP server](mcp-server.md) | LLM-free MCP surface for external agents: transports, auth, tool groups |
| [E2E checklist](e2e-checklist.md) | Manual browser smoke test of the whole stack |

## Generation modes

| Doc | Contents |
|-----|----------|
| [Mesh generation](mesh-generation.md) | `cad`/`mesh`/`hybrid`/`auto` modes, adapters, candidate screening, post-processing |
| [FEM & TopOpt](fem-topopt.md) | Voxel FEM, SIMP topology optimization, load-case contract, limitations |
| [FEM verification](fem-verification.md) | femverify suite: benchmark cases, convergence trends, MMS, TopOpt invariants, CalculiX lane |
| [Organic structures](organic.md) | Pseudo-topology organic restyling — gyroid/trabecular/voronoi/branches (aesthetic, not structural) |
| [DfM checks](dfm.md) | FDM/CNC manufacturability heuristics |

## Internals & reference

| Doc | Contents |
|-----|----------|
| [Architecture](architecture.md) | Services, request flow, code and data layout, security model |
| [Agent loop](agent-loop.md) | State machine, tool sessions, step reference, budgets |
| [Prompts](prompts.md) | Prompt files under `backend/cadx/agent/prompts/` and their conventions |
| [HTTP API](api.md) | Backend REST endpoints grouped by router |
| [mesh-server API](mesh-server-api.md) | Sidecar contract (mesh/image/llm adapters, chat proxy, model management, queue) |
| [Evaluation](eval.md) | evalkit: suite cases/stages/gates, http+local drivers, baselines, cleanup |
| [Roadmap](roadmap.md) | Shipped milestones and remaining backlog |
| [Local production](local-production.md) | Single-user production hardening: updater/watchdog, job recovery, autostart, backups |
| [Releasing](releasing.md) | Maintainer runbook: version contract, screenshots, release workflow, rulesets |
| [Release notes](release-notes/) | Curated per-release notes (`0.1.0-alpha.1`) |
| [TopOpt audit 2026-10](audit-topopt-2026-10.md) | Point-in-time defect-class audit of the FEM/TopOpt pipeline (historical record — findings since fixed) |

## Repo root

- [`CHANGELOG.md`](../CHANGELOG.md) — keep-a-changelog release history
- [`SECURITY.md`](../SECURITY.md) — security policy & vulnerability reporting
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — dev setup, conventions, test rules
- [`AGENTS.md`](../AGENTS.md) — working rules for AI agents in this repo
- [`install.sh`](../install.sh) — bootstrap/in-tree installer

## Component READMEs

- [`mesh_server/README.md`](../mesh_server/README.md) — the multi-model
  inference sidecar: adapters, install scripts, env configuration, CLI.
- [`AGENTS.md`](../AGENTS.md) — working rules for coding agents in this repo.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — development setup and conventions.
- [`capture_screenshots.py`](capture_screenshots.py) — regenerate all
  `screenshots/` PNGs/GIFs against a running stack (Playwright; `--list`
  prints candidate demo ids, `all` captures + assembles + installs).
