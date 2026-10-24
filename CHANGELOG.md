# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - 2026-10-25

First public alpha — local, single-user CAD agent that turns images and text
into verified, parametric, manufacturing-ready 3D models.

### Added

- Agent-driven CAD generation loop: generate → execute → measure → critique →
  repair, producing parametric build123d code with STEP/STL/3MF/GLB exports.
- Image-to-3D mesh sidecar (`mesh_server`) with Hunyuan3D / TRELLIS.2 /
  TripoSG / TripoSR adapters and a deterministic `MESH_FAKE` mode.
- Voxel FEM verification with nominal/eroded/dilated export realizations
  (`sf_verified` headline metric) and a FEM-independent load-path certificate.
- SIMP/BESO topology optimization and lattice infill pipelines with
  export-structure verification.
- Mesh ops pipeline (`cad/mesh_ops.py`) with deterministic replay.
- React/Vite/Three.js web UI: projects, versions, chat agent, jobs, stats,
  settings (provider/models/agent/mesh/fem/system tabs), wizards, editors.
- MCP server surface (`cadx/mcpserver/`) — deterministic, LLM-free tools.
- EvalKit: durable live-API evaluation (`make eval`) + in-process scratch
  evaluation (`make eval-quick`), suites under `evals/suite/`.
- One-command local stack (`make start`) with GPU tier autodetect
  (mlx/cuda/rocm/none), update/watchdog/autostart helpers, `install.sh`.
- Full test suite: backend unit + cad tests, mesh-server tests, frontend
  vitest; `AGENTS.md`/`CONTRIBUTING.md`/`docs/` developer documentation.

### Known limitations

- Verified on macOS (Apple Silicon, MLX) only — Linux/Windows GPU tiers are
  untested; CPU tier works everywhere Docker does.
- Requires a local LLM backend (LM Studio / llama.cpp / vLLM) or an API
  provider key for generation; no bundled model weights.
- Alpha quality: interfaces, eval suites and settings keys may still change.
