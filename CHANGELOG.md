# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.2] - 2026-10-25

End-to-end validated on the live Docker stack: the `robot_arm_mount` eval
case (codegen → FEM → TopOpt → refine → export) passes all gates, including
the FEM verify on the refined mesh.

### Added

- Anchor `pad_dia` + boss detection: TopOpt auto-keeps now preserve boss
  discs and bore rings as round prisms instead of rectangular patches.
- Load-case anchor rebinding: out-of-range or semantically mismatched
  `hole_index` references are deterministically rebound to the correct
  anchors; anchor tables are injected into the load-extraction prompt.
- Refine verify FEM anchored on the *refined* mesh: bore/anchor regions
  bind by (axis, dia, center) hint without `hole_index`.
- `eval_mode` flag so eval runs never park on `waiting_for_input`.
- Version list UI: parts grouped, newest-first ordering.
- EvalKit: `fem_loadcases` gate with couple-magnitude weighting and
  `moment_mag_nm` band check; `depth_class` filter + axis-correct
  positions for hole gates; `refine_verify` gate asserting the refine
  job's own feature-cut verification.

### Fixed

- `_effective_dia` no longer subtracts the thread pitch twice from an
  already-declared pilot/tap-drill diameter (M12 bore restored to nominal).
- Rim-fit hole measurement interpolates the 50%-blocked ring crossing
  instead of the last fully-clear ring — under-measurement of bores next
  to occluding geometry (e.g. flange bores near the boss) fixed.
- Load-case region categories corrected: `g_load` binds to `all` (body
  force, not skin), broad fixtures with ≥2 mount holes rebind to the
  mount holes, moments rebind to threaded anchors; passive pinning now
  skips `g_load`/`all` regions consistently (incl. resample path).
- Refine channel plugs run before drills + through-channel sweep;
  signed entry axes for declared features (bottom-face taps).
- Queue-side refine timeout now counts exec-class feature finishing
  alongside FEM + STEP export stages.
- Mesh `_ensure_watertight` stitches hairline boundary cracks.
- Keep-restore no longer silently skips declared keep regions whose axis
  lacks an explicit sign (unsigned axes now resolve both directions).
- Refine seats feature entry on the mouth face instead of interior
  seals; exporters skip the helical groove for non-helical taps and use
  ring-aware entry geometry.

### Changed

- `intake_loads`/`intake_system`/`codegen_system` prompts: anchor tables,
  envelope-overlap rule, forbidden load/fixture anti-patterns.

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
