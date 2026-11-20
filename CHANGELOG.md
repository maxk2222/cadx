# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Mesh sidecar llama.cpp concurrency suite (CUDA/RTX-class hosts):
  `llm_kv_unified` (`--kv-unified` — `-c` becomes one KV pool shared
  dynamically by all parallel slots instead of a static ctx/N split,
  vLLM-style), `llm_kv_per_slot`, `llm_kv_cache_type` (`--cache-type-k/v`,
  e.g. `q8_0`), `llm_flash_attn`, `llm_gpu_layers` (`-ngl`, default -1 =
  full offload) and `llm_prompt_cache_mb` → `--cache-ram` +
  `--cache-idle-slots`. All flags are feature-probed against
  `llama-server --help` (cached) so older binaries never receive unknown
  args; `mesh_server/scripts/install_llm_cuda.sh` installs a current
  CUDA `llama-server` build (sm_120 needs CUDA ≥ 12.8, matching the
  updated `Dockerfile.cuda` toolchain). The `llama_cpp.server` python
  fallback gains `--n_gpu_layers`/`--cache_type_*`/`--flash_attn` — its
  previous `n_gpu_layers=0` default would have run CUDA models on CPU.
  Adapters carry `kv_bytes_per_token` for hybrid-arch KV-budget logs,
  and `server_env()` lets backends pass env knobs (mlx `APC_ENABLED`
  via `llm_apc`, vlm-only `--kv-quant-scheme` via `llm_kv_scheme`).
- Media-part capability gating: `video_url`/`input_video`/`video` parts
  join image parts on the vision path, `input_audio`/`audio_url`/`audio`
  parts require `supports_audio` (registerable via
  `POST /v1/models/register` `audio: true` — e.g. Qwen-Omni); audio
  requests no longer silently match vision-only models.
- Start wizard goal step: the new-part wizard now opens with a
  **Part / Anleitung** choice. The Part branch is unchanged (describe →
  images → mode → review); the Guide branch walks through manual PDF +
  CAD-part uploads, optional agent notes and a review step exposing all
  guide run options, then hands off to the shared job-progress view.
- Guide run options end-to-end: `POST /projects/{id}/guide` (and the
  wizard review step) accept `language` (BCP-47 output pin — locks
  `detect_language` so PDF autodetect cannot override it) and `audience`
  (`beginner` | `intermediate` | `pro` — an audience text-variant rewrite
  is applied before the artifacts are emitted, shared with the post-hoc
  REST rewrite via `kiara/guide/rewrite.py`).
- AssetManager accepts `.pdf`/`.glb`/`.gltf` uploads with per-kind size
  caps matching the server (PDF 100 MB, CAD 50 MB, other 25 MB).
- KIARA-style 3D guide editor overhaul (`GuideEditor`):
  - Full annotation model end-to-end — `GuideAnnotation` supports
    `kind: "arrow" | "text"` with `position`/`label`/`color`/`size`, backed
    by new REST endpoints (`POST/PATCH/DELETE
    /versions/{id}/guide/annotations[/…]`) with step/part validation and
    renumbering on step insert/delete. Arrows render with `Html` label
    placards in preview *and* editor via a shared `guideAnnotations`
    renderer; free text fields are selectable/movable in-scene.
  - Collapsible right rail with a scrollable **Steps** editor (title /
    text / display text / safety hint for every step, annotation counts,
    per-step delete) and an **Objects** tab (instance list, numeric
    transforms, quick rotations, align-height/flatten, annotation list).
  - Step media workflow: pick from extracted guide media, copy a project
    image asset in, upload a file, or capture a **viewport snapshot** —
    via new `POST /versions/{id}/guide/media` (multipart upload or
    `{asset_id}` copy, path-safe unique names).
  - Step duplicate, isolate-step ghosting, staggered slide-in animation
    preview along each step's arrow axis, Delete-key annotation removal,
    red error vs green info feedback. Server step ops flush the pending
    doc diff first so debounced saves can't hit shifted step numbers.
- Guides as first-class citizens across the frontend:
  - Persistent guide preview card on the project page — cumulative 3D
    assembly (`GuideScene`, latest step), live-refreshing while a guide job
    builds, multi-guide picker, and an editor button opening `GuideEditor`
    as a modal in place.
  - Guide entry points everywhere: Parts-section "Guide" button and an
    *or create a guide* link deep-linking to `/?goal=guide&project=<id>`;
    the wizard adopts the existing project (no draft project, no
    delete-on-reset, no silent rename).
  - VersionPage guide focus: defaults to the guide tab, purple `G{n}`
    header chip, part-only actions/editors/compare/master hidden;
    `EditPage` shows an informational state for guide versions; the
    project parts list stays visible in the inspector.
  - Guide hygiene: guide companion parts filtered out of *into part*
    pickers; PDF assets visible but unselectable as part inputs.
  - Mid-run guide notes REST endpoint `POST /versions/{id}/guide/notes`
    (same `user_notes.jsonl` format as the chat tool) plus a note composer
    on the guide tab while the job runs.
  - Neutral wording: nav button, chat hints and a "build the assembly
    guide from my PDF" suggestion; guide icons in version tree and jobs
    list.

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
- MCP server surface (`kiara/mcpserver/`) — deterministic, LLM-free tools.
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
