# Configuration

All application configuration lives in the app database and is edited on the
**Settings** page (`/settings`) or via `GET|PUT /api/settings`. Changes apply
to *new* jobs — a running job keeps the runtime snapshot it was started with.
The settings status cards and the sidebar health dots poll
`GET /api/settings/health` (a real code+vision probe of the configured
provider), `/api/mesh/health` and `/api/images/models` — the sidecar serves
all three adapter families (mesh, image and llm). The **Models** tab manages
sidecar models directly (downloads, weight deletion, HF-repo registration,
queue/resident view — proxied under `/api/mesh/*`) and holds the Hugging Face
access token; the provider tab's model
dropdowns are fed by `GET /api/settings/llm-models`.

Run options are merged in this order, later layers winning:

```
code defaults < saved runtime settings < project settings < per-version options
```

The tables below list the code defaults (`RuntimeSettings` in
`backend/cadx/runtime_config.py`). Only non-default values are persisted in
the database. Secret values (`llm_api_key`, `tripo_api_key`, `image_api_key`,
`hf_token`) are stored Fernet-encrypted (`enc:v1:` prefix); the key comes
from the `CADX_SETTINGS_KEY` env var or an auto-generated
`{DATA_DIR}/.settings.key` file (mode 0600).

## LLM provider

| Setting | Default | Notes |
|---------|---------|-------|
| `llm_backend` | `external` | `external` = `llm_base_url` provider; `sidecar` = the OpenAI-compatible endpoint on the local mesh-server (`mesh_base_url`, no auth) |
| `llm_sidecar_model` | `""` | single local model for both roles when `llm_backend=sidecar`; empty = `llm_model_*` names apply there too (unknown ids fall back to `auto`) |
| `llm_base_url` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `llm_api_key` | — | write-only in the UI; never logged or returned by the API |
| `llm_model_vision` | `gpt-4o` | needs image input for intake/critique |
| `llm_model_code` | `gpt-4o` | code generation |
| `llm_json_mode` | `json_object` | `json_object` / `json_schema` / `none`; client auto-retries without `response_format` on 400s |
| `llm_temperature_code` | `0.2` | sampling temperature for code steps (0–2) |
| `llm_context_tokens` | `0` | context window override: `0` = auto-detect from the provider (`/models` → vLLM `max_model_len` / OpenRouter `context_length` / sidecar `context_len`, then native LM Studio `/api/v1/models`, llama.cpp `/props`, Ollama `/api/show`, static name table, provider error bodies); `>0` = manual override for both roles. Detections cache in the `llm_caps` AppSetting row and surface as `llm_detected` in GET /settings |
| `llm_max_tokens` | `0` | completion cap per call: `0` = auto → `clamp(ctx//4, 2048, 128000)` additionally clamped so prompt+completion fits the detected window; `>0` = hard cap (still clamped to fit); falls back to 8000 when no context is known |
| `agent_context_budget_frac` | `0.8` | fraction of the detected context window a prompt may occupy before tool-loop history condenses / chat history summarises early — bounds the resend before the provider rejects with a context error |
| `llm_timeout_s` | `180` | per-request timeout |
| `llm_stream` | `true` | stream completions |
| `llm_idle_timeout_s` | `180` | abort a stalled stream with no content |
| `llm_total_timeout_s` | `1800` | hard cap for one streamed response |
| `llm_degenerate_stall_s` | `60` | abort a stream emitting only whitespace/indentation (invisible to the byte-progress watchdog) |
| `llm_extra_body` | `""` | extra JSON object merged into every request body (provider-specific flags) |
| `llm_tools` | `auto` | `auto` probes native tool support per role; `on` is strict; `off` uses legacy JSON |
| `llm_thinking` | `auto` | `off` / `auto` / `always` — adaptive reasoning (`agent/thinking.py`); `auto` enables thinking for hard steps and escalates only on stall/regression |
| `llm_thinking_body_on` / `llm_thinking_body_off` | `""` | provider-dialect extra_body JSON merged when thinking resolves on/off; `""` = model default |
| `llm_thinking_body_medium` / `llm_thinking_body_high` | `""` | per-escalation-stage overrides — level 1 merges `medium`, level 2 `high`, each falling back to `body_on` when empty |
| `llm_thinking_hard_steps` | `""` | CSV step names that always get thinking under `auto`; empty by default — escalation is stall/regression-driven |
| `llm_thinking_escalate_after` | `2` | iterations without progress before `auto` raises `ctx.thinking_level` (medium, then high) for the rest of the job; a dq regression ≥10 also escalates |
| `llm_thinking_max_calls` | `8` | per-job cap on thinking-on calls (`0` disables thinking) |

## Agent loop

| Setting | Default | Notes |
|---------|---------|-------|
| `agent_max_iterations` | `3` | outer generate → critique loop iterations |
| `agent_max_repairs` | `3` | repair attempts per iteration |
| `agent_pass_score` | `80` | critique score needed for `succeeded` |
| `agent_dim_tolerance_pct` / `agent_dim_tolerance_mm` | `2` / `0.5` | bbox tolerance vs stated dims |
| `agent_clarification_mode` | `assume` | `ask` pauses the job on blocking questions (`waiting_for_input`) |
| `agent_max_input_images` | `8` | cap on images sent to intake |
| `agent_max_tokens_per_job` | `400000` | prompt-token budget per job |
| `agent_max_wall_s` | `900` | wall-clock budget per job (s) |
| `agent_default_gen_mode` | `auto` | `auto` / `cad` / `mesh` / `hybrid` / `topopt` |
| `agent_plan_mode` | `full` | `full` = catalog-aware tool session, `lite` = one short non-tool call, `auto` = lite for text-only cad jobs |
| `agent_param_autofix` | `true` | retry dim errors via `params_override` without an LLM call |
| `agent_patch_mode` | `auto` | `auto` / `always` / `never` — patch vs full regeneration |
| `agent_patch_retries` | `2` | patch attempts before falling back to fresh codegen |
| `agent_feature_feedback` | `true` | feed measured features (holes, rounds, …) into improve steps |
| `agent_critique_reask` | `true` | re-ask once when a failing critique lists no discrepancies |
| `agent_sigcheck` | `true` | static AST signature gate on generated code (pre-execution) |
| `agent_staged_build` | `auto` | feature-by-feature codegen sessions (`add_block`/`checkpoint`/`verify`); `auto` = on when tool sessions are |
| `agent_parametrize` | `true` | best-effort mesh → build123d parametric reconstruction (one extra executor task per mesh job) |
| `agent_codegen_samples` | `1` | best-of-N sampled first codegen attempt (light-verified, best wins) |
| `agent_quickref_context` | `false` | inline the runtime-introspected signature reference into codegen/repair context |
| `agent_design_audit` | `false` | PARAMS ±ε robustness audit before finishing |
| `agent_max_approaches` | `2` | strategy-chain budget (e.g. `hybrid → cad` fallback) |
| `agent_stall_patience` | `2` | stop after this many iterations without improvement (0 = never) |
| `agent_switch_score` / `agent_switch_after_iters` | `60` / `4` | strategy-switch trigger thresholds |

## Tool sessions

| Setting | Default | Notes |
|---------|---------|-------|
| `agent_tool_max_calls` | `10` | tool turns per edit/codegen session |
| `agent_tool_max_calls_info` | `6` | lookup turns per intake/plan/critique session |
| `agent_tool_max_calls_improve` | `6` | tool budget for feature/lint-only improve sessions |
| `agent_tool_max_turns` | `24` | hard turn cap per tool session |
| `agent_tool_max_turns_edit` | `12` | turn cap for edit/repair sessions (each verify pays the full pipeline) |
| `agent_tool_churn_limit` | `6` | nudge → abort after this many edits without a check |
| `agent_session_condense_turns` | `12` | condense tool-session history past this many exchanges (`0` = off) |
| `agent_tool_verify` | `true` | allow bounded sandbox `verify`/`render_model` in edit sessions |
| `agent_tool_verify_max` | `2` | sandbox executions per session |
| `agent_build_verify_max` | `8` | sandbox `verify` calls per staged build |
| `agent_tool_max_calls_build` | `16` | tool-call budget for staged build sessions |
| `agent_mesh_tools` | `true` | master switch for the mesh-refine session |
| `agent_mesh_max_ops` | `24` | `MeshOp`s per refine session |
| `agent_mesh_preview_max` | `4` | `mesh_preview` renders per session |
| `agent_mesh_refine_rounds` | `1` | issues-seeded repair rounds after a failed critique |
| `agent_mesh_shape_guard_iou` | `0.8` | refuse refine ops below this silhouette IoU (or >15% volume loss on a non-cut op); `0` = off |
| `agent_mesh_hard_fail_score` / `agent_mesh_hard_fail_limit` | `30` / `3` | consecutive hard fails before the mesh loop stops |
| `agent_struct_tools` | `true` | master switch for the optimize session |
| `agent_topopt_max_runs` | `6` | `topopt_run` calls per optimize session |
| `agent_fem_max_runs` | `4` | `analyze_fem` calls per optimize session |

## Chat

Chat-agent session settings — see [chat.md](chat.md).

| Setting | Default | Notes |
|---------|---------|-------|
| `chat_model_role` | `vision` | `vision` receives tool-result images (renders, concept images); `code` is text-only |
| `chat_tool_max_calls` | `30` | tool calls per chat turn |
| `chat_max_turns` | `40` | turn cap for a running chat turn |
| `chat_spawn_timeout_s` | `2700` | wait cap on spawn tools (`wait=true`) |
| `chat_history_messages` | `60` | message history sent to the model |
| `chat_image_max` | `8` | images per user message |

## Mesh generation

| Setting | Default | Notes |
|---------|---------|-------|
| `mesh_backend` | `sidecar` | `sidecar` = local mesh-server; `tripo` = Tripo3D cloud (no sidecar needed) |
| `tripo_api_key` | — | write-only; bearer token for `mesh_backend=tripo` (`tripo_api_key_set` in GET) |
| `tripo_api_base` | `https://api.tripo3d.ai/v2/openapi` | Tripo3D OpenAPI base |
| `tripo_model` | `v2.5-20250123` | Tripo3D `model_version` for `image_to_model` tasks |
| `mesh_base_url` | `""` | mesh-server URL; seeded once from `MESH_BASE_URL` env |
| `mesh_model` | `auto` | adapter id or `auto`; seeded once from `MESH_MODEL` |
| `mesh_timeout_s` | `900` | client-side generation timeout |
| `mesh_stall_timeout_s` | `300` | abort polling a server job whose (progress,message) hasn't moved |
| `mesh_min_free_mb` | `0` | refuse to submit when the sidecar host reports less free RAM (`0` = off) |
| `mesh_decimate_faces` | `30000` | decimation target in the baseline pipeline |
| `mesh_max_faces_for_cad` | `30000` | face cap for the hybrid work-solid — keep ≤60k, OCCT booleans scale with facet count |
| `mesh_smooth_after_flatten_iters` | `10` | smoothing after `flatten_bottom` |
| `mesh_candidates_per_iter` | `2` | best-of-N screening candidates per iteration |
| `mesh_candidate_steps` | `15` | diffusion steps for screening candidates |
| `mesh_candidate_adapters` | `""` | CSV adapter ids for screening; `""` spreads over all available |
| `mesh_heightfield_flat` | `true` | deterministic heightfield candidate for flat parts |
| `mesh_prep_images` | `true` | preprocess inputs (bg removal, padding) |
| `mesh_cache_enabled` | `true` | cache generations by image+params key |

## Image generation

| Setting | Default | Notes |
|---------|---------|-------|
| `image_backend` | `sidecar` | `sidecar` = local mesh-server; `openai` = OpenAI-compatible images API |
| `image_openai_model` | `gpt-image-1` | model for `image_backend=openai` (`gpt-image-1` / `dall-e-3` / `dall-e-2` or a custom deployment id) |
| `image_api_base` | `""` | images API base; empty = `llm_base_url` |
| `image_api_key` | — | write-only; empty = `llm_api_key` (`image_api_key_set` in GET) |
| `hf_token` | — | write-only; Hugging Face token forwarded to the sidecar as `X-HF-Token` (applied as `HF_TOKEN` for downloads/model loads — gated repos, higher rate limits); seeded once from `HF_TOKEN` (`hf_token_set` in GET) |
| `image_model` | `auto` | text-to-image adapter; seeded once from `MESH_IMAGE_MODEL` |
| `image_edit_model` | `auto` | adapter for reference-image edits; `auto` = edit ladder |
| `image_default_strength` | `0.6` | img2img denoise when the request has no `strength` |
| `image_steps` / `image_guidance` | adapter default | diffusion steps / cfg scale |
| `image_max_side` | `1024` | width/height clamp — OOM prevention |
| `image_max_count` | `4` | per-request clamp |
| `image_autogen_mesh` | `true` | synthesise hero images for mesh/hybrid runs without inputs |
| `image_tool_budget` | `4` | `generate_concept_image` calls per tool session |
| `image_timeout_s` | `300` | client-side generation timeout |

## FEM & topology optimization

| Setting | Default | Notes |
|---------|---------|-------|
| `fem_voxel_mm` | `1.5` | voxel pitch (adaptively coarsened to stay under `fem_max_elements`) |
| `fem_max_elements` | `150000` | element cap |
| `fem_spsolve_dofs` | `120000` | above this, SSOR-CG replaces the direct solver |
| `fem_max_effective_cases` | `8` | cap on direction-cone load-case expansion |
| `fem_timeout_s` | `600` | executor task timeout |
| `topopt_timeout_s` | `2700` | executor task timeout |
| `topopt_max_iterations` | `80` | SIMP iteration cap |
| `topopt_default_volfrac` | `0.4` | target volume fraction |
| `topopt_default_rmin_mm` | `3` | sensitivity filter radius |
| `topopt_smooth_iterations` | `6` | surface smoothing on the extracted mesh |
| `topopt_min_member_mm` | `0` | post-thicken members below this size (0 = off) |
| `topopt_auto_keep` | `true` | hold detected fastener anchors solid |
| `topopt_auto_keep_margin_mm` | `3` | margin around auto-kept anchors |
| `topopt_multires_min_cells` | `4000` | below this many cells the `multires` coarse pass is skipped |
| `topopt_multires` | `true` | coarse→fine multires SIMP warm start (~3× faster on big parts) |
| `topopt_auto_refine` | `true` | auto-enqueue a feature-finishing refine child after a succeeded topopt with features/anchors |
| `refine_helical_threads` | `true` | re-cut derivable-pitch threads as real helical V-grooves (tap-drill bore kept as fallback) |

## Executor

| Setting | Default | Notes |
|---------|---------|-------|
| `exec_timeout_s` | `180` | sandbox run timeout |
| `exec_mem_limit_mb` | `0` (auto) | sandbox memory cap (enforced via rlimits on Linux); FEM/TopOpt
fail fast with a memory estimate when the voxelized domain would exceed
this budget instead of being OOM-killed mid-iteration. `0` = 60 % of the
detected (cgroup-aware) memory, clamped to 2048–32768 MB |
| `exec_threads` | `0` (auto) | threads one CPU-bound stage may use — OCC meshing, BLAS/SciPy,
llvmpipe raster. Auto = min(detected cpus, 8), exported as
`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`LP_NUM_THREADS`
on the executor and inside every sandbox child |
| `exec_pool_workers` | `0` (auto) | fork-pool workers for parallel FEM load cases + the render/DfM/export
pipeline fan-out. Auto = min(4, cpus − 1); `1` disables pools entirely |
| `audit_parallel_runs` | `0` (auto) | parallel ±ε sandbox runs in `audit_params` — forces the cold
subprocess path (the zygote serializes all runs anyway). Auto = min(4, cpus − 1) |
| `analytic_step_timeout_s` | `240` | wall-clock budget for the isolated per-feature analytic-STEP subprocess; on timeout the faceted STEP ships |
| `exec_warm_runner` | `true` | pre-warmed zygote fork per sandbox run — identical isolation, cold-subprocess fallback on failure |

Detection lives in `cadx/cad/resources.py` and is cgroup-aware: `cpus:` in
docker-compose is a CFS quota (`cpu.max`), not a cpuset — affinity alone
would report the whole host. `detect_cpus()` = min(affinity, quota);
`detect_mem_mb()` = cgroup memory limit else physical RAM. The executor
publishes its detected values to the settings API (`resources_detected` in
GET /settings, shown on the System tab); `RLIMIT_CPU` scales with the
thread budget because it counts CPU-seconds across all threads.

`start.sh` additionally auto-sizes the compose container caps via
`scripts/lib/host_detect.sh` (docker VM budget on Desktop, host resources
on native dockerd) into `.env`: `CADX_EXECUTOR_CPUS`/`CADX_EXECUTOR_MEM`,
`CADX_WORKER_CPUS`/`CADX_WORKER_MEM`, `CADX_EXECUTOR_PIDS` — existing keys
win, so pinning a value in `.env` is a permanent override. The cad queue's
`max_jobs` scales as `max(2, detected_cpus // 4)`; pin `CADX_CAD_MAX_JOBS`
to override.

## Project settings and per-run options

`ProjectSettings` (project edit dialog) and the run-options panel override the
runtime defaults per project/version: `units`, `process` (`fdm` / `sla` /
`cnc3` / `cnc2d`), `material`, `clarification_mode`, `max_iterations`,
`max_repairs`, `pass_score`, `dim_tolerance_pct`, `dim_tolerance_mm`,
`build_volume`, `min_wall_mm`, `nozzle_mm`, `tool_radius_mm`,
`thread_compensation_mm`, `include_fasteners`, `gen_mode`, `mesh_model`,
`mesh_steps`, `mesh_guidance`, `mesh_resolution`, `mesh_seed`, plus the FEM /
TopOpt toggles `fem`, `fem_note`, `fem_material`, `fem_voxel_mm`, `topopt`,
`topopt_volfrac`, `topopt_iterations`.

Provider backends and models can also be overridden per project and per run
(RunOptions / OptionsPanel): `llm_backend` (`external`/`sidecar`),
`mesh_backend` (`sidecar`/`tripo`), `image_backend` (`sidecar`/`openai`),
`mesh_model`, `image_model`, `image_edit_model`. Merge order is unchanged:
version options win over project settings, which win over runtime defaults.

## Environment variables

### Backend (`backend/cadx/config.py`)

Infrastructure-only bootstrap settings; everything else lives in the DB.

| Variable | Default | Notes |
|----------|---------|-------|
| `DATA_DIR` | `./data` | projects, library, models, caches |
| `DATABASE_URL` | `postgresql://cadx:cadx@localhost:5432/cadx` | set by compose |
| `REDIS_URL` | `redis://localhost:6379/0` | set by compose |
| `LOG_LEVEL` | `INFO` | structlog level |
| `AGENT_JOB_TIMEOUT_S` | `14400` | arq outer cap for one agent job |
| `MESH_BASE_URL` | — | seeds `mesh_base_url` **once** on first boot |
| `MESH_MODEL` | — | seeds `mesh_model` once |
| `MESH_IMAGE_MODEL` | — | seeds `image_model` once |
| `HF_TOKEN` | — | seeds `hf_token` once |
| `CADX_SETTINGS_KEY` | — | Fernet key encrypting stored secrets; default: auto-generated `{DATA_DIR}/.settings.key` (0600) |
| `U2NET_HOME` | `~/.u2net` (`/data/rembg` on compose) | rembg model cache |
| `CADX_MCP_TOKEN` | — | optional bearer token gating the `/mcp` HTTP endpoint |

### mesh-server (`mesh_server/meshserver/config.py`)

| Variable | Default | Notes |
|----------|---------|-------|
| `MESH_HOST` / `MESH_PORT` | `127.0.0.1` / `8765` | bind address |
| `MESH_DEVICE` | `auto` | `auto` / `cuda` / `rocm` / `mps` / `mlx` / `cpu` |
| `MESH_MODEL_DIR` | `~/.cache/meshserver/models` | weight cache |
| `MESH_JOB_DIR` | `/tmp/meshserver-jobs` | job scratch space |
| `MESH_JOB_TTL_S` | `3600` | result retention |
| `MESH_MAX_QUEUE` | `8` | queued jobs cap (one GPU job at a time) |
| `MESH_UNLOAD_IDLE_S` | `0` | unload weights after idle seconds (0 = never) |
| `MESH_FAKE` | `0` | enable the deterministic fake adapters |
| `MESH_ALLOW_CPU` | `0` | allow real adapters on CPU (slow) |
| `MESH_DEFAULT_MODEL` | `auto` | mesh adapter for `model=auto` requests |
| `MESH_IMAGE_MODEL` | `auto` | image adapter for `model=auto` requests |
| `MESH_BG_REMOVAL` | `auto` | `auto` / `always` / `never` (rembg/u2net on inputs) |
| `MESH_TRIPOSG_MPS` | `0` | experimental MPS path for TripoSG |
| `TORCH_ROCM_INDEX` | `…/whl/rocm6.4` | wheel index used by `install_rocm.sh` |
| `MESH_FLUX_KONTEXT` | `0` | enable the gated FLUX.1-Kontext edit adapter |
| `MESH_FLUX_MLX_QUANT` | `4` | mflux weight quantization bits (4/8/16) |
| `MESH_CUDA_OFFLOAD` | `auto` | `auto` / `model` / `sequential` / `none` diffusers CPU offload |
| `MESH_CUDA_OFFLOAD_VRAM_GB` | `24` | free-VRAM threshold for `auto` offload |
| `MESH_MLX_MEM_LIMIT_GB` | `0` | unified-memory cap (0 = unlimited) |
| `MESH_TRELLIS_MIN_VRAM_GB` | `20` | VRAM threshold for the setup TRELLIS.2 pick |
| `MESH_JOB_STALL_S` | `600` | fail a running job whose (progress,message) hasn't moved |
| `MESH_JOB_CANCEL_IDLE_S` | `60` | idle threshold after which a client abort is recorded as a stall |
| `MESH_JOBS_SUBPROCESS` | `1` | run each generation in `python -m meshserver.gen_worker` (killable, RSS-capped) |
| `MESH_JOB_MAX_RSS_MB` | `24576` | hard RSS cap for the generation subprocess (0 = off) |
| `MESH_JOB_MIN_FREE_MB` | `0` | refuse new jobs below this much free host RAM (0 = off) |
| `MESH_MEM_HEADROOM_MB` | `4096` | host RAM kept free when admitting a model (memory ledger) |
| `MESH_WARM_WORKERS` | `1` | keep `gen_worker --serve` children resident between same-model jobs |
| `MESH_WARM_IDLE_S` | `300` | warm worker self-exit after this much stdin silence |
| `MESH_WARM_MAX` | `2` | max concurrent warm workers |
| `MESH_QUEUE_FAIR_S` | `120` | jobs older than this beat model-affinity ordering |
| `MESH_QUEUE_MEM_WAIT_S` | `300` | queued job waiting on memory admission fails after this (0 = never) |
| `MESH_LLM_PIN` | `0` | pin loaded llm servers — never evicted under memory pressure |
| `MESH_LLM_LOAD_TIMEOUT_S` | `300` | readiness bound for a managed llm server subprocess |
| `MESH_LLM_CTX` | `16384` | context size passed to llama.cpp servers — the effective served window; reported to CADX as `context_len` in `/v1/models` (beats the adapter catalog max) |
| `MESH_LLM_PYTHON` | — | interpreter for mlx llm servers (else `mesh_server/.venv-llm/bin/python`, else shared venv) |
| `LLAMA_SERVER_BIN` | — | explicit `llama-server` binary (else PATH, else `python -m llama_cpp.server`) |

## See also

- [Getting started](getting-started.md) — install and first run
- [Agent loop](agent-loop.md) — how budgets and guards interact
- [Troubleshooting](troubleshooting.md)
