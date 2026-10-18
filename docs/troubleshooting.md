# Troubleshooting

Diagnostics first, then per-area fixes. Feature-specific notes stay in the
feature docs — this page is the index of common failures.

## Diagnostics

```bash
scripts/status.sh              # health of all components        (make status)
scripts/setup.sh --doctor      # GPU tier, adapter availability, env overrides
docker compose logs -f --tail=200            # service logs      (make logs)
tail -f logs/*.log                            # native --dev mode logs
```

In the UI: **Settings** shows provider health, mesh-server reachability and
adapter licenses/errors; a version's **job log** shows every pipeline step and
its duration.

## LLM provider

- **`Test provider` fails** — check `llm_base_url` (must end without a trailing
  `/chat/completions`; the path is appended), API key, and that the model name
  exists on the endpoint. For Ollama/LM Studio/vLLM make sure the container
  can reach the host — use `http://host.docker.internal:<port>/v1` from the
  docker stack.
- **Structured-output errors** — the client retries without `response_format`
  on 400s; if the provider is very weak set `llm_json_mode=none` and
  `llm_tools=off` and expect more repair iterations.
- **Tool calls rejected** — with `llm_tools=auto` a provider that 400s on
  `tools` is recorded per role and falls back to legacy JSON; `llm_tools=on`
  surfaces the error instead.
- **Weak local model loops forever** — raise `agent_tool_max_calls`,
  `agent_tool_max_turns` and `agent_max_tokens_per_job` as described in
  [recommended settings for local ~27B models](agent-loop.md#recommended-settings-for-local-27b-models);
  the churn/stall guards will still bound runaway sessions.

## Mesh server

- **`mesh server not configured`** — set `mesh_base_url` on the Settings page.
  Compose seeds `http://host.docker.internal:8765` (native sidecar) on first
  boot; the `gpu` compose profile needs `http://mesh-server:8765`.
- **`no adapter available … on device 'cpu'`** — no GPU detected and the fake
  adapter is off; install an adapter (see
  [`mesh_server/README.md`](../mesh_server/README.md)) or run
  `scripts/start.sh --fake`.
- **Wrong up/front axis** — change the `up`/`front` post-params and re-run
  params (no LLM call); the orientation source is recorded in
  `mesh.orientation_source` (`llm` / `heuristic` / `parent`).
- **Wrong size** — scale comes from *stated* spec dims (uniform only); correct
  via the `scale` post-param. Aspect mismatches appear as `mesh.dim_issues`.
- **Hybrid boolean failures** — faceted solids stress OCCT; the pipeline
  retries with stronger decimation, then falls back to mesh-only
  (`needs_review`).
- **Adapter OOMs** — the adapter is unloaded and badlisted (`last_error` in
  `/v1/models` and in Settings); `auto` picks the next adapter. On CUDA see
  `MESH_CUDA_OFFLOAD`; on Apple Silicon the lever is 4-bit `mflux`
  quantization.
- **rembg download** — `MESH_BG_REMOVAL=auto` only uses an already-cached
  model; set `always` to allow the one-time download.
- **AMD GPU not detected (tier `none`)** — ROCm support is Linux-only. The
  probe needs the `amdgpu` driver: `/dev/kfd` plus a `/dev/dri/renderD*` node,
  or `amd-smi`/`rocm-smi`/`rocminfo` on PATH. In containers pass
  `--device=/dev/kfd --device=/dev/dri` (see the `gpu-rocm` compose profile)
  and make sure the user is in the `render`/`video` groups.
- **torch doesn't see the AMD GPU** — verify `python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"`.
  ROCm torch reuses the `cuda` namespace (`hip` must be set). A plain PyPI
  torch is CUDA/CPU-only — install the ROCm wheel via
  `mesh_server/scripts/install_rocm.sh` or `TORCH_ROCM_INDEX=<rocm index>`.
- **`HSA_OVERRIDE_GFX_VERSION`** — gfx archs not covered by the installed
  wheel (e.g. gfx1031 on an RX 6700) can masquerade as a supported sibling:
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` (RDNA2) / `11.0.0` (other RDNA3) /
  `12.0.1` (other RDNA4). `setup.sh --doctor` prints the detected gfx arch.
- **TRELLIS.2 never appears on ROCm** — intended: its extensions
  (flash-attn, CuMesh, spconv, o-voxel) are CUDA-only. Use `triposg` (or the
  Hunyuan3D-2 torch adapters via `setup.sh --extra hunyuan`).
- **ROCm OOM / fragmentation** — set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (already the default in `Dockerfile.rocm`); `MESH_CUDA_OFFLOAD` applies to
  ROCm too since diffusers offload rides on the cuda namespace.

## Jobs

- **Job stuck `running`** — a worker restart reaps orphaned `running` jobs; if
  a job is genuinely hung use `cancel` then `restart` (a restart replays the
  same inputs as a sibling job).
- **Job refused to start** — jobs refuse to start below ~2 GB free disk on the
  `DATA_DIR` volume.
- **`waiting_for_input`** — answer the clarification question in the UI (or
  `POST /api/jobs/{id}/answer`); the job then resumes.
- **Mesh generation cancels slowly** — model calls are not interruptible
  mid-flight; the job is marked cancelled and its result discarded when the
  call returns.

## Generation quality

- **Dimensionally wrong parts** — state explicit dims in the prompt; check
  `Version.issues` and the measured-features table for what the check found.
- **Holes look filled in the viewer** — verify the geometry itself, not the
  render: a nominally through hole closed by a thin membrane is detected as
  `blocked` in the measured hole groups.
- **Mesh/hybrid STEP looks faceted** — expected for the base body: exports
  from tessellated geometry are marked `tessellated`. TopOpt/refine children
  additionally export an analytic STEP whose re-cut bores/threads are real
  cylindrical/helical faces — the isolated `export_analytic_step_isolated`
  subprocess degrades per-feature to the faceted STEP on timeout, with an
  honest warning.

## See also

- [Getting started](getting-started.md)
- [Mesh generation](mesh-generation.md) — pipeline details
- [Configuration](configuration.md)
