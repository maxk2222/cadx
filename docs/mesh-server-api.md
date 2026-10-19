# mesh-server HTTP API

Contract between `cadx` (worker → `agent/mesh_client.py` /
`agent/image_client.py`, LLM via `llm/client.py`) and the `mesh_server/`
sidecar. Both sides keep their own pydantic models — there is
**no code sharing** between the two projects; this document is the contract.

The sidecar is a multi-model inference host with three adapter families:
`mesh` (image→3D), `image` (text-to-image / image edit) and `llm`
(OpenAI-compatible chat completions).

Base URL: `MESH_BASE_URL`, e.g. `http://localhost:8765` (native) or
`http://mesh-server:8765` (compose `gpu` profile).

The backend forwards the Hugging Face token stored in WebUI Settings →
Models → Hugging Face as an `X-HF-Token` request header on every call; a
middleware applies it process-wide as `HF_TOKEN`, so managed weight
downloads and adapter loads against gated repos authenticate transparently.
`HF_TOKEN` in the sidecar's own environment works the same for non-backend
callers.

## Endpoints

### `GET /v1/health`

```json
{
  "ok": true,
  "device": "mlx",                 // cuda | mps | mlx | cpu | fake
  "backend": "local",
  "models": [MeshModelInfo],
  "mem_free_mb": 41000             // host RAM available to new jobs
}
```

### `GET /v1/models`

```json
{ "models": [MeshModelInfo] }
```

`MeshModelInfo`:

```json
{
  "id": "hunyuan3d-2.1-mlx",
  "name": "Hunyuan3D-2.1 Shape (MLX)",
  "device": "mlx",
  "backend": "mlx",
  "kind": "mesh",                   // "mesh" | "image" | "llm" — adapter family
  "supports_multiview": false,
  "max_images": 1,
  "supports_edit": false,           // image: accepts an init image
  "requires_image": false,          // image: edit-only — never wins a txt2img auto pick
  "max_count": 4,                   // image: max result images per job
  "license_note": "Tencent Hunyuan Community License — …",
  "loaded": false,
  "available": true,                // deps + weights usable on this host
  "last_error": null,               // set after a load/OOM failure ("badlisted")
  // model-management fields (llm family + downloadable adapters)
  "downloaded": true,               // weights present in the local cache
  "downloading": false,             // a download job is in flight
  "size_gb": null,                  // on-disk weights size once known
  "est_mem_mb": 5600,               // estimated resident footprint (measured peak wins)
  "supports_vision": true,          // llm: accepts image_url content parts
  "supports_tools": true,           // llm: accepts OpenAI tool calls
  "pinned": false,                  // never evicted under memory pressure
  "custom": false                   // registered via POST /v1/models/register
}
```

### `POST /v1/jobs`

`multipart/form-data` with:

- `json` — a part (filename `params.json`, `application/json`) containing `GenParams`
- `images` — one part per image (hero image first; up to `max_images` used)

`GenParams`:

```json
{
  "model": "auto",          // adapter id or "auto" (server picks by device)
  "seed": 42,
  "subject": "round domed furniture knob",
  "image_roles": ["front", "back"],   // front|back|left|right|iso|other (aligned with images)
  "steps": 30,              // optional inference steps
  "guidance": 5.0,          // optional
  "resolution": 256,        // optional octree/voxel hint
  "batch": [{"seed": 1}, {"seed": 2}],   // optional: N variants in ONE job —
                                         // candidate screening shares the load
  "control_type": "bbox",   // optional conditioning (hunyuan3d-omni):
                            // bbox | voxel | point | pose
  "control_input": [60, 40, 12]          // bbox = [dx,dy,dz] mm; ignored elsewhere
}
```

Response `200`: `{ "job_id": "…" }` — `422` if no image, `429` when the queue
is full or free host RAM is below `MESH_JOB_MIN_FREE_MB`. One generation job
runs at a time; the rest queue.

### `GET /v1/jobs/{id}`

```json
{
  "status": "queued|running|succeeded|failed|cancelled",
  "progress": 0.42,
  "eta_s": null,
  "message": "diffusion steps",
  "model": "hunyuan3d-2.1-mlx",
  "backend": "mlx",
  "seed": 42,
  "steps": 30,
  "guidance": 5.0,
  "resolution": 256,
  "duration_s": 12.3,
  "triangles": 18420,
  "busy": true,            // subprocess liveness: child CPU >2% recently
  "outputs": [0, 2],       // batch jobs: succeeded variant indices
  "error": null
}
```

### `GET /v1/jobs/{id}/mesh`

`200` → GLB bytes (`model/gltf-binary`) in **generator-normalized space**
(not mm, not oriented — the cad-executor pipeline handles that). `?i=N`
selects batch variant `N` (see `outputs` on the job status); `409` while the
job is not `succeeded`, `404` unknown id/variant.

### `DELETE /v1/jobs/{id}`

Cancels a queued job and deletes its result. A running job is marked
cancelled; its result is discarded (model calls are not interruptible
mid-flight). `200 {"ok": true, "deleted": bool}`.

## Text-to-image jobs

Second adapter family (`kind: "image"` in `/v1/models`): FLUX.1-schnell via
`mflux` on Apple Silicon, FLUX.1-schnell / SDXL / SD1.5 via diffusers on CUDA,
`fake-image` for tests. Same job queue — a running mesh job blocks image jobs
and vice versa; residency is governed by the memory ledger (see
[Queue & memory](#queue--memory)) — several models may stay loaded as long
as they fit, LRU residents are evicted when a newcomer needs the room.

### `POST /v1/image-jobs`

Plain JSON body `ImageGenParams` (no multipart — there are no inputs):

```json
{
  "prompt": "studio photo of a red gear",
  "negative": "",
  "model": "auto",          // image adapter id or "auto"
  "seed": 42,
  "count": 1,               // 1..max_count per adapter
  "width": 768,
  "height": 768,
  "steps": 4,               // optional
  "guidance": 3.5           // optional
}
```

Response `200`: `{ "job_id": "…" }` — `422` on invalid params, `429` when the
queue is full. With no usable adapter (none installed / all badlisted) the
submitted job lands in `failed` with a "no image adapter available" error.

### `POST /v1/image-edit-jobs`

Multipart variant carrying an init image for edit-capable adapters
(`supports_edit` in `/v1/models`): fields `json` (same `ImageGenParams`,
plus optional `strength` 0..1) and `image` (the PNG/JPEG to edit). Job
lifecycle + result download are the same as `/v1/image-jobs`. `model="auto"`
picks only adapters that accept an init image.

### `GET /v1/image-jobs/{id}`

Same shape as mesh jobs plus `"images": <n>` (result count) and `"backend"`.

### `GET /v1/image-jobs/{id}/image?i=0`

`200` → PNG bytes of result image `i` (0-indexed). `409` while the job is not
`succeeded`, `404` unknown id, `400` index out of range.

### `DELETE /v1/image-jobs/{id}`

Same semantics as mesh job deletion.

## LLM chat completions

Third adapter family (`kind: "llm"`): local LLMs served as managed
OpenAI-compatible **server subprocesses** on a loopback port —
`mlx-lm`/`mlx-vlm` (`python -m mlx_lm.server` / `python -m
mlx_vlm.server`, Apple Silicon) and `llama.cpp` (`llama-server` binary or
`python -m llama_cpp.server`, GGUF weights; a `gguf_mmproj` projector file
enables vision). Vision adapters spawn `mlx_vlm.server`, not
`mlx_lm.server` — mlx_lm's server has no Qwen-family vision processor and
would silently drop image parts; the adapter picks the module itself
(`vision=True` → `mlx_vlm`). The proxy resolves the requested model, loads
its server through the registry's single-flight `get()` (concurrent
requests converge on ONE load), rewrites `model` to the id the inner server
actually serves, then relays the response verbatim — `stream: true` bodies
pass through as SSE.

#### Catalog

The built-in catalog is **platform-gated** — MLX quants on darwin-arm64,
llama.cpp GGUFs (each with a `mmproj-BF16` vision projector) on
CUDA/ROCm/other hosts. All entries are natively multimodal
(image-text-to-text), 262k context:

| id (darwin-arm64 / other) | weights | size |
|---|---|---|
| `qwen3.5-9b-mlx` / `qwen3.5-9b-gguf` | `mlx-community/Qwen3.5-9B-4bit` / `unsloth/Qwen3.5-9B-GGUF` Q4_K_M | ~6 GB |
| `qwen3.8-27b-mlx` / `qwen3.8-27b-gguf` | `mlx-community/Qwen3.8-27B-4bit` / `unsloth/Qwen3.8-27B-GGUF` Q4_K_M | ~16/~17 GB |
| `qwen3.6-35b-a3b-mlx` / `qwen3.6-35b-a3b-gguf` | `mlx-community/Qwen3.6-35B-A3B-4bit` / `unsloth/Qwen3.6-35B-A3B-GGUF` Q4_K_M (MoE) | ~20/~21 GB |
| `qwen3.5-122b-a10b-mlx` / `qwen3.5-122b-a10b-gguf` | `mlx-community/Qwen3.5-122B-A10B-4bit` / `unsloth/…-GGUF` split Q4_K_M (MoE; MLX needs ≥96 GB RAM) | ~70 GB |
| `qwen3.8-flash-next-mlx` (darwin only) | `mlx-community/Qwen3.8-Flash-Next-4bit` (≥128 GB RAM) | ~112 GB |

Image parts arrive as OpenAI `image_url` data-URIs; there is no model-side
image-count limit — context is the bound (the CADX worker caps at 8
images/message anyway). User-registered repos are added via
`POST /v1/models/register` and persisted in
`model_dir/custom_models.json`. Adapter subprocess logs land in
`job_dir/llm-<id>.log`.

#### Install

`mesh_server/scripts/install_llm.sh` sets up the backend deps. On
darwin-arm64 it creates a dedicated `mesh_server/.venv-llm` holding
`mlx-lm>=0.32` + `mlx-vlm>=0.7` — both need `transformers>=5`, which
conflicts with the shared venv's `transformers<5` pin (torch/diffusers
extras). The mlx adapter resolves its interpreter as `MESH_LLM_PYTHON` →
`mesh_server/.venv-llm/bin/python` → `sys.executable`; a legacy
`mlx-lm<0.30` stays in the shared venv (`llm` extra) as fallback for older
architectures. On other hosts the script installs the `llm` extra
(`llama-cpp-python[server]`) into the shared venv — a system
`llama-server` binary or `LLAMA_SERVER_BIN` also works. GGUF vision needs
a llama.cpp build that supports the `qwen3_5` architecture family plus
`--mmproj` (added automatically for `gguf_mmproj` entries).

`HF_TOKEN` in the environment — or the token stored via WebUI settings,
which the backend forwards per request — makes downloads of gated repos
work transparently.

### `POST /v1/chat/completions`

Standard OpenAI chat-completions body (`model`, `messages`, `tools`,
`temperature`, `stream`, …) — forwarded as-is, only `model` is rewritten.

Model resolution (`select_llm`): an explicit catalog id wins; `"auto"` (or
an empty/unknown id — e.g. an external provider's model name leaking
through) falls back to the first available **and downloaded** adapter.
Vision requests need `supports_vision`; check `/v1/models` first.

Responses:

- `200` — upstream JSON or `text/event-stream` relayed verbatim.
- `404` — `{"error": {"message": …}}` when no llm adapter can serve the
  request (empty catalog or nothing downloaded).
- `503` — load failure/timeout; may carry `"loading": true` while another
  request's load is still in flight. `502` when the spawned server is
  unreachable.

## Model management

Weight downloads run on a `DownloadManager` thread per model — no model is
loaded while weights stream in, HF transfers resume partial files, and a
cancel just flags the job back to `idle`. Only llm adapters are
downloadable today (`hf_repo` snapshot / `gguf_files`); mesh/image adapters
report `downloaded` via availability and reject `POST …/download` with
`400`.

### `POST /v1/models/{id}/download`

Start (idempotent — an in-flight download returns its current status, an
already-downloaded model short-circuits to `done`). `404` unknown id.

### `GET /v1/models/{id}/download`

```json
{
  "status": "idle|downloading|done|error",
  "bytes_done": 0,
  "bytes_total": null,
  "error": null
}
```

### `DELETE /v1/models/{id}/download`

Flag a running download to stop → `{"ok": bool}` (`false` when nothing is
downloading).

### `DELETE /v1/models/{id}/weights`

Remove cached weights from disk. Guarded — refuses while the model is
loaded or downloading → `{"deleted": false, "error": "…"}`; on success
`{"deleted": true}`.

### `POST /v1/models/register`

Persist a custom HuggingFace repo as an llm catalog entry (effective
immediately, survives restarts):

```json
{
  "repo": "org/name",               // required
  "kind": "llm",                    // only "llm" is accepted today
  "backend": "mlx-lm",              // "mlx-lm"|"llama.cpp"|"" — "" = auto by device
  "name": "friendly name",
  "vision": false,
  "tools": true,
  "est_mem_mb": null                // admission estimate until measured
}
```

`200` → the created `ModelInfo`; `422` on an invalid repo or an entry
incomplete for its backend. For `llama.cpp` the server tries to infer the
GGUF file list from the repo (Q4_K_M preferred, `mmproj` detected →
`vision: true`); entries that stay unresolvable are skipped on load.

### `DELETE /v1/models/{id}/custom`

Remove a registered custom model → `{"ok": bool}`; `409` while the model
is loaded, `{"ok": false}` when the id isn't a custom entry.

## Queue & memory

Generation jobs share a model-affine scheduling queue and a **memory
ledger** (`model_dir/model_stats.json` persists measured peak RSS per
model):

- **Admission**: a queued job runs only when `free_mb ≥ est_mb +
  MESH_MEM_HEADROOM_MB`, where `est_mb = max(declared est_mem_mb,
  1.1 × measured peak)`. When it doesn't fit, idle warm workers then LRU
  registry residents are evicted; `pinned` models (llm with `MESH_LLM_PIN`)
  are never evicted. Jobs that can't fit wait as `"waiting for memory"` and
  fail with `insufficient host memory` after `MESH_QUEUE_MEM_WAIT_S`
  instead of OOM-crashing (or head-of-line blocking the queue).
- **Scheduling**: pick score = `waited_s` plus a `MESH_QUEUE_FAIR_S` bonus
  when the job's `resolved_model` matches the last-used model or an idle
  warm worker — same-model bursts batch on resident weights, but a job
  older than the fairness window always wins, so affinity can't starve.
- **Warm workers** (`MESH_WARM_WORKERS`): `gen_worker --serve` children
  keep one resolved model resident and serve subsequent same-model jobs via
  stdin/stdout (saves the per-job reload); bounded by `MESH_WARM_MAX`, they
  self-exit after `MESH_WARM_IDLE_S` of stdin silence and the parent falls
  back to a cold subprocess on death or model mismatch.
- **Multi-residency**: several models (any kind, incl. llm servers) may
  stay loaded simultaneously while they fit — the old single-resident
  invariant is replaced by memory admission.

### `GET /v1/queue`

```json
{
  "jobs": [
    {
      "id": "…",
      "kind": "mesh",               // "mesh" | "image"
      "model": "auto",              // requested
      "resolved_model": "triposg",  // adapter picked at submit/pick time
      "status": "queued",
      "waited_s": 12.3,
      "est_mem_mb": 6200,
      "message": "queued"           // "waiting for memory" while blocked
    }
  ],
  "free_mb": 41000,
  "residents": [
    {"id": "qwen3.5-9b-mlx", "mb": 5200, "kind": "model", "pinned": true, "idle_s": 3.1},
    {"id": "warm:triposg", "mb": 6100, "kind": "warm", "pinned": false, "idle_s": 41.0}
  ]
}
```

## VRAM / OOM behaviour

- Residency is admission-controlled (see [Queue & memory](#queue--memory)):
  models only coexist while `MESH_MEM_HEADROOM_MB` stays free; non-pinned
  LRU residents are evicted for newcomers, `unload_all()` is the OOM
  recovery path.
- OOM during load or inference → job `failed` with a human-readable message
  ("VRAM exhausted — try a smaller model or fewer/smaller images"), the
  adapter is `unload()`ed and badlisted (`last_error` in `/v1/models`).
- `model: "auto"` selection skips badlisted adapters (automatic fallback
  flux → sdxl → sd15); an explicitly requested model surfaces its real error.
- `MESH_CUDA_OFFLOAD=auto|model|sequential|none` controls diffusers
  CPU offload on CUDA (default `auto` = model offload when VRAM is tight).
- Request `width`/`height`/`count` are clamped per adapter before generation.

## New environment variables

In addition to the existing `MESH_*` knobs (see
[configuration.md](configuration.md) for the full table):

| Variable | Default | Notes |
|----------|---------|-------|
| `MESH_MEM_HEADROOM_MB` | `4096` | host RAM kept free when admitting a model |
| `MESH_WARM_WORKERS` | `1` | keep `gen_worker --serve` children resident between jobs |
| `MESH_WARM_IDLE_S` | `300` | warm worker self-exit after stdin silence |
| `MESH_WARM_MAX` | `2` | max concurrent warm workers |
| `MESH_QUEUE_FAIR_S` | `120` | fairness window — jobs older than this beat model affinity |
| `MESH_QUEUE_MEM_WAIT_S` | `300` | queued job waiting on memory fails after this (0 = never) |
| `MESH_JOB_MIN_FREE_MB` | `0` | refuse new jobs (→ `429`) when host free RAM is below this (0 = off) |
| `MESH_LLM_PIN` | `0` | pin loaded llm servers — never evicted for memory pressure |
| `MESH_LLM_LOAD_TIMEOUT_S` | `300` | readiness bound for a managed llm server subprocess |
| `MESH_LLM_CTX` | `16384` | context size passed to llama.cpp servers |
| `MESH_LLM_PYTHON` | `""` | interpreter for mlx llm servers (else `mesh_server/.venv-llm/bin/python`, else the shared venv) |
| `LLAMA_SERVER_BIN` | `""` | explicit `llama-server` binary (else PATH, else `llama_cpp.server`) |

## Semantics

- Meshes are cleaned on the server (floaters, normals, holes) but **not**
  scaled/oriented — CADX's `cad/mesh_ops.py` owns the frame contract.
- Results expire after `MESH_JOB_TTL_S` (default 1 h).
- The client cancels the server job whenever its own agent job is cancelled.

## Error behaviour expected by the backend

- `GET /v1/health` non-200/timeout → `mesh_available = false` → mode falls
  back to CAD.
- `status == "failed"` → `MeshGenError` → iteration retry with a new seed.
- Poll timeout (`MESH_TIMEOUT_S`) → `MeshGenError`; client issues `DELETE`.

## See also

- [Mesh generation](mesh-generation.md) — how the backend drives this API
- [`mesh_server/README.md`](../mesh_server/README.md) — adapters and setup
