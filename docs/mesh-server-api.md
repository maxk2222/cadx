# mesh-server HTTP API

Contract between `cadx` (worker → `agent/mesh_client.py`) and the
`mesh_server/` sidecar. Both sides keep their own pydantic models — there is
**no code sharing** between the two projects; this document is the contract.

Base URL: `MESH_BASE_URL`, e.g. `http://localhost:8765` (native) or
`http://mesh-server:8765` (compose `gpu` profile).

## Endpoints

### `GET /v1/health`

```json
{
  "ok": true,
  "device": "mlx",                 // cuda | mps | mlx | cpu | fake
  "backend": "local",
  "models": [MeshModelInfo]
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
  "kind": "mesh",                   // "mesh" | "image" — second adapter family
  "supports_multiview": false,
  "max_images": 1,
  "license_note": "Tencent Hunyuan Community License — …",
  "loaded": false,
  "available": true,                // deps + weights usable on this host
  "last_error": null                // set after a load/OOM failure ("badlisted")
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
  "resolution": 256         // optional octree/voxel hint
}
```

Response `200`: `{ "job_id": "…" }` — `422` if no image, `429` when the queue
is full. One generation job runs at a time; the rest queue.

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
  "error": null
}
```

### `GET /v1/jobs/{id}/mesh`

`200` → GLB bytes (`model/gltf-binary`) in **generator-normalized space**
(not mm, not oriented — the cad-executor pipeline handles that).
`409` while the job is not `succeeded`, `404` unknown id.

### `DELETE /v1/jobs/{id}`

Cancels a queued job and deletes its result. A running job is marked
cancelled; its result is discarded (model calls are not interruptible
mid-flight). `200 {"ok": true, "deleted": bool}`.

## Text-to-image jobs

Second adapter family (`kind: "image"` in `/v1/models`): FLUX.1-schnell via
`mflux` on Apple Silicon, FLUX.1-schnell / SDXL / SD1.5 via diffusers on CUDA,
`fake-image` for tests. Same job queue — a running mesh job blocks image jobs
and vice versa; at most one heavyweight model is resident at a time (loading
a model unloads all others).

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

Response `200`: `{ "job_id": "…" }` — `422` on invalid params, `503` when no
image adapter is installed.

### `GET /v1/image-jobs/{id}`

Same shape as mesh jobs plus `"images": <n>` (result count) and `"backend"`.

### `GET /v1/image-jobs/{id}/image?i=0`

`200` → PNG bytes of result image `i` (0-indexed). `409` while the job is not
`succeeded`, `404` unknown id/index.

### `DELETE /v1/image-jobs/{id}`

Same semantics as mesh job deletion.

## VRAM / OOM behaviour

- One heavyweight model resident: loading a model unloads all others.
- OOM during load or inference → job `failed` with a human-readable message
  ("VRAM exhausted — try a smaller model or fewer/smaller images"), the
  adapter is `unload()`ed and badlisted (`last_error` in `/v1/models`).
- `model: "auto"` selection skips badlisted adapters (automatic fallback
  flux → sdxl → sd15); an explicitly requested model surfaces its real error.
- `MESH_CUDA_OFFLOAD=auto|model|sequential|none` controls diffusers
  CPU offload on CUDA (default `auto` = model offload when VRAM is tight).
- Request `width`/`height`/`count` are clamped per adapter before generation.

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
