# mesh-server HTTP API

Contract between `cadagent` (worker → `agent/mesh_client.py`) and the
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
  "supports_multiview": false,
  "max_images": 1,
  "license_note": "Tencent Hunyuan Community License — …",
  "loaded": false,
  "available": true                 // deps + weights usable on this host
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

## Semantics

- Meshes are cleaned on the server (floaters, normals, holes) but **not**
  scaled/oriented — cadagent's `cad/mesh_ops.py` owns the frame contract.
- Results expire after `MESH_JOB_TTL_S` (default 1 h).
- The client cancels the server job whenever its own agent job is cancelled.

## Error behaviour expected by the backend

- `GET /v1/health` non-200/timeout → `mesh_available = false` → mode falls
  back to CAD.
- `status == "failed"` → `MeshGenError` → iteration retry with a new seed.
- Poll timeout (`MESH_TIMEOUT_S`) → `MeshGenError`; client issues `DELETE`.
