# mesh-server

Local image→3D mesh generation sidecar for cadagent. The backend
(`backend/cadagent/agent/mesh_client.py`) talks to it over HTTP — like an
Ollama-style service — so the cad-executor stays sandboxed and the heavy model
runtime lives in its own uv project/venv.

## Quick start (macOS / Apple Silicon)

```bash
# from the repo root
make mesh-sync        # creates mesh_server/.venv with the right extras (mlx on arm64)
make mesh-server      # serves on http://127.0.0.1:8765
make mesh-models      # downloads the default weights
```

Then point cadagent at it:

```bash
# .env
MESH_BASE_URL=http://localhost:8765
```

Check status:

```bash
cadagent mesh-health          # or: curl http://localhost:8765/v1/health
cadagent mesh-gen photo.png -o out.glb
```

## Adapters / models

| id | model | device | license | install |
|----|-------|--------|---------|---------|
| `trellis2` | TRELLIS.2-4B (shape only) | CUDA | **MIT** | `scripts/install_trellis2.sh` / `Dockerfile.cuda` |
| `triposg` | TripoSG 1.5B | CUDA, MPS (exp.) | **MIT** | `scripts/install_triposg.sh` |
| `hunyuan3d-2.1-mlx` | Hunyuan3D-2.1 Shape | MLX (arm64 Mac) | Tencent Community* | `scripts/install_hunyuan_mlx.sh` |
| `hunyuan3d-2mini` / `-turbo` | Hunyuan3D-2 mini Shape | torch MPS | Tencent Community* | `scripts/install_hunyuan_torch.sh` |
| `hunyuan3d-2mv` / `-mv-turbo` | Hunyuan3D-2 multi-view | torch MPS | Tencent Community* | same |
| `fake` | deterministic test fixture | — | MIT | always available with `MESH_FAKE=1` |

\* **The Tencent Hunyuan Community License excludes the EU/UK/South Korea.**
Check whether it applies to you; TRELLIS.2 and TripoSG are MIT-licensed
alternatives.

`model=auto` picks: `trellis2` on CUDA, `hunyuan3d-2.1-mlx` on arm64 Macs with
MLX, `hunyuan3d-2mv` when ≥2 view-tagged images are given on MPS, otherwise
`hunyuan3d-2mini`/`triposg`.

## HTTP API

See `docs/mesh-server-api.md`. In short:

```
GET  /v1/health
GET  /v1/models
POST /v1/jobs            # multipart: field "json" (GenParams) + "images"
GET  /v1/jobs/{id}
GET  /v1/jobs/{id}/mesh  # GLB bytes
DELETE /v1/jobs/{id}
```

## Configuration (env)

| var | default | meaning |
|-----|---------|---------|
| `MESH_HOST` / `MESH_PORT` | `127.0.0.1` / `8765` | bind address |
| `MESH_DEVICE` | `auto` | `auto \| cuda \| mps \| mlx \| cpu` |
| `MESH_MODEL_DIR` | `~/.cache/meshserver/models` | weight cache |
| `MESH_JOB_TTL_S` | `3600` | result retention |
| `MESH_MAX_QUEUE` | `8` | queued jobs cap (one GPU job at a time) |
| `MESH_UNLOAD_IDLE_S` | `0` (never) | unload weights after idle seconds |
| `MESH_FAKE` | `0` | enable the fake adapter |
| `MESH_ALLOW_CPU` | `0` | allow real adapters on CPU (slow) |
| `MESH_BG_REMOVAL` | `auto` | `auto \| always \| never` (rembg/u2net) |
| `MESH_TRIPOSG_MPS` | `0` | experimental MPS path for TripoSG |

## CUDA deployment

```bash
docker build -f mesh_server/Dockerfile.cuda -t meshserver:cuda .
docker run --gpus all -p 8765:8765 -v mesh_models:/models meshserver:cuda
```

Or via the repo's `gpu` compose profile:

```bash
docker compose --profile gpu up -d mesh-server
# then MESH_BASE_URL=http://mesh-server:8765 for the worker
```

First run downloads weights into `MESH_MODEL_DIR`/`HF_HOME` — mount a volume.

## In-process CLI

```bash
meshserver models                      # adapter availability on this device
meshserver generate photo.png -o out.glb --model auto --seed 42
meshserver download hunyuan3d-2.1-mlx  # prefetch weights
meshserver health
```
