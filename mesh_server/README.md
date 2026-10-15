# mesh-server

Local image→3D mesh generation sidecar for CADX. The backend
(`backend/cadx/agent/mesh_client.py`) talks to it over HTTP — like an
Ollama-style service — so the cad-executor stays sandboxed and the heavy model
runtime lives in its own uv project/venv.

## Quick start (macOS / Apple Silicon)

```bash
# from the repo root — auto-detects the GPU tier and installs the adapter
../scripts/setup.sh     # (or: make autosetup / scripts/start.sh for the full stack)
make mesh-server        # serves on http://127.0.0.1:8765

# manual equivalent:
make mesh-sync          # creates mesh_server/.venv with the right extras (mlx on arm64)
make mesh-models        # downloads the default weights
```

Then point CADX at it — set `mesh_base_url` to `http://localhost:8765`
in **Settings** (the app stores it in its database; `start.sh` seeds the
right value on first boot, e.g. `http://host.docker.internal:8765` for the
docker stack reaching a native sidecar).

Check status:

```bash
cadx mesh-health          # or: curl http://localhost:8765/v1/health
cadx mesh-gen photo.png -o out.glb
```

## Adapters / models

| id | model | device | license | install |
|----|-------|--------|---------|---------|
| `trellis2` | TRELLIS.2-4B (shape only) | CUDA only | **MIT** | `scripts/install_trellis2.sh` / `Dockerfile.cuda` |
| `triposg` | TripoSG 1.5B | CUDA, ROCm, MPS (exp.) | **MIT** | `scripts/install_triposg.sh` (`install_rocm.sh` on AMD) |
| `hunyuan3d-2.1-mlx` | Hunyuan3D-2.1 Shape | MLX (arm64 Mac) | Tencent Community* | `scripts/install_hunyuan_mlx.sh` |
| `hunyuan3d-2mini` / `-turbo` | Hunyuan3D-2 mini Shape | torch MPS, ROCm | Tencent Community* | `scripts/install_hunyuan_torch.sh` |
| `hunyuan3d-2mv` / `-mv-turbo` | Hunyuan3D-2 multi-view | torch MPS, ROCm | Tencent Community* | same |
| `fake` | deterministic test fixture | — | MIT | always available with `MESH_FAKE=1` |

\* **The Tencent Hunyuan Community License excludes the EU/UK/South Korea.**
Check whether it applies to you; TRELLIS.2 and TripoSG are MIT-licensed
alternatives.

`model=auto` picks: `trellis2` on NVIDIA CUDA, `triposg` on AMD ROCm (single
image) or `hunyuan3d-2mv` (multi-view, if installed), `hunyuan3d-2.1-mlx` on
arm64 Macs with MLX, `hunyuan3d-2mv` when ≥2 view-tagged images are given on
MPS, otherwise `hunyuan3d-2mini`/`triposg`.

A second adapter family serves local **text-to-image and image editing**
(`kind: "image"` in `/v1/models`): `flux-schnell-mlx`, `flux-schnell`,
`sdxl`, `sd15`, `qwen-image-edit`, `flux-kontext-mlx` (opt-in via
`MESH_FLUX_KONTEXT=1`) and `fake-image`. See
[`docs/image-generation.md`](../docs/image-generation.md).

## HTTP API

See [`docs/mesh-server-api.md`](../docs/mesh-server-api.md). In short:

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
| `MESH_DEVICE` | `auto` | `auto \| cuda \| rocm \| mps \| mlx \| cpu` |
| `MESH_MODEL_DIR` | `~/.cache/meshserver/models` | weight cache |
| `MESH_JOB_DIR` | `/tmp/meshserver-jobs` | job scratch space |
| `MESH_JOB_TTL_S` | `3600` | result retention |
| `MESH_MAX_QUEUE` | `8` | queued jobs cap (one GPU job at a time) |
| `MESH_UNLOAD_IDLE_S` | `0` (never) | unload weights after idle seconds |
| `MESH_FAKE` | `0` | enable the fake adapters |
| `MESH_ALLOW_CPU` | `0` | allow real adapters on CPU (slow) |
| `MESH_DEFAULT_MODEL` | `auto` | mesh adapter for `model=auto` requests |
| `MESH_IMAGE_MODEL` | `auto` | image adapter for `model=auto` requests |
| `MESH_BG_REMOVAL` | `auto` | `auto \| always \| never` (rembg/u2net) |
| `MESH_TRIPOSG_MPS` | `0` | experimental MPS path for TripoSG |
| `MESH_FLUX_KONTEXT` | `0` | enable the gated FLUX.1-Kontext edit adapter |
| `MESH_FLUX_MLX_QUANT` | `4` | mflux weight quantization (4/8/16 bit) |
| `MESH_CUDA_OFFLOAD` | `auto` | `auto \| model \| sequential \| none` diffusers CPU offload |
| `MESH_CUDA_OFFLOAD_VRAM_GB` | `24` | free-VRAM threshold for `auto` offload |
| `MESH_MLX_MEM_LIMIT_GB` | `0` | unified-memory cap (0 = unlimited) |

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

## AMD (ROCm) deployment

Linux host with the `amdgpu` driver (`/dev/kfd` + `/dev/dri/renderD*`).
Native install goes through `scripts/install_rocm.sh` (ROCm torch wheel →
TripoSG; `HUNYUAN=1` adds the Hunyuan3D-2 torch adapters). Wheel index
override: `TORCH_ROCM_INDEX`.

```bash
docker build -f mesh_server/Dockerfile.rocm -t meshserver:rocm .
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    --group-add render --ipc=host -p 8765:8765 -v mesh_models:/models \
    meshserver:rocm
# or: docker compose --profile gpu-rocm up -d mesh-server-rocm
```

Notes: ROCm torch reuses the `cuda` device namespace — adapters report
`device: rocm` via `torch.version.hip`. The CUDA-only `diso` flash decoder is
replaced by the skimage marching-cubes path automatically. gfx archs not
covered by the installed wheel (e.g. gfx1031) need `HSA_OVERRIDE_GFX_VERSION`.

First run downloads weights into `MESH_MODEL_DIR`/`HF_HOME` — mount a volume.

## In-process CLI

```bash
meshserver models                      # adapter availability on this device
meshserver generate photo.png -o out.glb --model auto --seed 42
meshserver download hunyuan3d-2.1-mlx  # prefetch weights
meshserver image --prompt "a red gear" --model flux-schnell-mlx
meshserver health
```

## See also

- [`docs/mesh-server-api.md`](../docs/mesh-server-api.md) — the HTTP contract
- [`docs/mesh-generation.md`](../docs/mesh-generation.md) — how CADX uses it
- [`docs/configuration.md`](../docs/configuration.md) — all env vars
