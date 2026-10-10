# Mesh generation (v2: cad | mesh | hybrid | auto)

cadagent can generate geometry in three ways and lets a planner pick:

| mode | what runs | output |
|------|-----------|--------|
| `cad` | build123d code → B-Rep | analytic STEP, all 6 exports |
| `mesh` | image→3D model → trimesh pipeline | STL/3MF/GLB (+ tessellated STEP when convertible) |
| `hybrid` | mesh → faceted B-Rep solid → build123d booleans | organic shape + exact features, all exports (STEP tessellated) |
| `auto` | planner decides (`plan.approach`) | — |

`auto` is the default (`AGENT_DEFAULT_GEN_MODE`, project settings,
`GenerateRequest.mode`, CLI `--mode`). Guard rails: `mesh`/`hybrid` need
**at least one image** and a **reachable mesh server** — otherwise the run
falls back to CAD with a note in the job log and `resolved_mode="cad"`.

## Architecture

```
worker ──HTTP──► mesh-server (own uv project, runs NATIVELY)
   │               adapters: trellis2 | triposg | hunyuan3d-2.1-mlx | hunyuan3d-2(mini|mv) | fake
   │  GLB (normalized space)
   ▼
cad-executor  process_mesh → clean → decimate → orient → scale →
                            analysis → renders → DfM(mesh) → exports
   │            (make_solid → mesh_to_solid → solid.brep for hybrid)
   ▼
hybrid only:  runner_entry injects load_mesh_solid() + MESH_BBOX
              → normal CAD loop adds exact features via booleans
```

- The **worker** calls the mesh server (`public` network); the **executor**
  only sees files under `DATA_DIR` (internal, sealed).
- Mesh post-params (scale, up, front, rotate_z, decimate, flatten, mirror)
  are stored as `version.params` — the params job re-runs the pipeline
  **without any LLM call** (`MeshPostParams` in `cad/mesh_ops.py`).
- Mesh versions have `code = None` — the "show code" UI is hidden and
  `model.py` is only written when code exists.

## Models & hardware

| adapter id | model | device | license |
|---|---|---|---|
| `trellis2` | TRELLIS.2-4B (shape only) | CUDA ≥ ~24 GB | **MIT** |
| `triposg` | TripoSG 1.5B | CUDA, MPS exp. | **MIT** |
| `hunyuan3d-2.1-mlx` | Hunyuan3D-2.1 shape | arm64 Mac (MLX) | Tencent* |
| `hunyuan3d-2mini` / `-turbo` | Hunyuan3D-2 mini | torch MPS | Tencent* |
| `hunyuan3d-2mv` / `-mv-turbo` | Hunyuan3D-2 multi-view | torch MPS | Tencent* |
| `fake` | deterministic fixture | any | MIT |

\* **The Tencent Hunyuan Community License excludes the EU/UK/South Korea.**
Check applicability; TRELLIS.2/TripoSG are MIT defaults where available.
Model + license are surfaced in Settings → "Mesh generator" and in
`version.mesh.generation.model`.

`model="auto"` chooses: `trellis2` on CUDA, `hunyuan3d-2.1-mlx` on arm64 Macs
with MLX, the `mv` variant when ≥2 view-tagged images exist, otherwise
`hunyuan3d-2mini`/`triposg`.

## Install

```bash
# macOS (native — docker has no GPU access)
make mesh-sync
mesh_server/scripts/install_hunyuan_mlx.sh     # or install_hunyuan_torch.sh
make mesh-server                               # :8765
# .env: MESH_BASE_URL=http://localhost:8765 (native dev)
#       compose handles host.docker.internal automatically

# CUDA box
docker build -f mesh_server/Dockerfile.cuda -t meshserver:cuda .
docker run --gpus all -p 8765:8765 -v mesh_models:/models meshserver:cuda
# or: docker compose --profile gpu up -d mesh-server
#     → MESH_BASE_URL=http://mesh-server:8765 for the worker
```

## Flow details

**mesh**: `MESH_GENERATE` (seed = `mesh_seed` or per-version base + iteration)
→ `ORIENT` (6 candidates rendered in the executor, vision LLM picks
`MeshOrientation{up,front}`; heuristic fallback; parent orientation is reused
on edits) → `PROCESS_MESH` → dim-check → `CRITIQUE` → best candidate →
exports. Iterations vary the seed; `edit` = regenerate, `params` =
post-params rerun.

**hybrid**: same mesh prep in `iterations/0` → `mesh_to_solid` (decimated to
`MESH_MAX_FACES_FOR_CAD` first; one retry at ≤15k faces) → CAD loop with
`load_mesh_solid()`/`MESH_BBOX` injected into the sandbox → the code contract
restricts hybrid code to boolean ops (no fillet/edge-select on mesh faces).
`mesh_to_solid` failure → `resolved_mode="mesh"` fallback; generation failure
→ CAD. Edits reuse `versions/<parent>/mesh/solid.brep` unless
`regenerate_mesh=true`.

## Exports & artifacts per version

```
versions/<vid>/mesh/mesh_raw.glb    raw generator output (ArtifactKind mesh_raw)
versions/<vid>/mesh/mesh_clean.stl  post-processed (mm, oriented, scaled)
versions/<vid>/mesh/solid.brep      hybrid: faceted solid fed into build123d
versions/<vid>/mesh/mesh.json       provenance: model, seed, orientation, post-params
versions/<vid>/iterations/0/        hybrid mesh prep (generation + orient + solid)
versions/<vid>/iterations/<n>/      per-iteration runs
```

STEP export for mesh/hybrid results is **tessellated** (faceted), marked
`{"tessellated": true}` on the artifact and labelled in the UI.

## Troubleshooting

- **`mesh server not configured (MESH_BASE_URL)`** — set it; on compose the
  worker gets `http://host.docker.internal:8765` automatically (native
  sidecar) or `http://mesh-server:8765` (`gpu` profile).
- **`no adapter available … on device 'cpu'`** — no GPU detected and
  `MESH_FAKE` off; install an adapter (see `mesh_server/README.md`) or run
  `MESH_FAKE=1`.
- **wrong up/front axis** — change the `up`/`front` post-params and re-run
  params (no LLM); the orientation source is recorded
  (`mesh.orientation_source`: `llm | heuristic | parent`).
- **wrong size** — scale comes from *stated* spec dims (uniform only);
  correct via the `scale` post-param. Aspect mismatches appear as
  `mesh.dim_issues` / dim warnings.
- **hybrid boolean failures** — faceted solids stress OCCT; the system
  retries with stronger decimation, then falls back to mesh-only
  (`needs_review`).
- **rembg download** — `MESH_BG_REMOVAL=auto` only uses an already-cached
  model; set `always` to allow the one-time ~1 GB download.
