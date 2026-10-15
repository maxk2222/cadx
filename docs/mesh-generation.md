# Mesh generation

Generation modes: `cad | mesh | hybrid | topopt | auto`.

CADX can generate geometry in three ways and lets a planner pick:

| mode | what runs | output |
|------|-----------|--------|
| `cad` | build123d code → B-Rep | analytic STEP, all 6 exports |
| `mesh` | image→3D model → trimesh pipeline | STL/3MF/GLB (+ tessellated STEP when convertible) |
| `hybrid` | mesh → faceted B-Rep solid → build123d booleans | organic shape + exact features, all exports (STEP tessellated) |
| `auto` | planner decides (`plan.approach`) | — |

`auto` is the default (`agent_default_gen_mode` runtime setting, project settings,
`GenerateRequest.mode`, CLI `--mode`). Guard rails: `mesh`/`hybrid` need
**at least one image** and a **reachable mesh server** — otherwise the run
falls back to CAD with a note in the job log and `resolved_mode="cad"`.

![Hybrid result — photo-derived mesh body with exact countersunk holes](screenshots/version-hybrid.png)

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
  plus the recorded transform **ops** are stored as `version.params` — the
  params job re-runs the pipeline **without any LLM call**
  (`MeshPostParams`/`MeshOp` in `cad/mesh_ops.py`).
- Mesh versions have `code = None` — the "show code" UI is hidden and
  `model.py` is only written when code exists.

## Models & hardware

| adapter id | model | device | license |
|---|---|---|---|
| `trellis2` | TRELLIS.2-4B (shape only) | CUDA ≥ ~24 GB | **MIT** |
| `triposg` | TripoSG 1.5B | CUDA, MPS exp. | **MIT** |
| `triposg-scribble` | TripoSG-scribble (sketch+text) | CUDA, MPS exp. | **MIT** (experimental) |
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
`hunyuan3d-2mini`/`triposg`. Note: `hunyuan3d-2mv` on torch-MPS has been
observed to stall mid-generation under memory pressure (job pinned at ~10 %
progress, worker polling until the stall watchdog fires) — on MPS hosts
prefer `hunyuan3d-2.1-mlx` or a `mini`/`turbo` variant as `mesh_model`. When `DATA_DIR/mesh_bench.json` exists (written
by `mesh_server/scripts/bench_adapters.py`), `auto` additionally ranks
available adapters by their measured per-class results (`agent/model_pick.py`).
Rows include `roughness_deg` (mean dihedral on non-horizontal faces —
stair-stepping proxy) and the corpus carries `detailed_gear.json`
(14-tooth silhouette + bore) to catch fine-detail loss; the flat-class
pick score weights edge sharpness slightly.

## Generation inputs & candidates

- **View roles** come from intake `image_roles`, overridden by the measure
  step's deterministic plan-view classification for flat parts.
- **Synthetic back view**: a flat part without a real bottom view gets a
  filled-silhouette PNG rendered from the measured outline (`back` role) —
  anchors the underside without inventing features.
- **Best-of-N screening** (`mesh_candidates_per_iter`, default 2): cheap
  low-step candidates are scored deterministically (aspect + footprint IoU +
  height agreement + hole count) before the full run; all-hopeless
  candidates skip generation entirely and degrade to CAD.
- **Multi-adapter candidates** (`mesh_candidate_adapters`, csv): candidates
  spread over several models; the winning adapter is reused for the full
  generation. "" spreads over all available adapters automatically.
- **Heightfield candidate** (`mesh_heightfield_flat`, default on): flat parts
  get a deterministic watertight column-solid built from the measured
  outline + stated height (optional luminance relief). It competes on score
  and stands alone when no candidate screening runs — this eliminates the
  classic "standing banana" failure for flat parts structurally.

## Install

`scripts/setup.sh` auto-detects the GPU tier and installs the matching
adapter — see [Getting started](getting-started.md) for the full flow. Manual
alternatives:

```bash
# macOS (native — docker has no GPU access)
make mesh-sync
mesh_server/scripts/install_hunyuan_mlx.sh     # or install_hunyuan_torch.sh
make mesh-server                               # :8765
# native dev: Settings → mesh_base_url = http://localhost:8765
# docker: start.sh seeds host.docker.internal automatically

# CUDA box
docker build -f mesh_server/Dockerfile.cuda -t meshserver:cuda .
docker run --gpus all -p 8765:8765 -v mesh_models:/models meshserver:cuda
# or: docker compose --profile gpu up -d mesh-server
#     → MESH_BASE_URL=http://mesh-server:8765 for the worker
# native (no container): scripts/setup.sh picks trellis2 (>=20 GB VRAM,
# override via MESH_TRELLIS_MIN_VRAM_GB) or triposg automatically
mesh_server/scripts/install_trellis2.sh   # TRELLIS.2-4B on the CUDA box
mesh_server/scripts/install_triposg.sh    # TripoSG (+ -scribble via same pkg)
```

## Adapter benchmark

```bash
cd mesh_server && uv run python scripts/bench_adapters.py \
    --out <DATA_DIR>/mesh_bench.json          # all available adapters
uv run python scripts/bench_adapters.py --adapters triposg,hunyuan3d-2.1-mlx \
    --seeds 2                                # subset / more rolls
```

Measures per model × part class (`flat` | `organic`): success, latency,
footprint IoU, aspect/height error, watertightness, hole recall. The backend
uses the report for `auto` adapter selection; stale reports are harmless
(absent → static heuristic).

## Flow details

**mesh**: `MESH_GENERATE` (seed = `mesh_seed` or per-version base + iteration)
→ `ORIENT` (6 candidates rendered in the executor, vision LLM picks
`MeshOrientation{up,front}`; heuristic fallback; parent orientation is reused
on edits) → `PROCESS_MESH` → `MESH_REFINE` shape session → dim-check →
`CRITIQUE` → issues-seeded `MESH_REFINE` repair round(s) → best candidate →
exports. Remaining iterations vary the seed (a refine session can also force
a re-roll via `regenerate`); `edit` = regenerate, `params` =
post-params+ops rerun.

**MESH_REFINE** (`agent/steps/mesh_refine.py`, workbench in
`agent/mesh_tools.py`) is a tool session on the vision role: the agent
inspects the baseline-processed mesh (`mesh_inspect`), renders it
(`mesh_preview`, capped at `agent_mesh_preview_max`), applies bounded
transforms — `set_up`, `rotate` (any axis), `mirror`, `scale` (uniform or
per-axis), `flatten_bottom`, `slice`, `crop`, `decimate`, `smooth`,
`keep_largest`, `drop_small`, `repair`, `clip_height` (Z-clamp + cap),
`symmetrize` (mirror best half), `remesh` (voxel rebuild), `drill_holes`
(explicit positions or `from_anchors` using the measured hole anchors),
`shell` (hollow + opening), `boolean_add`/`boolean_cut` with
box/cylinder/sphere primitives — plus inspection tools `mesh_measure`
(bbox/volume/watertight), `mesh_section` (cross-section image),
`mesh_overlay` (mesh footprint vs measured outline) — and commits with
`done`. Every call is
validated through the op's pydantic model (`OPS_REGISTRY`), applied to a
worker-side trimesh state, and the mesh is re-grounded after each op so
session preview and executor output can never drift. The session only
*records* ops — the authoritative result is re-produced by a
`process_mesh` run in the executor. Limits: `agent_mesh_tools`,
`agent_mesh_max_ops`, `agent_mesh_preview_max`, `agent_mesh_refine_rounds`;
unavailable sessions fall back to the plain seed-rotation loop.

**hybrid**: same mesh prep in `iterations/0` → `mesh_to_solid` (decimated to
`mesh_max_faces_for_cad` first; one retry at ≤15k faces) → CAD loop with
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
versions/<vid>/mesh/mesh.json       provenance: model, seed, orientation, post-params, ops, ops_report
versions/<vid>/iterations/0/        hybrid mesh prep (generation + orient + solid)
versions/<vid>/iterations/<n>/      per-iteration runs
```

STEP export for mesh/hybrid results is **tessellated** (faceted), marked
`{"tessellated": true}` on the artifact and labelled in the UI.

## Troubleshooting

Common mesh-server and pipeline failures are indexed in
[troubleshooting.md](troubleshooting.md). Mesh-specific quick hits:

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
  model; set `always` to allow the one-time download.

## See also

- [mesh-server API](mesh-server-api.md) — sidecar contract
- [Image generation](image-generation.md) — text-to-image adapters + `mesh_regen_edit`
- [Agent loop](agent-loop.md) — MESH_GENERATE / ORIENT / MESH_REFINE steps
- [`mesh_server/README.md`](../mesh_server/README.md) — adapter install details
