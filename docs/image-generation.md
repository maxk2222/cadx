# Local image generation

CADX can synthesise images with a **local text-to-image model** hosted on the
same `mesh_server` sidecar as the image→3D generators. Three use cases:

1. **Concept images in the asset manager** — "Generate image" button creates
   `generated_image` assets that feed the agent like uploaded references.
2. **Automatic hero images** — when a mesh/hybrid run has no input images, the
   worker generates a hero reference *before* mode resolution
   (`image_autogen_mesh`, on by default), so mesh mode works from text alone.
3. **Agent concept tool** — `generate_concept_image` in info/planning tool
   sessions lets the LLM ideate shapes; results are shown to the model and
   stored as assets (budget: `image_tool_budget` calls per session).

## Setup

`scripts/setup.sh` installs the tier's image adapter automatically (skip with
`--skip-image`):

| tier  | adapter            | model                        | install script               |
|-------|--------------------|------------------------------|------------------------------|
| mlx   | `flux-schnell-mlx` | FLUX.1-schnell via `mflux`, 4-bit | `install_image_mlx.sh`    |
| cuda  | `flux-schnell`     | FLUX.1-schnell via diffusers | `install_image_torch.sh`     |
| cuda  | `sdxl` / `sd15`    | fallbacks for small VRAM     | (same)                       |
| mlx   | `flux-kontext-mlx` | FLUX.1-Kontext-dev — instruction image **editing** (mflux, 4-bit) | `MESH_FLUX_KONTEXT=1` opt-in |
| fake  | `fake-image`       | deterministic test adapter   | `MESH_FAKE=1`                |

FLUX.1-schnell is **Apache-2.0** and ungated on HuggingFace. Weights download
on first use (~12 GB 4-bit MLX, ~24 GB fp16 diffusers). SDXL (~7 GB) and SD1.5
(~4 GB) are the fallbacks for tight memory.

### Image editing (img2img)

Edit jobs carry an init image + `strength` (`POST /v1/image-edit-jobs`,
multipart). `sdxl`/`sd15` run img2img via `AutoPipelineForImage2Image`
(shared weights, `supports_edit`); `flux-kontext-mlx` does instruction
editing and is edit-only (`requires_image`, gated on `MESH_FLUX_KONTEXT=1`
so a bare availability probe never triggers the ~12 GB download). The mesh
pipeline uses this for `mesh_regen_edit` — see `mesh-generation.md`.

Manual check:

```bash
backend/.venv/bin/cadx image-gen "studio photo of a red gear" --model auto
# or straight on the sidecar
mesh_server/.venv/bin/meshserver image --prompt "a red gear" --model flux-schnell-mlx
```

## Runtime settings (Settings → Mesh)

| setting              | default | meaning                                              |
|----------------------|---------|------------------------------------------------------|
| `image_model`        | `auto`  | adapter id; `auto` = device preference + OOM fallback |
| `image_steps`        | empty   | diffusion steps (FLUX 4, SDXL ~30)                   |
| `image_guidance`     | empty   | cfg scale                                            |
| `image_max_side`     | 1024    | clamp on width/height — OOM prevention               |
| `image_max_count`    | 4       | clamp per request                                    |
| `image_autogen_mesh` | on      | synthesise hero images for mesh/hybrid w/o inputs    |
| `image_tool_budget`  | 4       | `generate_concept_image` calls per tool session      |
| `image_timeout_s`    | 300     | client-side generation timeout                       |

`MESH_IMAGE_MODEL` seeds `image_model` once at deploy (like `MESH_MODEL`);
afterwards the DB row is authoritative.

## VRAM / OOM

Mesh and image models **share one residency slot** — loading one unloads all
others, so they never compete for VRAM at the same time. On OOM the adapter is
unloaded, badlisted (`last_error` in `/v1/models`, Settings shows it) and
`auto` selection moves to the next adapter; explicitly requested models return
the real error (no silent downgrade).

CUDA extras: `MESH_CUDA_OFFLOAD=auto|model|sequential|none` enables diffusers
CPU offload (`auto` = on when VRAM is tight; `sequential` gets FLUX down to
~6-8 GB at a speed cost). On Apple Silicon unified memory the lever is the
4-bit `mflux` quantisation (~10-12 GB), not CPU offload.

## Provenance

Generated assets carry `kind=generated_image` and a `gen_params` JSON column
(prompt, model, seed, steps, duration, source: `hero_autogen` | `concept_tool`
| `api`). They render with an `ai` badge in the asset manager and work
everywhere images do — mesh input, `view_image`, edit/derivative, selection
for generation runs.

## API

`POST /api/projects/{id}/images` → generates synchronously and returns the
created `Asset[]` (`201`). `GET /api/images/models` lists image adapters.
Sidecar endpoints are documented in `docs/mesh-server-api.md` ("Text-to-image
jobs").
