# Anleitungsbuilder — PDF manual → Unity AR guide

Turns an uploaded instruction PDF plus uploaded CAD parts into a
`unity_guide.json` consumable by an AR app. Missing parts are generated
through the normal sandboxed CAD pipeline. Everything happens inside one
chat — the agent asks clarifying questions (`waiting_for_input`) and the
user can feed supplemental notes mid-run.

Reference implementation: KIARA `ar_guide_agent_v4` (ported, adapted —
see `docs/anleitungsbuilder-masterplan.md` §3 "Verified platform facts"
for the platform differences that drove the port).

## Entry points

Start-page wizard (`/`): pick **Anleitung** on the goal step, then the
wizard walks through manual PDF + CAD parts → optional notes → review
with all run options (missing-part generation, LLM images, output
language, audience rewrite). It POSTs `/projects/{id}/guide` and hands
off to the shared job-progress view — `waiting_for_input` questions are
answered inline.

REST: `POST /projects/{id}/guide` accepts `pdf_asset_id`,
`cad_asset_ids`, `prompt`, `allow_generation`, `attach_images`, plus
`language` (BCP-47, "" = autodetect the manual's language — a set value
locks `detect_language`) and `audience` (`""` | `beginner` |
`intermediate` | `pro`, applies a text-variant rewrite before the
artifacts are emitted — the shipped `unity_guide.json` already carries
the variant and `_original` keeps the untranslated text).

Editing REST (durable state under `version_dir/…/guide`, every mutation
regenerates `unity_guide.json`):
- `PATCH/POST/DELETE …/guide/steps[/{n}]` — step fields
  (title/text/display_text/safety_hint/media_image), insert via
  `insert_at`, delete migrates that step's instances+annotations to the
  previous step and renumbers everything (incl. `assembly.step_instances`
  and annotation `step` refs).
- `PATCH/POST/DELETE …/guide/instances[/{iid}]` — placed-instance
  transform/step/color/opacity.
- `POST/PATCH/DELETE …/guide/annotations[/{idx}]` — annotations are
  index-addressed: `kind:"arrow"` (`part_id` accepts instance-or-part id
  and resolves to the part, `approach_axis` vec3 or named `direction`,
  optional `label`) and `kind:"text"` (world `position` + `text`), both
  with `color`/`size`.
- `POST …/guide/media` — step-photo sources: multipart file upload or
  `{asset_id}` copy of a project image asset; both land in
  `guide/output/media` with path-safe deduped names and are served via
  `GET …/guide/media/{name}`.
- `POST …/guide/notes` — same `user_notes.jsonl` contract as the chat
  tool (works before `state.json` exists).

Chat tools (project-bound chats):
- `start_guide_builder {pdf_asset_id, cad_asset_ids?, prompt?, wait?}` —
  spawns the guide job (`spawn_guide`).
- `attach_guide_note {job_id, note}` — supplemental info mid-run, lands in
  `user_notes.jsonl` which every mission reads via `list_user_notes`.
- `answer_questions` — resumes a parked guide job (generic
  `enqueue_resume` dispatches by the job's payload func).
- `job_status` — progress as usual.

Uploads: `POST /projects/{id}/assets` accepts `.pdf` (100 MB),
`.glb/.gltf` (50 MB, `AssetKind.cad_glb`), plus the existing
`.step/.stp/.stl/.dxf` reference kinds. The chat composer accept list
covers all of them.

## Pipeline (one durable `JobType.guide` job)

`run_guide_job` (agent queue) → `guide.guide_runner.run_guide` runs the
missions in order, each a bounded `run_session` with a mission-scoped
`GuideToolSession` (tool allowlist per mission, `guide/tools.py`):

| Mission | Tools | Output |
|---|---|---|
| `doc` | prepare_pdf, page text/images/figures, classify, part names | `state.document` |
| `steps` | create/update/reorder steps + sub_steps + media | ordered `state.steps` (dual text; `sub_steps` required in cad mode only — blender instances come from `cadx_step` tags) |
| `parts` | register_uploaded_part, match/alias, generate_missing_part | `state.parts` w/ ports+bbox |
| `assembly` | find_matching_ports, solve_*_preview, place_instance (candidate_id only), collision/render/validate | `state.assembly` |
| `review` | validators + `write_review_report` | `state.review` |
| `annotate` | add_arrow | `state.annotations` |
| emit | (no LLM) | `output/unity_guide.json` artifact |

The review verdict drives a bounded repair loop: blocker findings reopen
the flagged mission (`guide_review_max_iter`, default 3 — blender mode
uses `guide_review_max_iter_blender`, default 5, since each repair costs
a free-form build; score target `guide_review_score_target`, default 8).
Findings name review-schema missions (`doc|steps|parts|assembly|build|
annotate`); `_repair_owner` maps them onto the mode's order — in blender
mode `parts`/`assembly` collapse onto `build` (the only mission that can
move scene geometry), in cad mode `build` collapses onto `assembly`.

## Blender mode (`guide_mode`)

`POST /guide` accepts `mode: cad|blender|auto` (default `cad`; chat tool
`start_guide_builder` takes the same `mode` arg). `blender` replaces the
`parts`+`assembly` missions with one free-build mission `build`; `auto`
picks blender only when zero CAD assets were uploaded AND a Blender MCP
endpoint answers a ping — otherwise it falls back honestly to cad. The
resolved mode persists on `state.guide_mode` so resumes stay consistent.

| Mission | Tools | Output |
|---|---|---|
| `build` | blender_ping/exec/screenshot/register_part/sync_scene/render_step/save/export_scene + register_uploaded_part/match/alias/mark_context | `state.parts` (GLB, bbox) + `state.assembly` (scene read-back) |

The agent models/places/renders with full bpy freedom inside a running
Blender instance (BlenderMCP addon, JSON/TCP `:9876`). The semantic
contract is scene tags: every placed object carries
`obj['cadx_part']='part_id'`, `obj['cadx_step']=N` (0 = initial state),
`obj['cadx_premount']=True`. `blender_sync_scene` and the scene gate
(`blender_check.run_scene_gate`, fires at `review` instead of the cad
lint's `assembly` gate) read `matrix_world` back deterministically —
planlint itself is skipped in blender mode: its checks guard the CAD
sub_step/target/pin grammar for port resolution, which the blender
pipeline never consumes (scene gate + review stay the authorities).
instances, bboxes and the manifest come from measured scene facts, never
from LLM claims. Buried non-premounted instances block — the blocker
carries a least-escape `obj.location` delta (Blender axes) so the repair
session acts instead of re-measuring; joint interpenetration stays a
warning in `state.blender_manifest`.

Session-death safety nets for the free-form mission: `blender_exec`
counts calls since the last durable checkpoint and nudges (then forces)
`blender_sync_scene`, so tagged work reaches `state` even mid-session.
`blender_render_step` auto-frames an orthographic camera to the visible
tagged objects (`direction` arg selects the bearing) — no manual camera
setup. And a session dying after the durable state already satisfies the
contract (steps exist, parts registered, instances synced, every step
rendered) completes the mission from scene truth — the scene gate still
verifies physical quality before review.

**Units**: 1 Blender unit = 1 mm (`blender_exec` enforces
`scale_length=0.001` so glTF export + `normalize_mesh_units` compose
exactly). Blender is Z-up like CADX — `matrix_world` IS the mm transform.

**Networking/files**: worker→`host.docker.internal:9876` (compose sets
`extra_hosts`; env `CADX_BLENDER_HOST`/`CADX_BLENDER_PORT`). Blender
writes only host paths — `CADX_DATA_HOST_ROOT` is the host-side absolute
path of this repo's `./data` (compose defaults to `${PWD}/data`). Native
dev-worker: `localhost:9876`, root = `storage.data_root()`. Blender must
be open with the addon enabled (server auto-starts on .blend load);
`blender_ping` reports unreachability honestly and the mission may
`ask_user` to park until it is.

Extra artifacts in blender mode: `guide.blend`, `assembly_blender.glb`,
`blender_manifest.json` (tagged-object map, overlap report, renders).

## Key contracts

- **Dual text**: `GuideStep.text` = detailed internal description;
  `display_text` = short AR label (validated, ≤240 chars).
- **State**: `GuideState` JSON, atomic save under
  `storage.version_dir(project, guide_version)/guide/` — keyed by the
  guide Version id, never the (transient) job id. Resume reuses it via
  `job.resume_state["guide_dir"]`.
- **Placement**: the LLM never supplies transforms —
  `solve_*_preview` returns server-cached candidates; `place_instance`
  consumes a `candidate_id`. Port frames/gates ported from KIARA's
  p5 solver (`guide/solver.py`, mm/Z-up).
- **Collisions**: manifold3d boolean intersection (`guide/scene.py`) —
  no FCL dependency.
- **Heavy CAD work**: `extract_ports` + `render_assembly` run on the
  executor `cad-light` lane (`cad/guide_exec.py`), never in the worker.
- **Missing parts**: `generate_missing_part` → `spawn_generate` → normal
  sandboxed codegen job; parked child questions surface to the agent so
  it can relay them via `ask_user`.
- **Units/frame**: CADX mm/Z-up RH → Unity m/Y-up LH via
  `(x, z, y)·0.001` and `R_u = M·R·Mᵀ` (det −1 map) — golden-tested in
  `tests/guide/test_unity.py`.
- **Annotations**: per-step `arrows` carry `origin`/`direction` where
  `direction` always points AT the target (tip lands on the fixed port /
  part center; tail stands off by the anchored part's bbox extent along
  the axis — never a fixed offset that would bury inside large parts) +
  `label`/`label_position`. Per-step `texts` (`kind:"text"` annotations)
  carry `text` + `position` (freely settable via `add_text`, REST PATCH or
  the editor translate gizmo; unset → deterministic fallback above the
  step's placed bbox) + a constant `rotation` identity quaternion — every
  text field shares one upright orientation, only position varies.

## Runtime settings

`guide_pdf_dpi` (200), `guide_max_pages` (80), `guide_image_budget`
(24), `guide_review_score_target` (8.0), `guide_review_max_iter` (3).

## Files

```
backend/cadx/guide/            # the whole pipeline package
backend/cadx/cad/guide_exec.py # executor tasks (extract_ports, render_assembly)
backend/cadx/agent/prompts/guide_*.md  # mission system prompts
backend/cadx/spawn.py          # spawn_guide + generalized enqueue_resume
backend/cadx/agent/worker_jobs.py      # run_guide_job (agent queue)
tests/guide/                   # state/pdf/solver/unity/tools tests
```

Requires `CADX_AGENT_MAX_JOBS ≥ 2` when missing-part generation is
expected (the guide job waits on a child agent job; default is 8).

## Implementation deviations (vs masterplan v2)

- **`guide_status` chat tool dropped** — the generic `job_status` tool
  already returns step/progress/questions; a second status path would
  duplicate it.
- **`ArtifactKind.unity_guide` added** — the plan only listed
  job/version/asset enum values; the emit artifact needs its own kind
  (`register_artifacts` handles `guide:*` keys).
- **Step ids are `guide_<mission>`** — colons break i18next key lookup
  (`jobs:step.guide:doc`), so mission steps use underscores; labels for
  all 9 locales live in `jobs.json.step`.
- **`user_notes.jsonl`** — `attach_guide_note` appends to a sidecar the
  running missions pick up via `list_user_notes` (simpler than routing
  notes through the DB while a job is mid-flight).
- **Evalkit `guide` stage type deferred** — the suite harness only knows
  generate/edit/fem/… spawn routes; a guide stage needs PDF upload +
  answer-loop + new gates in `evalkit/http.py`. Tracked as future work.
- **`wait_job_terminal` reuse** — `generate_missing_part` waits on the
  child job via the existing spawn helper instead of duplicating poll
  logic; parked child questions forward to the agent for `ask_user`.
