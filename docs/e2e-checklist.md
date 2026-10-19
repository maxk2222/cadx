# E2E checklist (manual, browser)

Run against `make up` (full compose stack) at http://localhost:8080.

## Setup

- [ ] `docker compose up -d` → all 6 services healthy (`api`, `worker`,
      `cad-executor`, `postgres`, `redis`, `frontend`)
- [ ] Settings page → enter provider base URL + API key + models →
      "Test provider" returns OK

## Generate

- [ ] Create project (name, units mm, process FDM)
- [ ] Upload 2 photos + 1 drawing, add captions, see thumbnails
- [ ] Prompt with dimensions, e.g. "120x80x10mm plate, 4× M4 countersunk
      holes, 5mm corner radii" → Generate
- [ ] Job log streams progress live (SSE): intake → plan → codegen →
      execute → analyze → critique
- [ ] Version appears in tree with thumbnail and status badge
- [ ] Contact sheet + 7 renders visible in gallery
- [ ] Critique score and dimension table shown; DfM panel lists warnings

## Verify output

- [ ] Download STEP → open in FreeCAD: 4.5mm holes with 90° countersink,
      overall dims correct
- [ ] Download 3MF → opens in slicer (OrcaSlicer/Bambu)
- [ ] GLB renders in the built-in 3D viewer (orbit works)
- [ ] `model.py` shown in code panel; `params.json` values match prompt
- [ ] DXF/SVG present for prismatic parts

## Edit & params

- [ ] Edit version: "add 2× M3 heat-set insert holes" (+ optional image) →
      child version created, parent code only minimally changed, PARAMS
      names stable
- [ ] Params panel: change `screw` M4-0.7 → M5-0.8 → re-run → new child
      version with updated holes, no LLM call (fast)
- [ ] Delete a leaf version → removed from tree and disk
- [ ] Cancel a running job → status `cancelled`, partial iteration kept

## Version page & Jobs UI

- [ ] Version overview → **compare** toggle + version picker → side-by-side
      `a`/`b` viewers and a ComparePanel: params diff table, volume/bbox and
      added/removed material deltas; skipped OCC booleans show a
      `boolean diff` warning with the partial result
- [ ] Viewer toolbar → measure tool → click two surface points → distance
      in mm; `Esc` exits
- [ ] Side panel → **Edit feature** on a CAD version with a through-hole →
      resize the diameter → `kind=edit` child version re-cuts the hole
- [ ] Jobs page → status filter chips + job-type dropdown narrow the list;
      a failed row's **retry** re-runs the job; *show more* paginates older
      history
- [ ] A `waiting_for_input` job shows its first pending question inline on
      the Jobs page; answering on the version page resumes it

## Mesh & hybrid modes

Prereq: mesh server running — easiest via `./scripts/start.sh` (auto-detects
the GPU tier and installs+starts the native mesh-server); alternatives:
`make mesh-server`, compose `--profile gpu`, or `start.sh --fake` for the
deterministic fake. Settings → "Mesh & Image" tab shows the device; "Local
models" lists/downloads/removes the sidecar models.

- [ ] Generate with mode **mesh** + a reference photo → version shows a mesh
      badge, STL/3MF/GLB exports download, no `model.py` code panel,
      `mesh.json` provenance visible (model, seed, orientation)
- [ ] Generate with mode **hybrid** → code panel contains
      `load_mesh_solid()` usage; STEP export downloads and is labelled
      *tessellated* in the UI
- [ ] Mode **auto** on an organic prompt → planner picks mesh/hybrid
      (`resolved_mode` badge + job-log note shows the decision reason)
- [ ] Mode **mesh** with no images → falls back to CAD with a log note
- [ ] Mesh version → Params panel: change `scale`/`front` → child version
      without any LLM call; `mesh_raw` reused from the parent
- [ ] Hybrid version → Edit with *regenerate mesh* unchecked → parent's
      `solid.brep` reused; checked → new mesh generation
- [ ] Stop the mesh server mid-generation → job fails cleanly / falls back;
      version is `failed`, not stuck `running`

## User library, BOM & agent tools

- [ ] On the Library page, upload STEP/STL, rename/caption, download, preview,
      and delete a test entry.
- [ ] Generate with a library body/template selected; inspect Version detail
      and confirm the selected library IDs are preserved. Save a finished
      version to the library as both a STEP body and (when code exists) template.
- [ ] Edit a version: inherited library parts are selected; toggle one off/on
      and confirm the child reflects exactly the new selection. Confirm an
      explicit empty selection clears inherited parts.
- [ ] Inspect Quality: a `bd_warehouse` screw/insert used by the model appears
      in the BOM with class/type/size/count.
- [ ] With `llm_tools=auto`, inspect the trace/job activity for per-step tool
      calls. Force one model role to reject tools and confirm only that role
      falls back; `llm_tools=off` still runs legacy behavior.

## FEM & topology optimization

FEM/TopOpt defaults live under Settings → "FEM & topology optimization". Use
explicit supports and loads; the numerical results are engineering estimates,
not certification.

- [ ] Generate a CAD part with `Options → FEM check`; state fixtures and loads
      in the note/prompt and confirm the stored version has a report artifact.
- [ ] Inspect the Quality tab: each load case shows max von Mises stress,
      displacement, safety factor, solver status, and any warnings.
- [ ] Manually optimize a completed version with `Topology-optimize this part`;
      enter fixed/load faces, force magnitude/direction, material and target
      volume fraction; confirm a child version and job progress are created.
- [ ] Confirm the optimized child is `topopt`, exposes FEM results, and has
      density/history and mesh exports in Files; compare volume against the
      target while remembering boundary regularization can add a small amount.
- [ ] Generate in `TopOpt` mode with explicit load/fixture requirements;
      confirm the agent creates a base version then runs SIMP and records its
      worst safety factor.
- [ ] Omit load cases for TopOpt and confirm the job reports a clear skip/error
      rather than presenting an unverified optimized result.
- [ ] Structural editor → Optimize: switch **Solver** to `mma`, set
      *MinMax* with 2+ load cases, a displacement cap, and/or a σ cap →
      confirm the optimized child's `topopt/history.json` records
      `solver: "mma"` and the constraint diagnostics.
- [ ] Enable *self-supporting* + a **Print dir** and confirm the optimized
      mesh has no unsupported overhangs in that direction.
- [ ] **Feature finishing**: optimize a part with holes/anchors, then
      "Finish features" (or `auto_refine_features`) → `refine` child version
      re-cuts exact bores/threads; the report lists
      `requested_dia_mm`/`measured_dia_mm`/`dia_deviation_mm` per feature.
      Toggle *auto planar restore* / *helical threads* to exercise the flags.

## Executor-image tests (D3)

`tests/cad` must also pass inside the executor image. The `executor-test`
Dockerfile target adds dev deps on top of `cad-executor`:

```bash
docker build --target executor-test -t cadx-executor-test backend
docker run --rm -v "$PWD/backend/tests:/app/tests:ro" -v "$PWD/data:/data" \
  cadx-executor-test python -m pytest /app/tests/cad -x -q
```

The sandboxed CAD contract tests (build123d, manifold3d, trimesh, nlopt,
scikit-image) must all pass there — the executor image is what generated
code and FEM/TopOpt actually run in.

## Eval benchmarks

`cadx eval` runs scripted FakeLLM benchmarks through the real agent loop.
TopOpt coverage: `topopt_mma_minmax_scripted` (solver=mma, case_objective=
minmax, displacement cap, penal_init, min_gap_mm) and
`topopt_filters_scripted` (heaviside, self_supporting, multires,
auto_plane_restore). `--repeat N` for variance control.

## Robustness

- [ ] Switch LLM base URL to Ollama (`http://host.docker.internal:11434/v1`)
      with a model that lacks JSON mode → graceful fallback / clear error
- [ ] Prompt an impossible part → job ends `failed`/`needs_review` with a
      useful error, not a crash
- [ ] Upload more than `agent_max_input_images` images → capped cleanly

## Teardown

- [ ] `docker compose down` clean; `data/` retains projects on next `up`

## See also

- [Getting started](getting-started.md) — stack setup for this checklist
- [Troubleshooting](troubleshooting.md) — when a step fails
