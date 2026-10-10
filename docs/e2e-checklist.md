# E2E checklist (manual, browser)

Run against `make up` (full compose stack) at http://localhost:8080.

## Setup

- [ ] `docker compose up -d` → all 6 services healthy (`api`, `worker`,
      `cad-executor`, `postgres`, `redis`, `frontend`)
- [ ] Settings page → enter `LLM_BASE_URL` + `LLM_API_KEY` + models →
      "Test connection" returns OK

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

## Mesh & hybrid modes

Prereq: mesh server running (native: `make mesh-server`, or compose
`--profile gpu`, or `MESH_FAKE=1` for the deterministic fake). Settings →
"Mesh generator" shows device + available models.

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

## Robustness

- [ ] Switch LLM base URL to Ollama (`http://host.docker.internal:11434/v1`)
      with a model that lacks JSON mode → graceful fallback / clear error
- [ ] Prompt an impossible part → job ends `failed`/`needs_review` with a
      useful error, not a crash
- [ ] Upload >`AGENT_MAX_INPUT_IMAGES` images → capped cleanly

## Teardown

- [ ] `docker compose down` clean; `data/` retains projects on next `up`
