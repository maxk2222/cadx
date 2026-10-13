# Web GUI

The React/Vite application exposes the generation, editing, analysis, and
library workflows through four pages: Projects, Project, Version, and Settings.
The UI is served by nginx at `:8080` in Docker mode or Vite at `:5173` in
native development mode.

## User workflows

### Projects and generation

- Create, rename, search, and delete projects.
- Upload and manage image/reference assets.
- Start a generation with a text prompt, selected assets, mode, per-run options,
and optional reusable library parts.
- Modes: `Auto`, `CAD`, `Mesh`, `Hybrid`, `TopOpt`. Mesh and Hybrid require an
  available mesh server and at least one image; TopOpt generates a base CAD
  design then optimizes it.
- Per-run options include manufacturing process, material, FEM analysis,
  topology optimization/target volume, and standard fasteners.

### Version detail

The Version page provides:

- Overview: 3D viewer, dimensions/volume/validity, renders, exports.
- Spec: extracted PartSpec and construction plan.
- Code: generated source and parameter controls.
- Quality: DfM warnings, measured issues, vision critique, BOM, and FEM report
  including stress/deformation artifacts when available.
- Files: all version artifacts, downloads, and saving successful/reviewed
  versions to the reusable part library.
- Side panel: job progress/log and history, prompt/image edits, mode/options,
  inherited or selected library parts, parameter-only reruns, and manual
  topology optimization for eligible versions.

Edits create child versions. Library selections are carried forward by default
and can be changed in the edit form. The API distinguishes an omitted selection
(inherit) from an explicit empty list (clear).

### Mesh editor (`/versions/<id>/edit`)

A dedicated page for visual, mesh-level editing of any finished version. The
tool panel builds a stack of validated `MeshOp` operations — the same
`OPS_REGISTRY` replay the executor uses for agent-side mesh refinement:

- **Orient / Rotate / Mirror / Scale** (uniform or per-axis)
- **Flatten bottom, Slice, Crop box, Clip height** — slice and clip planes and
  crop corners can be picked on the model surface *or* dragged with an
  on-screen transform gizmo (mm coordinates)
- **Smooth, Smooth-preserve, Decimate, Remesh, Symmetrize, Shell**
- **Drill holes** — vertical holes at picked XY points (dia / depth /
  countersink), optionally from measured anchors
- **Boolean add/cut** with a box/cylinder/sphere primitive — place by clicking
  the surface, then fine-tune with a move/rotate gizmo (ghost preview)
- **Keep largest / Drop small bodies / Repair**
- **Measure** — two-click surface distance (not recorded)

The op stack supports per-op removal, reorder, duplication, inline arg
editing, undo and reset; a failing op is highlighted in red using the
`failed_op` index + per-op report returned by the preview.

**Preview** runs the same synchronous baseline + ops replay the executor
uses, but in-process (no queue round-trip) with a cached baseline, and swaps
the viewer to a server-generated GLB (face count, bbox, watertight stats).
Stacks containing only rigid ops past the recorded prefix (orient, rotate,
mirror, scale) are evaluated instantly client-side — the model is
transformed locally, no request at all ("instant preview" badge). With
*auto* enabled, geometry-op stacks re-preview automatically ~0.7 s after the
last change. **Apply as new version** creates an immutable `mesh_edit` child
version: for mesh-mode parents the full baseline + ops replay runs on
`mesh/mesh_raw.glb`, for every other parent the ops run directly on the
finished STL (`ops_only` — no re-orient/re-scale). Drafts are autosaved on
the version (`options.ui_editor.ops`, merged server-side) so reopening the
editor restores the stack.

### Structural editor (FEM / TopOpt regions)

The structural editor paints boundary regions directly on the model surface
with a brush (per-region radius, mm) and supports the full `RegionSpec`
vocabulary:

- **Support regions** (blue) — fixed nodes per load case
- **Load regions** (red) — force vector `[fx,fy,fz]` N *or* pressure MPa per
  load (pressure follows the region's stored surface normal)
- **Keep regions** (green) — TopOpt design-preserve volumes

Region kinds: `points` (painted — points + radius + averaged surface
normal), `face`/`surface` (+axis), `area` (XY footprint), `cylinder`, `box`,
`hole` (measured anchor holes, 1-based index) and `all`. Geometric regions
offer click-to-place centers and an expandable detail row (center, diameter/
edge, margin, label). Painted regions render as vertex-coloured fills on the
model — the live approximation of the voxel-domain selection — and support
an **eraser** and per-stroke **undo**.

Three actions share the declared cases: **Check regions** (server-side voxel
resolution → hit counts, warnings, and a 3D overlay of the *resolved* cell
selection — `fem-preview` returns subsampled cell centres per case; legacy
2D PNGs via `render_pngs`), **FEM check** (standalone stress analysis
written to the same version, `needs_review` when safety factor < 1), and
**Optimize** (SIMP TopOpt child version with keep regions). Selections,
material and volume fraction autosave under `options.ui_editor` (merged —
the mesh-editor ops draft and structural state no longer clobber each
other).

### Settings and reusable part library

Settings includes LLM/provider configuration, health status, the standard-part
catalog, and the user library. The library supports STEP/STL upload, download,
rename/caption editing, deletion, and saving a version as either a STEP body or
parametric code template. Parts may be pinned on generation/edit or discovered
and attached by the agent's library tools. See `docs/parts-library.md`.

## Agent visibility

The UI consumes `VersionDetail` and artifact endpoints; fields are displayed in
context rather than duplicated across pages. Agent tool calls and their results
are captured in backend trace records for debugging. The current JobLog shows
phase-level progress, not individual tool-call payloads. Tool schemas are
step-scoped: edit tools are not available in critique, and critique library
tools are read-only.

## Verification checklist

Run the commands from the repository root:

```bash
cd frontend && npx tsc --noEmit && npm run build
cd ../backend && ../.venv/bin/uv run pytest tests/unit/test_api.py tests/unit/test_library.py -q
```

Manual smoke test:

1. Open Settings, upload a STEP/STL body, edit its name/caption, download it,
   and remove it.
2. Generate a CAD model with a library part selected; confirm its id is stored
   in version details. Save the resulting version as a template and confirm it
   appears in Settings and the project picker.
3. Edit a version, confirm its selected library parts are preselected, change
   the selection, and verify the child version reflects the new selection.
4. Enable FEM or TopOpt options, complete a run with explicit load cases, and
   inspect the Quality and Files tabs. Trigger manual topology optimization on
   a completed non-TopOpt version and confirm it opens a child version.
5. Confirm the job log continues polling while queued/running and stops polling
   after completion.

## Scope and limitations

The GUI is for a local, single-user deployment; it does not provide
authentication or multi-user library isolation. FEM/TopOpt values are
engineering estimates, not certification. The mesh editor operates on
tessellated geometry — it records replayable `MeshOp` transforms, it does not
patch parametric code; booleans are capped at 200k faces (decimate first).
Region painting selects near-surface voxel cells within the brush radius —
thin features below ~2 voxel pitches may not resolve reliably. See
`docs/fem-topopt.md`.