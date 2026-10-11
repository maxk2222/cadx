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
engineering estimates, not certification. The manual optimization dialog
currently represents simple face supports and axial face forces; richer region
loads are available through the API/spec contract. See `docs/fem-topopt.md`.