# Hybrid mesh generation — implemented

This file used to describe v2 plans; mesh + hybrid generation shipped.
See `docs/mesh-generation.md` (architecture, modes, flow) and
`docs/mesh-server-api.md` (sidecar contract). Historical notes:

- The planner chooses `approach: build123d | mesh | hybrid` (`auto` mode).
- Mesh generation is a separate HTTP sidecar (`mesh_server/`) — not a queue
  task — because it needs GPU/MLX which Docker Desktop cannot provide.
- Hybrid is *CAD-hosted*: `process_mesh` also emits a faceted B-Rep
  (`mesh_to_solid`), injected into the sandbox as `load_mesh_solid()` so
  build123d booleans keep precise features exact.

## Remaining backlog

- **manifold3d boolean fallback**: mesh-hosted boolean ops when OCCT
  struggles with dense faceted solids (alternative to the CAD-hosted path).
- **Hollowing / shell thickness**: voxel-remesh + offset for print-ready
  hollow meshes (marked as stretch goal).
- **Text→image stage**: text-only prompts without reference images for the
  mesh path (currently falls back to CAD by design).
- **Textures/PBR**: the generation pipeline is shape-only; texture bake is
  out of scope for manufacturing output.
- **Mesh-native edits**: in-painting / partial regeneration instead of full
  regenerate + CAD features.

## Ideas backlog (`docs/ideas.md` candidates)

- bd_warehouse extensions: DIN 912 length tables, more insert vendors,
  imperial fasteners.
- Assemblies: multi-body versions with joints/constraints.
- Slicer hand-off: OrcaSlicer CLI profile push, printability preview.
- KiCad/PCB outline import → automatic enclosure shells.
- G-code / CAM preview for CNC process.
- Photo → dimension auto-calibration (ArUco/credit-card scale reference).
