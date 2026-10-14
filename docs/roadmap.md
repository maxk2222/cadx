# Roadmap & backlog

Current capabilities and what is intentionally not built yet. Shipped features
have their own docs — this file is only for orientation and ideas.

## Shipped

- Generation modes `auto` / `cad` / `mesh` / `hybrid` / `topopt` with strategy
  chains and deterministic fallbacks — [mesh-generation.md](mesh-generation.md)
- Agent tool sessions (native OpenAI function calling) with budgeted
  edit/check/verify loops — [agent-loop.md](agent-loop.md)
- Voxel FEM + SIMP topology optimization with agent-driven parameter search —
  [fem-topopt.md](fem-topopt.md)
- Local text-to-image and image editing (FLUX/SDXL/Qwen adapters) —
  [image-generation.md](image-generation.md)
- bd_warehouse catalog + `cadlib` helpers + user part library + BOM —
  [parts-library.md](parts-library.md)
- Mesh editor and structural editor in the WebGUI — [webgui-ux.md](webgui-ux.md)
- Mesh refinement sessions replayed as deterministic `MeshOp` lists
- Benchmark/eval harness — [eval.md](eval.md)

## Backlog

- **manifold3d boolean fallback** — mesh-hosted booleans when OCCT struggles
  with dense faceted solids (alternative to the CAD-hosted hybrid path).
- **Assemblies** — multi-body versions with named parts, joints/constraints
  and interference checks.
- **Mesh-native edits** — in-painting / partial regeneration instead of full
  regenerate + CAD features.
- **Textures/PBR** — the pipeline is shape-only; texture baking is out of
  scope for manufacturing output.
- **Photo→dimension auto-calibration** — ArUco / credit-card scale reference.
- **Slicer hand-off** — OrcaSlicer CLI profile push, printability preview.
- **KiCad/PCB outline import** → automatic enclosure shells.
- **G-code / CAM preview** for the CNC process.

## Explicit non-goals

- Multi-user operation / auth — CADX is a local single-user system.
- Certification-grade structural analysis — FEM/TopOpt is design feedback
  only; see the limits in [fem-topopt.md](fem-topopt.md).
