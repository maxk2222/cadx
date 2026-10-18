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
- manifold3d boolean fallback — mesh-hosted rescue booleans behind OCC,
  wired via `cadlib.ops.set_boolean_backend` (`cad/mesh_boolean.py`)
- Chat agent with the full spawn/inspect tool surface — [chat.md](chat.md)
- MCP server exposing the deterministic tool surface (stdio + `/mcp`,
  LLM-free) — [mcp-server.md](mcp-server.md)
- Organic restyle (gyroid/trabecular/voronoi/branches) — [organic.md](organic.md)
- Version compare endpoint + side-by-side UI
  (`GET /versions/{a}/compare/{b}`) — [webgui-ux.md](webgui-ux.md)
- Job finish/restart controls (Jobs page + `/api/jobs/{id}/finish|restart`)
- Structural editor pick tools (face/edge/hole) — [webgui-ux.md](webgui-ux.md)
- Benchmark/eval harness — [eval.md](eval.md)

## Backlog

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
