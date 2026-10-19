# Prompts

Prompt sources live in `backend/cadx/agent/prompts/` and are loaded with
explicit UTF-8 encoding. They are assembled per step — the LLM never sees a
growing conversation, only the slices relevant to the current call.

## Files

| File | Used by | Content |
|------|---------|---------|
| `intake_system.md`   | INTAKE   | Multimodal spec extraction: dimension table with value+source (`stated`/`estimated`/`missing`), open questions, units discipline. |
| `intake_features.md` | INTAKE (re-ask) | Re-ask when the first spec came back incomplete: enumerate every named geometric feature with type/dims/position/count. |
| `intake_loads.md`    | INTAKE (loads) | Derive `spec.load_cases` (fixed regions + loads) when the user described forces/mounting/optimization but no load cases were produced. |
| `plan_system.md`     | PLAN     | Ordered construction steps, PARAMS table schema, standard-parts selection, manufacturing notes — plus `approach` (`build123d|mesh|hybrid`) used by `auto` mode. |
| `plan_lite_system.md` | PLAN (lite) | Plan-lite variant (`agent_plan_mode` lite/auto, loaded in `steps/plan.py`): one short non-tool call — ≤8 geometrically explicit steps + PARAMS table, `cad` approach fixed. |
| `mesh_orient_system.md` | ORIENT | Vision call on the 6-candidate contact sheet: pick `up`/`front` axes for the generated mesh (`MeshOrientation`). |
| `codegen_system.md`  | CODEGEN  | build123d coding rules for generated `model.py`. |
| `codegen_tools.md`   | CODEGEN (tools) | Interactive variant: look up parts/helpers, draft, `check_snippet`, finish with `submit_code`. |
| `codegen_build_system.md` | CODEGEN (staged build) | Feature-by-feature build session (`agent_staged_build`): one `add_block` per feature, sandbox `verify` after every block, `checkpoint`/`rollback` on failure. |
| `mesh_refine_system.md` | MESH_REFINE | Tool-session system prompt for mesh transforms: frame contract, re-grounding discipline, op vocabulary. |
| `optimize_system.md` | OPTIMIZE | Tool-session system prompt for the structural-optimization session (`analyze_fem`/`topopt_run` parameter search). |
| `image_hero.md`      | hero autogen | Text-to-image template producing a clean studio reference photo for mesh runs without input images. |
| `code_examples.md`   | CODEGEN  | Few-shot model.py example (PARAMS, counterbored plate, chamfer) — appended to the codegen system prompt. |
| `code_contract.md`   | CODEGEN/REPAIR/IMPROVE | Hard contract (rules only, no output format): `PARAMS` dict at top, part assigned to `result` (or `part`/`model`), mm units, coordinate frame (X=length, Y=width, Z=height, part on XY plane, centered). Hybrid addendum: `load_mesh_solid()`/`MESH_BBOX` are injected and only boolean ops may touch the mesh solid (no fillet/edge-select on mesh faces). |
| `code_contract_hybrid.md` | CODEGEN/REPAIR/IMPROVE (hybrid) | Appended when a mesh solid exists: `load_mesh_solid()`/`MESH_BBOX`, boolean-only edits, no fillet/edge-select on mesh faces, `cadlib.mesh_*` helpers. |
| `code_contract_responses.md` | +CODEGEN/REPAIR/IMPROVE (legacy) | Output-channel contract for non-tool calls: `CodeResponse` / `PatchResponse` JSON shapes. |
| `code_contract_responses_tools.md` | +CODEGEN (tool sessions) | Output-channel contract with tools: call `submit_code`. |
| `tools_protocol.md`   | all tool sessions | How tool calls work, line numbering (`NNN │`), session lifecycle, `done`/`submit_code` contract. |
| `repair_system.md`   | REPAIR   | Patch-first fixing: diagnosed error (category/failing line/hint) + spec/plan/measured analysis + failing code → `PatchResponse`; keep PARAMS names stable. |
| `repair_tools.md`    | REPAIR (tools) | Interactive repair: read/search the numbered code, surgical line/param edits, `check_code`, optional `verify`, then `done`. |
| `improve_system.md`  | IMPROVE  | Same contract as repair but the input is the best-so-far code + structured issue list (dims, measured features, critique, DfM); minimal edits that preserve all unrelated features. |
| `improve_tools.md`   | IMPROVE (tools) | Same issues input but interactive: fix listed issues with minimal edits, re-check before `done`. |
| `critique_system.md` | CRITIQUE | Vision judge: compare renders/contact sheet against inputs + measured facts; emit score, `feature_verdicts` per spec feature, concrete discrepancies with fix_hints, confidence. |
| `build123d_reference.md` | CODEGEN/REPAIR | Condensed API cheatsheet: workplane idioms, fillet/chamfer, Hole/CounterBoreHole, patterns, algebra mode pitfalls. |
| `bd_warehouse_reference.md` | PLAN/CODEGEN | Standard parts usage: screw/nut/washer/bearing class names, thread sizes, countersink angles, clearance/heat-set inserts. |
| `cadlib_reference.md`  | CODEGEN | Reviewed deterministic helpers (`cadx/cad/cadlib/`): hole grids, bosses, ribs, snap fits, knurl rings, dovetail slots, insert pockets, bearing seats — callable inside the sandbox. |
| `tools_info.md`        | info sessions (intake/plan) | Optional exploratory tools before the final JSON: `search_catalog`/`lookup_part`/`cadlib_lookup`, plus `search_library`/`library_part`/`attach_library_part` to reuse saved user parts. |
| `tools_inspect.md`     | CRITIQUE (tools) | Read-only inspection of the candidate code (`read_lines`/`search_code`) and catalog/library lookups for cross-checking dimensions. |
| `dfm_rules.md`       | PLAN/CODEGEN | Process rules (FDM walls/overhangs, CNC pockets/inside radii) so the model is born manufacturable. |
| `chat_system.md`     | chat turn  | Chat-agent system prompt (`chat_worker.py`): conversational CAD engineer with the full spawn/inspect tool surface — see [chat.md](chat.md). |

## Conventions baked into prompts

- **Coordinate frame**: X = length, Y = width, Z = height; part sits on the
  XY plane, XY-centered. Enables deterministic bbox comparison and swapped-
  axis detection.
- **`expected_bbox`**: the model must declare intended outer dimensions so
  ANALYZE can compare intent vs measured result.
- **PARAMS-first**: every dimension the user stated becomes a param with
  min/max; standard-part sizes (e.g. `"screw": "M4-0.7"`) stay string params.
- **Minimal diffs on edit**: parent code is shown; instructions forbid
  rewrites unless necessary and require PARAMS name stability.
- **Standard parts**: prompts point at bd_warehouse classes rather than
  asking the LLM to model threads by hand.

## Tuning

If generation quality drops (see [eval.md](eval.md)), prefer editing the
reference files first — they are the cheapest lever. Keep `code_contract.md`
consistent with `cad/allowlist.py`: anything not allowed there must not be
suggested by the prompts.

## See also

- [Agent loop](agent-loop.md) — which step loads which file
- [Parts library](parts-library.md) — catalog/cadlib/library prompt content
