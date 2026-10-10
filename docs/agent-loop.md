# Agent loop

State machine in `backend/cadagent/agent/orchestrator.py`:

```
INTAKE → CLARIFY? → PLAN → resolve_mode →
  cad:    [CODEGEN → EXECUTE → REPAIR → ANALYZE → CRITIQUE] × K → FINALIZE
  mesh:   [MESH_GENERATE → ORIENT → PROCESS_MESH → ANALYZE → CRITIQUE] × K → FINALIZE
  hybrid: MESH_GENERATE → ORIENT → PROCESS_MESH(+solid) → cad loop with load_mesh_solid()
```

`resolve_mode` (in the orchestrator) maps the requested mode (`auto` via the
planner's `plan.approach`, or explicit `cad|mesh|hybrid`) to the effective
`resolved_mode`, recording the reason in the job log. Without images or a
reachable mesh server, mesh/hybrid fall back to CAD.

Each box is a step module in `agent/steps/` that takes `AgentCtx` and returns
a typed result. Every step writes to the `Trace` (persisted as `AgentTrace`
rows) and emits SSE progress events.

## Steps

| Step | LLM? | Purpose |
|------|------|---------|
| INTAKE  | yes (vision) | Extract `Spec`: name, summary, units, dimensions (x/y/z, each with value+source), open questions. Sees all input images. |
| CLARIFY | — | If spec has open questions and project mode is `ask`, the job ends `waiting_for_input` with questions. `assume` mode skips this and lets the LLM state assumptions. Resume continues at PLAN via `/jobs/{id}/resume`. |
| PLAN    | yes | Construction plan: ordered steps, PARAMS table (name/value/unit/min/max), standard-parts list (bd_warehouse), notes. |
| CODEGEN | yes | Produce `model.py` honoring the code contract + `expected_bbox`. |
| EXECUTE | no (cad queue) | Sandbox-run the code: shape, analysis, renders, DfM, exports. |
| REPAIR  | yes | On exec failure or validation error, feed error + code back; bounded by `agent_max_repairs`. |
| ANALYZE | no | Compare measured bbox vs spec dims with `dim_tolerance_pct/mm`; detect swapped axes. |
| CRITIQUE| yes (vision) | Contact sheet + input images → score 0–100, discrepancies, matches. |
| MESH_GENERATE | no (HTTP) | `MeshPort.generate` → mesh-server job → raw GLB into `iterations/<n>/`. Iterations vary the seed; on a `major` critique failure the hero image rotates. |
| ORIENT | yes (vision) | Once per version: 6 up-axis candidates are rendered in the executor and the vision model picks `MeshOrientation{up,front}`; heuristic/parent fallback otherwise. |
| PROCESS_MESH | no (cad queue) | `run_mesh_pipeline`: clean → decimate → orient → scale-to-spec → analysis → renders → DfM(mesh) → exports; `make_solid` additionally writes `_run/model.brep` (faceted solid) for hybrid. |
| FINALIZE| no | Pick best candidate `(no dim errors, −major issues, score)`, copy artifacts, write exports + `model.py`/`params.json`, set status. |

## Edit flow

`POST /versions/{id}/edit` creates a child version (same project, `parent_id`
set). The parent `ParentContext` carries its code, params, spec summary and
contact sheet so the LLM makes *minimal diffs* and keeps PARAMS names stable.
History is a version tree (`VersionTree` in the UI).

For `mesh` parents an edit *regenerates* the mesh (new seed/prompt/images) —
the parent's orientation is reused. For `hybrid` parents the parent's
`mesh/solid.brep` is reused as the base body unless `regenerate_mesh=true`.

## Params-only re-run

`POST /versions/{id}/params` re-executes the parent code with a `PARAMS`
override — no LLM at all. Result is a new child version. For `resolved_mode
== "mesh"` parents it re-runs the mesh pipeline with merged `MeshPostParams`
(scale, up, front, rotate_z, decimate, flatten_bottom, mirror_x) on the
parent's `mesh_raw.glb`. For hybrid parents the code re-runs against the
parent's `mesh/solid.brep`.

## Termination & status

- `succeeded` — score ≥ `agent_pass_score` and no unresolved dim mismatches.
- `needs_review` — executed fine but below score or with dim deviations.
- `failed` — repair budget exhausted / fatal error.
- `waiting_for_input` — clarification pending.
- `cancelled` — via `POST /jobs/{id}/cancel` (checked between steps).

Budgets: `agent_max_iterations` (outer loop), `agent_max_repairs` (per
iteration), `agent_max_wall_s`, `agent_max_tokens_per_job`.

## LLM calling discipline

- Every step = fresh `system` + `user` messages; no chat accumulation.
- `llm_json_mode`: `json_object` (default) → `json_schema` → `none`
  (plain prompt); client auto-retries without `response_format` on 400s.
- Structured parsing (`llm/structured.py`) extracts JSON even from fenced
  prose, validates against Pydantic schemas, and re-asks once on failure.
- `LLM_FAKE_SCRIPT` env swaps in `FakeLLM` replaying a JSON script —
  deterministic end-to-end runs in tests/compose without a provider.
