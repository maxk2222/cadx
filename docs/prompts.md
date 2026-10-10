# Prompts

Prompt sources live in `backend/cadagent/agent/prompts/` and are loaded with
explicit UTF-8 encoding. They are assembled per step — the LLM never sees a
growing conversation, only the slices relevant to the current call.

## Files

| File | Used by | Content |
|------|---------|---------|
| `intake_system.md`   | INTAKE   | Multimodal spec extraction: dimension table with value+source (`stated`/`estimated`/`missing`), open questions, units discipline. |
| `plan_system.md`     | PLAN     | Ordered construction steps, PARAMS table schema, standard-parts selection, manufacturing notes — plus `approach` (`build123d|mesh|hybrid`) used by `auto` mode. |
| `mesh_orient_system.md` | ORIENT | Vision call on the 6-candidate contact sheet: pick `up`/`front` axes for the generated mesh (`MeshOrientation`). |
| `codegen_system.md`  | CODEGEN  | build123d coding rules for generated `model.py`. |
| `code_contract.md`   | CODEGEN/REPAIR | Hard contract: `PARAMS` dict at top, part assigned to `result` (or `part`/`model`), mm units, coordinate frame (X=length, Y=width, Z=height, part on XY plane, centered), JSON response shape `{reasoning, params, expected_bbox, code}`. Hybrid addendum: `load_mesh_solid()`/`MESH_BBOX` are injected and only boolean ops may touch the mesh solid (no fillet/edge-select on mesh faces). |
| `repair_system.md`   | REPAIR   | Minimal-diff fixing using error text + failing code; keep PARAMS names stable. |
| `critique_system.md` | CRITIQUE | Vision judge: compare renders/contact sheet against inputs; emit score, discrepancies, matches, confidence. |
| `build123d_reference.md` | CODEGEN/REPAIR | Condensed API cheatsheet: workplane idioms, fillet/chamfer, Hole/CounterBoreHole, patterns, algebra mode pitfalls. |
| `bd_warehouse_reference.md` | PLAN/CODEGEN | Standard parts usage: screw/nut/washer/bearing class names, thread sizes, countersink angles, clearance/heat-set inserts. |
| `dfm_rules.md`       | PLAN/CODEGEN | Process rules (FDM walls/overhangs, CNC pockets/inside radii) so the model is born manufacturable. |

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

If generation quality drops (see `docs/eval.md`), prefer editing the
reference files first — they are the cheapest lever. Keep `code_contract.md`
consistent with `cad/allowlist.py`: anything not allowed there must not be
suggested by the prompts.
