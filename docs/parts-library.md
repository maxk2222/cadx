# Parts library

![Library page — reusable STEP/STL bodies, code templates and the bd_warehouse catalog](screenshots/library.png)

Three complementary layers make standard and reusable geometry available to
the agent — the goal is that generated code calls verified building blocks
instead of re-inventing threads, holes and fasteners on every run.

## 1. bd_warehouse catalog (`cad/catalog.py`)

The full bd_warehouse 0.4.x catalog is introspected once at import and cached:
~15 categories, 100+ classes — screws/nuts/washers, clearance/tap/threaded/
insert holes, ISO + trapezoidal threads, bearings + press-fit seats, spur/
helical/rack/worm gears, O-rings, retaining rings + grooves, shaft keys,
bushings, sprockets, pipes, flanges, OpenBuilds profiles.

- `build_catalog()` → `{category: {class_name: entry}}` with `kind`
  (`normed`|`parametric`), valid `types`→`sizes` tables or constructor params.
- `search(query)` — name/type/size search for tools + UI.
- `lookup_part(class, type, size, length)` — instantiates the class and returns
  real published dims (clearance/tap hole diameters, head dims, …).
- `validate_part_ref(...)` — checks class/type/size strings.
- `catalog_excerpt(names)` / `class_listing()` — prompt-ready bounded excerpts.
- `nominal_hole_diameter(spec)` — feeds `cad/features.py` with real norm tables
  instead of hardcoded dicts.

**Agent visibility**: PLAN gets `catalog_listing_block()` (all class names);
CODEGEN/REPAIR/IMPROVE get `catalog_block()` — a bounded excerpt restricted to
the classes named in `plan.standard_parts` and `spec.features[].standard`, so
context stays small. In tool sessions the model calls `search_catalog` /
`lookup_part` itself.

**Static validation**: `cad/partspec.py` (`check_standard_part_calls`) parses
generated code and rejects fabricated class names (e.g. `HexScrew`), unknown
type/size strings and wrong call signatures *before* the executor runs —
errors flow into the repair loop like contract violations.

API: `GET /api/catalog`, `GET /api/catalog/search?q=`, `GET /api/catalog/lookup`.

## 2. cadlib (`cad/cadlib/`)

Reviewed deterministic helper functions for recurring geometry — importable
in generated code (`import cadlib` is allowlisted in the sandbox; the package
ships inside the executor image):

- cutouts: `hole_grid` (clearance/tap/countersink/counterbore), `keyhole_slot`,
  `t_slot`, context cutters
- shapes: `boss`, `rib`, `mounting_ear`, `snap_fit_cantilever`, `living_hinge`,
  `knurl_ring`, `filleted_box`, `l_bracket`, `dovetail_slot`, `text_label`
- standard features: `insert_pocket` (heat-set inserts), `bearing_seat`
  (+circlip groove)

Documented for the LLM via `prompts/cadlib_reference.md` (in the codegen
system prompt) and the `cadlib_lookup` tool. cadlib internals are constrained
by the same runtime import guard as generated code.

## 3. User library (`LibraryPart` + `data/library/`)

Reusable parts the user owns — two kinds:

| kind | content | agent usage |
|------|---------|-------------|
| `step`/`stl` | uploaded or saved body file | `load_reference("<part_id>")` in generated code |
| `template` | parametric build123d code + PARAMS | shown via `library_block`/tool result to adapt |

Sources: upload via API/UI, or "Save to library" on a finished version
(STEP export or the parametric code as template).

**Agent visibility**: the user can pin parts per version (UI picker →
`version.library_part_ids` → `_load_library()` injects them as reference
assets + templates at job start). Additionally the agent can self-serve via
tool session:

- `search_library(query, kind?)` — matches name/caption/filename
- `library_part(part_id)` — details; templates return their code
- `attach_library_part(part_id)` — file parts become `load_reference` bodies
  (available from the next EXECUTE onward); templates land in
  `ctx.library_templates`. Attached ids merge into
  `version.library_part_ids` on persist.

API: `POST/GET/DELETE /api/library`, `GET /api/library/{id}/file`,
`PATCH /api/library/{id}`, `POST /api/library/from-version/{version_id}`.

## BOM (`bom.json` + `Version.bom`)

`runner_entry.py` scans the executed namespace for `bd_warehouse` objects and
groups them by `(class, type, size)` — using the semantic `*_size` attributes
(`screw_size`, `bearing_size`, `insert_size`, …), since `.size` is an
OCP-internal number on most classes. Counts reflect `hole_locations` where
the part drove hole operations. The BOM lands in `Version.bom` + `bom.json`
and renders in the quality tab.

## See also

- [Prompts](prompts.md) — how catalog/cadlib content reaches the model
- [Agent loop](agent-loop.md) — the `search_library`/`attach_library_part` tools
- [HTTP API](api.md) — `/catalog` and `/library` endpoints
