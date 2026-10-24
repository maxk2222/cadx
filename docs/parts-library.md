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

**Generic constructor validation**: `catalog.constructor_kwargs(class_name,
fastener_type=, size=, length=, parameters=, simple=, default_length=)` builds
a validated kwarg dict for any registered class — unknown parameters are
rejected, `Literal[...]` choices are enforced *with their real type* (a
`Literal[150, 300]` validates and injects the int), missing required params
are hard errors, and norm type/size pairs must exist in the size table.
Known mapping rules: `size` maps onto the class's `size` param, or onto the
sole remaining required numeric param (e.g. `ShaftKey.size` →
`shaft_diameter`); `fastener_type` maps onto `fastener_type`/`bearing_type`/
`o_ring_type`/`bottle_type`; `material` choices are intersected with what the
installed `bd_materials` registry actually resolves (Pipe's `iron`/`pvc`/
`steel` literals are unbuildable upstream and are filtered out).

API: `GET /api/catalog`, `GET /api/catalog/search?q=`, `GET /api/catalog/lookup`.

## 1a. Deterministic standard-part recipes (`cad/standard_parts.py`)

For clearly-standard parts the agent can skip free-form codegen entirely:
`PartSpec.standard_parts` is a list of `StandardPartSpec` entries
(`class_name` + `fastener_type`/`size` + `length_mm`/`parameters` + quantity,
placement, axis, `threaded`, `covers_features`). Two paths consume it:

- **`create_standard_part` tool** — validates the spec against the catalog
  and returns the measured norm data plus a sandbox-ready code recipe.
- **codegen step 1 shortcut** — when `is_standard_only(spec)` holds (every
  declared feature is covered by a standard part), the recipe is emitted
  directly with zero LLM calls.

`build_standard_part_code(parts)` emits a parametric script: every value
lands in `PARAMS` (parametric like any other candidate), `CADX_FEATURES`
carries the structured feature records (thread pitch/length/diameter/axis/
position, catalog geometry adjustments), `CADX_BOM` declares the bill of
materials — the runner prefers these declarations over namespace scanning.
Classes whose required params take host objects (`fastener`, `bearing`,
`ring`, `key` — hole/seat/groove ops) are reported as *context helpers, not
standalone parts*.

### Thread level-of-detail: `CADX_DETAIL_MODE`

The runner injects `CADX_DETAIL_MODE` (`"preview"` | `"export"`) — an
executor-controlled flag, never an LLM boolean:

- **preview** (`verify`/`render_model`/iteration runs): threadable classes
  build with `simple=True` (fast core body); thread classes without a
  `simple` switch get a plain cylinder proxy. The `CADX_FEATURES` thread
  records feed a helix-line overlay in the render views plus a `thread_detail`
  close-up camera — the thread is clearly *visible* as a preview cue without
  paying for real helix geometry. `preview_thread_count` reports the marker
  count.
- **export** (final selected candidate): the orchestrator re-executes the
  winning code once with `detail_mode="export"`, so fasteners/thread classes
  build `simple=False` — the real ISO helix. Multi-solid catalog parts are
  fused per part into single solids — chunk-wise: a real thread arrives as
  one solid per helix turn (≈76 on an M10×120 full thread), and a single
  N-way boolean over all of them runs O(n²) section work that crashed OCC in
  the executor (exit -11); `_cadx_fuse_solids` fuses contiguous groups of ≤8
  instead, which also preserves the helical BSPLINE faces the runner's
  thread check counts on (`glue=True` collapses them — never use it here).
  `thread_count` must then match the declared expected count or the run
  fails — a smooth-shaft preview can never ship as the final artifact. All
  four formats (STEP/STL/3MF/GLB) are produced from that real geometry.

`CADX_FEATURES` coordinates are emitted in the **final part frame** — the
placement transform (axis rotation + XY seating lift) is baked into
`position_mm`/`start`, so the helix overlay lands on the rendered body even
when the recipe flips a head-datum part upside down (`start` then measures
axis distance from the head datum, not the catalog's local z).

`verify_spec` has a dedicated `thread_features` check: preview requires a
thread marker or real thread, export requires the real helix — FAIL
otherwise. Timing reference (M10×150 ISO 10642, cold): preview ≈ 2.2 s,
export ≈ 7.3 s sandbox build.

### Applied-feature records

`cadlib.finish_edges(shape, edges, kind="fillet"|"chamfer", size=...)` is the
reporting wrapper around `safe_fillet_shape`/`safe_chamfer_shape`: it returns
`(result, {"requested", "applied"})` and **raises** when the radius collapsed
to 0 instead of silently succeeding — generated code appends the record to
`CADX_FEATURES` so the run reports what actually landed.

Known limits: context-helper classes (holes/seats/grooves) can't be emitted
as standalone parts; `Pipe` needs `length_mm` (its `path` Edge is generated
by the recipe); `CounterSunkScrew` gets a corrected norm-table head profile
(the catalog class's filleted head deviates from the table dims — the
adjustment is reported as a `catalog_geometry_adjustment` feature record).

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

A declared `CADX_BOM` in generated code (emitted by the standard-part recipe
builder) is authoritative — explicit part instances, not guesses. Otherwise
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
