"""Tool schemas and the session executor for LLM tool-calling sessions.

Each step gets a `ToolSession` holding the mutable state the tools act on
(a code buffer for edit sessions, a read-only listing for inspection, a
`MeshWorkbench` for mesh sessions, or nothing for info-only steps).
`session.execute(name, args)` dispatches a tool call and returns a
JSON-serialisable dict — errors are results, never exceptions, so the model
can self-correct.

Tool sets per step:
  edit    repair/improve — read/search/replace/insert/set_param + info + library + check + verify/render + done
  codegen fresh code — info + library + check_snippet + submit_code
  build   staged codegen — read/edit buffer + add_block/checkpoint/rollback + verify/render + done
  info    intake/plan — catalog/cadlib lookups + library search/attach (final answer = content JSON)
  inspect critique — read-only code view + catalog/library lookups (no attach)
  mesh    mesh_refine — mesh inspect/preview + transform ops + undo/reset + done/regenerate
  optimize structural — analyze_fem + topopt_run probes + done(run_index)
`view_image` (re-view an uploaded reference image) is offered in every kind.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import re
import tokenize
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from cadx import storage
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx, InputAsset
from cadx.cad import catalog, codeedits
from cadx.cad.allowlist import check_code
from cadx.cad.partspec import check_standard_part_calls
from cadx.logging import get_logger

log = get_logger(__name__)

VerifyFn = Callable[[str], Awaitable[dict[str, Any]]]
StructFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

_DONE_TOOLS = ("done", "submit_code", "regenerate")


def _fn(name: str, desc: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _s(desc: str) -> dict[str, Any]:
    return {"type": "string", "description": desc}


def _i(desc: str) -> dict[str, Any]:
    return {"type": "integer", "description": desc}


# --------------------------------------------------------------------------- #
# tool schemas
# --------------------------------------------------------------------------- #

CODE_READ_TOOLS = [
    _fn(
        "read_lines",
        "Read a line range of the current code file (1-based, inclusive). "
        "Returns numbered lines. Keep ranges small and targeted.",
        {"start": _i("first line, >= 1"), "end": _i("last line, inclusive")},
        ["start", "end"],
    ),
    _fn(
        "search_code",
        "Find lines matching a literal string (or regex when regex=true). "
        "Returns matching line numbers with context — use it to locate the "
        "code you need before reading a range.",
        {
            "pattern": _s("literal text or regex to find"),
            "context": _i("context lines around each hit (default 2)"),
            "regex": {"type": "boolean", "description": "treat pattern as regex"},
        },
        ["pattern"],
    ),
]

CODE_EDIT_TOOLS = [
    _fn(
        "replace_lines",
        "Replace lines start..end (1-based inclusive) of the current code with "
        "new text. Applies immediately; line numbers shift afterwards.",
        {
            "start": _i("first line to replace"),
            "end": _i("last line to replace, inclusive"),
            "code": _s("replacement text"),
        },
        ["start", "end", "code"],
    ),
    _fn(
        "insert_after",
        "Insert text after the given 1-based line (0 = top of file).",
        {"line": _i("line number"), "code": _s("text to insert")},
        ["line", "code"],
    ),
    _fn(
        "insert_before",
        "Insert text before the given 1-based line.",
        {"line": _i("line number"), "code": _s("text to insert")},
        ["line", "code"],
    ),
    _fn(
        "find_replace",
        "Replace an exact verbatim code block — `find` must occur exactly "
        "ONCE in the file (include enough surrounding lines to be unique).",
        {"find": _s("verbatim text to find"), "replace": _s("replacement text")},
        ["find", "replace"],
    ),
    _fn(
        "set_param",
        "Set one entry of the top-level PARAMS dict (name -> new literal "
        "value). Prefer this over editing the line manually for pure value "
        "changes.",
        {
            "name": _s("PARAMS key"),
            "value": {
                "type": ["string", "number", "boolean"],
                "description": "new literal value",
            },
        },
        ["name", "value"],
    ),
]

INFO_TOOLS = [
    _fn(
        "search_catalog",
        "Search the bd_warehouse standard-parts catalog by name, type or size "
        "string (e.g. 'countersunk screw', 'M4', 'iso4762'). Returns matching "
        "classes with valid type/size tables.",
        {"query": _s("search text")},
        ["query"],
    ),
    _fn(
        "lookup_part",
        "Instantiate a catalog part and get its real published dimensions "
        "(clearance/tap hole diameters, head dims, …). Use exact catalog "
        "type/size strings — invalid values return an error listing valid ones.",
        {
            "class_name": _s("catalog class, e.g. CounterSunkScrew"),
            "fastener_type": _s("norm type string, e.g. iso10642"),
            "size": _s("size string, e.g. M4-0.7"),
            "length": {"type": "number", "description": "part length in mm"},
        },
        ["class_name"],
    ),
    _fn(
        "cadlib_lookup",
        "Show the signature and docstring of a cadlib helper function (e.g. "
        "hole_grid, snap_fit_cantilever) — prefer these helpers over "
        "hand-rolled geometry.",
        {"name": _s("cadlib function name")},
        ["name"],
    ),
    _fn(
        "quickref",
        "Runtime-introspected API signature digest of the INSTALLED CAD "
        "packages — the ground truth for method names, parameter order and "
        "valid enum/size strings (beats memory and the prose reference docs "
        "when they disagree). Topics: 'build123d' (core ops + enums), "
        "'cadlib' (all reviewed helpers), 'bd_warehouse' (standard parts with "
        "type/size tables), or a dotted path like 'build123d.extrude', "
        "'bd_warehouse.fastener.CounterSunkScrew', 'cadlib.hole_grid'.",
        {"topic": _s("topic name or dotted path to an object")},
        ["topic"],
    ),
]

IMAGE_TOOLS = [
    _fn(
        "view_image",
        "Re-view an uploaded reference image at full detail. The context "
        "labels them 'Image N (asset <id>)' — pass that asset_id. Use when "
        "the initial image context was capped or you need a closer look at "
        "one specific view.",
        {"asset_id": _s("asset id from the image labels")},
        ["asset_id"],
    ),
]

CONCEPT_IMAGE_TOOLS = [
    _fn(
        "generate_concept_image",
        "Generate a design-concept image with the LOCAL image model "
        "(FLUX/SDXL/Qwen on the mesh server) and view it. Plain call = "
        "text-to-image for ideating shapes, proportions or style. Pass "
        "reference_asset_id (an existing image/generated_image asset id) for "
        "img2img variations or instruction edits ('make it matte black', "
        "'add cooling fins') — 'strength' controls how much of the reference "
        "is kept (low = close, high = strong change). With mark_sketch the "
        "result is tagged as a sketch so *-scribble mesh adapters use it as "
        "the conditioning drawing. The image is stored as a project asset "
        "(kind=generated_image) and may serve as a reference for "
        "image-to-3D generation. Limited calls per session.",
        {
            "prompt": _s("what to render — subject, view, style; or the edit instruction"),
            "count": _i("number of variants, 1 or 2 (default 1)"),
            "seed": _i("optional seed for reproducibility"),
            "reference_asset_id": _s("optional id of an existing image asset to vary or edit"),
            "strength": _s("img2img denoise strength 0..1 (only with reference_asset_id)"),
            "mark_sketch": _s(
                "tag result as a sketch/scribble for sketch-conditioned mesh adapters"
            ),
        },
        ["prompt"],
    ),
]

LIBRARY_TOOLS = [
    _fn(
        "search_library",
        "Search the user's reusable part library — uploaded STEP/STL bodies "
        "and saved parametric code templates — by name, caption or filename. "
        "Returns matching parts with their ids.",
        {
            "query": _s("search text"),
            "kind": _s("optional filter: step | stl | template"),
        },
        ["query"],
    ),
    _fn(
        "library_part",
        "Show one library part in detail: templates return their complete "
        "parametric code + PARAMS, file parts their filename and how to load "
        "them via load_reference().",
        {"part_id": _s("library part id from search_library")},
        ["part_id"],
    ),
    _fn(
        "attach_library_part",
        "Select a library part for this generation: file parts (STEP/STL) "
        "become reference bodies loadable via load_reference(part_id) in the "
        "generated code; template parts attach their parametric code for you "
        "to adapt.",
        {"part_id": _s("library part id from search_library")},
        ["part_id"],
    ),
]

# inspect sessions may search/inspect but never attach
LIBRARY_INFO_TOOLS = LIBRARY_TOOLS[:2]

CHECK_TOOLS = [
    _fn(
        "check_code",
        "Run the static contract check on the CURRENT code (imports, PARAMS "
        "dict, result assignment, standard-part call validity). Returns errors "
        "without executing.",
        {},
        [],
    ),
]

CHECK_SNIPPET_TOOLS = [
    _fn(
        "check_snippet",
        "Run the static contract check on a code string BEFORE submitting it.",
        {"code": _s("complete code file to check")},
        ["code"],
    ),
]

VERIFY_TOOLS = [
    _fn(
        "verify",
        "Execute the CURRENT code in the CAD sandbox and return the outcome: "
        "error diagnosis on failure, measured geometry (bbox, holes, "
        "fillets…) on success. Limited calls — use once, after your edits, "
        "before done.",
        {},
        [],
    ),
]

RENDER_TOOLS = [
    _fn(
        "render_model",
        "Execute the CURRENT code and LOOK at the result — returns the "
        "measured analysis plus the labeled contact-sheet render as an "
        "image. Shares the sandbox-execution budget with verify; use it "
        "when you need to SEE the geometry, not just its numbers.",
        {},
        [],
    ),
]

# verified parametric feature edits (M9): offered in "edit" sessions only
# when the executor exposes edit_feature — dry-run only, the session's code
# submit remains the way geometry actually changes
FEATURE_EDIT_TOOLS = [
    _fn(
        "list_editable_features",
        "List recognised hole features on the LAST VERIFIED solid of this "
        "session: feature_id, kind, diameter, axis/opening/depth, and whether "
        "the hole is editable (plain through-holes only — countersunk, "
        "counterbored or spotfaced bores are reported but flagged "
        "non-editable). Requires a successful verify() first.",
        {},
        [],
    ),
    _fn(
        "edit_feature",
        "Resize a recognised through-hole to a new diameter on the last "
        "verified solid — runs the executor's transactional gate (volume "
        "annulus, envelope, sibling holes, re-recognition) and returns the "
        "gate report. This is a CHECK: it does not commit geometry — bake "
        "the accepted resize into the code buffer afterwards.",
        {
            "feature_id": _i("feature_id from list_editable_features"),
            "diameter": {
                "type": "number",
                "description": "new bore diameter in mm, e.g. 8.0",
            },
        },
        ["feature_id", "diameter"],
    ),
]

DONE_TOOLS = [
    _fn(
        "done",
        "Finish the session: the current buffer becomes the result. Call only "
        "after your edits are applied and (ideally) verified.",
        {
            "reasoning": _s("short summary of what you changed"),
            "expected_bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "[L, W, H] in mm of the expected result",
            },
        },
        [],
    ),
]

# staged-build session tools (build kind): the buffer grows feature-by-feature
BUILD_TOOLS = [
    _fn(
        "add_block",
        "Append ONE feature's code as a named block at the end of the file "
        "(base body, holes, fillets, …). The block is marked `## block: "
        "<name>` so verify errors map back to it. Names must be unique — "
        "to change an existing block use the edit tools, do not re-add it.",
        {
            "name": _s("short unique block name, e.g. 'base', 'mounting_holes'"),
            "code": _s(
                "python code for this one feature — may use variables defined by earlier blocks"
            ),
        },
        ["name", "code"],
    ),
    _fn(
        "checkpoint",
        "Save the current buffer under a name so `rollback(name)` can restore "
        "it later. Call after a stage verifies clean — limited checkpoints "
        "per session.",
        {"name": _s("unique checkpoint name, e.g. 'base_ok'")},
        ["name"],
    ),
    _fn(
        "rollback",
        "Restore the buffer to a named checkpoint — the cheap way out of a "
        "failed stage instead of undoing many edits by hand. Counts as an "
        "edit (the buffer changes).",
        {"name": _s("checkpoint name from an earlier checkpoint() call")},
        ["name"],
    ),
]

_BLOCK_RE = re.compile(r"^## block:\s*(\S+)", re.M)
_MAX_CHECKPOINTS = 8

# --------------------------------------------------------------------------- #
# structural session tools (optimize step)                                    #
# --------------------------------------------------------------------------- #

_AXIS = {"type": "string", "enum": ["x", "y", "z"], "description": "cartesian axis"}
_SIGNED_AXIS = {
    "type": "string",
    "enum": ["+x", "-x", "+y", "-y", "+z", "-z"],
    "description": "signed axis",
}
_VEC3 = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
    "description": "[x, y, z] in mm",
}

_REGION_SCHEMA = {
    "type": "object",
    "description": "geometric region: surface (part skin facing axis, follows "
    "curves), area (XY sub-area via polygon/x_range/y_range/center+diameter_mm, "
    "optionally restricted to the surface facing axis), hole/anchor (measured "
    "feature anchor via hole_index — any axis; use list_anchors first), bore "
    "(the bore WALL of an anchor — for bearing/preload loads), face (legacy "
    "surface alias), cylinder (around an axis — needs axis+diameter_mm), box "
    "(center+diameter_mm edge) or all",
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "face",
                "surface",
                "area",
                "cylinder",
                "box",
                "hole",
                "anchor",
                "bore",
                "all",
            ],
        },
        "axis": _s("signed axis, e.g. -z (face normal or cylinder axis)"),
        "center": {**_VEC3, "description": "optional mm hint, e.g. hole position"},
        "diameter_mm": {"type": "number", "description": "cylinder diameter / box edge"},
        "margin_mm": {"type": "number", "description": "selection depth, default 2"},
        "polygon": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "area: XY polygon vertices in mm",
        },
        "x_range": {
            "type": "array",
            "items": {"type": "number"},
            "description": "area: [min, max] mm in X",
        },
        "y_range": {
            "type": "array",
            "items": {"type": "number"},
            "description": "area: [min, max] mm in Y",
        },
        "hole_index": {
            "type": "integer",
            "description": "hole/anchor/bore: 1-based index into the anchor table "
            "(call list_anchors to see it)",
        },
        "fix_mode": {
            "type": "string",
            "enum": ["all", "normal"],
            "description": "fixed regions only: 'all' = encastre (default); "
            "'normal' = pin only the axis DOF — roller/sliding/symmetry",
        },
        "material": _s(
            "keep regions only: override material inside the region "
            "(e.g. 'aluminum' for a threaded insert)"
        ),
        "description": _s("label, e.g. 'M4 bore'"),
    },
}

_LOAD_SCHEMA = {
    "type": "object",
    "description": "one load on a region — combine freely: force_n (total force, "
    "split over region nodes), pressure_mpa (on face/surface/area), bearing_n "
    "(cosine-distributed bolt/bearing force on a bore wall — needs a "
    "hole/anchor/bore region), moment_nm (pure couple [mx,my,mz] in N·m), "
    "preload_n (bolt pretension — axial squeeze on the two bore mouths), "
    "remote_point [x,y,z] (apply force_n at that point — adds the induced "
    "moment to the region), g_load [ax,ay,az] (body force in units of g — "
    "gravity on the region's mass), direction_cone_deg (robust TopOpt: sample "
    "the force over a cone of directions)",
    "properties": {
        "region": _REGION_SCHEMA,
        "force_n": {
            **_VEC3,
            "description": "[fx, fy, fz] in Newton — also the bearing direction",
        },
        "pressure_mpa": {
            "type": "number",
            "description": "uniform pressure on a face region",
        },
        "bearing_n": {
            "type": "number",
            "description": "total bearing force on a bore wall (N) — hole/anchor/bore region",
        },
        "bearing_dir": {**_VEC3, "description": "bearing direction (default: force_n)"},
        "moment_nm": {**_VEC3, "description": "pure couple [mx,my,mz] in N·m"},
        "preload_n": {
            "type": "number",
            "description": "bolt pretension along the bore axis (N)",
        },
        "remote_point": {
            **_VEC3,
            "description": "apply force_n at this point — adds lever moment",
        },
        "g_load": {**_VEC3, "description": "body accel in g units, e.g. [0,0,-1]"},
        "direction_cone_deg": {
            "type": "number",
            "description": "robust TopOpt: sample force directions within this cone (deg)",
        },
    },
}

_LOAD_CASES_SCHEMA = {
    "type": "array",
    "description": "load cases; each needs at least one fixed region and one load",
    "items": {
        "type": "object",
        "properties": {
            "name": _s("case name"),
            "weight": {
                "type": "number",
                "description": "relative objective weight for TopOpt (default 1)",
            },
            "fixed": {
                "type": "array",
                "items": _REGION_SCHEMA,
                "description": "constrained regions — fix_mode 'all' (encastre) "
                "or 'normal' (roller/symmetry on the axis DOF)",
            },
            "loads": {"type": "array", "items": _LOAD_SCHEMA},
        },
        "required": ["fixed", "loads"],
    },
}

STRUCT_TOOLS = [
    _fn(
        "list_anchors",
        "List the part's measured feature anchors: holes, bores, inserts, "
        "bearing seats on ANY axis — each with index, 3D centre, axis, "
        "diameter, depth, kind and bolt size. hole_index in hole/anchor/bore "
        "regions refers to this table. Call FIRST — never guess coordinates "
        "when a measured anchor exists. Free (no FEM budget).",
        {},
        [],
    ),
    _fn(
        "check_load",
        "Sanity-check a load case WITHOUT solving: resolved node counts, "
        "resultant force and applied moment about the support centroid, "
        "warnings for empty regions or unstable supports. Call after writing "
        "each case — cheaper than a full preview/fem run.",
        {
            "load_cases": _LOAD_CASES_SCHEMA,
        },
        ["load_cases"],
    ),
    _fn(
        "preview_regions",
        "Render which cells your regions select — fixed=blue, load=red, "
        "keep=green — plus node counts and the resolved total force/moment. "
        "Call BEFORE the first topopt_run to verify load placement; does "
        "not consume FEM/TopOpt budget.",
        {
            "load_cases": _LOAD_CASES_SCHEMA,
            "keep_regions": {
                "type": "array",
                "items": _REGION_SCHEMA,
                "description": "regions that will be forced solid",
            },
        },
        ["load_cases"],
    ),
    _fn(
        "analyze_fem",
        "Run FEM on the BASE part with the given load cases — returns per-case "
        "safety factor, max von-Mises stress and displacement, warnings, and "
        "stress renders. Use to check the design before optimizing or to "
        "compare load-case variants.",
        {
            "load_cases": _LOAD_CASES_SCHEMA,
            "material": _s("material name, e.g. PLA, PETG, aluminum — empty = spec default"),
            "voxel_mm": {"type": "number", "description": "FEM voxel pitch in mm"},
        },
        ["load_cases"],
    ),
    _fn(
        "topopt_run",
        "Run SIMP topology optimization on the base part, then FEM-verify the "
        "result. Returns run_index, achieved volume fraction, iterations, "
        "worst safety factor and a render of the optimized shape. Vary "
        "target_volume_fraction / keep_regions / material between runs and "
        "pick the best one with done(run_index).",
        {
            "target_volume_fraction": {
                "type": "number",
                "description": "0.1-0.9 fraction of material to keep",
            },
            "max_iterations": {"type": "integer", "description": "SIMP iteration cap"},
            "keep_regions": {
                "type": "array",
                "items": _REGION_SCHEMA,
                "description": "regions forced to stay solid (e.g. mounting bores; "
                "set material for insert zones)",
            },
            "void_regions": {
                "type": "array",
                "items": _REGION_SCHEMA,
                "description": "regions forced EMPTY (forbidden material: cable "
                "runs, swing space, clearance zones)",
            },
            "keep_mode": {
                "type": "string",
                "enum": ["manual", "auto"],
                "description": "'auto' adds a keep ring around EVERY detected anchor",
            },
            "min_member_mm": {
                "type": "number",
                "description": "enforce minimum member thickness (e.g. 1.6 for FDM)",
            },
            "density_threshold": {
                "type": "number",
                "description": "iso threshold 0.2-0.9 for mesh extraction (default 0.5)",
            },
            "self_supporting": {
                "type": "boolean",
                "description": "AM filter — cells need 45° support from below; "
                "use when the part is printed without supports",
            },
            "print_direction": {
                "type": "string",
                "enum": ["+x", "-x", "+y", "-y", "+z", "-z"],
                "description": "build direction for self_supporting (default +z)",
            },
            "filter": {
                "type": "string",
                "enum": ["cone", "heaviside", "helmholtz", "helmholtz+heaviside"],
                "description": "density filter — 'heaviside' gives crisp 0/1 "
                "fields (threshold-robust extraction, slower); 'helmholtz' is "
                "the PDE filter (consistent feature size at domain boundaries, "
                "no thinned edge members); 'helmholtz+heaviside' combines both",
            },
            "robust": {
                "type": "boolean",
                "description": "robust formulation (Sigmund/Wang): optimize the "
                "WORST of eroded/intermediate/dilated designs — print-tolerant "
                "and threshold-insensitive; forces a heaviside-bearing filter "
                "('heaviside' or 'helmholtz+heaviside'), ~3x cost",
            },
            "robust_delta_eta": {
                "type": "number",
                "description": "robust eta half-width 0-0.45 (default 0.25)",
            },
            "robust_designs": {
                "type": "string",
                "enum": ["edi", "e"],
                "description": "'edi' = eroded+intermediate+dilated; 'e' = eroded "
                "only (~half the extra FEM cost)",
            },
            "symmetry": {
                "type": "string",
                "enum": ["xy", "xz", "yz"],
                "description": "mirror the design about that domain mid-plane — "
                "each iteration the density is averaged with its reflection so "
                "the printed part comes out symmetric",
            },
            "multires": {
                "type": "boolean",
                "description": "coarse SIMP at 2x pitch warm-starts the fine "
                "grid — ~3x faster on big parts",
            },
            "solver": {
                "type": "string",
                "enum": ["oc", "mma"],
                "description": "optimizer — 'oc' (fast, volume-only) or 'mma' "
                "(enables real constraints below; slower but required for "
                "minmax/stress/displacement)",
            },
            "case_objective": {
                "type": "string",
                "enum": ["weighted", "minmax"],
                "description": "'minmax' optimizes the WORST load case instead "
                "of the weighted average — use for uncertain loads (needs mma)",
            },
            "max_displacement_mm": {
                "type": "number",
                "description": "mma: hard cap on max nodal displacement",
            },
            "max_stress_mpa": {
                "type": "number",
                "description": "mma: P-norm von-Mises stress limit (e.g. yield/2)",
            },
            "min_gap_mm": {
                "type": "number",
                "description": "warn when enclosed gaps are narrower than this "
                "(would fuse in printing) — defaults to min_member_mm",
            },
            "penal_init": {
                "type": "number",
                "description": "penal continuation: start exponent < penal "
                "(softer start escapes grayscale local minima)",
            },
            "load_cases": _LOAD_CASES_SCHEMA,
            "material": _s("material name — empty = spec default"),
            "voxel_mm": {"type": "number", "description": "voxel pitch in mm"},
        },
        [],
    ),
    _fn(
        "refine_features",
        "Feature finishing on a completed topopt_run: re-cuts EXACT bores, "
        "printed threads (tap-drill), inserts, bearing seats, countersinks and "
        "counterbores at measured anchor positions on the optimized mesh, and "
        "restores keep-region surfaces that voxel smoothing shrank (needs the "
        "original part). Returns a NEW run_index you can pass to done(). "
        "Each run can be refined once.",
        {
            "run_index": {
                "type": "integer",
                "description": "the topopt_run to finish",
            },
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "description": "FeatureSpec subset: {kind, anchor_index|dia|pos, "
                    "axis, depth, thread_spec, cbore_dia, cbore_depth, csk_dia, "
                    "csk_angle, length, section}. Empty → auto-resolve from the "
                    "spec's declared features + scanned anchors.",
                },
                "description": "explicit features to re-cut; omit for auto-resolution",
            },
            "restore_planes": {
                "type": "boolean",
                "description": "union original keep-region surfaces back (default on)",
            },
            "load_cases": _LOAD_CASES_SCHEMA,
        },
        ["run_index"],
    ),
]

OPTIMIZE_DONE_TOOLS = [
    _fn(
        "done",
        "Finish the session. Pass run_index of the best topopt_run (as "
        "returned by it), or -1 to keep the base part unoptimized.",
        {
            "run_index": {
                "type": "integer",
                "description": "chosen topopt run, or -1 to skip optimization",
            },
            "reasoning": _s("why this run / why skip"),
        },
        ["run_index"],
    ),
]

# --------------------------------------------------------------------------- #
# mesh session tools (mesh_refine step)                                        #
# --------------------------------------------------------------------------- #

MESH_INFO_TOOLS = [
    _fn(
        "mesh_inspect",
        "Measured facts about the CURRENT working mesh: bbox, volume, "
        "watertight, triangle count, flat-base ratio, body count, detected "
        "holes/dimples, measured anchors, ops applied so far.",
        {},
        [],
    ),
    _fn(
        "mesh_preview",
        "Render the current working mesh (iso/front/top views) and see the "
        "result as images. Limited calls — use to verify your transforms.",
        {},
        [],
    ),
    _fn(
        "mesh_measure",
        "Compare the working mesh against the spec: per-axis bbox deltas (mm), "
        "detected tunnels vs measured anchor positions (dist + dia), "
        "missing-hole count. No image output — pure numbers.",
        {},
        [],
    ),
    _fn(
        "mesh_section",
        "Rasterised cross-section of the mesh at an axis-aligned plane — "
        "inspect internal structure, wall thickness, blind features. "
        "Counts as a preview call.",
        {
            "axis": _AXIS,
            "at_mm": {
                "type": "number",
                "description": "plane position in mm (default: bbox centre)",
            },
        },
        ["axis"],
    ),
    _fn(
        "mesh_overlay",
        "Top-view overlay image: measured outline (red) vs current mesh "
        "footprint (blue) vs anchor positions (green). Best single check for "
        "footprint/hole alignment.",
        {},
        [],
    ),
    _fn(
        "mesh_regen_edit",
        "Text-steered regeneration round-trip: edits the conditioning image "
        "with a local image-edit model (your prompt describes the change, e.g. "
        "'add deeper knurled grip grooves'), regenerates the mesh from the "
        "edited image with the SAME model+seed, and re-baselines it. Use for "
        "surface-detail/shape changes that ops cannot express. ALL recorded "
        "ops are discarded — re-apply needed transforms afterwards. Fails "
        "cleanly when no edit-capable image adapter is installed.",
        {
            "prompt": {
                "type": "string",
                "description": "edit instruction for the reference image",
            },
            "strength": {
                "type": "number",
                "description": "img2img strength 0.3..0.9 (default 0.6; lower "
                "keeps more of the original shape)",
            },
        },
        ["prompt"],
    ),
]

MESH_TRANSFORM_TOOLS = [
    _fn(
        "mesh_set_up",
        "Re-orient: the named source axes of the CURRENT mesh map to +Z (up) "
        "and +Y (front). Use when the part is lying on its side/back.",
        {"up": _SIGNED_AXIS, "front": _SIGNED_AXIS},
        ["up"],
    ),
    _fn(
        "mesh_rotate",
        "Rotate around a cartesian axis through the bbox centre (degrees). "
        "After every transform the part is re-centred on XY.",
        {"axis": _AXIS, "deg": {"type": "number", "description": "rotation in degrees"}},
        ["axis", "deg"],
    ),
    _fn(
        "mesh_mirror",
        "Mirror along a cartesian axis (left/right handedness flip).",
        {"axis": _AXIS},
        ["axis"],
    ),
    _fn(
        "mesh_scale",
        "Uniform (`factor`) or per-axis (`x`/`y`/`z`) scaling about the bbox "
        "centre. Per-axis is THE fix for aspect-ratio mismatches.",
        {
            "factor": {"type": "number", "description": "uniform multiplier"},
            "x": {"type": "number", "description": "X-axis multiplier"},
            "y": {"type": "number", "description": "Y-axis multiplier"},
            "z": {"type": "number", "description": "Z-axis multiplier"},
        },
        [],
    ),
    _fn(
        "mesh_flatten_bottom",
        "Cut + cap the mesh flat at `height_mm` above its lowest point — "
        "creates a printable flat base.",
        {"height_mm": {"type": "number", "description": "height above lowest point, mm"}},
        ["height_mm"],
    ),
    _fn(
        "mesh_slice",
        "Cut + cap with an axis-aligned plane at `at_mm`; keep 'low' or 'high' "
        "side. Coordinates are mm in the current frame (part rests on XY).",
        {
            "axis": _AXIS,
            "at_mm": {"type": "number", "description": "plane position in mm"},
            "keep": {"type": "string", "enum": ["low", "high"]},
        },
        ["axis", "at_mm", "keep"],
    ),
    _fn(
        "mesh_crop",
        "Keep only geometry inside the min/max box (mm).",
        {"min": _VEC3, "max": _VEC3},
        ["min", "max"],
    ),
    _fn(
        "mesh_decimate",
        "Reduce triangle count to ~`faces` (quadric). Needed before boolean "
        "ops on dense generated meshes.",
        {"faces": {"type": "integer", "description": "target triangle count"}},
        ["faces"],
    ),
    _fn(
        "mesh_smooth",
        "Laplacian/Taubin smoothing — softens faceting/noise.",
        {
            "iterations": {"type": "integer", "description": "1-20, default 5"},
            "scheme": {"type": "string", "enum": ["laplacian", "taubin"]},
        },
        [],
    ),
    _fn(
        "mesh_smooth_preserve",
        "Taubin smoothing with the flat bottom pinned — softens voxel "
        "striations while keeping the base plane and XY envelope. Use after "
        "mesh_flatten_bottom.",
        {"iterations": {"type": "integer", "description": "1-30, default 10"}},
        [],
    ),
    _fn(
        "mesh_keep_largest",
        "Drop all but the largest disconnected body (removes floaters).",
        {},
        [],
    ),
    _fn(
        "mesh_drop_small",
        "Drop disconnected bodies below `min_volume_frac` of the largest body.",
        {"min_volume_frac": {"type": "number", "description": "0 < f < 0.5"}},
        ["min_volume_frac"],
    ),
    _fn(
        "mesh_repair",
        "Re-run the cleaning pass: merge vertices, fix normals, fill holes, drop tiny floaters.",
        {},
        [],
    ),
    _fn(
        "mesh_clip_height",
        "Cut + cap everything above `z_max` mm — a hard ceiling on height "
        "(e.g. a mesh that grew too tall vs the spec).",
        {"z_max": {"type": "number", "description": "max height in mm"}},
        ["z_max"],
    ),
    _fn(
        "mesh_symmetrize",
        "Mirror the better half across the symmetry plane at the bbox centre. "
        "For parts that must be left/right symmetric — picks the bigger half "
        "by default (keep: auto|pos|neg).",
        {
            "axis": _AXIS,
            "keep": {"type": "string", "enum": ["auto", "pos", "neg"]},
        },
        [],
    ),
    _fn(
        "mesh_remesh",
        "Uniform voxel remesh — watertight by construction, evens topology. "
        "Use before booleans on organic meshes or when faces are uneven.",
        {"voxel_mm": {"type": "number", "description": "voxel pitch in mm (0.05-5)"}},
        ["voxel_mm"],
    ),
    _fn(
        "mesh_shell",
        "Hollow the part: erode an inner solid by `thickness_mm` and subtract "
        "it. ONLY when the spec explicitly asks for a hollow part or wall "
        "thickness — NEVER as a substitute for drilling a bore (hollowing a "
        "solid part destroys it). `opening` slices one face open.",
        {
            "thickness_mm": {"type": "number", "description": "wall thickness mm"},
            "opening": {"type": "string", "enum": ["none", "bottom", "top"]},
        },
        ["thickness_mm"],
    ),
    _fn(
        "mesh_drill_holes",
        "Drill straight holes along x/y/z — with "
        "from_anchors=true the measured anchor positions/diameters are used "
        "directly (never retype coordinates). An explicit holes list may add "
        "extra holes; depth_mm for blind holes, csk_dia/csk_angle_deg for "
        "countersinks. `axis` sets the drill direction: 'z' (default, "
        "position = x,y), 'x' (position = y,z coords via x,y fields), 'y' "
        "(position = x,z). Use the critique's hole axis — e.g. a bore through "
        "a lengthwise grip is axis='x' centered at the section centre.",
        {
            "from_anchors": {
                "type": "boolean",
                "description": "drill all measured anchor holes",
            },
            "holes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "dia": {"type": "number"},
                        "depth_mm": {"type": "number"},
                        "axis": {"type": "string", "enum": ["x", "y", "z"]},
                        "csk_dia": {"type": "number"},
                        "csk_angle_deg": {"type": "number"},
                    },
                    "required": ["x", "y", "dia"],
                },
            },
        },
        [],
    ),
    _fn(
        "mesh_radial_profile",
        "Sculpt the OUTER shape along an axis without primitives: scales the "
        "cross-section radially at stations (0..1 along `axis`). r1/r2 are "
        "factors on the two cross-section axes (axis=x → r1=Y, r2=Z). Use for "
        "ergonomic waists, tapered ends, or rounding an oval tube. Apply "
        "BEFORE bore/hole cuts — the factors scale every vertex, including "
        "existing bore walls.",
        {
            "axis": {**_AXIS, "description": "profile axis, default x"},
            "stations": {
                "type": "array",
                "description": "2-32 stations, factors interpolate linearly",
                "items": {
                    "type": "object",
                    "properties": {
                        "at": {"type": "number", "description": "0..1 along axis"},
                        "r1": {"type": "number", "description": "factor on 1st radial axis"},
                        "r2": {"type": "number", "description": "factor on 2nd radial axis"},
                    },
                    "required": ["at"],
                },
            },
        },
        ["axis", "stations"],
    ),
    _fn(
        "mesh_silhouette_carve",
        "Visual-hull carve: removes material OUTSIDE the reference silhouette "
        "only — it never fills. `view` is the projection direction (+x/-x/+y/-y, "
        "same convention as the mesh_view_mask silhouette check); `outline` is "
        "the reference silhouette polygon in view-plane mm (u horizontal, v "
        "vertical/up). Use for generator overhang vs the reference; a too-deep "
        "waist needs mesh_radial_profile_auto instead.",
        {
            "view": {"type": "string", "enum": ["+x", "-x", "+y", "-y"]},
            "outline": {
                "type": "array",
                "description": "3-500 polygon points [[u, v], ...] in view-plane mm",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "tolerance_mm": {
                "type": "number",
                "description": "grow the carve prism by this (default 0.5)",
            },
        },
        ["view", "outline"],
    ),
    _fn(
        "mesh_radial_profile_auto",
        "Rebuild a revolve cross-section from a smooth station profile — the "
        "complement of silhouette_carve: sets the OUTER radius at each station "
        "to radius_mm, so a too-deep waist/neck is filled, not just pinched. "
        "stations come from the side silhouette (t = 0..1 along `axis`). "
        "blend_mm gaussian-smooths transitions (0 = linear between stations; a "
        "few mm keeps the waist soft). Apply BEFORE bore/hole cuts.",
        {
            "axis": {**_AXIS, "description": "profile axis, default x"},
            "stations": {
                "type": "array",
                "description": "2-32 stations {t, radius_mm}",
                "items": {
                    "type": "object",
                    "properties": {
                        "t": {"type": "number", "description": "0..1 along axis"},
                        "radius_mm": {
                            "type": "number",
                            "description": "target outer radius in mm",
                        },
                    },
                    "required": ["t", "radius_mm"],
                },
            },
            "blend_mm": {
                "type": "number",
                "description": "smoothing length along the axis in mm (default 0)",
            },
        },
        ["axis", "stations"],
    ),
    _fn(
        "mesh_boolean_add",
        "Union a primitive with the part — e.g. add a base plate "
        "(box [x,y,z] centred at [0,0,h/2]) or a boss (cylinder).",
        {
            "shape": {"type": "string", "enum": ["box", "cylinder", "sphere"]},
            "size": {"type": "number", "description": "box: uniform edge length mm"},
            "size_xyz": {**_VEC3, "description": "box: [x, y, z] extents mm"},
            "radius": {"type": "number", "description": "cylinder/sphere radius mm"},
            "height": {"type": "number", "description": "cylinder height mm"},
            "center": _VEC3,
            "axis": {**_AXIS, "description": "cylinder axis, default z"},
            "rotate_z_deg": {"type": "number", "description": "box yaw around Z"},
        },
        ["shape", "center"],
    ),
    _fn(
        "mesh_boolean_cut",
        "Subtract a primitive from the part — e.g. drill a hole (cylinder), "
        "cut a pocket (box) or slice off a region.",
        {
            "shape": {"type": "string", "enum": ["box", "cylinder", "sphere"]},
            "size": {"type": "number", "description": "box: uniform edge length mm"},
            "size_xyz": {**_VEC3, "description": "box: [x, y, z] extents mm"},
            "radius": {"type": "number", "description": "cylinder/sphere radius mm"},
            "height": {"type": "number", "description": "cylinder height mm"},
            "center": _VEC3,
            "axis": {**_AXIS, "description": "cylinder axis, default z"},
            "rotate_z_deg": {"type": "number", "description": "box yaw around Z"},
        },
        ["shape", "center"],
    ),
]

MESH_UNDO_TOOLS = [
    _fn("mesh_undo", "Remove the last transform op and rebuild the state.", {}, []),
    _fn("mesh_reset", "Clear ALL ops and return to the baseline mesh.", {}, []),
]

REGENERATE_TOOLS = [
    _fn(
        "regenerate",
        "Give up on this mesh and request a NEW generation with a different "
        "seed. Only when the shape itself is wrong — not for fixable issues.",
        {
            "subject_hint": _s("optional improved subject description for the generator"),
            "note": _s("why regeneration is needed"),
        },
        [],
    ),
]


SUBMIT_CODE_TOOLS = [
    _fn(
        "submit_code",
        "Submit the complete code file as the answer. Call once, when the code is ready.",
        {
            "code": _s("complete build123d code file"),
            "expected_bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "[L, W, H] in mm of the expected result",
            },
            "params": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": _s("PARAMS key"),
                        "unit": _s("unit, e.g. mm or deg"),
                        "min": {"type": "number", "description": "sensible minimum"},
                        "max": {"type": "number", "description": "sensible maximum"},
                        "description": _s("what this parameter controls"),
                    },
                },
                "description": "optional metadata for PARAMS entries (unit/min/max/description)",
            },
            "reasoning": _s("short design rationale"),
        },
        ["code"],
    ),
]


# --------------------------------------------------------------------------- #
# lazy-elision guard: refuse "# … rest unchanged" placeholder submissions      #
# --------------------------------------------------------------------------- #

_COMMENT_LINE_RE = re.compile(r"^\s*(#.*)$")

# matches on the comment BODY (after '#', stripped, case-insensitive) —
# anchored phrase shapes so a legit note like "# keep wall thickness" survives
_ELISION_RE = re.compile(
    r"(?:"
    r"^\s*(?:\.\.\.|…)|"  # '# …' / '# ... rest unchanged'
    r"\brest\s+of\s+(?:the\s+)?(?:code|file|implementation|class|function|script)\b|"
    r"\b(?:previous|existing|original|earlier)\s+"
    r"(?:code|implementation|logic|parts?|sections?|lines?)\b|"
    r"\b(?:rest|code|implementation|remainder|sections?)\s+"
    r"(?:is\s+|remains?\s+|stays?\s+)?un(?:changed|modified)\b|"
    r"^(?:unchanged|unmodified|as before|same as before|ditto)\W*$|"
    r"\b(?:omitted|elided|skipped|removed|truncated)\s+for\s+brevity\b|"
    r"\btodo\b[^\n]{0,15}\b(?:rest|remaining|unchanged)\b|"
    r"\b(?:same|identical)\s+as\s+(?:before|above|earlier|previous)\b|"
    r"\bkeep\s+the\s+rest\b"
    r")",
    re.IGNORECASE,
)


def elision_marker(code: str) -> str | None:
    """First lazy-elision comment found in `code` ("# ... rest", "# unchanged",
    "# previous code", "# TODO rest" …) or None.

    Comments come from ``tokenize`` so '#' inside string literals is never
    mistaken for one; code that fails to tokenize falls back to comment-only
    lines (the contract check will flag that code on other grounds anyway).
    """
    comments: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                comments.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        comments = [m.group(1) for ln in code.splitlines() if (m := _COMMENT_LINE_RE.match(ln))]
    for c in comments:
        if _ELISION_RE.search(c.lstrip("#").strip()):
            return c.strip()
    return None


def _elision_error(marker: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": (
            f"refused: comment '{marker[:80]}' looks like a placeholder for "
            "omitted code — the executor runs the file VERBATIM. Submit the "
            "COMPLETE code with every section written out (no '# ... rest' "
            "shortcuts)."
        ),
    }


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #


class ToolSession:
    """Mutable state + dispatch for one tool-calling session."""

    def __init__(
        self,
        ctx: AgentCtx,
        *,
        kind: Literal["edit", "codegen", "build", "info", "inspect", "mesh", "optimize"],
        code: str = "",
        verify_fn: VerifyFn | None = None,
        # full-pipeline variant of verify_fn (renders+dfm) — used by
        # render_model only; verify itself runs the light pipeline
        render_fn: VerifyFn | None = None,
        workbench: Any = None,
        fem_fn: StructFn | None = None,
        topopt_fn: StructFn | None = None,
        preview_fn: StructFn | None = None,
        anchors_fn: StructFn | None = None,
        refine_fn: StructFn | None = None,
    ) -> None:
        self.ctx = ctx
        self.kind = kind
        self.buffer = code
        self.verify_fn = verify_fn
        self.render_fn = render_fn
        self.workbench = workbench
        self.fem_fn = fem_fn
        self.topopt_fn = topopt_fn
        self.preview_fn = preview_fn
        self.anchors_fn = anchors_fn
        self.refine_fn = refine_fn
        self._refined_runs: set[int] = set()
        settings = ctx.runtime_settings
        # build sessions verify once per feature block — they get the larger
        # staged-build budget instead of the edit/repair one
        verify_max = (
            settings.agent_build_verify_max if kind == "build" else settings.agent_tool_verify_max
        )
        self.verify_left = verify_max if (verify_fn and settings.agent_tool_verify) else 0
        self.fem_left = (
            settings.agent_fem_max_runs if (fem_fn and settings.agent_struct_tools) else 0
        )
        # generate_concept_image calls left (each call may yield up to 2 images)
        self.image_left = (
            settings.image_tool_budget if (ctx.image is not None and ctx.image_available) else 0
        )
        self.topopt_left = (
            settings.agent_topopt_max_runs if (topopt_fn and settings.agent_struct_tools) else 0
        )
        # topopt run results (index, metrics, stored executor result/iteration)
        self.struct_runs: list[dict[str, Any]] = []
        self.edits_applied = 0
        self.submitted: dict[str, Any] | None = None
        self.tool_log: list[dict[str, Any]] = []
        self._done_reasks = 0
        self._unchanged_warned = False
        # build sessions: named buffer snapshots for rollback()
        self._checkpoints: dict[str, str] = {}
        # sha1 of every image file already sent to the model — an identical
        # re-render is stubbed to text instead of resending the same bytes
        self._sent_image_hashes: set[str] = set()

    # -- schema selection --------------------------------------------------- #

    def _verify_render_tools(self) -> list[dict[str, Any]]:
        """verify + render_model schemas — only while the shared sandbox
        budget lasts. No render callback → render_model could only fall back
        to the light verify and return no image, so it isn't offered."""
        if self.verify_left <= 0:
            return []
        out = list(VERIFY_TOOLS)
        if self.render_fn is not None:
            out += RENDER_TOOLS
        return out

    def tools(self) -> list[dict[str, Any]]:
        if self.kind == "edit":
            out = [
                *CODE_READ_TOOLS,
                *CODE_EDIT_TOOLS,
                *INFO_TOOLS,
                *LIBRARY_TOOLS,
                *CHECK_TOOLS,
                *self._verify_render_tools(),
            ]
            if getattr(self.ctx.executor, "edit_feature", None) is not None:
                out += FEATURE_EDIT_TOOLS
            return out + DONE_TOOLS + IMAGE_TOOLS
        if self.kind == "codegen":
            return [
                *INFO_TOOLS,
                *LIBRARY_TOOLS,
                *CHECK_SNIPPET_TOOLS,
                *SUBMIT_CODE_TOOLS,
                *IMAGE_TOOLS,
            ]
        if self.kind == "build":
            out = [
                *CODE_READ_TOOLS,
                *CODE_EDIT_TOOLS,
                *INFO_TOOLS,
                *LIBRARY_TOOLS,
                *CHECK_SNIPPET_TOOLS,
            ]
            out += self._verify_render_tools()
            return out + DONE_TOOLS + BUILD_TOOLS + IMAGE_TOOLS
        if self.kind == "inspect":
            return [*CODE_READ_TOOLS, *INFO_TOOLS, *LIBRARY_INFO_TOOLS, *IMAGE_TOOLS]
        if self.kind == "mesh":
            out = list(MESH_INFO_TOOLS)
            if self.workbench is None or self.workbench.preview_left <= 0:
                out = [t for t in out if t["function"]["name"] != "mesh_preview"]
            if self.ctx.image is None:
                # no image sidecar → the edit round-trip cannot work
                out = [t for t in out if t["function"]["name"] != "mesh_regen_edit"]
            transforms = MESH_TRANSFORM_TOOLS
            if self.workbench is not None and self.workbench.orientation_locked:
                # measurement-backed orientation is fixed — don't offer tools
                # that would let the model overturn it again
                transforms = [
                    t
                    for t in transforms
                    if t["function"]["name"] not in {"mesh_set_up", "mesh_rotate", "mesh_mirror"}
                ]
            return out + transforms + MESH_UNDO_TOOLS + DONE_TOOLS + REGENERATE_TOOLS + IMAGE_TOOLS
        if self.kind == "optimize":
            out = [
                t
                for t in STRUCT_TOOLS
                if (t["function"]["name"] == "analyze_fem" and self.fem_left > 0)
                or (t["function"]["name"] == "topopt_run" and self.topopt_left > 0)
                or (t["function"]["name"] == "preview_regions" and self.preview_fn)
                or (t["function"]["name"] == "check_load" and self.preview_fn)
                or (t["function"]["name"] == "list_anchors" and self.anchors_fn)
                or (
                    t["function"]["name"] == "refine_features"
                    and self.refine_fn
                    and self.struct_runs
                )
            ]
            return out + OPTIMIZE_DONE_TOOLS + IMAGE_TOOLS
        # info sessions (intake/plan): catalog/library lookups + image gen
        out = [*INFO_TOOLS, *LIBRARY_TOOLS, *IMAGE_TOOLS]
        if self.image_left > 0:
            out += CONCEPT_IMAGE_TOOLS
        return out

    def allowed_names(self) -> set[str]:
        """Tool names this session offers — guard against hallucinated calls
        (e.g. submit_code in a critique session would end it unusably)."""
        return {t["function"]["name"] for t in self.tools()}

    # -- dispatch ------------------------------------------------------------ #

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in self.allowed_names():
            result = {"ok": False, "error": f"tool '{name}' is not available in this session"}
            self.tool_log.append({"name": name, "args": _truncate_args(args), "result": result})
            return result
        try:
            result = await self._dispatch(name, args)
        except Exception as e:  # a failing tool is a result, not an exception
            log.warning("tool_session.tool_error", tool=name, error=str(e))
            result = {"ok": False, "error": f"tool '{name}' failed: {e}"}
        self.tool_log.append(
            {"name": name, "args": _truncate_args(args), "result": _truncate(result)}
        )
        return result

    async def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name.startswith("mesh_") and self.workbench is not None:
            wb_res = await self.workbench.handle(name, args)
            return wb_res if isinstance(wb_res, dict) else {"ok": False, "error": "bad tool result"}
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown tool '{name}'"}
        if inspect.iscoroutinefunction(handler):
            res: dict[str, Any] = await handler(args)
        else:
            res = handler(args)
        return res

    # -- code reading -------------------------------------------------------- #

    def _require_buffer(self) -> dict[str, Any] | None:
        if not self.buffer:
            return {"ok": False, "error": "no code loaded in this session"}
        return None

    def _tool_read_lines(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._require_buffer():
            return err
        return codeedits.read_lines(self.buffer, int(a.get("start", 1)), int(a.get("end", 1)))

    def _tool_search_code(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._require_buffer():
            return err
        return codeedits.search_code(
            self.buffer,
            str(a.get("pattern", "")),
            context=int(a.get("context", 2) or 2),
            regex=bool(a.get("regex", False)),
        )

    # -- code editing (edit kind only) ---------------------------------------- #

    def _sigcheck_enabled(self) -> bool:
        return bool(getattr(self.ctx.runtime_settings, "agent_sigcheck", False))

    def _sig_findings(self, code: str) -> tuple[list[str], list[str]]:
        """Signature-gate findings for a code buffer — never raises, never
        blocks (a mid-build buffer may legitimately be incomplete)."""
        if not self._sigcheck_enabled() or not code.strip():
            return [], []
        from cadx.cad.sigcheck import check_code_signatures

        rep = check_code_signatures(code)
        return rep.errors, rep.warnings

    def _apply(self, res: dict[str, Any]) -> dict[str, Any]:
        if res.get("ok") and "code" in res:
            self.buffer = res.pop("code")
            self.edits_applied += 1
            # signature gate on the NEW buffer — findings are surfaced so the
            # model fixes them before verify/done, but the edit still lands
            sig_errors, sig_warnings = self._sig_findings(self.buffer)
            if sig_errors:
                res["sig_errors"] = sig_errors[:10]
                res["note"] = (
                    "signature gate flagged API errors in the buffer — fix them "
                    "before verify/done (the edit was still applied)"
                )
            if sig_warnings:
                res["sig_warnings"] = sig_warnings[:10]
        return res

    def _edit_kind(self) -> dict[str, Any] | None:
        if self.kind not in ("edit", "build"):
            return {"ok": False, "error": "editing is not available in this session"}
        return self._require_buffer()

    def _tool_replace_lines(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._edit_kind():
            return err
        start, end, text = int(a["start"]), int(a["end"]), str(a.get("code", ""))
        n_lines = len(self.buffer.splitlines())
        if not text.strip() and (end - start + 1) >= max(5, int(0.7 * n_lines)):
            # weak models sometimes "fix" by wiping the file — refuse, the
            # buffer is the session's only copy
            return {
                "ok": False,
                "error": "refused: that would clear most of the file — "
                "replace only the lines that need changing",
            }
        return self._apply(codeedits.replace_lines(self.buffer, start, end, text))

    def _tool_insert_after(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._edit_kind():
            return err
        return self._apply(
            codeedits.insert_after_line(self.buffer, int(a["line"]), str(a.get("code", "")))
        )

    def _tool_insert_before(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._edit_kind():
            return err
        return self._apply(
            codeedits.insert_before_line(self.buffer, int(a["line"]), str(a.get("code", "")))
        )

    def _tool_find_replace(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._edit_kind():
            return err
        find, replace = str(a.get("find", "")), str(a.get("replace", ""))
        n_lines = len(self.buffer.splitlines())
        if (
            not replace.strip()
            and find.strip()
            and len(find.splitlines()) >= max(5, int(0.7 * n_lines))
        ):
            return {
                "ok": False,
                "error": "refused: that would clear most of the file — "
                "replace only the lines that need changing",
            }
        return self._apply(codeedits.find_replace(self.buffer, find, replace))

    def _tool_set_param(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._edit_kind():
            return err
        val = a.get("value")
        if not isinstance(val, str | int | float | bool):
            return {"ok": False, "error": "value must be string, number or boolean"}
        return self._apply(codeedits.set_param(self.buffer, str(a.get("name", "")), val))

    # -- staged build (build kind only) ----------------------------------------- #

    def _block_names(self) -> list[str]:
        """`## block: <name>` markers currently in the buffer, in order."""
        return _BLOCK_RE.findall(self.buffer)

    def _block_at_line(self, line: int) -> str | None:
        """Name of the block section containing 1-based `line` (the last
        `## block:` marker at or above it), or None for scaffold code."""
        name: str | None = None
        for i, ln in enumerate(self.buffer.splitlines(), 1):
            if i > line:
                break
            m = re.match(r"## block:\s*(\S+)", ln)
            if m:
                name = m.group(1)
        return name

    def _tool_add_block(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.kind != "build":
            return {"ok": False, "error": "add_block is only available in build sessions"}
        name = re.sub(r"[^\w-]+", "_", str(a.get("name", "")).split("\n", 1)[0].strip())
        if not name:
            return {"ok": False, "error": "name is required"}
        blocks = self._block_names()
        if name in blocks:
            return {
                "ok": False,
                "error": f"block '{name}' already exists — edit it in place "
                "(search_code/read_lines + replace_lines) or pick a new name",
                "blocks": blocks,
            }
        code = str(a.get("code", "")).rstrip()
        if not code.strip():
            return {"ok": False, "error": "code is required"}
        text = f"\n## block: {name}\n{code}"
        res = self._apply(
            codeedits.insert_after_line(self.buffer, len(self.buffer.splitlines()), text)
        )
        if res.get("ok"):
            res["block"] = name
            res["blocks"] = self._block_names()
        return res

    def _tool_checkpoint(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.kind != "build":
            return {"ok": False, "error": "checkpoint is only available in build sessions"}
        name = str(a.get("name", "")).strip()
        if not name:
            return {"ok": False, "error": "name is required"}
        if name in self._checkpoints:
            return {"ok": False, "error": f"checkpoint '{name}' already exists"}
        if len(self._checkpoints) >= _MAX_CHECKPOINTS:
            return {
                "ok": False,
                "error": f"checkpoint limit reached ({_MAX_CHECKPOINTS})",
                "checkpoints": sorted(self._checkpoints),
            }
        self._checkpoints[name] = self.buffer
        return {
            "ok": True,
            "checkpoint": name,
            "checkpoints": sorted(self._checkpoints),
            "lines_total": len(self.buffer.splitlines()),
        }

    def _tool_rollback(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.kind != "build":
            return {"ok": False, "error": "rollback is only available in build sessions"}
        name = str(a.get("name", "")).strip()
        saved = self._checkpoints.get(name)
        if saved is None:
            return {
                "ok": False,
                "error": f"unknown checkpoint '{name}'",
                "checkpoints": sorted(self._checkpoints),
            }
        res = self._apply({"ok": True, "code": saved})
        res["restored"] = name
        res["lines_total"] = len(self.buffer.splitlines())
        res["blocks"] = self._block_names()
        return res

    # -- info tools ------------------------------------------------------------ #

    def _tool_search_catalog(self, a: dict[str, Any]) -> dict[str, Any]:
        query = str(a.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        hits = catalog.search(query)
        return {"ok": True, "results": hits, "count": len(hits)}

    def _tool_lookup_part(self, a: dict[str, Any]) -> dict[str, Any]:
        try:
            data = catalog.lookup_part(
                str(a.get("class_name", "")),
                fastener_type=a.get("fastener_type") or None,
                size=str(a["size"]) if a.get("size") is not None else None,
                length=float(a["length"]) if a.get("length") is not None else None,
            )
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "part": data}

    def _tool_cadlib_lookup(self, a: dict[str, Any]) -> dict[str, Any]:
        try:
            from cadx.cad import cadlib  # backend package path
        except ImportError:
            import cadlib  # type: ignore[no-redef]  # sandbox layout

        name = str(a.get("name", ""))
        fn = getattr(cadlib, name, None)
        if fn is None or not callable(fn):
            return {
                "ok": False,
                "error": f"'{name}' not found in cadlib. "
                f"Available: {', '.join(n for n in dir(cadlib) if not n.startswith('_'))}",
            }
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"
        doc = inspect.getdoc(fn) or "no docstring"
        return {"ok": True, "name": name, "signature": f"{name}{sig}", "doc": doc[:1500]}

    def _tool_quickref(self, a: dict[str, Any]) -> dict[str, Any]:
        """Signature digest — synchronous host-side introspection, cached
        per installed package versions in cadx.agent.quickref."""
        from cadx.agent.quickref import quickref

        topic = str(a.get("topic", "")).strip()
        if not topic:
            return {"ok": False, "error": "topic is required"}
        text = quickref(topic)
        if text.startswith(("unknown quickref topic", "cannot resolve")):
            return {"ok": False, "error": text}
        return {"ok": True, "topic": topic, "reference": text}

    # -- image tools ------------------------------------------------------------ #

    def _tool_view_image(self, a: dict[str, Any]) -> dict[str, Any]:
        """Return the image file via `_images` — toolloop forwards it as a
        user message with image parts (tool messages can't carry images)."""
        aid = str(a.get("asset_id", "")).strip()
        images = [x for x in self.ctx.assets if x.kind in IMAGE_INPUT_KINDS]
        match = next((x for x in images if x.asset_id == aid), None)
        if match is None and aid:
            cands = [x for x in images if x.asset_id.startswith(aid)]
            if len(cands) == 1:
                match = cands[0]
        if match is None:
            return {
                "ok": False,
                "error": f"unknown image asset '{aid}'",
                "available": [x.asset_id for x in images],
            }
        if not match.path.exists():
            return {"ok": False, "error": f"image file missing: {match.filename}"}
        return {
            "ok": True,
            "asset_id": match.asset_id,
            "filename": match.filename,
            "caption": match.caption,
            "_images": [str(match.path)],
        }

    async def _tool_generate_concept_image(self, a: dict[str, Any]) -> dict[str, Any]:
        """Local text-to-image call — budget-capped per session; the result is
        persisted as a generated_image asset AND forwarded via `_images`."""
        if self.image_left <= 0:
            return {"ok": False, "error": "concept image budget exhausted"}
        prompt = str(a.get("prompt", "")).strip()
        if not prompt:
            return {"ok": False, "error": "prompt is required"}
        count = max(1, min(2, int(a.get("count", 1) or 1)))

        # optional reference image — must be one of this project's image assets
        # the agent already knows (ctx.assets); the id travels in the trace
        # either way, but the bytes never leave the machine
        ref_id = str(a.get("reference_asset_id") or "").strip()
        ref_asset: InputAsset | None = None
        if ref_id:
            ref_asset = next(
                (
                    x
                    for x in self.ctx.assets
                    if x.asset_id == ref_id and x.kind in IMAGE_INPUT_KINDS
                ),
                None,
            )
            if ref_asset is None:
                return {
                    "ok": False,
                    "error": f"reference_asset_id '{ref_id}' is not an image asset of this job",
                    "calls_left": self.image_left,
                }
        strength = a.get("strength")
        mark_sketch = bool(a.get("mark_sketch"))
        self.image_left -= 1

        try:
            from cadx.agent.concept_image import generate_concept_images

            out = await generate_concept_images(
                self.ctx,
                prompt,
                count=count,
                seed=int(a["seed"]) if a.get("seed") is not None else None,
                ref_asset=ref_asset,
                strength=float(strength) if strength is not None else None,
                mark_sketch=mark_sketch,
                source="concept_tool",
            )
            return {
                "ok": True,
                "calls_left": self.image_left,
                "assets": [{"asset_id": x.asset_id, "filename": x.filename} for x in out],
                "note": "image(s) shown below and stored as generated_image assets — "
                "usable as image inputs for mesh generation",
                "_images": [str(x.path) for x in out],
            }
        except Exception as e:
            log.warning("tool_session.concept_image_failed", error=str(e))
            return {
                "ok": False,
                "error": f"image generation failed: {e}",
                "calls_left": self.image_left,
            }

    # -- library tools ---------------------------------------------------------- #

    async def _fetch_part(self, part_id: str) -> Any:
        from cadx.db.models import LibraryPart
        from cadx.db.session import session_scope

        try:
            pid = uuid.UUID(part_id)
        except ValueError:
            return None
        async with session_scope() as db:
            return await db.get(LibraryPart, pid)

    async def _tool_search_library(self, a: dict[str, Any]) -> dict[str, Any]:
        from sqlmodel import select

        from cadx.db.models import LibraryPart
        from cadx.db.session import session_scope

        q = str(a.get("query", "")).strip().lower()
        kind = str(a.get("kind", "")).strip().lower()
        async with session_scope() as db:
            rows = (await db.execute(select(LibraryPart))).scalars().all()
        hits: list[dict[str, Any]] = []
        for p in rows:
            if kind and p.kind != kind:
                continue
            hay = f"{p.name} {p.caption} {p.filename}".lower()
            if q and not all(t in hay for t in q.split()):
                continue
            hits.append(
                {
                    "id": str(p.id),
                    "name": p.name,
                    "kind": p.kind,
                    "filename": p.filename,
                    "caption": p.caption,
                    "params": sorted((p.params or {}).keys()) or None,
                }
            )
            if len(hits) >= 20:
                break
        out: dict[str, Any] = {"ok": True, "count": len(hits), "results": hits}
        if not hits:
            out["hint"] = "no library parts match — model the geometry yourself"
        return out

    async def _tool_library_part(self, a: dict[str, Any]) -> dict[str, Any]:
        part = await self._fetch_part(str(a.get("part_id", "")))
        if part is None:
            return {"ok": False, "error": "library part not found"}
        out: dict[str, Any] = {
            "ok": True,
            "id": str(part.id),
            "name": part.name,
            "kind": part.kind,
            "filename": part.filename,
            "caption": part.caption,
        }
        if part.kind == "template":
            out["params"] = part.params or {}
            out["code"] = (part.code or "")[:6000]
        else:
            out["usage"] = f'load_reference("{part.id}")  # returns the stored part solid'
        return out

    async def _tool_attach_library_part(self, a: dict[str, Any]) -> dict[str, Any]:
        part = await self._fetch_part(str(a.get("part_id", "")))
        if part is None:
            return {"ok": False, "error": "library part not found"}
        pid = str(part.id)
        if pid in self.ctx.library_attached or any(x.asset_id == pid for x in self.ctx.assets):
            return {"ok": True, "part_id": pid, "note": "already attached"}
        if part.kind == "template":
            self.ctx.library_templates.append(
                {
                    "name": part.name,
                    "caption": part.caption,
                    "code": part.code or "",
                    "params": part.params or {},
                }
            )
            usage = "template code attached — adapt it, don't copy verbatim"
        else:
            self.ctx.assets.append(
                InputAsset(
                    asset_id=pid,
                    kind="reference_step" if part.kind == "step" else "reference_stl",
                    path=storage.resolve(part.path),
                    caption=f"{part.name} — {part.caption}".rstrip(" —"),
                    filename=part.filename,
                )
            )
            usage = f'file part attached — load it via load_reference("{pid}")'
        self.ctx.library_attached.append(pid)
        return {
            "ok": True,
            "part_id": pid,
            "name": part.name,
            "kind": part.kind,
            "usage": usage,
        }

    # -- checks ---------------------------------------------------------------- #

    def _contract_errors(self, code: str) -> list[str]:
        chk = check_code(code, sigcheck=self._sigcheck_enabled())
        errs = list(chk.errors) + list(chk.sig_errors)
        if not chk.errors:
            errs += check_standard_part_calls(code)
        return errs

    def _tool_check_code(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._require_buffer():
            return err
        errs = self._contract_errors(self.buffer)
        out: dict[str, Any] = {"ok": not errs, "errors": errs[:10]}
        _, sig_warnings = self._sig_findings(self.buffer)
        if sig_warnings:
            out["sig_warnings"] = sig_warnings[:10]
        return out

    def _tool_check_snippet(self, a: dict[str, Any]) -> dict[str, Any]:
        code = str(a.get("code", ""))
        errs = self._contract_errors(code)
        out: dict[str, Any] = {"ok": not errs, "errors": errs[:10]}
        _, sig_warnings = self._sig_findings(code)
        if sig_warnings:
            out["sig_warnings"] = sig_warnings[:10]
        return out

    # -- verify ---------------------------------------------------------------- #

    def _exec_diag(self, res: dict[str, Any], label: str) -> dict[str, Any]:
        """Failure shape shared by verify/render: structured error diagnosis."""
        from cadx.cad.errordiag import diagnose

        diag = diagnose(res.get("error"), res.get("traceback"), self.buffer)
        out: dict[str, Any] = {
            "ok": False,
            label: "execution failed",
            "category": diag.category,
            "error": (res.get("error") or "")[:400],
            "line": diag.failing_line,
            "excerpt": diag.code_excerpt,
            "hint": diag.hint,
        }
        # staged build: attribute the failing line to its `## block:` section
        # so the model knows WHICH feature to fix/rollback, not just the line
        if (
            self.kind == "build"
            and diag.failing_line is not None
            and (blk := self._block_at_line(diag.failing_line))
        ):
            out["block"] = blk
        # error-triggered doc injection: when the error names a real symbol
        # (AttributeError/NameError/TypeError), attach its introspected
        # signature so the repair doesn't guess the API again
        try:
            from cadx.agent.docfetch import docs_for_error

            docs = docs_for_error(
                res.get("error"), res.get("traceback"), self.buffer, diag.failing_line
            )
            if docs:
                out["docs"] = docs
        except Exception as e:  # doc injection is best-effort, never breaks the result
            log.warning("docfetch.failed", error=str(e))
        geo = res.get("geometry_at_failure")
        if geo:
            out["geometry_at_failure"] = geo
            out["note"] = (
                "geometry_at_failure shows where the built bodies ACTUALLY are "
                "— a body at the origin that should be elsewhere means the "
                "placement was discarded; fix positioning before selectors"
            )
        return out

    async def _tool_verify(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.verify_fn is None or self.verify_left <= 0:
            return {"ok": False, "error": "verify not available (budget exhausted or disabled)"}
        if err := self._require_buffer():
            return err
        self.verify_left -= 1
        try:
            res = await self.verify_fn(self.buffer)
        except Exception as e:
            return {"ok": False, "error": f"verify execution failed: {e}"}
        if res.get("ok"):
            ana = res.get("analysis") or {}
            out: dict[str, Any] = {
                "ok": True,
                "verify": "code executed successfully",
                "bbox_size": ana.get("bbox_size"),
                "volume": ana.get("volume"),
                "is_valid": ana.get("is_valid"),
                "watertight": ana.get("watertight"),
                "warnings": ana.get("warnings"),
            }
            if res.get("features"):
                out["features"] = res["features"]
            return out
        return self._exec_diag(res, "verify")

    async def _tool_render_model(self, a: dict[str, Any]) -> dict[str, Any]:
        """verify + vision: same sandbox run, but the contact-sheet render is
        attached via `_images` so the model can see the geometry. Uses the
        full (non-light) pipeline — renders are the point."""
        fn = self.render_fn or self.verify_fn
        if fn is None or self.verify_left <= 0:
            return {"ok": False, "error": "render not available (budget exhausted or disabled)"}
        if err := self._require_buffer():
            return err
        self.verify_left -= 1
        try:
            res = await fn(self.buffer)
        except Exception as e:
            return {"ok": False, "error": f"render execution failed: {e}"}
        if not res.get("ok"):
            return self._exec_diag(res, "render")
        images: list[str] = []
        if sheet := res.get("contact_sheet"):
            with contextlib.suppress(ValueError):
                images.append(str(storage.resolve(sheet)))
        ana = res.get("analysis") or {}
        out: dict[str, Any] = {
            "ok": True,
            "bbox_size": ana.get("bbox_size"),
            "volume": ana.get("volume"),
            "is_valid": ana.get("is_valid"),
            "watertight": ana.get("watertight"),
        }
        if images:
            out["_images"] = images
            out["render"] = "contact sheet attached — compare with the spec and references"
        else:
            out["render"] = "executed ok, but the executor produced no render"
        return out

    # -- verified feature edits (M9) -------------------------------------------- #

    def _session_brep_rel(self) -> str | None:
        """Newest verified B-Rep this session produced — ``_run/model.brep``
        under any iteration/attempt dir of the working version."""
        v_dir = storage.version_dir(self.ctx.project_id, self.ctx.version_id)
        it_root = v_dir / "iterations"
        candidates: list[Any] = []
        if it_root.is_dir():
            candidates.extend(it_root.glob("*/_run/model.brep"))
            candidates.extend(it_root.glob("*/*/_run/model.brep"))
        if not candidates:
            return None
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return storage.relpath(newest)

    async def _feature_edit_exec(self, extra: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self.ctx.executor, "edit_feature", None)
        if fn is None:
            return {"ok": False, "error": "edit_feature not available in this session kind"}
        source = self._session_brep_rel()
        if source is None:
            return {
                "ok": False,
                "error": "no verified B-Rep in this session yet — run verify() first",
            }
        payload = {
            "job_id": str(self.ctx.job_id),
            "project_id": str(self.ctx.project_id),
            "version_id": str(self.ctx.version_id),
            "iteration": 0,
            "source_brep": source,
            "dry_run": True,  # gate check only — nothing is committed
            "settings": self.ctx.settings.model_dump(),
            "runtime_settings": self.ctx.runtime_settings.executor_values(),
            **extra,
        }
        res = await fn(payload)
        return res if isinstance(res, dict) else {"ok": False, "error": "bad executor result"}

    async def _tool_list_editable_features(self, a: dict[str, Any]) -> dict[str, Any]:
        res = await self._feature_edit_exec({"feature_id": None})
        if not res.get("ok"):
            return res
        return {"ok": True, "features": res.get("features") or []}

    async def _tool_edit_feature(self, a: dict[str, Any]) -> dict[str, Any]:
        try:
            feature_id: int | str = int(str(a.get("feature_id")))
        except (TypeError, ValueError):
            feature_id = str(a.get("feature_id") or "")
        try:
            diameter = float(a.get("diameter"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "diameter must be a number (mm)"}
        res = await self._feature_edit_exec(
            {"feature_id": feature_id, "parameter": "diameter", "value": diameter}
        )
        if res.get("ok"):
            res["note"] = (
                "gate passed — this diameter is geometrically safe; express the "
                "same resize in the code buffer before submit_code/done"
            )
        return res

    # -- structural tools (optimize kind) --------------------------------------- #

    @staticmethod
    def _coerce_obj_list(raw: Any) -> Any:
        """Models often send a single object or a JSON string where a list is
        required — accept `{...}` → `[{...}]` and `'[{...}]'` → parsed."""
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                return raw
            return [parsed] if isinstance(parsed, dict) else parsed
        return raw

    @staticmethod
    def _fem_compact(fem: dict[str, Any]) -> dict[str, Any]:
        """Compact FEM report for the model: per-case numbers + warnings."""
        return {
            "material": (fem.get("material") or {}).get("name"),
            "grid": {k: fem.get("grid", {}).get(k) for k in ("pitch", "elements", "dofs")},
            "cases": [
                {
                    "name": c.get("name"),
                    "safety_factor": c.get("safety_factor"),
                    "max_vm_mpa": c.get("max_vm_mpa"),
                    "max_disp_mm": c.get("max_disp_mm"),
                    "converged": c.get("converged"),
                    "reactions_n": c.get("reactions"),
                    "warnings": c.get("warnings") or [],
                }
                for c in fem.get("cases") or []
            ],
            "warnings": fem.get("warnings") or [],
            "note": fem.get("note"),
        }

    async def _tool_preview_regions(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.preview_fn is None:
            return {"ok": False, "error": "preview_regions not available"}
        cases = self._coerce_obj_list(a.get("load_cases"))
        if not isinstance(cases, list) or not cases:
            return {"ok": False, "error": "load_cases must be a non-empty list"}
        res = await self.preview_fn(
            {**a, "load_cases": cases, "keep_regions": self._coerce_obj_list(a.get("keep_regions"))}
        )
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "preview failed"}
        prev = res.get("preview") or {}
        out: dict[str, Any] = {
            "ok": True,
            "pitch_mm": prev.get("pitch_mm"),
            "solid_elements": prev.get("solid_elements"),
            "cases": prev.get("cases") or [],
        }
        images = [str(p) for p in res.get("_images") or []]
        if images:
            out["_images"] = images
            out["preview"] = (
                "region renders attached — blue=fixed red=load green=keep; "
                "check the load lands where you intended"
            )
        return out

    async def _tool_list_anchors(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.anchors_fn is None:
            return {"ok": False, "error": "list_anchors not available"}
        res = await self.anchors_fn(a)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "anchor scan failed"}
        anchors = res.get("anchors") or []
        return {
            "ok": True,
            "n_anchors": len(anchors),
            "anchors": [
                {
                    k: a2.get(k)
                    for k in (
                        "index",
                        "kind",
                        "axis",
                        "x",
                        "y",
                        "z",
                        "dia",
                        "depth_mm",
                        "bolt_size",
                        "name",
                        "source",
                    )
                    if a2.get(k) is not None
                }
                for a2 in anchors
            ],
            "note": "hole_index in hole/anchor/bore regions refers to these indices",
        }

    async def _tool_check_load(self, a: dict[str, Any]) -> dict[str, Any]:
        """Region resolution + resultant force/moment WITHOUT a render or a
        solve — the cheap equilibrium sanity check."""
        if self.preview_fn is None:
            return {"ok": False, "error": "check_load not available"}
        cases = self._coerce_obj_list(a.get("load_cases"))
        if not isinstance(cases, list) or not cases:
            return {"ok": False, "error": "load_cases must be a non-empty list"}
        res = await self.preview_fn({**a, "load_cases": cases, "render_pngs": False})
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "check failed"}
        prev = res.get("preview") or {}
        return {
            "ok": True,
            "pitch_mm": prev.get("pitch_mm"),
            "solid_elements": prev.get("solid_elements"),
            "cases": prev.get("cases") or [],
        }

    async def _tool_analyze_fem(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.fem_fn is None or self.fem_left <= 0:
            return {"ok": False, "error": "analyze_fem not available (budget exhausted)"}
        cases = self._coerce_obj_list(a.get("load_cases"))
        if not isinstance(cases, list) or not cases:
            return {"ok": False, "error": "load_cases must be a non-empty list"}
        self.fem_left -= 1
        res = await self.fem_fn({**a, "load_cases": cases})
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "FEM failed"}
        fem = res.get("fem") or {}
        out = {"ok": True, **self._fem_compact(fem)}
        images = [str(p) for p in res.get("_images") or []]
        if images:
            out["_images"] = images
        out["fem_calls_left"] = self.fem_left
        return out

    async def _tool_topopt_run(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.topopt_fn is None or self.topopt_left <= 0:
            return {"ok": False, "error": "topopt_run not available (budget exhausted)"}
        self.topopt_left -= 1
        res = await self.topopt_fn(
            {
                **a,
                "load_cases": self._coerce_obj_list(a.get("load_cases")),
                "keep_regions": self._coerce_obj_list(a.get("keep_regions")),
            }
        )
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "optimization failed"}
        top = res.get("topopt") or {}
        fem = res.get("fem") or {}
        sfs = [
            c.get("safety_factor")
            for c in fem.get("cases") or []
            if c.get("safety_factor") is not None
        ]
        idx = len(self.struct_runs)
        self.struct_runs.append(
            {
                "index": idx,
                "iteration": res.get("_iteration"),
                "res": res,
                "params": a,
            }
        )
        out: dict[str, Any] = {
            "ok": True,
            "run_index": idx,
            "volfrac_target": top.get("volfrac_target"),
            "volfrac_achieved": top.get("volfrac_achieved"),
            "iterations": top.get("iterations"),
            "converged": top.get("converged"),
            "worst_safety_factor": min(sfs) if sfs else None,
            "sf_mask": top.get("sf_mask"),
            "sf_export": top.get("sf_export"),
            "min_member_mm": top.get("min_member_mm"),
            "thin_fraction": top.get("thin_fraction"),
            "n_components": top.get("n_components"),
            "cases": self._fem_compact(fem).get("cases"),
            "warnings": (top.get("warnings") or [])[:10],
            "topopt_calls_left": self.topopt_left,
        }
        images = [str(p) for p in res.get("_images") or []]
        if images:
            out["_images"] = images
        # structured run row for the job's live table — emit is a noop in
        # non-job contexts (MCP/chat) and must never break the tool result
        worst_sf = top.get("sf_export") or (min(sfs) if sfs else None)
        emit = getattr(self.ctx, "emit", None)
        if emit is not None:
            with contextlib.suppress(Exception):
                await emit(
                    "optimize",
                    res.get("_iteration") or 0,
                    f"run {idx} done — volfrac {top.get('volfrac_achieved')}, SF {worst_sf}",
                    run={
                        "index": idx,
                        "type": "topopt",
                        "volfrac": top.get("volfrac_target"),
                        "achieved": top.get("volfrac_achieved"),
                        "worst_sf": worst_sf,
                    },
                )
        return out

    async def _tool_refine_features(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.refine_fn is None:
            return {"ok": False, "error": "refine_features not available"}
        idx = a.get("run_index")
        if (
            not isinstance(idx, int)
            or isinstance(idx, bool)
            or not (0 <= idx < len(self.struct_runs))
        ):
            return {
                "ok": False,
                "error": f"run_index must be 0..{len(self.struct_runs) - 1} "
                "(a completed topopt_run)",
            }
        if idx in self._refined_runs:
            return {"ok": False, "error": f"run {idx} already refined"}
        res = await self.refine_fn(
            {
                **a,
                "load_cases": self._coerce_obj_list(a.get("load_cases")),
                "features": self._coerce_obj_list(a.get("features")),
                "_run": self.struct_runs[idx],
            }
        )
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "refine failed"}
        self._refined_runs.add(idx)
        rep = (res.get("features") or {}).get("refine") or {}
        fem = res.get("fem") or {}
        sfs = [
            c.get("safety_factor")
            for c in fem.get("cases") or []
            if c.get("safety_factor") is not None
        ]
        new_idx = len(self.struct_runs)
        self.struct_runs.append(
            {
                "index": new_idx,
                "iteration": res.get("_iteration"),
                "res": res,
                "params": {"refine_of": idx, "features": a.get("features") or "auto"},
            }
        )
        out: dict[str, Any] = {
            "ok": True,
            "run_index": new_idx,
            "refined_from_run": idx,
            "features_applied": rep.get("applied") or [],
            "features_skipped": rep.get("skipped") or [],
            "restored_regions": rep.get("restored_regions") or 0,
            "worst_safety_factor": min(sfs) if sfs else None,
            "cases": self._fem_compact(fem).get("cases") if fem else None,
            "warnings": (rep.get("warnings") or [])[:10],
        }
        images = [str(p) for p in res.get("_images") or []]
        if images:
            out["_images"] = images
        emit = getattr(self.ctx, "emit", None)
        if emit is not None:
            with contextlib.suppress(Exception):
                await emit(
                    "optimize",
                    res.get("_iteration") or 0,
                    f"run {new_idx} done — refined run {idx}, SF {min(sfs) if sfs else None}",
                    run={
                        "index": new_idx,
                        "type": "refine",
                        "refined_from": idx,
                        "applied": len(rep.get("applied") or []),
                        "worst_sf": min(sfs) if sfs else None,
                    },
                )
        return out

    # -- submission ------------------------------------------------------------ #

    def _tool_done(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.kind == "optimize":
            idx = int(a.get("run_index", -1))
            if idx < -1 or idx >= len(self.struct_runs):
                return {
                    "ok": False,
                    "error": f"run_index must be -1 or 0..{len(self.struct_runs) - 1}",
                }
            self.submitted = {
                "chosen_run": idx,
                "reasoning": str(a.get("reasoning", "")),
            }
            return {"ok": True, "done": True}
        if self.kind == "mesh":
            if self.workbench is None or self.workbench.work is None:
                return {"ok": False, "error": self.workbench.error or "no mesh loaded"}
            self.submitted = {
                "ops": [o.model_dump() for o in self.workbench.ops],
                "reasoning": str(a.get("reasoning", "")),
                "expected_bbox": a.get("expected_bbox") or [],
            }
            return {"ok": True, "done": True}
        if err := self._require_buffer():
            return err
        if marker := elision_marker(self.buffer):
            return _elision_error(marker)
        if self.edits_applied == 0 and not self._unchanged_warned:
            self._unchanged_warned = True
            return {
                "ok": False,
                "error": "no edits applied — make a change, or call done again "
                "to confirm the code is already correct",
            }
        errs = self._contract_errors(self.buffer)
        if errs and self._done_reasks < 2:
            self._done_reasks += 1
            return {
                "ok": False,
                "error": "contract check failed — fix before done",
                "errors": errs[:10],
            }
        self.submitted = {
            "code": self.buffer,
            "reasoning": str(a.get("reasoning", "")),
            "expected_bbox": a.get("expected_bbox") or [],
            "unchanged": self.edits_applied == 0,
            "contract_errors": errs if errs else None,
        }
        return {"ok": True, "done": True}

    def _tool_submit_code(self, a: dict[str, Any]) -> dict[str, Any]:
        code = str(a.get("code", ""))
        if marker := elision_marker(code):
            return _elision_error(marker)
        errs = self._contract_errors(code)
        if errs and self._done_reasks < 2:
            self._done_reasks += 1
            return {
                "ok": False,
                "error": "contract check failed — fix and submit again",
                "errors": errs[:10],
            }
        self.submitted = {
            "code": code,
            "reasoning": str(a.get("reasoning", "")),
            "expected_bbox": a.get("expected_bbox") or [],
            "params": [p for p in (a.get("params") or []) if isinstance(p, dict)],
            "contract_errors": errs if errs else None,
        }
        self.buffer = code
        return {"ok": True, "submitted": True}

    def _tool_regenerate(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.kind != "mesh":
            return {"ok": False, "error": "regenerate is only available in mesh sessions"}
        self.submitted = {
            "regenerate": True,
            "subject_hint": str(a.get("subject_hint", "")),
            "note": str(a.get("note", "")),
            "ops": [o.model_dump() for o in (self.workbench.ops if self.workbench else [])],
        }
        return {"ok": True, "regenerate": True}

    @property
    def is_done(self) -> bool:
        return self.submitted is not None


def _truncate(v: Any, limit: int = 1200) -> Any:
    s = json.dumps(v, default=str)
    if len(s) <= limit:
        return v
    return s[:limit] + "…"


def _truncate_args(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        sv = str(v)
        out[k] = sv if len(sv) <= 300 else sv[:300] + "…"
    return out
