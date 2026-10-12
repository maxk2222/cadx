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
  info    intake/plan — catalog/cadlib lookups + library search/attach (final answer = content JSON)
  inspect critique — read-only code view + catalog/library lookups (no attach)
  mesh    mesh_refine — mesh inspect/preview + transform ops + undo/reset + done/regenerate
  optimize structural — analyze_fem + topopt_run probes + done(run_index)
`view_image` (re-view an uploaded reference image) is offered in every kind.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import shutil
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
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
        "Generate a design-concept image with the LOCAL text-to-image model "
        "(FLUX/SDXL on the mesh server) and view it. Use to ideate shapes, "
        "proportions or style before committing to a modeling plan. The image "
        "is stored as a project asset (kind=generated_image) and may serve as "
        "a reference for image-to-3D generation. Limited calls per session.",
        {
            "prompt": _s("what to render — subject, view, style"),
            "count": _i("number of variants, 1 or 2 (default 1)"),
            "seed": _i("optional seed for reproducibility"),
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
    "optionally restricted to the surface facing axis), hole (measured anchor "
    "via hole_index), face (legacy surface alias), cylinder (around an axis — "
    "needs axis+diameter_mm), box (center+diameter_mm edge) or all",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["face", "surface", "area", "cylinder", "box", "hole", "all"],
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
            "description": "hole: 1-based index into the measured anchors",
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
            "fixed": {
                "type": "array",
                "items": _REGION_SCHEMA,
                "description": "fully constrained regions",
            },
            "loads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "region": _REGION_SCHEMA,
                        "force_n": {
                            **_VEC3,
                            "description": "[fx, fy, fz] in Newton",
                        },
                        "pressure_mpa": {
                            "type": "number",
                            "description": "uniform pressure on a face region",
                        },
                    },
                },
            },
        },
        "required": ["fixed", "loads"],
    },
}

STRUCT_TOOLS = [
    _fn(
        "preview_regions",
        "Render which cells your regions select — fixed=blue, load=red, "
        "keep=green — plus node counts and the resolved total force. "
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
                "description": "regions forced to stay solid (e.g. mounting bores)",
            },
            "load_cases": _LOAD_CASES_SCHEMA,
            "material": _s("material name — empty = spec default"),
            "voxel_mm": {"type": "number", "description": "voxel pitch in mm"},
        },
        [],
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
        "watertight, triangle count, flat-base ratio, body count, ops applied "
        "so far.",
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
# session
# --------------------------------------------------------------------------- #


class ToolSession:
    """Mutable state + dispatch for one tool-calling session."""

    def __init__(
        self,
        ctx: AgentCtx,
        *,
        kind: Literal["edit", "codegen", "info", "inspect", "mesh", "optimize"],
        code: str = "",
        verify_fn: VerifyFn | None = None,
        workbench: Any = None,
        fem_fn: StructFn | None = None,
        topopt_fn: StructFn | None = None,
        preview_fn: StructFn | None = None,
    ) -> None:
        self.ctx = ctx
        self.kind = kind
        self.buffer = code
        self.verify_fn = verify_fn
        self.workbench = workbench
        self.fem_fn = fem_fn
        self.topopt_fn = topopt_fn
        self.preview_fn = preview_fn
        settings = ctx.runtime_settings
        self.verify_left = (
            settings.agent_tool_verify_max if (verify_fn and settings.agent_tool_verify) else 0
        )
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

    # -- schema selection --------------------------------------------------- #

    def tools(self) -> list[dict[str, Any]]:
        if self.kind == "edit":
            out = [
                *CODE_READ_TOOLS,
                *CODE_EDIT_TOOLS,
                *INFO_TOOLS,
                *LIBRARY_TOOLS,
                *CHECK_TOOLS,
            ]
            if self.verify_left > 0:
                out += VERIFY_TOOLS + RENDER_TOOLS
            return out + DONE_TOOLS + IMAGE_TOOLS
        if self.kind == "codegen":
            return [
                *INFO_TOOLS,
                *LIBRARY_TOOLS,
                *CHECK_SNIPPET_TOOLS,
                *SUBMIT_CODE_TOOLS,
                *IMAGE_TOOLS,
            ]
        if self.kind == "inspect":
            return [*CODE_READ_TOOLS, *INFO_TOOLS, *LIBRARY_INFO_TOOLS, *IMAGE_TOOLS]
        if self.kind == "mesh":
            out = list(MESH_INFO_TOOLS)
            if self.workbench is None or self.workbench.preview_left <= 0:
                out = [t for t in out if t["function"]["name"] != "mesh_preview"]
            return (
                out
                + MESH_TRANSFORM_TOOLS
                + MESH_UNDO_TOOLS
                + DONE_TOOLS
                + REGENERATE_TOOLS
                + IMAGE_TOOLS
            )
        if self.kind == "optimize":
            out = [
                t
                for t in STRUCT_TOOLS
                if (t["function"]["name"] == "analyze_fem" and self.fem_left > 0)
                or (t["function"]["name"] == "topopt_run" and self.topopt_left > 0)
                or (t["function"]["name"] == "preview_regions" and self.preview_fn)
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

    def _apply(self, res: dict[str, Any]) -> dict[str, Any]:
        if res.get("ok") and "code" in res:
            self.buffer = res.pop("code")
            self.edits_applied += 1
        return res

    def _edit_kind(self) -> dict[str, Any] | None:
        if self.kind != "edit":
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

    # -- info tools ------------------------------------------------------------ #

    def _tool_search_catalog(self, a: dict[str, Any]) -> dict[str, Any]:
        hits = catalog.search(str(a.get("query", "")))
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
        self.image_left -= 1

        import tempfile

        from cadx.agent.image_client import ImageGenRequest

        s = self.ctx.runtime_settings
        tmp = Path(tempfile.mkdtemp(prefix="cadx-concept-"))
        try:
            side = min(768, s.image_max_side)
            seed = a.get("seed")
            if seed is None:
                # deterministic per job+prompt so retries stay reproducible
                h = hashlib.sha256(f"{self.ctx.job_id}:{prompt}".encode()).digest()
                seed = int.from_bytes(h[:4]) % 100_000
            req = ImageGenRequest(
                prompt=prompt,
                model=s.image_model,
                seed=int(seed),
                count=count,
                width=side,
                height=side,
                steps=s.image_steps,
                guidance=s.image_guidance,
                out_dir=tmp,
            )
            result = await self.ctx.image.generate(req, None, self.ctx.check_cancel)
            from cadx.agent.image_assets import persist_generated_images
            from cadx.db.session import session_scope

            async with session_scope() as db:
                assets = persist_generated_images(
                    db,
                    self.ctx.project_id,
                    result.image_paths,
                    prompt=prompt,
                    caption=f"concept — {prompt[:120]}",
                    gen_params={
                        "prompt": prompt,
                        "model": result.model,
                        "backend": result.backend,
                        "seed": result.seed,
                        "steps": result.steps,
                        "guidance": result.guidance,
                        "duration_s": result.duration_s,
                        "source": "concept_tool",
                        "job_id": self.ctx.job_id,
                    },
                )
                await db.commit()
            out_assets: list[dict[str, Any]] = []
            for asset in assets:
                self.ctx.assets.append(
                    InputAsset(
                        asset_id=str(asset.id),
                        kind="generated_image",
                        path=storage.resolve(asset.path),
                        caption=asset.caption,
                        filename=asset.filename,
                    )
                )
                self.ctx.generated_asset_ids.append(str(asset.id))
                out_assets.append({"asset_id": str(asset.id), "filename": asset.filename})
            return {
                "ok": True,
                "model": result.model,
                "seed": result.seed,
                "calls_left": self.image_left,
                "assets": out_assets,
                "note": "image(s) shown below and stored as generated_image assets — "
                "usable as image inputs for mesh generation",
                "_images": [str(storage.resolve(asset.path)) for asset in assets],
            }
        except Exception as e:
            log.warning("tool_session.concept_image_failed", error=str(e))
            return {
                "ok": False,
                "error": f"image generation failed: {e}",
                "calls_left": self.image_left,
            }
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
        chk = check_code(code)
        errs = list(chk.errors)
        if chk.ok:
            errs += check_standard_part_calls(code)
        return errs

    def _tool_check_code(self, a: dict[str, Any]) -> dict[str, Any]:
        if err := self._require_buffer():
            return err
        errs = self._contract_errors(self.buffer)
        return {"ok": not errs, "errors": errs[:10]}

    def _tool_check_snippet(self, a: dict[str, Any]) -> dict[str, Any]:
        errs = self._contract_errors(str(a.get("code", "")))
        return {"ok": not errs, "errors": errs[:10]}

    # -- verify ---------------------------------------------------------------- #

    def _exec_diag(self, res: dict[str, Any], label: str) -> dict[str, Any]:
        """Failure shape shared by verify/render: structured error diagnosis."""
        from cadx.cad.errordiag import diagnose

        diag = diagnose(res.get("error"), res.get("traceback"), self.buffer)
        return {
            "ok": False,
            label: "execution failed",
            "category": diag.category,
            "error": (res.get("error") or "")[:400],
            "line": diag.failing_line,
            "excerpt": diag.code_excerpt,
            "hint": diag.hint,
        }

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
        attached via `_images` so the model can see the geometry."""
        if self.verify_fn is None or self.verify_left <= 0:
            return {"ok": False, "error": "render not available (budget exhausted or disabled)"}
        if err := self._require_buffer():
            return err
        self.verify_left -= 1
        try:
            res = await self.verify_fn(self.buffer)
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

    # -- structural tools (optimize kind) --------------------------------------- #

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
        cases = a.get("load_cases")
        if not isinstance(cases, list) or not cases:
            return {"ok": False, "error": "load_cases must be a non-empty list"}
        res = await self.preview_fn(dict(a))
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

    async def _tool_analyze_fem(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.fem_fn is None or self.fem_left <= 0:
            return {"ok": False, "error": "analyze_fem not available (budget exhausted)"}
        cases = a.get("load_cases")
        if not isinstance(cases, list) or not cases:
            return {"ok": False, "error": "load_cases must be a non-empty list"}
        self.fem_left -= 1
        res = await self.fem_fn(dict(a))
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
        res = await self.topopt_fn(dict(a))
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
            "cases": self._fem_compact(fem).get("cases"),
            "warnings": (top.get("warnings") or [])[:10],
            "topopt_calls_left": self.topopt_left,
        }
        images = [str(p) for p in res.get("_images") or []]
        if images:
            out["_images"] = images
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
