"""Deterministic error diagnosis for generated CAD code failures.

Classifies an execution failure into a category, extracts the failing line from
the generated model.py and attaches a targeted fix hint — so even a weak model
gets "what broke, where, and what to try" instead of a raw traceback dump.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LINE_RE = re.compile(r'File "model\.py", line (\d+)')


@dataclass
class ErrorDiag:
    category: str = "runtime"
    failing_line: int | None = None
    code_excerpt: str = ""
    hint: str = ""
    summary: str = ""


def _failing_line(traceback: str | None, error: str | None) -> int | None:
    for src in (traceback, error):
        if not src:
            continue
        hits = _LINE_RE.findall(src)
        if hits:
            return int(hits[-1])
        m = re.search(r"line (\d+)", src)
        if m:
            return int(m.group(1))
    return None


def _excerpt(code: str | None, line: int | None, ctx: int = 2) -> str:
    if not code or line is None:
        return ""
    lines = code.splitlines()
    lo = max(0, line - 1 - ctx)
    hi = min(len(lines), line + ctx)
    out = []
    for i in range(lo, hi):
        marker = ">>>" if i == line - 1 else "   "
        out.append(f"{marker} {i + 1:4d} | {lines[i]}")
    return "\n".join(out)


_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"safe_fillet\(\).*active BuildPart|safe_fillet_shape\(\).*outside BuildPart", re.I
        ),
        "api-misuse",
        "Fillet helper used in the wrong execution mode. `safe_fillet` edits the active "
        "BuildPart and returns only a radius; use it as a statement inside the builder. "
        "For algebra mode use `part, applied_r = safe_fillet_shape(part, edges, radius)`.",
    ),
    (
        re.compile(
            r"executor task failed.*(?:validation error|RuntimeSettings|pydantic)"
            r"|validation error for RuntimeSettings",
            re.I | re.S,
        ),
        "infra",
        "Sandbox infrastructure error (executor/runtime-settings mismatch) — "
        "not a code problem; do not patch generated code for this.",
    ),
    (
        re.compile(r"SyntaxError|IndentationError", re.I),
        "syntax",
        "Fix the syntax at the marked line — unclosed bracket/quote or bad indentation.",
    ),
    (
        re.compile(r"not allowed in generated model code|is not allowed", re.I),
        "forbidden-import",
        "Only these imports exist in the sandbox: build123d, bd_warehouse, math, "
        "numpy, typing, dataclasses. Remove the forbidden import/call and rewrite "
        "the logic with allowed APIs.",
    ),
    (
        re.compile(r"missing top-level PARAMS|no `result` assigned|contains no solids", re.I),
        "contract",
        "The file must define `PARAMS = {...}` and assign a build123d solid to "
        "`result`. Add whatever is missing.",
    ),
    (
        re.compile(
            r"Failed creating a (fillet|chamfer)|no suitable edges|edge.*(empty|selected)",
            re.I,
        ),
        "occ-fillet-empty",
        "The fillet/chamfer ran on an EMPTY edge selection — the selector found "
        "no edges (stale list after translate/locate, wrong filter, or the edges "
        "were consumed by a prior operation). Fix: re-run edges()/filter_by() "
        "AFTER all positioning, widen the selector, or wrap with "
        "cadlib.safe_fillet(edges, r) which shrinks the radius until it fits.",
    ),
    (
        re.compile(
            r"Face can only be created with closed wires|No objects to create a hull"
            r"|Failed to.*loft",
            re.I,
        ),
        "occ-loft",
        "loft/sweep failed — a section wire isn't closed or no valid section "
        "reached the kernel. For arched handles/bridges use "
        "`cadlib.arch_bridge(span_x, apex_z, width_y, thickness)`; for "
        "straight connectors `cadlib.strut(p0, p1, w, t)`.",
    ),
    (
        re.compile(r"BRep_API: command not done|StdFail_NotDone|BRepFilletAPI", re.I),
        "occ-fillet",
        "fillet/chamfer failed: radius/length too large for the selected edges, or "
        "edges already consumed by a previous feature. Fix: reduce the radius, "
        "apply it to fewer edges (e.g. only vertical: edges().filter_by(Axis.Z)), "
        "or drop the fillet entirely.",
    ),
    (
        re.compile(r"mesh_cut removed no material|mesh_fuse added no material", re.I),
        "hybrid-boolean-noop",
        "The cutter's bounding box doesn't overlap the body or the boolean "
        "removed <1e-4 volume — check feature coordinates against MESH_BBOX "
        "(the part is centred at origin, so 'N mm from an end' is NOT an "
        "absolute coordinate). For trimming the mesh to the declared envelope "
        "use `cadlib.clip_to_box(body, x=..., y=..., z_min=...)` — it skips "
        "sides that need no cut instead of raising.",
    ),
    (
        re.compile(r"TopoDS_FrozenShape|frozen shape", re.I),
        "occ-frozen",
        "fillet/chamfer hit an immutable/faceted OCC shape. Wrap with "
        "`cadlib.safe_fillet(edges, r)` or fillet the analytic sub-part BEFORE "
        "`mesh_fuse` — never fillet a mesh solid.",
    ),
    (
        re.compile(r"unexpected keyword argument 'normal'|keyword arguments.*\bnormal\b", re.I),
        "api-plane-normal",
        "build123d `Plane` has no `normal=` kwarg — use `Plane(z_dir=...)`. For "
        "a sketch plane ON a curve use `Plane(origin=pt, x_dir=in_plane_dir, "
        "z_dir=tangent)`. For arched handles/bridges use `cadlib.arch_bridge("
        "span_x, apex_z, width_y, thickness)`.",
    ),
    (
        re.compile(r"unexpected keyword argument 'profile'", re.I),
        "api-sweep-profile",
        "build123d `sweep` takes `sections`/`path` — there is no `profile=` "
        "kwarg. Use `loft(sections=[sk1, sk2, ...])`/`sweep(sections=[...], "
        "path=wire)`, or `cadlib.arch_bridge` for arched handles.",
    ),
    (
        re.compile(
            r"BOPAlgo|boolean.*fail|empty result|no common|cannot fuse|did not produce", re.I
        ),
        "occ-boolean",
        "Boolean operation failed: the tool solid probably doesn't fully intersect "
        "the base, or the faces are coplanar/tangent. Fix: oversize the tool body "
        "by ~0.5 mm, move it slightly into the base, or use builder-mode "
        "Hole()/extrude(amount=-...) instead of a raw boolean.",
    ),
    (
        re.compile(
            r"invalid (size|fastener_type|type)|does not exist in.*table|KeyError.*size", re.I
        ),
        "bd-warehouse",
        "bd_warehouse rejected the size/type string. Pick the nearest valid entry "
        "from the reference tables (e.g. size='M4-0.7', valid fastener_type from "
        "the docs in context).",
    ),
    (
        re.compile(r"load_reference|reference asset.*not found", re.I),
        "reference",
        "load_reference(asset_id) failed — use only asset_ids listed under "
        "REFERENCE BODIES and keep the returned solid unchanged (position/scale "
        "it, don't edit its topology).",
    ),
    (
        re.compile(
            r"load_mesh_solid|MESH_BBOX|MESH_PROFILE|spine_y|spine_x|top_z"
            r"|anchor_holes|ANCHOR_HOLES|IMAGE_PROFILE|load_image_profile",
            re.I,
        ),
        "hybrid-mesh",
        "Hybrid mesh boolean failed. Prefer simple convex primitives as the "
        "tool body, positioned from MESH_BBOX, slightly oversized. Use direct "
        "`result = mesh - tool` / `mesh + tool` (algebra mode), not builder mode.",
    ),
    (
        re.compile(r"TimeoutError|timed out|SIGXCPU|CPU time limit|killed", re.I),
        "timeout",
        "Execution exceeded the time limit — simplify: fewer pattern copies, "
        "smaller fillets, coarser revolve/sweep segments, avoid huge GridLocations.",
    ),
    (
        re.compile(
            r"(cut|fuse|intersect|boolean).*(Curve|Ellipse|Circle|Rectangle|Geom2d|2D)"
            r"|expected.*(solid|TopoDS_Solid)|incompatible.*(shape|type)",
            re.I,
        ),
        "hybrid-2d",
        "A 2D shape (Ellipse/Circle/Rectangle/curve) cannot be a boolean tool on "
        "a solid. Use extruded primitives instead: Cylinder for round holes, Box "
        "for pockets, or extrude(...) a sketch first.",
    ),
    (
        re.compile(r"Unexpected keyword arguments", re.I),
        "api-kwargs",
        "A constructor got keyword arguments it does not accept. build123d "
        "signatures: Rot(0, 90, 0) Euler angles, or Rot(axis=Axis.Y, angle=90), "
        "or Rot(Axis.Y, 90) — never a=/about=; Pos(x, y, z) positional only; "
        "Cylinder(radius=, height=, align=); Box(length=, width=, height=). "
        "Compare the call with the reference idioms.",
    ),
    (
        re.compile(r"has no attribute '(removeSplitter|unify|makeSolid)'", re.I),
        "api-missing-method",
        "That method name is an OpenSCAD/CadQuery idiom — it does not exist in "
        "build123d. To merge coplanar faces/edges left over from boolean ops use "
        "`part = part.clean()`; to convert a shell/mesh to a solid use "
        "`Solid(...)`/`Solid.make_solid(...)` — or drop the call entirely.",
    ),
    (
        re.compile(
            r"(BoundBox|Vector)' object has no attribute|has no attribute "
            r"'(boundingBox|xlen|xmax|xsize|xmin|ylen|ymax|ysize|ymin|zlen|zmax|zsize|zmin)'",
            re.I,
        ),
        "api-bbox",
        "build123d bounding-box API: `bb = part.bounding_box()`; `bb.min`/`bb.max` "
        "and `bb.size` are Vectors — uppercase `.X`/`.Y`/`.Z` or index `[0]`/`[1]`/"
        "`[2]` (OCP names like xlen/xmax/size.x do NOT exist). Simplest: "
        "`cadlib.bbox_dims(part)` returns [dx, dy, dz] in mm.",
    ),
    (
        re.compile(r"AttributeError", re.I),
        "api-misuse",
        "Wrong attribute/method name — check the build123d reference in context. "
        "Common: Edge/Face has no `.pos` (use `.center()`), Vector uses "
        "uppercase `.X`/`.Y`/`.Z` (not `.x`), ShapeList uses "
        ".filter_by/.sort_by not .filter(), Axis uses Axis.X not 'x'.",
    ),
    (
        re.compile(r"TypeError|ValueError", re.I),
        "api-args",
        "Wrong argument types/values — compare the call with the verified idioms "
        "in the build123d reference (Hole takes RADIUS not diameter; Box align "
        "tuple; GridLocations(spacing_x, spacing_y, nx, ny)).",
    ),
    (
        re.compile(r"NameError", re.I),
        "name",
        "Undefined name — a variable/function is used before definition, or a "
        "needed import is missing (must come from the allow-list).",
    ),
]


def diagnose(error: str | None, traceback: str | None, code: str | None = None) -> ErrorDiag:
    """Classify an executor failure and extract the failing code region."""
    blob = "\n".join(x for x in (error, traceback) if x)
    category, hint = "runtime", "Read the last traceback line and fix the root cause."
    for pat, cat, h in _RULES:
        if pat.search(blob):
            category, hint = cat, h
            break
    line = _failing_line(traceback, error)
    excerpt = _excerpt(code, line)
    last = ""
    for ln in reversed(blob.strip().splitlines()):
        if ln.strip():
            last = ln.strip()
            break
    summary = f"[{category}] {last[:300]}"
    return ErrorDiag(
        category=category,
        failing_line=line,
        code_excerpt=excerpt,
        hint=hint,
        summary=summary,
    )


def diag_block(
    diag: ErrorDiag,
    error: str | None,
    traceback: str | None,
    stdout: str | None,
    geometry: list[dict] | None = None,
) -> str:
    """Render the diagnosis for the repair prompt."""
    lines = ["EXECUTION FAILED.", f"diagnosis: {diag.summary}", f"hint: {diag.hint}"]
    if diag.code_excerpt:
        lines.append("failing code:\n" + diag.code_excerpt)
    if geometry:
        rows = [
            "  {name}: bbox X[{bmin[0]:.1f},{bmax[0]:.1f}] "
            "Y[{bmin[1]:.1f},{bmax[1]:.1f}] Z[{bmin[2]:.1f},{bmax[2]:.1f}] "
            "solids={solids} vol={volume_mm3}mm^3".format(
                name=g["name"],
                bmin=g["bbox_min"],
                bmax=g["bbox_max"],
                solids=g["solids"],
                volume_mm3=g["volume_mm3"],
            )
            for g in geometry
            if isinstance(g, dict) and "bbox_min" in g
        ]
        if rows:
            lines.append(
                "GEOMETRY AT FAILURE (actual bboxes of the built shapes — "
                "compare against the intended positions before touching "
                "selectors; a body at the origin that should be elsewhere "
                "means the placement was discarded):\n" + "\n".join(rows)
            )
    if error:
        lines.append("error:\n" + error[-2000:])
    if traceback:
        lines.append("traceback (trimmed):\n" + traceback[-2000:])
    if stdout:
        lines.append("stdout tail:\n" + stdout[-800:])
    return "\n\n".join(lines)
