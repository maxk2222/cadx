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
        re.compile(r"BRep_API: command not done|StdFail_NotDone|BRepFilletAPI", re.I),
        "occ-fillet",
        "fillet/chamfer failed: radius/length too large for the selected edges, or "
        "edges already consumed by a previous feature. Fix: reduce the radius, "
        "apply it to fewer edges (e.g. only vertical: edges().filter_by(Axis.Z)), "
        "or drop the fillet entirely.",
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
        re.compile(r"load_mesh_solid|MESH_BBOX", re.I),
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
        re.compile(r"AttributeError", re.I),
        "api-misuse",
        "Wrong attribute/method name — check the build123d reference in context. "
        "Common: Edge/Face has no `.pos` (use `.center()`), ShapeList uses "
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
    diag: ErrorDiag, error: str | None, traceback: str | None, stdout: str | None
) -> str:
    """Render the diagnosis for the repair prompt."""
    lines = ["EXECUTION FAILED.", f"diagnosis: {diag.summary}", f"hint: {diag.hint}"]
    if diag.code_excerpt:
        lines.append("failing code:\n" + diag.code_excerpt)
    if error:
        lines.append("error:\n" + error[-2000:])
    if traceback:
        lines.append("traceback (trimmed):\n" + traceback[-2000:])
    if stdout:
        lines.append("stdout tail:\n" + stdout[-800:])
    return "\n\n".join(lines)
