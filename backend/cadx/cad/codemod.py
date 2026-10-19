"""Deterministic, comment-preserving rewrites of unambiguous API misuse.

Only rewrites mistakes that cannot be correct build123d code — currently the
``translate(x, y, z)`` three-positional-args form (the real signature is
``translate(vector, transform=False)``). Replacements happen on the SOURCE
TEXT using AST node offsets, so comments, formatting and line numbers survive
— ``cad/patch.py`` hunks downstream still line up. Unparseable code comes back
unchanged.
"""

from __future__ import annotations

import ast


def _byte_offsets(code: str) -> list[int]:
    """Absolute UTF-8 byte offset of each 1-based source line (ast col_offset
    counts bytes, not characters)."""
    offs: list[int] = [0]
    pos = 0
    for line in code.encode("utf-8").splitlines(keepends=True):
        pos += len(line)
        offs.append(pos)
    return offs


def fix_api_misuse(code: str) -> tuple[str, list[str]]:
    """Rewrite unambiguous build123d API misuse; returns (code, notes).

    Rule: ``<expr>.translate(a, b, c)`` — exactly 3 positional args, no
    keywords — becomes ``<expr>.translate((a, b, c))``. A locally defined
    ``translate`` function makes the call ambiguous: the file is left alone.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code, []
    if any(
        isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "translate"
        for n in ast.walk(tree)
    ):
        return code, []  # user-defined translate(...) — not our API

    hits: list[tuple[int, int, int]] = []  # (start_byte, end_byte, lineno)
    offs = _byte_offsets(code)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "translate"
            and len(node.args) == 3
            and not node.keywords
            and all(not isinstance(a, ast.Starred) for a in node.args)
        ):
            first, last = node.args[0], node.args[-1]
            hits.append(
                (
                    offs[first.lineno - 1] + first.col_offset,
                    offs[last.end_lineno - 1] + last.end_col_offset,  # type: ignore[operator]
                    node.lineno,
                )
            )
    if not hits:
        return code, []

    buf = code.encode("utf-8")
    applied: list[tuple[int, int]] = []  # replaced spans in ORIGINAL coords
    notes: list[str] = []
    # back to front so earlier offsets stay valid; a nested hit inside an
    # already-fixed arg span shifts that span's end by 2 bytes per wrap
    for start, end, lineno in sorted(hits, key=lambda h: -h[0]):
        contained = sum(1 for s, e in applied if s >= start and e <= end)
        if any(s < end and e > start and not (s >= start and e <= end) for s, e in applied):
            continue  # partial overlap — leave it rather than corrupt source
        end += 2 * contained
        buf = buf[:start] + b"(" + buf[start:end] + b")" + buf[end:]
        applied.append((start, end - 2 * contained))
        notes.append(f"codemod: translate(x, y, z) -> translate((x, y, z)) at line {lineno}")
    notes.reverse()  # ascending line order
    return buf.decode("utf-8"), notes
