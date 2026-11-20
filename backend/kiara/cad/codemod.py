"""Deterministic, comment-preserving rewrites of unambiguous API misuse.

Only rewrites mistakes that cannot be correct build123d code:

- ``<expr>.translate(a, b, c)`` — exactly 3 positional args, no keywords —
  becomes ``<expr>.translate((a, b, c))`` (the real signature is
  ``translate(vector, transform=False)``).
- ``Plane(..., normal=EXPR, ...)`` — renames the keyword to ``z_dir``;
  build123d ``Plane`` has no ``normal`` kwarg (the valid form is
  ``Plane(origin=..., x_dir=..., z_dir=...)``).
- ``sweep(profile=EXPR, ...)`` — renames ``profile`` to ``sections``; the
  real signature is ``sweep(sections, path, ...)``.

Replacements happen on the SOURCE TEXT using AST node offsets, so comments,
formatting and line numbers survive — ``cad/patch.py`` hunks downstream
still line up. Unparseable code comes back unchanged. A locally defined
symbol (``def translate`` / ``class Plane`` / ``def sweep`` / ...) makes the
matching call ambiguous: that rule — and only that rule — is skipped.
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


def _call_name(func: ast.expr) -> str | None:
    """Bare or attribute call name — ``Plane(...)`` and ``gp.Plane(...)`` both
    resolve to ``"Plane"`` so the rewrite also hits qualified calls."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _kw_rename_span(kw: ast.keyword, offs: list[int], name: str) -> tuple[int, int]:
    """Byte span of just the keyword NAME in ``name=value`` — ``kw.col_offset``
    marks the start of ``name=value``, so the name runs ``len(name)`` bytes
    from there (spaces around ``=`` stay untouched)."""
    start = offs[kw.lineno - 1] + kw.col_offset
    return start, start + len(name)


def fix_api_misuse(code: str) -> tuple[str, list[str]]:
    """Rewrite unambiguous build123d API misuse; returns (code, notes).

    All rule hits are collected against the ORIGINAL AST positions first,
    then applied back-to-front in a single byte pass — sequential text
    rewrites would invalidate the offsets of later rules.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code, []
    # names the file itself defines — a user symbol shadows the build123d
    # callable and makes the call legal, so its rule stays off
    defined = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }

    # (start_byte, end_byte, replacement-or-None-for-paren-wrap, lineno, note)
    hits: list[tuple[int, int, bytes | None, int, str]] = []
    offs = _byte_offsets(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if (
            name == "translate"
            and "translate" not in defined
            and isinstance(node.func, ast.Attribute)  # method call — never bare translate()
            and len(node.args) == 3
            and not node.keywords
            and all(not isinstance(a, ast.Starred) for a in node.args)
        ):
            first, last = node.args[0], node.args[-1]
            hits.append(
                (
                    offs[first.lineno - 1] + first.col_offset,
                    offs[last.end_lineno - 1] + last.end_col_offset,  # type: ignore[operator]
                    None,
                    node.lineno,
                    "translate(x, y, z) -> translate((x, y, z))",
                )
            )
        elif name == "Plane" and "Plane" not in defined:
            kw_names = {k.arg for k in node.keywords}
            kw = next((k for k in node.keywords if k.arg == "normal"), None)
            # a **kwargs spill or an existing z_dir= can carry the real value —
            # renaming then would produce a duplicate or hide the clash
            if kw is not None and "z_dir" not in kw_names and None not in kw_names:
                start, end = _kw_rename_span(kw, offs, "normal")
                hits.append(
                    (start, end, b"z_dir", node.lineno, "Plane(normal=...) -> Plane(z_dir=...)")
                )
        elif (
            name == "sweep"
            and "sweep" not in defined
            and not node.args  # positional sections already passed -> profile= is ambiguous
        ):
            kw_names = {k.arg for k in node.keywords}
            kw = next((k for k in node.keywords if k.arg == "profile"), None)
            if kw is not None and "sections" not in kw_names and None not in kw_names:
                start, end = _kw_rename_span(kw, offs, "profile")
                hits.append(
                    (
                        start,
                        end,
                        b"sections",
                        node.lineno,
                        "sweep(profile=...) -> sweep(sections=...)",
                    )
                )
    if not hits:
        return code, []

    buf = code.encode("utf-8")
    # replaced spans in ORIGINAL coords + their net byte delta — a rewrite
    # nested inside a later hit's span shifts that span's end
    applied: list[tuple[int, int, int]] = []
    notes: list[str] = []
    # back to front so earlier offsets stay valid
    for start, end, repl, lineno, what in sorted(hits, key=lambda h: -h[0]):
        if any(s < end and e > start and not (s >= start and e <= end) for s, e, _d in applied):
            continue  # partial overlap — leave it rather than corrupt source
        adj_end = end + sum(d for s, e, d in applied if s >= start and e <= end)
        if repl is None:
            buf = buf[:start] + b"(" + buf[start:adj_end] + b")" + buf[adj_end:]
            delta = 2
        else:
            buf = buf[:start] + repl + buf[adj_end:]
            delta = len(repl) - (end - start)
        applied.append((start, end, delta))
        notes.append(f"codemod: {what} at line {lineno}")
    notes.reverse()  # ascending line order
    return buf.decode("utf-8"), notes
