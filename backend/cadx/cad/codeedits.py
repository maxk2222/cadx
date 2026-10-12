"""Line-addressable code editing primitives for tool sessions.

The model addresses code by 1-based line numbers of the CURRENT session
buffer; every edit applies immediately and results report the new line count
so the model knows numbering shifted. All functions return dicts with
``ok``/``error`` — failures are tool results, never exceptions.
"""

from __future__ import annotations

import re
from typing import Any

_MAX_READ = 400
_MAX_MATCHES = 25
_PAD = 5  # width of the line-number gutter

Param = str | int | float | bool


def line_numbered(code: str, start: int = 1, end: int | None = None) -> str:
    """`NNN │ <line>` rendering — the shared line-number convention."""
    lines = code.splitlines()
    if end is None:
        end = len(lines)
    out = []
    for n in range(max(1, start), min(end, len(lines)) + 1):
        out.append(f"{n:>{_PAD}} │ {lines[n - 1]}")
    return "\n".join(out)


def _lines(code: str) -> list[str]:
    return code.splitlines()


def read_lines(code: str, start: int, end: int) -> dict[str, Any]:
    total = len(_lines(code))
    if total == 0:
        return {"ok": False, "error": "file is empty"}
    start = int(start)
    end = int(end)
    if start < 1 or end < start:
        return {"ok": False, "error": f"invalid range {start}-{end} (file has {total} lines)"}
    clamped = end > total or end - start + 1 > _MAX_READ
    end = min(end, total, start + _MAX_READ - 1)
    return {
        "ok": True,
        "content": line_numbered(code, start, end),
        "start": start,
        "end": end,
        "lines_total": total,
        "clamped": clamped,
    }


def search_code(
    code: str, pattern: str, *, context: int = 2, regex: bool = False
) -> dict[str, Any]:
    """Find lines matching `pattern`; returns numbered hits with context."""
    if not pattern:
        return {"ok": False, "error": "empty pattern"}
    lines = _lines(code)
    rx: re.Pattern[str] | None = None
    if regex:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return {"ok": False, "error": f"bad regex: {e}"}
    hits = [i for i, ln in enumerate(lines, 1) if (rx.search(ln) if rx else pattern in ln)]
    if not hits:
        return {"ok": True, "matches": [], "count": 0, "lines_total": len(lines)}
    truncated = len(hits) > _MAX_MATCHES
    hits = hits[:_MAX_MATCHES]
    ctx = max(0, int(context))
    blocks: list[dict[str, Any]] = []
    shown: list[int] = []
    for n in hits:
        lo, hi = max(1, n - ctx), min(len(lines), n + ctx)
        blocks.append({"line": n, "text": lines[n - 1], "context": line_numbered(code, lo, hi)})
        shown.extend(range(lo, hi + 1))
    return {
        "ok": True,
        "matches": blocks,
        "count": len(hits),
        "total_matches": len(hits) if not truncated else f">{_MAX_MATCHES}",
        "lines_total": len(lines),
        "truncated": truncated,
    }


def _mk_edit(code: str, new_code: str, note: str = "") -> dict[str, Any]:
    if new_code == code:
        return {"ok": False, "error": "edit produced identical code"}
    return {
        "ok": True,
        "code": new_code,
        "lines_total": len(_lines(new_code)),
        "note": note or "edit applied — line numbers refer to the new buffer",
    }


def replace_lines(code: str, start: int, end: int, new: str) -> dict[str, Any]:
    """Replace 1-based inclusive line range [start, end] with `new` text."""
    lines = _lines(code)
    start, end = int(start), int(end)
    if start < 1 or end < start or start > len(lines):
        return {
            "ok": False,
            "error": f"invalid range {start}-{end} (file has {len(lines)} lines)",
        }
    end = min(end, len(lines))
    out = lines[: start - 1] + new.splitlines() + lines[end:]
    return _mk_edit(code, "\n".join(out) + "\n")


def insert_after_line(code: str, line: int, text: str) -> dict[str, Any]:
    """Insert `text` after 1-based `line` (0 = top of file)."""
    lines = _lines(code)
    line = int(line)
    if line < 0 or line > len(lines):
        return {
            "ok": False,
            "error": f"line {line} out of range (file has {len(lines)} lines)",
        }
    out = lines[:line] + text.splitlines() + lines[line:]
    return _mk_edit(code, "\n".join(out) + "\n")


def insert_before_line(code: str, line: int, text: str) -> dict[str, Any]:
    """Insert `text` before 1-based `line`."""
    lines = _lines(code)
    line = int(line)
    if line < 1 or line > len(lines):
        return {
            "ok": False,
            "error": f"line {line} out of range (file has {len(lines)} lines)",
        }
    out = lines[: line - 1] + text.splitlines() + lines[line - 1 :]
    return _mk_edit(code, "\n".join(out) + "\n")


def find_replace(code: str, find: str, replace: str) -> dict[str, Any]:
    """Exact-match replace; `find` must occur exactly once."""
    if not find:
        return {"ok": False, "error": "find must not be empty"}
    n = code.count(find)
    if n == 0:
        hint = _closest_hint(code, find)
        return {"ok": False, "error": f"find text not found. {hint}".strip()}
    if n > 1:
        return {
            "ok": False,
            "error": f"find text matches {n} locations — include more context to be unique",
        }
    return _mk_edit(code, code.replace(find, replace, 1))


def _closest_hint(code: str, find: str) -> str:
    """Best-effort pointer to where the model probably meant."""
    import difflib

    first = next((ln for ln in find.splitlines() if ln.strip()), "")
    if not first:
        return ""
    best_n, best_ratio = 0, 0.0
    for i, ln in enumerate(code.splitlines(), 1):
        r = difflib.SequenceMatcher(None, first.strip(), ln.strip()).ratio()
        if r > best_ratio:
            best_n, best_ratio = i, r
    if best_ratio < 0.6:
        return ""
    return f"Closest line ({best_ratio:.0%} similar): {best_n}"


def set_param(code: str, name: str, value: Param) -> dict[str, Any]:
    """Set one entry inside the top-level `PARAMS = {...}` dict literal.

    Replaces the `"name": <literal>` pair on its line; preserves the line's
    trailing comment. Fails (with hint) when the key is absent.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "error": f"code does not parse: {e}"}
    # locate the PARAMS dict literal line range via AST
    dict_node: ast.Dict | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            dict_node = node.value
            break
    if dict_node is None:
        return {"ok": False, "error": "no top-level PARAMS dict literal found"}
    key_node = None
    for k in dict_node.keys:
        if isinstance(k, ast.Constant) and k.value == name:
            key_node = k
            break
    if key_node is None:
        return {
            "ok": False,
            "error": (
                f"PARAMS has no key '{name}' — add it inside the dict "
                f"(lines {dict_node.lineno}-{dict_node.end_lineno}) with insert_after"
            ),
        }
    lines = _lines(code)
    ln = key_node.lineno  # key and value start on the same line in our style
    line = lines[ln - 1]
    m = re.match(rf"^(\s*)[\"']?{re.escape(name)}[\"']?\s*:\s*(.*?)(,\s*)?(#.*)?$", line)
    if not m:
        return {"ok": False, "error": f"could not parse PARAMS entry on line {ln}"}
    quote = '"' if line.lstrip().startswith('"') or '"' in line else "'"
    new_val = f"{quote}{value}{quote}" if isinstance(value, str) else repr(value)
    sep = m.group(3) or ", "  # comma + original whitespace before any comment
    comment = m.group(4) or ""
    lines[ln - 1] = f"{m.group(1)}{quote}{name}{quote}: {new_val}{sep}{comment}"
    new_code = "\n".join(lines) + "\n"
    res = _mk_edit(code, new_code)
    if res.get("ok"):
        res["param"] = {name: value}
    return res


def apply_line_ops(code: str, ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply several line-anchored ops atomically against the original numbering.

    Ops: {"op": "replace", "start": s, "end": e, "code": "..."} |
         {"op": "insert_after"|"insert_before", "line": n, "code": "..."}
    Line numbers refer to the buffer BEFORE any op — applied bottom-up so
    earlier line numbers stay valid.
    """
    lines = _lines(code)

    def span(op: dict[str, Any]) -> int:
        return int(op.get("start", op.get("line", 0)))

    ordered = sorted(enumerate(ops), key=lambda t: span(t[1]), reverse=True)
    applied = 0
    for _, op in ordered:
        kind = op.get("op")
        text = str(op.get("code", op.get("text", "")))
        if kind == "replace":
            s, e = int(op["start"]), int(op["end"])
            if s < 1 or e < s or s > len(lines):
                return {"ok": False, "error": f"op range {s}-{e} out of range ({len(lines)})"}
            lines[s - 1 : e] = text.splitlines()
        elif kind in ("insert_after", "insert_before"):
            n = int(op["line"])
            if n < 0 or n > len(lines):
                return {"ok": False, "error": f"op line {n} out of range ({len(lines)})"}
            pos = n if kind == "insert_after" else n - 1
            lines[pos:pos] = text.splitlines()
        else:
            return {"ok": False, "error": f"unknown op '{kind}'"}
        applied += 1
    res = _mk_edit(code, "\n".join(lines) + "\n")
    if res.get("ok"):
        res["applied"] = applied
    return res
