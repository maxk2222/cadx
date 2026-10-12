"""Apply PatchResponse edits to generated model code.

Each PatchEdit is an exact-text replacement: `find` must occur exactly once in
the code. Deterministic, verifiable — failed hunks are reported back to the
model instead of silently producing broken code.
"""

from __future__ import annotations

from cadx.cad.allowlist import check_code
from cadx.llm.schemas import PatchEdit


def _ws_tolerant_match(code: str, find: str) -> str | None:
    """Locate `find` in `code` ignoring per-line leading/trailing whitespace.

    Returns the raw code span matching it (so `replace` keeps the file's
    indentation style), or None when there is no unique match.
    """
    needle = [ln.strip() for ln in find.splitlines() if ln.strip()]
    if not needle:
        return None
    lines = code.splitlines(keepends=True)
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    hits: list[str] = []
    n = len(needle)
    for j in range(len(nonempty) - n + 1):
        if [lines[i].strip() for i in nonempty[j : j + n]] == needle:
            # raw span from first to last matched line (keeps inner blank lines)
            hits.append("".join(lines[nonempty[j] : nonempty[j + n - 1] + 1]))
    return hits[0] if len(hits) == 1 else None


def apply_edits(code: str, edits: list[PatchEdit]) -> tuple[str | None, list[str]]:
    """Apply find/replace hunks in order. Returns (patched_code, errors).

    All-or-nothing per edit: an edit whose `find` is missing or ambiguous is
    reported and skipped; remaining edits still apply. Callers should retry with
    the errors or request full_code when errors is non-empty.
    """
    errors: list[str] = []
    out = code
    for i, edit in enumerate(edits, start=1):
        find = edit.find
        if not find.strip():
            errors.append(f"edit {i}: empty `find` block")
            continue
        n = out.count(find)
        if n == 0:
            # tolerant retry: weak models drift on whitespace/blank lines —
            # match per-line stripped before declaring a miss
            raw = _ws_tolerant_match(out, find)
            if raw is not None:
                find = raw
                n = 1
        if n == 0:
            errors.append(
                f"edit {i}: `find` block not found in the code "
                f"(starts with {find.strip()[:60]!r}...)"
            )
            continue
        if n > 1:
            errors.append(
                f"edit {i}: `find` block matches {n} locations — include more "
                "surrounding lines to make it unique"
            )
            continue
        out = out.replace(find, edit.replace, 1)
    return out, errors


def apply_patch(
    code: str,
    edits: list[PatchEdit],
    *,
    validate: bool = True,
) -> tuple[str | None, list[str]]:
    """Apply edits and re-validate the result against the code contract.

    Returns (patched_code, errors). patched_code is None when validation fails —
    never hand contract-breaking code to the executor.
    """
    patched, errors = apply_edits(code, edits)
    if errors:
        return None, errors
    if patched is None:
        return None, ["no edits applied"]
    if patched == code:
        return None, ["patch produced identical code — the edits changed nothing"]
    if validate:
        check = check_code(patched)
        if not check.ok:
            return None, ["patched code fails the contract check: " + "; ".join(check.errors[:6])]
    return patched, []


def diff_summary(old: str, new: str, max_lines: int = 20) -> str:
    """One-line-per-change summary for traces/logging (not a real diff)."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    changes = []
    for i, (a, b) in enumerate(zip(old_lines, new_lines, strict=False), start=1):
        if a != b:
            changes.append(f"line {i}: {a.strip()[:50]!r} -> {b.strip()[:50]!r}")
            if len(changes) >= max_lines:
                changes.append("...")
                break
    added = len(new_lines) - len(old_lines)
    if added > 0:
        changes.append(f"+{added} lines")
    elif added < 0:
        changes.append(f"{-added} lines removed")
    return "\n".join(changes) or "no line changes"
