"""Apply PatchResponse edits to generated model code.

Each PatchEdit is an exact-text replacement: `find` must occur exactly once in
the code. Deterministic, verifiable — failed hunks are reported back to the
model instead of silently producing broken code.
"""

from __future__ import annotations

import difflib

from cadx.cad.allowlist import check_code
from cadx.llm.schemas import PatchEdit
from cadx.logging import get_logger

log = get_logger(__name__)


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


def _nows_match(code: str, find: str) -> str | None:
    """Locate `find` ignoring ALL whitespace — catches blocks the model
    reformatted (single-line dict vs multi-line literal, re-wrapped args).
    Returns the raw code span covering the match when unique, else None.
    """
    needle = "".join(find.split())
    if len(needle) < 16:
        return None  # short needles collide too easily
    hay = "".join(code.split())
    if hay.count(needle) != 1:
        return None
    idx = hay.index(needle)
    lines = code.splitlines(keepends=True)
    pos = 0
    start = end = -1
    for i, ln in enumerate(lines):
        nws = sum(1 for c in ln if not c.isspace())
        if start < 0 and pos + nws > idx:
            start = i
        if pos + nws >= idx + len(needle):
            end = i
            break
        pos += nws
    if start < 0 or end < 0:
        return None
    return "".join(lines[start : end + 1])


def _fuzzy_match(code: str, find: str, *, threshold: float = 0.9) -> str | None:
    """Locate `find` in `code` via line-window similarity — the last resort for
    models that paraphrase the find block (line merges/splits, small edits).

    Compares stripped-joined line windows (length n-1..n+1) against the
    stripped-joined find text. Applies only when the best window reaches
    `threshold` AND is clearly better than any non-overlapping alternative —
    ambiguous fuzzy matches are worse than a reported miss.
    """
    needle_lines = [ln.strip() for ln in find.splitlines() if ln.strip()]
    if not needle_lines:
        return None
    needle = "".join(needle_lines)
    lines = code.splitlines(keepends=True)
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    n = len(needle_lines)
    scored: list[tuple[float, int, int]] = []  # ratio, start_line, end_line (inclusive)
    # allow windows up to n+4 lines: a single-line find often targets a
    # multi-line reformatted block (dicts, wrapped calls)
    for k in range(max(1, n - 1), min(n + 4, len(nonempty)) + 1):
        for j in range(len(nonempty) - k + 1):
            win = nonempty[j : j + k]
            cand = "".join(lines[i].strip() for i in win)
            ratio = difflib.SequenceMatcher(None, needle, cand).ratio()
            scored.append((ratio, win[0], win[-1]))
    if not scored:
        return None
    best = max(scored)
    if best[0] < threshold:
        return None
    second_disjoint = max(
        (r for r, s, e in scored if e < best[1] or s > best[2]),
        default=0.0,
    )
    if second_disjoint > best[0] - 0.05:
        return None  # another location matches nearly as well — refuse to guess
    return "".join(lines[best[1] : best[2] + 1])


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
            if raw is None:
                # model reformatted the block (line merges/wraps) —
                # whitespace-insensitive substring still finds it verbatim
                raw = _nows_match(out, find)
                if raw is not None:
                    log.info("patch.nows_applied", edit=i)
            if raw is None:
                # last resort: the model paraphrased the block (merged lines,
                # reordered keys) — accept a single clearly-best fuzzy window
                raw = _fuzzy_match(out, find)
                if raw is not None:
                    log.info("patch.fuzzy_applied", edit=i)
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
