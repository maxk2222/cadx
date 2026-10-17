"""docfetch — error-triggered doc injection (DAG: retrieve ON error, not always-on).

When generated code fails on an API symbol — ``AttributeError: 'Edge' object
has no attribute 'radius'``, ``NameError: name 'Pos' is not defined``, or a
TypeError arity/kwarg failure — the failing name is extracted here and
resolved against the runtime signature digests in ``cadx.agent.quickref``.
The resulting ≤400-char digest rides along on the verify/repair error result
so the model gets the CORRECT signature instead of guessing the same wrong
API again for several repair rounds.

Pure best-effort: no symbol resolves → empty string → zero overhead.
"""

from __future__ import annotations

import re

from cadx.agent.quickref import symbol_digest
from cadx.logging import get_logger

log = get_logger(__name__)

# 'Edge' object has no attribute 'radius' | type object 'Edge' has no attribute
# 'x' | module 'build123d' has no attribute 'x'
_ATTR_ERR = re.compile(
    r"(?:'(?P<obj>[\w.]+)'\s+object|type object '(?P<tobj>[\w.]+)'|module '(?P<mod>[\w.]+)')"
    r"\s+has no attribute '(?P<attr>\w+)'"
)
# NameError: name 'Pos' is not defined
_NAME_ERR = re.compile(r"name '(?P<name>[\w.]+)' is not defined")
# TypeError arity/kwarg failures naming the callee:
#   f() takes 2 positional arguments but 3 were given
#   f() missing 1 required positional argument: 'x'
#   f() got an unexpected keyword argument 'x'  |  f() takes no keyword arguments
_TYPE_CALLEE = re.compile(
    r"(?P<fn>[\w.]+)\(\)\s+"
    r"(?:takes\b|missing \d+ required positional argument|"
    r"got (?:an unexpected|some) keyword argument)"
)
_CALL_RE = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")


def _lookup(callee: str, max_chars: int) -> str:
    """Resolve one callee — 'owner.attr' tries the owner first so the digest
    can name the member's kind, then falls back to the bare attribute name."""
    owner, _, attr = callee.rpartition(".")
    if owner:
        digest = symbol_digest(attr, owner=owner, max_chars=max_chars)
        if digest:
            return digest
    return symbol_digest(attr if owner else callee, max_chars=max_chars)


def _call_names(code: str, line: int) -> list[str]:
    """Call expressions on the 1-based failing source line, e.g.
    'r = edge.radius(fillet)' → ['edge.radius', 'fillet']."""
    lines = code.splitlines()
    if not (1 <= line <= len(lines)):
        return []
    return [m.group(1) for m in _CALL_RE.finditer(lines[line - 1])]


def docs_for_error(
    error: str | None,
    traceback: str | None = None,
    code: str | None = None,
    line: int | None = None,
    *,
    max_chars: int = 400,
) -> str:
    """Return a compact quickref digest for the symbol implicated by an exec
    error, or "" when nothing resolves.

    Resolution order: AttributeError owner+attribute, NameError name,
    TypeError callee, then the call names on the failing source line."""
    blob = "\n".join(x for x in (error, traceback) if x)
    if not blob.strip():
        return ""
    m = _ATTR_ERR.search(blob)
    if m is not None:
        owner = m.group("obj") or m.group("tobj") or m.group("mod") or ""
        attr = m.group("attr")
        digest = symbol_digest(attr, owner=owner, max_chars=max_chars) or symbol_digest(
            attr, max_chars=max_chars
        )
        if digest:
            return digest
    m = _NAME_ERR.search(blob)
    if m is not None:
        digest = symbol_digest(m.group("name").split(".")[0], max_chars=max_chars)
        if digest:
            return digest
    m = _TYPE_CALLEE.search(blob)
    if m is not None:
        digest = _lookup(m.group("fn"), max_chars)
        if digest:
            return digest
    if code and line is not None:
        for callee in reversed(_call_names(code, line)):
            digest = _lookup(callee, max_chars)
            if digest:
                return digest
    return ""
