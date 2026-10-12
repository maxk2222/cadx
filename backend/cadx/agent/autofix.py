"""Deterministic param autofix: correct wrong PARAMS values without an LLM call.

Runs when the numeric dimension check reports mismatches but the model's code
is otherwise fine — the classic "spec says 8 mm, PARAMS has 10" failure. Fixes
by re-executing with `params_override` (the code itself is untouched).

Conservative by design: only fires when the axis->param mapping is unambiguous.
A wrong guess would waste an executor run, not corrupt anything — but skipped
fixes cost nothing either, so we prefer skipping over guessing.
"""

from __future__ import annotations

import re
from typing import Any

from cadx.llm.schemas import PartSpec

_AXIS_KEYS: list[tuple[int, str, list[str]]] = [
    (0, "length/X", [r"length", r"len", r"^x$|_x$|^x_", r"long"]),
    (1, "width/Y", [r"width", r"wid", r"depth", r"dep", r"^y$|_y$|^y_"]),
    (2, "height/Z", [r"height", r"thick", r"^z$|_z$|^z_", r"tall"]),
]

_MAX_FIXES_PER_RUN = 3


def _axis_params(params: dict[str, Any], axis_idx: int) -> list[tuple[str, float]]:
    """Numeric PARAMS whose name plausibly controls the given axis."""
    pats = _AXIS_KEYS[axis_idx][2]
    out = []
    for k, v in params.items():
        if not isinstance(v, int | float) or isinstance(v, bool):
            continue
        kl = k.lower()
        if any(re.search(p, kl) for p in pats):
            out.append((k, float(v)))
    return out


def propose_fixes(
    spec: PartSpec,
    params_used: dict[str, Any],
    analysis: dict[str, Any],
    tol_pct: float,
    tol_mm: float,
) -> dict[str, float]:
    """Return {param_key: new_value} overrides, or {} when no confident fix.

    Confidence rule: exactly one numeric PARAM maps to the mismatched axis AND
    (its value ≈ the wrong measured size — i.e. the model literally encoded the
    wrong number — or no other numeric PARAM could plausibly drive that axis).
    """
    actual = [float(v) for v in analysis.get("bbox_size", [])]
    if len(actual) < 3:
        return {}
    stated = [
        (spec.length_x, 0),
        (spec.width_y, 1),
        (spec.height_z, 2),
    ]
    fixes: dict[str, float] = {}
    for dim, idx in stated:
        if dim.source != "stated":
            continue
        tol = max(dim.value * tol_pct / 100.0, tol_mm)
        if abs(actual[idx] - dim.value) <= tol:
            continue  # axis already within tolerance
        candidates = _axis_params(params_used, idx)
        if len(candidates) != 1:
            continue  # ambiguous — let the LLM handle it
        key, cur = candidates[0]
        value_matches_actual = abs(cur - actual[idx]) <= max(0.05 * actual[idx], 0.5)
        # Only autofix when the param literally holds the wrong measurement
        # (strong signal it's THE axis driver), not a derived/composed value.
        if not value_matches_actual:
            continue
        fixes[key] = float(dim.value)
        if len(fixes) >= _MAX_FIXES_PER_RUN:
            break
    return fixes
