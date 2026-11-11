"""INTAKE — extract a PartSpec from images + text (vision model)."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from cadx.agent.context import (
    images_block,
    parent_block,
    references_block,
    system_prompt,
    user_block,
)
from cadx.agent.thinking import thinking_body
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx
from cadx.llm.schemas import (
    Dim,
    Feature,
    LoadCase,
    OpenQuestion,
    OptimizeSpec,
    PartSpec,
    StandardPartSpec,
)
from cadx.logging import get_logger

log = get_logger(__name__)

STEP = "intake"


class _LoadCasesOut(BaseModel):
    load_cases: list[LoadCase] = Field(default_factory=list)
    optimize: OptimizeSpec | None = None


_ANCHOR_INTENT_RE = re.compile(
    r"\bm\d+\b|flange|corner|boss|seat|insert|tapped?|thread|gewinde|"
    r"mount|plate|bracket|\barm\b|joint|rib|rim|lug|\bear\b",
    re.IGNORECASE,
)


def _anchor_intent_tokens(text: str) -> set[str]:
    """Distinctive tokens a case name / region description carries — bolt
    sizes normalise to 'm<number>' to match anchor bolt_size/name."""
    low = (text or "").lower()
    out = {f"m{m}" for m in re.findall(r"\bm(\d+)\b", low)}
    out |= {t.group(0).lower() for t in _ANCHOR_INTENT_RE.finditer(low)}
    return out


class _Rebinder:
    """State for one `hole_index` rebind pass over a spec's load cases."""

    def __init__(self, anchors: list[Any], spec: Any | None = None) -> None:
        self.anchors = anchors
        self.spec = spec
        self.n = len(anchors)
        self.a_toks = [
            _anchor_intent_tokens(f"{a.name or ''} {a.bolt_size or ''}") for a in anchors
        ]
        self.notes: list[str] = []

    def _best_matches(self, tokens: set[str]) -> list[int]:
        """Indices (1-based) of anchors sharing the most intent tokens."""
        best = 0
        out: list[int] = []
        for i, t in enumerate(self.a_toks):
            s = len(tokens & t)
            if s > best:
                best, out = s, [i + 1]
            elif s == best and s:
                out.append(i + 1)
        return out

    def _feature_pos_match(self, idx: int) -> int | None:
        """Out-of-range `hole_index` rescue via spec feature position.

        Weak models sometimes count `spec.features` positions instead of
        anchor indices — observed keep_regions #8/#9 on a 7-anchor part
        where `spec.features[8]`/`[9]` (0-based) ARE the intended flange
        holes. Try the 0-based then the 1-based feature position; when it
        is hole-like and names exactly one anchor, that is the intended
        anchor. Ambiguous or non-hole hits return None — a dead index
        stays honest."""
        feats = list(getattr(self.spec, "features", None) or [])
        if not feats:
            return None
        from cadx.agent.anchors import _hole_like

        for j in (idx, idx - 1):
            if not (0 <= j < len(feats)):
                continue
            f = feats[j]
            if not _hole_like(f):
                continue
            name = " ".join(str(getattr(f, "name", "") or "").lower().split())
            if not name:
                continue
            hits = [
                i + 1
                for i, a in enumerate(self.anchors)
                if " ".join(str(getattr(a, "name", "") or "").lower().split()) == name
            ]
            if len(hits) == 1:
                return hits[0]
        return None

    def _rebind(self, r: Any, where: str, *, is_fixed: bool) -> Any:
        if r is None or getattr(r, "hole_index", None) is None:
            return r
        if str(getattr(r, "kind", "")) not in ("hole", "anchor", "bore"):
            return r
        idx = int(r.hole_index)
        in_range = 1 <= idx <= self.n
        # a case name describes the LOAD side ("arm load on flange holes") —
        # fixtures named by it are coincidence: only out-of-range fixed
        # refs get re-pointed; an in-range fixture stays where it is.
        if is_fixed and in_range:
            return r
        cur = self.a_toks[idx - 1] if in_range else set()
        tokens = _anchor_intent_tokens(f"{where} {getattr(r, 'description', '') or ''}")
        cur_score = len(tokens & cur)
        matches = self._best_matches(tokens) if tokens else []
        best_score = len(tokens & self.a_toks[matches[0] - 1]) if matches else 0
        if best_score <= cur_score:
            if not in_range:
                m = self._feature_pos_match(idx)
                if m is not None:
                    self.notes.append(
                        f"{where}: hole_index {idx} out of range (max {self.n}) — "
                        f"rebound to #{m} '{self.anchors[m - 1].name}' "
                        "(spec feature position)"
                    )
                    return r.model_copy(update={"hole_index": m})
                avail = ", ".join(
                    f"#{i + 1} {a.name or ''}".strip() for i, a in enumerate(self.anchors)
                )
                self.notes.append(
                    f"{where}: hole_index {idx} has no anchor (only {self.n}: {avail}) "
                    "— the region will match no nodes; re-point or drop it"
                )
            return r
        names = ", ".join(f"#{i} '{self.anchors[i - 1].name}'" for i in matches[:4])
        self.notes.append(
            f"{where}: hole_index {idx} "
            + ("out of range " if not in_range else "pointed at the wrong anchor ")
            + f"(max {self.n}) — rebound to {names}"
        )
        return r.model_copy(update={"hole_index": matches[0]})

    def _split(self, ld: Any, r: Any, where: str, matches: list[int]) -> list[Any]:
        self.notes.append(
            f"{where}: load split over anchors "
            + ", ".join(f"#{m} '{self.anchors[m - 1].name}'" for m in matches[:4])
        )
        return [
            ld.model_copy(update={"region": r.model_copy(update={"hole_index": m})})
            for m in matches[:4]
        ]

    def _rebind_load(self, ld: Any, r: Any, where: str, case_name: str) -> list[Any]:
        """One load entry → one-or-more rewritten entries (plural intent
        splits across every anchor the text names)."""
        tokens = _anchor_intent_tokens(f"{case_name} {r.description or ''}")
        rebound = self._rebind(r, where, is_fixed=False)
        # plural intent ("holes", "both", "all") + several matching anchors
        # → one load entry per anchor — a single bore region can't span two
        plural = bool(
            re.search(
                r"holes|bohrungen|both|all\b|beide",
                f"{case_name} {r.description or ''}",
                re.I,
            )
        )
        anchor_ref = getattr(r, "hole_index", None) is not None and str(getattr(r, "kind", "")) in (
            "hole",
            "anchor",
            "bore",
        )
        if rebound is r and anchor_ref:
            matches = self._best_matches(tokens)
            if plural and len(matches) > 1:
                return self._split(ld, r, where, matches)
        elif rebound is not r and plural:
            extra = [m for m in self._best_matches(tokens) if m != rebound.hole_index]
            if extra:
                out = [ld.model_copy(update={"region": rebound})]
                out += [
                    ld.model_copy(update={"region": r.model_copy(update={"hole_index": m})})
                    for m in extra[:3]
                ]
                return out
        return [ld.model_copy(update={"region": rebound}) if rebound is not r else ld]

    def _face_fix_candidates(self, spec: PartSpec, axis: str, used_loads: set[int]) -> list[int]:
        """Mounting-hole anchors whose bore mouths open on the named face
        side — encastre on a skin is wrong when bolts carry the fixture."""
        letter = axis.lstrip("+-")
        neg = axis.startswith("-")
        dim = {"x": spec.length_x, "y": spec.width_y, "z": spec.height_z}.get(letter)
        try:
            half = float(dim.value or 0) / 2.0 if dim is not None else None
        except (TypeError, ValueError, AttributeError):
            half = None
        out: list[int] = []
        for i, a in enumerate(self.anchors):
            if str(a.axis or "").lstrip("+-") != letter:
                continue
            if a.kind not in ("hole", ""):
                continue  # tapped/threaded bores are load seats, not bolts
            if (i + 1) in used_loads:
                continue  # already a load region in this case
            if half:
                axc = a.z
                if axc is not None:
                    on_side = float(axc) <= half if neg else float(axc) >= half
                    if not on_side:
                        continue
            out.append(i + 1)
        return out

    def _sanitize_case_geometry(self, spec: PartSpec, case: Any) -> None:
        """Region category repairs the index pass can't express: a `g_load`
        is a body force — a skin region (face/plane/bore wall…) is a category
        error; a whole-face encastre fixture on a bolted part pins the skin
        instead of the holes (~27% of the domain at coarse pitch); a pure
        moment on an untapped clearance hole has nothing to react it."""
        where = f"case '{case.name}'"
        for ld in case.loads or []:
            r = getattr(ld, "region", None)
            if r is None:
                continue
            if getattr(ld, "g_load", None) and str(r.kind) not in ("all", "cylinder", "box"):
                self.notes.append(
                    f"{where}: g_load on '{r.kind}' region → 'all' (body force needs material volume)"
                )
                ld.region = r.model_copy(update={"kind": "all"})
                r = ld.region
            if getattr(ld, "moment_nm", None) and str(r.kind) in ("hole", "anchor", "bore"):
                tapped = [
                    i + 1
                    for i, a in enumerate(self.anchors)
                    if a.kind in ("thread", "tap", "insert", "bearing_seat")
                ]
                if tapped:
                    big = max(tapped, key=lambda i: float(self.anchors[i - 1].dia or 0))
                    cur = (
                        self.anchors[r.hole_index - 1]
                        if r.hole_index and 1 <= r.hole_index <= self.n
                        else None
                    )
                    if r.hole_index != big and (
                        cur is None or cur.kind not in ("thread", "tap", "insert", "bearing_seat")
                    ):
                        self.notes.append(
                            f"{where}: moment_nm on untapped bore #{r.hole_index} → "
                            f"#{big} '{self.anchors[big - 1].name}' (only a threaded seat reacts torque)"
                        )
                        ld.region = r.model_copy(update={"hole_index": big})
        used_loads = {
            int(getattr(getattr(ld, "region", None), "hole_index", 0) or 0)
            for ld in case.loads or []
        }
        new_fixed: list[Any] = []
        for r in case.fixed or []:
            if (
                str(r.kind) in ("plane", "face", "surface", "border")
                and getattr(r, "fix_mode", "all") == "all"
                and str(r.axis or "") in ("+x", "-x", "+y", "-y", "+z", "-z")
            ):
                cands = self._face_fix_candidates(spec, str(r.axis), used_loads)
                if len(cands) >= 2:
                    self.notes.append(
                        f"{where}: whole-face {r.kind} {r.axis} encastre → holes "
                        + ", ".join(f"#{i}" for i in cands[:6])
                        + " (mount bolts carry the fixture, not the skin)"
                    )
                    new_fixed += [
                        r.model_copy(
                            update={
                                "kind": "hole",
                                "axis": "",
                                "hole_index": i,
                                "center": [],
                            }
                        )
                        for i in cands[:6]
                    ]
                    continue
            new_fixed.append(r)
        case.fixed = new_fixed

    def run(self, spec: PartSpec) -> list[str]:
        for case in spec.load_cases:
            where = f"case '{case.name}'"
            case.fixed = [self._rebind(r, where, is_fixed=True) for r in (case.fixed or [])]
            new_loads: list[Any] = []
            for ld in case.loads or []:
                r = getattr(ld, "region", None)
                if r is None:
                    new_loads.append(ld)
                    continue
                new_loads += self._rebind_load(ld, r, where, case.name or "")
            if new_loads:
                case.loads = new_loads
            self._sanitize_case_geometry(spec, case)
        if spec.optimize is not None:
            spec.optimize.keep_regions = [
                self._rebind(r, "keep_regions", is_fixed=False)
                for r in (spec.optimize.keep_regions or [])
            ]
        return self.notes


def rebind_case_anchors(spec: PartSpec) -> list[str]:
    """Re-point load-case `hole_index` refs at the anchors they were meant for.

    `hole_index` counts the spec's hole-like features top to bottom — a weak
    model routinely miscounts (observed: `bore #8` on a 7-anchor part — dead
    case — and `bore #4` for the M12 seat — a 15 N·m torque applied to a
    corner mounting hole). The spec-order anchor table is authoritative:
    when the case text names an anchor the index does NOT point at (e.g.
    "M12"/"flange" in the case name vs the indexed anchor's feature name),
    rewire it; a load naming several anchors splits into one entry each.
    Out-of-range refs with no name match stay — the FEM warning is honest.
    """
    from cadx.agent.anchors import _spec_anchors

    anchors = _spec_anchors(spec)
    if not anchors:
        return []
    return _Rebinder(anchors, spec).run(spec)


_AXES = "xyz"
_MODE_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")


def _region_mode_mask(r: Any) -> int:
    """6-bit mask of rigid-body modes a fixed region can pin (tx..tz, rx..rz),
    estimated from kind/fix_mode/axis without a mesh. A hole wall pins
    everything it is told to (encastre); 'bearing' on a bore is the collar
    the solver implements (everything except sliding along the bore axis);
    'normal' pins the one axis DOF. Plane+normal pins the named letters."""
    fix = str(getattr(r, "fix_mode", "all") or "all")
    kind = str(getattr(r, "kind", "") or "")
    if fix == "all":
        return 0b111111
    ax = str(getattr(r, "axis", "") or "+z").lstrip("+-").lower()
    if kind in ("hole", "anchor", "bore") and fix == "bearing":
        idx = _AXES.index(ax[0]) if ax and ax[0] in _AXES else 2
        return 0b111111 & ~(1 << idx)
    # fix_mode 'normal': pin the axis DOF only
    idx = _AXES.index(ax[0]) if ax and ax[0] in _AXES else 2
    return 1 << idx


def lint_support_modes(spec: PartSpec) -> list[str]:
    """Per-case under-constraint lint: union the mode masks of all fixed
    regions; below 6 pinned rigid-body modes the FEM solve is singular.
    Surfaces the free modes by name so the agent can add a real support
    instead of discovering 'unsolvable' at FEM time."""
    notes: list[str] = []
    for case in spec.load_cases or []:
        if not case.fixed:
            continue
        pinned = 0
        for r in case.fixed:
            pinned |= _region_mode_mask(r)
        free = [m for i, m in enumerate(_MODE_NAMES) if not (pinned >> i & 1)]
        if free:
            notes.append(
                f"case '{case.name}': supports leave rigid modes free ({', '.join(free)}) "
                "— the case will be unsolvable; use fix_mode 'all'/'bearing' "
                "or add a second support"
            )
    return notes


_SUPPORT_HOLE_KINDS = ("hole", "anchor", "bore")


def normalize_support_modes(spec: PartSpec) -> list[str]:
    """Upgrade fix_mode 'normal' on hole/anchor/bore fixed regions in
    under-constrained cases. 'normal' on a bore pins only the axial DOF —
    never a real support (roller/symmetry stays valid on plane/face kinds,
    which this never touches). Prefer 'bearing' (collar: pins the two DOFs
    normal to the bore axis, axial slide stays free); escalate to 'all'
    (bolted seat) when the union still leaves rigid modes free — e.g. N
    coaxial mounting bores where the collar keeps the axis free."""
    notes: list[str] = []
    for case in spec.load_cases or []:
        fixed = list(case.fixed or [])
        if not fixed:
            continue
        pinned = 0
        for r in fixed:
            pinned |= _region_mode_mask(r)
        if pinned == 0b111111:
            continue  # already constrained — a stray 'normal' hole adds a
            # harmless axial pin; preserve the declared intent
        normal_holes = [
            r
            for r in fixed
            if str(getattr(r, "fix_mode", "all") or "all") == "normal"
            and str(getattr(r, "kind", "") or "") in _SUPPORT_HOLE_KINDS
        ]
        if not normal_holes:
            continue  # under-constrained for another reason — lint reports it
        # rebuild the mask as it would stand AFTER a collar upgrade — the
        # 'normal' axial pin is lost, so seeding from `pinned` would keep a
        # DOF the new support no longer provides
        probe = 0
        for r in fixed:
            probe |= _region_mode_mask(
                r.model_copy(update={"fix_mode": "bearing"}) if r in normal_holes else r
            )
        target = "bearing" if probe == 0b111111 else "all"
        case.fixed = [
            r.model_copy(update={"fix_mode": target}) if r in normal_holes else r for r in fixed
        ]
        names = ", ".join(str(getattr(r, "description", "") or r.kind) for r in normal_holes[:4])
        notes.append(
            f"case '{case.name}': fix_mode 'normal' on {names} pins only the "
            f"bore-axial DOF — upgraded to '{target}' "
            + (
                "(collar — axial slide stays free)"
                if target == "bearing"
                else "(bolted seat — 'normal' on a hole is never a support)"
            )
        )
    return notes


def drop_dead_anchor_refs(spec: PartSpec) -> list[str]:
    """Drop regions whose `hole_index` points past the spec anchor table.

    The rebinder rewires miscounts it can identify by name/position; what is
    left is a genuinely dead reference the agent cannot repair post-intake
    (load_cases are not editable mid-loop). A dead region only ever produces
    a 'matched no nodes' warning that poisons the whole case — dropping it
    lets the case evaluate on its VALID supports/loads, or fail with the
    honest free-mode lint. Loads lose their target entirely, so the load
    entry itself is dropped."""
    from cadx.agent.anchors import _spec_anchors

    n = len(_spec_anchors(spec) or [])
    if n == 0:
        return []  # no anchors to validate against — unbound refs resolve by hint
    notes: list[str] = []

    def _dead(r: Any) -> bool:
        idx = int(getattr(r, "hole_index", 0) or 0)
        return idx > n or idx < 0  # 0/None means unbound — kept (resolved by hint)

    for case in spec.load_cases or []:
        where = f"case '{case.name}'"
        kept = []
        for r in case.fixed or []:
            if _dead(r):
                notes.append(
                    f"{where}: fixed region hole_index {r.hole_index} dropped — "
                    f"only {n} anchor(s) exist; case evaluates on remaining supports"
                )
            else:
                kept.append(r)
        case.fixed = kept
        kept_l = []
        for ld in case.loads or []:
            r = getattr(ld, "region", None)
            if r is not None and _dead(r):
                notes.append(
                    f"{where}: load '{getattr(ld, 'description', '') or 'unnamed'}' dropped — "
                    f"its region hole_index {r.hole_index} exceeds the {n} anchor(s)"
                )
            else:
                kept_l.append(ld)
        case.loads = kept_l
    if spec.optimize is not None:
        spec.optimize.keep_regions = [r for r in (spec.optimize.keep_regions or []) if not _dead(r)]
    return notes


def lint_region_semantics(spec: PartSpec) -> list[str]:
    """Cavity-pressure semantics: a `cavity` region only resolves on a sealed
    interior void — an open neck/nozzle floods the mask empty, which produces
    a silent no-load case at FEM time. When a case puts pressure inside a
    vessel that has an opening (hydro test with capped ports), the region
    must carry `seal_openings_mm` so the solver treats the port as capped."""
    notes: list[str] = []
    for case in spec.load_cases or []:
        regions = list(case.fixed or []) + [
            getattr(ld, "region", None) for ld in (case.loads or [])
        ]
        for r in regions:
            if r is None or str(getattr(r, "kind", "") or "") != "cavity":
                continue
            if getattr(r, "seal_openings_mm", None) is None:
                notes.append(
                    f"case '{case.name}': cavity region '{getattr(r, 'description', '') or 'inner'}' "
                    "resolves only a SEALED interior void — an open nozzle floods it "
                    "empty (load matches nothing). If the vessel is open and the "
                    "pressure case models a capped/hydro test, set "
                    "seal_openings_mm to the largest opening Ø in mm"
                )
    return notes


_MAX_ASSUMPTION_NOTES = 6

_HOLE_REGION_KINDS = ("hole", "anchor", "bore")


def _anchor_plane_coord(a: Any, letter: str) -> float | None:
    """In-plane coordinate of an anchor on one axis — the AXIAL coordinate
    of a bore is not envelope evidence (a side bore may sit on the face)."""
    ax = str(getattr(a, "axis", "+z") or "+z").lstrip("+-")[-1:]
    if ax == letter:
        return None
    pos = getattr(a, "pos", None) or []
    idx = {"x": 0, "y": 1, "z": 2}[letter]
    if len(pos) > idx and pos[idx] is not None:
        return float(pos[idx])
    if letter == "x":
        return float(getattr(a, "x", 0.0) or 0.0)
    if letter == "y":
        return float(getattr(a, "y", 0.0) or 0.0)
    return None  # z without pos is not resolvable


def _expand_envelope_dim(spec: PartSpec, field: str, letter: str, req: float) -> str | None:
    """Expand one envelope dim to `req` — or emit the honest warning when
    the implied size is absurd (parse error) instead of trusting it."""
    dim = getattr(spec, field, None)
    try:
        declared = float(getattr(dim, "value", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if declared <= 0 or req <= declared:
        return None
    req += 2.0 if letter == "z" else 4.0  # headroom / edge margin
    if req > 2000.0:
        return (
            f"spec envelope {field}={declared} cannot contain declared anchors "
            f"(positions imply ≥{req:.0f} mm) — envelope/positions need review"
        )
    if req > declared * 10.0:
        return (
            f"spec envelope {field}={declared} vs anchors imply ≥{req:.0f} mm "
            "(>10x — likely a parse error, kept as declared)"
        )
    try:
        dim.value = round(req, 2)
    except (AttributeError, TypeError):
        return None
    return (
        f"spec envelope {field} expanded {declared:g}→{dim.value:g} mm — declared "
        "anchors don't fit otherwise (self-contradictory spec reconciled)"
    )


def lint_envelope_fit(spec: PartSpec) -> list[str]:
    """Declared envelope must contain the declared anchors.

    A weak model occasionally lets a catalog-part dimension bleed into the
    top-level envelope (observed: SocketHeadCapScrew 18.27 mm governing
    `length_x` while its own holes sat at x=±42) — the spec is then
    self-contradictory and any honest part fails `envelope_*` regardless of
    code quality. Positions are resolved on the IN-PLANE axes of each bore
    (a side-face bore legitimately sits AT the envelope face on its own
    axis, so its axial coordinate is not evidence). Only ever EXPANDS the
    envelope to fit declared intent — shrinking would silently cut
    declared geometry; expansions >10x are almost certainly a parse error,
    so those only warn."""
    from cadx.agent.anchors import _spec_anchors, _spec_bounds

    anchors = _spec_anchors(spec)
    bounds = _spec_bounds(spec)
    if not anchors or bounds is None:
        return []

    need: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
    for a in anchors:
        r = max(float(getattr(a, "dia", 0.0) or 0.0) / 2.0, 0.0)
        for letter in ("x", "y", "z"):
            c = _anchor_plane_coord(a, letter)
            if c is not None:
                need[letter] = max(need[letter], abs(c) + r)

    notes: list[str] = []
    for field, letter in (("length_x", "x"), ("width_y", "y"), ("height_z", "z")):
        req = 2.0 * need[letter] if letter != "z" else need[letter]
        if req > 0 and (note := _expand_envelope_dim(spec, field, letter, req)):
            notes.append(note)
    return notes


def _anchor_hint_update(r: Any, anchors: list[Any]) -> dict[str, Any] | None:
    """Region-update dict materializing a `hole_index` ref into a geometric
    (axis, diameter_mm, center) hint — and unbinding the index once the
    effective hint is complete. Returns None when the region can't be
    safely re-pointed (unresolvable anchor position, estimated coords)."""
    idx = int(getattr(r, "hole_index", 0) or 0)
    if str(getattr(r, "kind", "") or "") not in _HOLE_REGION_KINDS:
        return None
    if not (1 <= idx <= len(anchors)):
        return None
    a = anchors[idx - 1]
    if str(getattr(a, "source", "") or "") == "spec-est":
        return None  # estimated position is weaker evidence than the index
    letter = str(getattr(a, "axis", "+z") or "+z").lstrip("+-")[-1:]
    ax_idx = {"x": 0, "y": 1, "z": 2}[letter]
    plane_idx = [i for i in range(3) if i != ax_idx]
    pos = getattr(a, "pos", None) or []
    center: list[float] | None = None
    if len(pos) == 3 and all(pos[i] is not None for i in plane_idx):
        # in-plane slots resolve — the axial slot may stay None
        center = [float(v) if v is not None else 0.0 for v in pos]
    elif letter == "z":
        # a declared (0,0) bore is a real position — the source check
        # above already filtered mere estimates
        center = [float(a.x or 0.0), float(a.y or 0.0), 0.0]
    if center is None:
        return None  # no resolvable in-plane position — index is all we have
    upd: dict[str, Any] = {}
    if not getattr(r, "axis", None):
        upd["axis"] = a.axis
    if not getattr(r, "diameter_mm", None) and getattr(a, "dia", 0.0):
        upd["diameter_mm"] = float(a.dia)
    if not getattr(r, "center", None):
        upd["center"] = center
    # unbind the index once the effective region carries a complete hint
    # (centre + axis letter agreeing with the anchor) — whether the fields
    # came from the model or were just materialized. A conflicting
    # model-set axis keeps the index: the two candidates disagree and
    # neither is clearly authoritative.
    final_axis = upd.get("axis") or getattr(r, "axis", None)
    final_center = upd.get("center") or getattr(r, "center", None)
    if final_center and str(final_axis or "").lstrip("+-")[-1:] == letter:
        upd["hole_index"] = None
    return upd or None


def materialize_anchor_hints(spec: PartSpec) -> list[str]:
    """Fill empty axis/diameter_mm/center on `hole_index` regions from the
    spec anchor table, then unbind the index.

    `hole_index` binds 1-based into the MEASURED anchor list at FEM time,
    but intake writes it spec-order — when measured count/order diverges
    (extra holes found, blind bores missed) the index pins the WRONG bore
    or a phantom anchor whose ring has no nodes ("matched no nodes" →
    unsolvable case). Writing the intended anchor's (axis, dia, in-plane
    centre) into the region and clearing the index routes binding through
    the deterministic (axis+dia+centre) hint match instead — strictly more
    robust under count/order drift. Regions whose spec anchor has no
    resolvable position keep their index (measured order is the only
    remaining evidence)."""
    from cadx.agent.anchors import _spec_anchors

    anchors = _spec_anchors(spec)
    if not anchors:
        return []
    notes: list[str] = []

    def _repoint(r: Any, where: str) -> Any:
        idx = int(getattr(r, "hole_index", 0) or 0)
        upd = _anchor_hint_update(r, anchors)
        if upd is None:
            return r
        if "hole_index" in upd and idx:
            notes.append(
                f"{where}: hole_index {idx} → anchor hint "
                f"(axis={upd.get('axis') or r.axis}, pos={upd.get('center') or r.center}) — "
                "spec-order index would bind the wrong measured bore"
            )
        return r.model_copy(update=upd)

    for case in spec.load_cases or []:
        where = f"case '{case.name}'"
        case.fixed = [_repoint(r, where) for r in (case.fixed or [])]
        for ld in case.loads or []:
            r = getattr(ld, "region", None)
            if r is not None:
                ld.region = _repoint(r, where)
    if spec.optimize is not None:
        spec.optimize.keep_regions = [
            _repoint(r, "keep_regions") for r in (spec.optimize.keep_regions or [])
        ]
    return notes


def sanitize_load_cases(spec: PartSpec) -> None:
    """Drop degenerate cases (no supports and no loads) a weak model emits as
    `{"name":"","fixed":[],"loads":[]}` — truthy lists of those would skip the
    extraction fallback and produce unsolvable FEM runs. Also clamps bogus
    negative case weights back to neutral, rebinds `hole_index` region
    refs onto the anchors the case text actually names, and upgrades
    `fix_mode 'normal'` on hole-kind supports in under-constrained cases —
    a bore 'normal' pins only the axial DOF and is never a real support."""
    lint_notes = (
        lint_envelope_fit(spec)
        + rebind_case_anchors(spec)
        + materialize_anchor_hints(spec)
        + normalize_support_modes(spec)
        + lint_support_modes(spec)
        + lint_region_semantics(spec)
        + drop_dead_anchor_refs(spec)
    )
    for note in lint_notes:
        log.info("intake.spec_lint", note=note)
    if lint_notes:
        prior = spec.assumptions or []
        # the recovery path sanitizes a second time — dedupe new-vs-prior so
        # the same diagnostics can't pile up across calls
        spec.assumptions = prior + [n for n in dict.fromkeys(lint_notes) if n not in prior]
    # the assumptions tail serializes into the codegen prompt via spec_block —
    # bound it so per-case lint diagnostics can't dilute the generation
    # instructions; legit (upstream) assumptions stay ahead of the lint notes.
    # "researched:" notes are pipeline-side sourced evidence (pre-intake
    # research step) — they never spend the model's assumption budget and
    # must survive this trim on every (re-)sanitize pass.
    if spec.assumptions:
        uniq = list(dict.fromkeys(spec.assumptions))
        researched = [n for n in uniq if n.startswith("researched:")]
        spec.assumptions = [n for n in uniq if not n.startswith("researched:")][
            :_MAX_ASSUMPTION_NOTES
        ] + researched
    kept: list[LoadCase] = []
    for c in spec.load_cases:
        if not c.fixed and not c.loads:
            continue
        if c.weight <= 0:
            c.weight = 1.0
        kept.append(c)
    spec.load_cases = kept


_MATERIAL_RE = re.compile(
    r"\b(petg|pet-g|pla|abs|asa|nylon|pa6|pa12|\bpa\b|pc|resin|"
    r"aluminium|aluminum|alu|steel|stahl|s235)\b",
    re.IGNORECASE,
)


def sanitize_material(spec: PartSpec, prompt: str) -> None:
    """Fill `spec.material` when the prompt names a material the model dropped.

    FEM/topopt fall back to the default (PLA) on an empty field — a declared
    "PETG" or "aluminium" then silently verifies against wrong E/yield. Scans
    the prompt and the generated part name for the material table's known
    tokens; leaves an explicitly set value untouched."""
    if (spec.material or "").strip():
        return
    for m in _MATERIAL_RE.finditer(f"{prompt or ''} {spec.name or ''}"):
        token = m.group(1).lower()
        try:
            from cadx.cad.fem.materials import resolve_material

            # keep the literal token (alias-aware) — display names like
            # "PA (nylon)" do not re-resolve downstream
            resolve_material(token)
            spec.material = token
            return
        except Exception:
            continue


_MM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm\b", re.IGNORECASE)
_DIMSET_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)\s*mm",
    re.IGNORECASE,
)


def _prompt_mm_values(prompt: str) -> list[float]:
    """Explicit millimetre values mentioned in the user prompt, largest first."""
    vals: set[float] = set()
    for m in _MM_RE.finditer(prompt):
        vals.add(float(m.group(1).replace(",", ".")))
    for m in _DIMSET_RE.finditer(prompt):
        vals.update(float(g.replace(",", ".")) for g in m.groups())
    # unitless "A x B x C" triplets also carry numbers — the shared parser
    # already rejects thread/size designations ("M10x1.5x100", "Ø26x8x4")
    for t in _dimset_triplets(prompt):
        vals.update(t)
    return sorted(vals, reverse=True)


def sanitize_dims(spec: PartSpec, prompt: str) -> list[str]:
    """Recover an overall dimension the model dropped or misread.

    Weak models routinely lose an explicitly stated size ("approx 130 mm
    long") or substitute a feature size — downstream, scale_to_spec then fits
    the generated mesh to the wrong magnitude (a 130 mm grip becomes a 40 mm
    nugget). When the largest explicit mm value in the prompt exceeds every
    stated dim by a wide margin and length_x is not itself stated, re-inject
    it as the stated length."""
    notes: list[str] = []
    vals = _prompt_mm_values(prompt)
    if not vals:
        return notes
    # the largest prompt number is not automatically an envelope — "two
    # holes 30 mm apart" must not re-state length_x=30 (observed on the
    # NEMA-17 case: sanitize_stated_dims downgraded the label, then this
    # recovery re-run resurrected it). Only envelope-context values count.
    # And only X-evidenced ones may re-state length_x: "Ø80 x 120 tall"
    # puts its largest number on Z, not on the part's length.
    x_vals = _prompt_x_extent_values(prompt)
    env_val = next(
        (v for v in vals if v in x_vals and _dim_stated_in_prompt(v, prompt)),
        None,
    )
    if env_val is None:
        return notes
    lx = spec.length_x
    if lx.source == "stated" and lx.value > 0:
        return notes  # the model stated an overall length — trust it
    stated_max = max(
        (
            float(d.value)
            for d in (spec.length_x, spec.width_y, spec.height_z)
            if d is not None and d.source == "stated" and d.value > 0
        ),
        default=0.0,
    )
    if stated_max >= env_val:
        # a stated dim at/above the envelope value means the model already
        # pinned the overall size — env_val is a sub-dimension of it.
        return notes
    notes.append(
        f"prompt states ~{env_val:g} mm but intake had length_x="
        f"{lx.value:g} {lx.source} — re-stated"
    )
    spec.length_x = lx.model_copy(update={"value": env_val, "source": "stated"})
    return notes


_AXIS_COORD_RE = re.compile(r"\b([xyz])\s*=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_AXIS_RANGE_RE = re.compile(
    r"\b([xyz])\s*=\s*([+-]?\d+(?:\.\d+)?)\s*(?:bis|to|\u2013|\u2014|\.\.)\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_APEX_RE = re.compile(
    r"\b(?:scheitel|apex|peak|oberkante|top)\s*(?:bei|at)?\s*([xyz])?\s*=?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _feature_axis_extent(spec: PartSpec, axis: str) -> float:
    """Largest |coordinate| the features claim along `axis` — from position
    ranges ("x=-60 bis -30"), single coords and apex notes ("Scheitel z=30").
    For z (0-based frame) the extent IS the max coordinate; for x/y the
    caller compares against half the declared span."""
    extent = 0.0
    for f in spec.features:
        pos = str(f.position or "")
        for m in _AXIS_RANGE_RE.finditer(pos):
            if m.group(1).lower() == axis:
                extent = max(extent, abs(float(m.group(2))), abs(float(m.group(3))))
        for m in _AXIS_COORD_RE.finditer(pos):
            if m.group(1).lower() == axis:
                extent = max(extent, abs(float(m.group(2))))
        m = _APEX_RE.search(pos)
        if m and (m.group(1) or "z").lower() == axis:
            extent = max(extent, float(m.group(2)))
    return extent


def sanitize_dim_echo(spec: PartSpec, prompt: str) -> list[str]:
    """Recover a declared bbox dim that echoes another axis's value while the
    spec's own feature positions contradict it (observed: '120 x 40 x 30'
    mis-parsed as height_z=120 duplicating length_x, while every feature
    said z<=30). Only fires on the echo signature + a large contradiction —
    features legitimately do not always span the full envelope, so a mere
    mismatch is not enough. The replacement must appear in the prompt text.
    """
    notes: list[str] = []
    dims = {"x": spec.length_x, "y": spec.width_y, "z": spec.height_z}
    stated = {
        ax: float(d.value)
        for ax, d in dims.items()
        if d is not None and d.source == "stated" and float(d.value or 0) > 0
    }
    if len(stated) < 2:
        return notes
    prompt_vals = set(_prompt_mm_values(prompt or ""))
    for ax, v in stated.items():
        echoes = any(v == w for other, w in stated.items() if other != ax)
        if not echoes:
            continue
        extent = _feature_axis_extent(spec, ax)
        if not extent:
            continue
        # z positions are 0-based extents; x/y are centred half-extents
        expected = extent if ax == "z" else extent * 2.0
        if v <= expected * 1.5:
            continue
        # prompt values no stated dim claims — the strongest ground truth for
        # what the echoing dim was meant to be (e.g. "120 x 40 x 90" parsed
        # as z=120: 90 is the orphaned truth)
        unmatched = [p for p in prompt_vals if not any(abs(p - w) < 1.0 for w in stated.values())]
        if unmatched:
            new_v = min(unmatched, key=lambda p: abs(p - expected))
        elif not prompt_vals or any(abs(expected - p) < 1.0 for p in prompt_vals):
            new_v = expected  # feature extent has footing or nothing to check against
        else:
            continue  # prompt states dims and the smaller extent is not among them
        notes.append(
            f"{ax} dim {v:g} duplicates another axis but features only reach "
            f"{expected:g} — re-stated {ax}={new_v:g}"
        )
        d = dims[ax]
        setattr(
            spec,
            {"x": "length_x", "y": "width_y", "z": "height_z"}[ax],
            d.model_copy(update={"value": new_v}),
        )
    return notes


_DIMSET_ANY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)


def _dimset_triplets(prompt: str) -> list[tuple[float, float, float]]:
    """Explicit ``A x B x C`` triplets in the prompt (mm optional).

    Thread/size designations like ``M10x1.5x100`` or ``Ø26x8x4`` are
    excluded — the first number glued to a leading M/m/Ø is a nominal
    diameter, not an envelope."""
    trips: list[tuple[float, float, float]] = []
    for m in _DIMSET_ANY_RE.finditer(prompt or ""):
        before = (prompt or "")[: m.start()].rstrip()
        if before[-1:].upper() in ("M", "\u00d8", "\u2300"):
            continue
        trips.append(tuple(float(g.replace(",", ".")) for g in m.groups()))
    return trips


def sanitize_dimset(spec: PartSpec, prompt: str) -> list[str]:
    """Raise an understated envelope dim to an explicit prompt triplet.

    Weak models smear a feature dimension into an envelope dim — observed
    twice: "Steel cube 40 x 40 x 40, bore 20 deep" parsed as height_z=20
    (the bore depth), producing a 40x40x20 slab. An explicitly written
    triplet is strong evidence for the envelope. Correction is raise-only:
    a stated dim larger than the triplet member is left alone — prompts may
    carry feature triplets ("base plate 100 x 80 x 10" in a taller part)
    and the smear failure mode is always understatement. When several
    triplets appear, the largest-volume one anchors the check."""
    notes: list[str] = []
    trips = _dimset_triplets(prompt)
    if not trips:
        return notes
    env = max(trips, key=lambda t: t[0] * t[1] * t[2])
    dims = {"x": spec.length_x, "y": spec.width_y, "z": spec.height_z}
    for ax, want in zip("xyz", env, strict=True):
        d = dims[ax]
        if d is None:
            continue
        have = float(d.value or 0)
        if have > 0 and want <= have + max(0.5, 0.02 * have):
            continue
        notes.append(
            f"{ax} dim {have:g} ({d.source}) below prompt triplet "
            f"{' x '.join(f'{v:g}' for v in env)} — re-stated {ax}={want:g}"
        )
        setattr(
            spec,
            {"x": "length_x", "y": "width_y", "z": "height_z"}[ax],
            d.model_copy(update={"value": want, "source": "stated"}),
        )
    return notes


_ENV_PAIR_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
_ENV_KW_RE = re.compile(
    r"\b(?:width|wide|length|long|height|high|tall|depth|deep|thick(?:ness)?|"
    r"envelope|overall|footprint|bbox|size|outer|span|"
    r"breit(?:e)?|l\u00e4nge|lang|h\u00f6he|hoch|tiefe?|dick(?:e|er)?|"
    r"durchmesser|abmessung(?:en)?|umriss)\b",
    re.I,
)
_NON_ENV_RE = re.compile(
    r"\b(?:apart|spacing|pitch|distance|offset|centered|centred|"
    r"hole|holes|bore|drill|thread|tap|screw|bolt|insert|nut|slot|"
    r"abstand|versatz|l\u00f6ch|loch|bohr|gewinde|schraub|zentri)\b",
    re.I,
)
_NEAR_CHARS = 20

_LEN_AFTER_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:mm\s*)?(?:long|length|lang(?:e)?)\b", re.IGNORECASE
)
_LEN_BEFORE_RE = re.compile(
    r"\b(?:length|länge|lang)\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(?:mm\b)?", re.IGNORECASE
)
_DIA_PREFIX_RE = re.compile(r"[\u00d8\u2300]\s*(\d+(?:[.,]\d+)?)")


def _prompt_x_extent_values(prompt: str) -> set[float]:
    """Prompt numbers that pin the X extent specifically.

    A generic "stated" value is not enough — 'Ø80 x 120 tall' puts the
    120 on Z, and re-stating it as length_x would double the part's
    length (observed failure mode). Only first triplet members ("L x W x H"),
    first pair members ("L x W"), Ø-prefixed values (diameter spans X) and
    explicit length keywords count as X evidence."""
    text = prompt or ""
    out: set[float] = set()
    for t in _dimset_triplets(text):
        out.add(t[0])
    for m in _ENV_PAIR_RE.finditer(text):
        out.add(float(m.group(1).replace(",", ".")))
    for m in _DIA_PREFIX_RE.finditer(text):
        out.add(float(m.group(1).replace(",", ".")))
    for rx in (_LEN_AFTER_RE, _LEN_BEFORE_RE):
        for m in rx.finditer(text):
            out.add(float(m.group(1).replace(",", ".")))
    return out


def _dim_stated_in_prompt(value: float, prompt: str) -> bool:
    """Whether ``value`` is an envelope-bearing number in ``prompt``.

    Membership in an explicit ``A x B x C`` / ``A x B`` extent, a ``Ø``
    prefix, or a size keyword next to the number counts. A number that
    only appears in feature contexts ("30 mm apart", "holes", "M3") is
    NOT a stated envelope — it is a spacing, pattern or bore spec that
    must never pin the part envelope."""
    text = prompt or ""
    if any(value in t for t in _dimset_triplets(text)):
        return True
    for m in _ENV_PAIR_RE.finditer(text):
        if value in {
            float(m.group(1).replace(",", ".")),
            float(m.group(2).replace(",", ".")),
        }:
            return True
    weak = False
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)", text):
        try:
            if abs(float(m.group(1).replace(",", ".")) - value) > 0.001:
                continue
        except ValueError:
            continue
        near = text[max(0, m.start() - _NEAR_CHARS) : m.end() + _NEAR_CHARS]
        if "\u00d8" in near[:_NEAR_CHARS] or "\u2300" in near[:_NEAR_CHARS]:
            return True
        if _ENV_KW_RE.search(near):
            return True
        wide = text[max(0, m.start() - 60) : m.end() + 60]
        if not _NON_ENV_RE.search(wide):
            weak = True
    return weak


def _feature_axis_extents(spec: PartSpec) -> dict[str, float]:
    """Largest span the declared features need per axis — singles, ±pairs
    and centre-spread endpoints folded through _feat_coords. x/y are
    centred so the span is twice the largest |coord|; z is 0-based."""
    ext: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
    for f in spec.features or []:
        coords = _feat_coords(f"{f.position or ''} {f.description or ''}")
        for ax, vals in coords.items():
            ext[ax] = max(ext[ax], max((abs(v) for v in vals), default=0.0))
    return ext


def _dim_envelope_evidenced(ax: str, v: float, prompt: str) -> bool:
    """Is ``v`` an explicitly stated envelope number for axis ``ax`` in the
    prompt? Only ``L x W x H`` triplet positions count — pairs/Ø values are
    local feature dims ("boss dia 25 x 15"), never envelope claims. A kept
    'stated' that turns out wrong still fails verify_spec loudly, which is
    the intended outcome over a silent UNVERIFIED."""
    if not (prompt or "").strip():
        return False
    idx = {"x": 0, "y": 1, "z": 2}[ax]
    return any(len(t) > idx and t[idx] == v for t in _dimset_triplets(prompt))


def sanitize_stated_dims(spec: PartSpec, prompt: str) -> list[str]:
    """Downgrade ``source="stated"`` envelope dims the spec cannot honour.

    Two honest-label audits: (1) a stated value must be an envelope-bearing
    prompt number — "two Ø4.5 through-holes about 30 mm apart" produced
    ``length_x=30 stated`` and verify_spec honestly FAILED a correct 48 mm
    part; (2) a stated dim the spec's own feature extents contradict is a
    wrong-axis assignment — "length_x=20 stated" (the prompt's "~20 mm deep"
    flange value, a Y extent) while features declare x=±15.5 → x needs ≥31.
    'stated' is a hard contract in verify_spec; a guessed or misplaced dim
    keeps ``inferred`` so the check degrades to UNVERIFIED instead of
    enforcing the model's own assumption. Runs after the dim upgrades so
    only surviving labels are audited; the prompt audit is skipped without
    a prompt so edit / resume paths keep earlier stated dims."""
    notes: list[str] = []
    ext = _feature_axis_extents(spec)
    for ax, attr in (("x", "length_x"), ("y", "width_y"), ("z", "height_z")):
        d = getattr(spec, attr, None)
        if d is None or getattr(d, "source", "") != "stated":
            continue
        try:
            v = float(d.value or 0)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        need = 2.0 * ext[ax] if ax in "xy" else ext[ax]
        if need > v + max(0.5, 0.02 * v):
            # the dim contradicts the spec's own feature extents. Normally the
            # dim is the wrong-axis guess — downgrade it. But when the value is
            # an explicitly prompt-stated envelope number, the FEATURE is the
            # offender (arm r24: flange authored at x=50..58 against stated 100
            # — downgrading hid the contradiction: envelope_x went UNVERIFIED
            # and the loop shipped the 108 mm part). Keep 'stated' so
            # verify_spec fails loudly and the improve loop repairs the feature.
            if _dim_envelope_evidenced(ax, v, prompt):
                notes.append(
                    f"{ax} dim {v:g} stated but own features reach "
                    f"{ax}={ext[ax]:g} (need ≥{need:g}) — prompt-stated "
                    f"envelope kept; features must obey it"
                )
                continue
            notes.append(
                f"{ax} dim {v:g} stated but own features reach "
                f"{ax}={ext[ax]:g} (need ≥{need:g}) — wrong-axis value, "
                f"downgraded to inferred"
            )
            setattr(spec, attr, d.model_copy(update={"source": "inferred"}))
            continue
        if (prompt or "").strip() and not _dim_stated_in_prompt(v, prompt):
            notes.append(
                f"{ax} dim {v:g} labelled 'stated' but never appears as an "
                f"envelope value in the prompt — downgraded to inferred"
            )
            setattr(spec, attr, d.model_copy(update={"source": "inferred"}))
    return notes


_VOID_FEATURE_TYPES = {
    "hole",
    "bore",
    "void",
    "slot",
    "pocket",
    "cutout",
    "tunnel",
    "channel",
    "opening",
    "insert",
    "thread",
    "bearing_seat",
    "through_hole",
    "blind_hole",
    "countersink_hole",
    "counterbore_hole",
    "clearance_hole",
    "tapped_hole",
    "notch",
    "groove",
    "recess",
    "cavity",
}
_SOLID_LEAD_RE = re.compile(r"\b(?:solid|massiv|voll)\b", re.I)
_SOLID_NOUN_RE = re.compile(
    r"\b(?:shaft|body|boss|rib|plate|web|wall|flange|collar|spacer|lug|arm|"
    r"post|stud|pin|block|slug|rod|bar|core|deck|frame|gusset|hub|"
    r"welle|koerper|k\u00f6rper|rippe|platte|flansch|bolzen|steg|wand)\b",
    re.I,
)
_VOID_CONTEXT_RE = re.compile(
    r"\b(?:hole|holes|bore|void|opening|drill|drilled|tap|tapped|thread|"
    r"slot|pocket|cutout|cut\s*out|through|throughout|clearance|recess|"
    r"cavity|tunnel|channel|notch|groove|bohrung|loch|aussparung|"
    r"durchgang|senkung|gewinde|fenster)\b",
    re.I,
)


def sanitize_feature_kinds(spec: PartSpec) -> list[str]:
    """Re-type void-kind features that actually describe solid material.

    Weak intake models label a declared SOLID element as ``hole`` when its
    only dimension is a diameter — observed: "shaft body … dia 16" became
    a 16 mm hole feature; the verify layer then demanded the missing void
    and failed the part the model built correctly (the critique itself
    flagged the finding as a false positive). A void type plus an
    unambiguous solid description — 'solid', or a body noun in the name
    (shaft/body/boss/plate/…) with no void word anywhere in the text —
    is retyped ``other`` so the spec stops requesting a hole that was
    never meant to exist. Anything with a void word in name or
    description ("hole for the shaft") is left alone.
    """
    notes: list[str] = []
    for f in spec.features or []:
        ftype = str(getattr(f, "type", "") or "").lower()
        if ftype not in _VOID_FEATURE_TYPES:
            continue
        name = str(getattr(f, "name", "") or "")
        desc = str(getattr(f, "description", "") or "")
        if _VOID_CONTEXT_RE.search(f"{name} {desc}"):
            continue
        if not (_SOLID_LEAD_RE.search(f"{name} {desc}") or _SOLID_NOUN_RE.search(name)):
            continue
        f.type = "other"
        notes.append(
            f"feature '{name or ftype}' described solid material but was "
            f"typed {ftype} — retyped 'other'"
        )
    return notes


_FEAT_AXIS_RE = re.compile(
    r"\balong\s*(?:the\s*)?[+-]?\s*([xyz])\b|\b([xyz])\s*[-_]?\s*axis\b|\bvertical\b",
    re.I,
)


def _feat_axis_word(text: str) -> str | None:
    m = _FEAT_AXIS_RE.search(text)
    if not m:
        return None
    if m.group(0).lower() == "vertical":
        return "z"
    return (m.group(1) or m.group(2) or "").lower() or None


def _feat_coords(text: str) -> dict[str, frozenset[float]]:
    """Per-axis coordinate values a feature declares — ``x=v`` singles,
    ``x=±v`` symmetric pairs and ``z=c±s`` centre-spread endpoints all
    reduce to point values so differently phrased declarations of the same
    pattern compare equal."""
    from cadx.cad.poshint import _CENTER_SPREAD_RE, _POS_COORD_RE, SYM_COORD_RE

    vals: dict[str, set[float]] = {}
    masked = [m.span() for m in _CENTER_SPREAD_RE.finditer(text)]
    masked += [m.span() for m in SYM_COORD_RE.finditer(text)]
    for m in _CENTER_SPREAD_RE.finditer(text):
        ax, c, s = m.group(1).lower(), float(m.group(2)), float(m.group(3))
        vals.setdefault(ax, set()).update((c - s, c + s))
    for m in SYM_COORD_RE.finditer(text):
        ax, v = m.group(1).lower(), float(m.group(2))
        vals.setdefault(ax, set()).update((v, -v))
    for m in _POS_COORD_RE.finditer(text):
        if any(s <= m.start() < e for s, e in masked):
            continue
        vals.setdefault(m.group(1).lower(), set()).add(float(m.group(2)))
    return {ax: frozenset(v) for ax, v in vals.items()}


_FEATURE_INDEX_RE = re.compile(r"^(.*?)[\s_#-]*\d+$")


def _feat_base_name(name: str | None) -> str:
    """Feature name minus a trailing enumeration index — 'foot lug 2'
    and 'foot lug 3' share base 'foot lug', which marks distinct
    instances the model numbered, not a re-declaration."""
    m = _FEATURE_INDEX_RE.match((name or "").strip().lower())
    return (m.group(1) if m else (name or "").strip().lower()).strip()


def dedup_spec_features(spec: PartSpec) -> list[str]:
    """Merge features the model declared twice under different phrasing.

    Observed on the NEMA-17 case: the spec listed the 4xM3 pattern and the
    2xM4 foot holes twice each ("x=±15.5, z=26±15.5" vs "(±15.5, ±15.5)"),
    inflating the declared feature count so spec_features honestly failed
    against the 3 real measured groups. Two entries are duplicates when
    type, bore axis word, count and dims agree and no shared coordinate
    axis contradicts — an axis declared in only one phrasing cannot
    contradict. With no shared axis at all there is no coordinate
    evidence, so only merge a whole-pattern re-declaration (count>1) or
    identically-named entries — index-enumerated names ('foot lug 2' vs
    'foot lug 3') are distinct placed instances, never duplicates.
    First entry wins; the merge is a sanitize note, not a feature_drop —
    no intent is lost."""
    kept: list[Feature] = []
    seen: list[tuple[tuple, dict[str, frozenset[float]], str]] = []
    notes: list[str] = []
    for f in spec.features or []:
        text = f"{f.position or ''} {f.description or ''}"
        dims = f.dims or {}
        head = (
            (f.type or "").strip().lower(),
            _feat_axis_word(text),
            int(f.count or 1),
            dims.get("diameter"),
            dims.get("depth"),
        )
        coords = _feat_coords(text)
        base = _feat_base_name(f.name)
        dup = False
        for prev_head, prev_coords, prev_name in seen:
            if prev_head != head:
                continue
            shared = set(coords) & set(prev_coords)
            if shared:
                if all(coords[ax] == prev_coords[ax] for ax in shared):
                    dup = True
                    break
                continue
            prev_base = _feat_base_name(prev_name)
            if base == prev_base and (f.name or "") != (prev_name or ""):
                continue
            if int(f.count or 1) > 1 or base == prev_base:
                dup = True
                break
        if dup:
            notes.append(
                f"duplicate feature dropped: '{(f.name or f.type)[:60]}' — "
                f"{head[0] or 'feature'} on axis {head[1] or '?'} already declared"
            )
            continue
        seen.append((head, coords, f.name or ""))
        kept.append(f)
    spec.features = kept
    return notes


def sanitize_prompt_axis_extents(spec: PartSpec, prompt: str) -> list[str]:
    """Raise an understated envelope dim to fit explicit prompt coordinates.

    Observed: "sphere D=60 ... top pole at z=60" parsed as height_z=25 (the
    bore depth) — codegen then centred the bore at z=12.5 inside a 25 mm
    part. Prompt ``z=``/``x=``/``y=`` tokens are explicit user statements;
    any coordinate exceeding the declared span is a spec contradiction.
    Only ever raises a dim — a coordinate inside the envelope is just a
    feature position. z is 0-based so the extent IS the height; x/y are
    centred so the span is twice the largest |coord|."""
    notes: list[str] = []
    ext: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
    for m in _AXIS_RANGE_RE.finditer(prompt or ""):
        ax = m.group(1).lower()
        ext[ax] = max(ext[ax], abs(float(m.group(2))), abs(float(m.group(3))))
    for m in _AXIS_COORD_RE.finditer(prompt or ""):
        ext[m.group(1).lower()] = max(ext[m.group(1).lower()], abs(float(m.group(2))))
    dims = {"x": spec.length_x, "y": spec.width_y, "z": spec.height_z}
    for ax, e in ext.items():
        if not e:
            continue
        want = e if ax == "z" else 2.0 * e
        d = dims[ax]
        if d is None:
            continue
        have = float(d.value or 0)
        tol = max(0.5, 0.02 * max(have, want))
        if want <= have + tol:
            continue
        notes.append(
            f"{ax} dim {have:g} ({d.source}) below explicit prompt coordinate "
            f"{ax}={e:g} — re-stated {ax}={want:g}"
        )
        setattr(
            spec,
            {"x": "length_x", "y": "width_y", "z": "height_z"}[ax],
            d.model_copy(update={"value": want, "source": "stated"}),
        )
    return notes


_FULL_THREAD_RE = re.compile(
    r"vollgewinde|voll(?:es|en)?\s*gewinde|ganz(?:es|en)?\s*gewinde|"
    r"full[\s_-]*thread|entire(?:ly)?\s*thread|thread(?:ed)?\s+(?:over\s+)?"
    r"(?:the\s+)?(?:full|entire|whole)\s+length|"
    r"über\s+(?:die\s+)?(?:ganze|gesamte)\s+länge\s*(?:gewinde|thread|ver(?:sehen|telt)|gängig)?|"
    r"gewinde\s+über\s+(?:die\s+)?(?:ganze|gesamte)\s+länge",
    re.I,
)


def resolve_thread_intents(spec: PartSpec) -> list[str]:
    """Map free-text thread-reach wording onto StandardPartSpec knobs.

    Intake may write "Vollgewinde"/"full thread" only inside a covered
    feature's description — bd_warehouse would then emit the norm's partial
    thread (e.g. ISO 888 b=2d+6 = 32 mm on an M10x100) while the user asked
    for the full shank. Detect the wording (and a covered feature's numeric
    thread_length/length dim before dims get stripped) and set the recipe
    knobs the code emitter honours."""
    if not spec.standard_parts:
        return []
    from cadx.cad import catalog
    from cadx.cad.standard_parts import (
        _is_intrinsic_fastener_feature,
        covered_feature_indices,
    )

    covered = covered_feature_indices(spec, keep_host_voids=True)
    # an intrinsically-covered thread feature is only attributable when a
    # single screw-class part exists — otherwise explicit covers_features
    # has to disambiguate which part's reach it means
    screw_parts = [
        part
        for part in spec.standard_parts
        if (catalog.find_entry(part.class_name) or {}).get("category") == "screws"
    ]
    notes: list[str] = []
    for part in spec.standard_parts:
        if (
            part.full_thread is not None
            or part.thread_length_mm is not None
            or (part.parameters or {}).get("thread_length") is not None
        ):
            continue
        covers = set(part.covers_features or [])
        want_full = False
        want_len: float | None = None
        for i, feature in enumerate(spec.features):
            if covered and i not in covered:
                continue
            owned = bool(feature.name and feature.name in covers) or (
                len(screw_parts) == 1
                and part is screw_parts[0]
                and _is_intrinsic_fastener_feature(feature)
            )
            if not owned:
                continue
            if not ("thread" in str(feature.type or "").lower() or feature.thread_spec):
                continue
            text = f"{feature.name or ''} {feature.description or ''}"
            if _FULL_THREAD_RE.search(text):
                want_full = True
            for key in ("thread_length", "length"):
                raw = (feature.dims or {}).get(key)
                if isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0:
                    want_len = max(want_len or 0.0, float(raw))
        if not (want_full or want_len):
            continue
        if want_len:
            part.thread_length_mm = want_len
            notes.append(
                f"{part.class_name}: covered feature asks thread length "
                f"{want_len:g} mm — recipe emits a thread-length override"
            )
        if want_full:
            part.full_thread = True
            notes.append(
                f"{part.class_name}: 'Vollgewinde/full thread' wording detected "
                "— recipe emits full shank thread"
            )
    return notes


def sanitize_standard_part_names(spec: PartSpec) -> list[str]:
    """Canonicalise near-miss standard_part class names.

    Weak intake models invent plausible-but-wrong names ('HexScrew' for
    HexHeadScrew); an unresolvable name silently drops the whole
    standard-part path — no recipe, no catalog dims, no catalog-covered
    feature dims — and the request's own numbers then fight the geometry
    at verify time (m8_hex: stated x=50 vs the 13 mm hex head). The
    resolver only accepts unambiguous token-cover matches."""
    from cadx.cad import catalog

    notes: list[str] = []
    for part in spec.standard_parts or []:
        canonical = catalog.resolve_class_name(
            str(getattr(part, "class_name", "") or ""),
            fastener_type=getattr(part, "fastener_type", None),
        )
        if canonical and canonical != part.class_name:
            notes.append(f"standard part '{part.class_name}' resolved to '{canonical}'")
            part.class_name = canonical
    return notes


def sanitize_standard_part_dims(spec: PartSpec) -> list[str]:
    if len(spec.standard_parts) != 1 or spec.standard_parts[0].quantity != 1:
        return []
    from cadx.cad.standard_parts import standard_part_bbox

    part = spec.standard_parts[0]
    try:
        expected = standard_part_bbox(part)
    except Exception as e:
        question = OpenQuestion(
            question=f"Complete the catalog parameters for {part.class_name}: {str(e)[:180]}",
            why_it_matters="The registered standard component cannot be generated without these values.",
            default_assumption="Use only the exact catalog class/type/size and required parameters.",
            blocking=True,
        )
        if not any(q.question == question.question for q in spec.open_questions):
            spec.open_questions.append(question)
        return [f"catalog dimensions unavailable for {part.class_name}"]
    notes: list[str] = []
    for field, value, axis in (
        ("length_x", expected[0], "X"),
        ("width_y", expected[1], "Y"),
        ("height_z", expected[2], "Z"),
    ):
        dim = getattr(spec, field)
        tolerance = max(0.25, abs(value) * 0.02)
        if dim.source == "stated" and abs(float(dim.value) - value) > tolerance:
            nominal = [
                float(v)
                for v in (
                    getattr(part, "length_mm", None),
                    getattr(part, "thread_length_mm", None),
                )
                if isinstance(v, int | float) and float(v) > 0
            ]
            if any(abs(float(dim.value) - v) <= tolerance for v in nominal):
                # the stated number restates the fastener's catalog nominal
                # length (under-head shank length — never a bbox extent):
                # not this axis's envelope at all, so there is no real
                # conflict to escalate — the catalog bbox governs.
                setattr(spec, field, Dim(value=value, unit="mm", source="assumed"))
                notes.append(
                    f"{axis}={dim.value:g} restates the {part.class_name} nominal "
                    f"length, not an envelope extent — catalog {axis}={value:g} governs"
                )
                continue
            question_text = (
                f"The {part.class_name} catalog dimensions give {axis}={value:g} mm, "
                f"but the request states {axis}={dim.value:g} mm. Which should govern?"
            )
            existing = next((q for q in spec.open_questions if q.question == question_text), None)
            if existing is not None and not existing.blocking:
                # resolved with the default ("keep catalog dims") — apply it,
                # otherwise verify keeps flagging stated-vs-catalog forever
                setattr(
                    spec,
                    field,
                    Dim(value=value, unit="mm", source="assumed"),
                )
                notes.append(
                    f"{part.class_name} catalog {axis}={value:g} mm governs (question resolved)"
                )
                continue
            question = OpenQuestion(
                question=question_text,
                why_it_matters="A catalog part cannot keep both dimensions and the stated norm.",
                default_assumption="Keep the exact catalog dimensions.",
                blocking=True,
            )
            if existing is None:
                spec.open_questions.append(question)
            notes.append(f"{part.class_name} catalog {axis} dimension conflicts with stated size")
            continue
        if dim.source != "stated" and abs(float(dim.value) - value) > tolerance:
            setattr(
                spec,
                field,
                Dim(value=value, unit="mm", source="inferred"),
            )
            notes.append(f"{part.class_name} catalog dimensions set {axis}={value:g} mm")
    if notes:
        assumption = f"Nominal dimensions for {part.class_name} come from the installed catalog."
        if assumption not in spec.assumptions:
            spec.assumptions.append(assumption)
    notes.extend(_sanitize_covered_features(spec, part.class_name))
    return notes


def _sanitize_covered_features(spec: PartSpec, class_name: str) -> list[str]:
    """Hallucinated dims on catalog-covered features (a stated "head_diameter
    10" vs the real dk=18.8) contradict the norm table forever — the catalog
    governs every intrinsic dimension, so strip them instead of letting
    critique/improve fight the standard's geometry. Thread-pitch conflicts
    are still caught by is_standard_only via the free-text thread_spec."""
    from cadx.cad.standard_parts import covered_feature_indices

    covered = covered_feature_indices(spec, keep_host_voids=True)
    notes: list[str] = []
    if not covered:
        return notes
    for i, feature in enumerate(spec.features):
        if i in covered and feature.dims:
            feature.dims = {}
            notes.append(f"catalog {class_name} governs '{feature.name}' dimensions")
    # verification questions that quote a covered feature's hallucinated
    # numbers ("is the head diameter <= 10 mm?") can never pass — the
    # catalog answer is norm-defined. Keep questions about the part as a
    # whole (seating, orientation, presence).
    covered_words = {
        w
        for i, feature in enumerate(spec.features)
        if i in covered
        for w in re.findall(r"[a-zA-Z]{4,}", f"{feature.name or ''} {feature.type or ''}")
    }
    if not covered_words:
        return notes

    def _about_covered(question_text: str) -> bool:
        q_words = set(re.findall(r"[a-zA-Z]{4,}", question_text.lower()))
        return any(qw[:5] == cw.lower()[:5] for qw in q_words for cw in covered_words)

    kept = [vq for vq in spec.verification_questions if not _about_covered(str(vq))]
    for vq in spec.verification_questions:
        if _about_covered(str(vq)):
            notes.append(f"dropped catalog-internal check: {str(vq)[:80]}")
    if len(kept) != len(spec.verification_questions):
        spec.verification_questions = kept
    return notes


def normalize_feature_positions(spec: PartSpec) -> list[str]:
    """Rewrite model-computed coords in ``feature.position`` when they
    contradict the stated end offset — intake likes to append its own
    arithmetic ("75 mm from -x end (x=20, y=0)") and gets it wrong
    (75-45=30, not 20). The end hint carries the user's stated semantics,
    so the coord token is rewritten to the end-resolved value, leaving one
    consistent position for codegen/verify/anchors to read. Multi-offset
    hints ("20 and 75mm from -x end") are left untouched — a single coord
    cannot alias two positions. Returns change notes."""
    from cadx.cad.poshint import parse_position_hints, resolve_end_hint

    notes: list[str] = []
    try:
        bounds = {
            "x": (-float(spec.length_x.value) / 2, float(spec.length_x.value) / 2),
            "y": (-float(spec.width_y.value) / 2, float(spec.width_y.value) / 2),
        }
    except (AttributeError, TypeError, ValueError):
        return notes  # no stated envelope — end hints unresolvable
    for f in spec.features:
        pos = str(f.position or "").strip()
        if not pos:
            continue
        coords, _faces, ends, _nears = parse_position_hints(pos)
        if not coords or not ends:
            continue
        new = pos
        for m in _AXIS_COORD_RE.finditer(pos):
            ax = m.group(1).lower()
            if ax not in bounds:
                continue
            end_vals = {
                resolve_end_hint(ax, str(e["end"]), float(e["offset_mm"]), *bounds[ax])
                for e in ends
                if str(e["axis"]) == ax
            }
            if len(end_vals) != 1:
                continue  # multi-offset or none — no single truth to rewrite to
            v = next(iter(end_vals))
            if abs(float(m.group(2)) - v) <= 1.0:
                continue  # coord agrees with the end offset — consistent already
            new = new.replace(m.group(0), f"{ax}={v:g}", 1)
            notes.append(
                f"position of '{f.name or f.type}' rewritten — {m.group(0)} "
                f"contradicted the stated end offset (now {ax}={v:g})"
            )
        if new != pos:
            f.position = new
    return notes


# --------------------------------------------------------------------------- #
# degenerate-spec gate                                                          #
# --------------------------------------------------------------------------- #

_TOOL_NAMES = {
    "create_standard_part",
    "search_catalog",
    "lookup_part",
    "cadlib_lookup",
    "search_library",
    "library_part",
    "attach_library_part",
    "view_image",
    "generate_concept_image",
}
_FEATURE_WORDS = re.compile(
    r"\b(loch|löcher|loecher|bohrung|bohrungen|hole|holes|bore|bores|langloch|"
    r"slot|slots|nut|groove|tasche|pocket|rippe|rippen|rib|ribs|steg|web|"
    r"radius|radien|fillet|verrund|verrundung|fase|chamfer|senk|senkung|"
    r"countersunk|countersink|counter|gewinde|thread|insert|boss|zapfen|"
    r"ausschnitt|cutout|aussparung|recess)\w*",
    re.I,
)
_STANDARD_PART_RE = re.compile(
    r"\bM\d+(?:[x-]\d+(?:\.\d+)?)?\s*(?:schraube|screw|bolt|mutter|nut)\b|"
    r"\b(?:heat[- ]set insert|gewindeeinsatz|heatsetnut|heat-set-nut)\b",
    re.I,
)


_HOST_TYPES = ("pad", "boss", "tab", "lug", "plate", "mount", "flange", "base")
# quantifiers that promise a feature PER host/side — "je eine Bohrung",
# "one hole in each lug", "an beiden Enden" — making a missing mirror-side
# feature a real spec gap instead of an asymmetric design choice
_PER_HOST_QUANT_RE = re.compile(
    r"\b(je|jeweils|jeder|jede|each|per|both|beiden|beidseitig|either)\w*", re.I
)
# hole words as SUBSTRINGS — German compounds ("Durchgangsbohrung") break
# \b-based matching, and a false positive only costs one cheap re-ask gate
_HOLE_WORD_RE = re.compile(r"loch|bohr|hole|bore|gewinde|thread", re.I)


# P1 feature-coverage gate: every feature-word CLASS the prompt names must
# map to >= 1 spec feature whose name/type/description carries a class
# keyword. The empty-features gap only fires when the model dropped
# EVERYTHING — a spec that kept plates+holes but silently lost the named
# slot/rib/fillet still reached codegen (observed: "Langloch … verrunden …
# Verstärkungsrippen" -> spec with foot, leg and 2 holes only).
_FEATURE_CLASSES: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    # (class label, prompt-word triggers, required keywords in feature text)
    (
        "hole",
        frozenset({"hole", "bore", "loch", "bohrung", "durchgang"}),
        frozenset({"hole", "bore", "loch", "bohrung", "durchgang"}),
    ),
    (
        "slot",
        frozenset({"slot", "langloch", "nut"}),
        frozenset({"slot", "langloch", "nut"}),
    ),
    (
        "rib",
        frozenset({"rib", "rippe", "ribs", "rippen", "steg", "web", "gusset"}),
        frozenset({"rib", "rippe", "gusset", "steg", "web"}),
    ),
    (
        "fillet",
        frozenset({"fillet", "radius", "radien", "verrund"}),
        frozenset({"fillet", "radius", "rund", "verrund"}),
    ),
    (
        "chamfer",
        frozenset({"chamfer", "fase"}),
        frozenset({"chamfer", "fase"}),
    ),
    (
        "pocket",
        frozenset({"pocket", "tasche", "aussparung", "recess"}),
        frozenset({"pocket", "tasche", "ausspar", "recess"}),
    ),
    (
        "thread",
        frozenset({"thread", "gewinde"}),
        frozenset({"thread", "gewinde"}),
    ),
    (
        "boss",
        frozenset({"boss", "zapfen"}),
        frozenset({"boss", "zapfen"}),
    ),
    (
        "countersink",
        frozenset({"countersink", "senk", "countersunk", "counterbore"}),
        frozenset({"counter", "senk"}),
    ),
)

# bare-substring triggers for the raw-prompt scan — catches German compounds
# the boundary-anchored _FEATURE_WORDS structurally misses
# ("Verstärkungsrippen" glues the noun inside). Only unambiguous stems are
# scanned bare: "senk" would false-trip on "senkrecht", "fase" on "Faser",
# "nut" on "Minuten"-like words — those stay word-matched only.
_CLASS_TEXT_TRIGGERS: dict[str, tuple[str, ...]] = {
    "hole": ("bohrung", "durchgang", "loch", "hole", "bore"),
    "slot": ("langloch", "slot"),
    "rib": ("rippen", "rippe", "gusset", "ribs", "steg"),
    "fillet": ("verrund", "fillet", "radius", "radien"),
    "chamfer": ("chamfer",),
    "pocket": ("aussparung", "tasche", "pocket", "recess"),
    "thread": ("gewinde", "thread"),
    "boss": ("zapfen", "boss"),
    "countersink": ("countersink", "counterbore", "countersunk", "senkbohrung", "senkung"),
}

_UMLAUTS = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "Ä": "a", "Ö": "o", "Ü": "u", "ß": "ss"})

# _FEATURE_WORDS swallows whole words on stem matches ("senk" + \w* eats
# "senkrechter") — these inflected false friends would fire phantom
# classes (a "senkrechter Schenkel" is the vertical leg, no countersink;
# "Nutzen"/"nützlich" is no Nut, "Faser" no Fase)
_WORD_FALSE_FRIENDS = ("senkrecht", "faser", "nutz")


def _fold_feature_text(text: str) -> str:
    """Fold umlauts and their digraph spellings so "Löcher"/"Loecher" hit
    the same class triggers as "Loch"."""
    t = text.lower().translate(_UMLAUTS)
    return t.replace("ae", "a").replace("oe", "o").replace("ue", "u")


def _prompt_feature_classes(prompt: str, words: list[str]) -> dict[str, str]:
    """Feature classes the prompt names -> {class: evidence word}.

    `words` are the _FEATURE_WORDS matches — each joins a class when it
    starts with a trigger (inflection tails ride along) or is one inside a
    longer trigger ("counter" inside "countersink"). The raw-text scan
    additionally catches compounds the regex cannot see; its triggers are
    the long, unambiguous stems only. The scan claims longest-first:
    a shorter stem inside a claimed trigger ("loch" inside "langloch",
    "bohrung" inside "senkbohrung") belongs to the more specific class —
    a Langloch is a slot, not an extra hole."""
    found: dict[str, str] = {}
    for w in words:
        fw = _fold_feature_text(w)
        if any(fw.startswith(bad) for bad in _WORD_FALSE_FRIENDS):
            continue
        for cls, triggers, _required in _FEATURE_CLASSES:
            if cls in found:
                continue
            if any(fw.startswith(t) or (len(fw) >= 5 and fw in t) for t in triggers):
                found[cls] = w
    hay = _fold_feature_text(prompt or "")
    trig_cls = [
        (t, cls) for cls, _trig, _req in _FEATURE_CLASSES for t in _CLASS_TEXT_TRIGGERS[cls]
    ]
    for t, cls in sorted(trig_cls, key=lambda p: -len(p[0])):
        if t not in hay:
            continue
        if cls not in found:
            found[cls] = t
        # the trigger's text is claimed either way — a shorter stem inside
        # it must not fire a second class afterwards
        hay = hay.replace(t, " ")
    return found


def _feature_class_gaps(spec: PartSpec, words: list[str], prompt: str) -> list[str]:
    """Prompt-named feature classes with NO matching spec feature (P1).

    Conservative: a class only reports when it is completely absent — one
    feature whose name/type/description carries any class keyword covers
    it (counts and variants are not audited). A fired gap re-triggers the
    extract_features re-ask like the empty-features case."""
    if not spec.features:
        return []
    present = _prompt_feature_classes(prompt, words)
    if not present:
        return []
    have = [
        _fold_feature_text(f"{f.name or ''} {f.type or ''} {f.description or ''}")
        for f in spec.features
    ]
    gaps: list[str] = []
    for cls, _trig, required in _FEATURE_CLASSES:
        word = present.get(cls)
        if word is None:
            continue
        if any(any(k in h for k in required) for h in have):
            continue
        gaps.append(f"prompt names {cls} ({word}) but spec.features has none")
    return gaps


def _axis_side(text: str, axis: str) -> int | None:
    """Sign (-1/+1) a feature position claims on `axis`, from absolute
    coords, ranges or face hints. None = unresolved."""
    from cadx.agent.spechost import parse_ranges
    from cadx.cad.poshint import parse_position_hints

    coords, faces, _ends, _nears = parse_position_hints(text)
    v = coords.get(axis)
    if v is None:
        rng = parse_ranges(text).get(axis)
        if rng is not None:
            v = (rng[0] + rng[1]) / 2.0
    if v is None:
        for fc in faces:
            if len(fc) == 2 and fc[1] == axis:
                v = 1.0 if fc[0] == "+" else -1.0
                break
    if v is None or abs(v) < 1e-6:
        return None
    return 1 if v > 0 else -1


def _mirror_host_gaps(spec: PartSpec) -> list[str]:
    """Hole declared for only one side of a mirrored host pair (observed:
    'je eine Durchgangsbohrung in den Laschen' -> intake emitted pads at
    x=±45 but only the -x through-hole). Host = pad/boss/tab/lug-type
    feature; binding = sign of the resolved axis position."""
    from cadx.agent.anchors import _hole_like

    gaps: list[str] = []
    for axis in ("x", "y"):
        host_sides: set[int] = set()
        hole_sides: set[int] = set()
        unresolved_holes = 0
        for f in spec.features:
            text = f"{f.position or ''} {f.description or ''}".strip()
            if not text:
                continue
            side = _axis_side(text, axis)
            if _hole_like(f):
                if side is None:
                    unresolved_holes += 1
                else:
                    hole_sides.add(side)
            elif str(f.type or "").lower() in _HOST_TYPES and side is not None:
                host_sides.add(side)
        if host_sides == {-1, 1} and hole_sides and len(hole_sides) < len(host_sides):
            present = f"+{axis}" if 1 in hole_sides else f"-{axis}"
            missing = f"-{axis}" if 1 in hole_sides else f"+{axis}"
            gaps.append(
                f"hole-like features resolve to the {present} side only while "
                f"host features span both ±{axis} — the {missing} hole appears missing"
            )
    return gaps


def spec_gaps(spec: PartSpec, prompt: str) -> list[str]:
    """Deterministic plausibility check. Returns reasons (empty = ok)."""
    gaps: list[str] = []
    if spec.name.strip().lower() in _TOOL_NAMES or "catalog search" in spec.summary.lower():
        gaps.append("spec name/summary is a tool echo")
    if _STANDARD_PART_RE.search(prompt or "") and not spec.standard_parts:
        gaps.append("prompt names a normed component but spec.standard_parts is empty")
    words = sorted({m.group(0).lower() for m in _FEATURE_WORDS.finditer(prompt or "")})
    if words and not spec.features:
        gaps.append(f"prompt names features ({', '.join(words[:8])}) but spec.features is empty")
    elif spec.features:
        # per-class coverage: a named class with NO matching feature is a
        # silent drop, not a design choice (P1)
        gaps.extend(_feature_class_gaps(spec, words, prompt or ""))
    if (
        spec.features
        and _HOLE_WORD_RE.search(prompt or "")
        and _PER_HOST_QUANT_RE.search(prompt or "")
    ):
        gaps.extend(_mirror_host_gaps(spec))
    return gaps


class _FeaturesOut(BaseModel):
    """Focused re-ask payload: every feature the prompt names, plus identity."""

    name: str = ""
    summary: str = ""
    features: list[Feature] = Field(default_factory=list)
    standard_parts: list[StandardPartSpec] = Field(default_factory=list, max_length=24)


async def extract_features(ctx: AgentCtx, spec: PartSpec) -> _FeaturesOut:
    """Second-chance extraction when the intake spec dropped named features.

    Small one-shot call (prompt + current spec as JSON + images) — cheaper
    than re-running the full intake, same pattern as extract_load_cases."""
    user_text = json.dumps(
        {
            "prompt": ctx.prompt,
            "current_spec": spec.model_dump(
                include={
                    "name",
                    "summary",
                    "length_x",
                    "width_y",
                    "height_z",
                    "features",
                    "standard_parts",
                }
            ),
        },
        ensure_ascii=False,
    )
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    user_content += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("intake_features.md")},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = await ctx.llm.chat(
            "vision",
            messages,
            json_schema=_FeaturesOut,
            temperature=0.0,
            extra_body=thinking_body(ctx, STEP),
        )
        parsed = resp.parsed if isinstance(resp.parsed, _FeaturesOut) else _FeaturesOut()
        await ctx.trace.record(
            step="intake_features",
            iteration=0,
            model=resp.model,
            messages=messages,
            response_text=resp.text,
            parsed=parsed.model_dump(),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )
        return parsed
    except Exception:
        log.warning("intake.features_retry_failed", exc_info=True)
        return _FeaturesOut()


async def extract_load_cases(
    ctx: AgentCtx, spec: PartSpec
) -> tuple[list[LoadCase], OptimizeSpec | None]:
    """Focused second-chance extraction: the main intake emitted no usable
    `load_cases` although the user described forces/optimization. Small
    one-shot call on prompt+spec — cheaper than re-running full intake.
    Also recovers the `optimize` block when intake left it empty."""
    from cadx.agent.anchors import _spec_anchors

    # the model cannot see the mesh — hand it the resolved anchor table so
    # `hole_index` refs land on real anchors instead of a miscount
    anchors = _spec_anchors(spec)
    anchor_table = [
        {
            "hole_index": i + 1,
            "name": a.name,
            "kind": a.kind,
            "axis": a.axis,
            "dia": a.dia,
            "pad_dia": a.pad_dia,
            "bolt_size": a.bolt_size,
            "pos": [round(p, 2) if p is not None else None for p in (a.pos or [a.x, a.y, a.z])],
        }
        for i, a in enumerate(anchors)
    ]
    user_text = json.dumps(
        {
            "prompt": ctx.prompt,
            "spec": spec.model_dump(exclude={"load_cases", "optimize"}),
            "anchor_table": anchor_table,
        },
        ensure_ascii=False,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("intake_loads.md")},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    try:
        resp = await ctx.llm.chat(
            "vision",
            messages,
            json_schema=_LoadCasesOut,
            temperature=0.0,
            extra_body=thinking_body(ctx, STEP),
        )
        parsed = resp.parsed if isinstance(resp.parsed, _LoadCasesOut) else _LoadCasesOut()
        await ctx.trace.record(
            step=STEP,
            iteration=0,
            model=resp.model,
            messages=messages,
            response_text=resp.text,
            parsed=parsed.model_dump(),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )
        return parsed.load_cases, parsed.optimize
    except Exception:
        log.warning("intake.load_cases_retry_failed", exc_info=True)
        return [], None


def _repair_image_roles(spec: PartSpec, ctx: AgentCtx) -> None:
    """Bind blank image_roles entries to assets positionally.

    Local vision models regularly emit one entry per image but leave
    `asset_id` (and sometimes `role`) blank. Entries are one-per-image in
    image order, so a count match lets us recover the ids the model meant.
    """
    imgs = [a for a in ctx.assets if a.kind in IMAGE_INPUT_KINDS]
    shown = imgs[: ctx.runtime_settings.agent_max_input_images]
    if not shown or len(spec.image_roles) != len(shown):
        return
    if all(not r.asset_id for r in spec.image_roles):
        for r, a in zip(spec.image_roles, shown, strict=False):
            r.asset_id = a.asset_id


def _sanitize_spec(spec: PartSpec, ctx: AgentCtx) -> PartSpec:
    sanitize_load_cases(spec)
    sanitize_dims(spec, ctx.prompt or "")
    sanitize_material(spec, ctx.prompt or "")
    for note in sanitize_dim_echo(spec, ctx.prompt or ""):
        log.info("intake.dim_echo_fixed", note=note)
    # audit model-authored 'stated' labels BEFORE the upgrades below set
    # their own stated dims from derived values (axis extents use 2x|coord|
    # which never appears verbatim in the prompt)
    for note in sanitize_stated_dims(spec, ctx.prompt or ""):
        log.info("intake.stated_dim_fixed", note=note)
    # explicit prompt triplets / axis coordinates are the strongest
    # stated evidence — they run last so dim_echo's feature-extent
    # guesses cannot silently undo them
    for note in sanitize_dimset(spec, ctx.prompt or ""):
        log.info("intake.dimset_fixed", note=note)
    for note in sanitize_prompt_axis_extents(spec, ctx.prompt or ""):
        log.info("intake.axis_extents_fixed", note=note)
    for note in sanitize_feature_kinds(spec):
        log.info("intake.feature_kind_fixed", note=note)
    for note in dedup_spec_features(spec):
        log.info("intake.feature_deduped", note=note)
    if not spec.process:
        spec.process = ctx.settings.process
    if ctx.research is not None and spec.research is None:
        # sourced research facts join the auditable assumptions tail —
        # spec.research keeps the full brief (product, facts, sources)
        spec.research = ctx.research
        sourced = [
            f"researched: {f.claim} = {f.value} {f.unit}".rstrip() + f" [{f.source}]"
            for f in ctx.research.facts
            if f.claim and f.value
        ]
        for note in sourced[:_MAX_ASSUMPTION_NOTES]:
            if note[:300] not in spec.assumptions:
                spec.assumptions.append(note[:300])
    ctx.spec = spec
    return spec


def research_block(brief: Any) -> str:
    """Render the research brief as intake context — sourced facts the
    spec should build on (they land in spec.assumptions after sanitize)."""
    facts = "\n".join(
        f"- {f.claim}: {f.value} {f.unit} (source: {f.source})".rstrip()
        for f in brief.facts
        if f.claim and f.value
    )
    cons = "\n".join(f"- {c}" for c in brief.constraints if c)
    open_q = "\n".join(f"- {q}" for q in brief.open_questions if q)
    out = ["RESEARCHED FACTS (authoritative sources — treat as stated dims,"]
    if brief.product:
        out.append(f"about: {brief.product}):")
    else:
        out.append("):")
    if facts:
        out += ["facts:", facts]
    if cons:
        out += ["design constraints:", cons]
    if open_q:
        out += ["open questions (sources disagreed — note in assumptions):", open_q]
    return "\n".join(out)


async def run(ctx: AgentCtx) -> PartSpec:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_block(ctx.prompt, ctx.settings, ctx.mode)}
    ]
    if ctx.research is not None:
        user_content.append({"type": "text", "text": research_block(ctx.research)})
    ref = references_block(ctx.assets)
    if ref:
        user_content.append({"type": "text", "text": ref})
    if ctx.mode == "edit" and (ctx.parent.spec or ctx.parent.code):
        user_content += parent_block(ctx.parent.code, ctx.parent.spec, ctx.parent.contact_sheet)
    user_content += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)

    use_tools = tools_enabled(ctx, "vision")
    system = system_prompt("intake_system.md")
    if use_tools:
        system += "\n\n" + system_prompt("tools_info.md")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    if use_tools:
        session = ToolSession(ctx, kind="info")
        res = await run_session(
            ctx,
            "vision",
            messages,
            session=session,
            json_schema=PartSpec,
            max_calls=ctx.runtime_settings.agent_tool_max_calls_info,
            step=STEP,
            temperature=0.0,
        )
        if res.response is None or not isinstance(res.response.parsed, PartSpec):
            from cadx.llm.structured import LLMFormatError

            raise LLMFormatError("intake session produced no spec")
        spec = res.response.parsed
        spec.feature_drops = []
        _repair_image_roles(spec, ctx)
        return _sanitize_spec(spec, ctx)
    resp = await ctx.llm.chat(
        "vision",
        messages,
        json_schema=PartSpec,
        temperature=0.0,
        extra_body=thinking_body(ctx, STEP),
    )
    assert isinstance(resp.parsed, PartSpec)
    await ctx.trace.record(
        step=STEP,
        iteration=0,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed=resp.parsed.model_dump(),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    spec = resp.parsed
    spec.feature_drops = []
    _repair_image_roles(spec, ctx)
    return _sanitize_spec(spec, ctx)
