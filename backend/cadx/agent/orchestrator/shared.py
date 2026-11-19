"""Shared context for orchestrator submodules — dataclasses, issue building,
feedback, patch/exec helpers used by more than one pipeline cluster.

Split out of the former `orchestrator.py` monolith; `cadx.agent.orchestrator`
re-exports every name so import paths stay stable."""

from __future__ import annotations

import inspect
import itertools
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cadx import storage
from cadx.agent.steps import critique as critique_step
from cadx.agent.types import AgentCtx
from cadx.cad.fem.analyze import fem_feedback_lines
from cadx.cad.spec_features import (
    active_features,
)
from cadx.llm.schemas import (
    CodeResponse,
    Critique,
    PartSpec,
)
from cadx.logging import get_logger

log = get_logger(__name__)


@dataclass
class Candidate:
    iteration: int
    code_response: CodeResponse | None
    exec_result: dict[str, Any]
    critique: Critique | None
    dim_issues: list[str]
    mesh_post: dict[str, Any] | None = None  # MeshPostParams used (mesh/hybrid modes)
    feature_findings: list[dict[str, Any]] = field(default_factory=list)
    hybrid_missing: bool = False  # code dropped load_mesh_solid()
    # declared parametric replacement (`cad_fallback = "<reason>"` in the
    # code) — the model openly gave up the mesh body; not a contract
    # violation, and when such a candidate wins the outcome resolves "cad"
    # so scoring/reporting stays honest (B20b)
    cad_fallback: str | None = None
    # critique floor reached with major discrepancies — a wrecked candidate must
    # never claim "best" just for keeping a contract call (load_mesh_solid);
    # observed: score-0 faceted mesh part outranked every viable analytic one
    wrecked: bool = False
    # deterministic quality from the executor's verify_spec report (0..100) —
    # keeps a progress gradient where the saturated critique score flatlines
    dq: int = 0

    @property
    def measured_feature_count(self) -> int:
        """Geometric features actually measured on this candidate's result.

        Ground truth for the regression penalty: a rewrite that drops
        previously-built holes/rounds must not outrank an iteration that
        still has them — LLM critiques miss losses too often.
        """
        f = self.exec_result.get("features") or {}
        return (
            int(f.get("total_holes") or 0)
            + int(f.get("countersinks") or 0)
            + int(f.get("counterbores") or 0)
            + len(f.get("rounds") or [])
            + int(f.get("chamfer_faces") or 0)
        )

    @property
    def rank(self) -> tuple[int, int, int, int, int, int, int]:
        major = (
            sum(1 for d in self.critique.discrepancies if d.severity == "major")
            if self.critique
            else 99
        )
        score = self.critique.score if self.critique else 0
        # hard gates first (fractured solid, dim misses, critique wreck), then
        # the hybrid contract: load_mesh_solid() only outranks among viable
        # candidates — it must not rescue a wrecked mesh attempt over a
        # healthy parametric one. After that the measured verify_spec
        # gradient — the critique score saturates at 0 long before the part
        # is actually hopeless, and the raw feature count is only a tiebreak
        # now (extra holes drilled through a gusset must not be rewarded)
        return (
            0 if _multi_solid(self.exec_result) else 1,
            0 if self.dim_issues else 1,
            0 if self.wrecked else 1,
            0 if self.hybrid_missing else 1,
            self.dq,
            -major,
            score,
            self.measured_feature_count,
        )
@dataclass
class RunOutcome:
    status: str = "failed"  # succeeded | needs_review | failed | waiting_for_input | cancelled
    spec: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    code: str | None = None
    params: dict[str, Any] | None = None
    params_schema: list[dict[str, Any]] | None = None
    analysis: dict[str, Any] | None = None
    critique: dict[str, Any] | None = None
    dfm: dict[str, Any] | None = None
    bom: list[dict[str, Any]] | None = None
    mesh: dict[str, Any] | None = None
    fem: dict[str, Any] | None = None
    resolved_mode: str | None = None
    mode_note: str = ""
    score: int | None = None
    iterations: int = 0
    best_iteration: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # kind|view -> relpath
    questions: list[dict[str, Any]] | None = None
    resume_state: dict[str, Any] | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, Any] | None = None  # measured geometry (holes, rounds, chamfers)
    error: str | None = None
    fallback_to_cad: bool = False  # mesh loop hard-failed on a dimensioned spec
    switch_requested: bool = False  # plateaued — try the next strategy approach
    art_dir: str | None = None  # winning attempt dir (T8.7) — topopt/export read artifacts here
def _annotate_anchor_sources(
    findings: list[dict[str, Any]], anchors: dict[str, Any] | None
) -> None:
    """Stamp ``anchor_source`` on anchor-position findings (``hole <index>``)
    so ``build_issues`` can tell spec/image-declared anchors from
    mesh-scan-only detections (B22). Mutates ``findings`` in place."""
    src_by_idx = {
        str(a.get("index")): str(a.get("source") or "")
        for a in (anchors or {}).get("holes") or []
        if a.get("index") is not None
    }
    if not src_by_idx:
        return
    for f in findings:
        m = re.match(r"hole\s+(\S+)", str(f.get("feature") or ""))
        if m and m.group(1) in src_by_idx:
            f["anchor_source"] = src_by_idx[m.group(1)]
def _clean_losing_attempts(ctx: AgentCtx, keep_rel: str | None) -> int:
    """Drop sibling attempt dirs that lost the bake-off — each keeps ~3MB
    of renders/STLs the promoted copies at version root already cover.
    Scoped to the winner's own iteration dir: FEM/topopt attempts under
    other iterations may still be referenced by their result artifacts."""
    if not keep_rel:
        return 0
    try:
        keep = storage.resolve(keep_rel).resolve()
    except (ValueError, OSError):
        return 0
    it_dir = keep.parent
    if it_dir.parent.name != "iterations":
        return 0
    removed = 0
    for sub in it_dir.iterdir():
        if sub.is_dir() and sub.resolve() != keep:
            shutil.rmtree(sub, ignore_errors=True)
            removed += 1
    if removed:
        log.info("agent.attempt_cleanup", removed=removed, kept=keep_rel)
    return removed
def _copy_best_artifacts(
    ctx: AgentCtx, iteration: int, exec_result: dict[str, Any]
) -> dict[str, str]:
    """Copy renders/contact sheet/thumbnail from the winning attempt to version root."""
    it_dir = (
        storage.resolve(exec_result["art_dir"])
        if exec_result.get("art_dir")
        else (storage.iteration_dir(ctx.project_id, ctx.version_id, iteration))
    )
    v_dir = storage.version_dir(ctx.project_id, ctx.version_id)
    out: dict[str, str] = {}

    r_dir = storage.ensure(v_dir / "renders")
    for view, rel in (exec_result.get("renders") or {}).items():
        src = it_dir / "renders" / Path(rel).name
        if src.exists():
            dst = r_dir / f"{view}.png"
            shutil.copy2(src, dst)
            out[f"render:{view}"] = storage.relpath(dst)
    for name, kind in (("contact_sheet.png", "contact_sheet"), ("thumbnail.png", "thumbnail")):
        src = it_dir / name
        if src.exists():
            dst = v_dir / name
            shutil.copy2(src, dst)
            out[kind] = storage.relpath(dst)
    if exec_result.get("dfm"):
        dfm_path = v_dir / "dfm.json"
        dfm_path.write_text(json.dumps(exec_result["dfm"], indent=2), encoding="utf-8")
        out["dfm_json"] = storage.relpath(dfm_path)
    if exec_result.get("bom"):
        bom_path = v_dir / "bom.json"
        bom_path.write_text(json.dumps(exec_result["bom"], indent=2), encoding="utf-8")
        out["bom_json"] = storage.relpath(bom_path)
    return out
def _copy_fem_artifacts(ctx: AgentCtx, iteration: int) -> dict[str, str]:
    """Copy fem/ + topopt/ iteration outputs to the version root."""
    it_dir = storage.iteration_dir(ctx.project_id, ctx.version_id, iteration)
    v_dir = storage.version_dir(ctx.project_id, ctx.version_id)
    out: dict[str, str] = {}
    for sub in ("fem", "topopt"):
        src_dir = it_dir / sub
        if not src_dir.exists():
            continue
        dst_dir = storage.ensure(v_dir / sub)
        for f in src_dir.iterdir():
            if f.is_file():
                dst = dst_dir / f.name
                shutil.copy2(f, dst)
                out[f"fem:{f.stem}"] = storage.relpath(dst)
    return out
def _copy_mesh_artifacts(
    project_id: Any, version_id: Any, iteration: int, art_dir: str | None = None
) -> dict[str, str]:
    """Copy the version-level mesh intermediates (raw GLB, cleaned STL, solid BREP).

    Files already present in versions/<vid>/mesh/ (e.g. inherited from a parent
    on hybrid reuse) are registered as artifacts too. `art_dir` points at the
    attempt subdir holding `_run/` for the winning candidate.
    """
    it_dir = storage.iteration_dir(project_id, version_id, iteration)
    run_dir = storage.resolve(art_dir) if art_dir else it_dir
    mesh_dir = storage.ensure(storage.version_dir(project_id, version_id) / "mesh")
    out: dict[str, str] = {}
    for src, dst_name, key in (
        (it_dir / "mesh_raw.glb", "mesh_raw.glb", "mesh_raw"),
        (run_dir / "_run" / "model.stl", "mesh_clean.stl", "mesh_clean"),
        (run_dir / "_run" / "model.brep", "solid.brep", "mesh_solid"),
    ):
        dst = mesh_dir / dst_name
        if src.exists():
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            out[key] = storage.relpath(dst)
        elif dst.exists():
            out[key] = storage.relpath(dst)
    return out
def _multi_solid(exec_result: dict[str, Any] | None) -> bool:
    """Deterministic hard gate: specs describe one joined part."""
    return int(((exec_result or {}).get("analysis") or {}).get("solid_count") or 1) > 1
def _resolve_issue_conflicts(
    issues: list[dict[str, Any]], spec: PartSpec | None
) -> list[dict[str, Any]]:
    """Post-pass: keep the issue list self-consistent when mesh-scan anchors
    demand features the spec never declared (B22).

    Observed: a phantom mesh-only anchor produced 'hole 3 missing — cut it'
    while the spec declared exactly two bores — improve was told to add and
    not-add the same hole. Anchors without spec/image provenance never
    outrank the spec: their demands are downgraded to an advisory
    "mesh-detected (not spec-declared)" note, and dropped outright when
    another issue already negates the same feature (fixed count / removal)."""
    if spec is None or not issues:
        return issues

    def _mesh_only(it: dict[str, Any]) -> bool:
        src = str(it.get("anchor_source") or "")
        return (
            it.get("source") == "features"
            and bool(src)
            and ("spec" not in src and "image" not in src)
        )

    def _text(it: dict[str, Any]) -> str:
        return " ".join(str(it.get(k) or "") for k in ("what", "where", "fix_hint"))

    out: list[dict[str, Any]] = []
    for it in issues:
        if not _mesh_only(it):
            out.append(it)
            continue
        token = _ISSUE_FEATURE_RE.search(_text(it).lower())
        dropped = False
        if token:
            # hard contradiction: another (non-mesh-only) issue says this
            # feature class must not exist / is capped by a fixed count —
            # the spec-authoritative side wins
            for other in issues:
                if other is it or _mesh_only(other):
                    continue
                ot = _text(other).lower()
                if token.group(0) in ot and _ISSUE_NEGATION_RE.search(ot):
                    log.info(
                        "agent.issue_conflict",
                        dropped=str(it.get("what"))[:160],
                        kept=str(other.get("what"))[:160],
                    )
                    dropped = True
                    break
        if dropped:
            continue
        out.append(
            {
                **it,
                "severity": "minor",
                "what": f"mesh-detected (not spec-declared): {it.get('what', '')}",
            }
        )
    return out
def _verify_spec_issue(
    exec_result: dict[str, Any] | None, *, stage: str | None = None
) -> dict[str, Any] | None:
    """Canonical M6 conformance verdict — folded into ``Version.issues`` as ONE
    summary entry (same shape as the ``design_audit`` report entry). The
    per-source issues still carry the detail; this is the consolidated
    PASS/FAIL/UNVERIFIED verdict over the executor's measured evidence."""
    report = (exec_result or {}).get("verify_spec")
    if not isinstance(report, dict) or report.get("status") not in ("PASS", "FAIL", "UNVERIFIED"):
        return None
    summary = str(report.get("summary") or "verify_spec conformance report")
    return {
        "source": "verify_spec",
        "severity": "major" if report.get("status") == "FAIL" else "minor",
        "what": f"[{stage}] {summary}" if stage else summary,
        "stage": stage,
        "report": report,
    }
def _write_mesh_json(project_id: Any, version_id: Any, info: dict[str, Any]) -> None:
    mesh_dir = storage.ensure(storage.version_dir(project_id, version_id) / "mesh")
    clean_info = {k: v for k, v in info.items() if not k.startswith("_")}
    (mesh_dir / "mesh.json").write_text(json.dumps(clean_info, indent=2), encoding="utf-8")
def build_issues(
    critique: Critique | None,
    dim_issues: list[str],
    feature_findings: list[dict[str, Any]],
    dfm: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
    spec: PartSpec | None = None,
) -> list[dict[str, Any]]:
    """Structured issue list for the IMPROVE step + trace/UI persistence."""
    issues: list[dict[str, Any]] = []
    # deterministic gate: specs describe one part — a fractured result must
    # never look acceptable just because dims/watertightness are fine
    solid_count = int((analysis or {}).get("solid_count") or 1)
    if solid_count > 1:
        issues.append(
            {
                "source": "solids",
                "severity": "major",
                "what": f"part consists of {solid_count} disconnected solids — "
                "the spec describes a single joined part",
                "fix_hint": "rebuild so all features are fused into one body "
                "(boolean additions must overlap the base body — a rib/gusset "
                "that only touches a face stays a separate solid; use "
                "cadlib.rib() and sink it partly into the wall)",
            }
        )
    for i in dim_issues:
        issues.append({"source": "dims", "severity": "major", "what": f"dimension mismatch — {i}"})
    for f in feature_findings:
        issue = {
            "source": "features",
            "severity": f.get("severity", "minor"),
            "what": f"{f.get('feature')}: expected {f.get('expected')}, actual {f.get('actual')}",
            "fix_hint": f.get("fix_hint", ""),
        }
        # anchor-derived findings carry provenance ("spec"/"image"/"mesh…")
        # — the conflict pass below needs it to spot mesh-only demands
        if f.get("anchor_source"):
            issue["anchor_source"] = str(f["anchor_source"])
        issues.append(issue)
    if critique:
        for d in critique.discrepancies:
            issues.append(
                {
                    # M6: spec-derived yes/no check failures keep their own
                    # source instead of collapsing into generic "critique"
                    "source": ("verify_question" if d.source == "verify_question" else "critique"),
                    "severity": d.severity,
                    "what": d.what,
                    "where": d.where,
                    "fix_hint": d.fix_hint,
                }
            )
        for v in critique.feature_verdicts:
            if v.status in ("missing", "wrong"):
                issues.append(
                    {
                        "source": "critique",
                        "severity": "major" if v.status == "missing" else "minor",
                        "what": f"feature '{v.name}' is {v.status}"
                        + (f" — {v.note}" if v.note else ""),
                    }
                )
    if dfm:
        # a mesh-derived/organic solid cannot be "reoriented" by editing CAD
        # params — an overhang there is a print-process note, not a geometry
        # defect that the improve loop could ever fix
        mesh_derived = (analysis or {}).get("source") == "mesh" or int(
            (analysis or {}).get("face_count") or 0
        ) > 2000
        for w in dfm.get("warnings", []):
            if w.get("severity") == "high":
                sev = "minor" if mesh_derived and w.get("code") == "overhang" else "major"
                issues.append({"source": "dfm", "severity": sev, "what": w["message"]})
    return _resolve_issue_conflicts(issues, spec)[:12]
def compare_dims(
    spec: PartSpec, analysis: dict[str, Any], tol_pct: float, tol_mm: float
) -> list[str]:
    """Numeric axis-wise dimension check + swapped-axes detection."""
    issues: list[str] = []
    actual = [float(v) for v in analysis["bbox_size"]]
    stated: list[tuple[Any, int, str]] = [
        (spec.length_x, 0, "length/X"),
        (spec.width_y, 1, "width/Y"),
        (spec.height_z, 2, "height/Z"),
    ]
    stated_dims = [d for d, _, _ in stated if d.source == "stated"]
    for dim, idx, name in stated:
        if dim.source != "stated":
            continue
        tol = max(dim.value * tol_pct / 100.0, tol_mm)
        if abs(actual[idx] - dim.value) > tol:
            issues.append(
                f"{name}: spec {dim.value:g} mm vs actual {actual[idx]:.2f} mm (tol {tol:.2f})"
            )
    if issues and len(stated_dims) >= 2:
        tol = lambda v: max(v * tol_pct / 100.0, tol_mm)  # noqa: E731
        for perm in itertools.permutations(actual):
            if all(abs(perm[i] - d.value) <= tol(d.value) for i, d in enumerate(stated_dims)):
                issues.append(
                    "note: dimensions match under a different axis order — "
                    "check axis assignment (X/Y possibly swapped)"
                )
                break
    # a rib/gusset protruding past the outer face inflates the bbox — name the
    # usual suspect so the repair prompt doesn't chase the base body
    has_ribs = any("rib" in str(getattr(f, "type", "")).lower() for f in active_features(spec))
    if issues and has_ribs:
        issues.append(
            "note: the spec includes ribs — check they sit INSIDE the envelope "
            "(proud ribs widen the part past the stated width/length)"
        )
    return issues
log = get_logger(__name__)
_LIGHTWEIGHT_RE = re.compile(
    r"leichtbau|lightweight|topolog|material\s+spar|save material|weight sav|reduce weight",
    re.I,
)
_LOAD_HINT_RE = re.compile(
    r"optimi|topolog|leicht|light|gewicht|weight|"
    r"kraft|force|belast|last\b|\bloads?\b|load\s*case|"
    r"\d+\s*(?:N|kg|kn|nm|nmm|bar|mbar|psi|pa|kpa|mpa)\b|pressur|druck|"
    r"\bstress|spannung|structur|statik|tragf|\btorque|drehmoment|\bvibrat|"
    r"\bfix(?:ed|ture|ation)|\bsupport|\bmount|montier|befestig|\bbolt|\bscrew|\bclamp",
    re.I,
)
_FEM_SMOKE_TIMEOUT_S = (
    900.0  # queue wait + coarse solve bound — smoke feedback must never stall the loop
)
_CRITIQUE_PREV_ISSUES = "prev_issues" in inspect.signature(critique_step.run).parameters
_FEM_LIGHTWEIGHT_KW = "lightweight_requested" in inspect.signature(fem_feedback_lines).parameters
_ISSUE_FEATURE_RE = re.compile(r"hole|bore|slot|boss|rib|gusset|fillet|chamfer|pocket|insert", re.I)
_ISSUE_NEGATION_RE = re.compile(
    r"\b(?:remove|delete|drop|extra|spurious|phantom|superfluous|"
    r"must not|should not|do not|not (?:in|part of|declared in)|"
    r"too many|only \d+|exactly \d+|entfern\w*|weglassen|überzählig)\b",
    re.I,
)
_CANDIDATE_VIABLE_SCORE = 0.25
_ADAPTER_CANONICAL_UP = {"hunyuan": "y"}
_FLAT_BOTTOM_CUE = re.compile(
    r"(flat|planar|flush|ebene?|flach|plan\b)[^.;]{0,40}"
    r"(back|bottom|-z|underside|unter|base|r[üu]e?ck|rear)"
    r"|(back|bottom|-z|underside|unter|base|r[üu]e?ck|rear)[^.;]{0,40}"
    r"(flat|planar|flush|ebene?|flach|plan\b)",
    re.IGNORECASE,
)
