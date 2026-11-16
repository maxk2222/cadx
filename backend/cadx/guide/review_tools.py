"""Review-mission tools — the bounded quality gate before annotate/emit.

The review LLM inspects geometry (validators + renders), step order and
coverage, then writes ONE ReviewReport (score 0..10 + findings carrying
the responsible mission). The runner turns blocker findings into repair
iterations (≤ guide_review_max_iter) targeting the flagged mission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cadx.guide.state import ReviewFinding, ReviewReport
from cadx.guide.tools import _arr, _fn, _num
from cadx.logging import get_logger

if TYPE_CHECKING:
    from cadx.guide.session import GuideToolSession

log = get_logger(__name__)

SCHEMAS: dict[str, dict[str, Any]] = {
    "write_review_report": _fn(
        "write_review_report",
        "Submit the final review for this iteration: overall score 0..10 "
        "plus findings. Every finding MUST name the mission that owns the "
        "fix (doc|steps|parts|assembly|build|annotate) — blocker findings "
        "restart that mission in the repair loop (bounded). Blender-mode "
        "runs: spatial/placement/joint faults belong to 'build' (or "
        "'assembly'/'parts' — both map onto build).",
        {
            "score": _num("overall guide quality 0..10"),
            "findings": _arr(
                {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["blocker", "warning", "note"]},
                        "mission": {
                            "type": "string",
                            "enum": [
                                "doc",
                                "steps",
                                "parts",
                                "assembly",
                                "build",
                                "annotate",
                            ],
                        },
                        "step": {"type": "integer"},
                        "text": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["severity", "mission", "text"],
                },
                "findings — empty list means clean",
            ),
            "geometry_ok": {
                "type": "boolean",
                "description": "did geometry validation pass (joints+collisions)",
            },
            "visual_comparison": {
                "type": "string",
                "description": "MANDATORY: expected-vs-observed verdict from "
                "comparing the end-state render with the manual figure(s) — "
                "one line per branch/part-group: '<part>: expected <manual> — "
                "observed <render> — match|deviation'. A score without this "
                "evidence is rejected.",
            },
        },
        ["score", "findings", "visual_comparison"],
    ),
}


def _audit_evidence(s: GuideToolSession) -> tuple[bool, list[str]]:
    """Physical audit findings as readable evidence lines — the reviewer's
    score cannot contradict what the deterministic audit measured. Stays
    advisory (clean) when the scene can't be built."""
    evidence: list[str] = []
    try:
        from cadx.guide.assembly_tools import joint_geometry_audit

        audit = joint_geometry_audit(s)
        for b in audit["bad_joints"]:
            evidence.append(f"joint {b['joint']}: gap={b['gap_mm']}mm nd={b['normal_dot']}")
        for b in audit["buried_instances"]:
            evidence.append(
                f"{b['instance']} is {round(b['buried_frac'] * 100)}% inside other parts"
            )
        for d in audit.get("disconnected_instances", []):
            evidence.append(d["issue"])
        # declared mates that never materialized / joints that landed on
        # parts the sub-step did not declare — topology deviations the
        # geometry checks alone can't see (a clean mate on the wrong part)
        for m in audit.get("unfulfilled_mates", []):
            evidence.append(m["issue"])
        for m in audit.get("unexpected_mates", []):
            evidence.append(m["issue"])
        for m in audit.get("bridge_incomplete", []):
            evidence.append(m["issue"])
        # a joint on a SYNTHESIZED thread site (nested stub emitted inside an
        # enclosing socket when the mesh models no real threaded end) is
        # geometrically legal but semantically unverified — the classic
        # wrong-interface failure (fitting sunk into an oversized opening
        # instead of seated on the pipe end) passes gap/alignment cleanly.
        # It cannot earn a ship-grade score without the reviewer explicitly
        # judging it against the manual figure.
        for ap in audit.get("approximated_joints", []):
            evidence.append(
                f"joint {ap['joint']}: approximated thread-site interface "
                f"(nested stub inside an enclosing socket) — verify against "
                f"the manual figure; if the fitting sits inside an oversized "
                f"opening this is a blocker on 'assembly'"
            )
    except Exception:
        pass
    return (not evidence, evidence)


async def write_review_report(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    score = max(0.0, min(10.0, float(a.get("score", 0.0))))
    findings = []
    for f in a.get("findings") or []:
        try:
            sev = str(f.get("severity", "note"))
            mission = str(f.get("mission", ""))
            if sev not in ("blocker", "warning", "note"):
                sev = "note"
            if mission not in ("doc", "steps", "parts", "assembly", "build", "annotate"):
                mission = (
                    "build" if s.state.guide_mode == "blender" else "assembly"
                )
            findings.append(
                ReviewFinding(
                    severity=sev,
                    mission=mission,
                    step=int(f["step"]) if f.get("step") is not None else None,
                    text=str(f.get("text", ""))[:500],
                    evidence=str(f.get("evidence", ""))[:300],
                )
            )
        except Exception:
            continue
    # mandatory visual evidence: a score without an explicit
    # expected-vs-observed layout comparison is superficial judgement —
    # exactly how a render showing pipes through the casing wall earned 8.0
    visual = str(a.get("visual_comparison") or "").strip()
    if len(visual) < 40:
        return {
            "ok": False,
            "error": "missing visual_comparison — render the end state "
            "(render_assembly), open the manual's overview figure "
            "(get_figure_image/get_page_image) and write one "
            "expected-vs-observed line per branch/part-group before "
            "scoring",
        }
    n_blockers = sum(1 for f in findings if f.severity == "blocker")
    if score < 8.0 and n_blockers == 0:
        return {
            "ok": False,
            "error": "score below 8 must name at least one BLOCKER finding "
            "with its owning mission — a low score without a flagged fault "
            "is not actionable; mark the fault(s) severity='blocker'",
        }
    if score >= 8.0 and n_blockers:
        return {
            "ok": False,
            "error": "score ≥8 contradicts open blocker findings — either "
            "fix the score to <8 or downgrade the findings (severity "
            "'warning'/'note')",
        }
    # deterministic backstop: the SAME joint/burial audit the validator
    # reports runs here — a reviewer cannot wave through geometry that
    # measurably fails (buried parts, misaligned/open joints). Passing
    # score while physical findings exist is a contradiction — return
    # ok:false and force an honest verdict instead.
    audit_ok, audit_evidence = _audit_evidence(s)
    if score >= 8.0 and not audit_ok:
        return {
            "ok": False,
            "error": "score ≥8 contradicts the physical audit — these are "
            "BLOCKER findings against mission 'assembly'; report them "
            "honestly (score ≤7): " + "; ".join(audit_evidence[:5]),
            "bad_joints": audit_evidence,
        }
    it = int(s.state.review.iteration) + 1
    s.state.review = ReviewReport(
        score=score,
        findings=findings,
        geometry_ok=bool(a["geometry_ok"]) if a.get("geometry_ok") is not None else audit_ok,
        iteration=it,
        visual_comparison=visual[:2000],
    )
    return {
        "ok": True,
        "score": score,
        "iteration": it,
        "blockers": sum(1 for f in findings if f.severity == "blocker"),
        "audit_ok": audit_ok,
        "note": "done_mission ends the review — the runner decides whether a "
        "repair iteration is needed (score/blockers vs bounds)",
    }


def _done_guard(s: GuideToolSession) -> dict[str, Any] | None:
    if s.state.review.iteration == 0 or (
        s.state.review.iteration > 0 and not s.state.review.findings and s.state.review.score == 0
    ):
        return {"ok": False, "error": "call write_review_report first"}
    return None


from cadx.guide.tools import DONE_GUARDS  # noqa: E402

DONE_GUARDS["review"] = _done_guard

HANDLERS = {
    "write_review_report": write_review_report,
}
