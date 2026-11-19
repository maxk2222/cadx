"""CAD codegen pipeline — `_run_cad` and its attempt/feedback machinery.

Split out of the former `orchestrator.py` monolith; `cadx.agent.orchestrator`
re-exports every name so import paths stay stable."""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import time
from typing import Any

from cadx import storage
from cadx.agent.autofix import propose_fixes
from cadx.agent.executor_client import exec_error
from cadx.agent.orchestrator.shared import (
    _CRITIQUE_PREV_ISSUES,
    _FEM_LIGHTWEIGHT_KW,
    _FEM_SMOKE_TIMEOUT_S,
    _LIGHTWEIGHT_RE,
    Candidate,
    RunOutcome,
    _annotate_anchor_sources,
    _clean_losing_attempts,
    _copy_best_artifacts,
    _copy_fem_artifacts,
    _copy_mesh_artifacts,
    _multi_solid,
    _verify_spec_issue,
    _write_mesh_json,
    build_issues,
    compare_dims,
)
from cadx.agent.quality import det_quality
from cadx.agent.specdrops import feature_drop_issues
from cadx.agent.steps import codegen as codegen_step
from cadx.agent.steps import critique as critique_step
from cadx.agent.steps import improve as improve_step
from cadx.agent.steps import intake as intake_step
from cadx.agent.types import AgentCtx
from cadx.cad.allowlist import (
    check_code,
    declared_cad_fallback,
    mesh_solid_flows_to_result,
)
from cadx.cad.codemod import fix_api_misuse
from cadx.cad.features import compare_features
from cadx.cad.fem.analyze import _FEM_DEGENERATE_SF, FemReport, fem_feedback_lines
from cadx.cad.partspec import check_standard_part_calls
from cadx.cad.patch import apply_patch, diff_summary
from cadx.cad.spec_features import (
    active_features,
    dropped_feature_indices,
    feature_drop_records,
)
from cadx.llm.schemas import (
    CodeResponse,
    Critique,
    Discrepancy,
    PartSpec,
    PatchResponse,
)
from cadx.logging import get_logger

log = get_logger(__name__)


def _attempt_history(entries: list[str], keep: int = 6) -> str:
    """The edit-side counterpart of _repair_error_history: which code changes
    were already applied this job. Without it an identical issue list +
    identical base code regenerates the identical failed patch — the main
    mechanical driver of repair/improve loops."""
    if not entries:
        return ""
    lines = ["PREVIOUS ATTEMPTS this job — edits already tried, do NOT repeat them:"]
    lines += ["  " + e for e in entries[-keep:]]
    return "\n".join(lines)
def _exec_error_signature(exec_result: dict[str, Any]) -> str:
    """Job-wide exec-failure signature: errordiag category + the last error
    line with numbers normalised — the same root cause counts as one
    signature even when line numbers or values churn between repair
    attempts. Returns "" when there is no error text."""
    err = str(exec_result.get("error") or "").strip()
    if not err:
        return ""
    from cadx.cad.errordiag import diagnose

    cat = diagnose(err, exec_result.get("traceback")).category
    last = next((ln.strip() for ln in reversed(err.splitlines()) if ln.strip()), "")
    norm = re.sub(r"0x[0-9a-fA-F]+|\d+(?:\.\d+)?", "#", last)
    norm = re.sub(r"\s+", " ", norm)
    return f"{cat}:{norm[:160]}"
def _exec_payload(
    ctx: AgentCtx,
    code: str,
    iteration: int,
    *,
    full_exports: bool,
    subdir: str = "",
    light: bool = False,
    detail_mode: str | None = None,
    params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": ctx.job_id,
        "project_id": str(ctx.project_id),
        "version_id": str(ctx.version_id),
        "iteration": iteration,
        "subdir": subdir,
        "code": code,
        "params_override": params_override,
        "references": [
            {"asset_id": a.asset_id, "path": str(a.path)}
            for a in ctx.assets
            if a.kind in ("reference_step", "reference_stl")
        ],
        "settings": ctx.settings.model_dump(),
        "runtime_settings": ctx.runtime_settings.executor_values(),
        "full_exports": full_exports,
        "detail_mode": detail_mode or ("export" if full_exports else "preview"),
        "light": light,
        "name": "model",
    }
    if ctx.hybrid_solid is not None:
        payload["mesh_solid"] = str(ctx.hybrid_solid)
        payload["mesh_bbox"] = ctx.hybrid_mesh_info.get("bbox_size")
    # the executor task forwards `anchors` verbatim — it is the only free-form
    # dict channel into run_pipeline, so the minimal spec sub-dict for the
    # executor-side verify_spec report rides inside it (the runner ignores
    # unknown anchor keys; ctx.anchors itself is never mutated)
    anchors = dict(ctx.anchors or {})
    if ctx.spec is not None:
        dropped_names = {
            str((record.get("feature") or {}).get(key) or "").strip().casefold()
            for record in feature_drop_records(ctx.spec)
            for key in ("name", "type")
        }
        dropped_names.discard("")
        if dropped_names and isinstance(anchors.get("holes"), list):
            holes = []
            for anchor in anchors["holes"]:
                item = dict(anchor) if isinstance(anchor, dict) else anchor
                source = str(item.get("source") or "") if isinstance(item, dict) else ""
                independent = any(
                    token in source.split("+")
                    for token in ("image", "image-approx", "mesh", "scan")
                )
                name = (
                    str(item.get("name") or "").strip().casefold() if isinstance(item, dict) else ""
                )
                if "spec" in source and not independent and name in dropped_names:
                    item["dropped_feature"] = True
                holes.append(item)
            anchors["holes"] = holes
        anchors["spec"] = {
            "name": ctx.spec.name,
            "length_x": ctx.spec.length_x.model_dump(),
            "width_y": ctx.spec.width_y.model_dump(),
            "height_z": ctx.spec.height_z.model_dump(),
            "features": [f.model_dump() for f in active_features(ctx.spec)],
            "feature_drops": feature_drop_records(ctx.spec),
            "standard_parts": [part.model_dump() for part in ctx.spec.standard_parts],
            "min_wall_mm": ctx.settings.min_wall_mm,
        }
    if anchors:
        payload["anchors"] = anchors
    return payload
def _export_lost_holes(
    preview_groups: list[dict[str, Any]],
    export_groups: list[dict[str, Any]],
) -> list[str]:
    """Preview hole groups with no export counterpart — an export-only code
    path (a `CADX_DETAIL_MODE` fork) dropped geometry the preview verified.

    Matching is deliberately loose: the helical thread groove can shift a
    measured bore diameter and re-group members, so a group counts as kept
    when ANY export group shares its axis letter and a diameter within
    1.2 mm or 12%. Elongated slots are skipped — they were never the
    thread-path failure this gate exists for."""
    lost: list[str] = []
    for g in preview_groups:
        if g.get("elongated"):
            continue
        dia = float(g.get("diameter") or 0.0)
        ax = str(g.get("axis") or g.get("through_axis") or "").lstrip("+-")
        kept = False
        for e in export_groups:
            if e.get("elongated"):
                continue
            e_dia = float(e.get("diameter") or 0.0)
            e_ax = str(e.get("axis") or e.get("through_axis") or "").lstrip("+-")
            if ax == e_ax and dia > 0 and abs(e_dia - dia) <= max(1.2, dia * 0.12):
                kept = True
                break
        if not kept:
            lost.append(f"Ø{dia:g}/{ax or '?'}x{int(g.get('count') or 1)}")
    return lost
def _has_degenerate_case(fem_res: dict[str, Any]) -> bool:
    """A solved case reporting sf > _FEM_DEGENERATE_SF verifies nothing — the
    declared load barely stresses the part (marine propeller r25: a 0.1 MPa
    uniform pressure on a Ø140 hub returned sf≈1419 and the gate bound
    sf_max=300 failed it). Same class of spec defect as an unsolvable case:
    the declaration is wrong, not the solver, so it earns one re-extract."""
    if not fem_res.get("ok"):
        return False
    cases = ((fem_res.get("fem") or {}).get("cases")) or []
    return any((c.get("safety_factor") or 0.0) > _FEM_DEGENERATE_SF for c in cases)
def _has_unsolvable_case(fem_res: dict[str, Any]) -> bool:
    """The fem smoke returned ok but at least one declared case reported no
    solvable result — a spec-declaration defect (regions resolving to zero
    force / empty masks), not a solver failure. Even one dead case fails the
    downstream stage gate (forbidden warnings), so partial sets repair too."""
    if not fem_res.get("ok"):
        return False
    cases = ((fem_res.get("fem") or {}).get("cases")) or []
    return any(c.get("safety_factor") is None for c in cases)
def _issue_source_rollup(issues: list[dict[str, Any]], *, exclude: set[str] | None = None) -> str:
    """`{source}:{count}` rollup over the actionable issue list — the same
    filter the improve step applies (no measurement-artifact / infra
    entries). `exclude` drops sources already covered elsewhere (the
    critique emit's own major/minor count covers critique +
    verify_question discrepancies)."""
    ex = exclude or set()
    counts: dict[str, int] = {}
    for i in issues or []:
        if i.get("artifact") or i.get("source") == "infra":
            continue
        src = str(i.get("source") or "critique")
        if src in ex:
            continue
        counts[src] = counts.get(src, 0) + 1
    return ", ".join(f"{s}:{n}" for s, n in sorted(counts.items()))
def _log_attempt(log_entries: list[str], prefix: str, old: str, new: str) -> None:
    """Record one applied code change (bounded diffstat, capped log)."""
    diff = diff_summary(old, new, 4).replace("\n", "; ")
    log_entries.append(f"{prefix}: {diff[:200]}")
    del log_entries[:-12]
def _measurement_authoritative(ctx: AgentCtx) -> bool:
    """True when at least one measurement was taken from an explicitly
    labelled plan view with clean quality (set by the measure step)."""
    return any(bool(m.get("authoritative")) for m in (ctx.measurements or {}).values())
def _note_exec_error(sig_counts: dict[str, int], sig: str) -> bool:
    """Increment the job-wide signature tally; True once `sig` has been seen
    twice or more — the caller's cue to switch the repair to full_code."""
    if not sig:
        return False
    sig_counts[sig] = sig_counts.get(sig, 0) + 1
    return sig_counts[sig] >= 2
def _params_vs_spec(
    spec: PartSpec, literal_params: dict[str, Any], *, mode: str = "cad"
) -> list[dict[str, Any]]:
    """Cheap CAD consistency check: stated part dims should appear in PARAMS.

    Mesh/hybrid base dimensions come from the normalized source mesh; their
    code PARAMS usually describe only a bore/cut and cannot represent the
    whole envelope.
    """
    issues: list[dict[str, Any]] = []
    if mode != "cad" or not literal_params:
        return issues
    vals = [float(v) for v in literal_params.values() if isinstance(v, int | float)]
    for dim, name in (
        (spec.length_x, "length/X"),
        (spec.width_y, "width/Y"),
        (spec.height_z, "height/Z"),
    ):
        if dim.source != "stated":
            continue
        if vals and not any(abs(v - dim.value) <= 0.2 * dim.value for v in vals):
            issues.append(
                {
                    "source": "contract",
                    "severity": "minor",
                    "what": (
                        f"spec {name}={dim.value:g}mm does not appear in PARAMS — "
                        "possibly hardcoded wrong or derived incorrectly"
                    ),
                    "fix_hint": f"ensure a PARAMS value controls {name}={dim.value:g}mm",
                }
            )
    return issues
def _regression_critique(best: Candidate | None, analysis: dict[str, Any]) -> Critique | None:
    """Deterministic rebuild-collapse verdict — skips the vision critique.

    When a rebuild (improve patch or fresh codegen) produces a fraction of the
    best candidate's body (observed: single-tab plate at ~10% of the volume,
    full 10-minute critique just to say 'the arch is gone'), the measured
    metrics already prove the regression. Returns a synthesized Critique whose
    fix_hint points back at the working structure, or None when the candidate
    is not a clear collapse.
    """
    if best is None:
        return None
    b_ana = (best.exec_result or {}).get("analysis") or {}
    b_vol = float(b_ana.get("volume") or 0)
    a_vol = float(analysis.get("volume") or 0)
    b_bb = b_ana.get("bbox_size") or []
    a_bb = analysis.get("bbox_size") or []
    if b_vol <= 0 or a_vol <= 0 or len(b_bb) != 3 or len(a_bb) != 3:
        return None
    collapsed_axes = sum(1 for a, b in zip(a_bb, b_bb, strict=True) if a < 0.5 * b)
    if a_vol >= 0.30 * b_vol and collapsed_axes < 2:
        return None
    bb = "x".join(f"{float(x):.0f}" for x in a_bb), "x".join(f"{float(x):.0f}" for x in b_bb)
    return Critique(
        score=10,
        passed=False,
        confidence=1.0,
        discrepancies=[
            Discrepancy(
                what=(
                    f"catastrophic regression vs the best candidate: bbox "
                    f"{bb[0]} vs {bb[1]} mm, volume {a_vol:.0f} vs "
                    f"{b_vol:.0f} mm³ — the rebuilt code lost the main body"
                ),
                where="whole part",
                severity="major",
                source="spec",
                fix_hint=(
                    "keep the previous working structure (the mesh/CAD base "
                    "plus every feature block) and apply ONLY the minimal "
                    "change the listed issues ask for — do not rewrite or "
                    "re-trim the part from scratch"
                ),
            )
        ],
    )
def _repair_error_history(sig_counts: dict[str, int], attempts: int) -> str:
    """Deterministic 'error memory' for the repair context — the distinct
    exec-error signatures seen this job and how often each repeated
    (Reflexion-style recall without an extra LLM call)."""
    if not sig_counts:
        return ""
    lines = ["ERROR HISTORY this job — approaches that produced these already failed:"]
    for sig, n in sorted(sig_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]:
        lines.append(f"  {n}x {sig}")
    lines.append(
        f"repair attempts so far: {attempts} — repeating a failed approach "
        "reproduces the same signature; change strategy"
    )
    return "\n".join(lines)
def _resolve_patch(
    base_code: str, resp: PatchResponse, patch_mode: str
) -> tuple[str | None, list[str]]:
    """Turn a PatchResponse into code: prefer edits; full_code fallback/explicit."""
    if resp.full_code and (not resp.edits or patch_mode == "never"):
        return resp.full_code, []
    if not resp.edits:
        return None, ["model returned neither edits nor full_code"]
    patched, errors = apply_patch(base_code, resp.edits)
    if patched is not None:
        return patched, []
    if resp.full_code:
        check = check_code(resp.full_code)
        if check.ok:
            return check.code or resp.full_code, errors
        errors = [*errors, "full_code also fails contract: " + "; ".join(check.errors[:4])]
    return None, errors
async def _run_cad(  # noqa: C901
    ctx: AgentCtx,
    outcome: RunOutcome,
    spec: PartSpec,
    plan: Any,
    t0: float,
    cancelled: Any,
    finish_early: Any,
    *,
    max_iterations: int,
    max_repairs: int,
    pass_score: int,
    tol_pct: float,
    tol_mm: float,
    switch_check: Any = None,
    start_iteration: int = 1,
) -> RunOutcome:
    s = ctx.runtime_settings
    patch_mode = s.agent_patch_mode
    best: Candidate | None = None
    best_issues: list[dict[str, Any]] = []  # issue list belonging to `best`
    # issue-fingerprint stall memory (P10): an issue whose normalised text
    # survives 2+ consecutive iterations while the measured geometry did
    # not move is a measurement artifact, not an actionable defect — it
    # gets flagged (stays visible for honesty) but stops driving improve
    prev_issue_fps: dict[str, int] = {}
    prev_features_hash: str | None = None
    # progress tracker — a later candidate can outrank `best` on fewer dim
    # issues/majors while the critique score stays flat; measured dq moves
    # count as progress too so a saturated score-0 plateau no longer aborts
    # a run that is actually converging (observed: bracket stalled out at
    # iteration 2 while it had just added the missing arm holes)
    last_progress_it = start_iteration - 1
    best_dq = -1
    best_score = -1
    # stall escape: ONE fresh codegen from the issue list before finalising —
    # improve patches circle the same broken code, a clean regen breaks free
    fresh_retry_used = False
    force_fresh = False
    issues: list[dict[str, Any]] = []
    iterations_done = 0
    improve_failures = 0
    # patch errors from a failed REPAIR round — forwarded to the next improve
    # session so it knows the previous patch attempt missed the code text
    pending_patch_errors: list[str] | None = None
    # job-wide exec-error signature tally: the SAME root cause reappearing
    # across repair attempts (incl. across iterations, unlike the local
    # prev_err_sig check) means patching isn't converging — the next repair
    # regenerates full_code instead and the history digest below goes into
    # the repair context as deterministic failure memory
    repair_sig_counts: dict[str, int] = {}
    stall_sigs_announced: set[str] = set()
    # applied-edit memory injected into improve/repair contexts — without it
    # identical issue+code prompts deterministically regenerate the same patch
    attempt_log: list[str] = []
    repair_calls = 0
    last_escalation_it = -1

    def _escalate_thinking(reason: str) -> bool:
        """Raise the reasoning level one step (0 -> medium -> high) and latch
        `ctx.thinking_escalated` for the rest of the job —
        `llm_thinking="auto"` then merges the level's thinking payload into every
        subsequent call (run_session re-reads it each turn). At most one
        raise per iteration; returns True only on an actual raise so
        callers emit once."""
        nonlocal last_escalation_it
        if last_escalation_it == it or ctx.thinking_level >= 2:
            return False
        ctx.thinking_level += 1
        ctx.thinking_escalated = True
        last_escalation_it = it
        log.info(
            "agent.thinking_escalated",
            iteration=it,
            reason=reason,
            level=ctx.thinking_level,
        )
        return True

    def issues_text(items: list[dict[str, Any]]) -> str:
        from cadx.agent.context import issues_block

        # issues_block prefixes its own header — strip it for the codegen path
        return "\n".join(issues_block(items).splitlines()[1:])

    for it in range(start_iteration, start_iteration + max_iterations):
        iterations_done = it
        if await cancelled():
            outcome.status = "cancelled"
            return outcome
        if best is not None and await finish_early():
            await ctx.emit("finalize", it, "finish requested — keeping best candidate")
            break
        if ctx.tokens_used() > ctx.budget_tokens or time.monotonic() - t0 > s.agent_max_wall_s:
            await ctx.emit("budget", it, "budget exceeded — finalising best candidate")
            break

        # attempt counter: every exec/process within this iteration gets its
        # own artifact subdir so a stale candidate's files are never
        # overwritten by a later repair/verify/autofix pass (stale-score bug)
        art_attempt = itertools.count(1)

        # ---------- obtain code: fresh codegen vs targeted improve ----------
        base_code = best.code_response.code if best and best.code_response else ""
        # artifact-flagged issues (same fingerprint 2+ iterations on
        # unchanged geometry) and executor infra errors stay on the
        # persisted issue list for honesty, but they carry no code change
        # — feeding them back just re-spends an iteration reproducing the
        # same "fix" (P10 / infra)
        actionable = [i for i in issues if not i.get("artifact") and i.get("source") != "infra"]
        if issues and not actionable:
            await ctx.emit(
                "improve",
                it,
                "only measurement-artifact / infra issues remain — converged",
            )
            break
        use_improve = (
            bool(base_code)
            and it > start_iteration
            and bool(actionable)
            and improve_failures < 2
            and not force_fresh
        )
        code_resp: CodeResponse | None = None
        # measured geometry of the best-so-far candidate — the improve
        # session and the fresh-codegen stall escape both need it
        features = (best.exec_result.get("features") or None) if best else None
        if use_improve:
            rollup = _issue_source_rollup(issues)
            await ctx.emit(
                "improve",
                it,
                f"fixing {len(actionable)} issue(s) on the best candidate"
                + (f" ({rollup})" if rollup else ""),
            )
            patch_errors: list[str] | None = pending_patch_errors
            pending_patch_errors = None
            for attempt in range(1, s.agent_patch_retries + 1):
                try:
                    presp = await improve_step.run(
                        ctx,
                        spec,
                        plan,
                        base_code=base_code,
                        issues=actionable,
                        features=features,
                        patch_errors=patch_errors,
                        attempts=attempt_log,
                        iteration=it,
                        attempt=attempt,
                        verify_fn=_session_verify_fn(ctx, it, art_attempt),
                        render_fn=_session_render_fn(ctx, it, art_attempt),
                        render_existing_fn=_session_render_existing_fn(ctx, it, art_attempt),
                    )
                except Exception as e:
                    log.warning("agent.improve_failed", iteration=it, error=repr(e))
                    break
                new_code, patch_errors = _resolve_patch(base_code, presp, patch_mode)
                if new_code is not None:
                    _log_attempt(attempt_log, f"it{it} improve", base_code, new_code)
                    code_resp = CodeResponse(
                        reasoning=presp.reasoning,
                        params=presp.params,
                        expected_bbox=presp.expected_bbox,
                        code=new_code,
                    )
                    break
            if code_resp is None:
                improve_failures += 1
                attempt_log.append(f"it{it} improve patch did not apply")
                del attempt_log[:-12]
                await ctx.emit("improve", it, "patch failed — falling back to fresh codegen next")
                continue
        else:
            await ctx.emit("codegen", it, f"writing CAD code (iteration {it}/{max_iterations})")
            try:
                code_resp = await codegen_step.run(
                    ctx,
                    spec,
                    plan,
                    iteration=it,
                    feedback=issues_text(actionable) if actionable else None,
                    verify_fn=_session_verify_fn(ctx, it, art_attempt),
                    render_fn=_session_render_fn(ctx, it, art_attempt),
                    render_existing_fn=_session_render_existing_fn(ctx, it, art_attempt),
                    features=features,
                )
                force_fresh = False
            except Exception as e:
                issues = [
                    {
                        "source": "codegen",
                        "severity": "major",
                        "what": f"code generation failed: {e}",
                    }
                ]
                log.warning("agent.codegen_failed", iteration=it, error=repr(e))
                continue

        code = code_resp.code
        code, codemod_notes = fix_api_misuse(code)
        if codemod_notes:
            log.info("agent.codemod", iteration=it, notes=codemod_notes)
            await ctx.emit("execute", it, codemod_notes[0])

        # ---------- execute with patch-based repairs ----------
        # The static contract check runs inside the repair loop: violations are
        # treated like exec failures (they'd fail in the executor anyway) and go
        # through the same patch-repair path without burning an executor call.
        exec_result: dict[str, Any] = {}
        chk = check_code(code)
        prev_err_sig = ""
        for repair_n in range(max_repairs + 1):
            if await cancelled():
                outcome.status = "cancelled"
                return outcome
            if best is not None and await finish_early():
                break
            chk = check_code(code)
            if chk.code:
                code = chk.code  # escape-decoded — the tree the contract validated
            part_errors = check_standard_part_calls(code) if chk.ok else []
            if chk.ok and not part_errors:
                await ctx.emit(
                    "execute",
                    it,
                    "executing in sandbox"
                    if repair_n == 0
                    else f"repair attempt {repair_n}/{max_repairs}",
                )
                exec_result = await ctx.executor.execute(
                    _exec_payload(ctx, code, it, full_exports=False, subdir=f"a{next(art_attempt)}")
                )
            else:
                exec_result = {
                    "ok": False,
                    "error": "contract check failed: " + "; ".join((chk.errors + part_errors)[:8]),
                    "traceback": None,
                    "stdout": "",
                }
            if exec_result.get("ok"):
                break
            if repair_n < max_repairs:
                # identical failure twice in a row — patching isn't converging,
                # a fresh codegen pass has better odds than more repairs
                err_sig = (exec_result.get("error") or "")[:200]
                # job-wide signature tally: the same root cause seen twice
                # across repair attempts switches the next repair to full_code
                sig = _exec_error_signature(exec_result)
                if sig.startswith("infra:"):
                    # executor-side failure (payload/schema skew, sandbox
                    # outage) — no model patch can fix infrastructure, so
                    # don't burn repair calls on it
                    await ctx.emit(
                        "repair",
                        it,
                        "executor infrastructure error — skipping model repair",
                    )
                    break
                stalled = _note_exec_error(repair_sig_counts, sig)
                if err_sig and err_sig == prev_err_sig:
                    await ctx.emit("repair", it, "same error twice — repair not converging")
                    break
                prev_err_sig = err_sig
                if stalled and sig not in stall_sigs_announced:
                    stall_sigs_announced.add(sig)
                    log.info(
                        "agent.repair_stall",
                        iteration=it,
                        signature=sig,
                        seen=repair_sig_counts[sig],
                    )
                    await ctx.emit(
                        "repair",
                        it,
                        "same exec error across repairs — switching to full_code",
                    )
                await ctx.emit("repair", it, "fixing execution error")
                history = _repair_error_history(repair_sig_counts, repair_calls)
                att = _attempt_history(attempt_log)
                if att:
                    history = f"{history}\n\n{att}" if history else att
                try:
                    repair_calls += 1
                    presp = await codegen_step.repair(
                        ctx,
                        code,
                        spec,
                        plan,
                        exec_result.get("error"),
                        exec_result.get("traceback"),
                        exec_result.get("stdout"),
                        iteration=it,
                        repair_n=repair_n + 1,
                        verify_fn=_session_verify_fn(ctx, it, art_attempt),
                        prefer_full_code=stalled,
                        geometry=exec_result.get("geometry_at_failure"),
                        render_fn=_session_render_fn(ctx, it, art_attempt),
                        render_existing_fn=_session_render_existing_fn(ctx, it, art_attempt),
                        error_history=history or None,
                    )
                except Exception as e:
                    log.warning("agent.repair_failed", iteration=it, error=str(e))
                    break
                new_code, perrs = _resolve_patch(code, presp, patch_mode)
                if new_code is None:
                    # hunks missed AND no usable full_code — retry once asking
                    # for the complete file before giving up on this repair
                    await ctx.emit("repair", it, "patch hunks missed — retrying with full_code")
                    try:
                        repair_calls += 1
                        presp = await codegen_step.repair(
                            ctx,
                            code,
                            spec,
                            plan,
                            exec_result.get("error"),
                            exec_result.get("traceback"),
                            exec_result.get("stdout"),
                            iteration=it,
                            repair_n=repair_n + 1,
                            verify_fn=_session_verify_fn(ctx, it, art_attempt),
                            prefer_full_code=True,
                            geometry=exec_result.get("geometry_at_failure"),
                            render_fn=_session_render_fn(ctx, it, art_attempt),
                            render_existing_fn=_session_render_existing_fn(ctx, it, art_attempt),
                            error_history=history or None,
                        )
                        new_code, perrs = _resolve_patch(code, presp, "never")
                    except Exception as e:
                        log.warning("agent.repair_failed", iteration=it, error=str(e))
                if new_code is None:
                    await ctx.emit("repair", it, f"patch did not apply ({len(perrs)} hunk errors)")
                    log.warning("agent.repair_patch_failed", iteration=it, errors=perrs[:3])
                    # the next improve session patches best.code — tell it the
                    # repair's patch attempt already missed the text
                    pending_patch_errors = perrs
                    break
                _log_attempt(attempt_log, f"it{it} repair{repair_n}", code, new_code)
                code = new_code
                code, codemod_notes = fix_api_misuse(code)
                if codemod_notes:
                    log.info("agent.codemod", iteration=it, notes=codemod_notes)
                    await ctx.emit("repair", it, codemod_notes[0])

        if not exec_result.get("ok"):
            if _exec_error_signature(exec_result).startswith("infra:"):
                # not a code problem — record it as infra so the issue
                # list stays honest while the patcher skips it
                exec_issue: dict[str, Any] = {
                    "source": "infra",
                    "severity": "major",
                    "what": "sandbox infrastructure error — not a code problem: "
                    + str(exec_result.get("error") or "")[:200],
                }
            else:
                exec_issue = {
                    "source": "exec",
                    "severity": "major",
                    "what": (
                        "the last patch produced code that fails to execute — "
                        "avoid that change and write simpler geometry"
                    ),
                    "fix_hint": (exec_result.get("error") or "")[:400],
                }
            issues = [*best_issues, exec_issue]
            continue

        # repairs may have patched the code — keep the response in sync so the
        # persisted version holds the EXECUTED code, not the broken original
        code_resp = code_resp.model_copy(update={"code": code})

        pre_issues = _params_vs_spec(
            spec, chk.params, mode=ctx.resolved_mode or outcome.resolved_mode or "cad"
        )
        # static lint warnings (discarded transforms, no-op rotations, 2D
        # shapes in booleans) are probable bugs — surface them to the
        # improve/repair steps as minor issues
        for w in chk.warnings:
            pre_issues.append({"source": "lint", "severity": "minor", "what": w})
        # runner warnings (e.g. hybrid sliver-collapse) tell improve the code
        # worked but used a fragile pattern worth fixing in the next pass
        for w in exec_result.get("warnings") or []:
            pre_issues.append(
                {
                    "source": "exec",
                    "severity": "major",
                    "what": str(w),
                    "fix_hint": "wrap mesh booleans in cadlib.mesh_cut()/mesh_fuse() "
                    "so sliver fragments are dropped at the op, not the result",
                }
            )
        try:
            from cadx.cad.standard_parts import expected_thread_count

            expected_threads = expected_thread_count(spec)
        except Exception as e:
            expected_threads = 0
            log.warning("standard_part.thread_spec_failed", error=str(e)[:160])
        preview_threads = int(exec_result.get("preview_thread_count") or 0)
        if expected_threads and preview_threads < expected_threads:
            pre_issues.append(
                {
                    "source": "feature",
                    "severity": "major",
                    "what": (
                        f"{expected_threads} thread feature(s) are specified but only "
                        f"{preview_threads} have visible preview metadata"
                    ),
                    "fix_hint": (
                        "keep the thread in the parametric geometry and add a "
                        "CADX_FEATURES thread record; preview geometry is display-only, "
                        "final export must use the real helix"
                    ),
                }
            )
        elif expected_threads:
            # the preview exec never exercises `CADX_DETAIL_MODE == "export"`
            # branches — probe the same code once in export detail (light:
            # geometry + feature measurement, no renders/exports) so a broken
            # export path fails THIS candidate while alternates can still win,
            # instead of only at the finalize rebuild
            probe = await ctx.executor.execute(
                _exec_payload(
                    ctx,
                    code,
                    it,
                    full_exports=False,
                    subdir=f"a{next(art_attempt)}",
                    detail_mode="export",
                    light=True,
                )
            )
            real_threads = int(probe.get("thread_count") or 0)
            fail_what = ""
            if not probe.get("ok"):
                # an executor-side outage is not a code defect — skip the
                # probe verdict rather than punish the candidate for infra
                if not _exec_error_signature(probe).startswith("infra:"):
                    fail_what = str(probe.get("error") or "unknown executor error").strip()
            elif real_threads < expected_threads:
                fail_what = f"{real_threads}/{expected_threads} real thread feature(s) built"
            else:
                missing = _export_lost_holes(
                    (exec_result.get("features") or {}).get("hole_groups") or [],
                    (probe.get("features") or {}).get("hole_groups") or [],
                )
                if missing:
                    fail_what = (
                        f"{len(missing)} hole group(s) measured in preview are gone in "
                        f"export ({', '.join(missing[:3])}) — an export-only code path "
                        "dropped geometry the preview verified"
                    )
            if fail_what:
                pre_issues.append(
                    {
                        "source": "exec",
                        "severity": "major",
                        "what": "export-detail rebuild fails where the preview passed: "
                        + fail_what[:400],
                        "fix_hint": (
                            "cadlib.tapped_hole switches preview/export internally — assign "
                            "the cutter (`_tap = cadlib.tapped_hole(...)`) and place it at "
                            "the face it drills from (mouth z=0, cuts to -length). Never "
                            "fork on `CADX_DETAIL_MODE` — the preview never exercises the "
                            "export branch, so defects there ship unverified"
                        ),
                    }
                )
        if (
            code_resp.expected_bbox
            and len(code_resp.expected_bbox) >= 3
            and spec.length_x.source == spec.width_y.source == spec.height_z.source == "stated"
        ):
            eb = code_resp.expected_bbox[:3]
            for dim, idx, name in (
                (spec.length_x, 0, "X"),
                (spec.width_y, 1, "Y"),
                (spec.height_z, 2, "Z"),
            ):
                tol = max(dim.value * tol_pct / 100.0, tol_mm)
                if abs(float(eb[idx]) - dim.value) > tol:
                    pre_issues.append(
                        {
                            "source": "contract",
                            "severity": "minor",
                            "what": (
                                f"declared expected_bbox {name}={eb[idx]:g}mm already "
                                f"violates spec {name}={dim.value:g}mm"
                            ),
                            "fix_hint": f"make the {name} dimension {dim.value:g}mm",
                        }
                    )
        elif code_resp.expected_bbox and len(code_resp.expected_bbox) < 3:
            # malformed declaration — silently skipping it hides sloppy output
            pre_issues.append(
                {
                    "source": "contract",
                    "severity": "minor",
                    "what": f"expected_bbox has {len(code_resp.expected_bbox)} "
                    "values — must be [x, y, z] in mm",
                    "fix_hint": "declare expected_bbox as [length, width, height]",
                }
            )

        # ---------- numeric checks + deterministic param autofix ----------
        analysis = exec_result.get("analysis") or {}
        dim_issues = compare_dims(spec, analysis, tol_pct, tol_mm)
        if dim_issues and s.agent_param_autofix:
            fixes = propose_fixes(
                spec, exec_result.get("params_used") or {}, analysis, tol_pct, tol_mm
            )
            if fixes:
                await ctx.emit(
                    "autofix",
                    it,
                    "trying param override: " + ", ".join(f"{k}={v:g}" for k, v in fixes.items()),
                )
                payload = _exec_payload(
                    ctx, code, it, full_exports=False, subdir=f"a{next(art_attempt)}"
                )
                payload["params_override"] = fixes
                auto_res = await ctx.executor.execute(payload)
                if auto_res.get("ok"):
                    new_dims = compare_dims(spec, auto_res.get("analysis") or {}, tol_pct, tol_mm)
                    if not new_dims:
                        await ctx.emit("autofix", it, "dimension fix applied without LLM")
                        exec_result = auto_res
                        analysis = exec_result.get("analysis") or {}
                        dim_issues = []
                    else:
                        await ctx.emit("autofix", it, "param override did not resolve dims")

        # ---------- feature verification (feedback only) ----------
        feature_findings: list[dict[str, Any]] = []
        features = exec_result.get("features") or {}
        if s.agent_feature_feedback and features and active_features(spec):
            try:
                from cadx.cad.features import MeasuredFeatures
                from cadx.cad.standard_parts import covered_feature_indices

                covered = covered_feature_indices(spec, keep_host_voids=True)
                dropped = dropped_feature_indices(spec)
                feature_findings = [
                    f.model_dump()
                    for f in compare_features(
                        [
                            f
                            for i, f in enumerate(spec.features)
                            if i not in covered and i not in dropped
                        ],
                        MeasuredFeatures(**features),
                    )
                ]
            except Exception as e:
                log.warning("features.compare_failed", error=str(e)[:200])
        if s.agent_feature_feedback and features and ctx.anchors:
            try:
                from cadx.cad.features import (
                    MeasuredFeatures,
                    compare_anchor_positions,
                )

                anchor_findings = [
                    f.model_dump()
                    for f in compare_anchor_positions(ctx.anchors, MeasuredFeatures(**features))
                ]
                _annotate_anchor_sources(anchor_findings, ctx.anchors)
                feature_findings += anchor_findings
            except Exception as e:
                log.warning("features.anchor_compare_failed", error=str(e)[:200])
        # declared connection spans (executor samples the centreline — the
        # critic cannot tell a detached strut from a connected one)
        for sp in features.get("spans") or []:
            if isinstance(sp, dict) and not sp.get("ok", True):
                feature_findings.append(
                    {
                        "feature": str(sp.get("name") or "span"),
                        "expected": "material along the declared connection path "
                        + " → ".join(str(p) for p in (sp.get("points") or [])),
                        "actual": f"only {float(sp.get('coverage') or 0.0):.0%} of the "
                        "declared span is inside material — the connection is detached",
                        "severity": "major",
                        "fix_hint": "the feature does not reach its declared endpoints — "
                        "rebuild it so both ends embed into the bodies it must connect",
                    }
                )
        # declared plate footprints: material at the host's band height must
        # not sprawl past the declared x/y ranges — a mesh-inherited pad
        # looks "flat enough" to the critic while being 2x the declared size
        for pc in features.get("plates") or []:
            if isinstance(pc, dict) and not pc.get("ok", True):
                d = pc.get("declared") or {}
                m = pc.get("measured_span_mm") or {}
                dx, dy = d.get("x") or (0, 0), d.get("y") or (0, 0)
                feature_findings.append(
                    {
                        "feature": str(pc.get("name") or "plate"),
                        "expected": (
                            f"footprint {dx[1] - dx[0]:.0f}x{dy[1] - dy[0]:.0f} mm "
                            f"(x {dx[0]:g}..{dx[1]:g}, y {dy[0]:g}..{dy[1]:g})"
                        ),
                        "actual": f"material spans {m.get('x'):g}x{m.get('y'):g} mm "
                        f"at z={pc.get('z_mid'):g}",
                        "severity": "minor",
                        "fix_hint": "trim the host plate to the declared footprint — "
                        "cut the excess material outside the declared box within the "
                        "plate's z band (mesh_cut with a frame cutter), do NOT touch "
                        "the band's connection to the rest of the body",
                    }
                )
        f_iou = (analysis or {}).get("footprint_iou")
        if f_iou is not None and f_iou < 0.85:
            # only an authoritative plan-view measurement (or an explicit
            # use_image_profile plan) may demand an outline rebuild — an
            # advisory measurement downgrades the finding to a hint.
            # Severity scales with the gap: a generative contour around
            # 0.6-0.85 follows the silhouette but differs in detail — flooring
            # the score to 0 for that is wrong (observed: knife scale with
            # IoU 0.81, all hard gates green, score 0). Only a grossly wrong
            # silhouette (<0.6) is major.
            authoritative = _measurement_authoritative(ctx) or bool(
                getattr(plan, "use_image_profile", False)
            )
            major = authoritative and f_iou < 0.6
            feature_findings.append(
                {
                    "feature": "footprint",
                    "expected": "matches the measured plan-view outline",
                    "actual": f"footprint IoU {f_iou:.2f}",
                    "severity": "major" if major else "minor",
                    "fix_hint": (
                        "footprint diverges from the measured outline — rebuild "
                        "the body from the measured outline via "
                        "load_image_profile() / profile_body()"
                        if major
                        else "footprint follows the outline with detail drift — "
                        "refine proportions locally, do NOT rebuild from the outline"
                    ),
                }
            )

        # ---------- FEM check (only when the spec declares loads) ----------
        fem_issues: list[dict[str, Any]] = []

        async def _fem_smoke(
            it: int, exec_result: dict[str, Any], fem_issues: list[dict[str, Any]]
        ) -> None:
            if not spec.load_cases:
                return
            # FEM reads the exec's STL — must come from the attempt dir the
            # current exec_result was written to, not the shared iteration dir
            it_rel = str(
                exec_result.get("art_dir")
                or storage.relpath(storage.iteration_dir(ctx.project_id, ctx.version_id, it))
            )
            await ctx.emit("fem", it, f"verifying {len(spec.load_cases)} load case(s)")
            payload = {
                "job_id": ctx.job_id,
                "project_id": str(ctx.project_id),
                "version_id": str(ctx.version_id),
                "iteration": it,
                "mesh_rel": f"{it_rel}/renders/mesh.stl",
                "load_cases": [c.model_dump() for c in spec.load_cases],
                # spec-ordered anchor table so hole_index resolves
                # against declaration order — scanned holes still
                # merge in afterwards (scan_anchors is on)
                "anchors": (ctx.anchors or {}).get("holes") or None,
                "material": ctx.settings.fem_material or spec.material,
                # iteration verify is a smoke check (solvable? sane SF?) —
                # a coarser pitch keeps it inside the task timeout on
                # long parts; the optimize phase re-verifies at full res
                "voxel_mm": max(1.5, ctx.settings.fem_voxel_mm or 1.5),
                "runtime_settings": ctx.runtime_settings.executor_values(),
            }

            async def _smoke() -> dict[str, Any]:
                # queue wait + solve both eat the iteration's codegen budget —
                # a smoke that outlives this bound is skipped honestly, not
                # allowed to stall the loop (heavy-lane contention made prop
                # smoke calls sit queued ~20min then die on cancellation)
                try:
                    return await asyncio.wait_for(
                        ctx.executor.fem_analyze(payload), timeout=_FEM_SMOKE_TIMEOUT_S
                    )
                except TimeoutError:
                    return {
                        "ok": False,
                        "error": f"fem smoke timed out after {_FEM_SMOKE_TIMEOUT_S}s — skipped",
                    }
                except Exception as e:
                    return {"ok": False, "error": str(e)}

            fem_res = await _smoke()
            # all-unsolvable declarations (regions resolving to zero force /
            # no nodes) are a spec defect the loop cannot otherwise repair —
            # load_cases are not editable mid-loop. Re-extract them once and
            # re-verify: bounded by the ctx latch, audited via emit.
            if (
                (_has_unsolvable_case(fem_res) or _has_degenerate_case(fem_res))
                and not ctx.load_repair_done
                and not await cancelled()
            ):
                ctx.load_repair_done = True
                cases, _opt = await intake_step.extract_load_cases(ctx, spec)
                if cases:
                    spec.load_cases = cases
                    intake_step.sanitize_load_cases(spec)
                    payload["load_cases"] = [c.model_dump() for c in spec.load_cases]
                    await ctx.emit(
                        "fem",
                        it,
                        f"declared cases unsolvable/degenerate — re-extracted "
                        f"{len(cases)} case(s), re-verifying",
                    )
                    fem_res = await _smoke()
            if fem_res.get("ok") and fem_res.get("fem"):
                exec_result["fem"] = fem_res["fem"]
                sf_min = spec.optimize.safety_factor_min if spec.optimize else 1.5
                rep = FemReport(**fem_res["fem"])
                worst = rep.worst_safety_factor
                # "could be lightened" advisories only belong on the issue
                # list when lightweighting was actually requested (P9)
                lw_requested = bool(spec.optimize and spec.optimize.enabled) or bool(
                    _LIGHTWEIGHT_RE.search(ctx.prompt or "")
                )
                fb_kwargs = {"lightweight_requested": lw_requested} if _FEM_LIGHTWEIGHT_KW else {}
                for severity, line, hint in fem_feedback_lines(rep, sf_min, **fb_kwargs):
                    fem_issues.append(
                        {
                            "source": "fem",
                            "severity": severity,
                            "what": line,
                            "fix_hint": hint,
                        }
                    )
                await ctx.emit(
                    "fem",
                    it,
                    f"done: worst safety factor {worst}"
                    if worst is not None
                    else "done: no solvable case",
                    metrics={"worst_sf": float(worst)} if worst is not None else None,
                )
            else:
                await ctx.emit("fem", it, f"skipped: {exec_error(fem_res, 'no result')}")

        # FEM smoke + vision critique are independent — run concurrently.
        fem_task = asyncio.create_task(_fem_smoke(it, exec_result, fem_issues))

        # ---------- vision critique ----------
        # deterministic regression guard: when a rebuild collapses the part vs
        # the best candidate (observed: improve pass returning a single tab
        # plate, volume 10% of best) the vision call only restates the obvious
        # at ~10 min of reasoning cost — synthesize the verdict instead. The
        # fix_hint is also more useful: it tells improve to keep the working
        # structure instead of "fix the plate".
        critique = _regression_critique(best, analysis)
        if critique is not None:
            await ctx.emit("critique", it, "regression vs best candidate — skipping vision call")
        else:
            await ctx.emit("critique", it, "comparing renders with your images")
            sheet_rel = exec_result.get("contact_sheet")
            sheet = storage.resolve(sheet_rel) if sheet_rel else None
            try:
                critique = await critique_step.run(
                    ctx,
                    spec,
                    sheet,
                    analysis,
                    dim_issues,
                    exec_result.get("dfm"),
                    iteration=it,
                    feature_findings=feature_findings,
                    features=features,
                    code=code,
                    # the previous iteration's issue list — lets the critic
                    # see which issues were already reported (P10)
                    **({"prev_issues": best_issues} if _CRITIQUE_PREV_ISSUES else {}),
                )
            except Exception as e:
                log.warning("agent.critique_failed", iteration=it, error=repr(e))
                critique = Critique(score=0, passed=False, confidence=0.0)
        await fem_task

        cand = Candidate(it, code_resp, exec_result, critique, dim_issues)
        cand.dq = det_quality(exec_result, dim_issues)
        cand.feature_findings = feature_findings
        # hybrid guard: an improve pass that silently drops load_mesh_solid()
        # regresses to a parametric part and no longer IS the organic body —
        # flag it deterministically instead of relying on the critique to
        # notice the missing mesh. Exception: the code may DECLARE the
        # replacement (`cad_fallback = "<reason>"`) — that candidate is then
        # scored as honest CAD, and the run resolves "cad" if it wins (B20b)
        hybrid_missing = ctx.hybrid_solid is not None and not mesh_solid_flows_to_result(
            code_resp.code
        )
        fb_reason = declared_cad_fallback(code_resp.code) if hybrid_missing else None
        if hybrid_missing and fb_reason is not None:
            cand.cad_fallback = fb_reason
            hybrid_missing = False
            log.info("agent.cad_fallback_declared", iteration=it, reason=fb_reason[:200])
        cand.hybrid_missing = hybrid_missing
        cand.wrecked = critique.score <= getattr(s, "agent_mesh_hard_fail_score", 30) and any(
            d.severity == "major" for d in critique.discrepancies
        )
        hybrid_issues = (
            [
                {
                    "source": "hybrid",
                    "severity": "major",
                    "what": "hybrid part does not use the mesh body — load_mesh_solid() "
                    "is missing or its result never reaches `result`",
                    "fix_hint": "build ON `mesh = load_mesh_solid()` (mesh_fuse/mesh_cut "
                    "for every feature) and derive `result` from it; a parked/unused "
                    "mesh variable does not satisfy the hybrid contract",
                }
            ]
            if hybrid_missing
            else []
        )
        cur_issues = (
            pre_issues
            + fem_issues
            + hybrid_issues
            + build_issues(
                critique,
                dim_issues,
                feature_findings,
                exec_result.get("dfm"),
                exec_result.get("analysis"),
                spec=spec,
            )
        )
        # issue-fingerprint stall (P10): an issue whose number-normalised
        # text appears in 2+ consecutive iterations while the measured
        # geometry did not change is a phantom — a measurement artifact,
        # not a code defect. Flag it (the issue stays on the persisted
        # list for honesty) so the improve feed above skips it. Two guards:
        # the hash covers EVERY measured feature class (an old 3-key subset
        # reported "geometry unchanged" while fillets/bosses/pockets moved),
        # and only LLM-subjective repeats are phantom candidates — a
        # deterministic finding (measured features, dims, FEM, DFM, solids)
        # that verify_spec re-measures at gate time is either a real defect
        # the patcher must keep seeing or a detector bug the gate fails on
        # anyway, so suppressing it either strands a fixable defect or saves
        # nothing (observed: a never-built spec fillet flagged as artifact
        # after it2, patcher never saw it again, codegen gate doomed).
        flaggable = {"critique", "verify_question"}
        feat_hash = json.dumps(
            {k: v for k, v in features.items() if k not in ("recognition_ran", "approximate")},
            sort_keys=True,
            default=str,
        )
        cur_fps: dict[str, int] = {}
        stalled_issues = 0
        for issue in cur_issues:
            fp = re.sub(r"[\d.]+", "#", (issue.get("what") or "").lower().strip())
            cur_fps[fp] = prev_issue_fps.get(fp, 0) + 1
            if (
                cur_fps[fp] >= 2
                and issue.get("source") in flaggable
                and feat_hash == prev_features_hash
            ):
                issue["artifact"] = True
                stalled_issues += 1
        if stalled_issues:
            log.info("agent.issue_stall", iteration=it, count=stalled_issues)
            await ctx.emit(
                "improve",
                it,
                f"{stalled_issues} issue(s) unchanged 2+ iterations — "
                "likely a measurement artifact",
            )
        prev_issue_fps = cur_fps
        prev_features_hash = feat_hash
        best_replaced = best is None or cand.rank > best.rank
        best, best_issues = _select_best_candidate(best, best_issues, cand, cur_issues)
        # Keep the live issue list paired with the selected code even when a
        # passing candidate breaks out of the loop immediately below.
        issues = best_issues
        cur_score = cand.critique.score if cand.critique else 0
        # progress = a new best OR a rise in either quality number — a
        # saturated critique score (0 after enough majors) must not read as
        # a stall while verify_spec keeps improving
        if best_replaced or cand.dq > best_dq or cur_score > best_score:
            last_progress_it = it
            best_dq = max(best_dq, cand.dq)
            best_score = max(best_score, cur_score)
        elif it - last_progress_it >= s.llm_thinking_escalate_after and _escalate_thinking("stall"):
            await ctx.emit("improve", it, f"stalled — reasoning level {ctx.thinking_level}")
        # regression: a candidate far below the best measured quality means
        # the last change went backwards — reason about it before patching on
        if (
            best is not None
            and cand is not best
            and cand.dq <= best.dq - 10
            and _escalate_thinking("regression")
        ):
            await ctx.emit(
                "improve",
                it,
                f"quality regressed (dq {cand.dq} vs best {best.dq}) — "
                f"reasoning level {ctx.thinking_level}",
            )

        major = [d for d in critique.discrepancies if d.severity == "major"]
        # the major/minor count covers critique + verify_question sourced
        # issues — other sources (verify_spec, dims, fem, …) are rolled up
        # separately so "0 issues" here never collides with "fixing N" there
        extra = _issue_source_rollup(issues, exclude={"critique", "verify_question"})
        n_extra = sum(
            1
            for i in issues
            if not i.get("artifact")
            and str(i.get("source") or "critique") not in {"critique", "verify_question", "infra"}
        )
        await ctx.emit(
            "critique",
            it,
            f"score {critique.score}/100 — {len(major)} major, "
            f"{len(critique.discrepancies) - len(major)} minor discrepancies"
            + (f" (+{n_extra} issues: {extra})" if extra else ""),
            score=critique.score,
        )

        if (
            critique.passed
            and critique.score >= pass_score
            and not dim_issues
            and not major
            and not fem_issues
            and not hybrid_missing
            and not _multi_solid(exec_result)
        ):
            await ctx.emit("done", it, "all checks passed")
            break
        issues = best_issues

        # strategy plateau hook — a second approach may do better than more
        # iterations on a candidate stuck below the switch score
        if switch_check is not None and switch_check(
            it - start_iteration + 1,
            best.critique.score if best is not None and best.critique else None,
        ):
            await ctx.emit("plan", it, "score plateaued — switching approach")
            outcome.switch_requested = True
            break

        # stall early-exit: no progress for `agent_stall_patience` iterations
        # (progress = new best, dq rise or score rise — see above). Before
        # finalising, a stalled run gets ONE fresh codegen from the issue
        # list: improve patches keep circling the same broken code, a clean
        # regen conditioned on the accumulated issues is the cheaper escape.
        stalled = (
            best is not None
            and bool(s.agent_stall_patience)
            and it - last_progress_it >= s.agent_stall_patience
        )
        if stalled and not fresh_retry_used and it + 1 < start_iteration + max_iterations:
            best_score_val = best.critique.score if best is not None and best.critique else 0
            if best_score_val < pass_score:
                fresh_retry_used = True
                force_fresh = True
                last_progress_it = it
                await ctx.emit(
                    "plan",
                    it,
                    "stalled — one fresh codegen from the issue list before finalising",
                )
                continue
        if stalled:
            await ctx.emit(
                "finalize",
                it,
                f"no improvement for {s.agent_stall_patience} iterations — keeping best",
            )
            break

    outcome.iterations = iterations_done

    # ---------- FINALIZE ----------
    if best is None:
        outcome.error = "all iterations failed to produce executable code"
        return outcome

    final_exec = best.exec_result
    expected_threads = 0
    try:
        from cadx.cad.standard_parts import expected_thread_count

        expected_threads = expected_thread_count(spec)
    except Exception as e:
        log.warning("standard_part.thread_spec_failed", error=str(e)[:160])

    if expected_threads:
        await ctx.emit(
            "finalize",
            best.iteration,
            "rebuilding selected candidate with real thread geometry",
        )
        base_code = best.code_response.code if best.code_response else ""
        params_ovr = best.exec_result.get("params_used") or None

        async def _export_rebuild(code: str, subdir: str = "final_export") -> dict[str, Any]:
            return await ctx.executor.execute(
                _exec_payload(
                    ctx,
                    code,
                    best.iteration,
                    full_exports=False,
                    subdir=subdir,
                    detail_mode="export",
                    params_override=params_ovr,
                )
            )

        exact = await _export_rebuild(base_code)
        if base_code and (
            not exact.get("ok") or int(exact.get("thread_count") or 0) < expected_threads
        ):
            # export-only failure class: the preview build passed, so the defect
            # is usually a fragile construction that only the real helical groove
            # exposes (helical bore edges defeat `edges() >> Axis.Z` selections —
            # that operator sorts, it does not filter). Feed it back as ONE
            # bounded improve pass — a failed export must never silently ship
            # preview geometry.
            err_text = (
                str(exact.get("error") or "").strip()
                or f"only {int(exact.get('thread_count') or 0)}/{expected_threads} "
                "real thread feature(s) built"
            )
            await ctx.emit(
                "improve",
                best.iteration,
                "export-detail rebuild failed — one repair pass",
            )
            issue = {
                "source": "exec",
                "severity": "major",
                "what": (
                    "EXPORT-detail rebuild of the selected candidate failed while "
                    "the preview build passed: " + err_text[:400]
                ),
                "fix_hint": (
                    "export detail runs the identical code but cadlib.tapped_hole "
                    "builds a real helical groove, adding helical bore edges — "
                    "select fillet/chamfer edges with edges_at(axis='z', geom='line') "
                    "(never `edges() >> Axis.Z`, which sorts without filtering), and "
                    "keep the tapped_hole cutter assignment so its thread tag survives"
                ),
            }
            new_code: str | None = None
            patch_errors: list[str] | None = None
            for attempt in range(1, s.agent_patch_retries + 1):
                try:
                    presp = await improve_step.run(
                        ctx,
                        spec,
                        plan,
                        base_code=base_code,
                        issues=[issue],
                        features=best.exec_result.get("features") or None,
                        patch_errors=patch_errors,
                        attempts=attempt_log,
                        iteration=best.iteration,
                        attempt=attempt,
                        verify_fn=_session_verify_fn(ctx, best.iteration, art_attempt),
                        render_fn=_session_render_fn(ctx, best.iteration, art_attempt),
                        render_existing_fn=_session_render_existing_fn(
                            ctx, best.iteration, art_attempt
                        ),
                    )
                except Exception as e:
                    log.warning("agent.finalize_repair_failed", error=repr(e))
                    break
                new_code, patch_errors = _resolve_patch(base_code, presp, patch_mode)
                if new_code is not None:
                    _log_attempt(attempt_log, "export repair", base_code, new_code)
                    break
            if new_code is not None:
                # preview sanity: a repair that breaks the preview build cannot
                # become the export candidate
                sanity = await ctx.executor.execute(
                    _exec_payload(
                        ctx,
                        new_code,
                        best.iteration,
                        full_exports=False,
                        subdir=f"a{next(art_attempt)}",
                        light=True,
                        params_override=params_ovr,
                    )
                )
                if sanity.get("ok"):
                    retry = await _export_rebuild(new_code, subdir="final_export_repair")
                    if retry.get("ok") and int(retry.get("thread_count") or 0) >= expected_threads:
                        if best.code_response is not None:
                            best.code_response = best.code_response.model_copy(
                                update={"code": new_code}
                            )
                        exact = retry
                        await ctx.emit(
                            "finalize",
                            best.iteration,
                            "export repair applied — real thread geometry verified",
                        )
        if not exact.get("ok"):
            outcome.error = (
                "final real-thread build failed; preview geometry was not exported: "
                + str(exact.get("error") or "unknown executor error")[:400]
            )
            outcome.status = "failed"
            return outcome
        actual_threads = int(exact.get("thread_count") or 0)
        if actual_threads < expected_threads:
            outcome.error = (
                f"final export contains {actual_threads} real thread feature(s), "
                f"but {expected_threads} were requested; preview geometry was not exported"
            )
            outcome.status = "failed"
            outcome.issues.append(
                {
                    "source": "feature",
                    "severity": "major",
                    "what": outcome.error,
                }
            )
            return outcome
        final_exec = exact
        await ctx.emit(
            "finalize",
            best.iteration,
            f"verified {actual_threads} real thread feature(s) for export",
        )

    await ctx.emit("finalize", best.iteration, "exporting all formats")
    exports = await ctx.executor.export_all(
        {
            "job_id": ctx.job_id,
            "project_id": str(ctx.project_id),
            "version_id": str(ctx.version_id),
            "iteration": best.iteration,
            "art_dir": final_exec.get("art_dir"),
            "name": "model",
        }
    )
    if exports.get("ok") is False:
        outcome.error = str(exports.get("error") or "final exports failed")[:500]
        outcome.status = "failed"
        return outcome
    if expected_threads:
        missing_formats = [
            fmt
            for fmt in ("step", "stl", "3mf", "glb")
            if not (exports.get("exports") or {}).get(fmt)
        ]
        if missing_formats:
            outcome.error = (
                "real-thread export incomplete; missing "
                + ", ".join(missing_formats)
                + ". No smooth-thread fallback was accepted."
            )
            outcome.status = "failed"
            return outcome
    artifacts = _copy_best_artifacts(ctx, best.iteration, final_exec)
    for fmt, rel in (exports.get("exports") or {}).items():
        if rel:
            artifacts[f"export:{fmt}"] = rel
    artifacts.update(_copy_fem_artifacts(ctx, best.iteration))

    final_dim_issues = compare_dims(spec, final_exec.get("analysis") or {}, tol_pct, tol_mm)
    outcome.status = (
        "succeeded"
        if (
            best.critique
            and best.critique.passed
            and not best.dim_issues
            and not final_dim_issues
            and (expected_threads > 0 or not _multi_solid(final_exec))
        )
        else "needs_review"
    )
    outcome.art_dir = final_exec.get("art_dir")
    _clean_losing_attempts(ctx, outcome.art_dir)
    outcome.fem = final_exec.get("fem") or best.exec_result.get("fem")
    outcome.code = best.code_response.code if best.code_response else None
    outcome.params = dict(final_exec.get("params_used") or {})
    outcome.params_schema = (
        [p.model_dump() for p in best.code_response.params] if best.code_response else None
    )
    outcome.analysis = final_exec.get("analysis")
    outcome.features = final_exec.get("features")
    outcome.critique = best.critique.model_dump() if best.critique else None
    if outcome.critique is not None:
        outcome.critique["feature_findings"] = best.feature_findings
        outcome.critique["issues"] = issues
    outcome.dfm = final_exec.get("dfm")
    outcome.bom = final_exec.get("bom")
    outcome.score = best.critique.score if best.critique else None
    outcome.best_iteration = best.iteration
    # declared CAD fallback won (B20b): the final part is an honest
    # parametric replacement, not a hybrid — resolve/report "cad" so the
    # outcome never claims a mesh body the code discarded. The plan is NOT
    # rewritten mid-loop; only the final mode is corrected here (before the
    # hybrid artifact copy below so it correctly stays empty)
    if best.cad_fallback:
        outcome.resolved_mode = "cad"
        ctx.resolved_mode = "cad"
        outcome.mode_note += f" | declared CAD fallback: {best.cad_fallback[:120]}"
        log.info(
            "agent.cad_fallback_resolved",
            iteration=best.iteration,
            reason=best.cad_fallback[:200],
        )
    # canonical verify_spec verdict heads the persisted issue list; the
    # per-source issues after it carry the per-finding detail
    vs_entry = _verify_spec_issue(final_exec, stage="cad")
    drop_issues = feature_drop_issues(spec)
    outcome.issues = ([vs_entry] if vs_entry else []) + drop_issues + issues
    if ctx.resolved_mode == "hybrid" and ctx.mesh_info:
        prep_it = int(ctx.mesh_info.get("_prep_it", 1))
        artifacts.update(
            _copy_mesh_artifacts(
                ctx.project_id, ctx.version_id, prep_it, ctx.mesh_info.get("_prep_art_dir")
            )
        )
        outcome.mesh = {k: v for k, v in ctx.mesh_info.items() if not k.startswith("_")}
        _write_mesh_json(ctx.project_id, ctx.version_id, ctx.mesh_info)
    outcome.artifacts = artifacts
    outcome.usage = dict(getattr(ctx.llm, "total_usage", {}))
    return outcome
def _select_best_candidate(
    best: Candidate | None,
    best_issues: list[dict[str, Any]],
    candidate: Candidate,
    candidate_issues: list[dict[str, Any]],
) -> tuple[Candidate, list[dict[str, Any]]]:
    if best is None or candidate.rank > best.rank:
        return candidate, candidate_issues
    return best, best_issues
def _session_render_existing_fn(ctx: AgentCtx, iteration: int, attempt: Any):
    """render_model fast path: render PNGs from the mesh.stl a previous
    exec already produced — no sandbox rerun. The session only calls this
    when the buffer hash proves the source attempt ran identical code."""

    def run(source_art_dir: str):
        payload = _exec_payload(ctx, "", iteration, full_exports=False, subdir=f"a{next(attempt)}")
        payload["render_from"] = source_art_dir
        return ctx.executor.execute(payload)

    return run
def _session_render_fn(ctx: AgentCtx, iteration: int, attempt: Any):
    """render_model callback: same sandbox run as verify but through the FULL
    pipeline — the point is the contact-sheet image (verify runs light)."""

    def run(code: str):
        return ctx.executor.execute(
            _exec_payload(ctx, code, iteration, full_exports=False, subdir=f"a{next(attempt)}")
        )

    return run
def _session_verify_fn(ctx: AgentCtx, iteration: int, attempt: Any):
    """verify-tool callback: execute session buffer code in the sandbox.

    `attempt` is a shared counter — every sandbox exec writes artifacts and
    must get its own subdir so a verify can't clobber the candidate files.
    """

    def run(code: str):
        return ctx.executor.execute(
            _exec_payload(
                ctx, code, iteration, full_exports=False, subdir=f"a{next(attempt)}", light=True
            )
        )

    return run
