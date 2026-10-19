"""CODEGEN / REPAIR — produce build123d code (code model)."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from cadx.agent.context import (
    anchors_block,
    bindings_block,
    catalog_block,
    codegen_system,
    library_block,
    measurements_block,
    numbered_code_block,
    plan_block,
    references_block,
    request_block,
    spec_block,
)
from cadx.agent.docfetch import docs_for_error
from cadx.agent.thinking import thinking_body
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession, VerifyFn, elision_marker
from cadx.agent.types import AgentCtx
from cadx.cad.allowlist import check_code
from cadx.cad.errordiag import diag_block, diagnose
from cadx.cad.partspec import check_standard_part_calls
from cadx.llm.schemas import (
    CodeResponse,
    ModelingPlan,
    PartSpec,
    PatchResponse,
)
from cadx.logging import get_logger

log = get_logger(__name__)

STEP = "codegen"

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# seed buffer for staged-build sessions: the minimal valid frame the model
# grows via add_block — the contract (PARAMS + `result`) is enforced on done
_BUILD_SEED = '''"""build123d part — assembled feature-by-feature via add_block."""

from build123d import *

PARAMS = {
    # "name": default_value,  # one entry per tunable dimension
}

# Feature blocks land below this comment. The final buffer must be a complete
# runnable script that assigns the finished solid to `result`.
'''


def _extract_code_block(text: str) -> str | None:
    """Pull a fenced ```python block out of a prose answer, if present."""
    if not text:
        return None
    m = _CODE_FENCE.search(text)
    return m.group(1).strip() if m else None


def _drop_elided(presp: PatchResponse, *, step: str) -> PatchResponse:
    """Drop lazy-elided code from a PatchResponse BEFORE it can be applied:
    ``full_code``/edit-``replace`` text carrying placeholder comments
    ("# ... rest unchanged") is not real code — removing it degrades the
    answer to "patch didn't apply", which the caller already retries with a
    prefer_full_code pass."""
    from cadx.logging import get_logger

    log = get_logger(__name__)
    if presp.full_code and (marker := elision_marker(presp.full_code)):
        log.info("codegen.elided_full_code", step=step, marker=marker[:80])
        presp.full_code = None
        presp.reasoning = (
            presp.reasoning + " [dropped: full_code contained an elision placeholder]"
        ).strip()
    kept = [e for e in presp.edits if elision_marker(e.replace) is None]
    if len(kept) != len(presp.edits):
        log.info("codegen.elided_edits", step=step, dropped=len(presp.edits) - len(kept))
        presp.edits = kept
    return presp


def _quickref_part(ctx: AgentCtx) -> dict[str, Any] | None:
    """Runtime-introspected signature digest for the user message — gated by
    the agent_quickref_context runtime setting (the quickref TOOL is always
    available; this pins the most relevant digests into context)."""
    if not ctx.runtime_settings.agent_quickref_context:
        return None
    from cadx.agent.quickref import context_block

    text = context_block()
    return {"type": "text", "text": text} if text else None


def _user_parts(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    feedback: str | None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    rb = request_block(ctx.prompt)
    if rb:
        parts.append({"type": "text", "text": rb})
    parts.append({"type": "text", "text": spec_block(spec)})
    if plan is not None:
        parts.append({"type": "text", "text": plan_block(plan)})
    bb = bindings_block(spec)
    if bb:
        parts.append({"type": "text", "text": bb})
    if qr := _quickref_part(ctx):
        parts.append(qr)
    cat = catalog_block(plan, spec)
    if cat:
        parts.append({"type": "text", "text": cat})
    if ctx.hybrid_solid is not None:
        bbox = ctx.hybrid_mesh_info.get("bbox_size") or "unknown"
        tris = ctx.hybrid_mesh_info.get("triangles") or "?"
        feats = (
            "\nHybrid CAD features to implement: " + "; ".join(plan.hybrid_cad_features)
            if plan and plan.hybrid_cad_features
            else ""
        )
        parts.append(
            {
                "type": "text",
                "text": (
                    "HYBRID MODE: a generated organic mesh is already converted to a B-Rep "
                    f"solid and available via `load_mesh_solid()` (faceted solid, ~{tris} "
                    f"triangles). It is oriented and scaled: rests on the XY plane, centred "
                    f"at origin, bbox LxWxH = {bbox} mm (also in `MESH_BBOX`). "
                    "Use it as the base body and add the exact CAD features via boolean "
                    "ops — the mesh supplies the organic shape, your code supplies the "
                    "precision features." + feats
                ),
            }
        )
    ref = references_block(ctx.assets)
    if ref:
        parts.append({"type": "text", "text": ref})
    lib = library_block(ctx.library_templates)
    if lib:
        parts.append({"type": "text", "text": lib})
    mm = measurements_block(ctx.measurements)
    if mm:
        parts.append({"type": "text", "text": mm})
    ab = anchors_block(ctx.anchors)
    if ab:
        parts.append({"type": "text", "text": ab})
    if ctx.mode == "edit" and ctx.parent.code:
        parts.append(
            {
                "type": "text",
                "text": (
                    "PREVIOUS VERSION code — modify it minimally; keep PARAMS names "
                    "stable unless the change requires otherwise:\n"
                    f"```python\n{ctx.parent.code}\n```"
                ),
            }
        )
    if feedback:
        parts.append(
            {
                "type": "text",
                "text": ("PREVIOUS ATTEMPT — issues to fix, ALL of them:\n" + feedback),
            }
        )
    return parts


async def run(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    *,
    iteration: int,
    feedback: str | None = None,
    verify_fn: VerifyFn | None = None,
    render_fn: VerifyFn | None = None,
) -> CodeResponse:
    if tools_enabled(ctx, "code"):
        n = int(ctx.runtime_settings.agent_codegen_samples)
        # best-of-N on the FIRST codegen attempt only — repairs, improves and
        # later iterations are conditioned on a specific failure/best code and
        # stay single-shot
        if n > 1 and iteration == 1:
            return await _codegen_tournament(
                ctx,
                spec,
                plan,
                n=n,
                iteration=iteration,
                feedback=feedback,
                verify_fn=verify_fn,
                render_fn=render_fn,
            )
        return await _codegen_session(
            ctx,
            spec,
            plan,
            iteration=iteration,
            feedback=feedback,
            verify_fn=verify_fn,
            render_fn=render_fn,
        )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": codegen_system(
                plan=plan, spec=spec, fresh=True, hybrid=ctx.hybrid_solid is not None
            ),
        },
        {"role": "user", "content": _user_parts(ctx, spec, plan, feedback)},
    ]
    resp = await ctx.llm.chat(
        "code", messages, json_schema=CodeResponse, extra_body=thinking_body(ctx, STEP)
    )
    assert isinstance(resp.parsed, CodeResponse)
    await ctx.trace.record(
        step=STEP,
        iteration=iteration,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed={
            "reasoning": resp.parsed.reasoning,
            "expected_bbox": resp.parsed.expected_bbox,
            "params": [p.model_dump() for p in resp.parsed.params],
            "code": "<code>",
        },
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    if marker := elision_marker(resp.parsed.code):
        from cadx.llm.structured import LLMFormatError

        raise LLMFormatError(
            f"codegen returned elided code ('{marker[:80]}') — complete file required"
        )
    return resp.parsed


async def repair(
    ctx: AgentCtx,
    prev_code: str,
    spec: PartSpec,
    plan: ModelingPlan | None,
    error: str | None,
    traceback: str | None,
    stdout: str | None,
    *,
    iteration: int,
    repair_n: int,
    verify_fn: VerifyFn | None = None,
    prefer_full_code: bool = False,
    geometry: list[dict[str, Any]] | None = None,
    render_fn: VerifyFn | None = None,
    error_history: str | None = None,
) -> PatchResponse:
    """Ask the model to fix broken code — tool session when the provider
    supports it, else one-shot PatchResponse (edits preferred, full_code
    fallback).

    ``error_history`` is the orchestrator's deterministic digest of distinct
    exec-error signatures seen job-wide — appended to the repair context so a
    repeated failure carries memory of what already failed (no extra LLM
    call). The failing symbol's introspected quickref digest rides along too
    when the error names one."""
    from cadx.agent.context import system_prompt

    diag = diagnose(error, traceback, prev_code)
    docs = docs_for_error(error, traceback, prev_code, diag.failing_line)
    diag_text = diag_block(diag, error, traceback, stdout, geometry=geometry)
    if docs:
        diag_text += (
            "\n\nCORRECT API (introspected from the installed packages — "
            "trust this over memory):\n" + docs
        )
    if error_history:
        diag_text += "\n\n" + error_history
    if tools_enabled(ctx, "code") and ctx.edit_sessions_exhausted < 1:
        return await _repair_session(
            ctx,
            prev_code,
            spec,
            plan,
            diag_text,
            iteration=iteration,
            repair_n=repair_n,
            verify_fn=verify_fn,
            max_calls=ctx.runtime_settings.agent_tool_max_calls,
            render_fn=render_fn,
        )
    if prefer_full_code:
        user = (
            "Fix this code. Return a PatchResponse JSON object with `full_code` "
            "containing the COMPLETE corrected file — earlier `edits` hunks did "
            "not match the code text, do NOT return `edits`."
        )
    else:
        user = (
            "Fix this code. Return a PatchResponse JSON object — prefer `edits` for "
            "localised fixes, `full_code` only when the fix is structural."
        )
    parts: list[dict[str, Any]] = [{"type": "text", "text": user}]
    rb = request_block(ctx.prompt)
    if rb:
        parts.append({"type": "text", "text": rb})
    if spec is not None:
        parts.append({"type": "text", "text": spec_block(spec)})
    if plan is not None:
        parts.append({"type": "text", "text": plan_block(plan)})
    bb = bindings_block(spec)
    if bb:
        parts.append({"type": "text", "text": bb})
    if qr := _quickref_part(ctx):
        parts.append(qr)
    cat = catalog_block(plan, spec)
    if cat:
        parts.append({"type": "text", "text": cat})
    # APC ordering: the per-attempt code + diagnosis is the dynamic tail of
    # the prompt — every block above is byte-stable across repair attempts
    # on the same job, so the prefix cache survives repair_1 -> repair_2+
    parts.append(
        {
            "type": "text",
            "text": "PREVIOUS CODE:\n```python\n" + prev_code + "\n```\n\n" + diag_text,
        }
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt("repair_system.md")
            + "\n\n"
            + codegen_system(
                plan=plan, spec=spec, fresh=False, hybrid=ctx.hybrid_solid is not None
            ),
        },
        {"role": "user", "content": parts},
    ]
    resp = await ctx.llm.chat(
        "code",
        messages,
        json_schema=PatchResponse,
        temperature=0.1,
        extra_body=thinking_body(ctx, f"repair_{repair_n}"),
    )
    assert isinstance(resp.parsed, PatchResponse)
    await ctx.trace.record(
        step=f"repair_{repair_n}",
        iteration=iteration,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed={
            "reasoning": resp.parsed.reasoning,
            "diag": diag.category,
            "n_edits": len(resp.parsed.edits),
            "has_full_code": bool(resp.parsed.full_code),
        },
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    return _drop_elided(resp.parsed, step=f"repair_{repair_n}")


async def _repair_session(
    ctx: AgentCtx,
    prev_code: str,
    spec: PartSpec | None,
    plan: ModelingPlan | None,
    diag_text: str,
    *,
    iteration: int,
    repair_n: int,
    verify_fn: VerifyFn | None,
    max_calls: int,
    render_fn: VerifyFn | None = None,
) -> PatchResponse:
    from cadx.agent.context import system_prompt

    session = ToolSession(
        ctx, kind="edit", code=prev_code, verify_fn=verify_fn, render_fn=render_fn
    )
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": "Fix this code using the tools — minimal edit only."},
    ]
    rb = request_block(ctx.prompt)
    if rb:
        parts.append({"type": "text", "text": rb})
    if spec is not None:
        parts.append({"type": "text", "text": spec_block(spec)})
    if plan is not None:
        parts.append({"type": "text", "text": plan_block(plan)})
    bb = bindings_block(spec)
    if bb:
        parts.append({"type": "text", "text": bb})
    if qr := _quickref_part(ctx):
        parts.append(qr)
    cat = catalog_block(plan, spec)
    if cat:
        parts.append({"type": "text", "text": cat})
    # APC ordering: numbered code + diagnosis change every attempt — they go
    # LAST so the stable head (system + request/spec/plan/context) stays a
    # shared prefix across repair sessions
    parts.append({"type": "text", "text": numbered_code_block(prev_code, "PREVIOUS CODE")})
    parts.append({"type": "text", "text": diag_text})
    system = "\n\n".join(
        [
            system_prompt("repair_tools.md"),
            system_prompt("tools_protocol.md"),
            codegen_system(plan=plan, spec=spec, fresh=False, hybrid=ctx.hybrid_solid is not None),
        ]
    )
    res = await run_session(
        ctx,
        "code",
        [
            {"role": "system", "content": system},
            {"role": "user", "content": parts},
        ],
        session=session,
        max_calls=max_calls,
        step=f"repair_{repair_n}",
        iteration=iteration,
        temperature=0.1,
    )
    if res.exhausted:
        ctx.edit_sessions_exhausted += 1
    if res.code and res.code != prev_code and elision_marker(res.code) is None:
        sub = res.submitted or {}
        return PatchResponse(
            reasoning=str(sub.get("reasoning", "")),
            expected_bbox=list(sub.get("expected_bbox") or []),
            full_code=res.code,
        )
    # last resort: the model ignored the tools but answered with a fenced
    # code block — still usable, better than failing the whole iteration
    fenced = _extract_code_block(res.response.text if res.response else "")
    if fenced and fenced != prev_code and elision_marker(fenced) is None:
        return PatchResponse(
            reasoning="model answered with a fenced code block instead of tool edits",
            full_code=fenced,
        )
    return PatchResponse(reasoning="tool session produced no changes")


async def _codegen_session(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    *,
    iteration: int,
    feedback: str | None,
    verify_fn: VerifyFn | None = None,
    render_fn: VerifyFn | None = None,
    out: dict[str, Any] | None = None,
) -> CodeResponse:
    """Fresh generation via tool session.

    Default: the model may look up catalog parts, check its draft with
    check_snippet, then submit_code. With ``agent_staged_build`` on (or
    "auto" — the default — while tools are enabled) it runs a *build*
    session instead: a seeded buffer grown feature-by-feature via
    add_block, sandbox-verified per stage, checkpointed, finished by done.
    A plain content answer is still accepted (CodeResponse schema).

    ``out`` (best-of-N tournament): when given, the session object is stored
    under ``out["session"]`` so the caller can mine its tool_log for verify
    evidence without changing the return contract."""
    from cadx.agent.context import codegen_build_system, codegen_tools_system
    from cadx.llm.schemas import ParamDef
    from cadx.llm.structured import LLMFormatError

    staged = ctx.runtime_settings.agent_staged_build == "on" or (
        ctx.runtime_settings.agent_staged_build == "auto" and tools_enabled(ctx, "code")
    )
    if staged:
        session = ToolSession(
            ctx,
            kind="build",
            code=_BUILD_SEED,
            verify_fn=verify_fn,
            render_fn=render_fn,
        )
        system = codegen_build_system(
            plan=plan, spec=spec, fresh=True, hybrid=ctx.hybrid_solid is not None
        )
    else:
        session = ToolSession(ctx, kind="codegen")
        system = codegen_tools_system(
            plan=plan, spec=spec, fresh=True, hybrid=ctx.hybrid_solid is not None
        )
    if out is not None:
        out["session"] = session
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": _user_parts(ctx, spec, plan, feedback)},
    ]
    res = await run_session(
        ctx,
        "code",
        messages,
        session=session,
        json_schema=CodeResponse,
        # staged builds spend ~2 calls per feature (add_block + verify) plus
        # checkpoints — a one-shot budget would strand a 5-feature part
        max_calls=(
            ctx.runtime_settings.agent_tool_max_calls_build
            if staged
            else ctx.runtime_settings.agent_tool_max_calls
        ),
        step=STEP,
        iteration=iteration,
    )
    if (
        res.submitted
        and res.submitted.get("code")
        and elision_marker(str(res.submitted["code"])) is None
    ):
        code = str(res.submitted["code"])
        chk = check_code(code)
        # PARAMS literals in the code are authoritative; the model's `params`
        # arg only supplies metadata (unit/min/max/description)
        meta = {
            str(p.get("name")): p
            for p in (res.submitted.get("params") or [])
            if isinstance(p, dict) and p.get("name")
        }

        def _num(x: Any) -> float | None:
            return float(x) if isinstance(x, int | float) else None

        params = [
            ParamDef(
                name=k,
                value=v,
                unit=str(meta.get(k, {}).get("unit") or "mm"),
                min=_num(meta.get(k, {}).get("min")),
                max=_num(meta.get(k, {}).get("max")),
                description=str(meta.get(k, {}).get("description") or ""),
            )
            for k, v in (chk.params or {}).items()
            if isinstance(v, str | int | float | bool)
        ]
        return CodeResponse(
            reasoning=str(res.submitted.get("reasoning", "")),
            params=params,
            expected_bbox=list(res.submitted.get("expected_bbox") or []),
            code=code,
        )
    if res.response is not None and isinstance(res.response.parsed, CodeResponse):
        return res.response.parsed
    # build-session rescue: the model ran out of budget without `done` but
    # leaves a grown buffer — submit it (it goes through the same contract
    # gate as any codegen; a broken buffer gets a repair session, which is
    # strictly better than throwing the work away)
    if staged and res.code and res.code != _BUILD_SEED and elision_marker(res.code) is None:
        return CodeResponse(
            reasoning="staged build session ended without done — buffer submitted as-is",
            params=[],
            expected_bbox=[],
            code=str(res.code),
        )
    raise LLMFormatError("codegen session produced no code")


# --------------------------------------------------------------------------- #
# best-of-N first-attempt codegen (agent_codegen_samples > 1)
# --------------------------------------------------------------------------- #

# tools that executed the session buffer in the sandbox — a PASS on the LAST
# one is free verification evidence for the candidate
_EXEC_CHECK_TOOLS = frozenset({"verify", "render_model"})
# tools that mutate the buffer — an exec check is stale once one lands after it
_BUFFER_EDIT_TOOLS = frozenset(
    {
        "add_block",
        "replace_lines",
        "insert_after",
        "insert_before",
        "find_replace",
        "set_param",
        "rollback",
        "submit_code",
    }
)


@dataclass
class _Candidate:
    """One tournament entry: the produced response plus its own evidence."""

    index: int
    resp: CodeResponse
    session: ToolSession | None
    # None = never verified; True/False = sandbox outcome of the final code
    verified: bool | None = None
    # "session" = the session's own verify/render tool (free); "exec" = one
    # post-hoc light verify_fn run on the submitted code
    verify_source: str = ""
    contract_errors: list[str] = field(default_factory=list)


def _in_session_verify(session: ToolSession | None, code: str) -> bool | None:
    """Sandbox outcome the candidate already earned inside its own session.

    True/False when the LAST verify/render tool call ran against the final
    buffer (no buffer-mutating tool call after it — a build session's verify
    on the submitted code is conclusive, pass or fail). None when no exec
    tool ran, its logged result was truncated, edits landed after the last
    exec, or the verified buffer isn't the code being scored (stale evidence
    — the caller re-verifies the submitted code)."""
    if session is None or session.buffer.strip() != code:
        return None
    last_exec: dict[str, Any] | None = None
    last_exec_i = -1
    last_edit_i = -1
    for i, entry in enumerate(session.tool_log):
        name = entry.get("name")
        if name in _EXEC_CHECK_TOOLS:
            last_exec, last_exec_i = entry, i
        elif name in _BUFFER_EDIT_TOOLS:
            last_edit_i = i
    if last_exec is None or last_edit_i > last_exec_i:
        return None
    res = last_exec.get("result")
    if not isinstance(res, dict):  # _truncate() may have stringified it
        return None
    return bool(res.get("ok"))


def _candidate_contract_errors(ctx: AgentCtx, code: str) -> list[str]:
    """Contract errors for scoring — mirrors ToolSession._contract_errors
    (static check + standard-part call validity, agent_sigcheck honoured)."""
    chk = check_code(code, sigcheck=bool(ctx.runtime_settings.agent_sigcheck))
    errs = list(chk.errors) + list(chk.sig_errors)
    if not chk.errors:
        errs += check_standard_part_calls(code)
    return errs


def _candidate_score(c: _Candidate) -> tuple[int, int, int, int, int]:
    """Score tuple for _pick_candidate — higher is better, lexicographic."""
    code = (c.resp.code or "").strip()
    verified_rank = 1 if c.verified else (-1 if c.verified is False else 0)
    return (
        verified_rank,  # sandbox PASS beats everything; proven FAIL sinks below unknown
        int(bool(code) and not c.contract_errors),  # parseable + contract-clean
        int(bool(code)),  # produced code at all
        -len(c.contract_errors),  # tiebreak: fewer contract errors
        -len(code),  # tiebreak: shorter code
    )


def _pick_candidate(candidates: list[_Candidate]) -> tuple[_Candidate, str]:
    """Pick the winning candidate and a short human reason for the emit line.

    Order: sandbox-verified PASS > parseable + contract-clean > has code >
    nothing; ties break on fewer contract errors then shorter code (max keeps
    the earlier index on a full tie — deterministic). A verified FAIL ranks
    BELOW 'never verified': provably broken code is worse than unproven."""
    winner = max(candidates, key=_candidate_score)
    code = (winner.resp.code or "").strip()
    if winner.verified:
        why = "verified in-session" if winner.verify_source == "session" else "verified"
    elif code and not winner.contract_errors:
        why = "contract-clean, unverified"
    elif code:
        why = f"{len(winner.contract_errors)} contract error(s)"
    else:
        why = "no code"
    return winner, why


async def _codegen_tournament(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    *,
    n: int,
    iteration: int,
    feedback: str | None,
    verify_fn: VerifyFn | None,
    render_fn: VerifyFn | None,
) -> CodeResponse:
    """Best-of-N codegen (AlphaCode-lite generate-then-filter).

    Runs N independent ``_codegen_session`` calls via asyncio.gather — each
    builds its OWN ToolSession and message list (never shared). This is NOT
    the vLLM ``n`` param: chat() parses only choices[0], and the parallel
    requests share the APC prompt prefix anyway, so extra cost is ~linear in
    completion tokens only.

    Budget: each candidate keeps the FULL per-session max_calls — halving it
    would strand staged builds mid-part; the worst case (n x max_calls) is
    bounded and every LLM call is already accounted inside chat().

    Scoring costs nothing extra where possible: a build session's own verify
    tool already ran the code through the sandbox, so its tool_log verdict is
    reused; only candidates whose final code was never executed pay one
    light ``verify_fn`` call. A candidate that raises is skipped — the
    tournament only fails when ALL candidates fail."""
    slots: list[dict[str, Any]] = [{} for _ in range(n)]

    async def _one(i: int) -> CodeResponse:
        return await _codegen_session(
            ctx,
            spec,
            plan,
            iteration=iteration,
            feedback=feedback,
            verify_fn=verify_fn,
            render_fn=render_fn,
            out=slots[i],
        )

    results = await asyncio.gather(*(_one(i) for i in range(n)), return_exceptions=True)

    candidates: list[_Candidate] = []
    first_error: BaseException | None = None
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            first_error = first_error or r
            log.warning("codegen.candidate_failed", candidate=i, error=str(r))
            continue
        session = slots[i].get("session")
        candidates.append(
            _Candidate(
                index=i,
                resp=r,
                session=session if isinstance(session, ToolSession) else None,
            )
        )
    if not candidates:
        # every candidate failed — surface the first failure exactly like a
        # single-attempt call would
        assert first_error is not None
        raise first_error

    for c in candidates:
        code = (c.resp.code or "").strip()
        c.verified = _in_session_verify(c.session, code)
        if c.verified is not None:
            c.verify_source = "session"
        elif verify_fn is not None and code:
            try:
                c.verified = bool((await verify_fn(code)).get("ok"))
                c.verify_source = "exec"
            except Exception as e:
                log.warning("codegen.candidate_verify_failed", candidate=c.index, error=str(e))
        c.contract_errors = _candidate_contract_errors(ctx, code)

    winner, why = _pick_candidate(candidates)
    summary = [
        {
            "i": c.index,
            "verified": c.verified,
            "verify_source": c.verify_source or None,
            "contract_errors": len(c.contract_errors),
            "code_len": len(c.resp.code or ""),
            "score": list(_candidate_score(c)),
        }
        for c in candidates
    ]
    log.info("codegen.best_of_n", n=n, iteration=iteration, winner=winner.index, candidates=summary)
    await ctx.emit(
        "codegen", iteration, f"best of {n} candidates (winner: #{winner.index + 1} {why})"
    )
    await ctx.trace.event(
        "codegen.best_of_n",
        step=STEP,
        iteration=iteration,
        n=n,
        winner=winner.index,
        reason=why,
        candidates=summary,
    )
    return winner.resp
