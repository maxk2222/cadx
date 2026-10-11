"""MESH_REFINE step — interactive mesh-transformation tool session.

Runs twice in the mesh loop: once after the deterministic baseline processing
("shape pass" — the agent may accept as-is via `done` with no ops) and again,
seeded with the critique issues, as the repair path instead of a blind
re-generation with a new seed.

Session kind "mesh" on the vision role (preview renders come back as images).
Falls back to `None` when tools are off / the vision model lost tool support —
the orchestrator then keeps the previous seed-rotation behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cadagent import storage
from cadagent.agent.context import (
    dfm_block,
    images_block,
    issues_block,
    renders_block,
    spec_block,
    system_prompt,
)
from cadagent.agent.mesh_tools import MeshWorkbench
from cadagent.agent.toolloop import run_session, tools_enabled
from cadagent.agent.toolspecs import ToolSession
from cadagent.agent.types import AgentCtx
from cadagent.cad.mesh_ops import MeshOp, MeshPostParams
from cadagent.llm.schemas import PartSpec
from cadagent.logging import get_logger

log = get_logger(__name__)

STEP = "meshrefine"


@dataclass
class RefineResult:
    ops: list[MeshOp] = field(default_factory=list)
    regenerate: bool = False
    subject_hint: str = ""
    changed: bool = False  # ops differ from the session's initial ops
    reasoning: str = ""


def _parts(
    ctx: AgentCtx,
    spec: PartSpec,
    *,
    issues: list[dict[str, Any]],
    contact_sheet: Path | None,
    analysis: dict[str, Any] | None,
    dfm: dict[str, Any] | None,
    initial_ops: list[MeshOp],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "A generative mesh model produced this part. The deterministic "
                "baseline (clean, orient, scale-to-spec) is already applied — "
                "your tools act on that result. Inspect it, transform it, "
                "preview it, then call done when the part matches the spec. "
                "Only call regenerate when the generated shape itself is wrong."
            ),
        }
    ]
    if issues:
        parts.append({"type": "text", "text": issues_block(issues)})
    if initial_ops:
        parts.append(
            {
                "type": "text",
                "text": "OPS ALREADY APPLIED (you may undo/reset or build on them):\n"
                + "\n".join(f"- {o.op}({o.args})" for o in initial_ops),
            }
        )
    parts.append({"type": "text", "text": spec_block(spec)})
    parts += renders_block(contact_sheet, analysis)
    dfm_txt = dfm_block(dfm)
    if dfm_txt:
        parts.append({"type": "text", "text": dfm_txt})
    parts.append({"type": "text", "text": "USER INPUT IMAGES (for reference):"})
    parts += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)
    return parts


async def run(
    ctx: AgentCtx,
    spec: PartSpec,
    mesh_rel: str,
    post: MeshPostParams,
    *,
    iteration: int,
    initial_ops: list[MeshOp] | None = None,
    issues: list[dict[str, Any]] | None = None,
    contact_sheet: Path | None = None,
    analysis: dict[str, Any] | None = None,
    dfm: dict[str, Any] | None = None,
) -> RefineResult | None:
    """Run the refine session. Returns None when unavailable — caller falls back."""
    s = ctx.runtime_settings
    if not getattr(s, "agent_mesh_tools", True) or not tools_enabled(ctx, "vision"):
        return None
    initial = list(initial_ops or [])

    wb = MeshWorkbench(
        ctx,
        raw_path=storage.resolve(mesh_rel),
        post=post,
        spec=spec,
        iteration=iteration,
        initial_ops=initial,
        preview_budget=getattr(s, "agent_mesh_preview_max", 4),
        max_ops=getattr(s, "agent_mesh_max_ops", 24),
    )
    if wb.work is None:
        log.warning("mesh.refine_no_workbench", error=wb.error)
        return None

    session = ToolSession(ctx, kind="mesh", workbench=wb)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt("mesh_refine_system.md")
            + "\n\n"
            + system_prompt("tools_protocol.md"),
        },
        {
            "role": "user",
            "content": _parts(
                ctx,
                spec,
                issues=issues or [],
                contact_sheet=contact_sheet,
                analysis=analysis,
                dfm=dfm,
                initial_ops=initial,
            ),
        },
    ]

    try:
        res = await run_session(
            ctx,
            "vision",
            messages,
            session=session,
            json_schema=None,
            max_calls=s.agent_tool_max_calls,
            step=STEP,
            iteration=iteration,
            temperature=0.0,
        )
    except Exception as e:
        log.warning("mesh.refine_session_failed", error=str(e))
        return None

    sub = res.submitted
    if sub is None:
        # model ended with a plain answer instead of done/regenerate — treat
        # ops applied so far as its result when they changed anything
        if wb.ops != initial:
            return RefineResult(
                ops=list(wb.ops),
                changed=True,
                reasoning="session ended without done — keeping applied ops",
            )
        return None
    if sub.get("regenerate"):
        return RefineResult(
            ops=list(wb.ops),
            regenerate=True,
            subject_hint=str(sub.get("subject_hint", "")),
            reasoning=str(sub.get("note", "")),
        )
    ops = [MeshOp.model_validate(o) for o in sub.get("ops", [])]
    return RefineResult(
        ops=ops,
        changed=ops != initial,
        reasoning=str(sub.get("reasoning", "")),
    )
