"""ORIENT step: choose the generated mesh's up/front axes.

Renders the 6 up-axis candidates (in the cad-executor, where offscreen
rendering is set up), shows them to the vision model together with the user's
reference images, and asks for a MeshOrientation. Falls back to a stability
heuristic when rendering or the LLM call fails.
"""

from __future__ import annotations

import contextlib
from typing import Any

from cadx import storage
from cadx.agent.context import images_block, system_prompt
from cadx.agent.thinking import thinking_body
from cadx.agent.types import AgentCtx
from cadx.llm.images import image_part
from cadx.llm.schemas import MeshOrientation
from cadx.logging import get_logger

log = get_logger(__name__)


def _heuristic(mesh_rel: str) -> dict[str, str]:
    """Stability-based guess when the LLM path is unavailable."""
    try:
        import numpy as np

        from cadx.cad.mesh_ops import default_front, load_mesh, suggest_up

        mesh = load_mesh(storage.resolve(mesh_rel))
        up = suggest_up(mesh)
        front = default_front(up, np.asarray(mesh.extents))
        return {"up": up, "front": front}
    except Exception:
        return {"up": "+z", "front": "+y"}


def _valid(o: MeshOrientation) -> bool:
    return o.up[-1] != o.front[-1]


async def run(ctx: AgentCtx, iteration: int, mesh_rel: str) -> dict[str, str]:
    """Returns {"up": ..., "front": ...} in generator-axis space."""
    await ctx.emit("orient", iteration, "choosing mesh orientation")

    # render candidates in the executor
    try:
        cand = await ctx.executor.process_mesh(
            {
                "job_id": ctx.job_id,
                "project_id": str(ctx.project_id),
                "version_id": str(ctx.version_id),
                "iteration": iteration,
                "mode": "candidates",
                "mesh_rel": mesh_rel,
                "settings": ctx.settings.model_dump(),
            }
        )
    except Exception as e:
        log.warning("mesh.orient_candidates_failed", error=str(e))
        cand = {}

    if not cand.get("ok"):
        choice = _heuristic(mesh_rel)
        await ctx.emit("orient", iteration, f"heuristic orientation {choice['up']}")
        ctx.mesh_info["orientation_source"] = "heuristic"
        return choice

    # ask the vision model
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The grid shows the SAME generated mesh rotated into 6 candidate "
                "orientations; the label above each tile says which mesh axis points UP (+Z). "
                "Using the user's reference images and description, pick the orientation "
                "where the part is upright and its front faces the viewer. "
                f"Part: {ctx.spec.name if ctx.spec else ctx.prompt[:80]}. "
                f"Description: {(ctx.spec.summary if ctx.spec else ctx.prompt)[:400]}. "
                "The candidate render stats: "
                + ", ".join(f"{k}={v}" for k, v in (cand.get("stats") or {}).items())
            ),
        }
    ]
    sheet = storage.resolve(cand["sheet"]) if cand.get("sheet") else None
    if sheet and sheet.exists():
        parts.append({"type": "text", "text": "ORIENTATION CANDIDATES:"})
        with contextlib.suppress(Exception):
            parts.append(image_part(sheet))
    parts.append({"type": "text", "text": "USER REFERENCE IMAGES:"})
    parts += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)

    try:
        resp = await ctx.llm.chat(
            "vision",
            [
                {"role": "system", "content": system_prompt("mesh_orient_system.md")},
                {"role": "user", "content": parts},
            ],
            json_schema=MeshOrientation,
            temperature=0.0,
            extra_body=thinking_body(ctx, "mesh_orient"),
        )
        await ctx.trace.record(
            step="orient",
            iteration=iteration,
            model=resp.model,
            messages=[{"role": "user", "content": "[orientation candidates + refs]"}],
            response_text=resp.text,
            parsed=resp.parsed.model_dump() if resp.parsed else None,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )
        if resp.parsed is not None:
            o = MeshOrientation.model_validate(resp.parsed)
            if _valid(o) and o.confidence >= 0.3:
                await ctx.emit("orient", iteration, f"up={o.up} front={o.front} ({o.reason[:60]})")
                return {"up": o.up, "front": o.front}
            log.warning("mesh.orient_rejected", orientation=o.model_dump())
    except Exception as e:
        log.warning("mesh.orient_llm_failed", error=str(e))

    choice = _heuristic(mesh_rel)
    await ctx.emit("orient", iteration, f"heuristic orientation {choice['up']}")
    ctx.mesh_info["orientation_source"] = "heuristic"
    return choice
