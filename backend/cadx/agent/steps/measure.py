"""MEASURE step — deterministic plan-view measurement of uploaded images.

Runs after INTAKE: for each image whose role is top/bottom/drawing (or an
unroled image whose silhouette aspect matches the spec), measure the outline
and hole positions in Frame mm. Results feed anchors (hole positions for CAD),
mesh orientation/clipping and the critique's position check.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cadx import storage
from cadx.agent.types import AgentCtx
from cadx.llm.schemas import PartSpec
from cadx.logging import get_logger
from cadx.vision.measure import measure_plan_view, render_overlay

log = get_logger(__name__)

STEP = "measure"
PLAN_ROLES = {"top", "bottom", "drawing"}
_DIA_RE = re.compile(
    r"(?:Ø|\bdia(?:meter)?\b|\bdurchmesser\b)\s*(\d+(?:[.,]\d+)?)\s*mm", re.IGNORECASE
)


def _reference_from_prompt(prompt: str) -> float | None:
    m = _DIA_RE.search(prompt or "")
    return float(m.group(1).replace(",", ".")) if m else None


def _candidate_role(asset: Any, roles: dict[str, str], spec: PartSpec) -> str | None:
    """Planned role for an asset, or 'top' when its silhouette aspect matches."""
    if (r := roles.get(asset.asset_id)) in PLAN_ROLES:
        return r
    if r is not None:
        return None
    # fallback: unroled image with alpha whose aspect fits length/width ±25%
    try:
        import numpy as np
        from PIL import Image

        a = np.asarray(Image.open(asset.path).convert("RGBA"))[..., 3]
        if float((a < 128).mean()) <= 0.02:
            return None
        cols = np.flatnonzero((a >= 128).any(axis=0))
        rows = np.flatnonzero((a >= 128).any(axis=1))
        aspect = (cols[-1] - cols[0] + 1) / max(rows[-1] - rows[0] + 1, 1)
        stated = spec.length_x.value / max(spec.width_y.value, 1e-9)
        # orientation-agnostic: a portrait top view of a 119x38 part has
        # aspect ~0.32 while stated length/width is ~3.1 — compare long/short
        fit = max(aspect, 1 / aspect) / max(stated, 1 / stated)
        if 0.8 <= fit <= 1.25:
            return "top"
    except Exception:
        return None
    return None


async def run(ctx: AgentCtx, spec: PartSpec) -> dict[str, dict[str, Any]]:
    """Measure plan-view images. Returns {asset_id: ImageMeasurement dict}."""
    images = [a for a in ctx.assets if a.kind == "image"]
    if not images or spec.length_x.value <= 0:
        return {}
    roles = {r.asset_id: r.role for r in spec.image_roles}
    ref_mm = _reference_from_prompt(ctx.prompt)
    out_dir = storage.ensure(storage.version_dir(ctx.project_id, ctx.version_id) / "measure")
    out: dict[str, dict[str, Any]] = {}
    for asset in images:
        role = _candidate_role(asset, roles, spec)
        if role is None:
            continue
        try:
            m = measure_plan_view(
                Path(asset.path),
                length_mm=spec.length_x.value,
                width_mm=spec.width_y.value or None,
                role=role,
                reference_mm=ref_mm,
            )
            m.asset_id = asset.asset_id
            (out_dir / f"{asset.asset_id}.json").write_text(m.model_dump_json(indent=2))
            try:
                render_overlay(Path(asset.path), m, out_dir / f"{asset.asset_id}_overlay.png")
            except Exception as e:
                log.warning("measure.overlay_failed", error=str(e)[:120])
            out[asset.asset_id] = m.model_dump()
            await ctx.emit(
                STEP,
                0,
                f"measured {role} view: {m.bbox_mm[0]:.1f} x {m.bbox_mm[1]:.1f} mm, "
                f"{len(m.holes)} holes",
            )
        except Exception as e:
            await ctx.emit(STEP, 0, f"measurement skipped for {asset.filename}: {e}")
            log.warning("measure.failed", asset=asset.filename, error=str(e)[:200])
    if not out:
        await ctx.emit(STEP, 0, "no measurable plan view found")
    return out
