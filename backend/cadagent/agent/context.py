"""Message builders for agent LLM calls.

Convention: all static reference docs go into the system message; all dynamic
content into a single user message built from blocks.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from cadagent.agent.types import InputAsset
from cadagent.config import get_settings
from cadagent.db.models import ProjectSettings
from cadagent.llm.images import image_part
from cadagent.llm.schemas import PartSpec

PROMPTS_DIR = Path(__file__).parent / "prompts"


def read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def system_prompt(name: str) -> str:
    return read_prompt(name)


def codegen_system() -> str:
    return "\n\n".join(
        read_prompt(n)
        for n in (
            "codegen_system.md",
            "code_contract.md",
            "build123d_reference.md",
            "bd_warehouse_reference.md",
            "dfm_rules.md",
        )
    )


def user_block(prompt: str, settings: ProjectSettings, mode: str) -> str:
    lines = [
        f"USER REQUEST: {prompt}",
        "",
        f"process: {settings.process} | material: {settings.material or 'unspecified'}",
        f"units: {settings.units} | include_fasteners: {settings.include_fasteners}",
        f"thread_compensation_mm: {settings.thread_compensation_mm}",
        f"build_volume_mm: {settings.build_volume}",
    ]
    if mode == "edit":
        lines.append("MODE: edit of an existing version (previous spec/code/render provided)")
    return "\n".join(lines)


def images_block(assets: list[InputAsset]) -> list[dict[str, Any]]:
    """Content parts for input images (capped at AGENT_MAX_INPUT_IMAGES)."""
    max_n = get_settings().agent_max_input_images
    parts: list[dict[str, Any]] = []
    n = 0
    for a in assets:
        if a.kind != "image":
            continue
        n += 1
        if n > max_n:
            parts.append(
                {"type": "text", "text": f"(further {len(assets) - max_n} images omitted)"}
            )
            break
        caption = a.caption or "no caption"
        parts.append({"type": "text", "text": f"Image {n} (asset {a.asset_id}): {caption}"})
        try:
            parts.append(image_part(a.path))
        except Exception:
            parts.append({"type": "text", "text": f"[image {a.asset_id} failed to load]"})
    return parts


def references_block(assets: list[InputAsset]) -> str:
    refs = [a for a in assets if a.kind in ("reference_step", "reference_stl")]
    docs = [a for a in assets if a.kind == "reference_dxf"]
    lines: list[str] = []
    if refs:
        lines.append("REFERENCE BODIES available via load_reference(asset_id):")
        for a in refs:
            lines.append(f"- asset_id={a.asset_id}  file={a.filename}  caption={a.caption or '-'}")
    if docs:
        lines.append("Attached DXF drawings (2D references — match their dimensions):")
        for a in docs:
            lines.append(f"- file={a.filename}  caption={a.caption or '-'}")
    return "\n".join(lines)


def spec_block(spec: PartSpec) -> str:
    s: str = spec.model_dump_json(indent=2)
    return "PART SPEC (JSON):\n" + s


def plan_block(plan: Any) -> str:
    s: str = plan.model_dump_json(indent=2)
    return "MODELING PLAN (JSON):\n" + s


def renders_block(
    contact_sheet: Path | None, analysis: dict[str, Any] | None
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if analysis:
        size = analysis.get("bbox_size", "?")
        parts.append(
            {
                "type": "text",
                "text": (
                    f"GENERATED MODEL — measured bbox {size}, "
                    f"volume {analysis.get('volume')} mm^3, "
                    f"solids {analysis.get('solid_count')}, "
                    f"valid {analysis.get('is_valid')}, watertight {analysis.get('watertight')}, "
                    f"warnings {analysis.get('warnings')}"
                ),
            }
        )
    if contact_sheet is not None and contact_sheet.exists():
        parts.append({"type": "text", "text": "CONTACT SHEET of the generated model:"})
        try:
            parts.append(image_part(contact_sheet))
        except Exception:
            parts.append({"type": "text", "text": "[contact sheet failed to load]"})
    else:
        parts.append({"type": "text", "text": "(no renders available — judge numerically)"})
    return parts


def parent_block(
    parent_code: str | None, parent_spec: PartSpec | None, sheet: Path | None
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if parent_spec:
        parts.append(
            {
                "type": "text",
                "text": "PREVIOUS VERSION spec:\n" + parent_spec.model_dump_json(indent=2),
            }
        )
    if parent_code:
        parts.append(
            {"type": "text", "text": "PREVIOUS VERSION code:\n```python\n" + parent_code + "\n```"}
        )
    if sheet is not None and sheet.exists():
        parts.append({"type": "text", "text": "PREVIOUS VERSION contact sheet:"})
        with contextlib.suppress(Exception):
            parts.append(image_part(sheet))
    return parts


def error_block(error: str | None, traceback: str | None, stdout: str | None) -> str:
    lines = ["EXECUTION FAILED."]
    if error:
        lines.append("error:\n" + error[-3000:])
    if traceback:
        lines.append("traceback (trimmed):\n" + traceback[-3000:])
    if stdout:
        lines.append("stdout tail:\n" + stdout[-1000:])
    return "\n\n".join(lines)


def dfm_block(dfm: dict[str, Any] | None) -> str:
    if not dfm or not dfm.get("warnings"):
        return ""
    lines = ["DFM WARNINGS:"]
    for w in dfm["warnings"][:20]:
        lines.append(f"- [{w['severity']}] {w['code']}: {w['message']}")
    return "\n".join(lines)
