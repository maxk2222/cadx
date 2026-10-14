"""Message builders for agent LLM calls.

Convention: all static reference docs go into the system message; all dynamic
content into a single user message built from blocks.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

from cadx.agent.types import IMAGE_INPUT_KINDS, InputAsset
from cadx.db.models import ProjectSettings
from cadx.llm.images import image_part
from cadx.llm.schemas import PartSpec

PROMPTS_DIR = Path(__file__).parent / "prompts"


def read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def system_prompt(name: str) -> str:
    return read_prompt(name)


# helpers that are always relevant regardless of the spec's feature list
_CADLIB_ALWAYS = {"through_hole", "profile_body"}


def _cadlib_sections(spec: PartSpec | None) -> str:
    """cadlib_reference.md filtered to the helpers the spec actually needs.

    Sections are delimited by `### <helper_name>`; a section is kept when its
    helper name shares a word with a spec feature type/name. With no spec (or
    no features) the full reference is returned — codegen from scratch gets
    the complete toolbox."""
    text = read_prompt("cadlib_reference.md")
    if spec is None or not spec.features:
        return text
    words: set[str] = set()
    for f in spec.features:
        for w in re.split(r"[^a-z]+", f"{f.type} {f.name}".lower()):
            if w:
                words.add(w)
    # shared vocabulary: feature wording -> helper names
    for alias, helpers in {
        "hole": {"through_hole", "hole_grid"},
        "bore": {"through_hole"},
        "insert": {"insert_pocket", "hole_grid"},
        "screw": {"hole_grid", "boss"},
        "boss": {"boss"},
        "rib": {"rib"},
        "gusset": {"rib"},
        "ear": {"mounting_ear"},
        "slot": {"keyhole_slot", "t_slot", "dovetail_slot"},
        "keyhole": {"keyhole_slot"},
        "bearing": {"bearing_seat"},
        "hinge": {"living_hinge"},
        "snap": {"snap_fit_cantilever"},
        "knurl": {"knurl_ring"},
        "text": {"text_label"},
        "label": {"text_label"},
        "bracket": {"l_bracket"},
        "box": {"filleted_box"},
        "outline": {"profile_body", "closed_polyline"},
        "profile": {"profile_body", "closed_polyline"},
    }.items():
        if alias in words:
            words.update(helpers)
    keep = words | _CADLIB_ALWAYS

    lines = text.split("\n")
    out: list[str] = []
    cur_ok = True
    section: list[str] = []

    def _flush() -> None:
        if section and cur_ok:
            out.extend(section)

    for line in lines:
        if line.startswith("### "):
            _flush()
            name = line[4:].strip().split("(")[0].strip()
            cur_ok = name in keep
            section = [line]
        else:
            section.append(line)
    _flush()
    return "\n".join(out)


def _reference_docs(
    *, plan: Any | None = None, spec: PartSpec | None = None, fresh: bool = True
) -> str:
    """Staged reference docs: build123d always; bd_warehouse only when the
    plan/spec references standard parts; cadlib filtered to needed helpers;
    full examples only for fresh codegen (repairs get the terse contract)."""
    parts = [read_prompt("build123d_reference.md")]
    needs_bd = bool(getattr(plan, "standard_parts", None)) or any(
        getattr(f, "standard", "") for f in (spec.features if spec else [])
    )
    if needs_bd:
        parts.append(read_prompt("bd_warehouse_reference.md"))
    parts.append(_cadlib_sections(spec))
    if fresh:
        parts.append(read_prompt("code_examples.md"))
    parts.append(read_prompt("dfm_rules.md"))
    return "\n\n".join(p for p in parts if p)


def codegen_system(
    *, plan: Any | None = None, spec: PartSpec | None = None, fresh: bool = True
) -> str:
    return "\n\n".join(
        [
            read_prompt("codegen_system.md"),
            read_prompt("code_contract.md"),
            read_prompt("code_contract_responses.md"),
            _reference_docs(plan=plan, spec=spec, fresh=fresh),
        ]
    )


def codegen_tools_system(
    *, plan: Any | None = None, spec: PartSpec | None = None, fresh: bool = True
) -> str:
    """Codegen system prompt for tool sessions — submit_code replaces the
    CodeResponse channel, the file contract stays identical."""
    return "\n\n".join(
        [
            read_prompt("codegen_tools.md"),
            read_prompt("tools_protocol.md"),
            read_prompt("code_contract.md"),
            read_prompt("code_contract_responses_tools.md"),
            _reference_docs(plan=plan, spec=spec, fresh=fresh),
        ]
    )


_PROCESS_LABELS = {
    "fdm": "FDM 3D printing",
    "sla": "SLA resin printing",
    "cnc3": "CNC 3-axis machining",
    "cnc2d": "CNC 2D cutting",
}


def user_block(prompt: str, settings: ProjectSettings, mode: str) -> str:
    proc = _PROCESS_LABELS.get(settings.process, settings.process)
    lines = [
        f"USER REQUEST: {prompt}",
        "",
        f"process: {settings.process} ({proc}) | material: {settings.material or 'unspecified'}",
        f"units: {settings.units} | include_fasteners: {settings.include_fasteners}",
        f"thread_compensation_mm: {settings.thread_compensation_mm}",
        f"build_volume_mm: {settings.build_volume}",
    ]
    reqs: list[str] = []
    if settings.fem:
        req = "- run FEM structural analysis"
        req += (
            f" ({settings.fem_note})"
            if settings.fem_note
            else " — infer reasonable load cases and fixtures"
        )
        reqs.append(req)
    if settings.topopt:
        vf = (
            f"target volume fraction {settings.topopt_volfrac:.2f}"
            if settings.topopt_volfrac
            else "a sensible volume fraction"
        )
        reqs.append(f"- apply topology optimization ({vf})")
    if reqs:
        lines += [
            "",
            "USER-SELECTED REQUIREMENTS — honor these; set spec.process, spec.material,",
            "spec.load_cases and spec.optimize accordingly:",
            *reqs,
        ]
    if mode == "edit":
        lines.append("MODE: edit of an existing version (previous spec/code/render provided)")
    return "\n".join(lines)


def images_block(assets: list[InputAsset], max_n: int) -> list[dict[str, Any]]:
    """Content parts for input images (capped by the configured image budget)."""
    parts: list[dict[str, Any]] = []
    n = 0
    for a in assets:
        if a.kind not in IMAGE_INPUT_KINDS:
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


def library_block(templates: list[dict[str, Any]]) -> str:
    """Prompt block for reusable code templates from the user library.

    File-based library parts already appear in references_block (they are
    injected as reference assets); this block only covers `template` parts
    whose parametric code the model may adapt.
    """
    if not templates:
        return ""
    lines = [
        "LIBRARY TEMPLATES — reviewed parametric code you may adapt (rename "
        "PARAMS keys only if needed; do not copy verbatim if dims differ):",
    ]
    for t in templates:
        code = (t.get("code") or "").strip()
        if len(code) > 4000:
            code = code[:4000] + "\n# ... (truncated)"
        lines.append(f"\n### {t['name']}" + (f" — {t['caption']}" if t.get("caption") else ""))
        lines.append(f"```python\n{code}\n```")
    return "\n".join(lines)


def request_block(prompt: str) -> str:
    """Verbatim user request — authoritative when it conflicts with the spec."""
    p = (prompt or "").strip()
    if not p:
        return ""
    return (
        "USER REQUEST (verbatim — every feature and dimension named here MUST be "
        "implemented; if the PART SPEC below misses something, the request wins):\n" + p
    )


def spec_block(spec: PartSpec) -> str:
    s: str = spec.model_dump_json(indent=2)
    return "PART SPEC (JSON):\n" + s


def catalog_block(plan: Any | None, spec: PartSpec | None) -> str:
    """Bounded catalog excerpt for the standard parts named in plan + spec.

    Class names come from ``plan.standard_parts``; additionally any class name
    or norm type (e.g. "iso10642") mentioned in a feature's ``standard`` string
    pulls that class into the excerpt.
    """
    import re

    from cadx.cad import catalog

    names: list[str] = list(getattr(plan, "standard_parts", None) or [])
    reg = catalog.registry()
    for f in (spec.features if spec else []) or []:
        std = (getattr(f, "standard", "") or "").lower()
        if not std:
            continue
        norm = re.sub(r"[^a-z0-9]", "", std)
        for cls_name in reg:
            if cls_name.lower() in std:
                names.append(cls_name)
                continue
            entry = catalog.find_entry(cls_name)
            for t in (entry or {}).get("types", {}):
                if t != "default" and re.sub(r"[^a-z0-9]", "", t.lower()) in norm:
                    names.append(cls_name)
                    break
    try:
        return catalog.catalog_excerpt(names)
    except Exception:
        return ""


def catalog_listing_block() -> str:
    """Compact list of all catalog class names — lets the planner pick real ones."""
    try:
        from cadx.cad import catalog

        return catalog.class_listing()
    except Exception:
        return ""


def plan_block(plan: Any) -> str:
    s: str = plan.model_dump_json(indent=2)
    return "MODELING PLAN (JSON):\n" + s


def numbered_code_block(code: str, label: str = "CURRENT CODE") -> str:
    """Line-numbered code view — makes ranges addressable for tool sessions."""
    from cadx.cad.codeedits import line_numbered

    return f"{label} (line-numbered — address edits by line range):\n```python\n{line_numbered(code)}\n```"


def renders_block(
    contact_sheet: Path | None, analysis: dict[str, Any] | None
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if analysis:
        size = analysis.get("bbox_size", "?")
        # watertight_after_repair is the export-quality signal — the raw
        # pre-repair flag must not tank the score when the shipped file is clean
        wt = analysis.get("watertight_after_repair", analysis.get("watertight"))
        parts.append(
            {
                "type": "text",
                "text": (
                    f"GENERATED MODEL — measured bbox {size}, "
                    f"volume {analysis.get('volume')} mm^3, "
                    f"solids {analysis.get('solid_count')}, "
                    f"valid {analysis.get('is_valid')}, watertight {wt}, "
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


def dfm_block(dfm: dict[str, Any] | None) -> str:
    if not dfm or not dfm.get("warnings"):
        return ""
    lines = ["DFM WARNINGS:"]
    for w in dfm["warnings"][:20]:
        lines.append(f"- [{w['severity']}] {w['code']}: {w['message']}")
    return "\n".join(lines)


def issues_block(issues: list[dict[str, Any]]) -> str:
    """Render the structured issue list the model must fix — ALL of them."""
    if not issues:
        return ""
    lines = ["ISSUES TO FIX — address every one, change nothing else:"]
    for i, it in enumerate(issues, start=1):
        src = it.get("source", "")
        sev = it.get("severity", "")
        head = f"[{sev}] " if sev else ""
        where = f" @ {it['where']}" if it.get("where") else ""
        hint = f" -> {it['fix_hint']}" if it.get("fix_hint") else ""
        lines.append(f"{i}. {src}:{head}{it.get('what', '')}{where}{hint}")
    return "\n".join(lines)


def measured_features_block(features: dict[str, Any] | None) -> str:
    """Compact measured-geometry summary (ground truth) for improve/critique."""
    if not features:
        return ""
    lines = ["MEASURED GEOMETRY (ground truth from the solid):"]
    for g in features.get("hole_groups") or []:
        lines.append(
            f"- {g['count']}x hole Ø{g['diameter']}mm, axis {g['axis']}, {g['depth_class']}"
        )
    lines.append(f"- total holes: {features.get('total_holes', 0)}")
    if features.get("countersinks"):
        lines.append(f"- countersink cones: {features['countersinks']}")
    if features.get("counterbores"):
        lines.append(f"- stepped counterbores: {features['counterbores']}")
    if features.get("chamfer_faces"):
        lines.append(f"- chamfer-like 45° faces: {features['chamfer_faces']}")
    if features.get("rounds"):
        rs = sorted({round(float(r), 1) for r in features["rounds"]})
        lines.append(f"- fillet-like rounds (r mm): {rs}")
    if features.get("approximate"):
        lines.append("(approximate — faceted mesh source)")
    return "\n".join(lines)


def measurements_block(measurements: dict[str, Any] | None) -> str:
    """Image-measurement summary (plan-view silhouette + holes) for prompts."""
    if not measurements:
        return ""
    lines = [
        "MEASURED PLAN VIEW(S) (entries marked authoritative are ground truth; "
        "the rest are hints only):"
    ]
    for asset_id, m in measurements.items():
        bb = m.get("bbox_mm") or []
        holes = m.get("holes") or []
        hole_txt = " | ".join(
            f"#{i + 1} ({h['x']:.1f}, {h['y']:.1f}) Ø{h['dia']:.1f}" for i, h in enumerate(holes)
        )
        lines.append(
            f"- asset {asset_id} ({m.get('role', 'top')}): outline "
            f"{bb[0]:.1f} x {bb[1]:.1f} mm, confidence {m.get('confidence', 0):.2f}"
            f" | authoritative: {'yes' if m.get('authoritative') else 'no'}"
        )
        if hole_txt:
            lines.append(f"  holes: {hole_txt}")
        for w in (m.get("warnings") or [])[:3]:
            lines.append(f"  warning: {w}")
    return "\n".join(lines)


def anchors_block(anchors: dict[str, Any] | None) -> str:
    """Reconciled hole anchors (image+mesh) — index table for region kinds."""
    if not anchors:
        return ""
    holes = anchors.get("holes") or []
    if not holes:
        return ""
    lines = [
        "MEASURED ANCHORS (authoritative hole positions — use `hole` regions "
        "with these indexes, never retype coordinates):"
    ]
    for h in holes:
        name = f" '{h['name']}'" if h.get("name") else ""
        lines.append(
            f"#{h['index']}{name} ({h['x']:.1f}, {h['y']:.1f}) Ø{h['dia']:.1f} [{h['source']}]"
        )
    if anchors.get("mirror_x"):
        lines.append("note: the mesh was mirrored to match the top view")
    return "\n".join(lines)
