"""Visual prompt building blocks for synthesised reference images.

Feature names in a PartSpec are code jargon (`mounting_tab_-x`) — useless
as conditioning for a text-to-image model. These helpers turn the spec's
feature list into short, deterministic, natural-language clauses plus the
pose/negative tail pieces of the hero-image prompt.
"""

from __future__ import annotations

from typing import Any

from cadx.llm.schemas import PartSpec

MAX_FEATURES = 8
INTENT_MAX_CHARS = 420
DESC_MAX_CHARS = 60
POSITION_MAX_CHARS = 45
# total character budget for the feature clause list — t2i models truncate
# prompts (Z-Image at 512 tokens), so an overlong feature dump silently loses
# the tail features (observed: holes dropped → verify retried uselessly)
FEATURES_MAX_CHARS = 420

_HOLE_TYPES = ("hole", "through_hole")
_AXIS_WORDS = {"z": "vertical", "x": "horizontal (X-axis)", "y": "horizontal (Y-axis)"}

NEGATIVE_PROMPT = "text, watermark, extra objects, human hands, multiple parts, cropped object"


def _clip(text: str, limit: int) -> str:
    """Clamp to ~`limit` chars, cut at the last sentence boundary so the
    result never ends mid-sentence (falls back to a word boundary)."""
    t = " ".join(str(text or "").split())
    if len(t) <= limit:
        return t
    cut = max(
        t.rfind(". ", 0, limit),
        t.rfind("; ", 0, limit),
        t.rfind("! ", 0, limit),
        t.rfind("? ", 0, limit),
    )
    if cut > 0:
        return t[:cut].rstrip(" ,;.")
    word = t.rfind(" ", 0, limit)
    return (t[:word] if word > 0 else t[:limit]).rstrip(" ,;.")


def _label(f: Any) -> str:
    """Human-readable feature label: `description` is already natural
    language, `name` is the code-jargon fallback with underscores stripped."""
    desc = str(getattr(f, "description", "") or "").strip()
    if desc:
        return desc
    return str(getattr(f, "name", "") or getattr(f, "type", "") or "").replace("_", " ").strip()


def visual_features(spec: PartSpec | None) -> str:
    """Comma-separated visual clauses for the t2i prompt, max 8 features.

    Repeated holes with a known axis get the explicit
    "{count} round {axis-word} through-holes" template — a photo model
    understands that better than a bare label. `position` text is appended
    as a clipped " at {position}" tail.
    """
    if spec is None:
        return ""
    clauses: list[str] = []
    for f in list(getattr(spec, "features", None) or [])[:MAX_FEATURES]:
        ftype = str(getattr(f, "type", "") or "").lower()
        axis = str(getattr(f, "axis", "") or "").lower()
        count = int(getattr(f, "count", 1) or 1)
        if ftype in _HOLE_TYPES and count > 1 and axis in _AXIS_WORDS:
            clause = f"{count} round {_AXIS_WORDS[axis]} through-holes"
            desc = str(getattr(f, "description", "") or "").strip()
            if desc:
                clause += f" — {_clip(desc, DESC_MAX_CHARS)}"
        else:
            label = _clip(_label(f), DESC_MAX_CHARS)
            if not label:
                continue
            clause = f"{count}x {label}" if count > 1 else label
        pos = _clip(getattr(f, "position", "") or "", POSITION_MAX_CHARS)
        if pos:
            clause += f" at {pos}"
        if sum(len(c) + 2 for c in clauses) + len(clause) > FEATURES_MAX_CHARS:
            break
        clauses.append(clause)
    return ", ".join(clauses)


def proportion_hint(spec: PartSpec | None) -> str:
    """Qualitative silhouette descriptor derived from the spec dims — t2i
    models do not translate '120 x 40 x 30 mm' into aspect ratio (observed:
    a flat 120x40x30 handle rendered as a tall U-arch ~4x too high). Giving
    the ratio plus a plain-English qualifier fixes the proportion the mesh
    later inherits."""
    if spec is None:
        return ""
    try:
        lx = float(spec.length_x.value or 0)
        wy = float(spec.width_y.value or 0)
        hz = float(spec.height_z.value or 0)
    except (AttributeError, TypeError, ValueError):
        return ""
    if lx <= 0 or wy <= 0 or hz <= 0:
        return ""
    unit = min(lx, wy, hz)
    ratio = f"{lx / unit:.4g} : {wy / unit:.4g} : {hz / unit:.4g}"
    footprint = max(lx, wy)
    qualifiers: list[str] = []
    if max(lx, wy, hz) / unit < 1.35:
        qualifiers.append("roughly equal dimensions in all directions")
    if hz / footprint < 0.35:
        qualifiers.append(
            "very low and flat — the height is only a small fraction of the footprint"
        )
    elif hz / footprint < 0.6:
        qualifiers.append("low in height relative to the footprint")
    elif hz / footprint > 2.0:
        qualifiers.append("tall and slender — the height dominates the footprint")
    if lx / wy > 2.5:
        qualifiers.append("strongly elongated in one direction")
    desc = ", ".join(qualifiers) if qualifiers else "moderate proportions"
    return f"roughly {ratio} length-to-width-to-height — {desc}"


def rest_pose_hint(spec: PartSpec | None) -> str:
    """Pose clause for the hero prompt — parts rest on z=0 per the
    coordinate contract, so the photo should show them lying flat."""
    return "lying flat on the surface"


def hero_prompt_parts(spec: PartSpec | None, prompt: str, subject: str) -> dict[str, str]:
    """Format values for prompts/image_hero.md — every key the template
    references, in one place so the caller stays a plain .format()."""
    return {
        "subject": subject,
        "intent": _clip(prompt, INTENT_MAX_CHARS),
        "features_visual": visual_features(spec),
        "proportions": proportion_hint(spec),
        "pose": rest_pose_hint(spec),
        "negative": NEGATIVE_PROMPT,
    }
