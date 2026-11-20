"""Doc-mission tools — PDF preparation into ``state.document``.

Owns ONLY the DocumentIndex: rendering, page text, figure crops, language,
page classification, detected part names. Ported tool-for-tool from
KIARA's DocAnalyst surface (v4 tools/doc_tools.py).
"""

from __future__ import annotations

import re as _re_mod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiara.guide import pdfdoc
from kiara.guide.state import DocumentIndex, PageSummary
from kiara.guide.tools import _arr, _fn, _i, _s
from kiara.logging import get_logger

if TYPE_CHECKING:
    from kiara.guide.session import GuideToolSession

log = get_logger(__name__)

SCHEMAS: dict[str, dict[str, Any]] = {
    "prepare_pdf": _fn(
        "prepare_pdf",
        "Render the instruction PDF to page images, extract per-page text "
        "and figure crops, and build the document index. MUST be called "
        "first — every other doc tool needs it.",
        {
            "dpi": _i("render resolution (default 200)"),
            "max_pages": _i("cap pages processed (default: all, hard cap from settings)"),
        },
        [],
    ),
    "detect_language": _fn(
        "detect_language",
        "Set the document language (BCP-47 code like 'de'/'en') used for display_text generation.",
        {"language": _s("language code")},
        ["language"],
    ),
    "classify_pages": _fn(
        "classify_pages",
        "Assign a classification to each page so later missions can find "
        "steps/parts-lists quickly. Batch all pages in ONE call.",
        {
            "labels": _arr(
                {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "classification": {
                            "type": "string",
                            "enum": ["cover", "toc", "parts_list", "steps", "appendix", "unknown"],
                        },
                    },
                    "required": ["page", "classification"],
                },
                "one {page, classification} per page",
            )
        },
        ["labels"],
    ),
    "set_detected_part_names": _fn(
        "set_detected_part_names",
        "Set the canonical part names referenced by the manual (from the "
        "parts list page and step texts). These names drive CAD matching. "
        "Include the Stückliste quantity when the parts list gives one — "
        "as '2x Rohr DN22', 'Rohr DN22 (2x)' or {name, qty}.",
        {"names": _arr(
            _s("part name as written in the manual, optionally with quantity"),
            "part names",
        )},
        ["names"],
    ),
}


def _doc_required(s: GuideToolSession) -> dict[str, Any] | None:
    if s.state.document is None:
        return {"ok": False, "error": "call prepare_pdf first"}
    return None


async def prepare_pdf(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    pdf_assets = [x for x in s.guide_assets if x.get("kind") == "pdf"]
    if not pdf_assets:
        return {"ok": False, "error": "no PDF asset attached to this guide run"}
    pdf_path = Path(str(pdf_assets[0]["path"]))
    if not pdf_path.exists():  # noqa: ASYNC240
        return {"ok": False, "error": f"pdf file missing: {pdf_path}"}

    rt = s.ctx.runtime_settings
    dpi = int(a.get("dpi") or getattr(rt, "guide_pdf_dpi", 200))
    cap = int(getattr(rt, "guide_max_pages", 80))
    max_pages = int(a.get("max_pages") or cap)
    max_pages = min(max_pages, cap) if cap > 0 else max_pages

    pages_dir = s.state_dir / "pages"
    figs_dir = s.state_dir / "figures"
    try:
        pages = pdfdoc.render_pdf(pdf_path, pages_dir, dpi=dpi, max_pages=max_pages)
    except Exception as e:
        return {"ok": False, "error": f"pdf render failed: {e}"}
    texts = pdfdoc.extract_page_texts(pdf_path, max_pages=max_pages)
    figs = pdfdoc.extract_figures(pdf_path, figs_dir, dpi=dpi, max_pages=max_pages)
    figs_by_page: dict[int, list[dict[str, Any]]] = {}
    for f in figs:
        figs_by_page.setdefault(int(f["page"]), []).append(f)

    doc = DocumentIndex(
        pdf_path=str(pdf_path),
        pdf_asset_id=str(pdf_assets[0]["id"]),
        dpi=dpi,
        page_count=len(pages),
        pages=[
            PageSummary(
                page=pno,
                image_path=str(p),
                text=texts.get(pno, ""),
                figures=figs_by_page.get(pno, []),
            )
            for pno, p, _w, _h in pages
        ],
    )
    truncated = len(pages) == max_pages and doc.page_count == max_pages
    s.state.document = doc
    return {
        "ok": True,
        "page_count": doc.page_count,
        "figures": len(figs),
        "pages_truncated": truncated,
        "note": (
            "page text is embedded where present; use get_page_image/"
            "get_figure_image to read pages visually"
        ),
    }


async def detect_language(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    if err := _doc_required(s):
        return err
    lang = str(a.get("language", "")).strip().lower()[:10] or "de"
    assert s.state.document is not None
    if s.state.language_locked:
        # the user pinned the output language at spawn — autodetect may
        # only record the manual's own language, never the output one
        s.state.document.language = s.state.language
        return {"ok": True, "language": s.state.language, "locked": True}
    s.state.document.language = lang
    s.state.language = lang
    return {"ok": True, "language": lang}


async def classify_pages(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    if err := _doc_required(s):
        return err
    assert s.state.document is not None
    doc = s.state.document
    valid = {"cover", "toc", "parts_list", "steps", "appendix", "unknown"}
    applied = 0
    for lbl in a.get("labels") or []:
        try:
            page = int(lbl.get("page", 0))
            cls = str(lbl.get("classification", "unknown"))
        except Exception:
            continue
        if cls not in valid or page < 1 or page > len(doc.pages):
            continue
        doc.pages[page - 1].classification = cls
        applied += 1
    return {
        "ok": True,
        "classified": applied,
        "pages": [{"page": p.page, "classification": p.classification} for p in doc.pages],
    }


_PREMOUNT_WORDS = _re_mod.compile(
    r"vormontiert|premount|pre-mount|preinstall|pre-install|bereits montiert|"
    r"already mounted|factory.?fit", _re_mod.I
)


def _split_name_qty(item: Any) -> tuple[str, int | None, int]:
    """Accept '2x Rohr DN22', 'Rohr DN22 (2x)', 'Rohr DN22 x2',
    'Rohr DN22 — 2 Stück', 'Adapter 2 + 2 (vormontiert)' or {name, qty} —
    manuals write the Menge into the line itself; keep the canonical
    name clean of the count. Returns (name, total_qty, premounted_qty):
    split notation sums kit + device-mounted copies and reports the
    mounted side separately."""
    import re as _re

    if isinstance(item, dict):
        name = str(item.get("name", "")).strip()
        try:
            qty = int(item.get("qty") or item.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0
        return name, (qty if qty > 0 else None), 0
    raw = str(item).strip()
    # "2 + 2 (vormontiert)" split: kit copies + copies already on the device
    m = _re.match(
        r"^(.+?)[\s\-:,;]*(\d+)\s*\+\s*(\d+)\s*[\(\[]?\s*"
        r"(vormontiert|premount|pre-mount|preinstall|pre-install|"
        r"bereits montiert|already mounted|factory.?fit)\w*[\)\]]?\s*$",
        raw,
        _re.I,
    )
    if m:
        name, kit_n, pm_n = m.group(1), int(m.group(2)), int(m.group(3))
        return name.strip(" ,-:;"), kit_n + pm_n, pm_n
    for pat, qty_first in (
        (r"^(\d+)\s*[x\u00d7]\s+(.+)$", True),  # "2x Rohr DN22"
        (r"^(.+?)\s*[\(\[]\s*(\d+)\s*(?:x|\u00d7)\s*[\)\]]\s*$", False),  # "Rohr (2x)"
        (r"^(.+?)\s*(?:x|\u00d7)\s*(\d+)\s*$", False),  # "Rohr x2"
        (r"^(.+?)[\-\u2013\u2014:,]+\s*(\d+)\s*(?:St\u00fcck|Stk\.?|pcs?\.?|x)\s*$", False),  # "Rohr — 2 Stück"
    ):
        m = _re.match(pat, raw, _re.I)
        if m:
            name, qty = (m.group(2), m.group(1)) if qty_first else (m.group(1), m.group(2))
            return name.strip(), int(qty), 0
    # "N vormontiert" — the whole count sits on the device already
    if _PREMOUNT_WORDS.search(raw):
        m2 = _re.match(r"^(.+?)[\s\-:]*(\d+)\s*[\(\[]?", raw, _re.I)
        if m2:
            name2 = _PREMOUNT_WORDS.sub("", m2.group(1)).strip(" ,()[]")
            return name2, int(m2.group(2)), int(m2.group(2))
    return raw, None, 0


async def set_detected_part_names(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    if err := _doc_required(s):
        return err
    from kiara.guide.assembly_tools import _norm2

    names: list[str] = []
    qtys: dict[str, int] = {}
    pm_qtys: dict[str, int] = {}
    for item in a.get("names") or []:
        name, qty, pm_qty = _split_name_qty(item)
        if not name:
            continue
        names.append(name)
        if qty:
            qtys[_norm2(name)] = qty
        if pm_qty:
            pm_qtys[_norm2(name)] = pm_qty
    if not names:
        return {"ok": False, "error": "names list is empty"}
    assert s.state.document is not None
    s.state.document.detected_part_names = names[:60]
    s.state.document.detected_part_qty.update(qtys)
    s.state.document.detected_premount_qty.update(pm_qtys)
    return {
        "ok": True,
        "detected_part_names": s.state.document.detected_part_names,
        "detected_part_qty": qtys or None,
        "detected_premount_qty": pm_qtys or None,
    }


HANDLERS = {
    "prepare_pdf": prepare_pdf,
    "detect_language": detect_language,
    "classify_pages": classify_pages,
    "set_detected_part_names": set_detected_part_names,
}
