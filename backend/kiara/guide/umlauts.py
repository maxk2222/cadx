"""De-ASCII-fy German guide text.

LLM missions occasionally emit ASCII digraph spellings ("Gehaeuse",
"Waermepumpe") even though the manual's own vocabulary carries proper
umlauts. This module restores them — but ONLY against witnessed evidence:
a digraph token is replaced solely when it exactly matches the digraph
spelling of an umlaut word seen in the run's own sources (catalog part
names, context names, master notes, the source PDF's text layer). Real
German words that legitimately contain ae/oe/ue/ss ("Anschluss", "neues")
never collide because their digraph form is not in the vocabulary — the
map is a no-op without evidence, which keeps it honest and generalizable
to any manual.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kiara.logging import get_logger

log = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")

_ASCII_DIGRAPHS = (
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
    ("ß", "ss"),
    ("Ä", "Ae"),
    ("Ö", "Oe"),
    ("Ü", "Ue"),
)


def _digraph_form(word: str) -> str:
    out = word
    for umlaut, digraph in _ASCII_DIGRAPHS:
        out = out.replace(umlaut, digraph)
    # "ÖFFNUNG" digraphs to "OEFFNUNG", not "OeFFNUNG"
    return out.upper() if word.isupper() else out


def build_umlaut_map(*texts: str | None) -> dict[str, str]:
    """digraph spelling -> proper umlaut word, from witnessed vocabulary."""
    umap: dict[str, str] = {}
    for text in texts:
        for word in _WORD_RE.findall(text or ""):
            if not any(c in word for c in "äöüÄÖÜß"):
                continue
            umap.setdefault(_digraph_form(word), word)
            cap = word[0].upper() + word[1:]
            umap.setdefault(_digraph_form(cap), cap)
    return umap


def restore_umlauts(text: str | None, umap: dict[str, str]) -> str | None:
    if not text or not umap:
        return text
    return _WORD_RE.sub(lambda m: umap.get(m.group(0), m.group(0)), text)


def state_umlaut_map(state: Any) -> dict[str, str]:
    """Evidence vocabulary for one guide run: part names, context names,
    master/user notes, and the source PDF's own text layer (the manual is
    the terminology ground truth the LLM ASCII-fied)."""
    sources: list[str] = [str(p.name or "") for p in getattr(state, "parts", [])]
    sources.extend(str(n) for n in getattr(state, "context_names", []) or [])
    sources.extend(str(n) for n in getattr(state, "master_notes", []) or [])
    doc = getattr(state, "document", None)
    pdf_path = getattr(doc, "pdf_path", None) if doc is not None else None
    if pdf_path and Path(pdf_path).is_file():
        try:
            from kiara.guide.pdfdoc import extract_page_texts

            sources.extend(extract_page_texts(Path(pdf_path)).values())
        except Exception as exc:  # PDF evidence is additive — never block emit
            log.warning("guide.umlaut_pdf_failed", error=str(exc)[:200])
    return build_umlaut_map(*sources)
