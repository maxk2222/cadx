"""PDF ingestion for the Anleitungsbuilder.

Port of KIARA's ``pdf_img_extraction/pdf_to_images.py`` — renders each page
to a PNG (VLM input), extracts embedded per-page text (OCR-lite; scanned
pages stay readable through the page image the VLM receives), and crops
figure blocks when PyMuPDF is available.

Backend cascade: pypdfium2 (primary, permissive licence) → PyMuPDF/fitz
(also drives figure cropping) → pdf2image (needs system poppler).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cadx.logging import get_logger

log = get_logger(__name__)

DEFAULT_DPI = 200
DEFAULT_FMT = "png"


def _dedup_variant_text(text: str) -> str:
    """Drop a whole-page duplicated text variant.

    Some PDFs carry multiple text layers per page — the visible layer
    plus hidden variants (an OCR layer, revised wording, an imposition
    duplicate). The model then reads every step twice with DIFFERENT
    wording and reconciles them differently each run — a structural
    source of plan variance. The reliable signature: the duplicated
    layer restarts mid-page with the page's own opening line; the tail
    then mirrors the head. When that restart is found, keep the first
    variant only. Conservative by design — a normal manual never
    reprints its first line mid-page, and the head-mirror check rejects
    coincidences; legit repeated step wording ("…links"/"…rechts") is
    untouched. If a variant restarts with a different heading than the
    page's first line, it stays (status quo) — precision over recall.
    """
    import difflib

    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    lines = [" ".join(ln.split()).lower() for ln in raw_lines]
    if len(lines) < 7:
        return text

    def _sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).quick_ratio()

    for k in range(3, len(lines) - 3):
        # the duplicate layer restarts with the page's own FIRST line —
        # near-exact (OCR may typo a char) but strict enough that the next
        # numbered heading ("Schritt 5:" vs "Schritt 4:" = 0.9) can't fake it
        if _sim(lines[k], lines[0]) < 0.95:
            continue
        # confirm it is a real restart, not a one-off echoed header: the
        # opening line repeats AGAIN later, or another head line recurs in
        # the tail — either way the page content re-runs from here
        head = set(lines[:k])
        tail = lines[k + 1 :]
        second = any(_sim(lines[k], tl) >= 0.95 for tl in tail) or any(
            any(_sim(hl, tl) >= 0.8 for tl in tail) for hl in head
        )
        if not second:
            continue
        log.info(
            "pdf.dedup_variant",
            kept_lines=k,
            dropped_lines=len(lines) - k,
        )
        return "\n".join(raw_lines[:k]) + "\n"
    return text


def render_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = DEFAULT_DPI,
    max_pages: int = 0,
    image_format: str = DEFAULT_FMT,
) -> list[tuple[int, Path, int, int]]:
    """Render every page to ``out_dir/page_NNNN.<fmt>``.

    Returns [(page_number, path, width, height)] — raises RuntimeError when
    no backend is usable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    last_err: Exception | None = None

    # --- 1) pypdfium2 (primary — bundled pdfium, permissive licence) ---
    try:
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(str(pdf_path))
        limit = len(pdf) if max_pages <= 0 else min(len(pdf), max_pages)
        out: list[tuple[int, Path, int, int]] = []
        for i in range(limit):
            pil = pdf[i].render(scale=scale).to_pil()
            p = out_dir / f"page_{i + 1:04d}.{image_format}"
            pil.save(p)
            out.append((i + 1, p, pil.width, pil.height))
        pdf.close()
        return out
    except ImportError:
        pass
    except Exception as e:
        last_err = e

    # --- 2) PyMuPDF (fitz) ---
    try:
        import pymupdf as fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        limit = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
        mat = fitz.Matrix(scale, scale)
        out = []
        for i in range(limit):
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            p = out_dir / f"page_{i + 1:04d}.{image_format}"
            pix.save(str(p))
            out.append((i + 1, p, int(pix.width), int(pix.height)))
        doc.close()
        return out
    except ImportError:
        pass
    except Exception as e:
        last_err = e

    # --- 3) pdf2image (needs system poppler) ---
    try:
        from pdf2image import convert_from_path  # type: ignore

        images = convert_from_path(str(pdf_path), dpi=dpi, fmt=image_format)
        out = []
        for i, img in enumerate(images[: max_pages or len(images)]):
            p = out_dir / f"page_{i + 1:04d}.{image_format}"
            img.save(p)
            out.append((i + 1, p, img.width, img.height))
        if out:
            return out
    except ImportError:
        pass
    except Exception as e:
        last_err = e

    raise RuntimeError(
        f"no PDF render backend usable (last error: {last_err!r}) — install pypdfium2 or PyMuPDF"
    )


def extract_page_texts(pdf_path: Path, max_pages: int = 0) -> dict[int, str]:
    """Embedded text per page — {page_number: text}. Best-effort; a page
    with no text layer yields "" (the VLM still gets the page image)."""
    # 1) pypdfium2
    try:
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(str(pdf_path))
        limit = len(pdf) if max_pages <= 0 else min(len(pdf), max_pages)
        out: dict[int, str] = {}
        for i in range(limit):
            page = pdf[i]
            try:
                tp = page.get_textpage()
                out[i + 1] = tp.get_text_range() or ""
            except Exception:
                out[i + 1] = ""
            finally:
                page.close()
        pdf.close()
        return {k: _dedup_variant_text(v) for k, v in out.items()}
    except ImportError:
        pass
    except Exception:
        pass

    # 2) PyMuPDF
    try:
        import pymupdf as fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        limit = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
        out = {i + 1: (doc.load_page(i).get_text() or "") for i in range(limit)}
        doc.close()
        return {k: _dedup_variant_text(v) for k, v in out.items()}
    except ImportError:
        pass
    except Exception:
        pass

    # 3) PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError as e:
        raise RuntimeError("no PDF text backend usable — install pypdfium2") from e
    reader = PdfReader(str(pdf_path))
    limit = len(reader.pages) if max_pages <= 0 else min(len(reader.pages), max_pages)
    out = {}
    for i in range(limit):
        try:
            out[i + 1] = reader.pages[i].extract_text() or ""
        except Exception:
            out[i + 1] = ""
    return {k: _dedup_variant_text(v) for k, v in out.items()}


def extract_figures(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = DEFAULT_DPI,
    max_pages: int = 0,
    *,
    min_area_ratio: float = 0.01,
    max_images_per_page: int = 8,
) -> list[dict[str, Any]]:
    """Crop image blocks from each page (PyMuPDF only).

    Returns [{"page", "image_path", "bbox", "index"}] sorted per page —
    empty list when fitz is unavailable or no figures exist.
    """
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        log.info("guide.figures_no_fitz")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log.warning("guide.figures_open_failed", error=str(e)[:200])
        return []
    limit = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
    mat = fitz.Matrix(scale, scale)
    out: list[dict[str, Any]] = []
    for i in range(limit):
        out.extend(
            _page_figures(
                doc.load_page(i),
                i + 1,
                mat,
                out_dir,
                min_area_ratio=min_area_ratio,
                max_images=max_images_per_page,
                fitz=fitz,
            )
        )
    doc.close()
    return out


def _figure_rects(page: Any, page_area: float, min_area_ratio: float, fitz: Any) -> list[Any]:
    """Deduped, area-filtered image rects of one page (layout blocks → xref)."""
    rects: list[Any] = []
    # A) layout-dict image blocks
    try:
        for b in (page.get_text("dict") or {}).get("blocks", []):
            if isinstance(b, dict) and b.get("type") == 1 and len(b.get("bbox") or ()) == 4:
                rects.append(fitz.Rect(b["bbox"]))
    except Exception:
        pass
    # B) xref rects fallback
    if not rects:
        try:
            for img in page.get_images(full=True):
                rects.extend(page.get_image_rects(img[0]))
        except Exception:
            pass
    # dedupe + area filter, top→bottom / left→right
    uniq: list[Any] = []
    seen: set[tuple[float, float, float, float]] = set()
    for r in rects:
        try:
            if r is None or r.is_empty or r.get_area() / page_area < min_area_ratio:
                continue
            key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        except Exception:
            continue
    uniq.sort(key=lambda r: (float(r.y0), float(r.x0)))
    return uniq


def _page_figures(
    page: Any,
    page_no: int,
    mat: Any,
    out_dir: Path,
    *,
    min_area_ratio: float,
    max_images: int,
    fitz: Any,
) -> list[dict[str, Any]]:
    page_area = float(page.rect.get_area()) if page.rect else 0.0
    if page_area <= 0:
        return []
    out: list[dict[str, Any]] = []
    for k, r in enumerate(_figure_rects(page, page_area, min_area_ratio, fitz)[:max_images], 1):
        try:
            pix = page.get_pixmap(matrix=mat, clip=r, alpha=False)
            p = out_dir / f"page_{page_no:04d}_fig_{k:02d}.png"
            pix.save(str(p))
            out.append(
                {
                    "page": page_no,
                    "image_path": str(p),
                    "bbox": [float(r.x0), float(r.y0), float(r.x1), float(r.y1)],
                    "index": k,
                }
            )
        except Exception:
            continue
    return out
