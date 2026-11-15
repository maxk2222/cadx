"""PDF ingestion tests — self-contained minimal-PDF fixtures.

The hand-rolled PDF below is a valid single-content-stream-per-page
document; it exercises text extraction without shipping a binary fixture.
"""

from pathlib import Path

import pytest

from cadx.guide import pdfdoc


def _make_pdf(texts: list[str]) -> bytes:
    """Build a minimal valid PDF with one Helvetica text line per page."""
    n = len(texts)
    # object numbering: 1 catalog, 2 pages, then per page: page obj + content obj + font
    objs: dict[int, bytes] = {}
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(n))
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
    font_id = 3 + n * 2
    for i, text in enumerate(texts):
        content = f"BT /F1 18 Tf 50 750 Td ({text}) Tj ET".encode()
        objs[3 + i * 2] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {4 + i * 2} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()
        objs[4 + i * 2] = (
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
    objs[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref_at = len(out)
    max_id = max(objs)
    out += f"xref\n0 {max_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, max_id + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def _backend_available() -> bool:
    try:
        import pypdfium2  # noqa: F401

        return True
    except ImportError:
        try:
            import fitz  # noqa: F401

            return True
        except ImportError:
            return False


pytestmark = pytest.mark.skipif(
    not _backend_available(), reason="no PDF backend (pypdfium2/fitz) installed"
)


def test_render_pdf_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(_make_pdf(["Schritt 1", "Schritt 2", "Schritt 3"]))
    out = pdfdoc.render_pdf(pdf, tmp_path / "pages", dpi=100)
    assert len(out) == 3
    for _page_no, path, w, h in out:
        assert path.exists() and path.suffix == ".png"
        assert w > 100 and h > 100
    assert [p for p, *_ in out] == [1, 2, 3]


def test_extract_page_texts(tmp_path: Path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(_make_pdf(["Schraube M8", "Rohr einsetzen"]))
    texts = pdfdoc.extract_page_texts(pdf)
    assert "Schraube" in texts[1]
    assert "Rohr" in texts[2]


def test_max_pages_cap(tmp_path: Path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(_make_pdf([f"S{i}" for i in range(10)]))
    out = pdfdoc.render_pdf(pdf, tmp_path / "pages", dpi=72, max_pages=3)
    assert len(out) == 3


def test_extract_figures_empty_graceful(tmp_path: Path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(_make_pdf(["nur text"]))
    figs = pdfdoc.extract_figures(pdf, tmp_path / "figs")
    assert isinstance(figs, list)  # [] without fitz or without image blocks


def test_render_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        pdfdoc.render_pdf(tmp_path / "nope.pdf", tmp_path / "out")


def test_dedup_variant_text_drops_second_layer() -> None:
    """A page whose text layer restarts with its own first line carries a
    duplicated variant (OCR layer / revised wording) — keep the first."""
    text = (
        "Schritt 4:\n"
        "Teile: Rohr l = 70 mm\n"
        "Rohr in den linken oberen Adapter einfuehren.\n"
        "Schritt 5:\n"
        "Teile: Rohrbogen, Muffe auf Spitzende\n"
        "Bogen in den Adapter am linken unteren Anschluss einsetzen.\n"
        "Schritt 4:\n"
        "Sie benoetigen: Rohr l = 7 mm\n"
        "Stecken Sie das Rohr in die Bohrung des Adapters ein.\n"
        "Schritt 5:\n"
        "Sie benoetigen: Rohrbogen Anschluss 1x aussen 1x innen\n"
        "Bogen in die untere Anschlussstelle stecken.\n"
    )
    out = pdfdoc._dedup_variant_text(text)
    assert "Muffe auf Spitzende" in out
    assert "1x aussen" not in out
    assert out.count("Schritt 4:") == 1


def test_dedup_variant_text_keeps_distinct_steps() -> None:
    """Similar step wording ('links' vs 'rechts') is not a duplicated layer —
    a normal manual must survive untouched."""
    text = (
        "Schritt 1:\n"
        "Schrauben Sie den Adapter an das linke Anschlussrohr.\n"
        "Schritt 2:\n"
        "Schrauben Sie den Adapter an das rechte Anschlussrohr.\n"
        "Schritt 3:\n"
        "Stecken Sie das lange Rohr in den linken Bogen.\n"
        "Schritt 4:\n"
        "Stecken Sie das lange Rohr in den rechten Bogen.\n"
        "Schritt 5:\n"
        "Pruefen Sie alle Verbindungen auf festen Sitz.\n"
    )
    assert pdfdoc._dedup_variant_text(text) == text


def test_dedup_variant_text_first_line_repeat_without_mirror_keeps() -> None:
    """A single echoed first line (running footer/header quirk) without
    further head-line recurrences is not a layer restart."""
    text = (
        "Montageanleitung\n"
        "Schritt 1: Adapter setzen.\n"
        "Schritt 2: Rohr stecken.\n"
        "Schritt 3: Bogen montieren.\n"
        "Schritt 4: Rohr verlaengern.\n"
        "Schritt 5: Dichtung pruefen.\n"
        "Montageanleitung\n"
        "Anhang: Werkzeugliste folgt auf der naechsten Seite.\n"
    )
    assert pdfdoc._dedup_variant_text(text) == text
