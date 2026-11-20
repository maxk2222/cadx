"""PDF ingestion tests — self-contained minimal-PDF fixtures.

The hand-rolled PDF below is a valid single-content-stream-per-page
document; it exercises text extraction without shipping a binary fixture.
"""

from pathlib import Path

import pytest

from kiara.guide import pdfdoc


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
