"""make_contact_sheet: dynamic rows once ghost (xray) tiles join the 7 views."""

from pathlib import Path

from PIL import Image

from cadx.cad.contact_sheet import COLS, TILE, make_contact_sheet


def _png(path: Path, rgb: tuple[int, int, int] = (150, 150, 150)) -> Path:
    Image.new("RGB", (8, 8), rgb).save(path)
    return path


def test_sheet_with_ghost_tiles_has_three_rows(tmp_path):
    renders = {
        v: _png(tmp_path / f"{v}.png")
        for v in (
            "front",
            "top",
            "right",
            "iso",
            "back",
            "bottom",
            "left",
            "iso_ghost",
            "front_ghost",
        )
    }
    sheet = make_contact_sheet(renders, tmp_path / "sheet.png", ["bbox: 1 x 2 x 3"])
    assert sheet is not None
    assert Image.open(sheet).size == (COLS * TILE, 3 * TILE)


def test_sheet_without_ghosts_keeps_row_layout(tmp_path):
    renders = {v: _png(tmp_path / f"{v}.png") for v in ("front", "iso")}
    sheet = make_contact_sheet(renders, tmp_path / "sheet.png", [])
    assert sheet is not None
    # rows derive from ORDER (9 tiles + stats = 10 cells -> 3 rows) regardless
    # of which tiles are missing — empty cells just keep the border
    assert Image.open(sheet).size == (COLS * TILE, 3 * TILE)


def test_sheet_none_without_renders(tmp_path):
    assert make_contact_sheet({}, tmp_path / "s.png", ["x"]) is None
    assert not (tmp_path / "s.png").exists()
