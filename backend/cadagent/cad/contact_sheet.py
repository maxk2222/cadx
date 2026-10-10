"""Compose the 7 view renders + a stats tile into one labelled contact sheet PNG."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ORDER = ["front", "top", "right", "iso", "back", "bottom", "left"]
TILE = 512
COLS = 4


def make_contact_sheet(
    renders: dict[str, Path], out_path: Path, stats_lines: list[str]
) -> Path | None:
    tiles = [p for v in ORDER if (p := renders.get(v)) and p.exists()]
    if not tiles:
        return None

    rows = 2
    sheet = Image.new("RGB", (COLS * TILE, rows * TILE), "white")
    draw = ImageDraw.Draw(sheet)

    for i, view in enumerate(ORDER[: COLS * rows - 1]):
        col, row = i % COLS, i // COLS
        img_path = renders.get(view)
        if img_path and img_path.exists():
            img = Image.open(img_path).convert("RGB").resize((TILE, TILE))
            sheet.paste(img, (col * TILE, row * TILE))
        draw.rectangle(
            [col * TILE, row * TILE, (col + 1) * TILE - 1, (row + 1) * TILE - 1],
            outline="#cccccc",
        )

    # stats tile (last cell)
    col, row = (COLS * rows - 1) % COLS, (COLS * rows - 1) // COLS
    x0, y0 = col * TILE, row * TILE
    draw.rectangle([x0, y0, x0 + TILE - 1, y0 + TILE - 1], fill="#f5f5f5", outline="#cccccc")
    for i, line in enumerate(stats_lines[:18]):
        draw.text((x0 + 16, y0 + 16 + i * 26), line, fill="#222222")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
