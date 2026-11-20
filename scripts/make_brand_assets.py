#!/usr/bin/env python3
"""Generate KIARA web brand assets from the source PNGs.

Sources (checked in — see assets/brand/):
    KIARA_HO_BW_Transparent_White.png   hexagon mark + KIARA wordmark, white
    KIARA_HO_BWC_Transparent_White.png  same + tagline ("KI-basierte
                                        AR-Anleitungen"), used for hero

Outputs under frontend/public/:
    brand/logo.png        banner logo on dark card (opaque, for README/docs)
    brand/hero.png        wide banner with tagline variant (README/docs)
    brand/wordmark.png    full logo with transparent background (sidebar)
    brand/icon.png        square app-icon mark (transparent)
    favicon.ico           16/32/48 multi-res
    favicon-{16,32,48}.png, apple-touch-icon.png (180),
    icon-{192,512}.png

The hexagon mark is split from the wordmark source at the first wide
alpha gap between the mark and the "K" — no manual crop coordinates.
The GUI is dark-only, so transparent assets stay transparent; opaque
variants (PWA/touch icons, README banner) composite onto the app-dark
base so the white glyphs stay readable on light themes.

Usage: backend/.venv/bin/python scripts/make_brand_assets.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SRC_WORDMARK = ROOT / "assets" / "brand" / "KIARA_HO_BW_Transparent_White.png"
SRC_TAGLINE = ROOT / "assets" / "brand" / "KIARA_HO_BWC_Transparent_White.png"
OUT = ROOT / "frontend" / "public"

DARK_BG = (9, 9, 11, 255)  # app background — opaque variants sit on this


def content_bbox(im: Image.Image) -> tuple[int, int, int, int]:
    """Alpha bounding box of visible content."""
    return im.getchannel("A").getbbox() or (0, 0, im.width, im.height)


def hexagon_mark(im: Image.Image, gap: int = 40) -> Image.Image:
    """Left-of-first-gap crop: the hexagon mark ends at the first wide
    alpha column gap before the 'K' glyph."""
    a = np.asarray(im.getchannel("A"), dtype=float).sum(axis=0)
    nz = np.where(a > 5)[0]
    run_start = None
    for x in range(int(nz[0]), int(nz[-1])):
        if a[x] <= 5:
            if run_start is None:
                run_start = x
        elif run_start is not None and x - run_start > gap:
            return im.crop((0, 0, run_start, im.height)).crop(
                im.crop((0, 0, run_start, im.height)).getchannel("A").getbbox()
            )
        elif run_start is not None:
            run_start = None
    return im


def square_tile(mark: Image.Image, size: int, bg: tuple | None) -> Image.Image:
    """Mark centered on a (possibly transparent) square with padding —
    icon/favicon geometry in one place."""
    tile = Image.new("RGBA", (size, size), bg or (0, 0, 0, 0))
    inner = int(size * 0.82)
    m = mark.resize(
        (inner, max(1, int(mark.height * inner / mark.width))), Image.LANCZOS
    )
    tile.alpha_composite(m, ((size - m.width) // 2, (size - m.height) // 2))
    return tile


def main() -> None:
    (OUT / "brand").mkdir(parents=True, exist_ok=True)

    logo = Image.open(SRC_WORDMARK).convert("RGBA")
    tagline = Image.open(SRC_TAGLINE).convert("RGBA")
    mark = hexagon_mark(logo)

    # --- transparent wordmark (sidebar) — same height as the old asset ---
    logo_c = logo.crop(content_bbox(logo))
    wm_small = logo_c.resize(
        (int(logo_c.width * 192 / logo_c.height), 192), Image.LANCZOS
    )
    wm_small.save(OUT / "brand" / "wordmark.png", optimize=True)

    # --- banner logo on dark card (opaque, for README/docs/sidebar) ---
    pad = 40
    bb = content_bbox(logo)
    logo_c2 = logo.crop(
        (
            max(0, bb[0] - pad),
            max(0, bb[1] - pad),
            min(logo.width, bb[2] + pad),
            min(logo.height, bb[3] + pad),
        )
    )
    card = Image.new("RGBA", logo_c2.size, DARK_BG)
    card.alpha_composite(logo_c2)
    banner = card.convert("RGB")
    if banner.width > 1200:
        banner = banner.resize(
            (1200, int(banner.height * 1200 / banner.width)), Image.LANCZOS
        )
    banner.save(OUT / "brand" / "logo.png", optimize=True)

    # --- hero bar: tagline variant on rounded dark card (README/docs) ---
    HW, HH = 1600, 400
    hero = Image.new("RGBA", (HW, HH), DARK_BG)
    tc = tagline.crop(content_bbox(tagline))
    lh = HH - 120
    logo_h = tc.resize((int(tc.width * lh / tc.height), lh), Image.LANCZOS)
    hero.alpha_composite(logo_h, ((HW - logo_h.width) // 2, (HH - logo_h.height) // 2))
    mask = Image.new("L", (HW, HH), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, HW, HH], radius=32, fill=255)
    hero.putalpha(mask)
    hero.save(OUT / "brand" / "hero.png", optimize=True)

    # --- hexagon mark + favicon set ---
    square_tile(mark, 256, None).save(OUT / "brand" / "icon.png", optimize=True)
    for size in (16, 32, 48):
        square_tile(mark, size, None).save(OUT / f"favicon-{size}.png", optimize=True)
    square_tile(mark, 180, DARK_BG).convert("RGB").save(
        OUT / "apple-touch-icon.png", optimize=True
    )
    for size in (192, 512):
        square_tile(mark, size, DARK_BG).convert("RGB").save(
            OUT / f"icon-{size}.png", optimize=True
        )
    square_tile(mark, 256, None).save(
        OUT / "favicon.ico",
        sizes=[(16, 16), (32, 32), (48, 48)],
    )
    print("done ->", OUT)


if __name__ == "__main__":
    main()
