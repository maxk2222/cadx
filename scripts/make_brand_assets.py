#!/usr/bin/env python3
"""Generate CADX web brand assets from the source PNGs.

Sources (edit SRC_* if they move):
    ~/Downloads/CADX-Logo.png      landscape logo, dark background
    ~/Downloads/CADX_Favicon.png   rounded app-icon tile

Outputs under frontend/public/:
    brand/logo.png        cropped banner logo (opaque, for README/docs/sidebar)
    brand/wordmark.png    full logo with transparent background (sidebar)
    brand/icon.png        square app-icon tile (opaque)
    favicon.ico           16/32/48 multi-res
    favicon-{16,32,48}.png, apple-touch-icon.png (180),
    icon-{192,512}.png

Transparency for wordmark.png uses a border flood fill: only the contiguous
outer background becomes alpha=0, dark regions enclosed by the glowing X stay
opaque. If it fringes, just use logo.png instead.

Usage: backend/.venv/bin/python scripts/make_brand_assets.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC_LOGO = Path.home() / "Downloads" / "CADX_Logo.png"  # transparent RGBA source
DARK_BG = (10, 15, 30, 255)  # base for the opaque banner card
SRC_ICON = Path.home() / "Downloads" / "CADX_favicon_transparent.png"
OUT = ROOT / "frontend" / "public"

BG_THRESH = 48  # flood-fill color tolerance


def content_bbox(im: Image.Image, lum: int = 36) -> tuple[int, int, int, int]:
    """Bounding box of visible content — alpha bbox for RGBA, luminance
    threshold against the dark background otherwise."""
    if im.mode == "RGBA":
        return im.getchannel("A").getbbox() or (0, 0, im.width, im.height)
    mask = im.convert("L").point(lambda p: 255 if p > lum else 0)
    return mask.getbbox() or (0, 0, im.width, im.height)


def border_transparent(im: Image.Image, thresh: int = BG_THRESH) -> Image.Image:
    """Flood-fill the contiguous outer background to transparent."""
    work = im.convert("RGB").copy()
    w, h = work.size
    step = 3
    seeds = [(x, 0) for x in range(0, w, step)] + [(x, h - 1) for x in range(0, w, step)]
    seeds += [(0, y) for y in range(0, h, step)] + [(w - 1, y) for y in range(0, h, step)]
    for seed in seeds:
        if work.getpixel(seed) != (255, 0, 255):
            ImageDraw.floodfill(work, seed, (255, 0, 255), thresh=thresh)
    rgba = im.convert("RGBA")
    px = work.load()
    mask = Image.new("L", work.size, 255)
    mp = mask.load()
    for y in range(h):
        for x in range(w):
            if px[x, y] == (255, 0, 255):
                mp[x, y] = 0
    rgba.putalpha(mask.filter(ImageFilter.GaussianBlur(1.2)))
    return rgba


def main() -> None:
    (OUT / "brand").mkdir(parents=True, exist_ok=True)

    logo = Image.open(SRC_LOGO).convert("RGBA")
    icon = Image.open(SRC_ICON).convert("RGBA")
    # opaque variant on app-dark background for touch/PWA icons
    icon_op = Image.new("RGBA", icon.size, (9, 9, 11, 255))
    icon_op.alpha_composite(icon)
    icon_op = icon_op.convert("RGB")

    # --- banner logo: crop to content, composite on dark card, resize ---
    # (the silver "CAD" glyphs need a dark backdrop to stay readable on
    # light themes — e.g. GitHub README)
    bb = content_bbox(logo)
    pad = 40
    bb = (
        max(0, bb[0] - pad),
        max(0, bb[1] - pad),
        min(logo.width, bb[2] + pad),
        min(logo.height, bb[3] + pad),
    )
    logo_c = logo.crop(bb)
    card = Image.new("RGBA", logo_c.size, DARK_BG)
    card.alpha_composite(logo_c)
    banner = card.convert("RGB")
    if banner.width > 1200:
        banner = banner.resize((1200, int(banner.height * 1200 / banner.width)), Image.LANCZOS)
    banner.save(OUT / "brand" / "logo.png", optimize=True)

    # --- hero bar: full-width dark banner with centered logo (README/docs) ---
    # rounded transparent corners so it reads as a card on light themes
    HW, HH = 1600, 400
    hero = Image.new("RGBA", (HW, HH), DARK_BG)
    lh = HH - 120
    logo_h = logo_c.resize((int(logo_c.width * lh / logo_c.height), lh), Image.LANCZOS)
    hero.alpha_composite(logo_h, ((HW - logo_h.width) // 2, (HH - logo_h.height) // 2))
    mask = Image.new("L", (HW, HH), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, HW, HH], radius=32, fill=255)
    hero.putalpha(mask)
    hero.save(OUT / "brand" / "hero.png", optimize=True)

    # --- transparent wordmark (sidebar-sized) ---
    wm_small = logo_c.resize((int(logo_c.width * 192 / logo_c.height), 192), Image.LANCZOS)
    wm_small.save(OUT / "brand" / "wordmark.png", optimize=True)

    # --- transparent X mark + favicon set ---
    icon.resize((256, 256), Image.LANCZOS).save(OUT / "brand" / "icon.png", optimize=True)
    # favicons keep the transparent X; PWA/touch icons get an opaque dark base
    for size in (16, 32, 48):
        icon.resize((size, size), Image.LANCZOS).save(OUT / f"favicon-{size}.png", optimize=True)
    icon_op.resize((180, 180), Image.LANCZOS).save(OUT / "apple-touch-icon.png", optimize=True)
    for size in (192, 512):
        icon_op.resize((size, size), Image.LANCZOS).save(OUT / f"icon-{size}.png", optimize=True)
    icon.save(
        OUT / "favicon.ico",
        sizes=[(16, 16), (32, 32), (48, 48)],
    )
    print("done ->", OUT)


if __name__ == "__main__":
    main()
