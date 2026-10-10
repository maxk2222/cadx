"""Input image preprocessing: EXIF fix, square crop/pad, optional bg removal."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

TARGET = 1024


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGBA") if img.mode == "P" else img


def remove_bg(img: Image.Image) -> Image.Image:
    """Subject isolation via rembg (u2net). Returns RGBA with transparent bg."""
    from rembg import remove  # type: ignore[import-not-found]

    out = remove(img.convert("RGB"))
    if isinstance(out, (bytes, bytearray)):
        import io

        out = Image.open(io.BytesIO(out))
    return out.convert("RGBA")


def to_square(img: Image.Image, size: int = TARGET) -> Image.Image:
    """Pad to square on white (RGB) or transparent (RGBA), preserving aspect."""
    img.thumbnail((size, size), Image.LANCZOS)
    has_alpha = img.mode == "RGBA" and img.getextrema()[3][0] < 255
    bg_mode = "RGBA" if has_alpha else "RGB"
    bg_val = (255, 255, 255, 0) if has_alpha else (255, 255, 255)
    canvas = Image.new(bg_mode, (size, size), bg_val)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def _rembg_cached() -> bool:
    """True when a rembg model is already on disk — avoids a surprise ~1 GB
    download inside a request. 'always' still attempts it."""
    models_dir = Path.home() / ".rembg" / "models"
    return models_dir.is_dir() and any(models_dir.rglob("*.onnx"))


def prep_images(paths: list[Path], bg_mode: str = "auto") -> list[Image.Image]:
    """bg_mode: auto (rembg only when installed AND model cached), always, never."""
    out: list[Image.Image] = []
    for p in paths:
        img = load_image(p)
        do_bg = bg_mode == "always"
        if bg_mode == "auto" and _rembg_cached():
            try:
                import rembg  # noqa: F401

                do_bg = True
            except Exception:
                do_bg = False
        if do_bg:
            try:
                img = remove_bg(img)
            except Exception:
                pass
        out.append(to_square(img))
    return out
