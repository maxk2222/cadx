"""Image preparation for LLM vision input (EXIF-rotate, downscale, JPEG)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

MAX_EDGE = 1568
JPEG_QUALITY = 85

# Uploaded reference images are normalized to this bound; the LLM path
# (prepare()) downscales further to MAX_EDGE. 2048 keeps headroom for the
# mesh pipeline and previews.
UPLOAD_MAX_EDGE = 2048
UPLOAD_JPEG_QUALITY = 90


def _flatten(img: Image.Image) -> Image.Image:
    """Composite onto white so transparent areas don't turn black."""
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        base = Image.new("RGBA", img.size, (255, 255, 255, 255))
        base.alpha_composite(img.convert("RGBA"))
        return base.convert("RGB")
    return img.convert("RGB")


def normalize(
    data: bytes, *, max_edge: int = UPLOAD_MAX_EDGE, quality: int = UPLOAD_JPEG_QUALITY
) -> tuple[bytes, int, int] | None:
    """Normalize an uploaded raster image → (jpeg_bytes, width, height).

    Applies EXIF rotation, takes the first frame of animated formats and
    downscales to `max_edge` (never upscales). Returns None when PIL cannot
    decode the data (e.g. HEIC without plugin) — the caller stores raw bytes.
    """
    try:
        img: Image.Image = Image.open(io.BytesIO(data))
        img.seek(0)  # animated formats (gif): first frame only
        img = _flatten(ImageOps.exif_transpose(img))
        w, h = img.size
        if max(w, h) > max_edge:
            scale = max_edge / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), *img.size
    except Exception:
        return None


def prepare(path: Path) -> bytes:
    img = _flatten(ImageOps.exif_transpose(Image.open(path)))
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > MAX_EDGE:
        scale = MAX_EDGE / long_edge
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def to_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(prepare(path)).decode()


def image_part(path: Path, detail: str = "high") -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": to_data_url(path), "detail": detail},
    }
