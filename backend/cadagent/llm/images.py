"""Image preparation for LLM vision input (EXIF-rotate, downscale, JPEG)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

MAX_EDGE = 1568
JPEG_QUALITY = 85


def prepare(path: Path) -> bytes:
    img: Image.Image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
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
