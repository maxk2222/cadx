"""Image preparation for the mesh generator.

The image->3D models key on the isolated object: an off-centre subject, a
busy background or a non-square frame skews the reconstruction. We
background-remove (when the image has no alpha channel of its own), crop to
the object with a small margin and pad to a transparent square — feeding the
generator exactly what it was trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from cadx.logging import get_logger

log = get_logger(__name__)


def _object_bbox(im: object) -> tuple[int, int, int, int] | None:
    """Bounding box of non-transparent pixels (alpha > 8)."""
    from PIL import Image

    assert isinstance(im, Image.Image)
    alpha = im.getchannel("A")
    return alpha.point(lambda a: 255 if a > 8 else 0).getbbox()


def prepare_for_meshgen(path: Path, out_path: Path, *, size: int = 1024) -> Path:
    """Crop+pad an input image for the mesh generator.

    Loads the image as RGBA, removes the background via rembg when there is
    no meaningful alpha channel, crops to the object with an 8 % margin and
    pads onto a transparent square. Returns `out_path`; on any failure the
    original `path` is returned unchanged.
    """
    from PIL import Image, ImageFilter

    try:
        im = Image.open(path).convert("RGBA")
        alpha = im.getchannel("A")
        extrema = cast("tuple[float, float]", alpha.getextrema())
        lo, hi = extrema
        if hi - lo < 16:  # no real transparency — try background removal
            try:
                import rembg

                # u2net (~170 MB) instead of the 1 GB default model — the
                # worker container's RAM budget can't afford bria-rmbg-2.0.
                session = rembg.new_session("u2net")
                small = im.copy()
                small.thumbnail((768, 768), Image.Resampling.LANCZOS)
                cut = rembg.remove(small, session=session)
                if isinstance(cut, Image.Image):
                    mask = cut.getchannel("A").resize(im.size, Image.Resampling.LANCZOS)
                    im = im.copy()
                    im.putalpha(mask)
            except Exception as e:
                log.info("vision.prep.rembg_skipped", error=str(e)[:160])
        bbox = _object_bbox(im)
        if bbox is None:
            return path
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        m = int(0.08 * max(w, h))
        x0, y0 = max(0, x0 - m), max(0, y0 - m)
        x1, y1 = min(im.width, x1 + m), min(im.height, y1 + m)
        crop = im.crop((x0, y0, x1, y1))
        side = max(crop.width, crop.height)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        # soften the alpha edge so hard cut lines don't read as geometry
        soft = canvas.getchannel("A").filter(ImageFilter.GaussianBlur(1))
        canvas.putalpha(soft)
        canvas.thumbnail((size, size), Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, format="PNG")
        return out_path
    except Exception as e:
        log.warning("vision.prep.failed", path=str(path), error=str(e)[:200])
        return path
