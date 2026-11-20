"""Content validation for project media uploads.

The browser/client-provided MIME type is NEVER trusted. Each allowed asset
type gets a lightweight signature check; the returned content type is derived
from the VERIFIED file content and is what gets stored in the media index.

Honest limitation (documented, not hidden): text-based CAD formats
(.obj, ASCII .stl, ASCII .fbx) carry no magic bytes and are only checked as
non-empty + UTF-8-decodable header. We do not claim full CAD validation.
"""

from __future__ import annotations

import codecs
import json
import struct
import warnings
from pathlib import Path

from PIL import Image, ImageOps

from kiara.guides.paths_cfg import settings
from kiara.guides.upload_errors import InvalidUploadError

# Pillow format name -> (accepted extensions, content type)
_PIL_FORMATS = {
    "PNG": ({".png"}, "image/png"),
    "JPEG": ({".jpg", ".jpeg"}, "image/jpeg"),
    "WEBP": ({".webp"}, "image/webp"),
}


def _validate_image(path: Path, ext: str) -> str:
    try:
        with warnings.catch_warnings():
            # Dekompressionsbomben sind ein Fehler, keine Warnung.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                fmt = (img.format or "").upper()
                width, height = img.size

                if fmt not in _PIL_FORMATS:
                    raise InvalidUploadError(f"unsupported image format '{fmt or 'unknown'}'")

                allowed_exts, content_type = _PIL_FORMATS[fmt]
                if ext not in allowed_exts:
                    raise InvalidUploadError(
                        f"image content is {fmt} but filename extension is '{ext}'"
                    )

                if width <= 0 or height <= 0:
                    raise InvalidUploadError("image has invalid dimensions")
                if width * height > settings.MEDIA_MAX_IMAGE_PIXELS:
                    raise InvalidUploadError(
                        f"image dimensions {width}x{height} exceed the "
                        f"{settings.MEDIA_MAX_IMAGE_PIXELS}-pixel limit"
                    )

                img.verify()
                return content_type
    except InvalidUploadError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidUploadError("image rejected: decompression bomb") from exc
    except Exception as exc:  # Pillow wirft diverse Fehlertypen bei kaputten Dateien
        raise InvalidUploadError(f"invalid image file: {exc}") from exc


_EXIF_ORIENTATION_TAG = 0x0112


def upright_image_in_place(path: Path) -> None:
    """Turn a still image's pixels the way its EXIF orientation says, over the file at `path`.

    C2 §2n: a browser obeys that note and Unity's image loader doesn't. The result has upright pixels
    and no EXIF at all, so a date or GPS note in that file goes too. Any other image keeps its bytes
    exactly, and anything animated is left alone.
    """
    try:
        # `_validate_image`'s verify() uses its handle up, so this one is new.
        with Image.open(path) as img:
            if getattr(img, "is_animated", False):
                return
            if img.getexif().get(_EXIF_ORIENTATION_TAG) not in range(2, 9):
                return
            # The turned copy has no format, and the temp file's name has no image extension.
            fmt = img.format
            icc = img.info.get("icc_profile")
            upright = ImageOps.exif_transpose(img)
    except Exception as exc:  # Pillow wirft diverse Fehlertypen bei kaputten Dateien
        raise InvalidUploadError(f"invalid image file: {exc}") from exc

    # Outside that `try`: the picture is read by now, so a failure here is the server's, like a full disk.
    options = {"quality": 95} if fmt == "JPEG" else {}
    upright.save(path, format=fmt, icc_profile=icc, **options)


def _read_head(path: Path, n: int) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def _validate_wav(path: Path) -> str:
    head = _read_head(path, 12)
    if len(head) < 12 or head[0:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise InvalidUploadError("invalid WAV file (RIFF/WAVE signature missing)")
    return "audio/wav"


def _validate_mp4(path: Path) -> str:
    head = _read_head(path, 12)
    if len(head) < 12 or head[4:8] != b"ftyp":
        raise InvalidUploadError("invalid MP4 file (ISO-BMFF 'ftyp' box missing)")
    return "video/mp4"


def _is_text_head(path: Path) -> bool:
    head = _read_head(path, 4096)
    if not head:
        return False
    try:
        # Not final: a multi-byte character cut off at the end of the head is fine.
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
        return True
    except UnicodeDecodeError:
        # Can still be single-byte text such as Latin-1. NUL and other control bytes mean it isn't.
        return all(byte >= 0x20 or byte in b"\t\n\f\r" for byte in head)


def _validate_cad(path: Path, ext: str) -> str:  # noqa: C901 - ported suite contract logic
    size = path.stat().st_size
    if size == 0:
        raise InvalidUploadError("empty CAD file")

    if ext == ".glb":
        if _read_head(path, 4) != b"glTF":
            raise InvalidUploadError("invalid GLB file ('glTF' magic missing)")
        return "model/gltf-binary"

    if ext == ".gltf":
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidUploadError("invalid glTF file (not valid JSON)") from exc
        if not isinstance(doc, dict) or "asset" not in doc:
            raise InvalidUploadError("invalid glTF file (missing 'asset')")
        return "model/gltf+json"

    if ext == ".stl":
        head = _read_head(path, 84)
        # ASCII-STL beginnt mit "solid"; Binär-STL: 80-Byte-Header + uint32-Dreieckszahl,
        # Gesamtgröße muss 84 + n*50 sein.
        if head[:5] == b"solid" and _is_text_head(path):
            return "model/stl"
        if len(head) >= 84:
            (tri_count,) = struct.unpack("<I", head[80:84])
            if size == 84 + tri_count * 50:
                return "model/stl"
        raise InvalidUploadError("invalid STL file (neither ASCII nor well-formed binary)")

    if ext == ".fbx":
        if _read_head(path, 18) == b"Kaydara FBX Binary":
            return "application/octet-stream"
        if _is_text_head(path):  # ASCII-FBX: nur Leichtgewichts-Check (dokumentierte Grenze)
            return "application/octet-stream"
        raise InvalidUploadError("invalid FBX file")

    if ext == ".obj":
        if _is_text_head(path):  # Leichtgewichts-Check (dokumentierte Grenze)
            return "model/obj"
        raise InvalidUploadError("invalid OBJ file (not readable as text)")

    raise InvalidUploadError(f"unhandled CAD extension '{ext}'")


def _validate_hands(path: Path) -> str:
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InvalidUploadError("invalid hand recording (not valid JSON)") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("frameList"), list):
        raise InvalidUploadError("invalid hand recording (missing 'frameList')")
    return "application/json"


def validate_media_content(path: Path, asset_type: str, ext: str) -> str:
    """Validate the file at `path` for its asset type; return the verified content type."""
    e = str(ext or "").lower()

    if asset_type == "images":
        return _validate_image(path, e)
    if asset_type == "audio":
        return _validate_wav(path)
    if asset_type == "videos":
        return _validate_mp4(path)
    if asset_type == "cad_models":
        return _validate_cad(path, e)
    if asset_type == "hands":
        return _validate_hands(path)

    raise InvalidUploadError(f"unknown asset type '{asset_type}'")
