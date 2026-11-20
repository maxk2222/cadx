"""C2 §2l: the canonical check for a model in the shared store (R1, U14).

File logic only, with no web framework: Slices 7 and 8 call it too. A refusal is an
`UnsupportedMediaTypeError` whose text is written for the person who uploaded the file.

If nothing needs changing, the canonical file is exactly the source's bytes. Otherwise the output
depends only on the input, so the same input always gives the same file and hash.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, NoReturn
from urllib.parse import unquote_to_bytes

from PIL import Image

from kiara.guides.paths_cfg import settings
from kiara.guides.upload_errors import UnsupportedMediaTypeError

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_COPY_CHUNK_BYTES = 1024 * 1024

# §2l: the longest side a texture keeps; the 452-vertex arrow is 9.4 MB without this rule.
MAX_TEXTURE_SIDE = 1024
_JPEG_QUALITY = 90

ALLOWED_REQUIRED_EXTENSIONS = frozenset(
    {"KHR_mesh_quantization", "KHR_texture_transform", "KHR_materials_unlit"}
)
# Refused whether required or only used: R1 keeps these out of a canonical file altogether (U14).
_COMPRESSION_EXTENSIONS = {
    "KHR_draco_mesh_compression": "Draco",
    "KHR_texture_basisu": "KTX",
    "EXT_meshopt_compression": "meshopt",
    "KHR_meshopt_compression": "meshopt",
}
# A normalized accessor's min and max hold the stored integers (KHR_mesh_quantization).
_NORMALIZED_DIVISORS = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
_IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
_MALFORMED_ERRORS = (AttributeError, IndexError, KeyError, OverflowError, RecursionError, TypeError, ValueError, struct.error)
_MALFORMED_REASON = "This GLB's contents are malformed, so it can't be read."


@dataclass(frozen=True)
class CanonicalGlb:
    sha256: str
    size_bytes: int
    bounds: dict[str, float]
    warnings: list[str]
    changed: bool


@dataclass(frozen=True)
class _Container:
    doc: dict[str, Any]
    bin_start: int | None
    bin_length: int


def _refuse(reason: str) -> NoReturn:
    raise UnsupportedMediaTypeError(reason)


def canonicalize_glb(source: Path, target: Path) -> CanonicalGlb:
    """Check the GLB at `source` and write its canonical form to `target`."""
    try:
        return _canonicalize(source, target)
    except _MALFORMED_ERRORS as exc:
        # Whatever a hand-made file gets wrong inside its JSON ends here, never as a server error.
        raise UnsupportedMediaTypeError(_MALFORMED_REASON) from exc


def _canonicalize(source: Path, target: Path) -> CanonicalGlb:
    with source.open("rb") as handle:
        container = _read_container(handle, source.stat().st_size)
        doc = container.doc

        _check_extensions(doc)
        _check_buffers(doc, container)
        _check_meshes(doc)
        bounds = _scene_bounds(doc)

        warnings = _remove_external_images(doc)
        replacements, uri_rewritten = _scale_images(doc, handle, container)

        changed = bool(warnings or replacements or uri_rewritten)
        if changed:
            _write_glb(target, doc, handle, container, replacements)

    if not changed:
        shutil.copyfile(source, target)

    return CanonicalGlb(
        sha256=_sha256_file(target),
        size_bytes=target.stat().st_size,
        bounds=bounds,
        warnings=warnings,
        changed=changed,
    )


def apply_unit_scale(source: Path, target: Path, factor: float) -> None:
    """
    §2l: one new node with a uniform scale becomes the default scene's only root, over its old roots.
    No vertex is rewritten and other scenes stay as they are. A factor of 1 is a plain copy.
    """
    if factor == 1:
        shutil.copyfile(source, target)
        return

    try:
        with source.open("rb") as handle:
            container = _read_container(handle, source.stat().st_size)
            _add_scale_root(container.doc, factor)
            _write_glb(target, container.doc, handle, container, {})
    except _MALFORMED_ERRORS as exc:
        raise UnsupportedMediaTypeError(_MALFORMED_REASON) from exc


def replace_invalid_text(source: Path, target: Path) -> bool:
    """
    §2l, for a converted file only: Assimp copies a name that isn't UTF-8 into the JSON as raw bytes.
    Those become U+FFFD in a new file at `target`. If there are none, nothing is written and this is False.
    """
    try:
        with source.open("rb") as handle:
            file_size = source.stat().st_size
            json_bytes = _read_json_chunk(handle, file_size)
            try:
                json_bytes.decode("utf-8")
            except UnicodeDecodeError:
                container = _parse_container(json_bytes.decode("utf-8", errors="replace"), handle, file_size)
                _write_glb(target, container.doc, handle, container, {})
                return True
            return False
    except _MALFORMED_ERRORS as exc:
        raise UnsupportedMediaTypeError(_MALFORMED_REASON) from exc


def _add_scale_root(doc: dict[str, Any], factor: float) -> None:
    scene = _item(doc, "scenes", doc.get("scene", 0))
    roots = scene.get("nodes")
    if not roots:
        _refuse("This GLB's default scene shows no geometry.")

    nodes = doc["nodes"]
    nodes.append({"name": "UnitScale", "scale": [factor, factor, factor], "children": roots})
    scene["nodes"] = [len(nodes) - 1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


# ── The container ────────────────────────────────────────────────────────────────────────────────


def _reject_json_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


def _read_chunk_header(handle: BinaryIO) -> tuple[int, int]:
    head = handle.read(8)
    if len(head) < 8:
        _refuse("This GLB is incomplete: a chunk header is cut off.")
    return struct.unpack("<II", head)


def _read_json_chunk(handle: BinaryIO, file_size: int) -> bytes:
    header = handle.read(12)
    if len(header) < 12 or header[:4] != _GLB_MAGIC:
        _refuse("This file isn't a GLB: its 'glTF' header is missing.")

    version, declared_length = struct.unpack("<II", header[4:])
    if version != 2:
        _refuse(f"This GLB is version {version}; only version 2 can be used.")
    if declared_length != file_size:
        _refuse(
            f"This GLB declares {declared_length} bytes but the file has {file_size}, "
            "so it's incomplete or damaged."
        )

    json_length, json_type = _read_chunk_header(handle)
    position = 20 + json_length
    if json_type != _CHUNK_JSON or position > file_size:
        _refuse("This GLB has no complete JSON chunk.")
    return handle.read(json_length)


def _parse_container(json_text: str, handle: BinaryIO, file_size: int) -> _Container:
    """The document and where its binary chunk is. `handle` stands just after the JSON chunk."""
    position = handle.tell()
    doc = json.loads(json_text, parse_constant=_reject_json_constant)
    if not isinstance(doc, dict):
        _refuse("This GLB's JSON chunk isn't a glTF document.")

    bin_start: int | None = None
    bin_length = 0
    if position < file_size:
        chunk_length, chunk_type = _read_chunk_header(handle)
        if chunk_type == _CHUNK_BIN:
            if position + 8 + chunk_length > file_size:
                _refuse("This GLB is incomplete: its binary chunk is cut off.")
            bin_start = position + 8
            bin_length = chunk_length

    return _Container(doc=doc, bin_start=bin_start, bin_length=bin_length)


def _read_container(handle: BinaryIO, file_size: int) -> _Container:
    return _parse_container(_read_json_chunk(handle, file_size).decode("utf-8"), handle, file_size)


def _item(doc: dict[str, Any], key: str, index: Any) -> Any:
    """A top-level glTF array's entry. A negative index would silently find the wrong one in Python."""
    items = doc.get(key)
    if (
        not isinstance(items, list)
        or not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(items)
    ):
        raise IndexError(f"{key}[{index}]")
    return items[index]


def _is_embedded(uri: str) -> bool:
    return uri.lower().startswith("data:")


def _decode_data_uri(uri: str) -> bytes:
    header, payload = uri.split(",", 1)
    if header.lower().endswith(";base64"):
        return base64.b64decode(payload, validate=True)
    return unquote_to_bytes(payload)


# ── The refusals ─────────────────────────────────────────────────────────────────────────────────


def _check_extensions(doc: dict[str, Any]) -> None:
    for name in [*(doc.get("extensionsRequired") or []), *(doc.get("extensionsUsed") or [])]:
        if name in _COMPRESSION_EXTENSIONS:
            _refuse(
                f"This GLB uses {_COMPRESSION_EXTENSIONS[name]} compression ({name}), "
                "which the headset can't read. Export it without compression."
            )

    for name in doc.get("extensionsRequired") or []:
        if name not in ALLOWED_REQUIRED_EXTENSIONS:
            _refuse(f"This GLB requires the extension {name}, which the headset can't read.")


def _check_buffers(doc: dict[str, Any], container: _Container) -> None:
    buffers = doc.get("buffers") or []
    for index, buffer in enumerate(buffers):
        uri = buffer.get("uri")
        if uri is None:
            if index != 0 or container.bin_start is None or buffer["byteLength"] > container.bin_length:
                _refuse(f"This GLB's buffer {index} has no matching binary data.")
        elif not _is_embedded(uri):
            _refuse(
                f"This GLB's buffer {index} points at a separate file, so it isn't self-contained. "
                "Export it with everything embedded."
            )

    for index, view in enumerate(doc.get("bufferViews") or []):
        buffer = _item(doc, "buffers", view["buffer"])
        offset = view.get("byteOffset", 0)
        if offset < 0 or view["byteLength"] < 0:
            raise ValueError(f"bufferViews[{index}]")
        if "uri" not in buffer and offset + view["byteLength"] > buffer["byteLength"]:
            _refuse(f"This GLB's buffer view {index} runs past the end of its data.")


def _position_box(doc: dict[str, Any], accessor_index: Any) -> tuple[list[float], list[float]] | None:
    """The accessor's corners in model units, or None if it doesn't declare them."""
    accessor = _item(doc, "accessors", accessor_index)
    low, high = accessor.get("min"), accessor.get("max")
    for values in (low, high):
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        ):
            return None

    divisor = _NORMALIZED_DIVISORS.get(accessor.get("componentType")) if accessor.get("normalized") else None
    if divisor is None:
        return [float(v) for v in low], [float(v) for v in high]
    return [max(v / divisor, -1.0) for v in low], [max(v / divisor, -1.0) for v in high]


def _check_meshes(doc: dict[str, Any]) -> None:
    meshes = doc.get("meshes")
    if not meshes:
        _refuse("This GLB contains no mesh.")

    for mesh_index, mesh in enumerate(meshes):
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            if "POSITION" in attributes and _position_box(doc, attributes["POSITION"]) is None:
                _refuse(
                    f"Mesh {mesh_index} of this GLB has no min and max on its POSITION data, "
                    "so its size can't be worked out."
                )


# ── Bounds ───────────────────────────────────────────────────────────────────────────────────────


def _node_matrix(node: dict[str, Any]) -> list[list[float]]:
    if "matrix" in node:
        m = node["matrix"]
        if len(m) != 16:
            raise ValueError("matrix")
        # glTF stores columns.
        return [[float(m[column * 4 + row]) for column in range(4)] for row in range(4)]

    tx, ty, tz = node.get("translation", (0.0, 0.0, 0.0))
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    sx, sy, sz = node.get("scale", (1.0, 1.0, 1.0))
    return [
        [(1 - 2 * (y * y + z * z)) * sx, 2 * (x * y - z * w) * sy, 2 * (x * z + y * w) * sz, float(tx)],
        [2 * (x * y + z * w) * sx, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z - x * w) * sz, float(ty)],
        [2 * (x * z - y * w) * sx, 2 * (y * z + x * w) * sy, (1 - 2 * (x * x + y * y)) * sz, float(tz)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][column] for k in range(4)) for column in range(4)] for row in range(4)]


def _scene_bounds(doc: dict[str, Any]) -> dict[str, float]:
    """The extents of the whole default scene, in metres: each mesh's box corners, through its nodes."""
    scene = _item(doc, "scenes", doc.get("scene", 0))
    low = [math.inf] * 3
    high = [-math.inf] * 3
    seen: set[int] = set()

    def visit(node_index: Any, parent: list[list[float]]) -> None:
        node = _item(doc, "nodes", node_index)
        if node_index in seen:
            _refuse("This GLB's node tree isn't a tree: a node appears twice.")
        seen.add(node_index)

        world = _multiply(parent, _node_matrix(node))
        if "mesh" in node:
            for primitive in _item(doc, "meshes", node["mesh"])["primitives"]:
                if "POSITION" not in primitive["attributes"]:
                    continue
                box_low, box_high = _position_box(doc, primitive["attributes"]["POSITION"])
                for corner_x in (box_low[0], box_high[0]):
                    for corner_y in (box_low[1], box_high[1]):
                        for corner_z in (box_low[2], box_high[2]):
                            for axis in range(3):
                                value = (
                                    world[axis][0] * corner_x
                                    + world[axis][1] * corner_y
                                    + world[axis][2] * corner_z
                                    + world[axis][3]
                                )
                                low[axis] = min(low[axis], value)
                                high[axis] = max(high[axis], value)

        for child in node.get("children") or []:
            visit(child, world)

    for root in scene.get("nodes") or []:
        visit(root, _IDENTITY)

    if low[0] > high[0]:
        _refuse("This GLB's default scene shows no geometry.")

    extents = [round(high[axis] - low[axis], 6) for axis in range(3)]
    if not all(math.isfinite(value) for value in extents):
        raise ValueError("bounds")
    return {"x": extents[0], "y": extents[1], "z": extents[2]}


# ── Images ───────────────────────────────────────────────────────────────────────────────────────


def _texture_sources(texture: dict[str, Any]) -> list[dict[str, Any]]:
    """Every dict that names an image for this texture: itself, and EXT_texture_webp and the like."""
    holders = [texture] if "source" in texture else []
    for extension in (texture.get("extensions") or {}).values():
        if isinstance(extension, dict) and "source" in extension:
            holders.append(extension)
    return holders


def _remap_texture_slots(value: Any, texture_map: dict[int, int]) -> None:
    """Slots also live inside extensions such as KHR_materials_specular, so this walks the lot."""
    if isinstance(value, dict):
        for key in list(value):
            if key == "extras":
                continue
            child = value[key]
            if isinstance(child, dict) and isinstance(child.get("index"), int):
                if child["index"] in texture_map:
                    child["index"] = texture_map[child["index"]]
                else:
                    del value[key]
            else:
                _remap_texture_slots(child, texture_map)
    elif isinstance(value, list):
        for item in value:
            _remap_texture_slots(item, texture_map)


def _remove_external_images(doc: dict[str, Any]) -> list[str]:
    """§2k: Assimp writes paths from someone else's PC. Each goes, with its texture and every slot."""
    images = doc.get("images") or []
    removed = {
        index for index, image in enumerate(images) if "uri" in image and not _is_embedded(image["uri"])
    }
    if not removed:
        return []

    warnings = [
        "Removed the external image '{}' and its texture slots: it isn't in the file.".format(
            re.split(r"[\\/]", images[index]["uri"])[-1]
        )
        for index in sorted(removed)
    ]

    image_map: dict[int, int] = {}
    kept_images = []
    for index, image in enumerate(images):
        if index not in removed:
            image_map[index] = len(kept_images)
            kept_images.append(image)

    texture_map: dict[int, int] = {}
    kept_textures = []
    for index, texture in enumerate(doc.get("textures") or []):
        holders = _texture_sources(texture)
        if any(holder["source"] in removed for holder in holders):
            continue
        for holder in holders:
            holder["source"] = image_map[holder["source"]]
        texture_map[index] = len(kept_textures)
        kept_textures.append(texture)

    # An empty array isn't valid glTF.
    for key, kept in (("images", kept_images), ("textures", kept_textures)):
        if kept:
            doc[key] = kept
        else:
            doc.pop(key, None)

    for material in doc.get("materials") or []:
        _remap_texture_slots(material, texture_map)

    return warnings


def _image_bytes(doc: dict[str, Any], image: dict[str, Any], handle: BinaryIO, container: _Container) -> bytes:
    if "uri" in image:
        return _decode_data_uri(image["uri"])

    view = _item(doc, "bufferViews", image["bufferView"])
    buffer = _item(doc, "buffers", view["buffer"])
    offset = view.get("byteOffset", 0)
    if "uri" in buffer:
        return _decode_data_uri(buffer["uri"])[offset : offset + view["byteLength"]]
    handle.seek(container.bin_start + offset)
    return handle.read(view["byteLength"])


def _scaled_image(data: bytes, image_index: int) -> tuple[bytes, str] | None:
    """The image at the size limit as (bytes, mime type), or None if it's within it."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            if width * height > settings.MEDIA_MAX_IMAGE_PIXELS:
                _refuse(
                    f"Image {image_index} of this GLB has {width}x{height} pixels, more than the "
                    f"{settings.MEDIA_MAX_IMAGE_PIXELS} allowed."
                )
            longest = max(width, height)
            if longest <= MAX_TEXTURE_SIDE:
                return None

            size = (
                max(1, round(width * MAX_TEXTURE_SIDE / longest)),
                max(1, round(height * MAX_TEXTURE_SIDE / longest)),
            )
            has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
            resized = img.convert("RGBA" if has_alpha else "RGB").resize(size, Image.LANCZOS)

            out = io.BytesIO()
            if has_alpha:
                resized.save(out, format="PNG")
                return out.getvalue(), "image/png"
            resized.save(out, format="JPEG", quality=_JPEG_QUALITY)
            return out.getvalue(), "image/jpeg"
    except UnsupportedMediaTypeError:
        raise
    except (OSError, ValueError, SyntaxError, EOFError, struct.error, IndexError, Image.DecompressionBombError) as exc:
        # What Pillow raises for bad data. A server fault such as MemoryError isn't the file's doing.
        raise UnsupportedMediaTypeError(f"Image {image_index} of this GLB can't be read.") from exc


def _scale_images(
    doc: dict[str, Any], handle: BinaryIO, container: _Container
) -> tuple[dict[int, bytes], bool]:
    """Scale what's over the limit. Returns new bytes by bufferView, and whether any URI changed."""
    replacements: dict[int, bytes] = {}
    uri_rewritten = False

    for index, image in enumerate(doc.get("images") or []):
        scaled = _scaled_image(_image_bytes(doc, image, handle, container), index)
        if scaled is None:
            continue
        data, mime_type = scaled
        image["mimeType"] = mime_type

        if "uri" not in image:
            view = _item(doc, "bufferViews", image["bufferView"])
            if "uri" not in _item(doc, "buffers", view["buffer"]):
                replacements[image["bufferView"]] = data
                continue
            # Only buffer 0 is rebuilt, so an image in an embedded buffer moves into its own URI.
            del image["bufferView"]

        image["uri"] = f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
        uri_rewritten = True

    return replacements, uri_rewritten


# ── Writing ──────────────────────────────────────────────────────────────────────────────────────


def _lay_out_binary_chunk(
    doc: dict[str, Any], container: _Container, replacements: dict[int, bytes]
) -> tuple[list[tuple[int, int, bytes | None, int]], int]:
    """
    Where each piece of the binary chunk goes, as (new offset, old offset, new bytes, length), and
    the chunk's data length. Without replacements the chunk stays exactly as it was.
    """
    if not replacements:
        if container.bin_start is None:
            return [], 0
        return [(0, 0, None, container.bin_length)], container.bin_length

    pieces: list[tuple[int, int, bytes | None, int]] = []
    cursor = 0
    for index, view in enumerate(doc.get("bufferViews") or []):
        if "uri" in _item(doc, "buffers", view["buffer"]):
            continue

        old_offset = view.get("byteOffset", 0)
        data = replacements.get(index)
        length = len(data) if data is not None else view["byteLength"]
        # A view that wasn't on a 4-byte boundary keeps its remainder: an accessor inside it may
        # rely on that to land on its own component size. All others stay 4-byte aligned.
        remainder = 0 if data is not None else old_offset % 4
        new_offset = (cursor + 3) // 4 * 4 + remainder

        pieces.append((new_offset, old_offset, data, length))
        view["byteOffset"] = new_offset
        view["byteLength"] = length
        cursor = new_offset + length

    doc["buffers"][0]["byteLength"] = cursor
    return pieces, cursor


def _copy_range(handle: BinaryIO, start: int, length: int, out: BinaryIO) -> None:
    handle.seek(start)
    remaining = length
    while remaining > 0:
        chunk = handle.read(min(remaining, _COPY_CHUNK_BYTES))
        if not chunk:
            raise ValueError("binary chunk ended early")
        out.write(chunk)
        remaining -= len(chunk)


def _write_glb(
    target: Path,
    doc: dict[str, Any],
    handle: BinaryIO,
    container: _Container,
    replacements: dict[int, bytes],
) -> None:
    pieces, bin_data_length = _lay_out_binary_chunk(doc, container, replacements)

    json_bytes = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)

    has_bin = container.bin_start is not None
    bin_chunk_length = bin_data_length + (-bin_data_length % 4)
    total_length = 12 + 8 + len(json_bytes) + (8 + bin_chunk_length if has_bin else 0)

    with target.open("wb") as out:
        out.write(struct.pack("<4sII", _GLB_MAGIC, 2, total_length))
        out.write(struct.pack("<II", len(json_bytes), _CHUNK_JSON))
        out.write(json_bytes)

        if has_bin:
            out.write(struct.pack("<II", bin_chunk_length, _CHUNK_BIN))
            written = 0
            for new_offset, old_offset, data, length in pieces:
                out.write(b"\x00" * (new_offset - written))
                if data is not None:
                    out.write(data)
                else:
                    _copy_range(handle, container.bin_start + old_offset, length, out)
                written = new_offset + length
            out.write(b"\x00" * (bin_chunk_length - written))
