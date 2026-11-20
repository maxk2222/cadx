"""C2 §2l Slice 7: a non-GLB model upload becomes a GLB in metres, through the pinned Assimp image.

Sync functions, run through the store's threadpool path and never inside its lock. The container sees
one folder, `work_dir`, as `/work`; it holds a copy of the upload under a fixed name. A refusal is an
`UnsupportedMediaTypeError` whose text is for the person who uploaded the file and holds no server path.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import struct
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn
from uuid import uuid4

from kiara.guides.glb_tools import apply_unit_scale, replace_invalid_text
from kiara.guides.media_validation import validate_media_content
from kiara.guides.paths import media_cap_bytes
from kiara.guides.paths_cfg import settings
from kiara.guides.upload_errors import (
    ConverterUnavailableError,
    InvalidUploadError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)

log = logging.getLogger(__name__)

# STL and OBJ carry no units, so these keep today's assumptions (`poseTransform.ts`).
_FIXED_UNIT_FACTORS = {".stl": 0.001, ".obj": 1.0}
# FBX's UnitScaleFactor counts centimetres.
_FBX_CENTIMETRES_TO_METRES = 0.01

_DOCKER_CHECK_TIMEOUT_SECONDS = 15
_DOCKER_REMOVE_TIMEOUT_SECONDS = 30
# Docker's own failure, then a command in the image that can't be run or isn't there.
_CONVERTER_NOT_STARTED_EXITS = (125, 126, 127)
_OUT_OF_MEMORY_EXIT = 137
_MAX_REASON_CHARS = 300
# Assimp can print an empty `ERROR:` line, which isn't a reason.
_ASSIMP_ERROR_RE = re.compile(r"ERROR:[ \t]*([^\s][^\r\n]*)")

_FBX_BINARY_PREFIX = b"Kaydara FBX Binary"
_FBX_BINARY_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_FBX_NODES_OFFSET = 27
_FBX_WIDE_HEADER_VERSION = 7500
_FBX_NARROW_NODE = struct.Struct("<IIIB")
_FBX_WIDE_NODE = struct.Struct("<QQQB")
_FBX_UNIT_FIELD = b"UnitScaleFactor"
_FBX_UNIT_VALUE_INDEX = 4
_FBX_SCALAR_SIZES = {b"Y": 2, b"C": 1, b"I": 4, b"F": 4, b"D": 8, b"L": 8}
_FBX_NUMBER_FORMATS = {b"D": "<d", b"F": "<f", b"I": "<i", b"L": "<q"}
_FBX_BLOB_TYPES = (b"S", b"R")
_FBX_ARRAY_TYPES = (b"f", b"d", b"l", b"i", b"b", b"c")

_FBX_ASCII_SETTINGS_RE = re.compile(rb"^\s*GlobalSettings:")
# The opening quote keeps `"OriginalUnitScaleFactor"` out. The value is the line's last field.
_FBX_ASCII_UNIT_RE = re.compile(rb'^\s*P:\s*"UnitScaleFactor"\s*,.*,\s*([^,\s]+)\s*$')
_FBX_ASCII_NUMBER_RE = re.compile(rb"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def convert_upload_to_glb(source: Path, ext: str, work_dir: Path) -> Path:
    """The upload at `source` as a GLB in metres, written to `work_dir`. `ext` is in lower case."""
    try:
        validate_media_content(source, "cad_models", ext)
    except InvalidUploadError as exc:
        raise UnsupportedMediaTypeError(str(exc)) from exc

    # Before anything to do with Docker: a file refused here never starts a container.
    factor = _unit_factor(source, ext)
    _require_converter()

    shutil.copyfile(source, work_dir / f"input{ext}")
    converted = _run_assimp(ext, work_dir)

    cap = media_cap_bytes("cad_models")
    if converted.stat().st_size > cap:
        raise PayloadTooLargeError(f"This model is over the {cap}-byte limit for models once converted.")

    readable = work_dir / "readable.glb"
    if replace_invalid_text(converted, readable):
        log.info("Model conversion: part names that weren't UTF-8 had their invalid bytes replaced.")
        converted = readable

    scaled = work_dir / "scaled.glb"
    apply_unit_scale(converted, scaled, factor)
    return scaled


def _unit_factor(source: Path, ext: str) -> float:
    if ext in _FIXED_UNIT_FACTORS:
        return _FIXED_UNIT_FACTORS[ext]
    return fbx_unit_factor(source)


# ── FBX units ────────────────────────────────────────────────────────────────────────────────────


def fbx_unit_factor(path: Path) -> float:
    """Metres per FBX unit, read from the file. Refused rather than guessed: that's what the store removes."""
    with path.open("rb") as handle:
        is_binary = handle.read(len(_FBX_BINARY_PREFIX)) == _FBX_BINARY_PREFIX
    value = _read_binary_unit_scale(path) if is_binary else _read_ascii_unit_scale(path)

    factor = value * _FBX_CENTIMETRES_TO_METRES
    if not math.isfinite(factor) or factor <= 0:
        _refuse_unit(f"This FBX's UnitScaleFactor is {value}, which can't be used as a size.")
    return factor


def _refuse_unit(reason: str) -> NoReturn:
    raise UnsupportedMediaTypeError(reason)


def _refuse_missing_unit() -> NoReturn:
    _refuse_unit("This FBX has no UnitScaleFactor in its GlobalSettings, so its size can't be worked out.")


def _refuse_damaged_fbx() -> NoReturn:
    _refuse_unit("This FBX file is damaged: its structure doesn't fit inside the file.")


@dataclass(frozen=True)
class _FbxNode:
    name: bytes
    property_count: int
    properties_start: int
    children_start: int
    end: int


def _fbx_read(handle: BinaryIO, position: int, length: int, limit: int) -> bytes:
    if position + length > limit:
        _refuse_damaged_fbx()
    handle.seek(position)
    data = handle.read(length)
    if len(data) != length:
        _refuse_damaged_fbx()
    return data


def _fbx_nodes(handle: BinaryIO, start: int, limit: int, wide: bool) -> Iterator[_FbxNode]:
    """The node records from `start` until a null record. Every step moves forward, so it can't loop."""
    header = _FBX_WIDE_NODE if wide else _FBX_NARROW_NODE
    position = start
    while position < limit:
        end, property_count, property_bytes, name_length = header.unpack(
            _fbx_read(handle, position, header.size, limit)
        )
        if end == 0 and property_count == 0 and property_bytes == 0 and name_length == 0:
            return

        properties_start = position + header.size + name_length
        children_start = properties_start + property_bytes
        if end <= position or end > limit or children_start > end:
            _refuse_damaged_fbx()

        name = _fbx_read(handle, position + header.size, name_length, limit)
        yield _FbxNode(name, property_count, properties_start, children_start, end)
        position = end


def _fbx_child(handle: BinaryIO, start: int, limit: int, wide: bool, name: bytes) -> _FbxNode:
    for node in _fbx_nodes(handle, start, limit, wide):
        if node.name == name:
            return node
    _refuse_missing_unit()


def _fbx_property(handle: BinaryIO, position: int, limit: int) -> tuple[bytes, int, int]:
    """One property's type code, where its data starts and where the next property starts."""
    code = _fbx_read(handle, position, 1, limit)
    data_start = position + 1

    if code in _FBX_SCALAR_SIZES:
        length = _FBX_SCALAR_SIZES[code]
    elif code in _FBX_BLOB_TYPES:
        (length,) = struct.unpack("<I", _fbx_read(handle, data_start, 4, limit))
        data_start += 4
    elif code in _FBX_ARRAY_TYPES:
        # Element count, encoding, then the stored size, whichever the encoding.
        _count, _encoding, length = struct.unpack("<III", _fbx_read(handle, data_start, 12, limit))
        data_start += 12
    else:
        _refuse_damaged_fbx()

    if data_start + length > limit:
        _refuse_damaged_fbx()
    return code, data_start, data_start + length


def _fbx_unit_value(handle: BinaryIO, node: _FbxNode) -> float | None:
    """The value if this `P` is the UnitScaleFactor, otherwise None."""
    limit = node.children_start
    if node.property_count == 0:
        return None

    code, data_start, position = _fbx_property(handle, node.properties_start, limit)
    if (
        code != b"S"
        or position - data_start != len(_FBX_UNIT_FIELD)
        or _fbx_read(handle, data_start, len(_FBX_UNIT_FIELD), limit) != _FBX_UNIT_FIELD
    ):
        return None

    if node.property_count <= _FBX_UNIT_VALUE_INDEX:
        _refuse_missing_unit()
    for _index in range(1, _FBX_UNIT_VALUE_INDEX):
        _code, _data_start, position = _fbx_property(handle, position, limit)

    code, data_start, next_position = _fbx_property(handle, position, limit)
    if code not in _FBX_NUMBER_FORMATS:
        _refuse_unit("This FBX's UnitScaleFactor isn't a number.")
    (value,) = struct.unpack(_FBX_NUMBER_FORMATS[code], _fbx_read(handle, data_start, next_position - data_start, limit))
    return float(value)


def _read_binary_unit_scale(path: Path) -> float:
    """A walk of the node tree, not a byte search, so `OriginalUnitScaleFactor` can't match."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        head = _fbx_read(handle, 0, _FBX_NODES_OFFSET, file_size)
        if head[: len(_FBX_BINARY_MAGIC)] != _FBX_BINARY_MAGIC:
            _refuse_damaged_fbx()
        (version,) = struct.unpack("<I", head[len(_FBX_BINARY_MAGIC) :])
        wide = version >= _FBX_WIDE_HEADER_VERSION

        settings_node = _fbx_child(handle, _FBX_NODES_OFFSET, file_size, wide, b"GlobalSettings")
        properties = _fbx_child(handle, settings_node.children_start, settings_node.end, wide, b"Properties70")
        for node in _fbx_nodes(handle, properties.children_start, properties.end, wide):
            if node.name == b"P":
                value = _fbx_unit_value(handle, node)
                if value is not None:
                    return value
    _refuse_missing_unit()


def _read_ascii_unit_scale(path: Path) -> float:
    inside_settings = False
    depth = 0
    with path.open("rb") as handle:
        for line in handle:
            if not inside_settings:
                if not _FBX_ASCII_SETTINGS_RE.match(line):
                    continue
                inside_settings = True

            match = _FBX_ASCII_UNIT_RE.match(line)
            if match:
                if not _FBX_ASCII_NUMBER_RE.fullmatch(match.group(1)):
                    _refuse_unit("This FBX's UnitScaleFactor isn't a number.")
                return float(match.group(1))

            depth += line.count(b"{") - line.count(b"}")
            if depth <= 0 and b"}" in line:
                break
    _refuse_missing_unit()


# ── Docker ───────────────────────────────────────────────────────────────────────────────────────


def _docker(arguments: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *arguments],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _no_docker_error() -> ConverterUnavailableError:
    return ConverterUnavailableError(
        "Docker isn't installed on the server, so this file can't be converted. .glb files still upload."
    )


def _converter_not_started_error() -> ConverterUnavailableError:
    return ConverterUnavailableError(
        "Docker couldn't start the converter on the server, so this file can't be converted. "
        ".glb files still upload."
    )


def _require_converter() -> None:
    image = settings.ASSIMP_DOCKER_IMAGE
    try:
        result = _docker(["image", "inspect", image], _DOCKER_CHECK_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise _no_docker_error() from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning(f"Model conversion: 'docker image inspect' failed: {exc}")
        raise ConverterUnavailableError(
            "Docker isn't answering on the server, so this file can't be converted. .glb files still upload."
        ) from exc

    if result.returncode == 0:
        return

    log.warning(f"Model conversion: 'docker image inspect {image}' exit {result.returncode}: {result.stderr.strip()}")
    if "no such image" in result.stderr.lower():
        raise ConverterUnavailableError(
            f"The converter image {image} isn't built on the server, so this file can't be converted. "
            f".glb files still upload. Build it once with: docker build -t {image} backend/tools/assimp"
        )
    raise ConverterUnavailableError(
        "Docker isn't running on the server, so this file can't be converted. "
        "Start Docker and try again; .glb files still upload."
    )


def _converter_command(ext: str) -> list[str]:
    return ["assimp", "export", f"/work/input{ext}", "/work/output.glb"]


def _plain_path(path: Path) -> str:
    """Docker can't take the long-path prefix `resolve_safe_path` adds on Windows."""
    raw = str(path)
    if raw.startswith("\\\\?\\UNC\\"):
        return "\\\\" + raw[len("\\\\?\\UNC\\") :]
    if raw.startswith("\\\\?\\"):
        return raw[len("\\\\?\\") :]
    return raw


def _run_arguments(container_name: str, ext: str, work_dir: Path) -> list[str]:
    arguments = [
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--network",
        "none",
        "--memory",
        "2g",
        "--cpus",
        "2",
        "-v",
        f"{_plain_path(work_dir)}:/work",
    ]
    # So the files can be removed afterwards. Windows has neither call, and doesn't need them.
    if hasattr(os, "getuid"):
        arguments += ["--user", f"{os.getuid()}:{os.getgid()}"]
    return [*arguments, settings.ASSIMP_DOCKER_IMAGE, *_converter_command(ext)]


def _force_remove_container(container_name: str) -> None:
    try:
        _docker(["rm", "-f", container_name], _DOCKER_REMOVE_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning(f"Model conversion: could not remove container '{container_name}': {exc}")


def _assimp_failure_reason(result: subprocess.CompletedProcess[str]) -> str:
    # Assimp prints the error straight after its progress counter, so it isn't at the start of a line.
    match = _ASSIMP_ERROR_RE.search(f"{result.stderr}\n{result.stdout}")
    if not match:
        return "This file couldn't be converted."
    detail = match.group(1).strip().replace("/work/", "")[:_MAX_REASON_CHARS]
    return f"This file couldn't be converted: {detail}"


def _run_assimp(ext: str, work_dir: Path) -> Path:
    container_name = f"kiara-assimp-{uuid4().hex}"
    limit = settings.ASSIMP_TIMEOUT_SECONDS
    arguments = _run_arguments(container_name, ext, work_dir)

    log.info(f"Model conversion: docker {' '.join(arguments)}")
    started = time.monotonic()
    try:
        result = _docker(arguments, limit)
    except FileNotFoundError as exc:
        raise _no_docker_error() from exc
    except OSError as exc:
        log.warning(f"Model conversion: could not start 'docker run': {exc}")
        raise _converter_not_started_error() from exc
    except subprocess.TimeoutExpired as exc:
        _force_remove_container(container_name)
        log.warning(f"Model conversion: '{container_name}' stopped after {limit} s.")
        raise UnsupportedMediaTypeError(f"Converting this file took longer than {limit} seconds.") from exc

    elapsed = time.monotonic() - started
    log.info(f"Model conversion: '{container_name}' exit {result.returncode} after {elapsed:.1f} s.")

    if result.returncode in _CONVERTER_NOT_STARTED_EXITS:
        log.warning(f"Model conversion: Docker couldn't run '{container_name}': {result.stderr.strip()}")
        raise _converter_not_started_error()
    if result.returncode == _OUT_OF_MEMORY_EXIT:
        raise UnsupportedMediaTypeError("Converting this file used too much memory.")

    output = work_dir / "output.glb"
    if result.returncode != 0 or not output.is_file():
        log.warning(f"Model conversion: '{container_name}' failed: {result.stderr.strip()} {result.stdout.strip()}")
        raise UnsupportedMediaTypeError(_assimp_failure_reason(result))
    return output
