import copy
import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensures that the parent directory of the given file path exists.
    Returns the normalized Path object.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def now_iso() -> str:
    """
    Returns a UTC timestamp in ISO 8601 format: `YYYY-MM-DDTHH:MM:SS.mmmZ`.

    The millisecond precision and the `Z` suffix are deliberate: this is exactly the
    format Unity writes (`SaveDataUtility.cs`, `"yyyy-MM-ddTHH:mm:ss.fffZ"`), which is
    also JavaScript's native `Date.toISOString()` format. Do NOT "simplify" this back to
    a bare `datetime.isoformat()` — that emits 6-digit microseconds and `+00:00`, which
    makes backend-written timestamps textually differ from client-written ones for the
    same instant.
    """
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_legacy_id() -> str:
    """
    Returns a 24-character hex string.

    This intentionally resembles a Mongo-style ObjectId string length,
    without depending on MongoDB or bson.
    """
    return secrets.token_hex(12)


def read_json_file(path: str | Path, default: Any = None) -> Any:
    """
    Reads JSON data from a file.
    Returns a deep copy of `default` if the file does not exist.

    Raises:
        ValueError: if the file exists but contains invalid JSON.
    """
    target_path = Path(path)

    if not target_path.exists():
        return copy.deepcopy(default)

    try:
        with target_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file '{target_path}': {e}") from e


def write_json_file(path: str | Path, data: Any) -> Path:
    """
    Writes JSON data directly to a file.
    Ensures the parent directory exists first.

    Note:
        This is a simple direct write. For safer production writes,
        prefer `atomic_write_json`.
    """
    target_path = ensure_parent_dir(path)

    with target_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return target_path


def atomic_write_json(path: str | Path, data: Any) -> Path:
    """
    Writes JSON data atomically by writing to a temporary file
    in the same directory and then replacing the target file.

    This reduces the risk of partial/corrupted writes.
    """
    target_path = ensure_parent_dir(path)
    target_dir = target_path.parent

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_dir,
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        json.dump(data, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.write("\n")
        temp_path = Path(tmp_file.name)

    temp_path.replace(target_path)
    return target_path


# ── C1 Slice 2 (§7a.2) — canonical hashing ───────────────────────────────────────────────────────
#
# The server is the ONLY authority for a guide's content hash (§7a intro, §1c "no client-computed
# revision number trusted by the server"). These helpers sit ALONGSIDE atomic_write_json — they do
# not replace it; every canonical guide commit still goes through atomic_write_json exactly once.

#: Transport/storage metadata that never participates in the authoring hash (§7a.1).
#: `translations` and `unityGuidePath` are excluded separately as *derived* content: they are
#: retained and returned, but a translation must not advance the authoring revision (F7).
HASH_EXCLUDED_TOP_LEVEL_KEYS = frozenset(
    {
        "_id",
        "databaseId",
        "version",
        "baseVersion",
        "contentHash",
        "createdAt",
        "updatedAt",
        "translations",
        "unityGuidePath",
    }
)


def canonical_json_bytes(data: Any) -> bytes:
    """
    UTF-8 canonical JSON: object keys recursively sorted, array order preserved,
    stable separators, no ASCII escaping (§7a.2).

    This is a pure serialization of whatever it is handed. It does no field filtering —
    `authoring_content_hash` decides what is meaningful.
    """
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def authoring_projection(doc: dict[str, Any]) -> dict[str, Any]:
    """
    The meaningful authoring content of a guide document (§7a.1).

    Everything the raw validated document carries — including unknown extensions the backend has
    no schema for — is kept, EXCEPT transport/storage metadata and the server-derived subtrees.
    This deliberately operates on the raw dict, never on a narrow DTO reconstruction, so an
    unknown field added by a newer client still changes the hash.
    """
    return {
        key: value
        for key, value in (doc or {}).items()
        if key not in HASH_EXCLUDED_TOP_LEVEL_KEYS
    }


def authoring_content_hash(doc: dict[str, Any]) -> str:
    """
    SHA-256 over the canonical JSON of the authoring projection. Hex digest.
    """
    return hashlib.sha256(canonical_json_bytes(authoring_projection(doc))).hexdigest()
