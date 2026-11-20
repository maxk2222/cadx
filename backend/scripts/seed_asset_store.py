"""C2 §2l Slice 8: creates or updates the built-in models in the store from a folder of Unity exports.

    python backend/scripts/seed_asset_store.py <folder>

`<folder>` holds what `BuiltInModelExporter` wrote: `manifest.json` and `<prefabName>.glb` for each model.
Uses the store under `DATA_DIR`, like the backend. Run it with the backend stopped: the store's lock
doesn't reach another process. Running it twice changes nothing.

A model that can't go in is reported and left alone, and the next one still runs. Models already done
stay. Anything wrong with the manifest stops the run before any model. Any failure exits with 1.
A model with an unreadable `asset.json` counts as not there: new folder, new `assetId`, the old folder stays.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiara.guides.asset_store import AssetConflictError, seed_built_in_asset  # noqa: E402
from kiara.guides.upload_errors import UploadError  # noqa: E402

MANIFEST_FILE = "manifest.json"
_TEXT_FIELDS = ("prefabName", "displayName", "category")


class ManifestError(Exception):
    """The manifest can't be used, so no model is touched."""


def read_manifest(folder: Path) -> list[dict[str, Any]]:
    path = folder / MANIFEST_FILE
    if not path.is_file():
        raise ManifestError(f"No {MANIFEST_FILE} in {folder}.")

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{MANIFEST_FILE} can't be read: {exc}") from exc

    models = manifest.get("models") if isinstance(manifest, dict) else None
    if not isinstance(models, list) or not models:
        raise ManifestError(f'{MANIFEST_FILE} has no "models" list with anything in it.')

    for index, model in enumerate(models):
        if not isinstance(model, dict) or "defaultScale" not in model:
            raise ManifestError(f"{MANIFEST_FILE}: model {index} has no defaultScale.")
        for field in _TEXT_FIELDS:
            if not isinstance(model.get(field), str):
                raise ManifestError(f"{MANIFEST_FILE}: model {index} has no {field}.")

    repeated = [name for name, count in Counter(model["prefabName"] for model in models).items() if count > 1]
    if repeated:
        raise ManifestError(f"{MANIFEST_FILE} lists {', '.join(repeated)} more than once.")
    return models


def describe_conflict(error: AssetConflictError) -> str:
    existing = error.detail["existing"]
    kind = "built-in" if existing.get("builtIn") else "uploaded"
    what = f"{kind} asset {existing['assetId']} '{existing['displayName']}'"
    if error.detail["reason"] == "name":
        return f"its display name is taken by {what}"
    return f"it has the same content as {what}"


def seed_model(folder: Path, model: dict[str, Any]) -> str:
    """Prints this model's line and returns its outcome, or `failed`."""
    prefab_name = model["prefabName"]
    try:
        record, outcome, warnings = seed_built_in_asset(
            folder / f"{prefab_name}.glb",
            prefab_name,
            model["displayName"],
            model["category"],
            model["defaultScale"],
        )
    except AssetConflictError as exc:
        print(f"failed    {prefab_name}: {describe_conflict(exc)}. Nothing changed.")
        return "failed"
    except (UploadError, OSError) as exc:
        print(f"failed    {prefab_name}: {exc} Nothing changed.")
        return "failed"

    print(f"{outcome:<9} {prefab_name}: {record['assetId']}, {record['sizeBytes']} bytes, {record['currentHash'][:12]}")
    for warning in warnings:
        print(f"          warning: {warning}")
    return outcome


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: seed_asset_store.py <folder>", file=sys.stderr)
        return 1

    folder = Path(argv[0])
    try:
        models = read_manifest(folder)
    except ManifestError as exc:
        print(f"Nothing was seeded. {exc}", file=sys.stderr)
        return 1

    outcomes = Counter(seed_model(folder, model) for model in models)
    print(
        f"{len(models)} models: {outcomes['created']} created, {outcomes['updated']} updated, "
        f"{outcomes['unchanged']} unchanged, {outcomes['failed']} failed."
    )
    return 1 if outcomes["failed"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
