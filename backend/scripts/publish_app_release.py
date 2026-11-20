"""S2-AP1 (F-040): puts a new headset build where the in-app updater finds it.

    python backend/scripts/publish_app_release.py --platform ml2 --apk <path> --version-name 0.23.0 --version-code 23

Copies the APK into DATA_DIR/app_releases/ as KIARA_<ML2|Quest>_<version>.apk, works out its size and SHA-256
and writes that platform's entry in latest.json, keeping the other platform's entry. The version code must be
higher than the published one (--force allows an equal or lower one, for tests). The APK must contain the native
libraries of its headset (lib/x86_64/ for ml2, lib/arm64-v8a/ for quest), so a Quest build can't be published
as ML2 by mistake. The version code is not read from the APK: the headset checks it before installing.

Any failure exits with 1 and leaves latest.json as it was. Republishing the same version name (with --force)
replaces that APK file before latest.json is written; if that last write fails, the script says so and a second
run puts both back in step.
Contract: planning/SPEC_S2AP1_implementation.md §2 "Server parts".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiara.guides import paths  # noqa: E402
from kiara.guides.persistence import atomic_write_json  # noqa: E402

LATEST_FILE_NAME = "latest.json"
PLATFORM_LABELS = {"ml2": "ML2", "quest": "Quest"}
PLATFORM_ABI = {"ml2": "lib/x86_64/", "quest": "lib/arm64-v8a/"}


class PublishError(Exception):
    """The release can't be published; nothing was changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_latest(latest_path: Path) -> dict[str, Any]:
    if not latest_path.exists():
        return {}
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublishError(f"{latest_path} can't be read: {exc}") from exc
    if not isinstance(latest, dict):
        raise PublishError(f"{latest_path} is not a JSON object.")
    return latest


def _check_abi(apk: Path, platform: str) -> None:
    try:
        with zipfile.ZipFile(apk) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise PublishError(f"{apk} is not a readable APK: {exc}") from exc
    wanted = PLATFORM_ABI[platform]
    if not any(name.startswith(wanted) for name in names):
        raise PublishError(f"{apk.name} has no {wanted} libraries, so it isn't a {PLATFORM_LABELS[platform]} build.")


def publish(platform: str, apk: Path, version_name: str, version_code: int, force: bool = False) -> dict[str, Any]:
    if platform not in PLATFORM_LABELS:
        raise PublishError(f"Unknown platform {platform!r}; use ml2 or quest.")
    if not version_name.strip():
        raise PublishError("--version-name is empty.")
    if version_code <= 0:
        raise PublishError("--version-code must be positive.")
    if not apk.is_file():
        raise PublishError(f"{apk} doesn't exist.")
    _check_abi(apk, platform)

    releases_dir = paths.APP_RELEASES_DIR
    releases_dir.mkdir(parents=True, exist_ok=True)
    latest_path = releases_dir / LATEST_FILE_NAME
    latest = _read_latest(latest_path)

    current = latest.get(platform)
    if (
        isinstance(current, dict)
        and isinstance(current.get("versionCode"), int)
        and not force
        and version_code <= current["versionCode"]
    ):
            raise PublishError(
                f"{platform} already has version code {current['versionCode']}; "
                f"{version_code} isn't higher (use --force for a test)."
            )

    final_name = f"KIARA_{PLATFORM_LABELS[platform]}_{version_name.strip()}.apk"
    if Path(final_name).name != final_name:
        raise PublishError(f"--version-name {version_name!r} makes no plain file name.")
    final_path = releases_dir / final_name
    # Not ending in .apk, so the download route can never serve a half-written copy.
    temp_path = releases_dir / f".{final_name}.part"

    try:
        shutil.copyfile(apk, temp_path)
        size = temp_path.stat().st_size
        sha256 = _sha256(temp_path)
        temp_path.replace(final_path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise PublishError(f"Copying {apk} failed: {exc}") from exc

    entry = {
        "versionName": version_name.strip(),
        "versionCode": version_code,
        "file": final_name,
        "size": size,
        "sha256": sha256,
    }
    latest[platform] = entry
    try:
        atomic_write_json(latest_path, latest)
    except OSError as exc:
        raise PublishError(
            f"Writing {latest_path} failed: {exc}. {final_name} is already in place; run the script again."
        ) from exc
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a KIARA headset build for the in-app updater.")
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORM_LABELS))
    parser.add_argument("--apk", required=True, type=Path)
    parser.add_argument("--version-name", required=True)
    parser.add_argument("--version-code", required=True, type=int)
    parser.add_argument("--force", action="store_true", help="allow a version code that isn't higher (tests only)")
    args = parser.parse_args(argv)

    try:
        entry = publish(args.platform, args.apk, args.version_name, args.version_code, args.force)
    except PublishError as exc:
        print(f"Not published: {exc}", file=sys.stderr)
        return 1

    print(
        f"Published {args.platform} {entry['versionName']} (code {entry['versionCode']}): "
        f"{entry['file']}, {entry['size']} bytes, sha256 {entry['sha256']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
