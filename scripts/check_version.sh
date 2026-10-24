#!/usr/bin/env bash
# Version consistency check: VERSION file ↔ package manifests ↔ git tag.
# Package manifests carry the BASE version (X.Y.Z); VERSION carries the full
# release string (X.Y.Z or X.Y.Z-prerelease). A tag must equal VERSION.
#
# Usage:
#   scripts/check_version.sh [--tag TAG] [--no-color] [-h|--help]
#     --tag TAG   expected tag (default: $GITHUB_REF_NAME or `git describe
#                 --tags --exact-match` when HEAD is tagged; skipped otherwise)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib/ui.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

TAG="${GITHUB_REF_NAME:-}"
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --tag)      TAG="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)          say err "unknown arg: $1"; usage; exit 2 ;;
  esac
done

VER="$(tr -d '[:space:]' < VERSION)"
BASE="${VER%%-*}"
step_begin "VERSION = $VER (base $BASE)"

fail=0
check() { # <label> <actual> <expected>
  if [ "$2" = "$3" ]; then
    say ok "$1 = $2"
  else
    say err "$1 = $2 (expected $3)"; fail=1
  fi
}

pyproject_version() { # <path>
  sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$1" | head -1
}

check "backend/pyproject.toml"    "$(pyproject_version backend/pyproject.toml)"    "$BASE"
check "mesh_server/pyproject.toml" "$(pyproject_version mesh_server/pyproject.toml)" "$BASE"
check "frontend/package.json"     "$(sed -n 's/^  "version": "\([^"]*\)".*/\1/p' frontend/package.json | head -1)" "$BASE"

# prerelease suffix sanity: only letters/digits/hyphen/dots
case "$VER" in
  *[^A-Za-z0-9.-]*) say err "VERSION has invalid characters"; fail=1 ;;
esac
case "$VER" in
  "$BASE") ;;  # no suffix — fine
  "$BASE"-*) ;;  # X.Y.Z-suffix — fine
  *) say err "VERSION '$VER' is not X.Y.Z[-suffix]"; fail=1 ;;
esac

if [ -z "$TAG" ]; then
  TAG="$(git describe --tags --exact-match 2>/dev/null || true)"
fi
if [ -n "$TAG" ]; then
  check "git tag" "${TAG#v}" "$VER"
else
  say info "no tag context — tag check skipped"
fi

if [ "$fail" -eq 0 ]; then step_done; else step_fail; exit 1; fi
