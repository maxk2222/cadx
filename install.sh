#!/usr/bin/env bash
# CADX installer — fetch a release tarball OR run in-tree, then setup + start.
#
# Usage:
#   curl -fsSL <release-asset>/install.sh | bash -s -- [--channel alpha|stable]
#   ./install.sh [flags]                       # inside a checkout/tarball
#
# Modes:
#   bootstrap   script is standalone → resolve tag, download tarball +
#               SHA256SUMS.txt, verify, extract into --dir, re-exec in-tree.
#   in-tree     script lives in the source tree → no fetch; setup + start.
#   --update    refresh an installed --dir to the channel's tag: downloads
#               the tarball, copies it over the install (preserving data/,
#               .env*, run/, logs/, venvs, node_modules, .git), backs up
#               .env, then re-runs setup + start.
#
# Flags:
#   --channel alpha|stable   release channel (default: stable; falls back to
#                            DEFAULT_TAG when nothing matches — the
#                            releases/latest endpoint ignores prereleases,
#                            hence the list API)
#   --tag vX.Y.Z             explicit release tag (bootstrap/--update)
#   --dir PATH               install dir (default: ./cadx; with --update the
#                            existing install to refresh)
#   --dev                    also set up native frontend dev (needs node ≥20)
#   --fake / --tier T / --skip-mesh / --skip-image   passed to setup.sh
#   --no-start               setup only — don't launch the stack
#   --dry-run                print the plan, change nothing
#   --repo owner/name        GitHub repo (default: maxk2222/cadx)
#   -y|--yes  -q|--quiet  -v|--verbose  --no-color  -h|--help
#
# Kept bash-3.2-compatible (macOS system bash). Requires: curl, tar,
# python3 ≥3.11, git, docker + compose plugin.
set -euo pipefail

REPO="maxk2222/cadx"
DEFAULT_TAG="v0.1.0-alpha.1"
DEFAULT_CHANNEL="stable"

# ---------- minimal output (bootstrap can't rely on scripts/lib yet) ----------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ -z "${CI:-}" ]; then
  C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_B=$'\033[36m'; C_0=$'\033[0m'
else
  C_R=""; C_G=""; C_Y=""; C_B=""; C_0=""
fi
QUIET=0; VERBOSE=0
info() { [ "$QUIET" = 1 ] || echo "${C_B}[install]${C_0} $*"; }
ok()   { [ "$QUIET" = 1 ] || echo "${C_G}[install]${C_0} $*"; }
warn() { echo "${C_Y}[install]${C_0} $*" >&2; }
err()  { echo "${C_R}[install]${C_0} $*" >&2; }
die()  { err "$*"; exit 1; }

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

# ---------- args ----------
CHANNEL="$DEFAULT_CHANNEL"; TAG=""; DIR=""; UPDATE=0; DEV=0; DRY=0
NO_START=0; FAKE=0; TIER=""; SKIP_MESH=0; SKIP_IMAGE=0; YES=0
# shellcheck disable=SC2015
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
IN_TREE=0
[ -f "$SELF_DIR/scripts/setup.sh" ] && [ -f "$SELF_DIR/docker-compose.yml" ] && IN_TREE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --channel)     CHANNEL="$2"; shift 2 ;;
    --tag)         TAG="$2"; shift 2 ;;
    --dir)         DIR="$2"; shift 2 ;;
    --update)      UPDATE=1; shift ;;
    --dev)         DEV=1; shift ;;
    --fake)        FAKE=1; shift ;;
    --tier)        TIER="$2"; shift 2 ;;
    --skip-mesh)   SKIP_MESH=1; shift ;;
    --skip-image)  SKIP_IMAGE=1; shift ;;
    --no-start)    NO_START=1; shift ;;
    --dry-run)     DRY=1; shift ;;
    --repo)        REPO="$2"; shift 2 ;;
    -y|--yes)      YES=1; shift ;;
    -q|--quiet)    QUIET=1; shift ;;
    -v|--verbose)  VERBOSE=1; shift ;;
    --no-color)    C_R=""; C_G=""; C_Y=""; C_B=""; C_0=""; export NO_COLOR=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done
case "$CHANNEL" in alpha|stable) ;; *) die "--channel must be alpha|stable" ;; esac

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }

preflight_tools() { # shared tool check — docker/git/python needed everywhere
  need_cmd curl; need_cmd tar
  need_cmd python3
  python3 -c "import sys;sys.exit(0 if sys.version_info[:2]>=(3,11) else 1)" \
    || die "python3 ≥3.11 required"
  need_cmd git; need_cmd docker
  docker compose version >/dev/null 2>&1 || die "docker compose plugin required"
  if [ "$DEV" = 1 ]; then
    need_cmd node
    node -e 'process.exit(process.versions.node.split(".")[0]>=20?0:1)' \
      || die "node ≥20 required for --dev"
  fi
}

resolve_tag() { # <channel> → release tag_name via list API (latest skips prereleases)
  local body; body="$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=10")" \
    || die "could not list releases for $REPO"
  local want_pre=False; [ "$CHANNEL" = "alpha" ] && want_pre=True
  local tag; tag="$(printf '%s' "$body" | python3 -c "
import json,sys
for r in json.load(sys.stdin):
    if r.get('draft'): continue
    if bool(r.get('prerelease'))==$want_pre:
        print(r['tag_name']); break
" 2>/dev/null || true)"
  if [ -z "$tag" ]; then
    warn "no $CHANNEL release found — falling back to $DEFAULT_TAG"
    tag="$DEFAULT_TAG"
  fi
  printf '%s' "$tag"
}

fetch_release() { # <tag> <destdir> — download + verify + extract into destdir
  local tag="$1" dest="$2" ver asset base tmp
  ver="${tag#v}"
  asset="cadx-$ver.tar.gz"
  base="https://github.com/$REPO/releases/download/$tag"
  tmp="$(mktemp -d)"
  info "downloading $asset ($tag)"
  curl -fSL --retry 3 -o "$tmp/$asset" "$base/$asset" || die "download failed: $asset"
  if curl -fsSL -o "$tmp/SHA256SUMS.txt" "$base/SHA256SUMS.txt" 2>/dev/null; then
    local exp got
    exp="$(awk '{print $1}' "$tmp/SHA256SUMS.txt" | head -1)"
    if command -v sha256sum >/dev/null 2>&1; then
      got="$(sha256sum "$tmp/$asset" | awk '{print $1}')"
    else
      got="$(shasum -a 256 "$tmp/$asset" | awk '{print $1}')"
    fi
    [ "$exp" = "$got" ] || die "SHA256 mismatch for $asset — aborting"
    info "SHA256 verified"
  else
    warn "no SHA256SUMS.txt on the release — skipping verification"
  fi
  mkdir -p "$dest"
  # release tarballs carry a cadx-<ver>/ root — strip it (BSD + GNU tar)
  tar xzf "$tmp/$asset" -C "$dest" --strip-components 1
  rm -rf "$tmp"
}

copy_over() { # <srcdir> <dstdir> — portable copy preserving excluded paths.
  # find prunes venvs/node_modules/.git at ANY depth; awk drops top-level
  # data/run/logs/.env*/.src/dist — but keeps .env.example (tracked template,
  # must update). Works with BSD + GNU tar (-T -).
  local src="$1" dst="$2"
  (cd "$src" && find . -type d \
      \( -name .venv -o -name ".venv-*" -o -name node_modules -o -name .git \) \
      -prune -o -type f -print \
    | awk '!/^\.\/(data\/|run\/|logs\/|mesh_server\/\.src\/|frontend\/dist\/)/ && ( !/^\.\/\.env($|\.)/ || /^\.\/\.env\.example$/ )' \
    | tar cf - -T -) | (cd "$dst" && tar xf -)
}

setup_and_start() { # <srcdir> — in-tree phase: preflight, .env, setup, start
  local src="$1"
  cd "$src"
  preflight_tools
  uname -s | grep -qi linux && \
    warn "Linux tiers are untested in this alpha — macOS/mlx is the verified platform"

  # disk preflight via the shipped tier table (rough need by GPU tier)
  local need_mb=1200 free_mb
  if [ -f scripts/lib/gpu_detect.sh ]; then
    # shellcheck disable=SC1091
    . scripts/lib/gpu_detect.sh
    need_mb="$(tier_disk_need_mb "${TIER:-$(gpu_detect_tier 2>/dev/null || echo none)}")"
  fi
  free_mb="$(df -m . | awk 'NR==2{print $4}')"
  [ "${free_mb:-0}" -lt "$need_mb" ] && \
    die "not enough free disk: ${free_mb} MiB free, ~${need_mb} MiB needed"

  if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    local pw; pw="$(python3 -c 'import secrets;print(secrets.token_hex(16))')"
    # -i with a backup suffix is portable across BSD and GNU sed
    sed -i.bak "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$pw/" .env && rm -f .env.bak
    info ".env created with a generated POSTGRES_PASSWORD"
  fi
  mkdir -p run
  printf '{"tag":"%s","channel":"%s","installed_at":"%s"}\n' \
    "${TAG:-local}" "$CHANNEL" "$(date -u +%FT%TZ)" > run/install.json

  local sargs=()
  [ "$YES" = 1 ] && sargs+=(--yes);        [ "$DEV" = 1 ] && sargs+=(--dev)
  [ "$FAKE" = 1 ] && sargs+=(--fake);      [ -n "$TIER" ] && sargs+=(--tier "$TIER")
  [ "$SKIP_MESH" = 1 ] && sargs+=(--skip-mesh)
  [ "$SKIP_IMAGE" = 1 ] && sargs+=(--skip-image)
  [ "$QUIET" = 1 ] && sargs+=(--quiet);    [ "$VERBOSE" = 1 ] && sargs+=(--verbose)
  bash scripts/setup.sh "${sargs[@]}"
  if [ "$NO_START" = 0 ]; then
    bash scripts/start.sh --no-setup
  else
    info "--no-start: run scripts/start.sh when ready"
  fi
  ok "done — CADX installed in $src"
}

reexec_args() { # flags to forward into the extracted/updated copy
  local a=()
  [ -n "$TAG" ] && a+=(--tag "$TAG"); a+=(--channel "$CHANNEL")
  [ "$DEV" = 1 ] && a+=(--dev);            [ "$FAKE" = 1 ] && a+=(--fake)
  [ -n "$TIER" ] && a+=(--tier "$TIER")
  [ "$SKIP_MESH" = 1 ] && a+=(--skip-mesh); [ "$SKIP_IMAGE" = 1 ] && a+=(--skip-image)
  [ "$NO_START" = 1 ] && a+=(--no-start);  [ "$YES" = 1 ] && a+=(--yes)
  [ "$QUIET" = 1 ] && a+=(--quiet);        [ "$VERBOSE" = 1 ] && a+=(--verbose)
  printf '%s\n' "${a[@]+"${a[@]}"}"
}

# ---------- mode dispatch ----------
if [ "$DRY" = 1 ]; then
  if [ "$UPDATE" = 1 ]; then M=update
  elif [ "$IN_TREE" = 1 ] && [ -z "$DIR" ]; then M=in-tree
  else M=bootstrap; fi
  info "dry-run: mode=$M channel=$CHANNEL tag=${TAG:-<resolve>} dir=${DIR:-$SELF_DIR} fake=$FAKE"
  exit 0
fi

if [ "$UPDATE" = 1 ]; then
  [ -z "$DIR" ] && die "--update needs --dir <existing install>"
  [ -f "$DIR/run/install.json" ] || die "$DIR is not a CADX install (no run/install.json)"
  preflight_tools
  [ -z "$TAG" ] && TAG="$(resolve_tag)"
  info "updating $DIR → $TAG"
  mkdir -p "$DIR/run"
  local_ts="$(date +%Y%m%d-%H%M%S)"
  if [ -f "$DIR/.env" ]; then
    cp "$DIR/.env" "$DIR/run/env-backup-$local_ts.env"
    info ".env backed up → run/env-backup-$local_ts.env"
  fi
  tmpd="$(mktemp -d)"; fetch_release "$TAG" "$tmpd/src"
  copy_over "$tmpd/src" "$DIR"; rm -rf "$tmpd"
  mapfile -t ra < <(reexec_args) 2>/dev/null || ra=()
  # bash-3.2 fallback when mapfile is absent
  [ ${#ra[@]} -eq 0 ] && while IFS= read -r l; do ra+=("$l"); done < <(reexec_args)
  exec bash "$DIR/install.sh" "${ra[@]}"
fi

if [ "$IN_TREE" = 1 ] && [ -z "$DIR" ]; then
  setup_and_start "$SELF_DIR"
  exit 0
fi

# ---------- bootstrap ----------
preflight_tools
DIR="${DIR:-./cadx}"
[ -z "$TAG" ] && TAG="$(resolve_tag)"
info "installing CADX $TAG (channel $CHANNEL) → $DIR"
fetch_release "$TAG" "$DIR"
ok "extracted → $DIR"
mapfile -t ra < <(reexec_args) 2>/dev/null || ra=()
[ ${#ra[@]} -eq 0 ] && while IFS= read -r l; do ra+=("$l"); done < <(reexec_args)
exec bash "$DIR/install.sh" "${ra[@]}"
