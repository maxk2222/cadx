#!/usr/bin/env bash
# CADX shared installer helpers for mesh_server/scripts/install_*.sh —
# deduplicates the venv/pip/uv/clone/HF-download boilerplate.
#
# Usage (from a mesh_server install script):
#   ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
#   source scripts/lib/pkg.sh
#   need_venv mesh_server/.venv          # or rely on default VENV
#   pkg_detect                           # sets INSTALL(), PY
#   clone_or_update <repo> <dir> [ref]
#   hf_snapshot <repo> [allow_patterns...]
#
# Env honoured: VENV (default mesh_server/.venv), DRY (dry-run print).

_PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# ui.sh is optional-but-expected: fall back to plain echo primitives
if [ -f "$_PKG_ROOT/scripts/lib/ui.sh" ]; then
  source "$_PKG_ROOT/scripts/lib/ui.sh"
else
  say()  { printf '== %s\n' "$*"; }
  info() { printf '   %s\n' "$*"; }
  ok()   { printf '   ok %s\n' "$*"; }
  warn() { printf '! %s\n' "$*" >&2; }
  err()  { printf 'ERR %s\n' "$*" >&2; }
  die()  { err "$*"; exit 1; }
  run_step() { local l="$1"; shift; say "$l"; "$@"; }
  : "${DRY:=0}"
fi

PY=""
INSTALL() { die "pkg_detect not called"; }

need_venv() {
  # need_venv [venv-dir] — defaults to $VENV or mesh_server/.venv
  local v="${1:-${VENV:-mesh_server/.venv}}"
  if [ ! -d "$v" ]; then
    die "venv $v not found — run 'make mesh-sync' first"
  fi
  VENV="$v"
  PY="$v/bin/python"
}

pkg_detect() {
  # pick the best package installer into the mesh_server venv:
  # venv pip -> `uv` on PATH -> repo .venv uv
  : "${VENV:=mesh_server/.venv}"
  PY="$VENV/bin/python"
  if "$PY" -m pip --version >/dev/null 2>&1; then
    INSTALL() { "$PY" -m pip install "$@"; }
  elif command -v uv >/dev/null 2>&1; then
    INSTALL() { uv pip install --python "$PY" "$@"; }
  elif [ -x ".venv/bin/uv" ]; then
    INSTALL() { .venv/bin/uv pip install --python "$PY" "$@"; }
  else
    die "neither pip nor uv available — install uv or add pip to $VENV"
  fi
}

clone_or_update() {
  # clone_or_update <repo> <dir> [ref] [recursive] — shallow clone or
  # fetch+checkout; recursive=1 pulls submodules instead.
  local repo="$1" dir="$2" ref="${3:-main}" recurse="${4:-0}"
  mkdir -p "$(dirname "$dir")"
  if [ "$DRY" = "1" ]; then
    info "+ clone/update $repo -> $dir"
    return 0
  fi
  if [ -d "$dir/.git" ]; then
    if [ "$recurse" = "1" ]; then
      git -C "$dir" pull --recurse-submodules
    else
      git -C "$dir" fetch --depth 1 origin "$ref" \
        && git -C "$dir" checkout -q FETCH_HEAD
    fi
  else
    if [ "$recurse" = "1" ]; then
      git clone --recursive "$repo" "$dir"
    else
      git clone --depth 1 "$repo" "$dir"
    fi
  fi
}

hf_snapshot() {
  # hf_snapshot <repo> [allow_pattern ...] — download via venv python.
  # Disables tqdm bars when non-TTY so the log file stays readable.
  local repo="$1"; shift || true
  [ "$DRY" = "1" ] && { info "+ hf snapshot_download $repo $*"; return 0; }
  local no_tty=1
  [ -t 1 ] && no_tty=0
  HF_HUB_DISABLE_PROGRESS_BARS="$no_tty" "$PY" - "$repo" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, patterns = sys.argv[1], sys.argv[2:]
kw = {"allow_patterns": patterns} if patterns else {}
try:
    snapshot_download(repo, **kw)
    print("ok", repo)
except Exception as e:
    print("FAILED", repo, e)
    sys.exit(1)
PY
}

install_req_filtered() {
  # install_req_filtered <requirements.txt> <drop_regex> — install an
  # upstream requirements file minus filtered lines (numpy pins, CUDA-only
  # extensions like diso).
  local file="$1" drop="$2" tmp
  tmp=$(mktemp)
  grep -v -E "$drop" "$file" > "$tmp" || true
  INSTALL -r "$tmp"
  rm -f "$tmp"
}
