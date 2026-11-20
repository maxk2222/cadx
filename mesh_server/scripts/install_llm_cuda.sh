#!/usr/bin/env bash
# Install the prebuilt CUDA llama-server binary (ggml-org/llama.cpp release).
# Usage: mesh_server/scripts/install_llm_cuda.sh [build-tag]
#
# Why the binary: the gguf adapter prefers `llama-server` on PATH /
# LLAMA_SERVER_BIN over `python -m llama_cpp.server` — only the binary gives
# real continuous batching (unified KV pool across --parallel slots, mmproj
# vision/video input, --cache-type quant). llama-cpp-python stays a serial
# CPU fallback.
#
# Blackwell note: RTX 50xx (sm_120) needs a CUDA >=12.8 build; the script
# fetches the ubuntu-cuda-12.8 asset which carries sm_120 cubins.
#
# Env honoured: LLAMA_RELEASE_TAG (e.g. b12090 — default: latest release),
#   LLAMA_BIN_DIR (default ~/.local/share/meshserver/llama-cpp),
#   DRY (dry-run print).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

: "${LLAMA_BIN_DIR:=$HOME/.local/share/meshserver/llama-cpp}"
: "${DRY:=0}"
REPO="ggml-org/llama.cpp"

if ! command -v curl >/dev/null 2>&1; then
  say "curl is required"; exit 1
fi

TAG="${1:-${LLAMA_RELEASE_TAG:-}}"
if [ -z "$TAG" ]; then
  say "resolving latest llama.cpp release tag"
  TAG="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -m1 '"tag_name"' | sed -E 's/.*"(b[0-9]+)".*/\1/')"
fi
[ -n "$TAG" ] || { say "could not resolve a release tag — pass one explicitly"; exit 1; }
info "release tag: $TAG"

# Prefer the cudart-* stand-alone asset — it bundles libggml-cuda +
# cudart/cublas .so files next to the binaries, so no local CUDA toolkit
# install is required. Fall back to the slim cuda-12.8 asset (needs the
# system cuda-12.8 cudart/cublas packages).
ASSETS=(
  "cudart-llama-${TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"
  "llama-${TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"
)

mkdir -p "$LLAMA_BIN_DIR"
FETCHED=""
for ASSET in "${ASSETS[@]}"; do
  URL="https://github.com/$REPO/releases/download/${TAG}/${ASSET}"
  say "fetching $ASSET"
  if [ "$DRY" = "1" ]; then
    info "+ curl -fL $URL | tar -xz -C $LLAMA_BIN_DIR"
    FETCHED="$ASSET"
    break
  fi
  if curl -fL "$URL" | tar -xz -C "$LLAMA_BIN_DIR"; then
    FETCHED="$ASSET"
    break
  fi
  info "asset missing — trying next variant"
done
[ -n "$FETCHED" ] || { say "no llama.cpp CUDA asset could be fetched"; exit 1; }
info "installed from: $FETCHED"

case "$FETCHED" in
  cudart-*) : ;;  # runtime libs bundled next to the binaries
  *) ldconfig -p 2>/dev/null | grep -q "libcudart.so.12" || \
     info "warn: libcudart.so.12 not in ldconfig — install cuda-cudart-12-8 + libcublas-12-8" ;;
esac

BIN="$(find "$LLAMA_BIN_DIR" -name llama-server -type f | head -1)"
if [ -z "$BIN" ]; then
  say "llama-server not found in the archive — check $LLAMA_BIN_DIR"; exit 1
fi
chmod +x "$BIN"
ok "llama-server: $BIN"

if [ "$DRY" != "1" ]; then
  "$BIN" --version 2>/dev/null | head -2 || true
fi

# PATH convenience — the adapter probes shutil.which("llama-server") first
LINK_DIR="$HOME/.local/bin"
if [ -d "$LINK_DIR" ] || mkdir -p "$LINK_DIR"; then
  ln -sf "$BIN" "$LINK_DIR/llama-server" 2>/dev/null \
    && ok "symlinked $LINK_DIR/llama-server" \
    || info "could not symlink into $LINK_DIR — set LLAMA_SERVER_BIN=$BIN"
fi

info ""
info "verify:  llama-server --list-devices   (should show CUDA0, sm_120 on RTX 50xx)"
info "custom:  export LLAMA_SERVER_BIN=$BIN"
