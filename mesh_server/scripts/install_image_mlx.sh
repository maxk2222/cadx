#!/usr/bin/env bash
# Install the FLUX.1-schnell text-to-image adapter (mflux) on Apple Silicon.
# Usage: mesh_server/scripts/install_image_mlx.sh
set -euo pipefail

VENV="${VENV:-mesh_server/.venv}"
if [ ! -d "$VENV" ]; then
  echo "venv $VENV not found — run 'make mesh-sync' first" >&2
  exit 1
fi
PY="$VENV/bin/python"

if "$PY" -m pip --version >/dev/null 2>&1; then
  INSTALL() { "$PY" -m pip install "$@"; }
elif command -v uv >/dev/null 2>&1; then
  INSTALL() { uv pip install --python "$PY" "$@"; }
elif [ -x ".venv/bin/uv" ]; then
  INSTALL() { .venv/bin/uv pip install --python "$PY" "$@"; }
else
  echo "neither pip nor uv available — install uv or add pip to $VENV" >&2
  exit 1
fi

echo "== installing image-mlx deps (mflux + mlx) =="
if [ -d "mesh_server" ]; then
  (cd mesh_server && uv pip install --python "../$PY" -e '.[image-mlx]' -q) 2>/dev/null \
    || INSTALL 'mflux>=0.9' 'mlx>=0.20' 'safetensors>=0.4'
else
  INSTALL 'mflux>=0.9' 'mlx>=0.20' 'safetensors>=0.4'
fi

echo "== done =="
echo "first run downloads FLUX.1-schnell weights (~12 GB, 4-bit) from HF to ~/.cache/mflux"
echo "test: $PY -m meshserver.cli image --prompt 'a red gear' --model flux-schnell-mlx"
