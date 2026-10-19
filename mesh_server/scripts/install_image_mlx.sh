#!/usr/bin/env bash
# Install the FLUX.1-schnell text-to-image adapter (mflux) on Apple Silicon.
# Usage: mesh_server/scripts/install_image_mlx.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

VENV="${VENV:-mesh_server/.venv}"
need_venv
pkg_detect

say "installing image-mlx deps (mflux + mlx)"
if [ -d "mesh_server" ]; then
  (cd mesh_server && uv pip install --python "../$VENV/bin/python" -e '.[image-mlx]' -q) 2>/dev/null \
    || INSTALL 'mflux>=0.9' 'mlx>=0.20' 'safetensors>=0.4'
else
  INSTALL 'mflux>=0.9' 'mlx>=0.20' 'safetensors>=0.4'
fi

ok "done"
info "first run downloads FLUX.1-schnell weights (~12 GB, 4-bit) from HF to ~/.cache/mflux"
info "test: $VENV/bin/python -m meshserver.cli image --prompt 'a red gear' --model flux-schnell-mlx"
