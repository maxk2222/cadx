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
# mflux's declared deps are unsatisfiable in this shared venv (it wants
# transformers>=5.5 / mlx<0.31, torch extras want transformers<5 /
# mlx_arsenal needs mlx>=0.31) — install the extra's dep list normally and
# mflux itself with --no-deps. qwen_edit_mlx patches mflux's tf5-era
# Qwen2Tokenizer workaround so it loads under transformers 4.x.
if [ -d "mesh_server" ]; then
  (cd mesh_server && uv pip install --python "../$VENV/bin/python" -e '.[image-mlx]' -q) 2>/dev/null \
    || INSTALL 'mlx>=0.32' 'safetensors>=0.4' 'transformers>=4.45,<5' \
        'sentencepiece>=0.2' 'opencv-python>=4.10' 'piexif>=1.1' 'platformdirs>=4.0' 'toml>=0.10'
else
  INSTALL 'mlx>=0.32' 'safetensors>=0.4' 'transformers>=4.45,<5' \
    'sentencepiece>=0.2' 'opencv-python>=4.10' 'piexif>=1.1' 'platformdirs>=4.0' 'toml>=0.10'
fi
INSTALL --no-deps 'mflux>=0.14'

ok "done"
info "first run downloads FLUX.1-schnell weights (~12 GB, 4-bit) from HF to ~/.cache/mflux"
info "test: $VENV/bin/python -m meshserver.cli image --prompt 'a red gear' --model flux-schnell-mlx"
