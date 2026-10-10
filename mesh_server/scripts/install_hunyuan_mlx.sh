#!/usr/bin/env bash
# Install the community MLX port of Hunyuan3D-2.1 (shape stage) on Apple Silicon.
# Usage: mesh_server/scripts/install_hunyuan_mlx.sh
set -euo pipefail

# pinned for reproducibility — bump deliberately
MLX_FORK_REPO="https://github.com/dgrauet/Hunyuan3D-2.1-mlx.git"
MLX_FORK_COMMIT="main"

VENV="${VENV:-mesh_server/.venv}"
if [ ! -d "$VENV" ]; then
  echo "venv $VENV not found — run 'make mesh-sync' first" >&2
  exit 1
fi
PY="$VENV/bin/python"

echo "== installing mlx extras =="
"$PY" -m pip install 'mlx>=0.20' safetensors scikit-image 'PyMCubes>=0.1.6' scipy rembg onnxruntime

echo "== installing hy3dshape (MLX fork) =="
"$PY" -m pip install "git+${MLX_FORK_REPO}@${MLX_FORK_COMMIT}#subdirectory=hy3dshape" \
  || "$PY" -m pip install "git+${MLX_FORK_REPO}@${MLX_FORK_COMMIT}"

echo "== downloading weights (shape stage only) =="
"$PY" -m huggingface_hub.cli download tencent/Hunyuan3D-2.1 \
  --include 'hunyuan3d-dit-v2-1/*' || \
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns=["hunyuan3d-dit-v2-1/*"])
EOF

echo "done — 'meshserver models' should list hunyuan3d-2.1-mlx as available"
