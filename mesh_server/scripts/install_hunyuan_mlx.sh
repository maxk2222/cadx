#!/usr/bin/env bash
# Install the community MLX port of Hunyuan3D-2.1 (shape stage) on Apple Silicon.
# Usage: mesh_server/scripts/install_hunyuan_mlx.sh
set -euo pipefail

# pinned for reproducibility — bump deliberately
MLX_FORK_REPO="https://github.com/dgrauet/Hunyuan3D-2.1-mlx.git"
MLX_FORK_COMMIT="main"
MLX_WEIGHTS_REPO="dgrauet/hunyuan3d-2.1-mlx"

VENV="${VENV:-mesh_server/.venv}"
SRC="${MESH_HUNYUAN_MLX_SRC:-mesh_server/.src/Hunyuan3D-2.1-mlx}"
if [ ! -d "$VENV" ]; then
  echo "venv $VENV not found — run 'make mesh-sync' first" >&2
  exit 1
fi
PY="$VENV/bin/python"

# uv-managed venvs ship without pip — prefer uv when available
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

echo "== installing mlx deps =="
# NOTE: the fork's hy3dshape __init__ eagerly imports the torch pipeline and
# postprocessors, so torch stack + image/mesh utils are required even when only
# the MLX path is used.
INSTALL 'mlx>=0.31' 'mlx-arsenal>=0.10.1' safetensors 'scipy>=1.12' 'scikit-image>=0.24'
INSTALL 'torch>=2.3' torchvision diffusers 'transformers>=4.45' accelerate
INSTALL einops omegaconf opencv-python-headless pygltflib xatlas pymeshlab rembg onnxruntime

echo "== cloning $MLX_FORK_REPO =="
mkdir -p "$(dirname "$SRC")"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch --depth 1 origin "$MLX_FORK_COMMIT" && git -C "$SRC" checkout FETCH_HEAD
else
  git clone --depth 1 "$MLX_FORK_REPO" "$SRC"
fi

echo "== downloading converted MLX weights ($MLX_WEIGHTS_REPO) =="
"$PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    "$MLX_WEIGHTS_REPO",
    allow_patterns=[
        "config.json", "split_model.json",
        "image_encoder.safetensors", "dit.safetensors", "vae.safetensors",
    ],
)
EOF

echo "done — 'meshserver models' should list hunyuan3d-2.1-mlx as available"
