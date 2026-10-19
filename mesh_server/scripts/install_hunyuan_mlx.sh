#!/usr/bin/env bash
# Install the community MLX port of Hunyuan3D-2.1 (shape stage) on Apple Silicon.
# Usage: mesh_server/scripts/install_hunyuan_mlx.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

# pinned for reproducibility — bump deliberately
MLX_FORK_REPO="https://github.com/dgrauet/Hunyuan3D-2.1-mlx.git"
MLX_FORK_COMMIT="main"
MLX_WEIGHTS_REPO="dgrauet/hunyuan3d-2.1-mlx"

VENV="${VENV:-mesh_server/.venv}"
SRC="${MESH_HUNYUAN_MLX_SRC:-mesh_server/.src/Hunyuan3D-2.1-mlx}"
need_venv
pkg_detect

say "installing mlx deps"
# NOTE: the fork's hy3dshape __init__ eagerly imports the torch pipeline and
# postprocessors, so torch stack + image/mesh utils are required even when only
# the MLX path is used.
INSTALL 'mlx>=0.31' 'mlx-arsenal>=0.10.1' safetensors 'scipy>=1.12' 'scikit-image>=0.24'
INSTALL 'torch>=2.3' torchvision diffusers 'transformers>=4.45' accelerate
INSTALL einops omegaconf opencv-python-headless pygltflib xatlas pymeshlab rembg onnxruntime

say "cloning $MLX_FORK_REPO"
clone_or_update "$MLX_FORK_REPO" "$SRC" "$MLX_FORK_COMMIT"

say "downloading converted MLX weights ($MLX_WEIGHTS_REPO)"
hf_snapshot "$MLX_WEIGHTS_REPO" \
  "config.json" "split_model.json" \
  "image_encoder.safetensors" "dit.safetensors" "vae.safetensors"

ok "done — 'meshserver models' should list hunyuan3d-2.1-mlx as available"
