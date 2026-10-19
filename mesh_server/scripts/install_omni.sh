#!/usr/bin/env bash
# Install Hunyuan3D-Omni (controllable image-to-3D, torch) for MPS/CUDA.
# Usage: mesh_server/scripts/install_omni.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

OMNI_REPO="https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git"
OMNI_COMMIT="main"
VENV="${VENV:-mesh_server/.venv}"
SRC="${MESH_HUNYUAN_OMNI_SRC:-mesh_server/.src/Hunyuan3D-Omni}"
need_venv
pkg_detect

say "torch + pipeline deps"
# Same pinned stack as install_hunyuan_torch.sh — the Omni checkpoint shares
# the Hunyuan3D-2.x Dinov2 conditioner layout. Upstream requirements.txt pins
# a Linux/CUDA-only set (cupy-cuda12x, deepspeed, torchaudio+cu124, tf 4.46,
# diffusers 0.30) — intentionally NOT installed wholesale here.
INSTALL 'torch>=2.3' 'transformers==4.57.3' 'diffusers==0.31.0' 'huggingface_hub<1' \
  accelerate rembg onnxruntime einops opencv-python 'scikit-image>=0.24' omegaconf \
  'trimesh>=4.0' tqdm

say "cloning $OMNI_REPO"
clone_or_update "$OMNI_REPO" "$SRC" "$OMNI_COMMIT"
# the clone has no packaging metadata — the adapter resolves it via
# MESH_HUNYUAN_OMNI_SRC / .src/Hunyuan3D-Omni and sys.path, no `pip install -e`

say "downloading weights"
hf_snapshot "tencent/Hunyuan3D-Omni" || warn "tencent/Hunyuan3D-Omni download failed"

ok "done — 'meshserver models' should list hunyuan3d-omni"
