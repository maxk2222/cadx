#!/usr/bin/env bash
# Install Step1X-3D geometry (stepfun-ai/Step1X-3D, Apache-2.0) — CUDA only.
# Usage: mesh_server/scripts/install_step1x.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

STEP1X_REPO="https://github.com/stepfun-ai/Step1X-3D.git"
STEP1X_COMMIT="main"
VENV="${VENV:-mesh_server/.venv}"
SRC="${MESH_STEP1X_SRC:-mesh_server/.src/Step1X-3D}"
need_venv
pkg_detect

say "cloning $STEP1X_REPO"
clone_or_update "$STEP1X_REPO" "$SRC" "$STEP1X_COMMIT"

say "installing Step1X-3D requirements"
# Upstream requirements.txt pins a Linux/CUDA training stack (deepspeed,
# nvdiffrast, pytorch3d, cupy-cuda12x, sageattention, mosaicml-streaming) —
# the geometry inference pipeline only needs the subset below (the adapter
# binds a stub package that skips the eager data/systems imports, so the
# training-only deps are NOT required). pytorch_lightning is needed:
# utils/base.py's BaseModule derives from it.
INSTALL 'torch>=2.4' 'diffusers>=0.30,<0.33' 'transformers>=4.45,<5' \
  'huggingface_hub<1' accelerate rembg onnxruntime \
  'pytorch-lightning>=2.2' 'pymeshlab' 'scikit-image>=0.24' \
  einops omegaconf opencv-python-headless timm 'trimesh>=4.0' \
  'safetensors>=0.4' plyfile joblib jaxtyping typeguard \
  bs4 sentencepiece
# the clone has no packaging metadata — the adapter resolves it via
# MESH_STEP1X_SRC / .src/Step1X-3D and sys.path, no `pip install -e`

say "downloading weights"
hf_snapshot "stepfun-ai/Step1X-3D" || warn "stepfun-ai/Step1X-3D download failed"

ok "done — 'meshserver models' should list step1x-3d (CUDA hosts only)"
