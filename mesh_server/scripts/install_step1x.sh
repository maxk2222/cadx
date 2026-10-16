#!/usr/bin/env bash
# Install Step1X-3D geometry (stepfun-ai/Step1X-3D, Apache-2.0) — CUDA only.
# Usage: mesh_server/scripts/install_step1x.sh
set -euo pipefail

STEP1X_REPO="https://github.com/stepfun-ai/Step1X-3D.git"
STEP1X_COMMIT="main"
VENV="${VENV:-mesh_server/.venv}"
SRC="${MESH_STEP1X_SRC:-mesh_server/.src/Step1X-3D}"

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

echo "== cloning $STEP1X_REPO =="
mkdir -p "$(dirname "$SRC")"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch --depth 1 origin "$STEP1X_COMMIT" && git -C "$SRC" checkout FETCH_HEAD
else
  git clone --depth 1 "$STEP1X_REPO" "$SRC"
fi

echo "== installing Step1X-3D requirements =="
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

echo "== downloading weights =="
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
try:
    snapshot_download("stepfun-ai/Step1X-3D")
    print("ok stepfun-ai/Step1X-3D")
except Exception as e:
    print("FAILED stepfun-ai/Step1X-3D", e)
EOF

echo "done — 'meshserver models' should list step1x-3d (CUDA hosts only)"
