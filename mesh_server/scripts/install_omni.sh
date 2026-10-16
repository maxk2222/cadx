#!/usr/bin/env bash
# Install Hunyuan3D-Omni (controllable image-to-3D, torch) for MPS/CUDA.
# Usage: mesh_server/scripts/install_omni.sh
set -euo pipefail

OMNI_REPO="https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git"
OMNI_COMMIT="main"
VENV="${VENV:-mesh_server/.venv}"
SRC="${MESH_HUNYUAN_OMNI_SRC:-mesh_server/.src/Hunyuan3D-Omni}"

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

echo "== torch + pipeline deps =="
# Same pinned stack as install_hunyuan_torch.sh — the Omni checkpoint shares
# the Hunyuan3D-2.x Dinov2 conditioner layout. Upstream requirements.txt pins
# a Linux/CUDA-only set (cupy-cuda12x, deepspeed, torchaudio+cu124, tf 4.46,
# diffusers 0.30) — intentionally NOT installed wholesale here.
INSTALL 'torch>=2.3' 'transformers==4.57.3' 'diffusers==0.31.0' 'huggingface_hub<1' \
  accelerate rembg onnxruntime einops opencv-python 'scikit-image>=0.24' omegaconf \
  'trimesh>=4.0' tqdm

echo "== cloning $OMNI_REPO =="
mkdir -p "$(dirname "$SRC")"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch --depth 1 origin "$OMNI_COMMIT" && git -C "$SRC" checkout FETCH_HEAD
else
  git clone --depth 1 "$OMNI_REPO" "$SRC"
fi
# the clone has no packaging metadata — the adapter resolves it via
# MESH_HUNYUAN_OMNI_SRC / .src/Hunyuan3D-Omni and sys.path, no `pip install -e`

echo "== downloading weights =="
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
try:
    snapshot_download("tencent/Hunyuan3D-Omni")
    print("ok tencent/Hunyuan3D-Omni")
except Exception as e:
    print("FAILED tencent/Hunyuan3D-Omni", e)
EOF

echo "done — 'meshserver models' should list hunyuan3d-omni"
