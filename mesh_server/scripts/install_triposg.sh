#!/usr/bin/env bash
# Install VAST-AI TripoSG 1.5B (CUDA; MPS experimental via MESH_TRIPOSG_MPS=1).
# Usage: mesh_server/scripts/install_triposg.sh
set -euo pipefail

TRIPOSG_REPO="https://github.com/VAST-AI-Research/TripoSG.git"
VENV="${VENV:-mesh_server/.venv}"
SRC="${TRIPOSG_SRC:-mesh_server/.src/TripoSG}"

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

echo "== cloning $TRIPOSG_REPO =="
mkdir -p "$(dirname "$SRC")"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 "$TRIPOSG_REPO" "$SRC"
fi

echo "== installing TripoSG requirements =="
INSTALL 'torch>=2.3' diffusers transformers accelerate rembg onnxruntime
if [ -f "$SRC/requirements.txt" ]; then
  INSTALL -r "$SRC/requirements.txt"
fi
INSTALL -e "$SRC"

echo "== downloading weights =="
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("VAST-AI/TripoSG")
EOF

echo "done — 'meshserver models' should list triposg"
