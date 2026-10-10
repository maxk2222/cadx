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

echo "== cloning $TRIPOSG_REPO =="
mkdir -p "$(dirname "$SRC")"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 "$TRIPOSG_REPO" "$SRC"
fi

echo "== installing TripoSG requirements =="
"$PY" -m pip install 'torch>=2.3' diffusers transformers accelerate rembg onnxruntime
if [ -f "$SRC/requirements.txt" ]; then
  "$PY" -m pip install -r "$SRC/requirements.txt"
fi
"$PY" -m pip install -e "$SRC"

echo "== downloading weights =="
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("VAST-AI/TripoSG")
EOF

echo "done — 'meshserver models' should list triposg"
