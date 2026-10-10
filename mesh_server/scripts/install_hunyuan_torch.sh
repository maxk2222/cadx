#!/usr/bin/env bash
# Install the official Hunyuan3D-2 repo (hy3dgen) for torch/MPS on Apple Silicon.
# Usage: mesh_server/scripts/install_hunyuan_torch.sh
set -euo pipefail

HUNYUAN_REPO="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
HUNYUAN_COMMIT="main"
VENV="${VENV:-mesh_server/.venv}"
SRC="${HUNYUAN_SRC:-mesh_server/.src/Hunyuan3D-2}"

if [ ! -d "$VENV" ]; then
  echo "venv $VENV not found — run 'make mesh-sync' first" >&2
  exit 1
fi
PY="$VENV/bin/python"

echo "== torch + pipeline deps =="
"$PY" -m pip install 'torch>=2.3' diffusers transformers accelerate rembg onnxruntime

echo "== cloning $HUNYUAN_REPO =="
mkdir -p "$(dirname "$SRC")"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch --depth 1 origin "$HUNYUAN_COMMIT" && git -C "$SRC" checkout FETCH_HEAD
else
  git clone --depth 1 "$HUNYUAN_REPO" "$SRC"
fi
"$PY" -m pip install -e "$SRC"

echo "== downloading weights =="
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("tencent/Hunyuan3D-2mini", "tencent/Hunyuan3D-2mv"):
    try:
        snapshot_download(repo)
        print("ok", repo)
    except Exception as e:
        print("FAILED", repo, e)
EOF

echo "done — 'meshserver models' should list hunyuan3d-2mini/-mv variants"
