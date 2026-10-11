#!/usr/bin/env bash
# Install Microsoft TRELLIS.2-4B on a CUDA box (>=24 GB VRAM recommended).
# Usage: mesh_server/scripts/install_trellis2.sh
set -euo pipefail

TRELLIS_REPO="https://github.com/microsoft/TRELLIS.2.git"
VENV="${VENV:-mesh_server/.venv}"
SRC="${TRELLIS_SRC:-mesh_server/.src/TRELLIS.2}"

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

echo "== cloning $TRELLIS_REPO (recursive submodules!) =="
mkdir -p "$(dirname "$SRC")"
if [ ! -d "$SRC/.git" ]; then
  git clone --recursive "$TRELLIS_REPO" "$SRC"
else
  git -C "$SRC" pull --recurse-submodules
fi

echo "== TRELLIS.2 setup (basic + flash-attn + cumesh + o-voxel + flexgemm) =="
cd "$SRC"
. ./setup.sh --new-env --basic --flash-attn --cumesh --o-voxel --flexgemm
cd - >/dev/null
INSTALL -e "$SRC"

echo "== downloading weights =="
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("microsoft/TRELLIS.2-4B")
EOF

echo "done — 'meshserver models' should list trellis2 on a CUDA device"
