#!/usr/bin/env bash
# Install VAST-AI TripoSG 1.5B (CUDA; MPS experimental via MESH_TRIPOSG_MPS=1).
# Usage: mesh_server/scripts/install_triposg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

TRIPOSG_REPO="https://github.com/VAST-AI-Research/TripoSG.git"
VENV="${VENV:-mesh_server/.venv}"
SRC="${TRIPOSG_SRC:-mesh_server/.src/TripoSG}"
need_venv
pkg_detect

say "cloning $TRIPOSG_REPO"
clone_or_update "$TRIPOSG_REPO" "$SRC" main

say "installing TripoSG requirements"
INSTALL 'torch>=2.3' diffusers transformers accelerate rembg onnxruntime 'scikit-image>=0.24'
if [ -f "$SRC/requirements.txt" ]; then
  # upstream requirements pin numpy==1.22.3 (conflicts with our >=1.26 —
  # drop on every tier) and list diso, a CUDA-only extension that fails to
  # build on ROCm/MPS/CPU (the adapter injects a skimage shim instead).
  _drop='^[[:space:]]*(numpy[<=>!~ ]|diso([<=>!~ ]|$))'
  if "$PY" - <<'PY' 2>/dev/null
import torch, sys
sys.exit(0 if torch.cuda.is_available() and getattr(torch.version, "hip", None) is None else 1)
PY
  then
    _drop='^[[:space:]]*numpy[<=>!~ ]'   # real CUDA torch: keep diso
  else
    info "non-CUDA torch detected — skipping diso"
  fi
  install_req_filtered "$SRC/requirements.txt" "$_drop"
fi
INSTALL -e "$SRC"

say "downloading weights"
hf_snapshot "VAST-AI/TripoSG"

ok "done — 'meshserver models' should list triposg"
