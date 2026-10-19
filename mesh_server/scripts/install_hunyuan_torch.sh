#!/usr/bin/env bash
# Install the official Hunyuan3D-2 repo (hy3dgen) for torch/MPS on Apple Silicon.
# Usage: mesh_server/scripts/install_hunyuan_torch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

HUNYUAN_REPO="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
HUNYUAN_COMMIT="main"
VENV="${VENV:-mesh_server/.venv}"
SRC="${HUNYUAN_SRC:-mesh_server/.src/Hunyuan3D-2}"
need_venv
pkg_detect

say "torch + pipeline deps"
# transformers>=5 renames Dinov2 modules (q_proj/gate_proj) — the Hunyuan3D
# checkpoints (incl. 2mv) use the pre-v5 layout, and diffusers>=0.32 forces
# huggingface_hub>=1 which conflicts with transformers<5. Pin the tested set.
INSTALL 'torch>=2.3' 'transformers==4.57.3' 'diffusers==0.31.0' 'huggingface_hub<1' accelerate rembg onnxruntime 'gradio<6'

say "cloning $HUNYUAN_REPO"
clone_or_update "$HUNYUAN_REPO" "$SRC" "$HUNYUAN_COMMIT"
INSTALL -e "$SRC"

say "downloading weights"
# partial failures are tolerated per-repo (mini is the useful default)
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("tencent/Hunyuan3D-2mini", "tencent/Hunyuan3D-2mv"):
    try:
        snapshot_download(repo)
        print("ok", repo)
    except Exception as e:
        print("FAILED", repo, e)
EOF

ok "done — 'meshserver models' should list hunyuan3d-2mini/-mv variants"
