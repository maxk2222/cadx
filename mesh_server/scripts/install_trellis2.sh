#!/usr/bin/env bash
# Install Microsoft TRELLIS.2-4B on a CUDA box (>=24 GB VRAM recommended).
# Usage: mesh_server/scripts/install_trellis2.sh
#
# Everything lands in mesh_server/.venv — the adapter imports trellis2
# in-process, so deps must live in that env (upstream's setup.sh --new-env
# conda path is unusable for us). GPU-arch aware: sm_120 (Blackwell/RTX 50xx)
# needs torch+cu128 — upstream's cu124 pin has no sm_120 kernels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

TRELLIS_REPO="https://github.com/microsoft/TRELLIS.2.git"
VENV="${VENV:-mesh_server/.venv}"
SRC="${TRELLIS_SRC:-mesh_server/.src/TRELLIS.2}"
need_venv
pkg_detect

say "cloning $TRELLIS_REPO (recursive submodules!)"
clone_or_update "$TRELLIS_REPO" "$SRC" main 1

# ---- torch stack by GPU arch ---------------------------------------------
# upstream pins torch 2.6.0+cu124, which ships no sm_120 kernels ("no kernel
# image available"). Blackwell (compute_cap >= 12.0) gets the cu128 line.
CC_MAJOR="$("$PY" - <<'PY' 2>/dev/null || true
import json, subprocess
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    ).decode().split()[0]
    print(out.split(".")[0])
except Exception:
    pass
PY
)"
if [ "${CC_MAJOR:-0}" -ge 12 ]; then
  TORCH_SPEC="torch==2.7.1" TVISION_SPEC="torchvision==0.22.1"
  TORCH_IDX="https://download.pytorch.org/whl/cu128"
  CUDA_MM="12.8" ARCH_LIST="12.0"
else
  TORCH_SPEC="torch==2.6.0" TVISION_SPEC="torchvision==0.21.0"
  TORCH_IDX="https://download.pytorch.org/whl/cu124"
  CUDA_MM="12.4" ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
fi
say "torch stack: $TORCH_SPEC $TVISION_SPEC ($TORCH_IDX, compute_cap major=${CC_MAJOR:-?})"
INSTALL "$TORCH_SPEC" "$TVISION_SPEC" --index-url "$TORCH_IDX"

# ---- nvcc / CUDA toolkit matching torch's CUDA ---------------------------
# The CUDA-only extensions need an nvcc whose major.minor equals torch's.
# Resolution order: $CUDA_HOME/bin/nvcc -> /usr/local/cuda*/bin/nvcc -> PATH.
pick_nvcc() {
  local c
  for c in "${NVCC:-}" "$CUDA_HOME/bin/nvcc" /usr/local/cuda/bin/nvcc \
           /usr/local/cuda-*/bin/nvcc "$(command -v nvcc || true)"; do
    [ -x "$c" ] || continue
    if "$c" --version 2>/dev/null | grep -q "release ${CUDA_MM%.*}\.${CUDA_MM#*.}"; then
      echo "$c"; return 0
    fi
  done
  return 1
}
NVCC_BIN="$(pick_nvcc || true)"
[ -n "$NVCC_BIN" ] || die "no nvcc matching CUDA $CUDA_MM found — install the CUDA $CUDA_MM toolkit or set CUDA_HOME/NVCC"
export CUDA_HOME="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
export PATH="$CUDA_HOME/bin:$PATH"
# conda-packaged toolkits keep headers/libs under targets/<triple>/ —
# host-side .cpp compiles need them on the search path too
TGT_INCLUDE="$(dirname "$(dirname "$NVCC_BIN")")/targets/x86_64-linux/include"
TGT_LIB="$(dirname "$(dirname "$NVCC_BIN")")/targets/x86_64-linux/lib"
[ -d "$TGT_INCLUDE" ] && export CPATH="$TGT_INCLUDE${CPATH:+:$CPATH}"
[ -d "$TGT_LIB" ] && export LIBRARY_PATH="$TGT_LIB:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
info "nvcc: $NVCC_BIN (CUDA_HOME=$CUDA_HOME)"

# nvcc rejects host gcc > 13 — pin a supported compiler when needed
if command -v g++-13 >/dev/null 2>&1; then
  GXX_VER="$(g++ -dumpversion 2>/dev/null | cut -d. -f1)"
  if [ "${GXX_VER:-0}" -gt 13 ]; then
    export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"
  fi
fi
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"

say "basic deps"
INSTALL imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
  trimesh 'transformers>=4.45,<5' tensorboard pandas lpips zstandard kornia timm
INSTALL 'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8'
INSTALL ninja psutil packaging wheel setuptools

say "CUDA extensions (cumesh, flex_gemm, o_voxel, flash-attn)"
mkdir -p /tmp/extensions
clone_or_update "https://github.com/JeffreyXiang/CuMesh.git" /tmp/extensions/CuMesh main 1
INSTALL --no-build-isolation /tmp/extensions/CuMesh
clone_or_update "https://github.com/JeffreyXiang/FlexGEMM.git" /tmp/extensions/FlexGEMM main 1
INSTALL --no-build-isolation /tmp/extensions/FlexGEMM
cp -r "$SRC/o-voxel" /tmp/extensions/o-voxel
INSTALL --no-build-isolation /tmp/extensions/o-voxel
INSTALL --no-build-isolation 'flash-attn==2.7.3'

say "registering trellis2 package (upstream ships no pyproject — .pth link)"
SITE_PKGS="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROOT/$SRC" > "$SITE_PKGS/trellis2.pth"
"$PY" -c 'from trellis2.pipelines import Trellis2ImageTo3DPipeline' \
  || die "trellis2 import failed after .pth registration"

say "downloading weights"
hf_snapshot "microsoft/TRELLIS.2-4B"

ok "done — 'meshserver models' should list trellis2 on a CUDA device"
