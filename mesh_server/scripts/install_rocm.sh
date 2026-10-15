#!/usr/bin/env bash
# AMD ROCm tier installer: ROCm-enabled torch first, then TripoSG.
#
# ROCm torch builds reuse the cuda device namespace (torch.version.hip set),
# so the regular adapter install scripts work unchanged once a HIP-enabled
# torch is present — their 'torch>=x.y' requirement is already satisfied.
#
# Usage:
#   mesh_server/scripts/install_rocm.sh            # rocm torch + triposg
#   HUNYUAN=1 ... install_rocm.sh                  # also Hunyuan3D-2 torch
#
# Env:
#   TORCH_ROCM_INDEX   wheel index (default: pytorch.org rocm6.4)
#   PYTORCH_INDEX_URL  alias honoured by pip/uv directly (set if you prefer)
set -euo pipefail

VENV="${VENV:-mesh_server/.venv}"
TORCH_ROCM_INDEX="${TORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(uname -s)" != "Linux" ]; then
  echo "ROCm tier is Linux-only (amdgpu/KFD) — on this host use the mlx or fake tier" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "venv $VENV not found — run 'make mesh-sync' first" >&2
  exit 1
fi
PY="$VENV/bin/python"

if "$PY" -m pip --version >/dev/null 2>&1; then
  INSTALL() { "$PY" -m pip install "$@"; }
  REINSTALL=(--force-reinstall)
elif command -v uv >/dev/null 2>&1; then
  INSTALL() { uv pip install --python "$PY" "$@"; }
  REINSTALL=(--reinstall-package torch)
elif [ -x ".venv/bin/uv" ]; then
  INSTALL() { .venv/bin/uv pip install --python "$PY" "$@"; }
  REINSTALL=(--reinstall-package torch)
else
  echo "neither pip nor uv available — install uv or add pip to $VENV" >&2
  exit 1
fi

echo "== ROCm torch =="
echo "   index: $TORCH_ROCM_INDEX  (override via TORCH_ROCM_INDEX)"
# a satisfied 'torch>=x' spec alone is NOT enough: a CPU/CUDA build would be
# kept silently — verify the HIP flavour and force-replace anything else
if "$PY" -c "import sys, torch; sys.exit(0 if getattr(torch.version, 'hip', None) else 1)" \
    2>/dev/null; then
  echo "   ROCm torch already installed — skipping"
else
  INSTALL "${REINSTALL[@]}" "torch>=2.6" --index-url "$TORCH_ROCM_INDEX"
fi

echo "== sanity check =="
"$PY" - <<'EOF' || true
import torch
hip = getattr(torch.version, "hip", None)
ok = torch.cuda.is_available() and hip
print(f"   torch {torch.__version__}  hip={hip}  cuda_ns_available={torch.cuda.is_available()}")
if not ok:
    print("   WARNING: torch does not see the GPU — check amdgpu driver, ROCm")
    print("   install, /dev/kfd permissions and HSA_OVERRIDE_GFX_VERSION")
else:
    print("   gpu:", torch.cuda.get_device_name(0))
EOF

# gfx arch hint: wheels cover gfx900/906/908/90a/942/1030/110x/120x — anything
# else (e.g. gfx1031) needs HSA_OVERRIDE_GFX_VERSION to masquerade as a
# supported sibling
GFX="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1 || true)"
case "$GFX" in
  ""|gfx900|gfx906|gfx908|gfx90a|gfx942|gfx1030|gfx1100|gfx1101|gfx1102|gfx1200|gfx1201) ;;
  *) echo "   NOTE: gfx arch '$GFX' is not in the standard wheel support list —"
     echo "         you may need HSA_OVERRIDE_GFX_VERSION (e.g. 11.0.0 for gfx103x,"
     echo "         12.0.1 for gfx12xx)" ;;
esac

echo "== TripoSG (mesh adapter) =="
VENV="$VENV" bash "$ROOT/scripts/install_triposg.sh"

if [ "${HUNYUAN:-0}" = "1" ]; then
  echo "== Hunyuan3D-2 torch adapters =="
  VENV="$VENV" bash "$ROOT/scripts/install_hunyuan_torch.sh"
fi

echo "done — 'meshserver models' should list triposg on device 'rocm'"
