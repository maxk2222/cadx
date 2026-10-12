#!/usr/bin/env bash
# Install the diffusers text-to-image adapters (FLUX.1-schnell / SDXL / SD1.5)
# for CUDA (and torch fallbacks on other tiers).
# Usage: mesh_server/scripts/install_image_torch.sh
set -euo pipefail

VENV="${VENV:-mesh_server/.venv}"
if [ ! -d "$VENV" ]; then
  echo "venv $VENV not found — run 'make mesh-sync' first" >&2
  exit 1
fi
PY="$VENV/bin/python"

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

echo "== installing image-torch deps (diffusers stack) =="
INSTALL 'torch>=2.3' 'diffusers>=0.30,<0.36' 'transformers>=4.45,<5' \
  'huggingface_hub<1' 'accelerate>=0.33' 'sentencepiece>=0.2' 'protobuf' 'safetensors>=0.4'

echo "== done =="
echo "first run downloads model weights from HF (flux-schnell ~24 GB fp16, sdxl ~7 GB)"
echo "small GPUs: MESH_CUDA_OFFLOAD=sequential lowers peak VRAM to ~6-8 GB"
echo "test: $PY -m meshserver.cli image --prompt 'a red gear' --model flux-schnell"
