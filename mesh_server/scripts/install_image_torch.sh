#!/usr/bin/env bash
# Install the diffusers text-to-image adapters (FLUX.1-schnell / SDXL / SD1.5)
# for CUDA (and torch fallbacks on other tiers).
# Usage: mesh_server/scripts/install_image_torch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

VENV="${VENV:-mesh_server/.venv}"
need_venv
pkg_detect

say "installing image-torch deps (diffusers stack)"
INSTALL 'torch>=2.3' 'diffusers>=0.30,<0.36' 'transformers>=4.45,<5' \
  'huggingface_hub<1' 'accelerate>=0.33' 'sentencepiece>=0.2' 'protobuf' 'safetensors>=0.4'

ok "done"
info "first run downloads model weights from HF (flux-schnell ~24 GB fp16, sdxl ~7 GB)"
info "small GPUs: MESH_CUDA_OFFLOAD=sequential lowers peak VRAM to ~6-8 GB"
info "test: $VENV/bin/python -m meshserver.cli image --prompt 'a red gear' --model flux-schnell"
