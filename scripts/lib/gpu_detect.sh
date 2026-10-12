#!/usr/bin/env bash
# GPU tier detection for CADX mesh-server auto-setup.
#
# Usage:
#   source scripts/lib/gpu_detect.sh   # -> gpu_detect_tier / gpu_vram_mb / gpu_default_model
#   scripts/lib/gpu_detect.sh          # -> prints JSON {tier, vram_mb, model}
#
# Tiers:
#   mlx   - Apple Silicon (Darwin arm64) -> Hunyuan3D-2.1 via community MLX port
#   cuda  - local NVIDIA GPU             -> TRELLIS.2 (>=20 GB VRAM) else TripoSG
#   none  - no usable GPU                -> no real model (optional MESH_FAKE=1)
#
# Override for testing: GPU_TIER=mlx|cuda|none|fake
# TRELLIS.2 VRAM threshold: MESH_TRELLIS_MIN_VRAM_GB (default 20)

gpu_detect_tier() {
  if [ -n "${GPU_TIER:-}" ]; then
    echo "$GPU_TIER"
    return
  fi
  if [ "$(uname -sm 2>/dev/null)" = "Darwin arm64" ]; then
    echo "mlx"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "cuda"
    return
  fi
  echo "none"
}

gpu_vram_mb() {
  # total VRAM of the first GPU in MiB, 0 when unavailable
  command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' ' || echo 0
}

gpu_default_model() {
  # adapter id to install/select for a tier
  local tier="${1:-$(gpu_detect_tier)}"
  case "$tier" in
    mlx)
      echo "hunyuan3d-2.1-mlx" ;;
    cuda)
      local vram_g=$(( $(gpu_vram_mb) / 1024 ))
      local min_g=${MESH_TRELLIS_MIN_VRAM_GB:-20}
      if [ "$vram_g" -ge "$min_g" ]; then
        echo "trellis2"
      else
        echo "triposg"
      fi ;;
    fake)
      echo "fake" ;;
    *)
      echo "" ;;
  esac
}

gpu_install_script() {
  # which installer covers the tier's default model
  local tier="${1:-$(gpu_detect_tier)}"
  case "$tier" in
    mlx)  echo "install_hunyuan_mlx.sh" ;;
    cuda)
      local model; model="$(gpu_default_model "$tier")"
      if [ "$model" = "trellis2" ]; then
        echo "install_trellis2.sh"
      else
        echo "install_triposg.sh"
      fi ;;
    *)    echo "" ;;
  esac
}

gpu_default_image_model() {
  # text-to-image adapter id for a tier (same sidecar as the mesh adapters)
  local tier="${1:-$(gpu_detect_tier)}"
  case "$tier" in
    mlx)  echo "flux-schnell-mlx" ;;
    cuda)
      # FLUX.1-schnell via diffusers — the adapter itself applies CPU offload
      # on smaller cards (MESH_CUDA_OFFLOAD), so schnell works everywhere;
      # sdxl remains selectable as the light fallback.
      echo "flux-schnell" ;;
    fake) echo "fake-image" ;;
    *)    echo "" ;;
  esac
}

gpu_image_install_script() {
  local tier="${1:-$(gpu_detect_tier)}"
  case "$tier" in
    mlx)  echo "install_image_mlx.sh" ;;
    cuda) echo "install_image_torch.sh" ;;
    *)    echo "" ;;
  esac
}

# --- CLI mode ---
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  _tier=$(gpu_detect_tier)
  _model=$(gpu_default_model "$_tier")
  _vram=$(gpu_vram_mb)
  _script=$(gpu_install_script "$_tier")
  printf '{"tier":"%s","vram_mb":%s,"default_model":"%s","install_script":"%s"}\n' \
    "$_tier" "${_vram:-0}" "$_model" "$_script"
fi
