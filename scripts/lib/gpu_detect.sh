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
#   rocm  - local AMD GPU (Linux, ROCm)  -> TripoSG (TRELLIS.2 is CUDA-only)
#   none  - no usable GPU                -> no real model (optional MESH_FAKE=1)
#
# Override for testing: GPU_TIER=mlx|cuda|rocm|none|fake
# TRELLIS.2 VRAM threshold: MESH_TRELLIS_MIN_VRAM_GB (default 20)

_rocm_present() {
  [ "$(uname -s 2>/dev/null)" = "Linux" ] || return 1
  if command -v amd-smi >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1 \
      || command -v rocminfo >/dev/null 2>&1; then
    return 0
  fi
  # amdgpu KFD device + a DRM render node is enough even without ROCm tools
  [ -c /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1
}

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
  if _rocm_present; then
    echo "rocm"
    return
  fi
  echo "none"
}

gpu_vram_mb() {
  # total VRAM of the first GPU in MiB, 0 when unavailable
  if command -v nvidia-smi >/dev/null 2>&1; then
    local v
    v=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | tr -d ' ')
    [ -n "$v" ] && { echo "$v"; return; }
  fi
  # AMD: sysfs works without ROCm userspace tools
  for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -r "$f" ] || continue
    local bytes; bytes=$(cat "$f" 2>/dev/null)
    [ -n "$bytes" ] && [ "$bytes" -eq "$bytes" ] 2>/dev/null || continue
    echo $(( bytes / 1048576 ))
    return
  done
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showmeminfo vram --csv 2>/dev/null \
      | awk -F, 'NR>1 {gsub(/[^0-9.]/,"",$2); print int($2/1048576); exit}' && return
  fi
  echo 0
}

gpu_default_model() {
  # adapter id to install/select for a tier
  local tier="${1:-$(gpu_detect_tier)}"
  case "$tier" in
    mlx)
      echo "hunyuan3d-2.1-mlx" ;;
    rocm)
      # TRELLIS.2 is CUDA-only (flash-attn/CuMesh/spconv/o-voxel) — TripoSG
      # is the MIT-licensed default on AMD regardless of VRAM size
      echo "triposg" ;;
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
    rocm) echo "install_rocm.sh" ;;   # ROCm torch first, then delegates to install_triposg.sh
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
    cuda|rocm)
      # FLUX.1-schnell via diffusers — the adapter itself applies CPU offload
      # on smaller cards (MESH_CUDA_OFFLOAD), so schnell works everywhere;
      # sdxl remains selectable as the light fallback. Runs on ROCm unchanged
      # (ROCm torch exposes the cuda device namespace).
      echo "flux-schnell" ;;
    fake) echo "fake-image" ;;
    *)    echo "" ;;
  esac
}

tier_disk_need_mb() {
  # rough free-disk requirement (MiB) for the tier's default adapters —
  # deps + model weights; used by setup.sh preflight.
  local tier="${1:-$(gpu_detect_tier)}" model
  model=$(gpu_default_model "$tier")
  case "$tier:$model" in
    cuda:trellis2)  echo 32000 ;;   # torch cu + trellis deps + ~10GB weights
    cuda:*)         echo 12000 ;;   # triposg
    rocm:*)         echo 16000 ;;   # rocm torch wheels are chunky
    mlx:*)          echo 19000 ;;   # mlx deps + .venv-llm (~2GB) + flux weights (~12GB)
    *)              echo 800 ;;     # fake/none — tiny deps only
  esac
}

gpu_image_install_script() {
  local tier="${1:-$(gpu_detect_tier)}"
  case "$tier" in
    mlx)  echo "install_image_mlx.sh" ;;
    cuda|rocm) echo "install_image_torch.sh" ;;
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
