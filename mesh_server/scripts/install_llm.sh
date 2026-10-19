#!/usr/bin/env bash
# Install local-LLM backends.
# Usage: mesh_server/scripts/install_llm.sh
#
# mlx tier (Apple Silicon): the modern multimodal Qwen generation needs
# mlx-lm>=0.32 + mlx-vlm — both require transformers>=5, which conflicts
# with the torch extras pinned to <5 in the shared mesh venv. They
# therefore live in a dedicated venv at mesh_server/.venv-llm that the
# mlx adapter auto-resolves (MESH_LLM_PYTHON overrides).
# A legacy mlx-lm<0.30 stays in the shared venv as fallback for older
# architectures.
# Other tiers: llama-cpp-python[server] in the shared venv (compiles
# llama.cpp — needs cmake; alternatively install a system llama-server
# binary, e.g. brew install llama.cpp, or set LLAMA_SERVER_BIN).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

LLM_VENV="mesh_server/.venv-llm"

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  say "installing llm deps into dedicated $LLM_VENV (mlx-lm + mlx-vlm)"
  if [ ! -x "$LLM_VENV/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
      run_step "create $LLM_VENV" uv venv "$LLM_VENV" --python 3.12
    elif [ -x ".venv/bin/uv" ]; then
      run_step "create $LLM_VENV" .venv/bin/uv venv "$LLM_VENV" --python 3.12
    else
      run_step "create $LLM_VENV" python3 -m venv "$LLM_VENV"
    fi
  fi
  LPY="$LLM_VENV/bin/python"
  if [ "$DRY" = "1" ]; then
    info "+ $LLM_VENV pip install mlx-lm>=0.32 mlx-vlm>=0.7"
  elif command -v uv >/dev/null 2>&1; then
    run_step "mlx-lm + mlx-vlm" uv pip install --python "$LPY" 'mlx-lm>=0.32' 'mlx-vlm>=0.7'
  elif [ -x ".venv/bin/uv" ]; then
    run_step "mlx-lm + mlx-vlm" .venv/bin/uv pip install --python "$LPY" 'mlx-lm>=0.32' 'mlx-vlm>=0.7'
  else
    run_step "mlx-lm + mlx-vlm" "$LPY" -m pip install 'mlx-lm>=0.32' 'mlx-vlm>=0.7'
  fi
  ok "llm venv ready: $LLM_VENV"
  info "the mlx adapter resolves it automatically (MESH_LLM_PYTHON overrides)"
else
  need_venv
  pkg_detect
  say "installing llm deps (llama.cpp) into shared $VENV"
  if [ -d "mesh_server" ]; then
    (cd mesh_server && uv pip install --python "../$VENV/bin/python" -e '.[llm]' -q) 2>/dev/null \
      || INSTALL 'llama-cpp-python[server]>=0.2.90' 'httpx>=0.27'
  else
    INSTALL 'llama-cpp-python[server]>=0.2.90' 'httpx>=0.27'
  fi
  ok "llm extras installed"
  info "gguf models additionally need llama-server on PATH or LLAMA_SERVER_BIN"
fi
