#!/usr/bin/env bash
# Install local-LLM backends into the mesh_server venv.
# Usage: mesh_server/scripts/install_llm.sh
#
# mlx tier (Apple Silicon): mlx-lm — pinned <0.30 because newer releases
# require transformers>=5, conflicting with the torch extras' <5 pin.
# Other tiers: llama-cpp-python[server] (compiles llama.cpp — needs cmake;
# alternatively install a system llama-server binary, e.g. brew install
# llama.cpp, or set LLAMA_SERVER_BIN).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh
need_venv
pkg_detect

say "installing llm deps (mlx-lm / llama.cpp)"
if [ -d "mesh_server" ]; then
  (cd mesh_server && uv pip install --python "../$VENV/bin/python" -e '.[llm]' -q) 2>/dev/null \
    || INSTALL 'mlx-lm>=0.24,<0.30; sys_platform=="darwin"' \
        'llama-cpp-python[server]>=0.2.90; sys_platform!="darwin"' 'httpx>=0.27'
else
  INSTALL 'mlx-lm>=0.24,<0.30; sys_platform=="darwin"' \
    'llama-cpp-python[server]>=0.2.90; sys_platform!="darwin"' 'httpx>=0.27'
fi
ok "llm extras installed"
info "gguf models additionally need llama-server on PATH or LLAMA_SERVER_BIN"
