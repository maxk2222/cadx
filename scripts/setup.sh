#!/usr/bin/env bash
# cadagent one-shot installer — detects the GPU tier and installs everything.
# Idempotent: re-running only performs the missing steps.
#
# Usage:
#   scripts/setup.sh [flags]
#
# Flags:
#   --tier mlx|cuda|none|fake   override GPU detection (testing)
#   --dev                       also install frontend deps for native dev mode
#   --fake                      enable the deterministic test adapter (tier none)
#   --extra mps                 additionally install torch/MPS adapters (Apple Silicon)
#   --skip-mesh                 do not install any mesh adapter
#   --dry-run                   print the plan, change nothing
#   --doctor                    print diagnostics and exit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/gpu_detect.sh

# ---------- flags ----------
TIER_OVERRIDE=""
DEV=0; FAKE=0; SKIP_MESH=0; DRY=0; DOCTOR=0; EXTRA_MPS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier)   TIER_OVERRIDE="$2"; shift 2 ;;
    --dev)    DEV=1; shift ;;
    --fake)   FAKE=1; shift ;;
    --extra)  [ "$2" = "mps" ] && EXTRA_MPS=1; shift 2 ;;
    --skip-mesh) SKIP_MESH=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --doctor) DOCTOR=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

[ -n "$TIER_OVERRIDE" ] && export GPU_TIER="$TIER_OVERRIDE"
TIER=$(gpu_detect_tier)
[ "$FAKE" = 1 ] && [ "$TIER" = "none" ] && TIER=fake
MODEL=$(gpu_default_model "$TIER")

say()  { printf '\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
run()  { if [ "$DRY" = 1 ]; then echo "  + $*"; else eval "$@"; fi; }

mesh_model_available() {
  local id="$1"
  mesh_server/.venv/bin/python - <<PY 2>/dev/null | grep -q "^1$"
import json
from meshserver.config import get_config
from meshserver.registry import Registry
reg = Registry(get_config())
avail = {m.id: m.available for m in reg.models()}
print(1 if avail.get("$id") else 0)
PY
}

if [ "$DOCTOR" = 1 ]; then
  say "doctor"
  info "tier:              $TIER  (override: ${GPU_TIER:-none})"
  info "vram:              $(gpu_vram_mb) MiB"
  info "default model:     ${MODEL:-<none>}"
  info "install script:    $(gpu_install_script "$TIER")"
  info "backend venv:      $([ -x .venv/bin/python ] && echo ok || echo MISSING)"
  info "mesh venv:         $([ -x mesh_server/.venv/bin/python ] && echo ok || echo MISSING)"
  info "node_modules:      $([ -d frontend/node_modules ] && echo ok || echo MISSING)"
  info "legacy .env:       $([ -f .env ] && echo 'present — will be imported once at next start' || echo none)"
  if [ -x mesh_server/.venv/bin/meshserver ]; then
    info "adapters:"
    mesh_server/.venv/bin/python - <<'PY' 2>/dev/null
from meshserver.config import get_config
from meshserver.registry import Registry
reg = Registry(get_config())
print("   device:", reg.device)
for m in reg.models():
    print(f"   {'x' if m.available else '-'} {m.id}  ({m.device})")
PY
  fi
  exit 0
fi

say "cadagent setup — tier: $TIER, model: ${MODEL:-none} $([ "$DRY" = 1 ] && echo '(dry-run)')"

# ---------- 1. prerequisites ----------
say "prerequisites"
MISSING=0
command -v python3 >/dev/null || { warn "python3 >= 3.11 required"; MISSING=1; }
[ "$(python3 -c 'import sys;print(sys.version_info>=(3,11))' 2>/dev/null)" = "True" ] \
  || { warn "python3 >= 3.11 required (found $(python3 --version 2>&1))"; MISSING=1; }
command -v docker >/dev/null && docker compose version >/dev/null 2>&1 \
  || warn "docker + compose plugin required for the docker stack (only needed in docker mode)"
[ "$DEV" = 1 ] && { command -v node >/dev/null || warn "node >= 20 required for --dev"; }
[ "$MISSING" = 1 ] && exit 1
info "ok"

# ---------- 2. backend venv ----------
say "backend env"
if [ ! -x .venv/bin/uv ]; then
  run "python3 -m venv .venv && .venv/bin/python -m pip install -q uv"
fi
run ".venv/bin/uv --directory backend lock -q"
run ".venv/bin/uv --directory backend sync -q"
info "ok"

# ---------- 3. mesh defaults ----------
# app runtime settings live in the DB (Settings page) — the launcher seeds
# mesh_base_url/mesh_model on first boot from these detected values.
case "$TIER" in
  mlx|cuda|fake)
    info "mesh default: $MODEL — start.sh will seed it into the app settings"
    ;;
  none)
    info "no GPU tier — mesh generation disabled (CAD fallback active)"
    ;;
esac

# ---------- 4. mesh-server venv + adapter ----------
if [ "$SKIP_MESH" = 0 ] && [ "$TIER" != "none" ]; then
  say "mesh-server env"
  if [ ! -x mesh_server/.venv/bin/python ]; then
    run "cd mesh_server && ../.venv/bin/uv venv .venv"
  fi
  run "cd mesh_server && ../.venv/bin/uv pip install -e '.' --python .venv/bin/python -q"
  if [ "$TIER" != "fake" ]; then
    SCRIPT=$(gpu_install_script "$TIER")
    if mesh_model_available "$MODEL"; then
      info "adapter '$MODEL' already available — skipping install"
    else
      say "installing adapter: $MODEL"
      run "VENV=mesh_server/.venv bash mesh_server/scripts/$SCRIPT"
    fi
    if [ "$EXTRA_MPS" = 1 ]; then
      say "extra: torch/MPS adapters (hunyuan3d-2mini/-mv)"
      run "VENV=mesh_server/.venv bash mesh_server/scripts/install_hunyuan_torch.sh"
    fi
  else
    info "fake tier — no adapter to install (MESH_FAKE=1)"
  fi
else
  [ "$SKIP_MESH" = 1 ] && info "--skip-mesh: skipping mesh-server"
fi

# ---------- 5. frontend (dev only) ----------
if [ "$DEV" = 1 ]; then
  say "frontend deps"
  run "cd frontend && npm ci --no-fund --no-audit"
fi

# ---------- 6. marker ----------
if [ "$DRY" = 0 ]; then
  mkdir -p run logs
  cat > run/setup.json <<EOF
{"tier":"$TIER","model":"$MODEL","dev":$DEV,"ts":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
fi

say "setup complete"
info "next: scripts/start.sh$([ "$DEV" = 1 ] && echo ' --dev')"
