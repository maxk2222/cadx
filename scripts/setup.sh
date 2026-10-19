#!/usr/bin/env bash
# CADX one-shot installer — detects the GPU tier and installs everything.
# Idempotent: re-running only performs the missing steps.
#
# Usage:
#   scripts/setup.sh [flags]
#
# Flags:
#   --tier mlx|cuda|rocm|none|fake   override GPU detection (testing)
#   --dev                       also install frontend deps for native dev mode
#   --fake                      enable the deterministic test adapter (tier none)
#   --extra mps                 additionally install torch/MPS adapters (Apple Silicon)
#   --extra hunyuan             additionally install Hunyuan3D-2 torch adapters (ROCm tier;
#                               adds multiview — note the Tencent license restrictions)
#   --skip-mesh                 do not install any mesh adapter
#   --skip-image                do not install a text-to-image adapter
#   --dry-run                   print the plan, change nothing
#   --doctor                    print diagnostics and exit
#   -y, --yes                   assume yes for interactive confirmations
#   -q, --quiet                 only errors + summary
#   -v, --verbose               stream command output live instead of spinners
#   --no-color                  disable ANSI colors/spinners
#   -h, --help                  show this help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/gpu_detect.sh
source scripts/lib/pkg.sh   # also pulls in ui.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

# ---------- flags ----------
TIER_OVERRIDE=""
DEV=0; FAKE=0; SKIP_MESH=0; SKIP_IMAGE=0; DOCTOR=0; EXTRA_MPS=0; EXTRA_HUNYUAN=0
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --tier)   TIER_OVERRIDE="$2"; shift 2 ;;
    --dev)    DEV=1; shift ;;
    --fake)   FAKE=1; shift ;;
    --extra)  case "$2" in
                mps)     EXTRA_MPS=1 ;;
                hunyuan) EXTRA_HUNYUAN=1 ;;
                *) err "unknown --extra: $2"; exit 2 ;;
              esac; shift 2 ;;
    --skip-mesh)  SKIP_MESH=1; shift ;;
    --skip-image) SKIP_IMAGE=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    --doctor)     DOCTOR=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done

[ -n "$TIER_OVERRIDE" ] && export GPU_TIER="$TIER_OVERRIDE"
TIER=$(gpu_detect_tier)
[ "$FAKE" = 1 ] && [ "$TIER" = "none" ] && TIER=fake
if [ "$EXTRA_HUNYUAN" = 1 ] && [ "$TIER" != "rocm" ]; then
  warn "--extra hunyuan only applies to the rocm tier (on mlx use --extra mps)"
fi
MODEL=$(gpu_default_model "$TIER")
IMG_MODEL=$(gpu_default_image_model "$TIER")

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

# ---------- doctor ----------
if [ "$DOCTOR" = 1 ]; then
  say "doctor — $(uname -sm), bash ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"
  info "tier:              $TIER  (override: ${GPU_TIER:-none})"
  info "vram:              $(gpu_vram_mb) MiB"
  if [ "$TIER" = "rocm" ]; then
    GFX=$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1 || true)
    [ -z "$GFX" ] && GFX=$(amd-smi list 2>/dev/null | grep -io 'gfx[0-9a-f]*' | head -1 || true)
    info "gfx arch:          ${GFX:-unknown (install rocminfo for details)}"
  fi
  info "default model:     ${MODEL:-<none>}"
  info "install script:    $(gpu_install_script "$TIER")"
  info "image model:       ${IMG_MODEL:-<none>}"
  info "image installer:   $(gpu_image_install_script "$TIER")"
  info "disk free:         $(disk_free_mb .) MiB   (need ~$(tier_disk_need_mb "$TIER") MiB)"
  say "tools"
  for t in python3 git curl docker node npm uv; do
    if command -v "$t" >/dev/null 2>&1; then
      info "$(printf '%-18s' "$t") $(command -v "$t")"
    else
      info "$(printf '%-18s' "$t") MISSING"
    fi
  done
  command -v docker >/dev/null 2>&1 && \
    info "$(printf '%-18s' "docker daemon") $(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo 'not running')"
  say "environment"
  info "backend venv:      $([ -x .venv/bin/python ] && echo ok || echo MISSING)"
  info "mesh venv:         $([ -x mesh_server/.venv/bin/python ] && echo ok || echo MISSING)"
  info "node_modules:      $([ -d frontend/node_modules ] && echo ok || echo MISSING)"
  info ".env overrides:    $([ -f .env ] && echo 'present — used by docker compose for POSTGRES_* etc.' || echo none)"
  info "last setup log:    $(ls -t logs/setup-*.log 2>/dev/null | head -1 || echo none)"
  if [ -f run/setup.json ]; then
    info "setup marker:      $(cat run/setup.json)"
  fi
  if [ -x mesh_server/.venv/bin/meshserver ]; then
    say "adapters"
    mesh_server/.venv/bin/python - <<'PY' 2>/dev/null | grep -E '^   '
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

log_init setup
trap_setup "setup"

say "CADX setup — tier: $TIER · model: ${MODEL:-none}$([ "$DRY" = 1 ] && echo ' · dry-run')"
info "log: $LOG_FILE"
case "$TIER" in
  mlx|cuda|rocm|fake)
    # app runtime settings live in the DB (Settings page) — the launcher
    # seeds mesh_base_url/mesh_model on first boot from detected values.
    info "mesh default: $MODEL — start.sh will seed it into the app settings"
    [ -n "$IMG_MODEL" ] && info "image default: $IMG_MODEL"
    ;;
  none)
    info "no GPU tier — mesh generation disabled (CAD fallback active)"
    ;;
esac

steps_total 7

# ---------- 1. prerequisites ----------
step_begin "prerequisites"
MISSING=0
need_cmd python3 || MISSING=1
[ "$(python3 -c 'import sys;print(sys.version_info>=(3,11))' 2>/dev/null)" = "True" ] \
  || { warn "python3 >= 3.11 required (found $(python3 --version 2>&1))"; MISSING=1; }
need_cmd git "needed to clone model adapter sources" || MISSING=1
need_cmd curl "needed for health checks and downloads" || MISSING=1
if command -v docker >/dev/null 2>&1; then
  docker info >/dev/null 2>&1 \
    || warn "docker CLI present but daemon not reachable (needed for the docker stack)"
  docker compose version >/dev/null 2>&1 \
    || warn "docker compose plugin missing (only needed in docker mode)"
else
  warn "docker + compose plugin required for the docker stack (only needed in docker mode)"
fi
if [ "$DEV" = 1 ]; then
  if command -v node >/dev/null 2>&1; then
    NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)
    [ "$NODE_MAJOR" -ge 20 ] 2>/dev/null \
      || { warn "node >= 20 required for --dev (found $(node --version))"; MISSING=1; }
  else
    warn "node >= 20 required for --dev"; MISSING=1
  fi
fi
# disk space vs tier estimate
NEED_MB=$(tier_disk_need_mb "$TIER"); FREE_MB=$(disk_free_mb .)
if [ "$FREE_MB" -lt "$NEED_MB" ]; then
  warn "low disk: ${FREE_MB} MiB free, tier '$TIER' needs ~${NEED_MB} MiB (deps + model weights)"
  confirm "continue anyway?" n || die "aborted — free up disk space or use --skip-mesh/--skip-image"
fi
# HF reachability is only a warning — the failure would surface at download time
if [ "$SKIP_MESH" = 0 ] && [ "$TIER" != "none" ] && [ "$TIER" != "fake" ]; then
  curl -sf --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null \
    || warn "huggingface.co unreachable — adapter weight downloads will fail"
fi
[ "$MISSING" = 1 ] && { step_fail "missing prerequisites"; exit 1; }
step_done "$(python3 --version 2>&1 | awk '{print $2}') · ${FREE_MB} MiB free"

# ---------- 2. backend venv ----------
if [ ! -x .venv/bin/uv ]; then
  run_step "backend venv" bash -c 'python3 -m venv .venv && .venv/bin/python -m pip install -q uv' \
    || exit 1
else
  step_begin "backend venv"; step_skip ".venv already present"
fi
run_step "backend deps" bash -c '.venv/bin/uv --directory backend lock -q && .venv/bin/uv --directory backend sync -q' \
  || exit 1

# ---------- 3. mesh-server venv ----------
if [ "$SKIP_MESH" = 0 ] && [ "$TIER" != "none" ]; then
  if [ ! -x mesh_server/.venv/bin/python ]; then
    run_step "mesh-server venv" bash -c 'cd mesh_server && ../.venv/bin/uv venv .venv' \
      || exit 1
  fi
  run_step "mesh-server deps" bash -c 'cd mesh_server && ../.venv/bin/uv pip install -e . --python .venv/bin/python -q' \
    || exit 1
else
  step_begin "mesh-server env"
  step_skip "$([ "$SKIP_MESH" = 1 ] && echo '--skip-mesh' || echo "tier=$TIER")"
fi

# ---------- 4. mesh adapter ----------
if [ "$SKIP_MESH" = 0 ] && [ "$TIER" != "none" ]; then
  case "$TIER" in
    fake)
      step_begin "mesh adapter"; step_skip "fake tier — no adapter to install (MESH_FAKE=1)"
      ;;
    *)
      SCRIPT=$(gpu_install_script "$TIER")
      MODEL_THERE=0
      if mesh_model_available "$MODEL"; then
        step_begin "mesh adapter"; step_skip "'$MODEL' already available"
        MODEL_THERE=1
      else
        # rocm: install_rocm.sh delegates to install_hunyuan_torch.sh when
        # HUNYUAN=1 — the hunyuan torch build needs the ROCm wheel first
        run_step "mesh adapter: $MODEL" \
          env "VENV=mesh_server/.venv" "HUNYUAN=$EXTRA_HUNYUAN" \
            bash "mesh_server/scripts/$SCRIPT" || exit 1
      fi
      if [ "$EXTRA_MPS" = 1 ]; then
        run_step "extra: torch/MPS adapters (hunyuan3d-2mini/-mv)" \
          env "VENV=mesh_server/.venv" bash mesh_server/scripts/install_hunyuan_torch.sh \
          || exit 1
      fi
      # rocm + --extra hunyuan on a re-run (triposg already present ⇒ rocm
      # torch already in place): install the hunyuan adapters directly
      if [ "$EXTRA_HUNYUAN" = 1 ] && [ "$TIER" = "rocm" ] && [ "$MODEL_THERE" = 1 ]; then
        if mesh_model_available "hunyuan3d-2mini"; then
          step_begin "extra: hunyuan adapters"; step_skip "already available"
        else
          run_step "extra: Hunyuan3D-2 torch on ROCm" \
            env "VENV=mesh_server/.venv" bash mesh_server/scripts/install_hunyuan_torch.sh \
            || exit 1
        fi
      fi
      ;;
  esac
else
  step_begin "mesh adapter"
  step_skip "$([ "$SKIP_MESH" = 1 ] && echo '--skip-mesh' || echo 'no GPU tier — CAD fallback active')"
fi

# ---------- 5. image adapter ----------
IMG_SCRIPT=$(gpu_image_install_script "$TIER")
if [ "$SKIP_MESH" = 0 ] && [ "$SKIP_IMAGE" = 0 ] && [ -n "$IMG_SCRIPT" ] && [ -n "$IMG_MODEL" ]; then
  if mesh_model_available "$IMG_MODEL"; then
    step_begin "image adapter"; step_skip "'$IMG_MODEL' already available"
  else
    run_step "image adapter: $IMG_MODEL" \
      env "VENV=mesh_server/.venv" bash "mesh_server/scripts/$IMG_SCRIPT" || exit 1
  fi
else
  step_begin "image adapter"
  step_skip "$([ "$SKIP_IMAGE" = 1 ] && echo '--skip-image' || echo 'none for this tier')"
fi

# ---------- 6. frontend deps (dev only) ----------
if [ "$DEV" = 1 ]; then
  run_step "frontend deps" npm ci --prefix frontend --no-fund --no-audit || exit 1
else
  step_begin "frontend deps"; step_skip "only installed with --dev"
fi

# ---------- marker + summary ----------
if [ "$DRY" = 0 ]; then
  mkdir -p run logs
  cat > run/setup.json <<EOF
{"tier":"$TIER","model":"$MODEL","dev":$DEV,"ts":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
fi

print_summary
say "setup complete"
info "next: scripts/start.sh$([ "$DEV" = 1 ] && echo ' --dev')"
