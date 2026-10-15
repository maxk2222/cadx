#!/usr/bin/env bash
# CADX launcher — runs setup if needed, then starts the full stack
# (docker by default, --dev for native) plus the mesh-server natively.
#
# Usage:
#   scripts/start.sh [--dev] [--tier T] [--fake] [--no-mesh] [--no-setup] [--bind-local]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/gpu_detect.sh

DEV=0; NO_MESH=0; NO_SETUP=0; BIND_LOCAL=0; TIER_OVERRIDE=""; FAKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dev)        DEV=1; shift ;;
    --tier)       TIER_OVERRIDE="$2"; shift 2 ;;
    --fake)       FAKE=1; shift ;;
    --no-mesh)    NO_MESH=1; shift ;;
    --no-setup)   NO_SETUP=1; shift ;;
    --bind-local) BIND_LOCAL=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$TIER_OVERRIDE" ] && export GPU_TIER="$TIER_OVERRIDE"
[ "$FAKE" = 1 ] && export GPU_TIER="${GPU_TIER:-fake}"

say()  { printf '\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
die()  { warn "$*"; exit 1; }

TIER=$(gpu_detect_tier)
MODEL=$(gpu_default_model "$TIER")
IMG_MODEL=$(gpu_default_image_model "$TIER")
mkdir -p run logs

# ---------- 1. setup check ----------
need_setup=0
[ ! -x .venv/bin/python ] && need_setup=1
[ ! -x mesh_server/.venv/bin/python ] && [ "$TIER" != "none" ] && need_setup=1
if [ "$need_setup" = 1 ] && [ "$NO_SETUP" = 0 ]; then
  say "first run — executing setup"
  args=( )
  [ "$DEV" = 1 ] && args+=(--dev)
  [ -n "$TIER_OVERRIDE" ] && args+=(--tier "$TIER_OVERRIDE")
  [ "$FAKE" = 1 ] && args+=(--fake)
  bash scripts/setup.sh "${args[@]}"
fi
[ -x .venv/bin/python ] || die "backend venv missing and setup skipped — run scripts/setup.sh"

# ---------- 2. mesh-server config ----------
# MESH_BASE_URL / MESH_MODEL are exported as SEED env: the app copies them into
# the DB runtime row on first boot; afterwards the Settings page is the source.
MESH_PORT=${MESH_PORT:-8765}
MESH_URL=""
if [ "$NO_MESH" = 0 ] && [ "$TIER" != "none" ]; then
  if [ "$DEV" = 1 ] || [ "$BIND_LOCAL" = 1 ]; then
    MESH_HOST="127.0.0.1"
    MESH_URL="http://localhost:$MESH_PORT"
  else
    # docker workers reach the host via host.docker.internal (docker0 gw on
    # linux, vpnkit on macOS) — linux needs a non-loopback bind
    if [ "$(uname -s)" = "Linux" ]; then
      MESH_HOST="0.0.0.0"
      warn "binding mesh-server to 0.0.0.0 (required so docker workers can reach it; visible on your LAN)"
    else
      MESH_HOST="127.0.0.1"
    fi
    MESH_URL="http://host.docker.internal:$MESH_PORT"
  fi
  export MESH_BASE_URL="$MESH_URL" MESH_MODEL="${MESH_MODEL:-$MODEL}"
  export MESH_IMAGE_MODEL="${MESH_IMAGE_MODEL:-$IMG_MODEL}"
  [ "$TIER" = "fake" ] && export MESH_FAKE=1
fi

# ---------- 3. start stack ----------
if [ "$DEV" = 1 ]; then
  say "infra (docker postgres+redis)"
  docker compose -f docker-compose.dev.yml up -d postgres redis
  say "migrations"
  (cd backend && ../.venv/bin/uv run alembic upgrade head >/dev/null)

  start_bg() { # name workdir command...
    local name="$1" dir="$2"; shift 2
    if [ -f "run/$name.pid" ] && kill -0 "$(cat "run/$name.pid")" 2>/dev/null; then
      info "$name already running (pid $(cat "run/$name.pid"))"
      return
    fi
    ( cd "$dir" && exec "$@" ) > "logs/$name.log" 2>&1 &
    echo $! > "run/$name.pid"
    info "$name started (pid $!, logs/$name.log)"
  }

  say "services (native)"
  start_bg api      backend  ../.venv/bin/uv run uvicorn cadx.main:app --port 8000
  start_bg worker   backend  ../.venv/bin/uv run arq cadx.queue.AgentWorkerSettings
  start_bg executor backend  ../.venv/bin/uv run arq cadx.queue.CadWorkerSettings
  start_bg web      frontend npm run dev -- --port 5173
  APP_URL="http://localhost:5173"
else
  say "docker stack"
  docker compose up -d --build
  say "waiting for api"
  # api is not published on :8000 in docker mode — probe through nginx
  for i in $(seq 1 40); do
    curl -sf http://localhost:8080/api/health >/dev/null 2>&1 && break
    sleep 3
  done
  curl -sf http://localhost:8080/api/health >/dev/null 2>&1 \
    || warn "api not healthy yet — check 'docker compose logs api'"
  APP_URL="http://localhost:8080"
fi

# ---------- 4. mesh-server ----------
if [ "${MESH_HOST:-}" != "" ]; then
  say "mesh-server ($TIER${MODEL:+, model=$MODEL})"
  PIDFILE=run/mesh-server.pid
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    info "already running (pid $(cat "$PIDFILE"))"
  else
    # ROCm: forward gfx override / allocator tuning when the user set them —
    # required for gfx archs not covered by the installed torch wheel
    MESH_HOST="$MESH_HOST" MESH_PORT="$MESH_PORT" MESH_DEVICE="$TIER" \
      MESH_FAKE="${MESH_FAKE:-}" \
      HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
      nohup mesh_server/.venv/bin/meshserver serve > logs/mesh-server.log 2>&1 &
    echo $! > "$PIDFILE"
  fi
  for i in $(seq 1 30); do
    curl -sf "http://localhost:$MESH_PORT/v1/health" >/dev/null 2>&1 && break
    sleep 2
  done
  if curl -sf "http://localhost:$MESH_PORT/v1/health" >/dev/null 2>&1; then
    info "mesh-server up at $MESH_URL (logs/mesh-server.log)"
  else
    warn "mesh-server did not answer /v1/health — check logs/mesh-server.log"
  fi
else
  info "mesh-server skipped ($([ "$NO_MESH" = 1 ] && echo '--no-mesh' || echo "tier=$TIER"))"
fi

say "ready"
info "app:   $APP_URL"
info "api:   http://localhost:8000"
[ "${MESH_HOST:-}" != "" ] && info "mesh:  $MESH_URL  (tier=$TIER)"
info "stop:  scripts/stop.sh    status: scripts/status.sh"
