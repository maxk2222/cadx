#!/usr/bin/env bash
# CADX launcher — runs setup if needed, then starts the full stack
# (docker by default, --dev for native) plus the mesh-server natively.
#
# Usage:
#   scripts/start.sh [--dev] [--tier T] [--fake] [--no-mesh] [--no-setup]
#                    [--no-build] [--bind-local] [--dry-run] [--open] [--yes]
#                    [--quiet|--verbose] [--no-color] [-h|--help]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/gpu_detect.sh
source scripts/lib/host_detect.sh
source scripts/lib/ui.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

DEV=0; NO_MESH=0; NO_SETUP=0; NO_BUILD=0; BIND_LOCAL=0; OPEN=0; TIER_OVERRIDE=""; FAKE=0
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --dev)        DEV=1; shift ;;
    --tier)       TIER_OVERRIDE="$2"; shift 2 ;;
    --fake)       FAKE=1; shift ;;
    --no-mesh)    NO_MESH=1; shift ;;
    --no-setup)   NO_SETUP=1; shift ;;
    --no-build)   NO_BUILD=1; shift ;;
    --bind-local) BIND_LOCAL=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    --open)       OPEN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done
[ -n "$TIER_OVERRIDE" ] && export GPU_TIER="$TIER_OVERRIDE"
[ "$FAKE" = 1 ] && export GPU_TIER="${GPU_TIER:-fake}"

log_init start
trap_setup "start"

TIER=$(gpu_detect_tier)
MODEL=$(gpu_default_model "$TIER")
IMG_MODEL=$(gpu_default_image_model "$TIER")
mkdir -p run logs
# run/mode records how the stack was started — update.sh restarts the same way
echo "$([ "$DEV" = 1 ] && echo dev || echo docker)" > run/mode

say "CADX start — tier: $TIER · mode: $([ "$DEV" = 1 ] && echo dev || echo docker)"
info "log: $LOG_FILE"

# ---------- 1. setup check ----------
step_begin "setup check"
need_setup=0
[ ! -x .venv/bin/python ] && need_setup=1
[ ! -x mesh_server/.venv/bin/python ] && [ "$TIER" != "none" ] && need_setup=1
if [ "$need_setup" = 1 ] && [ "$NO_SETUP" = 0 ]; then
  step_done "first run — executing setup"
  args=( )
  [ "$DEV" = 1 ] && args+=(--dev)
  [ -n "$TIER_OVERRIDE" ] && args+=(--tier "$TIER_OVERRIDE")
  [ "$FAKE" = 1 ] && args+=(--fake)
  [ "$YES" = 1 ] && args+=(--yes)
  bash scripts/setup.sh "${args[@]}"
elif [ "$need_setup" = 1 ]; then
  step_fail "env missing and --no-setup given"
  die "backend venv missing — run scripts/setup.sh"
else
  step_done "environment present"
fi
[ -x .venv/bin/python ] || die "backend venv missing and setup skipped — run scripts/setup.sh"

# ---------- 2. preflight ----------
step_begin "preflight"
PF_FAIL=0
# docker is needed in both modes (full stack or infra-only dev stack)
if ! command -v docker >/dev/null 2>&1; then
  warn "docker not installed — required for postgres/redis$([ "$DEV" = 0 ] && echo ' and the app stack')"
  PF_FAIL=1
elif ! docker info >/dev/null 2>&1; then
  warn "docker daemon not reachable — start Docker Desktop / dockerd first"
  PF_FAIL=1
fi
if [ "$DEV" = 1 ]; then
  need_cmd node "required for --dev (vite)" || PF_FAIL=1
  need_cmd npm || PF_FAIL=1
  pidfile_alive api  || check_port 8000 "api"  || PF_FAIL=1
  pidfile_alive web  || check_port 5173 "vite" || PF_FAIL=1
else
  # a port-8080 listener that IS our own compose frontend is not a conflict
  if docker compose ps --format '{{.Name}}' 2>/dev/null | grep -q frontend; then
    info "app stack already running — will refresh"
  else
    check_port 8080 "app" || PF_FAIL=1
  fi
fi
if [ "$NO_MESH" = 0 ] && [ "$TIER" != "none" ]; then
  if ! pidfile_alive mesh-server && ! port_free "${MESH_PORT:-8765}"; then
    if curl -sf --max-time 3 "http://localhost:${MESH_PORT:-8765}/v1/health" >/dev/null 2>&1; then
      info "mesh-server already answering on :${MESH_PORT:-8765} (foreign process) — will adopt, not restart"
      MESH_ADOPT=1
    else
      check_port "${MESH_PORT:-8765}" "mesh-server" || PF_FAIL=1
    fi
  fi
fi
[ "$PF_FAIL" = 1 ] && { step_fail "resolve the problems above"; exit 1; }
step_done

# ---------- 2b. updater token/url ----------
# The host-side updater daemon authenticates with a bearer token that must be
# identical for the daemon (env/file), the native dev api (inherited env) and
# the docker api (.env -> compose env). Precedence: .env > env > generated.
step_begin "updater token"
if [ ! -s run/updater.token ]; then
  python3 -c 'import secrets;print(secrets.token_hex(24))' > run/updater.token
  chmod 600 run/updater.token
fi
export CADX_UPDATE_TOKEN
CADX_UPDATE_TOKEN="${CADX_UPDATE_TOKEN:-$(cat run/updater.token)}"
export CADX_UPDATE_URL="http://127.0.0.1:${UPDATE_PORT:-8766}"
if [ "$DEV" = 0 ]; then
  touch .env
  if grep -q '^CADX_UPDATE_TOKEN=' .env; then
    CADX_UPDATE_TOKEN=$(grep '^CADX_UPDATE_TOKEN=' .env | tail -1 | cut -d= -f2-)
    export CADX_UPDATE_TOKEN
    printf '%s\n' "$CADX_UPDATE_TOKEN" > run/updater.token
    chmod 600 run/updater.token
  else
    printf 'CADX_UPDATE_TOKEN=%s\n' "$CADX_UPDATE_TOKEN" >> .env
  fi
  export CADX_UPDATE_URL="http://host.docker.internal:${UPDATE_PORT:-8766}"
fi
step_done "token ready"

# ---------- 3. mesh-server config ----------
# MESH_BASE_URL / MESH_MODEL are exported as SEED env: the app copies them
# into the DB runtime row on first boot; afterwards the Settings page wins.
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

# ---------- 4. start stack ----------
if [ "$DEV" = 1 ]; then
  run_step "infra (postgres+redis)" \
    docker compose -f docker-compose.dev.yml up -d postgres redis || exit 1
  run_step "migrations" \
    bash -c 'cd backend && ../.venv/bin/uv run alembic upgrade head' || exit 1

  step_begin "services (native)"
  pidfile_start api      backend  ../.venv/bin/uv run uvicorn cadx.main:app --port 8000
  pidfile_start worker   backend  ../.venv/bin/uv run arq cadx.queue.AgentWorkerSettings
  pidfile_start executor backend  ../.venv/bin/uv run arq cadx.queue.CadWorkerSettings
  pidfile_start web      frontend npm run dev -- --port 5173
  step_done
  APP_URL="http://localhost:5173"

  wait_for_url "http://localhost:8000/api/health" 60 api \
    || { warn "api not healthy — tail of logs/api.log:"; tail -15 logs/api.log >&2; }
  wait_for_url "http://localhost:5173" 60 vite \
    || { warn "vite not healthy — tail of logs/web.log:"; tail -15 logs/web.log >&2; }
  # arq workers have no HTTP port — verify they are still alive after boot
  if [ "$DRY" = 0 ]; then
    sleep 2
    for w in worker executor; do
      if pidfile_alive "$w"; then
        ok "$w alive (pid $(pidfile_pid "$w"))"
      else
        warn "$w died on boot — tail of logs/$w.log:"; tail -15 "logs/$w.log" >&2
      fi
    done
  fi
else
  # container cpu/mem caps follow the detected system (docker VM budget on
  # Desktop, host resources on native dockerd) — merged into .env only for
  # keys the user hasn't pinned, so overrides survive restarts
  step_begin "resource limits (auto)"
  host_detect_merge_env .env
  while IFS='=' read -r k v; do info "$k=$v"; done < <(host_detect_env)
  step_done
  if [ "$NO_BUILD" = 1 ]; then
    run_step "docker stack" docker compose up -d || exit 1
  else
    run_step "docker stack (build)" docker compose up -d --build || exit 1
  fi
  wait_for_url "http://localhost:8080/api/health" 120 api \
    || { warn "api not healthy yet — 'docker compose ps' and 'docker compose logs api':"; \
         docker compose ps 2>/dev/null; docker compose logs api --tail=20 2>/dev/null >&2 || true; }
  APP_URL="http://localhost:8080"
fi

# ---------- 5. mesh-server ----------
if [ "${MESH_HOST:-}" != "" ]; then
  if pidfile_alive mesh-server; then
    step_begin "mesh-server"
    step_skip "already running (pid $(pidfile_pid mesh-server))"
  elif [ "${MESH_ADOPT:-0}" = 1 ]; then
    step_begin "mesh-server"
    step_skip "already serving on :$MESH_PORT (adopted)"
  else
    step_begin "mesh-server ($TIER${MODEL:+, model=$MODEL})"
    if [ "$DRY" = 1 ]; then
      step_skip "dry-run: would nohup meshserver serve on $MESH_HOST:$MESH_PORT"
    else
      # ROCm: forward gfx override / allocator tuning when set — required for
      # gfx archs not covered by the installed torch wheel
      MESH_HOST="$MESH_HOST" MESH_PORT="$MESH_PORT" MESH_DEVICE="$TIER" \
        MESH_FAKE="${MESH_FAKE:-}" \
        MESH_JOB_MAX_RSS_MB="${MESH_JOB_MAX_RSS_MB:-}" \
        HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
        nohup mesh_server/.venv/bin/meshserver serve > logs/mesh-server.log 2>&1 &
      echo $! > run/mesh-server.pid
      step_done "pid $!"
    fi
  fi
  if wait_for_url "http://localhost:$MESH_PORT/v1/health" 60 mesh-server; then
    info "mesh-server at $MESH_URL (logs/mesh-server.log)"
    # adapter inventory — nice confirmation of what actually loaded
    if [ -x .venv/bin/python ] || command -v python3 >/dev/null 2>&1; then
      PYTHONPATH="" "$(command -v python3)" - "$MESH_PORT" <<'PY' 2>/dev/null || true
import json, sys, urllib.request
try:
    data = json.load(urllib.request.urlopen(
        f"http://localhost:{sys.argv[1]}/v1/models", timeout=5))
    avail = [m.get("id") for m in data.get("models", data if isinstance(data, list) else [])
             if m.get("available")]
    if avail:
        print("   adapters: " + ", ".join(avail))
except Exception:
    pass
PY
    fi
  else
    warn "mesh-server did not answer /v1/health — tail of logs/mesh-server.log:"
    tail -15 logs/mesh-server.log >&2 || true
  fi
else
  step_begin "mesh-server"
  step_skip "$([ "$NO_MESH" = 1 ] && echo '--no-mesh' || echo "tier=$TIER")"
fi

# ---------- 5b. updater daemon ----------
# Tiny stdlib http service that runs scripts/update.sh on the host so the web
# GUI can trigger a full-stack update (the api container cannot update itself).
step_begin "updater"
if pidfile_alive updater; then
  step_skip "already running (pid $(pidfile_pid updater))"
elif ! port_free "${UPDATE_PORT:-8766}"; then
  if curl -sf --max-time 3 "http://localhost:${UPDATE_PORT:-8766}/v1/health" >/dev/null 2>&1; then
    step_skip "already serving on :${UPDATE_PORT:-8766} (adopted)"
  else
    warn "port ${UPDATE_PORT:-8766} held by a foreign process — update button will be offline"
  fi
elif [ "$DRY" = 1 ]; then
  step_skip "dry-run: would start update_server.py"
else
  UPDATE_BIND="127.0.0.1"
  if [ "$DEV" = 0 ]; then
    # docker api reaches the host via host.docker.internal — on Linux that's
    # the docker0 gw, on macOS the host's LAN IP; a loopback bind is
    # unreachable either way. the bearer token gates all writes.
    UPDATE_BIND="0.0.0.0"
    warn "binding updater to 0.0.0.0 (required so the docker api can reach it; token-protected)"
  fi
  UPDATE_HOST="$UPDATE_BIND" UPDATE_PORT="${UPDATE_PORT:-8766}" \
    pidfile_start updater . python3 scripts/update_server.py
  wait_for_url "http://localhost:${UPDATE_PORT:-8766}/v1/health" 30 updater \
    || warn "updater did not answer — update button will be offline"
fi

# ---------- ready ----------
say "ready — service table"
printf '   %-22s %s\n' "app" "$APP_URL"
[ "$DEV" = 1 ] && printf '   %-22s %s\n' "api (native)" "http://localhost:8000"
[ "${MESH_HOST:-}" != "" ] && printf '   %-22s %s\n' "mesh-server" "$MESH_URL  (tier=$TIER)"
info "stop: scripts/stop.sh · status: scripts/status.sh"

if [ "$OPEN" = 1 ]; then
  if [ "$(uname -s)" = "Darwin" ]; then open "$APP_URL" || true
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$APP_URL" || true
  fi
fi
