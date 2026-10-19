#!/usr/bin/env bash
# Stop everything started by scripts/start.sh.
#
# Usage:
#   scripts/stop.sh [--purge] [--no-color] [-h|--help]
#     --purge   also remove docker volumes (deletes the database + data volumes)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/ui.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

PURGE=0
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --purge)   PURGE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done

say "stopping cadx"

# native processes — pidfile_stop verifies the pid still belongs to the
# expected service before signalling (PID-reuse guard)
for name in mesh-server api worker executor web; do
  if [ -f "run/$name.pid" ]; then
    pidfile_stop "$name"
  fi
done

if docker compose ps -q 2>/dev/null | grep -q .; then
  if [ "$PURGE" = 1 ]; then
    run_step "docker stack down --purge" docker compose down -v || true
    warn "volumes removed — postgres data is gone"
  else
    run_step "docker stack down" docker compose down || true
  fi
else
  info "docker stack not running"
fi
if [ -f docker-compose.dev.yml ] \
    && docker compose -f docker-compose.dev.yml ps -q 2>/dev/null | grep -q .; then
  run_step "dev infra down" docker compose -f docker-compose.dev.yml down || true
fi

# leftover port squatters without pidfiles — report, don't kill blindly
for port in 8080 8000 5173 8765; do
  owner=$(port_owner "$port")
  [ -n "$owner" ] && warn "port $port still held by: $owner"
done

ok "all stopped"
