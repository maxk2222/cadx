#!/usr/bin/env bash
# Stop everything started by scripts/start.sh.
# Usage: scripts/stop.sh [--purge]   (--purge also removes docker volumes)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

kill_pidfile() {
  local f="run/$1.pid"
  [ -f "$f" ] || return 0
  local pid; pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
    echo "stopped $1 (pid $pid)"
  fi
  rm -f "$f"
}

for name in mesh-server api worker executor web; do
  kill_pidfile "$name"
done

if docker compose ps -q 2>/dev/null | grep -q .; then
  if [ "$PURGE" = 1 ]; then
    docker compose down -v
    echo "docker stack down (volumes removed)"
  else
    docker compose down
    echo "docker stack down"
  fi
fi
if [ -f docker-compose.dev.yml ] && docker compose -f docker-compose.dev.yml ps -q 2>/dev/null | grep -q .; then
  docker compose -f docker-compose.dev.yml down
  echo "dev infra down"
fi
echo "all stopped"
