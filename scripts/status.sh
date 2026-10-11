#!/usr/bin/env bash
# Show status of the cadagent stack + mesh-server.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/gpu_detect.sh

check() { # label url
  if curl -sf --max-time 3 "$2" >/dev/null 2>&1; then
    printf '  \033[32mUP  \033[0m %-28s %s\n' "$1" "$2"
  else
    printf '  \033[31mDOWN\033[0m %-28s %s\n' "$1" "$2"
  fi
}

echo "tier: $(gpu_detect_tier)   model: $(gpu_default_model)"
echo "processes:"
for f in run/*.pid; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .pid); pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    printf '  \033[32mRUN \033[0m %-28s pid %s\n' "$name" "$pid"
  else
    printf '  \033[31mDEAD\033[0m %-28s pid %s\n' "$name" "$pid"
  fi
done
echo "http:"
check "app (nginx->api)" http://localhost:8080/api/health
check "api (native dev)" http://localhost:8000/api/health
check "vite (native dev)" http://localhost:5173
check "mesh-server"      http://localhost:8765/v1/health
echo "docker:"
docker compose ps --format '  {{.Name}}: {{.Status}}' 2>/dev/null || echo "  (not running)"
