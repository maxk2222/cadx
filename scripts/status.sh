#!/usr/bin/env bash
# Show status of the CADX stack + mesh-server.
#
# Usage:
#   scripts/status.sh [--watch] [--interval N] [--json] [--no-color] [-h|--help]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/gpu_detect.sh
source scripts/lib/ui.sh

usage() { sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; }

WATCH=0; INTERVAL=3; JSON=0
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --watch)    WATCH=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --json)     JSON=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done

http_check() { # label url -> prints state latency
  local label="$1" url="$2" out
  out=$(curl -sf --max-time 3 -o /dev/null -w '%{time_total}' "$url" 2>/dev/null)
  if [ -n "$out" ]; then
    printf '  %sUP  %s %-24s %-36s %sms\n' "$UI_G" "$UI_RST" "$label" "$url" \
      "$(python3 -c "print(int(float('$out')*1000))" 2>/dev/null || echo '?')"
  else
    printf '  %sDOWN%s %-24s %s\n' "$UI_R" "$UI_RST" "$label" "$url"
  fi
}

http_up() { curl -sf --max-time 3 "$1" >/dev/null 2>&1; }

proc_status() { # name -> "RUN pid etime" | "DEAD pid" | "-"
  local name="$1" pid etime
  pid=$(pidfile_pid "$name")
  if [ -z "$pid" ]; then echo "-"; return; fi
  if kill -0 "$pid" 2>/dev/null; then
    if pidfile_proc_matches "$name"; then
      etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
      echo "RUN $pid ${etime:-?}"
    else
      echo "STALE $pid (pid reused by foreign process)"
    fi
  else
    echo "DEAD $pid"
  fi
}

render() {
  local now
  now=$(date '+%Y-%m-%d %H:%M:%S')
  say "cadx status — $now"
  info "tier: $(gpu_detect_tier)   model: $(gpu_default_model)"

  say "native processes"
  local found=0 name st
  for name in api worker executor web mesh-server; do
    st=$(proc_status "$name")
    [ "$st" = "-" ] && continue
    found=1
    case "$st" in
      RUN*)   printf '  %sRUN  %s %-14s %s\n' "$UI_G" "$UI_RST" "$name" "${st#RUN }" ;;
      STALE*) printf '  %sWARN %s %-14s %s\n' "$UI_Y" "$UI_RST" "$name" "${st#STALE }" ;;
      *)      printf '  %sDEAD %s %-14s %s\n' "$UI_R" "$UI_RST" "$name" "${st#DEAD }" ;;
    esac
  done
  [ "$found" = 0 ] && info "none (no pidfiles)"

  say "http"
  http_check "app (nginx->api)" http://localhost:8080/api/health
  http_check "api (native dev)" http://localhost:8000/api/health
  http_check "vite (native dev)" http://localhost:5173
  http_check "mesh-server"      http://localhost:8765/v1/health

  say "docker"
  docker compose ps --format '  {{.Name}}: {{.Status}}' 2>/dev/null \
    | grep . || info "not running"

  if http_up http://localhost:8765/v1/health; then
    say "mesh adapters"
    python3 - <<'PY' 2>/dev/null || info "(could not query /v1/models)"
import json, urllib.request
data = json.load(urllib.request.urlopen("http://localhost:8765/v1/models", timeout=5))
models = data.get("models", data if isinstance(data, list) else [])
for m in models:
    mark = "x" if m.get("available") else "-"
    print(f"   {mark} {m.get('id')}  ({m.get('device')})")
PY
  fi

  say "resources"
  info "data/: $(du -sm data 2>/dev/null | awk '{print $1}') MiB   disk free: $(disk_free_mb .) MiB"
}

render_json() {
  python3 - <<'PY' 2>/dev/null || echo '{"error":"json render failed"}'
import json, os, signal, subprocess, time

def http(url):
    import urllib.request
    t0 = time.time()
    try:
        urllib.request.urlopen(url, timeout=3)
        return {"up": True, "ms": int((time.time() - t0) * 1000)}
    except Exception:
        return {"up": False}

def proc(name):
    f = f"run/{name}.pid"
    if not os.path.exists(f):
        return {"state": "absent"}
    pid = int(open(f).read().strip() or 0)
    try:
        os.kill(pid, 0)
        return {"state": "running", "pid": pid}
    except OSError:
        return {"state": "dead", "pid": pid}

ps = subprocess.run(
    ["docker", "compose", "ps", "--format", "{{.Name}} {{.Status}}"],
    capture_output=True, text=True)
out = {
    "tier": subprocess.run(
        ["bash", "-c", "source scripts/lib/gpu_detect.sh && gpu_detect_tier"],
        capture_output=True, text=True).stdout.strip(),
    "processes": {n: proc(n) for n in
                  ("api", "worker", "executor", "web", "mesh-server")},
    "http": {
        "app": http("http://localhost:8080/api/health"),
        "api": http("http://localhost:8000/api/health"),
        "vite": http("http://localhost:5173"),
        "mesh": http("http://localhost:8765/v1/health"),
    },
    "docker": [l for l in ps.stdout.splitlines() if l.strip()],
}
print(json.dumps(out, indent=2))
PY
}

if [ "$JSON" = 1 ]; then
  render_json
elif [ "$WATCH" = 1 ]; then
  while :; do
    clear
    render
    printf '\n  %srefresh every %ss — Ctrl-C to quit%s\n' "$UI_DIM" "$INTERVAL" "$UI_RST"
    sleep "$INTERVAL"
  done
else
  render
fi
