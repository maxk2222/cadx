#!/usr/bin/env bash
# CADX shared UI toolkit for setup/start/status/stop scripts.
# bash 3.2 compatible (macOS system bash) — no assoc arrays, no wait -n.
#
# Usage:
#   source scripts/lib/ui.sh          # auto-detects TTY/color
#   ui_init [name]                    # set log file context (logs/<name>-<ts>.log)
#
# Output primitives:  say info ok warn err die
# Steps:              step_begin <label> / step_done / step_skip / step_fail
# Live runs:          run_step <label> cmd args...
# Wait:               wait_for_url <url> <timeout_s> <label>
# Misc:               confirm port_owner port_free disk_free_mb need_cmd
#                     pidfile_alive pidfile_pid pidfile_start pidfile_stop
#                     last_log_line print_summary trap_setup
#
# Env:
#   NO_COLOR / CI / TERM=dumb  -> disables colors + spinners automatically
#   CADX_NO_SPIN=1             -> spinners off, plain log lines
#   QUIET / VERBOSE            -> set by --quiet/--verbose flags in callers

# ---------- terminal detection ----------

ui_detect() {
  UI_TTY=0
  if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ] && [ -z "${NO_COLOR:-}" ] \
      && [ -z "${CI:-}" ]; then
    UI_TTY=1
  fi
  UI_SPIN=$UI_TTY
  [ "${CADX_NO_SPIN:-0}" = "1" ] && UI_SPIN=0
  if [ "$UI_TTY" = 1 ]; then
    UI_B=$'\033[1m'; UI_DIM=$'\033[2m'; UI_RST=$'\033[0m'
    UI_R=$'\033[31m'; UI_G=$'\033[32m'; UI_Y=$'\033[33m'
    UI_Bl=$'\033[34m'; UI_C=$'\033[36m'; UI_M=$'\033[35m'
  else
    UI_B=""; UI_DIM=""; UI_RST=""
    UI_R=""; UI_G=""; UI_Y=""; UI_Bl=""; UI_C=""; UI_M=""
  fi
  # spinner glyphs: braille on UTF-8 locales, ASCII otherwise
  if printf '%s' "${LC_ALL:-${LANG:-}}" | grep -qi 'utf-8\|utf8'; then
    UI_FRAMES=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)
  else
    UI_FRAMES=('-' '\' '|' '/')
  fi
  : "${QUIET:=0}" "${VERBOSE:=0}" "${DRY:=0}"
}

ui_detect

# ---------- logging ----------

LOG_FILE=""
log_init() {
  # log_init <name> -> sets LOG_FILE=logs/<name>-<ts>.log, refreshes
  # logs/<name>-latest.log copy. Idempotent: re-init switches files.
  local name="$1" ts
  ts="$(date +%Y%m%d-%H%M%S)"
  mkdir -p logs
  LOG_FILE="logs/${name}-${ts}.log"
  : > "$LOG_FILE"
  ln -sf "$LOG_FILE" "logs/${name}-latest.log" 2>/dev/null \
    || cp "$LOG_FILE" "logs/${name}-latest.log" 2>/dev/null || true
}

_log_line() {
  # append a rendered line to the log file (when set)
  [ -n "$LOG_FILE" ] && printf '%s\n' "$*" >> "$LOG_FILE" || true
}

last_log_line() {
  # last_log_line [file] — normalize \r-progress (tqdm/git/docker) to lines,
  # return the most recent non-empty one, trimmed to terminal width.
  local f="${1:-$LOG_FILE}" cols
  [ -f "$f" ] || return 0
  cols=$(tput cols 2>/dev/null || echo 80)
  tail -c 8192 "$f" 2>/dev/null | tr '\r' '\n' | sed 's/[[:space:]]*$//' \
    | grep . | tail -1 | cut -c1-$((cols > 20 ? cols - 10 : 70))
}

# ---------- output primitives ----------

say()  { printf '%s== %s%s\n' "$UI_B" "$*" "$UI_RST"; _log_line "== $*"; }
info() { [ "$QUIET" = 1 ] || printf '   %s\n' "$*"; _log_line "   $*"; }
ok()   { printf '   %s✓%s %s\n' "$UI_G" "$UI_RST" "$*"; _log_line "   ok $*"; }
warn() { printf '%s!%s %s\n' "$UI_Y" "$UI_RST" "$*" >&2; _log_line "!  $*"; }
err()  { printf '%s✗%s %s\n' "$UI_R" "$UI_RST" "$*" >&2; _log_line "err $*"; }
die()  { err "$*"; exit 1; }

# ---------- steps ----------

STEP_I=0
STEP_N=0
STEP_LABEL=""
STEP_T0=0
STEP_LABELS=()
STEP_SECS=()
STEP_RESULTS=()

steps_total() { STEP_N="$1"; STEP_I=0; STEP_LABELS=(); STEP_SECS=(); STEP_RESULTS=(); }

_step_prefix() {
  if [ "$STEP_N" -gt 0 ]; then
    printf '[%d/%d]' "$STEP_I" "$STEP_N"
  else
    printf '*'
  fi
}

step_begin() {
  # step_begin <label>
  STEP_I=$((STEP_I + 1))
  STEP_LABEL="$1"
  STEP_T0=$(date +%s)
  if [ "$UI_SPIN" = 0 ]; then
    printf '%s%s %s%s\n' "$UI_B" "$(_step_prefix)" "$STEP_LABEL" "$UI_RST"
  else
    printf '%s%s %s%s ' "$UI_B" "$(_step_prefix)" "$STEP_LABEL" "$UI_RST"
  fi
  _log_line "-- step $(_step_prefix) $STEP_LABEL"
}

_step_record() {
  local result="$1" now
  now=$(date +%s)
  STEP_LABELS+=("$STEP_LABEL")
  STEP_SECS+=($((now - STEP_T0)))
  STEP_RESULTS+=("$result")
}

step_done() {
  # step_done [note]
  local note="${1:-}" elapsed
  elapsed=$(( $(date +%s) - STEP_T0 ))
  if [ "$UI_SPIN" = 0 ]; then
    printf '   %s✓%s %ss%s\n' "$UI_G" "$UI_RST" "${elapsed}" \
      "${note:+ — $note}"
  else
    printf '%s✓%s (%ss)%s\n' "$UI_G" "$UI_RST" "$elapsed" \
      "${note:+ — $note}"
  fi
  _log_line "-- done: $STEP_LABEL (${elapsed}s)${note:+ — $note}"
  _step_record "done"
}

step_skip() {
  local note="${1:-skipped}"
  if [ "$UI_SPIN" = 0 ]; then
    printf '   %s-%s %s\n' "$UI_DIM" "$UI_RST" "$note"
  else
    printf '%s-%s %s\n' "$UI_DIM" "$UI_RST" "$note"
  fi
  _log_line "-- skip: $STEP_LABEL — $note"
  _step_record "skip"
}

step_fail() {
  # step_fail [note] — prints ✗ plus the last lines of LOG_FILE
  local note="${1:-}" elapsed lines
  elapsed=$(( $(date +%s) - STEP_T0 ))
  if [ "$UI_SPIN" = 0 ]; then
    printf '   %s✗%s %ss%s\n' "$UI_R" "$UI_RST" "${elapsed}" \
      "${note:+ — $note}" >&2
  else
    printf '%s✗%s (%ss)%s\n' "$UI_R" "$UI_RST" "$elapsed" \
      "${note:+ — $note}" >&2
  fi
  if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    lines=$(tail -15 "$LOG_FILE" | sed 's/[[:space:]]*$//')
    [ -n "$lines" ] && printf '%s\n' "$lines" | sed 's/^/      /' >&2
    printf '   %ssee %s%s\n' "$UI_DIM" "$LOG_FILE" "$UI_RST" >&2
  fi
  _log_line "-- FAIL: $STEP_LABEL (${elapsed}s)${note:+ — $note}"
  _step_record "fail"
}

print_summary() {
  # print_summary — table of recorded steps with durations
  [ "${#STEP_LABELS[@]}" -eq 0 ] && return 0
  local i total=0 res mark
  say "summary"
  for i in "${!STEP_LABELS[@]}"; do
    res="${STEP_RESULTS[$i]}"
    case "$res" in
      done) mark="${UI_G}✓${UI_RST}" ;;
      skip) mark="${UI_DIM}-${UI_RST}" ;;
      fail) mark="${UI_R}✗${UI_RST}" ;;
      *)    mark="?" ;;
    esac
    printf '   %s %-44s %ss\n' "$mark" "${STEP_LABELS[$i]}" "${STEP_SECS[$i]}"
    total=$((total + STEP_SECS[$i]))
  done
  info "total: ${total}s"
}

# ---------- run_step: spinner + live log ----------

_ui_hide_cursor()  { [ "$UI_SPIN" = 1 ] && printf '\033[?25l' || true; }
_ui_show_cursor()  { [ "$UI_TTY" = 1 ] && printf '\033[?25h' || true; }

run_step() {
  # run_step <label> cmd args...
  # Runs cmd with stdout+stderr appended to LOG_FILE. On TTY shows a spinner
  # line with elapsed time and the command's latest log line. Returns the
  # command's exit code (does NOT exit itself).
  local label="$1"; shift
  local t0 rc line frame_i=0 pid
  step_begin "$label"
  t0=$(date +%s)

  if [ "$DRY" = "1" ]; then
    step_skip "dry-run: $*"
    return 0
  fi

  if [ -n "$LOG_FILE" ]; then
    printf '\n$ %s\n' "$*" >> "$LOG_FILE"
  fi

  if [ "$VERBOSE" = 1 ]; then
    if [ -n "$LOG_FILE" ]; then
      "$@" 2>&1 | tee -a "$LOG_FILE" && rc=0 || rc=$?
    else
      "$@" && rc=0 || rc=$?
    fi
    [ "$rc" = 0 ] && step_done || step_fail
    return "$rc"
  fi

  if [ -n "$LOG_FILE" ]; then
    "$@" >> "$LOG_FILE" 2>&1 &
  else
    "$@" > /dev/null 2>&1 &
  fi
  pid=$!

  if [ "$UI_SPIN" = 1 ]; then
    _ui_hide_cursor
    while kill -0 "$pid" 2>/dev/null; do
      frame_i=$(( (frame_i + 1) % ${#UI_FRAMES[@]} ))
      line=$(last_log_line)
      printf '\r\033[K%s %s %s%s %ss%s' \
        "$(_step_prefix)" "$label" \
        "$UI_C" "${UI_FRAMES[$frame_i]}" "$UI_RST" \
        "$(( $(date +%s) - t0 ))" "${line:+  ${UI_DIM}${line}${UI_RST}}"
      sleep 0.15 2>/dev/null || sleep 1
    done
    printf '\r\033[K'
    _ui_show_cursor
    # re-print step header (spinner overwrote the line)
    printf '%s%s %s%s ' "$UI_B" "$(_step_prefix)" "$label" "$UI_RST"
  else
    while kill -0 "$pid" 2>/dev/null; do
      sleep 0.3 2>/dev/null || sleep 1
    done
  fi

  wait "$pid" 2>/dev/null && rc=0 || rc=$?
  if [ "$rc" = 0 ]; then
    step_done
  else
    step_fail "exit $rc"
  fi
  return "$rc"
}

# ---------- wait_for_url ----------

wait_for_url() {
  # wait_for_url <url> <timeout_s> <label> — countdown wait, returns 0 on
  # first successful curl, 1 on timeout. Dry-run skips instantly.
  local url="$1" timeout="$2" label="${3:-service}" t0 elapsed
  [ "$DRY" = "1" ] && { info "dry-run: would wait for $label ($url)"; return 0; }
  t0=$(date +%s)
  if [ "$UI_SPIN" = 1 ]; then
    _ui_hide_cursor
    while :; do
      if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
        printf '\r\033[K'
        _ui_show_cursor
        ok "$label up ($url)"
        _log_line "wait_for_url ok: $url"
        return 0
      fi
      elapsed=$(( $(date +%s) - t0 ))
      [ "$elapsed" -ge "$timeout" ] && break
      printf '\r\033[K%s…%s waiting for %s %ss/%ss' \
        "$UI_C" "$UI_RST" "$label" "$elapsed" "$timeout"
      sleep 2
    done
    printf '\r\033[K'
    _ui_show_cursor
  else
    while :; do
      if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
        ok "$label up ($url)"
        _log_line "wait_for_url ok: $url"
        return 0
      fi
      elapsed=$(( $(date +%s) - t0 ))
      [ "$elapsed" -ge "$timeout" ] && break
      sleep 2
    done
  fi
  _log_line "wait_for_url TIMEOUT: $url after ${elapsed}s"
  return 1
}

# ---------- prompts ----------

confirm() {
  # confirm <question> [default y|n] — respects --yes (YES=1) and non-TTY
  local q="$1" default="${2:-n}" ans
  if [ "${YES:-0}" = 1 ] || [ "$UI_TTY" = 0 ]; then
    [ "$default" = "y" ]
    return
  fi
  printf '%s?%s %s [%s] ' "$UI_Y" "$UI_RST" "$q" "$default"
  read -r ans
  ans="${ans:-$default}"
  case "$ans" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

# ---------- system probes ----------

need_cmd() {
  # need_cmd <cmd> [hint] — returns 0 if present; prints warn + hint if not
  local c="$1" hint="${2:-}"
  if command -v "$c" >/dev/null 2>&1; then return 0; fi
  warn "missing required tool: $c${hint:+ — $hint}"
  return 1
}

port_owner() {
  # port_owner <port> -> "pid comm" of the listener, empty when free
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $2" "$1; exit}'
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $6; exit}'
  fi
}

port_free() {
  [ -z "$(port_owner "$1")" ]
}

disk_free_mb() {
  # disk_free_mb <path> — free space in MiB (POSIX df -m)
  local out
  out=$(df -m "$1" 2>/dev/null | awk 'NR==2 {print $4}')
  echo "${out:-0}"
}

check_port() {
  # check_port <port> <label> — warn (and fail caller) when held by a
  # foreign process; prints owner details.
  local port="$1" label="$2" owner
  owner=$(port_owner "$port")
  if [ -n "$owner" ]; then
    warn "port $port ($label) already in use by: $owner"
    return 1
  fi
  return 0
}

# ---------- pidfiles ----------

pidfile_pid()   { [ -f "run/$1.pid" ] && cat "run/$1.pid" 2>/dev/null; }
pidfile_alive() { local p; p=$(pidfile_pid "$1"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

pidfile_proc_matches() {
  # pidfile_proc_matches <name> — verify the recorded pid still runs the
  # expected kind of process (PID-reuse guard for stop.sh).
  local name="$1" p cmd
  p=$(pidfile_pid "$name"); [ -n "$p" ] || return 1
  cmd=$(ps -p "$p" -o command= 2>/dev/null) || return 1
  case "$name" in
    api)         printf '%s' "$cmd" | grep -q 'uvicorn\|cadx.main' ;;
    worker)      printf '%s' "$cmd" | grep -q 'arq.*AgentWorker\|cadx.queue' ;;
    executor)    printf '%s' "$cmd" | grep -q 'arq.*CadWorker\|cadx.queue' ;;
    web)         printf '%s' "$cmd" | grep -q 'vite\|npm' ;;
    mesh-server) printf '%s' "$cmd" | grep -q 'meshserver\|mesh_server' ;;
    *)           return 0 ;;
  esac
}

pidfile_start() {
  # pidfile_start <name> <workdir> cmd args...
  local name="$1" dir="$2"; shift 2
  if [ "$DRY" = "1" ]; then info "dry-run: would start $name ($*)"; return 0; fi
  mkdir -p run logs
  if pidfile_alive "$name"; then
    info "$name already running (pid $(pidfile_pid "$name"))"
    return 0
  fi
  ( cd "$dir" && exec "$@" ) > "logs/$name.log" 2>&1 &
  echo $! > "run/$name.pid"
  info "$name started (pid $!, logs/$name.log)"
}

pidfile_stop() {
  # pidfile_stop <name> — SIGTERM, grace, SIGKILL; verifies identity first
  local name="$1" pid
  [ -f "run/$name.pid" ] || return 0
  pid=$(pidfile_pid "$name")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    if pidfile_proc_matches "$name"; then
      kill "$pid" 2>/dev/null || true
      local i
      for i in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5 2>/dev/null || sleep 1
      done
      kill -9 "$pid" 2>/dev/null || true
      ok "stopped $name (pid $pid)"
    else
      warn "$name pid $pid is not a cadx process — leaving it alone (stale pidfile removed)"
    fi
  fi
  rm -f "run/$name.pid"
}

# ---------- failure traps ----------

TRAP_CONTEXT=""
trap_setup() {
  # trap_setup <context> — on unexpected non-zero exit: banner + log tail
  TRAP_CONTEXT="$1"
  trap '_ui_on_err $?' ERR
  trap '_ui_on_int' INT TERM
}

_ui_on_err() {
  local rc="$1"
  _ui_show_cursor
  printf '\n%s✗ %s failed%s — exit %s\n' \
    "$UI_R" "$TRAP_CONTEXT" "${STEP_LABEL:+ at step: $STEP_LABEL}" "$rc" >&2
  if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE" | sed 's/^/      /' >&2
    printf '   %sfull log: %s%s\n' "$UI_DIM" "$LOG_FILE" "$UI_RST" >&2
  fi
  exit "$rc"
}

_ui_on_int() {
  _ui_show_cursor
  printf '\n%s! interrupted%s\n' "$UI_Y" "$UI_RST" >&2
  exit 130
}

# ---------- flag helper ----------

# ui_common_flags <args...> — callers embed in their flag loops:
#   --quiet|--verbose|--no-color|--yes handled centrally via env vars.
ui_common_flags() {
  case "$1" in
    -q|--quiet)   QUIET=1 ;;
    -v|--verbose) VERBOSE=1 ;;
    --no-color)   NO_COLOR=1; UI_TTY=0; UI_SPIN=0; ui_detect ;;
    -y|--yes)     YES=1 ;;
    *)            return 1 ;;
  esac
  return 0
}

# ---------- selftest ----------

if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--selftest" ]; then
  cd "$(dirname "$0")/../.." || exit 1
  log_init uitest
  steps_total 4
  run_step "fast step" sleep 1
  run_step "noisy step" bash -c 'for i in 1 2 3 4 5; do echo "progress $i/5"; sleep 0.3; done'
  step_begin "skipped step"; step_skip "nothing to do"
  run_step "failing step" bash -c 'echo boom >&2; sleep 0.5; exit 3'
  print_summary
fi
