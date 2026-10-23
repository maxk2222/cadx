#!/usr/bin/env bash
# CADX updater — pulls the repo, re-runs the installer, restarts the stack.
# Works in docker mode (default) and native dev mode; the mode is read from
# run/mode (written by start.sh) with a pidfile fallback.
#
# Usage:
#   scripts/update.sh [--check] [--ref <branch|tag>] [--force] [--yes]
#                     [--rollback] [--no-restart] [--continue]
#                     [--dev|--docker] [--dry-run] [--quiet|--verbose|--no-color]
#                     [-h|--help]
#
#   --check      fetch + report update status as JSON (run/update-check.json)
#   --ref        target branch or tag (default: current branch's upstream)
#   --force      override dirty-tree refusal, active-job refusal and the
#                'already up to date' early exit
#   --rollback   reset --hard to the sha recorded before the last update
#                (run/update-prev.env) and rebuild/restart. WARNING: alembic
#                migrations are forward-only — rolling code back across a
#                migration can break the DB; back up data/ + pgdata first.
#   --no-restart pull + deps only, do not restart services
#   --continue   internal: resume after a self re-exec (script was updated)
#   --dev/--docker   force the restart mode (default: run/mode → detect)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/ui.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

# ---------- flags ----------
CHECK=0; ROLLBACK=0; NO_RESTART=0; CONTINUE=0; FORCE=0
REF=""; MODE_OVERRIDE=""
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --check)      CHECK=1; shift ;;
    --ref)        REF="$2"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    --rollback)   ROLLBACK=1; shift ;;
    --no-restart) NO_RESTART=1; shift ;;
    --continue)   CONTINUE=1; shift ;;
    --dev)        MODE_OVERRIDE=dev; shift ;;
    --docker)     MODE_OVERRIDE=docker; shift ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done

log_init update
trap_setup "update"

STATE_FILE="run/update-state.json"
PREV_ENV="run/update-prev.env"
CHECK_FILE="run/update-check.json"
REEXEC_MARK="run/update-reexec"
LOCK_DIR="run/update.lock"
LOCK_HELD=0

PREV_SHA=""; PREV_REF=""; NEW_SHA=""; TARGET_REF=""
U_STATE="idle"; U_PHASE=""; U_ERROR=""; U_ACTIVE_JOBS=0
# U_STARTED_AT survives the self re-exec via CADX_UPDATE_STARTED
U_STARTED_AT="${CADX_UPDATE_STARTED:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; U_FINISHED_AT=""
MODE="docker"

state_write() {
  # write the whole status document — consumed by update_server.py and the API.
  # --check runs lock-free (even during a real update) so it must never touch
  # the state file — its output goes to CHECK_FILE via write_check instead.
  [ "$CHECK" = 1 ] && return 0
  U_STATE="$U_STATE" U_PHASE="$U_PHASE" U_ERROR="$U_ERROR" \
  U_STARTED_AT="$U_STARTED_AT" U_FINISHED_AT="$U_FINISHED_AT" \
  U_ACTIVE_JOBS="$U_ACTIVE_JOBS" PREV_SHA="$PREV_SHA" NEW_SHA="$NEW_SHA" \
  TARGET_REF="$TARGET_REF" MODE="$MODE" LOG_FILE="$LOG_FILE" \
  python3 - "$STATE_FILE" <<'PY'
import json, os, sys

doc = {
    "state": os.environ["U_STATE"],
    "phase": os.environ["U_PHASE"],
    "error": os.environ["U_ERROR"] or None,
    "ref": os.environ["TARGET_REF"] or None,
    "prev_sha": os.environ["PREV_SHA"] or None,
    "new_sha": os.environ["NEW_SHA"] or None,
    "active_jobs": int(os.environ.get("U_ACTIVE_JOBS") or 0),
    "mode": os.environ["MODE"],
    "log": os.environ["LOG_FILE"] or None,
    "started_at": os.environ["U_STARTED_AT"],
    "finished_at": os.environ["U_FINISHED_AT"] or None,
    "ts": os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip(),
}
path = sys.argv[1]
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f, indent=2)
os.replace(tmp, path)
PY
}

set_phase() { U_PHASE="$1"; state_write; }

fail_phase() {
  # fail_phase <msg> — record state=failed, print tail + rollback hint, exit
  U_STATE="failed"; U_ERROR="$1"; U_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  state_write
  step_fail "$1"
  warn "to restore the previous checkout: scripts/update.sh --rollback"
  warn "(alembic migrations are forward-only — a rollback across a migration can break the DB)"
  exit 1
}

cleanup() { [ "$LOCK_HELD" = 1 ] && rm -rf "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

# ---------- mode detection ----------
detect_mode() {
  if [ -n "$MODE_OVERRIDE" ]; then MODE="$MODE_OVERRIDE"; return; fi
  if [ -f run/mode ]; then MODE=$(cat run/mode); return; fi
  if pidfile_alive api || pidfile_alive web; then MODE="dev"; else MODE="docker"; fi
  case "$MODE" in dev|docker) ;; *) MODE="docker" ;; esac
}

app_port() { [ "$MODE" = "dev" ] && echo 8000 || echo 8080; }

active_job_count() {
  # count non-terminal jobs via the app API; unreachable api -> 0
  local port out
  for port in $(app_port) 8080 8000; do
    out=$(curl -sf --max-time 5 "http://localhost:$port/api/jobs?active=true" 2>/dev/null) \
      && { printf '%s' "$out" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))' 2>/dev/null || echo 0; return; }
  done
  echo 0
}

# ---------- git helpers ----------
ref_exists() { # ref_exists <ref> -> 0 if remote branch or tag
  git show-ref --verify --quiet "refs/remotes/origin/$1" && return 0
  git show-ref --verify --quiet "refs/tags/$1" && return 0
  return 1
}

is_tag() { git show-ref --verify --quiet "refs/tags/$1"; }

write_check() {
  # write_check <behind> <ahead> <upstream> — upstream may be origin/<b> or a tag
  BEHIND="$1" AHEAD="$2" UPSTREAM="$3" python3 - "$CHECK_FILE" <<'PY'
import json, os, subprocess, sys

def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout

behind = int(os.environ["BEHIND"]); ahead = int(os.environ["AHEAD"])
upstream = os.environ["UPSTREAM"]
target = upstream.split("/", 1)[1] if upstream.startswith("origin/") else upstream
commits = []
if behind:
    for line in sh("git", "log", f"--format=%h%x09%s%x09%cs", f"HEAD..{upstream}").splitlines():
        sha, _, rest = line.partition("\t")
        subject, _, date = rest.partition("\t")
        commits.append({"sha": sha, "subject": subject, "date": date})
branches = sorted(
    b.strip().split("/", 1)[1]
    for b in sh("git", "branch", "-r", "--format=%(refname:short)").splitlines()
    if b.strip().startswith("origin/") and "HEAD" not in b
)
dirty = sh("git", "status", "--porcelain").splitlines()
current = sh("git", "branch", "--show-current").strip() or sh("git", "rev-parse", "--short", "HEAD").strip()
doc = {
    "behind": behind, "ahead": ahead, "commits": commits,
    "branches": branches, "current_ref": current, "target_ref": target,
    "current_sha": sh("git", "rev-parse", "--short", "HEAD").strip(),
    "target_sha": sh("git", "rev-parse", "--short", upstream).strip() or None,
    "update_available": behind > 0 or (current != target),
    "dirty": [l[3:] for l in dirty],
    "checked_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    .strftime("%Y-%m-%dT%H:%M:%SZ"),
}
tmp = sys.argv[1] + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f, indent=2)
os.replace(tmp, sys.argv[1])
PY
}

# ---------- preflight ----------
preflight() {
  step_begin "preflight"
  local fail=0
  git rev-parse --git-dir >/dev/null 2>&1 || { err "not a git repository"; fail=1; }
  git remote get-url origin >/dev/null 2>&1 || { err "no 'origin' remote"; fail=1; }
  need_cmd git || fail=1
  need_cmd python3 || fail=1
  need_cmd curl || fail=1
  if [ "$MODE" = "docker" ]; then
    docker info >/dev/null 2>&1 || { warn "docker daemon not reachable"; fail=1; }
  fi
  # dirty tree + active jobs only gate real updates — --check is read-only
  if [ "$CHECK" = 0 ]; then
    local dirty
    dirty=$(git status --porcelain | head -50)
    if [ -n "$dirty" ] && [ "$FORCE" = 0 ]; then
      warn "working tree is not clean:"
      printf '%s\n' "$dirty" | sed 's/^/      /' >&2
      err "refusing to update — commit/stash your changes or re-run with --force"
      fail=1
    fi
    U_ACTIVE_JOBS=$(active_job_count)
    # --yes counts as the explicit confirmation (the api gate or a human typed
    # it); without either flag a running job still blocks the update
    if [ "$U_ACTIVE_JOBS" -gt 0 ] && [ "$FORCE" = 0 ] && [ "${YES:-0}" = 0 ]; then
      err "$U_ACTIVE_JOBS generation job(s) still running — let them finish or pass --yes/--force"
      fail=1
    fi
  fi
  [ "$fail" = 1 ] && { step_fail "resolve the problems above"; exit 1; }
  step_done "mode=$MODE$([ "$U_ACTIVE_JOBS" -gt 0 ] && echo " · active_jobs=$U_ACTIVE_JOBS (forced)")"
}

# ---------- git fetch + target resolve ----------
fetch_and_resolve() {
  set_phase "fetch"
  run_step "git fetch" env GIT_TERMINAL_PROMPT=0 git fetch --tags --prune origin \
    || fail_phase "git fetch failed"
  local cur
  cur=$(git branch --show-current)
  if [ -z "$REF" ]; then
    REF=$(git rev-parse --abbrev-ref '@{u}' 2>/dev/null | sed 's|^origin/||')
    [ -z "$REF" ] && REF="$cur"
    # detached HEAD: fall back to origin/HEAD (usually main), then 'main'
    [ -z "$REF" ] && REF=$(git rev-parse --abbrev-ref origin/HEAD 2>/dev/null | sed 's|^origin/||')
    [ -z "$REF" ] && REF=main
  fi
  ref_exists "$REF" || fail_phase "unknown ref '$REF' (not a remote branch or tag)"
  TARGET_REF="$REF"
  local upstream="origin/$REF"
  is_tag "$REF" && upstream="$REF"
  local behind ahead
  behind=$(git rev-list --count "HEAD..$upstream" 2>/dev/null || echo 0)
  ahead=$(git rev-list --count "$upstream..HEAD" 2>/dev/null || echo 0)
  if [ "$CHECK" = 1 ]; then
    U_STATE="checking"; set_phase "check"
    write_check "$behind" "$ahead" "$upstream"
    step_done "$behind behind / $ahead ahead on $upstream"
    U_STATE="idle"; U_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set_phase "done"
    exit 0
  fi
  PREV_SHA=$(git rev-parse HEAD); PREV_REF="$cur"
  printf 'PREV_SHA=%s\nPREV_REF=%s\n' "$PREV_SHA" "$PREV_REF" > "$PREV_ENV"
  if [ "$behind" = 0 ] && [ "$cur" = "$REF" ] && [ "$FORCE" = 0 ]; then
    ok "already up to date ($upstream)"
    NEW_SHA=$PREV_SHA
    U_STATE="done"; U_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set_phase "done"
    exit 0
  fi
  info "$behind commit(s) behind $upstream"
}

# ---------- switch + pull ----------
do_pull() {
  set_phase "pull"
  local cur upstream="origin/$REF"
  cur=$(git branch --show-current)
  if [ "$cur" != "$REF" ]; then
    if is_tag "$REF"; then
      run_step "checkout tag $REF" git switch --detach "$REF" || fail_phase "checkout $REF failed"
    elif git show-ref --verify --quiet "refs/heads/$REF"; then
      run_step "switch to $REF" git switch "$REF" || fail_phase "git switch $REF failed"
    else
      run_step "track origin/$REF" git switch -c "$REF" --track "origin/$REF" \
        || fail_phase "could not create local branch $REF"
    fi
  fi
  is_tag "$REF" || {
    run_step "merge origin/$REF (ff-only)" git merge --ff-only "origin/$REF" \
      || fail_phase "fast-forward merge failed — local branch has diverged (resolve manually)"
  }
  NEW_SHA=$(git rev-parse HEAD)
  info "HEAD: ${PREV_SHA:0:10} -> ${NEW_SHA:0:10}"
}

# ---------- self re-exec when the updater itself changed ----------
maybe_reexec() {
  [ -f "$REEXEC_MARK" ] && return 0
  local changed
  changed=$(git diff --name-only "$PREV_SHA" "$NEW_SHA" -- \
    scripts/update.sh scripts/lib/ui.sh scripts/lib/pkg.sh \
    scripts/lib/gpu_detect.sh scripts/lib/host_detect.sh 2>/dev/null)
  [ -z "$changed" ] && return 0
  info "updater files changed — re-executing the updated script:"
  printf '%s\n' "$changed" | sed 's/^/      /'
  [ "$DRY" = 1 ] && return 0
  touch "$REEXEC_MARK"
  U_PHASE="reexec"; state_write
  export CADX_UPDATE_STARTED="$U_STARTED_AT"
  local args=(--continue --yes)
  [ "$FORCE" = 1 ] && args+=(--force)
  [ "$NO_RESTART" = 1 ] && args+=(--no-restart)
  [ "$MODE_OVERRIDE" = "dev" ] && args+=(--dev)
  [ "$MODE_OVERRIDE" = "docker" ] && args+=(--docker)
  [ "$VERBOSE" = 1 ] && args+=(--verbose)
  [ "$QUIET" = 1 ] && args+=(--quiet)
  [ "$DRY" = 1 ] && args+=(--dry-run)
  exec bash scripts/update.sh "${args[@]}"
}

# ---------- deps / restart / verify ----------
do_backup() {
  # pg_dump BEFORE stop/restart — alembic runs at api start, so the dump must
  # capture the pre-migration schema. run/update-backup.json makes the step
  # idempotent across an updater self-reexec (same NEW_SHA → skip).
  set_phase "backup"
  NEW_SHA=${NEW_SHA:-$(git rev-parse HEAD)}
  local marker="run/update-backup.json"
  if [ -f "$marker" ] && python3 - "$marker" "$NEW_SHA" <<'PY' 2>/dev/null
import json, sys
print(0 if json.load(open(sys.argv[1])).get("target") == sys.argv[2] else 1)
sys.exit(0 if json.load(open(sys.argv[1])).get("target") == sys.argv[2] else 1)
PY
  then
    step_begin "db backup"; step_skip "already taken for this update"
    return 0
  fi
  if [ "$DRY" = 1 ]; then step_begin "db backup"; step_skip "dry-run"; return 0; fi
  run_step "pg_dump (data/backups)" bash scripts/backup.sh \
    || warn "backup failed — update continues WITHOUT a database snapshot"
  if [ -f "$marker" ]; then :; else
    LATEST=$(ls -1t data/backups/cadx-*.dump 2>/dev/null | head -1 || true)
    [ -n "$LATEST" ] && cat > "$marker" <<EOF
{"file": "$LATEST", "target": "$NEW_SHA", "taken_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
  fi
}

do_deps() {
  set_phase "deps"
  rm -f "$REEXEC_MARK"
  local args=(--yes)
  [ "$MODE" = "dev" ] && args+=(--dev)
  # native mesh setup is pointless when the mesh-server runs as a gpu-profile
  # container (CUDA/ROCm compose tier) — skip the native venv/adapters then
  local gpu_mesh=0 s
  for s in mesh-server mesh-server-rocm; do
    docker compose --profile gpu --profile gpu-rocm ps -q "$s" 2>/dev/null \
      | grep -q . && gpu_mesh=1
  done
  [ "$gpu_mesh" = 1 ] && args+=(--skip-mesh --skip-image)
  run_step "installer (setup.sh)" bash scripts/setup.sh "${args[@]}" \
    || fail_phase "setup.sh failed — see $LOG_FILE"
}

do_restart() {
  [ "$NO_RESTART" = 1 ] && { step_begin "restart"; step_skip "--no-restart"; return 0; }
  set_phase "restart"
  run_step "stop" bash scripts/stop.sh || warn "stop.sh reported errors — continuing"
  local sargs=(--no-setup)
  [ "$MODE" = "dev" ] && sargs+=(--dev)
  run_step "start" bash scripts/start.sh "${sargs[@]}" \
    || fail_phase "start.sh failed — see $LOG_FILE"
}

do_verify() {
  set_phase "verify"
  local port ok=1
  port=$(app_port)
  if ! wait_for_url "http://localhost:$port/api/health" 180 api; then
    ok=0
  fi
  # mesh-server is advisory — the CAD path works without it
  if pidfile_alive mesh-server 2>/dev/null || ! port_free 8765; then
    wait_for_url "http://localhost:8765/v1/health" 60 mesh-server \
      || warn "mesh-server not healthy after restart"
  fi
  if [ "$ok" = 0 ]; then
    fail_phase "api did not come back healthy — stack state unknown"
  fi
  NEW_SHA=${NEW_SHA:-$(git rev-parse HEAD)}
  U_STATE="done"; U_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set_phase "done"
  say "update complete — ${PREV_SHA:0:10} -> ${NEW_SHA:0:10} (${TARGET_REF:-$PREV_REF})"
}

# ---------- rollback ----------
do_rollback() {
  [ -f "$PREV_ENV" ] || { err "no previous revision recorded (run/update-prev.env)"; exit 1; }
  source "$PREV_ENV"
  [ -z "$PREV_SHA" ] && { err "update-prev.env has no PREV_SHA"; exit 1; }
  warn "rolling back to $PREV_REF @ ${PREV_SHA:0:10}"
  warn "alembic migrations are forward-only — rolling back across a migration can break the DB"
  U_STATE="rolling-back"; TARGET_REF="$PREV_REF"; set_phase "rollback"
  local dirty
  dirty=$(git status --porcelain | head -50)
  if [ -n "$dirty" ] && [ "$FORCE" = 0 ]; then
    err "working tree dirty — cannot reset --hard safely; commit/stash or use --force"
    U_STATE="failed"; U_ERROR="dirty tree on rollback"; state_write
    exit 1
  fi
  [ -n "$PREV_REF" ] && [ "$(git branch --show-current)" != "$PREV_REF" ] \
    && { run_step "switch to $PREV_REF" git switch "$PREV_REF" \
         || fail_phase "git switch $PREV_REF failed"; }
  run_step "reset --hard $PREV_SHA" git reset --hard "$PREV_SHA" \
    || fail_phase "git reset failed"
  NEW_SHA=$(git rev-parse HEAD)
  # code rolled back — but alembic only moves forward. If the rollback
  # crossed a migration, the newest pre-update dump is the way back.
  local dump
  dump=$(ls -1t data/backups/cadx-*.dump 2>/dev/null | head -1 || true)
  if [ -n "$dump" ]; then
    warn "if the rolled-back code predates applied migrations, restore the DB:"
    warn "  scripts/restore_db.sh $dump   (or: cat $dump | docker compose exec -T postgres pg_restore -U cadx -d cadx --clean)"
  fi
  do_deps
  do_restart
  U_STATE="done"; U_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set_phase "rolled-back"
  say "rollback complete — now at ${NEW_SHA:0:10} ($PREV_REF)"
}

# ---------- main ----------
say "CADX update"
info "log: $LOG_FILE"
mkdir -p run logs

if [ "$CHECK" = 1 ]; then
  : # read-only status query — never takes the lock, safe during a running update
elif [ "$CONTINUE" = 0 ]; then
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    err "another update is already running (run/update.lock)"
    exit 1
  fi
  LOCK_HELD=1
  rm -f "$REEXEC_MARK"   # clear a stale marker from a crashed earlier run
else
  # same logical update after self re-exec — keep (and clean up) the lock
  [ -d "$LOCK_DIR" ] && LOCK_HELD=1
  [ -f "$PREV_ENV" ] && source "$PREV_ENV"
fi

detect_mode

if [ "$ROLLBACK" = 1 ]; then
  do_rollback
  exit 0
fi

if [ "$CONTINUE" = 0 ]; then
  preflight
  fetch_and_resolve
  do_pull
  maybe_reexec
else
  step_begin "resume"; step_done "continuing after updater self-update"
fi

do_backup
do_deps
do_restart
do_verify
print_summary
