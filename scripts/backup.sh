#!/usr/bin/env bash
# PostgreSQL backup → data/backups/cadx-<ts>.dump (custom pg_dump format) +
# sidecar .json metadata. Runs inside the postgres container — safe when the
# stack or postgres is down (skips with a warning, never fails the caller
# unless --required).
#
# Usage:
#   scripts/backup.sh [--dir PATH] [--keep N] [--required] [--no-color] [-h|--help]
#     --dir PATH   backup dir (default: data/backups)
#     --keep N     keep newest N dumps (default 10, 0 = keep all)
#     --required   exit non-zero when the backup could not be taken
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/ui.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

DIR="data/backups"; KEEP=10; REQUIRED=0
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --dir)      DIR="$2"; shift 2 ;;
    --keep)     KEEP="$2"; shift 2 ;;
    --required) REQUIRED=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done

say "cadx backup -> $DIR"
mkdir -p "$DIR"

# postgres container + credentials — compose env wins, .env seeds them
PG_CONTAINER=$(docker compose ps -q postgres 2>/dev/null || true)
if [ -z "$PG_CONTAINER" ]; then
  warn "postgres container not running — backup skipped"
  [ "$REQUIRED" = 1 ] && exit 1
  exit 0
fi
# shellcheck disable=SC1091
[ -f .env ] && set -a && . ./.env && set +a
POSTGRES_USER="${POSTGRES_USER:-cadx}"
POSTGRES_DB="${POSTGRES_DB:-cadx}"

TS=$(date +%Y%m%d-%H%M%S)
OUT="$DIR/cadx-$TS.dump"
META="$DIR/cadx-$TS.json"
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
BRANCH=$(git branch --show-current 2>/dev/null || echo "?")

step_begin "pg_dump $POSTGRES_DB"
if docker exec "$PG_CONTAINER" pg_dump -U "$POSTGRES_USER" -Fc "$POSTGRES_DB" > "$OUT" 2>"$OUT.err"; then
  rm -f "$OUT.err"
  step_done "$(du -sh "$OUT" | awk '{print $1}')"
else
  warn "pg_dump failed: $(tail -3 "$OUT.err" 2>/dev/null)"
  rm -f "$OUT" "$OUT.err"
  [ "$REQUIRED" = 1 ] && exit 1
  exit 0
fi

HEADS=$(cd backend && ../.venv/bin/uv run alembic heads 2>/dev/null | awk '{print $1}' | paste -sd, - || true)
cat > "$META" <<EOF
{
  "file": "$(basename "$OUT")",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_sha": "$SHA",
  "branch": "$BRANCH",
  "alembic_heads": "${HEADS:-unknown}",
  "db": "$POSTGRES_DB",
  "note": "alembic migrations are forward-only — code rollback requires restoring this dump"
}
EOF
info "metadata: $META"

if [ "$KEEP" -gt 0 ]; then
  ls -1t "$DIR"/cadx-*.dump 2>/dev/null | tail -n "+$((KEEP + 1))" | while read -r f; do
    rm -f "$f" "${f%.dump}.json"
    info "rotated out: $(basename "$f")"
  done
fi
ok "backup: $OUT"
