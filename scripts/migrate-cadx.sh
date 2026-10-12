#!/usr/bin/env bash
# CADX one-shot migration: clones the old `cadagent` docker volumes to the new
# `cadx` project names and renames the Postgres role + database inside them.
# Idempotent — safe to re-run; existing `cadx` volumes/roles are left alone.
#
# Usage:
#   scripts/migrate-cadx.sh            # prod volumes (+ dev volumes if present)
#
# Escape hatch instead of migrating: keep the old credentials by putting
#   POSTGRES_USER=cadagent POSTGRES_PASSWORD=cadagent POSTGRES_DB=cadagent
# in .env — then no volume clone or SQL rename is needed at all.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OLD=cadagent
NEW=cadx

say()  { printf '\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
die()  { warn "$*"; exit 1; }

command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "docker daemon not reachable"

# ---------- 1. stop the old stack (must be down before renaming the DB) -----
for proj in "$OLD" "$OLD-dev"; do
  if docker compose -p "$proj" ps -q 2>/dev/null | grep -q .; then
    say "stopping old '$proj' stack"
    docker compose -p "$proj" down
  fi
done
# also stop the new stack's consumers if a previous run left them up
docker compose stop api worker cad-executor >/dev/null 2>&1 || true

# ---------- 2. clone volumes -------------------------------------------------
clone_vol() { # $1 = old volume, $2 = new volume
  if ! docker volume inspect "$1" >/dev/null 2>&1; then
    info "volume $1 not present — skip"
    return
  fi
  if docker volume inspect "$2" >/dev/null 2>&1; then
    info "volume $2 already exists — skip"
    return
  fi
  say "cloning volume $1 → $2"
  docker volume create "$2" >/dev/null
  docker run --rm -v "$1":/from:ro -v "$2":/to alpine \
    sh -c 'cd /from && cp -a . /to'
}

clone_vol "${OLD}_pgdata"            "${NEW}_pgdata"
clone_vol "${OLD}_mesh_models"       "${NEW}_mesh_models"
clone_vol "${OLD}_rembg_models"      "${NEW}_rembg_models"
clone_vol "${OLD}-dev_pgdata-dev"    "${NEW}-dev_pgdata-dev"

# ---------- 3. rename Postgres role + database -------------------------------
# POSTGRES_* env only applies to freshly-initialised volumes — inside a cloned
# volume the superuser/db are still named "cadagent".
pg_rename() { # $1 = label, $2.. = docker compose command prefix
  local label="$1"; shift
  local su db_old db_new role_old role_new
  DC=("$@")   # compose prefix — nested functions can't see the outer $@

  say "starting postgres ($label)"
  "${DC[@]}" up -d postgres

  for _ in $(seq 1 60); do
    if "${DC[@]}" exec -T postgres pg_isready -q 2>/dev/null; then break; fi
    sleep 1
  done

  q() { "${DC[@]}" exec -T postgres psql -U "$su" -d postgres -tAc "$1" 2>/dev/null; }

  # figure out which superuser name is valid (handles partially-migrated state)
  su=$OLD
  "${DC[@]}" exec -T postgres psql -U "$OLD" -d postgres -tAc 'SELECT 1' >/dev/null 2>&1 \
    || su=$NEW
  "${DC[@]}" exec -T postgres psql -U "$su" -d postgres -tAc 'SELECT 1' >/dev/null 2>&1 \
    || { warn "no usable superuser ($OLD/$NEW) in $label postgres — skip"; return; }

  db_old=$(q "SELECT 1 FROM pg_database WHERE datname='$OLD'")
  db_new=$(q "SELECT 1 FROM pg_database WHERE datname='$NEW'")
  if [ "$db_old" = 1 ] && [ "$db_new" != 1 ]; then
    "${DC[@]}" exec -T postgres psql -U "$su" -d postgres \
      -c "ALTER DATABASE $OLD RENAME TO $NEW"
    info "database $OLD → $NEW"
  elif [ "$db_old" = 1 ] && [ "$db_new" = 1 ]; then
    warn "both databases exist — leaving as-is, fix manually"
  else
    info "database already $NEW (or absent) — skip"
  fi

  role_old=$(q "SELECT 1 FROM pg_roles WHERE rolname='$OLD'")
  role_new=$(q "SELECT 1 FROM pg_roles WHERE rolname='$NEW'")
  # a session user can't be renamed — create the new role instead of
  # ALTER ROLE … RENAME, hand over ownership, then drop the old one
  if [ "$role_new" != 1 ]; then
    "${DC[@]}" exec -T postgres psql -U "$su" -d postgres \
      -c "CREATE ROLE $NEW SUPERUSER LOGIN PASSWORD '$NEW'"
    info "role $NEW created (superuser)"
  fi
  if [ "$role_old" = 1 ]; then
    for db in "$NEW" postgres; do
      "${DC[@]}" exec -T postgres psql -U "$NEW" -d "$db" \
        -c "REASSIGN OWNED BY $OLD TO $NEW" >/dev/null 2>&1 || true
      "${DC[@]}" exec -T postgres psql -U "$NEW" -d "$db" \
        -c "DROP OWNED BY $OLD" >/dev/null 2>&1 || true
    done
    "${DC[@]}" exec -T postgres psql -U "$NEW" -d postgres \
      -c "ALTER DATABASE $NEW OWNER TO $NEW" >/dev/null 2>&1 || true
    "${DC[@]}" exec -T postgres psql -U "$NEW" -d postgres \
      -c "DROP ROLE $OLD" \
      && info "role $OLD dropped" \
      || warn "could not drop role $OLD — harmless, drop manually if desired"
  fi
  # keep the password aligned with the new DATABASE_URL default
  "${DC[@]}" exec -T postgres psql -U "$NEW" -d postgres \
    -c "ALTER ROLE $NEW WITH PASSWORD '$NEW'" >/dev/null \
    || warn "could not set $NEW password"

  "${DC[@]}" stop postgres
}

if docker volume inspect "${NEW}_pgdata" >/dev/null 2>&1; then
  pg_rename "prod" docker compose
fi

# dev stack postgres binds host :5432 — run it only after prod postgres stopped
if docker volume inspect "${NEW}-dev_pgdata-dev" >/dev/null 2>&1; then
  pg_rename "dev" docker compose -f docker-compose.dev.yml
fi

say "migration complete — start the stack with: docker compose up -d   (or: make start)"
info "old ${OLD}_* volumes were left untouched; remove them manually once verified:"
info "  docker volume rm ${OLD}_pgdata ${OLD}_mesh_models ${OLD}_rembg_models ${OLD}-dev_pgdata-dev"
