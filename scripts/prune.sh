#!/usr/bin/env bash
# Conservative disk cleanup — ages out rotated logs and stale temp artifacts
# and trims docker build garbage. NEVER touches: live database, run/ state,
# data/projects, data/library, current (non-rotated) logs, or anything a
# running job could still reference.
#
# Usage:
#   scripts/prune.sh [--dry-run] [--days N] [--aggressive] [--no-color] [-h|--help]
#     --dry-run     report what would be freed, delete nothing (default off)
#     --days N      age threshold for deletion (default 14)
#     --aggressive  also prune old eval archives + docker build cache
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/ui.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

DRY=0; DAYS=14; AGGRO=0
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --dry-run|-n) DRY=1; shift ;;
    --days)       DAYS="$2"; shift 2 ;;
    --aggressive) AGGRO=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done

say "cadx prune — days=$DAYS$([ "$DRY" = 1 ] && echo ' (dry-run)')"

MB() { du -sm "$1" 2>/dev/null | awk '{print $1+0}'; }
FREE0=$(disk_free_mb . 2>/dev/null || echo 0)

prune_dir() { # dir pattern days label — aged regular files only
  local dir="$1" pat="$2" days="$3" label="$4"
  [ -d "$dir" ] || return 0
  local n=0
  while IFS= read -r -d '' f; do
    n=$((n + 1))
    if [ "$DRY" = 1 ]; then info "  would rm $f"; else rm -f "$f"; fi
  done < <(find "$dir" -type f -name "$pat" -mtime "+$days" -print0 2>/dev/null)
  [ "$n" -gt 0 ] && info "$label: $n file(s) older than ${days}d $([ "$DRY" = 1 ] && echo '(dry)') || true"
  return 0
}

# 1. rotated launcher/service logs (logs/*-<ts>.log + mesh/updater tails).
#    the CURRENT unrotated logs (update.log, mesh-server.log, updater.log)
#    stay — they are the live tails the updater reads mtimes from
prune_dir logs "*-????????-??????.log" "$DAYS" "rotated run logs"
prune_dir logs "*.log.????????" "$DAYS" "rolled service logs"
# big live logs: truncate (not delete) beyond 100 MB — mtime is watchdog state
for f in logs/mesh-server.log logs/updater.log logs/update.log; do
  if [ -f "$f" ] && [ "$(du -sm "$f" | awk '{print $1}')" -gt 100 ]; then
    if [ "$DRY" = 1 ]; then info "  would tail-truncate $f (>100MB)"; else
      tail -c 50m "$f" > "$f.tmp" && mv "$f.tmp" "$f"
      info "truncated $f to last 50MB"
    fi
  fi
done

# 2. scratch evals — ephemeral by design (evals/scratch is gitignored)
if [ -d evals/scratch ] && [ "$(find evals/scratch -type f -mtime "+$DAYS" 2>/dev/null | head -1 | wc -l)" -gt 0 ]; then
  if [ "$DRY" = 1 ]; then info "  would clear evals/scratch aged entries"; else
    find evals/scratch -type f -mtime "+$DAYS" -delete 2>/dev/null || true
    find evals/scratch -type d -empty -delete 2>/dev/null || true
  fi
fi

# 3. data/tmp-ish dirs only — NEVER project/library payloads. data/tmp is the
#    documented scratch area for executor pipelines.
for d in data/tmp data/.tmp data/cache; do
  prune_dir "$d" "*" "$DAYS" "scratch $d"
done

# 4. docker: dead containers + dangling images only (compose-managed volumes
#    and tagged images stay)
if docker info >/dev/null 2>&1; then
  if [ "$DRY" = 1 ]; then
    info "  would run: docker system prune -f (containers, networks, dangling images)"
  else
    run_step "docker system prune" docker system prune -f || true
  fi
  if [ "$AGGRO" = 1 ]; then
    if [ "$DRY" = 1 ]; then
      info "  would run: docker builder prune (build cache)"
    else
      run_step "docker builder prune" docker builder prune -f --filter "until=168h" || true
    fi
    prune_dir evals/results "*.json" "$((DAYS * 4))" "eval archives"
  fi
else
  info "docker not reachable — skipping image/container prune"
fi

FREE1=$(disk_free_mb . 2>/dev/null || echo 0)
ok "prune done — free: ${FREE0}MiB -> ${FREE1}MiB$([ "$DRY" = 1 ] && echo ' (dry-run, nothing deleted)')"
