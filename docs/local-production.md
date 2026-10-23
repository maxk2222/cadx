# Local production — resilience & operations

How CADX stays alive on a single host: supervision, queue recovery, backups,
disk hygiene, health semantics. For the update flow itself see `updating.md`.

## Service topology

```
browser ──> frontend :8080 ──> api :8000 ──> postgres :5432 (data volume)
                                  │        └> redis :6379 (AOF, redisdata volume)
                                  ├─> worker      (arq "cadx" queue — agent jobs)
                                  └─> cad-executor (arq "cadx-exec" queue — CAD/FEM)
host:   mesh-server :8765 (native nohup, supervised)  ·  updater :8766 (watchdog)
```

All compose services carry `restart: unless-stopped`, resource caps, and log
rotation (`x-logging`, 10MB × 3). The **updater daemon doubles as the host
supervisor** — it is the only process that survives a full stack restart.

## Watchdog (scripts/update_server.py)

Ticks every `CADX_WATCHDOG_INTERVAL` (15s). Fully **suppressed while an
update/check/rollback is in flight** (`run/update-state.json` state ∈
running/checking/rolling-back) so it never races `docker compose down`.

- **Stalled update kill**: no output in the update log for
  `CADX_UPDATE_STALL_S` (30min) — measured by `max(state.mtime, log.mtime)`,
  never state-mtime alone — or absolute cap `CADX_UPDATE_MAX_S` (3h) →
  `SIGTERM` the process group (`start_new_session`), `SIGKILL` after 15s,
  state → `failed`.
- **mesh-server respawn**: only when supervised — `run/mesh-server.env`
  exists (written by `start.sh` when it nohup-spawns the server). A foreign
  process on :8765 without the env file is never adopted. Respawn after
  3 failed `/v1/health` probes using exactly the recorded env.
- **Compose healing**: `docker compose ps` — `exited` with ExitCode≠0 →
  `up -d <svc>`; `unhealthy` → `restart`. ExitCode 0 = intentional, never
  touched. A **fully-down stack is never resurrected** (user's `stop.sh`
  wins). Wedged workers (heartbeat age > 120s in `/api/health`) → restart.
- **Flap guard**: max 5 respawns/service per 10min — then suppressed until
  the window clears (crash-looping services report, don't thrash).
- State: `run/watchdog.json` (last actions, fail counters). Inspect via
  `scripts/status.sh`, `GET /v1/stack/status` (token), or
  `GET /api/system/stack`; manual heal: `POST /api/system/stack/restart/<svc>`
  (services: api, worker, cad-executor, frontend, mesh-server).

`stop.sh` removes `run/mesh-server.env` **before** SIGTERM — supervision is
dropped first, then the process, so a manual stop stays stopped.

## Job queue durability

- `Job.payload` (JSON column) stores the exact enqueue envelope
  `{"func", "args"}` at spawn time — recovery replays the payload verbatim
  instead of mapping `JobType` (ambiguous: `edit` serves two functions).
- Redis runs with **AOF** (`appendonly yes`, everysec) on the `redisdata`
  volume. Even after a total Redis loss, the **queue sweeper** (every 60s in
  the agent worker) re-enqueues `queued` jobs whose arq keys vanished, using
  deterministic `_job_id=<job.id>` dedup — a job already re-queued is a
  no-op. Jobs without a payload envelope (pre-migration) are failed
  explicitly rather than left hanging.
- Chat: pending user messages are re-enqueued per-message
  (`chat:<chat_id>:<msg_id>`); orphaned `turn_running` flags clear after the
  grace period. A failed enqueue surfaces as HTTP 503 — the pending row
  survives for the sweeper.
- `queued` rows younger than 90s are never touched (spawn grace window).

## Health semantics

`GET /api/health` → `ok` pinned to the DB session (compose healthcheck signal;
503 when DB is down so Docker/watchdog react). Detail: `db`, `redis`,
`workers.{agent,cad}.{age_s,stale}` (heartbeat TTL 45s; stale >60s →
`degraded: true`), `llm_configured`. Redis flapping never 503s the endpoint —
it shows as `degraded` + `redis:false` so the sweeper story is observable.

## Backups

`scripts/backup.sh` (`make backup`) → `data/backups/cadx-<ts>.dump` (pg_dump
-Fc) + `.json` sidecar (git sha, alembic heads). Keeps newest 10. Skips
cleanly when postgres is down (`--required` fails the caller). `update.sh`
runs it automatically before `deps`+`restart` — idempotent across updater
self-reexec via `run/update-backup.json`.

**Restore**: alembic is forward-only — a code rollback across a migration
needs `pg_restore`:

```sh
docker compose exec -T postgres pg_restore -U cadx -d cadx --clean --if-exists \
  < data/backups/cadx-<ts>.dump        # or: cat dump | docker exec -i …
```

## Disk hygiene

`scripts/prune.sh` (`make prune`, `--dry-run` first): aged rotated logs
(`logs/*-<ts>.log` >14d), >100MB live logs get tail-truncated (mtime is
watchdog state — never deleted), `evals/scratch` aged files, `data/tmp` /
`data/.tmp` / `data/cache` aged files, `docker system prune -f` (dangling
only). `--aggressive` adds builder cache (>168h) + old eval archives.
**Never** touches `data/projects`, `data/library`, `run/`, the DB, or any
current log.

## Frontend resilience

`ErrorBoundary` wraps the app in `main.tsx` (`inline` variant for panels);
`ModelCanvas` listens for `webglcontextlost` and offers a remount instead of
a frozen canvas. All strings i18n'd across the 9 locales.

## Autostart

`scripts/install_autostart.sh` (`make autostart`, `--uninstall`): launchd
agent (macOS) / systemd user unit (Linux) / `@reboot` cron fallback — starts
only the updater daemon; it supervises the rest.

## Operational cheat sheet

```
make start / stop / status   # stack lifecycle (status.sh shows watchdog)
make update                  # pull + backup + rebuild + restart + verify
make backup / prune          # data safety / disk hygiene
GET  /api/system/stack       # supervised-stack snapshot (updater down → stub)
POST /api/system/stack/restart/worker   # manual heal
```

## Failure modes & what happens

| failure | detection | reaction |
|---|---|---|
| api/worker/executor crash | docker `restart: unless-stopped` | container restart |
| worker event-loop wedge | heartbeat age >120s in /api/health | watchdog `compose restart` |
| container `unhealthy` | `compose ps` health | watchdog restart after 3 ticks |
| Redis data loss | sweeper finds queued rows w/o arq keys | payload replay, dedup-safe |
| update hangs | no log output 30min | killpg → state=failed |
| mesh-server dies | 3× /v1/health fail | respawn from env contract |
| user runs stop.sh | env contract removed + ExitCode 0 | watchdog stands down |
| disk fills | `_guarded` refuses jobs <2GB free | prune.sh recovers space |
