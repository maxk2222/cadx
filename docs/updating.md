# Updating CADX

CADX can update itself end-to-end: fetch from GitHub, reinstall dependencies,
rebuild docker images, run migrations and restart the whole stack — from the
terminal or from the web GUI (**Settings → System → Software update**).

```
make update            # full update: pull + deps + rebuild + restart + verify
make update-check      # how far behind origin am I? (read-only)
scripts/update.sh --ref dev        # switch channel to `dev`
scripts/update.sh --rollback       # reset --hard to the pre-update revision
scripts/update.sh --force          # override dirty-tree / active-job refusals
```

## How it works

```
browser ──> api :8000/8080 ──> updater daemon :8766 ──> scripts/update.sh
            (/api/system/*)      (host process,             (the real engine:
             bearer-token          survives the restart      git, setup.sh,
             proxy)                it drives)                stop/start, verify)
```

The api container only mounts `./data` — it cannot see the git repo, run
`git pull`, rebuild its own image, or survive its own restart. So the actual
update is driven by a small host-side daemon
(`scripts/update_server.py`, stdlib-only, port **8766**) that `start.sh`
launches alongside the mesh-server. The api merely proxies its endpoints.

## What an update does

`scripts/update.sh` runs these phases, logging to `logs/update-<ts>.log` and
publishing progress to `run/update-state.json`:

1. **preflight** — git repo + `origin` remote, required tools, docker daemon.
   Refuses when the working tree is dirty (prints the offending files) or when
   generation jobs are still active — `--force` overrides both.
2. **fetch** — `git fetch --tags --prune origin`, resolve the target ref
   (default: the current branch's upstream; `--ref` picks any remote branch or
   tag — e.g. switch `main` ↔ `dev`).
3. **pull** — `git switch` to the target (creating a tracking branch if
   needed) and `git merge --ff-only`. A diverged local branch fails safely.
   The pre-update sha is saved to `run/update-prev.env` for `--rollback`.
4. **re-exec** — if the update touched the updater itself
   (`update.sh`/`lib/*.sh`), the script re-executes the new version and
   resumes at the deps step.
5. **deps** — `scripts/setup.sh` re-runs idempotently: backend `uv sync`,
   frontend `npm install` (dev mode), mesh/image adapter + model-weight
   checks. Skipped when the mesh-server runs as a gpu-profile container.
6. **restart** — `scripts/stop.sh` + `scripts/start.sh --no-setup` in the same
   mode the stack runs in (`run/mode` → `--dev` or docker). Alembic migrations
   apply automatically on api boot.
7. **verify** — waits for `/api/health` (180 s) and the mesh-server; on
   failure the run is marked `failed` and `--rollback` is offered.

## GUI button

Settings → System shows the current version/commit/branch, a channel toggle
(`main`/`dev` and any other remote branches), the commits you're behind, a
live log tail while the update runs, and a restart banner. A badge on the
Settings nav item shows the behind-count. If generation jobs are active the
confirmation dialog warns and requires a second confirm. After a failed run a
**Roll back** button appears with a migration warning.

## Safety rules

- **Dirty tree** — refused with the offending paths listed; commit or stash,
  or `--force`. (`--check` stays read-only and always allowed.)
- **Active jobs** — refused by default; the GUI confirm dialog shows the
  count, `--force`/`confirm_jobs` proceeds anyway.
- **Locking** — `run/update.lock` serialises updates; the daemon additionally
  refuses while its last spawned process lives (HTTP 409).
- **Rollback** — `git reset --hard PREV_SHA` + deps + restart. Database
  migrations are **forward-only**: rolling code back across a migration can
  break the DB — back up `data/` and `pgdata` first.
- **Auth** — every daemon endpoint except `/v1/health` requires a bearer
  token (`run/updater.token`, also `.env` `CADX_UPDATE_TOKEN` for the docker
  api). In docker mode the daemon binds `0.0.0.0` so `host.docker.internal`
  can reach it; in dev mode it stays on loopback.
- **Restart survival** — `stop.sh` skips the updater pidfile while an update
  is running; the daemon also re-execs itself when an update replaced its own
  source.

## When the daemon is offline

The GUI card degrades to instructions and `/api/system/*` returns `503`
("updater daemon unreachable"). Fix by restarting the stack
(`scripts/start.sh`), or update manually: `scripts/update.sh`.

## Files

| Path | Purpose |
|------|---------|
| `scripts/update.sh` | update engine (all real work) |
| `scripts/update_server.py` | host daemon, port 8766 |
| `run/updater.token` | bearer token (0600) |
| `run/update-state.json` | live status for GUI polling |
| `run/update-check.json` | cached `--check` result (6 h staleness) |
| `run/update-prev.env` | pre-update sha/ref for `--rollback` |
| `run/update.lock` | mkdir lock serialising updates |
| `run/mode` | `docker` or `dev` — how to restart |
| `logs/update-*.log` | per-run logs; `updater-launch.log` for spawns |
| `backend/cadx/api/routers/system.py` | `/api/system/*` proxy |
| `frontend/src/components/UpdateCard.tsx` | Settings → System card |
