# Releasing

Maintainer runbook for cutting a release. Audience: repo owner only.

## Version contract

`VERSION` (repo root) is the single release string: `X.Y.Z` or
`X.Y.Z-suffix` (e.g. `0.1.0-alpha.1`). Package manifests carry the base
`X.Y.Z` only: `backend/pyproject.toml`, `mesh_server/pyproject.toml`,
`frontend/package.json`. `scripts/check_version.sh` verifies
tag ↔ VERSION ↔ manifests and runs in the release pipeline.

## Release checklist

1. `VERSION` bump + manifest versions + `CHANGELOG.md` entry +
   `docs/release-notes/<ver>.md`.
2. `scripts/check_version.sh` green.
3. Media fresh (see below), docs audit green, full test suites green.
4. Commit everything, tag `v<VERSION>` on `main`, push.
5. The `release.yml` workflow runs the gate → builds the tarball →
   `gh release create --prerelease --latest=false`.
6. Verify the release page: 3 assets (tarball, `SHA256SUMS.txt`,
   `install.sh`), notes rendered, marked as pre-release (not latest).

## Screenshots / GIFs

All assets under `docs/screenshots/` are produced by
`docs/capture_screenshots.py` (Playwright + Pillow) against a running
docker stack on `:8080`:

```bash
make up                                        # stack must be healthy
.venv/bin/uv run --with playwright --with pillow \
    python3 docs/capture_screenshots.py        # full run (png + gifs + install)
```

- `python3 docs/capture_screenshots.py --list` — prints projects/versions
  for the CONFIG block (DISPLAY_NAMES / P_* / V_* ids). Update CONFIG when
  the demo database changes; the script exits early on missing ids.
- Selectors key off **i18n strings** (e.g. `Continue`, tab names
  `Spec & Plan`/`Code`/`Quality`/`Files`) — when UI labels change, grep the
  script for the old literal and update.
- The run renames projects to DISPLAY_NAMES, seeds a demo chat, cleans up
  wizard drafts, masks endpoint URLs on the settings page, and restores
  everything in `finally`.
- Review every asset before committing: no keys/hosts/paths/personal data,
  current UI visible, `docs/screenshots/` ≤ ~10 MB total.

## Release workflow

`.github/workflows/release.yml` triggers on `v*` tags:
`test` gate (backend unit tests, mesh_server tests with `MESH_FAKE=1`,
frontend lint+tsc+test+build, `check_version.sh`, cad tests via executor
image) → `package` (tarball without `.env*`/`data/`/`run/`/`logs/`/venvs/
`node_modules`, `SHA256SUMS.txt`, `gh release create --prerelease`).

## Rulesets (post public-flip)

`.github/CODEOWNERS` (`* @maxk2222`) is tracked in the repo. After flipping
the repo public, create the owner-only ruleset (blocks direct pushes,
force-push and deletion on `main`/`dev`/`v*` for anyone but the admin
role; requires PR + code-owner review + green `test` check for everyone
else). See the JSON body in the release plan; verify with
`gh api repos/maxk2222/cadx/rulesets`. Enable the
`required_status_checks` rule only after the release workflow has run
green once (context name must match the check name).

## History hygiene

Commit messages never carry tool-attribution trailers — the public history
was normalized once (single identity, trailers purged, dead gif blobs
removed) and stays clean by convention, not by hook.
