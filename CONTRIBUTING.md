# Contributing

CADX is a local, single-user project. Contributions happen in three areas:
the Python backend/agent, the React frontend and the `mesh_server` sidecar.

## Setup

```bash
python3 -m venv .venv && .venv/bin/python -m pip install uv
make sync                      # backend venv + locked deps
make sync-fem                  # optional: CHOLMOD/AMG FEM solver tier
                               # (needs SuiteSparse headers, see fem-topopt.md)
cd frontend && npm install
```

Run the stack or individual services as described in
[docs/getting-started.md](docs/getting-started.md). For backend work the
minimal setup is `make infra` (postgres + redis) plus the native dev targets.

## Test, lint, types

```bash
make test        # backend suite (cd backend && uv run pytest tests -x -q)
make lint        # ruff check + format check + frontend tsc --noEmit (no eslint — run it separately)
make fmt         # autofix ruff issues
make eval        # evalkit suite via the live API (real versions under eval:* projects)
make eval-quick  # same suite in scratch mode — no API/Redis/DB writes
cd frontend && npm run lint   # eslint + eslint-plugin-react-hooks (eslint.config.js)
cd frontend && npm test       # vitest unit tests (~11 *.test.ts files)
cd mesh_server && uv run pytest     # sidecar suite
```

`backend/tests/cad` must also run inside the executor image
(`docker compose run cad-executor pytest tests/cad` — with the tests mounted).

## Conventions

The authoritative rules live in [`AGENTS.md`](AGENTS.md) — they apply to
humans and agents alike. The short version:

- Python: type hints everywhere, `ruff` line-length 100, async FastAPI
  handlers, pydantic v2 for all JSON, `structlog` instead of `print`, no
  business logic in routers.
- TypeScript strict; functional components; TanStack Query for server state;
  Tailwind only.
- Frontend is fully internationalised (`react-i18next`): user-facing strings go
  through `t()` / `i18n.t()` — never hardcode prose in components. Namespaced
  locale files live in `frontend/src/i18n/locales/<lang>/` (en is the
  reference; de/es/fr ship today); add the same key to every language (the
  `i18nParity` vitest test discovers languages from the locales dir, enforces
  key-set parity and flags unresolvable keys). To add a language: create
  `locales/<lang>/*.json` mirroring `en`, register it in
  `frontend/src/i18n/index.ts` (`SUPPORTED`, `LANG_LABEL`, `resources`) — the
  switcher and parity test pick it up automatically. Keep every `{{var}}`
  interpolation placeholder identical across languages.
- Backend strings (job log lines, warnings, DFM codes) stay English by design —
  only UI chrome is translated.
- Never put the LLM API key in DB/logs/traces.
- Every LLM prompt lives in `backend/cadx/agent/prompts/*.md` — never inline
  prompt text in Python.
- All file paths go through `cadx/storage.py`; DB writes use short
  transactions.
- LLM-generated code only ever runs through `cad/sandbox.py` inside the
  executor — never `exec()` in-process. Allowed imports: `build123d`,
  `bd_warehouse.*`, `cadlib`, `math`, `numpy`, `typing`, `dataclasses`.
- Shell scripts stay bash-3.2 compatible (macOS system bash) and source the
  shared helpers in `scripts/lib/` (`ui.sh` for output/steps/logs,
  `pkg.sh` for venv/installer/download boilerplate) instead of re-rolling
  `echo`/pip/uv handling. Every script accepts `--help`, `--quiet`,
  `--verbose` and `--no-color`.
- Commits: one per milestone step, descriptive message.

## Testing rules

- Agent-loop tests use `FakeLLM` — never call a real LLM in tests.
- Mesh work uses `FakeMeshClient` / `MESH_FAKE=1` for GPU-free runs.
- Renders use pyvista offscreen; tests must skip gracefully when rendering is
  unavailable.
- Extend `cadlib` deliberately — its internals are guarded by the same import
  rules as generated code.

## Documentation

Docs live in [`docs/`](docs/README.md). Keep them in sync with the code:
verify commands, settings keys and endpoints before merging, and prefer links
over duplicating facts between files.
