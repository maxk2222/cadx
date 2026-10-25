# Deferred Dependency Findings

Dependabot findings that are **not** fixed by lock refresh and are intentionally
deferred. Revisit each release; drop entries once the migration happens.

## Frontend — tailwindcss 3.x chain (moderate)

- `postcss-selector-parser@6.1.4` (needs 7.x) via `postcss-nested`,
  `postcss@8.4.31` (needs 8.5.6+ on that chain) — all come in through
  `tailwindcss ^3.4`.
- Fix requires the **Tailwind 3 → 4 migration** (config format, PostCSS plugin,
  utility renames). Build-time tooling only — the vulnerable selectors are never
  evaluated on end-user input — but plan a dedicated migration PR.

## mesh_server torch extras (install-time only)

The vulnerable packages live in the optional `cuda` / `rocm` / `mps` /
`image-torch` extras — they only materialize when a user installs the native
ML stack on Linux/macOS. They are **not** in the base `uv.lock` resolution for
the docker path and not in CI.

- `transformers 4.57.6`, pinned `>=4.45,<5` (4.x is EOL upstream; patches land
  on 5.x only). 5 alerts incl. CVE-2026-80047 which has **no patched version**
  yet on any line.
- `diffusers 0.31.0`, pinned `>=0.30,<0.32` (fix is 0.38.x — the pin exists
  because the hy3/trellis2/triposg adapters target the 0.31 API surface).
  CVE-2026-44513 / CVE-2026-45804.
- `diskcache 5.6.3` (transitive, no patched version exists).

**Action:** bumping these to the fixed lines is a real adapter migration
(transformers 5 + diffusers 0.38 API changes across
`mesh_server/adapters/{hy3,trellis2,triposg,image}`), not a lock refresh.
Schedule as its own workstream with the torch matrix.

## Fixed by `dac0838` (lock refresh only)

pillow→13.0.0 (both locks — 106 alerts), certifi→2026.10.23,
filelock→4.2.4, setuptools→84.2.1, rembg→2.0.86, vite→6.4.5,
vitest→4.1.12 (+tinypool/@vitest/mocker), react-router-dom→7.18.4,
esbuild→0.25.12.
