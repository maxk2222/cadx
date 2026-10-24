# Security Policy

## Supported versions

CADX is in early alpha. Only the latest pre-release receives fixes.

| Version         | Supported |
| --------------- | --------- |
| 0.1.0-alpha.x   | ✅ latest alpha only |
| older           | ❌ |

## Reporting a vulnerability

Please **do not** open public issues for security reports. Report privately
via GitHub: open a [security advisory](https://github.com/maxk2222/cadx/security/advisories/new)
on this repository, or contact the maintainer through the GitHub project page.

Include: affected version/commit, environment (OS, GPU tier), a description
of the issue and — where possible — a minimal reproduction. You will get an
acknowledgement; there is no formal SLA while the project is in alpha.

## Scope notes

- CADX is designed as a **local, single-user** tool. Exposing the API port
  beyond localhost is out of the supported deployment model.
- LLM API keys, HF tokens and other runtime secrets are stored
  **Fernet-encrypted at rest** (`backend/cadx/secrets_crypto.py`, `enc:v1:`
  prefix) — the key lives in `CADX_SETTINGS_KEY` or `data/.settings.key`.
  Never paste tokens into issues, logs, traces or screenshots.
- LLM-generated CAD code executes only inside the sandboxed executor
  (`cad/sandbox.py`) — report sandbox escapes with high priority.
