# Chat agent

![Chat page — conversation list and project binding](screenshots/chat.png)

A persistent conversation with the CADX agent: `/chat` in the web GUI (or
the **chat** button on a project page). You describe parts, attach reference
images (picker, paste, drag&drop), answer clarification questions — the agent
uses every CADX capability as a tool and shows results inline.

Chats may start **unbound** (no project): you can talk, browse the catalog
and library right away; when you ask for something to be built, the agent
creates a project itself (`create_project`) or binds an existing one
(`attach_project`, `list_projects`). Uploading an image into an unbound chat
lazily creates a project first; the header badge shows the binding and lets
you rebind manually. Project-bound tools are only exposed to the model while
a project is bound.

## What the agent can do in chat

| Tool | Purpose |
|------|---------|
| `generate_part` / `edit_version` | text/image → new version (cad/mesh/hybrid/topopt/auto), or a prompted edit of an existing version |
| `set_params` | re-render an existing version with new parameter values |
| `fem_check` / `optimize_version` / `topopt_resample` / `refine_version` | structural analysis, SIMP optimization, density re-extraction, post-optimization feature restore |
| `organic_restyle` | gyroid/trabecular/voronoi/branches restyling |
| `mesh_edit` | deterministic mesh ops (move/cut/scale/… on `OPS_REGISTRY`) |
| `generate_image` | concept images via the image sidecar (`source="chat"`, reusable as `asset_ids` for generation) |
| `answer_questions` | feed clarification answers into a `waiting_for_input` job |
| `job_status` / `wait_job` | poll or block on a spawned job |
| `update_project_settings` | whitelisted project settings (material, process, …) |
| `save_library_part` | save a finished version as STEP/template library part |
| `create_project` / `attach_project` / `list_projects` | project binding — available even while unbound |
| info tools | `list_versions`, `version_details`, `view_render`, `project_settings`, `job_status`, catalog + library lookups, `view_image` |

Spawned jobs are real `Job`/`Version` rows (shared helpers in `cadx/spawn.py`),
so they appear in the version tree, the Jobs page and the job SSE stream
exactly like GUI-triggered runs. The chat renders them as live cards and — on
success — attaches the version's contact sheet to the tool result so a vision
model can inspect the produced geometry.

## Architecture

```
POST /chats/{id}/messages          → ChatMessage(pending=True) + enqueue
arq "agent": run_chat_turn         → atomic claim → drain → tool loop → stream
GET  /chats/{id}/events (SSE)      → snapshot + message_delta/tool_*/turn_*
```

- `Chat.llm_messages` holds the sanitized OpenAI-format history across turns.
  Image parts persist as `{"type": "asset_ref"}` markers (no base64 in the DB)
  and are rehydrated per turn — as real image parts for the `vision` role, as
  caption text for the `code` role.
- **Crash safety:** pending user messages are moved into `llm_messages`
  *before* the first LLM call, so a worker crash never loses input.
- **Atomic turn claim:** `UPDATE chat SET turn_running=true WHERE NOT
  turn_running` — the API always enqueues; concurrent task invocations
  deduplicate on the flag.
- **Interrupt:** a new user message during a blocked tool call (`wait_job`,
  spawn+wait) returns control early — the spawned job keeps running and can be
  re-awaited via `wait_job`. `chat_spawn_timeout_s` caps the blocking time.
- **Streaming:** token deltas are published as `message_delta` events and the
  assistant row is persisted debounced (~1.5 s), so a page reload mid-turn
  shows the partial text.
- **Reaper:** worker startup clears orphaned `turn_running` flags and
  re-enqueues chats with unanswered pending messages (`reap_orphaned_chats`).

## SSE events (`chat:{id}`)

| Event | Payload |
|-------|---------|
| `snapshot` | `{chat, messages}` — full state on (re)connect |
| `message` | a persisted user message |
| `turn_start` | `{message_id}` — the streaming assistant row |
| `message_delta` | `{message_id, text}` — cumulative snapshot |
| `tool_start` / `tool_result` | `{name, args}` / `{name, ok, result}` |
| `chat_update` | `{chat}` — the binding/title changed (create_project, attach) |
| `message_update` | `{message_id, content}` — final text |
| `turn_end` | `{message_id, cancelled, exhausted}` |
| `error` | `{message}` |

## Settings (`chat_*`)

`chat_model_role` (`vision`|`code`), `chat_tool_max_calls`,
`chat_max_turns`, `chat_spawn_timeout_s`, `chat_history_messages`,
`chat_image_max` — all under *Settings → Agent → chat agent*.

The system prompt lives in `backend/cadx/agent/prompts/chat_system.md`.
