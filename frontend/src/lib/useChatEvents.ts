import { useEffect, useRef, useState } from "react";
import { chatEventsUrl } from "./api";
import type { Chat, ChatMessage } from "./types";

/** Live tool-call activity shown inside the current assistant turn. */
export interface ToolEvent {
  seq: number;
  name: string;
  args?: Record<string, unknown>;
  ok?: boolean;
  result?: Record<string, unknown>;
  done: boolean;
}

interface ChatStreamState {
  chat: Chat | null;
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  turnRunning: boolean;
  connected: boolean;
}

const mkMsg = (
  id: string,
  chatId: string,
  role: ChatMessage["role"],
  content: string,
  meta: ChatMessage["meta"] = {},
): ChatMessage => ({
  id,
  chat_id: chatId,
  role,
  content,
  asset_ids: [],
  meta,
  pending: false,
  created_at: new Date().toISOString(),
});

/** Subscribes to the chat SSE stream (`/chats/{id}/events` — snapshot-first,
 *  non-terminating). Reconnects automatically; `onEvent` fires for every
 *  event so the page can invalidate queries (e.g. refetch assets/jobs). */
export function useChatEvents(
  chatId: string | undefined,
  onEvent?: (type: string) => void,
): ChatStreamState {
  const [state, setState] = useState<ChatStreamState>({
    chat: null,
    messages: [],
    toolEvents: [],
    turnRunning: false,
    connected: false,
  });
  const cbRef = useRef(onEvent);
  useEffect(() => {
    cbRef.current = onEvent;
  }, [onEvent]);
  const seqRef = useRef(0);

  useEffect(() => {
    if (!chatId) return;
    setState({
      chat: null,
      messages: [],
      toolEvents: [],
      turnRunning: false,
      connected: false,
    });
    seqRef.current = 0;
    const es = new EventSource(chatEventsUrl(chatId));

    const upsert = (m: ChatMessage) =>
      setState((s) => {
        let i = s.messages.findIndex((x) => x.id === m.id);
        if (i < 0 && m.meta?.streaming)
          // adopt the synthetic placeholder instead of a second bubble
          i = s.messages.findIndex((x) => x.id === `tmp:${chatId}`);
        if (i < 0) return { ...s, messages: [...s.messages, m] };
        const msgs = [...s.messages];
        msgs[i] = m;
        return { ...s, messages: msgs };
      });

    const patch = (id: string, f: (m: ChatMessage) => ChatMessage) =>
      setState((s) => {
        let i = s.messages.findIndex((x) => x.id === id);
        if (i < 0) {
          // a snapshot missed turn_start (row not yet committed at read
          // time) — the tmp placeholder takes the real id on first patch
          i = s.messages.findIndex((x) => x.id === `tmp:${chatId}`);
          if (i < 0) return s;
          const msgs = [...s.messages];
          msgs[i] = f({ ...msgs[i], id });
          return { ...s, messages: msgs };
        }
        const msgs = [...s.messages];
        msgs[i] = f(msgs[i]);
        return { ...s, messages: msgs };
      });

    const handlers: Record<string, (d: unknown) => void> = {
      snapshot: (d) => {
        const snap = d as { chat: Chat; messages: ChatMessage[] };
        setState((s) => {
          const ids = new Set(snap.messages.map((m) => m.id));
          // keep streaming placeholders that raced ahead of the snapshot
          // read (amsg row committed after turn_running but before its row);
          // a turn that's already over can't still own a placeholder
          const extras = snap.chat.turn_running
            ? s.messages.filter((m) => !ids.has(m.id) && m.meta?.streaming)
            : [];
          let messages = [...snap.messages, ...extras];
          if (
            snap.chat.turn_running &&
            !messages.some((m) => m.meta?.streaming)
          ) {
            // turn claimed but the assistant row isn't in this snapshot —
            // deltas will adopt the real id via patch()
            messages = [
              ...messages,
              mkMsg(`tmp:${chatId}`, chatId, "assistant", "", {
                streaming: true,
              }),
            ];
          }
          return {
            ...s,
            chat: snap.chat,
            messages,
            turnRunning: snap.chat.turn_running,
            connected: true,
          };
        });
      },
      message: (d) => {
        const m = (d as { message: ChatMessage }).message;
        if (m) upsert(m);
      },
      turn_start: (d) => {
        const { message_id } = d as { message_id: string };
        setState((s) => ({
          ...s,
          turnRunning: true,
          toolEvents: [],
          chat: s.chat ? { ...s.chat, turn_running: true } : s.chat,
        }));
        upsert(
          mkMsg(message_id, chatId, "assistant", "", { streaming: true }),
        );
      },
      message_delta: (d) => {
        const { message_id, text } = d as { message_id: string; text: string };
        patch(message_id, (m) => ({ ...m, content: text }));
      },
      message_update: (d) => {
        const { message_id, content } = d as {
          message_id: string;
          content: string;
        };
        patch(message_id, (m) => ({ ...m, content }));
      },
      tool_start: (d) => {
        const t = d as { name: string; args?: Record<string, unknown> };
        setState((s) => ({
          ...s,
          toolEvents: [
            ...s.toolEvents,
            { seq: seqRef.current++, name: t.name, args: t.args, done: false },
          ],
        }));
      },
      tool_result: (d) => {
        const t = d as { name: string; ok: boolean; result?: Record<string, unknown> };
        setState((s) => {
          const evs = [...s.toolEvents];
          // merge into the newest open call with the same name
          for (let i = evs.length - 1; i >= 0; i--) {
            if (evs[i].name === t.name && !evs[i].done) {
              evs[i] = { ...evs[i], ok: t.ok, result: t.result, done: true };
              return { ...s, toolEvents: evs };
            }
          }
          return {
            ...s,
            toolEvents: [
              ...evs,
              { seq: seqRef.current++, name: t.name, ok: t.ok, result: t.result, done: true },
            ],
          };
        });
      },
      chat_update: (d) => {
        // the chat's binding changed (agent create_project/attach or the
        // UI attach endpoint) — refresh chat so header + composer react
        const { chat } = d as { chat: Chat };
        if (chat) setState((s) => ({ ...s, chat }));
      },
      turn_end: (d) => {
        const { message_id, meta, cancelled, exhausted } = d as {
          message_id: string;
          meta?: ChatMessage["meta"];
          cancelled?: boolean;
          exhausted?: boolean;
        };
        // meta carries the persisted tool_log/spawned/cancelled/exhausted —
        // merging it keeps chips, job cards and the 3D panel after the turn
        patch(message_id, (m) => ({
          ...m,
          meta: {
            ...m.meta,
            ...(meta ?? {}),
            streaming: false,
            ...(cancelled !== undefined ? { cancelled } : {}),
            ...(exhausted !== undefined ? { exhausted } : {}),
          },
        }));
        setState((s) => ({
          ...s,
          turnRunning: false,
          chat: s.chat ? { ...s.chat, turn_running: false } : s.chat,
        }));
      },
      error: () => {
        setState((s) => ({
          ...s,
          turnRunning: false,
          chat: s.chat ? { ...s.chat, turn_running: false } : s.chat,
          messages: s.messages.map((m) =>
            m.meta?.streaming
              ? { ...m, meta: { ...m.meta, streaming: false, error: "turn failed" } }
              : m,
          ),
        }));
      },
    };

    for (const [type, fn] of Object.entries(handlers)) {
      es.addEventListener(type, (e) => {
        try {
          fn(JSON.parse((e as MessageEvent).data));
        } catch (err) {
          console.warn(`chatEvents: malformed "${type}" event`, err);
        }
        cbRef.current?.(type);
      });
    }
    es.onopen = () => setState((s) => ({ ...s, connected: true }));
    es.onerror = () => setState((s) => ({ ...s, connected: false }));
    return () => es.close();
  }, [chatId]);

  return state;
}
