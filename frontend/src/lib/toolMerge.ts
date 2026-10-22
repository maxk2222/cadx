/**
 * Tool-event merge logic — the identity rules that keep persisted
 * `meta.tool_log` entries and live SSE tool events from double-rendering
 * the same call (e.g. after a mid-turn reload).
 *
 * `call_idx` is the server-side index into `meta.tool_log`, sent on
 * tool_start/tool_result since the meta-flush change. Older servers (or
 * tools that bypass the session publish path) send none — merging falls
 * back to positional/name matching.
 */
import type { ToolLogEntry } from "./types";
import type { ToolEvent } from "./useChatEvents";

export interface ToolResultPayload {
  name: string;
  ok: boolean;
  result?: Record<string, unknown>;
  call_idx?: number;
}

/** Merge a tool_result into the live event list: the newest OPEN entry with
 *  the same call_idx (fallback: same name) is closed; otherwise the result
 *  stands alone (start event lost). `seq` is consumed only on append. */
export function mergeToolResult(
  evs: ToolEvent[],
  t: ToolResultPayload,
  seq: number,
): { evs: ToolEvent[]; usedSeq: boolean } {
  for (let i = evs.length - 1; i >= 0; i--) {
    const match =
      !evs[i].done &&
      (t.call_idx != null
        ? evs[i].call_idx === t.call_idx
        : evs[i].name === t.name);
    if (match) {
      const next = [...evs];
      next[i] = { ...next[i], ok: t.ok, result: t.result, done: true };
      return { evs: next, usedSeq: false };
    }
  }
  return {
    evs: [
      ...evs,
      {
        seq,
        name: t.name,
        ok: t.ok,
        result: t.result,
        call_idx: t.call_idx,
        done: true,
      },
    ],
    usedSeq: true,
  };
}

/** Persisted meta.tool_log entry → the same chip shape live events use. */
export function toToolEvent(t: ToolLogEntry, i: number): ToolEvent {
  const r = (t.result ?? {}) as Record<string, unknown>;
  return {
    seq: i,
    call_idx: i,
    name: t.name,
    args: t.args,
    ok: r.ok !== false,
    result: r,
    done: true,
  };
}

/** The full chip list for a message: persisted entries first, then live
 *  events whose call_idx is beyond the persisted tail (unknown call_idx
 *  always passes — it can only be a call newer than the last flush). */
export function mergeToolLogs(
  persisted: ToolLogEntry[] | undefined,
  live: ToolEvent[] | undefined,
  streaming: boolean,
): ToolEvent[] {
  const metaLog = (persisted ?? []).map(toToolEvent);
  if (!streaming) return metaLog;
  return [
    ...metaLog,
    ...(live ?? []).filter(
      (t) => (t.call_idx ?? metaLog.length + t.seq) >= metaLog.length,
    ),
  ];
}
