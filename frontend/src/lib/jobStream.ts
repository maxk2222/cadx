/**
 * Pure event-application logic for the job SSE stream (`useJobEvents`).
 * Extracted so the reconnect/dedupe/cap semantics are testable without a DOM.
 *
 * Server contract (backend/cadx/api/routers/sse.py):
 *  - the first `status` event carries the ENTIRE cumulative job log — on every
 *    EventSource auto-reconnect the backlog is replayed and must REPLACE prior
 *    backlog lines, not append;
 *  - `status.status` is authoritative: terminal states and waiting_for_input
 *    mean the server is about to (or already did) close its side — the client
 *    must refetch the job row and close its EventSource;
 *  - after `question`/`done` the server breaks without an `end` frame, so the
 *    reconnect path above is the only guaranteed way to settle the job.
 */
import type { Job, JobLive } from "./types";

export interface JobEvent {
  type: string;
  step?: string;
  iteration?: number;
  message?: string;
  status?: string;
  score?: number;
  /** client-side receipt ms — used for log timestamps + step timing */
  t_ms?: number;
}

export const MAX_EVENTS = 600;

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
const SETTLED = new Set([...TERMINAL, "waiting_for_input"]);

export function isTerminalStatus(s: string | null | undefined): boolean {
  return !!s && TERMINAL.has(s);
}

/** terminal OR waiting_for_input — the server won't send anything more. */
export function isSettledStatus(s: string | null | undefined): boolean {
  return !!s && SETTLED.has(s);
}

/** cumulative `status.log` payload → backlog events (oldest first). */
export function parseLogBacklog(log: string | null | undefined): JobEvent[] {
  if (!log) return [];
  return String(log)
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => ({ type: "log", message: l }));
}

/**
 * Replace all prior backlog (`type:"log"`) events with the authoritative
 * snapshot — identical replays are idempotent; step/question/done events
 * already received are kept (they arrive live, after the backlog).
 */
export function applyBacklog(
  events: JobEvent[],
  log: string | null | undefined,
): JobEvent[] {
  const kept = events.filter((e) => e.type !== "log");
  return capEvents([...parseLogBacklog(log), ...kept]);
}

/** Bound the event list — oldest log lines go first; steps/questions/done stay. */
export function capEvents(events: JobEvent[], cap = MAX_EVENTS): JobEvent[] {
  if (events.length <= cap) return events;
  const out = [...events];
  let overflow = out.length - cap;
  for (let i = 0; i < out.length && overflow > 0; ) {
    if (out[i].type === "log") {
      out.splice(i, 1);
      overflow--;
    } else {
      i++;
    }
  }
  while (overflow-- > 0) out.shift();
  return out;
}

export function pushEvent(
  events: JobEvent[],
  ev: JobEvent,
  cap = MAX_EVENTS,
): JobEvent[] {
  return capEvents([...events, ev], cap);
}

export interface StatusPayload {
  status?: string;
  step?: string | null;
  live?: JobLive;
  progress?: number;
  log?: string;
}

export interface StatusResult {
  events: JobEvent[];
  /** job row is terminal/waiting — caller refetches it and closes the stream */
  settled: boolean;
  /** hard terminal — caller additionally fires the onDone callback (once) */
  terminal: boolean;
}

export function applyStatus(events: JobEvent[], d: StatusPayload): StatusResult {
  const terminal = isTerminalStatus(d.status);
  return {
    events: typeof d.log === "string" ? applyBacklog(events, d.log) : events,
    settled: terminal || d.status === "waiting_for_input",
    terminal,
  };
}

/**
 * Fetch the job row for the answers/refresh path. Returns null when the
 * caller's `isClosed` reports a stale context (jobId switch / unmount) so
 * late responses can't write into the new stream's state.
 */
export async function fetchJobRow(
  getJob: (id: string) => Promise<Job>,
  jobId: string,
  isClosed: () => boolean,
): Promise<{ job: Job; answers: string[] } | null> {
  const job = await getJob(jobId);
  if (isClosed()) return null;
  return { job, answers: (job.questions ?? []).map(() => "") };
}
