/**
 * Shared SSE consumer for job events: connects to /jobs/{id}/events,
 * keeps the event list, job row, structured `live` state and progress.
 * Used by JobLog (full UI) and VersionPage (live chip / preview sync).
 * Event-application logic lives in ./jobStream so it is unit-testable.
 */
import { useEffect, useRef, useState } from "react";
import { api, jobEventsUrl } from "./api";
import {
  applyStatus,
  fetchJobRow,
  isSettledStatus,
  isTerminalStatus,
  pushEvent,
  type JobEvent,
} from "./jobStream";
import type { Job, JobLive } from "./types";

export type { JobEvent };

export interface JobStream {
  events: JobEvent[];
  job: Job | null;
  setJob: (j: Job | null) => void;
  /** structured live state (max_iterations, scores, step timing, sub-progress) */
  live: JobLive;
  progress: number;
  answers: string[];
  setAnswers: (fn: (prev: string[]) => string[]) => void;
  /** SSE currently open — false while reconnecting or after the stream ended */
  connected: boolean;
  /** initial getJob failed (e.g. 404) — job stays null */
  jobError: string | null;
}

const parse = (e: MessageEvent, type: string) => {
  try {
    return JSON.parse(e.data) as Record<string, unknown>;
  } catch (err) {
    console.warn(`jobEvents: malformed "${type}" event`, err);
    return null;
  }
};

export function useJobEvents(jobId: string | null, onDone?: () => void): JobStream {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [live, setLive] = useState<JobLive>({});
  const [progress, setProgress] = useState(0);
  const [answers, setAnswers] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const doneRef = useRef(onDone);
  useEffect(() => {
    doneRef.current = onDone;
  }, [onDone]);

  useEffect(() => {
    setEvents([]);
    setJob(null);
    setLive({});
    setProgress(0);
    setAnswers([]);
    setConnected(false);
    setJobError(null);
    if (!jobId) return;
    let closed = false;
    let doneFired = false;
    const isClosed = () => closed;
    const fireDone = () => {
      if (!doneFired && !closed) {
        doneFired = true;
        doneRef.current?.();
      }
    };
    const syncJob = () =>
      fetchJobRow(api.getJob, jobId, isClosed)
        .then((r) => {
          if (!r) return;
          setJob(r.job);
          setAnswers(r.answers);
        })
        .catch(() => {});

    api
      .getJob(jobId)
      .then((j) => {
        if (closed) return;
        setJob(j);
        setProgress(j.progress);
        if (j.live) setLive(j.live);
        setAnswers((j.questions ?? []).map(() => ""));
      })
      .catch((e) => {
        if (closed) return;
        setJobError(/404|not found/i.test(String(e)) ? "job not found" : String(e));
      });
    const es = new EventSource(jobEventsUrl(jobId));

    es.addEventListener("status", (e) => {
      const d = parse(e as MessageEvent, "status");
      if (!d) return;
      setConnected(true);
      const st = d as { status?: string; live?: JobLive; progress?: number; log?: string };
      if (st.live) setLive(st.live);
      if (typeof st.progress === "number") setProgress(st.progress);
      // the log payload is an authoritative cumulative snapshot — replace
      // prior backlog lines so auto-reconnect replays don't duplicate
      setEvents((ev) => applyStatus(ev, st).events);
      if (isSettledStatus(st.status)) {
        syncJob();
        if (isTerminalStatus(st.status)) fireDone();
        setConnected(false);
        es.close();
      }
    });
    es.addEventListener("step", (e) => {
      const d = parse(e as MessageEvent, "step");
      if (!d) return;
      const st = d as { live?: JobLive; progress?: number };
      setEvents((ev) => pushEvent(ev, { ...d, t_ms: Date.now() } as JobEvent));
      if (st.live) setLive(st.live);
      if (typeof st.progress === "number") setProgress(st.progress);
      else setProgress((p) => Math.min(0.95, p + 0.05));
    });
    es.addEventListener("question", (e) => {
      const d = parse(e as MessageEvent, "question");
      if (!d) return;
      setEvents((ev) => pushEvent(ev, { ...d, t_ms: Date.now() } as JobEvent));
      syncJob();
    });
    es.addEventListener("done", (e) => {
      const d = parse(e as MessageEvent, "done");
      if (!d) return;
      setEvents((ev) => pushEvent(ev, { ...d, t_ms: Date.now() } as JobEvent));
      setProgress(1);
      syncJob();
      fireDone();
      setConnected(false);
      es.close();
    });
    es.addEventListener("end", () => {
      setConnected(false);
      es.close();
    });
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    return () => {
      closed = true;
      es.close();
    };
  }, [jobId]);

  return { events, job, setJob, live, progress, answers, setAnswers, connected, jobError };
}
