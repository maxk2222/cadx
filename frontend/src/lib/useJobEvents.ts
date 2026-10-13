/**
 * Shared SSE consumer for job events: connects to /jobs/{id}/events,
 * keeps the event list, job row, structured `live` state and progress.
 * Used by JobLog (full UI) and VersionPage (live chip / preview sync).
 */
import { useEffect, useRef, useState } from "react";
import { api, jobEventsUrl } from "./api";
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

export interface JobStream {
  events: JobEvent[];
  job: Job | null;
  setJob: (j: Job | null) => void;
  /** structured live state (max_iterations, scores, step timing, sub-progress) */
  live: JobLive;
  progress: number;
  answers: string[];
  setAnswers: (fn: (prev: string[]) => string[]) => void;
}

export function useJobEvents(jobId: string | null, onDone?: () => void): JobStream {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [live, setLive] = useState<JobLive>({});
  const [progress, setProgress] = useState(0);
  const [answers, setAnswers] = useState<string[]>([]);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    setEvents([]);
    setJob(null);
    setLive({});
    setProgress(0);
    setAnswers([]);
    if (!jobId) return;
    let closed = false;
    api
      .getJob(jobId)
      .then((j) => {
        if (closed) return;
        setJob(j);
        setProgress(j.progress);
        if (j.live) setLive(j.live);
        setAnswers((j.questions ?? []).map(() => ""));
      })
      .catch(() => {});
    const es = new EventSource(jobEventsUrl(jobId));

    es.addEventListener("status", (e) => {
      const d = JSON.parse(e.data);
      if (d.live) setLive(d.live);
      if (typeof d.progress === "number") setProgress(d.progress);
      if (d.log) {
        const lines = String(d.log).trim().split("\n").filter(Boolean);
        if (lines.length)
          // replayed backlog: no client timestamp (log lines aren't step events)
          setEvents((ev) => [...ev, ...lines.map((l) => ({ type: "log", message: l }))]);
      }
    });
    es.addEventListener("step", (e) => {
      const d = JSON.parse(e.data);
      setEvents((ev) => [...ev, { ...d, t_ms: Date.now() }]);
      if (d.live) setLive(d.live);
      if (typeof d.progress === "number") setProgress(d.progress);
      else setProgress((p) => Math.min(0.95, p + 0.05));
    });
    es.addEventListener("question", (e) => {
      const d = JSON.parse(e.data);
      setEvents((ev) => [...ev, { ...d, t_ms: Date.now() }]);
      api
        .getJob(jobId)
        .then((j) => {
          setJob(j);
          setAnswers((j.questions ?? []).map(() => ""));
        })
        .catch(() => {});
    });
    es.addEventListener("done", (e) => {
      const d = JSON.parse(e.data);
      setEvents((ev) => [...ev, { ...d, t_ms: Date.now() }]);
      setProgress(1);
      api
        .getJob(jobId)
        .then(setJob)
        .catch(() => {});
      if (!closed) doneRef.current?.();
      es.close();
    });
    es.addEventListener("end", () => es.close());
    es.onerror = () => {};

    return () => {
      closed = true;
      es.close();
    };
  }, [jobId]);

  return { events, job, setJob, live, progress, answers, setAnswers };
}
