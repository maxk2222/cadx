import { useEffect, useRef, useState } from "react";
import { api, jobEventsUrl } from "../lib/api";
import type { Job } from "../lib/types";

interface Event {
  type: string;
  step?: string;
  iteration?: number;
  message?: string;
  status?: string;
  score?: number;
  questions?: unknown[];
}

export default function JobLog({ jobId, onDone }: { jobId: string | null; onDone?: () => void }) {
  const [events, setEvents] = useState<Event[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [answers, setAnswers] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setEvents([]);
    setJob(null);
    if (!jobId) return;
    let closed = false;
    api.getJob(jobId).then(setJob).catch(() => {});
    const es = new EventSource(jobEventsUrl(jobId));
    es.addEventListener("status", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      if (d.log) setEvents((ev) => [...ev, ...d.log.trim().split("\n").map((l: string) => ({ type: "log", message: l }))]);
    });
    es.addEventListener("step", (e) => {
      setEvents((ev) => [...ev, JSON.parse((e as MessageEvent).data)]);
    });
    es.addEventListener("question", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setEvents((ev) => [...ev, { type: "question", message: "waiting for your answers" }]);
      api.getJob(jobId).then((j) => {
        setJob(j);
        setAnswers((j.questions ?? []).map(() => ""));
      });
    });
    es.addEventListener("done", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setEvents((ev) => [...ev, { type: "done", ...d }]);
      if (!closed) onDone?.();
      es.close();
    });
    es.addEventListener("end", () => es.close());
    es.onerror = () => {
      /* EventSource retries automatically */
    };
    return () => {
      closed = true;
      es.close();
    };
  }, [jobId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const waiting = job?.status === "waiting_for_input" && job.questions?.length;

  return (
    <div className="flex flex-col gap-2">
      <div className="h-56 overflow-y-auto rounded bg-zinc-900 p-2 font-mono text-xs leading-5">
        {events.map((e, i) => (
          <div key={i} className={e.type === "done" ? "text-emerald-400" : e.type === "question" ? "text-amber-400" : "text-zinc-300"}>
            {e.step ? `[${e.step}${e.iteration ? `:${e.iteration}` : ""}] ` : ""}
            {e.message ?? e.type}
            {e.type === "done" ? ` (${e.status}${e.score != null ? `, score ${e.score}` : ""})` : ""}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      {waiting ? (
        <div className="rounded border border-amber-700 bg-amber-950/40 p-3">
          <div className="mb-2 text-sm font-semibold text-amber-300">The agent needs answers:</div>
          {job!.questions!.map((q, i) => (
            <div key={i} className="mb-2">
              <div className="text-xs text-zinc-300">{q.question}</div>
              {q.why_it_matters ? <div className="text-[10px] text-zinc-500">{q.why_it_matters}</div> : null}
              <input
                className="mt-1 w-full rounded bg-zinc-800 px-2 py-1 text-sm"
                placeholder={q.default_assumption ? `default: ${q.default_assumption}` : ""}
                value={answers[i] ?? ""}
                onChange={(e) => setAnswers((a) => a.map((v, j) => (j === i ? e.target.value : v)))}
              />
            </div>
          ))}
          <button
            className="mt-1 rounded bg-amber-600 px-3 py-1 text-sm font-medium hover:bg-amber-500"
            onClick={() =>
              api
                .answerJob(jobId!, answers.map((a, i) => ({ answer: a || job!.questions![i].default_assumption || "use the default" })))
                .then((r) => {
                  setJob(null);
                  onDone?.();
                })
            }
          >
            Submit answers
          </button>
        </div>
      ) : null}
    </div>
  );
}
