import { Check, CircleDot, Flag, RotateCcw, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, jobEventsUrl } from "../lib/api";
import { fmtDuration } from "../lib/format";
import type { Job } from "../lib/types";
import { Btn, ProgressBar, useStickyBottom } from "./ui";

interface Event {
  type: string;
  step?: string;
  iteration?: number;
  message?: string;
  status?: string;
  score?: number;
}

const STEP_LABEL: Record<string, string> = {
  intake: "intake",
  plan: "plan",
  meshgen: "mesh gen",
  imagegen: "image gen",
  orient: "orient",
  meshprocess: "mesh process",
  meshrefine: "mesh refine",
  meshedit: "mesh edit",
  codegen: "codegen",
  improve: "improve",
  autofix: "autofix",
  fem: "fem",
  repair: "repair",
  execute: "build",
  critique: "critique",
  dfm: "dfm",
  finalize: "export",
  budget: "budget",
  done: "done",
};

function stepLabel(s: string) {
  return STEP_LABEL[s] ?? s;
}

function PipelineChips({ events, done }: { events: Event[]; done: boolean }) {
  const steps: { step: string; iteration: number }[] = [];
  const seen = new Set<string>();
  for (const e of events) {
    if (e.type !== "step" || !e.step) continue;
    const k = `${e.step}`;
    if (!seen.has(k)) {
      seen.add(k);
      steps.push({ step: e.step, iteration: e.iteration ?? 0 });
    }
  }
  if (!steps.length) return null;
  const last = steps[steps.length - 1];
  return (
    <div className="flex flex-wrap items-center gap-1">
      {steps.map((s, i) => {
        const active = !done && i === steps.length - 1;
        return (
          <span
            key={`${s.step}-${i}`}
            className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${
              active
                ? "bg-brand-900/60 text-brand-300"
                : "bg-zinc-800 text-zinc-400"
            }`}
          >
            {active ? (
              <CircleDot className="h-3 w-3 animate-pulse" />
            ) : (
              <Check className="h-3 w-3 text-emerald-500" />
            )}
            {stepLabel(s.step)}
          </span>
        );
      })}
      {done && last.step !== "done" && (
        <span className="flex items-center gap-1 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">
          <Check className="h-3 w-3 text-emerald-500" /> done
        </span>
      )}
    </div>
  );
}

export default function JobLog({
  jobId,
  jobs,
  onSelectJob,
  onDone,
}: {
  jobId: string | null;
  jobs?: Job[];
  onSelectJob?: (id: string) => void;
  onDone?: () => void;
}) {
  const [events, setEvents] = useState<Event[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [progress, setProgress] = useState(0);
  const [answers, setAnswers] = useState<string[]>([]);
  const [restarting, setRestarting] = useState(false);
  const [, setTick] = useState(0);
  const nav = useNavigate();
  const { boxRef: logBoxRef, onScroll } = useStickyBottom(events);

  useEffect(() => {
    setEvents([]);
    setJob(null);
    setProgress(0);
    setAnswers([]);
    if (!jobId) return;
    let closed = false;
    api.getJob(jobId).then((j) => {
      setJob(j);
      setProgress(j.progress);
      setAnswers((j.questions ?? []).map(() => ""));
    }).catch(() => {});
    const es = new EventSource(jobEventsUrl(jobId));
    es.addEventListener("status", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      if (d.log)
        setEvents((ev) => [
          ...ev,
          ...d.log
            .trim()
            .split("\n")
            .filter(Boolean)
            .map((l: string) => ({ type: "log", message: l })),
        ]);
    });
    es.addEventListener("step", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setEvents((ev) => [...ev, d]);
      setProgress((p) => Math.min(0.95, p + 0.05));
    });
    es.addEventListener("question", () => {
      setEvents((ev) => [...ev, { type: "question", message: "waiting for your answers" }]);
      api.getJob(jobId).then((j) => {
        setJob(j);
        setAnswers((j.questions ?? []).map(() => ""));
      });
    });
    es.addEventListener("done", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setEvents((ev) => [...ev, { type: "done", ...d }]);
      setProgress(1);
      api.getJob(jobId).then(setJob).catch(() => {});
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

  const running = job?.status === "running" || job?.status === "queued";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [running]);

  const waiting = job?.status === "waiting_for_input" && job.questions?.length;
  const isDone = ["succeeded", "failed", "cancelled"].includes(job?.status ?? "");

  return (
    <div className="flex flex-col gap-2">
      {(jobs?.length ?? 0) > 1 && (
        <select
          className="w-full rounded-md border border-zinc-700 bg-zinc-800/80 px-2 py-1 text-xs text-zinc-300"
          value={jobId ?? ""}
          onChange={(e) => onSelectJob?.(e.target.value)}
        >
          {jobs!.map((j) => (
            <option key={j.id} value={j.id}>
              {j.type} · {new Date(j.created_at).toLocaleTimeString()} · {j.status}
            </option>
          ))}
        </select>
      )}

      <PipelineChips events={events} done={isDone} />

      <div className="flex items-center gap-2">
        <ProgressBar
          value={progress}
          className="flex-1"
          tone={
            job?.status === "failed"
              ? "red"
              : job?.status === "waiting_for_input"
                ? "amber"
                : isDone
                  ? "green"
                  : "blue"
          }
        />
        <span className="w-14 text-right font-mono text-[10px] text-zinc-500">
          {job?.started_at ? fmtDuration(job.started_at, job.finished_at) : "—"}
        </span>
      </div>

      <div
        ref={logBoxRef}
        onScroll={onScroll}
        className="h-56 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950 p-2 font-mono text-[11px] leading-5"
      >
        {events.map((e, i) => (
          <div
            key={i}
            className={
              e.type === "done"
                ? e.status === "failed"
                  ? "text-red-400"
                  : "text-emerald-400"
                : e.type === "question"
                  ? "text-amber-400"
                  : /error|fail/i.test(e.message ?? "")
                    ? "text-red-300"
                    : "text-zinc-400"
            }
          >
            {e.step ? (
              <span className="text-zinc-600">
                [{e.step}
                {e.iteration ? `:${e.iteration}` : ""}]{" "}
              </span>
            ) : (
              ""
            )}
            {e.message ?? e.type}
            {e.type === "done"
              ? ` (${e.status}${e.score != null ? `, score ${e.score}` : ""})`
              : ""}
          </div>
        ))}
        {!events.length && (
          <div className="py-4 text-center text-zinc-600">
            {jobId ? "connecting to job stream…" : "no job selected"}
          </div>
        )}
        <div />
      </div>

      <div className="flex items-center justify-between text-[10px] text-zinc-600">
        <span>
          {job ? (
            <>
              {job.type} · {job.status}
              {job.step ? ` · ${stepLabel(job.step)}${job.iteration ? ` #${job.iteration}` : ""}` : ""}
              {job.cancel_requested ? " · cancel requested" : ""}
              {job.finish_requested ? " · finishing early" : ""}
            </>
          ) : (
            ""
          )}
        </span>
        {running && job && (
          <span className="flex items-center gap-2">
            {job.status === "running" && !job.finish_requested && (
              <button
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-amber-400 hover:bg-amber-950/50"
                title="stop after this iteration and keep the best result so far"
                onClick={() => api.finishJob(job.id).then(() => api.getJob(job.id)).then(setJob)}
              >
                <Flag className="h-3 w-3" /> finish &amp; keep best
              </button>
            )}
            <button
              className="flex items-center gap-1 rounded px-1.5 py-0.5 text-red-400 hover:bg-red-950/50"
              title="abort the job — discards the result"
              onClick={() => api.cancelJob(job.id).then(() => api.getJob(job.id)).then(setJob)}
            >
              <XCircle className="h-3 w-3" /> cancel job
            </button>
          </span>
        )}
        {isDone && job && (
          <button
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-brand-400 hover:bg-brand-950/50 disabled:opacity-40"
            title="re-run this job with the same inputs (new version)"
            disabled={restarting}
            onClick={() => {
              setRestarting(true);
              api
                .restartJob(job.id)
                .then((r) => nav(`/versions/${r.version_id}?job=${r.job_id}`))
                .catch(() => setRestarting(false));
            }}
          >
            <RotateCcw className="h-3 w-3" /> {restarting ? "restarting…" : "restart"}
          </button>
        )}
      </div>

      {waiting ? (
        <div className="rounded-lg border border-amber-700 bg-amber-950/40 p-3">
          <div className="mb-2 text-sm font-semibold text-amber-300">
            The agent needs answers:
          </div>
          {job!.questions!.map((q, i) => (
            <div key={i} className="mb-2">
              <div className="text-xs text-zinc-300">{q.question}</div>
              {q.why_it_matters ? (
                <div className="text-[10px] text-zinc-500">{q.why_it_matters}</div>
              ) : null}
              <input
                className="mt-1 w-full rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1 text-sm"
                placeholder={q.default_assumption ? `default: ${q.default_assumption}` : ""}
                value={answers[i] ?? ""}
                onChange={(e) =>
                  setAnswers((a) => a.map((v, j) => (j === i ? e.target.value : v)))
                }
              />
            </div>
          ))}
          <Btn
            variant="primary"
            className="mt-1 bg-amber-600 hover:bg-amber-500"
            onClick={() =>
              api
                .answerJob(
                  jobId!,
                  answers.map((a, i) => ({
                    answer: a || job!.questions![i].default_assumption || "use the default",
                  }))
                )
                .then((r) => {
                  setJob(null);
                  onSelectJob?.(r.job_id); // answering spawns a resume job
                  onDone?.();
                })
            }
          >
            Submit answers
          </Btn>
        </div>
      ) : null}
    </div>
  );
}
