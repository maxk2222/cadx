import { Check, CircleDot, Cpu, Flag, Gauge, RotateCcw, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { fmtDuration, fmtSecs, fmtTokens } from "../lib/format";
import { stepLabel } from "../lib/steps";
import type { Job, JobLive } from "../lib/types";
import type { JobStream } from "../lib/useJobEvents";
import { Btn, ProgressBar, useStickyBottom } from "./ui";

const TERMINAL = ["succeeded", "failed", "cancelled"];

function scoreTone(s: number) {
  return s >= 80 ? "text-emerald-400" : s >= 50 ? "text-amber-400" : "text-red-400";
}

/** Iteration-grouped pipeline steps with per-step durations and critique
 *  scores. Built from the structured Job.live state (step_history, current
 *  step, scores) — reconnects restore the full picture. Falls back to the
 *  plain unique-step chips for pre-live jobs (stream events only). */
function StepsTimeline({
  live,
  iteration,
  done,
  events,
}: {
  live: JobLive;
  iteration: number;
  done: boolean;
  events: { type: string; step?: string; iteration?: number }[];
}) {
  const hist = live.step_history ?? [];
  const hasLive = hist.length > 0 || !!live.step;

  if (!hasLive) {
    const steps: string[] = [];
    const seen = new Set<string>();
    for (const e of events) {
      if (e.type !== "step" || !e.step || seen.has(e.step)) continue;
      seen.add(e.step);
      steps.push(e.step);
    }
    if (!steps.length) return null;
    return (
      <div className="flex flex-wrap items-center gap-1">
        {steps.map((s, i) => {
          const active = !done && i === steps.length - 1;
          return (
            <span
              key={`${s}-${i}`}
              className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                active ? "bg-brand-900/60 text-brand-300" : "bg-zinc-800 text-zinc-400"
              }`}
            >
              {active ? (
                <CircleDot className="h-3 w-3 animate-pulse" />
              ) : (
                <Check className="h-3 w-3 text-emerald-500" />
              )}
              {stepLabel(s)}
            </span>
          );
        })}
      </div>
    );
  }

  type Entry = { step: string; dt?: number; active?: boolean };
  const groups = new Map<number, Entry[]>();
  const push = (it: number, e: Entry) => groups.set(it, [...(groups.get(it) ?? []), e]);
  for (const h of hist) push(h.it, { step: h.step, dt: h.dt_s });
  if (live.step) {
    const it = iteration > 0 ? iteration : (hist[hist.length - 1]?.it ?? 0);
    const dt = live.step_started_ts
      ? Math.max(0, (Date.now() - new Date(live.step_started_ts).getTime()) / 1000)
      : undefined;
    push(it, { step: live.step, dt, active: !done });
  }
  const scores = new Map((live.scores ?? []).map((s) => [s.it, s.score]));
  const its = [...groups.keys()].sort((a, b) => a - b);
  if (!its.length) return null;

  return (
    <div className="flex flex-col gap-1">
      {its.map((it) => (
        <div key={it} className="flex flex-wrap items-center gap-1">
          <span className="w-9 shrink-0 text-[9px] font-semibold uppercase tracking-wider text-zinc-600">
            {it > 0 ? `it ${it}` : "setup"}
          </span>
          {groups.get(it)!.map((s, i) => (
            <span
              key={i}
              className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                s.active ? "bg-brand-900/60 text-brand-300" : "bg-zinc-800 text-zinc-400"
              }`}
            >
              {s.active ? (
                <CircleDot className="h-3 w-3 animate-pulse" />
              ) : (
                <Check className="h-3 w-3 text-emerald-500" />
              )}
              {stepLabel(s.step)}
              {s.dt != null && (
                <span className="text-zinc-600">{fmtSecs(s.dt)}</span>
              )}
            </span>
          ))}
          {scores.has(it) && (
            <span
              className={`flex items-center gap-0.5 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-semibold ${scoreTone(scores.get(it)!)}`}
              title="critique score for this iteration"
            >
              <Gauge className="h-3 w-3" /> {scores.get(it)}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

export default function JobLog({
  jobId,
  stream,
  jobs,
  onSelectJob,
  onDone,
}: {
  jobId: string | null;
  stream: JobStream;
  jobs?: Job[];
  onSelectJob?: (id: string) => void;
  onDone?: () => void;
}) {
  const { events, job, setJob, live, progress, answers, setAnswers } = stream;
  const [restarting, setRestarting] = useState(false);
  const [, setTick] = useState(0);
  const nav = useNavigate();
  const { boxRef: logBoxRef, onScroll } = useStickyBottom(events);

  const running = job?.status === "running" || job?.status === "queued";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [running]);

  const waiting = job?.status === "waiting_for_input" && job.questions?.length;
  const isDone = TERMINAL.includes(job?.status ?? "");
  const t0 = events.find((e) => e.t_ms)?.t_ms;

  const elapsed_s = job?.started_at
    ? Math.max(
        0,
        ((job.finished_at ? new Date(job.finished_at).getTime() : Date.now()) -
          new Date(job.started_at).getTime()) /
          1000
      )
    : null;
  const eta_s =
    running && progress > 0.08 && progress < 0.95 && elapsed_s != null
      ? (elapsed_s * (1 - progress)) / progress
      : null;
  const scores = live.scores ?? [];
  const bestScore = scores.length ? Math.max(...scores.map((s) => s.score)) : null;
  const tokens = live.tokens
    ? (live.tokens.prompt_tokens ?? 0) + (live.tokens.completion_tokens ?? 0)
    : 0;

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

      {job && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-zinc-800 bg-zinc-900/60 px-2 py-1.5 text-[11px] text-zinc-400">
          <span className="flex items-center gap-1.5 font-medium text-zinc-300">
            {running && <CircleDot className="h-3 w-3 animate-pulse text-brand-400" />}
            {job.type} · {job.status}
          </span>
          {job.step && (
            <span className="text-brand-300">
              {stepLabel(job.step)}
              {live.sub_progress != null
                ? ` ${Math.round(live.sub_progress * 100)}%`
                : ""}
            </span>
          )}
          {live.max_iterations ? (
            <span
              className="font-mono font-semibold text-zinc-200"
              title="agent iteration / resolved budget"
            >
              it {Math.min(job.iteration, live.max_iterations)}/{live.max_iterations}
            </span>
          ) : job.iteration ? (
            <span className="font-mono text-zinc-300">it {job.iteration}</span>
          ) : null}
          <span className="flex-1" />
          {bestScore != null && (
            <span className={`font-mono ${scoreTone(bestScore)}`} title="best critique score">
              best {bestScore}
            </span>
          )}
          {elapsed_s != null && (
            <span className="font-mono" title="elapsed">
              {fmtSecs(elapsed_s)}
            </span>
          )}
          {eta_s != null && (
            <span className="font-mono text-zinc-500" title="rough estimate from progress">
              ~{fmtSecs(eta_s)} left
            </span>
          )}
          {tokens > 0 && (
            <span className="flex items-center gap-0.5 font-mono text-zinc-500" title="LLM tokens used">
              <Cpu className="h-3 w-3" />
              {fmtTokens(tokens)}
            </span>
          )}
        </div>
      )}

      <StepsTimeline
        live={live}
        iteration={job?.iteration ?? 0}
        done={isDone || !running}
        events={events}
      />

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
      {live.sub_label && live.sub_progress != null && (
        <div className="truncate font-mono text-[10px] text-zinc-500" title={live.sub_label}>
          ↳ {live.sub_label}
        </div>
      )}

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
            {e.t_ms != null && t0 != null && (
              <span className="text-zinc-700">+{fmtSecs((e.t_ms - t0) / 1000)} </span>
            )}
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
              {job.cancel_requested ? "cancel requested" : ""}
              {job.finish_requested ? "finishing early — keeping best" : ""}
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
