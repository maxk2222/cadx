import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useEffect, useMemo, useRef, useState } from "react";
import { isTerminalStatus } from "../../lib/jobStream";
import { fmtTokens } from "../../lib/liveStats";
import type { JobLive } from "../../lib/types";
import type { JobEvent } from "../../lib/useJobEvents";
import { useJobEvents } from "../../lib/useJobEvents";
import { StatArea } from "../charts";
import StatusBadge from "../StatusBadge";
import { PipelineStepper, StatChip, StepWaterfall } from "../ui";

/** JobEvent objects keep every field the server pushed (`pushEvent` spreads
 *  the payload) — `live` rides along on step events but isn't on the typed
 *  surface. */
type LiveEvent = JobEvent & { live?: JobLive };

const fmtMetric = (v: number) =>
  Number.isInteger(v) ? String(v) : v.toFixed(2);

/** Collapsible live monitor for the job behind the chat's 3D panel —
 *  pipeline steps, per-step waterfall, score/token charts and topopt runs.
 *  Works for finished jobs too: `useJobEvents` seeds state from the job row,
 *  so persisted `live` snapshots render without an open stream. */
export default function JobLivePanel({ jobId }: { jobId: string | null }) {
  const { t } = useTranslation("chat");
  const stream = useJobEvents(jobId);
  const { events, job, live, connected } = stream;
  const [open, setOpen] = useState(true);
  // default open while the job is live, collapsed for already-terminal jobs —
  // decided once when the job row first arrives (per job id)
  const seenJob = useRef<string | null>(null);
  useEffect(() => {
    if (job && job.id !== seenJob.current) {
      seenJob.current = job.id;
      setOpen(!isTerminalStatus(job.status));
    }
  }, [job]);

  /** cumulative token usage sampled per step event → stacked area series. */
  const tokSeries = useMemo(() => {
    const t0 = events.find((e) => e.t_ms != null)?.t_ms;
    if (t0 == null) return [];
    const rows: { t: number; prompt: number; completion: number; cached: number }[] =
      [];
    for (const e of events) {
      const tk = (e as LiveEvent).live?.tokens;
      if (!tk || e.t_ms == null) continue;
      rows.push({
        t: Math.round((e.t_ms - t0) / 1000),
        prompt: tk.prompt_tokens ?? 0,
        completion: tk.completion_tokens ?? 0,
        cached: tk.cached_tokens ?? 0,
      });
    }
    return rows;
  }, [events]);

  if (!jobId) return null;

  const scoreData = (live.scores ?? []).map((s) => ({ it: s.it, score: s.score }));
  const tok = live.tokens ?? {};
  const metricEntries = Object.entries(live.metrics ?? {});
  const runs = live.runs ?? [];
  const active = !!job && !isTerminalStatus(job.status);

  return (
    <div className="border-t border-edge">
      <button
        className="flex w-full items-center gap-2 px-3 py-2 text-left"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="text-xs font-semibold uppercase tracking-wider text-faint">
          {t("job_monitor")}
        </span>
        {job ? (
          <StatusBadge status={job.status} />
        ) : (
          <span className="text-[10px] text-faint">{t("connecting")}</span>
        )}
        {connected && active && (
          <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-400" />
        )}
        <span className="ml-auto text-faint">
          {open ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </span>
      </button>

      {open && (
        <div className="flex flex-col gap-3 px-3 pb-3">
          <PipelineStepper
            live={live}
            endMs={
              job && isTerminalStatus(job.status) && job.finished_at
                ? Date.parse(job.finished_at)
                : undefined
            }
          />
          <StepWaterfall
            live={live}
            endMs={
              job && isTerminalStatus(job.status) && job.finished_at
                ? Date.parse(job.finished_at)
                : undefined
            }
          />

          {scoreData.length > 1 && (
            <div>
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-faint">
                {t("critique_score")}
              </div>
              <StatArea
                data={scoreData}
                xKey="it"
                height={110}
                areas={[{ key: "score", name: t("score") }]}
              />
            </div>
          )}

          {tokSeries.length > 1 ? (
            <div>
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-faint">
                {t("tokens")}
              </div>
              <StatArea
                data={tokSeries}
                xKey="t"
                height={110}
                areas={[
                  { key: "prompt", name: t("tok_prompt"), color: "#7c9cf5" },
                  { key: "completion", name: t("tok_completion"), color: "#34d399" },
                  { key: "cached", name: t("tok_cached"), color: "#a1a1aa" },
                ]}
              />
            </div>
          ) : (
            (tok.prompt_tokens ?? 0) + (tok.completion_tokens ?? 0) > 0 && (
              <div className="grid grid-cols-3 gap-1.5">
                <StatChip
                  label={t("tok_prompt")}
                  value={fmtTokens(tok.prompt_tokens ?? 0)}
                />
                <StatChip
                  label={t("tok_completion")}
                  value={fmtTokens(tok.completion_tokens ?? 0)}
                />
                <StatChip
                  label={t("tok_cached")}
                  value={fmtTokens(tok.cached_tokens ?? 0)}
                  hint={tok.calls ? t("calls_n", { count: tok.calls }) : undefined}
                />
              </div>
            )
          )}

          {metricEntries.length > 0 && (
            <div className="grid grid-cols-2 gap-1.5">
              {metricEntries.map(([k, v]) => (
                <StatChip key={k} label={k} value={fmtMetric(v)} />
              ))}
            </div>
          )}

          {runs.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {runs.map((r, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 rounded border border-edge bg-panel px-1.5 py-0.5 font-mono text-[10px] text-muted"
                >
                  {r.type && (
                    <span className="rounded bg-brand-900/60 px-1 text-brand-300">
                      {r.type}
                    </span>
                  )}
                  {t("run_n", { n: r.index ?? i + 1 })}
                  {r.achieved != null && ` · volfrac ${r.achieved}`}
                  {r.worst_sf != null && ` · SF ${r.worst_sf}`}
                  {r.applied != null && ` · ${t("applied_n", { n: r.applied })}`}
                </span>
              ))}
            </div>
          )}

          {!job && !events.length && (
            <div className="py-2 text-center text-[11px] text-faint">
              {t("waiting_stream")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
