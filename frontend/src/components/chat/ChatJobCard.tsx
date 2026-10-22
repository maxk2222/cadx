import { useQuery } from "@tanstack/react-query";
import {
  ChevronDown,
  ChevronRight,
  ExternalLink,
  XCircle,
} from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { api, errMsg } from "../../lib/api";
import { etaSeconds, fmtDur, fmtTokens } from "../../lib/liveStats";
import { stepLabel } from "../../lib/steps";
import type { JobLive, SpawnedRef } from "../../lib/types";
import { useJobEvents } from "../../lib/useJobEvents";
import { LogRow } from "../JobLog";
import LivePreview from "../LivePreview";
import StatusBadge from "../StatusBadge";
import {
  PipelineStepper,
  ProgressRing,
  Sparkline,
  StatChip,
  useNow,
  useStickyBottom,
  useToast,
} from "../ui";

const TERMINAL = ["succeeded", "failed", "cancelled", "needs_review"];

/** Inline card for a job/version the chat agent spawned — subscribes to the
 *  job SSE stream for live state; the slow `useQuery` poll is only a fallback
 *  for terminal jobs whose stream already closed. Shows the live render while
 *  running and a static preview + deep link once finished. */
export default function ChatJobCard({ spawned }: { spawned: SpawnedRef }) {
  const { t } = useTranslation("chat");
  const toast = useToast();
  const [cancelling, setCancelling] = useState(false);
  const [logOpen, setLogOpen] = useState(false);
  const stream = useJobEvents(spawned.job_id ?? null);
  // live lines + replayed backlog; pinned to bottom while open
  const { boxRef: logBoxRef, onScroll: onLogScroll } = useStickyBottom(
    logOpen ? stream.events : null,
  );
  const job = useQuery({
    queryKey: ["job", spawned.job_id],
    queryFn: () => api.getJob(spawned.job_id!),
    enabled: !!spawned.job_id,
    // per-job SSE is the transport — the HTTP poll only runs while the
    // stream is down (degraded mode), never alongside it
    refetchInterval: (q) =>
      !stream.connected &&
      q.state.data &&
      !TERMINAL.includes(q.state.data.status)
        ? 15000
        : false,
  });
  const j = stream.job ?? job.data;
  const active = j ? !TERMINAL.includes(j.status) : true;
  const now = useNow(1000, active);
  const version = useQuery({
    queryKey: ["version", spawned.version_id],
    queryFn: () => api.getVersion(spawned.version_id!),
    enabled: !!spawned.version_id && !!j && TERMINAL.includes(j.status),
  });
  // prefer the stream's structured live state (fresh from SSE / synced job
  // row); fall back to the persisted snapshot on the polled job row so done
  // cards keep showing their stats without an open stream
  const streamLive = (stream.job?.live ?? stream.live) as JobLive;
  const live: JobLive = Object.keys(streamLive).length
    ? streamLive
    : ((job.data?.live ?? {}) as JobLive);
  const hasLive = Object.keys(live).length > 0;
  const progress = stream.job ? stream.progress : (j?.progress ?? 0);
  const eta = active ? etaSeconds(live, progress) : null;
  const scores = (live.scores ?? []).map((s) => s.score);
  const vid = spawned.version_id;

  const tok = live.tokens ?? {};
  const tokTotal = (tok.prompt_tokens ?? 0) + (tok.completion_tokens ?? 0);
  const tokCached = tok.cached_tokens ?? 0;
  const tokHint =
    tokCached > 0
      ? `${t("calls_n", { count: tok.calls ?? 0 })} · ${Math.round(
          (tokCached / Math.max(1, tok.prompt_tokens ?? 0)) * 100,
        )}% cached`
      : tok.calls
        ? t("calls_n", { count: tok.calls })
        : undefined;

  const elapsedS = j?.started_at
    ? ((j.finished_at ? Date.parse(j.finished_at) : now) -
        Date.parse(j.started_at)) /
      1000
    : null;

  const stepNow = live.step ?? j?.step;
  const stepElapsed =
    live.step && live.step_started_ts
      ? (now - Date.parse(live.step_started_ts)) / 1000
      : null;

  const logT0 = stream.events.find((e) => e.t_ms)?.t_ms;
  const fallbackTail = (j?.log ?? "")
    .trim()
    .split("\n")
    .filter(Boolean)
    .slice(-40);

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-1.5">
        <span className="text-xs font-medium text-zinc-300">
          {spawned.tool ?? spawned.kind ?? t("job")}
        </span>
        {j && <StatusBadge status={j.status} />}
        {stepNow && active && (
          <span className="flex min-w-0 items-center gap-1.5 text-xs text-brand-300">
            <span className="pulse-dot h-1.5 w-1.5 shrink-0 rounded-full bg-brand-400" />
            <span className="truncate">{stepLabel(stepNow)}</span>
            {stepElapsed != null && (
              <span className="shrink-0 font-mono text-zinc-500">
                {fmtDur(stepElapsed)}
              </span>
            )}
          </span>
        )}
        {j && ["queued", "running"].includes(j.status) && (
          <button
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-red-400 hover:bg-red-950/50 disabled:opacity-40"
            title={t("abort_title")}
            aria-label={t("abort_aria")}
            disabled={cancelling}
            onClick={() => {
              setCancelling(true);
              api
                .cancelJob(j.id)
                .then(() => job.refetch())
                .catch((e) => toast(errMsg(e), "error"))
                .finally(() => setCancelling(false));
            }}
          >
            <XCircle className="h-3 w-3" /> {t("common:action.cancel")}
          </button>
        )}
        {vid && (
          <Link
            to={`/versions/${vid}`}
            className="ml-auto flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300"
          >
            {t("version_link")}
            <ExternalLink className="h-3 w-3" />
          </Link>
        )}
      </div>

      {hasLive && (
        <div className="grid grid-cols-2 gap-1.5 px-3 pt-2 sm:grid-cols-4">
          {elapsedS != null && (
            <StatChip label={t("elapsed")} value={fmtDur(elapsedS)} />
          )}
          {tokTotal > 0 && (
            <StatChip label={t("tokens")} value={fmtTokens(tokTotal)} hint={tokHint} />
          )}
          {live.max_iterations != null && live.max_iterations > 0 && (
            <StatChip
              label={t("iteration")}
              value={`${Math.min(j?.iteration ?? 0, live.max_iterations)}/${live.max_iterations}`}
            />
          )}
          {eta != null && <StatChip label={t("eta")} value={`~${fmtDur(eta)}`} />}
        </div>
      )}

      {(hasLive || scores.length > 0) && (
        <div className="flex items-center gap-3 px-3 pt-2">
          <PipelineStepper
            live={live}
            compact
            endMs={
              !active && j?.finished_at ? Date.parse(j.finished_at) : undefined
            }
            className="min-w-0 flex-1"
          />
          {scores.length > 0 && (
            <Sparkline
              values={scores}
              width={72}
              height={22}
              refLine={80}
              className="shrink-0"
            />
          )}
        </div>
      )}

      {(live.runs?.length ?? 0) > 0 && (
        <div className="flex flex-wrap gap-1 px-3 pt-2">
          {live.runs!.map((r, i) => (
            <span
              key={i}
              className="rounded border border-zinc-800 bg-zinc-900 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400"
            >
              {t("run_n", { n: r.index ?? i + 1 })}
              {r.achieved != null && ` · volfrac ${r.achieved}`}
              {r.worst_sf != null && ` · SF ${r.worst_sf}`}
            </span>
          ))}
        </div>
      )}

      {vid && (
        <div className="relative m-3 aspect-video overflow-hidden rounded bg-zinc-900">
          {active ? (
            <LivePreview versionId={vid} active={active} />
          ) : version.data ? (
            <img
              src={api.previewUrl(vid, "iso")}
              alt={t("alt_preview")}
              className="h-full w-full object-contain"
            />
          ) : (
            <div className="h-full w-full" />
          )}
          <div
            className="absolute right-2 top-2 rounded-full bg-zinc-950/70 backdrop-blur-sm"
            title={t("job_progress")}
          >
            <ProgressRing value={progress} size={44} />
          </div>
        </div>
      )}
      {!vid && j && active && (
        <div className="flex justify-end px-3 pt-2">
          <ProgressRing value={progress} size={44} />
        </div>
      )}
      {j?.status === "failed" && j.log && (
        <pre className="mx-3 mb-3 max-h-24 overflow-auto rounded bg-red-950/40 p-2 text-xs text-red-300">
          {j.log.slice(-600)}
        </pre>
      )}
      {j?.status === "waiting_for_input" && j.questions && (
        <div className="mx-3 mb-3 rounded border border-amber-800/60 bg-amber-950/30 p-2">
          <div className="mb-1 text-xs font-medium text-amber-300">
            {t("agent_needs_input")}
          </div>
          <ul className="list-disc space-y-1 pl-4 text-xs text-amber-200/90">
            {j.questions.map((q, i) => (
              <li key={i}>{q.question}</li>
            ))}
          </ul>
        </div>
      )}

      {(stream.events.length > 0 || fallbackTail.length > 0 || active) && (
        <div className="border-t border-zinc-800">
          <button
            className="flex w-full items-center gap-1.5 px-3 py-1.5 text-[11px] text-zinc-500 transition-colors hover:text-zinc-300"
            onClick={() => setLogOpen((o) => !o)}
            aria-expanded={logOpen}
          >
            {logOpen ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            {t("log")}
            {stream.events.length > 0 && (
              <span className="font-mono text-zinc-600">
                {stream.events.length}
              </span>
            )}
            {active && (
              <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-400" />
            )}
          </button>
          {logOpen && (
            <div
              ref={logBoxRef}
              onScroll={onLogScroll}
              className="max-h-48 overflow-y-auto px-3 pb-2 font-mono text-[11px] leading-relaxed"
            >
              {stream.events.length > 0
                ? stream.events.map((e, i) => (
                    <LogRow key={i} e={e} t0={logT0} />
                  ))
                : fallbackTail.map((l, i) => (
                    <div key={i} className="text-zinc-400">
                      {l}
                    </div>
                  ))}
              {!stream.events.length && !fallbackTail.length && (
                <div className="text-zinc-600">{t("waiting_output")}</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
