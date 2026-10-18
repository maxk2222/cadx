import { useQuery } from "@tanstack/react-query";
import { ExternalLink, XCircle } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, errMsg } from "../../lib/api";
import { stepLabel } from "../../lib/steps";
import type { SpawnedRef } from "../../lib/types";
import LivePreview from "../LivePreview";
import StatusBadge from "../StatusBadge";
import { ProgressBar, useToast } from "../ui";

const TERMINAL = ["succeeded", "failed", "cancelled", "needs_review"];

/** Inline card for a job/version the chat agent spawned — polls job status
 *  slowly (chat already has an SSE stream; this is only a fallback), shows
 *  the live render while running and a static preview + deep link once
 *  finished. */
export default function ChatJobCard({ spawned }: { spawned: SpawnedRef }) {
  const toast = useToast();
  const [cancelling, setCancelling] = useState(false);
  const job = useQuery({
    queryKey: ["job", spawned.job_id],
    queryFn: () => api.getJob(spawned.job_id!),
    enabled: !!spawned.job_id,
    refetchInterval: (q) =>
      q.state.data && !TERMINAL.includes(q.state.data.status) ? 15000 : false,
  });
  const version = useQuery({
    queryKey: ["version", spawned.version_id],
    queryFn: () => api.getVersion(spawned.version_id!),
    enabled: !!spawned.version_id && !!job.data && TERMINAL.includes(job.data.status),
  });
  const j = job.data;
  const active = j ? !TERMINAL.includes(j.status) : true;
  const vid = spawned.version_id;

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-1.5">
        <span className="text-xs font-medium text-zinc-300">
          {spawned.tool ?? spawned.kind ?? "job"}
        </span>
        {j && <StatusBadge status={j.status} />}
        {j?.step && active && (
          <span className="truncate text-xs text-zinc-500">
            {stepLabel(j.step)}
          </span>
        )}
        {j && ["queued", "running"].includes(j.status) && (
          <button
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-red-400 hover:bg-red-950/50 disabled:opacity-40"
            title="abort this job"
            aria-label="cancel job"
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
            <XCircle className="h-3 w-3" /> cancel
          </button>
        )}
        {vid && (
          <Link
            to={`/versions/${vid}`}
            className="ml-auto flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300"
          >
            version
            <ExternalLink className="h-3 w-3" />
          </Link>
        )}
      </div>
      {j && active && (
        <div className="px-3 pt-2">
          <ProgressBar
            value={j.progress}
            tone={j.status === "waiting_for_input" ? "amber" : "blue"}
          />
        </div>
      )}
      {vid && (
        <div className="relative m-3 aspect-video overflow-hidden rounded bg-zinc-900">
          {active ? (
            <LivePreview versionId={vid} active={active} />
          ) : version.data ? (
            <img
              src={api.previewUrl(vid, "iso")}
              alt="version preview"
              className="h-full w-full object-contain"
            />
          ) : (
            <div className="h-full w-full" />
          )}
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
            the agent needs your input — answer in the chat
          </div>
          <ul className="list-disc space-y-1 pl-4 text-xs text-amber-200/90">
            {j.questions.map((q, i) => (
              <li key={i}>{q.question}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
