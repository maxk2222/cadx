import { useQuery } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../../lib/api";
import type { SpawnedRef } from "../../lib/types";
import LivePreview from "../LivePreview";
import StatusBadge from "../StatusBadge";
import { ProgressBar } from "../ui";

const TERMINAL = ["succeeded", "failed", "cancelled", "needs_review"];

/** Inline card for a job/version the chat agent spawned — polls job status,
 *  shows the live render while running and a static preview + deep link
 *  once finished. */
export default function ChatJobCard({ spawned }: { spawned: SpawnedRef }) {
  const job = useQuery({
    queryKey: ["job", spawned.job_id],
    queryFn: () => api.getJob(spawned.job_id!),
    enabled: !!spawned.job_id,
    refetchInterval: (q) =>
      q.state.data && !TERMINAL.includes(q.state.data.status) ? 2000 : false,
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
          <span className="truncate text-xs text-zinc-500">{j.step}</span>
        )}
        {vid && (
          <Link
            to={`/versions/${vid}`}
            className="ml-auto flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300"
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
