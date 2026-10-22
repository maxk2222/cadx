import { CheckCircle2, FolderOpen, PencilRuler, RotateCcw, XCircle } from "lucide-react";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Link, useNavigate } from "react-router-dom";
import { useJobEvents } from "../../lib/useJobEvents";
import JobLog from "../JobLog";
import LivePreview from "../LivePreview";
import { Btn, Card } from "../ui";

const TERMINAL = ["succeeded", "failed", "cancelled"];

/** Post-submit phase: live preview + full job stream (steps, scores, clarify
 *  questions via JobLog) and a terminal-state card that hands off to the
 *  version page. */
export default function WizardProgress({
  jobId,
  versionId,
  projectId,
  projectName,
  onSelectJob,
  onDone,
  onAdjustPrompt,
  onNewPart,
}: {
  jobId: string;
  versionId: string;
  projectId: string | null;
  projectName: string;
  onSelectJob: (id: string) => void;
  onDone?: () => void;
  onAdjustPrompt: () => void;
  onNewPart: () => void;
}) {
  const { t } = useTranslation("wizard");
  const nav = useNavigate();
  const stream = useJobEvents(jobId, onDone);
  const job = stream.job;
  const status = job?.status ?? "queued";
  const done = TERMINAL.includes(status);

  const errorText = useMemo(() => {
    if (status !== "failed" && status !== "cancelled") return "";
    const tail = (job?.log ?? "").trim().split("\n").filter(Boolean).slice(-6);
    return tail.join("\n");
  }, [status, job?.log]);

  return (
    <div>
      <h2 className="text-lg font-semibold">
        {done
          ? status === "succeeded"
            ? t("prog_ready")
            : status === "cancelled"
              ? t("prog_cancelled")
              : t("prog_failed")
          : t("prog_running")}
      </h2>
      {!done && <p className="mt-1 text-sm text-faint">{t("prog_hint")}</p>}
      {projectId && (
        <div className="mt-1 flex items-center gap-1.5 text-[11px] text-faint/80">
          <FolderOpen className="h-3 w-3" />
          {t("prog_project")}{" "}
          <Link to={`/projects/${projectId}`} className="text-brand-400 hover:underline">
            {projectName || projectId}
          </Link>
        </div>
      )}

      <div className="relative mt-4 h-[340px] overflow-hidden rounded-lg border border-edge bg-zinc-950">
        <LivePreview versionId={versionId} active={!done} />
        {!done && (
          <span className="absolute inset-0 -z-10 flex items-center justify-center text-xs text-faint/60">
            {t("version:waiting_render")}
          </span>
        )}
      </div>

      <div className="mt-3">
        <JobLog
          jobId={jobId}
          stream={stream}
          onSelectJob={onSelectJob}
          onDone={onDone}
        />
      </div>

      {done && status === "succeeded" && (
        <Card className="anim-fade-in-up mt-4 flex flex-wrap items-center gap-3 border-emerald-800/60 bg-emerald-950/20 p-4">
          <CheckCircle2 className="h-5 w-5 text-emerald-400" />
          <div className="min-w-0 flex-1 text-sm text-zinc-300">
            {t("prog_done_hint")}
          </div>
          <Btn onClick={() => nav(`/versions/${versionId}?job=${jobId}`)}>
            {t("prog_open")}
          </Btn>
          <Btn variant="ghost" onClick={onNewPart}>
            <RotateCcw className="h-4 w-4" /> {t("prog_another")}
          </Btn>
        </Card>
      )}

      {done && status !== "succeeded" && (
        <Card className="anim-fade-in-up mt-4 border-red-800/60 bg-red-950/20 p-4">
          <div className="flex items-center gap-2 text-sm font-medium text-red-300">
            <XCircle className="h-5 w-5" />
            {status === "cancelled" ? t("prog_cancelled_msg") : t("prog_failed_msg")}
          </div>
          {errorText && (
            <pre className="mt-2 max-h-32 overflow-y-auto whitespace-pre-wrap rounded bg-black/40 p-2 font-mono text-[11px] text-red-200/80">
              {errorText}
            </pre>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            <Btn onClick={onAdjustPrompt}>
              <PencilRuler className="h-4 w-4" /> {t("prog_adjust")}
            </Btn>
            <Btn
              variant="ghost"
              onClick={() => nav(`/versions/${versionId}?job=${jobId}`)}
            >
              {t("prog_open_version")}
            </Btn>
          </div>
        </Card>
      )}
    </div>
  );
}
