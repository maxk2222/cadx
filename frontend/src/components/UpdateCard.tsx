/** Software update card — check for updates, run a full-stack update, watch
 * the restart. Proxied via /api/system/* to the host-side updater daemon;
 * when it's offline the card degrades to instructions for the terminal. */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { RefreshCw, RotateCcw, ArrowUpCircle } from "lucide-react";

import { api, errMsg } from "../lib/api";
import { behindBadge, isUpdateActive, logTail, updatePhaseLabel } from "../lib/updateStatus";
import { Btn, Card, Chip, ConfirmModal, SectionTitle } from "./ui";

export default function UpdateCard() {
  const { t } = useTranslation("settings");
  const qc = useQueryClient();
  const [ref, setRef] = useState<string>("");
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [jobsWarn, setJobsWarn] = useState(0);
  const [rollbackOpen, setRollbackOpen] = useState(false);
  const [watching, setWatching] = useState(false);
  const [seenRestart, setSeenRestart] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);

  const version = useQuery({ queryKey: ["system", "version"], queryFn: api.systemVersion, retry: false });
  const check = useQuery({
    queryKey: ["update", "check"],
    queryFn: api.updateCheck,
    retry: false,
    staleTime: 60_000,
    enabled: version.data?.updater_ok !== false,
  });
  const status = useQuery({
    queryKey: ["update", "status"],
    queryFn: api.updateStatus,
    retry: false,
    // poll while an update is in flight OR while we expect one to have started
    refetchInterval: (q) => (watching || isUpdateActive(q.state.data) ? 2000 : false),
  });

  const active = isUpdateActive(status.data) || watching;
  const restarting = watching && status.isError; // api gone => mid-restart
  const done = status.data?.state === "done" && !watching;
  const failed = status.data?.state === "failed";

  // follow the log tail while it's growing
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [status.data?.log_tail]);

  // adopt an already-running update (page reload mid-update)
  useEffect(() => {
    if (status.data && isUpdateActive(status.data)) setWatching(true);
  }, [status.data]);

  // end watching once the daemon reports a terminal state and api is back
  useEffect(() => {
    if (!watching || !status.data) return;
    if (status.data.state === "done" || status.data.state === "failed") {
      setWatching(false);
      setSeenRestart(true);
      qc.invalidateQueries({ queryKey: ["system", "version"] });
      qc.invalidateQueries({ queryKey: ["update", "check"] });
    }
  }, [watching, status.data, qc]);

  const start = useMutation({
    mutationFn: (confirmJobs: boolean) =>
      api.updateStart({ ref: ref || undefined, confirm_jobs: confirmJobs }),
    onSuccess: () => {
      setConfirmOpen(false);
      setJobsWarn(0);
      setWatching(true);
      setSeenRestart(false);
      qc.invalidateQueries({ queryKey: ["update", "status"] });
    },
    onError: (e) => {
      const m = errMsg(e);
      const n = m.match(/(\d+) active job/);
      if (n) setJobsWarn(parseInt(n[1], 10)); // stays in the confirm modal
      else setConfirmOpen(false);
    },
  });

  const checkNow = useMutation({
    mutationFn: () => api.updateCheckRun(ref || undefined),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["update", "check"] }),
  });

  const rollback = useMutation({
    mutationFn: api.updateRollback,
    onSuccess: () => {
      setRollbackOpen(false);
      setWatching(true);
      qc.invalidateQueries({ queryKey: ["update", "status"] });
    },
  });

  const v = version.data;
  const c = check.data;
  const offline = version.isSuccess && v?.updater_ok === false;
  const behind = behindBadge(c?.behind);
  const branches = c?.branches?.length ? c.branches : ["dev", "main"];
  const targetRef = ref || c?.current_ref || branches[0];

  return (
    <Card className="p-4 lg:col-span-2">
      <SectionTitle>{t("upd_title")}</SectionTitle>

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-zinc-300">
          v{v?.version ?? "?"}
          {v?.git && (
            <>
              {" "}· <code className="text-zinc-400">{v.git.sha}</code> · {v.git.branch}
            </>
          )}
        </span>
        {(v?.git?.dirty ?? 0) > 0 && <Chip>{t("upd_dirty_n", { n: v?.git?.dirty })}</Chip>}
        {behind && <Chip>{t("upd_behind", { n: behind })}</Chip>}
        {offline && <Chip>{t("upd_offline_chip")}</Chip>}
      </div>

      {offline ? (
        <p className="mt-3 text-xs text-zinc-500">
          {t("upd_offline_pre")}{" "}
          <code className="text-zinc-300">scripts/start.sh</code> {t("upd_offline_mid")}{" "}
          <code className="text-zinc-300">scripts/update.sh</code>.
        </p>
      ) : (
        <>
          {/* channel toggle — remote branches from the last check */}
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <span className="text-xs text-zinc-500">{t("upd_channel")}</span>
            {branches.map((b) => (
              <label key={b} className="flex items-center gap-1.5 text-sm text-zinc-300">
                <input
                  type="radio"
                  name="update-ref"
                  checked={targetRef === b}
                  onChange={() => setRef(b)}
                  disabled={active}
                />
                {b}
              </label>
            ))}
          </div>

          {(c?.dirty?.length ?? 0) > 0 && (
            <p className="mt-2 text-xs text-amber-400">
              {t("upd_dirty_warn", { n: c?.dirty.length })}
            </p>
          )}

          {c?.commits?.length ? (
            <ul className="mt-2 max-h-32 space-y-1 overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950/60 p-2 text-xs text-zinc-400">
              {c.commits.slice(0, 8).map((cm) => (
                <li key={cm.sha} className="flex gap-2">
                  <code className="shrink-0 text-zinc-500">{cm.sha}</code>
                  <span className="truncate">{cm.subject}</span>
                  <span className="ml-auto shrink-0 text-zinc-600">{cm.date}</span>
                </li>
              ))}
              {c.commits.length > 8 && (
                <li className="text-zinc-600">{t("upd_more", { n: c.commits.length - 8 })}</li>
              )}
            </ul>
          ) : null}

          {/* progress area */}
          {active && (
            <div className="mt-3">
              <div className="mb-1.5 flex items-center gap-2 text-sm">
                <RefreshCw className="h-3.5 w-3.5 animate-spin text-brand-400" />
                <span className="text-zinc-200">
                  {restarting ? t("upd_restarting") : updatePhaseLabel(status.data ?? { state: "running", phase: "starting" })}
                </span>
                {status.data?.ref && <Chip>{status.data.ref}</Chip>}
              </div>
              <pre
                ref={logRef}
                className="max-h-40 overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950/80 p-2 text-[11px] leading-relaxed text-zinc-500"
              >
                {logTail(status.data?.log_tail) || t("upd_starting")}
              </pre>
            </div>
          )}

          {done && seenRestart && (
            <div className="mt-3 flex flex-wrap items-center gap-3 rounded-md border border-emerald-800/60 bg-emerald-950/40 p-3 text-sm">
              <span className="text-emerald-300">
                {t("upd_done_pre")} <code>{status.data?.new_sha?.slice(0, 10) ?? "?"}</code>
                {status.data?.ref ? ` (${status.data.ref})` : ""}
              </span>
              <Btn size="sm" variant="ghost" onClick={() => window.location.reload()}>
                {t("upd_reload")}
              </Btn>
            </div>
          )}

          {failed && (
            <div className="mt-3 rounded-md border border-red-800/60 bg-red-950/40 p-3">
              <p className="text-sm text-red-300">{status.data?.error ?? t("upd_failed")}</p>
              <pre className="mt-2 max-h-32 overflow-y-auto text-[11px] text-zinc-500">
                {logTail(status.data?.log_tail, 10)}
              </pre>
              <div className="mt-2 flex items-center gap-2">
                <Btn
                  size="sm"
                  variant="danger"
                  onClick={() => setRollbackOpen(true)}
                  disabled={!status.data?.prev_sha}
                >
                  <RotateCcw className="h-3.5 w-3.5" /> {t("upd_rollback_btn", { sha: status.data?.prev_sha?.slice(0, 10) ?? "?" })}
                </Btn>
                {status.data?.log && (
                  <span className="text-[11px] text-zinc-600">{t("upd_log")}: {status.data.log}</span>
                )}
              </div>
            </div>
          )}

          <div className="mt-3 flex flex-wrap gap-2">
            <Btn
              size="sm"
              variant="ghost"
              onClick={() => checkNow.mutate()}
              busy={checkNow.isPending}
              disabled={active}
            >
              <RefreshCw className="h-3.5 w-3.5" />
              {checkNow.isPending ? t("upd_checking") : t("upd_check")}
            </Btn>
            <Btn
              size="sm"
              onClick={() => setConfirmOpen(true)}
              disabled={active || (!!c && !c.update_available && behind === undefined)}
              title={c && !c.update_available ? t("upd_uptodate_hint") : undefined}
            >
              <ArrowUpCircle className="h-4 w-4" />
              {t("upd_btn")}
            </Btn>
            {start.isError && !jobsWarn && (
              <span className="self-center text-xs text-red-400">{errMsg(start.error)}</span>
            )}
          </div>
        </>
      )}

      {/* update confirm — extra sentence when active jobs are running */}
      <ConfirmModal
        open={confirmOpen}
        onClose={() => {
          setConfirmOpen(false);
          setJobsWarn(0);
        }}
        onConfirm={() => start.mutate(jobsWarn > 0)}
        title={t("upd_confirm_title")}
        confirmLabel={jobsWarn > 0 ? t("upd_confirm_jobs", { n: jobsWarn }) : t("upd_btn")}
        busy={start.isPending}
        body={
          <div className="space-y-2">
            <p>
                            {t("upd_body_a")} <code className="text-zinc-300">origin/{targetRef}</code>{" "}
              {t("upd_body_b")}
            </p>
            {jobsWarn > 0 && (
              <p className="text-amber-400">
                {t("upd_jobs_warn", { n: jobsWarn })}
              </p>
            )}
            {(c?.dirty?.length ?? 0) > 0 && (
              <p className="text-amber-400">
                {t("upd_dirty_body")}
              </p>
            )}
          </div>
        }
      />

      {/* rollback confirm */}
      <ConfirmModal
        open={rollbackOpen}
        onClose={() => setRollbackOpen(false)}
        onConfirm={() => rollback.mutate()}
        title={t("upd_rb_title")}
        confirmLabel={t("upd_rb_confirm")}
        busy={rollback.isPending}
        body={
          <div className="space-y-2">
            <p>
              {t("upd_rb_body_a")}{" "}
              <code className="text-zinc-300">{status.data?.prev_sha?.slice(0, 10)}</code>{" "}
              {t("upd_rb_body_b")}
            </p>
            <p className="text-amber-400">
              {t("upd_rb_warn")}
            </p>
          </div>
        }
      />
    </Card>
  );
}
