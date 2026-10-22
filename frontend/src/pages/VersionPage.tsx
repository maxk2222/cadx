import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import ArtifactsTable from "../components/ArtifactsTable";
import CodeView from "../components/CodeView";
import { defaultCompareTarget } from "../components/ComparePanel";
import OrganicEditor from "../components/OrganicEditor";
import PlanPanel from "../components/PlanPanel";
import SpecPanel from "../components/SpecPanel";
import StructuralEditor from "../components/StructuralEditor";
import Inspector from "../components/version/Inspector";
import OverviewTab from "../components/version/OverviewTab";
import QualityTab from "../components/version/QualityTab";
import SaveToLibrary from "../components/version/SaveToLibrary";
import VersionHeader, { VersionStatRow } from "../components/version/VersionHeader";
import {
  Btn,
  ConfirmModal,
  EmptyState,
  Skeleton,
  Tabs,
  useToast,
} from "../components/ui";
import { api, errMsg } from "../lib/api";
import type { GenMode, Job, RunOptions } from "../lib/types";
import { useJobEvents } from "../lib/useJobEvents";
import { useTitle } from "../lib/useTitle";

type Tab = "overview" | "spec" | "code" | "quality" | "files";

export default function VersionPage() {
  const { t } = useTranslation("version");
  const { id = "" } = useParams<{ id: string }>();
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const [tab, setTab] = useState<Tab>("overview");
  const [jobId, setJobId] = useState<string | null>(sp.get("job"));
  const [editPrompt, setEditPrompt] = useState("");
  const [editAssets, setEditAssets] = useState<Set<string>>(new Set());
  const [editLibraryParts, setEditLibraryParts] = useState<Set<string>>(new Set());
  const [editMode, setEditMode] = useState<GenMode>("auto");
  const [editOptions, setEditOptions] = useState<RunOptions>({});
  const [optFor, setOptFor] = useState("");
  const [regenMesh, setRegenMesh] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [optimizeOpen, setOptimizeOpen] = useState(false);
  const [organicOpen, setOrganicOpen] = useState(false);
  const [compareOpen, setCompareOpen] = useState(false);
  const [otherId, setOtherId] = useState<string | null>(null);

  const version = useQuery({
    queryKey: ["version", id],
    queryFn: () => api.getVersion(id),
    refetchInterval: (q) =>
      ["queued", "running", "waiting_for_input"].includes(
        q.state.data?.status ?? ""
      ) || !!q.state.data?.active_job
        ? 4000
        : false,
  });
  const v = version.data;

  const project = useQuery({
    queryKey: ["project", v?.project_id],
    queryFn: () => api.getProject(v!.project_id),
    enabled: !!v?.project_id,
  });
  const versions = useQuery({
    queryKey: ["versions", v?.project_id],
    queryFn: () => api.listVersions(v!.project_id),
    enabled: !!v?.project_id,
    refetchInterval: (q) =>
      (q.state.data ?? []).some(
        (x) =>
          ["queued", "running", "waiting_for_input"].includes(x.status) ||
          !!x.active_job
      )
        ? 4000
        : false,
  });
  const parts = useQuery({
    queryKey: ["parts", v?.project_id],
    queryFn: () => api.listParts(v!.project_id),
    enabled: !!v?.project_id,
  });
  const assets = useQuery({
    queryKey: ["assets", v?.project_id],
    queryFn: () => api.listAssets(v!.project_id),
    enabled: !!v?.project_id,
  });
  const library = useQuery({
    queryKey: ["library"],
    queryFn: api.listLibrary,
    staleTime: 30_000,
  });
  const jobs = useQuery({
    queryKey: ["jobs", id],
    queryFn: () => api.listJobs(id),
  });
  // detail of the version being compared against — gives us its glb artifact
  // for the side-by-side viewer (shares the ["version", id] cache shape)
  const otherVersion = useQuery({
    queryKey: ["version", otherId],
    queryFn: () => api.getVersion(otherId!),
    enabled: compareOpen && !!otherId,
  });
  useTitle(
    version.data
      ? `v${version.data.number}${version.data.name ? ` · ${version.data.name}` : ""} · ${project.data?.name ?? ""}`
      : t("title_fallback")
  );

  // keep the streamed job in sync with the URL — navigating between versions
  // or to a new ?job= must switch the stream, not keep the stale one. With no
  // ?job= param, fall back to the version's live job.
  const jobParam = sp.get("job");
  useEffect(() => {
    setJobId(jobParam ?? v?.active_job?.id ?? null);
  }, [id, jobParam, v?.active_job?.id]);

  // pick latest job once loaded (jobs are sorted newest-first)
  useEffect(() => {
    if (!jobId && jobs.data?.length) setJobId(jobs.data[0].id);
  }, [jobs.data, jobId]);

  // initialize edit options from the version once per version id
  useEffect(() => {
    if (v && v.id !== optFor) {
      setEditPrompt("");
      setEditAssets(new Set(v.asset_ids ?? []));
      setEditMode("auto");
      setRegenMesh(false);
      setEditOptions(v.options ?? {});
      setEditLibraryParts(new Set(v.library_part_ids ?? []));
      setOptFor(v.id);
    }
  }, [v, optFor]);

  // compare state is per-version — don't carry a stale otherId across pages
  useEffect(() => {
    setCompareOpen(false);
    setOtherId(null);
  }, [id]);

  // default compare target: the parent version, else the latest other
  useEffect(() => {
    if (!compareOpen || otherId || !v) return;
    setOtherId(
      defaultCompareTarget(
        (versions.data ?? []).filter((x) => x.id !== v.id),
        v.parent_id
      )
    );
  }, [compareOpen, otherId, v, versions.data]);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["version", id] });
    qc.invalidateQueries({ queryKey: ["jobs", id] });
    qc.invalidateQueries({ queryKey: ["versions", v?.project_id] });
    qc.invalidateQueries({ queryKey: ["live-renders", id] });
  };

  // single SSE stream shared between the JobLog card and the live chips —
  // one EventSource per selected job, drives both (no extra polling)
  const stream = useJobEvents(jobId, refresh);

  const edit = useMutation({
    mutationFn: () =>
      api.edit(id, editPrompt, [...editAssets], {
        mode: editMode,
        regenerate_mesh: regenMesh,
        options: editOptions,
        library_part_ids: [...editLibraryParts],
      }),
    onSuccess: (r) => nav(`/versions/${r.version_id}?job=${r.job_id}`),
  });

  const structuralStarted = (r: { version_id: string; job_id: string }) => {
    setOptimizeOpen(false);
    nav(`/versions/${r.version_id}?job=${r.job_id}`);
  };

  const rename = useMutation({
    mutationFn: (name: string) => api.updateVersion(id, { name }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["version", id] });
      qc.invalidateQueries({ queryKey: ["versions", v?.project_id] });
    },
    onError: (e) => toast(errMsg(e), "error"),
  });

  const del = useMutation({
    mutationFn: () => api.deleteVersion(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["versions", v?.project_id] });
      nav(`/projects/${v?.project_id}`);
    },
    onError: (e) => toast(errMsg(e), "error"),
  });

  const retry = useMutation({
    mutationFn: (jobId: string) => api.restartJob(jobId),
    onSuccess: (r) => nav(`/versions/${r.version_id}?job=${r.job_id}`),
    onError: (e) => toast(errMsg(e), "error"),
  });

  if (version.isLoading)
    return (
      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-5 p-6 xl:grid-cols-4">
        <div className="space-y-4 xl:col-span-3">
          <Skeleton className="h-4 w-56" />
          <Skeleton className="h-8 w-80" />
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-[460px]" />
        </div>
        <div className="space-y-4">
          <Skeleton className="h-64" />
          <Skeleton className="h-44" />
        </div>
      </div>
    );
  if (!v)
    return (
      <div className="mx-auto max-w-3xl p-6">
        <EmptyState title={t("not_found")} hint={t("not_found_hint")} />
      </div>
    );

  // a live job on this version — active_job survives a page reload (the
  // detail endpoint resolves it), the status check covers any residual gap
  const jobLive =
    !!v.active_job ||
    ["queued", "running", "waiting_for_input"].includes(v.status);
  const glb = v.artifacts.find((a) => a.kind === "glb");
  // newest job on this version (list is sorted newest-first) — restart target
  // when the version failed; active_job is null for terminal versions.
  const lastJobId = v.active_job?.id ?? jobs.data?.[0]?.id;
  const assetById = new Map((assets.data ?? []).map((as) => [as.id, as]));
  const libById = new Map((library.data ?? []).map((p) => [p.id, p]));
  const curJob: Job | undefined = (jobs.data ?? []).find((j) => j.id === jobId);
  const liveJob =
    stream.job &&
    ["queued", "running", "waiting_for_input"].includes(stream.job.status)
      ? stream.job
      : null;
  const dfmCount = v.dfm?.warnings?.length ?? 0;
  // compare: candidate versions (never self), the selected one, its glb url
  const others = (versions.data ?? []).filter((x) => x.id !== v.id);
  const otherSel = others.find((x) => x.id === otherId) ?? null;
  const otherGlb =
    otherVersion.data?.artifacts.find((a) => a.kind === "glb")?.url ?? null;

  return (
    <div className="anim-fade-in mx-auto grid max-w-7xl grid-cols-1 gap-5 p-6 lg:p-8 xl:grid-cols-4">
      {/* main column */}
      <div className="min-w-0 xl:col-span-3">
        <VersionHeader
          v={v}
          projectName={project.data?.name}
          projectId={v.project_id}
          part={(parts.data ?? []).find((p) => p.id === v.part_id) ?? null}
          compareOpen={compareOpen}
          canCompare={others.length > 0}
          onToggleCompare={() => {
            setCompareOpen((o) => !o);
            setTab("overview");
          }}
          onRename={(name) => rename.mutate(name)}
          onDelete={() => setDeleteOpen(true)}
        />

        <div className="mb-3 text-sm text-muted">{v.prompt}</div>

        {v.error && (
          <div className="mb-3 flex items-start gap-3 rounded-lg border border-red-800 bg-red-950/50 p-2.5 text-sm text-red-300">
            <span className="min-w-0 flex-1 break-words">{v.error}</span>
            {lastJobId && (
              <Btn
                variant="ghost"
                size="sm"
                className="shrink-0 border border-red-800/60 text-red-300"
                busy={retry.isPending}
                onClick={() => retry.mutate(lastJobId)}
              >
                <RotateCcw className="h-3 w-3" /> {t("retry_job")}
              </Btn>
            )}
          </div>
        )}

        <VersionStatRow
          v={v}
          liveJob={liveJob}
          curJob={curJob}
          liveMax={stream.live.max_iterations}
          stepStartedTs={stream.live.step_started_ts}
        />

        <Tabs
          className="mb-4"
          active={tab}
          onChange={setTab}
          tabs={[
            { id: "overview", label: t("tab_model") },
            { id: "spec", label: t("tab_spec") },
            ...(v.code != null
              ? [{ id: "code" as Tab, label: t("tab_code") }]
              : []),
            {
              id: "quality",
              label: (
                <span className="flex items-center justify-center gap-1">
                  {t("tab_quality")}
                  {dfmCount > 0 && (
                    <span className="rounded-full bg-amber-600 px-1.5 text-[10px] text-white">
                      {dfmCount}
                    </span>
                  )}
                </span>
              ),
            },
            {
              id: "files",
              label: t("tab_files", { count: v.artifacts.length }),
            },
          ]}
        />

        {tab === "overview" && (
          <OverviewTab
            v={v}
            jobLive={jobLive}
            liveJob={liveJob}
            streamConnected={stream.connected}
            glbUrl={glb?.url ?? null}
            compareOpen={compareOpen}
            others={others}
            otherId={otherId}
            setOtherId={setOtherId}
            otherSel={otherSel}
            otherVersion={otherVersion}
            otherGlb={otherGlb}
            assetById={assetById}
            libById={libById}
            versions={versions.data ?? []}
            onRefresh={refresh}
          />
        )}

        {tab === "spec" && (
          <div className="flex flex-col gap-4">
            {v.spec ? (
              <SpecPanel spec={v.spec} />
            ) : (
              <EmptyState title={t("no_spec")} hint={t("no_spec_hint")} />
            )}
            {v.plan ? (
              <PlanPanel plan={v.plan} />
            ) : (
              <EmptyState title={t("no_plan")} />
            )}
          </div>
        )}

        {tab === "code" && <CodeView versionId={v.id} />}

        {tab === "quality" && (
          <QualityTab v={v} onStructuralStarted={structuralStarted} />
        )}

        {tab === "files" && (
          <div className="flex flex-col gap-4">
            <ArtifactsTable artifacts={v.artifacts} />
            {(v.status === "succeeded" || v.status === "needs_review") &&
              (v.code || v.artifacts.some((a) => a.kind === "step")) && (
                <SaveToLibrary versionId={v.id} hasCode={!!v.code} />
              )}
          </div>
        )}
      </div>

      {/* side column — inspector */}
      <Inspector
        v={v}
        jobId={jobId}
        stream={stream}
        jobs={jobs.data}
        versions={versions.data ?? []}
        assets={assets.data ?? []}
        library={library.data ?? []}
        onSelectJob={setJobId}
        onDone={refresh}
        onEditVisual={() => nav(`/versions/${v.id}/edit`)}
        onEditStructural={() => setOptimizeOpen(true)}
        onEditOrganic={() => setOrganicOpen(true)}
        edit={{
          editPrompt,
          setEditPrompt,
          editAssets,
          setEditAssets,
          editMode,
          setEditMode,
          editOptions,
          setEditOptions,
          editLibraryParts,
          setEditLibraryParts,
          regenMesh,
          setRegenMesh,
          submitting: edit.isPending,
          submitError: edit.error,
          onSubmit: () => edit.mutate(),
        }}
      />

      <ConfirmModal
        open={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        onConfirm={() => del.mutate()}
        busy={del.isPending}
        title={t("delete_title")}
        body={
          <>
            {t("delete_body_pre")}
            <b className="text-fg">v{v.number}</b>
            {t("delete_body_post")}
          </>
        }
      />

      <StructuralEditor
        key={v.id}
        open={optimizeOpen}
        onClose={() => setOptimizeOpen(false)}
        version={v}
        canTopopt={v.resolved_mode !== "topopt"}
        onJobStarted={structuralStarted}
      />

      <OrganicEditor
        key={`org-${v.id}`}
        open={organicOpen}
        onClose={() => setOrganicOpen(false)}
        version={v}
        onJobStarted={structuralStarted}
      />
    </div>
  );
}
