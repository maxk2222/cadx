import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleDot, MessagesSquare, Pencil, Search, Sparkles, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useNavigate, useParams } from "react-router-dom";
import AssetManager from "../components/AssetManager";
import ModeSelector from "../components/ModeSelector";
import OptionsPanel from "../components/OptionsPanel";
import ProjectFormModal from "../components/ProjectFormModal";
import VersionTree from "../components/VersionTree";
import {
  Btn,
  Card,
  ConfirmModal,
  Crumbs,
  EmptyState,
  PageShell,
  SectionTitle,
  Skeleton,
  inputCls,
  useDebounced,
  useToast,
} from "../components/ui";
import { api, errMsg } from "../lib/api";
import type { GenMode, RunOptions, VersionSummary } from "../lib/types";
import { useTitle } from "../lib/useTitle";

const GEN_MODES: GenMode[] = ["auto", "cad", "mesh", "hybrid", "topopt"];

/** Version shown in the project hero render: the highest-scoring
 *  succeeded/needs_review version (newest wins ties), else simply the
 *  newest version so a first still-running/failed version still shows up. */
export function pickShowcaseVersion(
  versions: VersionSummary[]
): VersionSummary | null {
  if (!versions.length) return null;
  const done = versions.filter(
    (v) => v.status === "succeeded" || v.status === "needs_review"
  );
  const pool = done.length ? done : versions;
  return pool.reduce((a, b) => {
    const sa = a.score ?? -1;
    const sb = b.score ?? -1;
    return sb > sa || (sb === sa && b.number > a.number) ? b : a;
  });
}

/** Hero card on the project page: render of the best version, overlaid with
 *  the running version's live preview while a job is active (same pattern as
 *  the project-card thumbnail). */
function ProjectShowcase({
  versions,
  tick,
}: {
  versions: VersionSummary[];
  tick: number;
}) {
  const { t } = useTranslation("projects");
  const [bad, setBad] = useState<string | null>(null);
  const [liveOk, setLiveOk] = useState(false);
  const showcase = pickShowcaseVersion(versions);
  const running = versions.find(
    (v) =>
      !!v.active_job ||
      ["queued", "running", "waiting_for_input"].includes(v.status)
  );
  const shown = running ?? showcase;
  if (!shown) return null;
  const baseSrc = showcase ? `${api.previewUrl(showcase.id)}?t=${tick}` : "";
  return (
    <Card className="overflow-hidden p-0">
      <Link
        to={
          running?.active_job
            ? `/versions/${running.id}?job=${running.active_job.id}`
            : `/versions/${shown.id}`
        }
        title={`v${shown.number}${shown.name ? ` · ${shown.name}` : ""}`}
        className="group relative block h-56 bg-bg"
      >
        {baseSrc && bad !== baseSrc ? (
          <img
            key={baseSrc}
            src={baseSrc}
            alt=""
            className="h-full w-full object-contain"
            onError={() => setBad(baseSrc)}
          />
        ) : (
          <span className="flex h-full items-center justify-center text-xs text-faint/60">
            {t("no_render")}
          </span>
        )}
        {running && (
          <>
            <img
              src={`${api.previewUrl(running.id)}?t=${tick}`}
              alt=""
              className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-300 ${
                liveOk ? "opacity-100" : "opacity-0"
              }`}
              onLoad={() => setLiveOk(true)}
              onError={() => setLiveOk(false)}
            />
            <span className="absolute left-2 top-2 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-brand-300">
              <CircleDot className="h-3 w-3 animate-pulse" /> {t("live")}
            </span>
          </>
        )}
        <span className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-black/80 to-transparent px-3 pb-1.5 pt-6 text-xs text-fg">
          <span className="font-medium">v{shown.number}</span>
          {shown.name && <span className="truncate">{shown.name}</span>}
          <span className="ml-auto text-faint">
            {running
              ? t("showcase_live", { status: running.status })
              : shown.score != null
                ? t("showcase_scored", { status: shown.status, score: shown.score })
                : shown.status}
          </span>
        </span>
      </Link>
    </Card>
  );
}

export default function ProjectPage() {
  const { t } = useTranslation("projects");
  const { id = "" } = useParams<{ id: string }>();
  const nav = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const [prompt, setPrompt] = useState(
    () => localStorage.getItem(`draft:${id}`) ?? ""
  );
  const [mode, setMode] = useState<GenMode>("auto");
  const [options, setOptions] = useState<RunOptions>({});
  const [modeInit, setModeInit] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [selectedLib, setSelectedLib] = useState<Set<string>>(new Set());
  const [vFilter, setVFilter] = useState("");
  const vq = useDebounced(vFilter);
  const [editOpen, setEditOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setPrompt(localStorage.getItem(`draft:${id}`) ?? "");
    setMode("auto");
    setOptions({});
    setModeInit(false);
    setSelected(new Set());
    setSelectedLib(new Set());
    setError("");
  }, [id]);

  const project = useQuery({
    queryKey: ["project", id],
    queryFn: () => api.getProject(id),
  });
  useTitle(project.data?.name ?? t("title_fallback"));
  const assets = useQuery({
    queryKey: ["assets", id],
    queryFn: () => api.listAssets(id),
  });
  const library = useQuery({
    queryKey: ["library"],
    queryFn: api.listLibrary,
    staleTime: 30_000,
  });
  const versions = useQuery({
    queryKey: ["versions", id],
    queryFn: () => api.listVersions(id),
    refetchInterval: (q) =>
      (q.state.data ?? []).some((v) => ["queued", "running"].includes(v.status))
        ? 4000
        : false,
  });

  // adopt project default gen mode once loaded
  useEffect(() => {
    if (modeInit || !project.data) return;
    const gm = (project.data.settings as Record<string, unknown>)?.gen_mode;
    if (typeof gm === "string" && GEN_MODES.includes(gm as GenMode)) {
      setMode(gm as GenMode);
    }
    setModeInit(true);
  }, [project.data, modeInit]);

  const generate = useMutation({
    mutationFn: () =>
      api.generate(id, prompt, [...selected], mode, options, [...selectedLib]),
    onSuccess: (r) => {
      localStorage.removeItem(`draft:${id}`);
      nav(`/versions/${r.version_id}?job=${r.job_id}`);
    },
    onError: (e) => {
      setError(errMsg(e));
      toast(errMsg(e), "error");
    },
  });

  const openChat = useMutation({
    mutationFn: async () => {
      const list = await api.listChats(id);
      return list.length ? list[0] : api.createChat(id);
    },
    onSuccess: (c) => nav(`/chat/${c.id}`),
    onError: (e) => toast(errMsg(e), "error"),
  });

  const del = useMutation({
    mutationFn: () => api.deleteProject(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      toast(t("deleted_toast"), "success");
      nav("/projects");
    },
    onError: (e) => toast(errMsg(e), "error"),
  });

  const toggle = (aid: string) =>
    setSelected((s) => {
      const n = new Set(s);
      if (n.has(aid)) n.delete(aid);
      else n.add(aid);
      return n;
    });

  const onPromptChange = (v: string) => {
    setPrompt(v);
    localStorage.setItem(`draft:${id}`, v);
  };

  const filteredVersions = useMemo(() => {
    const list = versions.data ?? [];
    const needle = vq.trim().toLowerCase();
    if (!needle) return list;
    return list.filter(
      (v) =>
        v.prompt.toLowerCase().includes(needle) ||
        v.name.toLowerCase().includes(needle)
    );
  }, [versions.data, vq]);

  if (project.isLoading)
    return (
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-5 p-6 lg:grid-cols-3">
        <div className="flex flex-col gap-5 lg:col-span-2">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-72" />
          <Skeleton className="h-40" />
        </div>
        <Skeleton className="h-72" />
      </div>
    );
  if (!project.data)
    return (
      <div className="mx-auto max-w-3xl p-6">
        <EmptyState title={t("not_found")} hint={t("not_found_hint")} />
      </div>
    );
  const p = project.data;

  return (
    <PageShell
      title={p.name}
      subtitle={p.description}
      crumbs={
        <Crumbs
          items={[
            <Link key="p" to="/projects" className="hover:text-muted">
              {t("crumb_projects")}
            </Link>,
            <span key="n" className="font-medium text-fg">
              {p.name}
            </span>,
          ]}
        />
      }
      actions={
        <>
          <Btn
            variant="ghost"
            size="sm"
            onClick={() => openChat.mutate()}
            disabled={openChat.isPending}
          >
            <MessagesSquare className="h-3.5 w-3.5" /> {t("chat")}
          </Btn>
          <Btn variant="ghost" size="sm" onClick={() => setEditOpen(true)}>
            <Pencil className="h-3.5 w-3.5" /> {t("common:action.edit")}
          </Btn>
          <Btn variant="danger" size="sm" onClick={() => setDeleteOpen(true)}>
            <Trash2 className="h-3.5 w-3.5" /> {t("common:action.delete")}
          </Btn>
        </>
      }
    >
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
      <div className="flex flex-col gap-5 lg:col-span-2">

        <ProjectShowcase
          versions={versions.data ?? []}
          tick={versions.dataUpdatedAt}
        />

        <Card className="p-4">
          <SectionTitle
            right={
              <span className="text-[11px] normal-case tracking-normal text-faint">
                {t("ctrl_enter_gen")}
              </span>
            }
          >
            {t("describe_title")}
          </SectionTitle>
          <textarea
            className={`${inputCls} h-32 resize-y`}
            placeholder={t("describe_ph")}
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                if (prompt.trim()) generate.mutate();
              }
            }}
          />
          {error && <div className="mt-2 text-sm text-red-400">{error}</div>}
          <div className="mt-3">
            <ModeSelector value={mode} onChange={setMode} />
          </div>
          <div className="mt-2">
            <OptionsPanel value={options} onChange={setOptions} />
          </div>
          {(library.data ?? []).length > 0 && (
            <div className="mt-3">
              <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-faint">
                {t("library_parts")}{selectedLib.size ? ` — ${t("selected_n", { count: selectedLib.size })}` : ""}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {(library.data ?? []).map((p) => {
                  const on = selectedLib.has(p.id);
                  return (
                    <button
                      key={p.id}
                      type="button"
                      title={`${p.kind}: ${p.caption || p.filename || p.name}`}
                      className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                        on
                          ? "border-brand-600 bg-brand-950/60 text-brand-200"
                          : "border-edge2 bg-panel text-muted hover:border-faint hover:text-fg"
                      }`}
                      onClick={() =>
                        setSelectedLib((s) => {
                          const n = new Set(s);
                          if (n.has(p.id)) n.delete(p.id);
                          else n.add(p.id);
                          return n;
                        })
                      }
                    >
                      {p.name}
                      <span className="ml-1 text-[10px] text-faint">{p.kind}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
          <Btn
            className="mt-3 w-full py-2"
            onClick={() => generate.mutate()}
            busy={generate.isPending}
            disabled={!prompt.trim()}
          >
            <Sparkles className="h-4 w-4" />
            {t("generate")}{selected.size ? ` · ${t("assets_n", { count: selected.size })}` : ""}
          </Btn>
        </Card>

        <Card className="p-4">
          <SectionTitle
            right={
              selected.size > 0 && (
                <button
                  className="text-[11px] normal-case tracking-normal text-brand-400 hover:text-brand-300"
                  onClick={() => setSelected(new Set())}
                >
                  {t("clear_sel", { count: selected.size })}
                </button>
              )
            }
          >
            {t("ref_assets")}{selected.size ? ` — ${t("selected_n", { count: selected.size })}` : ""}
          </SectionTitle>
          <AssetManager
            projectId={id}
            assets={assets.data ?? []}
            selected={selected}
            onToggle={toggle}
            onAssetReplaced={(srcId, newId) =>
              setSelected((s) => {
                if (!s.has(srcId)) return s;
                const n = new Set(s);
                n.delete(srcId);
                n.add(newId);
                return n;
              })
            }
            onDeleted={(aid) =>
              setSelected((s) => {
                if (!s.has(aid)) return s;
                const n = new Set(s);
                n.delete(aid);
                return n;
              })
            }
          />
        </Card>
      </div>

      <Card className="h-fit p-3">
        <SectionTitle
          right={<span className="text-[11px] normal-case tracking-normal">{versions.data?.length ?? 0}</span>}
        >
          {t("versions_title")}
        </SectionTitle>
        {(versions.data?.length ?? 0) > 4 && (
          <div className="relative mb-2">
            <Search className="pointer-events-none absolute left-2 top-2 h-3.5 w-3.5 text-faint" />
            <input
              className={`${inputCls} py-1.5 pl-7 text-xs`}
              placeholder={t("filter_prompts_ph")}
              value={vFilter}
              onChange={(e) => setVFilter(e.target.value)}
            />
          </div>
        )}
        <VersionTree versions={filteredVersions} />
      </Card>

      {editOpen && <ProjectFormModal project={p} onClose={() => setEditOpen(false)} />}
      <ConfirmModal
        open={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        onConfirm={() => del.mutate()}
        busy={del.isPending}
        title={t("delete_title")}
        body={
          <>
            {t("delete_body_pre")}<b className="text-fg">{p.name}</b>{t("delete_body_post")}
          </>
        }
      />
    </div>
    </PageShell>
  );
}
