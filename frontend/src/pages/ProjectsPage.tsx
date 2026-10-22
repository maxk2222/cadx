import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleDot, FolderPlus, Pencil, Search, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useNavigate } from "react-router-dom";
import Logo from "../components/Logo";
import ProjectFormModal from "../components/ProjectFormModal";
import StatusBadge from "../components/StatusBadge";
import {
  Btn,
  ConfirmModal,
  EmptyState,
  IconBtn,
  PageShell,
  Skeleton,
  inputCls,
  useDebounced,
  useToast,
} from "../components/ui";
import { api, errMsg } from "../lib/api";
import { timeAgo } from "../lib/format";
import type { Project } from "../lib/types";
import { useTitle } from "../lib/useTitle";

/** Card thumbnail: latest render, overlaid with the live preview of the
 *  running version's newest iteration (if any). `tick` cache-busts the live
 *  image on each projects refetch. */
function ProjectThumb({ p, tick }: { p: Project; tick: number }) {
  const { t } = useTranslation("projects");
  const [ok, setOk] = useState(false);
  // thumbnail_url may point at the scan-based preview route which 404s when
  // the version truly has no image files — fall back to the placeholder
  const [badUrl, setBadUrl] = useState<string | null>(null);
  const live = !!p.running_version_id;
  return (
    <div className="relative flex h-32 items-center justify-center overflow-hidden bg-bg">
      {p.latest_thumbnail_url && badUrl !== p.latest_thumbnail_url ? (
        <img
          src={p.latest_thumbnail_url}
          alt=""
          className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
          onError={() => setBadUrl(p.latest_thumbnail_url)}
        />
      ) : (
        <span className="text-xs text-faint/60">{t("no_render")}</span>
      )}
      {live && (
        <>
          <img
            src={`${api.previewUrl(p.running_version_id!)}?t=${tick}`}
            alt=""
            className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-300 ${
              ok ? "opacity-100" : "opacity-0"
            }`}
            onLoad={() => setOk(true)}
            onError={() => setOk(false)}
          />
          <span className="absolute left-2 top-2 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-brand-300">
            <CircleDot className="h-3 w-3 animate-pulse" /> {t("live")}
          </span>
        </>
      )}
    </div>
  );
}

export default function ProjectsPage() {
  const { t } = useTranslation("projects");
  useTitle(t("title"));
  const qc = useQueryClient();
  const nav = useNavigate();
  const toast = useToast();
  const [search, setSearch] = useState("");
  const q = useDebounced(search);
  const [createOpen, setCreateOpen] = useState(false);
  const [editing, setEditing] = useState<Project | null>(null);
  const [deleting, setDeleting] = useState<Project | null>(null);

  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
    refetchInterval: (q) =>
      (q.state.data ?? []).some((p) => p.running_version_id) ? 4000 : false,
  });

  const del = useMutation({
    mutationFn: (id: string) => api.deleteProject(id),
    onSuccess: () => {
      setDeleting(null);
      qc.invalidateQueries({ queryKey: ["projects"] });
      toast(t("deleted_toast"), "success");
    },
    onError: (e) => toast(errMsg(e), "error"),
  });

  const filtered = useMemo(() => {
    const list = projects.data ?? [];
    const needle = q.trim().toLowerCase();
    if (!needle) return list;
    return list.filter(
      (p) =>
        p.name.toLowerCase().includes(needle) ||
        p.description.toLowerCase().includes(needle)
    );
  }, [projects.data, q]);

  return (
    <PageShell
      title={t("title")}
      subtitle={t("count_sub", { count: projects.data?.length ?? 0 })}
      actions={
        <>
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-faint" />
            <input
              className={`${inputCls} w-56 pl-8`}
              placeholder={t("search_ph")}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <Btn onClick={() => setCreateOpen(true)}>
            <FolderPlus className="h-4 w-4" /> {t("new_project")}
          </Btn>
        </>
      }
    >
      {projects.isLoading ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="overflow-hidden rounded-lg border border-edge">
              <Skeleton className="h-32 rounded-none" />
              <div className="space-y-2 p-3">
                <Skeleton className="h-4 w-2/3" />
                <Skeleton className="h-3 w-1/3" />
              </div>
            </div>
          ))}
        </div>
      ) : projects.isError ? (
        <EmptyState title={t("load_error")} hint={t("load_error_hint")} />
      ) : !filtered.length ? (
        <EmptyState
          icon={<Logo size={80} />}
          title={q ? t("empty_search") : t("empty")}
          hint={q ? undefined : t("empty_hint")}
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((p) => (
            <div
              key={p.id}
              className="panel panel-hover anim-fade-in-up group relative overflow-hidden"
            >
              <Link to={`/projects/${p.id}`} className="block">
                <ProjectThumb p={p} tick={projects.dataUpdatedAt} />
                <div className="p-3">
                  <div className="flex items-center gap-2">
                    <span className="min-w-0 flex-1 truncate font-medium">{p.name}</span>
                    {p.latest_version_status && (
                      <StatusBadge status={p.latest_version_status} />
                    )}
                  </div>
                  {p.description && (
                    <div className="mt-0.5 truncate text-sm text-faint">{p.description}</div>
                  )}
                  <div className="mt-1.5 text-xs text-faint/70">
                    {t("versions_count", { count: p.version_count })}
                    {p.parts_count > 1 && ` · ${t("parts_count", { count: p.parts_count })}`}
                    {" · "}
                    {t("updated", { at: timeAgo(p.updated_at) })}
                  </div>
                </div>
              </Link>
              <div className="absolute right-2 top-2 flex gap-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100">
                <IconBtn
                  title={t("rename")}
                  className="bg-black/60 backdrop-blur"
                  onClick={(e) => {
                    e.preventDefault();
                    setEditing(p);
                  }}
                >
                  <Pencil className="h-3.5 w-3.5" />
                </IconBtn>
                <IconBtn
                  title={t("delete_title")}
                  danger
                  className="bg-black/60 backdrop-blur"
                  onClick={(e) => {
                    e.preventDefault();
                    setDeleting(p);
                  }}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </IconBtn>
              </div>
            </div>
          ))}
        </div>
      )}

      {createOpen && (
        <ProjectFormModal
          onClose={() => setCreateOpen(false)}
          onCreated={(p) => nav(`/projects/${p.id}`)}
        />
      )}
      {editing && (
        <ProjectFormModal
          key={editing.id}
          project={editing}
          onClose={() => setEditing(null)}
        />
      )}
      <ConfirmModal
        open={!!deleting}
        onClose={() => setDeleting(null)}
        onConfirm={() => deleting && del.mutate(deleting.id)}
        busy={del.isPending}
        title={t("delete_title")}
        body={
          <>
            {t("delete_body_pre")}
            <b className="text-fg">{deleting?.name}</b>
            {t("delete_body_post")}
          </>
        }
      />
    </PageShell>
  );
}
