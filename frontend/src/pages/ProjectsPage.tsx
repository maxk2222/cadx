import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderPlus, Pencil, Search, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import ProjectFormModal from "../components/ProjectFormModal";
import StatusBadge from "../components/StatusBadge";
import {
  Btn,
  ConfirmModal,
  EmptyState,
  IconBtn,
  Spinner,
  inputCls,
  useDebounced,
} from "../components/ui";
import { api } from "../lib/api";
import { timeAgo } from "../lib/format";
import type { Project } from "../lib/types";

export default function ProjectsPage() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const [search, setSearch] = useState("");
  const q = useDebounced(search);
  const [createOpen, setCreateOpen] = useState(false);
  const [editing, setEditing] = useState<Project | null>(null);
  const [deleting, setDeleting] = useState<Project | null>(null);

  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });

  const del = useMutation({
    mutationFn: (id: string) => api.deleteProject(id),
    onSuccess: () => {
      setDeleting(null);
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
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
    <div className="mx-auto max-w-6xl p-6">
      <div className="mb-5 flex items-center gap-3">
        <h1 className="text-2xl font-bold">Projects</h1>
        <span className="text-sm text-zinc-500">{projects.data?.length ?? 0} total</span>
        <span className="flex-1" />
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-zinc-500" />
          <input
            className={`${inputCls} w-56 pl-8`}
            placeholder="search projects…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Btn onClick={() => setCreateOpen(true)}>
          <FolderPlus className="h-4 w-4" /> New project
        </Btn>
      </div>

      {projects.isLoading ? (
        <div className="flex justify-center py-16 text-zinc-500">
          <Spinner className="h-6 w-6" />
        </div>
      ) : projects.isError ? (
        <EmptyState title="Could not load projects" hint="Is the API running?" />
      ) : !filtered.length ? (
        <EmptyState
          title={q ? "No projects match your search" : "No projects yet"}
          hint={q ? undefined : "Create a project to start generating 3D models."}
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((p) => (
            <div
              key={p.id}
              className="group relative overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900 transition-colors hover:border-zinc-600"
            >
              <Link to={`/projects/${p.id}`} className="block">
                <div className="flex h-32 items-center justify-center overflow-hidden bg-zinc-950">
                  {p.latest_thumbnail_url ? (
                    <img
                      src={p.latest_thumbnail_url}
                      alt=""
                      className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                    />
                  ) : (
                    <span className="text-xs text-zinc-700">no render yet</span>
                  )}
                </div>
                <div className="p-3">
                  <div className="flex items-center gap-2">
                    <span className="min-w-0 flex-1 truncate font-medium">{p.name}</span>
                    {p.latest_version_status && (
                      <StatusBadge status={p.latest_version_status} />
                    )}
                  </div>
                  {p.description && (
                    <div className="mt-0.5 truncate text-sm text-zinc-500">{p.description}</div>
                  )}
                  <div className="mt-1.5 text-xs text-zinc-600">
                    {p.version_count} version{p.version_count === 1 ? "" : "s"} · updated{" "}
                    {timeAgo(p.updated_at)}
                  </div>
                </div>
              </Link>
              <div className="absolute right-2 top-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                <IconBtn
                  title="rename / edit"
                  className="bg-black/60 backdrop-blur"
                  onClick={(e) => {
                    e.preventDefault();
                    setEditing(p);
                  }}
                >
                  <Pencil className="h-3.5 w-3.5" />
                </IconBtn>
                <IconBtn
                  title="delete project"
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
        title="Delete project"
        body={
          <>
            Delete <b className="text-zinc-200">{deleting?.name}</b> and all of its versions,
            assets and files? This cannot be undone.
          </>
        }
      />
    </div>
  );
}

