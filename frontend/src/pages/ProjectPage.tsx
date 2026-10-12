import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Search, Sparkles, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
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
  SectionTitle,
  Spinner,
  inputCls,
  useDebounced,
  useToast,
} from "../components/ui";
import { api } from "../lib/api";
import type { GenMode, RunOptions } from "../lib/types";
import { useTitle } from "../lib/useTitle";

const GEN_MODES: GenMode[] = ["auto", "cad", "mesh", "hybrid", "topopt"];

export default function ProjectPage() {
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
  useTitle(project.data?.name ?? "Project");
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
      setError(String(e));
      toast(String(e), "error");
    },
  });

  const del = useMutation({
    mutationFn: () => api.deleteProject(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      toast("project deleted", "success");
      nav("/");
    },
    onError: (e) => toast(String(e), "error"),
  });

  const toggle = (aid: string) =>
    setSelected((s) => {
      const n = new Set(s);
      n.has(aid) ? n.delete(aid) : n.add(aid);
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
    return list.filter((v) => v.prompt.toLowerCase().includes(needle));
  }, [versions.data, vq]);

  if (project.isLoading)
    return (
      <div className="flex justify-center py-24 text-zinc-500">
        <Spinner className="h-6 w-6" />
      </div>
    );
  if (!project.data)
    return (
      <div className="mx-auto max-w-3xl p-6">
        <EmptyState title="Project not found" hint="It may have been deleted." />
      </div>
    );
  const p = project.data;

  return (
    <div className="mx-auto grid max-w-6xl grid-cols-1 gap-5 p-6 lg:grid-cols-3">
      <div className="flex flex-col gap-5 lg:col-span-2">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <Crumbs
              items={[
                <Link key="p" to="/" className="hover:text-zinc-300">
                  projects
                </Link>,
                <span key="n" className="font-medium text-zinc-200">
                  {p.name}
                </span>,
              ]}
            />
            <h1 className="mt-1 truncate text-2xl font-bold">{p.name}</h1>
            {p.description && (
              <div className="mt-0.5 text-sm text-zinc-500">{p.description}</div>
            )}
          </div>
          <div className="flex shrink-0 gap-1">
            <button
              className="flex items-center gap-1 rounded-md px-2 py-1 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              onClick={() => setEditOpen(true)}
            >
              <Pencil className="h-3.5 w-3.5" /> edit
            </button>
            <button
              className="flex items-center gap-1 rounded-md px-2 py-1 text-sm text-red-400 hover:bg-red-950/50"
              onClick={() => setDeleteOpen(true)}
            >
              <Trash2 className="h-3.5 w-3.5" /> delete
            </button>
          </div>
        </div>

        <Card className="p-4">
          <SectionTitle
            right={
              <span className="text-[11px] normal-case tracking-normal text-zinc-600">
                Ctrl+Enter to generate
              </span>
            }
          >
            Describe your part
          </SectionTitle>
          <textarea
            className={`${inputCls} h-32 resize-y`}
            placeholder={
              "Describe the part: shape, dimensions (X length × Y width × Z height), holes, threads, tolerances, material, process (FDM/CNC)…\n\nExample: Mounting plate 80×60×8mm, four M4 clearance holes at 10mm from corners, countersunk, PLA for FDM."
            }
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
              <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                Library parts{selectedLib.size ? ` — ${selectedLib.size} selected` : ""}
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
                          : "border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
                      }`}
                      onClick={() =>
                        setSelectedLib((s) => {
                          const n = new Set(s);
                          n.has(p.id) ? n.delete(p.id) : n.add(p.id);
                          return n;
                        })
                      }
                    >
                      {p.name}
                      <span className="ml-1 text-[10px] text-zinc-600">{p.kind}</span>
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
            Generate model{selected.size ? ` · ${selected.size} asset${selected.size === 1 ? "" : "s"}` : ""}
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
                  clear selection ({selected.size})
                </button>
              )
            }
          >
            Reference assets{selected.size ? ` — ${selected.size} selected` : ""}
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
          />
        </Card>
      </div>

      <Card className="h-fit p-3">
        <SectionTitle
          right={<span className="text-[11px] normal-case tracking-normal">{versions.data?.length ?? 0}</span>}
        >
          Versions
        </SectionTitle>
        {(versions.data?.length ?? 0) > 4 && (
          <div className="relative mb-2">
            <Search className="pointer-events-none absolute left-2 top-2 h-3.5 w-3.5 text-zinc-500" />
            <input
              className={`${inputCls} py-1.5 pl-7 text-xs`}
              placeholder="filter prompts…"
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
        title="Delete project"
        body={
          <>
            Delete <b className="text-zinc-200">{p.name}</b> and all of its versions, assets
            and files? This cannot be undone.
          </>
        }
      />
    </div>
  );
}
