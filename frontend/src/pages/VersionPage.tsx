import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Clock, Cpu, Gauge, Layers, Plus, Shrink, Trash2, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import ArtifactsTable from "../components/ArtifactsTable";
import CodeView from "../components/CodeView";
import DfmPanel from "../components/DfmPanel";
import ExportPanel from "../components/ExportPanel";
import FemPanel from "../components/FemPanel";
import JobLog from "../components/JobLog";
import ModeSelector from "../components/ModeSelector";
import OptimizeDialog from "../components/OptimizeDialog";
import OptionsPanel, { OptionChips } from "../components/OptionsPanel";
import ParamsPanel from "../components/ParamsPanel";
import PlanPanel from "../components/PlanPanel";
import RenderGallery from "../components/RenderGallery";
import SpecPanel from "../components/SpecPanel";
import StatusBadge from "../components/StatusBadge";
import VersionTree from "../components/VersionTree";
import Viewer3D from "../components/Viewer3D";
import {
  Btn,
  Card,
  Chip,
  ConfirmModal,
  Crumbs,
  EmptyState,
  SectionTitle,
  Spinner,
  Tabs,
  inputCls,
} from "../components/ui";
import { api } from "../lib/api";
import { fmtDuration, fmtTokens } from "../lib/format";
import type { GenMode, Job, RunOptions } from "../lib/types";

type Tab = "overview" | "spec" | "code" | "quality" | "files";

function SaveToLibrary({
  versionId,
  hasCode,
}: {
  versionId: string;
  hasCode: boolean;
}) {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [asTemplate, setAsTemplate] = useState(false);
  const [done, setDone] = useState(false);
  const save = useMutation({
    mutationFn: () => api.saveToLibrary(versionId, name.trim(), asTemplate),
    onSuccess: () => {
      setDone(true);
      setName("");
      qc.invalidateQueries({ queryKey: ["library"] });
    },
  });
  return (
    <Card className="p-4">
      <SectionTitle>Save to part library</SectionTitle>
      <div className="flex flex-wrap items-center gap-2">
        <input
          className={inputCls + " w-48 text-xs"}
          placeholder="part name…"
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setDone(false);
          }}
        />
        {hasCode && (
          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            <input
              type="checkbox"
              checked={asTemplate}
              onChange={(e) => setAsTemplate(e.target.checked)}
            />
            as code template (parametric)
          </label>
        )}
        <Btn
          variant="ghost"
          disabled={!name.trim()}
          busy={save.isPending}
          onClick={() => save.mutate()}
        >
          save {asTemplate ? "template" : "STEP body"}
        </Btn>
        {done && <span className="text-xs text-emerald-400">saved ✓</span>}
        {save.isError && (
          <span className="text-xs text-red-400">{String(save.error)}</span>
        )}
      </div>
      <div className="mt-1.5 text-[11px] text-zinc-600">
        Saved parts can be attached to new generations via the library picker on
        the project page; bodies are available to generated code through
        load_reference().
      </div>
    </Card>
  );
}

export default function VersionPage() {
  const { id = "" } = useParams<{ id: string }>();
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const qc = useQueryClient();
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
  const fileRef = useRef<HTMLInputElement>(null);

  const version = useQuery({
    queryKey: ["version", id],
    queryFn: () => api.getVersion(id),
    refetchInterval: (q) =>
      ["queued", "running"].includes(q.state.data?.status ?? "") ? 4000 : false,
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
      (q.state.data ?? []).some((x) => ["queued", "running"].includes(x.status))
        ? 4000
        : false,
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

  // pick latest job once loaded (jobs are sorted newest-first)
  useEffect(() => {
    if (!jobId && jobs.data?.length) setJobId(jobs.data[0].id);
  }, [jobs.data, jobId]);

  // initialize edit options from the version once per version id
  useEffect(() => {
    if (v && v.id !== optFor) {
      setEditPrompt("");
      setEditAssets(new Set());
      setEditMode("auto");
      setRegenMesh(false);
      setEditOptions(v.options ?? {});
      setEditLibraryParts(new Set(v.library_part_ids ?? []));
      setOptFor(v.id);
    }
  }, [v, optFor]);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["version", id] });
    qc.invalidateQueries({ queryKey: ["jobs", id] });
    qc.invalidateQueries({ queryKey: ["versions", v?.project_id] });
  };

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

  const optimize = useMutation({
    mutationFn: (body: Parameters<typeof api.optimize>[1]) => api.optimize(id, body),
    onSuccess: (r) => {
      setOptimizeOpen(false);
      nav(`/versions/${r.version_id}?job=${r.job_id}`);
    },
  });

  const del = useMutation({
    mutationFn: () => api.deleteVersion(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["versions", v?.project_id] });
      nav(`/projects/${v?.project_id}`);
    },
  });

  const uploadEditAssets = async (files: FileList | null) => {
    if (!files?.length || !v) return;
    const uploaded = await api.uploadAssets(v.project_id, Array.from(files));
    qc.invalidateQueries({ queryKey: ["assets", v.project_id] });
    setEditAssets((s) => new Set([...s, ...uploaded.map((u) => u.id)]));
  };

  if (version.isLoading)
    return (
      <div className="flex justify-center py-24 text-zinc-500">
        <Spinner className="h-6 w-6" />
      </div>
    );
  if (!v)
    return (
      <div className="mx-auto max-w-3xl p-6">
        <EmptyState title="Version not found" hint="It may have been deleted." />
      </div>
    );

  const active = ["queued", "running"].includes(v.status);
  const glb = v.artifacts.find((a) => a.kind === "glb");
  const a = v.analysis as {
    bbox_size?: number[];
    volume?: number;
    is_valid?: boolean;
    watertight?: boolean;
    triangle_count?: number;
  } | null;
  const curJob: Job | undefined = (jobs.data ?? []).find((j) => j.id === jobId);
  const dfmCount = v.dfm?.warnings?.length ?? 0;
  const usageEntries = Object.entries(v.usage ?? {}).filter(
    ([, n]) => typeof n === "number" && n > 0
  );

  return (
    <div className="mx-auto grid max-w-7xl grid-cols-1 gap-5 p-6 xl:grid-cols-4">
      {/* main column */}
      <div className="xl:col-span-3">
        <div className="mb-2">
          <Crumbs
            items={[
              <Link key="p" to="/" className="hover:text-zinc-300">
                projects
              </Link>,
              <Link
                key="proj"
                to={`/projects/${v.project_id}`}
                className="hover:text-zinc-300"
              >
                {project.data?.name ?? "project"}
              </Link>,
              <span key="v" className="font-medium text-zinc-200">
                v{v.number}
              </span>,
            ]}
          />
        </div>

        <div className="mb-2 flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-bold">v{v.number}</h1>
          <StatusBadge status={v.status} />
          <Chip tone="zinc" className="uppercase">{v.kind}</Chip>
          {(v.gen_mode || v.resolved_mode) && (
            <Chip
              tone={
                v.resolved_mode === "mesh"
                  ? "purple"
                  : v.resolved_mode === "hybrid"
                    ? "teal"
                    : "zinc"
              }
              title="requested → resolved generation mode"
            >
              {v.gen_mode}
              {v.resolved_mode && v.resolved_mode !== v.gen_mode
                ? ` → ${v.resolved_mode}`
                : ""}
            </Chip>
          )}
          <OptionChips options={v.options} />
          {v.score != null && (
            <Chip tone={v.score >= 80 ? "green" : v.score >= 50 ? "amber" : "red"}>
              <Gauge className="h-3 w-3" /> score {v.score}/100
            </Chip>
          )}
          <span className="flex-1" />
          <button
            className="flex items-center gap-1 rounded-md px-2 py-1 text-sm text-red-400 hover:bg-red-950/50"
            onClick={() => setDeleteOpen(true)}
          >
            <Trash2 className="h-3.5 w-3.5" /> delete
          </button>
        </div>

        <div className="mb-3 text-sm text-zinc-400">{v.prompt}</div>
        {v.error && (
          <div className="mb-3 rounded-lg border border-red-800 bg-red-950/50 p-2.5 text-sm text-red-300">
            {v.error}
          </div>
        )}

        {/* metrics strip */}
        <div className="mb-4 flex flex-wrap gap-1.5 text-xs">
          <Chip title="agent iterations">
            <Layers className="h-3 w-3" /> {v.iterations} iteration
            {v.iterations === 1 ? "" : "s"}
          </Chip>
          {curJob?.started_at && (
            <Chip title={`started ${curJob.started_at}`}>
              <Clock className="h-3 w-3" /> {fmtDuration(curJob.started_at, curJob.finished_at)}
            </Chip>
          )}
          {usageEntries.map(([k, n]) => (
            <Chip key={k} title={`LLM ${k}`}>
              <Cpu className="h-3 w-3" /> {k.replaceAll("_", " ")}: {fmtTokens(n as number)}
            </Chip>
          ))}
          {dfmCount > 0 && (
            <Chip tone="amber" title="manufacturability warnings">
              <TriangleAlert className="h-3 w-3" /> {dfmCount} DfM
            </Chip>
          )}
          {v.fem && (
            <Chip
              tone={
                Math.min(
                  ...v.fem.cases.map((c) => c.safety_factor ?? Infinity)
                ) < 1
                  ? "red"
                  : "purple"
              }
              title="FEM safety factor"
            >
              <Activity className="h-3 w-3" /> SF{" "}
              {Math.min(...v.fem.cases.map((c) => c.safety_factor ?? Infinity)).toFixed(2)}
            </Chip>
          )}
        </div>

        <Tabs
          className="mb-4"
          active={tab}
          onChange={setTab}
          tabs={[
            { id: "overview", label: "Model" },
            { id: "spec", label: "Spec & Plan" },
            ...(v.code != null ? [{ id: "code" as Tab, label: "Code" }] : []),
            {
              id: "quality",
              label: (
                <span className="flex items-center justify-center gap-1">
                  Quality
                  {dfmCount > 0 && (
                    <span className="rounded-full bg-amber-600 px-1.5 text-[10px] text-white">
                      {dfmCount}
                    </span>
                  )}
                </span>
              ),
            },
            { id: "files", label: `Files (${v.artifacts.length})` },
          ]}
        />

        {tab === "overview" && (
          <>
            <div className="mb-4 h-[460px] overflow-hidden rounded-lg border border-zinc-800">
              <Viewer3D url={glb ? glb.url : null} />
            </div>

            {v.mesh && (
              <Card className="mb-4 flex flex-wrap gap-x-4 gap-y-1 p-3 text-sm">
                <span className="text-zinc-500">mesh:</span>
                {v.mesh.generation?.model && (
                  <span title={v.mesh.generation.backend ?? ""}>
                    model: <b>{v.mesh.generation.model}</b>
                  </span>
                )}
                {v.mesh.generation?.seed != null && <span>seed: {v.mesh.generation.seed}</span>}
                {(v.mesh.triangles ?? v.mesh.generation?.triangles) != null && (
                  <span>triangles: {v.mesh.triangles ?? v.mesh.generation?.triangles}</span>
                )}
                {v.mesh.orientation && (
                  <span title={v.mesh.orientation_source ?? ""}>
                    orientation: {v.mesh.orientation.up ?? "?"} /{" "}
                    {v.mesh.orientation.front ?? "?"}
                  </span>
                )}
                {v.mesh.scale_factor != null && <span>scale ×{v.mesh.scale_factor}</span>}
                {v.mesh.generation?.duration_s != null && (
                  <span>gen: {v.mesh.generation.duration_s.toFixed(0)}s</span>
                )}
                {v.mesh.step_tessellated && (
                  <span
                    className="text-amber-400"
                    title="STEP export is tessellated (mesh-derived), not an analytic B-Rep"
                  >
                    STEP is tessellated
                  </span>
                )}
              </Card>
            )}

            {a && (
              <Card className="mb-4 flex flex-wrap gap-4 p-3 text-sm">
                {a.bbox_size && (
                  <span>
                    size: <b>{a.bbox_size.map((n) => n.toFixed(1)).join(" × ")}</b> mm
                  </span>
                )}
                {a.volume != null && <span>volume: {(a.volume / 1000).toFixed(1)} cm³</span>}
                {a.triangle_count != null && <span>triangles: {a.triangle_count}</span>}
                <span>valid: {String(a.is_valid)}</span>
                <span>watertight: {String(a.watertight)}</span>
              </Card>
            )}

            <RenderGallery artifacts={v.artifacts} />

            <div className="mt-4">
              <ExportPanel versionId={v.id} artifacts={v.artifacts} onRefresh={refresh} />
            </div>
          </>
        )}

        {tab === "spec" && (
          <div className="flex flex-col gap-4">
            {v.spec ? (
              <SpecPanel spec={v.spec} />
            ) : (
              <EmptyState title="No spec extracted" hint="The intake step did not run or produced no spec." />
            )}
            {v.plan ? (
              <PlanPanel plan={v.plan} />
            ) : (
              <EmptyState title="No plan available" />
            )}
          </div>
        )}

        {tab === "code" && <CodeView versionId={v.id} />}

        {tab === "quality" && (
          <div className="flex flex-col gap-4">
            {v.fem && <FemPanel fem={v.fem} artifacts={v.artifacts} />}
            <Card className="p-4">
              <SectionTitle>Manufacturability (DfM)</SectionTitle>
              {v.dfm ? (
                <DfmPanel warnings={v.dfm.warnings ?? []} />
              ) : (
                <div className="text-sm text-zinc-500">No DfM report for this version.</div>
              )}
            </Card>
            {!!v.bom?.length && (
              <Card className="p-4">
                <SectionTitle>Bill of materials (standard parts used)</SectionTitle>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-left text-zinc-500">
                      <th className="pb-1 pr-2">count</th>
                      <th className="pb-1 pr-2">class</th>
                      <th className="pb-1 pr-2">type</th>
                      <th className="pb-1">size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {v.bom.map((e, i) => (
                      <tr key={i} className="border-t border-zinc-800/60 text-zinc-300">
                        <td className="py-1 pr-2 font-mono">{e.count}×</td>
                        <td className="py-1 pr-2 font-mono">{e.part_class}</td>
                        <td className="py-1 pr-2 font-mono text-zinc-500">{e.type ?? "—"}</td>
                        <td className="py-1 font-mono text-zinc-500">{e.size ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            )}
            {!!v.issues?.length && (
              <Card className="p-4">
                <SectionTitle>Detected issues</SectionTitle>
                {v.issues.map((it, i) => (
                  <div key={i} className="mb-1 text-xs text-zinc-400">
                    <span
                      className={
                        it.severity === "major" ? "text-red-400" : "text-amber-400"
                      }
                    >
                      [{it.severity}]
                    </span>{" "}
                    <span className="text-zinc-500">({it.source})</span> {it.what}
                    {it.where ? ` @ ${it.where}` : ""}
                    {it.fix_hint ? (
                      <span className="text-zinc-600"> — fix: {it.fix_hint}</span>
                    ) : null}
                  </div>
                ))}
              </Card>
            )}
            {v.critique && (
              <Card className="p-4">
                <SectionTitle
                  right={
                    <Chip tone={v.critique.passed ? "green" : "amber"}>
                      score {v.critique.score}
                      {v.critique.passed ? " · passed" : " · not passed"}
                    </Chip>
                  }
                >
                  Critique
                </SectionTitle>
                {v.critique.discrepancies?.length > 0 && (
                  <div className="mb-2">
                    {v.critique.discrepancies.map((d, i) => (
                      <div key={i} className="mb-1 text-xs text-zinc-400">
                        <span
                          className={
                            d.severity === "major" ? "text-red-400" : "text-amber-400"
                          }
                        >
                          [{d.severity}]
                        </span>{" "}
                        {d.what}
                        {d.where ? ` @ ${d.where}` : ""}
                        {d.fix_hint ? (
                          <span className="text-zinc-600"> — fix: {d.fix_hint}</span>
                        ) : null}
                      </div>
                    ))}
                  </div>
                )}
                {!!(v.critique as { matches?: string[] }).matches?.length && (
                  <div>
                    <h4 className="mb-1 text-xs font-semibold text-zinc-400">Matches</h4>
                    <ul className="list-inside list-disc text-xs text-zinc-500">
                      {(v.critique as { matches?: string[] }).matches!.map((m, i) => (
                        <li key={i}>{m}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </Card>
            )}
          </div>
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

      {/* side column */}
      <div className="flex flex-col gap-4">
        <Card className="p-3">
          <SectionTitle>Agent activity</SectionTitle>
          <JobLog
            jobId={jobId}
            jobs={jobs.data}
            onSelectJob={setJobId}
            onDone={refresh}
          />
        </Card>

        {(v.status === "succeeded" || v.status === "needs_review") &&
          v.resolved_mode !== "topopt" && (
            <Btn
              variant="ghost"
              className="w-full border border-teal-800/60 text-teal-300 hover:bg-teal-950/40"
              onClick={() => setOptimizeOpen(true)}
              title="topology-optimize this part (SIMP + FEM verification)"
            >
              <Shrink className="h-4 w-4" /> Topology-optimize this part
            </Btn>
          )}

        {(v.status === "succeeded" || v.status === "needs_review") && (
          <Card className="p-3">
            <SectionTitle>Edit this model</SectionTitle>
            <textarea
              className={`${inputCls} h-20 resize-y`}
              placeholder="Describe the change… (e.g. 'add M5 mounting holes on the back', 'increase height to 15mm')"
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && editPrompt.trim()) {
                  e.preventDefault();
                  edit.mutate();
                }
              }}
            />
            <div className="my-2 flex flex-wrap gap-1">
              {(assets.data ?? [])
                .filter((as) => as.kind === "image")
                .map((as) => (
                  <button
                    key={as.id}
                    onClick={() =>
                      setEditAssets((s) => {
                        const n = new Set(s);
                        n.has(as.id) ? n.delete(as.id) : n.add(as.id);
                        return n;
                      })
                    }
                    className={`h-10 w-10 overflow-hidden rounded border-2 ${
                      editAssets.has(as.id) ? "border-blue-500" : "border-zinc-700"
                    }`}
                    title={as.filename}
                  >
                    <img src={as.url} className="h-full w-full object-cover" alt="" />
                  </button>
                ))}
              <button
                className="flex h-10 w-10 items-center justify-center rounded border border-dashed border-zinc-600 text-zinc-400 hover:border-zinc-400"
                onClick={() => fileRef.current?.click()}
                title="upload more reference images"
              >
                <Plus className="h-4 w-4" />
              </button>
              <input
                ref={fileRef}
                type="file"
                multiple
                hidden
                accept="image/*"
                onChange={(e) => uploadEditAssets(e.target.files)}
              />
            </div>
            <div className="my-2">
              <ModeSelector value={editMode} onChange={setEditMode} />
            </div>
            <div className="my-2">
              <OptionsPanel value={editOptions} onChange={setEditOptions} />
            </div>
            {(library.data ?? []).length > 0 && (
              <div className="my-2">
                <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                  Library parts{editLibraryParts.size ? ` — ${editLibraryParts.size} selected` : ""}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {(library.data ?? []).map((p) => {
                    const on = editLibraryParts.has(p.id);
                    return (
                      <button
                        key={p.id}
                        type="button"
                        title={`${p.kind}: ${p.caption || p.filename || p.name}`}
                        className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                          on
                            ? "border-blue-600 bg-blue-950/60 text-blue-200"
                            : "border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
                        }`}
                        onClick={() =>
                          setEditLibraryParts((s) => {
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
            {v.resolved_mode === "hybrid" && (
              <label className="my-2 flex items-center gap-2 text-xs text-zinc-400">
                <input
                  type="checkbox"
                  checked={regenMesh}
                  onChange={(e) => setRegenMesh(e.target.checked)}
                  className="accent-blue-500"
                />
                regenerate the organic base mesh (otherwise the parent mesh is reused)
              </label>
            )}
            {edit.isError && (
              <div className="mb-2 text-xs text-red-400">{String(edit.error)}</div>
            )}
            <Btn
              variant="success"
              className="w-full"
              onClick={() => edit.mutate()}
              busy={edit.isPending}
              disabled={!editPrompt.trim()}
            >
              Apply edit
            </Btn>
          </Card>
        )}

        {!!v.params_schema?.length && (
          <Card className="p-3">
            <ParamsPanel versionId={v.id} schema={v.params_schema} params={v.params} />
          </Card>
        )}

        <Card className="p-3">
          <SectionTitle>Versions</SectionTitle>
          <VersionTree versions={versions.data ?? []} activeId={v.id} />
        </Card>
      </div>

      <ConfirmModal
        open={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        onConfirm={() => del.mutate()}
        busy={del.isPending}
        title="Delete version"
        body={
          <>
            Delete <b className="text-zinc-200">v{v.number}</b> and its files? Child versions
            must be deleted first.
          </>
        }
      />

      <OptimizeDialog
        key={v.id}
        open={optimizeOpen}
        onClose={() => setOptimizeOpen(false)}
        busy={optimize.isPending}
        initial={v.spec?.load_cases}
        onSubmit={(body) => optimize.mutate(body)}
      />
      {optimize.isError && (
        <div className="fixed bottom-4 right-4 rounded-lg border border-red-800 bg-red-950/80 px-3 py-2 text-xs text-red-300">
          {String(optimize.error)}
        </div>
      )}
    </div>
  );
}
