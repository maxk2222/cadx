import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Asset, GenMode, Job, VersionDetail, VersionSummary } from "../lib/types";
import DfmPanel from "../components/DfmPanel";
import ModeSelector from "../components/ModeSelector";
import ExportPanel from "../components/ExportPanel";
import JobLog from "../components/JobLog";
import ParamsPanel from "../components/ParamsPanel";
import RenderGallery from "../components/RenderGallery";
import StatusBadge from "../components/StatusBadge";
import VersionTree from "../components/VersionTree";
import Viewer3D from "../components/Viewer3D";

export default function VersionPage() {
  const { id } = useParams<{ id: string }>();
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const [v, setV] = useState<VersionDetail | null>(null);
  const [versions, setVersions] = useState<VersionSummary[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [jobId, setJobId] = useState<string | null>(sp.get("job"));
  const [editPrompt, setEditPrompt] = useState("");
  const [editAssets, setEditAssets] = useState<Set<string>>(new Set());
  const [editMode, setEditMode] = useState<GenMode>("auto");
  const [regenMesh, setRegenMesh] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    if (!id) return;
    api.getVersion(id).then((d) => {
      setV(d);
      api.listVersions(d.project_id).then(setVersions);
      api.listAssets(d.project_id).then(setAssets);
    });
    api.listJobs(id).then((js) => {
      setJobs(js);
      if (!jobId && js.length) setJobId(js[0].id);
    });
  }, [id, jobId]);

  useEffect(load, [id]);

  const active = v && ["queued", "running"].includes(v.status);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [active, load]);

  const glb = v?.artifacts.find((a) => a.kind === "glb");
  const a = v?.analysis as { bbox_size?: number[]; volume?: number; is_valid?: boolean; watertight?: boolean } | null;

  const doEdit = async () => {
    if (!editPrompt.trim() || !id) return;
    setBusy(true);
    try {
      const r = await api.edit(id, editPrompt, [...editAssets], {
        mode: editMode,
        regenerate_mesh: regenMesh,
      });
      nav(`/versions/${r.version_id}?job=${r.job_id}`);
    } finally {
      setBusy(false);
    }
  };

  const uploadEditAssets = async (files: FileList | null) => {
    if (!files?.length || !v) return;
    const uploaded = await api.uploadAssets(v.project_id, Array.from(files));
    setEditAssets((s) => new Set([...s, ...uploaded.map((u) => u.id)]));
  };

  if (!v) return <div className="p-6 text-zinc-500">loading…</div>;

  return (
    <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 p-6 xl:grid-cols-4">
      {/* main column */}
      <div className="xl:col-span-3">
        <div className="mb-3 flex items-center gap-3">
          <Link to={`/projects/${v.project_id}`} className="text-sm text-zinc-400 hover:text-zinc-200">
            ← project
          </Link>
          <h1 className="text-xl font-bold">v{v.number}</h1>
          <StatusBadge status={v.status} />
          {(v.gen_mode || v.resolved_mode) && (
            <span
              className="rounded border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-zinc-300"
              title="requested → resolved generation mode"
            >
              {v.gen_mode}
              {v.resolved_mode && v.resolved_mode !== v.gen_mode ? ` → ${v.resolved_mode}` : ""}
            </span>
          )}
          {v.score != null && <span className="text-sm text-zinc-400">score {v.score}/100</span>}
          <span className="flex-1" />
          {active && jobs[0] && (
            <button
              className="rounded bg-red-700 px-2 py-1 text-xs hover:bg-red-600"
              onClick={() => api.cancelJob(jobs[0].id).then(load)}
            >
              cancel
            </button>
          )}
          <button
            className="text-sm text-red-400 hover:text-red-300"
            onClick={() => confirm("Delete this version?") && api.deleteVersion(v.id).then(() => nav(`/projects/${v.project_id}`))}
          >
            delete
          </button>
        </div>
        <div className="mb-3 text-sm text-zinc-400">{v.prompt}</div>
        {v.error && <div className="mb-3 rounded border border-red-800 bg-red-950/50 p-2 text-sm text-red-300">{v.error}</div>}

        <div className="mb-4 h-[420px] overflow-hidden rounded border border-zinc-800">
          <Viewer3D url={glb ? glb.url : null} />
        </div>

        {v.mesh && (
          <div className="mb-4 flex flex-wrap gap-x-4 gap-y-1 rounded border border-zinc-800 bg-zinc-900 p-3 text-sm">
            <span className="text-zinc-500">mesh:</span>
            {v.mesh.generation?.model && (
              <span title={v.mesh.generation.backend ?? ""}>model: <b>{v.mesh.generation.model}</b></span>
            )}
            {v.mesh.generation?.seed != null && <span>seed: {v.mesh.generation.seed}</span>}
            {(v.mesh.triangles ?? v.mesh.generation?.triangles) != null && (
              <span>triangles: {v.mesh.triangles ?? v.mesh.generation?.triangles}</span>
            )}
            {v.mesh.orientation && (
              <span title={v.mesh.orientation_source ?? ""}>
                orientation: {v.mesh.orientation.up ?? "?"} / {v.mesh.orientation.front ?? "?"}
              </span>
            )}
            {v.mesh.scale_factor != null && <span>scale ×{v.mesh.scale_factor}</span>}
            {v.mesh.step_tessellated && (
              <span className="text-amber-400" title="STEP export is tessellated (mesh-derived), not an analytic B-Rep">
                STEP is tessellated
              </span>
            )}
          </div>
        )}

        {a && (
          <div className="mb-4 flex flex-wrap gap-4 rounded border border-zinc-800 bg-zinc-900 p-3 text-sm">
            {a.bbox_size && (
              <span>
                size: <b>{a.bbox_size.map((n) => n.toFixed(1)).join(" × ")}</b> mm
              </span>
            )}
            {a.volume != null && <span>volume: {(a.volume / 1000).toFixed(1)} cm³</span>}
            <span>valid: {String(a.is_valid)}</span>
            <span>watertight: {String(a.watertight)}</span>
          </div>
        )}

        <RenderGallery artifacts={v.artifacts} />

        <div className="mt-4">
          <ExportPanel versionId={v.id} artifacts={v.artifacts} onRefresh={load} />
        </div>

        {v.code != null && (
          <div className="mt-4">
            <button className="text-sm text-zinc-400 underline" onClick={() => { setShowCode(!showCode); if (!code) api.getCode(v.id).then(setCode); }}>
              {showCode ? "hide generated code" : "show generated code"}
            </button>
            {showCode && <pre className="mt-2 max-h-96 overflow-auto rounded bg-zinc-900 p-3 text-xs">{code}</pre>}
          </div>
        )}
      </div>

      {/* side column */}
      <div className="flex flex-col gap-5">
        <JobLog jobId={jobId} onDone={load} />

        {(v.status === "succeeded" || v.status === "needs_review") && (
          <div className="rounded border border-zinc-800 p-3">
            <h3 className="mb-2 text-sm font-semibold text-zinc-300">Edit this model</h3>
            <textarea
              className="h-20 w-full rounded bg-zinc-800 p-2 text-sm"
              placeholder="Describe the change… (e.g. 'add M5 mounting holes on the back', 'increase height to 15mm')"
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
            />
            <div className="my-2 flex flex-wrap gap-1">
              {assets
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
                    className={`h-10 w-10 overflow-hidden rounded border-2 ${editAssets.has(as.id) ? "border-blue-500" : "border-zinc-700"}`}
                  >
                    <img src={as.url} className="h-full w-full object-cover" alt="" />
                  </button>
                ))}
              <button
                className="flex h-10 w-10 items-center justify-center rounded border border-dashed border-zinc-600 text-zinc-400"
                onClick={() => fileRef.current?.click()}
                title="upload more reference images"
              >
                +
              </button>
              <input ref={fileRef} type="file" multiple hidden accept="image/*" onChange={(e) => uploadEditAssets(e.target.files)} />
            </div>
            <div className="my-2">
              <ModeSelector value={editMode} onChange={setEditMode} />
            </div>
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
            <button
              onClick={doEdit}
              disabled={busy || !editPrompt.trim()}
              className="w-full rounded bg-emerald-600 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-50"
            >
              {busy ? "starting…" : "Apply edit"}
            </button>
          </div>
        )}

        <div className="rounded border border-zinc-800 p-3">
          <ParamsPanel versionId={v.id} schema={v.params_schema} params={v.params} />
        </div>

        {v.dfm && (
          <div className="rounded border border-zinc-800 p-3">
            <h3 className="mb-2 text-sm font-semibold text-zinc-300">Manufacturability (DfM)</h3>
            <DfmPanel warnings={v.dfm.warnings ?? []} />
          </div>
        )}

        {v.critique && v.critique.discrepancies?.length > 0 && (
          <div className="rounded border border-zinc-800 p-3">
            <h3 className="mb-2 text-sm font-semibold text-zinc-300">Critique findings</h3>
            {v.critique.discrepancies.map((d, i) => (
              <div key={i} className="mb-1 text-xs text-zinc-400">
                <span className={d.severity === "major" ? "text-red-400" : "text-amber-400"}>[{d.severity}]</span> {d.what}
                {d.where ? ` @ ${d.where}` : ""}
              </div>
            ))}
          </div>
        )}

        <div className="rounded border border-zinc-800 p-3">
          <h3 className="mb-2 text-sm font-semibold text-zinc-300">Versions</h3>
          <VersionTree versions={versions} activeId={v.id} />
        </div>
      </div>
    </div>
  );
}
