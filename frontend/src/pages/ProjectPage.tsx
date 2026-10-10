import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams, Link } from "react-router-dom";
import { api } from "../lib/api";
import type { Asset, GenMode, Project, VersionSummary } from "../lib/types";
import ModeSelector from "../components/ModeSelector";
import VersionTree from "../components/VersionTree";

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const nav = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [versions, setVersions] = useState<VersionSummary[]>([]);
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState<GenMode>("auto");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    if (!id) return;
    api.getProject(id).then((p) => {
      setProject(p);
      const gm = (p.settings as Record<string, unknown>)?.gen_mode;
      if (gm === "auto" || gm === "cad" || gm === "mesh" || gm === "hybrid") setMode(gm);
    }).catch(() => nav("/"));
    api.listAssets(id).then(setAssets).catch(() => {});
    api.listVersions(id).then(setVersions).catch(() => {});
  }, [id]);

  useEffect(load, [load]);
  // poll versions while any is active
  useEffect(() => {
    const active = versions.some((v) => ["queued", "running"].includes(v.status));
    if (!active) return;
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [versions, load]);

  const upload = async (files: FileList | null) => {
    if (!files?.length || !id) return;
    await api.uploadAssets(id, Array.from(files));
    load();
  };

  const generate = async () => {
    if (!prompt.trim() || !id) return;
    setBusy(true);
    setError("");
    try {
      const r = await api.generate(id, prompt, [...selected], mode);
      nav(`/versions/${r.version_id}?job=${r.job_id}`);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  const toggle = (aid: string) =>
    setSelected((s) => {
      const n = new Set(s);
      n.has(aid) ? n.delete(aid) : n.add(aid);
      return n;
    });

  if (!project) return <div className="p-6 text-zinc-500">loading…</div>;

  return (
    <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 p-6 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <div className="mb-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">{project.name}</h1>
          <button
            className="text-sm text-red-400 hover:text-red-300"
            onClick={() => confirm("Delete project and all files?") && api.deleteProject(id!).then(() => nav("/"))}
          >
            delete
          </button>
        </div>

        {/* assets */}
        <div
          className="mb-4 rounded border-2 border-dashed border-zinc-700 p-4 text-center text-sm text-zinc-400"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            upload(e.dataTransfer.files);
          }}
        >
          Drop reference images / STEP / STL / DXF here, or{" "}
          <button className="text-blue-400 underline" onClick={() => fileRef.current?.click()}>
            browse
          </button>
          <input ref={fileRef} type="file" multiple hidden accept="image/*,.step,.stp,.stl,.dxf" onChange={(e) => upload(e.target.files)} />
        </div>
        <div className="mb-4 grid grid-cols-4 gap-2">
          {assets.map((a) => (
            <button
              key={a.id}
              onClick={() => toggle(a.id)}
              className={`relative rounded border-2 ${selected.has(a.id) ? "border-blue-500" : "border-zinc-800"}`}
              title={`${a.filename}${a.caption ? ` — ${a.caption}` : ""}`}
            >
              {a.kind === "image" ? (
                <img src={a.url} className="h-20 w-full rounded object-cover" alt={a.filename} />
              ) : (
                <div className="flex h-20 items-center justify-center text-xs text-zinc-400">{a.filename}</div>
              )}
              <span className="absolute right-1 top-1 rounded bg-black/70 px-1 text-[9px]">{a.kind.replace("reference_", "")}</span>
            </button>
          ))}
        </div>

        {/* prompt */}
        <textarea
          className="mb-2 h-32 w-full rounded bg-zinc-800 p-3 text-sm"
          placeholder={
            "Describe the part: shape, dimensions (X length × Y width × Z height), holes, threads, tolerances, material, process (FDM/CNC)…\n\nExample: Mounting plate 80×60×8mm, four M4 clearance holes at 10mm from corners, countersunk, PLA for FDM."
          }
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        {error && <div className="mb-2 text-sm text-red-400">{error}</div>}
        <div className="mb-2">
          <ModeSelector value={mode} onChange={setMode} />
        </div>
        <button
          onClick={generate}
          disabled={busy || !prompt.trim()}
          className="w-full rounded bg-blue-600 py-2 font-semibold hover:bg-blue-500 disabled:opacity-50"
        >
          {busy ? "starting…" : `Generate model${selected.size ? ` (${selected.size} assets)` : ""}`}
        </button>
      </div>

      <div>
        <h2 className="mb-2 text-sm font-semibold text-zinc-300">Versions</h2>
        <VersionTree versions={versions} />
      </div>
    </div>
  );
}
