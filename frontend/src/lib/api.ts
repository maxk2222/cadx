import type {
  AnchorsResult,
  Asset,
  CatalogEntry,
  CatalogSearchHit,
  FemPreviewResult,
  GenMode,
  Job,
  JobListItem,
  LibraryPart,
  LiveRenders,
  LoadCase,
  MeshHealth,
  MeshModelInfo,
  MeshOp,
  MeshPreviewResult,
  OptimizeSpec,
  Project,
  ProviderTestResult,
  RegionSpec,
  RunOptions,
  Settings,
  SettingsUpdate,
  VersionDetail,
  VersionSummary,
} from "./types";

const BASE = "/api";

/** Friendly message for a failed response — JSON detail, else status text
 *  (proxies may answer with an HTML error page that must not reach the UI).
 *  FastAPI validation errors arrive as a detail array of {loc, msg}. */
async function errText(r: Response): Promise<string> {
  const body = await r.text();
  try {
    const d = (JSON.parse(body) as { detail?: unknown }).detail;
    if (Array.isArray(d)) {
      return d
        .map((x) => {
          const loc = Array.isArray(x?.loc)
            ? x.loc.filter((p: unknown) => p !== "body").join(".")
            : "";
          return loc ? `${loc}: ${x?.msg}` : String(x?.msg ?? x);
        })
        .join("; ");
    }
    if (typeof d === "string" && d) return d;
    if (d) return JSON.stringify(d).slice(0, 300);
  } catch {
    /* not JSON */
  }
  if (body.trimStart().startsWith("<")) return `${r.status} ${r.statusText}`;
  return `${r.status} ${body.slice(0, 300)}`;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, init);
  if (!r.ok) throw new Error(await errText(r));
  return r.json();
}

const json = (body: unknown) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<{ ok: boolean; llm_configured: boolean }>("/health"),

  listProjects: () => req<Project[]>("/projects"),
  getProject: (id: string) => req<Project>(`/projects/${id}`),
  createProject: (name: string, description = "", settings?: unknown) =>
    req<Project>("/projects", json({ name, description, settings })),
  updateProject: (id: string, body: Partial<{ name: string; description: string; settings: unknown }>) =>
    req<Project>(`/projects/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  deleteProject: (id: string) => req<{ ok: boolean }>(`/projects/${id}`, { method: "DELETE" }),

  listAssets: (pid: string) => req<Asset[]>(`/projects/${pid}/assets`),
  uploadAssets: async (pid: string, files: File[], caption = ""): Promise<Asset[]> => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const r = await fetch(`${BASE}/projects/${pid}/assets?caption=${encodeURIComponent(caption)}`, {
      method: "POST",
      body: fd,
    });
    if (!r.ok) throw new Error(await errText(r));
    return r.json();
  },
  deleteAsset: (id: string) => req<{ ok: boolean }>(`/assets/${id}`, { method: "DELETE" }),
  deriveAsset: async (id: string, png: Blob): Promise<Asset> => {
    const fd = new FormData();
    fd.append("file", png, "edit.png");
    const r = await fetch(`${BASE}/assets/${id}/derivative`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await errText(r));
    return r.json();
  },
  removeAssetBg: async (id: string): Promise<Blob> => {
    const r = await fetch(`${BASE}/assets/${id}/remove-bg`, { method: "POST" });
    if (!r.ok) throw new Error(await errText(r));
    return r.blob();
  },
  updateAsset: (id: string, patch: { caption?: string; role_hint?: string }) =>
    req<Asset>(`/assets/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  // local text-to-image generation (mesh-server image adapters)
  imageModels: () => req<MeshModelInfo[]>("/images/models"),
  generateImages: (
    pid: string,
    body: {
      prompt: string;
      negative?: string;
      model?: string;
      seed?: number;
      count?: number;
      width?: number;
      height?: number;
      steps?: number;
      guidance?: number;
      caption?: string;
      reference_asset_id?: string;
      strength?: number;
      mark_sketch?: boolean;
    },
  ) => req<Asset[]>(`/projects/${pid}/images`, json(body)),

  generate: (
    pid: string,
    prompt: string,
    asset_ids: string[],
    mode?: GenMode,
    options?: RunOptions,
    library_part_ids?: string[],
  ) =>
    req<{ version_id: string; job_id: string }>(
      `/projects/${pid}/generate`,
      json({ prompt, asset_ids, library_part_ids: library_part_ids ?? [], mode, options })
    ),
  edit: (vid: string, prompt: string, asset_ids: string[], opts?: { mode?: GenMode; regenerate_mesh?: boolean; options?: RunOptions; library_part_ids?: string[] }) =>
    req<{ version_id: string; job_id: string }>(
      `/versions/${vid}/edit`,
      json({ prompt, asset_ids, library_part_ids: opts?.library_part_ids ?? [], mode: opts?.mode, regenerate_mesh: opts?.regenerate_mesh ?? false, options: opts?.options })
    ),
  setParams: (vid: string, params: Record<string, unknown>) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/params`, json({ params })),
  reexport: (vid: string) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/reexport`, { method: "POST" }),
  optimize: (vid: string, body: { load_cases: LoadCase[]; optimize: Partial<OptimizeSpec>; material?: string }) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/optimize`, json(body)),
  topoptResample: (vid: string, body: { density_threshold?: number; min_member_mm?: number }) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/topopt-resample`, json(body)),
  refine: (vid: string, body: { features?: Record<string, unknown>[]; load_cases?: LoadCase[]; keep_regions?: RegionSpec[]; material?: string; voxel_mm?: number; restore_planes?: boolean; helical_threads?: boolean; auto_plane_restore?: boolean }) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/refine`, json(body)),
  anchors: (vid: string) => req<AnchorsResult>(`/versions/${vid}/anchors`),

  // visual mesh editor + structural editor
  meshPreview: (vid: string, body: { ops: MeshOp[]; post?: Record<string, unknown> }) =>
    req<MeshPreviewResult>(`/versions/${vid}/mesh-preview`, json(body)),
  meshEdit: (vid: string, body: { ops: MeshOp[]; post?: Record<string, unknown>; note?: string }) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/mesh-edit`, json(body)),
  femPreview: (vid: string, body: { load_cases: LoadCase[]; keep_regions?: RegionSpec[]; render_pngs?: boolean }) =>
    req<FemPreviewResult>(`/versions/${vid}/fem-preview`, json(body)),
  runFem: (vid: string, body: { load_cases: LoadCase[]; keep_regions?: RegionSpec[]; material?: string }) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/fem`, json(body)),
  saveEditorState: (vid: string, ui_editor: Record<string, unknown>) =>
    req<{ ok: boolean }>(`/versions/${vid}/editor-state`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ui_editor }),
    }),

  listVersions: (pid: string) => req<VersionSummary[]>(`/projects/${pid}/versions`),
  getVersion: (id: string) => req<VersionDetail>(`/versions/${id}`),
  previewUrl: (id: string, view?: string) =>
    `${BASE}/versions/${id}/preview${view ? `?view=${encodeURIComponent(view)}` : ""}`,
  liveRenders: (id: string) => req<LiveRenders>(`/versions/${id}/live-renders`),
  deleteVersion: (id: string) => req<{ ok: boolean }>(`/versions/${id}`, { method: "DELETE" }),
  getCode: (id: string) => fetch(`${BASE}/versions/${id}/code`).then((r) => r.text()),

  getJob: (id: string) => req<Job>(`/jobs/${id}`),
  listJobs: (vid: string) => req<Job[]>(`/versions/${vid}/jobs`),
  listJobsAll: (opts?: { active?: boolean; limit?: number }) => {
    const q = new URLSearchParams();
    if (opts?.active) q.set("active", "true");
    if (opts?.limit) q.set("limit", String(opts.limit));
    return req<JobListItem[]>(`/jobs${q.size ? `?${q}` : ""}`);
  },
  answerJob: (id: string, answers: { answer: string }[]) =>
    req<{ version_id: string; job_id: string }>(`/jobs/${id}/answer`, json({ answers })),
  cancelJob: (id: string) => req<{ ok: boolean }>(`/jobs/${id}/cancel`, { method: "POST" }),
  finishJob: (id: string) => req<{ ok: boolean }>(`/jobs/${id}/finish`, { method: "POST" }),
  restartJob: (id: string) =>
    req<{ version_id: string; job_id: string }>(`/jobs/${id}/restart`, { method: "POST" }),

  getSettings: () => req<Settings>("/settings"),
  updateSettings: (body: SettingsUpdate) =>
    req<Settings>("/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  testProvider: () => req<ProviderTestResult>("/settings/test", { method: "POST" }),

  getCatalog: () => req<Record<string, Record<string, CatalogEntry>>>("/catalog"),
  searchCatalog: (q: string) =>
    req<{ results: CatalogSearchHit[] }>(`/catalog/search?q=${encodeURIComponent(q)}`),
  lookupPart: (p: { class_name: string; fastener_type?: string; size?: string }) =>
    req<Record<string, unknown>>(
      `/catalog/lookup?${new URLSearchParams(
        Object.fromEntries(Object.entries(p).filter(([, v]) => v != null)) as Record<
          string,
          string
        >
      ).toString()}`
    ),

  listLibrary: () => req<LibraryPart[]>("/library"),
  uploadLibrary: async (files: File[], name = "", caption = ""): Promise<LibraryPart[]> => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const q = new URLSearchParams({ name, caption });
    const r = await fetch(`${BASE}/library?${q}`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await errText(r));
    return r.json();
  },
  saveToLibrary: (vid: string, name: string, asTemplate: boolean, caption = "") =>
    req<LibraryPart>(
      `/library/from-version/${vid}`,
      json({ name, caption, as_template: asTemplate })
    ),
  updateLibraryPart: (id: string, body: { name?: string; caption?: string }) =>
    req<LibraryPart>(`/library/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteLibraryPart: (id: string) =>
    req<{ ok: boolean }>(`/library/${id}`, { method: "DELETE" }),

  meshHealth: () => req<MeshHealth>("/mesh/health"),
  meshModels: () => req<MeshModelInfo[]>("/mesh/models"),
};

export function jobEventsUrl(jobId: string): string {
  return `${BASE}/jobs/${jobId}/events`;
}
