import type {
  Asset,
  GenMode,
  Job,
  MeshHealth,
  MeshModelInfo,
  Project,
  Settings,
  VersionDetail,
  VersionSummary,
} from "./types";

const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, init);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
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
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  deleteAsset: (id: string) => req<{ ok: boolean }>(`/assets/${id}`, { method: "DELETE" }),
  updateAsset: (id: string, caption: string) =>
    req<Asset>(`/assets/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ caption }),
    }),

  generate: (pid: string, prompt: string, asset_ids: string[], mode?: GenMode) =>
    req<{ version_id: string; job_id: string }>(
      `/projects/${pid}/generate`,
      json({ prompt, asset_ids, mode })
    ),
  edit: (vid: string, prompt: string, asset_ids: string[], opts?: { mode?: GenMode; regenerate_mesh?: boolean }) =>
    req<{ version_id: string; job_id: string }>(
      `/versions/${vid}/edit`,
      json({ prompt, asset_ids, mode: opts?.mode, regenerate_mesh: opts?.regenerate_mesh ?? false })
    ),
  setParams: (vid: string, params: Record<string, unknown>) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/params`, json({ params })),
  reexport: (vid: string) =>
    req<{ version_id: string; job_id: string }>(`/versions/${vid}/reexport`, { method: "POST" }),

  listVersions: (pid: string) => req<VersionSummary[]>(`/projects/${pid}/versions`),
  getVersion: (id: string) => req<VersionDetail>(`/versions/${id}`),
  deleteVersion: (id: string) => req<{ ok: boolean }>(`/versions/${id}`, { method: "DELETE" }),
  getCode: (id: string) => fetch(`${BASE}/versions/${id}/code`).then((r) => r.text()),

  getJob: (id: string) => req<Job>(`/jobs/${id}`),
  listJobs: (vid: string) => req<Job[]>(`/versions/${vid}/jobs`),
  answerJob: (id: string, answers: { answer: string }[]) =>
    req<{ version_id: string; job_id: string }>(`/jobs/${id}/answer`, json({ answers })),
  cancelJob: (id: string) => req<{ ok: boolean }>(`/jobs/${id}/cancel`, { method: "POST" }),

  getSettings: () => req<Settings>("/settings"),
  updateSettings: (body: Partial<Settings>) =>
    req<Settings>("/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),

  getCatalog: () => req<Record<string, Record<string, Record<string, string[]>>>>("/catalog"),

  meshHealth: () => req<MeshHealth>("/mesh/health"),
  meshModels: () => req<MeshModelInfo[]>("/mesh/models"),
};

export function jobEventsUrl(jobId: string): string {
  return `${BASE}/jobs/${jobId}/events`;
}
