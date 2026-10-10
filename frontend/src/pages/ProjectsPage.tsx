import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import type { Project } from "../lib/types";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  const load = () => api.listProjects().then(setProjects).catch(console.error);
  useEffect(() => {
    load();
  }, []);

  const create = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      await api.createProject(name.trim());
      setName("");
      load();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl p-6">
      <h1 className="mb-4 text-2xl font-bold">Projects</h1>
      <div className="mb-6 flex gap-2">
        <input
          className="flex-1 rounded bg-zinc-800 px-3 py-2"
          placeholder="new project name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && create()}
        />
        <button
          onClick={create}
          disabled={busy}
          className="rounded bg-blue-600 px-4 py-2 font-medium hover:bg-blue-500 disabled:opacity-50"
        >
          Create
        </button>
      </div>
      <div className="flex flex-col gap-2">
        {projects.map((p) => (
          <Link
            key={p.id}
            to={`/projects/${p.id}`}
            className="flex items-center justify-between rounded border border-zinc-800 bg-zinc-900 px-4 py-3 hover:border-zinc-600"
          >
            <div>
              <div className="font-medium">{p.name}</div>
              {p.description && <div className="text-sm text-zinc-500">{p.description}</div>}
            </div>
            <div className="text-sm text-zinc-500">{p.version_count} versions</div>
          </Link>
        ))}
        {!projects.length && <div className="text-zinc-500">No projects yet — create one above.</div>}
      </div>
    </div>
  );
}
