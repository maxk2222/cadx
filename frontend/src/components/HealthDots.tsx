import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../lib/api";

function Dot({ ok, warn, label, title }: { ok: boolean; warn?: boolean; label: string; title?: string }) {
  const cls = warn
    ? "bg-amber-500"
    : ok
      ? "bg-emerald-500"
      : "bg-red-500";
  return (
    <span className="flex items-center gap-1.5" title={title ?? label}>
      <span className={`h-2 w-2 rounded-full ${cls}`} />
      <span className="hidden text-[11px] text-zinc-500 sm:inline">{label}</span>
    </span>
  );
}

export default function HealthDots() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: false,
  });
  const mesh = useQuery({
    queryKey: ["meshHealth"],
    queryFn: api.meshHealth,
    refetchInterval: 30_000,
    retry: false,
  });

  const apiOk = health.data?.ok ?? false;
  const llmOk = health.data?.llm_configured ?? false;
  const meshOk = mesh.data?.ok ?? false;
  const meshErr = mesh.data?.error ?? mesh.error?.message;

  return (
    <Link
      to="/settings?tab=provider"
      className="flex items-center gap-3 rounded-md px-2 py-1 hover:bg-zinc-900"
      title="service status — open settings"
    >
      <Dot ok={apiOk} label="api" title={apiOk ? "API online" : "API unreachable"} />
      <Dot
        ok={llmOk}
        warn={apiOk && !llmOk}
        label="llm"
        title={llmOk ? "LLM configured" : "no LLM API key set"}
      />
      <Dot
        ok={meshOk}
        warn={!meshOk && (health.data?.ok ?? false)}
        label="mesh"
        title={meshOk ? `mesh server: ${mesh.data?.device}` : `mesh server offline${meshErr ? ` — ${meshErr}` : ""}`}
      />
    </Link>
  );
}
