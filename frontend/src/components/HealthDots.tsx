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
      <span className="hidden text-[10px] font-medium uppercase tracking-wider text-zinc-500 sm:inline">
        {label}
      </span>
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
  // shares its cache with the settings-page card / "test provider" button —
  // a slow poll here keeps the dot fresh between settings visits
  const llm = useQuery({
    queryKey: ["llmHealth"],
    queryFn: api.llmHealth,
    refetchInterval: 300_000,
    staleTime: 120_000,
    retry: false,
  });

  const apiOk = health.data?.ok ?? false;
  const llmConfigured = health.data?.llm_configured ?? false;
  const lp = llm.data;
  const meshOk = mesh.data?.ok ?? false;
  const meshErr = mesh.data?.error ?? mesh.error?.message;
  const models = mesh.data?.models ?? [];
  const nMesh = models.filter((m) => m.kind !== "image" && m.available).length;
  const nImg = models.filter((m) => m.kind === "image" && m.available).length;

  return (
    <Link
      to="/settings?tab=provider"
      className="grid grid-cols-2 gap-x-3 gap-y-1.5 rounded-md px-2 py-1 hover:bg-zinc-900"
      title="service status — open settings"
    >
      <Dot ok={apiOk} label="API" title={apiOk ? "API online" : "API unreachable"} />
      <Dot
        ok={!!lp?.ok}
        warn={llmConfigured ? !lp : true}
        label="LLM"
        title={
          lp?.ok
            ? `LLM online · ${lp.latency_ms}ms${lp.vision_ok ? " · vision" : ""}`
            : !llmConfigured
              ? "no LLM API key set"
              : lp
                ? `LLM unreachable${lp.error ? ` — ${lp.error}` : ""}`
                : "LLM status unknown (untested)"
        }
      />
      <Dot
        ok={meshOk}
        warn={!meshOk && apiOk}
        label="Mesh"
        title={meshOk ? `mesh server: ${mesh.data?.device} · ${nMesh} model(s)` : `mesh server offline${meshErr ? ` — ${meshErr}` : ""}`}
      />
      <Dot
        ok={meshOk && nImg > 0}
        warn={meshOk && nImg === 0}
        label="Image"
        title={
          !meshOk
            ? `image gen offline (mesh server unreachable${meshErr ? ` — ${meshErr}` : ""})`
            : nImg > 0
              ? `image gen: ${nImg} adapter(s)`
              : "image gen: no adapters installed"
        }
      />
    </Link>
  );
}
