import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation("common");
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
      title={t("health.hint")}
    >
      <Dot ok={apiOk} label="API" title={apiOk ? t("health.api_on") : t("health.api_off")} />
      <Dot
        ok={!!lp?.ok}
        warn={llmConfigured ? !lp : true}
        label="LLM"
        title={
          lp?.ok
            ? t("health.llm_on", { ms: lp.latency_ms, vision: lp.vision_ok ? ` · ${t("health.vision")}` : "" })
            : !llmConfigured
              ? t("health.llm_nokey")
              : lp
                ? t("health.llm_off", { e: lp.error ? ` — ${lp.error}` : "" })
                : t("health.llm_unknown")
        }
      />
      <Dot
        ok={meshOk}
        warn={!meshOk && apiOk}
        label="Mesh"
        title={meshOk ? t("health.mesh_on", { dev: mesh.data?.device, n: nMesh }) : t("health.mesh_off", { e: meshErr ? ` — ${meshErr}` : "" })}
      />
      <Dot
        ok={meshOk && nImg > 0}
        warn={meshOk && nImg === 0}
        label="Image"
        title={
          !meshOk
            ? t("health.img_off", { e: meshErr ? ` — ${meshErr}` : "" })
            : nImg > 0
              ? t("health.img_on", { n: nImg })
              : t("health.img_none")
        }
      />
    </Link>
  );
}
