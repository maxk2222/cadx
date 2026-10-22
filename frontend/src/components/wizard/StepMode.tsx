import { ArrowRight, Boxes, Check, Cpu, Image as ImageIcon, Shrink, Wand2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { GenMode, MeshHealth } from "../../lib/types";
import { Btn, Chip } from "../ui";

const MODE_CARDS: { id: GenMode; icon: typeof Cpu }[] = [
  { id: "auto", icon: Wand2 },
  { id: "cad", icon: Cpu },
  { id: "mesh", icon: ImageIcon },
  { id: "hybrid", icon: Boxes },
  { id: "topopt", icon: Shrink },
];

export default function StepMode({
  value,
  onChange,
  suggested,
  hasImages,
  health,
  loading,
  onNext,
  onBack,
}: {
  value: GenMode;
  onChange: (m: GenMode) => void;
  suggested: GenMode;
  hasImages: boolean;
  /** mesh-server health from the parent query — undefined while loading */
  health: MeshHealth | undefined;
  loading?: boolean;
  onNext: () => void;
  onBack: () => void;
}) {
  const { t } = useTranslation("wizard");
  const meshOk = health?.ok ?? false;

  return (
    <div>
      <h2 className="text-lg font-semibold">{t("mode_title")}</h2>
      <p className="mt-1 text-sm text-faint">{t("mode_hint")}</p>
      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {MODE_CARDS.map((m) => {
          // while health is still loading, don't flash-disable mesh modes
          const disabled =
            !loading && (m.id === "mesh" || m.id === "hybrid") && !meshOk;
          const active = value === m.id;
          const Icon = m.icon;
          return (
            <button
              key={m.id}
              type="button"
              disabled={disabled}
              title={
                disabled
                  ? `${t(`mode_${m.id}_title`)} — ${t("mesh_unreachable")}`
                  : t(`mode_${m.id}_title`)
              }
              onClick={() => onChange(m.id)}
              className={`relative rounded-lg border p-3 text-left transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-40 ${
                active
                  ? "glow-accent border-brand-600 bg-brand-950/40"
                  : "border-edge bg-panel/60 hover:border-edge2 hover:bg-panel"
              }`}
            >
              {active && (
                <span className="absolute right-2 top-2 flex h-4 w-4 items-center justify-center rounded-full bg-brand-500">
                  <Check className="h-3 w-3 text-white" />
                </span>
              )}
              <div className="flex items-center gap-2">
                <Icon
                  className={`h-4 w-4 ${active ? "text-brand-300" : "text-faint"}`}
                />
                <span className="font-medium">{t(`mode_${m.id}`)}</span>
                {m.id === suggested && (
                  <Chip tone="blue" className="ml-1">
                    {t("suggested")}
                  </Chip>
                )}
              </div>
              <div className="mt-1 text-xs text-muted">{t(`mode_${m.id}_desc`)}</div>
            </button>
          );
        })}
      </div>
      {(value === "mesh" || value === "hybrid") && !hasImages && (
        <div className="mt-2 text-[11px] text-amber-400">{t("no_images_warn")}</div>
      )}
      {value === "topopt" && (
        <div className="mt-2 text-[11px] text-faint">{t("topopt_hint")}</div>
      )}
      <div className="mt-6 flex items-center justify-between">
        <Btn variant="ghost" onClick={onBack}>
          {t("common:action.back")}
        </Btn>
        <Btn onClick={onNext}>
          {t("continue")} <ArrowRight className="h-4 w-4" />
        </Btn>
      </div>
    </div>
  );
}
