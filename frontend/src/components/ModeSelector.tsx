import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../lib/api";
import type { GenMode, MeshHealth } from "../lib/types";

const MODES: GenMode[] = ["auto", "cad", "mesh", "hybrid", "topopt"];

export default function ModeSelector({
  value,
  onChange,
}: {
  value: GenMode;
  onChange: (m: GenMode) => void;
}) {
  const { t } = useTranslation();
  const [health, setHealth] = useState<MeshHealth | null>(null);

  useEffect(() => {
    api.meshHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const meshOk = health?.ok ?? false;

  return (
    <div>
      <div className="flex rounded border border-edge2 bg-panel p-0.5">
        {MODES.map((m) => {
          const disabled = (m === "mesh" || m === "hybrid") && !meshOk;
          const active = value === m;
          const tip = t(`mode.${m}_h`);
          return (
            <button
              key={m}
              type="button"
              title={disabled ? `${tip} — ${t("mode.server_down")}` : tip}
              disabled={disabled}
              onClick={() => onChange(m)}
              className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
                active ? "bg-brand-600 text-white" : "text-muted hover:text-fg"
              } ${disabled ? "cursor-not-allowed opacity-40" : ""}`}
            >
              {t(`mode.${m}`)}
            </button>
          );
        })}
      </div>
      {/* inline disable reason — a title tooltip alone is invisible until hover */}
      {health != null && !meshOk && (
        <div className="mt-1 text-[11px] text-faint">
          {t("mode.mesh_down")}
        </div>
      )}
      {(value === "mesh" || value === "hybrid") && (
        <div className="mt-1 text-[11px] text-faint">
          {t("mode.need_image")}
        </div>
      )}
      {value === "auto" && !meshOk && (
        <div className="mt-1 text-[11px] text-faint">
          {t("mode.auto_down")}
        </div>
      )}
      {value === "topopt" && (
        <div className="mt-1 text-[11px] text-faint">
          {t("mode.topopt_hint")}
        </div>
      )}
    </div>
  );
}
