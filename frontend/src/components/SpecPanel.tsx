import { useTranslation } from "react-i18next";
import type { PartSpec, SpecDim } from "../lib/types";
import { Card, Chip, SectionTitle } from "./ui";

const SOURCE_TONE: Record<string, "green" | "blue" | "amber"> = {
  stated: "green",
  inferred: "blue",
  assumed: "amber",
};

function DimChip({ axis, dim }: { axis: string; dim?: SpecDim }) {
  const { t } = useTranslation("version");
  if (!dim?.value) return null;
  return (
    <Chip tone={SOURCE_TONE[dim.source ?? ""] ?? "zinc"} title={t("spec_source", { s: dim.source ?? "?" })}>
      <b>{axis}</b> {dim.value} {dim.unit ?? "mm"}
      <span className="text-[9px] opacity-70">{dim.source}</span>
    </Chip>
  );
}

export default function SpecPanel({ spec }: { spec: PartSpec }) {
  const { t } = useTranslation("version");
  const features = spec.features ?? [];
  const dims = [
    { axis: "X", dim: spec.length_x },
    { axis: "Y", dim: spec.width_y },
    { axis: "Z", dim: spec.height_z },
  ];

  return (
    <Card className="p-4">
      <SectionTitle>{t("spec_title")}</SectionTitle>
      {spec.summary && <p className="mb-3 text-sm text-fg">{spec.summary}</p>}

      <div className="mb-3 flex flex-wrap gap-1.5">
        {dims.map((d) => (
          <DimChip key={d.axis} axis={d.axis} dim={d.dim} />
        ))}
        {spec.process && <Chip>{t("spec_process")}: {spec.process}</Chip>}
        {spec.material && <Chip>{t("spec_material")}: {spec.material}</Chip>}
        {spec.symmetry && <Chip>{t("spec_symmetry")}: {spec.symmetry}</Chip>}
        {spec.units && spec.units !== "mm" && <Chip tone="amber">{t("spec_units")}: {spec.units}</Chip>}
      </div>

      {features.length > 0 && (
        <>
          <h4 className="mb-1.5 text-xs font-semibold text-muted">
            {t("spec_features_n", { n: features.length })}
          </h4>
          <div className="mb-3 overflow-x-auto rounded border border-edge">
            <table className="w-full text-left text-xs">
              <thead className="bg-panel2/60 text-muted">
                <tr>
                  <th className="px-2 py-1.5 font-medium">{t("spec_f_feature")}</th>
                  <th className="px-2 py-1.5 font-medium">{t("spec_f_type")}</th>
                  <th className="px-2 py-1.5 font-medium">{t("spec_f_count")}</th>
                  <th className="px-2 py-1.5 font-medium">{t("spec_f_pos")}</th>
                  <th className="px-2 py-1.5 font-medium">{t("spec_f_dims")}</th>
                </tr>
              </thead>
              <tbody>
                {features.map((f, i) => (
                  <tr key={i} className="border-t border-edge text-fg">
                    <td className="px-2 py-1.5">
                      <div className="font-medium">{f.name || "—"}</div>
                      {f.description && (
                        <div className="text-[10px] text-faint">{f.description}</div>
                      )}
                    </td>
                    <td className="px-2 py-1.5">
                      <span className="rounded bg-panel2 px-1 font-mono text-[10px]">
                        {f.type ?? t("spec_other")}
                      </span>
                    </td>
                    <td className="px-2 py-1.5">{f.count ?? 1}×</td>
                    <td className="px-2 py-1.5 text-muted">{f.position || "—"}</td>
                    <td className="px-2 py-1.5 text-muted">
                      {f.standard && <div className="text-[10px]">{f.standard}</div>}
                      {f.dims &&
                        Object.entries(f.dims)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {spec.orientation_note && (
        <div className="mb-2 text-xs text-faint">{t("spec_orientation")}: {spec.orientation_note}</div>
      )}
      {spec.tolerances_note && (
        <div className="mb-2 text-xs text-faint">{t("spec_tolerances")}: {spec.tolerances_note}</div>
      )}

      {!!spec.must_keep?.length && (
        <div className="mb-2">
          <h4 className="mb-1 text-xs font-semibold text-muted">{t("spec_must_keep")}</h4>
          <ul className="list-inside list-disc text-xs text-muted">
            {spec.must_keep.map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        </div>
      )}
      {!!spec.assumptions?.length && (
        <div className="mb-2">
          <h4 className="mb-1 text-xs font-semibold text-muted">{t("spec_assumptions")}</h4>
          <ul className="list-inside list-disc text-xs text-muted">
            {spec.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </div>
      )}
      {!!spec.open_questions?.length && (
        <div>
          <h4 className="mb-1 text-xs font-semibold text-muted">
            {t("spec_open_qs", { n: spec.open_questions.length })}
          </h4>
          <ul className="flex flex-col gap-1 text-xs text-muted">
            {spec.open_questions.map((q, i) => (
              <li key={i} className="rounded border border-edge bg-panel px-2 py-1">
                {q.blocking && <Chip tone="amber" className="mr-1">{t("spec_blocking")}</Chip>}
                {q.question}
                {q.default_assumption && (
                  <span className="text-faint"> — {t("spec_default")}: {q.default_assumption}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
