import { Activity, Gauge, TriangleAlert } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { VersionDetail } from "../../lib/types";
import AnchorsPanel from "../AnchorsPanel";
import DfmPanel from "../DfmPanel";
import FemPanel from "../FemPanel";
import TopOptPanel from "../TopOptPanel";
import { Card, Chip, SectionCollapsible, StatRow } from "../ui";

/** Quality tab — verdict header up front (score / DfM / SF / issues), then
 *  each report behind its own collapsible so the tab reads as a checklist
 *  instead of a 7-panel scroll. */
export default function QualityTab({
  v,
  onStructuralStarted,
}: {
  v: VersionDetail;
  onStructuralStarted: (r: { version_id: string; job_id: string }) => void;
}) {
  const { t } = useTranslation("version");
  const dfmCount = v.dfm?.warnings?.length ?? 0;
  const issueCount = v.issues?.length ?? 0;
  const sf = v.fem?.cases?.length
    ? Math.min(...v.fem.cases.map((c) => c.safety_factor ?? Infinity))
    : null;

  return (
    <div className="flex flex-col gap-3">
      {/* verdict strip */}
      <Card className="px-3 py-2.5">
        <StatRow
          items={[
            v.score != null && (
              <span key="score" className="flex items-center gap-1">
                <Gauge className="h-3 w-3" />
                {t("score", { n: v.score })}
              </span>
            ),
            v.critique && (
              <Chip
                key="crit"
                tone={v.critique.passed ? "green" : "amber"}
                className="py-0"
              >
                {v.critique.passed ? t("crit_pass") : t("crit_fail")} ·{" "}
                {v.critique.score}
              </Chip>
            ),
            dfmCount > 0 && (
              <span key="dfm" className="flex items-center gap-1 text-amber-300">
                <TriangleAlert className="h-3 w-3" />
                {t("dfm_count", { count: dfmCount })}
              </span>
            ),
            issueCount > 0 && (
              <span key="iss" className="text-red-300">
                {t("issues_count", { count: issueCount })}
              </span>
            ),
            sf != null && (
              <span
                key="sf"
                className={`flex items-center gap-1 ${sf < 1 ? "text-red-300" : "text-emerald-300"}`}
              >
                <Activity className="h-3 w-3" /> SF {sf.toFixed(2)}
              </span>
            ),
          ]}
        />
      </Card>

      {(!!v.mesh?.topopt || !!v.mesh?.refine) && (
        <SectionCollapsible
          title={t("q_topopt")}
          persistKey="vq:topopt"
          pad="p-4"
        >
          <TopOptPanel
            topopt={v.mesh!.topopt ?? {}}
            refine={v.mesh!.refine}
            versionId={v.id}
            onJobStarted={onStructuralStarted}
          />
        </SectionCollapsible>
      )}

      <SectionCollapsible title={t("q_anchors")} persistKey="vq:anchors" pad="p-4">
        <AnchorsPanel versionId={v.id} />
      </SectionCollapsible>

      {!!v.fem?.cases?.length && (
        <SectionCollapsible
          title={t("q_fem")}
          count={v.fem.cases.length}
          persistKey="vq:fem"
          pad="p-4"
        >
          <FemPanel fem={v.fem} artifacts={v.artifacts} />
        </SectionCollapsible>
      )}

      <SectionCollapsible
        title={t("q_dfm")}
        count={dfmCount || undefined}
        persistKey="vq:dfm"
        pad="p-4"
      >
        {v.dfm ? (
          <DfmPanel warnings={v.dfm.warnings ?? []} />
        ) : (
          <div className="text-sm text-faint">{t("q_no_dfm")}</div>
        )}
      </SectionCollapsible>

      {!!v.bom?.length && (
        <SectionCollapsible
          title={t("q_bom")}
          count={v.bom.length}
          defaultOpen={false}
          persistKey="vq:bom"
          pad="p-4"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-faint">
                  <th className="pb-1 pr-2">{t("bom_count")}</th>
                  <th className="pb-1 pr-2">{t("bom_class")}</th>
                  <th className="pb-1 pr-2">{t("bom_type")}</th>
                  <th className="pb-1">{t("bom_size")}</th>
                </tr>
              </thead>
              <tbody>
                {v.bom.map((e, i) => (
                  <tr
                    key={i}
                    className="border-t border-edge/60 text-fg"
                  >
                    <td className="py-1 pr-2 font-mono">{e.count}×</td>
                    <td className="py-1 pr-2 font-mono">{e.part_class}</td>
                    <td className="py-1 pr-2 font-mono text-faint">
                      {e.type ?? "—"}
                    </td>
                    <td className="py-1 font-mono text-faint">
                      {e.size ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCollapsible>
      )}

      {!!v.issues?.length && (
        <SectionCollapsible
          title={t("q_issues")}
          count={v.issues.length}
          persistKey="vq:issues"
          pad="p-4"
        >
          {v.issues.map((it, i) => (
            <div key={i} className="mb-1 text-xs text-muted">
              <span
                className={
                  it.severity === "major" ? "text-red-400" : "text-amber-400"
                }
              >
                [{it.severity}]
              </span>{" "}
              <span className="text-faint">({it.source})</span> {it.what}
              {it.where ? ` @ ${it.where}` : ""}
              {it.fix_hint ? (
                <span className="text-faint/80"> — fix: {it.fix_hint}</span>
              ) : null}
            </div>
          ))}
        </SectionCollapsible>
      )}

      {v.critique && (
        <SectionCollapsible
          title={t("q_critique")}
          persistKey="vq:critique"
          pad="p-4"
        >
          {v.critique.discrepancies?.length > 0 && (
            <div className="mb-2">
              {v.critique.discrepancies.map((d, i) => (
                <div key={i} className="mb-1 text-xs text-muted">
                  <span
                    className={
                      d.severity === "major" ? "text-red-400" : "text-amber-400"
                    }
                  >
                    [{d.severity}]
                  </span>{" "}
                  {d.what}
                  {d.where ? ` @ ${d.where}` : ""}
                  {d.fix_hint ? (
                    <span className="text-faint/80"> — fix: {d.fix_hint}</span>
                  ) : null}
                </div>
              ))}
            </div>
          )}
          {!!v.critique.matches?.length && (
            <div>
              <h4 className="mb-1 text-xs font-semibold text-muted">
                {t("crit_matches")}
              </h4>
              <ul className="list-inside list-disc text-xs text-faint">
                {v.critique.matches.map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
            </div>
          )}
        </SectionCollapsible>
      )}
    </div>
  );
}
