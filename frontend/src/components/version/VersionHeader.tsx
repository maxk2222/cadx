import { BadgeCheck, Gauge, GitCompareArrows, Package, Pencil, Trash2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { fmtDuration, fmtSecs, fmtTokens } from "../../lib/format";
import { stepLabel } from "../../lib/steps";
import type { Job, Part, VersionDetail } from "../../lib/types";
import StatusBadge from "../StatusBadge";
import { OptionChips } from "../OptionsPanel";
import { Btn, Chip, Crumbs, OverflowMenu, StatRow, inputCls } from "../ui";

/** crumbs + title + status chips + action row — kept separate so the page
 *  body stays a flat grid. Rename and delete live here; compare is a toggle
 *  owned by the page (it switches the overview layout). */
export default function VersionHeader({
  v,
  projectName,
  projectId,
  part,
  compareOpen,
  canCompare,
  onToggleCompare,
  onRename,
  onDelete,
  onToggleAccept,
  acceptBusy,
}: {
  v: VersionDetail;
  projectName: string | undefined;
  projectId: string;
  /** owning part — chip links back to its section on the project page */
  part?: Part | null;
  compareOpen: boolean;
  canCompare: boolean;
  onToggleCompare: () => void;
  onRename: (name: string) => void;
  onDelete: () => void;
  /** toggle the user-verified master flag (one per part) */
  onToggleAccept: () => void;
  acceptBusy?: boolean;
}) {
  const { t } = useTranslation("version");
  const [nameEdit, setNameEdit] = useState(false);
  const [nameDraft, setNameDraft] = useState("");

  return (
    <>
      <div className="mb-2">
        <Crumbs
          items={[
            <Link key="p" to="/projects" className="hover:text-muted">
              {t("crumb_projects")}
            </Link>,
            <Link
              key="proj"
              to={`/projects/${projectId}`}
              className="hover:text-muted"
            >
              {projectName ?? t("crumb_project")}
            </Link>,
            <span key="v" className="font-medium text-fg">
              v{v.number}
              {v.name ? ` · ${v.name}` : ""}
            </span>,
          ]}
        />
      </div>

      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h1 className="text-xl font-bold">v{v.number}</h1>
        {nameEdit ? (
          <input
            autoFocus
            className={`${inputCls} w-64 py-0.5 text-sm`}
            placeholder={t("name_placeholder")}
            maxLength={200}
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur();
              if (e.key === "Escape") setNameEdit(false);
            }}
            onBlur={() => {
              setNameEdit(false);
              const name = nameDraft.trim();
              if (name !== v.name) onRename(name);
            }}
          />
        ) : (
          <button
            type="button"
            title={t("rename_title")}
            onClick={() => {
              setNameDraft(v.name);
              setNameEdit(true);
            }}
            className="group flex items-center gap-1.5 rounded px-1.5 py-0.5 text-lg text-fg transition-colors hover:bg-panel2"
          >
            {v.name || (
              <span className="text-xs italic text-faint">{t("add_name")}</span>
            )}
            <Pencil className="h-3 w-3 text-faint/60 transition-colors group-hover:text-muted" />
          </button>
        )}
        <StatusBadge status={v.status} />
        <Chip tone="zinc" className="uppercase">
          {v.kind}
        </Chip>
        {part && (
          <Link to={`/projects/${projectId}?part=${part.id}`}>
            <Chip tone="blue" title={t("part_title")}>
              <Package className="h-3 w-3" /> P{part.number}
              {part.name ? ` · ${part.name}` : ""}
            </Chip>
          </Link>
        )}
        {(v.gen_mode || v.resolved_mode) && (
          <Chip
            tone={
              v.resolved_mode === "mesh"
                ? "purple"
                : v.resolved_mode === "hybrid"
                  ? "teal"
                  : "zinc"
            }
            title={t("mode_title")}
          >
            {v.gen_mode}
            {v.resolved_mode && v.resolved_mode !== v.gen_mode
              ? ` → ${v.resolved_mode}`
              : ""}
          </Chip>
        )}
        <OptionChips options={v.options} />
        {v.score != null && (
          <Chip tone={v.score >= 80 ? "green" : v.score >= 50 ? "amber" : "red"}>
            <Gauge className="h-3 w-3" /> {t("score", { n: v.score })}
          </Chip>
        )}
        {v.accepted_at && (
          <Chip tone="green" title={t("master_title")}>
            <BadgeCheck className="h-3 w-3" /> {t("master_badge")}
          </Chip>
        )}
        <span className="flex-1" />
        <Btn
          variant={v.accepted_at ? "success" : "subtle"}
          size="sm"
          aria-pressed={!!v.accepted_at}
          title={v.accepted_at ? t("unaccept_title") : t("accept_title")}
          busy={acceptBusy}
          onClick={onToggleAccept}
        >
          <BadgeCheck className="h-3.5 w-3.5" />
          {v.accepted_at ? t("unaccept") : t("accept")}
        </Btn>
        <Btn
          variant="subtle"
          size="sm"
          aria-pressed={compareOpen}
          title={canCompare ? t("compare_title") : t("compare_none")}
          onClick={onToggleCompare}
        >
          <GitCompareArrows className="h-3.5 w-3.5" /> {t("compare")}
        </Btn>
        <OverflowMenu
          title={t("more_actions")}
          items={[
            {
              icon: <Trash2 className="h-3.5 w-3.5" />,
              label: t("action_delete"),
              danger: true,
              onClick: onDelete,
            },
          ]}
        />
      </div>
    </>
  );
}

/** the metric strip — one compact dotted row instead of a chip soup.
 *  live values (stream) win over the stored job where available. */
export function VersionStatRow({
  v,
  liveJob,
  curJob,
  liveMax,
  stepStartedTs,
}: {
  v: VersionDetail;
  liveJob: Job | null;
  curJob: Job | undefined;
  liveMax: number | undefined;
  stepStartedTs: string | undefined;
}) {
  const { t } = useTranslation("version");
  const dfmCount = v.dfm?.warnings?.length ?? 0;
  const usageEntries = Object.entries(v.usage ?? {}).filter(
    ([, n]) => typeof n === "number" && n > 0
  );
  const sf = v.fem?.cases?.length
    ? Math.min(...v.fem.cases.map((c) => c.safety_factor ?? Infinity))
    : null;

  return (
    <StatRow
      className="mb-4"
      items={[
        <span key="it" title={t("stat_iterations")}>
          {liveJob && liveMax
            ? t("stat_it_live", {
                cur: Math.min(liveJob.iteration, liveMax),
                max: liveMax,
              })
            : t("stat_iter_n", { count: v.iterations })}
        </span>,
        liveJob?.step && (
          <span key="step" className="text-brand-300" title={t("stat_step")}>
            {stepLabel(liveJob.step)}
            {liveJob.iteration > 0 ? ` #${liveJob.iteration}` : ""}
          </span>
        ),
        (liveJob ?? curJob)?.started_at && (
          <span
            key="dur"
            title={t("stat_started", { at: (liveJob ?? curJob)!.started_at })}
          >
            {fmtDuration((liveJob ?? curJob)!.started_at, (liveJob ?? curJob)!.finished_at)}
            {liveJob && stepStartedTs && (
              <span className="text-faint">
                {" "}
                ·{" "}
                {fmtSecs(
                  Math.max(
                    0,
                    (Date.now() - new Date(stepStartedTs).getTime()) / 1000
                  )
                )}
              </span>
            )}
          </span>
        ),
        ...usageEntries.map(([k, n]) => (
          <span key={`u-${k}`} title={t("stat_llm", { name: k })}>
            {k.replaceAll("_", " ")}: {fmtTokens(n as number)}
          </span>
        )),
        dfmCount > 0 && (
          <span key="dfm" className="text-amber-300" title={t("stat_dfm")}>
            {t("stat_dfm_n", { count: dfmCount })}
          </span>
        ),
        sf != null && (
          <span
            key="sf"
            className={sf < 1 ? "text-red-300" : "text-purple-300"}
            title={t("stat_sf")}
          >
            SF {sf.toFixed(2)}
          </span>
        ),
      ]}
    />
  );
}
