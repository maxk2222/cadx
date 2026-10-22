import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router-dom";
import { api, errMsg } from "../../lib/api";
import { groupVersionsByPart } from "../../lib/parts";
import type { Part, VersionSummary } from "../../lib/types";
import { Card, SectionTitle, inputCls, useDebounced, useToast } from "../ui";
import PartCard from "./PartCard";

/** Right-column "Parts" section on the project page — one collapsible
 *  PartCard per part (own version tree each), an add button for empty
 *  parts, the prompt/name filter, `?part=<id>` deep-link scrolling, and a
 *  synthetic trailing card for versions whose part is unknown (brief
 *  fetch skew — the new part arrives with the next refresh). */
export default function PartsSection({
  projectId,
  versions,
  parts,
  activeId,
}: {
  projectId: string;
  versions: VersionSummary[];
  parts: Part[];
  /** currently viewed version — highlighted inside its part's tree */
  activeId?: string;
}) {
  const { t } = useTranslation("projects");
  const qc = useQueryClient();
  const toast = useToast();
  const [vFilter, setVFilter] = useState("");
  const vq = useDebounced(vFilter);
  const [searchParams] = useSearchParams();
  const focusPart = searchParams.get("part");
  const [newPartId, setNewPartId] = useState<string | null>(null);

  const { groups, orphans } = useMemo(
    () => groupVersionsByPart(versions, parts),
    [versions, parts],
  );

  const needle = vq.trim().toLowerCase();
  const match = (v: VersionSummary) =>
    !needle ||
    v.prompt.toLowerCase().includes(needle) ||
    v.name.toLowerCase().includes(needle);
  const shown = groups
    .map((g) => ({ ...g, versions: g.versions.filter(match) }))
    .filter(
      (g) =>
        !needle ||
        g.versions.length > 0 ||
        g.part.name.toLowerCase().includes(needle),
    );
  const shownOrphans = orphans.filter(match);

  const addPart = useMutation({
    mutationFn: () => api.createPart(projectId, ""),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ["parts", projectId] });
      qc.invalidateQueries({ queryKey: ["project", projectId] });
      qc.invalidateQueries({ queryKey: ["projects"] });
      setNewPartId(p.id);
    },
    onError: (e) => toast(errMsg(e), "error"),
  });

  const totalVersions = versions.length;

  return (
    <div className="flex flex-col gap-3">
      <Card className="p-3">
        <SectionTitle
          right={
            <button
              type="button"
              onClick={() => addPart.mutate()}
              disabled={addPart.isPending}
              title={t("part_add_title")}
              className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium normal-case tracking-normal text-brand-400 transition-colors hover:bg-panel2 hover:text-brand-300"
            >
              <Plus className="h-3.5 w-3.5" /> {t("part_add")}
            </button>
          }
        >
          {t("parts_title")}
          <span className="ml-1.5 font-normal normal-case tracking-normal text-faint/70">
            {parts.length}
          </span>
        </SectionTitle>
        {totalVersions > 4 && (
          <div className="relative mb-1">
            <Search className="pointer-events-none absolute left-2 top-2 h-3.5 w-3.5 text-faint" />
            <input
              className={`${inputCls} py-1.5 pl-7 text-xs`}
              placeholder={t("filter_prompts_ph")}
              value={vFilter}
              onChange={(e) => setVFilter(e.target.value)}
            />
          </div>
        )}
      </Card>

      {shown.map((g) => (
        <PartCard
          key={g.part.id}
          group={g}
          projectId={projectId}
          activeId={activeId}
          forceOpen={focusPart === g.part.id}
          startEditing={newPartId === g.part.id}
        />
      ))}

      {shownOrphans.length > 0 && (
        <PartCard
          group={{
            part: {
              id: "__orphans__",
              project_id: projectId,
              number: 0,
              name: t("parts_ungrouped"),
              version_count: shownOrphans.length,
              thumbnail_url: null,
              created_at: "",
            },
            versions: shownOrphans,
            active: false,
          }}
          projectId={projectId}
          activeId={activeId}
        />
      )}

      {parts.length === 0 && !versions.length && (
        <Card className="p-4 text-center text-xs italic text-faint">
          {t("parts_empty_hint")}
        </Card>
      )}
    </div>
  );
}
