import type { Part, VersionSummary } from "./types";

export interface PartGroup {
  part: Part;
  /** every version of this part, newest first */
  versions: VersionSummary[];
  /** a non-terminal job/version exists in this part's tree */
  active: boolean;
}

const NON_TERMINAL = new Set(["queued", "running", "waiting_for_input"]);

const descNumber = <T extends { number: number }>(a: T, b: T) => b.number - a.number;

/** Group a project's versions under their parts — newest first at every
 * level: part groups by part number desc, versions inside each group by
 * version number desc. Versions whose part_id has no matching Part row
 * (stale/deleted) collect in a trailing synthetic group so nothing
 * vanishes from the UI. */
export function groupVersionsByPart(
  versions: VersionSummary[],
  parts: Part[],
): { groups: PartGroup[]; orphans: VersionSummary[] } {
  const byPart = new Map<string, VersionSummary[]>();
  for (const v of versions) {
    const list = byPart.get(v.part_id) ?? [];
    list.push(v);
    byPart.set(v.part_id, list);
  }
  const known = new Set(parts.map((p) => p.id));
  const groups: PartGroup[] = parts
    .map((part) => {
      const vs = (byPart.get(part.id) ?? []).slice().sort(descNumber);
      return {
        part,
        versions: vs,
        active: vs.some((v) => NON_TERMINAL.has(v.status) || !!v.active_job),
      };
    })
    .sort((a, b) => descNumber(a.part, b.part));
  const orphans = versions.filter((v) => !known.has(v.part_id)).sort(descNumber);
  return { groups, orphans };
}
