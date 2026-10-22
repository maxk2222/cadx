import type { Part, VersionSummary } from "./types";

export interface PartGroup {
  part: Part;
  /** every version of this part (number-asc — VersionTree roots the null-parent ones) */
  versions: VersionSummary[];
  /** a non-terminal job/version exists in this part's tree */
  active: boolean;
}

const NON_TERMINAL = new Set(["queued", "running", "waiting_for_input"]);

/** Group a project's versions under their parts (number-asc). Versions whose
 * part_id has no matching Part row (stale/deleted) collect in a trailing
 * synthetic group so nothing vanishes from the UI. */
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
  const groups: PartGroup[] = parts.map((part) => {
    const vs = byPart.get(part.id) ?? [];
    return {
      part,
      versions: vs,
      active: vs.some((v) => NON_TERMINAL.has(v.status) || !!v.active_job),
    };
  });
  const orphans = versions.filter((v) => !known.has(v.part_id));
  return { groups, orphans };
}
