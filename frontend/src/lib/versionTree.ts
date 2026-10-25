import type { VersionSummary } from "./types";

export interface VersionRow {
  v: VersionSummary;
  depth: number;
}

const desc = (a: VersionSummary, b: VersionSummary) => b.number - a.number;

/** Flatten a version forest into display rows: roots first (newest on top),
 *  children nested under their parent — every sibling level sorted
 *  newest-first. `collapsed` hides a version's whole subtree.
 *
 *  Orphan safety: versions never reached by the walk (missing parent or a
 *  parent cycle) are appended at depth 0, sorted desc — unless they sit
 *  behind a collapsed ancestor, which is intentional hiding, not orphaning.
 */
export function buildVersionRows(
  versions: VersionSummary[],
  collapsed: ReadonlySet<string>,
): VersionRow[] {
  const byParent = new Map<string | null, VersionSummary[]>();
  for (const v of versions) {
    const key = v.parent_id ?? null;
    byParent.set(key, [...(byParent.get(key) ?? []), v]);
  }
  for (const kids of byParent.values()) kids.sort(desc);

  const rows: VersionRow[] = [];
  const walk = (parent: string | null, depth: number) => {
    for (const v of byParent.get(parent) ?? []) {
      rows.push({ v, depth });
      if (!collapsed.has(v.id)) walk(v.id, depth + 1);
    }
  };
  walk(null, 0);

  const seen = new Set(rows.map((r) => r.v.id));
  const byId = new Map(versions.map((v) => [v.id, v]));
  const hiddenByCollapse = (v: VersionSummary) => {
    const guard = new Set<string>();
    let p = v.parent_id;
    while (p) {
      if (collapsed.has(p)) return true;
      if (guard.has(p)) return false; // parent cycle — treat as orphan
      guard.add(p);
      p = byId.get(p)?.parent_id ?? null; // absent parent ends the chain
    }
    return false;
  };
  const orphans = versions
    .filter((v) => !seen.has(v.id) && !hiddenByCollapse(v))
    .sort(desc);
  for (const v of orphans) rows.push({ v, depth: 0 });
  return rows;
}
