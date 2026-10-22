/**
 * Query-cache sync for the global jobs lifecycle stream — the pure part of
 * `useJobsLive`, split out so the sync policy is unit-testable without a
 * DOM/EventSource.
 *
 *  - `applyJobsSnapshot`     — pushed active list → ["jobs","active"], so
 *                              the badge updates with zero HTTP refetches.
 *  - `invalidateJobDependents` — a lifecycle transition kicks every
 *                              "poll while active" query (they bootstrap
 *                              themselves off this refetch) plus the
 *                              per-version job lists keyed by version_id.
 */
import type { QueryClient } from "@tanstack/react-query";
import type { JobListItem } from "./types";

export function applyJobsSnapshot(qc: QueryClient, items: JobListItem[]) {
  qc.setQueryData(["jobs", "active"], items);
}

export function invalidateJobDependents(
  qc: QueryClient,
  pendingVersions: ReadonlySet<string>,
) {
  qc.invalidateQueries({ queryKey: ["jobs", "all"] });
  qc.invalidateQueries({ queryKey: ["versions"] });
  qc.invalidateQueries({ queryKey: ["version"] });
  qc.invalidateQueries({ queryKey: ["projects"] });
  for (const vid of pendingVersions)
    qc.invalidateQueries({ queryKey: ["jobs", vid] });
}
