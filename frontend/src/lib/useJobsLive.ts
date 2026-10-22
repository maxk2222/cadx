/**
 * Global job lifecycle stream (`GET /jobs/events`) — mounted once in App.
 *
 *  - `jobs`      → full refreshed active-job list pushed per transition;
 *                  written straight into ["jobs","active"] via setQueryData
 *                  so the badge updates with zero HTTP refetches.
 *  - `lifecycle` → single create/status transition {job_id, version_id,
 *                  status, job_type}; debounced invalidations bootstrap the
 *                  conditional polls on JobsPage/ProjectsPage/version lists
 *                  (they only poll while something is active — this is what
 *                  tells them activity started).
 *
 * The server replays a snapshot on (re)connect, so reconnects heal stale
 * views without client polling. Returns `connected` so callers can drop to
 * a degraded poll while the stream is down.
 */
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { jobsEventsUrl } from "./api";
import { applyJobsSnapshot, invalidateJobDependents } from "./jobsLiveSync";
import type { JobListItem } from "./types";

export function useJobsLive(): boolean {
  const qc = useQueryClient();
  const [connected, setConnected] = useState(false);
  const debRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const pendingVersions = new Set<string>();
    const debouncedInvalidate = (versionId?: string) => {
      if (versionId) pendingVersions.add(versionId);
      if (debRef.current) clearTimeout(debRef.current);
      // transitions arrive in bursts (queued→running→terminal) — one
      // trailing invalidation collapses them into a single refetch wave
      debRef.current = setTimeout(() => {
        invalidateJobDependents(qc, pendingVersions);
        pendingVersions.clear();
      }, 400);
    };

    const es = new EventSource(jobsEventsUrl());
    es.addEventListener("jobs", (e) => {
      try {
        const d = JSON.parse((e as MessageEvent).data) as {
          items: JobListItem[];
        };
        applyJobsSnapshot(qc, d.items);
      } catch (err) {
        console.warn("jobsEvents: malformed 'jobs' event", err);
      }
    });
    es.addEventListener("lifecycle", (e) => {
      try {
        const d = JSON.parse((e as MessageEvent).data) as {
          version_id?: string;
        };
        debouncedInvalidate(d.version_id);
      } catch {
        debouncedInvalidate();
      }
    });
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    return () => {
      if (debRef.current) clearTimeout(debRef.current);
      es.close();
    };
  }, [qc]);

  return connected;
}
