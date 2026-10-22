import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { applyJobsSnapshot, invalidateJobDependents } from "./jobsLiveSync";
import type { JobListItem } from "./types";

const item = (id: string, status: string): JobListItem =>
  ({ id, status }) as JobListItem;

describe("applyJobsSnapshot", () => {
  it("writes the pushed active list into the badge query cache", () => {
    const qc = new QueryClient();
    applyJobsSnapshot(qc, [item("j1", "running")]);
    expect(qc.getQueryData(["jobs", "active"])).toEqual([item("j1", "running")]);
    // terminal transition pushes an empty list → badge clears
    applyJobsSnapshot(qc, []);
    expect(qc.getQueryData(["jobs", "active"])).toEqual([]);
  });
});

describe("invalidateJobDependents", () => {
  it("kicks every poll-while-active query + the touched version's job list", () => {
    const qc = new QueryClient();
    const spy = vi.spyOn(qc, "invalidateQueries");
    invalidateJobDependents(qc, new Set(["v1", "v2"]));
    const keys = spy.mock.calls.map((c) => (c[0] as { queryKey: string[] }).queryKey);
    expect(keys).toContainEqual(["jobs", "all"]);
    expect(keys).toContainEqual(["versions"]);
    expect(keys).toContainEqual(["version"]);
    expect(keys).toContainEqual(["projects"]);
    expect(keys).toContainEqual(["jobs", "v1"]);
    expect(keys).toContainEqual(["jobs", "v2"]);
    // the badge cache is setQueryData-managed — never invalidated
    expect(keys).not.toContainEqual(["jobs", "active"]);
  });
});
