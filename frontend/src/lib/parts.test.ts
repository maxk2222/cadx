import { describe, expect, it } from "vitest";
import { groupVersionsByPart } from "./parts";
import type { Part, VersionSummary } from "./types";

const part = (over: Partial<Part> = {}): Part => ({
  id: "pt-1",
  project_id: "p",
  number: 1,
  name: "bracket",
  version_count: 0,
  thumbnail_url: null,
  created_at: "",
  ...over,
});

const vs = (over: Partial<VersionSummary> = {}): VersionSummary => ({
  id: "v-1",
  project_id: "p",
  part_id: "pt-1",
  parent_id: null,
  number: 1,
  kind: "generate",
  prompt: "a bracket",
  name: "",
  status: "succeeded",
  score: null,
  iterations: 1,
  error: null,
  gen_mode: "auto",
  resolved_mode: "cad",
  created_at: "",
  thumbnail_url: null,
  ...over,
});

describe("groupVersionsByPart", () => {
  it("groups versions under their part, newest first at every level", () => {
    const p1 = part({ id: "pt-1", number: 1 });
    const p2 = part({ id: "pt-2", number: 2, name: "housing" });
    const { groups, orphans } = groupVersionsByPart(
      [
        vs({ id: "v1", part_id: "pt-1", number: 1, status: "succeeded" }),
        vs({ id: "v2", part_id: "pt-1", number: 2, status: "running" }),
        vs({ id: "v3", part_id: "pt-2", number: 3, status: "failed" }),
      ],
      [p1, p2],
    );
    expect(orphans).toEqual([]);
    expect(groups).toHaveLength(2);
    // newest part (highest P number) leads regardless of input order
    expect(groups[0].part).toBe(p2);
    expect(groups[1].part).toBe(p1);
    // versions inside a group are newest-first
    expect(groups[1].versions.map((v) => v.id)).toEqual(["v2", "v1"]);
    expect(groups[1].active).toBe(true);
    expect(groups[0].active).toBe(false);
  });

  it("keeps parts without versions and collects orphaned versions", () => {
    const p = part({ id: "pt-1" });
    const { groups, orphans } = groupVersionsByPart(
      [vs({ id: "v-gone", part_id: "pt-gone" })],
      [p],
    );
    expect(groups[0].versions).toEqual([]);
    expect(groups[0].active).toBe(false);
    expect(orphans.map((v) => v.id)).toEqual(["v-gone"]);
  });

  it("flags waiting_for_input and active_job as active", () => {
    const p = part({ id: "pt-1" });
    const { groups } = groupVersionsByPart(
      [
        vs({ id: "w", part_id: "pt-1", status: "waiting_for_input" }),
        vs({
          id: "j",
          part_id: "pt-1",
          status: "queued",
          active_job: {
            id: "jb",
            type: "generate",
            status: "running",
            step: null,
            iteration: 0,
            progress: 0,
            max_iterations: null,
          },
        }),
      ],
      [p],
    );
    expect(groups[0].active).toBe(true);
  });
});
