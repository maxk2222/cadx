import { describe, expect, it } from "vitest";
import type { VersionSummary } from "./types";
import { buildVersionRows } from "./versionTree";

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

const ids = (rows: ReturnType<typeof buildVersionRows>) =>
  rows.map((r) => r.v.id);
const none = new Set<string>();

describe("buildVersionRows", () => {
  it("orders roots newest-first regardless of input order", () => {
    const rows = buildVersionRows(
      [
        vs({ id: "v1", number: 1 }),
        vs({ id: "v3", number: 3 }),
        vs({ id: "v2", number: 2 }),
      ],
      none,
    );
    expect(ids(rows)).toEqual(["v3", "v2", "v1"]);
    expect(rows.every((r) => r.depth === 0)).toBe(true);
  });

  it("nests children under their parent, newest sibling first", () => {
    const rows = buildVersionRows(
      [
        vs({ id: "root", number: 1 }),
        vs({ id: "kid-old", number: 2, parent_id: "root" }),
        vs({ id: "kid-new", number: 4, parent_id: "root" }),
        vs({ id: "grand", number: 3, parent_id: "kid-old" }),
      ],
      none,
    );
    expect(ids(rows)).toEqual(["root", "kid-new", "kid-old", "grand"]);
    expect(rows.map((r) => r.depth)).toEqual([0, 1, 1, 2]);
  });

  it("hides a collapsed subtree but keeps later roots", () => {
    const rows = buildVersionRows(
      [
        vs({ id: "v1", number: 1 }),
        vs({ id: "v2", number: 2, parent_id: "v1" }),
        vs({ id: "v3", number: 3, parent_id: "v2" }),
        vs({ id: "v4", number: 4 }),
      ],
      new Set(["v1"]),
    );
    expect(ids(rows)).toEqual(["v4", "v1"]);
  });

  it("appends orphans (missing parent) at depth 0, newest first", () => {
    const rows = buildVersionRows(
      [
        vs({ id: "v1", number: 1 }),
        vs({ id: "o1", number: 2, parent_id: "gone" }),
        vs({ id: "o2", number: 3, parent_id: "gone" }),
      ],
      none,
    );
    expect(ids(rows)).toEqual(["v1", "o2", "o1"]);
  });

  it("does not append versions hidden behind a collapsed ancestor", () => {
    const rows = buildVersionRows(
      [
        vs({ id: "v1", number: 1 }),
        vs({ id: "v2", number: 2, parent_id: "v1" }),
      ],
      new Set(["v1"]),
    );
    expect(ids(rows)).toEqual(["v1"]);
  });

  it("surfaces parent-cycle members as orphans", () => {
    const rows = buildVersionRows(
      [
        vs({ id: "a", number: 1, parent_id: "b" }),
        vs({ id: "b", number: 2, parent_id: "a" }),
      ],
      none,
    );
    expect(ids(rows).sort()).toEqual(["a", "b"]);
  });
});
