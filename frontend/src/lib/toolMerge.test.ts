import { describe, expect, it } from "vitest";
import { mergeToolLogs, mergeToolResult } from "./toolMerge";
import type { ToolEvent } from "./useChatEvents";
import type { ToolLogEntry } from "./types";

const open = (seq: number, name: string, call_idx?: number): ToolEvent => ({
  seq,
  name,
  call_idx,
  done: false,
});

const doneEntry = (name: string, ok = true): ToolLogEntry => ({
  name,
  args: {},
  result: { ok },
});

describe("mergeToolResult", () => {
  it("closes the matching open call by call_idx — same-name calls stay distinct", () => {
    // two repeated measure() calls: only the second is still open
    const evs = [
      { ...open(0, "measure", 0), done: true },
      open(1, "measure", 1),
      open(2, "measure", 2),
    ];
    const { evs: out } = mergeToolResult(
      evs,
      { name: "measure", ok: true, result: { v: 1 }, call_idx: 1 },
      3,
    );
    expect(out[1].done).toBe(true);
    expect(out[1].result).toEqual({ v: 1 });
    expect(out[2].done).toBe(false); // the newer open call untouched
  });

  it("falls back to name matching when no call_idx was sent", () => {
    const evs = [open(0, "measure"), open(1, "measure")];
    const { evs: out } = mergeToolResult(evs, { name: "measure", ok: true }, 2);
    expect(out[0].done).toBe(false); // newest open wins, not the oldest
    expect(out[1].done).toBe(true);
  });

  it("appends a closed entry when the start event was lost", () => {
    const { evs, usedSeq } = mergeToolResult(
      [],
      { name: "x", ok: false, result: { error: "e" }, call_idx: 4 },
      7,
    );
    expect(usedSeq).toBe(true);
    expect(evs).toHaveLength(1);
    expect(evs[0]).toMatchObject({ seq: 7, name: "x", ok: false, done: true, call_idx: 4 });
  });
});

describe("mergeToolLogs", () => {
  it("renders persisted entries only when the turn is done", () => {
    const log = mergeToolLogs([doneEntry("a"), doneEntry("b")], [open(9, "c")], false);
    expect(log.map((t) => t.name)).toEqual(["a", "b"]);
  });

  it("mid-turn: persisted + live entries merge without duplicates", () => {
    // reload mid-turn: snapshot restored calls 0-1, live stream only saw 1-2
    const persisted = [doneEntry("search"), doneEntry("measure")];
    const live = [
      { ...open(0, "measure", 1), done: true }, // already persisted — drop
      open(1, "fem", 2), // genuinely new
    ];
    const log = mergeToolLogs(persisted, live, true);
    expect(log.map((t) => t.call_idx)).toEqual([0, 1, 2]);
    expect(log.map((t) => t.name)).toEqual(["search", "measure", "fem"]);
  });

  it("live entries without call_idx always render (cannot be persisted)", () => {
    const log = mergeToolLogs([doneEntry("a")], [open(0, "b")], true);
    expect(log.map((t) => t.name)).toEqual(["a", "b"]);
  });

  it("handles an empty persisted log while streaming", () => {
    const log = mergeToolLogs(undefined, [open(0, "a", 0)], true);
    expect(log).toHaveLength(1);
  });
});
