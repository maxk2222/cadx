import { describe, expect, it } from "vitest";
import type { ChatMessage } from "./types";
import {
  computeWindow,
  estimateMessageHeight,
  ROW_GAP,
  tailWindow,
} from "./virtualList";

const slots = (n: number, h = 100) => Array.from({ length: n }, () => h + ROW_GAP);

const mkMsg = (content: string, meta: ChatMessage["meta"] = {}): ChatMessage => ({
  id: "m",
  chat_id: "c",
  role: "assistant",
  content,
  asset_ids: [],
  meta,
  pending: false,
  created_at: "",
});

describe("computeWindow", () => {
  it("renders everything when the whole list fits", () => {
    expect(computeWindow(slots(5), 0, 2000, 0)).toEqual({
      start: 0,
      end: 5,
      topPad: 0,
      bottomPad: 0,
    });
  });

  it("windows around the viewport plus overscan", () => {
    // 100 rows of 112px slots; viewport [5000, 6000], overscan 600 → rows
    // covering [4400, 6600]: row i spans [112i, 112(i+1))
    const w = computeWindow(slots(100), 5000, 1000, 600);
    expect(w.start).toBe(39); // row 39 straddles 4400
    expect(w.end).toBe(59); // row 58 ends at 6608 > 6600
    // spacers reproduce the hidden prefix/suffix minus their own flex gap
    expect(w.topPad).toBe(39 * 112 - ROW_GAP);
    expect(w.bottomPad).toBe((100 - 59) * 112 - ROW_GAP);
  });

  it("keeps row offsets exact with variable heights", () => {
    const s = [112, 212, 62, 112, 412];
    const w = computeWindow(s, 130, 100, 0);
    // row 0 covers [0,112), row 1 covers [112,324) — 130 lands in row 1
    expect(w.start).toBe(1);
    expect(w.topPad + ROW_GAP).toBe(112);
  });

  it("clamps at both ends", () => {
    const tail = computeWindow(slots(100), 1e9, 1000);
    expect(tail.end).toBe(100);
    expect(tail.bottomPad).toBe(0);
    const head = computeWindow(slots(100), -1e9, 1000);
    expect(head.start).toBe(0);
    expect(head.topPad).toBe(0);
  });

  it("handles the empty list", () => {
    expect(computeWindow([], 0, 800)).toEqual({
      start: 0,
      end: 0,
      topPad: 0,
      bottomPad: 0,
    });
  });
});

describe("tailWindow", () => {
  it("covers at least minRows of tail", () => {
    const w = tailWindow(slots(100), 20, 1800);
    expect(w.end).toBe(100);
    expect(w.start).toBe(80); // 20 rows × 112 = 2240 ≥ 1800
    expect(w.topPad).toBe(80 * 112 - ROW_GAP);
    expect(w.bottomPad).toBe(0);
  });
});

describe("estimateMessageHeight", () => {
  it("grows with content and attachments", () => {
    const short = estimateMessageHeight(mkMsg("hi"));
    const long = estimateMessageHeight(mkMsg("x".repeat(900)));
    const tools = estimateMessageHeight(
      mkMsg("hi", { tool_log: [{ name: "t" }, { name: "t" }] })
    );
    expect(short).toBeGreaterThanOrEqual(44);
    expect(long).toBeGreaterThan(short);
    expect(tools).toBeGreaterThan(short);
  });

  it("system rows are small", () => {
    const m = mkMsg("done");
    m.role = "system";
    expect(estimateMessageHeight(m)).toBe(24);
  });
});
