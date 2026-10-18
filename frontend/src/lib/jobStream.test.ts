import { describe, expect, it } from "vitest";
import {
  applyBacklog,
  applyStatus,
  capEvents,
  fetchJobRow,
  isSettledStatus,
  isTerminalStatus,
  parseLogBacklog,
  pushEvent,
  type JobEvent,
} from "./jobStream";
import type { Job } from "./types";

const job = (over: Partial<Job> = {}): Job => ({
  id: "j1",
  version_id: "v1",
  type: "generate",
  status: "succeeded",
  step: null,
  iteration: 1,
  progress: 1,
  log: "",
  live: null,
  questions: null,
  cancel_requested: false,
  finish_requested: false,
  started_at: null,
  finished_at: null,
  created_at: "2024-01-01T00:00:00Z",
  ...over,
});

describe("log backlog snapshot", () => {
  it("parses log lines into backlog events", () => {
    expect(parseLogBacklog("a\nb\n")).toEqual([
      { type: "log", message: "a" },
      { type: "log", message: "b" },
    ]);
    expect(parseLogBacklog("")).toEqual([]);
    expect(parseLogBacklog(undefined)).toEqual([]);
  });

  it("replays are idempotent — repeated status does not duplicate", () => {
    let ev: JobEvent[] = [];
    ev = applyBacklog(ev, "line1\nline2");
    ev = applyBacklog(ev, "line1\nline2\nline3");
    ev = applyBacklog(ev, "line1\nline2\nline3"); // identical reconnect replay
    expect(ev).toEqual([
      { type: "log", message: "line1" },
      { type: "log", message: "line2" },
      { type: "log", message: "line3" },
    ]);
  });

  it("keeps live step/question/done events behind the new backlog", () => {
    let ev = applyBacklog([], "l1");
    ev = pushEvent(ev, { type: "step", step: "codegen", t_ms: 1 });
    ev = applyBacklog(ev, "l1\nl2");
    expect(ev.map((e) => e.type)).toEqual(["log", "log", "step"]);
  });
});

describe("status settle detection", () => {
  it.each(["succeeded", "failed", "cancelled"])("terminal: %s", (s) => {
    const r = applyStatus([], { status: s, log: "" });
    expect(r.settled).toBe(true);
    expect(r.terminal).toBe(true);
    expect(isTerminalStatus(s)).toBe(true);
    expect(isSettledStatus(s)).toBe(true);
  });

  it("waiting_for_input settles the stream but is not done", () => {
    const r = applyStatus([], { status: "waiting_for_input", log: "" });
    expect(r.settled).toBe(true);
    expect(r.terminal).toBe(false);
    expect(isTerminalStatus("waiting_for_input")).toBe(false);
    expect(isSettledStatus("waiting_for_input")).toBe(true);
  });

  it("running/queued keep the stream open", () => {
    for (const s of ["running", "queued", undefined]) {
      const r = applyStatus([], { status: s, log: "x" });
      expect(r.settled).toBe(false);
      expect(r.terminal).toBe(false);
    }
  });
});

describe("closed guard", () => {
  it("returns the row + blank answers while open", async () => {
    const j = job({ questions: [{ question: "q?", why_it_matters: "", default_assumption: "", blocking: true }] });
    const r = await fetchJobRow(async () => j, "j1", () => false);
    expect(r?.job).toBe(j);
    expect(r?.answers).toEqual([""]);
  });

  it("suppresses stale writes after close", async () => {
    const r = await fetchJobRow(async () => job(), "j1", () => true);
    expect(r).toBeNull();
  });

  it("propagates fetch errors for the caller to swallow", async () => {
    await expect(
      fetchJobRow(async () => {
        throw new Error("404");
      }, "j1", () => false),
    ).rejects.toThrow("404");
  });
});

describe("events cap", () => {
  it("drops oldest log lines first, keeps steps/questions/done", () => {
    const cap = 10;
    let ev: JobEvent[] = [];
    for (let i = 0; i < 12; i++) ev = pushEvent(ev, { type: "log", message: `l${i}` }, cap);
    ev = pushEvent(ev, { type: "done", status: "succeeded" }, cap);
    expect(ev.length).toBe(cap);
    expect(ev.some((e) => e.type === "done")).toBe(true);
    // oldest two log lines dropped
    expect(ev[0]).toEqual({ type: "log", message: "l3" });
  });

  it("falls back to dropping oldest when nothing is droppable", () => {
    let ev: JobEvent[] = [];
    for (let i = 0; i < 8; i++)
      ev = pushEvent(ev, { type: "step", step: `s${i}` }, 5);
    expect(ev.length).toBe(5);
    expect(ev[0].step).toBe("s3");
  });

  it("caps oversized backlogs too", () => {
    const ev = capEvents(
      Array.from({ length: 700 }, (_, i) => ({ type: "log", message: `l${i}` })),
    );
    expect(ev.length).toBe(600);
    expect(ev[0].message).toBe("l100");
  });
});
